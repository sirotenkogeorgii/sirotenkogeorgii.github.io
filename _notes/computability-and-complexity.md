---
layout: default
title: Computability and Complexity
date: 2024-10-20
excerpt: A deep-dive into the formal foundations of computability, classic machine models, and the time-hierarchy lens on complexity theory.
tags:
  - computability
  - complexity
  - theory
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

<div class="note-callout">
  <p class="note-callout__title">Remark</p>
  <p>
    Classical deterministic mathematics emerges as the special case of
    probability theory where the sample space $\Omega$ contains only a single
    outcome. With $P(\{\omega\}) = 1$ every random variable collapses to a
    deterministic quantity. This perspective is helpful later when we discuss
    probabilistic computation: the usual discrete models are simply the
    zero-entropy corner of the same formalism.
  </p>
</div>

# Computability and Complexity

## Complexity

### Formalizing Computability

In theoretical computer science, the abstract concept of computability is given a precise mathematical foundation through the study of partial computable functions. From this core idea, related concepts such as computable functions and decidable sets are derived. While several equivalent models exist for formalizing computability—including register machines, µ-recursive functions, and the lambda calculus—this study will focus exclusively on the model provided by Turing machines. All these formalisms are provably identical, meaning they all define the same class of computable functions and decidable sets.

### The Turing Machine: A Formal Definition

The Turing machine serves as a foundational abstract model of a computing device. It consists of a finite set of states and one or more tapes that serve as its memory.

$\textbf{Definition 1 (Turing Machine):}$ Let $k$ be a nonzero natural number. A Turing machine with $k$ tapes, also called a $k$-tape Turing machine or $k$-tape TM, is a $6$-tuple of the form $M = (Q, \Sigma, \Gamma, \Delta, s, F)$ where

- $Q$ is the finite set of states,
- $\Sigma$ is the input alphabet,
- $\Gamma$ is the tape or working alphabet, where $\Sigma \subseteq \Gamma$ and the blank symbol $\square \in \Gamma \setminus \Sigma$,
- $\Delta$ is the transition relation of $M$, where for $\text{Mov} = \lbrace L, R, S \rbrace$, the relation $\Delta$ is a subset of $Q \times \Gamma^k \times Q \times \Gamma^k \times \text{Mov}^k$,
- $s \in Q$ is the initial state,
- $F \subseteq Q$ is the set of accepting states.

<div class="accordion">
  <details>
    <summary>"States" in CPU</summary>

## 1. “State” in the sense of complexity theory / automata

When I compared a Turing machine’s **state** to a CPU, I was talking about the **control logic** of the CPU, which you can model as a **finite-state machine**:

* The control unit has a **state register** (a bunch of flip-flops).
* On each clock cycle, this state register + some inputs (like parts of the instruction) determine:

  * what control signals to assert (read from register file, write to memory, etc.),
  * what the **next state** of the control unit will be.

Here:

* The set of possible control states is **finite** and **fixed by the design** of the CPU.
* Each state typically corresponds to a “phase” of instruction processing:

  * e.g. `FETCH`, `DECODE`, `EXECUTE`, `MEM_READ`, `MEM_WRITE`, `WRITE_BACK`, or finer-grained micro-steps.

This is the analog of a Turing machine’s (or DFA’s) **q ∈ Q**:

> A small piece of control info that decides “what kind of step am I doing now?”, **not** the entire memory contents.

In complexity theory, when we say “a machine has finitely many states”, we mean this *control* part, **not** the whole RAM/register contents.

## 2. How this connects back to Turing machines

For a Turing machine, you can think of:

* **Finite control state** (the “q” in TM formalism) ↔ CPU control unit’s FSM state (e.g. `FETCH`, `EXECUTE_ADD`, etc.).
* **Tape contents + head positions** ↔ CPU’s registers + main memory + program counter.

When we talk about space complexity and say **“the machine has only finitely many states”**, we mean:

* The **control states** are finite and don’t grow with input size.
* The “unbounded” or input-dependent information lives in **tape cells** (for TM) or **RAM/registers** (for a real CPU).

So:

* **In complexity theory**: “state” = the control state, a small, constant-sized piece of info.
  </details>
</div>

#### Structure of a $k$-tape Turing Machine

- **Memory.** The machine's memory consists of $k$ work tapes. Each tape is a doubly infinite sequence of cells, and each cell holds a single symbol from the tape alphabet $\Gamma$.
- **Heads.** Each tape has an associated read-write head. At any given time, each head is positioned on one cell of its tape. It can read the symbol in that cell and overwrite it with a new symbol.
- **Movement.** The heads on the $k$ tapes can move independently of one another. In a single step, a head can move one cell to the left ($L$), one cell to the right ($R$), or stay in the same position ($S$).



### How Turing Machines Compute

A Turing machine operates by executing a sequence of computational steps, beginning from an initial setup determined by an input word.

#### The Computation Process

The input to a Turing machine is a word $w$ over its input alphabet $\Sigma$. At the start of a computation:

1. The symbols of the input word $w$ are written consecutively on the first tape.
2. All other cells on the first tape, and all cells on the other $k-1$ tapes, are filled with the blank symbol, $\square$.
3. The machine is in its initial state, $s$.
4. The head of the first tape is positioned on the first symbol of the input word.

A computation is a sequence of steps, where each step involves executing an instruction from the machine's transition relation $\Delta$. This instruction determines the machine's next state, the symbols to be written on the tapes, and the movement of each head.

#### Instructions

The "program" of a Turing machine is its transition relation $\Delta$, which is a set of tuples called instructions. For a $k$-tape TM, an instruction has the form: $(q, a_1, \dots, a_k, q', a'_1, \dots, a'_k, Z_1, \dots, Z_k) \in Q \times \Gamma^k \times Q \times \Gamma^k \times \text{Mov}^k$ 

#### The intuitive meaning of such an instruction is as follows:

* Condition: The instruction can be executed if the machine is currently in state$ q$ and the symbols being read by the heads on tapes $1$ through $k$ are $a_1$ through $a_k$, respectively.
* Action: When executed, the machine transitions to state $q'$. Simultaneously, for each tape $i$ (from $1$ to $k$), the symbol $a_i$ under the head is overwritten with $a'_i$, and the head then performs the movement $Z_i \in \lbrace L, R, S \rbrace$.

To formalize this, we distinguish between the two parts of an instruction.

$\textbf{Definition 2 (Condition Part, Instruction part):}$ For an instruction of the form $(q, a_1, \dots, a_k, q', a'_1, \dots, a'_k, Z_1, \dots, Z_k)$, we refer to $(q, a_1, \dots, a_k)$ as its condition part, and to $(q', a'_1, \dots, a'_k, Z_1, \dots, Z_k)$ as its instruction part.

$\textbf{Remark 3:}$ The transition relation $\Delta$ of a Turing machine can also be understood as a relation between condition parts and instruction parts. For a $k$-tape Turing machine, $\Delta$ is thus a binary relation of the form  $\Delta \subseteq (Q \times \Gamma^k) \times (Q \times \Gamma^k \times \text{Mov}^k)$  which can be written in infix notation as  $(q, a_1, \dots, a_k) \Delta (q', a'_1, \dots, a'_k, Z_1, \dots, Z_k)$. The relation $\Delta$ is called right-unique if for each condition part $(q, a_1, \dots, a_k)$ there is at most one instruction part $(q', a'_1, \dots, a'_k, Z_1, \dots, Z_k)$ that satisfies the relation.

### Deterministic vs. Nondeterministic Machines

The nature of the transition relation $\Delta$ determines whether a Turing machine is deterministic or nondeterministic.

A general Turing machine, as defined above, does not require $\Delta$ to be right-unique. This means that for a given state and set of tape symbols, there might be multiple applicable instructions. Such a machine is called a nondeterministic Turing machine. When a computation reaches a point with multiple choices, it can branch, exploring several computational paths simultaneously.

$\textbf{Remark 5:}$ In the literature, the term *nondeterministic Turing machine* is used in two different senses. It may refer either to a *general Turing machine*, i.e., one that may or may not be deterministic, or to a Turing machine that is indeed not deterministic, i.e., has a transition relation that is not right-unique.

In contrast, a deterministic Turing machine has a transition relation that is right-unique. At any point in its computation, there is at most one applicable instruction. This ensures that for any given input, the machine follows a single, uniquely determined computational path.

$\textbf{Definition 4 (Deterministic Turing Machine):}$ A Turing machine $M$ is deterministic if its transition relation is right-unique. A $k$-tape Turing machine $(Q, \Sigma, \Gamma, \Delta, s, F)$ is deterministic if and only if for every state $q \in Q$ and every sequence of $k$ symbols $a_1, \dots, a_k$ from $\Gamma$, there is at most one instruction in $\Delta$ with condition part $(q, a_1, \dots, a_k)$.

### The Mechanics of Computation

To formally analyze Turing machine computations, we need precise definitions for the machine's state at any given moment (its configuration) and the transitions between these states.

#### Configurations: Snapshots of a Computation

A configuration is a complete snapshot of a Turing machine's status, capturing its current state, all tape contents, and the positions of all heads. Since a tape is infinite but can only contain a finite number of non-blank symbols, we need a way to represent its contents.

$\textbf{Definition 6 (Tape Inscriptions):}$ Let $M$ be a Turing machine with tape alphabet $\Gamma$. A *tape inscription* of $M$ is a function $f: \mathbb{Z} \to \Gamma$ such that $f(i) = \square$ for all but finitely many integers $i$. The *relevant part* $u_f$ of a tape inscription $f$ is a word over $\Gamma$. We let $u_f = \square$ if $f$ is the constant function with value $\square$. Otherwise, we let  
$$
u_f = f(\min I_f) \cdots f(\max I_f) \quad \text{where} \quad I_f = \lbrace i \in \mathbb{Z} : f(i) \neq \square \rbrace
$$
A tape inscription $f$ is represented by any word $u$ of the form 
$$
u = \square^{r_1} u_f \square^{r_2} \quad \text{with} \quad r_1, r_2 \in \mathbb{N}
$$

i.e., by any word u that equals the relevant part of the tape inscription $f$ plus at most finitely many leading and trailing blank symbols.

Using this representation for tape contents, we can formally define a configuration.

$\textbf{Definition 7 (Configuration):}$ Let $k$ be a nonzero natural number and let $M = (Q, \Sigma, \Gamma, \Delta, s, F)$ be a $k$-tape Turing machine. A configuration of $M$ is a tuple  
$$
(q, u_1, \dots, u_k, j_1, \dots, j_k) \in Q \times (\Gamma^+)^k \times \prod_{i=1,\dots,k} \lbrace 1, \dots, \lvert u_i \rvert \rbrace
$$
This tuple represents a situation where $q$ is the current state, and for each tape $i=1, \dots, k$:

* the word $u_i$ represents the relevant part of the inscription on tape $i$,
* the number $j_i$ indicates the head position on tape $i$, corresponding to the $j_i$-th symbol of $u_i$.

#### Computation Steps and Sequences

A computation unfolds as a sequence of configurations, where each transition from one configuration to the next is governed by an instruction from $\Delta$. This transition is called a computation step.

$\textbf{Definition 8 (Computation Step)}$: Let $C = (q, u_1, \dots, u_k, j_1, \dots, j_k)$ be a configuration of a $k$-tape Turing machine $M$. An instruction of $M$ of the form $(q, a_1, \dots, a_k, q', a'_1, \dots, a'_k, Z_1, \dots, Z_k)$ is *applicable* to $C$ if for $i=1, \dots, k$, the symbol at position $j_i$ of $u_i$ is equal to $a_i$. A configuration $C' = (q', u'_1, \dots, u'_k, j'_1, \dots, j'_k)$ is a *successor configuration* of $C$, written $C \xrightarrow{M} C'$, if there is an applicable instruction as above such that for $i=1, \dots, k$:

* $u'_i$ is defined by

  $$
  u'_i =
  \begin{cases}
    \square \tilde{u}_i & \text{if } Z_i = L \text{ and } j_i = 1, \\
    \tilde{u}_i \square & \text{if } Z_i = R \text{ and } j_i = \lvert u_i \rvert, \\
    \tilde{u}_i & \text{otherwise},
  \end{cases}
  $$

* $j'_i$ is defined by

  $$
  j'_i =
  \begin{cases}
    1 & \text{if } Z_i = L \text{ and } j_i = 1, \\
    j_i - 1 & \text{if } Z_i = L \text{ and } j_i \neq 1, \\
    j_i & \text{if } Z_i = S, \\
    j_i + 1 & \text{if } Z_i = R.
  \end{cases}
  $$

  Here $\tilde{u}_i$ denotes the word obtained from $u_i$ by replacing the symbol $a_i$ at position $j_i$ with $a'_i$.

The relation $\xrightarrow{M}$ is known as the successor relation or $1$-step relation of $M$. A sequence of such steps forms a partial computation.

$\textbf{Definition 9 (Partial Computations):}$ Let $M$ be a Turing machine. A *finite partial computation* of $M$ of *length $t$* is a finite sequence of configurations $C_0, \dots, C_t$ such that $C_i \xrightarrow{M} C_{i+1}$ for all $i = 0, \dots, t-1$. An *infinite partial computation* is an infinite sequence $C_0, C_1, \dots$ where this condition holds for all $i \ge 0$. We write $C \xrightarrow[M]{t} C'$ if there is a finite partial computation of length $t$ from $C$ to $C'$, and $C \xrightarrow[M]{*} C'$ if such a computation exists for some length $t$.

### The Full Computation Process

A full computation is a special type of partial computation that begins in a standardized initial configuration and proceeds until it can no longer continue.

$\textbf{Definition 10 (Initial and Halting Configuration, Computation):}$ Let $M = (Q, \Sigma, \Gamma, \Delta, s, F)$ be a Turing machine and let $w$ be a word over $\Sigma$.

* The *initial configuration* of $M$ on input $w$ is $(s, w\square, \square, \dots, \square, 1, \dots, 1)$.
* A *halting configuration* is a configuration $(q, u_1, \dots, u_k, j_1, \dots, j_k)$ to which no instruction in $\Delta$ is applicable, meaning $(q, u_1(j_1), \dots, u_k(j_k))$ is not the condition part of any instruction in $\Delta$.
* An *accepting configuration* is a halting configuration whose state is an accepting state (i.e., is in $F$).
* A *computation* of $M$ is a partial computation that starts with an initial configuration and is either infinite or ends in a halting configuration.
* A *terminating computation* is a finite computation.
* An *accepting computation* is a terminating computation that ends in an accepting configuration.

### The Computation Tree

For a nondeterministic machine, the set of all possible computations on a given input $w$ can be visualized as a tree.

$\textbf{Definition 11 (Computation Tree):}$ The computation tree of a Turing machine $M$ on input $w$ is a finite or infinite rooted tree labelled with configurations of $M$ such that:

* The root is labelled with the initial configuration of $M$ on input $w$.
* The children of an internal node labelled with configuration $C$ are labelled with the successor configurations of $C$. If there are $t > 0$ applicable instructions, the node has $t$ children, one for each resulting successor configuration.
* Leaf nodes are labelled exclusively with halting configurations.

A branch of the computation tree (a path starting at the root) corresponds to a single computation of $M$ on input $w$. Finite branches correspond to terminating computations, while infinite branches correspond to non-terminating computations. Since a TM has a finite number of instructions, its computation tree is always finitely branching. For a deterministic TM, the computation tree is simply a single path, as there is only one possible successor at each step.

### Recognizing and Deciding Languages

Turing machines can be used to define classes of languages based on their computational properties. The most fundamental concepts are acceptance and decision.

#### Acceptance and Recognized Languages

A Turing machine accepts an input word if at least one of its possible computational paths leads to an accepting state.

<!-- $\textbf{Definition 12 (Acceptance and Recognized Language):}$ Let $M$ be a Turing machine with input alphabet $\Sigma$, and let $w$ be a word over $\Sigma$. The Turing machine $M$ *accepts* the word $w$ if one of its computations on input $w$ is accepting. The language *recognized* by $M$ is  $L(M) = \lbrace w \in \Sigma^* : M \text{ accepts } w \rbrace$. -->
<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Recognized Language and Acceptance)</span></p>

Let $M$ be a Turing machine with input alphabet $\Sigma$, and let $w$ be a word over $\Sigma$. The Turing machine $M$ **accepts** the word $w$ if one of its computations on input $w$ is accepting. The language **recognized** by $M$ is  $L(M) = \lbrace w \in \Sigma^* : M \text{ accepts } w \rbrace$.

</div>

For a word $w$ to be in $L(M)$ (be accepted), the computation tree of $M$ on $w$ must have at least one accepting branch. It is possible for other branches to be infinite (non-terminating). The latter possibility is relevant in computability theory. This class of languages is known in computability theory as the recursively enumerable languages.

#### Decidable Languages and Total Turing Machines

A stronger condition is that a machine must halt on all inputs, whether it accepts them or not. This leads to the notion of a decidable language.

<!-- $\textbf{Definition 13 (Total Turing Machine):}$ A Turing machine is *total* if for all inputs all of its computations terminate. -->
<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Total Turing Machine)</span></p>

A Turing machine is **total** if for all inputs all of its computations terminate.

</div>

<!-- $\textbf{Definition 14 (Decidable Language):}$ A language is *decidable* if it is recognized by a total Turing machine (possibly nondeterministic). In this case, we also say that the language is *decided* by the Turing machine. -->
<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Decidable Language)</span></p>

A language is **decidable** if it is *recognized* by a total Turing machine (possibly nondeterministic). In this case, we also say that the language is *decided* by the Turing machine.

</div>

In the part on computation theory, languages that are recognized by some, not necessarily total Turing machine will be called recursively enumerable, and it will be shown that these languages form a strict superclass of the class of decidable languages.

### The Power of Nondeterminism

A central question in complexity theory is whether nondeterministic machines are more powerful than deterministic ones (the famous $P$ vs. $NP$ problem). In the context of computability without resource bounds, the answer is no: every language that can be decided by a nondeterministic TM can also be decided by a deterministic one.

The proof of this relies on the insight that for a total nondeterministic TM, the computation tree for any input must be finite. This can be shown using König's Lemma.

$\textbf{Theorem 15 (König’s Lemma):}$ A finitely branching rooted tree is infinite if and only if it has an infinite branch.

**Proof sketch.**: A tree with an infinite branch is clearly infinite. For the converse, let $T$ be an infinite, finitely branching rooted tree. We can inductively construct an infinite branch $v_0, v_1, \dots$. The key is to maintain the invariant that the subtree rooted at $v_i$ is infinite.

* Let $v_0$ be the root of $T$. The subtree at $v_0$ is $T$ itself, which is infinite.
* Assuming $v_i$ has been defined such that its subtree is infinite, we choose $v_{i+1}$ from its children. Since $v_i$ has only finitely many children, and the subtree at $v_i$ is infinite, at least one of its children must be the root of an infinite subtree. We choose such a child to be $v_{i+1}$. This process can be continued indefinitely, constructing an infinite branch.

Using this lemma, we can prove that nondeterminism adds no power for deciding languages.

$\textbf{Theorem 16:}$ Every decidable language is recognized by a total deterministic Turing machine.

**Proof sketch.**: Let $L$ be a decidable language. By definition, there is a total Turing machine $M$ that recognizes $L$.

1. Since $M$ is total, all of its computations terminate. This means that for any input $w$, all branches in the computation tree of $M$ on $w$ are finite.
2. The computation tree of any TM is finitely branching.
3. By König's Lemma, a finitely branching tree with only finite branches must be a finite tree. Therefore, the computation tree of $M$ on any input $w$ is finite.
4. We can construct a deterministic Turing machine $M'$ that, on input $w$, exhaustively explores the entire computation tree of $M$ on $w$ (e.g., via breadth-first or depth-first search) for an accepting computation by simulating all computations of $M$ on input $w$.
5. Since the tree is finite, this search is guaranteed to terminate. Thus, $M'$ is a total machine.
6. $M'$ is designed to accept if and only if its search finds an accepting configuration of $M$. Therefore, $L(M') = L(M) = L$. This shows that $L$ is recognized by a total deterministic Turing machine.

<div class="accordion">
  <details>
    <summary>Comment on the 4th step (Why the search is deterministic)</summary>
    <p>
      Even though $M$ may be <em>nondeterministic</em>, the <strong>simulation algorithm</strong> that $M'$ runs is <strong>deterministic</strong> because:
    </p>
    <ol>
      <li>
        $M'$ doesn’t “choose” among nondeterministic transitions. Instead, it <strong>enumerates them</strong> in a <em>fixed, deterministic order</em>.
      </li>
      <li>
        For example:
        <ul>
          <li>Each configuration of $M$ can be represented as a finite string (encoding of the state, tape contents, and head position).</li>
          <li>$M'$ maintains a queue or stack of configurations to explore.</li>
          <li>It processes them one by one — e.g., breadth-first or depth-first search.</li>
        </ul>
      </li>
      <li>
        Whenever $M'$ encounters a configuration that has multiple possible next configurations (because $M$ is nondeterministic), it just <strong>adds all of them</strong> to the queue in a <strong>predetermined order</strong> (say, lexicographic order of configurations or transition indices).
      </li>
    </ol>
    <p>
      Thus, although the <em>simulated machine</em> $M$ is nondeterministic, the <em>simulating machine</em> $M'$ follows a fully <strong>deterministic algorithm</strong> for enumerating and checking all branches.
    </p>
  </details>
</div>

<!-- *Comment on the 4th step (Why is the search deterministic)* -->


### Computable Functions

Beyond recognizing languages, Turing machines can also compute functions by transforming an input word into an output word.

#### From Language Recognition to Function Computation

To ensure a unique output for every input, function computation is defined using only total deterministic Turing machines. The output is defined based on the content of a designated tape when the machine halts.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Computable Functions)</span></p>

For a configuration $C = (q, u_1, \dots, u_k, j_1, \dots, j_k)$ of a $k$-tape TM, let $\text{out}(C)$ be the longest word on tape $k$ that starts at the current head position and extends to the right without containing any blank symbols. Formally,

$$
\text{out}(C) = u_k(j_k) u_k(j_k + 1) \cdots u_k(t)
$$

where $t = \max \lbrace i \leq \lvert u_k \rvert : u_k(j_k), \dots, u_k(i) \text{ all differ from } \square \rbrace$. If $u_k(j_k)$ is the blank symbol, $\text{out}(C)$ is the empty word.

The *function $f_M$ computed by a total deterministic Turing machine* $M$ is defined by $f_M(w) = \text{out}(C)$ for the halting configuration $C$ reached by $M$ on input $w$.

For alphabets $\Sigma$ and $\Sigma'$, a function $f : \Sigma^* \to \Sigma'^*$ is **computable** if it is computed by some total deterministic Turing machine.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Alternative definitons of computable function)</span></p>

A subset $S \subseteq \mathbb{N}$ is **computable** if there exists a **total computable function** $f:\mathbb{N}\to\lbrace 0,1\rbrace$ such that

$$
f(x)=
\begin{cases}
1 & \text{if } x\in S,\\
0 & \text{if } x\notin S.
\end{cases}
$$

Equivalently, $$S$ is computable $\iff$ its **indicator (characteristic) function** $\mathbf{1}_S$ is computable.

</div>

### Extending Computability to Other Domains

The concepts of decidable sets and computable functions, defined for words, can be extended to other domains like the natural numbers through the use of representations.

$\textbf{Definition 18 (Decidable sets and computable functions on other domains)}$: A *representation* of a set $A$ is an injective function $\text{repr} : A \to \Sigma^*$ for some alphabet $\Sigma$ such that the set of representations $\lbrace\text{repr}(x) : x \in A\rbrace$ is decidable. With respect to such a representation:

* a subset $X \subseteq A$ is decidable if the set $\lbrace \text{repr}(a) : a \in X\rbrace$ is decidable.
* a function $f : A \to A$ is *computable* if there is a computable function $f_{\text{repr}} : \Sigma^* \to \Sigma^*$ that maps the representation of any $x \in A$ to the representation of $f(x)$. That is, for all $x \in A$, we have $f_{\text{repr}}(\text{repr}(x)) = \text{repr}(f(x))$.

For example, the set of prime numbers is decidable because we can represent the natural number $n$ by the unary word $1^n$, and the language $\lbrace 1^n : n \text{ is prime}\rbrace$ is decidable.

$\textbf{Remark 19: Representations of natural numbers by binary words}$ A natural number n can be represented by its binary expansion $\text{bin}(n)$, such as $\text{bin}(5) = 101$. A bijective representation (one-to-one and onto) can be useful. One such representation maps the natural number n to the word $z_n$, where $z_0, z_1, z_2, \dots$ is the sequence of all binary words sorted in length-lexicographic order $(\lambda, 0, 1, 00, 01, \dots)$. This establishes an order isomorphism between $(\mathbb{N}, \le)$ and $(\lbrace 0,1\rbrace^*, \le_{\text{llex}})$.

$\textbf{Remark 20:}$ Sets, languages, and problems The terminology used often depends on the context of study:

* In computability theory, where resource bounds are not a concern, it is common to use unary representations (e.g., $n$ is represented by $1^n$). Turing machines are viewed as recognizing subsets of natural numbers, leading to terms like "decidable sets of natural numbers."
* In complexity theory, where computations are resource-bounded, inputs are typically binary words representing instances of a computational problem. Here, Turing machines are seen as recognizing languages or problems, both of which refer to arbitrary sets of binary words. (Note that in a general context, a "language" is any set of words over some alphabet).


## Time Complexity

### Deterministic Time

The analysis of algorithms and computational problems often centers on the resources they consume. One of the most critical resources is time. In the context of Turing machines, we formalize this by measuring the number of steps a machine takes to complete its computation.

$\textbf{Definition 21 (Running time):}$ The running time of a deterministic Turing machine $M$ on input $w$ is defined as

$$
\text{time}_M(w) =
\begin{cases}
t & \text{if the computation of } M \text{ on input } w \text{ terminates and has length } t, \\
\uparrow & \text{otherwise},
\end{cases}
$$

where the symbol $\uparrow$ denotes that the function value is undefined (i.e., the machine does not halt).

This definition allows us to quantify the performance of a specific machine on a specific input. To create broader complexity classes, we generalize this notion to a function of the input length, defining what it means for a machine to be bounded by a certain time complexity function.

$\textbf{Definition 22 (Time-bounded deterministic Turing machine):}$ A time bound is a computable function $t : \mathbb{N} \to \mathbb{N}$ with $t(n) \ge n$ for all $n$. For a time bound $t$, a deterministic Turing machine $M$ is $t(n)$-time-bounded or runs in time $t(n)$ if $M$ is total (halts on all inputs) and for almost all inputs $w$, it holds that $\text{time}_M(w) \le t(\lvert w \rvert)$.

The phrase "for almost all inputs" means the condition must hold for all but a finite number of inputs. This provides flexibility, allowing us to disregard a small number of exceptional cases, typically short inputs, which can be handled separately.

$\textbf{Remark 23}$ For a $t(n)$-time-bounded Turing machine, the time bound must be obeyed for almost all inputs, i.e., for all words over the input alphabet except for at most finitely many, say, for all inputs of length larger than or equal to some constant $b$. Note that such a Turing machine can be transformed into another Turing machine such that both Turing machines recognize the same language $L$, and the new machine runs in time at most $t(n) + 2b$ on all inputs. For the proof, call inputs of length at most $b-1$ small and call all other inputs long. It suffices to change the given Turing machine such that initially it scans the first $b$ symbols of its input and stores them in its state such that on every small input a halting configuration is reached and this configuration is accepting if only if the input is in $L$. On a large input, the new Turing machine goes back to the first symbol of the input and then proceeds as usual. Treating small inputs this way is referred to as table *lookup* or *hard-wiring*.

By default, time complexity is a measure of the machine's performance in the most demanding scenario for a given input length. This is known as worst-case time complexity. An alternative approach, average-case time complexity, considers the average running time over all inputs of a certain length. While potentially relevant for specific practical applications, the theory of average-case complexity is more intricate and less developed. Therefore, our focus will remain on worst-case complexity.

Convention 24 In the part about complexity theory, all languages are languages over the binary alphabet unless explicitly something different is stated. Accordingly, we will restrict attention to Turing machines with binary input alphabet.

### Deterministic Time Complexity Classes

Using the concept of time-bounded Turing machines, we can group languages and functions into complexity classes based on the resources required to decide or compute them.

$\textbf{Definition 25 (Deterministic time classes):}$ Let $t$ be a time bound.

The class of languages decidable in deterministic time $t(n)$ is
$\text{DTIME}(t(n)) = \lbrace L \subseteq \lbrace 0, 1\rbrace^* : L = L(M) \text{ for some deterministic } t(n)\text{-time-bounded Turing machine } M \rbrace$.
The class of functions computable in time $t(n)$ is
$\text{FTIME}(t(n)) = \lbrace f : \lbrace 0, 1\rbrace^* \to \lbrace 0, 1\rbrace^* : f = f_M \text{ for some deterministic } t(n)\text{-time-bounded Turing machine } M \rbrace$.
For a set $F$ of time bounds, we define
$$
\text{DTIME}(F) = \bigcup_{t(n) \in F} \text{DTIME}(t(n))
\quad\text{and}\quad
\text{FTIME}(F) = \bigcup_{t(n) \in F} \text{FTIME}(t(n)).
$$

The notation $\text{DTIME}_k(t(n))$ is used to specify the class of languages decidable by a $t(n)$-time-bounded $k$-tape Turing machine. These definitions can also be extended from languages over strings to subsets of natural numbers by using a standard representation, such as binary encoding.

$\textbf{Definition 26}$ Let $t$ be a time bound. A subset $A$ of the natural numbers is decidable in time $t(n)$ if there is a $t(n)$-time-bounded Turing machine that decides the set $\lbrace \text{bin}(n) : n \in A\rbrace$. A function $f : \mathbb{N} \to \mathbb{N}$ is computable in time $t(n)$ if there is a $t(n)$-time-bounded Turing machine that computes a function that maps $\text{bin}(n)$ to $\text{bin}(f(n))$.

This framework allows us to define some of the most fundamental and widely studied complexity classes.

$\textbf{Definition 27 (Some deterministic time classes):}$ Using the function classes:

* $\text{lin} = \lbrace n \mapsto c \cdot n + c : c \in \mathbb{N} \setminus \lbrace 0\rbrace \rbrace$
* $\text{poly} = \lbrace n \mapsto n^c + c : c \in \mathbb{N} \setminus \lbrace 0\rbrace \rbrace$
* $2^{\text{lin}} = \lbrace n \mapsto 2^{c \cdot n + c} : c \in \mathbb{N} \setminus \lbrace 0\rbrace \rbrace$
* $2^{\text{poly}} = \lbrace n \mapsto 2^{n^c + c} : c \in \mathbb{N} \setminus \lbrace 0\rbrace \rbrace$

we define the complexity classes:

* $\text{LIN} = \text{DTIME}(\text{lin})$
* $\text{P} = \text{DTIME}(\text{poly})$
* $\text{E} = \text{DTIME}(2^{\text{lin}})$
* $\text{EXP} = \text{DTIME}(2^{\text{poly}})$

We refer, for instance, to $P$ as the class of problems decidable in deterministic polynomial time and $E$ as the class of problems decidable in deterministic linear exponential time.

### Properties of Deterministic Time Classes

The definitions of complexity classes depend on a specific model of computation (the Turing machine) and its parameters (number of tapes, alphabet size). The following theorems demonstrate the robustness of these classes, showing that they are invariant under certain changes to the machine model.

### The Linear Speedup Theorem

The first theorem shows that constant factors in the running time do not change the fundamental complexity of a problem. We can always build a faster machine that solves the same problem, effectively making any constant-factor speedup possible.

$\textbf{Theorem 28 (Linear speedup):}$ Linear speedup Let $t$ be a time bound, and let $\alpha > 0$ be a real number, and let $k \ge 2$. Then it holds that
$$
\text{DTIME}_k(t(n)) \subseteq \text{DTIME}_k(\alpha \cdot t(n) + n)
\tag{2.1}
$$
and hence, in particular,
$$
\text{DTIME}(t(n)) \subseteq \text{DTIME}(\alpha \cdot t(n) + n).
\tag{2.2}
$$
Furthermore, every function $f$ in $\text{FTIME}(t(n))$ is in 
$$
\text{FTIME}(\alpha \cdot t(n) + n + \lvert f(n) \rvert).
\tag{2.3}
$$

**Proof.**: We will demonstrate the inclusion (2.1), from which (2.2) immediately follows. Given a language $L \in \text{DTIME}(t(n))$, there exists a $t(n)$-time-bounded Turing machine with some number of tapes, say $k \ge 2$, that recognizes $L$. By showing $L \in \text{DTIME}_k(\alpha \cdot t(n) + n)$, it follows that $L \in \text{DTIME}(\alpha \cdot t(n) + n)$.

To prove (2.1), let $L$ be a language recognized by a $t(n)$-time-bounded $k$-tape Turing machine $M$, where $k \ge 2$. We construct a new $k$-tape machine $M'$ that simulates $M$ (recognize the same language $L$), but runs faster. The core idea is to have $M'$ process larger chunks of data at each step.

Each symbol in the tape alphabet of $M'$ will represent a block of $d$ symbols from the tape alphabet of $M$, where $d$ is a constant we will choose later.

##### The computation of $M'$ proceeds in two phases:

1. Initialization Phase:
  * $M'$ translates its input from $M$'s alphabet $\Sigma$ to its own compressed alphabet $\Sigma'$.
  * It writes this new, compressed input onto its second tape.
  * Simultaneously, it overwrites the original input on the first tape with blank symbols.
  * Finally, it positions the head of the second tape at the beginning of the compressed input.
  * This phase takes $\underbrace{n}_{\text{read →}} + \underbrace{\lceil n/d \rceil}_{\text{go back ←}} + 2$ steps. After this, $M'$ will simulate $M$, treating its second tape as the primary input tape.
2. Simulation Phase:
  * $M'$ simulates $d$ steps of $M$ using just $7$ of its own steps. To do this, $M'$ uses its finite control (its state) to store key information about $M$'s current configuration: $M$'s current state, and for each tape, the head position of $M$ within the larger macro-symbol that $M'$ is currently scanning.
  * In each simulation cycle, $M'$ needs to know the contents of the cells $M$'s heads are on, as well as the adjacent cells, to determine $M$'s next $d$ moves. A "relevant" cell for $M'$ is one that is currently scanned or is immediately to its left or right.
  * $M'$ first reads all symbols on its relevant cells by moving one step left and two steps right ($3$ steps total), **storing this information in its state**.
  * With the complete local information ($M$'s state, head positions, and relevant cell contents), $M'$ can determine the outcome of the next $d$ steps of $M$. Since $d$ is a constant, all possible outcomes can be pre-computed and stored in $M'$'s transition function (a finite table lookup). This information includes $M$'s new state, the new content of the relevant cells, and the new head positions.
  * $M'$ then uses another $4$ steps to update its tape cells and move its heads to the new positions corresponding to $M'$s configuration after $d$ steps.

Timing Analysis: The initialization takes $n + \lceil n/d \rceil + 2$ steps. The simulation of $M$ takes at most $t(n)$ steps. Since $M'$ simulates $d$ steps of $M$ in $7$ steps, this phase requires at most $7 \lceil t(n)/d \rceil + 7$ steps.

The total time for $M'$ is at most $n + \frac{n}{d} + 2 + \frac{7 t(n)}{d} + 7$. Since the time bound $t(n) \ge n$, for almost all $n$, this is bounded by $\frac{8 t(n)}{d} + n + 9 \le \frac{9 t(n)}{d} + n$. We want this to be less than $\alpha \, t(n) + n$. We can achieve this by choosing a large enough constant $d$ such that $\frac{9}{d} < \alpha$. With such a $d$, the running time of $M'$ is bounded by $\alpha \, t(n) + n$, proving that $L \in \text{DTIME}_k(\alpha \cdot t(n) + n)$.

<div class="accordion">
  <details>
    <summary>Comment on the simulation phase (Why to scan left/right cells and Why is M′ deterministic)</summary>

    <p>
      We compress M’s tape into blocks of <code>d</code> symbols, so M′ sees one macro-cell per block.
      In the next <code>d</code> real steps of M:
    </p>
    <ul>
      <li>Each head can move at most <code>d</code> squares.</li>
      <li>Starting anywhere inside the current block, after <code>d</code> moves the head can only be in the same block or one of its immediate neighbors (it can’t skip two blocks).</li>
      <li>Moreover, M can read and write anywhere it visits during those <code>d</code> moves. That set of positions is contained within the current block plus at most one block to the left and one to the right.</li>
    </ul>

    <p>M is deterministic. So if we know:</p>
    <ul>
      <li>M’s current state,</li>
      <li>for each tape: the head’s offset inside its current block,</li>
      <li>for each tape: the contents of the three relevant macro-cells (left, current, right),</li>
    </ul>

    <p>
      then the next <code>d</code> moves of M are completely determined by its transition function. There are only finitely many possibilities for that local information (finite state set, finite alphabet, fixed <code>d</code>, finite offsets <code>0,…,d−1</code>), so M′ can precompute a macro-transition table:
    </p>

    <pre><code>(state, offsets, 3-block contents per tape)
→ (new state, updated 3-block contents, new offsets, which block is current)</code></pre>

    <p>
      During simulation, M′ just looks up the outcome and then writes the updated macro-cells and moves its heads accordingly — this is why simulating <code>d</code> steps costs <code>O(1)</code> steps for M′.
    </p>
  </details>
</div>


#### Alphabet Reduction

The size of the tape alphabet is another parameter of the Turing machine model. The following theorem shows that any computation can be performed by a Turing machine with a minimal binary alphabet, at the cost of only a constant factor slowdown.

$\textbf{Theorem 29:}$ Alphabet reduction Let $t$ be a time bound, let $k \ge 2$, and let $L$ be a language in $\text{DTIME}_k(t(n))$. There exists a deterministic $k$-tape Turing machine $M$ with tape alphabet $\lbrace0, 1, \square \rbrace$ that recognizes $L$ and runs in time $d \cdot t(n)$ for some constant $d$.

**Proof.**: Let $L$ be recognized by a deterministic $t(n)$-time-bounded $k$-tape machine $M$ with a tape alphabet $\lbrace a_1, \dots, a_r \rbrace$ of size $r$. We construct a machine $M'$ with a binary tape alphabet that simulates $M$.

The core idea is to encode each symbol $a_j$ from $M'$s alphabet as a unique binary string. We can use the encoding where $a_j$ is represented by the string $1^j0^{r-j}$.

1. Initialization Phase: On input $w, M'$ first translates $w$ into its binary-encoded form. Using a second tape, this can be done in $2r \lvert w \rvert$ steps: each symbol of $w$ is encoded with the sequence of the length $r$.
2. Simulation Phase: $M'$ simulates the computation of $M$ step-by-step. To simulate a single step of $M$, $M'$ must:
  * Read: Identify the symbol under each of $M$'s heads. This requires $M'$ to read the corresponding block of $r$ binary symbols on each of its tapes. While reading a block, $M'$ uses its state to remember its position within the block and the binary pattern it has seen so far.
  * Write/Move: Based on $M$'s transition function, $M'$ overwrites the binary blocks with the new encoded symbols and moves its heads accordingly. This involves moving across the $r$ cells of the block.

Simulating one step of $M$ requires reading and potentially writing $k$ blocks of $r$ symbols each. This takes a constant number of steps proportional to $r$. Thus, a single step of $M$ can be simulated in $d \cdot r$ steps of $M'$ for some constant $d$. The total running time of $M'$ is therefore bounded by a constant multiple of $t(n)$, as required.

#### Tape Reduction

A more significant change to the machine model is reducing the number of tapes. A multi-tape Turing machine can be simulated by a single-tape machine, but this incurs a more substantial, quadratic, increase in computation time.

$\textbf{Theorem 30:}$ Tape reduction Let $t$ be a time bound and let $L$ be a language in $\text{DTIME}_k(t(n))$, i.e., $L$ is recognized by a deterministic total $k$-tape Turing machine $M$ that runs in time $t(n)$. Then there exists a deterministic $1$-tape Turing machine $M'$ that recognizes $L$ and runs in time $d \cdot t^2(n)$ for some constant $d$.

**Proof.**: We construct a single-tape Turing machine $M'$ that simulates a $k$-tape machine $M$. The single tape of $M'$ is structured to represent all $k$ tapes of $M$ simultaneously using a system of "tracks".

Conceptually, the tape of $M'$ is partitioned into $2k$ tracks. Each cell on the tape of $M'$ contains a $2k$-tuple. For each of $M$'s tapes (say, tape $i$), two tracks on $M'$ are used: one to store the content of tape $i$ and another to mark the position of tape $i$'s head. The tape alphabet $\Gamma'$ of $M'$ consists of tuples containing $k$ symbols from $M$'s alphabet $\Gamma$ and $k$ symbols from $\lbrace \square, * \rbrace $, where $*$ is used as the head marker.

#### The simulation proceeds as follows:

1. Initialization: On input $w$, $M'$ first initializes its tape to represent the initial configuration of $M$. This involves writing $w$ onto the first track and placing head markers $(*)$ at the beginning of all head-position tracks.
2. Simulation of a Single Step: To simulate one step of $M$, $M'$ performs a subroutine:
  * (i) Scan and Read: $M'$ sweeps its single head across the entire active portion of its tape to find the $k$ head markers $(*)$. As it passes each marker, it notes the corresponding tape symbol from the content track and stores all $k$ symbols in its state.
  * (ii) Update and Write: After collecting all necessary information, $M'$ knows what transition $M$ will make. It then performs a second sweep across the tape to update the tape contents and move the head markers $(*)$ one position left or right, as dictated by $M$'s transition function. It also updates the state of $M$, which is stored in its own finite control.

Timing Analysis: The machine $M$ runs for at most $t(n)$ steps. This means that on any of its tapes, it can access at most $t(n)$ cells to the left and $t(n)$ cells to the right of the starting position. Therefore, the active portion of $M'$'s tape has a length of at most $2t(n)+1$.

Each simulation of a single step of $M$ requires $M'$ to make two full passes over this active portion. The time required for this is proportional to the length of the active tape, which is $O(t(n))$. So, simulating one step of $M$ takes at most $d \cdot t(n)$ steps for $M'$, for some constant $d$.

Since $M$ makes at most $t(n)$ steps in total, the entire simulation on $M'$ requires at most $t(n) \times (d \cdot t(n)) = d \cdot t^2(n)$ steps. Thus, $M'$ runs in time $O(t^2(n))$.

#### On the Requirement for Time Bounds

$\textbf{Remark 31: By definition, a time bound $t$ must satisfy $t(n)$}$ $\ge n$. The latter requirement indeed makes sense because for other $t$, a $t(n)$-time bounded Turing machine is restricted to read a prefix of its input of constant length, hence cannot recognize any interesting language. For a proof, let $t$ be a function where $t(n) < n$ for infinitely many $n$. Let $M$ be a Turing machine that is $t(n)$-bounded in the sense that for almost all $n$ or, equivalently, for some $n_0$ and all $n \ge n_0$, on inputs of length $n$, the running time of $M$ is at most $t(n)$. Pick $n_1 \ge n_0$ such that $t(n_1) < n_1$. Then on all inputs of length greater than $n_1$, $M$ reads at most the first $n_1$ symbols of its input. For a proof by contradiction, assume there is a word $w$ of length $\lvert w \rvert > n_1$ such that $M$ on input $w$ reads at least the first $n_1 + 1$ symbols of its input. Let $u$ be the prefix of $w$ of length $n_1$. Then $M$ scans on input $u$ at least $n_1 + 1$ cells of its input tape, hence makes at least $n_1 > t(n_1) = t(\lvert u \rvert)$ steps, which contradicts the choice of $n_0$ and $n_1$.

### Nondeterministic Time Complexity

### Nondeterministic Time-Bounded Turing Machines

The concept of time complexity can be extended from deterministic to nondeterministic Turing machines. A nondeterministic Turing machine may have multiple possible transitions from a given configuration, leading to a tree of possible computations. A word $w$ is accepted if at least one of these computational paths leads to an accepting state.

When analyzing the time complexity of a nondeterministic machine, we consider the length of the longest possible computation path.

$\textbf{Definition 32 (Nondeterministic time classes):}$ Let $t$ be a time bound. A Turing machine $M$ is $t(n)$-time bounded if $M$ is total (all computation paths halt) and for almost all inputs $w$, **all computations** of $M$ have length at most $t(\lvert w \rvert)$. The class of languages decidable in nondeterministic time $t(n)$ is:  $\text{NTIME}(t(n)) = \lbrace L \subseteq \lbrace 0, 1\rbrace^* : L = L(M) \text{ for a } t(n)\text{-time bounded Turing machine } M \rbrace$.

$\textbf{Remark 34}$ In the literature, one also finds a variant of the notion $t(n)$-time bounded Turing machine where the length bound $t(\lvert w \rvert)$ is required only for accepting computations, while nonaccepting computations may have arbitrary finite or even infinite length. For time-constructible time bounds $t$ with $t(n) \ge 2n$, the alternative definition is essentially equivalent to the one presented here. Here a function $t$ is time-constructible if the function $1^n \mapsto 1^{t(n)}$ can be computed in time $t(n)$. For such $t$, it is possible to equip a Turing machine that is $t(n)$-time bounded in the sense of the variant with a timing mechanism or clock that enforces termination after $t(n)$ steps on all computations such that the recognized language remains the same and the clocked Turing machine is $t(n)$-time bounded according to Definition 32.

### Nondeterministic Time Complexity Classes

Similar to the deterministic case, we can define major complexity classes based on nondeterministic time bounds.

$\textbf{Definition 33: Examples of nondeterministic time classes We define the complexity classes:}$

* $\text{NP}$ = $\text{NTIME}$($\text{poly}$)
* $\text{NE} = \text{NTIME}(2^{\text{lin}}) = \bigcup_{c \in \mathbb{N}} \text{NTIME}(2^{c n + c})$
* $\text{NEXP} = \text{NTIME}(2^{\text{poly}}) = \bigcup_{c \in \mathbb{N}} \text{NTIME}(2^{n^c + c})$

We refer, for example, to $NP$ as the class of problems decidable in nondeterministic polynomial time and to $NE$ as the class of problems decidable in nondeterministic linear exponential time.

### The theory of NP-completeness (SubMileStone)

### Reducibility and $NP$-Completeness

### Polynomial-Time Many-One Reducibility

To compare the relative difficulty of problems, we use the concept of reduction. A reduction is a way to solve one problem using an algorithm for another. If problem $A$ can be reduced to problem $B$, it means that $A$ is "no harder than" $B$.

$\textbf{Definition 35}$ A language $A$ is many-one reducible in polynomial time to a language $B$, denoted $A \le_p^m B$, if there exists a function $g \in \text{FP}$ (i.e., computable in deterministic polynomial time) such that for all $x \in \lbrace 0, 1\rbrace ^*$ it holds that $x \in A \iff g(x) \in B$.

This type of reduction, often called a $p$-$m$-reduction or Karp reduction, acts as a translation. To solve an instance $x$ of problem $A$, we can first compute $g(x)$ in polynomial time and then use a decider for $B$ to determine if $g(x) \in B$. If $B$ can be decided in polynomial time, then so can $A$. This relationship transfers difficulty upward: if $A$ is known to be hard, then $B$ must also be hard. Conversely, it transfers simplicity downward: if $B$ is easy, then $A$ must also be easy.

### Downward and Upward Closure

Complexity classes often exhibit closure properties with respect to reductions.

$\textbf{Definition 36 (Closed Downward, Closed Upward):}$ A set of languages is closed downward under $p$-$m$-reducibility if for every language $B$ in the class, every language $A \le_p^m B$ is in the class, too. A set of languages is closed upward under $p$-$m$-reducibility if for every language $A$ in the class, every language $B$ with $A \le_p^m B$ is in the class, too.

Many important complexity classes, such as $P$ and $NP$, are closed downward. This means that if a problem $B$ is in the class, any problem that is polynomially reducible to $B$ is also in that class.

$\textbf{Proposition 37 (Downward closure of $P$ and $NP$ under $p$-$m$-reducibility):}$ Let $A$ and $B$ be languages such that $A \le_p^m B$. Then it holds that:

* $B \in \text{P}$ implies $A \in \text{P}$,
* $B \in \text{NP}$ implies $A \in \text{NP}$.

These implications are logically equivalent to their contrapositions, i.e., to:

* $A \notin \text{P}$ implies $B \notin \text{P}$,
* $A \notin \text{NP}$ implies $B \notin \text{NP}$.

**Proof.**: We prove the first implication, $B \in \text{P} \implies A \in \text{P}$. The proof for $NP$ is very similar. Let $A \le_p^m B$ via a function $g \in \text{FP}$. Let $M_g$ be a deterministic Turing machine that computes $g$ in time $p_g(n)$ for some polynomial $p_g$. Since $B \in \text{P}$, there is a deterministic Turing machine $M_B$ that decides $B$ in time $p_B(n)$ for some polynomial $p_B$.

We can construct a deterministic Turing machine $M_A$ to decide $A$ as follows:

1. On input $x$, $M_A$ first runs $M_g$ to compute $g(x)$. This takes $p_g(\lvert x \rvert)$ time. The length of the output $g(x)$ is at most $p_g(\lvert x \rvert)$.
2. Next, $M_A$ simulates $M_B$ on the input $g(x)$. This takes $p_B(\lvert g(x) \rvert) \le p_B(p_g(\lvert x \rvert))$ time.
3. $M_A$ accepts if and only if $M_B$ accepts $g(x)$.

The total running time for $M_A$ is bounded by $p_g(\lvert x \rvert) + p_B(p_g(\lvert x \rvert))$. Since the composition of two polynomials is a polynomial, $M_A$ is a polynomially time-bounded machine. Therefore, $A \in \text{P}$.

### Properties of Reducibility

The $\le_p^m$ relation behaves like a partial order on languages.

$\textbf{Proposition 38}$ The relation $\le_p^m$ is reflexive and transitive.

**Proof.**:

* Reflexivity: For any language $A$, $A \le_p^m A$ via the identity function $g(x) = x$, which is computable in polynomial (linear) time.
* Transitivity: Suppose $A \le_p^m B$ via function $g_B$ and $B \le_p^m C$ via function $g_C$. Let $g_B$ be computable in time $p_B(n)$ and $g_C$ be computable in time $p_C(n)$. The reduction from $A$ to $C$ is the composition $g_C \circ g_B$. We have $x \in A \iff g_B(x) \in B \iff g_C(g_B(x)) \in C$. The time to compute $g_C(g_B(x))$ is the time to compute $g_B(x)$ (which is $p_B(n)$) plus the time to compute $g_C$ on the result (which is $p_C(\lvert g_B(x) \rvert) \le p_C(p_B(n))$). The total time is $p_B(n) + p_C(p_B(n))$, which is a polynomial. Thus, $A \le_p^m C$.

### $NP$-Hardness and $NP$-Completeness

Reducibility allows us to identify the "hardest" problems within a complexity class. For the class $NP$, these are the $NP$-complete problems.

$\textbf{Definition 39 (NP-complete languages):}$ 
* A language $B$ is $NP$-hard if all languages in $NP$ are $p$-$m$-reducible to 
* $B$ A language is $NP$-complete if it is $NP$-hard and belongs to $NP$.

An $NP$-complete problem is thus a problem that is both in $NP$ and is at least as hard as every other problem in $NP$.

There are further, provably different notions of $NP$-hardness and $NP$-completeness, which are defined in terms of reducibilities different from $p$-$m$-reducibility. For example, $A$ is Turing reducible to $B$ in polynomial time if $A$ can be decided by a polynomially time-bounded Turing machine that for arbitrary words $y$ can access the information whether $y$ is in $B$ by writing $y$ onto a special oracle tape. Accordingly, the notions just defined may be referred to more precisely as $NP$-hardness and $NP$-completeness with respect to $p$-$m$-reducibility.

### The $P$ versus $NP$ Problem

### Relationship Between $P$, $NP$, and $NP$-Complete Problems

The existence of $NP$-complete problems provides a powerful way to frame one of the most significant open questions in computer science: whether $P$ equals $NP$.

$\textbf{Theorem 40 The following statements are equivalent.}$ 
* (i) $P = NP$. 
* (ii) All $NP$-complete sets are in $P$. 
* (iii) There is an $NP$-complete set in $P$.

**Proof.**:

* (i) $\implies$ (ii): If $P$ = $NP$, then every language in $NP$ is also in $P$. By definition, $NP$-complete languages are in $NP$, so they must also be in $P$.
* (ii) $\implies$ (iii): This implication is trivial, assuming that $NP$-complete languages exist (which will be proven later).
* (iii) $\implies$ (i): We need to show that if some $NP$-complete language $L$ is in $P$, then $NP \subseteq P$. (The inclusion $P \subseteq NP$ is true by definition). Let $L$ be an NP-complete language such that $L \in P$. Since $L$ is $NP$-hard, every language $A \in NP$ is $p$-$m$-reducible to $L$ ($A \le_p^m L$). From Proposition 37, we know that if $A \le_p^m L$ and $L \in P$, then $A \in P$. Since this holds for any arbitrary language $A \in NP$, it follows that $NP \subseteq P$. Therefore, $P = NP$.

The Significance of the $P$ vs. $NP$ Problem

$\textbf{Remark 41}$ The class $P$ is a subclass of $NP$, but it is not known whether this inclusion is proper. Whether the two classes coincide is referred to as $P$ versus $NP$ problem. By Theorem 40, the two classes coincide if and only if some $NP$-complete language is in $P$. Common ways to express that a language is in $P$ are to say that the corresponding problem is feasible, can be solved efficiently, or has an efficient algorithm. Note that all three notions are also used in other contexts with different meanings. There are hundreds of practically relevant problems that are $NP$-complete, and for none of them an efficient algorithms is known. In fact, not even algorithms are known that are significantly faster than an exhaustive search over all possible solutions.

### A Criterion for Proving $NP$-Completeness

The property of transitivity provides a practical method for proving that new problems are $NP$-complete. Once we have one known $NP$-complete problem, we can use it as a starting point.

$\textbf{Remark 42}$ A language $B$ is $NP$-hard if and only if there is an $NP$-hard language $A$ with $A \le_p^m B$. In the special case of a language $B$ in $NP$, this implies that $B$ is $NP$-complete if and only if there is an $NP$-complete language $A$ such that $A \le_p^m B$. Both statements follow easily from the fact that $p$-$m$-reducibility is reflexive and transitive.

To establish that the first $NP$-complete problem exists, it will be shown that the satisfiability problem for propositional formulas ($\text{SAT}$) is $NP$-complete. Subsequently, the $NP$-completeness of other problems can be established by demonstrating a polynomial-time reduction from $\text{SAT}$ (or any other known $NP$-complete problem) to the new problem in question.


### Reducibility and Completeness

### Polynomial-Time Many-One Reducibility Degrees

The concept of reducibility allows us to compare the relative difficulty of computational problems. When two problems can be reduced to each other, they are considered equivalent in terms of their computational complexity. This equivalence can be formalized into distinct classes.

**p-m-equivalence**: Two languages $A$ and $B$ are $p$-$m$-equivalent, denoted $A \equiv_p^m B$, if $A \le_p^m B$ and $B \le_p^m A$.

**p-m-degree**: The $p$-$m$-degree of a language $A$ is the class of all languages that are $p$-$m$-equivalent to it:  $\text{deg}_p^m(A) = \lbrace B \subseteq \lbrace 0, 1 \rbrace ^* : A \equiv_p^m B\rbrace$ 

$\textbf{Remark 44:}$ The $p$-$m$-equivalence is an equivalence relation. Consequently, the $p$-$m$-degrees are the equivalence classes of this relation. This means that the $p$-$m$-degrees form a partition of the class of all binary languages. Recall that for any equivalence relation on a set $A$, its equivalence classes are mutually disjoint, and their union is equal to $A$. The relation of $p$-$m$-reducibility ($\le_p^m$) induces a nonstrict partial order on these $p$-$m$-degrees, where one degree is considered "smaller" than another if a language from the first degree can be $p$-$m$-reduced to a language in the second.

### Degrees within $P$ and $NP$

Within the class $NP$, the structure of $p$-$m$-degrees reveals important insights into the relationship between complexity classes.

$\textbf{Remark 45:}$ The set of all $NP$-complete languages forms a single $p$-$m$-degree. Similarly, the class $P$, excluding the trivial languages $\emptyset$ and $\lbrace 0, 1\rbrace^*$, also forms a $p$-$m$-degree. It has been shown that if $P \ne NP$, then the class $NP$ contains infinitely many distinct $p$-$m$-degrees. In fact, under this assumption, every finite partial order can be embedded into the structure of these degrees.

### An Anomaly of $P$-$M$-Reducibility

The formal definition of $p$-$m$-reducibility leads to some unusual properties, particularly concerning languages in $P$ and the two trivial languages, the empty set $\emptyset$ and the set of all strings $\lbrace 0, 1\rbrace^*$.

$\textbf{Remark 46:}$ Since the class $P$ is downward closed under $p$-$m$-reducibility, no language outside of $P$ can be $p$-$m$-reduced to a language in $P$. Conversely, every language in $P$ is $p$-$m$-reducible to any other language, with the exception of $\emptyset$ and $\lbrace 0, 1\rbrace^*$.

To prove this, let $A$ be a language in $P$ and let $B$ be any language such that $B \ne \emptyset$ and $B \ne \lbrace0, 1\rbrace^*$. Since $B$ is not trivial, we can select an element $x_1 \in B$ and an element $x_0 \notin B$. The $p$-$m$-reduction from $A$ to $B$ can be defined by the function that maps every $x \in A$ to $x_1$ and every $x \notin A$ to $x_0$. This function is computable in polynomial time because $A$ is in $P$.

$\textbf{Remark 47:}$ By definition, only the empty set can be $p$-$m$-reduced to the empty set. A similar restriction applies to the language $\lbrace 0,1 \rbrace^{*}$. As noted in Remark 46, every language in $P$ is reducible to all other languages. The special behavior of $\emptyset$ and $\lbrace 0,1 \rbrace^{*}$ is often considered an anomaly of the definition. For this reason, alternative definitions of $p$-$m$-reducibility are sometimes used to avoid these edge cases.

### The Satisfiability Problem ($\text{SAT}$)

To prove that a language is $NP$-hard, the standard method is to show that a known $NP$-hard language can be $p$-$m$-reduced to it. However, this technique requires a starting point: a first problem that is proven to be $NP$-hard from fundamental principles. The canonical first $NP$-hard problem is the Boolean Satisfiability Problem, or $\text{SAT}$.

### Propositional Formulas

We begin by formally defining the components of propositional logic.

$\textbf{Definition 48 (Propositional Formula):}$ Let $\Lambda = \lbrace \neg, \land, (, )\rbrace$ be a set of logical symbols and let Var be a countable set of variables disjoint from $\Lambda$. The set of propositional formulas over Var is a set of words over the alphabet $\text{Var} \cup \Lambda$, defined inductively:

* Base case: Every element of $\text{Var}$ is a propositional formula.
* Inductive step: If $\phi$ and $\phi'$ are propositional formulas, then so are $\neg \phi$ and $(\phi \land \phi')$.

In this context, elements of Var are propositional variables, while $\neg$ represents logical negation (NOT) and $\land$ represents logical conjunction (AND). Other logical operators can be introduced as shorthand. For instance, disjunction (OR), denoted by $\lor$, can be expressed using De Morgan's laws: $X \lor Y$ is shorthand for $\neg(\neg X \land \neg Y)$. Constants for true ($1$) and false ($0$) can also be included. For readability, standard rules for operator precedence are often used, allowing parentheses to be omitted.

The truth value of a formula, denoted $\text{val}(\phi)$, is either true ($1$) or false ($0$). This value is determined relative to an assignment $b : \text{Var} \to \lbrace 0, 1\rbrace$ that assigns a truth value to each variable. The truth value of a complex formula is defined inductively:

* Base case: For a variable $X \in \text{Var}$, $\text{val}(X) = b(X)$.
* Inductive step: For formulas $\phi$ and $\phi'$:
  * $\text{val}(\neg \phi) = 1 - \text{val}(\phi)$
  * $\text{val}(\phi \land \phi') = \min\lbrace \text{val}(\phi), \text{val}(\phi')\rbrace$

The set of variables that appear in a formula $\phi$ is denoted by $\text{var}(\phi)$. The truth value of $\phi$ depends only on the assignment of values to the variables in $\text{var}(\phi)$. The set $\text{var}(\phi)$ is defined inductively as:

* $\text{var}(X) = \lbrace X\rbrace$ for any $X \in \text{Var}$.
* $\text{var}(\neg \phi) = \text{var}(\phi)$.
* $\text{var}(\phi \land \phi') = \text{var}(\phi) \cup \text{var}(\phi')$.

### Satisfiability and Normal Forms

The central question of the satisfiability problem is whether a formula can be true.

$\textbf{Definition 49:}$

* A propositional formula is satisfiable if there exists an assignment of its variables that makes the formula true.
* A literal is a propositional variable or a negated propositional variable (e.g., $X$ or $\neg X$).
* A clause (specifically, a disjunctive clause) is a disjunction of literals (e.g., $X \lor \neg Y \lor Z$).
* A propositional formula is in conjunctive normal form (CNF) if it is a nonempty conjunction of clauses.
* A formula is in $k$-conjunctive normal form ($k$-CNF) if it is in CNF and each clause contains at most $k$ literals.

For example, the formula $(\neg X \lor Y \lor Z) \land (X \lor Z) \land (X \lor \neg Y \lor \neg Z)$ is in $3$-CNF.

This leads to the formal definition of the satisfiability problems as languages.

$\textbf{Definition 50: The Satisfiability Problem:}$ Assuming an appropriate representation of propositional formulas as binary strings, we define the following languages for any $k > 0$:

$\text{SAT} = \lbrace \, \phi \mid \phi \text{ is a satisfiable propositional formula in CNF} \,\rbrace$

$k\text{-SAT} = \lbrace \, \phi \mid \phi \text{ is a satisfiable propositional formula in k-CNF} \,\rbrace$

### $NP$-Completeness of $\text{SAT}$ and $\text{k-SAT}$

The languages $\text{SAT}$ and $\text{k-SAT}$ (for $k \ge 3$) are cornerstone examples of $NP$-complete problems.

#### $\text{SAT}$ is in $NP$

First, we must establish that these problems belong to the class $NP$.

$\textbf{Remark 51:}$ The language $\text{SAT}$ and the languages $\text{k-SAT}$ for $k > 0$ are members of the class $NP$.

A proof for this involves a nondeterministic Turing machine (NTM) that operates in polynomial time. Given an input string, the NTM first deterministically checks if it represents a valid formula $\phi$ in CNF (or $k$-CNF). If it is a valid formula, the machine proceeds to:

1. Guess: Nondeterministically guess an assignment for all variables occurring in $\phi$. This involves creating branches for each variable, assigning it $0$ in one branch and $1$ in another, effectively exploring all possible assignments.
2. Check: Deterministically evaluate the formula $\phi$ with the guessed assignment to see if it results in true.

If any assignment makes the formula true, at least one computational path of the NTM will accept. The length of an assignment is linear in the size of the formula, so both the guessing and checking phases can be completed in polynomial time. Therefore, $\text{SAT}$ and $\text{k-SAT}$ are in $NP$.

### Cook's Theorem: $\text{SAT}$ is $NP$-Complete

The proof that $\text{SAT}$ is not just in $NP$ but is $NP$-hard is a landmark result in computer science.

$\textbf{Theorem 52 (Cook’s Theorem)}$ : The language $\text{SAT}$ is $NP$-complete.

**Proof.**: We have already established in Remark 51 that $\text{SAT}$ is in $NP$. To prove that it is $NP$-complete, we must show it is $NP$-hard. This requires showing that every language $A \in \text{NP}$ is $p$-$m$-reducible to $\text{SAT}$.

Let $A$ be any language in $NP$. We will construct a function $g : x \mapsto \phi_x$, computable in polynomial time (i.e., $g \in \text{FP}$), that maps any binary word $x$ to a propositional formula $\phi_x$ in CNF. This construction will ensure that $x \in A$ if and only if $\phi_x$ is satisfiable.

Since $A \in \text{NP}$, there exists a nondeterministic Turing machine (NTM) $M = (Q, \Sigma, \Gamma, \Delta, s, F)$ and a polynomial $p$ such that $M$ recognizes $A$ within a time bound of $p(n)$, where $n$ is the input length. For simplicity, we can assume $M$ has only a single tape. Any multi-tape NTM can be converted to an equivalent single-tape NTM with only a polynomial increase in runtime, similar to the tape reduction construction in Theorem 30.

Given an input $x$ of length $n$, our goal is to construct a formula $\phi_x$ that is satisfiable if and only if $M$ accepts $x$. The formula will essentially describe the behavior of $M$ on input $x$. A satisfying assignment for $\phi_x$ will correspond directly to an accepting computation of $M$ on $x$.

An accepting computation is a sequence of configurations, starting with the initial configuration for input $x$ and ending in an accepting configuration. The length of this sequence is at most $p(n) + 1$. We will formalize a sequence of exactly $p(n)+1$ configurations, repeating the final configuration if the computation halts early. The formula $\phi_x$ will encode the rules of $M$'s transition relation, $\Delta$, to ensure that each configuration in the sequence legally follows from the previous one.

#### The overall structure of the proof is as follows:

1. If $M$ accepts $x$, there exists a valid sequence of configurations representing an accepting computation. This sequence will directly provide a satisfying assignment for the variables in $\phi_x$.
2. Conversely, if $\phi_x$ is satisfiable, any satisfying assignment can be used to decode a valid, accepting computation of M on input $x$.
3. The formula $\phi_x$ will be a conjunction of subformulas, each of which can be expressed in CNF. Therefore, $\phi_x$ itself can be written in CNF.
4. The construction of $\phi_x$ can be carried out in polynomial time with respect to $n = \lvert x \rvert$, because the number and size of the subformulas are polynomially bounded in $n$.

Together, these points establish that $A \le_p^m \text{SAT}$, proving that $\text{SAT}$ is $NP$-hard.

#### Technical Note on Logical Notation

In the construction, we will frequently express clauses as implications. The following logical equivalences are useful:

* An implication $\alpha \to \beta$ is equivalent to $\neg \alpha \lor \beta$.
* The negation of a conjunction $\neg(\gamma_1 \land \dots \land \gamma_r)$ is equivalent to a disjunction of negations $\neg \gamma_1 \lor \dots \lor \neg \gamma_r$.
* A clause of the form $L_1 \land \dots \land L_r \to L_{r+1}$ (where each $L_i$ is a literal) is equivalent to the disjunctive clause $\neg L_1 \lor \dots \lor \neg L_r \lor L_{r+1}$.

#### Formalization of the Turing Machine Computation

A computation of $M$ on input $x$ is a sequence of configurations over $p(n)+1$ time steps. Each configuration is defined by the tape content, head position, and current state.

#### We define index sets to refer to time steps, tape cells, states, and nondeterministic choices:

* Time steps: $I = \lbrace 0,1, \dots, p(n) \rbrace$
* Tape cells: $J = \lbrace -p(n)+1, \dots, p(n)+1 \rbrace$ (covers all cells the head can possibly reach)
* States: $K = \lbrace 0 1, \dots, t \rbrace$, where $t = \lvert Q \rvert$ and $Q= \lbrace q_1, \dots, q_t \rbrace$
* Instructions/Choices: $L = \lbrace 0, 1, \dots, d \rbrace$, where $d$ is the number of instructions in $\Delta$. The index $0$ represents the "choice" of repeating a halting configuration.
* We also define $I^- = I \setminus \lbrace p(n) \rbrace$ and $L^- = L \setminus \lbrace 0 \rbrace$.

We introduce the following propositional variables to describe the computation. The intended meaning of each variable being true is given below:

* $B_{i,j,a}$: At time step $i \in I$, tape cell $j \in J$ contains symbol $a \in \Gamma$.
* $P_{i,j}$: At time step $i \in I$, the tape head is positioned on cell $j \in J$.
* $Z_{i,k}$: At time step $i \in I$, the machine is in state $q_k \in Q$.
* $A_{i,0}$: At time step $i \in I$, the configuration is a halting configuration (i.e., the computation has already terminated).
* $A_{i',\ell}$: At time step $i' \in I^-$, the $\ell$-th instruction in $\Delta$ is executed to transition from configuration $i'$ to $i'+1$.

Constructing the Formula $\phi_x$

The formula $\phi_x$ is the conjunction of several subformulas, each enforcing a specific property of a valid accepting computation.

1. Start Configuration: The machine must start in the correct initial configuration at time $0$. The state is the start state $s = q_{k_0}$, the head is at cell $1$, the input $x=x(1)\dots x(n)$ is on the tape, and other cells are blank ($\Box$).  $Z_{0,k_0} \land P_{0,1} \land \left( \bigwedge_{j=1}^{n} B_{0,j,x(j)} \right) \land \left( \bigwedge_{j \in J \setminus \lbrace 1,...,n \rbrace} B_{0,j,\Box} \right) \quad (2.4) $
2. Termination and Acceptance: The computation must terminate, and the state at the final time step p(n) must be an accepting state.  $A_{p(n),0} \land \left( \bigvee_{k \in K : q_k \in F} Z_{p(n),k} \right) \quad (2.5)$ 
3. Uniqueness Constraints: At any time step $i$, the configuration must be well-defined.
  * Unique Inscription: Each tape cell contains at most one symbol.  $\bigwedge_{(i,j,a,a') \in I \times J \times \Gamma \times \Gamma : a < a'} (B_{i,j,a} \to \neg B_{i,j,a'}) \quad (2.6)$ 
  * Unique Head Position: The head is at most at one position.  $\bigwedge_{(i,j,j') \in I \times J \times J : j < j'} (P_{i,j} \to \neg P_{i,j'}) \quad (2.7)$ 
  * Unique State: The machine is in at most one state.  $\bigwedge_{(i,k,k') \in I \times K \times K : k < k'} (Z_{i,k} \to \neg Z_{i,k'}) \quad (2.8)$ 
  * Unique Instruction: At most one instruction is executed. No instruction is executed if and only if the configuration is halting ($A_{i,0}$ is true).  $\bigwedge_{(i,\ell,\ell') \in I \times L \times L : \ell < \ell'} (A_{i,\ell} \to \ne A_{i,\ell'}) \quad (2.9)$
4. Valid Transitions: The configuration at step $i+1$ must follow legally from the configuration at step $i$.
  * Writing only at the head position: Tape cells not under the head do not change.  $\bigwedge_{(i,j,a) \in I^- \times J \times \Gamma} (B_{i,j,a} \land \neg P_{i,j} \to B_{i+1,j,a}) \quad (2.10)$ 
  * Halting configurations remain unchanged: If a configuration is halting, all subsequent configurations are identical.  $\bigwedge_{(i,j,a) \in I^- \times J \times \Gamma} (A_{i,0} \land B_{i,j,a} \to B_{i+1,j,a}) \quad (2.11)$   $\bigwedge_{(i,j) \in I^- \times J} (A_{i,0} \land P_{i,j} \to P_{i+1,j}) \land \bigwedge_{(i,k) \in I^- \times K} (A_{i,0} \land Z_{i,k} \to Z_{i+1,k}) \quad (2.12)$ 
  * Instruction Execution: In every non-halting configuration, exactly one instruction must be executed. $\bigwedge_{i \in I} \left( \bigvee_{\ell \in L} A_{i,\ell} \right) \quad (2.13)$ 
  * For the $\ell$-th instruction in $\Delta$, let it be ($q_{k_{\ell}}, a_{\ell}, q_{k'_{\ell}}, a'_{\ell}, B_{\ell}$), where $B_{\ell}$ is the head movement ($L$, $R$, $S$). Let $\delta_{\ell}$ be $-1$, $1$, $0$ for $L$, $R$, $S$ respectively.
  * A configuration is not halting if an instruction applies to it.  $\bigwedge_{(i,j,\ell) \in I \times J \times L^-} (Z_{i,k_\ell} \land P_{i,j} \land B_{i,j,a_\ell} \to \neg A_{i,0}) \quad (2.14)$ 
  * If instruction $\ell$ is chosen, its preconditions must be met by configuration $i$.  $\bigwedge_{(i,\ell) \in I^- \times L^-} (A_{i,\ell} \to Z_{i,k_\ell}) \land \bigwedge_{(i,j,\ell) \in I^- \times J \times L^-} (A_{i,\ell} \land P_{i,j} \to B_{i,j,a_\ell}) \quad (2.15)$ 
  * If instruction $\ell$ is chosen at step $i$, the successor configuration $i+1$ must reflect its execution.
    * $\bigwedge_{(i,\ell) \in I^- \times L^-} (A_{i,\ell} \to Z_{i+1,k'_\ell}) \quad (2.16)$
    * $\bigwedge_{(i,\ell,j) \in I^- \times L^- \times J} (A_{i,\ell} \land P_{i,j} \to B_{i+1,j,a'_\ell}) \quad (2.17)$
    * $\bigwedge_{(i,\ell,j) \in I^- \times L^- \times J} (A_{i,\ell} \land P_{i,j} \to P_{i+1,j+\delta_\ell}) \quad (2.18)$

### Conclusion of the Proof

The full formula $\phi_x$ is the conjunction of all subformulas (2.4) through (2.18). Every accepting computation of $M$ on input $x$ defines a satisfying assignment for the variables in $\phi_x$. Conversely, any satisfying assignment for $\phi_x$ encodes a valid sequence of configurations. The values of the $A_{i,\ell}$ variables determine which instruction is executed at each step. Starting from the initial configuration (enforced by 2.4), one can inductively determine the entire sequence of configurations. The subformulas ensure this sequence represents a valid computation that eventually reaches an accepting state.

Thus, $x \in A$ if and only if $\phi_x$ is satisfiable. This completes the proof of Theorem 52. ∎

$\text{k-SAT}$ is $NP$-Complete for $k \ge 3$

Building on Cook's Theorem, we can show that variants of $\text{SAT}$ are also $NP$-complete.

$\textbf{Corollary 53:}$ For all $k \ge 3$, the language $\text{k-SAT}$ is $NP$-complete.

**Proof.**: As established in Remark 51, all $\text{k-SAT}$ languages are in $NP$. We need to show they are $NP$-hard for $k \ge 3$. We will demonstrate this for the case $k=3$. The result for $k > 3$ follows because $\text{3-SAT}$ can be easily $p$-$m$-reduced to $\text{k-SAT}$ (by padding clauses with dummy variables, or more simply, noting that any instance of $\text{3-SAT}$ is already an instance of $\text{k-SAT}$ for $k > 3$).

To show that $\text{3-SAT}$ is $NP$-hard, we use the transitivity of $p$-$m$-reducibility. By Cook's Theorem, $\text{SAT}$ is $NP$-hard. Therefore, if we can show that $\text{SAT} \le_p^m \text{3-SAT}$, it follows that $\text{3-SAT}$ is also $NP$-hard.

We need to construct a polynomial-time computable function that transforms a formula $\phi$ into a formula $\phi'$ such that $\phi \in \text{SAT}$ if and only if $\phi' \in \text{3-SAT}$.

Let $\phi$ be a given propositional formula.

* If $\phi$ is not in CNF, it cannot be in $\text{SAT}$. In this case, we map it to a fixed, unsatisfiable $3$-CNF formula (e.g., $(X) \land (\neg X)$).
* If $\phi$ is in CNF, it has the form $\phi \equiv C_1 \land C_2 \land \dots \land C_m$, where each $C_i$ is a clause. The formula $\phi$' is obtained by replacing each clause $C_i$ in $\phi$ with a new set of clauses $\kappa_i$, constructed as follows.

Let a clause $C_i$ be $(L_1^i \lor L_2^i \lor \dots \lor L_{k_i}^i)$, where $L_j^i$ are literals.

* If $k_i \le 3$, the clause is already in $3$-CNF form, so we can let $\kappa_i \equiv C_i$.
* If $k_i > 3$, we replace $C_i$ with a conjunction of new clauses that collectively are satisfiable if and only if $C_i$ is. We introduce $k_i - 3$ new, unique variables $Z_1^i, Z_2^i, \dots, Z_{k_i-3}^i$. The replacement formula $\kappa_i$ is:  $(L_1^i \lor L_2^i \lor Z_1^i) \land (\neg Z_1^i \lor L_3^i \lor Z_2^i) \land \dots \land (\neg Z_{k_i-2}^i \lor L_{k_i-1}^i \lor Z_{k_i-1}^i) \land (\neg Z_{k_i-1}^i \lor L_{k_i}^i)$  (The source document presents a slightly different but equivalent construction, which we will analyze): Let $C_i = (L_1^i \lor \dots \lor L_{k_i}^i)$. We introduce $k_i-1$ new variables $Z_1^i, \dots, Z_{k_i-1}^i$. The clause $C_i$ is replaced by $\kappa_i$:  $\kappa_i \equiv (L_1^i \lor Z_1^i) \land \left( \bigwedge_{j=2}^{k_i-1} (\neg Z_{j-1}^i \lor L_j^i \lor Z_j^i) \right) \land (\neg Z_{k_i-1}^i \lor L_{k_i}^i)$  The final formula is $\phi' = \kappa_1 \land \kappa_2 \land \dots \land \kappa_m$. This transformation introduces a polynomial number of new variables and clauses and is computable in polynomial time.

Now, we must show that $\phi$ is satisfiable if and only if $\phi$' is satisfiable.

($\Rightarrow$) Assume $\phi$ is satisfiable. Let b be a satisfying assignment for $\phi$. For each clause $C_i = (L_1^i \lor \dots \lor L_{k_i}^i)$ in $\phi$, at least one literal must be true under $b$. Let $t_i$ be the index of the first true literal in $C_i$. We can extend the assignment $b$ to the new variables $Z_j^i$ as follows:

* Set $Z_j^i$ to true for all $j < t_i$.
* Set $Z_j^i$ to false for all $j \ge t_i$. Under this extended assignment, every clause in every $\kappa_i$ becomes true, thus satisfying $\phi$'.

($\Leftarrow$) Assume $\phi$ is unsatisfiable. We will show that $\phi$' must also be unsatisfiable. Let $b$ be an arbitrary assignment for the variables in $\phi$'. Since $\phi$ is unsatisfiable, under the restriction of $b$ to the original variables, at least one clause $C_i$ in $\phi$ must be false. This means all literals $L_1^i$, $\dots$, $L_{k_i}^i$ are false. Now consider the corresponding formula $\kappa_i$ under assignment $b$:  $\kappa_i \equiv (L_1^i \lor Z_1^i) \land (\neg Z_1^i \lor L_2^i \lor Z_2^i) \land \dots \land (\neg Z_{k_i-1}^i \lor L_{k_i}^i)$.  Since $L_1^i$ is false, the first clause $(L_1^i \lor Z_1^i)$ implies that $Z_1^i$ must be true to satisfy $\kappa_i$. Now consider the second clause $(\neg Z_1^i \lor L_2^i \lor Z_2^i)$. Since $Z_1^i$ is true and $L_2^i$ is false, this clause implies $Z_2^i$ must be true. Propagating this logic forward, we find that for $\kappa_i$ to be true, $Z_1^i, Z_2^i, \dots, Z_{k_i-1}^i$ must all be true. However, the final clause is $(\neg Z_{k_i-1}^i \lor L_{k_i}^i)$. Since $Z_{k_i-1}^i$ must be true and $L_{k_i}^i$ is false, this final clause evaluates to false. Therefore, $\kappa_i$ is false. Since $\phi$' is a conjunction containing $\kappa_i$, $\phi$' is also false. As $b$ was an arbitrary assignment, this shows that $\phi$' is unsatisfiable.

Thus, we have shown that $\text{SAT}$ $\le_p^m$ $\text{3-SAT}$, which completes the proof. ∎

-----

### Further NP-Complete Languages

Many practical computational problems are known to correspond to $\text{NP}$-complete languages. Proving a problem is $\text{NP}$-complete demonstrates that it is among the hardest problems in the class $\text{NP}$, and an efficient (polynomial-time) algorithm for it is unlikely to exist. This section will demonstrate the $\text{NP}$-completeness of the clique problem.

#### The Clique Problem

A common problem in graph theory is identifying a "clique," which is a subgraph where every vertex is connected to every other vertex.

$\textbf{Definition 54 (Clique problem):}$ A subset $C$ of the vertex set $V$ of an undirected graph $G$ is called a **clique** of $G$ if there is an edge between all pairs of distinct vertices in $C$. The clique problem is the language

$$
\text{CLIQUE} = \lbrace \langle G, k \rangle : \text{there is a clique of size } k \text{ in the graph } G \rbrace
$$

where pairs of the form $\langle G, k \rangle$ are suitably represented by binary words, e.g., in the form $1^{k+1}0u$ where $u$ is the concatenation of the rows of the adjacency matrix of $G$.

#### NP-Completeness of CLIQUE

We will now prove that the $\text{CLIQUE}$ problem is not only in $\text{NP}$ but is also $\text{NP}$-hard, making it $\text{NP}$-complete.

$\textbf{Theorem 55:}$ The language $\text{CLIQUE}$ is $\text{NP}$-complete.

**Proof.**: The proof consists of two main parts:

1.  Show that $\text{CLIQUE} \in \text{NP}$.
2.  Show that $\text{CLIQUE}$ is $\text{NP}$-hard by demonstrating that a known $\text{NP}$-complete problem, $\text{3-SAT}$, is polynomial-time many-one reducible to $\text{CLIQUE}$ (i.e., $\text{3-SAT} \le_p^m \text{CLIQUE}$).

**Part 1: $\text{CLIQUE} \in \text{NP}$**

To show that $\text{CLIQUE}$ is in $\text{NP}$, we can construct a nondeterministic Turing machine that decides it in polynomial time. For an input of the form $\langle G, k \rangle$:

1.  The machine nondeterministically guesses a set of $k$ vertices from the graph $G$.
2.  It then deterministically checks if this set of $k$ vertices forms a clique. This involves checking for an edge between every pair of the $k$ chosen vertices, which can be done in polynomial time.

Since a potential solution (a set of $k$ vertices) can be verified in polynomial time, the language $\text{CLIQUE}$ is in the class $\text{NP}$.

**Part 2: $\text{CLIQUE}$ is $\text{NP}$-hard**

To demonstrate that $\text{CLIQUE}$ is $\text{NP}$-hard, we will construct a polynomial-time computable function that maps instances of $\text{3-SAT}$ to instances of $\text{CLIQUE}$. Let this function map a Boolean formula $\phi$ to a pair $\langle G_\phi, m \rangle$.

* If a formula $\phi$ is not in $3$-CNF (and thus not in $\text{3-SAT}$), it is mapped to a fixed pair $\langle G, m \rangle$ that is known not to be in $\text{CLIQUE}$.
* If a formula $\phi$ is in $3$-CNF with $m \ge 1$ clauses, it is mapped to a pair $\langle G_\phi, m \rangle$ constructed as follows:
* The vertex set of the graph $G_\phi$ consists of all occurrences of literals in $\phi$. Since there are $m$ clauses each with at most $3$ literals, there are at most $3m$ vertices.
* An edge is drawn between two vertices in $G_\phi$ **unless** one of two conditions is met:
1.  The two corresponding literal occurrences are in the same clause.
2.  The two literals are complementary (e.g., $x$ and $\neg x$).

This mapping is clearly computable in polynomial time. We must now show that $\phi$ is satisfiable if and only if $G_\phi$ contains a clique of size $m$.

($\Rightarrow$) **If $\phi$ is satisfiable, then $G_\phi$ has a clique of size $m$.** If $\phi$ is satisfiable, there exists a satisfying assignment. This assignment must make at least one literal true in each of the $m$ clauses. We can select one such true literal from each clause. This gives us a set of $m$ literal occurrences.

* These literals cannot be in the same clause (by our selection method).
* They cannot be complementary, because a single assignment cannot make both a variable and its negation true.
Therefore, by the construction of $G_\phi$, there is an edge between every pair of these $m$ selected literals. This set of $m$ vertices forms a clique of size $m$ in $G_\phi$.

($\Leftarrow$) **If $G_\phi$ has a clique of size $m$, then $\phi$ is satisfiable.** Suppose $G_\phi$ contains a clique of size $m$.

* By construction, no two vertices in a clique can come from the same clause. Since there are $m$ vertices in the clique and $m$ clauses in $\phi$, the clique must contain exactly one literal from each clause.
* Furthermore, by construction, no two literals in a clique can be complementary.
This means we have a set of $m$ literals (one from each clause) that are mutually consistent. We can construct an assignment that makes all literals in the clique true. This assignment satisfies at least one literal in every clause, and therefore, it satisfies the entire formula $\phi$.

Since we have shown that $\phi \in \text{3-SAT} \iff \langle G_\phi, m \rangle \in \text{CLIQUE}$, and the reduction is computable in polynomial time, we have established that $\text{3-SAT} \le_p^m \text{CLIQUE}$. Because $\text{CLIQUE}$ is in $\text{NP}$ and is $\text{NP}$-hard, it is $\text{NP}$-complete. ∎

-----

### An Alternative Characterization of NP

Languages in $\text{NP}$ like $\text{SAT}$ or $\text{CLIQUE}$ can be viewed as problems that check whether an instance possesses a certain property based on the existence of an "admissible solution."

* For $\text{CLIQUE}$, an instance is a graph $G$ and a number $k$. The **possible solutions** are all subsets of nodes. An **admissible solution** is a subset that is a clique of size $k$.
* For $\text{SAT}$, an instance is a $3$-CNF formula. The **possible solutions** are all possible variable assignments. An **admissible solution** is an assignment that makes the formula true.

In both cases, and for all problems in $\text{NP}$, two key properties hold:

1.  An instance has the property if and only if some possible solution is admissible.
2.  The admissibility of any given solution can be verified efficiently (in deterministic polynomial time).

This pattern provides a fundamental characterization of the class $\text{NP}$.

$\textbf{Proposition 56:}$ A language $L$ is in the class $\text{NP}$ if and only if there exists a language $B$ in $\text{P}$ and a polynomial $p$ such that for all binary words $w$, it holds that

$$
w \in L \text{ if and only if } \exists z \in \lbrace 0, 1\rbrace^* [(w, z) \in B \text{ and } \lvert z\rvert \le p(\lvert w \rvert)]
$$

In this formulation, for an instance $w$ of $L$, the binary words $z$ (of length at most $p(\lvert w \rvert)$) represent the **possible solutions**, and the language $B$ acts as a verifier, where $(w, z) \in B$ means that $z$ is an **admissible solution** for $w$.

**Proof sketch.**:

($\Rightarrow$) **If $L \in \text{NP}$, then such a $B$ and $p$ exist.**
Let $L$ be a language in $\text{NP}$. By definition, there is a nondeterministic Turing machine $M$ that recognizes $L$ and is $p_M(n)$-time-bounded for some polynomial $p_M$. Let the instruction set of $M$ have size $d$. We can represent each instruction $j$ (for $j = 1, \dots, d$) by the binary word $1^j0^{d-j}$.

A nondeterministic computation of $M$ on an input $w$ can be represented by a sequence of choices, which corresponds to the sequence of instructions executed. We can concatenate the binary representations of these instructions to form a single binary word $z$. For an input $w$ of length $n$, the length of any computation is at most $p_M(n)$. Thus, the length of the corresponding witness string $z$ is at most $p(n) = d \cdot p_M(n)$, which is a polynomial in $n$.

We can now define the language $B$ as:

$$
B = \lbrace \langle w, z \rangle : z \text{ represents an accepting computation of } M \text{ on input } w \rbrace
$$

The language $B$ is in $\text{P}$ because a deterministic machine can take $\langle w, z \rangle$ and simply simulate the single computational path of $M$ on $w$ dictated by the choices encoded in $z$, verifying in polynomial time whether it is an accepting computation.

Therefore, $w \in L$ if and only if there exists an accepting computation, which is equivalent to saying there exists a witness $z$ with $\lvert z \rvert \le p(\lvert w \rvert)$ such that $\langle w, z \rangle \in B$.

($\Leftarrow$) **If such a $B$ and $p$ exist, then $L \in \text{NP}$.**
Let $L$ be a language for which there exists a language $B \in \text{P}$ and a polynomial $p$ satisfying the condition. We can construct a nondeterministic Turing machine $M_L$ that recognizes $L$ in polynomial time:

On input $w$ of length $n$:

1.  Nondeterministically guess a length $\ell \le p(n)$ and a binary word $z$ of length $\ell$. This is analogous to how assignments were guessed for propositional formulas.
2.  Deterministically check whether $\langle w, z \rangle \in B$. Since $B \in \text{P}$, this check is performed by a deterministic polynomial-time Turing machine.
3.  Accept $w$ if and only if the check in step 2 succeeds.

This nondeterministic machine $M_L$ runs in polynomial time and recognizes $L$. Therefore, $L \in \text{NP}$. ∎

-----

## Space-Bounded Computation

While time complexity measures the number of steps a Turing machine takes, space complexity measures the amount of memory (tape cells) it uses. For space-bounded computations, especially those with sublinear bounds (e.g., $\log n$), it is essential to distinguish the memory used for input and output from the memory used for computation (work tapes). This leads to the model of an off-line Turing machine.

$\textbf{Definition 57 (Off-line Turing Machine):}$ An **off-line Turing machine** is a Turing machine with a separate input and output tape.

  * The **input tape** is read-only. The head can only access cells containing the input symbols and the two adjacent blank cells.
  * On the **output tape**, the head can only move to the right. Whenever a symbol is written, the head must advance one position to the right.
  * A Turing machine is called a **$k$-tape Turing machine** if it has $k$ work tapes in addition to the input and output tapes.

$\textbf{Remark 58 (Configurations of Off-line Turing machines):}$ Configurations for off-line Turing machines are defined as before (state, work tape contents, work tape head positions), but with adjustments for the special tapes:

  * **Output Tape:** Neither its content nor its head position is part of a configuration, as this information cannot influence future computational steps.
  * **Input Tape:** Only the position of the head is part of a configuration. For an input of length $n$, this position is an integer in $\lbrace 0, \dots, n+1\rbrace$.

This definition ensures that configurations remain snapshots of all information needed to proceed with a computation. For sublinear space bounds, this allows the representation of a configuration to be smaller than the input itself, which is advantageous in certain constructions.

$\textbf{Convention 59 (Space-Bounds and Off-line Turing machines):}$ In the context of space-bounded computations, all considered Turing machines are off-line Turing machines, unless explicitly stated otherwise.

-----

### Deterministic Space Complexity

We can now formally define space usage and the corresponding complexity classes.

$\textbf{Definition 60 (Space Usage):}$ The space usage of a deterministic Turing machine $M$ on input $w$ is:

$$
\text{space}_M(w) =
\begin{cases}
s & \text{if } M \text{ terminates on input } w \text{ and } s \text{ is the maximum} \\
& \text{number of cells accessed on a single work tape,} \\
\uparrow & \text{otherwise.}
\end{cases}
$$

$\textbf{Remark 61:}$ We write $\log x$ for the binary logarithm $\log_2 x$. When used in complexity bounds, $\log x$ may also refer to $\lceil \log_2 x \rceil$ or $\lfloor \log_2 x \rfloor$. For space bounds, we define $\log 0 = 1$.

$\textbf{Definition 62 (Space-Bounded Deterministic Turing Machine):}$ A **space bound** is a computable function $s: \mathbb{N} \to \mathbb{N}$ with $s(n) \ge \log n$ for all $n > 0$.

$\textbf{Definition 63 (Deterministic Space Classes):}$ Let $s$ be a space bound.

A deterministic Turing machine $M$ is **$s(n)$-space-bounded** if $M$ is total and for all but finitely many inputs $w$, it holds that $\text{space}_M(w) \le s(\lvert w\rvert)$.

The class of languages decidable in deterministic space $s(n)$ is:

$$
\text{DSPACE}(s(n)) = \lbrace L \subseteq \lbrace 0, 1\rbrace ^* : L = L(M) \text{ for a deterministic } s(n)\text{-space-bounded Turing machine } M \rbrace
$$

The class $\text{DSPACE}_k(s(n))$ is defined similarly but restricts machines to at most $k$ work tapes. For a set $F$ of space bounds, we define

$$
\text{DSPACE}(F) = \bigcup_{s \in F} \text{DSPACE}(s(n)).
$$

### Key Deterministic Space Classes

Using standard function classes, we define some of the most important space complexity classes.

$\textbf{Definition 64 (Examples of Deterministic Space Classes):}$ Using the function classes

  * $\text{log} = \lbrace n \mapsto c \cdot \log n + c : c \in \mathbb{N} \setminus \lbrace 0\rbrace \rbrace$
  * $\text{poly} = \lbrace n \mapsto n^c + c : c \in \mathbb{N} \setminus \lbrace 0\rbrace \rbrace$

we define the complexity classes:

  * $\text{LOG} = \text{DSPACE}(\text{log})$
  * $\text{PSPACE} = \text{DSPACE}(\text{poly})$
  * $\text{EXPSPACE} = \text{DSPACE}(2^{\text{poly}}) = \bigcup_{c \in \mathbb{N}} \text{DTIME}(2^{n^c+c})$

We refer to $\text{PSPACE}$ as the class of problems decidable in deterministic polynomial space.

-----

### Space-Bounded Function Computation

Off-line Turing machines can also be used to compute functions. The output is simply the word written on the output tape upon termination.

$\textbf{Definition 65 (Functions computed by space-bounded Turing machines):}$ Let $s$ be a space bound. The class of functions computable in deterministic space $s(n)$ is:

$$
\text{FSPACE}(s(n)) = \lbrace f : \lbrace 0, 1\rbrace^* \to \lbrace 0, 1\rbrace^* : f = f_M \text{ for a deterministic } s(n)\text{-space-bounded Turing machine } M \rbrace
$$

For a set $F$ of space bounds,

$$
\text{FSPACE}(F) = \bigcup_{s \in F} \text{FSPACE}(s(n))
$$

Similar to time complexity, space complexity classes are robust against changes in machine specifics like the number of tapes or constant factors in the space bound.

$\textbf{Theorem 66 (Linear Compression):}$ For every space bound $s$, it holds for all natural numbers $c$ and $k$ that

$$
\text{DSPACE}_k(c \cdot s(n)) \subseteq \text{DSPACE}_k(s(n))
$$

and therefore, in particular,

$$
\text{DSPACE}(c \cdot s(n)) \subseteq \text{DSPACE}(s(n))
$$

$\textbf{Theorem 67 (Alphabet Change):}$ Let $s$ be a space bound, and let $k \ge 2$. For every language $L$ in $\text{DSPACE}_k(s(n))$, there exists a deterministic $O(s(n))$-space-bounded $k$-tape Turing machine $M$ with tape alphabet $\lbrace 0, 1, \square \rbrace$ that recognizes $L$.

-----

### Nondeterministic Space Complexity

The concept of space bounds extends naturally to nondeterministic Turing machines.

$\textbf{Definition 68 (Nondeterministic Space Classes):}$ Let $s$ be a space bound. A Turing machine $M$ is **$s(n)$-space-bounded** if $M$ is total and, for almost all inputs $w$, **all computations** of $M$ visit at most $s(\lvert w \rvert)$ cells on each work tape. The class of languages decidable in nondeterministic space $s(n)$ is:

$$
\text{NSPACE}(s(n)) = \lbrace L \subseteq \lbrace 0, 1\rbrace^* : L = L(M) \text{ for an } s(n)\text{-space-bounded Turing machine } M \rbrace
$$

-----

### Key Nondeterministic Space Classes

$\textbf{Definition 69 (Examples of Nondeterministic Space Classes):}$ We define the complexity classes:

  * $\text{NLOG} = \text{NSPACE}(\text{log})$
  * $\text{NPSPACE} = \text{NSPACE}(\text{poly})$
  * $\text{NEXP} = \text{NSPACE}(2^{\text{poly}})$

We refer to $\text{NPSPACE}$ as the class of problems decidable in nondeterministic polynomial space.

-----

### Example: The Directed Path Problem (DirPath)

A canonical problem in $\text{NLOG}$ is determining reachability in a directed graph.

The problem is formalized as the language:

$$
\text{DirPath} = \lbrace \langle A, s, t \rangle : A \text{ is the adjacency matrix of a directed graph } G \text{ in which there exists a path from node } s \text{ to node } t \rbrace
$$

An instance $\langle A, s, t \rangle$ for a graph with $n$ nodes can be represented as $1^n0z_1z_2 \dots z_n01^s01^t$, where $z_i$ are the rows of the adjacency matrix.

To show that $\text{DirPath}$ is in $\text{NLOG}$, we can construct a $(c \log n + c)$-space bounded nondeterministic Turing machine. On input $\langle A, s, t \rangle$:

1.  The machine stores the current node, starting with $s$. Storing a node index from $1$ to $n$ requires $O(\log n)$ space.
2.  It nondeterministically guesses a next node to visit by following an edge from the current node.
3.  It keeps a counter to track the number of steps taken, also requiring $O(\log n)$ space.
4.  If the machine reaches node $t$ within $n$ steps, it accepts. If it takes more than $n$ steps without reaching $t$, it rejects, preventing infinite loops in cyclic graphs.

The machine only needs to store the current node and a step counter, both of which require $O(\log n)$ space. Thus, $\text{DirPath} \in \text{NLOG}$.

-----

### Bounding Computations of Space-Bounded Machines

For a time-bounded machine, the length of any computation is naturally bounded by the time limit. For a space-bounded machine, we can derive a powerful bound on the number of computational steps by considering the number of unique configurations.

$\textbf{Lemma 71 (Configurations of Space-Bounded Turing Machines):}$ Let $s$ be a space bound, and $M = (Q, \Sigma, \Gamma, \Delta, q_0, F)$ be an $s(n)$-space bounded Turing machine. Then there exists a constant $d$ that depends only on $M$ such that the following two statements hold for all $n$:

* (i) The number of possible configurations of $M$ on an input of length $n$ is at most $2^{d \cdot s(n)}$.
* (ii) The depth of the computation tree of $M$ on an input of length $n$ is less than $2^{d \cdot s(n)}$.

**Proof.**: Part (i) follows from a direct calculation of the components of a configuration: the state, the input head position, the work tape contents, and the work tape head positions. The number of possibilities for each component is bounded, and their product gives an upper bound of the form $2^{d \cdot s(n)}$ for some machine-dependent constant $d$. This part is proven in the exercises.

<div class="accordion">
  <details>
    <summary>Proof of ($i$)</summary>
    <p>
    Let $Q$ be the set of states, $\Gamma$ the working alphabet, $n$ the input length, and $k$ the number of work tapes.
    A configuration of an off-line TM consists of the current state ($\lvert Q \rvert$ choices),
    the position of the input head ($n+2$ choices, allowing for end markers), and,
    for each work tape, both the tape contents and the head position within those contents.

    On any work tape the machine ever scans at most $s(n)$ cells, so the number of
    admissible strings on that tape is $\sum_{i=0}^{s(n)} \lvert \Gamma \rvert^i \le c_{\text{tape}}\lvert \Gamma \rvert^{s(n)}$ for a constant $c_{\text{tape}}$ that depends only on $\Gamma$.
    The $s(n)+1$ possible head locations multiply this by another factor of $s(n)+1$.
    Hence one tape contributes at most $c_{\text{tape}}\lvert \Gamma \rvert^{s(n)}(s(n)+1)$ configurations, and all $k$ tapes together contribute at most $\big(c_{\text{tape}}\lvert \Gamma \rvert^{s(n)}(s(n)+1)\big)^k$.
    Including the choice of state and input-head position gives the bound
    \[
      \lvert Q\rvert(n+2) c_{\text{tape}}^k \lvert \Gamma \rvert^{k s(n)} (s(n)+1)^k.
    \]

    The exercise further notes that configurations on inputs of length $n$ can be encoded as binary words of length $c \cdot s(n)$ for some machine-dependent constant $c$.
    Because every input of length $n$ yields at least $n+2$ configurations (one per input-head position),
    we obtain $n+2 \le 2^{c s(n)}$ once $s(n) \ge \log_2(n+2)/c$; this is true for space bounds of interest.
    Moreover $(s(n)+1)^k \le 2^{c_1 s(n)}$ for a constant $c_1$ and all sufficiently large $n$, and $\lvert \Gamma \rvert^{k s(n)} = 2^{(\log_2 \lvert \Gamma \rvert) k s(n)}$.
    Combining these inequalities we get
    \[
      \lvert Q\rvert(n+2) c_{\text{tape}}^k \lvert \Gamma \rvert^{k s(n)} (s(n)+1)^k
      \le \underbrace{\lvert Q\rvert c_{\text{tape}}^k}_{\text{constant}} 2^{c s(n)} 2^{c_1 s(n)} 2^{(\log_2 \lvert \Gamma \rvert) k s(n)}
      = 2^{d s(n)}
    \]
    for $d = c + c_1 + (\log_2 \lvert \Gamma \rvert)k$.
    </p>
  </details>
</div>

For part (ii), let $\ell = 2^{d \cdot s(n)}$, which is the upper bound on the number of distinct configurations for an input of length $n$. We will prove (ii) by contradiction. Assume that on some input $w$ of length $n$, the computation tree of $M$ has a branch of length $\ell$. This branch corresponds to a computation path with $\ell + 1$ configurations.

By the pigeonhole principle, since there are more configurations in the sequence ($\ell+1$) than there are distinct possible configurations ($\le \ell$), at least one configuration must appear twice. Let's say configuration $C$ appears at step $i$ and again at a later step $j > i$.

This means the machine has entered a loop. The sequence of steps from configuration $C$ at step $i$ to the same configuration $C$ at step $j$ can be repeated infinitely. This implies that $M$ has a non-terminating computation on input $w$.

However, an $s(n)$-space bounded machine is required by definition to be total, meaning it must terminate on all inputs. This is a contradiction. Therefore, our initial assumption must be false, and no computation path can have a length of $\ell$ or more. The depth of the computation tree must be less than $2^{d \cdot s(n)}$. ∎



-----

### Relationships Between Time and Space Complexity

A fundamental aspect of complexity theory is understanding the relationships between different resource bounds (time vs. space) and computational modes (deterministic vs. nondeterministic).

$\textbf{Remark 72:}$ Let $t$ be any time bound. A $t(n)$-time bounded Turing machine can access at most $t(n)+1$ cells on each work tape. With minor adjustments, we can ensure it is $t(n)$-space-bounded. This gives the immediate inclusions:

$$
\text{DTIME}(t(n)) \subseteq \text{DSPACE}(t(n)) \text{ and } \text{NTIME}(t(n)) \subseteq \text{NSPACE}(t(n))
$$

The following theorem summarizes the key relationships between deterministic and nondeterministic time and space classes.

<div class="gd-grid">
  <figure>
    <img src="{{ '/assets/images/notes/computability-and-complexity/coco_theorem_73.png' | relative_url }}" alt="CPU + GPU system" loading="lazy">
    <figcaption>Illustration for Theorem 73</figcaption>
  </figure>
</div>

$\textbf{Theorem 73:}$ Let $t$ be a time bound. Then the following inclusions hold.
The relationships in the second and third columns also hold for an arbitrary space bound $s$ instead of $t$.

**Proof.**: The inclusions in the diagram can be justified as follows:

1.  **Vertical Inclusions:** The inclusions from the first row to the second row (e.g., $\text{DTIME}(t(n)) \subseteq \text{NTIME}(t(n))$) hold by definition, as any deterministic Turing machine is a special case of a nondeterministic one.
2.  **First Horizontal Inclusions:** The inclusions from the first column to the second (e.g., $\text{DTIME}(t(n)) \subseteq \text{DSPACE}(t(n))$) are immediate from Remark 72.
3.  **Second Horizontal Inclusions:** The inclusions from the second column to the third (e.g., $\text{DSPACE}(t(n)) \subseteq \text{DTIME}(2^{O(t(n))})$) are a direct consequence of Lemma 71. A deterministic machine can simulate a space-bounded machine by exploring its entire configuration graph. The number of configurations is bounded by $2^{O(t(n))}$, so the simulation takes time exponential in the space bound.
4.  **Diagonal Inclusions:** The diagonal inclusions are statements of other lemmas (Lemmas 75 and 76, not provided in the source context). ∎

These relationships give rise to a chain of inclusions among the major complexity classes.

$\textbf{Corollary 74:}$ The following inclusions hold.

$$
\text{LOG} \subseteq \text{NLOG} \subseteq \text{P} \subseteq \text{NP} \subseteq \text{PSPACE} \subseteq \text{NPSPACE} \subseteq \text{EXP} \subseteq \text{NEXP}
$$

This chain represents our current understanding of the hierarchy of these classes. While these inclusions are known, the strictness of many of them remains one of the greatest open questions in computer science.

* It is known from Savitch's Theorem that $\text{PSPACE} = \text{NPSPACE}$.
* It is known from hierarchy theorems (not covered here) that some of these inclusions are strict:
* $\text{LOG} \neq \text{PSPACE}$
* $\text{NLOG} \neq \text{NPSPACE}$
* $\text{P} \neq \text{EXP}$
* $\text{NP} \neq \text{NEXP}$
* Beyond these results and their direct consequences (like $\text{P} \neq \text{NEXP}$), it is not known which of the other inclusion relations are strict. The most famous of these is the $\text{P}$ versus $\text{NP}$ problem.

---

### Relationships Between Time and Space Complexity Classes

This chapter explores the fundamental relationships between deterministic and nondeterministic complexity classes, focusing on how time-bounded and space-bounded computations relate to one another. We will establish key inclusions, such as showing that any problem solvable in nondeterministic time can be solved in deterministic space. These relationships culminate in Savitch's Theorem, a landmark result demonstrating that nondeterministic polynomial space is equivalent to deterministic polynomial space ($\text{PSPACE} = \text{NPSPACE}$).

#### Nondeterministic Time vs. Deterministic Space

A foundational result connects nondeterministic time complexity with deterministic space complexity. It establishes that any language recognizable by a nondeterministic Turing machine within a time bound $t(n)$ can also be recognized by a deterministic Turing machine using a space bound of $t(n)$. This suggests that the parallelism inherent in nondeterminism can be simulated deterministically, provided sufficient memory is available.

$\textbf{Lemma 75:}$ Let $t$ be a time bound. Then $\text{NTIME}(t(n)) \subseteq \text{DSPACE}(t(n))$.

**Proof.**: Let $L$ be a language in $\text{NTIME}(t(n))$, recognized by a $t(n)$-time-bounded $k$-tape nondeterministic Turing machine (NTM) $M$. Let the tape alphabet of $M$ be $\Gamma$ and its transition relation be $\Delta$, with size $d$. We will construct a deterministic Turing machine (DTM) $D$ that recognizes $L$ within a space bound of $t(n)$. The tape alphabet of $D$ will include $\Gamma$ plus a set of $d$ new symbols, $Y = \lbrace y_1, \dots, y_d\rbrace$, where each $y_i$ corresponds to a unique instruction in $\Delta$.

The core idea is to represent a specific computation path of $M$ on an input $w$ as a sequence of instructions. A word $u = u(1) \dots u(t)$ over the alphabet $Y$ is said to **code a partial computation** $C_0, C_1, \dots, C_t$ on input $w$ if $C_0$ is the initial configuration and for each step $i$ from $1$ to $t-1$, the instruction $u(i)$ is applicable in configuration $C_{i-1}$ and yields configuration $C_i$. If a word $u$ codes a partial computation, we call it a **coding word**. Let $C_u$ denote the final configuration of the computation coded by $u$. Note that all prefixes of a coding word are also coding words. The empty word, $\lambda$, is a coding word, and $C_\lambda$ is the initial configuration.

The computation tree of $M$ on input $w$ represents all possible computation paths. The nodes are labeled with configurations, and the children of a node with configuration $C$ correspond to the configurations reachable from $C$ in one step. There is a natural bijection, $\pi$, between **coding words** and the nodes of this computation tree. The empty word $\lambda$ maps to the root. For a coding word $uy$, $\pi(uy)$ is the child of $\pi(u)$ corresponding to the application of instruction $y$. Thus, for any coding word $u$, the node $\pi(u)$ is labeled by the configuration $C_u$.

The DTM $D$ works by determining if any node in $M$'s computation tree is labeled with an accepting configuration. To do this, $D$ can simulate the partial computation of $M$ corresponding to any given word $u$ over $Y$. By using $k$ separate tapes to simulate the tapes of $M$, this simulation requires space proportional to the length of the computation, which is at most $t(n)$. During this simulation, $D$ can also detect if $u$ is not a valid coding word.

We present two constructions for the DTM $D$.

**First Construction of D**

This construction performs a **breadth-first-style search** on the computation tree.

1.  **Compute Tree Depth:** $D$ first determines the maximum depth of the computation tree, which is equivalent to the maximum length of a coding word. It initializes a counter $i=0$.
2.  **Iterate by Length:** For a given value of $i$, $D$ systematically generates all words $u$ over $Y$ of length $i$.
3.  **Simulate and Check:** For each generated word $u$, $D$ checks if it is a coding word by simulating the corresponding partial computation of $M$. If it is, $D$ obtains the final configuration $C_u$.
4.  **Termination Condition:** The process of incrementing $i$ stops when a length $i$ is found such that all words $u$ of that length are either non-coding or lead to a halting configuration $C_u$. This value of $i$ is set as the depth, $d$.
5.  **Final Search:** Once the maximum depth $d$ is determined, $D$ generates all words over $Y$ of length up to $d$. It accepts the input $w$ if and only if it finds a word that has a coding word prefix $u$ for which the configuration $C_u$ is accepting.

This process systematically checks all possible computation paths up to the maximum time bound $t(n)$.

<div class="accordion">
  <details>
    <summary>Note to the second step (Iterate by Length)</summary>
    <p>
    We don't store all words of the current length i, because it would require exponential space. We process words in this step one by one and check it.
    </p>
  </details>
</div>


**Second Construction of D**

This construction is more efficient and performs an exhaustive **depth-first search** on the computation tree of $M$, using a backtracking algorithm.

1.  **State Management:** $D$ uses a special **index tape** to store the current coding word $u$ being explored. Initially, the index tape contains the empty word, $\lambda$.
2.  **Iteration Process:** With a coding word $u$ on the index tape, $D$ computes the configuration $C_u$ and proceeds as follows:
    * If $C_u$ is an **accepting configuration**, $D$ terminates and accepts the input.
    * If $C_u$ is a **non-accepting halting configuration**, $D$ backtracks from $u$.
    * If $C_u$ is a **non-halting configuration**, $D$ finds the "least" instruction $y$ applicable in $C_u$, writes the new coding word $uy$ on the index tape, and iterates.
3.  **Backtracking Process:** To backtrack on a word $u = vy$, $D$ considers the configuration $C_v$.
    * It checks if there is an instruction $y' > y$ that is also applicable in $C_v$.
    * If such a $y'$ exists, $D$ chooses the least one, writes the new word $vy'$ on the index tape, and iterates from there.
    * If no such $y'$ exists, it means all branches from $C_v$ have been explored, so $D$ backtracks further on $v$.
4.  **Termination:** The entire search terminates either when an accepting configuration is found (and $D$ accepts) or when the process backtracks on the empty word (meaning all paths have been exhausted without finding an accepting state, and $D$ rejects).

To determine $C_u$ and $C_v$, $D$ simulates the partial computations of $M$ coded by $u$ and $v$. This depth-first search guarantees that all configurations in the computation tree are visited. The machine $D$ is deterministic and recognizes the same language as $M$. The space required is dominated by the need to store a configuration of $M$ and the current path (coding word) on the index tape, both of which are bounded by $O(t(n))$. Therefore, $L \in \text{DSPACE}(t(n))$. ∎

#### Nondeterministic Space vs. Deterministic Time

The next major result, known as Savitch's Theorem, provides a relationship in the other direction: from nondeterministic space to deterministic time. It shows that any problem solvable in nondeterministic space can be solved in deterministic time, though with an exponential increase in the time bound.

$\textbf{Lemma 76:}$ Let $s$ be a space bound. Then $\text{NSPACE}(s(n)) \subseteq \text{DTIME}(2^{O(s(n))})$.

**Proof.**: Let $L$ be a language in $\text{NSPACE}(s(n))$, recognized by an $s(n)$-space-bounded NTM $M$. We will construct a DTM $D$ that recognizes $L$ in $2^{c \cdot s(n)}$ time for some constant $c$.

A critical observation is that the computation trees of $M$ on an input $w$ of length $n$ can be extremely large. The depth could be as large as $2^{s(n)}$, and the total number of nodes could exceed $2^{2^{s(n)}}$. A simple depth-first search, as used in the previous proof, would be too slow for our deterministic time bound.

Instead of exploring the computation tree, $D$ constructs and analyzes the **configuration graph** of $M$ on input $w$. The nodes of this graph are the possible configurations of $M$, and a directed edge exists from configuration $C_1$ to $C_2$ if $C_2$ is a successor configuration of $C_1$. The problem of acceptance is then reduced to finding a path from the initial configuration to an accepting configuration within this graph.

By Lemma 71, the number of distinct configurations of $M$ on an input of length $n$ is at most $2^{d \cdot s(n)}$ for some constant $d$. We can represent each configuration by a word of length $d \cdot s(n)$ over a fixed alphabet of size $k$.

The algorithm for $D$ on input $w$ is as follows:

1.  **Generate All Configurations:** $D$ writes a list of all possible configuration representations (all words of length $d \cdot s(n)$) onto a special tape. This list contains $k^{d \cdot s(n)}$ entries, requiring at most $2^{d' \cdot s(n)}$ space for some constant $d'$.
2.  **Initialization:** All configurations in the list are initially marked as **unreached** and **unexpanded**, except for the initial configuration on input $w$, which is marked as **reached**.
3.  **Expansion Loop:** $D$ repeatedly performs the following expansion step as long as there is any configuration marked as **reached** and **unexpanded**:
  * i.  Find the first configuration $C$ on the tape that is marked as **reached** and **unexpanded**.
  * ii. Compute all successor configurations of $C$. For each successor, find it in the list and mark it as **reached**.
  * iii. Mark configuration $C$ as **expanded**.
4.  **Final Decision:** When the loop terminates (i.e., all reached configurations have been expanded), $D$ scans the list. It accepts $w$ if and only if any accepting configuration is marked as **reached**.

By a straightforward induction on path length, any configuration that is reachable from the initial state (i.e., any configuration that appears in the computation tree) will eventually be marked as **reached**. Therefore, $D$ recognizes the same language as $M$.

For the complexity analysis, there are at most $2^{d \cdot s(n)}$ configurations, so there will be at most $2^{d \cdot s(n)}$ expansion steps. A single expansion step involves searching and updating the list of configurations. The time for one step is polynomial in the size of the list, which is $2^{d' \cdot s(n)}$. Thus, the total running time of $D$ is bounded by $2^{d \cdot s(n)} \cdot (2^{d' \cdot s(n)})^t = 2^{(d+td') \cdot s(n)}$. This is of the form $2^{O(s(n))}$, completing the proof. ∎

#### Savitch's Theorem and Its Consequences

The previous results establish relationships between nondeterministic time and deterministic space, and between nondeterministic space and deterministic time. We now turn to a more direct comparison: nondeterministic space versus deterministic space. This leads to Savitch's Theorem, a cornerstone of complexity theory.

First, we must formally address some technical prerequisites for simulating space-bounded machines.

$\textbf{Definition 77 (Space-Constructible Functions):}$ A space bound $s$ is **space-constructible** if there exists an $s(n)$-space-bounded Turing machine $M$ that computes the function $1^n \mapsto 1^{s(n)}$.

$\textbf{Theorem 78 (Space-constructible Functions):}$ The functions in the function classes $\log$, $\text{lin}$, and $\text{poly}$ are all space-constructible. If the space bound $s(n)$ is space-constructible, then so is $n \mapsto 2^{s(n)}$.

Another important technical result is that constant factors in space bounds do not affect the power of the computational model.

$\textbf{Remark 79 (Linear compression):}$ Linear compression refers to the following fact: for all space bounds $s$ and all constants $c$, every $c \cdot s(n)$-space-bounded Turing machine can be transformed into an $s(n)$-space-bounded Turing machine that recognizes the same language; in case the given Turing machine is deterministic, the new one can be chosen to be deterministic, too. Consequently, it holds for all such $s$ and $c \ge 1$ that
$$
\text{NSPACE}(c \cdot s(n)) = \text{NSPACE}(s(n))
$$
$$
\text{DSPACE}(c \cdot s(n)) = \text{DSPACE}(s(n))
$$
This is achieved by encoding blocks of symbols from the original machine's tapes into single, more complex symbols on the new machine's tapes, or by using multiple work tapes to simulate one.

$\textbf{Theorem 80 (Savitch’s Theorem):}$ Let $s$ be a space-constructible space bound. Then $\text{NSPACE}(s(n)) \subseteq \text{DSPACE}(s^2(n))$.

This theorem has a profound corollary for polynomial space complexity classes.

$\textbf{Corollary 81:}$ It holds that $\text{PSPACE} = \text{NPSPACE}$.

**Proof of the Corollary.**: By definition, $\text{DSPACE}(s(n)) \subseteq \text{NSPACE}(s(n))$ for any space bound $s$, so $\text{PSPACE} \subseteq \text{NPSPACE}$. For the reverse inclusion, let $L \in \text{NPSPACE}$. This means $L$ is recognized by an NTM in space $p(n)$ for some polynomial $p$. By Savitch's Theorem, $L \in \text{DSPACE}(p^2(n))$. Since the square of a polynomial, $p^2(n)$, is also a polynomial, it follows that $L \in \text{PSPACE}$. Therefore, $\text{NPSPACE} \subseteq \text{PSPACE}$. ∎

**Proof of Savitch’s Theorem.**: Let $L \in \text{NSPACE}(s(n))$ be recognized by an NTM $N$. We can assume without loss of generality that for any input $x$ of length $n$:

* There is a unique accepting configuration, $K_{accept}(x)$.
* All computations have a length of at most $2^{\ell(n)}$, where $\ell(n) = d \cdot s(n)$ for some constant $d$. This is because there are at most $2^{d \cdot s(n)}$ distinct configurations, so any longer computation must contain a cycle.

The NTM $N$ accepts an input $x$ if and only if there is a computation path from the initial configuration $K_{initial}(x)$ to the accepting configuration $K_{accept}(x)$ of length at most $2^{\ell(n)}$. We denote this as:

$$
K_{initial}(x) \xrightarrow{\le 2^{\ell(n)}}_N K_{accept}(x) \quad (3.1)
$$

where $K \xrightarrow{\le t}_N K'$ means there is a computation of $N$ of length at most $t$ from configuration $K$ to $K'$.

We will construct a deterministic TM $M$ that decides if this condition holds using $O(s^2(n))$ space. The core of the proof is a recursive, divide-and-conquer algorithm. To check if

$$
K_1 \xrightarrow{\le 2^i}_N K_2,
$$

the algorithm checks for the existence of an intermediate configuration $K_{mid}$ such that:

$$
K_1 \xrightarrow{\le 2^{i-1}}_N K_{mid} \quad \text{and} \quad K_{mid} \xrightarrow{\le 2^{i-1}}_N K_2
$$

The machine $M$ checks this by iterating through all possible configurations $K_{mid}$ that obey the space bound $s(n)$. For each candidate $K_{mid}$, it recursively checks the two subproblems.

The process unfolds as follows:

1.  To solve the main problem (which is 3.1), $M$ iterates through all configurations $K$ and checks if both:

$$
i. K_{initial}(x) \xrightarrow{\le 2^{\ell(n)-1}}_N K
$$

and

$$
ii. K \xrightarrow{\le 2^{\ell(n)-1}}_N K_{accept}(x)
$$

2.  To check condition (i), $M$ recursively breaks it down further, looking for a configuration $K'$ such that:
    iii. $K_{initial}(x) \xrightarrow{\le 2^{\ell(n)-2}}_N K'$
    iv. $K' \xrightarrow{\le 2^{\ell(n)-2}}_N K$
3.  This process continues until the length of the computation to be checked is $2^0=1$ or $2^1=2$. These base cases can be checked directly by inspecting the transition function of $N$.

The depth of this recursion is $\ell(n) = d \cdot s(n)$. At each level of the recursion, the machine $M$ needs to store the configurations that form the start and end points of the current subproblem (e.g., $K_{initial}, K_{accept}, K, K'$, etc.). Since the recursion depth is $\ell(n)$, and each configuration of $N$ requires $O(s(n))$ space to store, the total space required for the recursion stack is $O(\ell(n) \cdot s(n)) = O(s(n) \cdot s(n)) = O(s^2(n))$.

By linear compression, any $O(s^2(n))$-space bounded DTM can be converted to a $\text{DSPACE}(s^2(n))$ machine. This completes the proof. ∎

#### The P versus NP Problem and PSPACE

The famous **P versus NP problem** asks whether deterministic polynomial-time computation is as powerful as nondeterministic polynomial-time computation.

A. Do the classes P and NP coincide?

This remains one of the greatest unsolved problems in computer science. It is not even known if PSPACE, the class of languages decidable in deterministic polynomial space, coincides with P or NP. A related open question concerns the closure of NP under complement.

B. Is the class NP closed under complement, i.e., does the complement $\lbrace 0, 1\rbrace^* \setminus L$ of any language $L$ in NP also belong to $\text{NP}$?

In the context of polynomial space, the analogous questions have been answered in the affirmative. Savitch's Theorem directly answers the space analogue of question A, showing $\text{PSPACE} = \text{NPSPACE}$. The answer to the space analogue of question B is also yes, a result proven by Immerman and Szelepcsényi.

#### A Complete Language for $\text{PSPACE}$

To study the intrinsic difficulty of a complexity class, we identify languages that are "hardest" within that class. This is formalized through the concepts of hardness and completeness.

$\textbf{Definition 82:}$ A **complexity class** is a set of languages over the binary alphabet. A language $B$ is **hard** for a complexity class if every language in the class is $p$-$m$-reducible to $B$. A language is **complete** for a complexity class if it is hard for the class and belongs to the class. A language that is complete for a complexity class $C$ is called **C-complete**.

We now introduce a type of logical formula whose evaluation problem is complete for $\text{PSPACE}$. This language, TQBF (True Quantified Boolean Formulas), serves a role for $\text{PSPACE}$ similar to what $\text{SAT}$ serves for $\text{NP}$.

$\textbf{Definition 83 (Quantified Propositional Formulas):}$ Let $\Lambda = \lbrace \neg, \land, (, ), \exists, \forall\rbrace$ and let $\text{Var}$ be a countable set of variables disjoint from $\Lambda$. The set of **quantified propositional formulas** over $\text{Var}$ is a set of words over the infinite alphabet $\text{Var} \cup \Lambda$ that is defined inductively as follows:

* **Base case.** All elements of $\text{Var}$ are quantified propositional formulas.
* **Inductive step.** If $\psi$ and $\psi'$ are quantified propositional formulas and $X$ is in $\text{Var}$, then $\neg\psi$, $(\psi \land \psi')$, and $\exists X \psi$ are quantified propositional formulas.

**Quantifier Primer.** Building on standard propositional logic, quantified propositional formulas extend the language with existential and universal quantifiers in addition to the usual connectives such as disjunction $(\lor)$ and implication $(\rightarrow)$. A universal quantification $\forall X \phi$ is shorthand for $\neg(\exists X \neg \phi)$. We also use the constants $0$ and $1$ for the logical values false and true, respectively, and assume conventional precedence so parentheses can be dropped when the intent is clear.

$\textbf{Remark 84 (Free and Bound Variables):}$ When quantifiers are present we distinguish between free and bound occurrences of variables. For the formula $\neg X \lor \exists X \forall Y (X \land Y)$ the variable $X$ appears three times: the first occurrence is free, the third is bound, and the second (immediately after the quantifier) is neither free nor bound. Formal definitions are omitted because the intuition is immediate for the prenex normal forms used below.

**Evaluating Truth Values.** The truth value of a quantified propositional formula $\psi$, denoted $\text{val}(\psi)$, is either $0$ (false) or $1$ (true). Evaluation follows two principles:

- Truth values are defined inductively over the structure of $\psi$.
- The value of $\psi$ is always relative to an assignment for its free variables.

Given a formula $\phi$ with free variable $X$ and an assignment $b$ to the other free variables, define $\phi_i$ as the result of replacing each free occurrence of $X$ by the constant $i \in \lbrace 0, 1\rbrace$. Then

- $\exists X \phi$ is true under $b$ iff at least one of $\phi_0$ or $\phi_1$ is true under $b$.
- $\forall X \phi$ is true under $b$ iff both $\phi_0$ and $\phi_1$ are true under $b$.

**Prenex Normal Form.** A standardized structure streamlines reasoning about quantified formulas.

$\textbf{Definition 85 (Prenex Normal Form):}$ A quantified propositional formula is in **prenex normal form** if it has the shape $Q_1 X_1 \cdots Q_m X_m \phi$ where the $X_i$ are mutually distinct variables, every $Q_i$ belongs to $\lbrace \exists, \forall\rbrace$, and $\phi$ is a quantifier-free propositional formula in conjunctive normal form (CNF).

Writing $\psi = Q_1 X_1 \cdots Q_m X_m \phi$ in prenex normal form assumes $\phi$ is in CNF and all occurrences of each variable inside $\phi$ are free. Any quantified propositional formula can be converted into an equivalent prenex normal form in deterministic polynomial time.

**The QBF Language.** The central decision problem for these formulas asks whether a sentence (a quantifier prefix followed by a quantifier-free matrix) is true.

$\textbf{Definition 87 (The Language QBF):}$ $\text{QBF} = \lbrace \psi : \psi \text{ is a true sentence in prenex normal form}\rbrace$.

$\textbf{Theorem 88 (QBF is PSPACE-Complete):}$ $\text{QBF}$ is $\text{PSPACE}$-complete, as captured by the following lemmas.

$\textbf{Lemma 89:}$ $\text{QBF} \in \text{PSPACE}$.

$\textbf{Lemma 90:}$ $\text{QBF}$ is $\text{PSPACE}$-hard.

**Proof of PSPACE-Completeness for QBF.**

$\textbf{Lemma 89 (Membership):}$ To place $\text{QBF}$ inside $\text{PSPACE}$, build a deterministic polynomial-space TM $M$ that evaluates any sentence $\psi = Q_1 X_1 \cdots Q_m X_m \phi$ of length $n$. Inputs that are not sentences in prenex normal form are rejected outright. The machine maintains three binary words $w_1, w_2, w_3$, each of length $m$, on separate tapes:

- Tape 1 ($w_1$) encodes the quantifier prefix, with bit $i$ equal to $1$ iff $Q_i$ is existential.
- Tape 2 ($w_2$) is a pointer of the form $1^i 0^{m-i}$ indicating the current quantifier.
- Tape 3 ($w_3$) stores the current truth assignment for $X_1, \dots, X_m$.

Initially, $w_1$ captures the full prefix, $w_2$ points to $Q_1$, and $w_3$ is the all-zero assignment. The evaluation procedure recursively branches on each quantifier, toggling the relevant bit of $w_3$ and combining subresults according to whether the quantifier is existential or universal. The recursion depth is $m \le n$, and only the current assignment and fixed bookkeeping are stored, so the total space is polynomial in $n$. Hence $\text{QBF} \in \text{PSPACE}$. ∎

<div class="accordion">
  <details>
    <summary>States vs tapes intuition</summary>

1. **States vs tapes in space-bounded TMs**

Think of a (space-bounded) Turing machine like a very dumb CPU with:

* a **finite control** = *the set of states*, and
* one or more **tapes** = *its RAM*.

The key points:

* **States are finite.**
  For a fixed TM, the set of states is some *fixed* finite set that does **not depend on the input length** (n). So the amount of information you can encode in the state is **O(1)** bits.

* **Tapes are where all “growing” information lives.**
  Anything whose size must grow with the input (counters up to (n), pointers into the input, partial solutions, stacks, assignments to variables, etc.) must be written on a tape, because that’s the only place where you can have **unbounded** memory.

So, informally:

* **In states you store:**

  * The current *phase* of the algorithm (“initializing”, “processing next quantifier”, “evaluating (\phi)”, “returning from recursion”, etc.).
  * A small constant number of *flags* (“I’m currently in the branch (X_i = 0)” vs “(X_i = 1)”, “so far the universal quantifier is satisfied” vs “already failed”, etc.).
  * Which transition rule to apply next.

  All of this is constant-size control info — it doesn’t scale with (n).

* **On tapes you store:**

  * Anything whose size **might depend on (n)**:

    * indices or pointers that can go up to (n) or (m),
    * counters that can be as big as (n),
    * strings/words of length up to polynomial in (n),
    * encodings of assignments to variables,
    * encodings of stacks / recursion information, if needed.

When we do **space complexity**, we **only** count the number of tape cells used (usually on the work tapes). The states are considered “free” in the sense that they’re a fixed constant for that machine.

---

2. **Applying this to Lemma 89 (QBF ∈ PSPACE)**

You have the lemma:

> Build a deterministic polynomial-space TM (M) that evaluates a QBF
> (\psi = Q_1 X_1 \cdots Q_m X_m ,\phi) of length (n).
> It maintains three binary words (w_1, w_2, w_3) of length (m) on three tapes:
>
> * Tape 1 ((w_1)): quantifier types (existential or universal)
> * Tape 2 ((w_2)): pointer to current quantifier
> * Tape 3 ((w_3)): current truth assignment to (X_1,\dots,X_m)

Let’s classify what is stored where.

### What is on tapes?

Besides the **input tape** (holding the formula (\psi)), we have 3 work tapes:

1. **Tape 1: (w_1) = quantifier prefix**

   * Length (m) word.
   * Bit (i = 1) iff (Q_i) is (\exists), and bit (i = 0) iff (Q_i) is (\forall).
   * This is information that depends on the input formula, so it must live on a tape.

2. **Tape 2: (w_2) = pointer to “where we are” in the prefix**

   * Encoding (1^i 0^{m-i}) means: “we are currently processing quantifier (Q_i)”.
   * Again, (i) can be as large as (m) (which is (O(n))), so it must be represented on a tape.

3. **Tape 3: (w_3) = current assignment to variables**

   * Length (m) word.
   * Bit (i) is the current truth value chosen for (X_i).
   * As we recurse/branch on quantifiers, we change bits in (w_3).
   * This is clearly input-dependent and unbounded in general; must be on a tape.

Additionally, when evaluating the propositional part (\phi) under the assignment in (w_3), the TM might use:

* A few extra **work cells** to scan (\phi): e.g., a small counter or a current “truth value so far”.
  All of those are again stored on tapes and take (O(\log n)) or (O(1)) space each.

Crucially: all these words/tapes have length (O(m) \le O(n)), so the total space is **polynomial in (n)**.

### What lives in the *states* here?

The states implement the **control flow** of the recursive “algorithm”:

* Whether we are:

  * checking that the input is in prenex normal form,
  * initializing (w_1, w_2, w_3),
  * processing a quantifier (Q_i),
  * at “leaf level” with all variables assigned and now evaluating (\phi),
  * combining results of two branches for a quantifier (existential / universal case),
  * moving the pointer to the next quantifier, or backtracking.

Concretely, the machine’s states might encode things like:

* `STATE_INIT` – set up (w_1, w_2, w_3).
* `STATE_AT_QI` – we are about to branch on quantifier (Q_i).
* `STATE_BRANCH_0` / `STATE_BRANCH_1` – we are exploring the subtree with (X_i = 0) or (X_i = 1).
* `STATE_EVAL_PHI` – evaluate (\phi) under current assignment (w_3).
* `STATE_RETURN_EXIST_OK` / `STATE_RETURN_EXIST_FAIL` – existential quantifier result is true/false.
* `STATE_RETURN_UNIV_OK` / `STATE_RETURN_UNIV_FAIL` – universal quantifier result is true/false.
* Plus a few states to move the heads around, reset bits, etc.

There are **finitely many** such states, independent of (n). They only encode:

* *Which phase* of the algorithm we’re in,
* A few *Boolean flags* (e.g., “did we already find a satisfying branch?”),
* Which operation should be done next (toggle a bit, move pointer, etc.).

All the “big” information — which quantifier index (i) we’re on, the actual quantifier types, the assignment to variables — is represented explicitly on tapes.

### And what about “recursion depth = m ≤ n”?

The lemma informally describes a **recursive** evaluation:

* For quantifier (Q_i):

  * Try assignment (X_i = 0) and recursively evaluate the rest,
  * Try (X_i = 1) and recursively evaluate the rest,
  * Combine according to ∃ or ∀.

In the actual TM:

* The **“depth”** is encoded just by:

  * where the pointer (w_2) currently points (which (i)), and
  * the current contents of (w_3) (current partial assignment).
* We **don’t** keep a separate call stack like in a programming language. We do a depth-first search:

  * modify (w_3) to set (X_i = 0),
  * move the pointer forward (update (w_2)),
  * evaluate the rest,
  * come back, modify (w_3) to set (X_i = 1),
  * evaluate again, etc.

All of this reuses the same (O(m)) space on tapes. The fact that the recursion depth is (m) explains why **time** can be exponential, but **space** stays linear in (m).

---

**Summary in one sentence:**

* In space-bounded TMs, *states* store only constant-size control information (phase, a few flags), while all data that can grow with input size — like the quantifier prefix, index of the current quantifier, and the current variable assignment in the QBF lemma — must be stored on the tapes, and that’s exactly how the QBF→PSPACE proof is structured.

  </details>
</div>

<div class="accordion">
  <details>
    <summary>Evaluating $\phi$ still fits in polynomial space</summary>

Good question — this is *exactly* where people get suspicious about “are we really just using polynomial space?”.

Short answer:
Yes, we do need some extra memory to evaluate (\phi) and store intermediate results — **but**:

* we only ever evaluate (\phi) for **one assignment at a time**,
* we only need to keep intermediate results for **one path in (\phi)** at a time,
* and that all fits in **(O(|\phi|) \le O(n))** extra space.

So we’re still safely in PSPACE.

---

## 1. What does “evaluating (\phi)” actually involve?

(\phi) is a *quantifier-free* Boolean formula over variables (X_1,\dots,X_m).
On a given assignment (w_3) (a bitstring telling you which (X_i) are true/false), the job of the machine is:

> Given (\phi) on the **input tape** + assignment (w_3) on **tape 3**, compute whether (\phi(w_3) = 1) or (0).

The input tape is read-only and **does not count** towards space complexity.
All extra “intermediate results” must go on work tapes, and we must show that’s only polynomially many cells.

---

## 2. Two ways to see that evaluating (\phi) is poly-space

### View A: Recursive evaluation / stack

Think of (\phi) as a syntax tree: leaves are literals, internal nodes are (\land,\lor,\lnot), etc.

A very natural algorithm:

1. Start at the root of (\phi).
2. Recursively evaluate its children.
3. Combine child values using the operator ((\land,\lor,\lnot), …).
4. Return a single bit (true/false).

To implement this on a Turing machine, you keep something like a **stack** of:

* “Where am I in (\phi)?” (position or subformula),
* “Have I already evaluated the left child?”,
* “What was the result of the child?” (just 1 bit),
* “What operator is this node?” ((\land,\lor,\lnot), etc.)

The key facts:

* The **depth** of this recursion/stack is at most the **size of (\phi)** (number of symbols), hence ≤ (n).
* Each stack frame is **constant-size** (a few bits/flags + maybe an index into the input).

So the total work-tape space for evaluating (\phi) is:

[
O(\text{depth}(\phi)) \cdot O(1) \le O(|\phi|) \le O(n).
]

Plus the assignment (w_3) (length (m \le n)). So overall still (O(n)) space.

Time can be exponential (because of the branching over quantifiers), but PSPACE cares only about **space**, not time.

---

### View B: Super-space-frugal evaluation by rescanning

If we want to be *extra* stingy with space, we can even avoid storing a big stack:

* For example, if (\phi) is in CNF (AND of clauses), we can:

  * scan the input, clause by clause,
  * for each clause, keep just:

    * one “clause is satisfied?” bit,
    * maybe an index while scanning,
  * use assignment (w_3) to tell if each literal is true.

We *never* need to store all clause truth values at once — we can:

* immediately reject if we find an unsatisfied clause,
* otherwise, keep going until the end.

Again, this uses **constant or logarithmic extra space** on top of the assignment, just at the cost of potentially rescanning the input multiple times.

---

## 3. How this fits back into the lemma

The lemma’s machine uses:

* Tape 1: (w_1) – quantifier types, length (m \le n)
* Tape 2: (w_2) – pointer, length (m \le n)
* Tape 3: (w_3) – assignment, length (m \le n)
* Plus **some extra cells** to evaluate (\phi) for the current (w_3), as above:

  * either a recursion stack of depth ≤ (|\phi| \le n), or
  * a few bits for a streaming check.

Total work-tape space used at any moment:

[
O(m) + O(\text{space for evaluating }\phi) \le O(n) + O(n) = O(n),
]

which is **polynomial** in the input length (n).

We do *not* store:

* all intermediate results for **all** assignments,
* or a full truth table,
* or a big DP table over subformulas × assignments,

because that would explode space. Instead, we:

* only keep data for **one assignment** (w_3) at a time (depth-first search through quantifiers),
* only keep data for **one “path” in (\phi)** at a time (recursive evaluation, or multiple rescans).

This is the classic **time–space tradeoff**: we’re happy to redo work (evaluate (\phi) again and again) to save space.

  </details>
</div>

$\textbf{Lemma 90 (Hardness).}$ For hardness, reduce any language $A \in \text{PSPACE}$ to $\text{QBF}$ via a polynomial-time computable map $g$. Let $M = (Q, \Sigma, \Gamma, \Delta, s, F)$ be a $p(n)$-space-bounded deterministic TM recognizing $A$, assumed to use a single work tape. A configuration on inputs of length $n$ is determined by:

- the current state $q_k \in Q = \lbrace q_1, \dots, q_k\rbrace$,
- the input-head position $j \in J = \lbrace 0, 1, \dots, n + 1\rbrace$,
- the work-tape head position $j' \in J' = \lbrace 1, \dots, p(n)\rbrace$,
- the work-tape contents $w \in \Gamma^{p(n)}$.

Introduce propositional variables $Z_k$, $P_j$, $P'\_{j'}$, and $B_{j', a}$ to encode these components, and let $V\_{conf}$ be the full set of such variables:

- $Z_k$: the state is $q_k$ , 
- $P_j$: the head on the input tape is at position $j$,
- $P_{j'}$: the head on the work tape reads position $j'$of $w$,
- $B_{j',a}$: $w(j)$ is equal to $a$.

For any assignment $K$ to $V_{conf}$, write $\psi(K)$ for the value of $\psi$ under $K$; expressions like $\exists K \phi(K)$ abbreviate quantification over all variables in $V\_{conf}$.

Following the Cook-Levin blueprint, create formulas $\text{Konf}_n$, $\text{Equal}_n$, and $\text{Succ}_n$ with the properties:

- $\text{Konf}_n(K)$ is true iff $K$ represents a valid configuration.
- $\text{Equal}_n(K, K')$ is true iff $K$ and $K'$ coincide.
- $\text{Succ}_n(K, K')$ is true iff $K'$ is the successor of $K$ (provided both are valid).

Define $\text{Comp}_n^i(K, K')$ to state that a computation path of length at most $2^i$ connects $K$ to $K'$:

- **Base case ($i = 0$):**

  $$
  \text{Comp}_n^0(K, K') \equiv \text{Konf}_n(K) \land \text{Konf}_n(K') \land (\text{Equal}_n(K, K') \lor \text{Succ}_n(K, K')).
  $$

- **Inductive step:**

  $$
  \text{Comp}_n^{i+1}(K, K') \equiv \exists \tilde{K} \forall K_1 \forall K_2 (\text{Konf}_n(\tilde{K}) \land ((\text{Equal}_n(K_1, K) \land \text{Equal}_n(K_2, \tilde{K})) \lor (\text{Equal}_n(K_1, \tilde{K}) \land \text{Equal}_n(K_2, K'))) \rightarrow \text{Comp}_n^i(K_1, K_2)).
  $$

This reuse of $\text{Comp}\_n^i$ keeps the overall formula size polynomial in $i$. For input $x$ of length $n$, let $K\_{initial}(x)$ and $K\_{accept}(x)$ denote the start and accepting configurations. Since $M$ runs in at most $2^{d \cdot p(n)}$ steps for some constant $d$, define


$$
\phi_x \equiv \text{Comp}_n^{d \cdot p(n)}(K_{initial}(x), K_{accept}(x)).
$$

Then $\phi_x$ is true iff $x \in A$, and $g:x \mapsto \phi_x$ is computable in polynomial time, so $\text{QBF}$ is $\text{PSPACE}$-hard. ∎

<div class="accordion">
  <details>
    <summary>Why not the Savitch-style definition of $\text{Comp}$</summary>

Your Savitch-style definition is *logically* fine, but as a **QBF encoding** it blows up the formula size exponentially, so it doesn’t give a polynomial-time reduction. The weird-looking definition in the notes is exactly a trick to avoid this blow-up while keeping the same meaning.

---

### 1. What you “expect” vs what they actually do

What you intuitively expect is something like (fixing the index to $i+1$):

\[
\text{Comp}_n^{i+1}(K,K')
\equiv
\exists \tilde{K}\Big(
\text{Konf}_n(\tilde{K})
\land \text{Comp}_n^{i}(K,\tilde{K})
\land \text{Comp}_n^{i}(\tilde{K},K')
\Big).
\tag{*}
\]

This is exactly analogous to Savitch:

> “There is a path of length $\le 2^{i+1}$ from $K$ to $K'$ iff there exists a middle configuration $\tilde{K}$ such that there is a path of length $\le 2^i$ from $K$ to $\tilde{K}$ and from $\tilde{K}$ to $K'$.”

**Semantically, this is totally correct.** So the issue is *not* that your definition is wrong logically. The issue is **size**.

---

### 2. Why your recursive definition is a problem for QBF

Let $\lvert\text{Comp}_n^i\rvert$ be the syntactic size (number of symbols) of the formula $\text{Comp}_n^i$.

* In your version $(*)$, $\text{Comp}_n^{i+1}$ **contains two copies** of $\text{Comp}_n^{i}$ as subformulas: one with arguments $(K,\tilde{K})$ and one with $(\tilde{K},K')$.

So the size satisfies roughly

\[
\lvert\text{Comp}_n^{i+1}\rvert = O(1) + 2 \cdot \lvert\text{Comp}_n^i\rvert.
\]

This solves to

\[
\lvert\text{Comp}_n^i\rvert = O(2^i).
\]

But we need to go up to $i = d \cdot p(n)$ so that paths of length up to $2^{d p(n)}$ are allowed. That gives

\[
\lvert\text{Comp}_n^{d p(n)}\rvert = 2^{\Theta(p(n))},
\]

i.e. *exponential* in the input size $n$.

And that is fatal: our reduction $g(x)$ must output a QBF formula $\phi_x$ of size **polynomial** in $\lvert x\rvert$. You cannot even *print* an exponentially long formula in polynomial time.

So:

* Your definition gives a **correct** specification of reachability,
* but it leads to an **exponentially large** formula, so it’s **not a valid polynomial-time reduction**.

This is the same “recursion tree vs DAG” phenomenon as in Savitch’s theorem, but now we care about **formula length**, not about space. Savitch’s algorithm is allowed to do exponentially many recursive calls, as long as it only uses polynomial *space*. Here, the QBF formula must be a *static object* whose length directly reflects how much “computation” we hard-wire into it.

---

### 3. What the weird $\text{Comp}$-definition is doing

Their inductive step is:

\[
\begin{aligned}
\text{Comp}_n^{i+1}(K, K') \equiv\;&
\exists \tilde{K}\;
\forall K_1 \forall K_2 \Big(
\big(
\text{Konf}_n(\tilde{K}) \land
(
(\text{Equal}_n(K_1, K) \land \text{Equal}_n(K_2, \tilde{K})) \\
&\qquad \lor
(\text{Equal}_n(K_1, \tilde{K}) \land \text{Equal}_n(K_2, K'))
)
\big)
\rightarrow
\text{Comp}_n^i(K_1, K_2)
\Big).
\end{aligned}
\]

The key idea: **only one syntactic occurrence** of $\text{Comp}_n^i$ appears — namely $\text{Comp}_n^i(K_1,K_2)$.

So now we have the size recurrence

\[
\lvert\text{Comp}_n^{i+1}\rvert = \lvert\text{Comp}_n^{i}\rvert + \text{poly}(n),
\]

hence

\[
\lvert\text{Comp}_n^{i}\rvert = \text{poly}(n) \cdot i = \text{poly}(n) \cdot p(n) = \text{poly}(n).
\]

Exactly what we need.

---

### 4. Why this formula has the *same meaning* as yours

Intuitively, they encode *both* requirements

* $\text{Comp}^i_n(K,\tilde{K})$
* and $\text{Comp}^i_n(\tilde{K},K')$

using the **same** subformula $\text{Comp}^i_n(K_1,K_2)$, and use universal quantifiers and implications to “route” the appropriate pair $(K_1,K_2)$ into that subformula.

Let’s unpack:

* We pick some $\tilde{K}$ (existential quantifier).
* Then we require that for **every** choice of configurations $(K_1,K_2)$, whenever

  * either $(K_1,K_2) = (K,\tilde{K})$, or
  * $(K_1,K_2) = (\tilde{K},K')$,

  then $\text{Comp}_n^i(K_1,K_2)$ holds.

Formally:

1. Consider the assignment where $K_1 = K$, $K_2 = \tilde{K}$. Then the antecedent of the implication becomes true (the $\text{Konf}_n$ and $\text{Equal}_n$ checks pass), so the implication forces $\text{Comp}_n^i(K,\tilde{K})$ to hold.
2. Consider the assignment where $K_1 = \tilde{K}$, $K_2 = K'$. Same story, it forces $\text{Comp}_n^i(\tilde{K},K')$ to hold.
3. For all other pairs $(K_1,K_2)$, the antecedent is false, so the implication is trivially true and imposes no extra constraints.

So from a *model-theoretic* point of view, their definition asserts exactly:

> There exists a configuration $\tilde{K}$ such that $\text{Comp}_n^i(K,\tilde{K})$ and $\text{Comp}_n^i(\tilde{K},K')$ both hold.

i.e. precisely what you wrote in $(*)$. The difference is purely **syntactic**: one copy of the big subformula vs two copies.

A much simpler toy version of the same trick:

* We want to encode $P(a) \land P(b)$ but we are only allowed one occurrence of $P(\cdot)$.
* We can write instead

  \[
  \forall x\,((x = a \lor x = b) \rightarrow P(x)).
  \]

  This is logically equivalent to $P(a)\land P(b)$ but only uses $P$ once.

Their construction is the same idea, lifted to configurations and to the reachability predicate $\text{Comp}_n^i$.

---

### 5. Summary

* Your Savitch-inspired definition of $\text{Comp}_n^{i+1}$ is **semantically correct**.
* The **problem** is that it causes the *formula length* to grow exponentially with $i$, so the reduction would not be polynomial-time.
* The given definition with $\exists \tilde{K}\,\forall K_1\forall K_2 (\dots \rightarrow \text{Comp}_n^i(K_1,K_2))$ is a clever syntactic trick:

  * it is **logically equivalent** to your intuitive version,
  * but uses only **one** copy of $\text{Comp}_n^i$, so the formula size stays **polynomial** in $\lvert x\rvert$.

If you like, next we can write both definitions side-by-side and explicitly derive the size recurrence and the equivalence in more formal detail.

  </details>
</div>

### Nondeterministic Space and Closure Under Complement

**Motivation.** Deterministic time and space classes are trivially closed under complement: if $M$ decides $L$, flipping its accepting and rejecting states decides $\bar{L}$. The same argument fails for nondeterministic time classes such as $\text{NP}$, where acceptance means at least one path accepts, and it remains open whether $\text{NP} = \text{co-NP}$. For nondeterministic space classes the situation is better, thanks to Immerman and Szelepcsényi. Both used a proof technique now known as $\text{inductive counting}$.

$\textbf{Definition (Closure Under Complement):}$ A complexity class is **closed under complement** if $L$ in the class implies $\bar{L} = \lbrace 0, 1\rbrace^* \setminus L$ is also in the class.

$\textbf{Theorem 91 (Immerman-Szelepcsényi):}$ Let $s$ be space-constructible. Then $\text{NSPACE}(s(n))$ is closed under complement.

$\textbf{Corollary 92:}$ The class of context-sensitive languages is closed under complement.

**Proof of the Corollary.** Over a binary alphabet the context-sensitive languages coincide with $\text{NSPACE}(n)$, so the corollary is an immediate special case of Theorem 91. The same holds for arbitrary alphabets after adjusting the definition of $\text{NSPACE}(s(n))$. ∎

**Proof of Theorem 91.** Let $L \in \text{NSPACE}(s(n))$ and let $N$ be an $s(n)$-space-bounded NTM recognizing $L$. Assume each $x \in L$ has a unique accepting configuration $K_{accept}(x)$. We will build an $O(s(n))$-space NTM $M$ for $\bar{L}$. For input $x$ of length $n$, configurations of $N$ consist of the input-head position, the work-tape contents (length at most $s(n)$), and the work-tape head position. Denote the set of configurations by $\text{Conf}_N(n)$. There exists a constant $d$ so each configuration can be encoded by a binary word of length $\ell(n) = d \cdot s(n)$, implying $\lvert \text{Conf}_N(n)\rvert \le 2^{\ell(n)}$. It is decidable in deterministic $O(s(n))$ space whether a word encodes a valid configuration and whether one configuration yields another in one step.

Let $\text{Conf}\_N(x, t)$ be the configurations reachable from $K_{start}(x)$ within $t$ steps, and write $k_t = \lvert \text{Conf}\_N(x, t)\rvert$. Define four $O(s(n))$-space NTMs $N_1, N_2, N_3, N_4$ for nondeterministic function computation:

- **$N_1$:** On input $(t, k_t, K)$ iterate over all configurations $K'$, checking (i) $K' \in \text{Conf}_N(x, t)$ and (ii) $K' \xrightarrow[N]{1} K$. Maintain a counter $a$ of the successful $K'$. If $a \ne k_t$, report an error. Otherwise output $1$ iff at least one $K'$ satisfies both properties, and $0$ otherwise. At least one computation path correctly guesses every subcomputation and produces the right answer without errors.
- **$N_2$:** On input $(t, k_t)$ iterate over all configurations $K$ and invoke $N_1(t, k_t, K)$ to determine whether $K \in \text{Conf}\_N(x, t+1)$. Count the number of configurations for which $N_1$ returns $1$. Errors from $N_1$ propagate; otherwise the count equals $k_{t+1}$.
- **$N_3$:** Starting from $k_0 = 1$ (only $K_{start}(x)$ is reachable in zero steps), repeatedly call $N_2$ to compute $k_1, k_2, \dots, k_{2^{\ell(n)}}$. Finally, call $N_1$ to check if $K_{accept}(x) \in \text{Conf}_N(x, 2^{\ell(n)})$ and output that result. Some computation path of $N_3$ therefore recognizes $L$.
- **$N_4$:** Replicate $N_3$ but flip the final output bit, swapping the accepting and rejecting halts on all error-free paths. This machine recognizes $\bar{L}$ within $O(s(n))$ space.

Adjusting $N_4$ so that output $1$ leads to an accepting halt and output $0$ to rejection yields an NTM that decides $\bar{L}$, establishing closure under complement for $\text{NSPACE}(s(n))$. ∎



---



## Probabilistic Turing Machines

[Alternative Definition of Probabilistic Turing Machines](/subpages/computability-and-complexity/probabilistic-turing-machine/)

### Introduction: The Auxiliary Tape Model

To understand randomized computation, we first introduce a more general model: a Turing machine equipped with an additional, special-purpose tape. This model serves as a foundation for both nondeterministic and probabilistic computation by reinterpreting the contents of this extra tape.

$\textbf{Definition (Turing machine with auxiliary tape):}$ A Turing machine with auxiliary tape is a deterministic Turing machine that has a special read-only **one-way** tape, its **auxiliary tape**. Initially, the cells of the auxiliary tape contain an infinite binary sequence, the **auxiliary sequence**, and the reading head is positioned on the first bit of this sequence. A Turing machine with auxiliary tape is **total** if it terminates on all inputs and auxiliary sequences, and it is **$t(n)$-time-bounded** for some time-bound $t(n)$ if it is total and runs for at most $t(\lvert w \rvert)$ steps on almost all inputs $w$ and all auxiliary sequences.

The key feature of the auxiliary tape is that it is **one-way**, meaning the head can only move to the right. This makes it suitable for modeling a stream of information, such as guesses or random bits.

<div class="accordion">
  <details>
    <summary>Note to the definition</summary>
    <p>
    So formally you can think of it as:
    <ul>
      <li>Input tape: contains the main input word $w$.</li>
      <li>Auxiliary tape: contains another word $a$ (the auxiliary sequence), read-only.</li>
      <li>Work tape: blank at the beginning, read/write, used for computation.</li>
    </ul>
    The machine’s behavior is then a function of <strong>both</strong> $a$ and $w$. We write it as $M(a,w)$.
    </p>
  </details>
</div>

<div class="accordion">
  <details>
    <summary>Why do we use an auxiliary tape?</summary>
    <p>We don’t “need” it in the sense of computability (a standard TM can simulate it), but we introduce this model because it’s very handy for definitions and proofs.</p>
    <p>Typical uses:</p>

    <ol>
      <li>
        <p><strong>Conditional Kolmogorov complexity</strong><br />
        When we define conditional complexity $K(x \mid y)$, the string $y$ is usually placed on a
        <strong>read-only auxiliary tape</strong>.<br />
        Intuition: “shortest program that outputs $x$ when it is given $y$ as extra information.”<br />
        Here:</p>
        <ul>
          <li>$x$ is the main output of the machine.</li>
          <li>$y$ is on auxiliary tape.</li>
          <li>The program (description) is on the normal input tape.</li>
        </ul>
      </li>

      <li>
        <p><strong>Advice / non-uniform complexity</strong><br />
        In $\text{P/poly}$, a polynomial-time machine gets an <strong>advice string</strong> that depends only on the input length $n$, not on the specific input:</p>
        <ul>
          <li>For inputs of length $n$, the same advice string $a_n$ is written on the auxiliary tape.</li>
          <li>The TM reads both the actual input $w$ and the advice $a_n$.</li>
        </ul>
      </li>

      <li>
        <p><strong>Parameterized algorithms</strong><br />
        Sometimes you want to model algorithms that take a main input $w$ and a parameter $k$.<br />
        You can view $k$ (or some encoding of it) as sitting on the auxiliary tape.</p>
      </li>
    </ol>

    <p>So: the auxiliary tape is a clean way to say <em>“the machine gets some additional read-only information besides the usual input.”</em></p>
  </details>
</div>

<div class="accordion">
  <details>
    <summary>What does it mean to terminate on all inputs and auxiliary sequences?</summary>
    <p>Normally, a TM is <strong>total</strong> if it halts on <strong>every input word</strong> $w$.</p>

    <p>
      Here, because the machine’s computation depends on <strong>two strings</strong> $w$ and $a$:
    </p>
    <ul>
      <li>$w$: input word on the normal input tape.</li>
      <li>$a$: auxiliary word on the auxiliary tape.</li>
    </ul>

    <p>“Total” now means:</p>

    <blockquote>
      <p>For <strong>every</strong> input word $w$ and for <strong>every</strong> auxiliary sequence $a$, the computation $M(w,a)$ halts after finitely many steps.</p>
    </blockquote>

    <p>So “terminates on auxiliary sequence” just means:</p>

    <ul>
      <li>If you fix some particular auxiliary tape content $a$ (any $a\in{0,1}^*$), then <strong>for every input $w$</strong> the machine still halts.</li>
      <li>It cannot loop forever for some unpleasant choice of auxiliary sequence.</li>
    </ul>
  </details>
</div>

<div class="accordion">
  <details>
    <summary>Time-boundedness and “run on auxiliary sequences”</summary>

    <p>The statement was:</p>

    <blockquote>
      <p>A Turing machine with auxiliary tape is time-bounded for some time bound if it is total and runs at most $t(\lvert w\rvert)$ steps on almost all inputs $w$ and <strong>all auxiliary sequences</strong>.</p>
    </blockquote>

    <p>Breakdown:</p>

    <ul>
      <li><strong>“all auxiliary sequences”</strong>:<br />
        For <em>every</em> possible string $a$ that could be written on the auxiliary tape.
      </li>
      <li><strong>“run on auxiliary sequences”</strong>:<br />
        For each choice of $a$, you consider the computation of the machine on input $w$ with that particular $a$ on its auxiliary tape. That whole computation is a “run” of the machine.
      </li>
    </ul>

    <p>
      Concretely:<br />
      Fix $a$. Then look at the function $w \mapsto M(w,a)$.<br />
      The time-boundedness condition says: for every such $a$, for all but finitely many $w$, the number of steps in this run is at most $t(\lvert w \rvert)$.
    </p>

    <ul>
      <li><strong>“almost all inputs $w$”</strong> = all but finitely many $w$.</li>
    </ul>
  </details>
</div>

<div class="accordion">
  <details>
    <summary>Does the content of the auxiliary tape typically depend on the content of the input tape?</summary>
    <p>Short answer: <strong>both models exist</strong>, but <strong>in most standard uses the auxiliary tape content is fixed independently of the input</strong>.</p>

    <p>Let’s go through the common cases.</p>

    <hr />

    <h1>✔ Two possible relationships between input and auxiliary tape</h1>

    <h3><strong>Case 1 — Auxiliary tape is <em>independent</em> of the input (most common)</strong></h3>

    <p>The auxiliary tape contains a string $a$ that is the same for <strong>all inputs $w$</strong>.</p>

    <p>This is the case in:</p>

    <h3><strong>1. Kolmogorov complexity (conditional complexity)</strong></h3>

    <p>In $K(x \mid y)$:</p>
    <ul>
      <li>The program is on the input tape.</li>
      <li>$y$ (the condition) is on the auxiliary tape.</li>
      <li>The machine sees <strong>both</strong>.</li>
    </ul>

    <p>
      But <strong>$y$ is <em>fixed</em> for the whole computation</strong>; it does <em>not</em> depend on the program or output.
      Every possible program $p$ must work with the same fixed auxiliary string $y$.
    </p>

    <hr />

    <h3><strong>2. Advice classes (P/poly)</strong></h3>

    <p>For all inputs of length $n$:</p>
    <ul>
      <li>The advice string $a_n$ is the same for <em>every</em> input of length $n$.</li>
      <li>The machine computes $M(w, a_n)$.</li>
    </ul>

    <p>Here the auxiliary tape content <strong>depends only on input length</strong>, not on the actual input.</p>

    <hr />

    <h3><strong>3. Machines in proofs (e.g., optimal universal machines)</strong></h3>

    <p>Often the auxiliary tape contains a universal constant, e.g.:</p>
    <ul>
      <li>A fixed description of a universal interpreter</li>
      <li>A fixed encoding of rules</li>
      <li>A fixed oracle string</li>
      <li>A parameter $k$ that does not change with $w$</li>
    </ul>

    <p>Again, <strong>the same auxiliary string for all runs</strong>.</p>

    <hr />

    <h1>✔ Case 2 — Auxiliary tape depends on the input</h1>

    <p>This is also allowed <strong>in the definition</strong>, but is used less frequently.</p>

    <p>You are allowed to think of a TM as taking <strong>two inputs</strong>:</p>

    <p>$$M(w, a)$$</p>

    <p>Both are arbitrary. You can choose $a$ depending on $w$.</p>

    <p>
      But this is usually <em>not</em> the intention when talking about:
    </p>
    <ul>
      <li>Turing machines with advice</li>
      <li>Conditional Kolmogorov complexity</li>
      <li>Universal machines</li>
      <li>Complexity classes</li>
    </ul>

    <p>In those contexts, the auxiliary string plays the role of:</p>
    <ul>
      <li>fixed additional information, or</li>
      <li>a length-dependent advice string.</li>
    </ul>

    <hr />

    <h1>✔ So which is the “typical” model?</h1>

    <p><strong>Typical default interpretation:</strong></p>

    <blockquote>
      <p>The auxiliary tape contains a fixed string (or a fixed family of strings depending only on input length), and the input tape contains the varying input.</p>
    </blockquote>

    <p>This matches:</p>
    <ul>
      <li>Kolmogorov complexity</li>
      <li>P/poly advice classes</li>
      <li>Universal simulation constructions</li>
      <li>Parameterized reductions</li>
      <li>Machines with oracle answers encoded on read-only tapes</li>
    </ul>

    <p>In other words:</p>

    <h3>➤ <strong>Auxiliary tape: fixed data</strong></h3>
    <h3>➤ <strong>Input tape: varying data</strong></h3>

    <p>
      This also explains why definitions often say:<br />
      “total on all inputs and all auxiliary sequences”
    </p>

    <p>
      —because mathematically the machine is a function of two arguments, even though in applications we fix one of them.
    </p>

    <hr />

    <h1>✔ Summary</h1>

    <table>
      <thead>
        <tr>
          <th>Context</th>
          <th>Does auxiliary tape depend on input?</th>
          <th>Typical?</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Conditional Kolmogorov complexity</td>
          <td><strong>No</strong>, fixed $y$</td>
          <td>✔ Yes</td>
        </tr>
        <tr>
          <td>Advice classes (P/poly)</td>
          <td>Depends only on length $n$</td>
          <td>✔ Yes</td>
        </tr>
        <tr>
          <td>Universal machines</td>
          <td>Fixed interpreter string</td>
          <td>✔ Yes</td>
        </tr>
        <tr>
          <td>General TM-with-aux definition</td>
          <td>Allowed</td>
          <td>—</td>
        </tr>
        <tr>
          <td>TM as a 2-argument machine</td>
          <td>Allowed $M(w,a)$</td>
          <td>—</td>
        </tr>
      </tbody>
    </table>

    <p>So <strong>typically the auxiliary tape is fixed</strong>, and the input tape varies.</p>

    <hr />

    <p>
      If you want, I can explain how this relates to the “almost all inputs” condition in the definition of optimal (prefix-free) Kolmogorov complexity.
    </p>
  </details>
</div>

$\textbf{Remark:}$ In what follows, we restrict attention to Turing machines with auxiliary tape that are time-bounded. Such a Turing machine can store all bits it reads from the auxiliary tape on some work tape while still obeying its time bound. So in connection with time bounds it is no real restriction to require the auxiliary tape to be read-only and one-way. In contrast, a space-bounded Turing machine can read more bits from the auxiliary tape than it can store on its tapes. More precisely, an $s(n)$-space-bounded Turing machine may, for example, run for as many as $2^{s(n)}$ steps and may read up to the same number of bits from the auxiliary tape.

For a time-bounded machine, the number of bits it can access from the auxiliary tape is naturally limited by its runtime.

$\textbf{Remark:}$ Let $t(n)$ be a time-bound. A $t(n)$-bounded Turing machine with auxiliary tape can read at most the first $t(n)$ bits on its auxiliary tape. In general, for a total Turing machine with auxiliary tape it follows by König’s lemma that there is a computable function $t$ such that on inputs of length $n$ at most the first $t(n)$ bits of the auxiliary sequence can be read. In these situations and with $t(n)$ understood from the context, we will refer to the prefix of length $t(n)$ of the auxiliary sequence as **auxiliary word**.

This model provides a powerful way to re-characterize familiar complexity classes like $\text{NP}$.

$\textbf{Remark 102:}$ Recall that a language $L$ is in the class $\text{NP}$ if and only if there exists a language $B$ in $\text{P}$ and a polynomial $p$ such that for all binary words $w$, it holds that $w \in L \text{ if and only if } \exists z \in \lbrace 0, 1\rbrace^* [(w , z) \in B \text{ \& } \lvert z \rvert \le p(\lvert w \rvert)]$. This equivalence can be reformulated in terms of Turing machines with auxiliary tape: a language $L$ is in the class $\text{NP}$ if and only if there is a polynomially time-bounded Turing machine $M$ with auxiliary tape such that for all binary words $w$, it holds that $w \in L \text{ if and only if } M \text{ accepts } w \text{ for some auxiliary word } z$.

### Formalizing Probabilistic Computation

We can now define a **probabilistic Turing machine** by treating the auxiliary tape as a source of random bits, as if generated by flipping a fair coin.

$\textbf{Definition (Probabilistic Turing machine):}$ A **probabilistic Turing machine** is a total Turing machine with auxiliary tape. Let $M$ be a probabilistic Turing machine and let $t$ be a computable function such that $M$ reads at most the first $t(n)$ bits on its random tape. The **acceptance probability** of $M$ on input $w$ is $\text{accept}_M(w) = \frac{\lvert \lbrace z \in \lbrace 0, 1\rbrace^{t(n)} : M \text{ accepts on input } w \text{ and random word } z\rbrace \rvert }{2^{t(n)}}$, and the **rejection probability** of $M$ on $w$ is $\text{reject}_M(w) = 1 - \text{accept}_M(w)$.

In this model, the machine doesn't simply accept or reject. Instead, it accepts with a certain probability. The decision rule is based on whether this probability crosses a specific threshold.

$\textbf{Definition (Acceptance and Recognized Language):}$ Let $M$ be a total probabilistic Turing machine. Then $M$ **accepts** an input $w$, if $\text{accept}_M(w) > \frac{1}{2}$. The language $L(M)$ **recognized** by $M$ is the set of all inputs that are accepted by $M$.

This probabilistic nature introduces the possibility of error.

$\textbf{Definition (Error Probability):}$ The **error probability** of $M$ on input $w$ with respect to a language $L$ is

$$
\text{error}^L_M(w) =
\begin{cases}
\text{reject}_M(w), & \text{if } w \text{ is in } L, \\
\text{accept}_M(w), & \text{otherwise},
\end{cases}
$$

and we say that $M$ recognizes $L$ with error probability $\text{error}^L_M(w)$. The **error probability** $\text{error}_M(w)$ of $M$ on input $w$ is equal to $\text{error}^{L(M)}_M(w)$.

$\textbf{Remark:}$ Probabilistic Turing machines can also be defined in terms of nondeterministic Turing machines that have computation trees where each inner node has exactly two children and for any given input, all leaves have the same depth. For a given input, the probability of acceptance is then defined as the ratio of accepting computations to all computations. This model of probabilistic Turing machine is equivalent to the one introduced above in the sense that the same languages can be recognized in both models with the same time bound up to a constant factor and with the same error probability.

### Probabilistic Complexity Classes

Based on these definitions, we can define two fundamental complexity classes for probabilistic polynomial-time computation.

$\textbf{Definition (Probabilistic Complexity Classes):}$ The class of languages decidable in **probabilistic polynomial time** is 

$$
\text{PP} = \lbrace L \subseteq \lbrace 0, 1\rbrace^* : L = L(M) \text{ for some polynomially time-bounded probabilistic Turing machine}\rbrace
$$ 

The class of languages decidable in **probabilistic polynomial time with bounded error** is 

$$
\text{BPP} = \lbrace L \subseteq \lbrace 0, 1\rbrace^* : L = L(M) \text{ for some polynomially time-bounded probabilistic Turing machine with error probability at most } \frac{1}{3}\rbrace
$$

The class $\text{PP}$ allows the acceptance probability for a correct answer to be arbitrarily close to $\frac{1}{2}$, whereas $\text{BPP}$ requires a constant gap (e.g., between $\frac{1}{3}$ and $\frac{2}{3}$), making it a more "reliable" class of computation.

These classes have clear relationships with other major complexity classes.

$\textbf{Theorem 108:}$ *It holds that $\text{BPP} \subseteq \text{PP} \subseteq \text{PSPACE}$.*

**Proof sketch.**: The first inclusion, $\text{BPP} \subseteq \text{PP}$, holds by definition. The second inclusion follows because a polynomially time-bounded probabilistic Turing machine can be simulated by a polynomially space-bounded deterministic Turing machine for all random words while counting the number of random words that result in acceptance. ∎

Specifically, to prove that $\text{PP} \subseteq \text{PSPACE}$ we construct a polynomially space-bounded deterministic Turing machine $M_1$, that uses/simulates the polynomially time-bounded Turing machine $M_2$ with auxiliary tape (which is deterministic by the definition of the TM with auxiliary tape). Given the input word w on the input tape, $M_1$ simulates $M_2$ on this input, but because $M_1$ does not have an auxiliary tape, one working tape $M_1$ serves as a auxiliary tape for $M_2$. $M_1$ writes all possible auxiliary words $z$ to the auxiliary tape one by one and for each such auxiliary word $M_2$ accepts or rejects. $M_2$ is total by definition, so we know for sure that it will either accept or reject a pair $(w,z)$. $M_1$ counts the number of accepted pairs $(w,z)$, updating its counter after each run of $M_2$ on a pair $(w,z)$: increment if is accepted, otherwise do nothing. Then if the number of accepted pairs is strictly greater than half of all pairs, the turing machine $M_1$ accepts the input $w$, rejecting otherwise. I divide the space analysis of $M_1$ into to parts: (i) space analysis of $M_2$ simulation and (ii) analysis of accepted pairs $(w,z)$ counting.

(i) Each run on each pair $(w,z)$ requires polynomial memory as $M_2$ is time-bounded by some polynomial $t(\lvert w \rvert)$ and $\text{DTIME}(t(\lvert w \rvert)) \subseteq \text{DSPACE}(t(\lvert w \rvert))$ $\implies$ one simulation of $M_2$ takes $t(\lvert w \rvert)$ memory cells. Construction of the auxiliary tape requires the same polynomial $t(\lvert w \rvert)$ memory cells, because $t(\lvert w \rvert)$ time-bounded TM with an auxiliary tape, can read at most of the first $t(\lvert w \rvert)$ bits on its auxiliary tape, that is why we consider auxiliary words of length $t(\lvert w \rvert)$ only. Each run of $M_2$ we reuse the same tapes and cells, thus $2t(\lvert w \rvert)$ memory cells for overall simulation of $M_2$.

(ii) Assuming that the alphabet of the auxiliary tape is binary ($\lbrace 0,1 \rbrace$), the total number of all possible pairs is $2^{t(\lvert w \rvert)}$, which greater than polynomial. That is why we will use binary counter, reducing the $2^{t(\lvert w \rvert)}$ memory cells that would unary counting used to $\log 2^{t(\lvert w \rvert)} = t(\lvert w \rvert)$ memory cells that binary counting uses. So, binary counter uses polynomial $t(\lvert w \rvert)$ number of memory cells. Technical detail: we skip the first accepted pair.

After the simulation and counting steps end, we check whether the most significant bit is $1$. If it is, then the number of of accepted pairs exceeds the number of rejected ones (we skipped the first accepted pair, then there cannot be a situation that counter shows only half of accepted pairs with the most significant bit $1$), implying that the fraction of accepted pairs is greater than $\frac{1}{2}$ and the polynomially time-bounded probabilistic Turing machine $M_2$ accepts the input word $w$.


Furthermore, these classes are robust under polynomial-time reductions.

$\textbf{Theorem 109:}$ *The complexity classes $\text{PP}$ and $\text{BPP}$ are closed downwards under $p$-$m$-reducibility.*

**Proof.**: We give the proof for the case of $\text{PP}$ and omit the essentially identical considerations for $\text{BPP}$. Let $A$ be reduced to a language $B$ in $\text{PP}$ via some function $g$ that is computable in time $p_g(n)$ for some polynomial $p_g$. Let $M$ be a $p_M(n)$-time bounded probabilistic Turing machine that recognizes $B$ where $p_M$ is a polynomial. Then $A$ is recognized by a probabilistic Turing machine $M_A$ that on input $w$ first computes $g(w)$, then simulates $M$ on input $g(w)$, and accepts if and only if $M$ accepts. Observe that $M_A$ can be chosen to be $(p_g(n) + p_M(p_g(n)))$-time-bounded, hence is polynomially time-bounded. ∎

The connection between $\text{NP}$ and $\text{PP}$ is also well-defined.

$\textbf{Theorem 110:}$ *It holds that $\text{NP} \subseteq \text{PP}$.*

**Proof.**: Let $L$ be any language in $\text{NP}$. By Remark 102, there is a polynomial $p$ and a deterministic $p(n)$-time-bounded Turing machine $M$ with auxiliary tape such that $w$ is in $L$ if and only if $M$ accepts $w$ for some auxiliary word $z$. Now we construct a new probabilistic TM $\text{P}$ that uses $\text{M}$ as a subroutine, plus a trick.  On input $w$, $P$ does:

1. Take a random word $r \in \lbrace 0,1\rbrace^{t(\lvert w \rvert)}$.
   * We pick $t(\lvert w \rvert) = p(\lvert w \rvert) + 1$ so that we can split it as: $r = b z, \quad b \in \lbrace 0,1 \rbrace,\ z \in \lbrace 0,1 \rbrace^{p(\lvert w\rvert)}.$

2. If the **first bit** $b = 1$, $P$ **always accepts**.

3. If the **first bit** $b = 0$, then $P$ runs $M$ on input $w$ with auxiliary word $z$, and accepts iff $M$ accepts.

The key is step 2: *“always accepts in case the first bit of the random word is 1”*. That’s the “probability boosting” trick increasing the probability of acceptance to $\frac{1}{2}$ regardless of the problem $w$ itself, then adjustment of the probability based on the problem $w$.

Then $L$ is recognized by the polynomially time-bounded probabilistic Turing machine $P$ that on input $w$ always accepts in case the first bit of the random word is $1$ and, otherwise, in case the random word has the form $0z$, accepts if and only if $M$ accepts $w$ with auxiliary word $z$. ∎

### Properties of the Class PP

The definition of $\text{PP}$ allows for an acceptance and error probability of exactly $\frac{1}{2}$, which is ambiguous: it could happend that the PTM accepts some input word $w$ with exactly a half of all random words. It means that $\text{accept}_M(w)=\text{reject}_M(w)=\frac{1}{2} \implies \text{error}^L_M(w) = \frac{1}{2}$. However, this can be avoided.

$\textbf{Lemma 111:}$ *Every language in $\text{PP}$ is recognized by a polynomially time-bounded probabilistic Turing machine that has error probability strictly less than $\frac{1}{2}$.*

**Proof.**: Let $L$ be a language in $\text{PP}$ and let $M$ be a probabilistic Turing machine that recognizes $L$ and is $p(n)$-time-bounded for some polynomial $p$. We construct a probabilistic Turing machine $M'$ as required. For almost all $n$, $M$ runs for at most $p(n)$ steps, hence reads at most $p(n)$ bits from its random tape on all inputs of length $n$. We can disregard all other inputs by hard-wiring their acceptance or rejection into $M'$.

On an input $w$ of length $n$, the Turing machine $M'$ first computes $p(n)$ and copies the first two blocks of $p(n)$ bits on its random tape to two special tapes, respectively. Then $M'$ works like $M$ on input $w$ while using the first special tape as random tape, and $M'$ accepts if and only if $M$ accepts and the bits on the second tape are not all $0$.

In case $w$ is not in $L(M)$, by construction either $\text{reject}\_{M'}(w) = \text{reject}\_M(w) = 1 \quad \text{or} \quad \text{reject}\_{M'}(w) > \text{reject}\_M(w) \ge \frac{1}{2}$. In case the word $w$ is in $L(M)$, it is accepted by $M$ on strictly more than half of all random words of length $p(n)$, hence is accepted by $M'$ for at least $(2^{p(n)-1} + 1)(2^{p(n)} - 1) = \underbrace{2^{2p(n)-1} + 2^{p(n)} - 1}_{>0}$ many random words of length $2p(n)$, i.e., for strictly more than half of the latter words. In summary, the polynomially time-bounded probabilistic Turing machine $M'$ recognizes the same language as $M$ and has error probability strictly less than $\frac{1}{2}$. ∎

<div class="accordion">
  <details>
    <summary>Alternative explanation</summary>
    <p>Let $L \in \text{PP}$. Then there exists a probabilistic Turing machine $M$ and a polynomial $p(n)$ such that</p>

    <ul>
      <li>$M$ runs in time at most $p(n)$ on all inputs of length $n$.</li>
      <li>and for every input $w$,
        $$
        w \in L \iff \Pr[M(w) \text{ accepts}] > \tfrac12 .
        $$
      </li>
    </ul>

    <p>Because $M$ runs for at most $p(n)$ steps, it can use at most $p(n)$ random bits.
    Thus, on inputs of length $n$, $M$ examines only the first $p(n)$ bits of its random tape.</p>

    <p>We now construct a new machine $M'$ that reduces the error below $1/2$.</p>

    <hr />

    <h2>Construction of $M'$</h2>

    <p>On input $w$ of length $n$, $M'$ does the following:</p>

    <ol>
      <li><strong>Compute $p(n)$</strong> and read <strong>exactly $2p(n)$</strong> random bits.</li>

      <li><strong>Split</strong> these random bits into two blocks:
        <ul>
          <li>Block A (length $p(n)$) — this block will be used exactly as the random tape for simulating $M$.</li>
          <li>Block B (length $p(n)$) — this block will serve as an extra check.</li>
        </ul>
      </li>

      <li><strong>Simulate $M$</strong> on input $w$ using Block A as the source of random bits.</li>

      <li><strong>Acceptance rule of $M'$:</strong>
        $M'$ accepts iff
        <ul>
          <li>$M$ accepts (based on Block A), <strong>and</strong></li>
          <li>Block B is <strong>not the all-zero string</strong>.</li>
        </ul>
      </li>
    </ol>

    <p>Thus, compared to $M$, $M'$ rejects some additional random strings (specifically those with Block B $= 0^{p(n)}$).</p>

    <hr />

    <h2>Correctness: error drops below $1/2$</h2>

    <h3>Case 1: $w \notin L$</h3>

    <p>Then $M$ accepts at most half of the Block-A strings:</p>

    <p>$$
    \Pr_A[M(w) \text{ accepts}] \le \tfrac12 .
    $$</p>

    <p>For $M'$ to accept, both conditions must hold:</p>

    <ol>
      <li>$M$ accepts (probability $\le \tfrac12$),</li>
      <li>Block B is not all-zero (probability $1 - 2^{-p(n)}$).</li>
    </ol>

    <p>Therefore,</p>

    <p>$$
    \Pr[M'(w)\text{ accepts}]
    = \Pr[M(w)\text{ accepts}] \cdot (1 - 2^{-p(n)})
    \le \tfrac12 .
    $$</p>

    <p>In fact, because $1 - 2^{-p(n)} < 1$, we even have</p>

    <p>$$
    \Pr[M'(w)\text{ accepts}] < \tfrac12.
    $$</p>

    <p>So on non-members, the acceptance probability stays $\le 1/2$ (and even drops).</p>

    <hr />

    <h3>Case 2: $w \in L$</h3>

    <p>Then $M$ accepts <strong>strictly more than half</strong> of the Block-A bitstrings:</p>

    <p>$$
    \#\{A : M(w) \text{ accepts on } A\} = 2^{p(n)-1} + k
    \qquad \text{for some } k\ge 1.
    $$</p>

    <p>For each such Block-A string, $M'$ accepts unless Block B is all zeros.
    Thus, for each good $A$, exactly $2^{p(n)} - 1$ choices of Block B lead to acceptance.</p>

    <p>Hence the total number of accepting random strings of length $2p(n)$ is:</p>

    <p>$$
    (2^{p(n)-1} + 1)(2^{p(n)} - 1)
    = 2^{2p(n)-1} + 2^{p(n)} - 1.
    $$</p>

    <p>This quantity is <strong>greater than half</strong> of all $2^{2p(n)}$ possible random strings, because:</p>

    <p>$$
    2^{2p(n)-1} + 2^{p(n)} - 1 > 2^{2p(n)-1}.
    $$</p>

    <p>Thus,</p>

    <p>$$
    \Pr[M'(w) \text{ accepts}] > \tfrac12.
    $$</p>

    <hr />

    <h2>Conclusion</h2>

    <p>For every input $w$:</p>

    <ul>
      <li>If $w \notin L$, then $M'$ accepts with probability <em>at most</em> (in fact, <em>less than</em>) $1/2$.</li>
      <li>If $w \in L$, then $M'$ accepts with probability <em>strictly more</em> than $1/2$.</li>
    </ul>

    <p>The running time of $M'$ remains polynomial.</p>

    <p>Therefore, $M'$ is a polynomial-time probabilistic Turing machine with error probability <strong>strictly below $1/2$</strong> that recognizes the same language $L$. ∎</p>

  </details>
</div>

$\textbf{Remark:}$ Recall that a time bound $t(n)$ is **time-constructible** if the function $1^n \mapsto 1^{t(n)}$ can be computed in time $O(t(n))$. For such $t$, every language that is recognized by a $t(n)$-time-bounded probabilistic Turing machine is recognized by an $O(t(n))$-time-bounded probabilistic Turing machine that has error probability strictly less than $\frac{1}{2}$. The latter assertion follows by essentially the same proof as Lemma 111.

This lemma allows us to prove an important closure property for $\text{PP}$.

$\textbf{Theorem 113:}$ *The complexity class $\text{PP}$ is closed under complement.*

**Proof.**: By Lemma 111, every language $L$ in $\text{PP}$ is recognized by a polynomially time-bounded probabilistic Turing machine that has error probability strictly less than $\frac{1}{2}$. 

$$
\text{P}[M \text{ accepts } w \mid w \notin L] < \frac{1}{2} \implies \text{P}[M \text{ rejects } w \mid w \notin L] > \frac{1}{2}
$$

$$
\text{P}[M \text{ accepts } w \mid w \in L] > \frac{1}{2} \implies \text{P}[M \text{ rejects } w \mid w \in L] < \frac{1}{2}
$$

Swapping acceptance and rejection in this machine yields a polynomially time-bounded probabilistic Turing machine that recognizes the complement of $L$. ∎

It is not known whether the complexity class $\text{PP}$ is closed under union or intersection. However, it is closed under another important set operation.

$\textbf{Theorem 114:}$ *The complexity class $\text{PP}$ is closed under symmetric difference.*

**Proof.**: Let $L$ and $L'$ be languages in $\text{PP}$ that are recognized by polynomially time-bounded probabilistic Turing machines $M$ and $M'$, respectively. For every input $w$, let the error probability of $M$ and $M'$ be written in the form $\text{error}\_M(w) = \frac{1}{2} - \epsilon\_w \quad \text{and} \quad \text{error}\_{M'}(w) = \frac{1}{2} - \epsilon'\_w$, where we can assume by Lemma 111 that $\epsilon\_w$ and $\epsilon'\_w$ both are strictly larger than $0$.

Now consider a probabilistic Turing machine $\tilde{M}$ that on input $w$ simulates $M$ and $M'$ on the same input while using independent random bits for the two simulations (both auxiliary tapes are bounded by some polynomials), and then accepts if and only if exactly one of the simulations accepted.

Then $\tilde{M}$ accepts or rejects correctly with respect to the symmetric difference of $L$ and $L'$ $iff$ both or neither of the simulations are correct with respect to $L$ and $L'$. The reason is that $\tilde{M}$ works as XOR over outputs of $M$ and $M'$ and XOR is invariant to all bits flip, but sensetive if only some bits are fliped (bit flipping is analogy to incorrect results of $M$,$M'$). Both are correct or both are incorrect happens with the probability

$$
\text{P}[M,M' \text{ correct } \lor M,M' \text{ incorrect }] = \left(\frac{1}{2} + \epsilon_w\right)\left(\frac{1}{2} + \epsilon'\_w\right) + \left(\frac{1}{2} - \epsilon_w\right)\left(\frac{1}{2} - \epsilon'\_w\right) = \frac{1}{2} + 2\epsilon_w \epsilon'\_w > \frac{1}{2},
$$

$$
\implies \text{P}[M \text{ accepts } w \mid w \in L] > \frac{1}{2}
$$

$$
\implies \text{P}[M \text{ rejects } w \mid w \notin L] > \frac{1}{2} \implies \text{P}[M \text{ accepts } w \mid w \notin L] = \text{error}_{\tilde{M}}^{\tilde{L}}(w) < \frac{1}{2}
$$

hence $\tilde{M}$ recognizes the symmetric difference of $L$ and $L'$. ∎

Like $\text{NP}$ has $\text{SAT}$, $\text{PP}$ also has a natural complete problem.

$\textbf{Definition 115 (The majority satisfiability problem):}$ The **majority satisfiability problem** is the language

$$
\text{MAJ} = \lbrace \phi : \phi \text{ is a propositional formula in conjunctive normal form that is satisfied by strictly more than half of the assignments to its variables}\rbrace
$$

$\textbf{Theorem 116:}$ *The majority satisfiability problem $\text{MAJ}$ is complete for $\text{PP}$.*

**Proof sketch.**:
First, to see that the language $\text{MAJ}$ is a member of $\text{PP}$, we specify a polynomially time-bounded probabilistic Turing machine $M$ that recognizes $\text{MAJ}$. $M$ rejects every input that is not a propositional formula in conjunctive normal form. Otherwise, in case its input is a formula $\phi$ in conjunctive normal form in $n$ variables, $M$ interprets the prefix of length $n$ of its random word in the natural way as an assignment to the variables of $\phi$ and accepts in case this assignment satisfies $\phi$. So the formula $\phi$ is in $\text{MAJ}$ if and only if it has acceptance probability strictly larger than $\frac{1}{2}$ if and only if it is in $L(M)$.

Second, to show that $\text{MAJ}$ is $\text{PP}$-hard, let $L$ be a language in $\text{PP}$. Fix some polynomially time-bounded probabilistic Turing machine $M$ that recognizes $L$. Similar to the construction in the proof of Cook’s Theorem, we construct a function $w \mapsto \phi_w$ computable in polynomial time such that $\phi_w$ is a propositional formula in conjunctive normal form and there is a one-to-one correspondence between satisfying assignments of $\phi_w$ and random words $z$ such that $M$ accepts on input $w$ and random word $z$. Consequently, $w$ is in $L$ if and only if $\phi$ is in $\text{MAJ}$. ∎

$\textbf{Remark:}$ It is not known whether there are complete languages for $\text{BPP}$.

### Properties of the Class BPP

The defining feature of $\text{BPP}$ is its bounded error. A remarkable property of this class is that this error can be made arbitrarily small through a process called **probability amplification**.

$\textbf{Theorem 117 (Probability amplification):}$ *For a language $L$ the following assertions are equivalent.* 

* The language $L$ is in $\text{BPP}$.
* The language $L$ is recognized by a polynomially time-bounded probabilistic Turing machine with error probability at most $\frac{1}{2} - \epsilon$ for some $\epsilon > 0$.
* For every polynomial $p$ there is a polynomially time-bounded probabilistic Turing machine with error probability at most $2^{-p(n)}$ that recognizes $L$.

**Proof.**: By definition, (iii) implies (i) and (i) implies (ii). In order to show that (ii) implies (iii), let $M$ be a probabilistic Turing machine as asserted to exist in (ii).

For a function $t$ to be specified later, let $M'$ be a probabilistic Turing machine that on an input $w$ of length $n$ runs $2t(n) + 1$ many simulations of $M$ on input $w$ while using independent random bits for each simulation. Then $M'$ accepts according to a majority vote among the simulations, i.e., $M'$ accepts if and only if at least $t(n) + 1$ simulations resulted in acceptance, and a similar equivalence holds for rejection. Accordingly, $M'$ makes an error with respect to recognizing $L$ $\iff$ at most $t(n)$ simulations are correct.

If we let $m = 2t(n) + 1$, then the probability that such an error occurs for a given input of length $n$ is at most

$$
\sum_{j=0}^{t(n)} \binom{m}{j} \left(\frac{1}{2} + \epsilon\right)^j \left(\frac{1}{2} - \epsilon\right)^{m-j} \le \sum_{j=0}^{t(n)} \binom{m}{j} \left(\frac{1}{2} + \epsilon\right)^{m/2} \left(\frac{1}{2} - \epsilon\right)^{m/2}
$$

$$
\le \left(\frac{1}{4} - \epsilon^2\right)^{m/2} \sum_{j=0}^{t(n)} \binom{m}{j} \le \left(\frac{1}{4} - \epsilon^2\right)^{m/2} 2^{m-1} \le \left(\frac{1}{4} - \epsilon^2\right)^{t(n)} 4^{t(n)} = (1 - 4\epsilon^2)^{t(n)},
$$

where the first inequality follows because of $j \le m/2$ and the last one because of $0 < \epsilon \le \frac{1}{2}$.

By the latter, fix a constant $c > 0$ such that $(1 - 4\epsilon^2)^c < \frac{1}{2}$, and given a polynomial $p$, let $t(n) = cp(n)$. Then $M'$ is polynomially time-bounded, has error probability at most $2^{-p(n)}$, because $(1 - 4\epsilon^2)^{cp(n)} < (\frac{1}{2})^{p(n)}=2^{-p(n)}$ and by construction recognizes $L$. ∎

<div class="accordion">
  <details>
    <summary>Comment 1</summary>
  </details>
</div>

<div class="accordion">
  <details>
    <summary>Comment 2</summary>
  </details>
</div>

The ability to amplify probability makes $\text{BPP}$ a very robust class with strong closure properties.

$\textbf{Theorem 118:}$ *The complexity class $\text{BPP}$ is closed under complement. The complexity class $\text{BPP}$ is closed under all binary set-theoretical operations, including union, intersection and symmetric difference.*

**Proof.**: Closure under complement follows by swapping acceptance and rejection of a polynomially time-bounded probabilistic Turing machine with error probability at most $\frac{1}{3}$ that recognizes a given language in $\text{BPP}$.

In order to demonstrate the closure under binary set-theoretical operators, let $L$ and $L'$ be languages in $\text{BPP}$. By the third item in Theorem 117, choose polynomially time-bounded probabilistic Turing machines $M$ and $M'$ that both have error probability of at most $\frac{1}{6}$ and recognize $L$ and $L'$, respectively. Now consider a polynomially time-bounded probabilistic Turing machine that on input $w$ simulates $M$ and $M'$ on input $w$ and then applies the given operator to the results of the simulations, e.g., in the case of intersection, accepts if and only if both simulations accepted. The error probability of this probabilistic Turing machine is at most the sum of the error probabilities of $M$ and $M'$, hence is at most $\frac{1}{3}$. ∎

<div class="accordion">
  <details>
    <summary>Comment on intersction proof</summary>
    <p>Yes, intersection can tolerate some individual errors.</p>
    <p>The proof knows that but doesn’t use this extra structure; it just gives a safe upper bound.</p>
    <p>$\text{Pr}[\text{Err}_{\text{intersection}}]\ge \text{Pr}[M]+\text{Pr}[M']$</p>
  </details>
</div>

### Relationship to Non-Uniform Complexity

$\text{BPP}$ also has a fascinating connection to a non-uniform complexity class, $\text{P/poly}$, which allows Turing machines access to an "advice string" that depends only on the input length.

$\textbf{Definition 119 (Advice Function and P/poly):}$ An **advice function** is a function that maps natural numbers to binary words. A Turing machine $M$ with auxiliary tape recognizes a language $L$ with advice function $a(n)$, if $M$ is total and for every input $w$ of length $n$, $M$ reads at most the first $\lvert a(n) \rvert$ bits on its auxiliary tape, and accepts $w$ on auxiliary word $a(n)$ if and only if $w$ is in $L$. The complexity class $\text{P/poly}$ contains exactly the languages over the binary alphabet that are recognized by a polynomially time-bounded Turing machine with auxiliary tape with some advice function.

In the notation $\text{P/poly}$, the terms $\text{P}$ and $\text{poly}$ refer, respectively, to the polynomial time bound and to the fact that the advice functions $a$ that witness membership in $\text{P/poly}$ can always be chosen such that the advice $a(n)$ has length $p(n)$ for some polynomial $p$.

The power of probability amplification allows us to show that any $\text{BPP}$ language can be solved by a machine with polynomial advice.

$\textbf{Theorem 120:}$ *The complexity class $\text{BPP}$ is a subset of $\text{P/poly}$.*

**Proof.**: Let $L$ be a language in $\text{BPP}$. Let $M$ be a $p(n)$-time-bounded probabilistic Turing machine that recognizes $L$ with error probability $2^{-2n}$. We construct an advice function $a(n)$ such that $M$ recognizes $L$ with advice $a(n)$. Fix some length $n$ and consider inputs of length $n$ and their corresponding random words of length $p(n)$. Say a random word is **bad** for an input if this random word results in an error of $M$ with respect to deciding whether the input is in $L$. For each of the $2^n$ inputs, at most a fraction of $2^{-2n}$ of all random words are bad. Consequently, at most a fraction of $2^n \cdot 2^{-2n} = 2^{-n}$ of all random words is bad for some input of length $n$. So it suffices to let $a(n)$ be equal to some random word that is **good**, i.e., not bad, for all inputs of length $n$. ∎


## The Polynomial Hierarchy and Complete Languages

This section explores the **polynomial hierarchy**, a hierarchy of complexity classes that generalizes the classes **P**, **NP**, and **co-NP**. It serves to classify problems that are not known to be in **NP** but are still solvable in polynomial space. We will also investigate the relationship between the probabilistic class **BPP** and this hierarchy, and conclude by examining methods for constructing complete languages for various complexity classes.

### The Polynomial-Time Hierarchy

The polynomial hierarchy is constructed using alternating existential and universal quantifiers, bounded by a polynomial in the input size.

$\textbf{Definition 120 (Polynomial-Time Hierarchy):}$ Let $k$ be a natural number. Then $\Sigma_k^p$ is the class of all languages $L$ over the binary alphabet for which there is a language $B$ in $\textbf{P}$ and a polynomial $p$ such that a binary word $w$ of length $n$ is in $L$ if and only if

$$
\exists_{y_1}^{p(n)} \forall_{y_2}^{p(n)} \exists_{y_3}^{p(n)} \cdots Q_k^{p(n)} y_k ((w , y_1, \ldots , y_k) \in B) \quad (4.1)
$$

where $Q_k$ is equal to $\exists$ if $k$ is odd and to $\forall$ if $k$ is even. Here $\exists_{y}^{\ell}$ is a short form of existential quantification over a binary word $y$ of length $\ell$, and similarly for $\forall^{\ell}$ and universal quantification.

The class $\Pi_k^p$ is defined literally the same except that (4.1) is replaced by

$$
\forall_{y_1}^{p(n)} \exists_{y_2}^{p(n)} \forall_{y_3}^{p(n)} \cdots Q_k^{p(n)} y_k ((w , y_1, \ldots , y_k) \in B) \quad (4.2)
$$

where now $Q_k$ is equal to $\forall$ if $k$ is odd and to $\exists$ if $k$ is even.

#### Levels and Properties of the Hierarchy

The hierarchy is built upon familiar complexity classes and possesses several key properties.

$\textbf{Remark 121:}$ It holds that $\Sigma_0^p = \Pi_0^p = \textbf{P}$ and $\Sigma_1^p = \textbf{NP}$. For every $k$, by definition a language is in $\Sigma_k^p$ if and only if the complement of the language is in $\Pi_k^p$. Furthermore, the classes $\Sigma_k^p$ and $\Pi_k^p$ are both subsets of $\Sigma_{k+1}^p$ and of $\Pi_{k+1}^p$, i.e., we have $\Sigma_k^p \cup \Pi_k^p \subseteq \Sigma_{k+1}^p \cap \Pi_{k+1}^p$.

The entire polynomial hierarchy, denoted **PH**, is the union of all its levels: $\text{PH} = \bigcup_k \Sigma_k^p$. This entire structure is contained within **PSPACE**.

$\textbf{Theorem 122:}$ The class **PH** is a subset of **PSPACE**.

*(Note: The proof of this theorem is similar to the proof that QBF is PSPACE-complete and is omitted here.)*

<div class="gd-grid">
  <figure>
    <img src="{{ '/assets/images/notes/computability-and-complexity/Polynomial_time_hierarchy.png' | relative_url }}" alt="Polynomial time hierarchy" loading="lazy">
  </figure>
</div>


<div class="note-callout">
<p class="note-callout__title">Remark 123 (Collapse of the Hierarchy)</p>
<p>
In case $\Sigma_k^p$ and $\Sigma_{k+1}^p$ coincide, then $\Sigma_k^p$ coincides with <strong>PH</strong>; this is referred to as a <strong>collapse of the polynomial hierarchy</strong> to $\Sigma_k^p$. It is not known whether <strong>PH</strong> coincides with <strong>PSPACE</strong>. If the latter were the case, then the polynomial hierarchy would collapse. The latter follows because the classes $\Sigma_k^p$ are all closed downwards under $p$-$m$-reducibility. In case the two classes were equal, some class $\Sigma_k^p$ would contain a <strong>PSPACE</strong>-complete language and thus the whole class <strong>PSPACE</strong>. Everything said about the classes $\Sigma^p$ holds accordingly also for the classes $\Pi^p$.
</p>
</div>

### BPP and the Polynomial Hierarchy

The complexity class **BPP** (Bounded-error Probabilistic Polynomial time) also has a place within the polynomial hierarchy. It is known to be contained within the second level.

$\textbf{Theorem 124:}$ It holds that $\textbf{BPP} \subseteq \Sigma_2^p$, that is, for every language $L$ in **BPP** there is a language $B$ in **P** and a polynomial $p$ such that for all binary words $w$ it holds that

$$
w \in L \text{ if and only if } \exists_{x}^{p(\lvert w \rvert)} \forall_{y}^{p(\lvert w \rvert)} ((w, x, y) \in B).
$$

As a direct consequence of this theorem and the fact that **BPP** is closed under complement, we can place **BPP** more precisely within the hierarchy.

$\textbf{Corollary 125:}$ The class **BPP** is a subset of $\Sigma_2^p \cap \Pi_2^p$.

**Proof.** Let $L$ be a language in **BPP**. By Theorem 124, $L$ is in $\Sigma_2^p$. Since **BPP** is closed under complement, the complement language $\bar{L}$ is also in **BPP** and thus is in $\Sigma_2^p$. Consequently, by the definition of the hierarchy (Remark 121), the language $L$ is in $\Pi_2^p$. Therefore, $L$ is in $\Sigma_2^p \cap \Pi_2^p$, and the corollary follows. ∎

#### Proof of Theorem 124

**Proof.** Let $L$ be a language in **BPP**. Let $M$ be a polynomially time-bounded probabilistic Turing machine with an error probability of at most $2^{-n}$ that recognizes $L$. In other words 

* For all $w \in L$: $\text{Pr}[\text{(M) accepts (w)}] \ge 1 - 2^{-n}$,
* For all $w \notin L$: $\text{Pr}[\text{(M) accepts (w)}] \le 2^{-n}$,

We can assume that the random words used by $M$ have a length of $p(n)$ for some polynomial $p$. We fix a value $n_0$ such that for all $n \ge n_0$, it holds that $p(n) < 2^n$. It must happen at some $n_0$, because exponential function $2^n$ and $p(n)$ is polynomial.

Fix an input word $w$ of length $n \ge n_0$. Let

$$U = \{r \in \{0,1\}^{p(n)} : M \text{ accepts } w \text{ when its random tape is } r \}$$

So:

* $U$ is the set of random strings that make $M$ accept $w$.
* If $w \in L$, then $\|U\|$ is **almost all** of ${0,1}^{p(n)}$.
* If $w \notin L$, then $\|U\|$ is **tiny** (at most a $2^{-n}$-fraction).

For the remainder of this proof, "word" will refer to a binary word of length $p(n)$ unless stated otherwise. We use the operator $\oplus$, which represents bitwise exclusive-or (parity). For any word $v$, the function $u \mapsto u \oplus v$ is a bijection on the set of all words. Let $U \oplus v = \lbrace u \oplus v : u \in U\rbrace$. Note that $\lvert U \rvert = \lvert U \oplus v \rvert$.

We will consider the following statement:


$$\exists v_1^{p(n)} \cdots \exists v_{p(n)}^{p(n)}\ \forall z^{p(n)} \bigl( z \in U \oplus v_1 \lor \cdots \lor z \in U \oplus v_{p(n)} \bigr) (4.3)$$


Informally:

> There exist $p(n)$ “shifts” $v_1,\ldots,v_{p(n)}$ such that **every** word $z$ is contained in at least one of the shifted good sets $U \oplus v_i$.

We will show:

* If $w \notin L$, then (4.3) is **false**.
* If $w \in L$, then (4.3) is **true**.

Thus, membership in $L$ is equivalent to the truth of this $\exists\forall$-statement about $U$.

**Case 1: $w$ is not in $L$.**
In this case, the set $U$ (and thus any set $U \oplus v$) contains at most $\dfrac\{2^{-n}}{2^n}$, because the acceptance probability is at most $2^{-n}$:

$$\frac{\|U\|}{2^{p(n)}} \le 2^{-n} \quad\Longrightarrow\quad \|U\| \le 2^{p(n)} \cdot 2^{-n}$$

Consider **any** choice of $v_1,\dots,v_{p(n)}$. The union

$$(U \oplus v_1) \cup \cdots \cup (U \oplus v_{p(n)})$$

has at most

$$p(n) \cdot \|U\| \le p(n) \cdot 2^{p(n)} \cdot 2^{-n}$$

elements.

Since we chose $n_0$ so that $p(n) < 2^n$ for $n \ge n_0$, we get:

$$p(n) \cdot \|U\| < 2^{p(n)}$$

Thus, even after taking the union of all $p(n)$ sets $U \oplus v_i$, we still **do not cover the entire space** ${0,1}^{p(n)}$, which has size $2^{p(n)}$.

The union of $p(n)$ such sets cannot comprise all words, because by our choice of $n_0$, we have $p(n) < 2^n$. Consequently, the following statement is **false**:

So for any choice of $v_1,\ldots,v_{p(n)}$, there is some word $z$ that is **not** in any $U \oplus v_i$. Therefore, the universal statement

$$\forall z\ (z \in U \oplus v_1 \lor \cdots \lor z \in U \oplus v_{p(n)})$$

is false for that particular choice of the $v_i$, and hence the whole formula (4.3) is false.

So: $w \notin L \quad\Rightarrow\quad \text{formula (4.3) is false.}$

**Case 2: $w$ is in $L$.**
In this case, we show that statement (4.3) is **true**. Fix some word $z$. Consider a random experiment where the bits of a word $v$ are determined by fair coin tosses. The word $z \oplus v$ is uniformly distributed among all words. Therefore, the probability that $z \oplus v$ is not in $U$ (or equivalently, that $z$ is not in $U \oplus v$) is at most $2^{-n}$.

Now, assume the bits of words $v_1, \ldots, v_{p(n)}$ are determined by fair coin tosses. The probability that a fixed word $z$ is in none of the sets $U \oplus v_i$ is at most $(2^{-n})^{p(n)} = 2^{-np(n)}$. The probability that *some* word $z$ among all $2^{p(n)}$ words is in none of the sets $U \oplus v_1$ through $U \oplus v_{p(n)}$ is at most:

$$
2^{p(n)} \cdot 2^{-np(n)} = 2^{-(n-1)p(n)} < 1.
$$

Since this probability is less than 1, the probability that the chosen $v_i$ are such that all words $z$ are in at least one set $U \oplus v_i$ is strictly greater than 0. This means there must exist such words $v_1, \ldots, v_{p(n)}$.

<div class="accordion">
<details>
<summary>Constructing the Language B</summary>
<p>
The given machine $M$ (the probabilistic one) does not decide the language $B$. We define $B$ using $M$, and then argue that there exists a deterministic polynomial-time Turing machine (call it $D_B$) that decides $B$.
</p>
<p>
The condition $(*)$ inside the quantifiers is decidable in polynomial time. Note that $z \in U \oplus v_i$ is equivalent to $z \oplus v_i \in U$. To check if a word is in $U$, we can simulate $M$ on input $w$ with that word as the random tape, which takes polynomial time.
</p>
<p>
We can now define the required language $B \in \textbf{P}$.
</p>
<ul>
<li>
For all binary words $w$ of length $n \ge n_0$, a tuple $(w, v_1, \ldots, v_{p(n)}, z)$ is in $B$ if and only if the components $v_1, \ldots, v_{p(n)}, z$ are words of length $p(\lvert w \rvert)$ that satisfy condition $(*)$ for the set $U$ corresponding to $w$. We basically encode the existential choices into a single string. The tuple ($v_1,\ldots,v_{p(n)}$) consists of $p(n)$ words, each of length $p(n)$. So in total it has $p(n)^2$ bits. That’s still polynomial in $n$. We can encode this whole tuple as a single word $x \in {0,1}^{p'(n)} \quad\text{for some polynomial } p'$.
</li>
<li>
For the finitely many words $w$ with $\|w\| < n_0$, we can hard-wire their behavior into $B$:
<ul>
<li>If $w$ is not in $L$, $B$ contains no tuple with $w$ as the first component.</li>
<li>If $w$ is in $L$, $B$ contains all tuples of the form $(w, 0^{p(\lvert w \rvert)}, \ldots, 0^{p(\lvert w \rvert)}, z)$ where $z$ is any binary word of length $p(\lvert w \rvert)$.</li>
</ul>
</li>
</ul>
</details>
</div>

<div class="accordion">
<details>
<summary>TODO: The fact that probability that some word $z$ is not in the sets is not 100%, does imply that such $v_1, ..., v_{p(n)}$ exist, because the probability that something exists does not mean that it actually exists?</summary>
<p>
</p>
</details>
</div>

This construction fulfills the requirements of Theorem 124. ∎

## Constructing Complete Languages

A key technique for understanding complexity classes is to identify **complete languages**, which are the "hardest" problems in that class. We will first examine a complete language for **PP** and then discuss a generic method for constructing such languages.

### A Complete Language for PP

$\textbf{Definition 126 (The Majority Satisfiability Problem):}$ The **majority satisfiability problem** is the language

$$
\textbf{MAJ} = \lbrace \phi : \phi \text{ is a propositional formula in CNF that is satisfied by strictly more than half of the assignments to its variables}\rbrace
$$

$\textbf{Theorem 127:}$ The majority satisfiability problem **MAJ** is complete for **PP**.

**Proof.** The proof has two parts: showing **MAJ** is in **PP** and showing **MAJ** is **PP**-hard.

1.  **MAJ is in PP:** To show that **MAJ** is a member of **PP**, we can specify a polynomially time-bounded probabilistic Turing machine $M$ that recognizes it.

* On an input that is not a propositional formula in conjunctive normal form (CNF), $M$ rejects.
* If the input is a CNF formula $\phi$ with $n$ variables, $M$ uses the first $n$ bits of its random word as an assignment to the variables of $\phi$.
* $M$ accepts if this assignment satisfies $\phi$.

The acceptance probability of $M$ for an input $\phi$ is the number of satisfying assignments divided by $2^n$. Therefore, $\phi$ is in **MAJ** if and only if its acceptance probability is strictly greater than $1/2$, which means $\phi$ is in the language $L(M)$.

2.  **MAJ is PP-hard:** To show that **MAJ** is **PP**-hard, we provide a polynomial-time reduction from any language $L \in \textbf{PP}$ to **MAJ**.

Let $L$ be a language in **PP**, recognized by a polynomially time-bounded PTM $M$ that uses random words of length $r(n)$ for an input of length $n$. Fix an input $w$ for $M$ of length $n$. Let $r = r(n)$. Similar to the proof of the Cook-Levin theorem, we construct a propositional formula $\phi_w$ in CNF that formalizes the computations of $M$ on input $w$. In addition to the standard variables, we use variables $Y_1, \ldots, Y_r$ for the bits of the random word and $P_{i,j}$ for the position of the head on the random tape.

An assignment to the random variables $Y_1, \ldots, Y_r$ corresponds to a specific random word. For any given random word, there is exactly one computation of $M$. Consequently, every accepting assignment to the random variables can be extended in a unique way to a satisfying assignment for the entire formula $\phi_w$. Let $a(w)$ be the number of accepting random words. This number is equal to the number of satisfying assignments of $\phi_w$.

The machine $M$ accepts $w$ if its acceptance probability is greater than $1/2$. This means $w \in L$ if and only if $a(w) / 2^r > 1/2$, which is equivalent to $a(w) > 2^{r-1}$.

Let the variables of $\phi_w$ be $Y_1, \ldots, Y_m$, where $m = m(n)$ is polynomially bounded. We can construct a new formula $\phi'\_w$. Let $Z$ be a new variable not in $\phi_w$. Let $\tau$ be the formula $Y_1 \land \cdots \land Y_{m-(r-1)}$. The formula $\tau$ is satisfied by exactly $2^{m - (m-(r-1))} = 2^{r-1}$ of the $2^m$ assignments to its variables.

Define $\phi'_w$ as:

$$
\phi'_w = (Z \land \phi_w(Y_1, \ldots, Y_m)) \lor (\neg Z \land \neg \tau(Y_1, \ldots, Y_m))
$$

This formula has $m+1$ variables.

  * If $Z$ is true, $\phi'_w$ is satisfied if $\phi_w$ is satisfied. There are $a(w)$ such assignments.
  * If $Z$ is false, $\phi'_w$ is satisfied if $\neg \tau$ is satisfied. There are $2^m - 2^{r-1}$ such assignments.

The total number of satisfying assignments for $\phi'_w$ is $a(w) + (2^m - 2^{r-1})$. The formula $\phi'_w$ is in **MAJ** if this number is greater than half of the total $2^{m+1}$ assignments:

$$
a(w) + 2^m - 2^{r-1} \> \frac{1}{2} 2^{m+1} = 2^m
$$

$$
a(w) - 2^{r-1} \> 0 \implies a(w) \> 2^{r-1}
$$

This is exactly the condition for $w \in L$. Thus, $w \in L \iff \phi'_w \in \textbf{MAJ}$. Finally, $\phi'_w$ can be transformed into an equivalent CNF formula in polynomial time using distributive laws. This completes the reduction. ∎


### A Generic Technique for Constructing Complete Languages

A general approach for creating complete languages involves simulating Turing machines that recognize languages within a given complexity class. To simplify these simulations, we can standardize the machines.

For this purpose, we define a **special Turing machine** as one having a single work tape and a tape alphabet of $\lbrace 0, 1, \square \rbrace$.

$\textbf{Remark 128:}$ By the results on tape reduction and on alphabet reduction, for every $t(n)$-time-bounded Turing machine there is a special Turing machine that is $c \cdot t^2(n)$-time-bounded for some constant $c$ such that both machines recognize the same language. Furthermore, if the given Turing machine is deterministic, the special Turing machine can be chosen to be deterministic, too.

We fix a binary representation for these special TMs. For a representation word $w$, we write $M_w$ for the machine it represents. This representation scheme has the following properties:

  * Given a binary word of length $n$, it can be checked in polynomial time whether it represents a TM and whether that TM is deterministic.
  * There exists a fixed universal Turing machine $U$ that can simulate any $M_w$ on a given input. The simulating computation's length exceeds the simulated one by at most a factor of $\lvert w\rvert$.

#### Examples of Generic Complete Languages

Using this framework, we can construct complete languages for several major complexity classes. A common technique involves padding the components of the input string, where for a binary word $w$, $\tilde{w}$ is obtained by doubling every bit (e.g., if $w=01$, then $\tilde{w}=0011$).

**A Complete Language for EXP**

Let

$$
C_{\text{EXP}} = \lbrace \tilde{w}01\tilde{x}01t : \text{the word } w \text{ represents a Turing machine that is deterministic and accepts } x \text{ in at most } 2^t \text{ steps}\rbrace.
$$

$\textbf{Proposition 129:}$ The language $C_{\text{EXP}}$ is **EXP**-complete.

**Proof sketch.**

* **Membership in EXP:** We must show that $C_{\text{EXP}}$ can be recognized by a $2^{poly(n)}$-time-bounded deterministic TM, let's call it $M_{\text{decider}}$. On an input of length $n$ of the form $\tilde{w}01\tilde{x}01t$, $M_{\text{decider}}$ first checks if $w$ represents a deterministic TM. If not, it rejects. Otherwise, it simulates the computation of $M_w$ on input $x$ for up to $2^t$ steps. It accepts if $M_w$ reaches an accepting configuration within this step bound. The simulation of $2^t$ steps of $M_w$ can be done in at most $\lvert w \rvert \cdot 2^t < 2^{2n-1}$ steps of $M_{\text{decider}}$. For some polynomial $p$, the total computation time on an input of length $n$ is at most $p(n) + 2^{2n-1}$, which is at most $2^{2n}$ for sufficiently large $n$. This is within the class **EXP**.
* **EXP-hardness:** Let $L$ be any language in **EXP**. By Remark 128, there is a polynomial $p$ such that $L$ is recognized by a $2^{p(n)}$-time-bounded deterministic special Turing machine, represented by a word $w$. Then $L$ is $p$-$m$-reducible to $C_{\text{EXP}}$ via the function $x \mapsto \tilde{w}01\tilde{x}01p(\lvert x\rvert)$. ∎

**An NP-Complete Language**

A similar construction yields an $\textbf{NP}$-complete language. Let

$$
C_{\text{NP}} = \lbrace \tilde{w}01\tilde{x}01t : \text{the word } w \text{ represents a Turing machine that on input } x \text{ has an accepting computation of length at most } t\rbrace.
$$

$\textbf{Proposition 130:}$ The language $C_{\text{NP}}$ is **NP**-complete.

**A PP-Complete Language**

This construction can be extended to probabilistic classes like **PP**. Here, we consider probabilistic Turing machines with an auxiliary tape. We define **pruning** a computation to length $t$ as a variant of the machine where a computation accepts if and only if an accepting configuration is reached within the first $t$ steps. Let

$$
C_{\text{PP}} = \lbrace \tilde{w}01\tilde{x}01t : \text{the word } w \text{ represents a probabilistic Turing machine that accepts input } x \text{ when all computations are pruned to length } t\rbrace.
$$

$\textbf{Proposition 131:}$ The language $C_{\text{PP}}$ is $\textbf{PP}$-complete.

*(Note: The proof is omitted here but was covered in the lecture.)*


## Interactive Proof Systems

[Some additional notes on Interactive Proof Systems](/subpages/computability-and-complexity/interactive-proof-systems/)

### Verifiers and Provers

An **interactive proof system** provides a framework for establishing the truth of a mathematical statement through a structured dialogue. It involves two distinct computational entities: a **prover** and a **verifier**. These parties communicate by exchanging messages regarding a common input string, with the goal of having the verifier decide whether the input belongs to a specific language.
The prover is conceptualized as a computationally unrestricted entity, capable of performing any calculation necessary to persuade the verifier. In contrast, the verifier is a computationally bounded machine—specifically, a probabilistic Turing machine that must operate within polynomial time. The verifier uses randomness to pose challenges to the prover and to evaluate its responses. The messages exchanged are polynomially bounded in the length of the input.

An interactive proof system establishes membership in a language through a dialogue between a computationally bounded verifier and a computationally unbounded prover. The verifier is a probabilistic Turing machine that runs in polynomial time, while the prover can perform arbitrary computation. Messages are polynomially bounded in the input size.

$\textbf{Definition 132 (Verifiers and provers):}$ A *message function* maps a tuple $(w, m_1, \ldots, m_i)$ of binary words to a binary word of length at most $p(\|w\|)$ for some fixed polynomial $p$. A *prover* is a message function. A *verifier* is a polynomial-time probabilistic Turing machine $M$ with special accepting and rejecting states that computes a message function depending on its random sequence.

For a verifier $V$ and a prover $P$, the messages exchanged on input $w$ and random word $r$ are $m_1, m_2, \ldots$ with

$$
m_1 = V^r(w), \quad m_2 = P(w, m_1), \quad m_3 = V^r(w, m_1, m_2), \ldots
$$

The interaction terminates at round $t$ if $t$ is minimal such that $V$ reaches either its accepting or rejecting state on computing $V^r(w, m_1, \ldots, m_{2t})$. $V^r(w,m_1,\dots,m_i)$ refers to the message that $V$ computes on input tuple $(w,m_1,\dots,m_i)$ and random word $r$. **Communication** is a finite sequence $m_1,\dots,m_{2t}$.

Let $t : \mathbb{N} \to \mathbb{N}$. A verifier $V$ is **total** and **$t(n)$-bounded** if for all inputs $w$, all random words, and all provers $P$, the interaction terminates by round $t(\|w\|)$.

Key consequences:

* Message lengths, **number of rounds**, and verifier computation per message are polynomially bounded in $\|w\|$.
* The messages sent by the prover do not depend on the random word of the verifier (the prover cannot see the verifier’s random word); its replies depend only on the input and previous messages.
* Any total verifier is $t(n)$-bounded for some computable $t$ (by bounding the computation tree depth).

> Q: In interactive proof systems, if we say that we have some polynomially bounded verifier $V$ that interactively recognizes a language $L$, does this polynomial bound mean that each interaction (round) of the verifier's execution is polynomially bounded, or is it a global bound for all rounds (the verifier's running time over all rounds is bounded by some polynomial)? 
> A: It is a global bound on the verifier's total running time (as a machine) over the entire interaction $\implies$ the number of rounds is at most polynomial in $n=\|w\|$.

$\textbf{Definition 133:}$ A language $L$ is recognized (interactively) by a total verifier $V$ if

1. For each $w \in L$, there **exists** a prover such that $V$ accepts with probability at least $\tfrac{2}{3}$.
2. For each $w \notin L$, **for all provers**, $V$ rejects with probability at least $\tfrac{2}{3}$.

Let $k \in \mathbb{N}$. The class $\text{IP}[k]$ contains all languages recognized by some $k$-bounded verifier; $\text{IP}$ contains those recognized by some polynomially bounded verifier.

Completeness is condition (1); soundness is condition (2). The $\tfrac{2}{3}$ bound can be amplified arbitrarily close to $1$ via repetition. It is immediate that $\text{BPP} = \text{IP}[0]$ and $\text{NP} \subseteq \text{IP}[1]$.

> Proving **completeness** we are allowed (and actually expected) to look for the best possible prover strategy. Proving **soundness** we are expected to consider the most malicious prover.

$\textbf{Remark 134:}$ If $L$ is recognized by a verifier $V$. Then for each input in $L$ some prover must result in an acceptance probability of at least $\frac{2}{3}$, where this prover *may depend on the input*. It can be shown, however, that there is a single prover $P$ *independent of the input* that yields acceptance probability at least $\tfrac{2}{3}$ for all $w \in L$. The pair $(V, P)$ is an interactive proof system for $L$. Remark 139 below shows that $P$ can be chosen polynomial-space-bounded.

### An Interactive Proof for Graph Non-Isomorphism

We adopt a fixed node-labeling convention.

Convention 136: All graphs with $n$ nodes have the node set $\lbrace 1, \ldots, n\rbrace$.

$\textbf{Definition 135:}$ Let $G = (V, E)$ and $G' = (V', E')$ be graphs. An *isomorphism* between graphs is a bijection $\pi : V \to V'$ such that $(u, v) \in E$ iff $(\pi(u), \pi(v)) \in E'$. Graphs are *isomorphic* if such a $\pi$ exists.

Languages:

* $\text{GI} = \lbrace (G, G') \mid G \text{ and } G' \text{ are isomorphic}\rbrace$
* $\text{GNI} = \lbrace (G, G') \mid G \text{ and } G' \text{ are nonisomorphic}\rbrace$

The language $\text{GI}$ is in $\text{NP}$ since for isomorphic graphs, it can be checked in deterministic polynomial time whether a given mapping between their node sets is an isomorphism. The problem $\text{GI}$ is one of the few languages in $\text{NP}$ arising naturally in practice that is neither known to be $\text{NP}$-complete nor known to be in $\text{P}$. The language $\text{GNI}$ is not known to be in $\text{NP}$.

$\textbf{Theorem 137:}$ $\text{GNI} \in \text{IP}[1]$.

*Proof.* Let the input be $(G_0, G_1)$ with the same number of nodes (otherwise reject immediately). The protocol is a single round:

1. **Verifier:** Choose $i \in \lbrace 0,1\rbrace$ uniformly and a random permutation $\pi$ of the nodes. Compute $H = \pi(G_i)$ and send $H$.
2. **Prover:** Return a bit $j'$ claiming whether $H$ is isomorphic to $G_{j'}$.
3. **Verifier:** Accept iff $j' = i$.

Completeness: If $(G_0, G_1) \in \text{GNI}$, $H$ is isomorphic to exactly one of $G_0$ or $G_1$. An unbounded prover can identify which and always answer correctly, so acceptance probability is $1$.

Soundness: If $(G_0, G_1) \notin \text{GNI}$, they are isomorphic, so $H$ reveals no information about $i$. The prover can do no better than guessing; acceptance probability is $\tfrac{1}{2}$. To meet the soundness requirement of $\ge \frac{2}{3}$ rejection probability, the protocol can be repeated twice in parallel.  The verifier accepts only if the prover answers correctly for both independent interactions. The probability of the prover succeeding twice is $\frac{1}{2} \times \frac{1}{2} = \frac{1}{4}$. Thus, the rejection probability is $1−\frac{1}{4}=\frac{3}{4}$, which satisfies the condition. Since the repetitions can be done in parallel, the protocol remains 1-bounded. $\square$

### The Equivalence of $\text{IP}$ and $\text{PSPACE}$

A landmark result in complexity theory is the characterization of $\textbf{IP}$ as being exactly equal to the class of problems solvable in polynomial space, $\textbf{PSPACE}$.

#### Proving $\text{IP}$ is a Subset of $\text{PSPACE}$

We first show that any problem with an interactive proof can be solved by a polynomial-space Turing machine.

$\textbf{Theorem 138:}$ $\text{IP} \subseteq \text{PSPACE}$.

*Proof.* Let $L \in \text{IP}$ with polynomially bounded verifier $V$ (bound $p$). On input $w$ of length $n$, the interaction can be represented as a game tree $T$ of depth $2p(n)$ whose nodes encode partial transcripts $m_1 \ldots m_t$. Each node has at most $2^{p(n)}$ children (all possible next messages).

Let $k(u)$ be the number of verifier random strings that lead to partial communication $u$ and eventual acceptance assuming an optimal prover from there. Then $w \in L$ iff $k(z) \ge \tfrac{2}{3} \cdot 2^{p(n)}$ for the root $z$ (root $z$ is the initial message computation: $z = m_1 = V^r(w)$).

Compute $k(u)$ recursively in polynomial space:

* If $u$ is at even depth (verifier turn, we count depth from 0), $k(u) = \sum_{v \text{ child of } u} k(v)$.
* If $u$ is at odd depth (prover turn), $k(u) = \max_{v \text{ child of } u} k(v)$.

Leaves can be evaluated by simulating $V$ on all $2^{p(n)}$ random strings using polynomial space. A DFS reuses space, so the overall computation fits in $\text{PSPACE}$. 

We show by induction on $j$ (from the leaves up) that for any node $u$ at depth $2p(n) - j$, $k(u)$ can be computed in polynomial space.

* **Base Case** $(j=0)$: The node $u$ is a leaf. To compute $k(u)$, we can iterate through all $2^{p(n)}$ possible random words. For each word, we simulate the verifier to check if it generates the verifier messages in $u$'s label and if it ultimately accepts. This requires space to store the current random word $(p(n))$, a counter (polynomial in $n$), and space to simulate $V$ (also polynomial).
* **Inductive Step** $(j>0)$: For an internal node $u$, we recursively compute $k(v)$ for each of its children $v$. By the induction hypothesis, each of these computations requires polynomial space. Since we can reuse the space for each recursive call, the total space needed is the space for one child's computation plus the space to manage the summation or maximization, which remains polynomial.

Since the entire computation can be performed in polynomial space, $L \in \text{PSPACE}$. Therefore, $\text{IP} \subseteq \text{PSPACE}$. $\square$

$\textbf{Remark 139:}$ Let $L$ be a language in $\text{IP}$, and let $V$ be a polynomially bounded verifier that recognizes $L$. Then there is a polynomially space-bounded prover $P$ such that the interactive proof system $(V,P)$ recognizes $L$, i.e., $V$ accepts all inputs in $L$ with probability at least $\frac{3}{2}$ when interacting with $P$. This follows because the optimal prover described in the proof of Theorem 138 makes decisions by computing the function $k(u)$, which we have shown can be done in deterministic polynomial space.

#### Proving PSPACE is a Subset of IP

We arithmetize logical formulas to design interactive proofs for $\text{PSPACE}$-complete problems.

$\textbf{Definition 140:}$ The counting $3$-satisfiability problem is

$$
\#3\text{-SAT} = \lbrace (\phi, k) : \phi \text{ is a 3-CNF formula with exactly } k \text{ satisfying assignments}\rbrace.
$$

$\textbf{Theorem 141:}$ #$3\text{-SAT} \in \text{IP}$.

*Proof*: We design an interactive protocol where a prover convinces a verifier that a 3-CNF formula $\phi$ with $n$ variables has exactly $k$ satisfying assignments.

<div class="gd-grid">
  <figure>
    <img src="{{ 'assets/images/notes/computability-and-complexity/Arithmetization.png' | relative_url }}" alt="a" loading="lazy">
    <figcaption>Arithmetization (playing with NotebookLM)</figcaption>
  </figure>
</div>

**Step 1: Arithmetization**

We map the formula $\phi$ to a polynomial $p_\phi(x_1, \ldots, x_n)$ over a field $\mathbb{F}$. A literal $X_i$ is mapped to the variable $x_i$, and a literal $\neg X_i$ is mapped to $(1-x_i)$. A clause, which is a disjunction of literals, is mapped to a polynomial such that it evaluates to $0$ if the clause is false and non-zero if it's true. For example, a clause $(L_1 \vee L_2 \vee L_3)$ becomes $1 - (1-p_{L_1})(1-p_{L_2})(1-p_{L_3})$. Assuming all clauses have exactly three literals, a clause such as $(\neg X_1 \vee X_4 \vee \neg X_7)$ is mapped to $1 - x_1(1-x_4)x_7$.

The entire formula $\phi = C_1 \wedge \ldots \wedge C_m$ is mapped to the product of the polynomials for its clauses:

$$p_\phi(x_1, \ldots, x_n) = \prod_{j=1}^m p_{C_j}(x_1, \ldots, x_n)$$

For any assignment of $\lbrace 0, 1\rbrace$ values to the variables $x_1, \ldots, x_n, p_\phi$ evaluates to $1$ if the assignment satisfies $\phi$, and $0$ otherwise. Consequently, the number of satisfying assignments, $k_\phi$, is given by the sum over all possible boolean assignments:

$$k_\phi = \sum_{x_1 \in \{0,1\}} \sum_{x_2 \in \{0,1\}} \cdots \sum_{x_n \in \{0,1\}} p_\phi(x_1, x_2, \ldots, x_n) \quad (5.2)$$

**Step 2: The Checksum Protocol**

The verifier needs to check if the given integer $k$ equals $k_\phi$. This is done iteratively. The protocol operates over a large finite field $\mathbb{F}_p$. Let's define a sequence of polynomials:

$$h_i(x_i) = \sum_{x_{i+1}\in\{0,1\}} \cdots \sum_{x_n\in\{0,1\}} p_\phi(r_1, \ldots, r_{i-1}, x_i, x_{i+1}, \ldots, x_n) $$

where $r_1, \ldots, r_{i-1}$ are values chosen in previous rounds. These definitions imply the following identities:

1. $k_\phi = h_1(0) + h_1(1)$
2. $h_i(r_i) = h_{i+1}(0) + h_{i+1}(1)$ for $i=1, \ldots, n-1$
3. $h_{n-1} = p_\phi(r_1, \ldots, r_{n-1}, 0) + p_\phi(r_1, \ldots, r_{n-1}, 1)$

The protocol proceeds in $n$ rounds. The verifier checks an initial claim from the prover and then, in each round, reduces a claim about a polynomial in $j$ variables to a claim about a polynomial in $j-1$ variables evaluated at a random point.

For technial reasons related to the verification of the construction of $V$, choose a constant $c$ such that $72n^4 \le 2^{cn}$ holds for all $n$. We then have for all $p \ge 2^{cn}$,

$$\dfrac{nd}{p} \le \dfrac{3mn}{2^{cn}} \le \dfrac{24n^4}{2^{cn}} \le \dfrac{1}{3}$$

where the second to last inequality follows because a formula in $3$-CNF in $n$ variables has not more than $8n^3$ mutually distinct clauses.

* **Setup**: On input $(\phi, k)$, the prover sends a large prime $p$. The verifier checks that $p$ is sufficiently large (e.g., $2^{cn+1} > p > 2^{cn}$ for some constant $c$) and rejects if not. All subsequent calculations are modulo $p$.
* **Round 1**: The prover sends a polynomial $h_1^P(x_1)$, claiming it is equal to $h_1(x_1)$. The verifier checks if $k \equiv h_1^P(0) + h_1^P(1) \pmod p$. If this check fails, the verifier rejects. Otherwise, it chooses a random value $r_1 \in \mathbb{F}_p$ (uniformly) and sends it to the prover. The prover must now convince the verifier that $h_1^P(r_1)$ is the correct value of $h_1(r_1)$. (Also in all rounds the verifier checks that the polynomial $h_1^P$ is of degree of $d$ and rejects if it is not.)
* **Round $i$ $(2 \le i \le n-1)$**: The prover sends a polynomial $h_i^P(x_i)$, claiming it equals $h_i(x_i)$ (with previous random values $r_1, \dots, r_{i-1}$ fixed). The verifier checks if the previous round's value is consistent: $h_{i-1}^P(r_{i-1}) \equiv h_i^P(0) + h_i^P(1) \pmod p$. If not, it rejects. Otherwise, it picks a new random value $r_i \in \mathbb{F}_p$ and sends it to the prover.
* **Round $n$ (Final Check)**: After round $n-1$, the verifier has values $r_1, \ldots, r_{n-1}$ and the prover's last claim $h_{n-1}^P(r_{n-1})$. The verifier can now compute the true value of $h_n(0) + h_n(1)$ by itself, which is $p_\phi(r_1, \ldots, r_{n-1}, 0) + p_\phi(r_1, \ldots, r_{n-1}, 1)$. It accepts if and only if the prover's claim matches this computed value. Checking whether $h^{P}\_{n-1}=h_{n-1}$ in the last round $n$ is equivalent to checking 

$$h^{P}_{n-1} = p_\phi(r_1, \ldots, r_{n-1}, 0) + p_\phi(r_1, \ldots, r_{n-1}, 1),$$

hence can actually be done by the verifier $V$ itself. 

**Step 3: Analysis**

* *Completeness*: If $k = k_\phi$, an honest prover can send the true polynomials $h_1, \ldots, h_{n-1}$. All checks will pass, and the verifier will accept with probability $1$.
* *Soundness*: If $k \ne k_\phi$, but the prover sends $h^P_{1}$ such that , then the prover must lie at some point. Suppose the first incorrect polynomial sent is $h_i^P \ne h_i$. This means that $h_{i-1}^P(r_{i-1})$ (which was correct by assumption) does not equal $h_i^P(0) + h_i^P(1)$. If the prover tries to maintain consistency, then $h_i^P$ must be a different polynomial from $h_i$. The verifier will only fail to detect this lie if its random choice $r_i$ happens to be a root of the non-zero polynomial $(h_i^P - h_i)$. A polynomial of degree $d$ has at most $d$ roots. The degree of $p_\phi$ is at most $d=3m$. The probability of picking such a root is at most $d/p$. By choosing $p$ to be sufficiently large, this probability can be made very small. Summing over all rounds, the total probability of fooling the verifier is low. $\square$

<div class="gd-grid">
  <figure>
    <img src="{{ 'assets/images/notes/computability-and-complexity/soundness_sketch.jpeg' | relative_url }}" alt="a" loading="lazy">
    <figcaption>Primarily soundness sketch</figcaption>
  </figure>
</div>


Great explanation of soundness:
<div class="accordion">
  <details markdown="1">
    <summary>Soundness</summary>

Here is an explanation of the **soundness proof** for the theorem stating that **$3$-SAT is in $\text{IP}$**.

The proof you provided describes the **Sumcheck Protocol**, a fundamental technique in interactive proofs. 

We can break the logic down into three key concepts: **The Initial Lie**, **The Forced Hand**, and **The Mathematical Trap**.

**1. The Core Logic: "The Lie Must Propagate"**
The soundness argument relies on a chain reaction. If the Prover starts with a false claim, they are forced to lie in every subsequent round to maintain consistency. Eventually, this chain of lies leads to a direct mathematical contradiction that the Verifier can easily check.

Here is the step-by-step breakdown:

**A. The Initial Lie (Round 1)**
Imagine the true number of satisfying assignments is $k_{\phi}$, but the dishonest Prover claims it is $k$ (where $k \neq k_{\phi}$).
 * The Prover sends a polynomial $h_1^P(x)$.
 * The Verifier checks: Does $h_1^P(0) + h_1^P(1) = k$?
 * If the Prover sends the true polynomial $h_1$, the sum would be $k_{\phi}$, and the check would fail immediately (since $k \neq k_{\phi}$).
 * **Therefore**: To pass this first check, the Prover **must** send a fake polynomial $h_1^P$ that is different from the true polynomial $h_1$.

**B. The Random Challenge (The Trap)**
The Verifier now picks a random value $r_1$ from the field $\mathbb{F}_p$ and asks the Prover to prove the value of the next round based on this $r_1$.
The "trap" relies on a fundamental property of polynomials:

> **Fact**: Two distinct polynomials of degree $d$ can equal each other at no more than $d$ points.

Since $h_1^P$ (the fake polynomial) and $h_1$ (the true polynomial) are different, they intersect at very few points relative to the huge size of the field $\mathbb{F}_p$.
 * **Scenario A (Bad for Verifier)**: The Verifier unluckily picks one of the few $r_1$ values where $h_1^P(r_1) = h_1(r_1)$. The Prover "gets away" with the lie because the values match.
 * **Scenario B (Good for Verifier)**: The Verifier picks any other $r_1$. Now, $h_1^P(r_1) \neq h_1(r_1)$.

**C. The Forced Hand (Recursion)**
If **Scenario B** happens (which is highly likely), the Prover is now stuck.
 * The protocol moves to Round $2$. The Prover must now prove that the sum for the next variable equals $h_1^P(r_1)$.
 * But we know $h_1^P(r_1)$ is the **wrong** value (it is not equal to the true value $h_1(r_1)$).
 * Just like in Round 1, to match this "wrong" target value, the Prover must send another fake polynomial $h_2^P$ that differs from the true $h_2$.
 * **The Result**: The lie propagates from round $i$ to round $i+1$.

**2. The Final Check (Round $n$)**
This game continues until the very last variable. If the Verifier has not been "unlucky" in any of the rounds, the Prover is forced to make a claim about the final value $h_{n-1}^P(r_{n-1})$ that is **false**.

At this point, there are no more sums to check. The Verifier can compute the true value directly using the original formula:

$$\text{True Value} = p_{\phi}(r_1, \dots, r_{n-1}, 0) + p_{\phi}(r_1, \dots, r_{n-1}, 1)$$

Because the lie propagated all the way to the end, the Prover's claimed value will not match this calculated true value. The Verifier sees the mismatch and **REJECTS**.

**3. The Probability Analysis (Why it works)**
The only way the Prover can cheat is if the Verifier gets "unlucky" and picks a random $r_i$ that happens to be a root of the difference polynomial ($h_i^P - h_i$).
 * **Max degree of polynomial**: The degree $d$ is bounded by $3m$ (roughly the size of the formula).
 * **Field size**: $p$ is chosen to be exponentially large ($2^{cn}$).
 * **Probability of error in ONE round**: The chance that a random $r_i$ is a "bad" point is at most $\frac{d}{p}$.
Since there are $n$ rounds, we sum the probabilities of failure (Union Bound). We are calculating the probability that the Verifier gets unlucky in Round $1$ ($E_1$) OR Round $2$ ($E_2$) OR $\dots$ OR Round $n$ ($E_n$) (also known as the Union Bound):

$$\text{Total Probability of Error} = P(E_1 \cup \dots \cup E_n) \le \sum_{i=1}^n \dfrac{d}{p} = \dfrac{nd}{p}$$

Because $p$ is exponentially large ($2^{cn}$) and $nd$ is polynomial in size $n \cdot 3m$, the fraction $\frac{nd}{p}$ is extremely close to $0$.
**Conclusion**: It is overwhelmingly likely that the Verifier will detect the lie, making the protocol **Sound**.
  </details>
</div>

<div class="accordion">
  <details markdown="1">
    <summary>Why Round 1 has two checks</summary>

The verification steps in Round 1 of the Checksum Protocol (Step 2 of the proof of Theorem 141) serve two distinct and essential purposes in the interactive proof for $\#3$-SAT, ensuring both **initial consistency** and **soundness** against a cheating prover.

The reason the verifier performs two seemingly related checks—one involving the evaluation of $h_1^P$ at boolean points (0 and 1) and one involving a random point ($r_1$)—is that the core strategy of this interactive proof relies on confirming the global correctness of the polynomial $h_1^P(x_1)$ over a large finite field $\mathbb{F}_p$.

### 1. Verification of Consistency (The Check at 0 and 1)

The first check verifies that the polynomial $h_1^P(x_1)$ provided by the prover is consistent with the initial claim that the formula $\phi$ has exactly $k$ satisfying assignments.

* **The Claim:** The definition of the polynomials $h_i(x_i)$ guarantees that the total number of satisfying assignments, $k_\phi$, is equal to the sum of $h_1(0)$ and $h_1(1)$.
* **The Check:** The verifier checks if **$k \equiv h_1^P(0) + h_1^P(1) \pmod p$**.
* **Purpose:** This check is necessary to maintain the connection to the original problem: if the prover submits the correct polynomial $h_1$, the sum of its values at 0 and 1 must equal $k$. If this check fails, the verifier rejects immediately.

### 2. Verification of Soundness (The Check at $r_1$)

After the initial consistency check passes, the verifier must determine if the polynomial $h_1^P(x_1)$ submitted by the prover is actually the true polynomial $h_1(x_1)$, or if it is a different, "cheating" polynomial that just happens to agree with the required sum $k$ at $\{0, 1\}$.

* **The Action:** The verifier chooses a **random value $r_1$** from the large finite field $\mathbb{F}_p$ and sends it to the prover.
* **The Goal:** The prover must subsequently convince the verifier that the claimed value $h_1^P(r_1)$ is the correct value of the true polynomial $h_1(r_1)$. This value $h_1^P(r_1)$ will then be used as the anchor for the consistency check in the subsequent round.
* **Purpose:** This randomized spot-check is the basis of the protocol's **soundness**. If the prover had submitted an incorrect polynomial $h_1^P$ (i.e., $h_1^P \ne h_1$), the polynomial representing the difference, $(h_1^P - h_1)$, is non-zero. Since a polynomial of degree $d$ (which is polynomial in $n$) has at most $d$ roots, the probability that the randomly chosen value $r_1$ happens to be one of the few points where $h_1^P(r_1) = h_1(r_1)$ is very small (at most $d/p$) when $p$ is chosen to be sufficiently large.

By checking the polynomial at the specific boolean inputs (0 and 1) and then verifying its value at a random, non-boolean input ($r_1$), the verifier ensures that the prover cannot hide an incorrect polynomial without a high probability of being caught. The verification process is essentially transforming a complex, global check (is $h_1^P$ identical to $h_1$?) into a local check (is $h_1^P$ correct at the random point $r_1$?) which carries a high probability of error detection due to the properties of polynomials over fields.
  </details>
</div>

<div class="accordion">
  <details markdown="1">
    <summary>What is the point of choosing value $r_i$ randomly?</summary>
The point of choosing the value $r_i$ randomly from the large finite field $\mathbb{F}_p$ in the Checksum Protocol (Step 2 of the proof of Theorem 141 for \#3-SAT) is to ensure the **soundness** of the interactive proof system. This randomization is the core mechanism that catches a dishonest prover.

Here is a comprehensive breakdown of the purpose of randomizing $r_i$:

### 1. Ensuring Soundness Against Cheating Provers

The primary goal of using randomness in the verifier's challenges is to guarantee that if the initial statement is false (i.e., if $k \ne k_\phi$), no prover, regardless of its computational power, can convince the verifier of the statement's truth with a high probability.

In the Checksum Protocol, the prover sends a claimed polynomial $h_i^P(x_i)$. If this polynomial is not the true polynomial $h_i(x_i)$, the prover is lying.

*   If $h_i^P(x_i)$ is incorrect, the difference between the true and claimed polynomial, $(h_i^P - h_i)$, is a non-zero polynomial.
*   A non-zero polynomial of degree $d$ (which is polynomial in $n$ for the Checksum Protocol) has at most $d$ roots.
*   When the verifier chooses a random $r_i$ from the large finite field $\mathbb{F}_p$, the probability that $r_i$ happens to be a root of the difference polynomial—meaning $h_i^P(r_i) = h_i(r_i)$ even though $h_i^P \ne h_i$—is very small, specifically at most $d/p$.

By choosing $p$ to be sufficiently large, this probability is made negligible, thereby ensuring that if the prover lies, the value $h_i^P(r_i)$ used in the subsequent round (Round $i+1$) will almost certainly be incorrect, leading to rejection in Round $i+1$.

### 2. Transforming Global Identity into Local Spot-Checks

The overall problem is checking if the provided polynomial $h_i^P$ is identical to the true polynomial $h_i$ over the entire field, which is difficult for a polynomial-time verifier. Randomization transforms this difficult global check into a sequence of manageable, probabilistic local checks:

*   The verifier does not need to check the entire structure of the polynomial $h_i^P(x_i)$.
*   Instead, the verifier focuses only on the claimed value $h_i^P(r_i)$ at a single, random point $r_i$. If the polynomial $h_i^P$ is incorrect, it is highly probable that the evaluation $h_i^P(r_i)$ will also be incorrect.

This process reduces a claim about a high-degree, multi-variable polynomial $p_\phi$ into a set of probabilistic checks on single-variable polynomials across $n$ rounds.

### 3. Maintaining Privacy and Independence

In the general framework of interactive proofs, the verifier uses randomness ("coin flips") to pose challenges and evaluate responses, and these choices are private to the verifier (though public-coin systems exist).

The random selection of $r_i$ ensures that the prover cannot predict the exact point the verifier will use to anchor the next round's verification. The prover's messages are a function of the input and prior messages, but not the verifier's secret random word. This prevents a cheating prover from constructing a fake polynomial $h_i^P$ that specifically agrees with $h_i$ only at predictable points.
  </details>
</div>

<div class="accordion">
  <details markdown="1">
    <summary>I do not understand why in the proof we evaluate the polynomial $h$ in random points, that are not binary. $p_{\phi}(r_1, ..., r_{n-1}, 0)$ or $p_{\phi}(r_1, ..., r_{n-1}, 1)$ can give us values that is different from $0$ or $1$, because $r_i$ are not necessary binary and we evaluate arithmetised formula in non binary values</summary>
This is a fundamental and excellent question about the core mechanism of the Checksum Protocol (Theorem 141) and the arithmetization technique used to prove PSPACE $\subseteq$ IP.

The reason the verifier evaluates the arithmetized formula $p_\phi$ at non-binary, random points ($r_1, \ldots, r_{n-1}$) and the resulting values are often non-zero and non-one is that **the protocol shifts the problem from checking boolean logic to checking polynomial identity over a large finite field $\mathbb{F}_p$**.

Here is a breakdown drawing on the sources:

### 1. The Role of Arithmetization and Boolean Input

The arithmetization step (Step 1 of Theorem 141) maps the boolean formula $\phi$ to a polynomial $p_\phi(x_1, \ldots, x_n)$.

*   **Boolean Inputs:** When $p_\phi$ is evaluated on **boolean inputs** (i.e., $x_i \in \{0, 1\}$), it correctly represents the satisfiability of the formula: $p_\phi$ evaluates to $1$ if the assignment satisfies $\phi$, and $0$ otherwise.
*   **The Goal:** The number of satisfying assignments, $k_\phi$, is the sum of $p_\phi$ over all possible boolean assignments. The verifier needs to check if the given $k$ equals this sum.

### 2. The Shift from Boolean Logic to Field Arithmetic (Introducing $r_i$)

The interactive proof protocol is designed to check the value of this massive sum iteratively, which requires checking polynomial identities. This is where the non-binary, random values $r_i$ come into play.

*   The goal of the protocol is to verify that the polynomial $h_i^P(x_i)$ submitted by the prover is actually the true polynomial $h_i(x_i)$.
*   The only way a computationally bounded verifier can probabilistically check this identity is through **randomized spot-checking**.

The verifier does not choose $r_i$ from $\{0, 1\}$ because:

*   **Soundness Requirement:** To detect a cheating prover who has submitted a polynomial $h_i^P$ that is different from the true polynomial $h_i$ (i.e., $h_i^P \ne h_i$), the verifier must choose a random point $r_i$ from the large field $\mathbb{F}_p$. If the verifier only checked points in $\{0, 1\}$, a cheating prover could easily construct a polynomial that agrees with the true polynomial only at the few binary points but is incorrect everywhere else.
*   **Polynomial Identity Testing:** By choosing $r_i$ randomly from a large field $\mathbb{F}_p$, the probability that a non-zero polynomial $(h_i^P - h_i)$ evaluates to zero at $r_i$ is negligible, ensuring high probability of error detection (soundness).

### 3. Why $p_\phi(r_1, \ldots, r_{n-1}, 0)$ is Not Necessarily 0 or 1

You are absolutely correct that when the polynomial $p_\phi$ is evaluated at inputs that include non-binary values $r_i \in \mathbb{F}_p$, the output will generally be an element of $\mathbb{F}_p$ that is neither 0 nor 1.

In the final step of the protocol, the verifier computes the true value:
$$p_\phi(r_1, \ldots, r_{n-1}, 0) + p_\phi(r_1, \ldots, r_{n-1}, 1) \pmod p$$.

*   This value is **not** meant to be a boolean truth value (0 or 1). Instead, it is the true arithmetic value of the specific polynomial $h_{n-1}(r_{n-1})$, which itself is derived from the arithmetized formula $p_\phi$ evaluated at the fixed random points $r_1, \ldots, r_{n-1}$.
*   This calculated field element serves as the **final numerical value** that the prover must match to confirm that the chain of algebraic identities (checked over $n$ rounds) holds up to the very end.

In essence, the protocol uses $r_i$ to **algebraically enforce consistency** across rounds, treating the problem purely as one of polynomial arithmetic over $\mathbb{F}_p$, rather than checking boolean assignments.

***

The situation is analogous to a bank audit using sampling. To check if a massive company's accounts balance, an auditor doesn't re-examine every single receipt (the boolean assignments). Instead, the auditor performs a sequence of spot checks on randomly chosen transactions ($r_i$ values). If the books are faked, the random spot check will almost certainly reveal an inconsistency, even though the randomly chosen numbers used for the check might not correspond to standard transaction numbers. The integrity of the entire system is verified by confirming the identity of the underlying mathematical structure across those random samples.
  </details>
</div>

<div class="accordion">
  <details markdown="1">
    <summary>Do I understand correctly that the main point of evaluating the polynomial given by prover at random points over the field is to verify the given polynomial correct?</summary>

Yes, your understanding is **correct**. The main point of having the prover evaluate the polynomial $h_i^P$ at a random value $r_i$ chosen from the large finite field $\mathbb{F}_p$ is precisely to **verify the correctness of the given polynomial** $h_i^P(x_i)$.

This process is essential for establishing the **soundness** of the Checksum Protocol.

Here is a detailed explanation of why this randomized evaluation is necessary for verification:

### 1. Verification of Identity via Random Spot-Checking

The fundamental problem the verifier faces is determining if the submitted polynomial $h_i^P$ is identical to the true, intended polynomial $h_i$. Since the verifier is computationally bounded (it operates in polynomial time), it cannot check the polynomial at every possible point in the field $\mathbb{F}_p$.

Instead, the verifier uses a randomized challenge:

*   If the prover submits a polynomial $h_i^P$ that is truly correct ($h_i^P = h_i$), then $h_i^P(r_i)$ will be the correct value, and the protocol maintains its completeness (acceptance probability $\ge 2/3$).
*   If the prover submits an incorrect polynomial ($h_i^P \ne h_i$), the difference between the two polynomials, $(h_i^P - h_i)$, is a non-zero polynomial.

### 2. The Soundness Guarantee

The randomness of $r_i$ ensures that if the polynomial is incorrect, the prover will be caught with high probability, which is the definition of soundness.

*   A non-zero polynomial of degree $d$ has at most $d$ roots.
*   Since $r_i$ is chosen uniformly at random from the large field $\mathbb{F}_p$, the probability that the random value $r_i$ is one of the few points where the cheating polynomial accidentally evaluates correctly (i.e., where $h_i^P(r_i) = h_i(r_i)$) is very small, specifically $\le d/p$.
*   By ensuring the prime $p$ is sufficiently large, the total probability of fooling the verifier across all $n$ rounds is made low.

The evaluation of $h_i^P(r_i)$ transforms the complex global problem of checking polynomial identity into a probabilistic local problem: if the prover can satisfy the next round's consistency check using the value $h_i^P(r_i)$, the verifier is highly confident that $h_i^P$ was the correct polynomial.

In summary, choosing $r_i$ randomly is the method by which the verifier's limited computational power is amplified to ensure that the claimed polynomial is correct with a high degree of probability, fulfilling the requirements for an interactive proof system.
  </details>
</div>

<div class="accordion">
  <details markdown="1">
    <summary>Does the prover give us new polynomial at each round?</summary>

Yes, the prover gives the verifier a **new polynomial** in virtually every round of the Checksum Protocol (Step 2 of the proof of Theorem 141) for proving \#3-SAT.

The protocol is structured across $n$ rounds, corresponding to the $n$ variables of the formula $\phi$. In each round $i$, the prover submits a claim about a different intermediate polynomial in the summation process:

1.  **Round 1:** The prover sends the polynomial $\mathbf{h_1^P(x_1)}$, claiming it is equal to the true polynomial $h_1(x_1)$.
2.  **Round $i$ (where $2 \le i \le n-1$):** The prover sends a different polynomial, $\mathbf{h_i^P(x_i)}$, claiming it equals the true polynomial $h_i(x_i)$.

### Why the Polynomial Must Change

The necessity for a new polynomial in each round stems from the definition of the polynomial sequence $h_i(x_i)$. These polynomials are defined iteratively to reduce the complexity of the original $n$-variable sum:

$$h_i(x_i) = \sum_{x_{i+1}\in\lbrace 0,1\rbrace} \cdots \sum_{x_n\in\lbrace 0,1\rbrace} p_\phi(r_1, \ldots, r_{i-1}, x_i, x_{i+1}, \ldots, x_n) \quad$$

*   $h_1(x_1)$ is a polynomial only of the variable $x_1$.
*   $h_2(x_2)$ is a polynomial only of the variable $x_2$, with the variable $x_1$ having been fixed to the random value $r_1$ chosen in the previous round.
*   In general, $h_i(x_i)$ is a polynomial only of the variable $x_i$, with all preceding variables ($x_1, \ldots, x_{i-1}$) having been substituted with the random field elements ($r_1, \ldots, r_{i-1}$) chosen by the verifier in prior rounds.

Because the summation structure changes and new random field elements are incorporated in the definition of $h_i$ at each step, the polynomial $h_i^P(x_i)$ submitted by the prover must change accordingly to maintain consistency with the claim about $k$.

This process transforms the verification of a massive, $n$-variable polynomial identity into a sequence of polynomially many checks on single-variable polynomials.
  </details>
</div>

<div class="accordion">
  <details markdown="1">
    <summary>Do I understand correctly that we check all $h_i$ recursively just to verify that prover understand the recursive structure of polynomial $h_i$?</summary>

That is an interesting way to frame the goal of the Checksum Protocol, but based on the sources, the primary reason for checking the polynomials $h_i$ recursively is not to verify the prover's "understanding" of the structure. Instead, the recursive checks are performed to ensure the **mathematical consistency** and **soundness** of the claimed solution.

Here is a breakdown of why the recursive structure is used:

### 1. The Prover's Nature

The Checksum Protocol operates under the assumption that the **prover is a computationally unrestricted entity**. The prover is conceptualized as being capable of performing any calculation necessary to persuade the verifier. Therefore, there is no need to test if the prover "understands" the recursive definitions of $h_i$; an optimal prover is assumed to always know the correct definition and the correct resulting polynomials.

### 2. The Goal: Consistency and Completeness

The recursive structure is defined by the necessary mathematical identities that relate the summation of the polynomial $p_\phi$ across variables:

$$h_i(r_i) = h_{i+1}(0) + h_{i+1}(1) \quad \text{for } i=1, \ldots, n-1 \quad$$

The verifier uses this structure to perform local checks in each round:

*   **Initial Consistency (Round 1):** The first check ensures that the initial polynomial $h_1^P$ is consistent with the ultimate claim $k$, checking $k \equiv h_1^P(0) + h_1^P(1) \pmod p$.
*   **Recursive Consistency (Round $i$):** For subsequent rounds, the check ensures that the polynomial $h_i^P$ submitted in the current round satisfies the identity required by the claimed value from the previous round ($h_{i-1}^P(r_{i-1})$).

This recursive checking guarantees **completeness**: if $k$ truly equals the number of satisfying assignments $k_\phi$, an honest prover can send the true polynomials $h_1, \ldots, h_{n-1}$, and all these consistency checks will pass, leading to acceptance with probability 1.

### 3. The Main Point: Soundness via Recursive Spot-Checks

The true power of the recursion, combined with the random challenges $r_i$, is to enforce **soundness**.

The recursive structure transforms the hard problem of verifying the huge sum $k_\phi$ into a sequence of probabilistic checks on $n$ different, single-variable polynomials.

If the prover submits an incorrect polynomial $h_i^P$ in Round $i$ (a lie), the verifier detects this lie by relying on the random challenge $r_i$ chosen in that round.

*   The random value $r_i$ ensures that $h_i^P(r_i)$ is incorrect with high probability (unless $r_i$ is one of the few roots of the difference polynomial).
*   This incorrect value $h_i^P(r_i)$ then becomes the target value for the consistency check in the **next round ($i+1$)**.

By checking the $h_i$ recursively, the verifier ensures that a lie introduced at any point in the polynomial sequence will be caught in a subsequent consistency check, thereby guaranteeing that no cheating prover can maintain a false claim $k \ne k_\phi$ all the way to the final step without being rejected with high probability.
  </details>
</div>

<div class="accordion">
  <details markdown="1">
    <summary>I mean the $h_i$ has nested (recursive) structure. If we just rely on the polynomial $h^P_1$, then the sum $h^P_1(0)+h^P_1(1)$ could coincide with $k_{\phi}$ by chance. To make sure that the given $h^P_1$ is indeed the function $h_1$, we check it recursively for all $h^P_i=h_i$. And if it's true for all $i$, then it's likely that the prover gave us the correct $h^P_1$</summary>

You have precisely captured the motivation and mechanism behind the recursive structure of the Checksum Protocol used in the proof of Theorem 141 (that \#3-SAT is in IP).

You are **correct** that relying only on the initial check $k \equiv h_1^P(0) + h_1^P(1) \pmod p$ is insufficient because a cheating prover could submit a polynomial $h_1^P$ that happens to satisfy this sum even though $h_1^P \ne h_1$. This is the central vulnerability that the recursive checking and randomization address.

The protocol uses recursion and randomized challenges to ensure that the initial claimed polynomial $h_1^P$ is correct, not just at the binary points, but across the entire domain of the finite field $\mathbb{F}_p$.

Here is how your interpretation aligns with the principles detailed in the sources:

### 1. The Insufficiency of the Initial Check

The initial check in Round 1 verifies if the claimed number of satisfying assignments, $k$, is consistent with the polynomial $h_1^P$ at the boolean inputs $\lbrace 0,1\rbrace$:
$$k \equiv h_1^P(0) + h_1^P(1) \pmod p \quad$$
As you noted, a false polynomial $h_1^P$ might satisfy this equation by chance, or by construction, if the prover ensures $h_1^P(0) + h_1^P(1)$ equals $k$.

### 2. The Verification Goal: Checking Polynomial Identity

The ultimate objective of the entire $n$-round protocol is to verify that $h_1^P$ is the true polynomial $h_1$. This is achieved by transforming the global check ($h_1^P = h_1$) into a sequence of local, verifiable claims over the large finite field $\mathbb{F}_p$.

### 3. Recursive Checks Enforce Consistency and Structure

The recursive structure of the protocol forces the prover to prove that $h_1^P$ is algebraically correct by linking it to the subsequent polynomials $h_2^P, h_3^P, \ldots$ using the fundamental identity $h_{i-1}(r_{i-1}) = h_i(0) + h_i(1)$.

In each Round $i$ (where $i \ge 2$), the consistency check ensures that the values match based on the required recursive structure:
$$h_{i-1}^P(r_{i-1}) \equiv h_i^P(0) + h_i^P(1) \pmod p \quad \text{}$$

If the prover submits a false $h_1^P$, they must maintain the appearance of correctness throughout the subsequent rounds.

### 4. Soundness and Probability (Making it "Likely")

The recursive checking alone is not enough; it is the **random selection of $r_i$** from the large field $\mathbb{F}_p$ that makes it "likely" that $h_1^P$ is correct, thereby ensuring the protocol's **soundness**.

*   If the prover successfully maintains consistency up to Round $i$ despite $h_i^P \ne h_i$, the verifier chooses a random $r_i$.
*   The only way a false $h_i^P$ can continue to fool the verifier is if the randomly chosen $r_i$ happens to be a root of the difference polynomial $(h_i^P - h_i)$.
*   Since the number of roots is small relative to the size of the field $p$, the probability that the prover succeeds is very low.

Therefore, the recursive structure is essential because it:
1.  **Algebraically enforces** the correct definition of $h_i$ (derived from the summation $k_\phi$) across all variables $x_1$ through $x_n$.
2.  **Sets up the randomized spot-check:** By checking the value $h_i^P(r_i)$ against the structure of $h_{i+1}^P$, the verifier forces the prover to evaluate the claimed polynomial at a point chosen randomly from the large field, proving global identity with high probability.

If all these checks pass successfully for all $i$ up to the final verifiable check, the verifier can conclude with high confidence that the initial polynomial $h_1^P$ was indeed the correct polynomial $h_1$.
  </details>
</div>

$\textbf{Theorem 142:}$ $\text{IP} = \text{PSPACE}$.

*Proof.* Theorem 138 gives $\text{IP} \subseteq \text{PSPACE}$. For the reverse direction, show a $\text{PSPACE}$-complete language has an interactive proof. QBF in $3$-CNF form ($3$-QBF) is PSPACE-complete.

$\textbf{Theorem 143:}$ $3\text{-QBF} \in \text{IP}$.

*Proof:* We adapt the arithmetization protocol from Theorem 141 for quantified formulas. Let 

$$\Psi = \forall X_1 \exists X_2 \ldots \phi(X_1, \ldots, X_n)$$

On input $\Psi$, the verifier $V$ can immediately reject in case $\Psi$ is not in 3-CNF prenex normal form, hence we can assume otherwise. Let also $\phi$ has $m$ clauses, $p_\phi$ has the same meaning as in Theorem 141. We can arithmetize this structure. The universal quantifier $\forall X_i$ corresponds to a product ($\prod_{x_i \in \lbrace 0,1\rbrace}$), and the existential quantifier $\exists X_i$ corresponds to a logical OR, which can be arithmetized using the operation $\alpha * \beta = 1 - (1-\alpha)(1-\beta)$. Then $\Psi$ is true if and only if the following expression evaluates to $1$: 

$$\prod_{x_1\in\lbrace 0,1\rbrace} \coprod_{x_2\in\lbrace 0,1\rbrace} \cdots \coprod_{x_n\in\lbrace 0,1\rbrace} p_\phi(x_1, \ldots, x_n) = 1 \quad (5.9)$$

where $\coprod$ denotes the arithmetic OR operation. Here $\prod$ is the usual product of polynomials, and given a polynomial $p$ in variables $x_1,\dots,x_t$, we let

$$\cdots \coprod_{x_t\in\lbrace 0,1\rbrace} p_\phi(x_1, \ldots, x_t) = p_\phi(x_1, \ldots, x_{t-1}, 0) * p_\phi(x_1, \ldots, x_{t-1}, 1)$$

A direct application of the checksum protocol fails because each operator ($\prod$ and $\coprod$) can double the degree of the polynomials involved, leading to an exponential degree overall. The solution is to apply a linearization operator at each step. For a polynomial $p$, let 

$$L_i p(x_1, \ldots, x_n) = x_i p(\ldots, x_{i-1}, 1, x_{i+1}, \ldots) + (1-x_i) p(\ldots, x_{i-1}, 0, x_{i+1}, \ldots)$$

<div class="gd-grid">
  <figure>
    <img src="{{ 'assets/images/notes/computability-and-complexity/linearization_example.jpeg' | relative_url }}" alt="a" loading="lazy">
    <figcaption>Example of linearization</figcaption>
  </figure>
</div>

This new polynomial $L_i p$ agrees with $p$ on all boolean inputs $\lbrace 0,1\rbrace^n$, but the degree of $x_i$ in $L_i p$ is at most $1$. By applying these operators repeatedly, we can keep the degree of all variables low throughout the protocol. By construction, $L_1,\dots,L_n p(x_1,\dots,x_{i-1},0,x_i,\dots,x_n)$ all variables have degree at most $1$. Thus in order to verify that (5.9) is equal to $1$, the checksum protocol can be applied in order to verify that:

$$1 = \prod_{x_1\in\lbrace 0,1\rbrace} \underbrace{  L_1 \coprod_{x_2\in\lbrace 0,1\rbrace} \cdots \prod_{x_{n-1}\in\lbrace 0,1\rbrace} L_1 \dots L_{n-1}  \coprod_{x_n\in\lbrace 0,1\rbrace} p_\phi(x_1, \ldots, x_n)}_{= h_1(x_1)} \quad (5.10)$$

The verifier's protocol is modified to check the arithmetized QBF expression with linearization.

* **Round 1 and 2**: to convince the verifier that (5.10) holds, the prover sends a prime $p$ (for modulo $p$) and then the prover sends a polynomial $h^P_1$ that is meant to coincide with $h_1$ in (5.10). The verifier checks $h^P_1(0) \cdot h^P_1(1) = 1$. Rejects in case this fails and, otherwise, selects a member $r_1$ in $\mathbb{F}_p$ uniformly at random and sends it to prover. In order to convince the verifier that 
 
  $$h^P_1(r_1) = L_1 \underbrace{ \coprod_{x_2\in\lbrace 0,1\rbrace} \cdots \prod_{x_{n-1}\in\lbrace 0,1\rbrace} L_1 \dots L_{n-1}  \coprod_{x_n\in\lbrace 0,1\rbrace} p_\phi(x_1, \ldots, x_n)}_{= h_2(x_1)}$$

  is true, the verifier sends the polynomial $h^P_2$ that is meant to coincide with $h_2$. The verifier then checks whether $h^P_1(r_1)$ is equal to $r_1h^P_2(1) + (1−r_1)h^P_2(0)$, rejects in case this fails and, otherwise, selects $r_2$ in $\mathbb{F}_p$ uniformly at random and sends it to the prover, which now has to convince the verifier that $h^P_2(r_2)$ is equal to $h_2(r_2)$.
* Similarly, in case the first operator of the currently considered expression is $\coprod$, the verifier checks whether $h_{i−1}(r_{i−1})=h^P_i(0)∗h^P_i(1)$. The verification remains essentially the same as in the proof of Theorem 141, in particular, each random choice of a member of $\mathbb{F}_p$ is unfortunate with probability at most $\frac{d}{p}$ for some constant $d$. Observe, however, that now there are more such choices, their number is quadratic instead of linear in $n$, hence the value of $c$ has to be adapted accordingly.

Why quadratic instead of linear in $n$:

$\textbf{Example: } \prod_{x_1\in\lbrace 0,1\rbrace} L_1 \coprod_{x_2\in\lbrace 0,1\rbrace} L_1 L_2 \prod_{x_3\in\lbrace 0,1\rbrace} p(x_1,x_2,x_3)$

$\textbf{Solution:}$

$$
1
= \prod_{x_1\in\{0,1\}}
   \underbrace{
     L_1\,
     \underbrace{
       \coprod_{x_2\in\lbrace 0,1\rbrace}
       \underbrace{
         L_1 
         \underbrace{
            L_2 
            \underbrace{
            \prod_{x_3\in\lbrace 0,1\rbrace} p(x_1,x_2,x_3)
            }_{h_5}
         }_{h_4}
       }_{h_3}
     }_{h_2}
   }_{h_1}
$$

General template (optional, for the $n$-variable case):

$$
h^P_1(z) 
= L_1 \underbrace{ \coprod_{x_2\in\lbrace 0,1\rbrace} \cdots 
\prod_{x_{n-1}\in\lbrace 0,1\rbrace} L_1 \dots L_{n-1}  
\coprod_{x_n\in\lbrace 0,1\rbrace} p_\phi(z, x_2, \ldots, x_n)}_{= h_2(z)}
$$

$\textbf{Round 1 (Operator: $\prod_{x_1}$)}$

Check: $1 \stackrel{?}{=} h^P_1(0)\cdot h^P_1(1)$

Action: If true, Verifier picks random $r_1$. New target is $h^P_1(r_1)$.

$\textbf{Round 2 (Operator: $L_1$)}$

$h^P_1(r_1) \stackrel{?}{=} r_1 \cdot h^P_2(1) + (1-r_1) \cdot h^P_2(0)$

Note: This uses the Linearization formula; $r_1$ acts as the weight.

Action: If true, Verifier picks a NEW random $r'_1$. New target is $h^P_2(r'_1)$.

$\textbf{Round 3 (Operator: $\coprod_{x_2}$)}$

Check: $h^P_2(r'_1) \stackrel{?}{=} 1 - (1-h^P_3(0))(1-h^P_3(1))$

Note: This uses the arithmetic OR formula $(a \lor b = 1-(1-a)(1-b))$ for $a,b\in\{0,1\}$.

Action: If true, Verifier picks random $r_2$. New target is $h^P_3(r_2)$.

$\textbf{Round 4 (Operator: $L_1$)}$

Check: $h^P_3(r_2) \stackrel{?}{=} r'_1 \cdot h^P_4(1) + (1-r'_1) \cdot h^P_4(0)$

Note: This linearizes again in $x_1$;~ $r'_1$ is the current point for $x_1$.

Action: If true, Verifier picks a NEW random $r''_1$. New target is $h^P_4(r''_1)$.

$\textbf{Round 5 (Operator: $L_2$)}$

Check: $h^P_4(r''_1) \stackrel{?}{=} r_2 \cdot h^P_5(1) + (1-r_2) \cdot h^P_5(0)$

Note: This linearizes in $x_2$;~ $r_2$ is the current point for $x_2$.

Action: If true, Verifier picks a NEW random $r'_2$. New target is $h^P_5(r'_2)$.

$\textbf{Round 6 (Operator: $\prod_{x_3}$)}$

Check: $h^P_5(r'_2) \stackrel{?}{=} h^P_6(0)\cdot h^P_6(1)$

Note: This is the product check for $x_3 \in \{0,1\}$.

Note: Actually we do not need to introduce $h^P_6$ explicitly, because  $h^P_6(x)=p(r''_1,r'_2,x)$.

Action: If true, Verifier picks random $r_3$. New target is $h^P_6(r_3)$.

$\textbf{Final Check}$

Verifier directly evaluates $p(r''_1, r'_2, r_3)$ and checks that $p(r''_1, r'_2, r_3) \stackrel{?}{=} h^P_6(r_3)$.

$\textbf{Note: }$ There is the same amount of computation per round, and the Linearization is made in general by the same template.


### Public Coins and Perfect Completeness

Interactive proofs can be made public-coin (all verifier randomness revealed) and perfectly complete (accept with probability $1$ on yes-instances).

Verifiers as defined in Definition 132 use **private coins**, i.e., there random word is not known to the prover. In the **public coin** model, all random bits read by the verifier are sent to the prover. The random bits need not be revealed all at once but may sent in successive rounds. Furthermore, a interactive proof system is said to have completeness 1 or **perfect** completeness, if every input in the recognized language is accepted with probability 1.

> **Remark 144**: The verifier from Theorem 143 can be modified to a public-coin verifier $V'$ with perfect completeness. Since every $L \in \text{IP}$ reduces to $3$-QBF, every language in $\text{IP}$ has a public-coin, perfectly complete interactive proof.

### Interactive Proof Systems with the Zero-Knowledge Property

Zero-knowledge proofs allow authentication without revealing secrets: the verifier learns nothing beyond the validity of the statement.

$\textbf{Definition 145:}$ An interactive proof system $(V, P)$ for $L$ is *zero-knowledge* if for every verifier $V'$ there is a probabilistic Turing machine $S$ (the simulator) running in expected polynomial time such that for every $w \in L$, the output distribution of $S$ equals the distribution of transcripts from the interaction of $V'$ with $P$ on input $w$.

The simulator shows that any transcript could have been generated without access to the prover’s secret, so no knowledge is leaked (perfect zero-knowledge).

$\text{GI}$, despite lacking a known $\text{NP}$ certificate, admits a zero-knowledge interactive proof.

$\textbf{Theorem 146:}$ $\text{GI}$ has a zero-knowledge interactive proof system.

*Proof.* For input $(G_0, G_1)$, the prover’s secret is an isomorphism if it exists.

1. **Prover:** Pick random $i \in \lbrace 0,1\rbrace$ and random permutation $\pi$. Send $H = \pi(G_i)$.
2. **Verifier:** Send random challenge bit $j$.
3. **Prover:** Return a permutation $\pi_P$ showing $H = \pi_P(G_j)$.
4. **Verifier:** Accept if $\pi_P$ is valid.

*Completeness*: If $G_0 \cong G_1$ via $\pi_0$, the prover always responds with a correct isomorphism (using $\pi$, $\pi \circ \pi_0$, or $\pi \circ \pi_0^{-1}$ depending on $i, j$), so acceptance probability is $1$. 

*Soundness*: If $G_0 \not\cong G_1$, $H$ is isomorphic to only one of them. With probability $\tfrac{1}{2}$ the verifier asks for the other, making acceptance impossible. Repetition reduces soundness error (by iterating the protocol twice, the acceptance probability is at most).

*Zero-knowledge*: It remains to show that the interactive proof system $(V,P)$ has the zeroknowledge property. Fix some verifer $V′$. Let $S$ be a probabilistic Turing machine that on input $(G_0,G_1)$ first chooses a bit $i$ and a permutation $π$ of the node set of $G_i$ uniformly at random and computes $H= π(G_i)$. Then $S$ computes the response $j$ of $V′ $by simulating $V′$ on receiving $H$ as first message of the prover. In case $j$ is not a bit, we can assume that $P$ responds by the empty word $λ$, and accordingly $S$ outputs $(H,j,λ)$. Now assume that $j$ is a bit. In case $j$ is equal to $i$, the simulator outputs $(H,j,π)$. In case $j$ differs from $i$, the simulator abandons the current simulation and restarts by choosing a new bit iand a new permutation $π$.

When demonstrating that $S$ is a simulator for $(V,P)$, we can assume that $G_0$ and $G_1$ are isomorphic and have the same node sets because there are no requirements on a simulator for inputs not in $\text{GI}$. But then for every graph that is isomorphic to $G_0$, the probability that this graph is sent as $H$ does not depend on the choice of $i$. On receiving the message $H$ from $P$, the verifier $V′$ has thus no information at all about the bit $i$, hence $j$ is chosen indepently of $i$. Since $i$ was chosen by tossing a fair coin, the probability that both bits are equal is exactly $\frac{1}{2}$. Thus a simulation where $i$ is equal to $j$ will occur after an expected number of iterations that is constant, hence the running time of $S$ is in expected polynomial time. Furthermore, the distribution of communications $(H,j,π)$ between $P$ and $V′$ is such that every graph $H$ occurs with a certain probability. The message $j$ is independent of $i$, and $π$ is the unique permutation such that $H= π(G_i)$. This coincides with the distribution of the output of $S$, hence $S$ is a simulator for $(V,P)$.

> **Note**: The distributions match because we do not actually know from the outside what was the graph $G_i$, that $H=\pi(G_i)$.

<div class="gd-grid">
  <figure>
    <img src="{{ 'assets/images/notes/computability-and-complexity/gi_has_zero_knowledge_ips.jpeg' | relative_url }}" alt="a" loading="lazy">
    <figcaption>Sketch of the proof</figcaption>
  </figure>
</div>

## Computability

## Unbounded Turing Machines

This chapter transitions from the resource-bounded computations central to complexity theory to the foundational principles of computability theory. In this domain, we are not concerned with how efficiently a problem can be solved, but rather with the more fundamental question of whether it can be solved at all. We will explore computations without any constraints on time or space, leading to the study of Turing machines that may not halt on all inputs.

### Recursively Enumerable Sets

In complexity theory, the focus is on decidable languages—those for which a Turing machine exists that halts on every possible input, providing a definitive "yes" or "no" answer. Such machines are called total Turing machines. In computability theory, we broaden our scope to include languages recognized by Turing machines that are not necessarily total.

A Turing machine $M$ accepts a word $w$ if there exists at least one computation path that leads to an accepting state. The set of all words accepted by $M$ constitutes the language it recognizes, denoted $L(M)$. Furthermore, a language is decidable, if it is recognized by some total Turing machine.

<!-- $\textbf{Definition 147 (Recursively Enumerable Languages):}$ A language $L$ is recursively enumerable, abbreviated as r.e., if $L$ is recognized by a Turing machine. -->
<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Recursively Enumerable Languages)</span></p>

A language $L$ is **recursively enumerable**, abbreviated as r.e., if $L$ is recognized by a Turing machine.

</div>

By this definition, every decidable language is also recursively enumerable. However, as we will see, the converse is not true. While nondeterministic Turing machines can provide significant speed-ups in complexity theory (e.g., the $\text{P}$ versus $\text{NP}$ problem), in computability theory, they offer no additional computational power. Any nondeterministic Turing machine can be simulated by a deterministic one that systematically explores its computation tree. Although this simulation incurs a significant resource cost (e.g., exponential in time), these costs are irrelevant when resource bounds are removed. Therefore, we can simplify our model without loss of generality.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Alternative definitons of r.e.)</span></p>

In **computability theory**, a set $S$ of natural numbers is called **computably enumerable (c.e.)** — also known as **recursively enumerable (r.e.)**, **semidecidable**, **partially decidable**, **listable**, **provable**, or **Turing-recognizable** — if:

**1. Halting characterization**

* There is an **algorithm** such that the set of input numbers **for which the algorithm halts** is **exactly $S$**:

$$\exists A \quad \text{ such that } S = \lbrace n \in \mathbb{N} : A(n) \text{ halts } \rbrace$$

*Or, equivalently:*

**2. Enumeration characterization**

* There is an **algorithm that enumerates the members of $S$**.
  That is, its output is a list of all members of $S$:
  
  $$s_1, s_2, s_3, \dots$$
  
  If $S$ is **infinite**, the algorithm will **run forever**, but **every element of $S$** will appear after a **finite amount of time**.
  **Note:** the elements **do not have to be listed in any particular order** (e.g., not necessarily from smallest to largest).

The **first condition** explains why the term **semidecidable** is used:
* If a number **is in** the set, you can **confirm this** by running the algorithm (it will **eventually halt**).
* If a number **is not in** the set, the algorithm may **run forever**, and **no definitive “no” answer** is returned.

A set that is **fully decidable** is a **decidable set** (or **computable** or **recursive**)**.

The **second condition** explains the term **computably enumerable**.

Finally, the abbreviations **c.e.** and **r.e.** are often used (even in print) instead of the full phrase.

</div>

> **Convention 148**: In what follows, i.e., in the part on computability theory, all Turing machines are meant to be deterministic.

A crucial concept for Turing machines that may not halt is their domain. The domain of a Turing machine $M$, denoted $\text{dom } M$, is the set of all input strings on which $M$ eventually halts.

<!-- $\textbf{Theorem 149:}$ A language is recursively enumerable $\iff$ it is equal to the domain of a Turing machine. -->
<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">($L$ is r.e. $\iff$ $L$ $\text{dom}$ TM)</span></p>

A language is recursively enumerable $\iff$ it is equal to the domain of a Turing machine.

</div>

*Proof*: Let $L$ be a language.

* First, assume $L$ is the domain of some Turing machine $M$. To show $L$ is recursively enumerable, we can modify $M$ into a new machine $M'$ that recognizes $L$. We do this by defining all of $M$'s halting states to be accepting states in $M'$. Now, $M'$ accepts an input $w$ $\iff$ $M$ halts on $w$. Thus, $L(M') = \text{dom } M = L$, which means $L$ is recursively enumerable.
* Conversely, assume $L$ is recursively enumerable, meaning there is a Turing machine $M$ that recognizes it, so $L = L(M)$. For any input to $M$, one of three outcomes is possible:
  1. $M$ halts in an accepting state (the input is in $L$).
  2. $M$ halts in a non-accepting state (the input is not in $L$).
  3. $M$ does not halt (the input is not in $L$). We can construct a new Turing machine $M'$ that halts $\iff$ the input is in $L$. To do this, we modify $M$ such that whenever it would enter a non-accepting halting configuration, it instead enters an infinite loop. This can be achieved by adding a transition from each non-accepting halt state to itself, leaving the tape unchanged. With this modification, $M'$ halts only on inputs that $M$ accepts. Therefore, $\text{dom } M' = L(M) = L$.

### Partial Computable Functions

Computability theory also extends the notion of computation from recognizing languages to computing functions. Because Turing machines are not guaranteed to halt, the functions they compute may not be defined for all inputs. These are known as partial functions.

> **Remark 150**: For sets $A$ and $B$, a function $f$ from $A$ to $B$ is by definition a subset of $A \times B$ that contains for every $x$ in $A$ exactly one pair of the form $(x, y)$, where then we let $f(x) = y$. A partial function from $A$ to $B$ is a subset of $A \times B$ that contains for every $x$ in $A$ at most one pair of the form $(x, y)$. If for given $x$ there is such a pair, the partial function $\alpha$ is defined at place $x$ (or on argument $x$), and we let $\alpha(x) = y$, also written as $\alpha(x) \downarrow = y$. Otherwise, the partial function $\alpha$ is undefined at place $x$, written as $\alpha(x) =\uparrow$. The domain $\text{dom } \alpha$ of $\alpha$ is the subset of $A$ of all places at which $\alpha$ is defined.

Recall that a total $k$-tape Turing machine computes a function by writing the output on its final tape. We now formalize this for machines that may not halt.

<!-- $\textbf{Definition (Partial function):}$ The partial function $\phi_M$ computed by a Turing machine $M$ is  
$$\phi_M (w) = \begin{cases} \text{out}(C) & \text{if the computation of } M \text{ on input } w \text{ reaches the halting configuration } C, \\ \uparrow & \text{otherwise.} \end{cases}$$
Observe that the partial function $\phi_M$ is defined exactly on the arguments $w$ such that $M$ terminates on input $w$. -->
<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Partial function)</span></p>

The **partial function** $\phi_M$ computed by a Turing machine $M$ is  

$$\phi_M (w) = \begin{cases} \text{out}(C) & \text{if the computation of } M \text{ on input } w \text{ reaches the halting configuration } C, \\ \uparrow & \text{otherwise.} \end{cases}$$

Observe that the partial function $\phi_M$ is defined exactly on the arguments $w$ such that $M$ terminates on input $w$.
</div>

> **Remark:** It is important to separate: 
> 1. **a partial function as a mathematical object**, from 
> 2. **a procedure/algorithm that tries to compute it**.
>
> **1. The General Case: Partial Functions (Pure Math)**
> 
> A **partial function** is a purely mathematical object. It is simply a mapping where some inputs in the domain might not have a corresponding output.
> * **Does it have a Turing Machine?** Not necessarily.
> * **How does it "behave" on undefined inputs?** It doesn't "behave" at all. It's just a set of pairs. If $x$ isn't in the domain, the pair simply doesn't exist. There is no "process," no "waiting," and no "machine."
>
> **2. The Specific Case: Partial Computable Functions**
>
> When we add the word "**computable**," we are specifically talking about the subset of partial functions that do have a Turing Machine associated with them.
> * For these functions, we do "offload" the logic to a machine.
> * The definition of a "Partial Computable Function" is literally: "A function for which there exists a Turing Machine that halts and returns the correct value for every $x$ in the domain, and fails to halt for every $x$ outside the domain."
>
> So in the case of **partial functions (not computable)**, we apply the pure math definition of "undefined".

> **Remark:** Do we work with general partial (non-computable) functions and therefore without corresponding Turing Machine in computability theory?
> In computability theory, **usually no**: we mostly work with **partial *computable*** functions, precisely *because* they have a corresponding effective procedure (Turing machine / program / partial recursive definition). But both notions exist, and it’s easy to mix them up:
> 
> **General partial functions**
> A general partial function is just a set-theoretic object
> 
> $$\alpha:\subseteq \mathbb{N}\to\mathbb{N},$$
> 
> with no promise it’s computable. There are vastly more of these than computable ones (uncountably many vs. countably many). For such an $\alpha$, there may be **no** Turing machine that computes it. 
> 
> In that setting, “$\alpha(x)$ is undefined” is purely mathematical: $x\notin\mathrm{dom}(\alpha)$. No “behavior”.
>
> Partial computable functions (the default in computability)
> 
> Here $\alpha$ is in the class usually denoted by $\varphi_e$, $\Phi_e$, etc.: there exists a TM $M$ such that
> 
> * if $\alpha(x)\downarrow$ (defined), $M(x)$ halts with output $\alpha(x)$,
> * if $\alpha(x)\uparrow$ (undefined), $M(x)$ does not halt.
> 
> **This is where “undefined = loops forever” is a correct computational interpretation.**


> **Remark:** 
> * For a **partial function** $\alpha$, saying “$\alpha(x)$ is undefined” simply means **$x$ is not in the domain** of $\alpha$. Formally, $\alpha:\subseteq \mathbb{N}\to\mathbb{N}$ and $x\notin\mathrm{dom}(\alpha)$.
> * In computability, if $\alpha$ is a **partial computable** function computed by some Turing machine/program $M$ in the standard way (halt = produce output), then:
> 
> $$\alpha(x)\ \text{undefined} \quad\Longleftrightarrow\quad M(x)\ \text{does not halt}.$$
> 
> So “loops forever on input $x$” is indeed equivalent to “the computed partial function is undefined at $x$”.
> 
> So, for a partial function $\alpha$, “$\alpha(x)$ is undefined” means $x\notin\mathrm{dom}(\alpha)$ (no value is assigned). In computability theory, when $\alpha$ is computed by a Turing machine $M$, this is equivalent to saying that $M$ does not halt on input $x$ (diverges/loops forever).

This leads to the formal definition of what it means for a partial function to be computable.

<!-- $\textbf{Definition (Partial computable function):}$ For alphabets $\Sigma_1$ and $\Sigma_2$, a partial function from $\Sigma'_1$ to $\Sigma'_2$ is partial computable if it equals the partial function $\phi_M$ computed by some Turing machine $M$.
A partial computable function that is defined for all inputs (i.e., is total) is simply called a computable function. -->
<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Partial computable function)</span></p>

For alphabets $\Sigma_1$ and $\Sigma_2$, a **partial function** from $\Sigma'_1$ to $\Sigma'_2$ is a **partial computable function** if it equals the partial function $\phi_M$ computed by some Turing machine $M$.

A partial computable function that is defined for all inputs (i.e., is total) is simply called a **computable function**.
</div>

### Computability on Natural Numbers

To facilitate a more mathematical treatment, we often work with natural numbers rather than strings. This is achieved by a simple encoding.

> **Remark 153**: The words over the unary alphabet $\lbrace 1\rbrace$ can be identified with the set $\mathbb{N}$ of natural numbers by associating a word $w$ with its length: the empty word is identified with $0$, the word $1$ with $1$, and so on.

Using this identification, we can directly translate our definitions from languages to sets of natural numbers and from functions on strings to functions on numbers.

$\textbf{Definition 154:}$ A set $A$ of natural numbers is recursively enumerable if the set $\lbrace 1^n : n \in A\rbrace$ is recursively enumerable. A partial function $\alpha : \mathbb{N} \to \mathbb{N}$ is partial computable if the partial function $1^n \mapsto 1^{\alpha(n)}$ is partial computable. The notions decidable set of natural numbers and computable function from $\mathbb{N}$ to $\mathbb{N}$ are defined likewise.

> **Convention 155**: In the part on computability theory, unless explicitly stated otherwise, the term set always refers to a set of natural numbers, and the terms partial function and function refer to a partial function and a function, respectively, from $\mathbb{N}$ to $\mathbb{N}$.

### Relations Between Notions of Effectiveness

The concepts of decidable sets, r.e. sets, computable functions, and partial computable functions are deeply interconnected. We can formalize these relationships using characteristic functions.

<!-- $\textbf{Definition 156:}$ Let $A$ be a set. The characteristic function $c_A$ and the partial characteristic function $\chi_A$ of $A$ are defined as follows:  

$$c_A(n) = \begin{cases} 1 & \text{if } n \in A, \\ 0 & \text{if } n \notin A, \end{cases} \quad \text{and} \quad \chi_A(n) = \begin{cases} 1 & \text{if } n \in A, \\ \uparrow & \text{if } n \notin A. \end{cases}$$ -->
<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Partial characteristic function)</span></p>

Let $A$ be a set. The characteristic function $c_A$ and the **partial characteristic function** $\chi_A$ of $A$ are defined as follows:  

$$c_A(n) = \begin{cases} 1 & \text{if } n \in A, \\ 0 & \text{if } n \notin A, \end{cases} \quad \text{and} \quad \chi_A(n) = \begin{cases} 1 & \text{if } n \in A, \\ \uparrow & \text{if } n \notin A. \end{cases}$$
</div>

These functions provide a direct link between sets and functions.

<!-- $\textbf{Proposition 157:}$ A set $A$ is decidable $\iff$ its characteristic function $c_A$ is computable. A set $A$ is recursively enumerable $\iff$ its partial characteristic function $\chi_A$ is partial computable. -->
<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Decidable set $\iff$ $c_A$ is computable, set r.e $\iff$ $\chi_A$ is partial computable)</span></p>

1. A set $A$ is **decidable** $\iff$ its **characteristic function $c_A$ is computable**. 
2. A set $A$ is **recursively enumerable** $\iff$ its **partial characteristic function $\chi_A$ is partial computable**.

</div>

*Proof*: See exercises.

The domain of a partial computable function provides another characterization of r.e. sets.

<!-- $\textbf{Proposition 158:}$ A set is recursively enumerable $\iff$ it is equal to the domain of a partial computable function. -->
<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Set r.e. $\iff$ set is $\text{dom}$ of partial computable function)</span></p>

A set $A \subseteq \mathbb{N}$ is recursively enumerable $\iff$ if there exists a partial computable function $f$ such that:

$$x \in A \iff f(x) \downarrow.$$

That is $A$ is equal to the domain of a partial computable function.

</div>

*Proof*: Let $A$ be a set.

* If $A$ is recursively enumerable, then by Proposition 157, its partial characteristic function $\chi_A$ is partial computable. By definition, the domain of $\chi_A$ is precisely the set $A$.
* Conversely, if $A$ is the domain of a partial computable function $\alpha$, then there exists a Turing machine $M$ that computes $\alpha$. The domain of $M$ is therefore $A$. By Theorem 149, a language that is the domain of a Turing machine is recursively enumerable.

To handle functions with multiple arguments, such as those on $\mathbb{N} \times \mathbb{N}$, we use a pairing function.

<!-- $\textbf{Definition 159:}$ A pairing function is a bijection from $\mathbb{N} \times \mathbb{N}$ to $\mathbb{N}$. Let $\langle \cdot, \cdot \rangle$ be an effective and effectively invertible pairing function (for details, see the exercises). For a subset $A$ of $\mathbb{N} \times \mathbb{N}$, let $\langle A \rangle = \lbrace\langle i, j \rangle : (i, j) \in A\rbrace$. A subset $A$ of $\mathbb{N} \times \mathbb{N}$ is recursively enumerable if $\langle A \rangle$ is recursively enumerable, and $A$ is decidable if $\langle A \rangle$ is decidable. A partial function $\alpha : \mathbb{N} \times \mathbb{N} \to \mathbb{N}$ is partial computable if the function $\langle x, y \rangle \mapsto \alpha(x, y)$ is partial computable, and $\alpha$ is computable if it is partial computable and total. -->

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Pairing function)</span></p>

A **pairing function** is a bijection $\langle\cdot,\cdot\rangle : \mathbb{N}\times\mathbb{N}\to\mathbb{N}$. Fix a pairing function $\langle\cdot,\cdot\rangle$ that is **effective** and **effectively invertible** (see the exercises).

For $A\subseteq \mathbb{N}\times\mathbb{N}$, define

$$\langle A\rangle = \lbrace\langle i,j\rangle : (i,j)\in A\rbrace\subseteq \mathbb{N}.$$

We call $A$ **recursively enumerable** if $\langle A\rangle$ is recursively enumerable, and **decidable** if $\langle A\rangle$ is decidable.

A partial function $\alpha:\mathbb{N}\times\mathbb{N}\to\mathbb{N}$ is **partial computable** if the (encoded) unary function

$$n \mapsto \alpha(x,y)\quad\text{where } n=\langle x,y\rangle$$

is partial computable. We call $\alpha$ **computable** if it is partial computable and total.

</div>

The graph of a function is the set of all its input-output pairs. The computability of a function is linked to the status of its graph.
<!-- $\textbf{Definition:}$ The graph of a partial function $\alpha : A \to B$ is  
$$\text{graph}(\alpha) = \lbrace (x, y) \in A \times B : \alpha(x) \downarrow = y\rbrace$$ -->

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Graph of a partial function)</span></p>

The **graph of a partial function** $\alpha : A \to B$ is  

$$\text{graph}(\alpha) = \lbrace (x, y) \in A \times B : \alpha(x) \downarrow = y\rbrace$$

</div>

<!-- $\textbf{Proposition 160:}$ For a partial function $\alpha$, the following statements are equivalent. 
* **(i)** The partial function $\alpha$ is partial computable. 
* **(ii)** The set $\text{graph}(\alpha)$ is recursively enumerable.

For a function $f$, the following statements are equivalent. 
* **(iii)** The function $f$ is computable. 
* **(iv)** The set $\text{graph}(f)$ is decidable. 
* **(v)** The set $\text{graph}(f)$ is recursively enumerable. -->
<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(name of Proposition)</span></p>

Let $\alpha$ be a partial function. The following are equivalent:

1. $\alpha$ is partial computable.
2. $\operatorname{graph}(\alpha)$ is recursively enumerable.

Let $f$ be a (total) function. The following are equivalent:

1. $f$ is computable.
2. $\operatorname{graph}(f)$ is decidable.
3. $\operatorname{graph}(f)$ is recursively enumerable.
  
</div>

*Proof*: See exercises.

Recursively enumerable sets can also be characterized as projections of decidable sets. Intuitively, this means an element $n$ is in an r.e. set $A$ if there exists some "witness" $i$ such that the pair $(n, i)$ belongs to a simpler, decidable set $B$.

<!-- $\textbf{Proposition 161:}$ A set $A$ is recursively enumerable $\iff$ $A$ is the projection of a decidable set. That is, for some decidable set $B$, it holds that   -->
<!-- $$A = \lbrace n : \text{there exists an } i \text{ such that } \langle n, i \rangle \text{ is in } B\rbrace$$  -->
<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(r.e. and decidable set projection)</span></p>

A set $A\subseteq \mathbb{N}$ is **recursively enumerable** $\iff$ it is the **projection of a decidable set**. 
That is, $A$ is **recursively enumerable** $\iff$ there exists a decidable set $B\subseteq \mathbb{N}\times\mathbb{N}$ such that

$$n\in A \Longleftrightarrow \exists m\in\mathbb{N} (n,m)\in B.$$

</div>

*Proof*:
* First, assume $A$ is the projection of a decidable set $B$. A procedure to compute $\chi_A(n)$ would be to search for an $i$ such that $\langle n, i \rangle \in B$. If such an $i$ is found, the procedure outputs $1$; otherwise, it searches forever. This procedure computes the partial characteristic function $\chi_A$, so $\chi_A$ is partial computable, and thus $A$ is recursively enumerable.
* Conversely, assume $A$ is recursively enumerable. Then $A$ is the domain of some Turing machine $M$. We can define a decidable set $B$ as the set of pairs $\langle n, t \rangle$ where $M$ halts on input $n$ within $t$ steps. $B$ is decidable because we can simulate $M$ on input $n$ for exactly $t$ steps. The set $A$ is then the projection of $B$, since $n \in A$ $\iff$ there exists some time $t$ such that $M$ halts on $n$ within $t$ steps, i.e., $\langle n, t \rangle \in B$.

Finally, the term "recursively enumerable" comes from an alternative characterization: a set is r.e. if its elements can be generated one by one by an effective procedure.

<!-- $\textbf{Definition 162:}$ An enumeration is a sequence $z_0, z_1, \dots$ of natural numbers. For such an enumeration, the set $\lbrace z_0, z_1, \dots\rbrace$ is enumerated by or has the enumeration $z_0, z_1, \dots$, and a natural number $n$ is enumerated if $n$ coincides with some $z_i$. An enumeration $z_0, z_1, \dots$ is effective if the function $i \mapsto z_i$ is computable. -->

<!-- $\textbf{Definition 162:}$ An **enumeration** is an infinite sequence $z_0, z_1, \ldots$ of natural numbers. This sequence is said to **enumerate** the set

$${z_0, z_1, \ldots}.$$

A natural number $n$ is **enumerated** by $z_0, z_1, \ldots$ if $n = z_i$ for some $i$.

The enumeration $z_0, z_1, \ldots$ is **effective** if the mapping $i \mapsto z_i$ is computable (i.e., if there is an algorithm that, given $i$, outputs $z_i$). -->

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Enumeration, Enumerate, Enumerated, Effective Enumeration)</span></p>

An **enumeration** is an infinite sequence $z_0, z_1, \ldots$ of natural numbers. This sequence is said to **enumerate** the set

$$\lbrace z_0, z_1, \ldots\rbrace.$$

A natural number $n$ is **enumerated** by $z_0, z_1, \ldots$ if $n = z_i$ for some $i$.

The enumeration $z_0, z_1, \ldots$ is **effective** if the mapping $i \mapsto z_i$ is computable (i.e., if there is an algorithm that, given $i$, outputs $z_i$, mapping the index directly to the element, allowing us to jump to it directly without enumeratign the whole sequence from the beginning one by one).

</div>

<!-- **Why is the idea of being able to list something is so important in computability theory?**
<div class="accordion">
  <details markdown="1">
    <summary>Why is the idea of being able to list something is so important in computability theory?</summary>

The importance of "listing" (enumeration) in computability theory cannot be overstated. It is the dividing line between **what we can know eventually** and **what we can never know.**

In a world where some programs run forever, being able to "list" elements is the only way to manage infinite sets without getting stuck in a permanent loop.

### 1. It Defines the "Boundary of Science"

In computability, we categorize sets based on how much "work" it takes to recognize them. The idea of listing things gives us the **three great tiers of knowledge**:

* **Tier 1: Decidable (The Gold Standard).** You can list the members and you can list the non-members. You have a definitive "Yes/No" answer for everything.
* **Tier 2: Enumerable (The "Wait and See").** You can list the members, but you can't list the non-members. You can confirm something is true, but you can never be certain something is false—you might just be "waiting longer."
* **Tier 3: Non-Enumerable (The Dark Zone).** You can't even list the members. These sets are completely beyond the reach of any algorithm.

### 2. It Solves the "Stuck Forever" Problem (Dovetailing)

The most practical reason listing is important is that it allows us to handle **partial functions**.

If you have an infinite number of programs and you want to find which ones halt, you cannot check them one by one. If program #1 loops forever, a standard check would "freeze" your computer at step 1.

The "Listing" idea (specifically **Dovetailing**) allows us to run all programs simultaneously in a "round-robin" fashion. Because we can list them, we can explore an infinite search space without ever getting trapped by a single infinite loop.

### 3. It Proves that "Hardware" can be "Software"

The idea of enumeration is what allows for the **Universal Turing Machine**.

* If we can **list** every possible Turing Machine $(\phi_0, \phi_1, \phi_3, \dots)$, then we can treat the "index" (the number in the list) as an input.
* This means one machine can simulate any other machine just by knowing its position in the list.
* **Without the ability to list programs, we could not have programmable computers.**

### 4. It Reveals the Gap Between Logic and Truth

The most famous application of enumeration is **Gödel’s Incompleteness Theorem**.

* Gödel showed that in any consistent mathematical system, the set of "Provable Statements" is **enumerable** (we can list them by checking every possible proof).
* However, he also showed that the set of "True Statements" is **not** enumerable.
* Because one list is "bigger" or "more complex" than the other, there will always be truths that have no proof. **Listing is the tool that proved the limits of human logic.**

### Summary Table: The Power of the List

| If you can... | Then the set is... | Human Equivalent |
| --- | --- | --- |
| **List members AND non-members** | **Decidable** | Perfect Knowledge. |
| **List members only** | **Enumerable** | "I'll tell you when I find it." |
| **List nothing** | **Immune/Co-immune** | "I have no idea where to start." |

### A Concrete Example: The Library of Babel

Imagine a library containing every possible book.

* Can you list all the books that contain the word "Apple"? **Yes.** (Eventually, you'll pull them off the shelf).
* Can you list all the books that **do not** contain the word "Apple"? **No.** Because to confirm a book doesn't have the word, you have to read to the very end. If some books are infinitely long (like a program that never halts), you will never finish checking.

This "one-way" nature of discovery is exactly why **Computably Enumerable** sets are the cornerstone of the field.


### It also enables universal machines and self-reference

A universal machine $U(e,x)$ exists because programs can be *numbered* and effectively listed. Then:
* programs become data (you can feed code to code),
* you can diagonalize over “the $e$-th program,”
* you get foundational results like the **Halting problem**, **Rice’s theorem**, and the **Recursion theorem**.

All of those depend on being able to range over *all* programs effectively.

### It gives robust “degrees” of unsolvability

c.e. sets are the first big class beyond decidable sets, and they have rich structure (reductions, completeness, Turing degrees). They’re a natural boundary where:
* search works (eventually find members),
* full decision fails (can’t always reject).

  </details>
</div> -->

**The "listing" (numbering) idea to prove that the Halting Problem is undecidable via a "Diagonalization" argument**
<div class="accordion">
  <details markdown="1">
    <summary>The "listing" (numbering) idea to prove that the Halting Problem is undecidable via a "Diagonalization" argument</summary>

To prove that the **Halting Problem** is undecidable, we use a "Diagonalization" argument. This technique was originally developed by Georg Cantor to show that some infinities are larger than others, but Alan Turing adapted it to show the limits of what computers can do.

The proof relies entirely on the **Listing (Numbering)** idea we discussed: because we can list every possible program, we can build a table of their behaviors.

### Step 1: The "List of All Programs" Table

Imagine we create a giant table.

* **The Rows** represent every possible program $(\phi_0, \phi_1, \phi_3, \dots)$.
* **The Columns** represent every possible input $(0, 1, 2, \dots)$.
* **The Cell $(i,j)$** tells us if program $i$ halts on input $j$.

|  | Input 0 | Input 1 | Input 2 | Input 3 | ... |
| --- | --- | --- | --- | --- | --- |
| $\phi_0$ | **Halt** | Loop | Halt | Loop | ... |
| $\phi_1$ | Halt | **Halt** | Loop | Halt | ... |
| $\phi_2$ | Loop | Loop | **Loop** | Halt | ... |
| $\phi_3$ | Halt | Halt | Halt | **Loop** | ... |

### Step 2: The Hypothetical "Halt Decider"

Suppose there exists a program called `Halt(program_index, input)`.
This program is perfect:

* It returns **True** if the program halts.
* It returns **False** if the program loops.
* It **never** loops itself (it is a total computable function).

### Step 3: Constructing the "Troublemaker" (Diagonalization)

Now, we create a new, weird program called $D$ (for Diagonal). $D$ takes an input $i$ and does the exact opposite of what the diagonal of our table says.

**The logic of $D(i)$:**

1. Ask the `Halt` decider: *"Does program $i$ halt on input $i$?"*
2. If the answer is **Yes** (it halts), $D$ enters an **infinite loop**.
3. If the answer is **No** (it loops), $D$ **halts** immediately.

Essentially, $D$ is designed to "disagree" with every program in the list at its own index.

### Step 4: The Contradiction

Since $D$ is a program, it must be somewhere in our original list! Let's say D is program number $k$ in the list ($\phi_k=D$).

Now, we ask the million-dollar question: **Does program $D$ halt on input $k$?**

* **Scenario A: $D(k)$ halts.**
According to $D$'s own code, if it sees that $\phi_k(k)$ halts, it must loop forever.
*Contradiction: It halted, but it was supposed to loop.*
* **Scenario B: $D(k)$ loops forever.**
According to $D$'s own code, if it sees that $\phi_k(k)$ loops, it must halt.
*Contradiction: It looped, but it was supposed to halt.*

### Conclusion

Because both scenarios lead to a logical impossibility (a program that halts if and only if it doesn't halt), our original assumption must be wrong.

**The assumption that a `Halt` decider exists is false.**

### Why the "Listing" was the key:

1. **Completeness:** We needed the list to be "Standard" so we could be sure program $D$ was actually in the list.
2. **Indexing:** We needed to be able to "feed a program its own number" ($D$ looking at index $k$), which is only possible because of universal numbering.
3. **Existence:** The listing property is what allowed us to treat the set of all programs as a single, organized mathematical object we could "diagonalize" across.
  </details>
</div>

**Rice's Theorem, which uses a similar logic to show that *any* non-trivial property of a program's behavior is undecidable**
<div class="accordion">
  <details markdown="1">
    <summary>Rice's Theorem, which uses a similar logic to show that *any* non-trivial property of a program's behavior is undecidable</summary>

If the Halting Problem is the "Grandfather" of undecidability, **Rice's Theorem** is the "Overlord." It takes the logic we just used for the Halting Problem and generalizes it to almost **everything** else.

Rice's Theorem states that **any non-trivial property of the partial computable functions is undecidable.**

### 1. What does "Non-Trivial Property" mean?

To understand the theorem, we have to define its two main conditions:

* **A Property of the Function (Behavioral):** We are looking at what the program *does* (its input/output mapping), not how it is written.
* *Example:* "Does this program halt on input 5?" (Behavioral)
* *Non-example:* "Does this program have more than 10 lines of code?" (Structural—this is actually decidable!)

* **Non-Trivial:** A property is non-trivial if it is true for *some* computable functions but not *all* of them.
* *Example:* "Does this program return 1 for all inputs?" (Some do, some don't. Non-trivial!)
* *Trivial example:* "Is this a partial computable function?" (All of them are. Trivial!)

### 2. The Big Revelation

Rice's Theorem tells us that you cannot write a program that takes the source code of another program and answers questions like:

* "Will this program ever output a 0?"
* "Is this program equivalent to this other program?"
* "Does this program compute the square of its input?"
* "Is the domain of this program infinite?"

**All of these are undecidable.** You can never write a "Perfect Debugger" or a "Perfect Compiler Optimizer" that works for all possible programs.

### 3. The Logic: Reduction to the Halting Problem

The proof of Rice's Theorem uses a "reduction." We show that if you could solve *any* non-trivial behavioral property, you could use that solution to solve the Halting Problem.

**The Setup:**

1. Assume we have a property $P$ that is non-trivial.
2. Suppose there is a program `CheckP(code)` that tells us if a program has property $P$.
3. We want to know if program $M$ halts on input $x$.

**The Trick:**
We create a new "wrapper" program $W$:

> **Program $W(y)$:**
> 1. First, run program $M$ on input $x$.
> 2. If it halts, run the logic for a function that *definitely has* property $P$.
> 3. If $M$ never halts, step 2 never happens.

**The Contradiction:**
If we give the code of $W$ to our assumed `CheckP` tool:

* If `CheckP(W)` says **Yes**, it means $M$ must have halted on $x$.
* If `CheckP(W)` says **No**, it means $M$ must have looped forever.

Since we already proved the Halting Problem is undecidable, `CheckP` **cannot exist.**

### 4. Why this matters for Computer Science

Rice's Theorem is why "Formal Verification" (proving a program is bug-free) is so difficult. It means:

* **Security:** You cannot write a program that perfectly detects all possible malware behaviors.
* **Optimization:** You cannot write a compiler that always finds the "most efficient" version of a program.
* **Education:** An "Auto-Grader" can never perfectly determine if a student's code is mathematically identical to the answer key for all possible inputs.

### Summary

If a question is about what a program **does** rather than how it is **formatted**, and the answer isn't "yes" for every single program, then a computer **cannot** answer that question for you.
  </details>
</div>

**This relates to "Kolmogorov Complexity"—the idea that some strings are so random they cannot be compressed by any program**
<div class="accordion">
  <details markdown="1">
    <summary>This relates to "Kolmogorov Complexity"—the idea that some strings are so random they cannot be compressed by any program</summary>

**Kolmogorov Complexity** ($K(s)$) is the ultimate limit of data compression. While Rice's Theorem deals with the limits of **what programs can do**, Kolmogorov Complexity deals with the limits of **how concisely information can be represented.**

The connection to the Halting Problem and Rice's Theorem is profound: **The "randomness" (complexity) of a string is uncomputable.**

### 1. What is Kolmogorov Complexity?

The Kolmogorov Complexity of a string $s$ is the **length of the shortest possible program** (in a fixed standard language) that produces $s$ as output and then halts.

* **Simple Strings:** Consider a string of 10,000 "A"s. You don't need 10,000 characters to describe it. You can just say: `print "A" * 10000`. The program is very short, so the complexity is **low**.
* **Complex (Random) Strings:** Consider a string of 10,000 truly random digits. There is no pattern to exploit. The shortest program to produce it is likely just: `print "4f92j..."`. The program is as long as the string itself. Its complexity is **high**.

### 2. The Berry Paradox: The Connection to Undecidability

The reason $K(s)$ relates to the Halting Problem is because of a logical trap called the **Berry Paradox**.

Imagine a program that tries to find the "first number that cannot be described in fewer than twenty words." But the phrase *"The first number that cannot be described in fewer than twenty words"* is itself a description of that number, and it only has **twelve words**.

In computability terms, this means:

> **You cannot compute $K(s)$.**

If you had a program that could compute the complexity of any string, you could use it to solve the Halting Problem. Conversely, because the Halting Problem is undecidable, you can never be certain if you've found the *shortest* possible program for a string—you might find a short program, but you can never prove that an even shorter one won't eventually halt and produce the same string.

### 3. "Randomness" is Just Lack of Compression

In this theory, a string is defined as **algorithmically random** if its Kolmogorov Complexity is approximately equal to its length ($K(s)\approx \lvert s \rvert$).

This leads to a shocking conclusion: **Most strings are random.**
Because there are fewer short programs than there are long strings, it is mathematically impossible for every string to have a "short" description. Most information in the universe is "incompressible."

### 4. Chaitin’s Constant ($\Omega$): The Ultimate Uncomputable Number

The bridge between Kolmogorov Complexity and the Halting Problem is **Chaitin’s Constant**, denoted by the Greek letter $\Omega$.

$\Omega$ represents the probability that a randomly constructed program will halt.

* To know the digits of $\Omega$, you would have to solve the Halting Problem for every possible program.
* Because the Halting Problem is undecidable, $\Omega$ is **transcendental** and **algorithmically random**.
* Knowing the first $N$ bits of $\Omega$ would allow you to solve the Halting Problem for all programs of length up to $N$.

### How it fits your "Inheritance Graph"

If we were to add these to our conceptual map of computability:

* **Halting Problem:** The "Base Error" (You can't know if it stops).
* **Rice's Theorem:** The "Generalization" (You can't know *any* behavioral property).
* **Kolmogorov Complexity:** The "Quantitative Limit" (You can't even know how much information is actually there).

| Concept | Question Asked | Answer Status |
| --- | --- | --- |
| **Halting Problem** | "Does this program stop?" | **Undecidable** |
| **Rice's Theorem** | "Does this program do X?" | **Undecidable** |
| **Kolmogorov Complexity** | "What is the shortest version of this?" | **Uncomputable** |
  </details>
</div>

**How to use these ideas to understand the "Incompleteness" of mathematics — specifically why there are true statements that can never be proven**
<div class="accordion">
  <details markdown="1">
    <summary>How to use these ideas to understand the "Incompleteness" of mathematics — specifically why there are true statements that can never be proven</summary>

The bridge between computer science and the "Incompleteness" of mathematics is one of the most beautiful connections in logic. It was Kurt Gödel who first proved this in 1931, but looking at it through the lens of **computability** (specifically the Halting Problem) makes it much easier to see why it *must* be true.

To understand why some truths can never be proven, we have to look at what a "proof" actually is in terms of an algorithm.

### 1. The "Proof Machine" (Enumeration)

In a formal mathematical system (like Arithmetic), a **proof** is just a finite sequence of logical steps following specific rules.
Because the rules are fixed, we can write a program to **list (enumerate) all possible proofs**:

1. The program starts with the basic axioms.
2. It applies every possible rule to those axioms to create new theorems.
3. It then applies the rules to *those* theorems, and so on.

This means the set of all **provable statements** is **computably enumerable (c.e.)**. If a statement has a proof, this "Proof Machine" will eventually find it and print it out.

### 2. The Link: Proofs vs. Halting

We can turn the Halting Problem into a mathematical statement. For any program $e$ and input $x$, the statement **"P_e(x) halts"** is a mathematical assertion.

* If the program **does** halt, we can prove it simply by running the program until it stops. The execution trace *is* the proof.
* But what if the program **does not** halt? To prove it *never* halts, we need a logical argument that covers an infinite amount of time.

### 3. The "Gap" (Incompleteness)

Now, imagine a mathematical system that is **Complete** (meaning every true statement has a proof). If such a system existed, we could solve the Halting Problem like this:

1. Take a program $M$ and input $x$.
2. Start two processes at the same time:
* **Process A:** Simulate the program $M(x)$ and wait for it to halt.
* **Process B:** Run the "Proof Machine" and look for a formal proof that says *"Program $M(x)$ never halts."*

If the system were complete, one of these two processes **must** eventually win. If the program halts, Process A finds out. If it doesn't halt, Process B finds the proof.

**The Contradiction:** This would give us a way to decide the Halting Problem! But we already proved (via Diagonalization) that no such decider exists. Therefore, **Process B must fail for some programs.** There must be programs that never halt, yet there is no formal proof within the system to prove it.

### 4. The Conclusion: Truth $\neq$ Provability

This leads to **Gödel’s First Incompleteness Theorem**:

> In any sufficiently powerful consistent mathematical system, there are statements that are **true** but **unprovable**.

* **Truth:** The program actually runs forever (a fact of reality).
* **Provability:** The logical rules of the system are not strong enough to "capture" that infinite behavior in a finite number of steps.

### Summary: The Computability Perspective

| Concept | Computability Equivalent | Mathematical Result |
| --- | --- | --- |
| **Provable Statements** | Computably Enumerable (c.e.) | We can list all "Yes" answers. |
| **True Statements** | Not Computably Enumerable | We cannot list all truths. |
| **Consistency** | The program doesn't lie. | We can't prove "Yes" and "No" for the same thing. |
| **Completeness** | Solving the Halting Problem. | **Impossible.** |

### The Final "Mind-Bending" Thought

If you find a statement that you can't prove or disprove (an **undecidable statement**), you can actually choose to *add* it as a new axiom. But as soon as you do, Rice's Theorem and the Halting Problem apply to your *new* system, creating a *new* set of true-but-unprovable statements.

**The "hole" in mathematics can be moved, but it can never be filled.**
  </details>
</div>

**The idea of enumeration and its connection to recursively enumerable sets**
<div class="accordion">
  <details markdown="1">
    <summary>The idea of enumeration and its connection to recursively enumerable sets</summary>

### Enumeration in computability

In our course’s terminology, an **enumeration** is simply an infinite sequence $z_0,z_1,\ldots$ of natural numbers. In computability theory we are usually interested in **effective enumerations**: sequences whose indexing map $i\mapsto z_i$ is **computable**, so that given $i$ we can algorithmically obtain (z_i). This captures the idea that, *given enough time, we can systematically list the members of a set*, even when we cannot decide membership in one step.

## The core picture: enumerations vs. effective enumerations

An enumeration $z_0,z_1,\ldots$ **enumerates** the set

$$\lbrace z_0,z_1,\ldots\rbrace.$$

It is **effective** if there exists an algorithm computing the function $i\mapsto z_i$. Equivalently, there is a procedure that can “jump” to the $i$-th listed element without having to treat the sequence as a mysterious object.

As usual:

* **Order doesn’t matter**
* **Duplicates are allowed**
* If the set is infinite, the process runs forever
* But each $x$ in the set appears at some finite stage

So instead of answering “is $x\in A$?”, you can “solve” $A$ by **systematically listing all the yes-instances**.

## Why it matters: semi-decision and c.e. sets

A set $A\subseteq\mathbb{N}$ is **computably enumerable (c.e.)** (a.k.a. **recursively enumerable**, **semidecidable**) precisely when it has an **effective enumeration** (in the above sense, allowing repetition). Intuitively:

* If $x\in A$, there is a computation that will eventually confirm it (by producing $x$ in the list, or by halting on input $x$).
* If $x\notin A$, the computation may run forever without ever giving a final “no”.

This is captured by the central equivalence:

$$
A\text{ is c.e.} \quad\Longleftrightarrow\quad \exists\ \text{a Turing machine that halts exactly on inputs in }A.
$$

* **YES instances** halt.
* **NO instances** may loop.

## The key technique: dovetailing

Effective enumeration often uses **dovetailing** (round-robin simulation) to avoid getting stuck on a single non-halting computation.

Classic example: the **halting set**

$$K=\lbrace\langle e,x\rangle:\varphi_e(x)\downarrow\rbrace.$$

Enumerate $K$ by running all computations $\varphi_e(x)$ in stages (one step each, then two steps each, etc.). Whenever one halts, output $\langle e,x\rangle$. This lists every halting pair—so $K$ is **c.e.**—even though $K$ is **not decidable**.

## “Enumeration” vs. “enumerating all programs”

Two related uses of the word appear frequently:

1. **Enumerating a set $A$**: giving an effective enumeration $z_0,z_1,\ldots$ whose range is $A$.
2. **Enumerating computable objects**: an effective list of programs/functions
   
   $$\varphi_0,\varphi_1,\varphi_2,\ldots$$
   
   where $e$ is an **index** (code) for a program—crucial for universal simulation and many theorems.

## Decidable vs. enumerable

* **Decidable (computable) set:** an algorithm always halts and answers **Yes/No** correctly.
* **c.e. (semidecidable) set:** there is an effective procedure that confirms **Yes** (halts / eventually lists $x$), but may run forever on **No**.

### One-sentence takeaway

**In our course’s language: a set is c.e. exactly when it admits an effective enumeration—an algorithmic listing of all its members—even though non-membership may not be decidable.**
  </details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Turing Machine $\iff$ Algorithm)</span></p>

In standard computability theory, 

$$\exists \text{ TM } \iff \exists \text{ Algorithm}$$

*as long as* “algorithm” means a fully precise, mechanical procedure (not a heuristic, randomized method, or “works in practice” routine).

* **Turing machine $\implies$ algorithm:** A Turing machine describes an effective procedure step by step, so it’s an algorithmic method.
* **Algorithm $\implies$ Turing machine:** Under the usual, formal interpretation of “algorithm” as an **effective method**, the Church–Turing thesis says any such procedure can be implemented by a Turing machine. So in textbooks, people freely swap:
  * “there exists an algorithm computing $f$”
  * “there exists a Turing machine computing $f$”
  * “$f$ is (partial) recursive / (partial) computable”

Important nuance: this equivalence is not a *theorem* inside mathematics unless you first **define** “algorithm” in one formal model. It’s either:
1. a matter of definition (“algorithm” = “Turing-computable”), or
2. an appeal to the Church–Turing thesis.

So: **in the context of r.e./decidable/partial computable, it’s safe to treat them as equivalent.**

</div>

<!-- $\textbf{Proposition 163:}$ A nonempty set $A$ is recursively enumerable $\iff$ $A$ has an effective enumeration. -->

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(r.e. and effective enumeration)</span></p>

A nonempty set $A\subseteq \mathbb{N}$ is recursively enumerable $\iff$ $A$ has an effective enumeration.

</div>

*Proof*:
* Assume $A$ has an effective enumeration $z_0, z_1, \dots$ produced by a computable function $i \mapsto z_i$. A Turing machine can recognize $A$ as follows: on input $n$, it computes $z_0, z_1, z_2, \dots$ in sequence. If it ever finds a $z_i$ such that $z_i = n$, it accepts. If no such $z_i$ exists, it runs forever. This machine recognizes $A$, so $A$ is r.e.
* Assume $A$ is a nonempty r.e. set. Let $M$ be a Turing machine that recognizes $A$. Fix an element $a \in A$. We can define an enumeration $z_0, z_1, \dots$ as follows. For each $i$, decode it as a pair $\langle n, t \rangle$. Simulate $M$ on input $n$ for $t$ steps. If $M$ accepts $n$ within $t$ steps, set $z_i = n$. Otherwise, set $z_i = a$. The function $i \mapsto z_i$ is computable, so this is an effective enumeration of $A$. $\square$

**More detailed proof:**
<div class="accordion">
  <details markdown="1">
    <summary>More detailed proof</summary>

Sure. The proposition is the classic equivalence:

* **r.e.** (“recursively enumerable”) means: there is a Turing machine $M$ that **halts and accepts** exactly the numbers in $A$ (and may run forever on numbers not in $A$).
* **effective enumeration** means: there is a **total computable function** $f(i)=z_i$ whose values list elements of $A$ (repetitions allowed).

I’ll unpack both directions and, especially, the second construction.

---

## 1. Effective enumeration $\Rightarrow$ r.e.

Assume $A$ has an effective enumeration $z_0,z_1,\dots$ with computable $i\mapsto z_i$.

We want a recognizer for $A$: a machine that, given input $n$,

* **accepts if $n\in A$**,
* **never halts if $n\notin A$**.

Algorithm:

1. Compute $z_0$. If $z_0=n$, accept.
2. Compute $z_1$. If $z_1=n$, accept.
3. Compute $z_2$. If $z_2=n$, accept.
4. Continue forever.

Why it works:

* If $n\in A$, then by definition of “enumeration”, $n=z_i$ for some $i$. When we reach that $i$, we accept and halt.
* If $n\notin A$, then no $z_i$ equals $n$. So we never accept; the machine just keeps going.

That’s exactly the definition of “recognizes $A$”, so $A$ is r.e.

---

## 2. r.e. (nonempty) $\Rightarrow$ effective enumeration

Now assume $A$ is **nonempty** and r.e. Let $M$ be a Turing machine that recognizes $A$.

Key difficulty:
If you try “enumerate by running $M$ on inputs $0,1,2,\dots$”, you get stuck because $M(n)$ might **never halt**, so you can’t move on. You need a way to “run all computations in parallel”.

### The trick: time-bounded simulation via pairing $\langle n,t\rangle$

For each index $i$, decode it into a pair $\langle n,t\rangle$ (so you range over all inputs $n$ and all time limits $t$).

Then define $z_i$ like this:

* Simulate $M(n)$ for **exactly $t$ steps**.
* If $M$ accepts within those $t$ steps, output $n$.
* Otherwise output some fixed $a\in A$.

So:

$$
z_i =
\begin{cases}
n & \text{if (M) accepts (n) within (t) steps}\
a & \text{otherwise.}
\end{cases}
$$

### Why we need $a\in A$ and why $A\neq\emptyset$

If $M$ does *not* accept within $t$ steps, we still need to output **some natural number** (because an enumeration is a total function $i\mapsto z_i$). We also need every output to be an element of $A$, because it’s supposed to enumerate $A$.

So we pick a known element $a\in A$ and use it as a safe fallback. This is exactly why the proof assumes **nonempty**.

### Why $i \mapsto z_i$ is computable

Given $i$:

1. Compute $(n,t)$ from $i$ using the inverse of the pairing function (effective invertibility).
2. Simulate $M(n)$ for $t$ steps. This is a finite simulation, so it always terminates.
3. Depending on whether acceptance happened, output $n$ or $a$.

All steps are effective, so $i\mapsto z_i$ is a **total computable function**.

### Why $\lbrace z_0,z_1,\dots\rbrace = A$

Two things to check:

**(i) Every output is in $A$.**

* If $z_i=n$, then $M$ accepted $n$, so $n\in A$.
* If $z_i=a$, then $a\in A$ by choice.

So all outputs lie in $A$.

**(ii) Every element of $A$ appears somewhere.**
Take any $x\in A$. Since $M$ recognizes $A$, $M(x)$ accepts after some finite number $T$ of steps.

Now consider the index (i) that decodes to $\langle x, T\rangle$. For that (i), the simulation runs $M(x)$ for $T$ steps, sees acceptance, and outputs $z_i=x$.

So every $x\in A$ appears at least once.

Together, (i) and (ii) mean the sequence effectively enumerates $A$.

---

## Intuition in one sentence

The second direction is “dovetailing”: instead of running $M(0)$, then $M(1)$, etc. (which can get stuck), you run **every input for every finite time bound**, guaranteeing you eventually witness any accepting computation.
  </details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Church–Turing thesis informal definition of r.e.)</span></p>

According to the Church–Turing thesis, every **effectively computable** function can be computed by a **Turing machine**. Consequently, a **recursively enumerable** set $S$ is **computably enumerable** *if and only if* there exists an algorithm that enumerates the elements of $S$.

**However, this characterization cannot be taken as a formal definition**, since the **Church–Turing thesis is an informal conjecture**, not a formally stated axiom.

</div>

Decidable sets correspond to a special, more structured type of enumeration.

<!-- $\textbf{Proposition 164:}$ A nonempty set $A$ is decidable $\iff$ $A$ has a nondecreasing effective enumeration. -->

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(decidability and nondecreasing effective enumeration)</span></p>

A nonempty set $A\subseteq \mathbb{N}$ is decidable $$\iff$$ $A$ has a nondecreasing effective enumeration.
</div>

*Proof*:

<!-- * Assume $A$ is decidable. If $A$ is finite, it is trivially decidable and has a nondecreasing enumeration. If infinite, we can construct a nondecreasing enumeration $z_0, z_1, \dots$. Let $z_0$ be the smallest element of $A$. For $i > 0$, let $z_i$ be the smallest element of $A$ that is greater than $z_{i-1}$. Since $A$ is decidable, we can find these elements effectively. The resulting function $i \mapsto z_i$ is computable.
* Conversely, assume $A$ has a nondecreasing effective enumeration $z_0, z_1, \dots$. If $A$ is finite, it is decidable. If $A$ is infinite, to decide membership for an input $n$, we generate the sequence $z_0, z_1, \dots$ until we find a $z_i$ such that $z_i \ge n$. Since the sequence is nondecreasing and infinite, such a $z_i$ must exist. We then check if $n = z_i$. If it is, $n \in A$; otherwise, $n \notin A$. This procedure always halts, so $A$ is decidable. -->
  
$(\Rightarrow)$ Assume $A$ is decidable and nonempty. Since membership in $A$ is decidable, we can effectively find the least element of $A$: search $0,1,2,\dots$ until the first $m$ with $m\in A$, and set $z_0=m$.

For $n\ge 1$, define

$$
z_n =
\begin{cases}
n, & \text{if } n\in A,\\
z_{n-1}, & \text{if } n\notin A.
\end{cases}
$$

Then $(z_n)\_{n\in\mathbb{N}}$ is nondecreasing (either it stays the same, or it jumps up to $n$). Moreover, every element of $A$ appears in the sequence (if $n\in A$, then $z_n=n$), and no value outside $A$ ever appears. Finally, the map $n\mapsto z_n$ is computable because we can decide whether $n\in A$ and we already computed $z_{n-1}$. Hence $A$ has a nondecreasing effective enumeration.

$(\Leftarrow)$ Conversely, suppose $z_0,z_1,\dots$ is a nondecreasing effective enumeration of $A$. If $A$ is finite, then it is decidable (e.g. compute the finite set of values it takes and hard-code membership).

If $A$ is infinite, then the nondecreasing enumeration cannot get stuck forever at a maximum value, so it must be unbounded: for every $n$ there exists $i$ with $z_i\ge n$. To decide whether a given $n$ lies in $A$, compute $z_0,z_1,\dots$ until you reach the least index $i$ such that $z_i\ge n$ (this search halts by unboundedness). Then accept iff $z_i=n$. Because $i\mapsto z_i$ is computable, this procedure is an algorithm, so $A$ is decidable. $\square$


A final, crucial theorem links decidability to the recursive enumerability of a set and its complement.

<!-- $\textbf{Proposition 165:}$ A set $A$ is decidable $\iff$ a set $A$ and its complement $\mathbb{N} \setminus A$ are recursively enumerable. -->

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(decidability and r.e.)</span></p>

A set $A\subseteq \mathbb{N}$ is decidable $\iff$ a set $A$ and its complement $\mathbb{N}\setminus A$ are recursively enumerable.

</div>

*Proof*: See exercises.

> **Remark 166**: By the equivalences shown in this section, the four concepts of decidable and recursively enumerable set as well as partial computable and computable functions can be mutually defined in terms of each other via purely set-theoretical definition not involving computability. Consequently, instead of introducing all four concepts separately via Turing machines, we could have done so for any one of them, then defining the other three concept without further use of Turing machines. This indicates that all four concepts formalize the same intuitive notion of effectiveness.

## Numberings

Having established the class of partial computable functions, we now turn to the question of how to refer to them. Since Turing machines can be described by finite strings, we can assign a unique natural number to every Turing machine, and by extension, to every partial computable function. This leads to the concept of a numbering.

For partial functions $\alpha$ and $\beta$, we write $\alpha(n) \simeq \beta(n)$ if the function values $\alpha(n)$ and $\beta(n)$ either are both undefined or are both defined and equal.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">($(\alpha_e)_{e\in\mathbb{N}}, \alpha$)</span></p>

For a sequence $(\alpha_e)_{e\in\mathbb{N}}=\alpha_0, \alpha_1, \dots$ of partial functions, there is a unique partial function $\alpha$ on $\mathbb{N}^2$ such that
 
$$\alpha(e, x) \simeq \alpha_e(x) \text{ for all } e, x \text{ in } \mathbb{N}. \quad (7.1)$$
 
Conversely, for every partial function $\alpha$ from $\mathbb{N}^2$ to $\mathbb{N}$, there is a unique sequence $\alpha_0, \alpha_1, \dots$ of partial functions such that (7.1) holds.
</div>

This correspondence allows us to treat a sequence of functions and its associated two-argument "principal" function interchangeably.

### Numberings of Partial Computable Functions

<!-- $\textbf{Definition 168:}$ Let $\alpha_0, \alpha_1, \dots$ be a sequence of partial functions on $\mathbb{N}$. Let $\alpha$ be a partial function on $\mathbb{N}^2$ such that (7.1) holds. Then $\alpha$ is called the principal function of the sequence $\alpha_0, \alpha_1, \dots$, and the sequence itself is called determined by $\alpha$. The sequence $\alpha_0, \alpha_1, \dots$ is a numbering of partial computable functions, or simply a numbering, if its principal function is partial computable. A numbering $\alpha$ is a numbering of or is universal for a set of partial functions if the latter set is equal to $\lbrace\alpha(e, \cdot) : e \in \mathbb{N}\rbrace$. A universal numbering is a numbering of the set of partial computable functions. -->
<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Numberings, Principal (universal) function, Universal numbering)</span></p>

Let $(\alpha_e)_{e\in\mathbb{N}}$ be a sequence of partial functions $\alpha_e:\mathbb{N}\rightharpoonup\mathbb{N}$. Define a partial function $\alpha:\mathbb{N}^2\rightharpoonup\mathbb{N}$ by

$$\alpha(e, x) \simeq \alpha_e(x) \text{ for all } e, x \text{ in } \mathbb{N}. \quad (7.1)$$

We call $\alpha$ the **principal (universal) function** of the sequence $(\alpha_e)$, and we say that the sequence $(\alpha_e)$ is **determined by** $\alpha$.

The sequence $(\alpha_e)$ is called a **numbering** (of partial functions) if its principal function $\alpha(e,x)$ is **partial computable**.

A numbering $(\alpha_e)$ is **universal for** a class $\mathcal{F}$ of partial functions if

$$\mathcal{F}=\lbrace\alpha_e : e\in\mathbb{N}\rbrace.$$

In particular, a **universal numbering** is a numbering that enumerates *all* partial computable functions.
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Numbering)</span></p>

In general, in computability theory, a **numbering** is an assignment of **natural numbers** to a set of objects such as functions, rational numbers, graphs, or words in some formal language. A numbering can be used to transfer the idea of computability and related concepts, which are originally defined on the natural numbers using **computable functions**, to these different types of objects.

Common examples of numberings include **Gödel numberings** in first-order logic, the **description numbers** that arise from **universal Turing machines** and **admissible numberings** of the set of partial computable functions.

In our course, a **numbering** is an effective way to use **natural numbers as indices (codes)** for **a family of partial functions**.

In your course’s setup, a **numbering** is an effective way to use natural numbers as **indices** (codes) for a family of partial functions.

Start with a sequence of partial functions

$$(\alpha_e)_{e\in\mathbb N},\quad \alpha_e:\mathbb N \rightharpoonup \mathbb N.$$

Define its **principal function** (also called the universal function for the sequence)

$$\alpha:\mathbb N^2 \rightharpoonup \mathbb N,\qquad \alpha(e,x)\simeq \alpha_e(x).$$

Then:
* $(\alpha_e)$ is a **numbering** iff its principal function $\alpha(e,x)$ is **partial computable**.

**Intuition**

A numbering is like a programming system:
* the index $e$ is the “program code,”
* $\alpha_e$ is the partial function computed by program $e$,
* and $\alpha(e,x)$ is the universal interpreter: “run program $e$ on input $x$.”

Saying $\alpha$ is partial computable means there is a single algorithm (a universal machine) that, given $e$ and $x$, simulates what program $e$ does on input $x$.

</div>

<figure>
  <img src="{{ '/assets/images/notes/computability-and-complexity/numbering_visualization.jpeg' | relative_url }}" alt="a" loading="lazy">
  <figcaption>A hopeless attempt to visualize what numbering is</figcaption>
</figure>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Numbering)</span></p>

Fix a universal Turing machine $U$. Let $\alpha_e(x)$ be “the output of machine with code $e$ on input $x$” (undefined if it doesn’t halt). Then $(\alpha_e)$ is a numbering because there is a single simulator $U$ computing $\alpha(e,x)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Numbering can have duplicates, and in computability we expect it to)</span></p>

**A numbering can have duplicates**, and in computability we *expect* it to.

**Does the definition allow duplicates?**

In your notes, a numbering is just a sequence $(\alpha_e)_{e\in\mathbb N}$ whose principal function $\alpha(e,x)$ is partial computable. There is **no requirement** that $e\mapsto \alpha_e$ is injective.

So duplicates are absolutely allowed.

**Why duplicates are natural**

Think “indices = program codes.” Different programs can compute the same function:
* add useless steps,
* add dead code,
* reorder computations,
* include an unused constant,
* etc.

So even with a very concrete “standard” model (Turing machines, RAM programs, λ-terms), you automatically get many syntactically different programs with the same behavior.

**But: duplicates are not *forced* by the definition**

You can design an effective numbering with **no duplicates** (each partial computable function appears exactly once). That would still be a valid numbering (principal function computable), but it would **not** be acceptable/admissible in the Gödel sense.

**What’s special about acceptable/admissible numberings**

For acceptable numberings, duplicates aren’t just “allowed”; they’re essentially **unavoidable**: by the padding lemma, every function has **infinitely many indices**.

* *Numbering* ⇒ duplicates **may** happen.
* *Acceptable numbering* ⇒ duplicates **must** happen (in fact infinitely many per function).

</div>

> **Remark 169**: For any numbering $\alpha_0, \alpha_1, \dots$, all $\alpha_e$ are partial computable. To prove this, choose a Turing machine that computes the principal function of the given numbering. By modifying this Turing machine so that it first transforms its input $x$ into $\langle e, x \rangle$ before performing the actual computation, we obtain a Turing machine that computes $\alpha_e$. As a consequence, a numbering $\alpha_0, \alpha_1, \dots$ is universal if and only if all partial computable functions occur among the $\alpha_e$.

Not every sequence of computable functions constitutes a numbering. A numbering requires that the principal function itself be computable.

> **Remark 170**: There exist sequences of computable functions that are not numberings. For a proof, first note that there are undecidable sets since there are uncountably many susbets of the natural numbers but only countable many Turing machines. Fix an undecidable set $A$ and consider the sequence $\alpha_0, \alpha_1, \dots$ of constant functions such that $\alpha(e)$ has constant value $0$ if $e$ is not in $A$, and has constant value $1$, otherwise. This sequence is not a numbering, as its principal function $\alpha$ is not partial computable. Otherwise, since $\alpha$ is total, $\alpha$ and then also the function $e \mapsto \alpha(e, e)$ were computable. This is a contradiction since, by construction, the latter function is the characteristic function of the undecidable set $A$.

<!-- <div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Turing Machine $\iff$ Algorithm)</span></p>
</div> -->

A fundamental result is that a universal numbering for all partial computable functions indeed exists.

[Universal Turing Machine](/subpages/computability-and-complexity/universal-turing-machine/)

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Universal Turing Machine theorem)</span></p>

There exists a universal numbering.

</div>

<!-- *Proof*: See lecture and exercises. -->
*Proof*:

**Claim.** There exists a numbering $(\alpha_e)_{e\in\mathbb N}$ that enumerates *all* partial computable functions.

**Step 1:** Effectively enumerate Turing machines

Fix once and for all:
1. a finite alphabet for describing Turing machines (states, tape symbols, directions, etc.), and
2. a concrete syntax for writing down a machine as a finite string over that alphabet.

Now enumerate **all finite strings** in (say) length-lexicographic order:

$$w_0, w_1, w_2, \dots$$

This enumeration is computable: given $e$, we can compute $w_e$.

Some strings encode a valid Turing machine under the chosen syntax; others do not. Define a sequence of machines $(M_e)_{e\in\mathbb N}$ by:

* if $w_e$ is a valid code of a Turing machine, let $M_e$ be that machine;
* otherwise, let $M_e$ be a fixed "dummy" machine that never halts on any input.

Then **every** Turing machine $M$ appears as $M_e$ for at least one index $e$ (namely, for any $e$ such that $w_e$ is a code of $M$).

**Step 2:** Define the induced principal function

Let $\phi(e,x)$ be the partial function:

$$\phi(e,x)\ \simeq\ \text{“the output of machine }M_e\text{ on input }x\text{”}.$$

Equivalently, $\phi_e(x) \simeq \phi(e,x)$.

So $\phi$ is the principal function of the sequence $(\phi_e)$ in the sense of (7.1).

**Step 3:** $\phi$ is partial computable (existence of a universal simulator)

We show there is a single Turing machine $U$ that computes $\phi$.

On input $\langle e,x\rangle$ (using any fixed computable pairing function):

1. Compute the string $w_e$. We do not look it up in an already-generated list. Remember that we make the enumeration $w_0, w_1, w_2, \dots$ effective, meaning that mapping $g(e)=w_e$ is computable ((total) computable function $g$).
2. Check whether $w_e$ is a well-formed code of a Turing machine.
   * If not, loop forever (so $\phi(e,x)$ is undefined, matching the dummy $M_e$).
3. If yes, interpret $w_e$ as the transition table of $M_e$, and **simulate** $M_e$ on input $x$, step by step:
   * maintain a coded representation of the simulated tape contents, head position, and state,
   * repeatedly apply the transition rule.
4. If the simulation halts with output $y$, then output $y$.

This simulation procedure is completely mechanical and uses only finite, local operations at each simulated step, so $U$ is a Turing machine. Therefore the function computed by $U$ is partial computable, and by construction it is exactly $\phi$. Hence $\phi$ is a partial computable principal function, so ($\phi_e$) is a **numbering**.

**Step 4:** The numbering is universal

Let $$f:\mathbb N\rightharpoonup\mathbb N$$ be any partial computable function. By definition, some Turing machine $M$ computes $f$.

Since every Turing machine occurs somewhere in the list $(M_e)$, pick $e$ with $M_e = M$. Then for all $x$,

$$\phi_e(x)\ \simeq\ \text{output of }M_e\text{ on }x\ \simeq\ \text{output of }M\text{ on }x\ \simeq\ f(x).$$

So $f = \phi_e$. Thus $\lbrace\phi_e : e\in\mathbb N\rbrace$ is exactly the class of all partial computable functions, i.e. the numbering is **universal**.

This proves Theorem 171. $\square$

This is exactly the construction the Definition 172 refers to: the effective list $(M_e)$ (obtained from enumerating codes) is the “standard enumeration of Turing machines,” and $(\phi_e)$ is the induced standard universal numbering of partial computable functions.

The construction in the proof of this theorem gives rise to a canonical numbering.

<!-- $\textbf{Definition 172:}$ The sequence $M_0, M_1, \dots$ constructed in the proof of Theorem 171 is called the standard enumeration of Turing machines. The derived universal numbering $\phi_1, \phi_2, \dots$, where $\phi_i = \phi_{M_i}$, is called the standard enumeration of partial computable functions. -->
<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Standard enumeration of Turing machines, Standard enumeration of partial computable functions)</span></p>

Let $M_0, M_1, M_2, \ldots$ be the sequence of Turing machines constructed in the proof of Theorem 171. We call this sequence the **standard enumeration of Turing machines**.

In derived universal numbering for each $i\in\mathbb{N}$, let $\phi_i$ denote the partial function computed by $M_i$ (that is, $\phi_i=\phi_{M_i}$). Then the sequence $\phi_0, \phi_1, \phi_2, \ldots$ is a universal numbering, and it is called the **standard enumeration of partial computable functions**.
</div>

While we can enumerate all partial computable functions, a famous diagonalization argument shows that we cannot do the same for the set of total computable functions.

<!-- $\textbf{Theorem 173:}$ There exists no numbering of the set of computable functions. -->
<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Numbering of set of computable functions does not exists)</span></p>

There exists no numbering of the set of computable functions.

</div>

*Proof.* Suppose, for a contradiction, that $(\alpha_e)\_{e\in\mathbb N}$ is a numbering whose range consists exactly of all total computable functions. Let $\alpha$ be its principal function, so

$$\alpha(e,x)=\alpha_e(x)\qquad\text{for all }e,x\in\mathbb N.$$

Because every $\alpha_e$ is total, $\alpha$ is total as well. By definition of “numbering”, $\alpha$ is partial computable; since it is total, it is (total) computable.

Define a new function $d:\mathbb N\to\mathbb N$ by

$$d(e)=\alpha(e,e)+1.$$

Since (\alpha) is computable, so is $d$; moreover $d$ is total.

We claim that $d$ is not equal to any $\alpha_e$. Indeed, fix $e$. Then

$$d(e)=\alpha(e,e)+1=\alpha_e(e)+1\neq \alpha_e(e),$$

so $d$ and $\alpha_e$ differ on input $e$. Therefore $d\neq \alpha_e$ for every $e$, contradicting the assumption that the list $(\alpha_e)$ contains all total computable functions.


Question: why cannot we use the same logic to disprove existence of universal numbering?

In the proof above, the *one* property that makes the diagonal argument go through is:

> **$\alpha(e,e)$ is defined for every $e$.**

That is exactly what you get when your list $(\alpha_e)$ contains **only total** functions: the principal function $\alpha(e,x)=\alpha_e(x)$ is then **total**, so $\alpha(e,e)$ always has a value, and the diagonal

$$d(e)=\alpha(e,e)+1$$

is a **total** computable function that disagrees with $\alpha_e$ on input $e$. Contradiction.

**Why it breaks for a universal numbering**

A *universal numbering* (like the standard $(\phi_e)$) enumerates **partial** computable functions. Then the principal function is only **partial**:

$$\alpha(e,x)\simeq \alpha_e(x),$$

and crucially, $\alpha(e,e)$ may be undefined.

If you try the same diagonal definition

$$d(e)\ \simeq\ 1+\alpha(e,e),$$

then:

* $d(e)$ is undefined whenever $\alpha(e,e)$ is undefined.
* So $d$ is **partial**, not total.
* And the diagonal “disagreement” step fails exactly in the undefined case:
  if $\alpha_e(e)\uparrow$, then $d(e)\uparrow$ too, so you do **not** get $d(e)\neq \alpha_e(e)$ at that $e$.

In fact, it is completely consistent for $d$ to appear somewhere in the list. Suppose $d=\alpha_k$. Then at the diagonal point $k$,

$$\alpha(k,k)=\alpha_k(k)=d(k)\simeq 1+\alpha(k,k).$$

This *forces* $\alpha(k,k)\uparrow$ (otherwise you’d get $y=1+y$). And once $\alpha(k,k)\uparrow$, both sides are undefined, so there is **no contradiction**.

**Why you can’t “fix” the diagonal to make it total**

You might try:

$$
g(e)=
\begin{cases}
1+\alpha(e,e) & \text{if }\alpha(e,e)\downarrow,\\
0 & \text{otherwise.}
\end{cases}
$$

This *would* be total and would diagonalize. But computing it requires deciding whether $\alpha(e,e)$ halts—i.e. deciding the diagonal halting set—so $g$ is not computable in general. That “totalization” is exactly where the halting problem blocks you.

**Summary**

* **Total case (no numbering of all total computable functions):** $\alpha$ is total, so the diagonal $d(e)=\alpha(e,e)+1$ is total and forces disagreement → contradiction.
* **Universal numbering case (partial functions):** $\alpha$ is partial, so the diagonal may be undefined exactly when needed, and the “disagree at $e$” step can fail → no contradiction. $\square$

### Halting Problems

Numberings provide a powerful tool for proving the existence of specific, concrete undecidable problems. The most famous of these is the Halting Problem.

<!-- $\textbf{Lemma 174:}$ Let $\alpha_0, \alpha_1, \dots$ be a numbering, and let $H_\alpha = \lbrace e : \alpha_e(e) \downarrow \rbrace$. The set $H_\alpha$ is recursively enumerable. If the numbering $\alpha_0, \alpha_1, \dots$ contains all computable functions, then the set $H_\alpha$ is not decidable. -->
<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(diagonal halting set is r.e.)</span></p>

Let $(\alpha_e)_{e\in\mathbb N}$ be a numbering, and let $\alpha$ be its principal function, so that

$$\alpha(e,x)\simeq \alpha_e(x).$$

Define the **diagonal halting set**

$$H_\alpha=\lbrace e\in\mathbb N : \alpha_e(e)\downarrow\rbrace.$$

Then $H_\alpha$ is recursively enumerable (r.e.). Moreover, if the numbering $(\alpha_e)$ contains every total computable function, then $H_\alpha$ is not decidable.

</div>

*Proof*:

**(1) $H_\alpha$ is r.e.**
Since $\alpha$ is partial computable, the partial function

$$d(e)\ \simeq\ \alpha(e,e)$$

is also partial computable. By definition,

$$\operatorname{dom}(d)=\lbrace e : \alpha(e,e)\downarrow\rbrace=\lbrace e : \alpha_e(e)\downarrow\rbrace=H_\alpha.$$

A set is r.e. $\iff$ it is the domain of some partial computable function. Hence $H_\alpha$ is r.e.

**(2) If $(\alpha_e)$ contains all total computable functions, then $H_\alpha$ is undecidable.**
We prove the contrapositive. Assume $H_\alpha$ is decidable. Define a total function $g:\mathbb N\to\mathbb N$ by

$$g(e)=
\begin{cases}
1+\alpha_e(e), & \text{if } e\in H_\alpha\ (\text{i.e. }\alpha_e(e)\downarrow), \\
0, & \text{if } e\notin H_\alpha\ (\text{i.e. }\alpha_e(e)\uparrow).
\end{cases}$$

Because $H_\alpha$ is decidable, we can compute $g(e)$ effectively: first decide whether $e\in H_\alpha$. If not, output $0$. If yes, then $\alpha_e(e)$ halts, so we simulate it to obtain its value and output one more. Therefore $g$ is total computable.

Now $g$ differs from every $\alpha_e$. Fix $e$.

* If $\alpha_e(e)\downarrow$, then $g(e)=1+\alpha_e(e)\neq \alpha_e(e)$.
* If $\alpha_e(e)\uparrow$, then $g(e)=0\downarrow$, so $g(e)$ is defined while $\alpha_e(e)$ is not.

In either case, $g\neq \alpha_e$. Thus $g$ is a total computable function not appearing in the numbering. Hence the numbering cannot contain all total computable functions. This proves the contrapositive, and therefore if $(\alpha_e)$ contains all total computable functions, $H_\alpha$ is not decidable. $\square$

Applying this lemma to the standard numbering gives us the classic Halting Problem.

<!-- $\textbf{Definition 175:}$ Let $\phi_0, \phi_1, \dots$ be the standard numbering of the partial computable functions. The sets  
$$H = H_{diag} = \lbrace e : \phi_e(e) \downarrow\rbrace \quad \text{and} \quad H_{gen} = \lbrace\langle e, x \rangle : \phi_e(x) \downarrow\rbrace$$
are called the diagonal halting problem, or simply the halting problem, and the general halting problem, respectively. -->
<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Standard numbering, Diagonal halting problem, General halting problem)</span></p>

Let $(\phi_e)_{e\in\mathbb{N}}$ be the standard numbering of the partial computable functions. Define

$$
H = H_{\mathrm{diag}} =\lbrace e\in\mathbb{N} : \phi_e(e)\downarrow \rbrace
\qquad
H_{\mathrm{gen}} =\lbrace \langle e,x\rangle \in\mathbb{N} : \phi_e(x)\downarrow \rbrace.
$$

The set ($H$ is called the **diagonal halting problem** (often simply the **halting problem**), and $H_{\mathrm{gen}}$ is called the **general halting problem**.

</div>

<!-- $\textbf{Theorem 176:}$ The halting problem $H$ is recursively enumerable but not decidable. The complement $\mathbb{N} \setminus H$ of the halting problem is not recursively enumerable. -->
<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(The halting problem $H$ is recursively enumerable but not decidable)</span></p>

The halting problem $H$ is recursively enumerable but not decidable. Furthermore, its complement $\mathbb{N}\setminus H$ is not recursively enumerable.
</div>

*Proof*: The standard numbering is universal and thus contains all computable functions. By Lemma 174, $H$ is recursively enumerable but not decidable. If the complement of $H$ were also recursively enumerable, then by Proposition 165, $H$ would be decidable, which is a contradiction. Therefore, the complement of $H$ is not recursively enumerable.

### Gödel Numberings

Some numberings are "better" than others in the sense that they are general enough to simulate any other numbering. These are called Gödel numberings.

<!-- $\textbf{Definition 177:}$ A numbering $\beta_0, \beta_1, \dots$ is called a Gödel numbering or acceptable numbering, if for every other numbering $\alpha_0, \alpha_1, \dots$ there exists a computable function $f$ such that for all $e$ it holds that  $\alpha_e = \beta_{f(e)}$.  -->

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Gödel numbering or acceptable numbering)</span></p>

A numbering $(\beta_e)\_{e\in\mathbb{N}}$ is called a **Gödel numbering** (or **acceptable numbering**) if for every numbering $(\alpha_e)\_{e\in\mathbb{N}}$ there exists a computable function $f$ such that, for all $e\in\mathbb{N}$,

$$\alpha_e = \beta_{f(e)}.$$

(Equivalently: every other numbering can be effectively translated into $(\beta_e)$.)

</div>

The function $f$ acts as a "compiler" or "translator" from indices in the $\alpha$ numbering to equivalent indices in the $\beta$ numbering.

> **Convention**: **“Gödel numbering” is often used as a synonym for “acceptable/admissible numbering”** (including this course), i.e., not *just* any coding, but a coding with the right **effectiveness** properties. Different areas use the word “Gödel numbering” at different strictness levels.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Admissible numbering)</span></p>

An **admissible numbering** (often also called an **acceptable numbering** or **Gödel numbering** in recursion theory) is a “good” way to assign indices $e\in\mathbb N$ to partial computable functions so that programs can be manipulated **effectively**. In computability theory, **admissible numberings** are **enumerations (numberings)** of the set of partial computable functions that can be converted *to and from* the **standard numbering**. These numberings are also called acceptable numberings and acceptable programming systems.

Rogers' equivalence theorem shows that all acceptable programming systems are equivalent to each other in the formal sense of numbering theory.

In the language of our definition (principal function $\alpha(e,x)$):

**Standard definition**

A numbering $(\alpha_e)_{e\in\mathbb N}$ of partial computable functions is **admissible** if:

1. **Universality:** it enumerates *all* partial computable functions:
   
   $$\lbrace\alpha_e : e\in\mathbb N\lbrace = \lbrace\text{all partial computable } \mathbb N\rightharpoonup\mathbb N\rbrace.$$

2. **(s)-(m)-(n) (parameter) property:** there is a **total computable** function $s(e,x)$ such that for all $e,x,y$,
   
   $$\alpha_{s(e,x)}(y)\ \simeq\ \alpha_e(\langle x,y\rangle).$$
   
   Intuition: from an index $e$ for a 2-argument program and a value $x$, you can **compute an index** for the 1-argument program where $x$ is “hardwired in.”

(Here $\langle x,y\rangle$ is any fixed computable pairing function.)

**Why this matters (intuition)**
* A *mere* universal numbering lists all computable partial functions, but it might be “unnatural” in the sense that basic operations like “fix the first input” might not be computable at the level of indices.
* Admissibility guarantees the numbering behaves like a reasonable programming system: you can compile/partial-evaluate effectively.

**Equivalent characterization (Rogers / translation property)**
A universal numbering $\alpha$ is admissible iff for **every** other numbering $\beta$ of the partial computable functions, there exists a **total computable translator** $t$ such that for all $e$,

$$\beta_e = \alpha_{t(e)}.$$

Meaning: any other effective programming system can be **compiled into** $\alpha$ by a computable map on codes.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Classic Gödel Numbering)</span></p>

In mathematical logic, a **Gödel numbering** is a function that assigns to each symbol and well-formed formula of some formal language a unique natural number, called its **Gödel number**. Kurt Gödel developed the concept for the proof of his incompleteness theorems.

A Gödel numbering can be interpreted as an encoding in which a number is assigned to each symbol of a mathematical notation, after which a sequence of natural numbers can then represent a sequence of symbols. These sequences of natural numbers can again be represented by single natural numbers, facilitating their manipulation in formal theories of arithmetic.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example:</span><span class="math-callout__name">(Symbols to unique natural numbers: classic Gödel numbering)</span></p>

**Gödel numbering** is a method for encoding symbols, formulas, and even whole proofs from a formal system (like arithmetic or logic) as **unique natural numbers**.

The idea is:

1. Assign each basic symbol (e.g., `0`, `S`, `+`, `=`, `(`, `)`, variables, etc.) a distinct natural number.
2. Encode a finite sequence of symbols (a string / formula) into a single natural number using a fixed, reversible scheme (commonly using prime powers).

A classic scheme is: if a formula is the symbol sequence $s_1, s_2, \dots, s_k$, and each symbol $s_i$ has code $c(s_i)$, define its Gödel number as

$$G(s_1 \dots s_k) = 2^{c(s_1)} \cdot 3^{c(s_2)} \cdot 5^{c(s_3)} \cdots p_k^{c(s_k)},$$

where $p_k$ is the $k$-th prime.

Because prime factorization is unique, you can decode the number back into the exact original sequence. This is what lets metamathematical statements like “this formula is provable” be translated into statements *about numbers* inside arithmetic.

</div>

<div class="pmf-grid">
  <figure>
    <img src="{{ '/assets/images/notes/computability-and-complexity/godel_numbering1.jpg' | relative_url }}" alt="a" loading="lazy">
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/computability-and-complexity/godel_numbering2.jpg' | relative_url }}" alt="a" loading="lazy">
  </figure>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Standard numbering is Gödel numbering)</span></p>

The standard numbering $\phi_0, \phi_1, \dots$ is a Gödel numbering.
</div>
<!-- $\textbf{Theorem 178:}$ The standard numbering $\phi_0, \phi_1, \dots$ is a Gödel numbering. -->

*Proof*: Let $\alpha_0, \alpha_1, \dots$ be an arbitrary numbering, and let its principal function $\alpha$ be computed by a Turing machine $M$. So, $M(\langle e, x \rangle) \simeq \alpha_e(x)$. For any given $e$, we can construct a new Turing machine $M'$ that, on input $x$, first transforms $x$ into the pair $\langle e, x \rangle$ and then simulates $M$. This new machine $M'$ computes the function $\alpha_e$. The process of constructing $M'$ from $M$ and $e$ is effective. The standard enumeration of Turing machines is constructed in such a way that we can computably find an index $f(e)$ for this machine $M'$. Thus, there exists a computable function $f$ such that $\phi_{f(e)} = \alpha_e$ for all $e$.

Not all universal numberings have this powerful translation property.

<!-- $\textbf{Theorem 179:}$ There is a universal numbering that is not a Gödel numbering. -->
<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Universal numbering is not Gödel numbering)</span></p>

There is a universal numbering that is not a Gödel numbering.
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Exists universal numbering that is not Gödel numbering)</span></p>

There exists a universal numbering that is not a Gödel numbering.

</div>

*Proof.* Let $(\phi_e)\_{e\in\mathbb N}$ be the standard numbering of the partial computable functions. Define a new sequence $(\psi_e)\_{e\in\mathbb N}$ as follows. For each $e=\langle i,t\rangle$, set

$$
\psi_{\langle i,t\rangle}(x)\ \simeq
\begin{cases}
\phi_i(x), & x\neq 0, \\
\uparrow, & x=0\ \text{and}\ t=0 \\
t-1, & x=0\ \text{and}\ t>0.
\end{cases}
$$

In words: $\psi_{\langle i,t\rangle}$ behaves like $\phi_i$ on all inputs $x\neq 0$, while its value at $0$ is independently controlled by $t$.

**Step 1: $(\psi_e)$ is a numbering.**
Its principal function $\psi(e,x)$ is partial computable: on input $(e,x)$, decode $e$ as $\langle i,t\rangle$. If $x\neq 0$, simulate $\phi_i(x)$. If $x=0$, output $\uparrow$ when $t=0$ and output $t-1$ when $t>0$. Hence $(\psi_e)$ is a numbering.

**Step 2: $(\psi_e)$ is universal.**
Fix $i$. We show that $\phi_i$ occurs somewhere in the $\psi$-list. Consider the value $\phi_i(0)$.

* If $\phi_i(0)\uparrow$, then we choose $e=\langle i,0\rangle$ and we have $\psi_{\langle i,0\rangle}(0)\uparrow$ and $\psi_{\langle i,0\rangle}(x)=\phi_i(x)$ for $x\neq 0$. Hence $\psi_{\langle i,0\rangle}=\phi_i$.
* If $\phi_i(0)\downarrow=y$, then we choose $e=\langle i,y+1\rangle$ and we have $\psi_{\langle i,y+1\rangle}(0)=y$ and again $\psi_{\langle i,y+1\rangle}(x)=\phi_i(x)$ for $x\neq 0$. Hence $\psi_{\langle i,y+1\rangle}=\phi_i$.

Thus every $\phi_i$ appears among the $\psi_e$, so $(\psi_e)$ is universal.

**Step 3: $(\psi_e)$ is not a Gödel numbering.**
Define a computable function $g$ such that, for every $e$,

$$
\phi_{g(e)}(x)\ \simeq
\begin{cases}
0, & \text{if } \phi_e(e)\downarrow,\\
\uparrow, & \text{if } \phi_e(e)\uparrow.
\end{cases}
$$

Such a $g$ exists and is computable: given $e$, effectively construct a Turing machine that ignores its input $x$ and outputs $0$ if $\phi_e(e)$ halts and loops otherwise (I guess such a TM just simulates $\phi_e(e)$ and simulation either diverges or produces some output and the TM outputs zero); Such a TM has its corresponding partial computable function, which is somewhere in the standard numbering. Let $g(e)$ be an index of this machine in the standard enumeration.

Now suppose, for contradiction, that $(\psi_e)$ were a Gödel numbering. Then there would exist a computable translation function $f$ such that

$$\phi_n = \psi_{f(n)}\qquad\text{for all }n.$$

Applying this with $n=g(e)$, we obtain

$$\phi_{g(e)} = \psi_{f(g(e))}\qquad\text{for all }e.$$

Write $f(g(e))=\langle i(e),t(e)\rangle$, and let $\mathrm{inv}_2(\langle u,v\rangle)=v$ denote the second projection. We now look at the value at input $0$.

* If $\phi_e(e)\downarrow$, then by construction $\phi_{g(e)}(0)=0$. Hence $\psi_{f(g(e))}(0)=0$. By the definition of $\psi$, this can happen only when $t(e)=1$, i.e. $\mathrm{inv}_2(f(g(e)))=1$.
* If $\phi_e(e)\uparrow$, then $\phi_{g(e)}(0)\uparrow$. Hence $\psi_{f(g(e))}(0)\uparrow$, which (by the definition of $\psi$) happens only when $t(e)=0$, i.e. $\mathrm{inv}_2(f(g(e)))=0$.

Therefore the function

$$h(e)\ :=\ \mathrm{inv}_2(f(g(e)))$$

is exactly the characteristic function of the diagonal halting set

$$H=\lbrace e:\phi_e(e)\downarrow\rbrace.$$

But $h$ is computable as a composition of computable functions ($\mathrm{inv}_2$, $f$, and $g$. This would make $H$ decidable, contradicting the undecidability of the halting problem. Hence $(\psi_e)$ cannot be a Gödel numbering. $\square$

<!-- *Proof*: Let $\phi_0, \phi_1, \dots$ be the standard numbering. We define a new sequence of partial functions $\psi_0, \psi_1, \dots$. For an index $e = \langle i, t \rangle$, we define $\psi_e$ as:  

$$\psi_{\langle i, t \rangle}(x) = \begin{cases} \phi_i(x) & \text{in case } x \ne 0, \\ \uparrow & \text{in case } x=0 \text{ and } t=0, \\ t-1 & \text{in case } x=0 \text{ and } t > 0. \end{cases}$$  

Essentially, $\psi_{\langle i, t \rangle}$ is a copy of $\phi_i$, except at input $0$, where its behavior is independently controlled by $t$.

* The sequence $\psi_0, \psi_1, \dots$ is a numbering because its principal function $\psi(e, x)$ is partial computable. To compute $\psi(e, x)$, we first decode $e$ into its components $i$ and $t$. If $x > 0$, we simulate $\phi_i(x)$. If $x=0$, we determine the output based on $t$.
* This numbering is universal. For any partial computable function $\phi_i$, we can find an index $e$ such that $\psi_e = \phi_i$. If $\phi_i(0)$ is undefined, we can choose $e = \langle i, 0 \rangle$. If $\phi_i(0)$ is defined and equals $y$, we can choose $e = \langle i, y+1 \rangle$.
* This numbering is not a Gödel numbering. We can construct a specific computable function $g$ such that for all $e$ and $x$:  
  
  $$\phi_{g(e)}(x) = \begin{cases} 0, & \text{in case } \phi_e(e) \downarrow, \\ \uparrow, & \text{otherwise.} \end{cases}$$ 
  
  The function $g$ is computable: given $e$, one can effectively construct the description of a Turing machine that outputs $0$ if $\phi_e(e)$ halts, and loops otherwise. $g(e)$ is the index of this machine in the standard enumeration.
* Now, assume for contradiction that $\psi$ were a Gödel numbering. Then there would be a computable function $f$ such that $\phi_e = \psi_{f(e)}$ for all $e$. Applying this to our constructed functions, we get $\phi_{g(e)} = \psi_{f(g(e))}$ for all $e$. Let's examine the behavior at input $0$.
  * If $e \in H$, then $\phi_e(e) \downarrow$. This means $\phi_{g(e)}(0) = 0$. So, $\psi_{f(g(e))}(0) = 0$. According to the definition of $\psi$, this happens only if the second component of the index $f(g(e))$ is $1$. Let $\text{inv}_2$ be the function that gives the second component of a pair (i.e., $\text{inv}_2(\langle x, y \rangle) = y$). Then $\text{inv}_2(f(g(e))) = 1$.
  * If $e \notin H$, then $\phi_e(e) \uparrow$. This means $\phi_{g(e)}(0) \uparrow$. So, $\psi_{f(g(e))}(0) \uparrow$. This happens only if the second component of $f(g(e))$ is $0$. Then $\text{inv}_2(f(g(e))) = 0$.
* This implies that the function $\text{inv}_2 \circ f \circ g$ is the characteristic function of the halting problem $H$. Since $f$, $g$, and $\text{inv}_2$ are all computable, their composition is also computable. But this would mean $H$ is decidable, which is a contradiction. Therefore, $\psi$ cannot be a Gödel numbering. -->

## Many-one Reducibility and Rice’s Theorem

To compare the difficulty of undecidable problems, we introduce the concept of reducibility. One of the most fundamental types is many-one reducibility.

### Many-one Reducibility

<!-- $\textbf{Definition 180:}$ A set $A$ is many-one reducible to a set $B$, abbreviated as $A$ is m-reducible to $B$ or $A \le_m B$, if there exists a computable function $f$ such that for all $x \in \mathbb{N}$, it holds that  $x \in A \text{ if and only if } f(x) \in B$.  -->
<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Many-one reduction)</span></p>

A set $A$ is **many-one reducible** to a set $B$ (also: **m-reducible**), written $A \le_m B$, if there exists a computable function $f$ such that for all $x \in \mathbb{N}$,

$$x \in A \iff f(x) \in B.$$

</div>

The function $f$ transforms instances of problem $A$ into instances of problem $B$ while preserving the yes/no answer. This means that if we can solve $B$, we can use $f$ to solve $A$.

> **Remark 181**: Let set $A$ be decidable and set $B$ be distinct from $\emptyset$ and $\mathbb{N}$. Then $A$ is m-reducible to $B$. In order to obtain a function $f$ that witnesses the latter, fix $b_0 \notin B$ and $b_1 \in B$, and let  
> 
> $$f(x) = \begin{cases} b_0, & \text{if } x \notin A, \\ b_1, & \text{if } x \in A. \end{cases}$$
> 
> On the other hand, there is just a single set that is m-reducible to $\emptyset$ since $A \le_m \emptyset$ implies $A = \emptyset$, and a similar statement holds for $\mathbb{N}$ in place of $\emptyset$. This is considered to be an anomaly and, accordingly, m-reducibility is sometimes defined such that, in addition to the relationships valid according to Definition 180, all decidable sets $A$ are m-reducible to $\emptyset$ and $\mathbb{N}$.


<!-- $\textbf{Proposition 182:}$ The relation m-reducibility is reflexive and transitive. -->
<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(m-reductions preserve decidability)</span></p>

The relation m-reducibility is **reflexive** and **transitive**.

</div>

*Proof*:

**Reflexive.** Let $A\subseteq\mathbb N$. Take the identity function $f(x)=x$. This function is computable and satisfies

$$x\in A \iff f(x)=x\in A.$$

Hence $A\le_m A$.

**Transitive.** Suppose $A\le_m B$ via a computable function $f$, and $B\le_m C$ via a computable function $g$. Define $h=g\circ f$, i.e. $h(x)=g(f(x)$). Since computable functions are closed under composition, $h$ is computable. Moreover, for every $x$,

$$x\in A \iff f(x)\in B \iff g(f(x))\in C \iff h(x)\in C.$$

Thus $A\le_m C$. $\square$

### M-reducibility allows us to infer properties about sets.

<!-- $\textbf{Proposition 183:}$ Let $A$ and $B$ be sets such that $A \le_m B$. 
* **(i)** If $B$ is decidable, then $A$ is also decidable. 
* **(ii)** If $A$ is undecidable, then $B$ is also undecidable. -->
<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(m-reductions preserve decidability/undecidability)</span></p>

Let $A$ and $B$ be sets such that $A \le_m B$. 
* **(i)** If $B$ is decidable, then $A$ is also decidable. 
* **(ii)** If $A$ is undecidable, then $B$ is also undecidable.

</div>

*Proof*:

Assume $A\le_m B$. Then there exists a computable function $f$ such that for all $x$,

$$x\in A \iff f(x)\in B.$$

**(i)** Suppose $B$ is decidable, and let $D_B$ be a decider for $B$. We build a decider $D_A$ for $A$ as follows: on input $x$,

1. compute $y=f(x)$;
2. run $D_B$ on input $y$ and output the same yes/no answer.

Because $f$ is computable and $D_B$ always halts, this procedure always halts. Correctness follows from the defining property of $f$:

$$D_A \text{ accepts } x \iff D_B \text{ accepts } f(x) \iff f(x)\in B \iff x\in A.$$

Hence $A$ is decidable.

**(ii)** This is the contrapositive of $i$. If $B$ were decidable, then $i$ would imply that $A$ is decidable. Therefore, if $A$ is undecidable, $B$ must be undecidable. $\square$

A key result is that the halting problem $H$ is a "hardest" problem among all r.e. sets with respect to m-reducibility.

<!-- $\textbf{Theorem 184:}$ Let $A$ and $B$ be sets. 
* **(i)** If $A \le_m B$ and $B$ is r.e., then $A$ is also r.e. 
* **(ii)** For every r.e. set $A$, it holds that $A \le_m H$. -->
<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(r.e. sets and many-one reductions to the halting problem)</span></p>

Let $A$ and $B$ be sets. 
* **(i)** If $A \le_m B$ and $B$ is r.e., then $A$ is also r.e. 
* **(ii)** For every r.e. set $A$, it holds that $A \le_m H$.

</div>

*Proof*: 

* **(i)** Let $A \le_m B$ via a computable function $f$, and let $B$ be r.e. Then $B$ is the domain of some partial computable function $\alpha$. The set $A$ is the domain of the partial computable function $\alpha \circ f : x \mapsto \alpha(f(x))$. Since $A$ is the domain of a partial computable function, it is r.e. 
* **(ii)** Let $A$ be an r.e. set. Then $A = \text{dom}(\alpha)$ for some partial computable function $\alpha$. Let $\phi_0, \phi_1, \dots$ be the standard numbering. Since it is a Gödel numbering, we can effectively construct a new numbering $\beta_0, \beta_1, \dots$ where for a given $e$, the function $\beta_e$ ignores its own input and simply computes $\alpha(e)$. That is, $\beta_e(x) \simeq \alpha(e)$. The principal function $(e,x) \mapsto \beta_e(x)$ is partial computable, so this is a valid numbering. Since the standard numbering is a Gödel numbering, there exists a computable function $f$ such that $\beta_e = \phi_{f(e)}$ for all $e$. This function $f$ witnesses that 
  
  $$A \le_m H:  e \in A \iff \alpha(e) \downarrow \iff \beta_e \text{ is total} \iff \beta_e(f(e)) \downarrow \iff \phi_{f(e)}(f(e)) \downarrow \iff f(e) \in H$$

**Another proof:**

**(i)**

Assume $A \le_m B$ via a computable function $f$, and assume $B$ is r.e. Then there exists a partial computable function $\alpha$ such that

$$B = \operatorname{dom}(\alpha).$$

Define a partial function $\gamma$ by

$$\gamma(x)\ \simeq\ \alpha(f(x)).$$

Since $f$ is computable and $\alpha$ is partial computable, their composition $\gamma=\alpha\circ f$ is partial computable. Moreover,

$$x\in A \iff f(x)\in B \iff \alpha(f(x))\downarrow \iff \gamma(x)\downarrow,$$

so $A=\operatorname{dom}(\gamma)$. Hence $A$ is r.e.

**(ii)**

Let $A$ be r.e. Then $A=\operatorname{dom}(\alpha)$ for some partial computable function $\alpha$.

For each $e$, define a partial computable function $\beta_e$ that ignores its input and simulates $\alpha(e)$; formally,

$$\beta_e(x)\ \simeq\ \alpha(e)\qquad(\text{for all }x).$$

Intuitively, $\beta_e(0)$ halts iff (\alpha(e)) halts, i.e. iff $e\in A$.

Because the standard numbering $(\phi_n)\_{n\in\mathbb N}$ is a Gödel numbering, there exists a computable function $f$ such that

$$\phi_{f(e)}=\beta_e\qquad\text{for all }e.$$

Now compute:

$$
e\in A
\iff \alpha(e)\downarrow
\iff \beta_e(0)\downarrow
\iff \phi_{f(e)}(0)\downarrow.
$$

To reduce to the *diagonal* halting set $H=\lbrace n:\phi_n(n)\downarrow\rbrace$, define a computable map $g$ by

$$g(e)=s(f(e)),$$

where $s$ is any computable “padding” function with the property that $\phi_{s(n)}(x)\simeq \phi_n(0)$ for all $x$. (Equivalently: from an index $n$ we can effectively produce an index of a machine that ignores its input and runs $\phi_n$ on input $0$.)

Then, for all $e$,

$$
e\in A
\iff \phi_{f(e)}(0)\downarrow
\iff \phi_{g(e)}(g(e))\downarrow
\iff g(e)\in H.
$$

Thus $A\le_m H$. $\square$

This theorem provides a complete characterization of recursively enumerable sets in terms of m-reducibility to the halting problem.

<!-- $\textbf{Corollary 185:}$ A set $A$ is r.e. if and only if $A \le_m H$. -->
<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(name of Corollary)</span></p>

A set $A$ is r.e. if and only if $A \le_m H$.

</div>

*Proof*: Immediate by Theorem 184.

<!-- $\textbf{Corollary 186:}$ It holds that $H \le_m H_{gen}$ and $H_{gen} \le_m H$. -->
<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(name of Corollary)</span></p>

It holds that $H \le_m H_{gen}$ and $H_{gen} \le_m H$.

</div>

*Proof*:

* The reduction $H \le_m H_{gen}$ is witnessed by the computable function $f(e) = \langle e, e \rangle$. We have $e \in H \iff \phi_e(e) \downarrow \iff \langle e, e \rangle \in H_{gen}$.
* The reduction $H_{gen} \le_m H$ follows directly from Corollary 185, because $H_{gen}$ is a recursively enumerable set.

M-reducibility also interacts cleanly with set complements. Let $\bar{X} = \mathbb{N} \setminus X$.

> **Remark 187**: If $A \le_m B$ via a function $f$, then also $\bar{A} \le_m \bar{B}$ via $f$ because we have 
> 
> $$x \in \bar{A} \iff x \notin A \iff f(x) \notin B \iff f(x) \in \bar{B}$$ 

This leads to an important non-reducibility result.

<!-- $\textbf{Corollary 188:}$ Let $A$ be an undecidable r.e. set. Then it holds that $A \not\le_m \bar{A}$ and $\bar{A} \not\le_m A$. In particular, we have 

$$H \not\le_m \bar{H} \text{ and } \bar{H} \not\le_m H$$  -->
<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(name of Corollary)</span></p>

Let $A$ be an undecidable r.e. set. Then neither set is many-one reducible to the other’s complement:

$$
A \not\le_m \overline{A}
\quad\text{and}\quad
\overline{A} \not\le_m A.
$$

In particular, we have

$$H \not\le_m \bar{H} \text{ and } \bar{H} \not\le_m H$$

</div>

*Proof*: The complement $\bar{A}$ cannot be r.e., otherwise $A$ would be decidable by Proposition 165. If we had $A \le_m \bar{A}$, since $\bar{A}$ is not r.e., this would imply by Theorem 184(i) that $A$ is not r.e., which contradicts that $A$ is an r.e. set. If we had $\bar{A} \le_m A$, since $A$ is r.e., this would imply that $\bar{A}$ is also r.e., which we know is false.

### Index Sets and Rice’s Theorem

Rice's Theorem is a powerful generalization of the undecidability of the halting problem. It states that any non-trivial property of the behavior of Turing machines (i.e., of the partial computable functions they compute) is undecidable. To formalize this, we use the notion of an index set.

<!-- $\textbf{Definition 189:}$ An index set is a set $I$ of natural numbers such that for all $e$ and $e'$ it holds that  

$$e \in I \text{ and } \phi_e = \phi_{e'} \implies e' \in I$$  

An index set is nontrivial if it differs from $\emptyset$ and $\mathbb{N}$. -->
<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Index Set)</span></p>

A set $I \subseteq \mathbb{N}$ is an **index set** if for all $e,e'$,

$$e \in I \ \text{and}\ \phi_e=\phi_{e'} \ \Longrightarrow\ e' \in I.$$

An index set is **nontrivial** if it is neither $\emptyset$ nor $\mathbb{N}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Expanded definition of Index set)</span></p>

Let $\varphi_e$ be a computable enumeration of all partial computable functions, and let $W_e$ be a computable enumeration of all computably enumerable (c.e.) sets.

Let $\mathcal{A}$ be a collection of partial computable functions. Define its **index set** by

$$A=\lbrace x:\varphi_x\in\mathcal{A}\rbrace.$$

More generally, a set $A\subseteq\mathbb{N}$ is an **index set** if whenever $x,y\in\mathbb{N}$ satisfy $\varphi_x\simeq \varphi_y$ (i.e., they compute the same partial function), we have

$$x\in A \iff y\in A.$$

Intuitively, index sets are subsets of $\mathbb{N}$ defined purely in terms of the functions their elements index.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Another view on index sets)</span></p>

An index set is a union of equivalence classes induced by $\simeq$ relation and some representative set of partial functions.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Index set as property of PCF)</span></p>

An index set can be viewed as a property of partial computable functions. 

For an index set $I$ and any partial computable function $\alpha$, either all or none of the indices $e$ with $\alpha = \phi_e$ are in $I$.

</div>
<!-- > **Remark 190**: An index set can be viewed as a property of partial computable functions. For an index set $I$ and any partial computable function $\alpha$, either all or none of the indices $e$ with $\alpha = \phi_e$ are in $I$. -->

<!-- $\textbf{Example 191:}$ The following sets are index sets: 
$$\lbrace e \in \mathbb{N} : \phi_e \text{ is total} \rbrace, \lbrace e \in \mathbb{N} : \phi_e(0) \uparrow\rbrace, \lbrace e \in \mathbb{N} : \text{dom}(\phi_e) \text{ is infinite} \rbrace$$ -->
<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(name of Example)</span></p>

The following sets are index sets: 

$$\lbrace e \in \mathbb{N} : \phi_e \text{ is total} \rbrace, \lbrace e \in \mathbb{N} : \phi_e(0) \uparrow\rbrace, \lbrace e \in \mathbb{N} : \text{dom}(\phi_e) \text{ is infinite} \rbrace$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Padding Lemma)</span></p>

The **Padding Lemma** is the formal way to say:

> From any program/index $e$, you can effectively produce infinitely many *different* indices that compute **the exact same** partial computable function.

**Standard statement (for an acceptable numbering)**

Let $(\varphi_e)\_{e\in\mathbb N}$ be an **acceptable (Gödel/admissible)** numbering of the partial computable functions. Then there exists a **total computable** function

$$p:\mathbb N^2 \to \mathbb N$$

(often chosen so that it’s injective in the second argument) such that for all $e,k$,

$$\varphi_{p(e,k)} = \varphi_e.$$

So $p(e,0), p(e,1), p(e,2),\dots$ are infinitely many “padded versions” of $e$: different indices, same computed function.

**Why it’s called “padding”**

In concrete models (Turing machines/programs), you can literally “pad” a program with useless extra code/comments/no-ops that don’t change its behavior but do change its code number.

### Proof idea (one clean computability-theory version)

Acceptable numberings are usually set up so you have the **s-m-n theorem** available. Define a partial computable 3-argument function that ignores the “padding parameter”:
[
g(e,k,x) ;=; \varphi_e(x).
]
This (g) is partial computable (it just simulates (e) on input (x), ignoring (k)). By **s-m-n**, there is a total computable function (s(e,k)) such that
[
\varphi_{s(e,k)}(x) ;=; g(e,k,x) ;=; \varphi_e(x).
]
Now set (p(e,k)=s(e,k)). That’s the padding function.

(If you want injectivity in (k), you can build it in by first pairing ((e,k)) into one number in a reversible computable way.)


</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Numbering can have duplicates, and in computability we expect it to)</span></p>

In an **acceptable/admissible** numbering, “infinitely many duplicates” is *exactly* the statement that each function’s index set is infinite.

Fix an acceptable numbering $(\varphi_e)$ and a partial computable function $\alpha$. The **index set** of $\alpha$ (sometimes called its *equivalence class* under extensional equality) is

$$E_\alpha = \lbrace e \in \mathbb{N} : \varphi_e = \alpha \rbrace.$$

If the numbering has **infinitely many indices** $e$ with $\varphi_e=\alpha$, then $E_\alpha$ contains infinitely many natural numbers, hence

$$\lvert E_\alpha\rvert = \infty.$$

And conversely, saying “$E_\alpha$ is infinite” means “$\alpha$ has infinitely many duplicate indices.”

Just one precision: the “infinite duplicates” fact isn’t true for *every* numbering; it’s a property of **acceptable** numberings (via the padding lemma).

</div>

<!-- $\textbf{Theorem 192 (Rice's Theorem):}$ Let $I$ be a nontrivial index set. Then it holds that 

$$H \le_m I \quad \text{or} \quad \bar{H} \le_m I$$  -->
<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Rice's Theorem)</span></p>

Let $I$ be a nontrivial index set. Then it holds that 

$$H \le_m I \quad \text{or} \quad \bar{H} \le_m I$$

</div>

*Proof of Rice's Theorem 1*: Let $I$ be a nontrivial index set. Let $\phi_\uparrow$ denote the everywhere undefined partial function. There are two cases for $I$.
* **Case 1**: I does not contain an index for $\phi_\uparrow$. Since $I$ is nontrivial, it is not empty, so there must be some index $e_0 \in I$. Let $\beta = \phi_{e_0}$. By our assumption, $\beta$ is not the everywhere undefined function. We will show $\bar{H} \le_m I$. To do this, we need a computable function $g$ such that $e \in \bar{H} \iff g(e) \in I$.
* Consider the function $g(e)$ which gives the index of a new Turing machine. This machine, on any input $x$, first simulates the computation of $\phi_e(e)$. If this simulation halts, the machine then proceeds to compute $\beta(x)$. If the simulation of $\phi_e(e)$ does not halt, the machine runs forever. The function computed by the machine with index $g(e)$ is:  
  
  $$\phi_{g(e)} = \begin{cases} \beta & \text{if } \phi_e(e)\uparrow \text{ (i.e., } e \in \bar{H}), \\ \phi_\uparrow & \text{if } \phi_e(e)\downarrow \text{ (i.e., } e \in H). \end{cases}$$

  The function $g$ is computable. Now we check the reduction:
  * If $e \in \bar{H}$, then $\phi_{g(e)} = \beta = \phi_{e_0}$. Since $e_0 \in I$ and $I$ is an index set, $g(e)$ must be in $I$.
  * If $e \in H$, then $\phi_{g(e)} = \phi_\uparrow$. By our case assumption, no index for $\phi_\uparrow$ is in $I$, so $g(e) \notin I$. Thus, $e \in \bar{H} \iff g(e) \in I$, so $\bar{H} \le_m I$.
* **Case 2**: I contains an index for $\phi_\uparrow$. In this case, the complement set $\bar{I}$ does not contain an index for $\phi_\uparrow$. Since $I$ is nontrivial, $\bar{I}$ is also a nontrivial index set. Applying the logic from Case 1 to $\bar{I}$, we find that $\bar{H} \le_m \bar{I}$. By Remark 187, this implies $H \le_m I$.

*Proof of Rice's Theorem 2.* Let $\phi_\uparrow$ denote the everywhere undefined partial computable function. Since $I$ is nontrivial, $I\neq\emptyset$ and $I\neq\mathbb N$. We distinguish two cases depending on whether $I$ contains an index of $\phi_\uparrow$.

**Case 1: $I$ contains no index of $\phi_\uparrow$**

Because $I\neq\emptyset$, choose some $e_0\in I$ and set $\beta=\phi_{e_0}$. In this case $\beta\neq \phi_\uparrow$ (otherwise $e_0$ would be an index of $\phi_\uparrow$, contradicting the case assumption).

We define a computable function $g$ such that

$$e\in \overline{H}\ \Longleftrightarrow\ g(e)\in I.$$

Given $e$, let $g(e)$ be an index of the following machine $M_e$: on input $x$,

1. simulate $\phi_e(e)$;
2. if the simulation halts, then **diverge** (loop forever);
3. if the simulation never halts, then compute and output $\beta(x)$.

Formally, the partial function computed by $M_e$ satisfies

$$
\phi_{g(e)} \simeq
\begin{cases}
\beta, & \text{if }\phi_e(e)\uparrow\ \ (e\in \overline{H}), \\
\phi_\uparrow, & \text{if }\phi_e(e)\downarrow\ \ (e\in H).
\end{cases}
$$

The map $e\mapsto g(e)$ is computable because from $e$ we can effectively write down a code for $M_e$ (using the usual indexing of Turing machines / the s-m-n idea).

Now check the reduction:
* If $e\in\overline{H}$, then $\phi_{g(e)}=\beta=\phi_{e_0}$. Since $e_0\in I$ and $I$ is an index set (membership depends only on the computed function), it follows that $g(e)\in I$.
* If $e\in H$, then $\phi_{g(e)}=\phi_\uparrow$. By the case assumption, no index of $\phi_\uparrow$ lies in $I$, hence $g(e)\notin I$.

Therefore $e\in\overline{H}\iff g(e)\in I$), so $\overline{H}\le_m I$.

**Case 2: $I$ contains an index of $\phi_\uparrow$**

Then $\overline{I}$ contains no index of $\phi_\uparrow$. Moreover, since $I$ is nontrivial, $\overline{I}$ is also nontrivial; and $\overline{I}$ is an index set because the defining property (“closed under functional equivalence”) is preserved under complement.

Applying Case 1 to $\overline{I}$, we obtain

$$\overline{H}\le_m \overline{I}.$$

By taking complements (equivalently: by composing with a computable Boolean negation of the membership test), this implies

$$H \le_m I.$$

In either case, $H \le_m I$ or $\overline{H}\le_m I$. $\square$

Since both $H$ and $\bar{H}$ are undecidable, **Rice's theorem directly implies that any nontrivial semantic property is undecidable.**

<div class="pmf-grid">
  <figure>
    <img src="{{ '/assets/images/notes/computability-and-complexity/rice_theorem_visualization.png' | relative_url }}" alt="a" loading="lazy">
    <figcaption>Sketch of the Rice's theorem proof</figcaption>
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/computability-and-complexity/rice_theorem_simple_mental_model.jpeg' | relative_url }}" alt="a" loading="lazy">
    <figcaption>Simple mental model of reduction function $g$, which maps functions that do not halt to $\beta$, and other functions to $\phi_\uparrow$</figcaption>
  </figure>
</div>

**About the right figure:**

**For the purposes of this reduction**, that’s a perfectly good mental note, with two small “precision tweaks”:
1. It’s not “all functions that do not halt,” it’s **all indices $e$ such that the specific computation $\varphi_e(e)$ does not halt** (i.e. $e \in \overline{H}$).
2. It’s also not “all functions in standard numbering,” but **all indices $e$ you feed into the reduction $g$**.

With that understood, the picture is exactly:
* If $\varphi_e(e)\uparrow$ (so $e\in\overline{H}$), then the constructed program $g(e)$ computes **the fixed function** $\beta=\varphi_{e_0}$.
* If $\varphi_e(e)\downarrow$ (so $e\in H$), then $g(e)$ computes **the fixed function** $\varphi_\uparrow$.

So **the image of $g$** (as a function on indices) is essentially just **two equivalence classes of programs**: those computing $\beta$ and those computing $\varphi_\uparrow$. That’s the whole “two-function” trick that makes the membership in $I$ encode the halting behavior.


<!-- $\textbf{Corollary 193:}$ Nontrivial index sets are not decidable. -->
<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(193)</span></p>

Nontrivial index sets are not decidable.

</div>

<!-- $\textbf{Corollary 194:}$ For every partial computable function $\alpha$ the set $\lbrace e : \alpha = \phi_e\rbrace$ is infinite. -->
<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(name of Corollary)</span></p>

For every partial computable function $\alpha$ the set $\lbrace e : \alpha = \phi_e\rbrace$ is infinite.

</div>

*Proof*: If this set were finite for some $\alpha$, it would be a decidable set. It is also nontrivial (it's not $\emptyset$ or $\mathbb{N}$) and is an index set by definition. This would contradict Corollary 193.

Not all properties related to Turing machine indices are index sets. For a property to be an index set, it must depend only on the function computed, not the index itself.

<!-- $\textbf{Example 196:}$ The set $A = \lbrace e : \text{dom}(\phi_e) \text{ has more than } e \text{ elements}\rbrace$ is not an index set. The set $A$ is r.e. Given $e$, we can effectively enumerate $\text{dom}(\phi_e)$ until at least $e + 1$ elements of $W_e$ have been enumerated. The latter enumeration works by a technique called dovetailing, which roughly amounts to simulate for $s = 0, 1, \dots$ the computation of $\phi_e(x)$ for all $x \le s$ and for $s$ computation steps, see the exercises for details.

* **Case I**: There exists $e \in A$ such that the domain of $\phi_e$ has finite size $m$. By Corollary 194, there exists an index $e' > m$ such that $\phi_e = \phi_{e'}$. Then $e' \notin A$, hence $A$ is not an index set.
* **Case II**: For all $e \in A$ the domain of $\phi_e$ is infinite. By definition, $A$ contains all indices $e$ such that $W_e$ is infinite, yielding a numbering of partial computable functions with infinite domains, contradicting Remark 195. For an effective enumeration $e_0, e_1, \dots$ of $A$, $\phi_{e_0}, \phi_{e_1}, \dots$ is such a numbering. -->
<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(name of Example)</span></p>

Let

$$A=\lbrace e : \operatorname{dom}(\phi_e)\ \text{has more than } e \text{ elements}\rbrace.$$

Then $A$ is **not** an index set.

Moreover, $A$ is r.e.: given $e$, we can effectively enumerate $\operatorname{dom}(\phi_e)$ until at least $e+1$ distinct elements have appeared. This can be done via **dovetailing**: for stages $s=0,1,2,\dots$, simulate the computations $\phi_e(x)$ for all $x\le s$, each for $s$ steps (see the exercises for details).

To show that $A$ is not an index set, distinguish two cases.

* **Case I.** There exists $e\in A$ such that $\operatorname{dom}(\phi_e)$ is finite of size $m$. By Corollary 194, there is an index $e'>m$ with $\phi_{e'}=\phi_e$. But then $e'\notin A$, so membership in $A$ is not preserved under equality of computed functions. Hence $A$ is not an index set.

* **Case II.** For every $e\in A$, the domain $\operatorname{dom}(\phi_e)$ is infinite. Then, by definition, $A$ contains every index $e$ such that $W_e$ is infinite. This would give an effective listing of partial computable functions with infinite domain, contradicting Remark 195: if $e_0,e_1,\dots$ is an effective enumeration of $A$, then $\phi_{e_0},\phi_{e_1},\dots$ would be such a numbering.
  
</div>

## Oracle Turing Machines and the Jump Operator

We now introduce a more powerful form of reducibility, Turing reducibility, based on a model of computation called an oracle Turing machine. This model allows us to ask "what if we had a magical black box that could solve problem $B$?" and then explore what other problems ($A$) we could solve with its help.

### Oracle Turing Machines and Turing Reducibility

<!-- $\textbf{Definition 197:}$ An oracle Turing machine is a Turing machine that, in addition to its working tapes, has a special tape on which the sequence of function values $B(0)B(1) \dots$ of the characteristic function of a set $B$ is continuously written. The set $B$, which can be considered as an additional input to the computation, is called the oracle, and the additional tape is called the oracle tape. The oracle tape is read-only, and initially, the read head is positioned on the cell containing the bit $B(0)$. The notation $M(x, B)$ is used for the result of the computation of the oracle Turing machine $M$ on inputs $x$ and oracle $B$, if the computation terminates, and $M(x, B) \uparrow$, if the computation does not terminate. The notations $M(x, B) \downarrow$ and $M(x, B) \downarrow = y$ are defined as for Turing machines. In all these notations, $M(x, B)$ can also be written as $M^B(x)$.
If an oracle Turing machine $M$ terminates for all inputs $x$ with a given oracle $B$, the notation $M(B)$ is used for the uniquely determined set $A$ with the characteristic function $c_A : x \mapsto M(x, B)$.
A set $A$ is Turing-reducible to a set $B$, briefly $A$ is T-reducible to $B$ or $A \le_T B$, if there is an oracle Turing machine $M$ such that $A = M(B)$, i.e.,  $A(x) = M(x, B) \text{ for all } x \in \mathbb{N}$.  -->
<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(name of Definition)</span></p>

An **oracle Turing machine** is a Turing machine that, in addition to its work tapes, has a special **oracle tape**. For a set $B$, the oracle tape contains the infinite binary sequence

$$B(0)B(1)B(2)\dots$$

(the values of the characteristic function of $B$). The set $B$ is called the **oracle**. The oracle tape is read-only, and initially the head scans the cell containing $B(0)$.

For an oracle machine $M$, input $x$, and oracle $B$, write:

* $M(x,B)$ for the output if the computation halts,
* $M(x,B)\uparrow$ if it does not halt,
* $M(x,B)\downarrow$ and $M(x,B)\downarrow = y$ with the usual meanings.

We also write $M^B(x)$ instead of $M(x,B)$.

If $M$ halts on every input $x$ when using oracle $B$, then $M(B)$ denotes the (unique) set $A$ whose characteristic function is $x \mapsto M(x,B)$.

A set $A$ is **Turing-reducible** to $B$, written $A \le_T B$, if there exists an oracle Turing machine $M$ such that $A=M(B)$, i.e.

$$A(x)=M(x,B)\quad\text{for all }x\in\mathbb{N}.$$

</div>

Turing reducibility is a more general notion than many-one reducibility.

<!-- $\textbf{Theorem 198:}$ For all sets $A$ and $B$, it holds that $A \le_m B$ implies $A \le_T B$ but the implication in the opposite direction is false in general. -->
<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(name of Theorem)</span></p>

For all sets $A,B$, if $A \le_m B$ then $A \le_T B$. In general, the converse implication fails.
</div>

*Proof*:

* The implication $A \le_m B \implies A \le_T B$ holds because if $f$ is the computable function for the m-reduction, an oracle Turing machine can compute $f(x)$ and then query the oracle for B at the single location $f(x)$ to get the answer.
* A counterexample for the reverse is given by the halting problem $H$ and its complement $\bar{H}$. For any set $A$, we have $A \le_T \bar{A}$, because an oracle for $\bar{A}$ can be used to decide membership in $A$ and vice versa. Thus, $H \le_T \bar{H}$. However, as shown in Corollary 188, we have $H \not\le_m \bar{H}$.

### Like m-reducibility, T-reducibility preserves decidability.

<!-- $\textbf{Proposition 199:}$ Let $A$ and $B$ be sets where $A \le_T B$. 
* **(i)** If $B$ is decidable, then $A$ is also decidable. 
* **(ii)** If $A$ is undecidable, then $B$ is also undecidable. -->
<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(name of Proposition)</span></p>

Let $A$ and $B$ be sets where $A \le_T B$. 
* **(i)** If $B$ is decidable, then $A$ is also decidable. 
* **(ii)** If $A$ is undecidable, then $B$ is also undecidable.
  
</div>

*Proof*: See the exercises.

### Relative Computations and the Jump Operator

The concept of an oracle allows us to "relativize" all the fundamental notions of computability. We can speak of sets being decidable or r.e. relative to an oracle.

<!-- $\textbf{Definition 200:}$ A set A is decidable with oracle $B$, if $A \le_T B$ holds. A set $A$ is recursively enumerable or r.e. with oracle $B$, if there exists an oracle Turing machine $M$ that has domain $A$ when its oracle is $B$, i.e., if  $A = \lbrace x \in \mathbb{N} : M(x, B) \downarrow\rbrace$.  -->
<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(name of Definition)</span></p>

Let $A,B \subseteq \mathbb{N}$.

* $A$ is **decidable relative to** (or **with oracle**) $B$ if $A \le_T B$.
* $A$ is **recursively enumerable relative to** $B$ if there exists an oracle Turing machine $M$ such that
  
  $$A=\lbrace x\in\mathbb{N} : M(x,B)\downarrow\rbrace.$$

</div>

If $A$ is decidable with oracle $B$, we also say $A$ is decidable in $B$ or decidable relative to $B$. The same terminology applies to r.e. sets. Most theorems from standard computability theory have direct analogues in this relativised world.

<!-- $\textbf{Definition 201:}$ $A$ function $f$ is computable with oracle $B$ if there is an oracle Turing machine that on input $i$ and oracle $B$ outputs $f(i)$. An enumeration $x_0, x_1, \dots$ is effective in $B$ if $x_i = f(i)$ for some function $f$ that is computable with oracle $B$. -->
<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(name of Definition)</span></p>

A function $f$ is **computable with oracle** $B$ if there exists an oracle Turing machine that, on input $i$ and oracle $B$, outputs $f(i)$.

An enumeration $x_0,x_1,\dots$ is **effective in** $B$ if $x_i=f(i)$ for some function $f$ computable with oracle $B$.

</div>

> **Remark 202**: Similar to the unrelativized case, it can be shown that a set $A$ is recursively enumerable with oracle $B$ if and only if there exists an enumeration $x_0, x_1, \dots$ that is effective in $B$.

<!-- $\textbf{Theorem 203:}$ Let A and B be sets. 
* **(i)** $A$ is decidable in $B$ if and only if $A$ and the complement $\bar{A}$ of $A$ are recursively enumerable in $B$. 
* **(ii)** $A$ is decidable in $B$ if and only if $A$ has a monotone enumeration $x_0 \le x_1 \le x_2$ \dots that is effective in $B$. -->
<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(name of theorem)</span></p>

Let $A,B$ be sets.

1. $A$ is decidable in $B$ $\iff$ both $A$ and $\overline{A}$ are r.e. in $B$.
2. $A$ is decidable in $B$ $\iff$ $A$ has a monotone enumeration $x_0 \le x_1 \le x_2 \le \dots$ that is effective in $B$.
</div>

Just as we can create a standard numbering of all Turing machines, we can create a standard numbering of all oracle Turing machines, which is also a Gödel numbering in the relativized sense. This allows us to define a relativized version of the halting problem.

<!-- $\textbf{Definition 205:}$ Let $B$ be a set and $M_0, M_1, \dots$ be the standard enumeration of all oracle Turing machines. The set  $H^B = \lbrace e : M_e(e, B) \downarrow \rbrace$  is called the (diagonal) halting problem relative to $B$, or the jump of $B$. The map $X \mapsto H^X$ is called the jump operator. -->
<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(name of definition)</span></p>

Let $B\subseteq \mathbb{N}$, and let $M_0,M_1,\dots$ be the standard enumeration of all oracle Turing machines. Define

$$H^B=\lbrace e : M_e(e,B)\downarrow\rbrace.$$

This set is called the **(diagonal) halting problem relative to $B$**, or the **jump** of $B$. The map $X \mapsto H^X$ is the **jump operator**.

</div>

The jump of $B$ is often denoted $B'$. The set $H^B$ plays the same role for sets that are r.e. in $B$ as the original halting problem $H$ does for r.e. sets.

<!-- $\textbf{Theorem 206:}$ For all sets $A$ and $B$, $A$ is recursively enumerable relative to the oracle $B$ if and only if $A$ is m-reducible to $H^B$. -->
<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(name of theorem)</span></p>

For all sets $A,B$, the set $A$ is r.e. relative to $B$ $\iff$ $A \le_m H^B$.

</div>

*Proof*: The proof is a direct relativization of the proof of Theorem 184 and Corollary 185.

* $(\impliedby)$ If $A \le_m H^B$ via computable function $f$, we can build an oracle TM for $A$. On input $x$ with oracle $B$, it computes $f(x)$ and then simulates the oracle TM $M_{f(x)}$ on input $f(x)$ with oracle $B$. This machine halts iff $f(x) \in H^B$, which happens iff $x \in A$. Thus, $A$ is r.e. in $B$.
* $(\implies)$ If $A$ is r.e. in $B$, there is an oracle TM $M$ such that $A$ is its domain with oracle $B$. We can create a computable function $h$ such that for any $x, M_{h(x)}$ is an oracle TM that on any input $y$ with any oracle $X$ simulates $M$ on input $x$ with oracle $X$. The function $h$ is computable as it just hard-wires $x$ into the description of $M$. Then we have:  $x \in A \iff M(x, B) \downarrow \iff M_{h(x)}(h(x), B) \downarrow \iff h(x) \in H^B$.  This shows $A \le_m H^B$ via the computable function $h$.

<!-- $\textbf{Corollary 207:}$ For all sets $B$, the halting problem relative to $B$ is recursively enumerable with oracle $B$. -->
<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(name of theorem)</span></p>

For every set $B$, the halting problem $H^B$ is r.e. relative to $B$.
</div>

The jump operator always produces a strictly more complex set. A set is always reducible to its own jump, but the jump is never reducible back to the original set.

<!-- $\textbf{Theorem 208:}$ Let $B$ be a set. Then $B$ is m-reducible to $H^B$ and thus decidable with oracle $H^B$, but $H^B$ is not decidable with oracle $B$, i.e., it holds  $B \le_m H^B \quad \text{and} \quad H^B \not\le_T B$.  -->
<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(name of theorem)</span></p>

Let $B$ be a set. Then $B \le_m H^B$, so $B$ is decidable relative to $H^B$. However, $H^B$ is not decidable relative to $B$; that is,

$$
B \le_m H^B
\quad\text{and}\quad
H^B \not\le_T B.
$$

</div>
*Proof*:

* To show $B \le_m H^B$, we need a computable function $h$. Let $h(x)$ be the index of an oracle TM $M_{h(x)}$ that, on any input $y$ and with any oracle $X$, halts if and only if $x \in X$. Then $x \in B \iff M_{h(x)}(h(x), B) \downarrow \iff h(x) \in H^B$. Thus $B \le_m H^B$.
* The proof that $H^B \not\le_T B$ is a relativized diagonalization argument, identical to the proof that the original halting problem is undecidable. If $H^B$ were decidable in $B$, then the function $g(e) = 1 + M_e(e, B)$ (if $e \in H^B$) and $0$ (otherwise) would be computable in $B$. But this function cannot be computed by any oracle TM $M_e$ with oracle $B$, as it differs from the function computed by $M_e$ on input $e$. This is a contradiction.

Finally, the jump operator is monotone with respect to Turing reducibility. If $A$ is no harder than $B$, then the jump of $A$ is no harder than the jump of $B$.

<!-- $\textbf{Theorem 209:}$ If a set $A$ is $T$-reducible to a set $B$, then the halting problem relative to $A$ is m-reducible to the halting problem relative to B, i.e., it holds  $A \le_T B \text{ implies } H^A \le_m H^B$.  -->
<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(name of theorem)</span></p>

If $A \le_T B$, then the halting problem relative to $A$ is many-one reducible to the halting problem relative to $B$; equivalently,

$$A \le_T B \ \Longrightarrow\ H^A \le_m H^B.$$

</div>

*Proof*: Let $A \le_T B$ via an oracle TM $M_{oracle}$. We want to show $H^A \le_m H^B$. We need a computable function $h$ such that $e \in H^A \iff h(e) \in H^B$. The function $h(e)$ produces the index of a new oracle TM $M_{h(e)}$. This machine, on input $y$ with oracle $X$, simulates the oracle TM $M_e$ on input $e$. Whenever $M_e$ makes an oracle query for some string $z$ (to what it thinks is oracle $A$), $M_{h(e)}$ pauses and uses its own oracle $X$ to simulate $M_{oracle}$ on input $z$ to get the answer. It then provides this answer back to the simulation of $M_e$. So,$ M_{h(e)}$ with oracle $X$ simulates $M_e$ with oracle $M_{oracle}(X)$. This means $M_e(e, A) \downarrow \iff M_e(e, M_{oracle}(B)) \downarrow \iff M_{h(e)}(h(e), B) \downarrow$. Therefore, we have the desired equivalence:  $e \in H^A \iff M_e(e, A) \downarrow \iff M_{h(e)}(h(e), B) \downarrow \iff h(e) \in H^B$.  The function $h$ is computable, so this establishes $H^A \le_m H^B$.
