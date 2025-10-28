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
## Computability and Complexity

### Formalizing Computability

In theoretical computer science, the abstract concept of computability is given a precise mathematical foundation through the study of partial computable functions. From this core idea, related concepts such as computable functions and decidable sets are derived. While several equivalent models exist for formalizing computability—including register machines, µ-recursive functions, and the lambda calculus—this study will focus exclusively on the model provided by Turing machines. All these formalisms are provably identical, meaning they all define the same class of computable functions and decidable sets.

### The Turing Machine: A Formal Definition

The Turing machine serves as a foundational abstract model of a computing device. It consists of a finite set of states and one or more tapes that serve as its memory.

**Turing Machine (TM).** Let $k$ be a nonzero natural number. A Turing machine with $k$ tapes, also called a $k$-tape Turing machine or $k$-tape TM, is a $6$-tuple of the form $M = (Q, \Sigma, \Gamma, \Delta, s, F)$ where

- $Q$ is the finite set of states,
- $\Sigma$ is the input alphabet,
- $\Gamma$ is the tape or working alphabet, where $\Sigma \subseteq \Gamma$ and the blank symbol $\square \in \Gamma \setminus \Sigma$,
- $\Delta$ is the transition relation of $M$, where for $\text{Mov} = \lbrace , R, S \rbrace$, the relation $\Delta$ is a subset of $Q \times \Gamma^k \times Q \times \Gamma^k \times \text{Mov}^k$,
- $s \in Q$ is the initial state,
- $F \subseteq Q$ is the set of accepting states.

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

The "program" of a Turing machine is its transition relation $\Delta$, which is a set of tuples called instructions. For a k-tape TM, an instruction has the form: $(q, a_1, \dots, a_k, q', a'_1, \dots, a'_k, Z_1, \dots, Z_k) \in Q \times \Gamma^k \times Q \times$ \Gamma^k \times \text{Mov}^k$ 

#### The intuitive meaning of such an instruction is as follows:

* Condition: The instruction can be executed if the machine is currently in state$ q$ and the symbols being read by the heads on tapes $1$ through $k$ are $a_1$ through $a_k$, respectively.
* Action: When executed, the machine transitions to state $q'$. Simultaneously, for each tape $i$ (from $1$ to $k$), the symbol $a_i$ under the head is overwritten with $a'_i$, and the head then performs the movement $Z_i \in \lbrace L, R, S \rbrace$.

To formalize this, we distinguish between the two parts of an instruction.

**Definition 2:** For an instruction of the form $(q, a_1, \dots, a_k, q', a'_1, \dots, a'_k, Z_1, \dots, Z_k)$, we refer to $(q, a_1, \dots, a_k)$ as its condition part, and to $(q', a'_1, \dots, a'_k, Z_1, \dots, Z_k)$ as its instruction part.

**Remark 3:** The transition relation $\Delta$ of a Turing machine can also be understood as a relation between condition parts and instruction parts. For a $k$-tape Turing machine, $\Delta$ is thus a binary relation of the form  $\Delta \subseteq (Q \times \Gamma^k) \times (Q \times \Gamma^k \times \text{Mov}^k)$  which can be written in infix notation as  $(q, a_1, \dots, a_k) \Delta (q', a'_1, \dots, a'_k, Z_1, \dots, Z_k)$. The relation $\Delta$ is called right-unique if for each condition part $(q, a_1, \dots, a_k)$ there is at most one instruction part $(q', a'_1, \dots, a'_k, Z_1, \dots, Z_k)$ that satisfies the relation.

### Deterministic vs. Nondeterministic Machines

The nature of the transition relation $\Delta$ determines whether a Turing machine is deterministic or nondeterministic.

A general Turing machine, as defined above, does not require $\Delta$ to be right-unique. This means that for a given state and set of tape symbols, there might be multiple applicable instructions. Such a machine is called a nondeterministic Turing machine. When a computation reaches a point with multiple choices, it can branch, exploring several computational paths simultaneously.

**Remark 5:** In the literature, the term nondeterministic Turing machine is used in two different senses. It may refer either to a general Turing machine, i.e., one that may or may not be deterministic, or to a Turing machine that is indeed not deterministic, i.e., has a transition relation that is not right-unique.

In contrast, a deterministic Turing machine has a transition relation that is right-unique. At any point in its computation, there is at most one applicable instruction. This ensures that for any given input, the machine follows a single, uniquely determined computational path.

**Definition 4:** Deterministic Turing Machine A Turing machine $M$ is deterministic if its transition relation is right-unique. A $k$-tape Turing machine $(Q, \Sigma, \Gamma, \Delta, s, F)$ is deterministic if and only if for every state $q \in Q$ and every sequence of $k$ symbols $a_1, \dots, a_k from \Gamma$, there is at most one instruction in $\Delta$ with condition part $(q, a_1, \dots, a_k)$.

### The Mechanics of Computation

To formally analyze Turing machine computations, we need precise definitions for the machine's state at any given moment (its configuration) and the transitions between these states.

#### Configurations: Snapshots of a Computation

A configuration is a complete snapshot of a Turing machine's status, capturing its current state, all tape contents, and the positions of all heads. Since a tape is infinite but can only contain a finite number of non-blank symbols, we need a way to represent its contents.

**Definition 6:** Tape Inscriptions Let $M$ be a Turing machine with tape alphabet $\Gamma$. A tape inscription of $M$ is a function $f: $\mathbb{Z} \to \Gamma$ such that $f(i) = \square$ for all but finitely many integers $i$. The relevant part u_f of a tape inscription f is a word over $\Gamma$. We let $u_f = \square$ if $f$ is the constant function with value $\square$. Otherwise, we let  $u_f = f(\min I_f) \cdots f(\max I_f) \quad \text{where} \quad I_f = \lbrace i \in \mathbb{Z} : f(i) \neq \square \rbrace$  A tape inscription $f$ is represented by any word $u$ of the form $u = \square^{r_1} u_f \square^{r_2}$ with $r_1, r_2 \in \mathbb{N}$.

Using this representation for tape contents, we can formally define a configuration.

**Definition 7:** Configuration Let $k$ be a nonzero natural number and let $M = (Q, \Sigma, \Gamma, \Delta, s, F)$ be a $k$-tape Turing machine. A configuration of $M$ is a tuple  $(q, u_1, \dots, u_k, j_1, \dots, j_k) \in Q \times (\Gamma^+)^k \times \prod_{i=1,\dots,k} \lbrace 1, \dots, \left|u_i\right| \rbrace$  This tuple represents a situation where $q$ is the current state, and for each tape $i=1, \dots, k$:

* the word $u_i$ represents the relevant part of the inscription on tape $i$,
* the number $j_i$ indicates the head position on tape $i$, corresponding to the $j_i$-th symbol of $u_i$.

#### c

A computation unfolds as a sequence of configurations, where each transition from one configuration to the next is governed by an instruction from $\Delta$. This transition is called a computation step.

**Definition 8: Computation Step Let C = (q, u_1, $\dots$, u_k, j_1, $\dots$, j_k)** be a configuration of a k-tape Turing machine M. An instruction of M of the form (q, a_1, $\dots$, a_k, q', a'_1, $\dots$, a'_k, Z_1, $\dots$, Z_k) is applicable to C if for i=1, $\dots$, k, the symbol at position j_i of u_i is equal to a_i. A configuration C' = (q', u'_1, $\dots$, u'_k, j'_1, $\dots$, j'_k) is a successor configuration of C, written C $\xrightarrow{M}$ C', if there is an applicable instruction as above such that for i=1, $\dots$, k:

* u'_i = $\begin{cases}$ $\square$ $\tilde{u}_i$ & $\text{if }$ Z_i = L $\text{ and }$ j_i = 1, \\ $\tilde{u}_i$ $\square$ & $\text{if }$ Z_i = R $\text{ and }$ j_i = |u_i|, \\ $\tilde{u}_i$ & $\text{otherwise}$, $\end{cases}$
* j'_i = $\begin{cases}$ 1 & $\text{if }$ Z_i = L $\text{ and }$ j_i = 1, \\ j_i - 1 & $\text{if }$ Z_i = L $\text{ and }$ j_i $\neq$ 1, \\ j_i & $\text{if }$ Z_i = S, \\ j_i + 1 & $\text{if }$ Z_i = R, $\end{cases}$ where $\tilde{u}_i$ is the word obtained from u_i by replacing the symbol a_i at position j_i with a'_i.

The relation $\xrightarrow{M}$ is known as the successor relation or 1-step relation of M. A sequence of such steps forms a partial computation.

**Definition 9:** Partial Computations Let M be a Turing machine. A finite partial computation of M of length t is a finite sequence of configurations C_0, $\dots$, C_t such that C_i $\xrightarrow{M}$ C_{i+1} for all i = 0, $\dots$, t-1. An infinite partial computation is an infinite sequence C_0, C_1, $\dots$ where this condition holds for all i $\ge$ 0. We write C $\xrightarrow$[M]{t} C' if there is a finite partial computation of length t from C to C', and C $\xrightarrow$[M]{*} C' if such a computation exists for some length t.

### The Full Computation Process

A full computation is a special type of partial computation that begins in a standardized initial configuration and proceeds until it can no longer continue.

**Definition 10:** Initial and Halting Configuration, Computation Let M = (Q, $\Sigma$, $\Gamma$, $\Delta$, s, F) be a Turing machine and let w be a word over $\Sigma$.

* The initial configuration of M on input w is (s, w$\square$, $\square$, $\dots$, $\square$, 1, $\dots$, 1).
* A halting configuration is a configuration (q, u_1, $\dots$, u_k, j_1, $\dots$, j_k) to which no instruction in $\Delta$ is applicable, meaning (q, u_1(j_1), $\dots$, u_k(j_k)) is not the condition part of any instruction in $\Delta$.
* An accepting configuration is a halting configuration whose state is an accepting state (i.e., is in F).
* A computation of M is a partial computation that starts with an initial configuration and is either infinite or ends in a halting configuration.
* A terminating computation is a finite computation.
* An accepting computation is a terminating computation that ends in an accepting configuration.

### The Computation Tree

For a nondeterministic machine, the set of all possible computations on a given input w can be visualized as a tree.

**Definition 11:** Computation Tree The computation tree of a Turing machine M on input w is a finite or infinite rooted tree labelled with configurations of M such that:

* The root is labelled with the initial configuration of M on input w.
* The children of an internal node labelled with configuration C are labelled with the successor configurations of C. If there are t > 0 applicable instructions, the node has t children, one for each resulting successor configuration.
* Leaf nodes are labelled exclusively with halting configurations.

A branch of the computation tree (a path starting at the root) corresponds to a single computation of M on input w. Finite branches correspond to terminating computations, while infinite branches correspond to non-terminating computations. Since a TM has a finite number of instructions, its computation tree is always finitely branching. For a deterministic TM, the computation tree is simply a single path, as there is only one possible successor at each step.

### Recognizing and Deciding Languages

Turing machines can be used to define classes of languages based on their computational properties. The most fundamental concepts are acceptance and decision.

#### Acceptance and Recognized Languages

A Turing machine accepts an input word if at least one of its possible computational paths leads to an accepting state.

**Definition 12:** Acceptance and Recognized Language Let M be a Turing machine with input alphabet $\Sigma$, and let w be a word over $\Sigma$. The Turing machine M accepts the word w if one of its computations on input w is accepting. The language recognized by M is  L(M) = \{ w $\in$ $\Sigma$^* : M $\text{ accepts }$ w \}

For a word w to be in L(M), the computation tree of M on w must have at least one accepting branch. It is possible for other branches to be infinite (non-terminating). This class of languages is known in computability theory as the recursively enumerable languages.

#### Decidable Languages and Total Turing Machines

A stronger condition is that a machine must halt on all inputs, whether it accepts them or not. This leads to the notion of a decidable language.

**Definition 13:** Total Turing Machine A Turing machine is total if for all inputs all of its computations terminate.

**Definition 14:** Decidable Language A language is decidable if it is recognized by a total Turing machine. In this case, we also say that the language is decided by the Turing machine.

It will later be shown that the class of decidable languages is a strict subclass of the recursively enumerable languages.

### The Power of Nondeterminism

A central question in complexity theory is whether nondeterministic machines are more powerful than deterministic ones (the famous P vs. NP problem). In the context of computability without resource bounds, the answer is no: every language that can be decided by a nondeterministic TM can also be decided by a deterministic one.

The proof of this relies on the insight that for a total nondeterministic TM, the computation tree for any input must be finite. This can be shown using König's Lemma.

**Theorem 15:** König’s Lemma A finitely branching rooted tree is infinite if and only if it has an infinite branch.

**Proof sketch.**: A tree with an infinite branch is clearly infinite. For the converse, let T be an infinite, finitely branching rooted tree. We can inductively construct an infinite branch v_0, v_1, $\dots$. The key is to maintain the invariant that the subtree rooted at v_i is infinite.

* Let v_0 be the root of T. The subtree at v_0 is T itself, which is infinite.
* Assuming v_i has been defined such that its subtree is infinite, we choose v_{i+1} from its children. Since v_i has only finitely many children, and the subtree at v_i is infinite, at least one of its children must be the root of an infinite subtree. We choose such a child to be v_{i+1}. This process can be continued indefinitely, constructing an infinite branch.

Using this lemma, we can prove that nondeterminism adds no power for deciding languages.

**Theorem 16:** Every decidable language is recognized by a total deterministic Turing machine.

**Proof sketch.**: Let L be a decidable language. By definition, there is a total Turing machine M that recognizes L.

1. Since M is total, all of its computations terminate. This means that for any input w, all branches in the computation tree of M on w are finite.
2. The computation tree of any TM is finitely branching.
3. By König's Lemma, a finitely branching tree with only finite branches must be a finite tree. Therefore, the computation tree of M on any input w is finite.
4. We can construct a deterministic Turing machine M' that, on input w, systematically explores the entire computation tree of M on w (e.g., via breadth-first or depth-first search).
5. Since the tree is finite, this search is guaranteed to terminate. Thus, M' is a total machine.
6. M' is designed to accept if and only if its search finds an accepting configuration of M. Therefore, L(M') = L(M) = L. This shows that L is recognized by a total deterministic Turing machine.

### Computable Functions

Beyond recognizing languages, Turing machines can also compute functions by transforming an input word into an output word.

#### From Language Recognition to Function Computation

To ensure a unique output for every input, function computation is defined using only total deterministic Turing machines. The output is defined based on the content of a designated tape when the machine halts.

**Definition 17:** Computable Functions For a configuration C = (q, u_1, $\dots$, u_k, j_1, $\dots$, j_k) of a k-tape TM, let $\text{out}$(C) be the longest word on tape k that starts at the current head position and extends to the right without containing any blank symbols. Formally, $\text{out}$(C) = u_k(j_k)u_k(j_k+1) $\cdots$ u_k(t) where t = $\max$\{i $\le$ |u_k| : u_k(j_k), $\dots$, u_k(i) $\text{ all differ from }$ $\square$\}. If u_k(j_k) is the blank symbol, $\text{out}$(C) is the empty word.

The function f_M computed by a total deterministic Turing machine M is defined by f_M(w) = $\text{out}$(C) for the halting configuration C reached by M on input w.

For alphabets $\Sigma$ and $\Sigma$', a function f: $\Sigma$^* $\to$ $\Sigma$'^* is computable if it is computed by some total deterministic Turing machine.

### Extending Computability to Other Domains

The concepts of decidable sets and computable functions, defined for words, can be extended to other domains like the natural numbers through the use of representations.

**Definition 18:** Decidable sets and computable functions on other domains A representation of a set A is an injective function $\text{repr}$: A $\to$ $\Sigma$^* for some alphabet $\Sigma$ such that the set of representations \{$\text{repr}$(x) : x $\in$ A\} is decidable. With respect to such a representation:

* a subset X of A is decidable if the set \{$\text{repr}$(a) : a $\in$ X\} is decidable.
* a function f: A $\to$ A is computable if there is a computable function f_{$\text{repr}$}: $\Sigma$^* $\to$ $\Sigma$^* that maps the representation of any x $\in$ A to the representation of f(x). That is, for all x $\in$ A, we have f_{$\text{repr}$}($\text{repr}$(x)) = $\text{repr}$(f(x)).

For example, the set of prime numbers is decidable because we can represent the natural number n by the unary word 1^n, and the language \{1^n : n $\text{ is prime}$\} is decidable.

**Remark 19:** Representations of natural numbers by binary words A natural number n can be represented by its binary expansion $\text{bin}$(n), such as $\text{bin}$(5) = 101. A bijective representation (one-to-one and onto) can be useful. One such representation maps the natural number n to the word z_n, where z_0, z_1, z_2, $\dots$ is the sequence of all binary words sorted in length-lexicographic order ($\lambda$, 0, 1, 00, 01, $\dots$). This establishes an order isomorphism between ($\mathbb{N}$, $\le$) and ($\{0,1\}$^*, $\le$_{$\text{llex}$}).

**Remark 20:** Sets, languages, and problems The terminology used often depends on the context of study:

* In computability theory, where resource bounds are not a concern, it is common to use unary representations (e.g., n is represented by 1^n). Turing machines are viewed as recognizing subsets of natural numbers, leading to terms like "decidable sets of natural numbers."
* In complexity theory, where computations are resource-bounded, inputs are typically binary words representing instances of a computational problem. Here, Turing machines are seen as recognizing languages or problems, both of which refer to arbitrary sets of binary words. (Note that in a general context, a "language" is any set of words over some alphabet).


## Computability and Complexity

## Time Complexity

### Deterministic Time

The analysis of algorithms and computational problems often centers on the resources they consume. One of the most critical resources is time. In the context of Turing machines, we formalize this by measuring the number of steps a machine takes to complete its computation.

**Definition 21:** Running time The running time of a deterministic Turing machine M on input w is defined as:  $\text{time}_M$(w) = $\begin{cases}$ t & $\text{the computation of }$ M $\text{ on input }$ w $\text{ terminates and has length }$ t, \ $\uparrow$ & $\text{otherwise}$, $\end{cases}$  where the symbol $\uparrow$ denotes that the function value is undefined (i.e., the machine does not halt).

This definition allows us to quantify the performance of a specific machine on a specific input. To create broader complexity classes, we generalize this notion to a function of the input length, defining what it means for a machine to be bounded by a certain time complexity function.

**Definition 22:** Time-bounded deterministic Turing machine A time bound is a computable function t : $\mathbb{N}$ $\to$ $\mathbb{N}$ with t(n) $\ge$ n for all n. For a time bound t, a deterministic Turing machine M is t(n)-time-bounded or runs in time t(n) if M is total (halts on all inputs) and for almost all inputs w, it holds that:  $\text{time}_M$(w) $\le$ t(|w|).

The phrase "for almost all inputs" means the condition must hold for all but a finite number of inputs. This provides flexibility, allowing us to disregard a small number of exceptional cases, typically short inputs, which can be handled separately.

**Remark 23 For a t(n)** -time-bounded Turing machine, the time bound must be obeyed for almost all inputs, i.e., for all words over the input alphabet except for at most finitely many, say, for all inputs of length larger than or equal to some constant b. Note that such a Turing machine can be transformed into another Turing machine such that both Turing machines recognize the same language L, and the new machine runs in time at most t(n) + 2b on all inputs. For the proof, call inputs of length at most b-1 small and call all other inputs long. It suffices to change the given Turing machine such that initially it scans the first b symbols of its input and stores them in its state such that on every small input a halting configuration is reached and this configuration is accepting if only if the input is in L. On a large input, the new Turing machine goes back to the first symbol of the input and then proceeds as usual. Treating small inputs this way is referred to as table lookup or hard-wiring.

By default, time complexity is a measure of the machine's performance in the most demanding scenario for a given input length. This is known as worst-case time complexity. An alternative approach, average-case time complexity, considers the average running time over all inputs of a certain length. While potentially relevant for specific practical applications, the theory of average-case complexity is more intricate and less developed. Therefore, our focus will remain on worst-case complexity.

Convention 24 In the part about complexity theory, all languages are languages over the binary alphabet unless explicitly something different is stated. Accordingly, we will restrict attention to Turing machines with binary input alphabet.

### Deterministic Time Complexity Classes

Using the concept of time-bounded Turing machines, we can group languages and functions into complexity classes based on the resources required to decide or compute them.

**Definition 25:** Deterministic time classes Let t be a time bound.

The class of languages decidable in deterministic time t(n) is:  $\text{DTIME}$(t(n)) = {L $\subseteq$ {0, 1}^* : L = L(M) $\text{ for some deterministic }$ t(n)$\text{-time-bounded Turing machine }$ M}.  The class of functions computable in time t(n) is:  $\text{FTIME}$(t(n)) = {f : {0, 1}^* $\to$ {0, 1}^* : f = f_M $\text{ for some deterministic }$ t(n)$\text{-time-bounded Turing machine }$ M}.  For a set F of time bounds, we define:  $\text{DTIME}$(F) = $\bigcup_{t(n) \in F}$ $\text{DTIME}$(t(n)) $\quad$ $\text{and}$ $\quad$ $\text{FTIME}$(F) = $\bigcup_{t(n) \in F}$ $\text{FTIME}$(t(n)). 

The notation $\text{DTIME}_k$(t(n)) is used to specify the class of languages decidable by a t(n)-time-bounded k-tape Turing machine. These definitions can also be extended from languages over strings to subsets of natural numbers by using a standard representation, such as binary encoding.

**Definition 26** Let t be a time bound. A subset A of the natural numbers is decidable in time t(n) if there is a t(n)-time-bounded Turing machine that decides the set \{$\text{bin}$(n) : n $\in$ A\}. A function f : $\mathbb{N}$ $\to$ $\mathbb{N}$ is computable in time t(n) if there is a t(n)-time-bounded Turing machine that computes a function that maps $\text{bin}$(n) to $\text{bin}$(f(n)).

This framework allows us to define some of the most fundamental and widely studied complexity classes.

**Definition 27: Some deterministic time classes Using the function classes:**

* $\text{lin}$ = \{n $\mapsto$ c $\cdot$ n + c : c $\in$ $\mathbb{N}$ $\setminus$ $\{0\}$\}
* $\text{poly}$ = \{n $\mapsto$ n^c + c : c $\in$ $\mathbb{N}$ $\setminus$ $\{0\}$\}
* 2^{$\text{lin}$} = \{n $\mapsto$ 2^{c $\cdot$ n + c} : c $\in$ $\mathbb{N}$ $\setminus$ $\{0\}$\}
* 2^{$\text{poly}$} = \{n $\mapsto$ 2^{n^c + c} : c $\in$ $\mathbb{N}$ $\setminus$ $\{0\}$\}

we define the complexity classes:

* $\text{LIN}$ = $\text{DTIME}$($\text{lin}$)
* $\text{P}$ = $\text{DTIME}$($\text{poly}$)
* $\text{E}$ = $\text{DTIME}$(2^{$\text{lin}$})
* $\text{EXP}$ = $\text{DTIME}$(2^{$\text{poly}$})

We refer, for instance, to P as the class of problems decidable in deterministic polynomial time and E as the class of problems decidable in deterministic linear exponential time.

### Properties of Deterministic Time Classes

The definitions of complexity classes depend on a specific model of computation (the Turing machine) and its parameters (number of tapes, alphabet size). The following theorems demonstrate the robustness of these classes, showing that they are invariant under certain changes to the machine model.

### The Linear Speedup Theorem

The first theorem shows that constant factors in the running time do not change the fundamental complexity of a problem. We can always build a faster machine that solves the same problem, effectively making any constant-factor speedup possible.

**Theorem 28:** Linear speedup Let t be a time bound, and let $\alpha$ > 0 be a real number, and let k $\ge$ 2. Then it holds that  $\text{DTIME}_k$(t(n)) $\subseteq$ $\text{DTIME}_k$($\alpha$ $\cdot$ t(n) + n) $\quad$ (2.1)  and hence, in particular,  $\text{DTIME}$(t(n)) $\subseteq$ $\text{DTIME}$($\alpha$ $\cdot$ t(n) + n) $\quad$ (2.2)  Furthermore, every function f in $\text{FTIME}$(t(n)) is in  $\text{FTIME}$($\alpha$ $\cdot$ t(n) + n + |f(n)|). $\quad$ (2.3)

**Proof.**: We will demonstrate the inclusion (2.1), from which (2.2) immediately follows. Given a language L $\in$ $\text{DTIME}$(t(n)), there exists a t(n)-time-bounded Turing machine with some number of tapes, say k $\ge$ 2, that recognizes L. By showing L $\in$ $\text{DTIME}_k$($\alpha$ $\cdot$ t(n) + n), it follows that L $\in$ $\text{DTIME}$($\alpha$ $\cdot$ t(n) + n).

To prove (2.1), let L be a language recognized by a t(n)-time-bounded k-tape Turing machine M, where k $\ge$ 2. We construct a new k-tape machine M' that simulates M but runs faster. The core idea is to have M' process larger chunks of data at each step.

Each symbol in the tape alphabet of M' will represent a block of d symbols from the tape alphabet of M, where d is a constant we will choose later.

#### The computation of M' proceeds in two phases:

1. Initialization Phase:
  * M' translates its input from M's alphabet $\Sigma$ to its own compressed alphabet $\Sigma$'.
  * It writes this new, compressed input onto its second tape.
  * Simultaneously, it overwrites the original input on the first tape with blank symbols.
  * Finally, it positions the head of the second tape at the beginning of the compressed input.
  * This phase takes n + $\lceil$ n/d $\rceil$ + 2 steps. After this, M' will simulate M, treating its second tape as the primary input tape.
2. Simulation Phase:
  * M' simulates d steps of M using just 7 of its own steps. To do this, M' uses its finite control (its state) to store key information about M's current configuration: M's current state, and for each tape, the head position of M within the larger macro-symbol that M' is currently scanning.
  * In each simulation cycle, M' needs to know the contents of the cells M's heads are on, as well as the adjacent cells, to determine M's next d moves. A "relevant" cell for M' is one that is currently scanned or is immediately to its left or right.
  * M' first reads all symbols on its relevant cells by moving one step left and two steps right (3 steps total), storing this information in its state.
  * With the complete local information (M's state, head positions, and relevant cell contents), M' can determine the outcome of the next d steps of M. Since d is a constant, all possible outcomes can be pre-computed and stored in M''s transition function (a finite table lookup). This information includes M's new state, the new content of the relevant cells, and the new head positions.
  * M' then uses another 4 steps to update its tape cells and move its heads to the new positions corresponding to M's configuration after d steps.

Timing Analysis: The initialization takes n + $\lceil$ n/d $\rceil$ + 2 steps. The simulation of M takes at most t(n) steps. Since M' simulates d steps of M in 7 steps, this phase requires at most 7 $\lceil$ t(n)/d $\rceil$ + 7 steps.

The total time for M' is at most n + $\frac{n}${d} + 2 + $\frac{7 t(n)}${d} + 7. Since the time bound t(n) $\ge$ n, for almost all n, this is bounded by:  $\frac{8 t(n)}${d} + n + 9 $\le$ $\frac{9 t(n)}${d} + n  We want this to be less than $\alpha$ t(n) + n. We can achieve this by choosing a large enough constant d such that $\frac{9}${d} < $\alpha$. With such a d, the running time of M' is bounded by $\alpha$ t(n) + n, proving that L $\in$ $\text{DTIME}_k$($\alpha$ $\cdot$ t(n) + n).

#### Alphabet Reduction

The size of the tape alphabet is another parameter of the Turing machine model. The following theorem shows that any computation can be performed by a Turing machine with a minimal binary alphabet, at the cost of only a constant factor slowdown.

**Theorem 29:** Alphabet reduction Let t be a time bound, let k $\ge$ 2, and let L be a language in $\text{DTIME}_k$(t(n)). There exists a deterministic k-tape Turing machine M with tape alphabet $\{0, 1, $\square$\}$ that recognizes L and runs in time d $\cdot$ t(n) for some constant d.

**Proof.**: Let L be recognized by a deterministic t(n)-time-bounded k-tape machine M with a tape alphabet $\{a_1, $\dots$, a_r\}$ of size r. We construct a machine M' with a binary tape alphabet that simulates M.

The core idea is to encode each symbol a_j from M's alphabet as a unique binary string. We can use the encoding where a_j is represented by the string 1^j0^{r-j}.

1. Initialization Phase: On input w, M' first translates w into its binary-encoded form. Using a second tape, this can be done in 2r|w| steps.
2. Simulation Phase: M' simulates the computation of M step-by-step. To simulate a single step of M, M' must:
  * Read: Identify the symbol under each of M's heads. This requires M' to read the corresponding block of r binary symbols on each of its tapes. While reading a block, M' uses its state to remember its position within the block and the binary pattern it has seen so far.
  * Write/Move: Based on M's transition function, M' overwrites the binary blocks with the new encoded symbols and moves its heads accordingly. This involves moving across the r cells of the block.

Simulating one step of M requires reading and potentially writing k blocks of r symbols each. This takes a constant number of steps proportional to r. Thus, a single step of M can be simulated in d $\cdot$ r steps of M' for some constant d. The total running time of M' is therefore bounded by a constant multiple of t(n), as required.

#### Tape Reduction

A more significant change to the machine model is reducing the number of tapes. A multi-tape Turing machine can be simulated by a single-tape machine, but this incurs a more substantial, quadratic, increase in computation time.

**Theorem 30:** Tape reduction Let t be a time bound and let L be a language in $\text{DTIME}_k$(t(n)), i.e., L is recognized by a deterministic total k-tape Turing machine M that runs in time t(n). Then there exists a deterministic 1-tape Turing machine M' that recognizes L and runs in time d $\cdot$ t^2(n) for some constant d.

**Proof.**: We construct a single-tape Turing machine M' that simulates a k-tape machine M. The single tape of M' is structured to represent all k tapes of M simultaneously using a system of "tracks".

Conceptually, the tape of M' is partitioned into 2k tracks. Each cell on the tape of M' contains a 2k-tuple. For each of M's tapes (say, tape i), two tracks on M' are used: one to store the content of tape i and another to mark the position of tape i's head. The tape alphabet $\Gamma$' of M' consists of tuples containing k symbols from M's alphabet $\Gamma$ and k symbols from $\{$\square$, *\}$, where * is used as the head marker.

#### The simulation proceeds as follows:

1. Initialization: On input w, M' first initializes its tape to represent the initial configuration of M. This involves writing w onto the first track and placing head markers (*) at the beginning of all head-position tracks.
2. Simulation of a Single Step: To simulate one step of M, M' performs a subroutine:
  * (i) Scan and Read: M' sweeps its single head across the entire active portion of its tape to find the k head markers (*). As it passes each marker, it notes the corresponding tape symbol from the content track and stores all k symbols in its state.
  * (ii) Update and Write: After collecting all necessary information, M' knows what transition M will make. It then performs a second sweep across the tape to update the tape contents and move the head markers (*) one position left or right, as dictated by M's transition function. It also updates the state of M, which is stored in its own finite control.

Timing Analysis: The machine M runs for at most t(n) steps. This means that on any of its tapes, it can access at most t(n) cells to the left and t(n) cells to the right of the starting position. Therefore, the active portion of M''s tape has a length of at most 2t(n)+1.

Each simulation of a single step of M requires M' to make two full passes over this active portion. The time required for this is proportional to the length of the active tape, which is O(t(n)). So, simulating one step of M takes at most d $\cdot$ t(n) steps for M', for some constant d.

Since M makes at most t(n) steps in total, the entire simulation on M' requires at most t(n) $\times$ (d $\cdot$ t(n)) = d $\cdot$ t^2(n) steps. Thus, M' runs in time O(t^2(n)).

#### On the Requirement for Time Bounds

**Remark 31: By definition, a time bound t must satisfy t(n)** $\ge$ n. The latter requirement indeed makes sense because for other t, a t(n)-time bounded Turing machine is restricted to read a prefix of its input of constant length, hence cannot recognize any interesting language. For a proof, let t be a function where t(n) < n for infinitely many n. Let M be a Turing machine that is t(n)-bounded in the sense that for almost all n or, equivalently, for some n_0 and all n $\ge$ n_0, on inputs of length n, the running time of M is at most t(n). Pick n_1 $\ge$ n_0 such that t(n_1) < n_1. Then on all inputs of length greater than n_1, M reads at most the first n_1 symbols of its input. For a proof by contradiction, assume there is a word w of length |w| > n_1 such that M on input w reads at least the first n_1 + 1 symbols of its input. Let u be the prefix of w of length n_1. Then M scans on input u at least n_1 + 1 cells of its input tape, hence makes at least n_1 > t(n_1) = t(|u|) steps, which contradicts the choice of n_0 and n_1.

### Nondeterministic Time Complexity

### Nondeterministic Time-Bounded Turing Machines

The concept of time complexity can be extended from deterministic to nondeterministic Turing machines. A nondeterministic Turing machine may have multiple possible transitions from a given configuration, leading to a tree of possible computations. A word w is accepted if at least one of these computational paths leads to an accepting state.

When analyzing the time complexity of a nondeterministic machine, we consider the length of the longest possible computation path.

**Definition 32: Nondeterministic time classes Let t be a time bound. A Turing machine M is t(n)** -time bounded if M is total (all computation paths halt) and for almost all inputs w, all computations of M have length at most t(|w|). The class of languages decidable in nondeterministic time t(n) is:  $\text{NTIME}$(t(n)) = {L $\subseteq$ {0, 1}^* : L = L(M) $\text{ for a }$ t(n)$\text{-time bounded Turing machine }$ M}.

**Remark 34 In the literature, one also finds a variant of the notion t(n)** -time bounded Turing machine where the length bound t(|w|) is required only for accepting computations, while nonaccepting computations may have arbitrary finite or even infinite length. For time-constructible time bounds t with t(n) $\ge$ 2n, the alternative definition is essentially equivalent to the one presented here. Here a function t is time-constructible if the function 1^n $\mapsto$ 1^{t(n)} can be computed in time t(n). For such t, it is possible to equip a Turing machine that is t(n)-time bounded in the sense of the variant with a timing mechanism or clock that enforces termination after t(n) steps on all computations such that the recognized language remains the same and the clocked Turing machine is t(n)-time bounded according to Definition 32.

### Nondeterministic Time Complexity Classes

Similar to the deterministic case, we can define major complexity classes based on nondeterministic time bounds.

**Definition 33: Examples of nondeterministic time classes We define the complexity classes:**

* NP = $\text{NTIME}$($\text{poly}$)
* NE = $\text{NTIME}$(2^{$\text{lin}$}) = $\bigcup$_{c $\in$ $\mathbb{N}$} $\text{NTIME}$(2^{cn+c})
* NEXP = $\text{NTIME}$(2^{$\text{poly}$}) = $\bigcup$_{c $\in$ $\mathbb{N}$} $\text{NTIME}$(2^{n^c+c})

We refer, for example, to NP as the class of problems decidable in nondeterministic polynomial time and to NE as the class of problems decidable in nondeterministic linear exponential time.

### Reducibility and NP-Completeness

### Polynomial-Time Many-One Reducibility

To compare the relative difficulty of problems, we use the concept of reduction. A reduction is a way to solve one problem using an algorithm for another. If problem A can be reduced to problem B, it means that A is "no harder than" B.

**Definition 35** A language A is many-one reducible in polynomial time to a language B, denoted A $\le_p^m$ B, if there exists a function g $\in$ $\text{FP}$ (i.e., computable in deterministic polynomial time) such that for all x $\in$ $\{0, 1\}$^*, it holds that:  x $\in$ A $\iff$ g(x) $\in$ B.

This type of reduction, often called a p-m-reduction or Karp reduction, acts as a translation. To solve an instance x of problem A, we can first compute g(x) in polynomial time and then use a decider for B to determine if g(x) $\in$ B. If B can be decided in polynomial time, then so can A. This relationship transfers difficulty upward: if A is known to be hard, then B must also be hard. Conversely, it transfers simplicity downward: if B is easy, then A must also be easy.

### Downward and Upward Closure

Complexity classes often exhibit closure properties with respect to reductions.

**Definition 36** A set of languages is closed downward under p-m-reducibility if for every language B in the class, every language A $\le_p^m$ B is in the class, too. A set of languages is closed upward under p-m-reducibility if for every language A in the class, every language B with A $\le_p^m$ B is in the class, too.

Many important complexity classes, such as P and NP, are closed downward. This means that if a problem B is in the class, any problem that is polynomially reducible to B is also in that class.

**Proposition 37:** Downward closure of P and NP under p-m-reducibility Let A and B be languages such that A $\le_p^m$ B. Then it holds that:

* B $\in$ $\text{P}$ implies A $\in$ $\text{P}$,
* B $\in$ $\text{NP}$ implies A $\in$ $\text{NP}$.

These implications are logically equivalent to their contrapositions, i.e., to:

* A $\notin$ $\text{P}$ implies B $\notin$ $\text{P}$,
* A $\notin$ $\text{NP}$ implies B $\notin$ $\text{NP}$.

**Proof.**: We prove the first implication, B $\in$ $\text{P}$ $\implies$ A $\in$ $\text{P}$. The proof for NP is very similar. Let A $\le_p^m$ B via a function g $\in$ $\text{FP}$. Let M_g be a deterministic Turing machine that computes g in time p_g(n) for some polynomial p_g. Since B $\in$ $\text{P}$, there is a deterministic Turing machine M_B that decides B in time p_B(n) for some polynomial p_B.

We can construct a deterministic Turing machine M_A to decide A as follows:

1. On input x, M_A first runs M_g to compute g(x). This takes p_g(|x|) time. The length of the output g(x) is at most p_g(|x|).
2. Next, M_A simulates M_B on the input g(x). This takes p_B(|g(x)|) $\le$ p_B(p_g(|x|)) time.
3. M_A accepts if and only if M_B accepts g(x).

The total running time for M_A is bounded by p_g(|x|) + p_B(p_g(|x|)). Since the composition of two polynomials is a polynomial, M_A is a polynomially time-bounded machine. Therefore, A $\in$ $\text{P}$.

### Properties of Reducibility

The $\le_p^m$ relation behaves like a partial order on languages.

**Proposition 38** The relation $\le_p^m$ is reflexive and transitive.

**Proof.**:

* Reflexivity: For any language A, A $\le_p^m$ A via the identity function g(x) = x, which is computable in polynomial (linear) time.
* Transitivity: Suppose A $\le_p^m$ B via function g_B and B $\le_p^m$ C via function g_C. Let g_B be computable in time p_B(n) and g_C be computable in time p_C(n). The reduction from A to C is the composition g_C $\circ$ g_B. We have:  x $\in$ A $\iff$ g_B(x) $\in$ B $\iff$ g_C(g_B(x)) $\in$ C.  The time to compute g_C(g_B(x)) is the time to compute g_B(x) (which is p_B(n)) plus the time to compute g_C on the result (which is p_C(|g_B(x)|) $\le$ p_C(p_B(n))). The total time is p_B(n) + p_C(p_B(n)), which is a polynomial. Thus, A $\le_p^m$ C.

### NP-Hardness and NP-Completeness

Reducibility allows us to identify the "hardest" problems within a complexity class. For the class NP, these are the NP-complete problems.

**Definition 39:** NP-complete languages A language B is NP-hard if all languages in NP are p-m-reducible to B. A language is NP-complete if it is NP-hard and belongs to NP.

An NP-complete problem is thus a problem that is both in NP and is at least as hard as every other problem in NP.

There are further, provably different notions of NP-hardness and NP-completeness, which are defined in terms of reducibilities different from p-m-reducibility. For example, A is Turing reducible to B in polynomial time if A can be decided by a polynomially time-bounded Turing machine that for arbitrary words y can access the information whether y is in B by writing y onto a special oracle tape. Accordingly, the notions just defined may be referred to more precisely as NP-hardness and NP-completeness with respect to p-m-reducibility.

### The P versus NP Problem

### Relationship Between P, NP, and NP-Complete Problems

The existence of NP-complete problems provides a powerful way to frame one of the most significant open questions in computer science: whether P equals NP.

**Theorem 40 The following statements are equivalent. (i)** $\text{P}$ = $\text{NP}$. (ii) All NP-complete sets are in $\text{P}$. (iii) There is an NP-complete set in $\text{P}$.

**Proof.**:

* (i) $\implies$ (ii): If $\text{P}$ = $\text{NP}$, then every language in NP is also in P. By definition, NP-complete languages are in NP, so they must also be in P.
* (ii) $\implies$ (iii): This implication is trivial, assuming that NP-complete languages exist (which will be proven later).
* (iii) $\implies$ (i): We need to show that if some NP-complete language L is in P, then $\text{NP}$ $\subseteq$ $\text{P}$. (The inclusion $\text{P}$ $\subseteq$ $\text{NP}$ is true by definition). Let L be an NP-complete language such that L $\in$ $\text{P}$. Since L is NP-hard, every language A $\in$ $\text{NP}$ is p-m-reducible to L (A $\le_p^m$ L). From Proposition 37, we know that if A $\le_p^m$ L and L $\in$ $\text{P}$, then A $\in$ $\text{P}$. Since this holds for any arbitrary language A $\in$ $\text{NP}$, it follows that $\text{NP}$ $\subseteq$ $\text{P}$. Therefore, $\text{P}$ = $\text{NP}$.

The Significance of the P vs. NP Problem

**Remark 41** The class P is a subclass of NP, but it is not known whether this inclusion is proper. Whether the two classes coincide is referred to as P versus NP problem. By Theorem 40, the two classes coincide if and only if some NP-complete language is in P. Common ways to express that a language is in P are to say that the corresponding problem is feasible, can be solved efficiently, or has an efficient algorithm. Note that all three notions are also used in other contexts with different meanings. There are hundreds of practically relevant problems that are NP-complete, and for none of them an efficient algorithms is known. In fact, not even algorithms are known that are significantly faster than an exhaustive search over all possible solutions.

### A Criterion for Proving NP-Completeness

The property of transitivity provides a practical method for proving that new problems are NP-complete. Once we have one known NP-complete problem, we can use it as a starting point.

**Remark 42** A language B is NP-hard if and only if there is an NP-hard language A with A $\le_p^m$ B. In the special case of a language B in NP, this implies that B is NP-complete if and only if there is an NP-complete language A such that A $\le_p^m$ B. Both statements follow easily from the fact that p-m-reducibility is reflexive and transitive.

To establish that the first NP-complete problem exists, it will be shown that the satisfiability problem for propositional formulas (SAT) is NP-complete. Subsequently, the NP-completeness of other problems can be established by demonstrating a polynomial-time reduction from SAT (or any other known NP-complete problem) to the new problem in question.


## Computability and Complexity

### Reducibility and Completeness

### Polynomial-Time Many-One Reducibility Degrees

The concept of reducibility allows us to compare the relative difficulty of computational problems. When two problems can be reduced to each other, they are considered equivalent in terms of their computational complexity. This equivalence can be formalized into distinct classes.

p-m-equivalence: Two languages A and B are p-m-equivalent, denoted A $\equiv_p^m$ B, if A $\le_p^m$ B and B $\le_p^m$ A.

p-m-degree: The p-m-degree of a language A is the class of all languages that are p-m-equivalent to it:  $\text{deg}_p^m$(A) = $\{B $\subseteq$ \{0, 1\}$^* : A $\equiv_p^m$ B\} 

**Remark 44:** The p-m-equivalence is an equivalence relation. Consequently, the p-m-degrees are the equivalence classes of this relation. This means that the p-m-degrees form a partition of the class of all binary languages. Recall that for any equivalence relation on a set A, its equivalence classes are mutually disjoint, and their union is equal to A. The relation of p-m-reducibility ($\le_p^m$) induces a nonstrict partial order on these p-m-degrees, where one degree is considered "smaller" than another if a language from the first degree can be p-m-reduced to a language in the second.

### Degrees within P and NP

Within the class NP, the structure of p-m-degrees reveals important insights into the relationship between complexity classes.

**Remark 45:** The set of all NP-complete languages forms a single p-m-degree. Similarly, the class P, excluding the trivial languages $\emptyset$ and $\{0, 1\}$^*, also forms a p-m-degree. It has been shown that if P $\ne$ NP, then the class NP contains infinitely many distinct p-m-degrees. In fact, under this assumption, every finite partial order can be embedded into the structure of these degrees.

### An Anomaly of P-M-Reducibility

The formal definition of p-m-reducibility leads to some unusual properties, particularly concerning languages in P and the two trivial languages, the empty set ($\emptyset$) and the set of all strings ($\{0, 1\}$^*).

**Remark 46:** Since the class P is downward closed under p-m-reducibility, no language outside of P can be p-m-reduced to a language in P. Conversely, every language in P is p-m-reducible to any other language, with the exception of $\emptyset$ and $\{0, 1\}$^*.

To prove this, let A be a language in P and let B be any language such that B $\ne$ $\emptyset$ and B $\ne$ $\{0, 1\}$^*. Since B is not trivial, we can select an element x_1 $\in$ B and an element x_0 $\notin$ B. The p-m-reduction from A to B can be defined by the function that maps every x $\in$ A to x_1 and every x $\notin$ A to x_0. This function is computable in polynomial time because A is in P.

**Remark 47:** By definition, only the empty set can be p-m-reduced to the empty set. A similar restriction applies to the language $\{0, 1\}$^*. As noted in Remark 46, every language in P is reducible to all other languages. The special behavior of $\emptyset$ and $\{0, 1\}$^* is often considered an anomaly of the definition. For this reason, alternative definitions of p-m-reducibility are sometimes used to avoid these edge cases.

### The Satisfiability Problem (SAT)

To prove that a language is NP-hard, the standard method is to show that a known NP-hard language can be p-m-reduced to it. However, this technique requires a starting point: a first problem that is proven to be NP-hard from fundamental principles. The canonical first NP-hard problem is the Boolean Satisfiability Problem, or SAT.

### Propositional Formulas

We begin by formally defining the components of propositional logic.

**Definition 48: Propositional Formula: Let $\Lambda$ = $\{$\neg$, $\land$, (, )** \}$ be a set of logical symbols and let Var be a countable set of variables disjoint from $\Lambda$. The set of propositional formulas over Var is a set of words over the alphabet Var $\cup$ $\Lambda$, defined inductively:

* Base case: Every element of Var is a propositional formula.
* Inductive step: If $\phi$ and $\phi$' are propositional formulas, then so are $\neg$$\phi$ and ($\phi$ $\land$ $\phi$').

In this context, elements of Var are propositional variables, while $\neg$ represents logical negation (NOT) and $\land$ represents logical conjunction (AND). Other logical operators can be introduced as shorthand. For instance, disjunction (OR), denoted by $\lor$, can be expressed using De Morgan's laws: X $\lor$ Y is shorthand for $\neg$($\neg$ X $\land$ $\neg$ Y). Constants for true (1) and false (0) can also be included. For readability, standard rules for operator precedence are often used, allowing parentheses to be omitted.

The truth value of a formula, denoted val($\phi$), is either true (1) or false (0). This value is determined relative to an assignment b: Var $\to$ $\{0, 1\}$ that assigns a truth value to each variable. The truth value of a complex formula is defined inductively:

* Base case: For a variable X $\in$ Var, val(X) = b(X).
* Inductive step: For formulas $\phi$ and $\phi$',
  * val($\neg$$\phi$) = 1 - val($\phi$)
  * val($\phi$ $\land$ $\phi$') = $\min$$\{val($\phi$), val($\phi$')\}$

The set of variables that appear in a formula $\phi$ is denoted by var($\phi$). The truth value of $\phi$ depends only on the assignment of values to the variables in var($\phi$). The set var($\phi$) is defined inductively as:

* var(X) = $\{X\}$ for any X $\in$ Var.
* var($\neg$$\phi$) = var($\phi$).
* var($\phi$ $\land$ $\phi$') = var($\phi$) $\cup$ var($\phi$').

### Satisfiability and Normal Forms

The central question of the satisfiability problem is whether a formula can be true.

**Definition 49:**

* A propositional formula is satisfiable if there exists an assignment of its variables that makes the formula true.
* A literal is a propositional variable or a negated propositional variable (e.g., X or $\neg$ X).
* A clause (specifically, a disjunctive clause) is a disjunction of literals (e.g., X $\lor$ $\neg$ Y $\lor$ Z).
* A propositional formula is in conjunctive normal form (CNF) if it is a nonempty conjunction of clauses.
* A formula is in k-conjunctive normal form (k-CNF) if it is in CNF and each clause contains at most k literals.

For example, the formula ($\neg$ X $\lor$ Y $\lor$ Z ) $\land$ (X $\lor$ Z ) $\land$ (X $\lor$ $\neg$ Y $\lor$ $\neg$ Z ) is in 3-CNF.

This leads to the formal definition of the satisfiability problems as languages.

**Definition 50: The Satisfiability Problem:** Assuming an appropriate representation of propositional formulas as binary strings, we define the following languages for any k > 0:

SAT = \{$\phi$ $\mid$ $\phi$ $\text{ is a satisfiable propositional formula in CNF}$\}

k-SAT = \{$\phi$ $\mid$ $\phi$ $\text{ is a satisfiable propositional formula in k-CNF}$\}

### NP-Completeness of SAT and k-SAT

The languages SAT and k-SAT (for k $\ge$ 3) are cornerstone examples of NP-complete problems.

#### SAT is in NP

First, we must establish that these problems belong to the class NP.

**Remark 51:** The language SAT and the languages k-SAT for k > 0 are members of the class NP.

A proof for this involves a nondeterministic Turing machine (NTM) that operates in polynomial time. Given an input string, the NTM first deterministically checks if it represents a valid formula $\phi$ in CNF (or k-CNF). If it is a valid formula, the machine proceeds to:

1. Guess: Nondeterministically guess an assignment for all variables occurring in $\phi$. This involves creating branches for each variable, assigning it 0 in one branch and 1 in another, effectively exploring all possible assignments.
2. Check: Deterministically evaluate the formula $\phi$ with the guessed assignment to see if it results in true.

If any assignment makes the formula true, at least one computational path of the NTM will accept. The length of an assignment is linear in the size of the formula, so both the guessing and checking phases can be completed in polynomial time. Therefore, SAT and k-SAT are in NP.

### Cook's Theorem: SAT is NP-Complete

The proof that SAT is not just in NP but is NP-hard is a landmark result in computer science.

**Theorem 52 (Cook’s Theorem)** : The language SAT is NP-complete.

**Proof.**: We have already established in Remark 51 that SAT is in NP. To prove that it is NP-complete, we must show it is NP-hard. This requires showing that every language A $\in$ $\text{NP}$ is p-m-reducible to SAT.

Let A be any language in NP. We will construct a function g: x $\mapsto$ $\phi_x$, computable in polynomial time (i.e., g $\in$ $\text{FP}$), that maps any binary word x to a propositional formula $\phi_x$ in CNF. This construction will ensure that x $\in$ A if and only if $\phi_x$ is satisfiable.

Since A $\in$ $\text{NP}$, there exists a nondeterministic Turing machine (NTM) M = (Q, $\Sigma$, $\Gamma$, $\Delta$, s, F) and a polynomial p such that M recognizes A within a time bound of p(n), where n is the input length. For simplicity, we can assume M has only a single tape. Any multi-tape NTM can be converted to an equivalent single-tape NTM with only a polynomial increase in runtime, similar to the tape reduction construction in Theorem 30.

Given an input x of length n, our goal is to construct a formula $\phi_x$ that is satisfiable if and only if M accepts x. The formula will essentially describe the behavior of M on input x. A satisfying assignment for $\phi_x$ will correspond directly to an accepting computation of M on x.

An accepting computation is a sequence of configurations, starting with the initial configuration for input x and ending in an accepting configuration. The length of this sequence is at most p(n) + 1. We will formalize a sequence of exactly p(n)+1 configurations, repeating the final configuration if the computation halts early. The formula $\phi_x$ will encode the rules of M's transition relation, $\Delta$, to ensure that each configuration in the sequence legally follows from the previous one.

#### The overall structure of the proof is as follows:

1. If M accepts x, there exists a valid sequence of configurations representing an accepting computation. This sequence will directly provide a satisfying assignment for the variables in $\phi_x$.
2. Conversely, if $\phi_x$ is satisfiable, any satisfying assignment can be used to decode a valid, accepting computation of M on input x.
3. The formula $\phi_x$ will be a conjunction of subformulas, each of which can be expressed in CNF. Therefore, $\phi_x$ itself can be written in CNF.
4. The construction of $\phi_x$ can be carried out in polynomial time with respect to n = |x|, because the number and size of the subformulas are polynomially bounded in n.

Together, these points establish that A $\le_p^m$ $\text{SAT}$, proving that SAT is NP-hard.

#### Technical Note on Logical Notation

In the construction, we will frequently express clauses as implications. The following logical equivalences are useful:

* An implication $\alpha$ $\to$ $\beta$ is equivalent to $\neg$ $\alpha$ $\lor$ $\beta$.
* The negation of a conjunction $\neg$($\gamma_1$ $\land$ $\dots$ $\land$ $\gamma_r$) is equivalent to a disjunction of negations $\neg$ $\gamma_1$ $\lor$ $\dots$ $\lor$ $\neg$ $\gamma_r$.
* A clause of the form L_1 $\land$ $\dots$ $\land$ L_r $\to$ L_{r+1} (where each L_i is a literal) is equivalent to the disjunctive clause $\neg$ L_1 $\lor$ $\dots$ $\lor$ $\neg$ L_r $\lor$ L_{r+1}.

#### Formalization of the Turing Machine Computation

A computation of M on input x is a sequence of configurations over p(n)+1 time steps. Each configuration is defined by the tape content, head position, and current state.

#### We define index sets to refer to time steps, tape cells, states, and nondeterministic choices:

* Time steps: I = $\{0, 1, $\dots$, p(n)\}$
* Tape cells: J = $\{-p(n)+1, $\dots$, p(n)+1\}$ (covers all cells the head can possibly reach)
* States: K = $\{1, $\dots$, t\}$, where t = |Q| and Q=$\{q_1, $\dots$, q_t\}$
* Instructions/Choices: L = $\{0, 1, $\dots$, d\}$, where d is the number of instructions in $\Delta$. The index 0 represents the "choice" of repeating a halting configuration.
* We also define I^- = I $\setminus$ $\{p(n)\}$ and L^- = L $\setminus$ $\{0\}$.

We introduce the following propositional variables to describe the computation. The intended meaning of each variable being true is given below:

* B_{i,j,a}: At time step i $\in$ I, tape cell j $\in$ J contains symbol a $\in$ $\Gamma$.
* P_{i,j}: At time step i $\in$ I, the tape head is positioned on cell j $\in$ J.
* Z_{i,k}: At time step i $\in$ I, the machine is in state q_k $\in$ Q.
* A_{i,0}: At time step i $\in$ I, the configuration is a halting configuration (i.e., the computation has already terminated).
* A_{i',$\ell$}: At time step i' $\in$ I^-, the $\ell$-th instruction in $\Delta$ is executed to transition from configuration i' to i'+1.

Constructing the Formula $\phi_x$

The formula $\phi_x$ is the conjunction of several subformulas, each enforcing a specific property of a valid accepting computation.

1. Start Configuration: The machine must start in the correct initial configuration at time 0. The state is the start state s = q_{k_0}, the head is at cell 1, the input x=x(1)$\dots$ x(n) is on the tape, and other cells are blank ($\Box$).  Z_{0,k_0} $\land$ P_{0,1} $\land$ $\left$( $\bigwedge_{j=1}^{n}$ B_{0,j,x(j)} $\right$) $\land$ $\left$( $\bigwedge$_{j $\in$ J $\setminus$ $\{1,...,n\}$} B_{0,j,$\Box$} $\right$) $\quad$ (2.4) 
2. Termination and Acceptance: The computation must terminate, and the state at the final time step p(n) must be an accepting state.  A_{p(n),0} $\land$ $\left$( $\bigvee_{k \in K : q_k \in F}$ Z_{p(n),k} $\right$) $\quad$ (2.5) 
3. Uniqueness Constraints: At any time step i, the configuration must be well-defined.
  * Unique Inscription: Each tape cell contains at most one symbol.  $\bigwedge_{(i,j,a,a') \in I \times J \times \Gamma \times \Gamma : a < a'}$ (B_{i,j,a} $\to$ $\neg$ B_{i,j,a'}) $\quad$ (2.6) 
  * Unique Head Position: The head is at most at one position.  $\bigwedge_{(i,j,j') \in I \times J \times J : j < j'}$ (P_{i,j} $\to$ $\neg$ P_{i,j'}) $\quad$ (2.7) 
  * Unique State: The machine is in at most one state.  $\bigwedge_{(i,k,k') \in I \times K \times K : k < k'}$ (Z_{i,k} $\to$ $\neg$ Z_{i,k'}) $\quad$ (2.8) 
  * Unique Instruction: At most one instruction is executed. No instruction is executed if and only if the configuration is halting (A_{i,0} is true).  $\bigwedge_{(i,\ell,\ell') \in I \times L \times L : \ell < \ell'}$ (A_{i,$\ell$} $\to$ $\neg$ A_{i,$\ell$'}) $\quad$ (2.9) 
4. Valid Transitions: The configuration at step i+1 must follow legally from the configuration at step i.
  * Writing only at the head position: Tape cells not under the head do not change.  $\bigwedge_{(i,j,a) \in I^- \times J \times \Gamma}$ (B_{i,j,a} $\land$ $\neg$ P_{i,j} $\to$ B_{i+1,j,a}) $\quad$ (2.10) 
  * Halting configurations remain unchanged: If a configuration is halting, all subsequent configurations are identical.  $\bigwedge_{(i,j,a) \in I^- \times J \times \Gamma}$ (A_{i,0} $\land$ B_{i,j,a} $\to$ B_{i+1,j,a}) $\quad$ (2.11)   $\bigwedge_{(i,j) \in I^- \times J}$ (A_{i,0} $\land$ P_{i,j} $\to$ P_{i+1,j}) $\land$ $\bigwedge_{(i,k) \in I^- \times K}$ (A_{i,0} $\land$ Z_{i,k} $\to$ Z_{i+1,k}) $\quad$ (2.12) 
  * Instruction Execution: In every non-halting configuration, exactly one instruction must be executed.  $\bigwedge_{i \in I}$ $\left$( $\bigvee_{\ell \in L}$ A_{i,$\ell$} $\right$) $\quad$ (2.13) 
  * For the $\ell$-th instruction in $\Delta$, let it be (q_{k_$\ell$}, a_$\ell$, q_{k'_$\ell$}, a'_$\ell$, B_$\ell$), where B_$\ell$ is the head movement (L, R, S). Let $\delta$_$\ell$ be -1, 1, 0 for L, R, S respectively.
  * A configuration is not halting if an instruction applies to it.  $\bigwedge_{(i,j,\ell) \in I \times J \times L^-}$ (Z_{i,k_$\ell$} $\land$ P_{i,j} $\land$ B_{i,j,a_$\ell$} $\to$ $\neg$ A_{i,0}) $\quad$ (2.14) 
  * If instruction $\ell$ is chosen, its preconditions must be met by configuration i.  $\bigwedge_{(i,\ell) \in I^- \times L^-}$ (A_{i,$\ell$} $\to$ Z_{i,k_$\ell$}) $\land$ $\bigwedge_{(i,j,\ell) \in I^- \times J \times L^-}$ (A_{i,$\ell$} $\land$ P_{i,j} $\to$ B_{i,j,a_$\ell$}) $\quad$ (2.15) 
  * If instruction $\ell$ is chosen at step i, the successor configuration i+1 must reflect its execution.  $\bigwedge_{(i,\ell) \in I^- \times L^-}$ (A_{i,$\ell$} $\to$ Z_{i+1,k'_$\ell$}) $\quad$ (2.16)   $\bigwedge_{(i,\ell,j) \in I^- \times L^- \times J}$ (A_{i,$\ell$} $\land$ P_{i,j} $\to$ B_{i+1,j,a'_$\ell$}) $\quad$ (2.17)   $\bigwedge_{(i,\ell,j) \in I^- \times L^- \times J}$ (A_{i,$\ell$} $\land$ P_{i,j} $\to$ P_{i+1,j+$\delta$_$\ell$}) $\quad$ (2.18) 

### Conclusion of the Proof

The full formula $\phi_x$ is the conjunction of all subformulas (2.4) through (2.18). Every accepting computation of M on input x defines a satisfying assignment for the variables in $\phi_x$. Conversely, any satisfying assignment for $\phi_x$ encodes a valid sequence of configurations. The values of the A_{i,$\ell$} variables determine which instruction is executed at each step. Starting from the initial configuration (enforced by 2.4), one can inductively determine the entire sequence of configurations. The subformulas ensure this sequence represents a valid computation that eventually reaches an accepting state.

Thus, x $\in$ A if and only if $\phi_x$ is satisfiable. This completes the proof of Theorem 52. ∎

k-SAT is NP-Complete for k $\ge$ 3

Building on Cook's Theorem, we can show that variants of SAT are also NP-complete.

**Corollary 53:** For all k $\ge$ 3, the language k-SAT is NP-complete.

**Proof.**: As established in Remark 51, all k-SAT languages are in NP. We need to show they are NP-hard for k $\ge$ 3. We will demonstrate this for the case k=3. The result for k>3 follows because 3-SAT can be easily p-m-reduced to k-SAT (by padding clauses with dummy variables, or more simply, noting that any instance of 3-SAT is already an instance of k-SAT for k > 3).

To show that 3-SAT is NP-hard, we use the transitivity of p-m-reducibility. By Cook's Theorem, SAT is NP-hard. Therefore, if we can show that $\text{SAT}$ $\le_p^m$ $\text{3-SAT}$, it follows that 3-SAT is also NP-hard.

We need to construct a polynomial-time computable function that transforms a formula $\phi$ into a formula $\phi$' such that $\phi$ $\in$ $\text{SAT}$ if and only if $\phi$' $\in$ $\text{3-SAT}$.

Let $\phi$ be a given propositional formula.

* If $\phi$ is not in CNF, it cannot be in SAT. In this case, we map it to a fixed, unsatisfiable 3-CNF formula (e.g., (X) $\land$ ($\neg$ X)).
* If $\phi$ is in CNF, it has the form $\phi$ $\equiv$ C_1 $\land$ C_2 $\land$ $\dots$ $\land$ C_m, where each C_i is a clause. The formula $\phi$' is obtained by replacing each clause C_i in $\phi$ with a new set of clauses $\kappa_i$, constructed as follows.

Let a clause C_i be (L_1^i $\lor$ L_2^i $\lor$ $\dots$ $\lor$ L_{k_i}^i), where L_j^i are literals.

* If k_i $\le$ 3, the clause is already in 3-CNF form, so we can let $\kappa_i$ $\equiv$ C_i.
* If k_i > 3, we replace C_i with a conjunction of new clauses that collectively are satisfiable if and only if C_i is. We introduce k_i - 3 new, unique variables Z_1^i, Z_2^i, $\dots$, Z_{k_i-3}^i. The replacement formula $\kappa_i$ is:  (L_1^i $\lor$ L_2^i $\lor$ Z_1^i) $\land$ ($\neg$ Z_1^i $\lor$ L_3^i $\lor$ Z_2^i) $\land$ $\dots$ $\land$ ($\neg$ Z_{k_i-2}^i $\lor$ L_{k_i-1}^i $\lor$ Z_{k_i-1}^i) $\land$ ($\neg$ Z_{k_i-1}^i $\lor$ L_{k_i}^i)  (The source document presents a slightly different but equivalent construction, which we will analyze): Let C_i = (L_1^i $\lor$ $\dots$ $\lor$ L_{k_i}^i). We introduce k_i-1 new variables Z_1^i, $\dots$, Z_{k_i-1}^i. The clause C_i is replaced by $\kappa_i$:  $\kappa_i$ $\equiv$ (L_1^i $\lor$ Z_1^i) $\land$ $\left$( $\bigwedge_{j=2}^{k_i-1}$ ($\neg$ Z_{j-1}^i $\lor$ L_j^i $\lor$ Z_j^i) $\right$) $\land$ ($\neg$ Z_{k_i-1}^i $\lor$ L_{k_i}^i)  The final formula is $\phi$' = $\kappa_1$ $\land$ $\kappa_2$ $\land$ $\dots$ $\land$ $\kappa_m$. This transformation introduces a polynomial number of new variables and clauses and is computable in polynomial time.

Now, we must show that $\phi$ is satisfiable if and only if $\phi$' is satisfiable.

($\Rightarrow$) Assume $\phi$ is satisfiable. Let b be a satisfying assignment for $\phi$. For each clause C_i = (L_1^i $\lor$ $\dots$ $\lor$ L_{k_i}^i) in $\phi$, at least one literal must be true under b. Let t_i be the index of the first true literal in C_i. We can extend the assignment b to the new variables Z_j^i as follows:

* Set Z_j^i to true for all j < t_i.
* Set Z_j^i to false for all j $\ge$ t_i. Under this extended assignment, every clause in every $\kappa_i$ becomes true, thus satisfying $\phi$'.

($\Leftarrow$) Assume $\phi$ is unsatisfiable. We will show that $\phi$' must also be unsatisfiable. Let b be an arbitrary assignment for the variables in $\phi$'. Since $\phi$ is unsatisfiable, under the restriction of b to the original variables, at least one clause C_i in $\phi$ must be false. This means all literals L_1^i, $\dots$, L_{k_i}^i are false. Now consider the corresponding formula $\kappa_i$ under assignment b:  $\kappa_i$ $\equiv$ (L_1^i $\lor$ Z_1^i) $\land$ ($\neg$ Z_1^i $\lor$ L_2^i $\lor$ Z_2^i) $\land$ $\dots$ $\land$ ($\neg$ Z_{k_i-1}^i $\lor$ L_{k_i}^i)  Since L_1^i is false, the first clause (L_1^i $\lor$ Z_1^i) implies that Z_1^i must be true to satisfy $\kappa_i$. Now consider the second clause ($\neg$ Z_1^i $\lor$ L_2^i $\lor$ Z_2^i). Since Z_1^i is true and L_2^i is false, this clause implies Z_2^i must be true. Propagating this logic forward, we find that for $\kappa_i$ to be true, Z_1^i, Z_2^i, $\dots$, Z_{k_i-1}^i must all be true. However, the final clause is ($\neg$ Z_{k_i-1}^i $\lor$ L_{k_i}^i). Since Z_{k_i-1}^i must be true and L_{k_i}^i is false, this final clause evaluates to false. Therefore, $\kappa_i$ is false. Since $\phi$' is a conjunction containing $\kappa_i$, $\phi$' is also false. As b was an arbitrary assignment, this shows that $\phi$' is unsatisfiable.

Thus, we have shown that $\text{SAT}$ $\le_p^m$ $\text{3-SAT}$, which completes the proof. ∎
