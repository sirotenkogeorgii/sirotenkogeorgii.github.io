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

## Turing Machines

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

1.0 Introduction: Formalizing the Notion of Computability

In the field of theoretical computer science, a primary objective is to move from the intuitive, informal notion of "computability" to a rigorous, mathematical formalization. This is achieved by defining a class of partial computable functions, from which related concepts such as total computable functions and decidable sets can be derived. While multiple models have been developed to capture this concept—including register machines, µ-recursive functions, and the lambda calculus—they are all provably equivalent in their computational power. In all these models, one obtains the same fundamental understanding of what it means for a function to be computable or a set to be decidable.

This monograph will focus exclusively on the Turing machine as the canonical model for this formalization. The purpose of this document is to provide a comprehensive, self-contained exploration of the Turing machine, beginning with its formal definition and proceeding through its operational mechanics to its profound implications for the theories of computability and computational complexity. We begin with the foundational mathematical definition of the machine itself.

2.0 The Formal Definition of a Turing Machine

The strategic importance of a precise, formal definition cannot be overstated. It is this mathematical rigor that enables the provable analysis of computation, allowing us to establish firm boundaries on what can and cannot be computed. The Turing machine is formally defined as a 6-tuple.

Definition 1 (Turing Machine) A Turing machine with k tapes, or a k-tape TM, is a 6-tuple of the form: M = (Q, Σ, Γ, ∆, s, F )

Each component of this tuple has a specific and critical role in defining the machine's structure and behavior:

* Q: The finite set of states.
* Σ: The input alphabet, which is the set of symbols that can be part of the initial input word.
* Γ: The tape or working alphabet. This is a superset of the input alphabet (Σ ⊆ Γ) that also includes a special blank symbol (□), which is not part of the input alphabet (□ ∈ Γ \ Σ).
* ∆: The transition relation, a subset of Q × Γk × Q × Γk × Movk, where Mov = {L, R, S}. The tuples in ∆ are the instructions of the machine.
* s: The initial state (s ∈ Q), which is the state the machine starts in.
* F: The set of accepting states (F ⊆ Q). If a computation halts in one of these states, the input is considered "accepted."

Intuitively, a k-tape Turing machine operates with a memory structure composed of k work tapes. Each tape is a doubly infinite sequence of cells, and every cell contains a symbol from the tape alphabet Γ. For each tape, there is an independent read-write head that can read the symbol in its current cell, overwrite it with a new symbol, and then move one cell to the left (L), one cell to the right (R), or stay in the same position (S).

The machine's behavior is governed by its set of instructions, which are tuples within the transition relation ∆. An instruction has the following structure:

(q, a1, . . . , ak , q′, a′ 1, . . . , a′ k , Z1, . . . , Zk)

The meaning of such an instruction is straightforward: if the machine is currently in state q and the heads on tapes 1 through k are reading the symbols a1 through ak, respectively, then the machine can execute this instruction. Upon execution, the state changes to q′, the symbols on the tapes are overwritten with a′1 through a′k, and each head performs its respective movement Z1 through Zk.

Per Definition 2, this instruction format is composed of two distinct parts:

* Condition Part: (q, a1, . . . , ak)
* Instruction Part: (q′, a′ 1, . . . , a′ k , Z1, . . . , Zk)

The nature of the relationship between these parts is what gives rise to the critical distinction between deterministic and nondeterministic machines.

3.0 Determinism versus Nondeterminism

The structure of the transition relation is a critical architectural feature that divides Turing machines into two fundamental classes: deterministic and nondeterministic. This distinction has significant consequences for computational models and lies at the heart of major open questions in complexity theory, such as the P versus NP problem.

A deterministic Turing machine is one where the computational path is uniquely determined at every step. This property is formalized by constraining the transition relation.

Definition 4 (Deterministic Turing Machine) A Turing machine M is deterministic if its transition relation is right-unique.

A relation is right-unique if, for every possible condition part, there exists at most one applicable instruction part. In other words, for any given state and combination of symbols under the read-write heads, there is never more than one valid instruction the machine can execute.

In contrast, a general or nondeterministic Turing machine does not have this restriction. In this model, the transition relation may contain multiple instructions with the same condition part. This means that from a single configuration, the machine may have several valid next steps, leading to a branching computation path. It is important to note a subtlety in terminology, as clarified in Remark 5: the term "nondeterministic Turing machine" can refer either to the general class of TMs (which may or may not be deterministic) or, more specifically, to a machine that is explicitly not deterministic.

The conceptual impact of this distinction is profound. A deterministic Turing machine follows a single, linear sequence of computations for any given input. A nondeterministic Turing machine, however, can be conceptualized as exploring a tree of possible computations simultaneously. We now turn to the formal representation of these computation paths as sequences of machine configurations.

4.0 The Mechanics of Computation

To rigorously analyze the behavior of a Turing machine, it is necessary to formalize its step-by-step execution. This is achieved by defining a configuration as a complete snapshot of the machine's status at a single moment and a computation step as the transition between two such configurations.

A key challenge is representing the infinite tape. At any point in a computation, all but a finite number of cells contain the blank symbol. This allows the state of an entire tape to be represented by a finite word.

Definition 6 (Tape inscriptions) A tape's contents can be formally described by a function f: Z → Γ where f(i) = □ for all but finitely many integers i. The non-blank content of such a tape can be represented by a finite word ui over Γ, which corresponds to the block of symbols between the leftmost and rightmost non-blank cells, inclusive.

This finite representation is crucial for the formal definition of a configuration.

Definition 7 (Configuration) A configuration of a k-tape Turing machine is a tuple: (q, u1, . . . , uk , j1, . . . , jk)

Each component of the configuration tuple captures a piece of the machine's state:

* q is the current state.
* ui is a word representing the non-blank portion of the inscription on tape i.
* ji is an integer indicating the head position on tape i, relative to the word ui.

A computation step is the formal transition from one configuration to a successor, denoted C −→ M C'.

Definition 8 (Computation Step) An instruction is applicable to a configuration C if the instruction's condition part matches the current state and the symbols being read by the heads in C. A configuration C' is a successor configuration of C if it is the result of applying an applicable instruction.

A partial computation is a sequence of configurations where each configuration is a successor of the one preceding it. Such a computation can be finite or infinite (Definition 9). The complete lifecycle of a computation, from start to finish, is defined by several key types of configurations.

* Initial Configuration: For a given input word w, the machine starts in a specific configuration: (s, w□, □, . . . , □, 1, . . . , 1). The input is on the first tape, followed by a blank symbol, all other tapes are blank, all heads are at the first position, and the machine is in the initial state s.
* Halting Configuration: A configuration (q, u1, . . . , uk , j1, . . . , jk) is a halting one if (q, u1(j1), . . . , uk(jk)) is not the condition part of any instruction in the transition relation ∆. The computation stops.
* Accepting Configuration: This is a halting configuration where the machine's state q is a member of the set of accepting states F.
* Terminating vs. Accepting Computation: A computation is terminating if it is finite (i.e., it reaches a halting configuration). An accepting computation is a terminating computation that ends in an accepting configuration.

The complete behavior of a machine on a given input w can be visualized as a computation tree (Definition 11). The root of this tree is the initial configuration. The children of any node are its successor configurations. The leaves of the tree are the halting configurations. Each branch from the root to a leaf or an infinite path represents one possible computation. Because applying two different instructions to a configuration results in different successor configurations, the labels of the children of any internal node in the tree are pairwise distinct. For a deterministic machine, this "tree" is simply a single path. The existence of an accepting computation within this tree is directly linked to the formal definition of language recognition.

5.0 Language Recognition and Decidability

In computability theory, the primary purpose of a Turing machine is not merely to perform calculations, but to serve as a formal device for recognizing languages, which are formally defined as sets of strings.

Definition 12 (Acceptance and Recognized Language) A Turing machine M accepts a word w if at least one of its computations on input w is an accepting computation. The language recognized by M, denoted L(M), is the set of all words that M accepts.

This definition highlights a key feature of nondeterminism: a word is accepted if any branch in its computation tree leads to an accepting configuration, even if other branches are infinite (non-terminating).

In complexity theory and for many practical considerations, we are interested in machines that are guaranteed to provide an answer. This leads to the concept of a total machine.

Definition 13 (Total Turing Machine) A Turing machine is total if all of its computations terminate for all possible inputs.

Using this, we can define a foundational class of languages.

Definition 14 (Decidable Language) A language is decidable if it is recognized by a total Turing machine.

A crucial question is whether the power of nondeterminism allows a total Turing machine to decide languages that a deterministic one cannot. For total machines without resource bounds, the answer is no. This is a fundamental result in computability theory.

Theorem 16 Every decidable language is recognized by a total deterministic Turing machine.

The proof of this theorem relies on the fact that the computation tree of a total Turing machine must be finite. Because the machine is total, every branch is finite. According to König’s Lemma (Theorem 15), a finitely branching tree with no infinite branches must be finite. A deterministic Turing machine can therefore be constructed to perform an exhaustive, systematic search of this finite tree. It simulates all possible computations of the original machine, and since the tree is finite, this search is guaranteed to terminate. The new machine accepts if and only if its search discovers an accepting configuration from the original machine's computation tree. This demonstrates that for decidability, determinism is as powerful as nondeterminism. We now transition from deciding set membership to computing function outputs.

6.0 Turing Machines as Function Computers

Beyond acting as language recognizers, Turing machines can be viewed as devices that compute functions. To serve this purpose, where a single, unambiguous output is required for every input, machines must be both deterministic (to ensure a unique output) and total (to ensure an output is always produced).

The output of a computation is derived from the machine's final tape in its unique halting configuration.

Definition 17 (Function Output) For a halting configuration C = (q, u1, . . . , uk , j1, . . . , jk), the output out(C) is the word uk(jk)uk(jk + 1) · · · uk(t), where t = max{i ≤ |uk| : uk(jk), . . . , uk(i) all differ from □}. If uk(jk) is the blank symbol, out(C) is the empty word.

A function is considered computable if such a machine can calculate it. The computable function fM is the function computed by a total deterministic Turing machine M, where fM(w) is the output derived from the halting configuration reached on input w.

The concepts of decidable sets and computable functions can be extended from words over an alphabet to other domains, such as the natural numbers. This is achieved through the use of formal representations.

Definition 18 (Computability on Other Domains) Using a representation scheme (e.g., representing the number n by its binary string bin(n) or a unary string 1n), a set is decidable if the set of its representations is a decidable language. Similarly, a function is computable if there exists a computable Turing machine that maps the representation of an input to the representation of its corresponding output.

Having established what can be computed, the next logical step is to analyze the resources—specifically, time—required for these computations.

7.0 An Introduction to Time Complexity

The move from computability theory to complexity theory marks a shift in focus from what is possible to compute to what is efficiently computable. The primary resource measured to quantify efficiency is the machine's running time.

Definition 20 (Running Time) The running time of a deterministic Turing machine M on input w, denoted timeM(w), is the length (number of steps) of the computation, provided it terminates.

This measure allows us to classify machines based on how their running time scales with the size of the input.

Definition 21 (Time-Bounded Deterministic Turing Machine) A deterministic Turing machine M runs in time t(n), where t is a computable function, if M is total and for almost all inputs w, its running time timeM(w) is bounded by t(|w|).

The analysis focuses on worst-case time complexity, meaning the running time is bounded by the function t(n) for the most time-consuming input of length n. This provides a guaranteed performance ceiling.

This framework allows for the definition of formal complexity classes that group together languages or functions that can be decided or computed within similar time bounds.

* DTIME(t(n)): The class of all languages decidable by a deterministic Turing machine in time t(n).
* FTIME(t(n)): The class of all functions computable by a deterministic Turing machine in time t(n).

From these, several major deterministic time classes are defined using sets of bounding functions:

* LIN (Deterministic Linear Time): DTIME(lin), where lin = {n ↦ c·n + c : c ∈ N \ {0}}
* P (Deterministic Polynomial Time): DTIME(poly), where poly = {n ↦ nc + c : c ∈ N \ {0}}
* E (Deterministic Linear Exponential Time): DTIME(2lin), where 2lin = {n ↦ 2^(c·n+c) : c ∈ N \ {0}}
* EXP (Deterministic Exponential Time): DTIME(2poly), where 2poly = {n ↦ 2^(n^c+c) : c ∈ N \ {0}}

A foundational theorem in this area demonstrates that these classes are robust and not sensitive to linear-scale improvements in machine efficiency.

8.0 The Linear Speedup Theorem

The Linear Speedup Theorem is a significant result demonstrating that constant-factor improvements in the running time of a Turing machine are not fundamental to the complexity of a problem. This theorem formally justifies the use of asymptotic (Big-O) notation and the focus on broad complexity classes like P, since any constant factor can be "sped up" away.

Theorem 27 (Linear Speedup Theorem) For any time bound t(n), any real number α > 0, and any machine with k ≥ 2 tapes, the following holds:

* DTIMEk(t(n)) ⊆ DTIMEk(α · t(n) + n)
* Every function in FTIME(t(n)) is also in FTIME(α · t(n) + n + |f (n)|).

The proof of this theorem is constructive. Given a Turing machine M that runs in time t(n), a new, faster machine M' is built to recognize the same language. The core technique involves "compressing" the tape data. A new machine M' is constructed where each symbol of its larger tape alphabet represents a block of m symbols from M's alphabet.

The simulation proceeds in two phases. First, M' takes n steps to convert its input of length n into the compressed format. Second, M' simulates the computation of M. To simulate m steps of M, M' first reads the cells relevant to the current computation—the cell under each head and its immediate neighbors. This requires a constant number of steps (e.g., 3). This local information is stored in M's finite state. From this state, M' can determine the outcome of m steps of M and update its own tape and head positions, which takes another constant number of steps (e.g., 4).

In total, M' simulates m steps of M in a constant number of its own steps (e.g., 7). The total running time for M' is therefore approximately n + (7/m) * t(n). By choosing a sufficiently large block size m such that 8/m < α, the simulation time (7/m) * t(n) plus any small overhead can be bounded by α · t(n) for almost all n. This demonstrates that any linear constant factor in the running time can be eliminated, affirming the robustness of our time complexity classes.

9.0 Conclusion

This monograph has traced the theoretical arc of the Turing machine, from its rigorous mathematical definition to its central role in the theories of computability and complexity. We began with the formal 6-tuple (Q, Σ, Γ, ∆, s, F ) that precisely defines the machine's components. From there, we explored the critical distinction between deterministic machines, which follow a single computational path, and nondeterministic machines, which explore a tree of possibilities.

The mechanics of computation were formalized through the concepts of configurations and computation steps, providing the language necessary to define decidable languages and computable functions. This foundation in computability—what is theoretically possible to compute—led directly to the principles of complexity theory, which asks what is efficiently computable. By introducing running time as a key resource, we established the framework of time complexity classes like P and EXP. Finally, the Linear Speedup Theorem reinforced the validity of these broad classes by showing that constant-factor efficiencies are not fundamental properties of a computational problem. The Turing machine remains the essential, enduring model in theoretical computer science, providing the formal basis for reasoning about the ultimate limits and practical efficiency of computation and framing seminal questions like the P versus NP problem.
