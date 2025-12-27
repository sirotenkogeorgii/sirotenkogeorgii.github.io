---
title: Universal Turing Machine
layout: default
noindex: true
---

# Universal Turing Machine

## Definiton

A **Universal Turing machine (UTM)** is a (theoretical) Turing machine that can **simulate any other Turing machine**.

Instead of being built to do one specific task, a UTM takes two inputs:

* a **description (“code”) of another Turing machine** $M$ — think of it like the program, and
* an **input string** $x$ for that machine — the data,

and then it runs **as if it were $M$ running on $x$**.

So you can write this idea as:

$$U(\langle M\rangle, x) = M(x)$$

where $\langle M\rangle$ means "an encoding of machine $M$."

### Why it matters

* It’s the mathematical core of the **stored-program computer** idea: one machine can run many different programs, as long as the programs are represented as data.
* It underpins the concept of a **general-purpose computer**.
* It’s central to results like the **Halting Problem**: if you can encode machines as data and simulate them, you can also reason about what can’t be decided mechanically.

### Tiny intuition

* A normal Turing machine is like a **single-purpose calculator** (built for one algorithm).
* A universal one is like a **computer + an interpreter**, where the "program" you feed it determines what it does.

## Mathematical Theory

Once you encode a Turing machine’s transition (action) table as a string, you can — at least in principle — build Turing machines that take those strings as input and reason about how *other* machines behave. But most interesting questions of that kind turn out to be **undecidable**: there is no general mechanical procedure that always gives the correct answer. A classic example is the **Halting Problem** — deciding whether an arbitrary Turing machine halts on a given input (or on all inputs) — which Turing proved is undecidable in his original paper. More broadly, **Rice’s theorem** says that every non-trivial property about what a Turing machine computes (i.e., about its output behavior) is undecidable.

A **universal Turing machine** can simulate any computation that is computable: it can compute any **recursive (total computable) function**, decide any **recursive language**, and accept any **recursively enumerable** language. Under the **Church–Turing thesis**, the set of problems a UTM can solve matches the set of problems solvable by an algorithm or any effective method of computation (in any reasonable formalization). Because of this, UTMs are used as a benchmark for comparing models of computation, and any system capable of simulating a UTM is called **Turing complete**.

There’s also a more abstract counterpart: a **universal function**, meaning a single computable function that can be used to compute any other computable function when given an appropriate description of it. The **UTM theorem** guarantees that such a universal function exists.

## UTM theorem

> There exists a single **computable “universal” function** $u(e,x)$ that can reproduce the behavior of *any* computable (more precisely: **partial computable**) function when you plug in the right code number $e$.

### Formal statement (typical version)

There is a **partial computable** function $u(e,x)$ such that for every **partial computable** function $f(x)$, there exists a natural number $e$ (think: an index / Gödel number / program code) with

$$f(x) \simeq u(e,x)\ \ \text{for all } x.$$


Here $\simeq$ means **partial equality**: for each $x$, either both sides are undefined, or both are defined and equal.

### Intuition

* $e$ is like “the program”
* $x$ is the program’s input
* $u$ is a universal interpreter: **run program $e$ on input $x$**

This is the function-version of a **universal Turing machine**, which takes an encoding $\langle M\rangle$ of a machine $M$ plus an input $w$, and simulates $M(w)$.

### What it gives you

A clean way to say: “we can effectively list all partial computable functions” by defining

$$\varphi_e(x) = u(e,x),$$

so $\varphi_0, \varphi_1, \varphi_2, \dots$ is an **effective enumeration** of all partial computable functions.

### Why it matters

It’s one of the foundational tools behind:
* formal"program-as-data" reasoning (machines/functions taking *descriptions* of other machines/functions),
* classic undecidability proofs (halting, Rice’s theorem style arguments),
* and results about acceptable Gödel numberings (often discussed alongside the s-m-n theorem / Rogers’ equivalence theorem).

## Sketch of building a UTM $U$

<figure>
  <img src="{{ '/assets/images/notes/computability-and-complexity/universal-turing-machine.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>Universal Turing Machine</figcaption>
</figure>

### 1. Encode a machine as data

Pick a scheme that turns any Turing machine $M$ into a string $\langle M\rangle$.

A TM $M$ is basically a finite set of transition rules like:

$$(q, a) \mapsto (q', b, D)$$

meaning: "if you’re in state $q$ reading symbol $a$, write $b$, move $D \in lbrace L,R,S \rbrace$, and go to state $q'$."

So your encoding needs to represent:

* states $q, q'$ (as numbers, say),
* symbols $a, b$,
* direction $D$,
* and a way to separate rules unambiguously.

Example *shape* (not the only way):
`rule# rule# rule# ...` where each rule looks like
`q,a -> q',b,D`.

### 2. Put both “program” and “data” on $U$’s tape(s)

Conceptually, $U$ takes two inputs:

* $\langle M\rangle$ = the “program”
* $x$ = the input for $M$

A common setup is a **multi-tape** UTM (simpler to explain):

* **Tape 1:** $\langle M\rangle$ (read-only “program tape”)
* **Tape 2:** a simulation of $M$’s tape contents
* **Tape 3:** bookkeeping (like current state, head position, temporary parsing)

(You *can* do it on one tape too; it’s just messier and slower.)

### 3. Maintain a full “instantaneous description” of $M$’s run

At any moment, to simulate one step of $M$, $U$ needs:

* the **current state** $q$ of $M$,
* the **symbol under $M$’s head**,
* the rest of $M$’s tape content (left and right),
* where $M$’s head is (implicitly, by how you store the tape).

A simple representation on a simulation tape is something like:

$$\text{(tape-left)}\ \underline{\text{current symbol}}\ \text{(tape-right)}$$

with a marker to show which cell is “under the head”.

### 4. The universal “fetch–decode–execute” loop

Now $U$ repeatedly performs this cycle:

### Step A: Read the simulated configuration

* Look at the simulation tape to get:

  * current state $q$
  * scanned symbol $a$

#### Step B: Find the matching rule in $\langle M\rangle$

* Scan the encoding $\langle M\rangle$ until you find the unique rule whose left side matches $(q,a)$.
* If no rule matches, that means $M$ has no move defined → simulation halts (or rejects, depending on the convention).

#### Step C: Apply the rule to the simulation tape

If the rule says $(q,a) \mapsto (q', b, D)$, then:

* write $b$ in the currently scanned cell of the simulation tape,
* update the stored state from $q$ to $q'$,
* move the head marker left/right/stay according to $D$,
* extend the tape with blanks if needed.

Then repeat.

That’s literally an interpreter: “read instruction table, pick the matching instruction, execute it.”

### 5. Why this works

Because:

* Every TM step depends only on $(\text{state}, \text{scanned symbol})$.
* $U$ can mechanically search the finite rule table in $\langle M\rangle$.
* $U$ can update a stored representation of tape + head + state.
  So $U$ reproduces $M$’s behavior step-for-step.

### 6. One-tape UTMs (what changes)

On a single tape, you typically store something like:

$$\langle M\rangle \# \text{encoded simulated tape with a head marker}$$

Then each simulated step involves lots of back-and-forth scanning:

* locate current state/head marker,
* scan back to the program region to find the matching rule,
* scan back to update the tape region,
* etc.

This is slower (polynomial overhead), but still universal.

### Encoding Turing Machine - Description Number

**In Turing’s original terminology**, the numerical encoding of a machine is its **description number** (often abbreviated **D.N.**).

A couple of small distinctions that help keep terms straight:
* **Standard description (S.D.)**: the machine written as a finite string over some fixed alphabet/syntax.
* **Description number (D.N.)**: a **natural number** obtained by recoding that standard description into digits (Turing gives a specific scheme in his paper).

In modern texts, people often write $\langle M\rangle$ for an **encoding** of $M$. Depending on the author, $\langle M\rangle$ might mean:
* the **string** (like Turing’s S.D.), or
* the **number** (like Turing’s D.N., i.e., a Gödel-number-style encoding).

So: if your $\langle M\rangle$ is explicitly a **number**, calling it a **description number** is perfectly appropriate (especially in a Turing-1936-flavored presentation).

More about description numbers: [Description Numbers](/subpages/computability-and-complexity/description-numbers/)