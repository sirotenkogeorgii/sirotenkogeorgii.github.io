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

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

For any numbering $\alpha_0, \alpha_1, \dots$, all $\alpha_e$ are partial computable. To prove this, choose a Turing machine that computes the principal function of the given numbering. By modifying this Turing machine so that it first transforms its input $x$ into $\langle e, x \rangle$ before performing the actual computation, we obtain a Turing machine that computes $\alpha_e$. As a consequence, a numbering $\alpha_0, \alpha_1, \dots$ is universal if and only if all partial computable functions occur among the $\alpha_e$.

</div>

Not every sequence of computable functions constitutes a numbering. A numbering requires that the principal function itself be computable.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(There exist sequences of computable functions that are not numberings)</span></p>

For a proof, first note that there are undecidable sets since there are uncountably many susbets of the natural numbers but only countable many Turing machines. Fix an undecidable set $A$ and consider the sequence $\alpha_0, \alpha_1, \dots$ of constant functions such that $\alpha(e)$ has constant value $0$ if $e$ is not in $A$, and has constant value $1$, otherwise. This sequence is not a numbering, as its principal function $\alpha$ is not partial computable. Otherwise, since $\alpha$ is total, $\alpha$ and then also the function $e \mapsto \alpha(e, e)$ were computable. This is a contradiction since, by construction, the latter function is the characteristic function of the undecidable set $A$.

</div>

A fundamental result is that a universal numbering for all partial computable functions indeed exists.

[Universal Turing Machine](/subpages/computability-and-complexity/universal-turing-machine/)

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Universal Turing Machine theorem)</span></p>

There exists a universal numbering.

</div>

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

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Standard enumeration of Turing machines, Standard enumeration of partial computable functions)</span></p>

Let $M_0, M_1, M_2, \ldots$ be the sequence of Turing machines constructed in the proof of Theorem 171. We call this sequence the **standard enumeration of Turing machines**.

In derived universal numbering for each $i\in\mathbb{N}$, let $\phi_i$ denote the partial function computed by $M_i$ (that is, $\phi_i=\phi_{M_i}$). Then the sequence $\phi_0, \phi_1, \phi_2, \ldots$ is a universal numbering, and it is called the **standard enumeration of partial computable functions**.
</div>

While we can enumerate all partial computable functions, a famous diagonalization argument shows that we cannot do the same for the set of total computable functions.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Numbering of set of computable functions does not exists)</span></p>

There exists no numbering of the set of computable functions.

</div>

*Proof.* Suppose, for a contradiction, that $(\alpha_e)\_{e\in\mathbb N}$ is a numbering whose range consists exactly of all total computable functions. Let $\alpha$ be its principal function, so

$$\alpha(e,x)=\alpha_e(x)\qquad\text{for all }e,x\in\mathbb N.$$

Because every $\alpha_e$ is total, $\alpha$ is total as well. By definition of “numbering”, $\alpha$ is partial computable; since it is total, it is (total) computable.

Define a new function $d:\mathbb N\to\mathbb N$ by

$$d(e)=\alpha(e,e)+1.$$

Since $\alpha$ is computable, so is $d$; moreover $d$ is total.

We claim that $d$ is not equal to any $\alpha_e$. Indeed, fix $e$. Then

$$d(e)=\alpha(e,e)+1=\alpha_e(e)+1\neq \alpha_e(e),$$

so $d$ and $\alpha_e$ differ on input $e$. Therefore $d\neq \alpha_e$ for every $e$, contradicting the assumption that the list $(\alpha_e)$ contains all total computable functions.


**Question:** why cannot we use the same logic to disprove existence of universal numbering?

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

<div class="math-callout math-callout--proposition" markdown="1">
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

*Proof*: Let $\alpha_0, \alpha_1, \dots$ be an arbitrary numbering, and let its principal function $\alpha$ be computed by a Turing machine $M$. So, $M(\langle e, x \rangle) \simeq \alpha_e(x)$. For any given $e$, we can construct a new Turing machine $M'$ that, on input $x$, first transforms $x$ into the pair $\langle e, x \rangle$ and then simulates $M$. This new machine $M'$ computes the function $\alpha_e$. The process of constructing $M'$ from $M$ and $e$ is effective. The standard enumeration of Turing machines is constructed in such a way that we can computably find an index $f(e)$ for this machine $M'$. Thus, there exists a computable function $f$ such that $\phi_{f(e)} = \alpha_e$ for all $e$. $\square$

Not all universal numberings have this powerful translation property.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Universal numbering is not Gödel numbering)</span></p>

There is a universal numbering that is not a Gödel numbering.
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Exists universal numbering that is not Gödel numbering)</span></p>

There exists a universal numbering that is not a Gödel numbering.

</div>
