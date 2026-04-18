## Oracle Turing Machines and the Jump Operator

We now introduce a more powerful form of reducibility, Turing reducibility, based on a model of computation called an oracle Turing machine. This model allows us to ask "what if we had a magical black box that could solve problem $B$?" and then explore what other problems ($A$) we could solve with its help.

### Oracle Turing Machines and Turing Reducibility

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Oracle Turing Machine)</span></p>

An **oracle Turing machine** is a Turing machine that, in addition to its work tapes, has a special **oracle tape**. For a set $B$, the oracle tape contains the infinite binary sequence

$$B(0)B(1)B(2)\dots$$

(the values of the characteristic function of $B$). The set $B$ is called the **oracle**. The oracle tape is read-only, and initially the head scans the cell containing $B(0)$.

For an oracle machine $M$, input $x$, and oracle $B$, write:

* $M(x,B)$ for the output if the computation halts,
* $M(x,B)\uparrow$ if it does not halt,
* $M(x,B)\downarrow$ and $M(x,B)\downarrow = y$ with the usual meanings.

We also write $M^B(x)$ instead of $M(x,B)$.

If $M$ halts on every input $x$ when using oracle $B$, then $M(B)$ denotes the (unique) set $A$ whose characteristic function $c_A: x \mapsto M(x,B)$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Turing-reducibility)</span></p>

A set $A$ is **Turing-reducible** to $B$, written $A \le_T B$, if there exists an oracle Turing machine $M$ such that $A=M(B)$, i.e.

$$A(x)=M(x,B)\quad\text{for all }x\in\mathbb{N}$$

</div>

Turing reducibility is a more general notion than many-one reducibility.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(name of Theorem)</span></p>

For all sets $A,B$, if $A \le_m B$ then $A \le_T B$. In general, the converse implication fails.

</div>

**Proof**:

* The implication 
  
  $$A \le_m B \implies A \le_T B$$
  
  holds because if $f$ is the computable function for the m-reduction, an oracle Turing machine can compute $f(x)$ and then query the oracle for $B$ at the single location $f(x)$ to get the answer.
* A counterexample for the reverse is given by the halting problem $H$ and its complement $\bar{H}$. For any set $A$, we have $A \le_T \bar{A}$, because an oracle for $\bar{A}$ can be used to decide membership in $A$ and vice versa. Thus, $H \le_T \bar{H}$. However, as shown in Corollary 188, we have $H \not\le_m \bar{H}$. $\square$

### Like m-reducibility, T-reducibility preserves decidability.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(name of Proposition)</span></p>

Let $A$ and $B$ be sets where $A \le_T B$. 
* **(i)** If $B$ is decidable, then $A$ is also decidable. 
* **(ii)** If $A$ is undecidable, then $B$ is also undecidable.
  
</div>

### Relative Computations and the Jump Operator

The concept of an oracle allows us to "relativize" all the fundamental notions of computability. We can speak of sets being decidable or r.e. relative to an oracle.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Relative decidability, Relative r.e.)</span></p>

Let $A,B \subseteq \mathbb{N}$.

* $A$ is **decidable relative to** (or **with oracle**) $B$ if $A \le_T B$.
* $A$ is **recursively enumerable relative to** $B$ if there exists an oracle Turing machine $M$ such that
  
  $$A=\lbrace x\in\mathbb{N} : M(x,B)\downarrow\rbrace$$

</div>

If $A$ is decidable with oracle $B$, we also say $A$ is decidable in $B$ or decidable relative to $B$. The same terminology applies to r.e. sets. Most theorems from standard computability theory have direct analogues in this relativised world.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Computable with oracle, Effective enumeration with oracle)</span></p>

A function $f$ is **computable with oracle** $B$ if there exists an oracle Turing machine that, on input $i$ and oracle $B$, outputs $f(i)$.

An enumeration $x_0,x_1,\dots$ is **effective in** $B$ if $x_i=f(i)$ for some function $f$ computable with oracle $B$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

Similar to the unrelativized case, it can be shown that a set $A$ is recursively enumerable with oracle $B$ if and only if there exists an enumeration $x_0, x_1, \dots$ that is effective in $B$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(name of theorem)</span></p>

Let $A,B$ be sets.

1. $A$ is decidable in $B$ $\iff$ both $A$ and $\overline{A}$ are r.e. in $B$.
2. $A$ is decidable in $B$ $\iff$ $A$ has a monotone enumeration $x_0 \le x_1 \le x_2 \le \dots$ that is effective in $B$.

</div>

Just as we can create a standard numbering of all Turing machines, we can create a standard numbering of all oracle Turing machines, which is also a Gödel numbering in the relativized sense. This allows us to define a relativized version of the halting problem.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

Similar to the construction for Turing machines, we obtain a standard enumeration $M_0, M_1, \dots$ of oracle Turing machines. First, this is an effective enumeration of Turing machines in the sense that there exists an oracle Turing machine $M$ such that for all $e$, $x$, and all oracles $X$ it holds that

$$M(\langle e, x\rangle, X) \simeq M_e(x, X)$$

Second, similar to a Gödel numbering, for every sequence $O_0, O_1, \dots$ of oracle Turing machines such that for some oracle Turing machine $O$ it holds that

$$O(\langle e, x\rangle, X) \simeq O_e(x, X)$$

there exists a computable function $f$ such that for all $e$, $x$, and all oracles $X$ it holds that

$$M_{f(e)}(x, X) \simeq O_e(x, X)$$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(name of definition)</span></p>

Let $B\subseteq \mathbb{N}$, and let $M_0,M_1,\dots$ be the standard enumeration of all oracle Turing machines. Define

$$H^B=\lbrace e : M_e(e,B)\downarrow\rbrace$$

This set is called the **(diagonal) halting problem relative to $B$**, or the **jump** of $B$. The map $X \mapsto H^X$ is the **jump operator**.

</div>

The jump of $B$ is often denoted $B'$. The set $H^B$ plays the same role for sets that are r.e. in $B$ as the original halting problem $H$ does for r.e. sets.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(name of theorem)</span></p>

For all sets $A,B$, the set $A$ is r.e. relative to $B$ $\iff$ $A \le_m H^B$.

</div>

**Proof**: The proof is a direct relativization of the proof of Theorem 184 and Corollary 185.

* $(\impliedby)$ If $A \le_m H^B$ via computable function $f$, we can build an oracle TM for $A$. On input $x$ with oracle $B$, it computes $f(x)$ and then simulates the oracle TM $M_{f(x)}$ on input $f(x)$ with oracle $B$. This machine halts iff $f(x) \in H^B$, which happens iff $x \in A$. Thus, $A$ is r.e. in $B$.
* $(\implies)$ If $A$ is r.e. in $B$, there is an oracle TM $M$ such that $A$ is its domain with oracle $B$. We can create a computable function $h$ such that for any $x, M_{h(x)}$ is an oracle TM that on any input $y$ with any oracle $X$ simulates $M$ on input $x$ with oracle $X$. The function $h$ is computable as it just hard-wires $x$ into the description of $M$. Then we have:  
 
$$x \in A \iff M(x, B) \downarrow \iff M_{h(x)}(h(x), B) \downarrow \iff h(x) \in H^B$$

This shows $A \le_m H^B$ via the computable function $h$. $\square$

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(name of theorem)</span></p>

For every set $B$, the halting problem $H^B$ is r.e. relative to $B$.

</div>

The jump operator always produces a strictly more complex set. A set is always reducible to its own jump, but the jump is never reducible back to the original set.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(name of theorem)</span></p>

Let $B$ be a set. Then $B \le_m H^B$, so $B$ is decidable relative to $H^B$. However, $H^B$ is not decidable relative to $B$; that is,

$$B \le_m H^B \quad\text{and}\quad H^B \not\le_T B$$

</div>

**Proof**:

* To show $B \le_m H^B$, we need a computable function $h$. Let $h(x)$ be the index of an oracle TM $M_{h(x)}$ that, on any input $y$ and with any oracle $X$, halts if and only if $x \in X$. Then 
  
  $$x \in B \iff M_{h(x)}(h(x), B) \downarrow \iff h(x) \in H^B$$
  
  Thus $B \le_m H^B$.
* The proof that $H^B \not\le_T B$ is a relativized diagonalization argument, identical to the proof that the original halting problem is undecidable. If $H^B$ were decidable in $B$, then the function $g(e) = 1 + M_e(e, B)$ (if $e \in H^B$) and $0$ (otherwise) would be computable in $B$. But this function cannot be computed by any oracle TM $M_e$ with oracle $B$, as it differs from the function computed by $M_e$ on input $e$. This is a contradiction. $\square$

Finally, the jump operator is monotone with respect to Turing reducibility. If $A$ is no harder than $B$, then the jump of $A$ is no harder than the jump of $B$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(name of theorem)</span></p>

If $A \le_T B$, then the halting problem relative to $A$ is many-one reducible to the halting problem relative to $B$; equivalently,

$$A \le_T B \ \Longrightarrow\ H^A \le_m H^B$$

</div>

**Proof**: Let $A \le_T B$ via an oracle TM $M_{oracle}$. We want to show $H^A \le_m H^B$. We need a computable function $h$ such that $e \in H^A \iff h(e) \in H^B$. The function $h(e)$ produces the index of a new oracle TM $M_{h(e)}$. This machine, on input $y$ with oracle $X$, simulates the oracle TM $M_e$ on input $e$. Whenever $M_e$ makes an oracle query for some string $z$ (to what it thinks is oracle $A$), $M_{h(e)}$ pauses and uses its own oracle $X$ to simulate $M_{oracle}$ on input $z$ to get the answer. It then provides this answer back to the simulation of $M_e$. So,$ M_{h(e)}$ with oracle $X$ simulates $M_e$ with oracle $M_{oracle}(X)$. This means 

$$M_e(e, A) \downarrow \iff M_e(e, M_{oracle}(B)) \downarrow \iff M_{h(e)}(h(e), B) \downarrow$$

Therefore, we have the desired equivalence: 

$$e \in H^A \iff M_e(e, A) \downarrow \iff M_{h(e)}(h(e), B) \downarrow \iff h(e) \in H^B$$

The function $h$ is computable, so this establishes $H^A \le_m H^B$. $\square$

Here are **clean, readable notes** on the pages, with the proofs rewritten more transparently.

### Arithmetic hierarchy, jumps, incompleteness, and oracle results

#### The jump operator and the sets $\emptyset^{[n]}$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Jump operator)</span></p>

We start with the empty set and repeatedly apply the **jump operator**:

$$X \mapsto H^X$$

where $H^X$ is the **halting problem relative to oracle $X$**.

</div>

So:
* $\emptyset^{[0]} = \emptyset$
* $\emptyset^{[1]} = \emptyset' = H^\emptyset$
* $\emptyset^{[2]} = (\emptyset')' = H^{\emptyset'}$
* and so on.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($n$-th jump of the empty set)</span></p>

$\emptyset^{[n]}$ is called the **$n$-th jump of the empty set**.

</div>


<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Intuition)</span></p>

Each jump makes the set strictly more complicated:

* $\emptyset$: trivial
* $\emptyset'$: ordinary halting problem
* $\emptyset''$: halting problem **with access to the halting problem**
* etc.

So the jumps form a natural hierarchy of increasing undecidability.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Each jump is enumerable in the previous one, but not decidable there)</span></p>

For all $n \ge 0$, the set $\emptyset^{[n+1]}$ is:
* **r.e. in** $\emptyset^{[n]}$, but
* **not decidable with oracle** $\emptyset^{[n]}$.

**Meaning:** If you are allowed to use $\emptyset^{[n]}$ as an oracle, then you can **enumerate** $\emptyset^{[n+1]}$, but you still cannot fully decide it.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Intuition)</span></p>

This is just the relativized halting problem fact:
* the halting problem relative to oracle $B$ is enumerable in $B$,
* but not decidable in $B$.

Take $B=\emptyset^{[n]}$.

**Important takeaway:** Each jump really is **strictly harder** than the previous one.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Arithmetic hierarchy $\Sigma_n$)</span></p>

A set $A \subseteq \mathbb N$ is called a **$\Sigma_n$ set** if there exists a decidable relation $R$ such that

$$x \in A \iff \exists y_1 \forall y_2 \cdots Q y_n; R(x,y_1,\dots,y_n)$$

where the quantifiers alternate, and:
* if $n$ is odd, the last quantifier is existential,
* if $n$ is even, the last quantifier is universal.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">($\Sigma_1$, $\Sigma_2$, $\Sigma_3$)</span></p>

* $\Sigma_1$: $\exists y, R(x,y)$
* $\Sigma_2$: $\exists y_1 \forall y_2, R(x,y_1,y_2)$
* $\Sigma_3$: $\exists y_1 \forall y_2 \exists y_3, R(\dots)$

with $R$ decidable.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Arithmetic sets)</span></p>

A set is called **arithmetical** if it belongs to some $\Sigma_n$.

The collection of all such levels is the **arithmetical hierarchy**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Key idea)</span></p>

The more alternations of quantifiers you need, the more complicated the set is.

Also, every $\Sigma_n$ set is automatically a $\Sigma_{n+1}$ set, because you can add a dummy quantifier. So the levels are nested.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Post’s Theorem)</span></p>

For all $n \ge 0$ and all sets $A$, the following are equivalent:
1. $A$ is a $\Sigma_{n+1}$ set.
2. $A$ is many-one reducible to $\emptyset^{[n+1]}$.
3. $A$ is r.e. in $\emptyset^{[n]}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why this is important)</span></p>

This theorem connects **three different ways** of measuring complexity:

* logical complexity: number of quantifier alternations,
* computability complexity: reducibility to jumps,
* oracle computability: being enumerable relative to an oracle.

So the jumps $\emptyset^{[n]}$ are exactly the canonical complete problems for the levels of the arithmetic hierarchy.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Consequence)</span></p>

The arithmetic hierarchy is **proper**:

$$\Sigma_n \subsetneq \Sigma_{n+1}$$

Why? Because $\emptyset^{[n+1]}$ belongs to level $\Sigma_{n+1}$, but by Lemma 234 it is not decidable from $\emptyset^{[n]}$, so it cannot already lie at a lower level.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(The set of true arithmetic sentences is not arithmetical)</span></p>

Let $T$ be the set of Gödel numbers of all **true arithmetic sentences**.

$T$ is **not arithmetical**.

This means $T$ is not in any finite level $\Sigma_n$ of the arithmetic hierarchy.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(What this means conceptually)</span></p>

No matter how many alternating number quantifiers you allow, that is still not enough to capture **all true arithmetic**.

So truth in arithmetic is more complicated than every finite arithmetical level.

</div>

**Proof:**

We prove by contradiction.

#### Step 1: assume $T$ is arithmetical

Then $T \in \Sigma_n$ for some $n$.

By Post’s theorem, this means:

$$T \le_m \emptyset^{[n]}$$

So there is a computable function $f$ such that

$$\varphi \in T \iff f(\varphi) \in \emptyset^{[n]}$$

In words: checking whether an arithmetic sentence is true can be reduced to membership in $\emptyset^{[n]}$.

#### Step 2: use arithmetization

Because $\emptyset^{[n+1]}$ is an arithmetical set, there is an arithmetic formula $\psi(x)$ such that

$$e \in \emptyset^{[n+1]} \iff \psi(\bar e)\text{ is true}$$

Here $\bar e$ is the numeral naming $e$.

This is the key bridge:

* numbers can encode formulas,
* arithmetic can talk about computations and sets,
* truth of a suitable formula can express membership in $\emptyset^{[n+1]}$.

#### Step 3: reduce $\emptyset^{[n+1]}$ to $T$

From the formula $\psi(\bar e)$, we get:

$$e \in \emptyset^{[n+1]} \iff \psi(\bar e)\in T$$

Because $T \le_m \emptyset^{[n]}$, we then obtain:

$$e \in \emptyset^{[n+1]} \iff f(\psi(\bar e)) \in \emptyset^{[n]}$$

So $\emptyset^{[n+1]}$ would be many-one reducible to $\emptyset^{[n]}$.

#### Step 4: contradiction

But this is impossible.

Why? Because $\emptyset^{[n+1]} = H^{\emptyset^{[n]}}$, the halting problem relative to $\emptyset^{[n]}$, and a halting problem relative to an oracle cannot be decidable from that same oracle.

Yet a many-one reduction to $\emptyset^{[n]}$ would be even stronger than oracle decidability.

Contradiction.

Therefore $T$ is **not arithmetical**.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The real core idea)</span></p>

If truth $T$ were arithmetical, then it would be too weak: it would sit at some finite level $n$. But arithmetic truth is powerful enough to express membership in **all** those finite jumps, and in particular in $\emptyset^{[n+1]}$. That would collapse the jump hierarchy, which cannot happen.

</div>

### Digression: oracle complexity classes

Now the text switches from computability/arithmetic hierarchy to **complexity theory with oracles**.

For an oracle $B$:

* $P^B$: polynomial-time deterministic machines with oracle $B$
* $NP^B$: polynomial-time nondeterministic machines with oracle $B$

The question is whether $P^B = NP^B$ or not.

The surprising point is: **both can happen, depending on the oracle**.

So oracle arguments alone cannot settle the ordinary $P$ vs $NP$ problem.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(There is an oracle $B$ with $P^B = NP^B$)</span></p>

There exists an oracle $B$ such that

$$P^B = NP^B$$

The proof chooses $B$ to be a language that is polynomial many-one complete for **PSPACE**, for example **QBF**.

</div>

**Proof:**

Take $B$ to be a PSPACE-complete language.

Then:

$$P^B \subseteq NP^B \subseteq PSPACE^B \subseteq PSPACE \subseteq P^B$$

Since we have inclusions from left to right and back to $P^B$, all classes are equal.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why each inclusion holds)</span></p>

**1. $P^B \subseteq NP^B$**

Trivial: deterministic computation is a special case of nondeterministic computation.

**2. $NP^B \subseteq PSPACE^B$**

Also standard: polynomial-time nondeterminism can be simulated in polynomial space.

**3. $PSPACE^B \subseteq PSPACE$**

This is the first nontrivial one.

Why? Because $B$ itself is in PSPACE.

So whenever a PSPACE machine with oracle $B$ wants to ask “is $x \in B$?”, it can just compute the answer itself in polynomial space.

Thus the oracle adds no extra power beyond PSPACE.

**4. $PSPACE \subseteq P^B$**

Because $B$ is PSPACE-complete.

Any PSPACE problem can be reduced in polynomial time to $B$, so a polynomial-time machine with oracle $B$ can solve all PSPACE problems.

**Conclusion**

$$P^B = NP^B = PSPACE^B = PSPACE$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why this matters)</span></p>

An oracle can make $P$ and $NP$ collapse.

So if someone tried to prove $P \neq NP$ using a method that still works relative to **every** oracle, that proof would fail.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(There is a decidable oracle $B$ with $P^B \ne NP^B$)</span></p>

There is even a **decidable** oracle $B$ such that

$$P^B \ne NP^B$$

This is the opposite of Theorem 238.

So some oracles make the classes equal, and some make them different.

</div>

#### Main construction in Theorem 239

We build two languages $A$ and $B$ such that:
* $A \in NP^B$
* $A \notin P^B$

The language $A$ is defined by:

$$A = \lbrace 0^n : \text{there exists some word of length } n \text{ in } B\rbrace$$

So to decide whether $0^n \in A$, you only need to know whether **at least one** string of length $n$ lies in $B$.

#### Why $A \in NP^B$

Given input $0^n$, an $\text{NP}$ machine can:
1. nondeterministically guess a word $y$ of length $n$,
2. ask the oracle whether $y \in B$,
3. accept if yes.

This is polynomial time.

So:

$$A \in NP^B$$

That part is easy.

####  How the diagonalization against $P^B$ works

We enumerate all deterministic oracle Turing machines:

$$M_0, M_1, M_2, \dots$$

Every polynomial-time deterministic oracle machine appears somewhere in this list.

We now want to make sure that **each such machine fails on at least one input**.

#### The rapidly growing lengths

Define:
* $n_0 = 1$
* $n_{i+1} = 2^{n_i}$

These lengths grow *extremely fast*.

#### Why this helps

At stage $i$, we only decide what happens on strings of lengths in the interval

$$n_i,\dots,n_{i+1}-1$$

So when we simulate a machine on input $0^{n_i}$, the information about shorter lengths is already fixed, and the much larger future lengths are still untouched.

This separation prevents later stages from messing up earlier diagonalization.

#### What happens at stage $i=\langle e,s\rangle$

We pair numbers $e,s$ into one stage index $i$.
So each machine $M_e$ gets attacked infinitely many times, once for each $s$.

At stage $i$:

* simulate $M_e$ on input $0^{n_i}$,
* using the current partial oracle $B$,
* for at most $2^{n_i}-1$ steps.

If the simulation does not finish in that time, call it **incomplete**.

Also note:

* in at most $2^{n_i}-1$ steps,
* the machine can make at most $2^{n_i}-1$ oracle queries.

So it can ask about **fewer than $2^{n_i}$** strings of length $n_i$.

But there are exactly $2^{n_i}$ strings of length $n_i$.

Therefore, at least one length-$n_i$ string is left unqueried.

That is the crucial counting trick.

#### The two cases at stage $i$

#### Case 1: $M_e$ accepts $0^{n_i}$, or simulation is incomplete

Do nothing.

Leave $A$ and $B$ unchanged at this stage.

#### Case 2: $M_e$ rejects $0^{n_i}$

Now we force $M_e$ to be wrong.

We do two things:

1. Put $0^{n_i}$ into $A$.
2. Put into $B$ some string of length $n_i$ that was **not queried** during the simulation.

Because that string was never queried, the machine could not have known this would happen.

After this modification:

* $0^{n_i} \in A$, because now $B$ contains a word of length $n_i$,
* but $M_e$ had rejected $0^{n_i}$.

So $M_e$ is wrong on that input.

#### Why every polynomial-time deterministic oracle machine fails

Take any deterministic oracle machine $M$ running in time $p(n)$, where $p$ is a polynomial.

Since the list $M_0,M_1,\dots$ is an effective enumeration, there is some $e$ such that $M=M_e$.

Now look at stages $i=\langle e,s\rangle$ with very large $s$.

Because $n_i$ grows so fast, eventually:

$$p(n_i) < 2^{n_i}$$

So for large enough $i$, the real computation of $M$ on $0^{n_i}$ finishes within our simulation limit $2^{n_i}-1$. Hence the stage-$i$ simulation is accurate.

At that stage, the construction guarantees:

$$A(0^{n_i}) \ne M(0^{n_i},B)$$

Therefore $M$ does **not** decide $A$.

Since $M$ was an arbitrary polynomial-time deterministic oracle machine, no machine in $P^B$ decides $A$.

Thus:

$$A \notin P^B$$

Combined with $A \in NP^B$, we get

$$P^B \ne NP^B$$

#### 14. Why the oracle $B$ is decidable

The theorem says $B$ can be chosen decidable, but the proof sketch does not explain this very explicitly. Here is the missing explanation.

At stage $i$, we only decide membership for strings whose lengths lie in:

$$[n_i, n_{i+1}-1]$$

So every string length is settled at some finite stage.

Given a string $x$ of length $m$:

1. find the unique $i$ such that
   
   $$n_i \le m < n_{i+1}$$

2. simulate the construction through stages $0,1,\dots,i$,
3. check whether $x$ was inserted into $B$.

This is a finite computation, so membership in $B$ is decidable.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Important note)</span></p>

This does **not** mean $B$ is easy to decide.
It only means it is decidable in principle.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Big-picture summary)</span></p>

**Arithmetic hierarchy side**

* The jumps $\emptyset^{[n]}$ form a strict hierarchy of harder and harder sets.
* Post’s theorem connects:
  * logical definability $(\Sigma_n)$,
  * reducibility to jumps,
  * oracle enumerability.
* The set $T$ of all true arithmetic sentences is **beyond the whole arithmetical hierarchy**.

**Complexity/oracle side**

* Relative to some oracle $B$, we can have:
  
  $$P^B = NP^B$$
  
* Relative to another oracle $B$, we can have:
  
  $$P^B \ne NP^B$$
  
* Therefore oracle methods alone cannot resolve the unrelativized $P$ vs $NP$ question.

</div>
