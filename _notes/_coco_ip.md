## Interactive Proof Systems

[Some additional notes on Interactive Proof Systems](/subpages/computability-and-complexity/interactive-proof-systems/)

### Verifiers and Provers

An **interactive proof system** provides a framework for establishing the truth of a mathematical statement through a structured dialogue. It involves two distinct computational entities: a **prover** and a **verifier**. These parties communicate by exchanging messages regarding a common input string, with the goal of having the verifier decide whether the input belongs to a specific language.
The prover is conceptualized as a computationally unrestricted entity, capable of performing any calculation necessary to persuade the verifier. In contrast, the verifier is a computationally bounded machine—specifically, a probabilistic Turing machine that must operate within polynomial time. The verifier uses randomness to pose challenges to the prover and to evaluate its responses. The messages exchanged are polynomially bounded in the length of the input.

An interactive proof system establishes membership in a language through a dialogue between a computationally bounded verifier and a computationally unbounded prover. The verifier is a probabilistic Turing machine that runs in polynomial time, while the prover can perform arbitrary computation. Messages are polynomially bounded in the input size.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Verifiers and provers)</span></p>

A *message function* maps a tuple $(w, m_1, \ldots, m_i)$ of binary words to a binary word of length at most $p(\|w\|)$ for some fixed polynomial $p$. A *prover* is a message function. A *verifier* is a polynomial-time probabilistic Turing machine $M$ with special accepting and rejecting states that computes a message function depending on its random sequence.

For a verifier $V$ and a prover $P$, the messages exchanged on input $w$ and random word $r$ are $m_1, m_2, \ldots$ with

$$
m_1 = V^r(w), \quad m_2 = P(w, m_1), \quad m_3 = V^r(w, m_1, m_2), \ldots
$$

The interaction terminates at round $t$ if $t$ is minimal such that $V$ reaches either its accepting or rejecting state on computing $V^r(w, m_1, \ldots, m_{2t})$. $V^r(w,m_1,\dots,m_i)$ refers to the message that $V$ computes on input tuple $(w,m_1,\dots,m_i)$ and random word $r$. **Communication** is a finite sequence $m_1,\dots,m_{2t}$.

Let $t : \mathbb{N} \to \mathbb{N}$. A verifier $V$ is **total** and **$t(n)$-bounded** if for all inputs $w$, all random words, and all provers $P$, the interaction terminates by round $t(\|w\|)$.

</div>

Key consequences:

* Message lengths, **number of rounds**, and verifier computation per message are polynomially bounded in $\|w\|$.
* The messages sent by the prover do not depend on the random word of the verifier (the prover cannot see the verifier’s random word); its replies depend only on the input and previous messages.
* Any total verifier is $t(n)$-bounded for some computable $t$ (by bounding the computation tree depth).

> Q: In interactive proof systems, if we say that we have some polynomially bounded verifier $V$ that interactively recognizes a language $L$, does this polynomial bound mean that each interaction (round) of the verifier's execution is polynomially bounded, or is it a global bound for all rounds (the verifier's running time over all rounds is bounded by some polynomial)? 
> A: It is a global bound on the verifier's total running time (as a machine) over the entire interaction $\implies$ the number of rounds is at most polynomial in $n=\|w\|$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Interactive Recognizability)</span></p>

A language $L$ is recognized (interactively) by a total verifier $V$ if

1. For each $w \in L$, there **exists** a prover such that $V$ accepts with probability at least $\tfrac{2}{3}$.
2. For each $w \notin L$, **for all provers**, $V$ rejects with probability at least $\tfrac{2}{3}$.

Let $k \in \mathbb{N}$. The class $\text{IP}[k]$ contains all languages recognized by some $k$-bounded verifier; $\text{IP}$ contains those recognized by some polynomially bounded verifier.

**Completeness** is condition (1); **soundness** is condition (2). The $\tfrac{2}{3}$ bound can be amplified arbitrarily close to $1$ via repetition. It is immediate that $\text{BPP} = \text{IP}[0]$ and $\text{NP} \subseteq \text{IP}[1]$.

</div>

> Proving **completeness** we are allowed (and actually expected) to look for the best possible prover strategy. Proving **soundness** we are expected to consider the most malicious prover.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

If $L$ is recognized by a verifier $V$. Then for each input in $L$ some prover must result in an acceptance probability of at least $\frac{2}{3}$, where this prover *may depend on the input*. It can be shown, however, that there is a single prover $P$ *independent of the input* that yields acceptance probability at least $\tfrac{2}{3}$ for all $w \in L$. The pair $(V, P)$ is an interactive proof system for $L$. Remark 139 below shows that $P$ can be chosen polynomial-space-bounded.

</div>

### An Interactive Proof for Graph Non-Isomorphism

We adopt a fixed node-labeling convention.

Convention 136: All graphs with $n$ nodes have the node set $\lbrace 1, \ldots, n\rbrace$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Graph Isomorphism)</span></p>

Let $G = (V, E)$ and $G' = (V', E')$ be graphs. An *isomorphism* between graphs is a bijection $\pi : V \to V'$ such that $(u, v) \in E$ iff $(\pi(u), \pi(v)) \in E'$. Graphs are *isomorphic* if such a $\pi$ exists.

Languages:

* $\text{GI} = \lbrace (G, G') \mid G \text{ and } G' \text{ are isomorphic}\rbrace$
* $\text{GNI} = \lbrace (G, G') \mid G \text{ and } G' \text{ are nonisomorphic}\rbrace$

</div>

The language $\text{GI}$ is in $\text{NP}$ since for isomorphic graphs, it can be checked in deterministic polynomial time whether a given mapping between their node sets is an isomorphism. The problem $\text{GI}$ is one of the few languages in $\text{NP}$ arising naturally in practice that is neither known to be $\text{NP}$-complete nor known to be in $\text{P}$. The language $\text{GNI}$ is not known to be in $\text{NP}$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name"></span></p>

$\text{GNI} \in \text{IP}[1]$.

</div>

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

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name"></span></p>

$\text{IP} \subseteq \text{PSPACE}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

Let $L$ be a language in $\text{IP}$, and let $V$ be a polynomially bounded verifier that recognizes $L$. Then there is a polynomially space-bounded prover $P$ such that the interactive proof system $(V,P)$ recognizes $L$, i.e., $V$ accepts all inputs in $L$ with probability at least $\frac{3}{2}$ when interacting with $P$. This follows because the optimal prover described in the proof of Theorem 138 makes decisions by computing the function $k(u)$, which we have shown can be done in deterministic polynomial space.

</div>

#### Proving PSPACE is a Subset of IP

We arithmetize logical formulas to design interactive proofs for $\text{PSPACE}$-complete problems.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name"></span></p>

The counting $3$-satisfiability problem is

$$
\#3\text{-SAT} = \lbrace (\phi, k) : \phi \text{ is a 3-CNF formula with exactly } k \text{ satisfying assignments}\rbrace.
$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name"></span></p>

#$3\text{-SAT} \in \text{IP}$.

</div>

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

* If the prover submits a polynomial $h_i^P$ that is truly correct ($h_i^P = h_i$), then $h_i^P(r_i)$ will be the correct value, and the protocol maintains its completeness (acceptance probability $\ge 2/3$).
* If the prover submits an incorrect polynomial ($h_i^P \ne h_i$), the difference between the two polynomials, $(h_i^P - h_i)$, is a non-zero polynomial.

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

$$h_i(x_i) = \sum_{x_{i+1}\in\lbrace 0,1\rbrace} \cdots \sum_{x_n\in\lbrace 0,1\rbrace} p_\phi(r_1, \ldots, r_{i-1}, x_i, x_{i+1}, \ldots, x_n)$$

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

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name"></span></p>

$\text{IP} = \text{PSPACE}$.

</div>

*Proof.* Theorem 138 gives $\text{IP} \subseteq \text{PSPACE}$. For the reverse direction, show a $\text{PSPACE}$-complete language has an interactive proof. QBF in $3$-CNF form ($3$-QBF) is PSPACE-complete.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name"></span></p>

$3\text{-QBF} \in \text{IP}$.

</div>

### Public Coins and Perfect Completeness

Interactive proofs can be made public-coin (all verifier randomness revealed) and perfectly complete (accept with probability $1$ on yes-instances).

Verifiers as defined in Definition 132 use **private coins**, i.e., there random word is not known to the prover. In the **public coin** model, all random bits read by the verifier are sent to the prover. The random bits need not be revealed all at once but may sent in successive rounds. Furthermore, a interactive proof system is said to have completeness 1 or **perfect** completeness, if every input in the recognized language is accepted with probability 1.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

The verifier from Theorem 143 can be modified to a public-coin verifier $V'$ with perfect completeness. Since every $L \in \text{IP}$ reduces to $3$-QBF, every language in $\text{IP}$ has a public-coin, perfectly complete interactive proof.

</div>

### Interactive Proof Systems with the Zero-Knowledge Property

Zero-knowledge proofs allow authentication without revealing secrets: the verifier learns nothing beyond the validity of the statement.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Zero-knowledge IPS)</span></p>

An interactive proof system $(V, P)$ for $L$ is *zero-knowledge* if for every verifier $V'$ there is a probabilistic Turing machine $S$ (the simulator) running in expected polynomial time such that for every $w \in L$, the output distribution of $S$ equals the distribution of transcripts from the interaction of $V'$ with $P$ on input $w$.

</div>

The simulator shows that any transcript could have been generated without access to the prover’s secret, so no knowledge is leaked (perfect zero-knowledge).

$\text{GI}$, despite lacking a known $\text{NP}$ certificate, admits a zero-knowledge interactive proof.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name"></span></p>

$\text{GI}$ has a zero-knowledge interactive proof system.

</div>

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
