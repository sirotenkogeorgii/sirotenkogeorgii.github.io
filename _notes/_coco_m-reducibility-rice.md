## Many-one Reducibility and Rice’s Theorem

To compare the difficulty of undecidable problems, we introduce the concept of reducibility. One of the most fundamental types is many-one reducibility.

### Many-one Reducibility

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Many-one reduction)</span></p>

A set $A$ is **many-one reducible** to a set $B$ (also: **m-reducible**), written $A \le_m B$, if there exists a computable function $f$ such that for all $x \in \mathbb{N}$,

$$x \in A \iff f(x) \in B.$$

</div>

The function $f$ transforms instances of problem $A$ into instances of problem $B$ while preserving the yes/no answer. This means that if we can solve $B$, we can use $f$ to solve $A$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

Let set $A$ be decidable and set $B$ be distinct from $\emptyset$ and $\mathbb{N}$. Then $A$ is m-reducible to $B$. In order to obtain a function $f$ that witnesses the latter, fix $b_0 \notin B$ and $b_1 \in B$, and let  
 
$$f(x) = \begin{cases} b_0, & \text{if } x \notin A, \\ b_1, & \text{if } x \in A. \end{cases}$$

On the other hand, there is just a single set that is m-reducible to $\emptyset$ since $A \le_m \emptyset$ implies $A = \emptyset$, and a similar statement holds for $\mathbb{N}$ in place of $\emptyset$. This is considered to be an anomaly and, accordingly, m-reducibility is sometimes defined such that, in addition to the relationships valid according to Definition 180, all decidable sets $A$ are m-reducible to $\emptyset$ and $\mathbb{N}$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(m-reductions preserve decidability)</span></p>

The relation m-reducibility is **reflexive** and **transitive**.

</div>

### M-reducibility allows us to infer properties about sets.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(m-reductions preserve decidability/undecidability)</span></p>

Let $A$ and $B$ be sets such that $A \le_m B$. 
* **(i)** If $B$ is decidable, then $A$ is also decidable. 
* **(ii)** If $A$ is undecidable, then $B$ is also undecidable.

</div>

A key result is that the halting problem $H$ is a "hardest" problem among all r.e. sets with respect to m-reducibility.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(r.e. sets and many-one reductions to the halting problem)</span></p>

Let $A$ and $B$ be sets. 
* **(i)** If $A \le_m B$ and $B$ is r.e., then $A$ is also r.e. 
* **(ii)** For every r.e. set $A$, it holds that $A \le_m H$.

</div>

**Proof**: 

* **(i)** Let $A \le_m B$ via a computable function $f$, and let $B$ be r.e. Then $B$ is the domain of some partial computable function $\alpha$. The set $A$ is the domain of the partial computable function $\alpha \circ f : x \mapsto \alpha(f(x))$. Since $A$ is the domain of a partial computable function, it is r.e. 
* **(ii)** Let $A$ be an r.e. set. Then $A = \text{dom}(\alpha)$ for some partial computable function $\alpha$. Let $\phi_0, \phi_1, \dots$ be the standard numbering. Since it is a Gödel numbering, we can effectively construct a new numbering $\beta_0, \beta_1, \dots$ where for a given $e$, the function $\beta_e$ ignores its own input and simply computes $\alpha(e)$. That is, $\beta_e(x) \simeq \alpha(e)$. The principal function $(e,x) \mapsto \beta_e(x)$ is partial computable, so this is a valid numbering. Since the standard numbering is a Gödel numbering, there exists a computable function $f$ such that $\beta_e = \phi_{f(e)}$ for all $e$. This function $f$ witnesses that 
  
  $$A \le_m H:  e \in A \iff \alpha(e) \downarrow \iff \beta_e \text{ is total} \iff \beta_e(f(e)) \downarrow \iff \phi_{f(e)}(f(e)) \downarrow \iff f(e) \in H$$

This theorem provides a complete characterization of recursively enumerable sets in terms of m-reducibility to the halting problem.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(name of Corollary)</span></p>

A set $A$ is r.e. if and only if $A \le_m H$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(name of Corollary)</span></p>

It holds that $H \le_m H_{gen}$ and $H_{gen} \le_m H$.

</div>

**Proof**:

* The reduction $H \le_m H_{gen}$ is witnessed by the computable function $f(e) = \langle e, e \rangle$. We have 
  
  $$e \in H \iff \phi_e(e) \downarrow \iff \langle e, e \rangle \in H_{gen}$$

* The reduction $H_{gen} \le_m H$ follows directly from Corollary above, because $H_{gen}$ is a recursively enumerable set. $\square$

M-reducibility also interacts cleanly with set complements. Let $\bar{X} = \mathbb{N} \setminus X$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

If $A \le_m B$ via a function $f$, then also $\bar{A} \le_m \bar{B}$ via $f$ because we have 
 
$$x \in \bar{A} \iff x \notin A \iff f(x) \notin B \iff f(x) \in \bar{B}$$ 

</div>

This leads to an important non-reducibility result.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(name of Corollary)</span></p>

Let $A$ be an undecidable r.e. set. Then

$$A \not\le_m \overline{A}\quad\text{and}\quad\overline{A} \not\le_m A$$

In particular, we have

$$H \not\le_m \bar{H} \text{ and } \bar{H} \not\le_m H$$

</div>

**Proof**: Proof. The set $\hat{A}$ cannot be r.e. because in this case A would be decidable. By Theorem above, then $\hat{A}$ cannot be $m$-reducible to $A$, hence also $A$ cannot be m-reducible to $\hat{A}$ by Remark above. $\square$

### Index Sets and Rice’s Theorem

Rice's Theorem is a powerful generalization of the undecidability of the halting problem. It states that any non-trivial property of the behavior of Turing machines (i.e., of the partial computable functions they compute) is undecidable. To formalize this, we use the notion of an index set.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Index Set)</span></p>

A set $I \subseteq \mathbb{N}$ is an **index set** if for all $e,e'$,

$$e \in I \ \land \ \phi_e=\phi_{e'} \ \Longrightarrow\ e' \in I$$

An index set is **nontrivial** if it is neither $\emptyset$ nor $\mathbb{N}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Expanded definition of Index set)</span></p>

Let $\varphi_e$ be a computable enumeration of all partial computable functions, and let $W_e$ be a computable enumeration of all computably enumerable (c.e.) sets.

Let $\mathcal{A}$ be a collection of partial computable functions. Define its **index set** by

$$A=\lbrace x:\varphi_x\in\mathcal{A}\rbrace$$

More generally, a set $A\subseteq\mathbb{N}$ is an **index set** if whenever $x,y\in\mathbb{N}$ satisfy $\varphi_x\simeq \varphi_y$ (i.e., they compute the same partial function), we have

$$x\in A \iff y\in A$$

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

$$g(e,k,x) = \varphi_e(x).$$

This $g$ is partial computable (it just simulates $e$ on input $x$, ignoring $k$). By **s-m-n**, there is a total computable function $s(e,k)$ such that

$$\varphi_{s(e,k)}(x) = g(e,k,x) = \varphi_e(x).$$

Now set $p(e,k)=s(e,k)$. That’s the padding function.

(If you want injectivity in $k$, you can build it in by first pairing $(e,k)$ into one number in a reversible computable way.)


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

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Rice's Theorem)</span></p>

Let $I$ be a nontrivial index set. Then it holds that 

$$H \le_m I \quad \text{or} \quad \bar{H} \le_m I$$

</div>

**Proof of Rice's Theorem 1**: Let $I$ be a nontrivial index set. Let $\phi_\uparrow$ denote the everywhere undefined partial function. There are two cases for $I$.
* **Case 1**: $I$ does not contain an index for $\phi_\uparrow$. Since $I$ is nontrivial, it is not empty, so there must be some index $e_0 \in I$. Let $\beta = \phi_{e_0}$. By our assumption, $\beta$ is not the everywhere undefined function. We will show $\bar{H} \le_m I$. To do this, we need a computable function $g$ such that 
  
  $$e \in \bar{H} \iff g(e) \in I$$

* Consider the function $g(e)$ which gives the index of a new Turing machine. This machine, on any input $x$, first simulates the computation of $\phi_e(e)$. If this simulation halts, the machine then proceeds to compute $\beta(x)$. If the simulation of $\phi_e(e)$ does not halt, the machine runs forever. The function computed by the machine with index $g(e)$ is:  
  
  $$\phi_{g(e)} = \begin{cases} \beta & \text{if } \phi_e(e)\uparrow \text{ (i.e., } e \in \bar{H}), \\ \phi_\uparrow & \text{if } \phi_e(e)\downarrow \text{ (i.e., } e \in H). \end{cases}$$

  The function $g$ is computable. Now we check the reduction:
  * If $e \in \bar{H}$, then $\phi_{g(e)} = \beta = \phi_{e_0}$. Since $e_0 \in I$ and $I$ is an index set, $g(e)$ must be in $I$.
  * If $e \in H$, then $\phi_{g(e)} = \phi_\uparrow$. By our case assumption, no index for $\phi_\uparrow$ is in $I$, so $g(e) \notin I$. Thus, $e \in \bar{H} \iff g(e) \in I$, so $\bar{H} \le_m I$.
* **Case 2**: I contains an index for $\phi_\uparrow$. In this case, the complement set $\bar{I}$ does not contain an index for $\phi_\uparrow$. Since $I$ is nontrivial, $\bar{I}$ is also a nontrivial index set. Applying the logic from Case 1 to $\bar{I}$, we find that $\bar{H} \le_m \bar{I}$. By Remark 187, this implies $H \le_m I$.

**Proof of Rice's Theorem 2** Let $\phi_\uparrow$ denote the everywhere undefined partial computable function. Since $I$ is nontrivial, $I\neq\emptyset$ and $I\neq\mathbb N$. We distinguish two cases depending on whether $I$ contains an index of $\phi_\uparrow$.

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
* If $e\in\overline{H}$, then $\phi_{g(e)}=\beta=\phi_{e_0}$. 
  * Since $e_0\in I$ and $I$ is an index set (membership depends only on the computed function), it follows that $g(e)\in I$.
* If $e\in H$, then $\phi_{g(e)}=\phi_\uparrow$. 
  * By the case assumption, no index of $\phi_\uparrow$ lies in $I$, hence $g(e)\notin I$.

Therefore $e\in\overline{H}\iff g(e)\in I$, so $\overline{H}\le_m I$.

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


<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(193)</span></p>

Nontrivial index sets are not decidable.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(name of Corollary)</span></p>

For every partial computable function $\alpha$ the set $\lbrace e : \alpha = \phi_e\rbrace$ is infinite.

</div>

**Proof**: If this set were finite for some $\alpha$, it would be a decidable set. It is also nontrivial (it's not $\emptyset$ or $\mathbb{N}$) and is an index set by definition. This would contradict Corollary 193.

Not all properties related to Turing machine indices are index sets. For a property to be an index set, it must depend only on the function computed, not the index itself.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(name of Example)</span></p>

Let

$$A=\lbrace e : \operatorname{dom}(\phi_e)\ \text{has more than } e \text{ elements}\rbrace$$

Then $A$ is **not** an index set.

Moreover, $A$ is r.e.: given $e$, we can effectively enumerate $\operatorname{dom}(\phi_e)$ until at least $e+1$ distinct elements have appeared. This can be done via **dovetailing**: for stages $s=0,1,2,\dots$, simulate the computations $\phi_e(x)$ for all $x\le s$, each for $s$ steps (see the exercises for details).

To show that $A$ is not an index set, distinguish two cases.

* **Case I.** There exists $e\in A$ such that $\operatorname{dom}(\phi_e)$ is finite of size $m$. By Corollary 194, there is an index $e'>m$ with $\phi_{e'}=\phi_e$. But then $e'\notin A$, so membership in $A$ is not preserved under equality of computed functions. Hence $A$ is not an index set.

* **Case II.** For every $e\in A$, the domain $\operatorname{dom}(\phi_e)$ is infinite. Then, by definition, $A$ contains every index $e$ such that $W_e$ is infinite. This would give an effective listing of partial computable functions with infinite domain, contradicting Remark 195: if $e_0,e_1,\dots$ is an effective enumeration of $A$, then $\phi_{e_0},\phi_{e_1},\dots$ would be such a numbering.
  
</div>
