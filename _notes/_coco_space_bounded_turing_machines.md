## Space-Bounded Computation

While time complexity measures the number of steps a Turing machine takes, space complexity measures the amount of memory (tape cells) it uses. For space-bounded computations, especially those with sublinear bounds (e.g., $\log n$), it is essential to distinguish the memory used for input and output from the memory used for computation (work tapes). This leads to the model of an off-line Turing machine.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Off-line Turing Machine)</span></p>

An **off-line Turing machine** is a Turing machine with a separate input and output tape.

  * The **input tape** is read-only. The head can only access cells containing the input symbols and the two adjacent blank cells.
  * On the **output tape**, the head can only move to the right. Whenever a symbol is written, the head must advance one position to the right.
  * A Turing machine is called a **$k$-tape Turing machine** if it has $k$ work tapes in addition to the input and output tapes.

</div>


<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Configurations of Off-line Turing machines)</span></p>

Configurations for off-line Turing machines are defined as before (state, work tape contents, work tape head positions), but with adjustments for the special tapes:

  * **Output Tape:** Neither its content nor its head position is part of a configuration, as this information cannot influence future computational steps.
  * **Input Tape:** Only the position of the head is part of a configuration. For an input of length $n$, this position is an integer in $\lbrace 0, \dots, n+1\rbrace$.

This definition ensures that configurations remain snapshots of all information needed to proceed with a computation. For sublinear space bounds, this allows the representation of a configuration to be smaller than the input itself, which is advantageous in certain constructions.

</div>

$\textbf{Convention 59 (Space-Bounds and Off-line Turing machines):}$ In the context of space-bounded computations, all considered Turing machines are off-line Turing machines, unless explicitly stated otherwise.

### Deterministic Space Complexity

We can now formally define space usage and the corresponding complexity classes.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Space Usage)</span></p>

The **space usage** of a deterministic Turing machine $M$ on input $w$ is:

$$
\text{space}_M(w) =
\begin{cases}
s & \text{if } M \text{ terminates on input } w \text{ and } s \text{ is the maximum} \\
& \text{number of cells accessed on a single work tape,} \\
\uparrow & \text{otherwise.}
\end{cases}
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

We write $\log x$ for the binary logarithm $\log_2 x$. When used in complexity bounds, $\log x$ may also refer to $\lceil \log_2 x \rceil$ or $\lfloor \log_2 x \rfloor$. For space bounds, we define $\log 0 = 1$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Space-Bounded Deterministic Turing Machine)</span></p>

A **space bound** is a computable function $s: \mathbb{N} \to \mathbb{N}$ with $s(n) \ge \log n$ for all $n > 0$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Deterministic Space Classes)</span></p>

Let $s$ be a space bound.

A deterministic Turing machine $M$ is **$s(n)$-space-bounded** if $M$ is total and for all but finitely many inputs $w$, it holds that $\text{space}_M(w) \le s(\lvert w\rvert)$.

The class of languages decidable in deterministic space $s(n)$ is:

$$
\text{DSPACE}(s(n)) = \lbrace L \subseteq \lbrace 0, 1\rbrace^{\ast} : L = L(M) \text{ for a deterministic } s(n)\text{-space-bounded Turing machine } M \rbrace
$$

The class $\text{DSPACE}_k(s(n))$ is defined similarly but restricts machines to at most $k$ work tapes. For a set $F$ of space bounds, we define

$$
\text{DSPACE}(F) = \bigcup_{s \in F} \text{DSPACE}(s(n)).
$$

</div>

### Key Deterministic Space Classes

Using standard function classes, we define some of the most important space complexity classes.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Examples of Deterministic Space Classes)</span></p>

Using the function classes

  * $\text{log} = \lbrace n \mapsto c \cdot \log n + c : c \in \mathbb{N} \setminus \lbrace 0\rbrace \rbrace$
  * $\text{poly} = \lbrace n \mapsto n^c + c : c \in \mathbb{N} \setminus \lbrace 0\rbrace \rbrace$

we define the complexity classes:

  * $\text{LOG} = \text{DSPACE}(\text{log})$
  * $\text{PSPACE} = \text{DSPACE}(\text{poly})$
  * $\text{EXPSPACE} = \text{DSPACE}(2^{\text{poly}}) = \bigcup_{c \in \mathbb{N}} \text{DTIME}(2^{n^c+c})$

We refer to $\text{PSPACE}$ as the class of problems decidable in deterministic polynomial space.

</div>

### Space-Bounded Function Computation

Off-line Turing machines can also be used to compute functions. The output is simply the word written on the output tape upon termination.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Functions computed by space-bounded Turing machines)</span></p>

Let $s$ be a space bound. The class of functions computable in deterministic space $s(n)$ is:

$$\text{FSPACE}(s(n)) = \lbrace f : \lbrace 0, 1\rbrace^* \to \lbrace 0, 1\rbrace^* : f = f_M \text{ for a deterministic } s(n)\text{-space-bounded Turing machine } M \rbrace$$

For a set $F$ of space bounds,

$$\text{FSPACE}(F) = \bigcup_{s \in F} \text{FSPACE}(s(n))$$

</div>

Similar to time complexity, space complexity classes are robust against changes in machine specifics like the number of tapes or constant factors in the space bound.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Linear Compression)</span></p>

For every space bound $s$, it holds for all natural numbers $c$ and $k$ that

$$\text{DSPACE}_k(c \cdot s(n)) \subseteq \text{DSPACE}_k(s(n))$$

and therefore, in particular,

$$\text{DSPACE}(c \cdot s(n)) \subseteq \text{DSPACE}(s(n))$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Alphabet Change)</span></p>

Let $s$ be a space bound, and let $k \ge 2$. For every language $L$ in $\text{DSPACE}_k(s(n))$, there exists a deterministic $O(s(n))$-space-bounded $k$-tape Turing machine $M$ with tape alphabet $\lbrace 0, 1, \square \rbrace$ that recognizes $L$.

</div>

### Nondeterministic Space Complexity

The concept of space bounds extends naturally to nondeterministic Turing machines.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Nondeterministic Space Classes)</span></p>

Let $s$ be a space bound. A Turing machine $M$ is **$s(n)$-space-bounded** if $M$ is total and, for almost all inputs $w$, **all computations** of $M$ visit at most $s(\lvert w \rvert)$ cells on each work tape. The class of languages decidable in nondeterministic space $s(n)$ is:

$$
\text{NSPACE}(s(n)) = \lbrace L \subseteq \lbrace 0, 1\rbrace^* : L = L(M) \text{ for an } s(n)\text{-space-bounded Turing machine } M \rbrace
$$

</div>

### Key Nondeterministic Space Classes

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Examples of Nondeterministic Space Classes)</span></p>

We define the complexity classes:

  * $\text{NLOG} = \text{NSPACE}(\text{log})$
  * $\text{NPSPACE} = \text{NSPACE}(\text{poly})$
  * $\text{NEXP} = \text{NSPACE}(2^{\text{poly}})$

We refer to $\text{NPSPACE}$ as the class of problems decidable in nondeterministic polynomial space.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(The Directed Path Problem)</span></p>

A canonical problem in $\text{NLOG}$ is determining **reachability in a directed graph**.

The problem is formalized as the language:

$$\text{DirPath} = \lbrace \langle A, s, t \rangle : A \text{ is the adjacency matrix of a directed graph } G \text{ in which there exists a path from node } s \text{ to node } t \rbrace$$

An instance $\langle A, s, t \rangle$ for a graph with $n$ nodes can be represented as $1^n0z_1z_2 \dots z_n01^s01^t$, where $z_i$ are the rows of the adjacency matrix.

To show that $\text{DirPath}$ is in $\text{NLOG}$, we can construct a $(c \log n + c)$-space bounded nondeterministic Turing machine. On input $\langle A, s, t \rangle$:

1.  The machine stores the current node, starting with $s$. Storing a node index from $1$ to $n$ requires $O(\log n)$ space.
2.  It nondeterministically guesses a next node to visit by following an edge from the current node.
3.  It keeps a counter to track the number of steps taken, also requiring $O(\log n)$ space.
4.  If the machine reaches node $t$ within $n$ steps, it accepts. If it takes more than $n$ steps without reaching $t$, it rejects, preventing infinite loops in cyclic graphs.

The machine only needs to store the current node and a step counter, both of which require $O(\log n)$ space. Thus, $\text{DirPath} \in \text{NLOG}$.

</div>

### Bounding Computations of Space-Bounded Machines

For a time-bounded machine, the length of any computation is naturally bounded by the time limit. For a space-bounded machine, we can derive a powerful bound on the number of computational steps by considering the number of unique configurations.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Configurations of Space-Bounded Turing Machines)</span></p>

Let $s$ be a space bound, and $M = (Q, \Sigma, \Gamma, \Delta, q_0, F)$ be an $s(n)$-space bounded Turing machine. Then there exists a constant $d$ that depends only on $M$ such that the following two statements hold for all $n$:

* (i) The number of possible **distinct** configurations of $M$ on an input of length $n$ is at most $2^{d \cdot s(n)}$.
* (ii) The depth of the computation tree of $M$ on an input of length $n$ is less than $2^{d \cdot s(n)}$.

</div>

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

However, an $s(n)$-space bounded machine is required by definition to be total, meaning it must terminate on all inputs. This is a contradiction. Therefore, our initial assumption must be false, and no computation path can have a length of $\ell$ or more. The depth of the computation tree must be less than $2^{d \cdot s(n)}$. $\square$

### Relationships Between Time and Space Complexity

A fundamental aspect of complexity theory is understanding the relationships between different resource bounds (time vs. space) and computational modes (deterministic vs. nondeterministic).

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

Let $t$ be any time bound. A $t(n)$-time bounded Turing machine can access at most $t(n)+1$ cells on each work tape. With minor adjustments, we can ensure it is $t(n)$-space-bounded. This gives the immediate inclusions:

$$\text{DTIME}(t(n)) \subseteq \text{DSPACE}(t(n)) \text{ and } \text{NTIME}(t(n)) \subseteq \text{NSPACE}(t(n))$$

</div>


The following theorem summarizes the key relationships between deterministic and nondeterministic time and space classes.

<figure>
  <img src="{{ '/assets/images/notes/computability-and-complexity/coco_theorem_73.png' | relative_url }}" alt="CPU + GPU system" loading="lazy">
  <figcaption>Illustration for Theorem 73</figcaption>
</figure>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name"></span></p>

Let $t$ be a time bound. Then the following inclusions hold.
The relationships in the second and third columns also hold for an arbitrary space bound $s$ instead of $t$.

</div>

**Proof.**: The inclusions in the diagram can be justified as follows:

1.  **Vertical Inclusions:** The inclusions from the first row to the second row (e.g., $\text{DTIME}(t(n)) \subseteq \text{NTIME}(t(n))$) hold by definition, as any deterministic Turing machine is a special case of a nondeterministic one.
2.  **First Horizontal Inclusions:** The inclusions from the first column to the second (e.g., $\text{DTIME}(t(n)) \subseteq \text{DSPACE}(t(n))$) are immediate from the previous Remark.
3.  **Second Horizontal Inclusions:** The inclusions from the second column to the third (e.g., $\text{DSPACE}(t(n)) \subseteq \text{DTIME}(2^{O(t(n))})$) are a direct consequence of the previous Lemma. A deterministic machine can simulate a space-bounded machine by exploring its entire configuration graph. The number of configurations is bounded by $2^{O(t(n))}$, so the simulation takes time exponential in the space bound.
4.  **Diagonal Inclusions:** The diagonal inclusions are statements of other lemmas (Lemmas below in the text). $\square$

These relationships give rise to a chain of inclusions among the major complexity classes.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name"></span></p>

The following inclusions hold.

$$\text{LOG} \subseteq \text{NLOG} \subseteq \text{P} \subseteq \text{NP} \subseteq \text{PSPACE} \subseteq \text{NPSPACE} \subseteq \text{EXP} \subseteq \text{NEXP}$$

</div>

This chain represents our current understanding of the hierarchy of these classes. While these inclusions are known, the strictness of many of them remains one of the greatest open questions in computer science.

* It is known from Savitch's Theorem that $\text{PSPACE} = \text{NPSPACE}$.
* It is known from hierarchy theorems (not covered here) that some of these inclusions are strict:
* $\text{LOG} \neq \text{PSPACE}$
* $\text{NLOG} \neq \text{NPSPACE}$
* $\text{P} \neq \text{EXP}$
* $\text{NP} \neq \text{NEXP}$
* Beyond these results and their direct consequences (like $\text{P} \neq \text{NEXP}$), it is not known which of the other inclusion relations are strict. The most famous of these is the $\text{P}$ versus $\text{NP}$ problem.

### Relationships Between Time and Space Complexity Classes

This chapter explores the fundamental relationships between deterministic and nondeterministic complexity classes, focusing on how time-bounded and space-bounded computations relate to one another. We will establish key inclusions, such as showing that any problem solvable in nondeterministic time can be solved in deterministic space. These relationships culminate in Savitch's Theorem, a landmark result demonstrating that nondeterministic polynomial space is equivalent to deterministic polynomial space ($\text{PSPACE} = \text{NPSPACE}$).

#### Nondeterministic Time vs. Deterministic Space

A foundational result connects nondeterministic time complexity with deterministic space complexity. It establishes that any language recognizable by a nondeterministic Turing machine within a time bound $t(n)$ can also be recognized by a deterministic Turing machine using a space bound of $t(n)$. This suggests that the parallelism inherent in nondeterminism can be simulated deterministically, provided sufficient memory is available.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name"></span></p>

Let $t$ be a time bound. Then $\text{NTIME}(t(n)) \subseteq \text{DSPACE}(t(n))$.

</div>

**Proof**: Let $L$ be a language in $\text{NTIME}(t(n))$, recognized by a $t(n)$-time-bounded $k$-tape nondeterministic Turing machine (NTM) $M$. Let the tape alphabet of $M$ be $\Gamma$ and its transition relation be $\Delta$, with size $d$. We will construct a deterministic Turing machine (DTM) $D$ that recognizes $L$ within a space bound of $t(n)$. The tape alphabet of $D$ will include $\Gamma$ plus a set of $d$ new symbols, $Y = \lbrace y_1, \dots, y_d\rbrace$, where each $y_i$ corresponds to a unique instruction in $\Delta$.

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

To determine $C_u$ and $C_v$, $D$ simulates the partial computations of $M$ coded by $u$ and $v$. This depth-first search guarantees that all configurations in the computation tree are visited. The machine $D$ is deterministic and recognizes the same language as $M$. The space required is dominated by the need to store a configuration of $M$ and the current path (coding word) on the index tape, both of which are bounded by $O(t(n))$. Therefore, $L \in \text{DSPACE}(t(n))$. $\square$

#### Nondeterministic Space vs. Deterministic Time

The next major result, known as Savitch's Theorem, provides a relationship in the other direction: from nondeterministic space to deterministic time. It shows that any problem solvable in nondeterministic space can be solved in deterministic time, though with an exponential increase in the time bound.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name"></span></p>

Let $s$ be a space bound. Then $\text{NSPACE}(s(n)) \subseteq \text{DTIME}(2^{O(s(n))})$.

</div>

**Proof**: Let $L$ be a language in $\text{NSPACE}(s(n))$, recognized by an $s(n)$-space-bounded NTM $M$. We will construct a DTM $D$ that recognizes $L$ in $2^{c \cdot s(n)}$ time for some constant $c$.

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

For the complexity analysis, there are at most $2^{d \cdot s(n)}$ configurations, so there will be at most $2^{d \cdot s(n)}$ expansion steps. A single expansion step involves searching and updating the list of configurations. The time for one step is polynomial in the size of the list, which is $2^{d' \cdot s(n)}$. Thus, the total running time of $D$ is bounded by $2^{d \cdot s(n)} \cdot (2^{d' \cdot s(n)})^t = 2^{(d+td') \cdot s(n)}$. This is of the form $2^{O(s(n))}$, completing the proof. $\square$

#### Savitch's Theorem and Its Consequences

The previous results establish relationships between nondeterministic time and deterministic space, and between nondeterministic space and deterministic time. We now turn to a more direct comparison: nondeterministic space versus deterministic space. This leads to Savitch's Theorem, a cornerstone of complexity theory.

First, we must formally address some technical prerequisites for simulating space-bounded machines.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Space-Constructible Functions)</span></p>

A space bound $s$ is **space-constructible** if there exists an $s(n)$-space-bounded Turing machine $M$ that computes the function $1^n \mapsto 1^{s(n)}$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Space-constructible Functions)</span></p>

The functions in the function classes $\log$, $\text{lin}$, and $\text{poly}$ are all space-constructible. If the space bound $s(n)$ is space-constructible, then so is $n \mapsto 2^{s(n)}$.

</div>

Another important technical result is that constant factors in space bounds do not affect the power of the computational model.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Linear compression)</span></p>

Linear compression refers to the following fact: for all space bounds $s$ and all constants $c$, every $c \cdot s(n)$-space-bounded Turing machine can be transformed into an $s(n)$-space-bounded Turing machine that recognizes the same language; in case the given Turing machine is deterministic, the new one can be chosen to be deterministic, too. Consequently, it holds for all such $s$ and $c \ge 1$ that

$$\text{NSPACE}(c \cdot s(n)) = \text{NSPACE}(s(n))$$

$$\text{DSPACE}(c \cdot s(n)) = \text{DSPACE}(s(n))$$

This is achieved by encoding blocks of symbols from the original machine's tapes into single, more complex symbols on the new machine's tapes, or by using multiple work tapes to simulate one.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Savitch’s Theorem)</span></p>

Let $s$ be a space-constructible space bound. Then $\text{NSPACE}(s(n)) \subseteq \text{DSPACE}(s^2(n))$.

</div>

This theorem has a profound corollary for polynomial space complexity classes.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name"></span></p>

It holds that $\text{PSPACE} = \text{NPSPACE}$.

</div>

**Proof of the Corollary**: By definition, $\text{DSPACE}(s(n)) \subseteq \text{NSPACE}(s(n))$ for any space bound $s$, so $\text{PSPACE} \subseteq \text{NPSPACE}$. For the reverse inclusion, let $L \in \text{NPSPACE}$. This means $L$ is recognized by an NTM in space $p(n)$ for some polynomial $p$. By Savitch's Theorem, $L \in \text{DSPACE}(p^2(n))$. Since the square of a polynomial, $p^2(n)$, is also a polynomial, it follows that $L \in \text{PSPACE}$. Therefore, $\text{NPSPACE} \subseteq \text{PSPACE}$. $\square$

**Proof of Savitch’s Theorem**: Let $L \in \text{NSPACE}(s(n))$ be recognized by an NTM $N$. We can assume without loss of generality that for any input $x$ of length $n$:

* There is a unique accepting configuration, $K_{accept}(x)$.
* All computations have a length of at most $2^{\ell(n)}$, where $\ell(n) = d \cdot s(n)$ for some constant $d$. This is because there are at most $2^{d \cdot s(n)}$ distinct configurations, so any longer computation must contain a cycle.

The NTM $N$ accepts an input $x$ if and only if there is a computation path from the initial configuration $K_{initial}(x)$ to the accepting configuration $K_{accept}(x)$ of length at most $2^{\ell(n)}$. We denote this as:

$$K_{initial}(x) \xrightarrow{\le 2^{\ell(n)}}_N K_{accept}(x) \quad (3.1)$$

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

$$(i) K_{initial}(x) \xrightarrow{\le 2^{\ell(n)-1}}_N K$$

and

$$(ii) K \xrightarrow{\le 2^{\ell(n)-1}}_N K_{accept}(x)$$

2.  To check condition (i), $M$ recursively breaks it down further, looking for a configuration $K'$ such that:
    iii. $K_{initial}(x) \xrightarrow{\le 2^{\ell(n)-2}}_N K'$
    iv. $K' \xrightarrow{\le 2^{\ell(n)-2}}_N K$
3.  This process continues until the length of the computation to be checked is $2^0=1$ or $2^1=2$. These base cases can be checked directly by inspecting the transition function of $N$.

The depth of this recursion is $\ell(n) = d \cdot s(n)$. At each level of the recursion, the machine $M$ needs to store the configurations that form the start and end points of the current subproblem (e.g., $K_{initial}, K_{accept}, K, K'$, etc.). Since the recursion depth is $\ell(n)$, and each configuration of $N$ requires $O(s(n))$ space to store, the total space required for the recursion stack is $O(\ell(n) \cdot s(n)) = O(s(n) \cdot s(n)) = O(s^2(n))$.

By linear compression, any $O(s^2(n))$-space bounded DTM can be converted to a $\text{DSPACE}(s^2(n))$ machine. This completes the proof. $\square$

#### The $\text{P}$ versus $\text{NP}$ Problem and PSPACE

The famous **$\text{P}$ versus $\text{NP}$ problem** asks whether deterministic polynomial-time computation is as powerful as nondeterministic polynomial-time computation.

A. Do the classes $\text{P}$ and $\text{NP}$ coincide?

This remains one of the greatest unsolved problems in computer science. It is not even known if $\text{PSPACE}$, the class of languages decidable in deterministic polynomial space, coincides with $\text{P}$ or $\text{NP}$. A related open question concerns the closure of $\text{NP}$ under complement.

B. Is the class $\text{NP}$ closed under complement, i.e., does the complement $\lbrace 0, 1\rbrace^* \setminus L$ of any language $L$ in $\text{NP}$ also belong to $\text{NP}$?

In the context of polynomial space, the analogous questions have been answered in the affirmative. Savitch's Theorem directly answers the space analogue of question A, showing $\text{PSPACE} = \text{NPSPACE}$. The answer to the space analogue of question B is also yes, a result proven by Immerman and Szelepcsényi.

#### A Complete Language for $\text{PSPACE}$

To study the intrinsic difficulty of a complexity class, we identify languages that are "hardest" within that class. This is formalized through the concepts of hardness and completeness.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Complexity class, hardness, completness)</span></p>

A **complexity class** is a set of languages over the binary alphabet. A language $B$ is **hard** for a complexity class if every language in the class is $p$-$m$-reducible to $B$. A language is **complete** for a complexity class if it is hard for the class and belongs to the class. A language that is complete for a complexity class $C$ is called **$\text{C}$-complete**.

</div>

We now introduce a type of logical formula whose evaluation problem is complete for $\text{PSPACE}$. This language, TQBF (True Quantified Boolean Formulas), serves a role for $\text{PSPACE}$ similar to what $\text{SAT}$ serves for $\text{NP}$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Quantified Propositional Formulas)</span></p>

Let $\Lambda = \lbrace \neg, \land, (, ), \exists, \forall\rbrace$ and let $\text{Var}$ be a countable set of variables disjoint from $\Lambda$. The set of **quantified propositional formulas** over $\text{Var}$ is a set of words over the infinite alphabet $\text{Var} \cup \Lambda$ that is defined inductively as follows:

* **Base case.** All elements of $\text{Var}$ are quantified propositional formulas.
* **Inductive step.** If $\psi$ and $\psi'$ are quantified propositional formulas and $X$ is in $\text{Var}$, then $\neg\psi$, $(\psi \land \psi')$, and $\exists X \psi$ are quantified propositional formulas.

</div>

**Quantifier Primer.** Building on standard propositional logic, quantified propositional formulas extend the language with existential and universal quantifiers in addition to the usual connectives such as disjunction $(\lor)$ and implication $(\rightarrow)$. A universal quantification $\forall X \phi$ is shorthand for $\neg(\exists X \neg \phi)$. We also use the constants $0$ and $1$ for the logical values false and true, respectively, and assume conventional precedence so parentheses can be dropped when the intent is clear.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Free and Bound Variables)</span></p>

When quantifiers are present we distinguish between free and bound occurrences of variables. For the formula $\neg X \lor \exists X \forall Y (X \land Y)$ the variable $X$ appears three times: the first occurrence is free, the third is bound, and the second (immediately after the quantifier) is neither free nor bound. Formal definitions are omitted because the intuition is immediate for the prenex normal forms used below.

**Evaluating Truth Values.** The truth value of a quantified propositional formula $\psi$, denoted $\text{val}(\psi)$, is either $0$ (false) or $1$ (true). Evaluation follows two principles:

- Truth values are defined inductively over the structure of $\psi$.
- The value of $\psi$ is always relative to an assignment for its free variables.

Given a formula $\phi$ with free variable $X$ and an assignment $b$ to the other free variables, define $\phi_i$ as the result of replacing each free occurrence of $X$ by the constant $i \in \lbrace 0, 1\rbrace$. Then

- $\exists X \phi$ is true under $b$ iff at least one of $\phi_0$ or $\phi_1$ is true under $b$.
- $\forall X \phi$ is true under $b$ iff both $\phi_0$ and $\phi_1$ are true under $b$.

</div>

**Prenex Normal Form.** A standardized structure streamlines reasoning about quantified formulas.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Prenex Normal Form)</span></p>

A quantified propositional formula is in **prenex normal form** if it has the shape $Q_1 X_1 \cdots Q_m X_m \phi$ where the $X_i$ are mutually distinct variables, every $Q_i$ belongs to $\lbrace \exists, \forall\rbrace$, and $\phi$ is a quantifier-free propositional formula in conjunctive normal form (CNF).

Writing $\psi = Q_1 X_1 \cdots Q_m X_m \phi$ in prenex normal form assumes $\phi$ is in CNF and all occurrences of each variable inside $\phi$ are free. Any quantified propositional formula can be converted into an equivalent prenex normal form in deterministic polynomial time.

</div>

**The QBF Language.** The central decision problem for these formulas asks whether a sentence (a quantifier prefix followed by a quantifier-free matrix) is true.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The Language $\text{QBF}$)</span></p>

$\text{QBF} = \lbrace \psi : \psi \text{ is a true sentence in prenex normal form}\rbrace$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">($\text{QBF}$ is $\text{PSPACE}$-Complete)</span></p>

$\text{QBF}$ is $\text{PSPACE}$-complete, as captured by the following lemmas.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name"></span></p>

$\text{QBF} \in \text{PSPACE}$.

</div>

**Proof:**

To place $\text{QBF}$ inside $\text{PSPACE}$, build a deterministic polynomial-space TM $M$ that evaluates any sentence $\psi = Q_1 X_1 \cdots Q_m X_m \phi$ of length $n$. Inputs that are not sentences in prenex normal form are rejected outright. The machine maintains three binary words $w_1, w_2, w_3$, each of length $m$, on separate tapes:

- Tape 1 ($w_1$) encodes the quantifier prefix, with bit $i$ equal to $1$ iff $Q_i$ is existential.
- Tape 2 ($w_2$) is a pointer of the form $1^i 0^{m-i}$ indicating the current quantifier.
- Tape 3 ($w_3$) stores the current truth assignment for $X_1, \dots, X_m$.

Initially, $w_1$ captures the full prefix, $w_2$ points to $Q_1$, and $w_3$ is the all-zero assignment. The evaluation procedure recursively branches on each quantifier, toggling the relevant bit of $w_3$ and combining subresults according to whether the quantifier is existential or universal. The recursion depth is $m \le n$, and only the current assignment and fixed bookkeeping are stored, so the total space is polynomial in $n$. Hence $\text{QBF} \in \text{PSPACE}$. $\square$

The recursion depth is $m \le n$ and each layer keeps only two values: result for the current var=0 and result for the current var=1.

```python
def evaluate_qbf(psi):
    """
    - Always evaluate BOTH branches for each quantified variable
    - Combine results with max (EXISTS) or min (FORALL)
    - Uses w3 as the current assignment; backtracking is done by overwriting w3[i]
    """

    if not is_prenex_qbf(psi):
        return False

    Q, X, phi = parse_prenex(psi)          # Q[i] in {"E","A"}
    m = len(Q)

    # w1: quantifier types (1 = EXISTS, 0 = FORALL)
    w1 = [1 if Q[i] == "E" else 0 for i in range(m)]

    # w3: current assignment (bits for X1..Xm)
    w3 = [0] * m

    def dfs(i):
        if i == m:
            return eval_phi(phi, X, w3)    # quantifier-free evaluation under current assignment

        # evaluate branch Xi = 0
        w3[i] = 0
        r0 = dfs(i + 1)

        # evaluate branch Xi = 1
        w3[i] = 1
        r1 = dfs(i + 1)

        # combine results depending on quantifier
        if w1[i] == 1:                     # EXISTS
            return max(r0, r1)             # booleans: max = OR
        else:                              # FORALL
            return min(r0, r1)             # booleans: min = AND

    return dfs(0)
```

<div class="accordion">
  <details markdown="1">
    <summary>Evaluating $\phi$ still fits in polynomial space</summary>

Good question — this is *exactly* where people get suspicious about “are we really just using polynomial space?”.

Short answer:
Yes, we do need some extra memory to evaluate $\phi$ and store intermediate results — **but**:

* we only ever evaluate $\phi$ for **one assignment at a time**,
* we only need to keep intermediate results for **one path in $\phi$** at a time,
* and that all fits in **$O(\lvert\phi\rvert) \le O(n)$** extra space.

So we’re still safely in PSPACE.

## 1. What does “evaluating $\phi$” actually involve?

$\phi$ is a *quantifier-free* Boolean formula over variables $X_1,\dots,X_m$.
On a given assignment $w_3$ (a bitstring telling you which $X_i$ are true/false), the job of the machine is:

> Given $\phi$ on the **input tape** + assignment $w_3$ on **tape 3**, compute whether $\phi(w_3) = 1$ or $0$.

The input tape is read-only and **does not count** towards space complexity.
All extra “intermediate results” must go on work tapes, and we must show that’s only polynomially many cells.

## 2. Two ways to see that evaluating $\phi$ is poly-space

### View A: Recursive evaluation / stack

Think of $\phi$ as a syntax tree: leaves are literals, internal nodes are $\land,\lor,\lnot$, etc.

A very natural algorithm:

1. Start at the root of $\phi$.
2. Recursively evaluate its children.
3. Combine child values using the operator $\land,\lor,\lnot,\dots$.
4. Return a single bit (true/false).

To implement this on a Turing machine, you keep something like a **stack** of:

* “Where am I in $\phi$?” (position or subformula),
* “Have I already evaluated the left child?”,
* “What was the result of the child?” (just 1 bit),
* “What operator is this node?” ($\land,\lor,\lnot$, etc.)

The key facts:

* The **depth** of this recursion/stack is at most the **size of $\phi$** (number of symbols), hence $\leq n$.
* Each stack frame is **constant-size** (a few bits/flags + maybe an index into the input).

So the total work-tape space for evaluating $\phi$ is:

$$O(\text{depth}(\phi)) \cdot O(1) \le O(\lvert\phi\rvert) \le O(n).$$

Plus the assignment $w_3$ (length $m \le n$). So overall still $O(n)$ space.

Time can be exponential (because of the branching over quantifiers), but PSPACE cares only about **space**, not time.

### View B: Super-space-frugal evaluation by rescanning

If we want to be *extra* stingy with space, we can even avoid storing a big stack:

* For example, if $\phi$ is in CNF (AND of clauses), we can:

  * scan the input, clause by clause,
  * for each clause, keep just:
    * one “clause is satisfied?” bit,
    * maybe an index while scanning,
  * use assignment $w_3$ to tell if each literal is true.

We *never* need to store all clause truth values at once — we can:

* immediately reject if we find an unsatisfied clause,
* otherwise, keep going until the end.

Again, this uses **constant or logarithmic extra space** on top of the assignment, just at the cost of potentially rescanning the input multiple times.

## 3. How this fits back into the lemma

The lemma’s machine uses:

* Tape 1: $w_1$ – quantifier types, length $m \le n$
* Tape 2: $w_2$ – pointer, length $m \le n$
* Tape 3: $w_3$ – assignment, length $m \le n$
* Plus **some extra cells** to evaluate $\phi$ for the current $w_3$, as above:

  * either a recursion stack of depth $\leq \lvert\phi\rvert \le n$, or
  * a few bits for a streaming check.

Total work-tape space used at any moment:

$$O(m) + O(\text{space for evaluating }\phi) \le O(n) + O(n) = O(n),$$

which is **polynomial** in the input length $n$.

We do *not* store:
* all intermediate results for **all** assignments,
* or a full truth table,
* or a big DP table over subformulas × assignments,

because that would explode space. Instead, we:

* only keep data for **one assignment** $w_3$ at a time (depth-first search through quantifiers),
* only keep data for **one “path” in $\phi$** at a time (recursive evaluation, or multiple rescans).

This is the classic **time–space tradeoff**: we’re happy to redo work (evaluate $\phi$ again and again) to save space.

  </details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name"></span></p>

$\text{QBF}$ is $\text{PSPACE}$-hard.

</div>

**Proof:**

For hardness, reduce any language $A \in \text{PSPACE}$ to $\text{QBF}$ via a polynomial-time computable map $g$. Let $M = (Q, \Sigma, \Gamma, \Delta, s, F)$ be a $p(n)$-space-bounded deterministic TM recognizing $A$, assumed to use a single work tape. A configuration on inputs of length $n$ is determined by:

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

  $$\text{Comp}_n^0(K, K') \equiv \text{Konf}_n(K) \land \text{Konf}_n(K') \land (\text{Equal}_n(K, K') \lor \text{Succ}_n(K, K'))$$

- **Inductive step:**

  $$
  \text{Comp}_n^{i+1}(K, K') \equiv \exists \tilde{K} \forall K_1 \forall K_2 (\text{Konf}_n(\tilde{K}) \land ((\text{Equal}_n(K_1, K) \land \text{Equal}_n(K_2, \tilde{K})) \lor (\text{Equal}_n(K_1, \tilde{K}) \land \text{Equal}_n(K_2, K'))) \rightarrow \text{Comp}_n^i(K_1, K_2))
  $$

This reuse of $\text{Comp}\_n^i$ keeps the overall formula size polynomial in $i$. For input $x$ of length $n$, let $K\_{initial}(x)$ and $K\_{accept}(x)$ denote the start and accepting configurations. Since $M$ runs in at most $2^{d \cdot p(n)}$ steps for some constant $d$, define


$$\phi_x \equiv \text{Comp}_n^{d \cdot p(n)}(K_{initial}(x), K_{accept}(x))$$

Then $\phi_x$ is true iff $x \in A$, and $g:x \mapsto \phi_x$ is computable in polynomial time, so $\text{QBF}$ is $\text{PSPACE}$-hard. $\square$

<div class="accordion">
  <details markdown="1">
    <summary>Why not the Savitch-style definition of $\text{Comp}$</summary>

Your Savitch-style definition is *logically* fine, but as a **QBF encoding** it blows up the formula size exponentially, so it doesn’t give a polynomial-time reduction. The weird-looking definition in the notes is exactly a trick to avoid this blow-up while keeping the same meaning.

### 1. What you “expect” vs what they actually do

What you intuitively expect is something like (fixing the index to $i+1$):

$$
\text{Comp}_n^{i+1}(K,K')
\equiv
\exists \tilde{K}\Big(
\text{Konf}_n(\tilde{K})
\land \text{Comp}_n^{i}(K,\tilde{K})
\land \text{Comp}_n^{i}(\tilde{K},K')
\Big)
$$

This is exactly analogous to Savitch:

> “There is a path of length $\le 2^{i+1}$ from $K$ to $K'$ iff there exists a middle configuration $\tilde{K}$ such that there is a path of length $\le 2^i$ from $K$ to $\tilde{K}$ and from $\tilde{K}$ to $K'$.”

**Semantically, this is totally correct.** So the issue is *not* that your definition is wrong logically. The issue is **size**.

### 2. Why your recursive definition is a problem for QBF

Let $\lvert\text{Comp}_n^i\rvert$ be the syntactic size (number of symbols) of the formula $\text{Comp}_n^i$.

* In your version $(*)$, $\text{Comp}_n^{i+1}$ **contains two copies** of $\text{Comp}_n^{i}$ as subformulas: one with arguments $(K,\tilde{K})$ and one with $(\tilde{K},K')$.

So the size satisfies roughly

$$\lvert\text{Comp}_n^{i+1}\rvert = O(1) + 2 \cdot \lvert\text{Comp}_n^i\rvert$$

This solves to

$$\lvert\text{Comp}_n^i\rvert = O(2^i)$$

But we need to go up to $i = d \cdot p(n)$ so that paths of length up to $2^{d p(n)}$ are allowed. That gives

$$\lvert\text{Comp}_n^{d p(n)}\rvert = 2^{\Theta(p(n))}$$

i.e. *exponential* in the input size $n$.

And that is fatal: our reduction $g(x)$ must output a QBF formula $\phi_x$ of size **polynomial** in $\lvert x\rvert$. You cannot even *print* an exponentially long formula in polynomial time.

So:

* Your definition gives a **correct** specification of reachability,
* but it leads to an **exponentially large** formula, so it’s **not a valid polynomial-time reduction**.

This is the same “recursion tree vs DAG” phenomenon as in Savitch’s theorem, but now we care about **formula length**, not about space. Savitch’s algorithm is allowed to do exponentially many recursive calls, as long as it only uses polynomial *space*. Here, the QBF formula must be a *static object* whose length directly reflects how much “computation” we hard-wire into it.

### 3. What the weird $\text{Comp}$-definition is doing

Their inductive step is:

$$
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
$$

The key idea: **only one syntactic occurrence** of $\text{Comp}_n^i$ appears — namely $\text{Comp}_n^i(K_1,K_2)$.

So now we have the size recurrence

$$\lvert\text{Comp}_n^{i+1}\rvert = \lvert\text{Comp}_n^{i}\rvert + \text{poly}(n)$$

hence

$$\lvert\text{Comp}_n^{i}\rvert = \text{poly}(n) \cdot i = \text{poly}(n) \cdot p(n) = \text{poly}(n)$$

Exactly what we need.

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

  $$\forall x\,((x = a \lor x = b) \rightarrow P(x))$$

  This is logically equivalent to $P(a)\land P(b)$ but only uses $P$ once.

Their construction is the same idea, lifted to configurations and to the reachability predicate $\text{Comp}_n^i$.

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

**Motivation.** Deterministic time and space classes are trivially closed under complement: if $M$ decides $L$, flipping its accepting and rejecting states decides $\bar{L}$. The same argument fails for nondeterministic time classes such as $\text{NP}$, where acceptance means at least one path accepts, and it remains open whether $\text{NP} = \text{co-NP}$. For nondeterministic space classes the situation is better, thanks to Immerman and Szelepcsényi. Both used a proof technique now known as **inductive counting**.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Closure Under Complement)</span></p>

A complexity class is **closed under complement** if $L$ in the class implies $\bar{L} = \lbrace 0, 1\rbrace^* \setminus L$ is also in the class.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Immerman-Szelepcsényi)</span></p>

Let $s$ be space-constructible. Then $\text{NSPACE}(s(n))$ is closed under complement.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name"></span></p>

The class of context-sensitive languages is closed under complement.

</div>

**Proof of the Corollary.** Over a binary alphabet the context-sensitive languages coincide with $\text{NSPACE}(n)$, so the corollary is an immediate special case of Theorem 91. The same holds for arbitrary alphabets after adjusting the definition of $\text{NSPACE}(s(n))$. $\square$

**Proof of Theorem 91.** Let $L \in \text{NSPACE}(s(n))$ and let $N$ be an $s(n)$-space-bounded NTM recognizing $L$. Assume each $x \in L$ has a unique accepting configuration $K_{accept}(x)$. We will build an $O(s(n))$-space NTM $M$ for $\bar{L}$. For input $x$ of length $n$, configurations of $N$ consist of the input-head position, the work-tape contents (length at most $s(n)$), and the work-tape head position. Denote the set of configurations by $\text{Conf}_N(n)$. There exists a constant $d$ so each configuration can be encoded by a binary word of length $\ell(n) = d \cdot s(n)$, implying $\lvert \text{Conf}_N(n)\rvert \le 2^{\ell(n)}$. It is decidable in deterministic $O(s(n))$ space whether a word encodes a valid configuration and whether one configuration yields another in one step.

Let $\text{Conf}\_N(x, t)$ be the configurations reachable from $K_{start}(x)$ within $t$ steps, and write $k_t = \lvert \text{Conf}\_N(x, t)\rvert$. Define four $O(s(n))$-space NTMs $N_1, N_2, N_3, N_4$ for nondeterministic function computation:

- **$N_1$:** On input $(t, k_t, K)$ iterate over all configurations $K'$, checking (i) $K' \in \text{Conf}_N(x, t)$ and (ii) $K' \xrightarrow[N]{1} K$. Maintain a counter $a$ of the successful $K'$. If $a \ne k_t$, report an error. Otherwise output $1$ iff at least one $K'$ satisfies both properties, and $0$ otherwise. At least one computation path correctly guesses every subcomputation and produces the right answer without errors.
- **$N_2$:** On input $(t, k_t)$ iterate over all configurations $K$ and invoke $N_1(t, k_t, K)$ to determine whether $K \in \text{Conf}\_N(x, t+1)$. Count the number of configurations for which $N_1$ returns $1$. Errors from $N_1$ propagate; otherwise the count equals $k_{t+1}$.
- **$N_3$:** Starting from $k_0 = 1$ (only $K_{start}(x)$ is reachable in zero steps), repeatedly call $N_2$ to compute $k_1, k_2, \dots, k_{2^{\ell(n)}}$. Finally, call $N_1$ to check if $K_{accept}(x) \in \text{Conf}_N(x, 2^{\ell(n)})$ and output that result. Some computation path of $N_3$ therefore recognizes $L$.
- **$N_4$:** Replicate $N_3$ but flip the final output bit, swapping the accepting and rejecting halts on all error-free paths. This machine recognizes $\bar{L}$ within $O(s(n))$ space.

Adjusting $N_4$ so that output $1$ leads to an accepting halt and output $0$ to rejection yields an NTM that decides $\bar{L}$, establishing closure under complement for $\text{NSPACE}(s(n))$. $\square$
