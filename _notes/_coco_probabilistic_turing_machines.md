## Probabilistic Turing Machines

[Alternative Definition of Probabilistic Turing Machines](/subpages/computability-and-complexity/probabilistic-turing-machine/)

### Introduction: The Auxiliary Tape Model

To understand randomized computation, we first introduce a more general model: a Turing machine equipped with an additional, special-purpose tape. This model serves as a foundation for both nondeterministic and probabilistic computation by reinterpreting the contents of this extra tape.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Turing machine with auxiliary tape)</span></p>

A Turing machine with auxiliary tape is a deterministic Turing machine that has a special read-only **one-way** tape, its **auxiliary tape**. Initially, the cells of the auxiliary tape contain an infinite binary sequence, the **auxiliary sequence**, and the reading head is positioned on the first bit of this sequence. A Turing machine with auxiliary tape is **total** if it terminates on all inputs and auxiliary sequences, and it is **$t(n)$-time-bounded** for some time-bound $t(n)$ if it is total and runs for at most $t(\lvert w \rvert)$ steps on almost all inputs $w$ and all auxiliary sequences.

The key feature of the auxiliary tape is that it is **one-way**, meaning the head can only move to the right. This makes it suitable for modeling a stream of information, such as guesses or random bits.

</div>

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

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

In what follows, we restrict attention to Turing machines with auxiliary tape that are time-bounded. Such a Turing machine can store all bits it reads from the auxiliary tape on some work tape while still obeying its time bound. So in connection with time bounds it is no real restriction to require the auxiliary tape to be read-only and one-way. In contrast, a space-bounded Turing machine can read more bits from the auxiliary tape than it can store on its tapes. More precisely, an $s(n)$-space-bounded Turing machine may, for example, run for as many as $2^{s(n)}$ steps and may read up to the same number of bits from the auxiliary tape.

</div>

For a time-bounded machine, the number of bits it can access from the auxiliary tape is naturally limited by its runtime.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

Let $t(n)$ be a time-bound. A $t(n)$-bounded Turing machine with auxiliary tape can read at most the first $t(n)$ bits on its auxiliary tape. In general, for a total Turing machine with auxiliary tape it follows by König’s lemma that there is a computable function $t$ such that on inputs of length $n$ at most the first $t(n)$ bits of the auxiliary sequence can be read. In these situations and with $t(n)$ understood from the context, we will refer to the prefix of length $t(n)$ of the auxiliary sequence as **auxiliary word**.

</div>

This model provides a powerful way to re-characterize familiar complexity classes like $\text{NP}$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

Recall that a language $L$ is in the class $\text{NP}$ if and only if there exists a language $B$ in $\text{P}$ and a polynomial $p$ such that for all binary words $w$, it holds that $w \in L \text{ if and only if } \exists z \in \lbrace 0, 1\rbrace^* [(w , z) \in B \text{ \& } \lvert z \rvert \le p(\lvert w \rvert)]$. This equivalence can be reformulated in terms of Turing machines with auxiliary tape: a language $L$ is in the class $\text{NP}$ if and only if there is a polynomially time-bounded Turing machine $M$ with auxiliary tape such that for all binary words $w$, it holds that $w \in L \text{ if and only if } M \text{ accepts } w \text{ for some auxiliary word } z$.

</div>

### Formalizing Probabilistic Computation

We can now define a **probabilistic Turing machine** by treating the auxiliary tape as a source of random bits, as if generated by flipping a fair coin.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Probabilistic Turing machine)</span></p>

A **probabilistic Turing machine** is a total Turing machine with auxiliary tape. Let $M$ be a probabilistic Turing machine and let $t$ be a computable function such that $M$ reads at most the first $t(n)$ bits on its random tape. The **acceptance probability** of $M$ on input $w$ is $\text{accept}_M(w) = \frac{\lvert \lbrace z \in \lbrace 0, 1\rbrace^{t(n)} : M \text{ accepts on input } w \text{ and random word } z\rbrace \rvert }{2^{t(n)}}$, and the **rejection probability** of $M$ on $w$ is $\text{reject}_M(w) = 1 - \text{accept}_M(w)$.

In this model, the machine doesn't simply accept or reject. Instead, it accepts with a certain probability. The decision rule is based on whether this probability crosses a specific threshold.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Acceptance and Recognized Language)</span></p>

Let $M$ be a total probabilistic Turing machine. Then $M$ **accepts** an input $w$, if $\text{accept}_M(w) > \frac{1}{2}$. The language $L(M)$ **recognized** by $M$ is the set of all inputs that are accepted by $M$.

</div>

This probabilistic nature introduces the possibility of error.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Error Probability)</span></p>

The **error probability** of $M$ on input $w$ with respect to a language $L$ is

$$
\text{error}^L_M(w) =
\begin{cases}
\text{reject}_M(w), & \text{if } w \text{ is in } L, \\
\text{accept}_M(w), & \text{otherwise},
\end{cases}
$$

and we say that $M$ recognizes $L$ with error probability $\text{error}^L_M(w)$. The **error probability** $\text{error}_M(w)$ of $M$ on input $w$ is equal to $\text{error}^{L(M)}_M(w)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

Probabilistic Turing machines can also be defined in terms of nondeterministic Turing machines that have computation trees where each inner node has exactly two children and for any given input, all leaves have the same depth. For a given input, the probability of acceptance is then defined as the ratio of accepting computations to all computations. This model of probabilistic Turing machine is equivalent to the one introduced above in the sense that the same languages can be recognized in both models with the same time bound up to a constant factor and with the same error probability.

</div>

### Probabilistic Complexity Classes

Based on these definitions, we can define two fundamental complexity classes for probabilistic polynomial-time computation.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Probabilistic Complexity Classes)</span></p>

The class of languages decidable in **probabilistic polynomial time** is 

$$
\text{PP} = \lbrace L \subseteq \lbrace 0, 1\rbrace^* : L = L(M) \text{ for some polynomially time-bounded probabilistic Turing machine}\rbrace
$$ 

The class of languages decidable in **probabilistic polynomial time with bounded error** is 

$$
\text{BPP} = \lbrace L \subseteq \lbrace 0, 1\rbrace^* : L = L(M) \text{ for some polynomially time-bounded probabilistic Turing machine with error probability at most } \frac{1}{3}\rbrace
$$

</div>

The class $\text{PP}$ allows the acceptance probability for a correct answer to be arbitrarily close to $\frac{1}{2}$, whereas $\text{BPP}$ requires a constant gap (e.g., between $\frac{1}{3}$ and $\frac{2}{3}$), making it a more "reliable" class of computation.

These classes have clear relationships with other major complexity classes.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name"></span></p>

It holds that $\text{BPP} \subseteq \text{PP} \subseteq \text{PSPACE}$.

</div>

**Proof sketch.**: The first inclusion, $\text{BPP} \subseteq \text{PP}$, holds by definition. The second inclusion follows because a polynomially time-bounded probabilistic Turing machine can be simulated by a polynomially space-bounded deterministic Turing machine for all random words while counting the number of random words that result in acceptance. ∎

Specifically, to prove that $\text{PP} \subseteq \text{PSPACE}$ we construct a polynomially space-bounded deterministic Turing machine $M_1$, that uses/simulates the polynomially time-bounded Turing machine $M_2$ with auxiliary tape (which is deterministic by the definition of the TM with auxiliary tape). Given the input word w on the input tape, $M_1$ simulates $M_2$ on this input, but because $M_1$ does not have an auxiliary tape, one working tape $M_1$ serves as a auxiliary tape for $M_2$. $M_1$ writes all possible auxiliary words $z$ to the auxiliary tape one by one and for each such auxiliary word $M_2$ accepts or rejects. $M_2$ is total by definition, so we know for sure that it will either accept or reject a pair $(w,z)$. $M_1$ counts the number of accepted pairs $(w,z)$, updating its counter after each run of $M_2$ on a pair $(w,z)$: increment if is accepted, otherwise do nothing. Then if the number of accepted pairs is strictly greater than half of all pairs, the turing machine $M_1$ accepts the input $w$, rejecting otherwise. I divide the space analysis of $M_1$ into to parts: (i) space analysis of $M_2$ simulation and (ii) analysis of accepted pairs $(w,z)$ counting.

(i) Each run on each pair $(w,z)$ requires polynomial memory as $M_2$ is time-bounded by some polynomial $t(\lvert w \rvert)$ and $\text{DTIME}(t(\lvert w \rvert)) \subseteq \text{DSPACE}(t(\lvert w \rvert))$ $\implies$ one simulation of $M_2$ takes $t(\lvert w \rvert)$ memory cells. Construction of the auxiliary tape requires the same polynomial $t(\lvert w \rvert)$ memory cells, because $t(\lvert w \rvert)$ time-bounded TM with an auxiliary tape, can read at most of the first $t(\lvert w \rvert)$ bits on its auxiliary tape, that is why we consider auxiliary words of length $t(\lvert w \rvert)$ only. Each run of $M_2$ we reuse the same tapes and cells, thus $2t(\lvert w \rvert)$ memory cells for overall simulation of $M_2$.

(ii) Assuming that the alphabet of the auxiliary tape is binary ($\lbrace 0,1 \rbrace$), the total number of all possible pairs is $2^{t(\lvert w \rvert)}$, which greater than polynomial. That is why we will use binary counter, reducing the $2^{t(\lvert w \rvert)}$ memory cells that would unary counting used to $\log 2^{t(\lvert w \rvert)} = t(\lvert w \rvert)$ memory cells that binary counting uses. So, binary counter uses polynomial $t(\lvert w \rvert)$ number of memory cells. Technical detail: we skip the first accepted pair.

After the simulation and counting steps end, we check whether the most significant bit is $1$. If it is, then the number of of accepted pairs exceeds the number of rejected ones (we skipped the first accepted pair, then there cannot be a situation that counter shows only half of accepted pairs with the most significant bit $1$), implying that the fraction of accepted pairs is greater than $\frac{1}{2}$ and the polynomially time-bounded probabilistic Turing machine $M_2$ accepts the input word $w$.


Furthermore, these classes are robust under polynomial-time reductions.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name"></span></p>

The complexity classes $\text{PP}$ and $\text{BPP}$ are closed downwards under $p$-$m$-reducibility.

</div>

**Proof.**: We give the proof for the case of $\text{PP}$ and omit the essentially identical considerations for $\text{BPP}$. Let $A$ be reduced to a language $B$ in $\text{PP}$ via some function $g$ that is computable in time $p_g(n)$ for some polynomial $p_g$. Let $M$ be a $p_M(n)$-time bounded probabilistic Turing machine that recognizes $B$ where $p_M$ is a polynomial. Then $A$ is recognized by a probabilistic Turing machine $M_A$ that on input $w$ first computes $g(w)$, then simulates $M$ on input $g(w)$, and accepts if and only if $M$ accepts. Observe that $M_A$ can be chosen to be $(p_g(n) + p_M(p_g(n)))$-time-bounded, hence is polynomially time-bounded. $\square$

The connection between $\text{NP}$ and $\text{PP}$ is also well-defined.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name"></span></p>

It holds that $\text{NP} \subseteq \text{PP}$.

</div>

**Proof.**: Let $L$ be any language in $\text{NP}$. By Remark 102, there is a polynomial $p$ and a deterministic $p(n)$-time-bounded Turing machine $M$ with auxiliary tape such that $w$ is in $L$ if and only if $M$ accepts $w$ for some auxiliary word $z$. Now we construct a new probabilistic TM $\text{P}$ that uses $\text{M}$ as a subroutine, plus a trick.  On input $w$, $P$ does:

1. Take a random word $r \in \lbrace 0,1\rbrace^{t(\lvert w \rvert)}$.
   * We pick $t(\lvert w \rvert) = p(\lvert w \rvert) + 1$ so that we can split it as: $r = b z, \quad b \in \lbrace 0,1 \rbrace,\ z \in \lbrace 0,1 \rbrace^{p(\lvert w\rvert)}.$

2. If the **first bit** $b = 1$, $P$ **always accepts**.

3. If the **first bit** $b = 0$, then $P$ runs $M$ on input $w$ with auxiliary word $z$, and accepts iff $M$ accepts.

The key is step 2: *“always accepts in case the first bit of the random word is 1”*. That’s the “probability boosting” trick increasing the probability of acceptance to $\frac{1}{2}$ regardless of the problem $w$ itself, then adjustment of the probability based on the problem $w$.

Then $L$ is recognized by the polynomially time-bounded probabilistic Turing machine $P$ that on input $w$ always accepts in case the first bit of the random word is $1$ and, otherwise, in case the random word has the form $0z$, accepts if and only if $M$ accepts $w$ with auxiliary word $z$. $\square$

### Properties of the Class PP

The definition of $\text{PP}$ allows for an acceptance and error probability of exactly $\frac{1}{2}$, which is ambiguous: it could happend that the PTM accepts some input word $w$ with exactly a half of all random words. It means that $\text{accept}_M(w)=\text{reject}_M(w)=\frac{1}{2} \implies \text{error}^L_M(w) = \frac{1}{2}$. However, this can be avoided.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name"></span></p>

Every language in $\text{PP}$ is recognized by a polynomially time-bounded probabilistic Turing machine that has error probability strictly less than $\frac{1}{2}$.

</div>

**Proof.**: Let $L$ be a language in $\text{PP}$ and let $M$ be a probabilistic Turing machine that recognizes $L$ and is $p(n)$-time-bounded for some polynomial $p$. We construct a probabilistic Turing machine $M'$ as required. For almost all $n$, $M$ runs for at most $p(n)$ steps, hence reads at most $p(n)$ bits from its random tape on all inputs of length $n$. We can disregard all other inputs by hard-wiring their acceptance or rejection into $M'$.

On an input $w$ of length $n$, the Turing machine $M'$ first computes $p(n)$ and copies the first two blocks of $p(n)$ bits on its random tape to two special tapes, respectively. Then $M'$ works like $M$ on input $w$ while using the first special tape as random tape, and $M'$ accepts if and only if $M$ accepts and the bits on the second tape are not all $0$.

In case $w$ is not in $L(M)$, by construction either $\text{reject}\_{M'}(w) = \text{reject}\_M(w) = 1 \quad \text{or} \quad \text{reject}\_{M'}(w) > \text{reject}\_M(w) \ge \frac{1}{2}$. In case the word $w$ is in $L(M)$, it is accepted by $M$ on strictly more than half of all random words of length $p(n)$, hence is accepted by $M'$ for at least $(2^{p(n)-1} + 1)(2^{p(n)} - 1) = \underbrace{2^{2p(n)-1} + 2^{p(n)} - 1}_{>0}$ many random words of length $2p(n)$, i.e., for strictly more than half of the latter words. In summary, the polynomially time-bounded probabilistic Turing machine $M'$ recognizes the same language as $M$ and has error probability strictly less than $\frac{1}{2}$. $\square$

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


<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

Recall that a time bound $t(n)$ is **time-constructible** if the function $1^n \mapsto 1^{t(n)}$ can be computed in time $O(t(n))$. For such $t$, every language that is recognized by a $t(n)$-time-bounded probabilistic Turing machine is recognized by an $O(t(n))$-time-bounded probabilistic Turing machine that has error probability strictly less than $\frac{1}{2}$. The latter assertion follows by essentially the same proof as Lemma 111.

</div>

This lemma allows us to prove an important closure property for $\text{PP}$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name"></span></p>

The complexity class $\text{PP}$ is closed under complement.

</div>

**Proof.**: By Lemma 111, every language $L$ in $\text{PP}$ is recognized by a polynomially time-bounded probabilistic Turing machine that has error probability strictly less than $\frac{1}{2}$. 

$$
\text{P}[M \text{ accepts } w \mid w \notin L] < \frac{1}{2} \implies \text{P}[M \text{ rejects } w \mid w \notin L] > \frac{1}{2}
$$

$$
\text{P}[M \text{ accepts } w \mid w \in L] > \frac{1}{2} \implies \text{P}[M \text{ rejects } w \mid w \in L] < \frac{1}{2}
$$

Swapping acceptance and rejection in this machine yields a polynomially time-bounded probabilistic Turing machine that recognizes the complement of $L$. $\square$

It is not known whether the complexity class $\text{PP}$ is closed under union or intersection. However, it is closed under another important set operation.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name"></span></p>

The complexity class $\text{PP}$ is closed under symmetric difference.

</div>

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

hence $\tilde{M}$ recognizes the symmetric difference of $L$ and $L'$. $\square$

Like $\text{NP}$ has $\text{SAT}$, $\text{PP}$ also has a natural complete problem.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The majority satisfiability problem)</span></p>

The **majority satisfiability problem** is the language

$$
\text{MAJ} = \lbrace \phi : \phi \text{ is a propositional formula in conjunctive normal form that is satisfied by strictly more than half of the assignments to its variables}\rbrace
$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(The majority satisfiability problem)</span></p>

The majority satisfiability problem $\text{MAJ}$ is complete for $\text{PP}$.

</div>

**Proof sketch.**:
First, to see that the language $\text{MAJ}$ is a member of $\text{PP}$, we specify a polynomially time-bounded probabilistic Turing machine $M$ that recognizes $\text{MAJ}$. $M$ rejects every input that is not a propositional formula in conjunctive normal form. Otherwise, in case its input is a formula $\phi$ in conjunctive normal form in $n$ variables, $M$ interprets the prefix of length $n$ of its random word in the natural way as an assignment to the variables of $\phi$ and accepts in case this assignment satisfies $\phi$. So the formula $\phi$ is in $\text{MAJ}$ if and only if it has acceptance probability strictly larger than $\frac{1}{2}$ if and only if it is in $L(M)$.

Second, to show that $\text{MAJ}$ is $\text{PP}$-hard, let $L$ be a language in $\text{PP}$. Fix some polynomially time-bounded probabilistic Turing machine $M$ that recognizes $L$. Similar to the construction in the proof of Cook’s Theorem, we construct a function $w \mapsto \phi_w$ computable in polynomial time such that $\phi_w$ is a propositional formula in conjunctive normal form and there is a one-to-one correspondence between satisfying assignments of $\phi_w$ and random words $z$ such that $M$ accepts on input $w$ and random word $z$. Consequently, $w$ is in $L$ if and only if $\phi$ is in $\text{MAJ}$. $\square$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

It is not known whether there are complete languages for $\text{BPP}$.

</div>

### Properties of the Class BPP

The defining feature of $\text{BPP}$ is its bounded error. A remarkable property of this class is that this error can be made arbitrarily small through a process called **probability amplification**.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Probability amplification)</span></p>

For a language $L$ the following assertions are equivalent:
* The language $L$ is in $\text{BPP}$.
* The language $L$ is recognized by a polynomially time-bounded probabilistic Turing machine with error probability at most $\frac{1}{2} - \epsilon$ for some $\epsilon > 0$.
* For every polynomial $p$ there is a polynomially time-bounded probabilistic Turing machine with error probability at most $2^{-p(n)}$ that recognizes $L$.

</div>

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

By the latter, fix a constant $c > 0$ such that $(1 - 4\epsilon^2)^c < \frac{1}{2}$, and given a polynomial $p$, let $t(n) = cp(n)$. Then $M'$ is polynomially time-bounded, has error probability at most $2^{-p(n)}$, because $(1 - 4\epsilon^2)^{cp(n)} < (\frac{1}{2})^{p(n)}=2^{-p(n)}$ and by construction recognizes $L$. $\square$

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

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name"></span></p>

The complexity class $\text{BPP}$ is closed under complement. The complexity class $\text{BPP}$ is closed under all binary set-theoretical operations, including union, intersection and symmetric difference.

</div>

**Proof.**: Closure under complement follows by swapping acceptance and rejection of a polynomially time-bounded probabilistic Turing machine with error probability at most $\frac{1}{3}$ that recognizes a given language in $\text{BPP}$.

In order to demonstrate the closure under binary set-theoretical operators, let $L$ and $L'$ be languages in $\text{BPP}$. By the third item in Theorem 117, choose polynomially time-bounded probabilistic Turing machines $M$ and $M'$ that both have error probability of at most $\frac{1}{6}$ and recognize $L$ and $L'$, respectively. Now consider a polynomially time-bounded probabilistic Turing machine that on input $w$ simulates $M$ and $M'$ on input $w$ and then applies the given operator to the results of the simulations, e.g., in the case of intersection, accepts if and only if both simulations accepted. The error probability of this probabilistic Turing machine is at most the sum of the error probabilities of $M$ and $M'$, hence is at most $\frac{1}{3}$. $\square$

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

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Advice Function and P/poly)</span></p>

An **advice function** is a function that maps natural numbers to binary words. A Turing machine $M$ with auxiliary tape recognizes a language $L$ with advice function $a(n)$, if $M$ is total and for every input $w$ of length $n$, $M$ reads at most the first $\lvert a(n) \rvert$ bits on its auxiliary tape, and accepts $w$ on auxiliary word $a(n)$ if and only if $w$ is in $L$. The complexity class $\text{P/poly}$ contains exactly the languages over the binary alphabet that are recognized by a polynomially time-bounded Turing machine with auxiliary tape with some advice function.

</div>

In the notation $\text{P/poly}$, the terms $\text{P}$ and $\text{poly}$ refer, respectively, to the polynomial time bound and to the fact that the advice functions $a$ that witness membership in $\text{P/poly}$ can always be chosen such that the advice $a(n)$ has length $p(n)$ for some polynomial $p$.

The power of probability amplification allows us to show that any $\text{BPP}$ language can be solved by a machine with polynomial advice.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name"></span></p>

The complexity class $\text{BPP}$ is a subset of $\text{P/poly}$.

</div>

**Proof.**: Let $L$ be a language in $\text{BPP}$. Let $M$ be a $p(n)$-time-bounded probabilistic Turing machine that recognizes $L$ with error probability $2^{-2n}$. We construct an advice function $a(n)$ such that $M$ recognizes $L$ with advice $a(n)$. Fix some length $n$ and consider inputs of length $n$ and their corresponding random words of length $p(n)$. Say a random word is **bad** for an input if this random word results in an error of $M$ with respect to deciding whether the input is in $L$. For each of the $2^n$ inputs, at most a fraction of $2^{-2n}$ of all random words are bad. Consequently, at most a fraction of $2^n \cdot 2^{-2n} = 2^{-n}$ of all random words is bad for some input of length $n$. So it suffices to let $a(n)$ be equal to some random word that is **good**, i.e., not bad, for all inputs of length $n$. $\square$

## The Polynomial Hierarchy and Complete Languages

This section explores the **polynomial hierarchy**, a hierarchy of complexity classes that generalizes the classes **$\text{P}$**, **$\text{NP}$**, and **co-$\text{NP}$**. It serves to classify problems that are not known to be in **$\text{NP}$** but are still solvable in polynomial space. We will also investigate the relationship between the probabilistic class **BPP** and this hierarchy, and conclude by examining methods for constructing complete languages for various complexity classes.

### The Polynomial-Time Hierarchy

The polynomial hierarchy is constructed using alternating existential and universal quantifiers, bounded by a polynomial in the input size.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Polynomial-Time Hierarchy)</span></p>

Let $k$ be a natural number. Then $\Sigma_k^p$ is the class of all languages $L$ over the binary alphabet for which there is a language $B$ in $\textbf{P}$ and a polynomial $p$ such that a binary word $w$ of length $n$ is in $L$ if and only if

$$
\exists_{y_1}^{p(n)} \forall_{y_2}^{p(n)} \exists_{y_3}^{p(n)} \cdots Q_k^{p(n)} y_k ((w , y_1, \ldots , y_k) \in B) \quad (4.1)
$$

where $Q_k$ is equal to $\exists$ if $k$ is odd and to $\forall$ if $k$ is even. Here $\exists_{y}^{\ell}$ is a short form of existential quantification over a binary word $y$ of length $\ell$, and similarly for $\forall^{\ell}$ and universal quantification.

The class $\Pi_k^p$ is defined literally the same except that (4.1) is replaced by

$$
\forall_{y_1}^{p(n)} \exists_{y_2}^{p(n)} \forall_{y_3}^{p(n)} \cdots Q_k^{p(n)} y_k ((w , y_1, \ldots , y_k) \in B) \quad (4.2)
$$

where now $Q_k$ is equal to $\forall$ if $k$ is odd and to $\exists$ if $k$ is even.

</div>

#### Levels and Properties of the Hierarchy

The hierarchy is built upon familiar complexity classes and possesses several key properties.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

It holds that $\Sigma_0^p = \Pi_0^p = \textbf{P}$ and $\Sigma_1^p = \textbf{NP}$. For every $k$, by definition a language is in $\Sigma_k^p$ if and only if the complement of the language is in $\Pi_k^p$. Furthermore, the classes $\Sigma_k^p$ and $\Pi_k^p$ are both subsets of $\Sigma_{k+1}^p$ and of $\Pi_{k+1}^p$, i.e., we have $\Sigma_k^p \cup \Pi_k^p \subseteq \Sigma_{k+1}^p \cap \Pi_{k+1}^p$.

</div>

The entire polynomial hierarchy, denoted **PH**, is the union of all its levels: $\text{PH} = \bigcup_k \Sigma_k^p$. This entire structure is contained within **PSPACE**.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name"></span></p>

The class **PH** is a subset of **PSPACE**.

</div>

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

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name"></span></p>

It holds that $\textbf{BPP} \subseteq \Sigma_2^p$, that is, for every language $L$ in **BPP** there is a language $B$ in **P** and a polynomial $p$ such that for all binary words $w$ it holds that

$$
w \in L \text{ if and only if } \exists_{x}^{p(\lvert w \rvert)} \forall_{y}^{p(\lvert w \rvert)} ((w, x, y) \in B).
$$

</div>

As a direct consequence of this theorem and the fact that **BPP** is closed under complement, we can place **BPP** more precisely within the hierarchy.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name"></span></p>

The class **BPP** is a subset of $\Sigma_2^p \cap \Pi_2^p$.

</div>

**Proof.** Let $L$ be a language in **BPP**. By Theorem 124, $L$ is in $\Sigma_2^p$. Since **BPP** is closed under complement, the complement language $\bar{L}$ is also in **BPP** and thus is in $\Sigma_2^p$. Consequently, by the definition of the hierarchy (Remark 121), the language $L$ is in $\Pi_2^p$. Therefore, $L$ is in $\Sigma_2^p \cap \Pi_2^p$, and the corollary follows. $\square$

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

This construction fulfills the requirements of Theorem 124. $\square$

## Constructing Complete Languages

A key technique for understanding complexity classes is to identify **complete languages**, which are the "hardest" problems in that class. We will first examine a complete language for **PP** and then discuss a generic method for constructing such languages.

### A Complete Language for $\text{PP}$


<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The Majority Satisfiability Problem)</span></p>

The **majority satisfiability problem** is the language

$$
\textbf{MAJ} = \lbrace \phi : \phi \text{ is a propositional formula in CNF that is satisfied by strictly more than half of the assignments to its variables}\rbrace
$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name"></span></p>

The majority satisfiability problem **MAJ** is complete for **PP**.

</div>
