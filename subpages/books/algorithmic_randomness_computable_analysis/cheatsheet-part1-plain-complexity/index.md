---
title: "Cheatsheet — Part 1: Foundations & Plain Kolmogorov Complexity"
layout: default
noindex: true
math: true
tags:
  - algorithmic-randomness
  - kolmogorov-complexity
  - computability
  - cantor-space
  - martin-löf-test
  - turing-machine
  - compressibility
  - cheatsheet
---

# Cheatsheet — Part 1: Foundations & Plain Kolmogorov Complexity

*Exam recap: definitions, formulas and key results only, covering everything up to (but excluding) §2.3 Prefix-Free Turing Machines. Full explanations live in the [main notes]({{ '/subpages/books/algorithmic_randomness_computable_analysis/' | relative_url }}).*

## Notation

* **Word:** $w \in \lbrace 0,1 \rbrace^{\ast}$, finite; **length** $l(w)$; empty word $\lambda$. ⚠️ $\lambda$ is overloaded — it is *both* the empty word and Lebesgue measure; read it off the context.
* **Binary representation:** $\;l(\mathrm{bin}(n)) = \lfloor \log\_2 n\rfloor + 1 = \log(n) + O(1)$.
* **Prefix** of an infinite sequence $A = a\_0a\_1a\_2\dots$:

$$A \upharpoonright n := a_0 \dots a_{n-1}.$$

* **Length-lexicographic enumeration** $\;(w\_0, w\_1, w\_2, \dots) = (\lambda, 0, 1, 00, 01, 10, 11, 000, \dots)$: order by length first, then lexicographically. There are $2^k$ words of length $k$, and the first index of a length-$k$ word is $2^k - 1$ — so a word of length $k$ has index $\ge 2^k - 1 \ge k$ (used in Thm. 1.3).

## Cantor space & Lebesgue measure

* **Cantor space** $\lbrace 0,1 \rbrace^{\omega}$ = infinite binary sequences. **Basic open set (cylinder)** for $\sigma \in \lbrace 0,1 \rbrace^{\ast}$:

$$[\![\sigma]\!] = \lbrace \sigma X : X \in \lbrace 0,1 \rbrace^{\omega} \rbrace \;\longleftrightarrow\; \text{dyadic interval } \bigl[0.\sigma,\; 0.\sigma + 2^{-l(\sigma)}\bigr] \subseteq [0,1].$$

* **Measure of a cylinder:** $\;\lambda([[\sigma]]) = 2^{-l(\sigma)}$ — each extra bit halves it.
* **Open set** = (countable) union of *pairwise disjoint* cylinders, with

$$S = ([\![\sigma_0]\!], [\![\sigma_1]\!], \dots), \qquad \lambda(S) = \sum_i 2^{-l(\sigma_i)}.$$

## The three intuitions of nonrandomness

| Intuition | "Nonrandom" means | Formal tool |
| :-- | :-- | :-- |
| **Compressibility** (§1.1) | has an essentially *shorter description* | Kolmogorov complexity $C$ |
| **Predictability** (§1.2) | a gambler can *bet* on its bits and win | martingales |
| **Untypicality** (§1.3) | lies in *effectively rare* sets | Martin-Löf tests |

**Running examples** (finite: length 2000):

| | finite | infinite | what is wrong |
| :-- | :-- | :-- | :-- |
| $X$ | $(01)^{1000}$ | $\tilde X = (01)^{\omega}$ | perfectly periodic |
| $Y$ | at most $200$ ones | $\tilde Y$: $\;b\_1 + \dots + b\_n \le n/5\;$ for $n \ge 5$ | skewed bit density |
| $Z$ | $z\_1 0\, z\_2 0 \dots z\_{1000} 0$ | $\tilde Z = z\_1 0\, z\_2 0\, z\_3 0 \dots$ | every second bit forced to $0$ |

## 1.1 Compressibility → Kolmogorov complexity

* **Complexity w.r.t. a machine $M$:**

$$C_M(\sigma) = \min\lbrace l(\tau) : M(\tau) = \sigma \rbrace \quad (\text{possibly undefined}).$$

* **Universal machine $U$:** for every $M$ there is a code $\rho\_M$ with $\;U(\rho\_M \tau) = M(\tau)\;$ for all $\tau$.
* **Kolmogorov complexity** = complexity w.r.t. a *fixed* universal $U$; it is a function $C : \lbrace 0,1 \rbrace^{\ast} \to \mathbb{N}$:

$$C(\sigma) = C_U(\sigma) = \min\lbrace l(\rho_M) + l(\tau) : M(\tau) = \sigma \rbrace.$$

* **Invariance / universality up to a constant** — the workhorse inequality, used in *every* upper-bound proof below:

$$\boxed{\,C(\sigma) \le C_M(\sigma) + c_M, \qquad c_M = l(\rho_M) \text{ depends on } M, \text{ not on } \sigma.\,} \tag{1}$$

  Recipe for any upper bound: **build a machine $M$, count the bits of its input, add $c\_M$.**

**Examples.** With $M(\mathrm{bin}(n)) = (01)^n$ and $L(s\_1\dots s\_n) = s\_1 0\, s\_2 0 \dots s\_n 0$:

$$C(X) \le 10 + c_M, \qquad C(Z) \le 1000 + c_L,$$

$$C(\tilde X \upharpoonright 2n) \le \log(n) + c_M, \qquad C(\tilde Z \upharpoonright 2n) \le n + c_L \qquad (c_M, c_L \text{ independent of } n).$$

### Why compressibility of prefixes fails as a definition of randomness

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(1.3 — Martin-Löf, 1966)</span></p>

For **every** infinite sequence $A$ and **every** constant $c$ there are **infinitely many** $n$ with

$$C(A \upharpoonright n) < n - c.$$

</div>

* **Proof idea:** enumerate $S$ in stages — at stage $n$ put in all words of length $n$ starting with $w\_n$. A machine mapping $\mathrm{bin}(n) \mapsto \sigma\_n$ gives $C(\sigma\_n) \le \log(n) + c\_M$, while $l(\sigma\_n) - \log(n) \to \infty$. Every $A$ meets $S$ infinitely often (index of $A\upharpoonright k$ is $\ge k$).
* **Consequence:** *no* sequence has all prefixes incompressible ⇒ "all prefixes are incompressible" is **vacuous** as a randomness notion for plain $C$. This is the first hint that plain complexity is the wrong tool and prefix-free complexity (§2.3) is needed.
* **Exercise 1.4** (sharpening): infinitely many $n$ with $\;C(X \upharpoonright n) \le n - \log(n) + O(1)$.

## 1.2 Predictability

A sequence is nonrandom if a gambler betting one dollar per bit gains **unbounded** capital.

* $\tilde X$: *every* bit is predictable ($X(2i) = 0$, $X(2i+1) = 1$).
* $\tilde Z$: every second bit is predictable ($Z(2i+1) = 0$).
* $\tilde Y$: always bet on $0$ — the ones are too sparse to hurt (for $Y$: $\approx 1800$ dollars profit over 2000 bits).

Formalized later via **martingales**.

## 1.3 Untypicality → Martin-Löf tests

**Finite:** a rare, easily describable and checkable property makes a word nonrandom.

* $X$ is one of a kind; $Z$'s property (even positions zero) is shared by $2^{1000}$ of $2^{2000}$ words; $Y$'s by

$$\binom{2000}{\le 200} \approx 1.1\cdot\binom{2000}{200} \approx 2^{938} \quad\text{of}\quad 2^{2000}.$$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(1.5 — Martin-Löf test)</span></p>

A **Martin-Löf test** $\mathcal{M} = (M\_0, M\_1, \dots)$ is a **uniformly enumerable** family of open sets ("layers") $M\_i = ([[\sigma\_0^i]], [[\sigma\_1^i]], \dots)$ with

$$\mu(M_i) = \sum_j 2^{-l(\sigma_j^i)} \le 2^{-i} \qquad \text{for every } i. \tag{2}$$

$A$ is **ML-nonrandom** iff some test **covers** it, i.e. $A \in M\_i$ for *every* $i$. Otherwise $A$ is **ML-random**.

</div>

Two moving parts, both essential: **effectivity** (uniformly c.e.) and **shrinking measure** ($\le 2^{-i}$).

**Example 1.6.** Covering $\tilde X$: $\;M\_i = ([[(01)^i]])$, so $\mu(M\_i) = 2^{-2i} \le 2^{-i}$. Covering $\tilde Z$:

$$M_i = \lbrace [\![a_1 0\, a_2 0 \dots a_i 0]\!] : a_1 \dots a_i \in \lbrace 0,1 \rbrace^{i} \rbrace, \qquad \lambda(M_i) = 2^i \cdot 2^{-2i} = 2^{-i}.$$

* **Pattern:** $2^i$ cylinders of measure $2^{-2i}$ each — the free bits cost measure, the forced bits buy it back.
* **Exercise 1.6:** build a test covering every $X$ that avoids the substring $111$.

## 2 Plain complexity: basic bounds

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(2.1 — Trivial upper bound)</span></p>

$$\exists c\ \forall w: \quad C(w) \le l(w) + c.$$

</div>

Proof: identity machine, $C\_{\mathrm{id}}(w) = l(w)$, then (1). *Never harder than writing the word out.*

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(2.2 — Computable processes do not increase complexity)</span></p>

For every computable $f :\subseteq \lbrace 0,1 \rbrace^{\ast} \to \lbrace 0,1 \rbrace^{\ast}$ there is $c\_f$ with

$$C(f(w)) < C(w) + c_f \qquad \text{for all } w \in \mathrm{dom}(f).$$

</div>

Proof: run $M := f \circ U$. The optimal code $\sigma\_w$ for $w$ already describes $f(w)$. **The constant depends on the process, never on the input.**

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(2.3 — Counting bound)</span></p>

$$\forall n: \quad \#\lbrace w : C(w) < n \rbrace \le 2^n - 1 < 2^n.$$

</div>

Proof: only $2^n - 1$ words of length $< n$ exist, and each is the optimal code of at most one word. **This is the only lower-bound tool in the chapter** — *pigeonhole on programs*; every "complexity is large" claim below (2.13, 2.16) goes through it.

* **Corollary 2.3.1 (layered counting):** among the $2^n$ words of length $n$, at most $2^{n-k+1} - 1$ have $C \le n - k$, for $k = 1, \dots, n$ — i.e. $\le n-1$: at most $2^n - 1$; $\le n-2$: at most $2^{n-1} - 1$; …; $\le 0$: at most $2^1 - 1$. In particular at least one word of each length $n$ has $C(w) \ge n$ — **incompressible words exist at every length**.

**Average complexity.** $\;g(n) := 2^{-n}\sum\_{l(w) = n} C(w)$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(2.4 — Average complexity is asymptotically the length)</span></p>

$$\frac{g(n)}{n} \xrightarrow[n\to\infty]{} 1.$$

</div>

* **Upper** side: Prop. 2.1 gives $g(n) \le n + c$.
* **Lower** side: spread the $2^n$ words over complexity strata via Cor. 2.3.1 and use $\sum\_{i \ge 1} i\,2^{-i} = 2$:

$$\frac{g(n)}{n} \ge 1 - 2^{-n} - \frac{1}{n}\sum_{i=1}^{n} i\,2^{-i} \;\ge\; 1 - 2^{-n} - \frac{2}{n} \longrightarrow 1.$$

* **Exercise 2.5** (sharper, constant-width band): $\;n - c\_m \le g(n) \le n + c\_M$, with $c\_m = 2 + \varepsilon$ workable.
* **Moral (bridge to §2.1):** most words of length $n$ are essentially incompressible — compressibility (Thm. 2.2) is the exception, not the rule. Which raises the effective question: *can incompressibility of a given word be checked algorithmically?* Answer below: essentially **no**.

### Two named exercises

* **Axiomatic characterization of $C$.** $(M\_0, M\_1, \dots)$ is **uniformly c.e.** if $\lbrace (x,i) : x \in M\_i \rbrace$ is c.e. Put $S\_n := \lbrace w : C(w) < n \rbrace$ (so $\lvert S\_n\rvert < 2^n$ by Prop. 2.3).
  * **(a)** $(S\_n)$ is uniformly c.e. — dovetail $U$ over all programs of length $< n$ and all $n$.
  * **(b)** For **every** uniformly c.e. $(V\_n)$ with $\lvert V\_n\rvert < 2^n$ there is $c$ with $\;C(w) < n + c$ for all $w \in V\_n$.
  * Proof of (b): let the program be the $n$-bit **index** of $w$ inside $V\_n$; the machine reads $n := l(p)$ off the *input length* (a plain machine sees its whole input), enumerates $V\_n$ to the $p$-th element. So $C\_M(w) \le n$, and (1) gives $c := c\_M + 1$.
  * **Punchline:** (b) says every admissible family sits inside a shifted $S$, i.e. $V\_n \subseteq S\_{n+c}$ — so $(S\_n)$ is the *largest* such family, equivalently $C$ is the *smallest* such complexity measure, up to an additive constant. This characterizes $C$ **without mentioning Turing machines**: it is the least function whose sublevel sets are uniformly c.e. and of size $< 2^n$.
* **Rate of growth.** $\;B(n) := \max\lbrace m \in \mathbb{N} : C(m) \le n \rbrace$.
  * **(a)** $B$ is **total**: at most $2^{n+1} - 1$ programs of length $\le n$, so the set is finite and has a maximum. *Totality comes from finiteness, not computability.*
  * **(b)** $B$ **grows faster than every partially computable** $f$: $\;f(n) \le B(n)$ for all but finitely many $n \in \mathrm{dom}(f)$, since $C(f(n)) \le C(n) + c\_f \le \log(n) + \tilde c < n$ eventually. Busy-beaver flavour: well-defined, not computable.

## 2.1 $C$ is upper-semicomputable (and nothing more)

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(2.6–2.7 — Computability and semicomputability via sets)</span></p>

| $f$ is … | iff this set is c.e. | name |
| :-- | :-- | :-- |
| **computable** | $G\_f = \lbrace (w, f(w)) \rbrace$ | graph |
| **lower**-semicomputable | $H\_f = \lbrace (w,y) : f(w) > y \rbrace$ | hypograph |
| **upper**-semicomputable | $E\_f = \lbrace (w,y) : f(w) < y \rbrace$ | epigraph |

</div>

* **Prop. 2.8:** computable $\iff$ lower- **and** upper-semicomputable. (Enumerate $H\_f, E\_f$ in parallel until some $y$ has $(w, y-1) \in H\_f$ and $(w,y+1) \in E\_f$, forcing $f(w) = y$.)
* **Prop. 2.9 (approximation form):** $f$ is upper-semicomputable $\iff$ there is a computable $F : \lbrace 0,1 \rbrace^{\ast}\times\mathbb{N} \to \mathbb{N}$ with

$$F(w,0) \ge F(w,1) \ge \dots \quad\text{and}\quad \lim_{n\to\infty} F(w,n) = f(w).$$

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(2.10 — $C$ is upper-semicomputable)</span></p>

$$F(w,t) := \min\lbrace l(\sigma) : U(\sigma)[t]\!\downarrow\, = w \rbrace$$

is computable, nonincreasing in $t$, and $\to C(w)$.

</div>

**Decreasing staircase:** run $U$ longer, discover shorter programs. At any finite $t$ you hold only an **upper bound** — you can never certify that no shorter program will still halt.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(2.11 — No computable unbounded lower bound)</span></p>

There is no computable $f :\subseteq \lbrace 0,1 \rbrace^{\ast} \to \mathbb{N}$ that is **unbounded** and satisfies $C(w) > f(w)$ on $\mathrm{dom}(f)$.

</div>

* **Proof (Berry-paradox shape):** let $M(\mathrm{bin}(n))$ dovetail $f$ over all words and output the first $\sigma$ with $f(\sigma)\!\downarrow\, > n$. Then

$$n < f(M(\mathrm{bin}(n))) < C(M(\mathrm{bin}(n))) \;\overset{\text{Thm. 2.2}}{\le}\; C(\mathrm{bin}(n)) + c_M \le \log(n) + c_M,$$

  false for $n > 2^{c\_M + 1}$.
* **Corollary 2.11.1:** $C$ is **not computable** (apply the theorem to $f(w) := C(w) - 1$).
* **The asymmetry in one line:** you can always *certify compressibility* (find a short program), never *certify incompressibility*.

## 2.2 Properties of plain $C$

### Stability under edits

| Edit | Effect | Result |
| :-- | :-- | :-- |
| add / change / delete the **last** bit | $\pm c$ for a fixed $c$ | Prop. 2.12 |
| insert / change / delete **one interior** bit | can raise $C$ by more than any $n$ | Prop. 2.13 |

**Prop. 2.13 construction.** Take the highly compressible $w\_n := (01)^{2^n}$, so $C(w\_n) \le \log(n) + c\_M$, and the family

$$W_n^{+} = \lbrace (01)^k 1 (01)^{2^n - k} : 0 \le k \le 2^n \rbrace \qquad (2^n + 1 \text{ distinct words}).$$

By the counting bound at most $2^n - 1$ words have $C < n$, so some $w\_n^{+} \in W\_n^{+}$ has $C(w\_n^{+}) \ge n$, hence

$$C(w_n^{+}) - C(w_n) \ge n - \log(n) - c_M \xrightarrow[n\to\infty]{} \infty.$$

Cases (ii) change and (iii) delete are analogous with $W\_n^{c} = \lbrace (01)^{k-1}11(01)^{2^n-k}\rbrace$ and $W\_n^{-} = \lbrace (01)^{k-1}1(01)^{2^n-k}\rbrace$.

### Concatenation: the parsing overhead

The obstruction: in a plain code $\tau\_x\tau\_y$ the decoder cannot tell **where the first code ends**.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(2.14 — Upper bound for concatenation)</span></p>

$$\boxed{\,C(xy) \le C(x) + 2\log C(x) + C(y) + c\,}$$

</div>

* **Construction:** feed $M$ the input $\;\mathrm{bin}(l(\tau\_x))\,01\,\tau\_x\tau\_y$; the separator $01$ makes the length field parseable, then split off the first $l(\tau\_x)$ bits and output $U(\tau\_1)U(\tau\_2)$.
* Where the **factor 2** comes from: the length field must itself be self-delimiting, since a bare $01$ can also occur *inside* $\mathrm{bin}(l(\tau\_x))$. The standard fix — double every bit of the length field, then terminate with $01$ — costs $2\log C(x) + 2$ bits, which is exactly the overhead term.
* **Exercise 2.15** (iterated-log refinement, for each fixed $n$):

$$C(xy) \le C(x) + \log C(x) + \log\log C(x) + \cdots + \underbrace{\log\cdots\log}_{n-1} C(x) + 2\underbrace{\log\cdots\log}_{n} C(x) + C(y) + c_n$$

  — encode the length, then the length of the length, and so on.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(2.16 — The overhead cannot be eliminated)</span></p>

For every $n$ there are words $x, y$ with

$$C(xy) > C(x) + \log C(x) + C(y) + n.$$

</div>

* **Proof sketch:** reuse the set $S$ from Thm. 1.3, whose members satisfy $C(\sigma) \le l(\sigma) - \log l(\sigma) + O(1)$. Take $w$ with $C(w) \ge l(w)$ (counting bound), let $x$ be a long prefix of $w$ lying in $S$, and $w = xy$. Then $C(x) + C(y)$ loses $\log l(x)$ against $l(w) \le C(xy)$.
* ⚠️ The lecture PDF's closing line of this proof is a **typo** — as printed, $\log l(x) - \log(l(x) - \log l(x)) \to 0$, not $\infty$. The intended claim (a logarithmic additive term is necessary) still stands; check the inequality against the lecturer's version before quoting it.

## One-glance summary of the bounds

| Result | Statement | Tool |
| :-- | :-- | :-- |
| (1) Invariance | $C(\sigma) \le C\_M(\sigma) + c\_M$ | universal machine |
| 2.1 | $C(w) \le l(w) + c$ | identity machine |
| 2.2 | $C(f(w)) < C(w) + c\_f$, $f$ computable | compose $f \circ U$ |
| 2.3 | $\lvert\lbrace w : C(w) < n\rbrace\rvert < 2^n$ | pigeonhole on programs |
| 2.4 | $g(n)/n \to 1$ | 2.1 + 2.3 |
| 2.10 | $C$ upper-semicomputable | run $U$ for $t$ steps |
| 2.11 / 2.11.1 | no computable unbounded lower bound; $C$ noncomputable | Berry paradox + 2.2 |
| 1.3 | $C(A\upharpoonright n) < n - c$ infinitely often | stagewise enumeration |
| 2.12 / 2.13 | last-bit edits cost $\pm c$; interior edits are unbounded | 2.3 on an edit family |
| 2.14 / 2.16 | $C(xy) \le C(x) + 2\log C(x) + C(y) + c$, and $\log$ is necessary | self-delimiting length field |

**Punchline.** Plain complexity $C$ is well-defined up to an additive constant, matches the length on average, and is approximable **only from above** — so it can be witnessed but never refuted algorithmically. Its two structural defects both trace back to the same source, that a plain machine's input is *not self-delimiting*: prefixes of every sequence dip below $n - c$ infinitely often (Thm. 1.3, killing prefix-compressibility as a randomness notion), and concatenation carries an unavoidable logarithmic parsing overhead (Prop. 2.16). Fixing that source is exactly the job of the prefix-free machines in §2.3.
