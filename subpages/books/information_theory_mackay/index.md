---
title: Information Theory, Inference, and Learning Algorithms
layout: default
noindex: true
---

# Information Theory, Inference, and Learning Algorithms

## About Chapter 1

### The Binomial Distribution

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Binomial Distribution)</span></p>

A bent coin has probability $f$ of coming up heads. The coin is tossed $N$ times. The number of heads $r$ has a **binomial distribution**:

$$P(r \mid f, N) = \binom{N}{r} f^r (1-f)^{N-r}.$$

The mean and variance are:

$$\mathcal{E}[r] \equiv \sum_{r=0}^{N} P(r \mid f, N) \, r = Nf, \qquad \text{var}[r] = Nf(1-f).$$

These follow from the fact that $r$ is the sum of $N$ independent Bernoulli random variables, each with mean $f$ and variance $f(1-f)$.

</div>

### Stirling's Approximation

Starting from the Poisson distribution $P(r \mid \lambda) = e^{-\lambda} \frac{\lambda^r}{r!}$ and using its Gaussian approximation for large $\lambda$, we can derive **Stirling's approximation** for the factorial function:

$$x! \simeq x^x \, e^{-x} \sqrt{2\pi x} \quad \Leftrightarrow \quad \ln x! \simeq x \ln x - x + \tfrac{1}{2} \ln 2\pi x.$$

Applying Stirling's approximation to the binomial coefficient:

$$\ln \binom{N}{r} \equiv \ln \frac{N!}{(N-r)!\, r!} \simeq (N-r) \ln \frac{N}{N-r} + r \ln \frac{N}{r}.$$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Binary Entropy Function)</span></p>

The **binary entropy function** is defined as:

$$H_2(x) \equiv x \log \frac{1}{x} + (1-x) \log \frac{1}{1-x},$$

where $\log$ denotes $\log_2$. Using this, the binomial coefficient can be approximated as:

$$\log \binom{N}{r} \simeq N H_2(r/N) \qquad \Leftrightarrow \qquad \binom{N}{r} \simeq 2^{N H_2(r/N)}.$$

A more accurate version includes the next-order correction:

$$\log \binom{N}{r} \simeq N H_2(r/N) - \tfrac{1}{2} \log \left[ 2\pi N \, \frac{N-r}{N} \, \frac{r}{N} \right].$$

</div>

## Chapter 1 -- Introduction to Information Theory

### 1.1 Communication over a Noisy Channel

The fundamental problem of communication is reproducing at one point either exactly or approximately a message selected at another point (Shannon, 1948). Examples of noisy communication channels include telephone lines, radio links, DNA replication, and disk drives. In all these cases, if we transmit data over the channel, there is some probability that the received message will differ from the transmitted message.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Binary Symmetric Channel)</span></p>

A **binary symmetric channel** (BSC) transmits each bit correctly with probability $(1-f)$ and incorrectly (flipped) with probability $f$, called the **noise level**:

$$P(y=0 \mid x=0) = 1-f, \quad P(y=0 \mid x=1) = f,$$

$$P(y=1 \mid x=0) = f, \quad P(y=1 \mid x=1) = 1-f.$$

</div>

There are two approaches to achieving reliable communication:
- **The physical solution:** improve the physical characteristics of the channel (more reliable components, higher-power signals, etc.). This increases cost.
- **The system solution:** accept the noisy channel and add communication *systems* -- an **encoder** before the channel and a **decoder** after it. The encoder adds **redundancy** to the source message $\mathbf{s}$, producing a transmitted message $\mathbf{t}$. The channel adds noise, yielding a received message $\mathbf{r}$. The decoder uses the known redundancy to infer the original signal.

**Information theory** is concerned with the theoretical limitations and potentials of such systems. **Coding theory** is concerned with the creation of practical encoding and decoding systems.

### 1.2 Error-Correcting Codes for the Binary Symmetric Channel

#### Repetition Codes

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Repetition Code $R_N$)</span></p>

A **repetition code** $R_N$ repeats every bit of the message $N$ times. For example, $R_3$ maps $0 \to 000$ and $1 \to 111$. The **rate** of the code is $R = 1/N$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Majority-Vote Decoding for $R_3$)</span></p>

The optimal decoder for $R_3$ (assuming a BSC and equal prior probabilities) examines each triplet of received bits and takes a **majority vote**. This is equivalent to choosing the hypothesis $\hat{s}$ that maximizes the **likelihood ratio**:

$$\frac{P(\mathbf{r} \mid s=1)}{P(\mathbf{r} \mid s=0)} = \prod_{n=1}^{N} \frac{P(r_n \mid t_n(1))}{P(r_n \mid t_n(0))},$$

where each factor equals $\gamma = \frac{1-f}{f}$ if $r_n = 1$ and $\frac{1}{\gamma}$ if $r_n = 0$. Since $\gamma > 1$ for $f < 0.5$, the hypothesis with the most "votes" wins.

</div>

The error probability of $R_3$ is dominated by two or more bits being flipped in a block of three:

$$p_\text{b} = p_\text{B} = 3f^2(1-f) + f^3 = 3f^2 - 2f^3.$$

For $f = 0.1$, the $R_3$ code gives $p_\text{b} \simeq 0.03$, reducing the error rate but also reducing the communication rate to $1/3$. To achieve $p_\text{b} \simeq 10^{-15}$, roughly $N \simeq 61$ repetitions are needed.

#### Block Codes -- the (7, 4) Hamming Code

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Block Code)</span></p>

A **block code** converts a sequence of source bits $\mathbf{s}$ of length $K$ into a transmitted sequence $\mathbf{t}$ of length $N$ bits, with $N > K$. In a **linear** block code, the extra $N - K$ bits are linear functions of the original $K$ bits; these are called **parity-check bits**.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">((7, 4) Hamming Code)</span></p>

The **(7, 4) Hamming code** transmits $N = 7$ bits for every $K = 4$ source bits, giving a rate $R = 4/7$. The first four transmitted bits $t_1 t_2 t_3 t_4$ are the source bits $s_1 s_2 s_3 s_4$. The three parity-check bits $t_5 t_6 t_7$ enforce even parity within three overlapping groups of bits.

The transmitted codeword is obtained via a linear operation $\mathbf{t} = \mathbf{G}^\mathsf{T} \mathbf{s}$ (modulo 2), where $\mathbf{G}^\mathsf{T}$ is the **generator matrix**:

$$\mathbf{G}^\mathsf{T} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \\ 1 & 1 & 1 & 0 \\ 0 & 1 & 1 & 1 \\ 1 & 0 & 1 & 1 \end{bmatrix}.$$

Any pair of the $16$ codewords differ from each other in at least three bits.

</div>

#### Decoding the (7, 4) Hamming Code -- Syndrome Decoding

The optimal decoder identifies the source vector $\mathbf{s}$ whose encoding $\mathbf{t}(\mathbf{s})$ differs from the received vector $\mathbf{r}$ in the fewest bits. A more efficient method uses **syndrome decoding**.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Syndrome and Parity-Check Matrix)</span></p>

The **syndrome** $\mathbf{z}$ is the pattern of parity-check violations. Writing $\mathbf{G}^\mathsf{T} = \begin{bmatrix} \mathbf{I}_4 \\ \mathbf{P} \end{bmatrix}$, the **parity-check matrix** is:

$$\mathbf{H} = \begin{bmatrix} \mathbf{P} & \mathbf{I}_3 \end{bmatrix} = \begin{bmatrix} 1 & 1 & 1 & 0 & 1 & 0 & 0 \\ 0 & 1 & 1 & 1 & 0 & 1 & 0 \\ 1 & 0 & 1 & 1 & 0 & 0 & 1 \end{bmatrix}.$$

All codewords $\mathbf{t} = \mathbf{G}^\mathsf{T} \mathbf{s}$ satisfy $\mathbf{H}\mathbf{t} = \mathbf{0}$. Given a received vector $\mathbf{r} = \mathbf{G}^\mathsf{T}\mathbf{s} + \mathbf{n}$, the syndrome is $\mathbf{z} = \mathbf{H}\mathbf{r} = \mathbf{H}\mathbf{n}$, and the decoding problem reduces to finding the most probable noise vector $\mathbf{n}$ satisfying $\mathbf{H}\mathbf{n} = \mathbf{z}$.

</div>

For the (7, 4) Hamming code with small noise level $f$, the optimal decoder:
- If the syndrome is all-zero: no action (the received vector is a codeword).
- Otherwise: unflip the single bit identified by the syndrome. Each of the seven non-zero syndromes maps uniquely to one of the seven bit positions.

The **probability of block error** $p_\text{B} = P(\hat{\mathbf{s}} \neq \mathbf{s})$ is the probability that two or more bits are flipped in a block of seven. The **probability of bit error** $p_\text{b} = \frac{1}{K} \sum_{k=1}^{K} P(\hat{s}_k \neq s_k)$ scales as:

$$p_\text{B} \simeq \binom{7}{2} f^2 = 21f^2, \qquad p_\text{b} \simeq \frac{3}{7} p_\text{B} \simeq 9f^2.$$

The Hamming code is completely symmetric in the protection it affords to all seven bits (source and parity bits alike). This can be proven by showing that the parity-check matrix has a **cyclic** structure: every row is a cyclic permutation of the top row.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Cyclic Code)</span></p>

A linear code is a **cyclic code** if there is an ordering of the bits $t_1 \ldots t_N$ such that the parity-check matrix is cyclic. The codewords of a cyclic code have the property that any cyclic permutation of a codeword is also a codeword.

</div>

#### Graphs Corresponding to Codes

Many codes can be conveniently expressed in terms of **bipartite graphs**. The (7, 4) Hamming code can be represented by a graph where 7 circles are the bit nodes and 3 squares are the parity-check nodes. Each parity-check node is connected to the bit nodes that participate in that parity constraint. The graph and the parity-check matrix $\mathbf{H}$ are directly related: each parity-check node corresponds to a row of $\mathbf{H}$, and each bit node to a column; for every 1 in $\mathbf{H}$ there is an edge between the corresponding nodes.

### 1.3 What Performance Can the Best Codes Achieve?

There appears to be a trade-off between the bit-error probability $p_\text{b}$ (which we want small) and the rate $R$ (which we want large). Shannon (1948) proved the remarkable result that the boundary between achievable and non-achievable points in the $(R, p_\text{b})$ plane meets the $R$ axis at a non-zero value $R = C$, the **capacity** of the channel.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Shannon's Noisy-Channel Coding Theorem -- Informal)</span></p>

For any channel, there exist codes that make it possible to communicate with **arbitrarily small** probability of error $p_\text{b}$ at non-zero rates. The maximum rate at which communication is possible with arbitrarily small $p_\text{b}$ is called the **capacity** $C$ of the channel.

For a binary symmetric channel with noise level $f$:

$$C(f) = 1 - H_2(f) = 1 - \left[ f \log_2 \frac{1}{f} + (1-f) \log_2 \frac{1}{1-f} \right].$$

For example, with $f = 0.1$, the capacity is $C \simeq 0.53$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Implications of Shannon's Theorem)</span></p>

Shannon's theorem asserts both **limitations** and **possibilities**:
- Reliable communication at any rate beyond the capacity is **impossible**.
- Reliable communication at all rates up to the capacity **is possible**.

For a BSC with $f = 0.1$ and $C \simeq 0.53$: instead of needing 60 noisy disk drives (using repetition codes) to build one reliable drive, Shannon's theorem says just two noisy drives suffice (since $1/2 < 0.53$), with *arbitrarily small* error probability -- provided the blocklength is large enough.

</div>

### 1.4 Summary

- The **(7, 4) Hamming Code:** by including three parity-check bits in a block of 7 bits, it is possible to detect and correct any single bit error in each block.
- **Shannon's noisy-channel coding theorem:** information can be communicated over a noisy channel at a non-zero rate with arbitrarily small error probability. The next chapters lay the foundations for this result by discussing how to *measure information content* and the related topic of *data compression*.

## Chapter 2 -- Probability, Entropy, and Inference

### 2.1 Probabilities and Ensembles

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Ensemble)</span></p>

An **ensemble** $X$ is a triple $(x, \mathcal{A}_X, \mathcal{P}_X)$, where the outcome $x$ is the value of a random variable taking on one of a set of possible values $\mathcal{A}_X = \lbrace a_1, a_2, \ldots, a_I \rbrace$, having probabilities $\mathcal{P}_X = \lbrace p_1, p_2, \ldots, p_I \rbrace$, with $P(x = a_i) = p_i$, $p_i \ge 0$, and $\sum_{a_i \in \mathcal{A}_X} P(x = a_i) = 1$.

</div>

**Probability of a subset.** If $T$ is a subset of $\mathcal{A}_X$:

$$P(T) = P(x \in T) = \sum_{a_i \in T} P(x = a_i).$$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Joint Ensemble)</span></p>

A **joint ensemble** $XY$ is an ensemble in which each outcome is an ordered pair $x, y$ with $x \in \mathcal{A}_X = \lbrace a_1, \ldots, a_I \rbrace$ and $y \in \mathcal{A}_Y = \lbrace b_1, \ldots, b_J \rbrace$. We call $P(x, y)$ the **joint probability** of $x$ and $y$. The two variables are not necessarily independent.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Rules of Probability)</span></p>

**Marginal probability:**

$$P(x = a_i) \equiv \sum_{y \in \mathcal{A}_Y} P(x = a_i, y), \qquad P(y) \equiv \sum_{x \in \mathcal{A}_X} P(x, y).$$

**Conditional probability:**

$$P(x = a_i \mid y = b_j) \equiv \frac{P(x = a_i, y = b_j)}{P(y = b_j)} \quad \text{if } P(y = b_j) \neq 0.$$

**Product rule** (chain rule):

$$P(x, y \mid \mathcal{H}) = P(x \mid y, \mathcal{H}) P(y \mid \mathcal{H}) = P(y \mid x, \mathcal{H}) P(x \mid \mathcal{H}).$$

**Sum rule:**

$$P(x \mid \mathcal{H}) = \sum_y P(x, y \mid \mathcal{H}) = \sum_y P(x \mid y, \mathcal{H}) P(y \mid \mathcal{H}).$$

**Bayes' theorem:**

$$P(y \mid x, \mathcal{H}) = \frac{P(x \mid y, \mathcal{H}) P(y \mid \mathcal{H})}{P(x \mid \mathcal{H})} = \frac{P(x \mid y, \mathcal{H}) P(y \mid \mathcal{H})}{\sum_{y'} P(x \mid y', \mathcal{H}) P(y' \mid \mathcal{H})}.$$

**Independence:** $X$ and $Y$ are independent ($X \perp Y$) iff $P(x, y) = P(x)P(y)$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Medical Test -- Bayes' Theorem)</span></p>

Jo has a test for a disease. The test is 95% reliable: $P(b{=}1 \mid a{=}1) = 0.95$, $P(b{=}0 \mid a{=}0) = 0.95$. The disease prevalence is $P(a{=}1) = 0.01$. Jo tests positive ($b = 1$). By Bayes' theorem:

$$P(a{=}1 \mid b{=}1) = \frac{P(b{=}1 \mid a{=}1) P(a{=}1)}{P(b{=}1 \mid a{=}1) P(a{=}1) + P(b{=}1 \mid a{=}0) P(a{=}0)} = \frac{0.95 \times 0.01}{0.95 \times 0.01 + 0.05 \times 0.99} = 0.16.$$

Despite the positive result, the probability of having the disease is only 16%.

</div>

### 2.2 The Meaning of Probability

Probabilities can be used in two ways:
1. To describe **frequencies of outcomes** in random experiments.
2. To describe **degrees of belief** in propositions that do not involve random variables.

Degrees of belief *can* be mapped onto probabilities if they satisfy simple consistency rules known as the **Cox axioms**. If a set of beliefs satisfies these axioms, they can be mapped onto probabilities satisfying $P(\text{false}) = 0$, $P(\text{true}) = 1$, $0 \le P(x) \le 1$, $P(x) = 1 - P(\overline{x})$, and $P(x, y) = P(x \mid y) P(y)$.

This more general use of probability to quantify beliefs is known as the **Bayesian viewpoint** (or *subjective* interpretation), since the probabilities depend on assumptions. Bayesians use probabilities to describe assumptions *and* to describe inferences given those assumptions.

### 2.3 Forward Probabilities and Inverse Probabilities

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Forward and Inverse Probability)</span></p>

- **Forward probability problems** involve a *generative model* that describes a process assumed to give rise to data; the task is to compute the probability distribution or expectation of some quantity that depends on the data.
- **Inverse probability problems** involve computing the conditional probability of one or more *unobserved variables* in the process, given the observed variables. This invariably requires Bayes' theorem.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Urn Problem -- Inverse Probability)</span></p>

There are eleven urns labelled $u \in \lbrace 0, 1, \ldots, 10 \rbrace$. Urn $u$ contains $u$ black balls and $10 - u$ white balls. Fred selects an urn $u$ at random and draws $N$ times with replacement, obtaining $n_B$ blacks. After $N = 10$ draws with $n_B = 3$ blacks, the posterior probability of $u$ given $n_B$ is:

$$P(u \mid n_B, N) = \frac{P(n_B \mid u, N) P(u)}{P(n_B \mid N)} = \frac{1}{P(n_B \mid N)} \cdot \frac{1}{11} \binom{N}{n_B} f_u^{n_B} (1 - f_u)^{N - n_B},$$

where $f_u \equiv u/10$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Terminology of Inverse Probability)</span></p>

In Bayes' theorem $P(\boldsymbol{\theta} \mid D, \mathcal{H}) = \frac{P(D \mid \boldsymbol{\theta}, \mathcal{H}) P(\boldsymbol{\theta} \mid \mathcal{H})}{P(D \mid \mathcal{H})}$, we identify:

$$\text{posterior} = \frac{\text{likelihood} \times \text{prior}}{\text{evidence}}.$$

- $P(u)$ is the **prior** probability.
- $P(n_B \mid u, N)$ is the **likelihood** (a function of $u$ for fixed data; *not* a probability distribution over $u$).
- $P(u \mid n_B, N)$ is the **posterior** probability.
- $P(n_B \mid N)$ is the **evidence** (or **marginal likelihood**).

Never say "the likelihood of the data." Always say "the likelihood of the parameters." The likelihood function is not a probability distribution.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Prediction by Marginalization)</span></p>

Given the posterior $P(u \mid n_B, N)$, the probability that the next drawn ball is black is obtained by **marginalizing** over the hypothesis $u$:

$$P(\text{ball}_{N+1} \text{ is black} \mid n_B, N) = \sum_u f_u \, P(u \mid n_B, N).$$

This is the correct predictive approach. It differs from the common (but inferior) practice of first selecting the most plausible hypothesis and then predicting as if that hypothesis were true. Marginalization produces slightly more moderate, less extreme predictions.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(The Likelihood Principle)</span></p>

Given a generative model for data $d$ given parameters $\boldsymbol{\theta}$, $P(d \mid \boldsymbol{\theta})$, and having observed a particular outcome $d_1$, all inferences and predictions should depend only on the function $P(d_1 \mid \boldsymbol{\theta})$. The details of other possible outcomes and their probabilities are irrelevant -- only the likelihood of the observed data matters.

</div>

### 2.4 Definition of Entropy and Related Functions

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Shannon Information Content)</span></p>

The **Shannon information content** of an outcome $x$ is defined to be:

$$h(x) = \log_2 \frac{1}{P(x)}.$$

It is measured in **bits**. Rare outcomes have high information content; certain outcomes have zero information content.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Entropy)</span></p>

The **entropy** of an ensemble $X$ is defined to be the average Shannon information content of an outcome:

$$H(X) \equiv \sum_{x \in \mathcal{A}_X} P(x) \log \frac{1}{P(x)},$$

with the convention $0 \times \log 1/0 \equiv 0$. Like information content, entropy is measured in bits. It is also called the **uncertainty** of $X$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Entropy)</span></p>

- $H(X) \ge 0$, with equality iff $p_i = 1$ for some $i$ (i.e., the outcome is certain).
- **Entropy is maximized by the uniform distribution:**

$$H(X) \le \log \lvert \mathcal{A}_X \rvert, \quad \text{with equality iff } p_i = 1 / \lvert \mathcal{A}_X \rvert \text{ for all } i.$$

- The **redundancy** of $X$ is $1 - \frac{H(X)}{\log \lvert \mathcal{A}_X \rvert}$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Joint Entropy)</span></p>

The **joint entropy** of $X, Y$ is:

$$H(X, Y) = \sum_{x, y \in \mathcal{A}_X, \mathcal{A}_Y} P(x, y) \log \frac{1}{P(x, y)}.$$

Entropy is **additive for independent random variables**:

$$H(X, Y) = H(X) + H(Y) \quad \text{iff} \quad P(x, y) = P(x) P(y).$$

</div>

### 2.5 Decomposability of the Entropy

The entropy function satisfies a recursive **decomposability** property. For any probability distribution $\mathbf{p} = \lbrace p_1, p_2, \ldots, p_I \rbrace$:

$$H(\mathbf{p}) = H(p_1, 1-p_1) + (1-p_1) H\!\left(\frac{p_2}{1-p_1}, \frac{p_3}{1-p_1}, \ldots, \frac{p_I}{1-p_1}\right).$$

More generally, for any partition into two groups at position $m$:

$$H(\mathbf{p}) = H\bigl[(p_1 + \cdots + p_m),\, (p_{m+1} + \cdots + p_I)\bigr] + (p_1 + \cdots + p_m) H\!\left(\frac{p_1}{p_1 + \cdots + p_m}, \ldots\right) + (p_{m+1} + \cdots + p_I) H\!\left(\frac{p_{m+1}}{p_{m+1} + \cdots + p_I}, \ldots\right).$$

This means we can compute the entropy by decomposing the random variable into successive revelations and summing the weighted entropies.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Decomposability of Entropy)</span></p>

A source produces a character $x$ from the alphabet $\mathcal{A} = \lbrace 0, 1, \ldots, 9, \texttt{a}, \texttt{b}, \ldots, \texttt{z} \rbrace$. With probability $1/3$, $x$ is a numeral (equiprobable); with probability $1/3$, $x$ is a vowel ($\texttt{a}, \texttt{e}, \texttt{i}, \texttt{o}, \texttt{u}$, equiprobable); with probability $1/3$, $x$ is one of the 21 consonants (equiprobable). Using decomposability:

$$H(X) = \log 3 + \tfrac{1}{3}(\log 10 + \log 5 + \log 21) = \log 3 + \tfrac{1}{3} \log 1050 \simeq \log 30 \text{ bits.}$$

</div>

### 2.6 Gibbs' Inequality

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Relative Entropy / Kullback--Leibler Divergence)</span></p>

The **relative entropy** or **Kullback--Leibler divergence** between two probability distributions $P(x)$ and $Q(x)$ defined over the same alphabet $\mathcal{A}_X$ is:

$$D_{\text{KL}}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}.$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Gibbs' Inequality)</span></p>

$$D_{\text{KL}}(P \| Q) \ge 0,$$

with equality only if $P = Q$. Note that the relative entropy is **not symmetric**: in general $D_{\text{KL}}(P \| Q) \neq D_{\text{KL}}(Q \| P)$, so it is not strictly a distance, although it is sometimes called the "KL distance."

</div>

Gibbs' inequality is arguably the most important inequality in this book. It can be proved using Jensen's inequality.

### 2.7 Jensen's Inequality for Convex Functions

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Convex Function)</span></p>

A function $f(x)$ is **convex** $\smile$ over $(a, b)$ if every chord of the function lies above the function:

$$f(\lambda x_1 + (1-\lambda) x_2) \le \lambda f(x_1) + (1-\lambda) f(x_2)$$

for all $x_1, x_2 \in (a, b)$ and $0 \le \lambda \le 1$. It is **strictly convex** if equality holds only for $\lambda = 0$ and $\lambda = 1$.

Examples of strictly convex functions: $x^2$, $e^x$, $e^{-x}$ for all $x$; $\log(1/x)$ and $x \log x$ for $x > 0$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Jensen's Inequality)</span></p>

If $f$ is a convex $\smile$ function and $x$ is a random variable, then:

$$\mathcal{E}[f(x)] \ge f(\mathcal{E}[x]).$$

If $f$ is strictly convex and $\mathcal{E}[f(x)] = f(\mathcal{E}[x])$, then $x$ is a constant. Jensen's inequality can be restated for a concave $\frown$ function with the inequality reversed.

**Physical interpretation:** if masses $p_i$ are placed on a convex curve $f(x)$ at locations $(x_i, f(x_i))$, the centre of gravity $(\mathcal{E}[x], \mathcal{E}[f(x)])$ lies above the curve.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Convexity and Maximization)</span></p>

If $f(\mathbf{x})$ is concave $\frown$ and there exists a point where $\frac{\partial f}{\partial x_k} = 0$ for all $k$, then $f(\mathbf{x})$ has its maximum at that point. The converse does not hold: a concave function may be maximized at a boundary point where the gradient is not zero.

</div>

## Chapter 3 -- More about Inference

Bayes' theorem provides the correct language for describing inference -- whether for decoding a message over a noisy channel (Chapter 1) or for any other inference problem. This chapter develops the Bayesian approach through several worked examples.

### 3.1 A First Inference Problem

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Inferring a Decay Constant)</span></p>

Unstable particles are emitted from a source and decay at a distance $x$ with an exponential distribution of characteristic length $\lambda$. Decays are only observable in a window from $x = 1$ cm to $x = 20$ cm. $N$ decays are observed at locations $\lbrace x_1, \ldots, x_N \rbrace$. What is $\lambda$?

The probability of one data point given $\lambda$ is:

$$P(x \mid \lambda) = \begin{cases} \frac{1}{\lambda} e^{-x/\lambda} / Z(\lambda) & 1 < x < 20 \\ 0 & \text{otherwise} \end{cases}$$

where $Z(\lambda) = \int_1^{20} \frac{1}{\lambda} e^{-x/\lambda} \, \mathrm{d}x = e^{-1/\lambda} - e^{-20/\lambda}$.

By Bayes' theorem, the posterior is:

$$P(\lambda \mid \lbrace x_1, \ldots, x_N \rbrace) \propto \frac{1}{(\lambda Z(\lambda))^N} \exp\!\left(-\sum_{n=1}^{N} x_n / \lambda\right) P(\lambda).$$

</div>

The key insight is that Bayes' theorem turns the likelihood function $P(\lbrace x \rbrace \mid \lambda)$ -- the probability of the data given the hypothesis -- into the posterior $P(\lambda \mid \lbrace x \rbrace)$ -- the probability of the hypothesis given the data. The posterior represents the unique and complete solution to the inference problem. There is no need to invent ad hoc "estimators."

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Advantages of Bayesian Inference)</span></p>

The Bayesian approach has several advantages:
1. **Objectivity given assumptions:** once assumptions $\mathcal{H}$ are made explicit, the inferences are objective, reproducible, and unique. Everyone with the same assumptions and data agrees on the posterior: $P(\lambda \mid D, \mathcal{H}) = \frac{P(D \mid \lambda, \mathcal{H}) P(\lambda \mid \mathcal{H})}{P(D \mid \mathcal{H})}$.
2. **Sensitivity analysis:** explicit assumptions are easier to criticize and modify. We can quantify the sensitivity of inferences to the details of assumptions (e.g., the prior).
3. **Model comparison:** when we are unsure which assumptions are best, we can treat this as another inference task. Given data $D$, we compare alternative assumptions $\mathcal{H}$ using: $P(\mathcal{H} \mid D, I) = \frac{P(D \mid \mathcal{H}, I) P(\mathcal{H} \mid I)}{P(D \mid I)}$.
4. **Predictions under uncertainty:** rather than choosing one model $\mathcal{H}^*$ and predicting from it, we marginalize over our uncertainty: $P(\mathbf{t} \mid D, I) = \sum_{\mathcal{H}} P(\mathbf{t} \mid D, \mathcal{H}, I) P(\mathcal{H} \mid D, I)$.

</div>

### 3.2 The Bent Coin

A bent coin is tossed $F$ times, producing a sequence $\mathbf{s}$ of heads ($\texttt{a}$) and tails ($\texttt{b}$). We wish to infer the bias $p_\texttt{a}$ and predict the next toss. This is the original inference problem studied by Thomas Bayes in 1763.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Bent Coin Model $\mathcal{H}_1$)</span></p>

Under model $\mathcal{H}_1$, the probability that $F$ tosses yield counts $\lbrace F_\texttt{a}, F_\texttt{b} \rbrace$ of the two outcomes is:

$$P(\mathbf{s} \mid p_\texttt{a}, F, \mathcal{H}_1) = p_\texttt{a}^{F_\texttt{a}} (1 - p_\texttt{a})^{F_\texttt{b}},$$

where $p_\texttt{b} \equiv 1 - p_\texttt{a}$. With a uniform prior $P(p_\texttt{a} \mid \mathcal{H}_1) = 1$ for $p_\texttt{a} \in [0, 1]$, the posterior is:

$$P(p_\texttt{a} \mid \mathbf{s}, F, \mathcal{H}_1) = \frac{p_\texttt{a}^{F_\texttt{a}} (1 - p_\texttt{a})^{F_\texttt{b}}}{P(\mathbf{s} \mid F, \mathcal{H}_1)},$$

where the normalizing constant (the **evidence** for model $\mathcal{H}_1$) is given by the **beta integral**:

$$P(\mathbf{s} \mid F, \mathcal{H}_1) = \int_0^1 p_\texttt{a}^{F_\texttt{a}} (1 - p_\texttt{a})^{F_\texttt{b}} \, \mathrm{d}p_\texttt{a} = \frac{\Gamma(F_\texttt{a} + 1)\,\Gamma(F_\texttt{b} + 1)}{\Gamma(F_\texttt{a} + F_\texttt{b} + 2)} = \frac{F_\texttt{a}!\, F_\texttt{b}!}{(F_\texttt{a} + F_\texttt{b} + 1)!}.$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Laplace's Rule)</span></p>

The predictive probability that the next toss is $\texttt{a}$, given data $\mathbf{s}$ with counts $F_\texttt{a}$ and $F_\texttt{b}$, is obtained by marginalizing over $p_\texttt{a}$:

$$P(\texttt{a} \mid \mathbf{s}, F) = \int \mathrm{d}p_\texttt{a} \; p_\texttt{a} \; P(p_\texttt{a} \mid \mathbf{s}, F) = \frac{F_\texttt{a} + 1}{F_\texttt{a} + F_\texttt{b} + 2}.$$

This is known as **Laplace's rule of succession**. It incorporates our uncertainty about $p_\texttt{a}$ into the prediction.

</div>

### 3.3 The Bent Coin and Model Comparison

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Model Comparison)</span></p>

Suppose a scientist proposes an alternative model $\mathcal{H}_0$: the source is a perfectly formed die with $p_\texttt{a} = p_0 = 1/6$ (no free parameter). To compare $\mathcal{H}_1$ (free parameter $p_\texttt{a}$) against $\mathcal{H}_0$, we use Bayes' theorem at the model level:

$$P(\mathcal{H}_1 \mid \mathbf{s}, F) = \frac{P(\mathbf{s} \mid F, \mathcal{H}_1) P(\mathcal{H}_1)}{P(\mathbf{s} \mid F)}, \qquad P(\mathcal{H}_0 \mid \mathbf{s}, F) = \frac{P(\mathbf{s} \mid F, \mathcal{H}_0) P(\mathcal{H}_0)}{P(\mathbf{s} \mid F)}.$$

The posterior probability ratio is:

$$\frac{P(\mathcal{H}_1 \mid \mathbf{s}, F)}{P(\mathcal{H}_0 \mid \mathbf{s}, F)} = \frac{P(\mathbf{s} \mid F, \mathcal{H}_1)}{P(\mathbf{s} \mid F, \mathcal{H}_0)} \cdot \frac{P(\mathcal{H}_1)}{P(\mathcal{H}_0)},$$

where $P(\mathbf{s} \mid F, \mathcal{H}_0) = p_0^{F_\texttt{a}} (1 - p_0)^{F_\texttt{b}}$ and $P(\mathbf{s} \mid F, \mathcal{H}_1) = \frac{F_\texttt{a}!\, F_\texttt{b}!}{(F_\texttt{a} + F_\texttt{b} + 1)!}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Evidence and Model Comparison)</span></p>

The **evidence** for a model is the normalizing constant of an earlier Bayesian inference:

$$P(\mathbf{s} \mid F, \mathcal{H}_1) = \int P(\mathbf{s} \mid p_\texttt{a}, F, \mathcal{H}_1) P(p_\texttt{a} \mid \mathcal{H}_1) \, \mathrm{d}p_\texttt{a}.$$

Key observations:
- The simpler model $\mathcal{H}_0$ (no free parameters) can lose by the biggest margin: the odds can be hundreds to one against it.
- The more complex model $\mathcal{H}_1$ can never lose by a large margin -- there is no data set that is *unlikely* given $\mathcal{H}_1$.
- With small data, neither model is overwhelmingly more probable. With more data, the evidence accumulates and the models can be clearly distinguished.

</div>

### 3.4 An Example of Legal Evidence

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Forensic Evidence)</span></p>

Two people left blood traces at a crime scene. Oliver, a suspect, has type O blood. The two blood traces are type O (frequency $p_\text{O} = 0.6$) and type AB (frequency $p_\text{AB} = 0.01$). Do these data support the hypothesis $S$ that Oliver was present?

Let $S$ = "the suspect and one unknown person were present" and $\bar{S}$ = "two unknown people from the population were present." The likelihoods are:

$$P(D \mid S, \mathcal{H}) = p_\text{AB}, \qquad P(D \mid \bar{S}, \mathcal{H}) = 2 \, p_\text{O} \, p_\text{AB}.$$

The likelihood ratio is:

$$\frac{P(D \mid S, \mathcal{H})}{P(D \mid \bar{S}, \mathcal{H})} = \frac{1}{2 p_\text{O}} = \frac{1}{1.2} = 0.83.$$

The data provide weak evidence **against** the hypothesis that Oliver was present. This may seem surprising, but the key insight is: the likelihood ratio depends only on the comparison of the suspect's blood type frequency ($n_\text{O}/N$) with the background population frequency ($p_\text{O}$). Since type O blood is common, its presence at the scene is not particularly indicative of the suspect.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(General Forensic Likelihood Ratio)</span></p>

In general, if $n_\text{O}$ type O stains and $n_\text{AB}$ type AB stains are found among $N$ total stains, the likelihood ratio for the suspect's presence is:

$$\frac{P(n_\text{O}, n_\text{AB} \mid S)}{P(n_\text{O}, n_\text{AB} \mid \bar{S})} = \frac{n_\text{O}/N}{p_\text{O}}.$$

The contribution depends simply on comparing the frequency of the suspect's blood type in the observed data with its background frequency. There is no dependence on the counts or frequencies of the *other* blood types.

</div>

## Chapter 4 -- The Source Coding Theorem

In this chapter we discuss how to measure the information content of the outcome of a random experiment. We examine the Shannon information content, the entropy, and prove Shannon's source coding theorem, which establishes the fundamental connection between entropy and data compression.

### 4.1 How to Measure the Information Content of a Random Variable?

Recall that an **ensemble** $X$ is a triple $(x, \mathcal{A}_X, \mathcal{P}_X)$, where the **outcome** $x$ is the value of a random variable taking on one of a set of possible values $\mathcal{A}_X = \lbrace a_1, a_2, \dots, a_I \rbrace$, having probabilities $\mathcal{P}_X = \lbrace p_1, p_2, \dots, p_I \rbrace$, with $P(x = a_i) = p_i$, $p_i \ge 0$ and $\sum_{a_i \in \mathcal{A}_X} P(x = a_i) = 1$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Shannon Information Content)</span></p>

The **Shannon information content** of an outcome $x = a_i$ is:

$$h(x = a_i) \equiv \log_2 \frac{1}{p_i}.$$

The less probable an outcome is, the greater its Shannon information content.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Entropy of an Ensemble)</span></p>

The **entropy** of an ensemble $X$ is the average Shannon information content:

$$H(X) = \sum_i p_i \log_2 \frac{1}{p_i}.$$

</div>

#### Information Content of Independent Random Variables

Why should $\log 1/p_i$ have anything to do with information content? The key property is **additivity** for independent random variables. If $x$ and $y$ are independent, meaning $P(x, y) = P(x)P(y)$, then the Shannon information content of the joint outcome is:

$$h(x, y) = \log \frac{1}{P(x,y)} = \log \frac{1}{P(x)P(y)} = \log \frac{1}{P(x)} + \log \frac{1}{P(y)} = h(x) + h(y).$$

Similarly, for independent random variables, entropy is additive:

$$H(X, Y) = H(X) + H(Y).$$

#### The Weighing Problem: Designing Informative Experiments

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(The Weighing Problem)</span></p>

You are given 12 balls, all equal in weight except for one that is either heavier or lighter. Using a three-outcome balance, the number of conceivable outcomes in three weighings is $3^3 = 27$, while the number of possible states is 24 (any of 12 balls, heavy or light). Since $3^2 < 24 < 3^3$, the problem might be solvable in three weighings but not in two.

The key insight is: at each step of an optimal procedure, the three outcomes ('left heavier', 'right heavier', and 'balance') should be **as close as possible to equiprobable**. This maximizes the information gained per weighing.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Most Informative Experiments)</span></p>

The outcome of a random experiment is guaranteed to be most informative if the probability distribution over outcomes is **uniform**. This agrees with the property that entropy $H(X)$ is maximized when all outcomes have equal probability $p_i = 1/\lvert \mathcal{A}_X \rvert$.

</div>

#### Guessing Games

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(The Game 'Sixty-Three')</span></p>

What is the smallest number of yes/no questions needed to identify an integer $x$ between 0 and 63? Since there are $64 = 2^6$ possibilities, six questions suffice. A reasonable strategy successively halves the search space:

1. Is $x \ge 32$?
2. Is $x \bmod 32 \ge 16$?
3. Is $x \bmod 16 \ge 8$?
4. Is $x \bmod 8 \ge 4$?
5. Is $x \bmod 4 \ge 2$?
6. Is $x \bmod 2 = 1$?

The answers, translated from $\lbrace \text{yes}, \text{no} \rbrace$ to $\lbrace 1, 0 \rbrace$, give the binary expansion of $x$. If all values are equally likely, each answer has Shannon information content $\log_2(1/0.5) = 1$ bit, and the total information gained is six bits.

</div>

#### The Game of Submarine

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(The Game of Submarine)</span></p>

In **submarine**, one player hides a submarine in one square of an $8 \times 8$ grid. The other player fires at squares, receiving 'miss' or 'hit' outcomes. If we hit on the first shot:

$$h(x) = h_{(1)}(\mathbf{y}) = \log_2 64 = 6 \text{ bits}.$$

One binary outcome can convey six bits because we have learnt the hiding place (one of 64 squares) in a single question. If the first shot misses, the Shannon information gained is only $\log_2 \frac{64}{63} = 0.0227$ bits. The total Shannon information content gained over all shots always sums to 6 bits, regardless of when the submarine is hit.

</div>

#### The Wenglish Language

Wenglish is a language similar to English, with a dictionary of $2^{15} = 32{,}768$ words, all of length 5, constructed by picking letters from the probability distribution over **a**...**z**. If we read one word at a time, the Shannon information content per word is $\log 32{,}768 = 15$ bits, giving 3 bits per character on average.

Reading character by character, rare initial letters like **z** ($p \approx 0.001$, Shannon info $\approx 10$ bits) allow immediate word identification, while common initial letters like **a** ($p \approx 0.0625$, Shannon info $\approx 4$ bits) require more characters. The total information content for all 5 characters is always exactly 15 bits.

### 4.2 Data Compression

The preceding examples justify the idea that the Shannon information content is a natural measure of information. Improbable outcomes convey more information than probable ones. We now discuss information content by considering how many bits are needed to describe the outcome of an experiment.

If we can show that we can compress data from a source into $L$ bits per source symbol and recover the data reliably, then we say that the average information content of that source is at most $L$ bits per symbol.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Raw Bit Content)</span></p>

The **raw bit content** of $X$ is:

$$H_0(X) = \log_2 \lvert \mathcal{A}_X \rvert.$$

$H_0(X)$ is a lower bound for the number of binary questions that are always guaranteed to identify an outcome from the ensemble $X$. It is additive: $H_0(X,Y) = H_0(X) + H_0(Y)$.

This measure does not include any probabilistic element -- it simply maps each outcome to a constant-length binary string.

</div>

There are only two ways a compressor can actually compress files:

1. A **lossy compressor** compresses some files, but maps some files to the same encoding. The probability that the source string is one of the confusable files is $\delta$ (probability of failure).
2. A **lossless compressor** maps all files to different encodings; if it shortens some files, it necessarily makes others longer.

### 4.3 Information Content Defined in Terms of Lossy Compression

We introduce a parameter $\delta$ describing the risk of our compression method: $\delta$ is the probability that there will be no name for an outcome $x$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Lossy Compression with 8 Symbols)</span></p>

Let $\mathcal{A}_X = \lbrace \mathbf{a}, \mathbf{b}, \mathbf{c}, \mathbf{d}, \mathbf{e}, \mathbf{f}, \mathbf{g}, \mathbf{h} \rbrace$ with $\mathcal{P}_X = \lbrace \tfrac{1}{4}, \tfrac{1}{4}, \tfrac{1}{4}, \tfrac{1}{16}, \tfrac{3}{64}, \tfrac{1}{64}, \tfrac{1}{64}, \tfrac{1}{64} \rbrace$. The raw bit content is 3 bits (8 binary names). But since $P(x \in \lbrace \mathbf{a}, \mathbf{b}, \mathbf{c}, \mathbf{d} \rbrace) = 15/16$, if we accept risk $\delta = 1/16$ of failure, we need only 4 names (2 bits).

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Smallest $\delta$-Sufficient Subset)</span></p>

The **smallest $\delta$-sufficient subset** $S_\delta$ is the smallest subset of $\mathcal{A}_X$ satisfying:

$$P(x \in S_\delta) \ge 1 - \delta.$$

$S_\delta$ can be constructed by ranking elements of $\mathcal{A}_X$ in order of decreasing probability and adding successive elements starting from the most probable until the total probability is $\ge (1-\delta)$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Essential Bit Content)</span></p>

The **essential bit content** of $X$ is:

$$H_\delta(X) = \log_2 \lvert S_\delta \rvert.$$

Note that $H_0(X)$ is the special case with $\delta = 0$ (if $P(x) > 0$ for all $x \in \mathcal{A}_X$).

</div>

#### Extended Ensembles

We now consider compressing **blocks** of $N$ i.i.d. symbols from a single ensemble $X$. Denote by $X^N$ the ensemble $(X_1, X_2, \dots, X_N)$. Since entropy is additive for independent variables, $H(X^N) = NH(X)$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Bent Coin Sequences)</span></p>

Consider a string of $N$ flips of a bent coin, $\mathbf{x} = (x_1, \dots, x_N)$, with $x_n \in \lbrace 0, 1 \rbrace$ and probabilities $p_0 = 0.9$, $p_1 = 0.1$. If $r(\mathbf{x})$ is the number of 1s in $\mathbf{x}$, then $P(\mathbf{x}) = p_0^{N - r(\mathbf{x})} p_1^{r(\mathbf{x})}$.

As $N$ increases, $\frac{1}{N} H_\delta(X^N)$ becomes an increasingly flat function of $\delta$, converging to $H(X)$ -- the entropy of a single variable. This motivates the source coding theorem.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Shannon's Source Coding Theorem)</span></p>

Let $X$ be an ensemble with entropy $H(X) = H$ bits. Given $\epsilon > 0$ and $0 < \delta < 1$, there exists a positive integer $N_0$ such that for $N > N_0$:

$$\left\lvert \frac{1}{N} H_\delta(X^N) - H \right\rvert < \epsilon.$$

**Verbal statement:** $N$ i.i.d. random variables each with entropy $H(X)$ can be compressed into more than $NH(X)$ bits with negligible risk of information loss, as $N \to \infty$; conversely if they are compressed into fewer than $NH(X)$ bits it is virtually certain that information will be lost.

</div>

### 4.4 Typicality

For long strings from $X^N$, the probability of a string $\mathbf{x}$ containing $r$ ones is $P(\mathbf{x}) = p_1^r (1 - p_1)^{N-r}$, and the number of 1s has a binomial distribution $P(r) = \binom{N}{r} p_1^r (1 - p_1)^{N-r}$ with mean $Np_1$ and standard deviation $\sqrt{Np_1(1-p_1)}$.

As $N$ gets bigger, the probability distribution of $r$ becomes more concentrated. While the range of possible values of $r$ grows as $N$, the standard deviation grows only as $\sqrt{N}$. That $r$ falls in a small range implies that the outcome $\mathbf{x}$ is most likely to fall in a corresponding small subset of outcomes called the **typical set**.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Typical Set)</span></p>

For an arbitrary ensemble $X$ with alphabet $\mathcal{A}_X$, the **typical set** $T_{N\beta}$ is defined as:

$$T_{N\beta} \equiv \left\lbrace \mathbf{x} \in \mathcal{A}_X^N : \left\lvert \frac{1}{N} \log_2 \frac{1}{P(\mathbf{x})} - H \right\rvert < \beta \right\rbrace.$$

The parameter $\beta$ controls how close the probability must be to $2^{-NH}$ for an element to be 'typical'. For any choice of $\beta$, the typical set contains almost all the probability as $N$ increases.

</div>

A typical string of $N$ symbols will contain about $p_1 N$ occurrences of the first symbol, $p_2 N$ of the second, etc. The probability of such a typical string is roughly:

$$P(\mathbf{x})_{\text{typ}} \simeq p_1^{(p_1 N)} p_2^{(p_2 N)} \cdots p_I^{(p_I N)},$$

so that the information content of a typical string is:

$$\log_2 \frac{1}{P(\mathbf{x})} \simeq N \sum_i p_i \log_2 \frac{1}{p_i} = NH.$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Asymptotic Equipartition Principle)</span></p>

For an ensemble of $N$ independent identically distributed random variables $X^N \equiv (X_1, X_2, \dots, X_N)$, with $N$ sufficiently large, the outcome $\mathbf{x} = (x_1, x_2, \dots, x_N)$ is almost certain to belong to a subset of $\mathcal{A}_X^N$ having only $2^{NH(X)}$ members, each having probability 'close to' $2^{-NH(X)}$.

Note: if $H(X) < H_0(X)$ then $2^{NH(X)}$ is a tiny fraction of $\lvert \mathcal{A}_X^N \rvert = \lvert \mathcal{A}_X \rvert^N = 2^{NH_0(X)}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Caveat on 'Asymptotic Equipartition')</span></p>

The elements of the typical set $T_{N\beta}$ do **not** truly have roughly the same probability. They are similar in probability only in the sense that their values of $\log_2 \frac{1}{P(\mathbf{x})}$ are within $2N\beta$ of each other. As $\beta$ decreases, $N$ must grow as $1/\beta^2$ to keep the typical set's probability near 1. Writing $\beta = \alpha/\sqrt{N}$, the ratio of the most probable to the least probable string in the typical set is of order $2^{\alpha\sqrt{N}}$, which grows exponentially. Thus we have 'equipartition' only in a weak sense.

</div>

### 4.5 Proofs

#### The Law of Large Numbers

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Mean and Variance)</span></p>

The **mean** and **variance** of a real random variable $u$ are:

$$\mathcal{E}[u] = \bar{u} = \sum_u P(u) u, \qquad \text{var}(u) = \sigma_u^2 = \mathcal{E}[(u - \bar{u})^2] = \sum_u P(u)(u - \bar{u})^2.$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Chebyshev's Inequality 1)</span></p>

Let $t$ be a non-negative real random variable, and let $\alpha$ be a positive real number. Then:

$$P(t \ge \alpha) \le \frac{\bar{t}}{\alpha}.$$

*Proof:* $P(t \ge \alpha) = \sum_{t \ge \alpha} P(t)$. Multiply each term by $t/\alpha \ge 1$ to obtain $P(t \ge \alpha) \le \sum_{t \ge \alpha} P(t) t / \alpha$. Add the non-negative missing terms: $P(t \ge \alpha) \le \sum_t P(t) t / \alpha = \bar{t}/\alpha$. $\square$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Chebyshev's Inequality 2)</span></p>

Let $x$ be a random variable, and let $\alpha$ be a positive real number. Then:

$$P\!\left((x - \bar{x})^2 \ge \alpha\right) \le \sigma_x^2 / \alpha.$$

*Proof:* Take $t = (x - \bar{x})^2$ and apply Chebyshev's inequality 1. $\square$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Weak Law of Large Numbers)</span></p>

Take $x$ to be the average of $N$ independent random variables $h_1, \dots, h_N$, having common mean $\bar{h}$ and common variance $\sigma_h^2$: $x = \frac{1}{N} \sum_{n=1}^{N} h_n$. Then:

$$P\!\left((x - \bar{h})^2 \ge \alpha\right) \le \frac{\sigma_h^2}{\alpha N}.$$

*Proof:* Since $\bar{x} = \bar{h}$ and $\sigma_x^2 = \sigma_h^2 / N$, the result follows directly from Chebyshev's inequality 2. $\square$

No matter how large $\sigma_h^2$ is, how small $\alpha$ is, and how small the desired probability that $(x - \bar{h})^2 \ge \alpha$, we can always achieve it by taking $N$ large enough.

</div>

#### Proof of the Source Coding Theorem

We apply the law of large numbers to the random variable $\frac{1}{N} \log_2 \frac{1}{P(\mathbf{x})}$ defined for $\mathbf{x}$ drawn from $X^N$. This can be written as the average of $N$ information contents $h_n = \log_2(1/P(x_n))$, each with mean $H = H(X)$ and variance $\sigma^2 \equiv \text{var}[\log_2(1/P(x_n))]$.

Define the typical set with parameters $N$ and $\beta$:

$$T_{N\beta} = \left\lbrace \mathbf{x} \in \mathcal{A}_X^N : \left[\frac{1}{N} \log_2 \frac{1}{P(\mathbf{x})} - H\right]^2 < \beta^2 \right\rbrace.$$

For all $\mathbf{x} \in T_{N\beta}$, the probability satisfies $2^{-N(H+\beta)} < P(\mathbf{x}) < 2^{-N(H-\beta)}$. By the law of large numbers, $P(\mathbf{x} \in T_{N\beta}) \ge 1 - \frac{\sigma^2}{\beta^2 N}$.

**Part 1:** $\frac{1}{N} H_\delta(X^N) < H + \epsilon$.

The size of $T_{N\beta}$ gives an upper bound on $H_\delta$. Since $\lvert T_{N\beta} \rvert \cdot 2^{-N(H+\beta)} < 1$, we get $\lvert T_{N\beta} \rvert < 2^{N(H+\beta)}$. Setting $\beta = \epsilon$ and choosing $N_0$ such that $\frac{\sigma^2}{\epsilon^2 N_0} \le \delta$, the set $T_{N\beta}$ witnesses that $H_\delta(X^N) \le \log_2 \lvert T_{N\beta} \rvert < N(H + \epsilon)$.

**Part 2:** $\frac{1}{N} H_\delta(X^N) > H - \epsilon$.

Suppose a rival smaller subset $S'$ with $\lvert S' \rvert \le 2^{N(H-2\beta)}$ claims to satisfy the definition of $S_\delta$. The probability of falling in $S'$ is:

$$P(\mathbf{x} \in S') = P(\mathbf{x} \in S' \cap T_{N\beta}) + P(\mathbf{x} \in S' \cap \overline{T_{N\beta}}).$$

The first term is at most $2^{N(H-2\beta)} \cdot 2^{-N(H-\beta)} = 2^{-N\beta}$, and the second term is at most $\frac{\sigma^2}{\beta^2 N}$. Setting $\beta = \epsilon/2$ and choosing $N_0$ large enough, $P(\mathbf{x} \in S') < 1 - \delta$, so $S'$ cannot be a $\delta$-sufficient subset. Therefore $H_\delta(X^N) > N(H - \epsilon)$. $\square$

### 4.6 Comments

The source coding theorem has two parts: $\frac{1}{N} H_\delta(X^N) < H + \epsilon$ and $\frac{1}{N} H_\delta(X^N) > H - \epsilon$.

**Part 1** tells us that even if $\delta$ is extremely small, the number of bits per symbol $\frac{1}{N} H_\delta(X^N)$ needed to specify a long $N$-symbol string $\mathbf{x}$ with vanishingly small error probability does not have to exceed $H + \epsilon$ bits. The number of bits required drops significantly from $H_0(X)$ to $(H + \epsilon)$.

**Part 2** tells us that even if $\delta$ is very close to 1 (errors made most of the time), the average number of bits per symbol needed must still be at least $H - \epsilon$ bits.

Together, these extremes tell us that regardless of our specific allowance for error, the number of bits per symbol needed to specify $\mathbf{x}$ is $H$ bits -- no more and no less.

#### Why Introduce the Typical Set?

The best choice of subset for block compression is (by definition) $S_\delta$, not the typical set. We introduced the typical set because *we can count it*: all its elements have 'almost identical' probability ($2^{-NH}$), and the whole set has probability almost 1, so it must have roughly $2^{NH}$ elements. Without this counting tool (which is very similar to $S_\delta$), it would have been hard to count how many elements are in $S_\delta$.

## Chapter 5 -- Symbol Codes

In the previous chapter we proved the *possibility* of data compression using fixed-length block codes. The block coding defined in the proof did not give a practical algorithm. In this chapter we study practical **variable-length symbol codes**, which encode one source symbol at a time. These codes are **lossless**: they are guaranteed to compress and decompress without any errors, but some outcomes may produce encoded strings longer than the original. The idea is to assign *shorter* encodings to more probable outcomes and *longer* encodings to less probable ones.

Analogy: imagine a rubber glove filled with water. If we compress two fingers, some other part must expand, because the total volume is constant. Similarly, when we shorten codewords for some outcomes, there must be other codewords that get longer, if the scheme is lossless.

### 5.1 Symbol Codes

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Binary Symbol Code)</span></p>

A **(binary) symbol code** $C$ for an ensemble $X$ is a mapping from the range of $x$, $\mathcal{A}_X = \lbrace a_1, \dots, a_I \rbrace$, to $\lbrace 0, 1 \rbrace^+$. We write $c(x)$ for the **codeword** corresponding to $x$, and $l(x)$ for its length, with $l_i = l(a_i)$.

The **extended code** $C^+$ is a mapping from $\mathcal{A}_X^+$ to $\lbrace 0, 1 \rbrace^+$ obtained by concatenation of the corresponding codewords, without punctuation:

$$c^+(x_1 x_2 \dots x_N) = c(x_1) c(x_2) \dots c(x_N).$$

</div>

Three basic requirements for a useful symbol code:
1. Any encoded string must have a **unique decoding**.
2. The symbol code must be **easy to decode**.
3. The code should achieve as much **compression** as possible.

#### Unique Decodability

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Uniquely Decodeable Code)</span></p>

A code $C(X)$ is **uniquely decodeable** if, under the extended code $C^+$, no two distinct strings have the same encoding:

$$\forall\; \mathbf{x}, \mathbf{y} \in \mathcal{A}_X^+, \quad \mathbf{x} \neq \mathbf{y} \;\Rightarrow\; c^+(\mathbf{x}) \neq c^+(\mathbf{y}).$$

</div>

#### Prefix Codes

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Prefix Code)</span></p>

A symbol code is called a **prefix code** if no codeword is a prefix of any other codeword. A word $c$ is a *prefix* of another word $d$ if there exists a tail string $t$ such that the concatenation $ct$ is identical to $d$.

A prefix code is also known as an **instantaneous** or **self-punctuating** code, because an encoded string can be decoded from left to right without looking ahead to subsequent codewords. Every prefix code is uniquely decodeable.

Prefix codes correspond to binary trees: *complete* prefix codes correspond to binary trees having no unused branches.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Prefix and Non-Prefix Codes)</span></p>

- $C_1 = \lbrace 0, 101 \rbrace$ is a prefix code: 0 is not a prefix of 101, nor is 101 a prefix of 0.
- $C_2 = \lbrace 1, 101 \rbrace$ is **not** a prefix code: 1 is a prefix of 101. However, $C_2$ *is* uniquely decodeable.
- $C_3 = \lbrace 0, 10, 110, 111 \rbrace$ is a prefix code.
- $C_4 = \lbrace 00, 01, 10, 11 \rbrace$ is a prefix code (fixed-length codes are always prefix codes).

</div>

#### Expected Length of a Symbol Code

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Expected Length)</span></p>

The **expected length** $L(C, X)$ of a symbol code $C$ for ensemble $X$ is:

$$L(C, X) = \sum_{x \in \mathcal{A}_X} P(x)\, l(x) = \sum_{i=1}^{I} p_i l_i,$$

where $I = \lvert \mathcal{A}_X \rvert$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Optimal Codelengths Equal Shannon Information Content)</span></p>

Let $\mathcal{A}_X = \lbrace \mathbf{a}, \mathbf{b}, \mathbf{c}, \mathbf{d} \rbrace$ with $\mathcal{P}_X = \lbrace 1/2, 1/4, 1/8, 1/8 \rbrace$ and code $C_3$: $\mathbf{a} \to 0$, $\mathbf{b} \to 10$, $\mathbf{c} \to 110$, $\mathbf{d} \to 111$. The entropy of $X$ is 1.75 bits, and the expected length $L(C_3, X) = 1.75$ bits. This is because the codeword lengths satisfy $l_i = \log_2(1/p_i)$, i.e., they equal the Shannon information contents.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Non-Prefix Uniquely Decodeable Codes)</span></p>

Code $C_6$: $\mathbf{a} \to 0$, $\mathbf{b} \to 01$, $\mathbf{c} \to 011$, $\mathbf{d} \to 111$ is **not** a prefix code (since $c(\mathbf{a}) = 0$ is a prefix of $c(\mathbf{b})$ and $c(\mathbf{c})$), yet it *is* uniquely decodeable. Its codewords are the reverses of $C_3$'s, and any string from $C_6$ is identical to a string from $C_3$ read backwards. However, $C_6$ is not easy to decode -- it requires looking ahead to find codeword boundaries.

</div>

### 5.2 What Limit Is Imposed by Unique Decodeability?

There is a constrained budget that we can spend on codewords, with shorter codewords being more expensive. A codeword of length $l$ has 'cost' $2^{-l}$. The total budget available for a uniquely decodeable code is 1.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Kraft Inequality)</span></p>

For any uniquely decodeable code $C(X)$ over the binary alphabet $\lbrace 0, 1 \rbrace$, the codeword lengths must satisfy:

$$\sum_{i=1}^{I} 2^{-l_i} \le 1,$$

where $I = \lvert \mathcal{A}_X \rvert$.

*Proof.* Define $S = \sum_i 2^{-l_i}$. Consider:

$$S^N = \left[\sum_i 2^{-l_i}\right]^N = \sum_{i_1=1}^{I} \cdots \sum_{i_N=1}^{I} 2^{-(l_{i_1} + l_{i_2} + \cdots + l_{i_N})}.$$

Let $A_l$ count how many strings $\mathbf{x}$ of length $N$ have encoded length $l$. Then $S^N = \sum_{l=Nl_{\min}}^{Nl_{\max}} 2^{-l} A_l$. Since $C$ is uniquely decodeable, $A_l \le 2^l$ (there are only $2^l$ distinct bit strings of length $l$). So $S^N \le \sum_{l} 1 \le Nl_{\max}$. If $S > 1$, then $S^N$ would grow exponentially, eventually exceeding $Nl_{\max}$ for large $N$. Contradiction. $\square$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Complete Code)</span></p>

A uniquely decodeable code that satisfies the Kraft inequality with **equality** ($\sum_i 2^{-l_i} = 1$) is called a **complete** code. Complete prefix codes correspond to binary trees having no unused branches.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Kraft Inequality and Prefix Codes)</span></p>

Given a set of codeword lengths that satisfy the Kraft inequality, there exists a uniquely decodeable **prefix code** with these codeword lengths.

This means we lose nothing by restricting attention to prefix codes: for any source there *is* an optimal symbol code that is also a prefix code.

</div>

### 5.3 What's the Most Compression That We Can Hope For?

We wish to minimize the expected length $L(C, X) = \sum_i p_i l_i$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Lower Bound on Expected Length)</span></p>

The expected length $L(C, X)$ of a uniquely decodeable code is bounded below by $H(X)$:

$$L(C, X) \ge H(X).$$

Equality is achieved only if the Kraft equality $z = \sum_{i'} 2^{-l_{i'}} = 1$ is satisfied **and** the codelengths satisfy $l_i = \log_2(1/p_i)$.

*Proof.* Define **implicit probabilities** $q_i \equiv 2^{-l_i}/z$ where $z = \sum_{i'} 2^{-l_{i'}}$, so that $l_i = \log 1/q_i - \log z$. Then:

$$L(C, X) = \sum_i p_i l_i = \sum_i p_i \log 1/q_i - \log z \ge \sum_i p_i \log 1/p_i = H(X),$$

where the inequality uses Gibbs' inequality ($\sum_i p_i \log 1/q_i \ge \sum_i p_i \log 1/p_i$, with equality iff $q_i = p_i$) and the Kraft inequality ($z \le 1$, so $-\log z \ge 0$). $\square$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Optimal Source Codelengths)</span></p>

The expected length is minimized and equal to $H(X)$ only if the codelengths are equal to the **Shannon information contents**:

$$l_i = \log_2(1/p_i).$$

This is achievable only when all probabilities are negative powers of 2. Conversely, any choice of codelengths $\lbrace l_i \rbrace$ implicitly defines a probability distribution $\lbrace q_i \rbrace$ via $q_i \equiv 2^{-l_i}/z$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Implicit Probabilities)</span></p>

Any choice of codelengths $\lbrace l_i \rbrace$ **implicitly** defines a probability distribution $\lbrace q_i \rbrace$:

$$q_i \equiv \frac{2^{-l_i}}{z}, \qquad z = \sum_{i'} 2^{-l_{i'}}.$$

If the code is complete ($z = 1$), the implicit probabilities are simply $q_i = 2^{-l_i}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Cost of Using the Wrong Codelengths)</span></p>

If the true probabilities are $\lbrace p_i \rbrace$ and we use a complete code with implicit probabilities $\lbrace q_i \rbrace$ (where $q_i = 2^{-l_i}$), the average length is:

$$L(C, X) = H(X) + \sum_i p_i \log \frac{p_i}{q_i} = H(X) + D_{\text{KL}}(\mathbf{p} \| \mathbf{q}),$$

i.e., the expected length exceeds the entropy by the **relative entropy** (Kullback--Leibler divergence) $D_{\text{KL}}(\mathbf{p} \| \mathbf{q})$.

</div>

### 5.4 How Much Can We Compress?

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Source Coding Theorem for Symbol Codes)</span></p>

For an ensemble $X$ there exists a prefix code $C$ with expected length satisfying:

$$H(X) \le L(C, X) < H(X) + 1.$$

*Proof.* Set the codelengths to integers slightly larger than the optimum: $l_i = \lceil \log_2(1/p_i) \rceil$, where $\lceil l^* \rceil$ denotes the smallest integer $\ge l^*$. The Kraft inequality is satisfied:

$$\sum_i 2^{-l_i} = \sum_i 2^{-\lceil \log_2(1/p_i) \rceil} \le \sum_i 2^{-\log_2(1/p_i)} = \sum_i p_i = 1.$$

Then the expected length satisfies:

$$L(C, X) = \sum_i p_i \lceil \log(1/p_i) \rceil < \sum_i p_i (\log(1/p_i) + 1) = H(X) + 1. \quad \square$$

</div>

### 5.5 Optimal Source Coding with Symbol Codes: Huffman Coding

Given a set of probabilities $\mathcal{P}$, how do we design an optimal prefix code? The answer is the **Huffman coding algorithm**, which builds the binary tree from its leaves.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Huffman Coding)</span></p>

1. Take the two least probable symbols in the alphabet. These two symbols will be given the longest codewords, which will have equal length and differ only in the last digit.
2. Combine these two symbols into a single symbol (with probability equal to the sum of the two), and repeat.

Since each step reduces the alphabet size by one, the algorithm assigns codewords to all symbols after $\lvert \mathcal{A}_X \rvert - 1$ steps. The codewords are obtained by concatenating the binary digits assigned at each step in reverse order.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Huffman Coding for Five Symbols)</span></p>

Let $\mathcal{A}_X = \lbrace \mathbf{a}, \mathbf{b}, \mathbf{c}, \mathbf{d}, \mathbf{e} \rbrace$ with $\mathcal{P}_X = \lbrace 0.25, 0.25, 0.2, 0.15, 0.15 \rbrace$.

| $a_i$ | $p_i$ | $h(p_i)$ | $l_i$ | $c(a_i)$ |
| --- | --- | --- | --- | --- |
| a | 0.25 | 2.0 | 2 | 00 |
| b | 0.25 | 2.0 | 2 | 10 |
| c | 0.2 | 2.3 | 2 | 11 |
| d | 0.15 | 2.7 | 3 | 010 |
| e | 0.15 | 2.7 | 3 | 011 |

The expected length is $L = 2.30$ bits, whereas the entropy is $H = 2.2855$ bits. $\square$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Optimality of Huffman Coding)</span></p>

There is no better symbol code for a source than the Huffman code. The proof relies on showing that giving the two least probable symbols equal-length codewords (the key step in the algorithm) cannot lead to a larger expected length than any other code. If a rival code assigned unequal lengths to the two least probable symbols, we could swap codewords to reduce expected length, contradicting the rival's optimality.

</div>

#### Constructing a Binary Tree Top-Down Is Suboptimal

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Greedy Top-Down vs. Huffman)</span></p>

For the ensemble $\mathcal{A}_X = \lbrace \mathbf{a}, \mathbf{b}, \mathbf{c}, \mathbf{d}, \mathbf{e}, \mathbf{f}, \mathbf{g} \rbrace$ with $\mathcal{P}_X = \lbrace 0.01, 0.24, 0.05, 0.20, 0.47, 0.01, 0.02 \rbrace$:

A greedy top-down method splits into subsets $\lbrace \mathbf{a}, \mathbf{b}, \mathbf{c}, \mathbf{d} \rbrace$ and $\lbrace \mathbf{e}, \mathbf{f}, \mathbf{g} \rbrace$ (each with probability $1/2$), giving expected length 2.53. The Huffman code yields expected length **1.97** -- significantly better.

The top-down approach (recursively bisecting into equiprobable subsets) is not always optimal because the constraint is not on *balanced splits* but on *total codeword cost*.

</div>

### 5.6 Disadvantages of the Huffman Code

#### Changing Ensemble

Huffman codes do not handle **changing ensemble probabilities** gracefully. If symbol frequencies vary with context (e.g., in English text the letter **u** is much more probable after a **q** than after an **e**), the Huffman code must be recomputed for each context, which is cumbersome. One workaround is to pre-compute a single probability distribution over the entire file, but this requires transmitting the code itself alongside the data, wasting bits.

#### The Extra Bit

A Huffman code achieves $H(X) \le L(C, X) < H(X) + 1$, so the overhead is between 0 and 1 bits per symbol. If $H(X)$ is large, this overhead is negligible. But for many applications the entropy may be close to 1 bit or even smaller per symbol, so the overhead $L(C,X) - H(X)$ can dominate the encoded file length.

For example, in the context `strings_of_ch`, one might predict the next nine symbols to be `aracters_` with probability 0.99. A Huffman code is obliged to use at least one bit per character, making a total cost of nine bits where virtually no information is being conveyed (0.13 bits in total). The entropy of English, given a good model, is about one bit per character, so a Huffman code is likely to be highly inefficient.

A traditional patch-up is to compress **blocks** of symbols (the extended sources $X^N$ from Chapter 4). The overhead per block is at most 1 bit, so the overhead per symbol is at most $1/N$ bits. However, this loses the elegant instantaneous decodeability and requires computing probabilities of all relevant strings.

#### Beyond Symbol Codes

The defects of Huffman codes are rectified by **arithmetic coding**, which dispenses with the restriction that each symbol must translate into an integer number of bits. Arithmetic coding is the topic of the next chapter.

### 5.7 Summary

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Chapter 5 Summary)</span></p>

- **Kraft inequality:** If a code is uniquely decodeable, its lengths must satisfy $\sum_i 2^{-l_i} \le 1$. For any lengths satisfying this, there exists a prefix code with those lengths.
- **Optimal source codelengths** are equal to the Shannon information contents: $l_i = \log_2 \frac{1}{p_i}$. Any choice of codelengths defines **implicit probabilities** $q_i = 2^{-l_i}/z$.
- **Relative entropy** $D_{\text{KL}}(\mathbf{p} \| \mathbf{q})$ measures how many bits per symbol are wasted by using a code whose implicit probabilities are $\mathbf{q}$ when the true distribution is $\mathbf{p}$.
- **Source coding theorem for symbol codes:** There exists a prefix code with $H(X) \le L(C, X) < H(X) + 1$.
- **Huffman coding** generates an optimal symbol code iteratively by combining the two least probable symbols at each step.

</div>

## Chapter 6 -- Stream Codes

This chapter discusses two data compression schemes: **arithmetic coding** and **Lempel--Ziv coding**. Arithmetic coding is a beautiful method that goes hand in hand with the philosophy that compression of data from a source entails probabilistic modelling of that source. Lempel--Ziv coding is a "universal" method, designed under the philosophy that a single compression algorithm should do a reasonable job for *any* source.

### 6.1 The Guessing Game

As a motivation for these two compression methods, consider the redundancy in a typical English text file. Such files have redundancy at several levels: ASCII characters with non-equal frequency, certain consecutive pairs of letters more probable than others, and entire words predictable given context and semantic understanding.

To illustrate the redundancy of English, we can imagine a **guessing game** in which an English speaker repeatedly attempts to predict the next character in a text file. For an alphabet of 27 symbols (26 upper case letters plus a space), the subject guesses the next character repeatedly until correct, and we record the number of guesses required.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Guessing Game)</span></p>

One sentence gave the following result when a human was asked to guess:

```
T H E R E - I S - N O - R E V E R S E - O N - A - M O T O R C Y C L E -
1 1 1 5 1 1 2 1 1 2 1 1 15 1 17 1 1 1 2 1 3 2 1 2 2 7 1 1 1 1 4 1 1 1 1 1
```

In many cases the next letter is guessed immediately in one guess. At the start of syllables, more guesses are needed.

</div>

The string of numbers '1, 1, 1, 5, 1, ...' is a time-varying mapping of the 27 letters onto the numbers $\lbrace 1, 2, \ldots, 27 \rbrace$. The total number of symbols has not been reduced, but since symbols like 1 and 2 are used much more frequently, this new string should be easy to compress.

The *uncompression* works by having an identical twin who plays the guessing game with us: stop him whenever he has made a number of guesses equal to the given number, and the correct letter is identified. Alternatively, with a single human and a window of context length $L$, we can tabulate answers for all $26 \times 27^L$ questions. Such a language model is called an $L$th order **Markov model**.

### 6.2 Arithmetic Codes

When we discussed variable-length symbol codes and the optimal Huffman algorithm, we concluded by pointing out two practical and theoretical problems with Huffman codes (section 5.6). These defects are rectified by **arithmetic codes**, invented by Elias, by Rissanen and by Pasco, and subsequently made practical by Witten *et al.* (1987).

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Arithmetic Code)</span></p>

In an **arithmetic code**, the probabilistic modelling is clearly separated from the encoding operation. The human predictor is replaced by a **probabilistic model** of the source. As each symbol is produced by the source, the probabilistic model supplies a **predictive distribution** over all possible values of the next symbol, that is, a list of positive numbers $\lbrace p_i \rbrace$ that sum to one.

The encoder makes use of the model's predictions to create a binary string. The decoder makes use of an identical twin of the model to interpret the binary string.

</div>

Let the source alphabet be $\mathcal{A}_X = \lbrace a_1, \ldots, a_I \rbrace$, and let the $I$th symbol $a_I$ have the special meaning 'end of transmission'. The source spits out a sequence $x_1, x_2, \ldots, x_n, \ldots$. A computer program is provided to the encoder that assigns a predictive probability distribution over $a_i$ given the sequence that has occurred thus far, $P(x_n = a_i \mid x_1, \ldots, x_{n-1})$. The receiver has an identical program that produces the same predictive probability distribution.

#### Concepts for Understanding Arithmetic Coding

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Binary Strings and Intervals)</span></p>

A binary transmission defines an interval within the real line from 0 to 1. For example, the string `01` is interpreted as a binary real number $0.01\ldots$, which corresponds to the interval $[0.01, 0.10)$ in binary, i.e., the interval $[0.25, 0.50)$ in base ten.

The longer string `01101` corresponds to a smaller interval $[0.01101, 0.01110)$. Because `01101` has the first string `01` as a prefix, the new interval is a sub-interval of $[0.01, 0.10)$.

A one-megabyte binary file ($2^{23}$ bits) is thus viewed as specifying a number between 0 and 1 to a precision of about two million decimal places.

</div>

We can also divide the real line $[0,1)$ into $I$ intervals of lengths equal to the probabilities $P(x_1 = a_i)$. We may then take each interval $a_i$ and subdivide it into intervals denoted $a_i a_1, a_i a_2, \ldots, a_i a_I$, such that the length of $a_i a_j$ is proportional to $P(x_2 = a_j \mid x_1 = a_i)$. Indeed the length of the interval $a_i a_j$ will be precisely the joint probability

$$P(x_1 = a_i,\, x_2 = a_j) = P(x_1 = a_i)\, P(x_2 = a_j \mid x_1 = a_i).$$

Iterating this procedure, the interval $[0,1)$ can be divided into a sequence of intervals corresponding to all possible finite length strings $x_1 x_2 \ldots x_N$, such that the length of an interval is equal to the probability of the string given the model.

#### Formulae Describing Arithmetic Coding

The intervals are defined in terms of the lower and upper **cumulative probabilities**:

$$Q_n(a_i \mid x_1, \ldots, x_{n-1}) \equiv \sum_{i'=1}^{i-1} P(x_n = a_{i'} \mid x_1, \ldots, x_{n-1}),$$

$$R_n(a_i \mid x_1, \ldots, x_{n-1}) \equiv \sum_{i'=1}^{i} P(x_n = a_{i'} \mid x_1, \ldots, x_{n-1}).$$

Starting with the first symbol, the intervals for '$a_1$', '$a_2$', and '$a_I$' are:

$$a_1 \leftrightarrow [Q_1(a_1), R_1(a_1)) = [0, P(x_1 = a_1)),$$

$$a_2 \leftrightarrow [Q_1(a_2), R_1(a_2)) = [P(x = a_1), P(x = a_1) + P(x = a_2)),$$

$$a_I \leftrightarrow [Q_1(a_I), R_1(a_I)) = [P(x_1 = a_1) + \ldots + P(x_1 = a_{I-1}), 1.0).$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Arithmetic Coding)</span></p>

Iterative procedure to find the interval $[u, v)$ for the string $x_1 x_2 \ldots x_N$:

```
u := 0.0
v := 1.0
p := v - u
for n = 1 to N {
    Compute the cumulative probabilities Q_n and R_n
    v := u + p * R_n(x_n | x_1, ..., x_{n-1})
    u := u + p * Q_n(x_n | x_1, ..., x_{n-1})
    p := v - u
}
```

To encode a string $x_1 x_2 \ldots x_N$, we locate the interval corresponding to $x_1 x_2 \ldots x_N$, and send a binary string whose interval lies within that interval.

</div>

#### Example: Compressing the Tosses of a Bent Coin

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Compressing `bbba☐`)</span></p>

Imagine watching a bent coin tossed some number of times with outcomes `a` and `b`, plus an end-of-file symbol `☐`. Let the source string be `bbba☐`. The probability of the next symbol given the string thus far:

| Context | $P(\mathtt{a})$ | $P(\mathtt{b})$ | $P(\text{☐})$ |
| --- | --- | --- | --- |
| (start) | 0.425 | 0.425 | 0.15 |
| `b` | 0.28 | 0.57 | 0.15 |
| `bb` | 0.21 | 0.64 | 0.15 |
| `bbb` | 0.17 | 0.68 | 0.15 |
| `bbba` | 0.28 | 0.57 | 0.15 |

The interval `b` is the middle 0.425 of $[0,1)$. The interval `bb` is the middle 0.567 of `b`, and so forth. The encoding works as follows: when the first `b` is observed, the encoder knows the string will start `01`, `10`, or `11`. The interval `bb` lies wholly within interval `1`, so the encoder can write the first bit: `1`. The third `b` narrows the interval a little. Only when `a` arrives can more bits be transmitted. The interval `bbba` lies within `1001`, so the encoder adds `001` to the `1`. When `☐` arrives, the encoding is completed by appending `11101`, giving the full encoding `100111101`.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Near-Optimality of Arithmetic Coding)</span></p>

The overhead required to terminate an arithmetic code message is never more than 2 bits, relative to the ideal message length given the probabilistic model $\mathcal{H}$, $h(\mathbf{x} \mid \mathcal{H}) = \log \frac{1}{P(\mathbf{x} \mid \mathcal{H})}$.

Arithmetic coding is very nearly optimal. The message length is always within two bits of the Shannon information content of the entire source string, so the expected message length is within two bits of the entropy of the entire message.

</div>

#### Decoding

The decoder receives the binary string (e.g., `100111101`) and passes along it one symbol at a time. First, the probabilities $P(\mathtt{a})$, $P(\mathtt{b})$, $P(\text{☐})$ are computed using the identical program, and the intervals `a`, `b`, `☐` are deduced. Once the first two bits `10` have been examined, it is certain that the original string started with `b`, since the interval `10` lies wholly within interval `b`. The decoder then uses the model to compute $P(\mathtt{a} \mid \mathtt{b})$, $P(\mathtt{b} \mid \mathtt{b})$, $P(\text{☐} \mid \mathtt{b})$ and deduce the boundaries of the intervals `ba`, `bb`, and `b☐`. Continuing, we decode the second `b` once we reach `1001`, the third `b` once we reach `100111`, and so forth, with the unambiguous identification of `bbba☐` once the whole binary string has been read.

#### The Big Picture

Notice that to communicate a string of $N$ letters, both the encoder and the decoder needed to compute only $N \lvert \mathcal{A} \rvert$ conditional probabilities -- the probabilities of each possible letter in each context actually encountered. This contrasts with Huffman coding with a large block size, where *all* block sequences must be considered.

Arithmetic coding is flexible: it can be used with any source alphabet and any encoded alphabet. The sizes of both alphabets can change with time. It can be used with any probability distribution, which can change utterly from context to context.

#### How the Probabilistic Model Might Make Its Predictions

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Bayesian Model for Bent Coin)</span></p>

The model can naturally be produced by a Bayesian model. The example above was generated using a simple model that always assigns a probability of 0.15 to `☐`, and assigns the remaining 0.85 to `a` and `b` divided in proportion to probabilities given by **Laplace's rule**:

$$P_{\text{L}}(\mathtt{a} \mid x_1, \ldots, x_{n-1}) = \frac{F_{\mathtt{a}} + 1}{F_{\mathtt{a}} + F_{\mathtt{b}} + 2},$$

where $F_{\mathtt{a}}(x_1, \ldots, x_{n-1})$ is the number of times that `a` has occurred so far, and $F_{\mathtt{b}}$ is the count of `b`s. This is a simple Bayesian model that expects and adapts to a non-equal frequency of use of `a` and `b`.

</div>

#### Details of the Bayesian Model

The model is described using parameters $p_{\text{☐}}$, $p_{\mathtt{a}}$ and $p_{\mathtt{b}}$:

1. It is assumed that the length of the string $l$ has an exponential probability distribution: $P(l) = (1 - p_{\text{☐}})^l p_{\text{☐}}$. This corresponds to assuming a constant probability $p_{\text{☐}}$ for the termination symbol `☐` at each character.

2. It is assumed that the non-terminal characters are selected independently at random from an ensemble with probabilities $\mathcal{P} = \lbrace p_{\mathtt{a}}, p_{\mathtt{b}} \rbrace$; the probability $p_{\mathtt{a}}$ is fixed throughout the string to some unknown value. The probability, given $p_{\mathtt{a}}$, that an unterminated string of length $F$ that contains $\lbrace F_{\mathtt{a}}, F_{\mathtt{b}} \rbrace$ counts of the two outcomes is the Bernoulli distribution: $P(\mathbf{s} \mid p_{\mathtt{a}}, F) = p_{\mathtt{a}}^{F_{\mathtt{a}}} (1 - p_{\mathtt{a}})^{F_{\mathtt{b}}}$.

3. We assume a uniform prior distribution for $p_{\mathtt{a}}$: $P(p_{\mathtt{a}}) = 1$, $p_{\mathtt{a}} \in [0, 1]$, and define $p_{\mathtt{b}} \equiv 1 - p_{\mathtt{a}}$.

The key result is the predictive distribution for the next symbol given the string so far. The probability that the next character is `a` or `b` (assuming it is not `☐`) is precisely **Laplace's rule**.

#### Comparison of Compression Methods

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Laplace vs. Dirichlet Models)</span></p>

Compare the expected message length when an ASCII file is compressed by the following three methods:

- **Huffman-with-header:** Read the whole file, find the empirical frequency of each symbol, construct a Huffman code, transmit the code by transmitting the lengths of the Huffman codewords, then transmit the file.

- **Arithmetic code using the Laplace model:**

$$P_{\text{L}}(\mathtt{a} \mid x_1, \ldots, x_{n-1}) = \frac{F_{\mathtt{a}} + 1}{\sum_{\mathtt{a}'} (F_{\mathtt{a}'} + 1)}.$$

- **Arithmetic code using a Dirichlet model:**

$$P_{\text{D}}(\mathtt{a} \mid x_1, \ldots, x_{n-1}) = \frac{F_{\mathtt{a}} + \alpha}{\sum_{\mathtt{a}'} (F_{\mathtt{a}'} + \alpha)},$$

where $\alpha$ is fixed to a number such as 0.01. A small value of $\alpha$ corresponds to a more responsive version of the Laplace model; $\alpha = 1$ reproduces the Laplace model.

</div>

### 6.3 Further Applications of Arithmetic Coding

#### Efficient Generation of Random Samples

Arithmetic coding not only offers a way to compress strings; it also offers a way to **generate random strings** from a model. To generate a sample, all we need to do is feed ordinary random bits into an arithmetic *decoder* for that model. An infinite random bit sequence corresponds to the selection of a point at random from the line $[0, 1)$, so the decoder will select a string at random from the assumed distribution. This arithmetic method is guaranteed to use very nearly the smallest number of random bits possible.

#### Efficient Data-Entry Devices

By inverting an arithmetic coder, we can obtain an information-efficient text entry device. In data compression, the aim is to map text into a *small* number of bits. In text entry, we want a small sequence of gestures to produce our intended text. The Dasher system zooms in on the unit interval to locate the interval corresponding to the intended string; a language model controls the sizes of the intervals such that probable strings are quick and easy to identify.

### 6.4 Lempel--Ziv Coding

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Lempel--Ziv Coding)</span></p>

The **Lempel--Ziv** algorithms, widely used for data compression (e.g., `compress` and `gzip`), are different in philosophy to arithmetic coding. There is no separation between modelling and coding, and no opportunity for explicit modelling.

The method of compression is to replace a substring with a **pointer** to an earlier occurrence of the same substring. The string is parsed into an ordered **dictionary** of substrings that have not appeared before. Each substring is encoded by giving a pointer to the earlier occurrence of its prefix and then sending the extra bit by which the new substring differs from the earlier one.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Basic Lempel--Ziv Encoding)</span></p>

If the string is `1011010100010...`, we parse it into a dictionary of substrings: $\lambda$, `1`, `0`, `11`, `01`, `010`, `00`, `10`, ... (where $\lambda$ is the empty substring). Each new substring is one bit longer than a substring that has occurred earlier. The code for this example:

| source substrings | $\lambda$ | `1` | `0` | `11` | `01` | `010` | `00` | `10` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| $s(n)$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| (pointer, bit) | | $(,1)$ | $(0,0)$ | $(01,1)$ | $(10,1)$ | $(100,0)$ | $(010,0)$ | $(001,0)$ |

The encoded string is `100011011000100010001000010`. In this simple case, the encoding is actually *longer* than the source string, because there was no obvious redundancy.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Improving the Basic Algorithm)</span></p>

One reason the basic algorithm lengthens some strings is that it transmits unnecessary bits. Once a substring in the dictionary has been joined by both of its children, it will not be needed again; so at that point we could drop it and shuffle the remaining substrings, reducing the length of subsequent pointer messages. A second unnecessary overhead is the transmission of the new bit the second time a prefix is used -- we can be sure of the identity of the next bit.

</div>

#### Theoretical Properties

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Universality of Lempel--Ziv)</span></p>

Given any **ergodic** source (i.e., one that is memoryless on sufficiently long timescales), the Lempel--Ziv algorithm can be proven *asymptotically* to compress down to the entropy of the source. This is why it is called a **universal** compression algorithm.

However, it achieves its compression only by *memorizing* substrings that have happened so that it has a short name for them the next time they occur. The asymptotic timescale on which this universal performance is achieved may, for many sources, be unfeasibly long.

</div>

#### Common Ground

In principle, one can design adaptive probabilistic models, and thence arithmetic codes, that are 'universal' -- models that will asymptotically compress *any source in some class* to within some factor (preferably 1) of its entropy. However, for practical purposes, such universal models can only be constructed if the class of sources is severely restricted.

### 6.5 Demonstration

#### Compression of a Text File

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Compression Comparison -- Text File)</span></p>

Comparison of compression algorithms applied to a text file (LaTeX source, 20,942 bytes):

| Method | Compressed size | Uncompression time |
| --- | --- | --- |
| Laplace model | 12,974 (61%) | 0.32 s |
| `gzip` | 8,177 (39%) | **0.01 s** |
| `compress` | 10,816 (51%) | 0.05 s |
| `bzip` | 7,495 (36%) | -- |
| `bzip2` | 7,640 (36%) | -- |
| `ppmz` | **6,800 (32%)** | -- |

</div>

#### Compression of a Sparse File

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Compression Comparison -- Sparse File)</span></p>

Comparison of compression algorithms applied to a random file of $10^6$ characters (99% `0`s and 1% `1`s). An ideal model would compress the file to about $10^6 H_2(0.01)/8 \simeq 10\,100$ bytes:

| Method | Compressed size |
| --- | --- |
| Laplace model | 14,143 (1.4%) |
| `gzip` | 20,646 (2.1%) |
| `gzip --best+` | 15,553 (1.6%) |
| `compress` | 14,785 (1.5%) |
| `bzip` | 10,903 (1.09%) |
| `bzip2` | 11,260 (1.12%) |
| `ppmz` | **10,447 (1.04%)** |

The Laplace model is quite well matched to this source. The benchmark arithmetic coder gives good performance, followed closely by `compress`; `gzip` is worst. The `ppmz` compressor compresses the best of all, but takes much more computer time.

</div>

### 6.6 Summary

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Chapter 6 Summary)</span></p>

In Chapters 4--6 we have studied three classes of data compression codes:

- **Fixed-length block codes** (Chapter 4): Mappings from a fixed number of source symbols to a fixed-length binary message. Only a tiny fraction of source strings are given an encoding. These codes were fun for identifying the entropy as the measure of compressibility but are of little practical use.

- **Symbol codes** (Chapter 5): Variable-length codes for each symbol, with codelengths being integer lengths determined by the symbol probabilities. Huffman's algorithm constructs an optimal symbol code. Every source string has a uniquely decodeable encoding. If the source is well matched to the assumed distribution, the expected length per character lies in $[H, H+1)$. The symbol code must emit at least one bit per source symbol; compression below one bit per symbol can be achieved only by blocking.

- **Stream codes** (Chapter 6): The distinctive property is that they are not constrained to emit at least one bit for every symbol read. So large numbers of source symbols may be coded into a smaller number of bits.
  - *Arithmetic codes* combine a probabilistic model with an encoding algorithm that identifies each string with a sub-interval of $[0,1)$ of size equal to the probability of that string under the model. This code is almost optimal: the compressed length closely matches the Shannon information content. Arithmetic codes fit with the philosophy that good compression requires *data modelling*, in the form of an adaptive Bayesian model.
  - *Lempel--Ziv codes* are adaptive in the sense that they memorize strings that have already occurred. They are built on the philosophy that we don't know the probability distribution of the source, and we want a compression algorithm that will perform reasonably well whatever that distribution is.

Both arithmetic codes and Lempel--Ziv codes will fail to decode correctly if any of the bits of the compressed file are altered. So if compressed files are to be stored or transmitted over noisy media, error-correcting codes will be essential. Reliable communication over unreliable channels is the topic of Part II.

</div>

## Chapter 7 -- Codes for Integers

This chapter is an aside on the coding of integers. To discuss the coding of integers we need some definitions.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Binary Representations of Integers)</span></p>

- **Standard binary representation** of a positive integer $n$: denoted by $c_{\text{b}}(n)$, e.g., $c_{\text{b}}(5) = \mathtt{101}$, $c_{\text{b}}(45) = \mathtt{101101}$.

- **Standard binary length** of a positive integer $n$: $l_{\text{b}}(n)$, the length of the string $c_{\text{b}}(n)$. For example, $l_{\text{b}}(5) = 3$, $l_{\text{b}}(45) = 6$.

- **Headless binary representation** of a positive integer $n$: denoted by $c_{\text{B}}(n)$, e.g., $c_{\text{B}}(5) = \mathtt{01}$, $c_{\text{B}}(45) = \mathtt{01101}$, and $c_{\text{B}}(1) = \lambda$ (the null string). This is obtained by stripping the leading `1` from the standard binary representation.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why Standard Binary Is Not Uniquely Decodeable)</span></p>

The standard binary representation $c_{\text{b}}(n)$ is *not* a uniquely decodeable code for integers since there is no way of knowing when an integer has ended. For example, $c_{\text{b}}(5)c_{\text{b}}(5)$ is identical to $c_{\text{b}}(45)$. It would be uniquely decodeable if we knew the standard binary length of each integer before it was received.

</div>

Two strategies can be distinguished for making uniquely decodeable codes for integers:

1. **Self-delimiting codes.** We first communicate somehow the length of the integer, $l_{\text{b}}(n)$, which is also a positive integer; then communicate the original integer $n$ itself using $c_{\text{B}}(n)$.

2. **Codes with 'end of file' characters.** We code the integer into blocks of length $b$ bits, and reserve one of the $2^b$ symbols to have the special meaning 'end of file'.

### The Unary Code

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Unary Code)</span></p>

The **unary code** encodes an integer $n$ by sending a string of $n-1$ zeros followed by a 1:

| $n$ | $c_{\text{U}}(n)$ |
| --- | --- |
| 1 | `1` |
| 2 | `01` |
| 3 | `001` |
| 4 | `0001` |
| 5 | `00001` |

The unary code has length $l_{\text{U}}(n) = n$. It is the optimal code for integers if the probability distribution over $n$ is $p_{\text{U}}(n) = 2^{-n}$.

</div>

### Self-Delimiting Codes

We can use the unary code to encode the *length* of the binary encoding of $n$ and make a self-delimiting code.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Code $C_\alpha$)</span></p>

**Code $C_\alpha$:** We send the unary code for $l_{\text{b}}(n)$, followed by the headless binary representation of $n$:

$$c_\alpha(n) = c_{\text{U}}[l_{\text{b}}(n)] \, c_{\text{B}}(n).$$

The codeword $c_\alpha(n)$ has length $l_\alpha(n) = 2 l_{\text{b}}(n) - 1$.

| $n$ | $c_{\text{b}}(n)$ | $l_{\text{b}}(n)$ | $c_\alpha(n)$ |
| --- | --- | --- | --- |
| 1 | `1` | 1 | `1` |
| 2 | `10` | 2 | `010` |
| 3 | `11` | 2 | `011` |
| 4 | `100` | 3 | `00100` |
| 5 | `101` | 3 | `00101` |
| 6 | `110` | 3 | `00110` |
| 45 | `101101` | 6 | `00000101101` |

We might equivalently view $c_\alpha(n)$ as consisting of a string of $(l_{\text{b}}(n) - 1)$ zeroes followed by the standard binary representation of $n$, $c_{\text{b}}(n)$.

</div>

The implicit probability distribution over $n$ for the code $C_\alpha$ is separable into the product of a probability distribution over the length $l$:

$$P(l) = 2^{-l},$$

and a uniform distribution over integers having that length:

$$P(n \mid l) = \begin{cases} 2^{-l+1} & l_{\text{b}}(n) = l \\ 0 & \text{otherwise}. \end{cases}$$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Improving on $C_\alpha$)</span></p>

For the code $C_\alpha$, the header that communicates the length always occupies the same number of bits as the standard binary representation of the integer (give or take one). If we are expecting to encounter large integers (large files), this representation seems suboptimal, since it leads to all files occupying a size that is double their original uncoded size. Instead of using the unary code to encode the length $l_{\text{b}}(n)$, we could use $C_\alpha$ itself.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Codes $C_\beta$, $C_\gamma$, $C_\delta$)</span></p>

By iterating the procedure of encoding the length using the previous code, we obtain a sequence of progressively more efficient codes:

**Code $C_\beta$:** We send the length $l_{\text{b}}(n)$ using $C_\alpha$, followed by the headless binary representation of $n$:

$$c_\beta(n) = c_\alpha[l_{\text{b}}(n)] \, c_{\text{B}}(n).$$

**Code $C_\gamma$:**

$$c_\gamma(n) = c_\beta[l_{\text{b}}(n)] \, c_{\text{B}}(n).$$

**Code $C_\delta$:**

$$c_\delta(n) = c_\gamma[l_{\text{b}}(n)] \, c_{\text{B}}(n).$$

| $n$ | $c_\beta(n)$ | $c_\gamma(n)$ |
| --- | --- | --- |
| 1 | `1` | `1` |
| 2 | `0100` | `01000` |
| 3 | `0101` | `01001` |
| 4 | `01100` | `0101100` |
| 5 | `01101` | `0101101` |
| 6 | `01110` | `0101110` |
| 45 | `0011001101` | `0111001101` |

</div>

### Codes with End-of-File Symbols

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Byte-Based Codes with End-of-File Symbols)</span></p>

We can also make byte-based representations. If we encode the number in some base, for example decimal, we can represent each digit in a byte. To represent a digit from 0 to 9 in a byte we need four bits. Because $2^4 = 16$, this leaves 6 extra four-bit symbols $\lbrace \mathtt{1010}, \mathtt{1011}, \mathtt{1100}, \mathtt{1101}, \mathtt{1110}, \mathtt{1111} \rbrace$ that correspond to no decimal digit. We can use these as end-of-file symbols to indicate the end of our positive integer.

A more efficient code would encode the integer into base 15, and use just the sixteenth symbol `1111` as the punctuation character. Generalizing, we can make similar byte-based codes for integers in bases 3 and 7, and in any base of the form $2^n - 1$.

These codes are almost complete. (Recall that a code is 'complete' if it satisfies the Kraft inequality with equality.) Their remaining inefficiency is that they provide the ability to encode the integer zero and the empty string, neither of which was required.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Encoding 'Claude Shannon')</span></p>

To illustrate these codes, consider a small file consisting of just 14 characters: `Claude Shannon`.

- If we map the ASCII characters onto seven-bit symbols (e.g., `C` $= 67$, `l` $= 108$, etc.), this 14-character file corresponds to the integer $n = 167\,987\,786\,364\,950\,891\,085\,602\,469\,870$ (decimal).

- The unary code for $n$ consists of this many (less one) zeroes, followed by a one. If all the oceans were turned into ink, and if we wrote a hundred bits with every cubic millimeter, there might be enough ink to write $c_{\text{U}}(n)$.

- The standard binary representation of $n$ is a length-98 sequence of bits.

</div>

### Comparing the Codes

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(No Universally Superior Code)</span></p>

Any complete code corresponds to a prior for which it is optimal; you should not say that any other code is superior to it. Other codes are optimal for other priors. These implicit priors should be thought about so as to achieve the best code for one's application.

One cannot, for free, switch from one code to another, choosing whichever is shorter. If one were to do this, it would be necessary to lengthen the message in some way to indicate which code is being used. If this is done by a single leading bit, the resulting code is suboptimal because it fails the Kraft equality.

</div>

Another way to compare codes for integers is to consider a sequence of probability distributions, such as monotonic probability distributions over $n \ge 1$, and rank the codes as to how well they encode *any* of these distributions. A code is called a **universal** code if for any distribution in a given class, it encodes into an average length that is within some factor of the ideal average length.

### Elias's Universal Code $C_\omega$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Elias's Encoder for $C_\omega$)</span></p>

Elias's 'universal code for integers' (Elias, 1975) effectively chooses from all the codes $C_\alpha, C_\beta, \ldots$ by sending a sequence of messages each of which encodes the length of the next message, and indicates by a single bit whether or not that message is the final integer. Because a length is a positive integer and all positive integers begin with '1', all the leading 1s can be omitted.

```
Write '0'
Loop {
    If ⌊log n⌋ = 0 halt
    Prepend c_b(n) to the written string
    n := ⌊log n⌋
}
```

The encoding is generated from right to left.

| $n$ | $c_\omega(n)$ |
| --- | --- |
| 1 | `0` |
| 2 | `100` |
| 3 | `110` |
| 4 | `101000` |
| 5 | `101010` |
| 6 | `101100` |
| 7 | `101110` |
| 8 | `1110000` |
| 16 | `10100100000` |
| 45 | `1011001101` |

</div>

## Part II -- Noisy-Channel Coding

## Chapter 8 -- Dependent Random Variables

In the preceding chapters on data compression we concentrated on random vectors $\mathbf{x}$ coming from separable probability distributions in which each component $x_n$ is independent of the others. This chapter considers **joint ensembles** in which the random variables are dependent. This material has two motivations: (1) real-world data have interesting correlations, so good data compression requires models that include dependences; (2) a noisy channel with input $x$ and output $y$ defines a joint ensemble in which $x$ and $y$ are dependent -- if they were independent, communication over the channel would be impossible.

### 8.1 More about Entropy

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Joint Entropy)</span></p>

The **joint entropy** of $X, Y$ is:

$$H(X, Y) = \sum_{xy \in \mathcal{A}_X \mathcal{A}_Y} P(x, y) \log \frac{1}{P(x, y)}.$$

Entropy is additive for independent random variables:

$$H(X, Y) = H(X) + H(Y) \quad \text{iff} \quad P(x, y) = P(x) P(y).$$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Conditional Entropy)</span></p>

The **conditional entropy of $X$ given $y = b_k$** is the entropy of the probability distribution $P(x \mid y = b_k)$:

$$H(X \mid y = b_k) \equiv \sum_{x \in \mathcal{A}_X} P(x \mid y = b_k) \log \frac{1}{P(x \mid y = b_k)}.$$

The **conditional entropy of $X$ given $Y$** is the average, over $y$, of the conditional entropy of $X$ given $y$:

$$H(X \mid Y) \equiv \sum_{y \in \mathcal{A}_Y} P(y) \left[ \sum_{x \in \mathcal{A}_X} P(x \mid y) \log \frac{1}{P(x \mid y)} \right] = \sum_{xy \in \mathcal{A}_X \mathcal{A}_Y} P(x, y) \log \frac{1}{P(x \mid y)}.$$

This measures the average uncertainty that remains about $x$ when $y$ is known. The **marginal entropy** of $X$ is another name for $H(X)$, used to contrast it with the conditional entropies.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Chain Rule for Entropy)</span></p>

From the product rule for probabilities we obtain the **chain rule for information content**:

$$\log \frac{1}{P(x, y)} = \log \frac{1}{P(x)} + \log \frac{1}{P(y \mid x)},$$

so $h(x, y) = h(x) + h(y \mid x)$. In words, the information content of $(x, y)$ equals the information content of $x$ plus the information content of $y$ given $x$.

The joint entropy, conditional entropy, and marginal entropy are related by:

$$H(X, Y) = H(X) + H(Y \mid X) = H(Y) + H(X \mid Y).$$

In words, the uncertainty of $(X, Y)$ is the uncertainty of $X$ plus the uncertainty of $Y$ given $X$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Mutual Information)</span></p>

The **mutual information** between $X$ and $Y$ is:

$$I(X; Y) \equiv H(X) - H(X \mid Y),$$

and satisfies $I(X; Y) = I(Y; X)$ and $I(X; Y) \ge 0$. It measures the average reduction in uncertainty about $x$ that results from learning the value of $y$; **or vice versa**, the average amount of information that $x$ conveys about $y$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Conditional Mutual Information)</span></p>

The **conditional mutual information** between $X$ and $Y$ given $z = c_k$ is the mutual information between $X$ and $Y$ in the joint ensemble $P(x, y \mid z = c_k)$:

$$I(X; Y \mid z = c_k) = H(X \mid z = c_k) - H(X \mid Y, z = c_k).$$

The **conditional mutual information** between $X$ and $Y$ given $Z$ is the average over $z$:

$$I(X; Y \mid Z) = H(X \mid Z) - H(X \mid Y, Z).$$

No other 'three-term entropies' are defined. For example, expressions such as $I(X; Y; Z)$ and $I(X \mid Y; Z)$ are illegal. However, one may put conjunctions of arbitrary numbers of variables in each of the three slots in $I(X; Y \mid Z)$ -- for example, $I(A, B; C, D \mid E, F)$ is fine: it measures how much information on average $c$ and $d$ convey about $a$ and $b$, assuming $e$ and $f$ are known.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Relationship between Entropy Quantities)</span></p>

The total entropy $H(X, Y)$ of a joint ensemble can be decomposed as:

$$H(X, Y) = H(X \mid Y) + I(X; Y) + H(Y \mid X),$$

where $H(X) = H(X \mid Y) + I(X; Y)$ and $H(Y) = H(Y \mid X) + I(X; Y)$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Conditioning Reduces Entropy)</span></p>

The conditional entropy $H(X \mid Y)$ satisfies:

$$H(X \mid Y) \le H(X),$$

with equality only if $P(y \mid x) = P(y)$ for all $x$ and $y$ (i.e., $X$ and $Y$ are independent). While it is possible for $H(X \mid y = b_k)$ to exceed $H(X)$ for a particular value $b_k$, the average $H(X \mid Y)$ is always less than or equal to $H(X)$. Data are helpful -- they do not increase uncertainty, on average.

**Proof.** We rewrite $H(X \mid Y)$ by applying Bayes' theorem:

$$H(X \mid Y) = \sum_{xy} P(x, y) \log \frac{1}{P(x \mid y)} = \sum_{xy} P(x) P(y \mid x) \log \frac{P(y)}{P(y \mid x) P(x)}.$$

Splitting the logarithm yields:

$$H(X \mid Y) = \sum_x P(x) \log \frac{1}{P(x)} + \sum_x P(x) \sum_y P(y \mid x) \log \frac{P(y)}{P(y \mid x)} = H(X) + \sum_x P(x) \left[ -D_{\mathrm{KL}}(P(y \mid x) \| P(y)) \right].$$

Since $D_{\mathrm{KL}} \ge 0$, we have $H(X \mid Y) \le H(X) + 0 = H(X)$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Symmetry and Non-Negativity of Mutual Information)</span></p>

Mutual information is symmetric and non-negative:

$$I(X; Y) = H(X) - H(X \mid Y) = H(Y) - H(Y \mid X).$$

**Proof of symmetry:**

$$I(X; Y) = \sum_{xy} P(x, y) \log \frac{P(x \mid y)}{P(x)} = \sum_{xy} P(x, y) \log \frac{P(x, y)}{P(x) P(y)}.$$

This expression is symmetric in $x$ and $y$, so $I(X; Y) = I(Y; X)$.

**Proof of non-negativity (via KL divergence):**

$$I(X; Y) = \sum_{x, y} P(x, y) \log \frac{P(x, y)}{P(x) P(y)} = D_{\mathrm{KL}}(P(x, y) \| P(x) P(y)) \ge 0,$$

with equality only if $P(x, y) = P(x) P(y)$, i.e., $X$ and $Y$ are independent.

**Proof of non-negativity (via Jensen's inequality):**

$$-\sum_{x, y} P(x, y) \log \frac{P(x) P(y)}{P(x, y)} \ge -\log \sum_{x, y} P(x, y) \cdot \frac{P(x) P(y)}{P(x, y)} \cdot 1 = -\log \sum_{x, y} P(x) P(y) = \log 1 = 0.$$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Entropy Distance)</span></p>

The **entropy distance** between two random variables can be defined as the difference between their joint entropy and their mutual information:

$$D_H(X, Y) \equiv H(X, Y) - I(X; Y).$$

This satisfies the axioms for a distance: $D_H(X, Y) \ge 0$, $D_H(X, X) = 0$, $D_H(X, Y) = D_H(Y, X)$, and $D_H(X, Z) \le D_H(X, Y) + D_H(Y, Z)$.

</div>

### 8.3 The Data-Processing Theorem

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Data-Processing Theorem)</span></p>

The data-processing theorem states that **data processing can only destroy information**. If three variables $W$, $D$, $R$ form a Markov chain:

$$W \to D \to R,$$

meaning $P(w, d, r) = P(w) P(d \mid w) P(r \mid d)$, then the average information that $R$ conveys about $W$ is less than or equal to the average information that $D$ conveys about $W$:

$$I(W; R) \le I(W; D).$$

**Proof.** For any joint ensemble $XYZ$, the chain rule for mutual information gives:

$$I(X; Y, Z) = I(X; Y) + I(X; Z \mid Y).$$

Since $W \to D \to R$ is a Markov chain, $W$ and $R$ are independent given $D$, so $I(W; R \mid D) = 0$. Applying the chain rule:

$$I(W; D, R) = I(W; D) \quad \text{and} \quad I(W; D, R) = I(W; R) + I(W; D \mid R),$$

so $I(W; R) = I(W; D) - I(W; D \mid R) \le I(W; D)$.

</div>

This theorem is as much a caution about our definition of 'information' as it is a caution about data processing.

### 8.4 Venn Diagrams for Entropies

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why Venn Diagrams Are Misleading for Entropies)</span></p>

Many texts depict entropies in the form of a Venn diagram, but this representation is misleading for at least two reasons:

1. Venn diagrams suggest that the 'sets' $H(X)$ and $H(Y)$ depict actual sets, but the objects and members of these 'sets' are unclear. This encourages inappropriate analogies -- for example, some students imagine that a random outcome $(x, y)$ corresponds to a point in the diagram, confusing entropies with probabilities.

2. For two random variables, all areas ($H(X \mid Y)$, $I(X; Y)$, $H(Y \mid X)$) are non-negative, so the Venn diagram works. But for three-variable ensembles, areas that appear positive may correspond to **negative quantities**. Consider $(X, Y, Z)$ where $x, y \in \lbrace 0, 1 \rbrace$ are independent binary variables and $z = x + y \bmod 2$. Then $I(X; Y) = 0$ (independent), but $I(X; Y \mid Z) = 1$ bit (knowing $z$ makes $X$ and $Y$ dependent). The 'area' that the Venn diagram labels as the three-way mutual information must be $-1$ bits.

The Venn diagram representation is only valid if one is aware that positive areas may represent negative quantities.

</div>

## Chapter 9 -- Communication over a Noisy Channel

### 9.1 The Big Picture

In Chapters 4--6, we discussed source coding with block codes, symbol codes, and stream codes. We implicitly assumed that the channel from the compressor to the decompressor was noise-free. Real channels are noisy. The aim of **channel coding** is to make the noisy channel behave like a noiseless channel. We assume the data to be transmitted has been through a good compressor, so the bit stream has no obvious redundancy. The channel code will put back redundancy of a special sort, designed to make the noisy received signal decodable.

The information transmitted is given by the **mutual information** between the source and the received signal: the entropy of the source minus the conditional entropy of the source given the received signal.

### 9.2 Review of Probability and Information

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Joint Distribution $XY$)</span></p>

Consider the joint distribution from exercise 8.6:

| $P(x,y)$ | $x{=}1$ | $x{=}2$ | $x{=}3$ | $x{=}4$ | $P(y)$ |
| --- | --- | --- | --- | --- | --- |
| $y{=}1$ | $1/8$ | $1/16$ | $1/32$ | $1/32$ | $1/4$ |
| $y{=}2$ | $1/16$ | $1/8$ | $1/32$ | $1/32$ | $1/4$ |
| $y{=}3$ | $1/16$ | $1/16$ | $1/16$ | $1/16$ | $1/4$ |
| $y{=}4$ | $1/4$ | $0$ | $0$ | $0$ | $1/4$ |
| $P(x)$ | $1/2$ | $1/4$ | $1/8$ | $1/8$ | |

The joint entropy is $H(X, Y) = 27/8$ bits. The marginal entropies are $H(X) = 7/4$ bits and $H(Y) = 2$ bits.

The conditional distributions $P(x \mid y)$ and their entropies are:

| $P(x \mid y)$ | $x{=}1$ | $x{=}2$ | $x{=}3$ | $x{=}4$ | $H(X \mid y)$/bits |
| --- | --- | --- | --- | --- | --- |
| $y{=}1$ | $1/2$ | $1/4$ | $1/8$ | $1/8$ | $7/4$ |
| $y{=}2$ | $1/4$ | $1/2$ | $1/8$ | $1/8$ | $7/4$ |
| $y{=}3$ | $1/4$ | $1/4$ | $1/4$ | $1/4$ | $2$ |
| $y{=}4$ | $1$ | $0$ | $0$ | $0$ | $0$ |

So $H(X \mid Y) = 11/8$ bits. Note that $H(X \mid y{=}4) = 0 < H(X)$ while $H(X \mid y{=}3) = 2 > H(X)$. Learning $y$ can *increase* our uncertainty about $x$ for specific values of $y$, but on average $H(X \mid Y) < H(X)$.

The mutual information is $I(X; Y) = H(X) - H(X \mid Y) = 7/4 - 11/8 = 3/8$ bits.

</div>

### 9.3 Noisy Channels

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Discrete Memoryless Channel)</span></p>

A **discrete memoryless channel** $Q$ is characterized by an input alphabet $\mathcal{A}_X$, an output alphabet $\mathcal{A}_Y$, and a set of conditional probability distributions $P(y \mid x)$, one for each $x \in \mathcal{A}_X$. These **transition probabilities** may be written in a matrix:

$$Q_{j|i} = P(y = b_j \mid x = a_i).$$

Each column of $\mathbf{Q}$ is a probability vector. The output distribution $\mathbf{p}_Y$ is obtained from the input distribution $\mathbf{p}_X$ by right-multiplication:

$$\mathbf{p}_Y = \mathbf{Q} \mathbf{p}_X.$$

</div>

Some useful model channels:

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Binary Symmetric Channel)</span></p>

The **binary symmetric channel** (BSC) has $\mathcal{A}_X = \lbrace 0, 1 \rbrace$, $\mathcal{A}_Y = \lbrace 0, 1 \rbrace$:

$$P(y = 0 \mid x = 0) = 1 - f, \quad P(y = 0 \mid x = 1) = f,$$

$$P(y = 1 \mid x = 0) = f, \quad P(y = 1 \mid x = 1) = 1 - f.$$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Binary Erasure Channel)</span></p>

The **binary erasure channel** (BEC) has $\mathcal{A}_X = \lbrace 0, 1 \rbrace$, $\mathcal{A}_Y = \lbrace 0, ?, 1 \rbrace$:

$$P(y = 0 \mid x = 0) = 1 - f, \quad P(y = 0 \mid x = 1) = 0,$$

$$P(y = {?} \mid x = 0) = f, \quad P(y = {?} \mid x = 1) = f,$$

$$P(y = 1 \mid x = 0) = 0, \quad P(y = 1 \mid x = 1) = 1 - f.$$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Noisy Typewriter)</span></p>

The **noisy typewriter** has $\mathcal{A}_X = \mathcal{A}_Y = \lbrace \texttt{A}, \texttt{B}, \ldots, \texttt{Z}, \texttt{-} \rbrace$ (27 letters arranged in a circle). When the typist attempts to type a letter, the output is the intended letter or one of its two neighbours, each with probability $1/3$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Z Channel)</span></p>

The **Z channel** has $\mathcal{A}_X = \lbrace 0, 1 \rbrace$, $\mathcal{A}_Y = \lbrace 0, 1 \rbrace$:

$$P(y = 0 \mid x = 0) = 1, \quad P(y = 0 \mid x = 1) = f,$$

$$P(y = 1 \mid x = 0) = 0, \quad P(y = 1 \mid x = 1) = 1 - f.$$

Input $0$ is always transmitted correctly, but input $1$ is flipped to $0$ with probability $f$.

</div>

### 9.4 Inferring the Input Given the Output

If we assume that the input $x$ to a channel comes from an ensemble $X$, then we obtain a joint ensemble $XY$ with the joint distribution:

$$P(x, y) = P(y \mid x) P(x).$$

Given a received symbol $y$, we can write down the posterior distribution of the input using Bayes' theorem:

$$P(x \mid y) = \frac{P(y \mid x) P(x)}{\sum_{x'} P(y \mid x') P(x')}.$$

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(BSC Posterior Inference)</span></p>

Consider a BSC with $f = 0.15$ and input ensemble $\mathcal{P}_X : \lbrace p_0 = 0.9, p_1 = 0.1 \rbrace$. If we observe $y = 1$:

$$P(x{=}1 \mid y{=}1) = \frac{0.85 \times 0.1}{0.85 \times 0.1 + 0.15 \times 0.9} = \frac{0.085}{0.22} = 0.39.$$

Thus '$x{=}1$' is still less probable than '$x{=}0$', although it is not as improbable as it was before.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Z Channel Posterior Inference)</span></p>

Consider a Z channel with $f = 0.15$ and $\mathcal{P}_X : \lbrace p_0 = 0.9, p_1 = 0.1 \rbrace$. If we observe $y = 1$:

$$P(x{=}1 \mid y{=}1) = \frac{0.85 \times 0.1}{0.85 \times 0.1 + 0 \times 0.9} = \frac{0.085}{0.085} = 1.0.$$

Given the output $y = 1$, we become certain of the input, since only $x = 1$ can produce output $1$ in the Z channel.

</div>

### 9.5 Information Conveyed by a Channel

We now consider how much information can be communicated through a channel. Assuming a particular input ensemble $X$, we measure how much information the output conveys about the input by the mutual information:

$$I(X; Y) \equiv H(X) - H(X \mid Y) = H(Y) - H(Y \mid X).$$

For computational purposes, it is often handy to evaluate $I(X; Y)$ as $H(Y) - H(Y \mid X)$ rather than $H(X) - H(X \mid Y)$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Mutual Information for the BSC)</span></p>

For the BSC with $f = 0.15$ and $\mathcal{P}_X : \lbrace p_0 = 0.9, p_1 = 0.1 \rbrace$:

$H(Y \mid X)$ is the same for each value of $x$: $H(Y \mid x{=}0) = H_2(0.15)$ and $H(Y \mid x{=}1) = H_2(0.15)$. The marginal probabilities are $P(y{=}0) = 0.78$ and $P(y{=}1) = 0.22$. So:

$$I(X; Y) = H(Y) - H(Y \mid X) = H_2(0.22) - H_2(0.15) = 0.76 - 0.61 = 0.15 \text{ bits}.$$

This may be contrasted with the entropy of the source $H(X) = H_2(0.1) = 0.47$ bits.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Mutual Information for the Z Channel)</span></p>

For the Z channel with $f = 0.15$ and $\mathcal{P}_X$ as above, $P(y{=}1) = p_1(1-f) = 0.085$:

$$I(X; Y) = H(Y) - H(Y \mid X) = H_2(0.085) - [0.9 \cdot H_2(0) + 0.1 \cdot H_2(0.15)] = 0.42 - 0.061 = 0.36 \text{ bits}.$$

The mutual information for the Z channel is bigger than for the BSC with the same $f$. The Z channel is a more reliable channel.

</div>

### 9.6 Channel Capacity

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Channel Capacity)</span></p>

The **capacity** of a channel $Q$ is defined as the maximum mutual information over all possible input distributions:

$$C(Q) = \max_{\mathcal{P}_X} I(X; Y).$$

The distribution $\mathcal{P}_X$ that achieves the maximum is called the **optimal input distribution**, denoted $\mathcal{P}_X^*$. There may be multiple optimal input distributions achieving the same value of $I(X; Y)$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Capacity of the BSC)</span></p>

For the BSC with noise level $f$, by symmetry the optimal input distribution is $\lbrace 0.5, 0.5 \rbrace$, giving:

$$C(Q_{\text{BSC}}) = H_2(0.5) - H_2(f) = 1 - H_2(f).$$

For $f = 0.15$: $C = 1 - 0.61 = 0.39$ bits.

To verify without invoking symmetry, write $I(X; Y) = H_2((1-f)p_1 + (1-p_1)f) - H_2(f)$. The only $p$-dependence is in the first term $H_2((1-f)p_1 + (1-p_1)f)$, which is maximized by setting the argument to $0.5$, i.e., $p_0 = 1/2$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Capacity of the Noisy Typewriter)</span></p>

For the noisy typewriter, the optimal input distribution is uniform over $x$, giving $C = \log_2 9$ bits. This is because we can find a **non-confusable subset** of 9 inputs (every third letter: B, E, H, ..., Z) such that any output can be uniquely decoded.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Capacity of the Z Channel)</span></p>

For the Z channel with noise level $f$, the mutual information is:

$$I(X; Y) = H_2(p_1(1-f)) - p_1 H_2(f).$$

This is a non-trivial function of $p_1$. For $f = 0.15$, it is maximized by $p_1^* \approx 0.445$, giving $C(Q_Z) \approx 0.685$. The optimal input distribution is *not* $\lbrace 0.5, 0.5 \rbrace$ -- we communicate slightly more information by using input symbol $0$ more frequently than $1$.

In general, the optimal input distribution for the Z channel is:

$$p_1^* = \frac{1/(1-f)}{1 + 2^{H_2(f)/(1-f)}}.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Capacity of the Binary Erasure Channel)</span></p>

For the BEC with erasure probability $f$, the optimal input distribution is $\lbrace 0.5, 0.5 \rbrace$ by symmetry. The conditional entropy $H(X \mid Y) = \sum_y P(y) H(X \mid y)$; $x$ is uncertain only when $y = \text{?}$, which occurs with probability $f$, so $H(X \mid Y) = f H_2(0.5) = f$. Thus:

$$C_{\text{BEC}} = H_2(0.5) - f H_2(0.5) = 1 - f.$$

The capacity is precisely $1 - f$, the fraction of the time the channel is reliable.

</div>

### 9.6.1 The Noisy-Channel Coding Theorem

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Block Code and Rate)</span></p>

An $(N, K)$ **block code** for a channel $Q$ is a list of $S = 2^K$ codewords:

$$\lbrace \mathbf{x}^{(1)}, \mathbf{x}^{(2)}, \ldots, \mathbf{x}^{(2^K)} \rbrace, \quad \mathbf{x}^{(s)} \in \mathcal{A}_X^N,$$

each of length $N$. Using this code we can encode a signal $s \in \lbrace 1, 2, \ldots, 2^K \rbrace$ as $\mathbf{x}^{(s)}$. The **rate** of the code is $R = K / N$ bits per channel use.

A **decoder** is a mapping from the set of length-$N$ strings of channel outputs, $\mathcal{A}_Y^N$, to a codeword label $\hat{s} \in \lbrace 0, 1, 2, \ldots, 2^K \rbrace$ (the extra symbol $\hat{s} = 0$ can indicate a 'failure').

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Probability of Error)</span></p>

The **probability of block error** for a code and decoder, given a probability distribution over the encoded signal $P(s_{\text{in}})$, is:

$$p_{\text{B}} = \sum_{s_{\text{in}}} P(s_{\text{in}}) P(s_{\text{out}} \ne s_{\text{in}} \mid s_{\text{in}}).$$

The **maximal probability of block error** is:

$$p_{\text{BM}} = \max_{s_{\text{in}}} P(s_{\text{out}} \ne s_{\text{in}} \mid s_{\text{in}}).$$

The **probability of bit error** $p_{\text{b}}$ is the average probability that a bit of $\mathbf{s}_{\text{out}}$ is not equal to the corresponding bit of $\mathbf{s}_{\text{in}}$ (averaging over all $K$ bits).

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Optimal Decoder)</span></p>

The **optimal decoder** for a channel code is the one that minimizes the probability of block error. It decodes an output $\mathbf{y}$ as the input $s$ that has maximum posterior probability $P(s \mid \mathbf{y})$:

$$\hat{s}_{\text{optimal}} = \operatorname{argmax}_s P(s \mid \mathbf{y}) = \operatorname{argmax}_s \frac{P(\mathbf{y} \mid s) P(s)}{\sum_{s'} P(\mathbf{y} \mid s') P(s')}.$$

A uniform prior distribution on $s$ is usually assumed, in which case the optimal decoder is also the **maximum likelihood decoder**: the decoder that maps $\mathbf{y}$ to the input $s$ that has maximum likelihood $P(\mathbf{y} \mid s)$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Shannon's Noisy-Channel Coding Theorem -- Part One)</span></p>

Associated with each discrete memoryless channel, there is a non-negative number $C$ (the channel capacity) with the following property: for any $\epsilon > 0$ and $R < C$, for large enough $N$, there exists a block code of length $N$ and rate $\ge R$ and a decoding algorithm, such that the maximal probability of block error is $< \epsilon$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Confirmation for the Noisy Typewriter)</span></p>

For the noisy typewriter, we can confirm the theorem using a block code of length $N = 1$: use only every third letter (B, E, H, ..., Z). These 9 letters form a non-confusable subset -- any output can be uniquely decoded. The rate is $\log_2 9$ bits, which equals the capacity $C$.

</div>

### 9.7 Intuitive Preview of Proof

To prove the noisy-channel coding theorem for any channel, we consider the **extended channel** corresponding to $N$ uses of the channel. The extended channel has $|\mathcal{A}_X|^N$ possible inputs $\mathbf{x}$ and $|\mathcal{A}_Y|^N$ possible outputs. The key intuition is that if $N$ is large, *an extended channel looks a lot like the noisy typewriter*.

Any particular input $\mathbf{x}$ is very likely to produce an output in a small subspace of the output alphabet -- the **typical output set**, given that input. So we can find a non-confusable subset of the inputs that produce essentially disjoint output sequences.

For a given $N$, imagine making an input sequence $\mathbf{x}$ by drawing it from an ensemble $X^N$:

- The total number of typical output sequences $\mathbf{y}$ is $\approx 2^{NH(Y)}$, all having similar probability.
- For any particular typical input sequence $\mathbf{x}$, there are about $2^{NH(Y \mid X)}$ probable sequences.

We restrict to a subset of typical inputs $\mathbf{x}$ such that the corresponding typical output sets do not overlap. The number of non-confusable inputs is bounded by dividing the size of the typical $\mathbf{y}$ set by the size of each typical-$\mathbf{y}$-given-typical-$\mathbf{x}$ set:

$$\text{number of non-confusable inputs} \le \frac{2^{NH(Y)}}{2^{NH(Y \mid X)}} = 2^{NI(X;Y)}.$$

The maximum value of this bound is achieved when $X$ is the ensemble that maximizes $I(X; Y)$, giving $\le 2^{NC}$ non-confusable inputs. Thus asymptotically up to $C$ bits per cycle, and no more, can be communicated with vanishing error probability.

### 9.9 Pattern Recognition as a Noisy Channel

Many pattern recognition problems can be framed in terms of communication channels. Consider recognizing handwritten digits: the author wishes to communicate a message from $\mathcal{A}_X = \lbrace 0, 1, 2, \ldots, 9 \rbrace$; this is the input to the channel. The output is a pattern of ink on paper.

One strategy for pattern recognition is **full probabilistic modelling** (or **generative modelling**): create a model for $P(y \mid x)$ for each input value $x$, then use Bayes' theorem to infer $x$ given $y$:

$$P(x \mid y) = \frac{P(y \mid x) P(x)}{\sum_{x'} P(y \mid x') P(x')}.$$

In addition to the channel model $P(y \mid x)$, one uses a prior probability distribution $P(x)$ -- for example, a language model that specifies the probability of the next character given the context.

## Chapter 10 -- The Noisy-Channel Coding Theorem

Before reading Chapter 10, Chapters 4 and 9 should be familiar. The cast of characters: $Q$ is the noisy channel, $C$ is the channel capacity, $X^N$ is the ensemble used to create a random code, $\mathcal{C}$ is a random code, $N$ is the codeword length, $\mathbf{x}^{(s)}$ is the $s$th codeword, $S = 2^K$ is the total number of codewords, $K = \log_2 S$ is the number of information bits, $R = K/N$ is the code rate, and $\hat{s}$ is the decoder's guess.

### 10.1 The Theorem

The noisy-channel coding theorem has three parts: two positive and one negative.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Noisy-Channel Coding Theorem)</span></p>

1. For every discrete memoryless channel, the channel capacity

   $$C = \max_{P_X} I(X; Y)$$

   has the following property: for any $\epsilon > 0$ and $R < C$, for large enough $N$, there exists a code of length $N$ and rate $\ge R$ and a decoding algorithm, such that the maximal probability of block error is $< \epsilon$.

2. If a probability of bit error $p_\text{b}$ is acceptable, rates up to $R(p_\text{b})$ are achievable, where

   $$R(p_\text{b}) = \frac{C}{1 - H_2(p_\text{b})}.$$

3. For any $p_\text{b}$, rates greater than $R(p_\text{b})$ are **not** achievable.

</div>

### 10.2 Jointly-Typical Sequences

The proof formalizes the intuitive preview from Chapter 9. Codewords $\mathbf{x}^{(s)}$ come from an ensemble $X^N$. We consider the random selection of one codeword and a corresponding channel output $\mathbf{y}$, defining a joint ensemble $(XY)^N$. A **typical-set decoder** decodes a received signal $\mathbf{y}$ as $s$ if $\mathbf{x}^{(s)}$ and $\mathbf{y}$ are *jointly typical*.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Joint Typicality)</span></p>

A pair of sequences $\mathbf{x}, \mathbf{y}$ of length $N$ are **jointly typical** (to tolerance $\beta$) with respect to the distribution $P(x, y)$ if all three of the following hold:

- $\mathbf{x}$ is typical of $P(\mathbf{x})$: $\left\lvert \frac{1}{N} \log \frac{1}{P(\mathbf{x})} - H(X) \right\rvert < \beta$,
- $\mathbf{y}$ is typical of $P(\mathbf{y})$: $\left\lvert \frac{1}{N} \log \frac{1}{P(\mathbf{y})} - H(Y) \right\rvert < \beta$,
- $\mathbf{x}, \mathbf{y}$ are jointly typical of $P(\mathbf{x}, \mathbf{y})$: $\left\lvert \frac{1}{N} \log \frac{1}{P(\mathbf{x}, \mathbf{y})} - H(X, Y) \right\rvert < \beta$.

The **jointly-typical set** $J_{N\beta}$ is the set of all jointly-typical sequence pairs of length $N$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Jointly-Typical Pair)</span></p>

For $N = 100$, $P(x)$ with $(p_0, p_1) = (0.9, 0.1)$, and $P(y \mid x)$ corresponding to a BSC with noise level 0.2: $\mathbf{x}$ has 10 ones (typical since $P(x=1) = 0.1$), $\mathbf{y}$ has 26 ones (typical since $P(y=1) = 0.26$), and $\mathbf{x}$ and $\mathbf{y}$ differ in 20 bits (the typical number of flips for this channel).

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Joint Typicality Theorem)</span></p>

Let $\mathbf{x}, \mathbf{y}$ be drawn from the ensemble $(XY)^N$ defined by $P(\mathbf{x}, \mathbf{y}) = \prod_{n=1}^{N} P(x_n, y_n)$. Then:

1. The probability that $\mathbf{x}, \mathbf{y}$ are jointly typical (to tolerance $\beta$) tends to 1 as $N \to \infty$.

2. The number of jointly-typical sequences $\lvert J_{N\beta} \rvert$ satisfies:

   $$\lvert J_{N\beta} \rvert \le 2^{N(H(X,Y) + \beta)}.$$

3. If $\mathbf{x}' \sim X^N$ and $\mathbf{y}' \sim Y^N$ are **independent** samples with the same marginal distributions as $P(\mathbf{x}, \mathbf{y})$, the probability that $(\mathbf{x}', \mathbf{y}')$ lands in the jointly-typical set is approximately $2^{-NI(X;Y)}$. Precisely:

   $$P((\mathbf{x}', \mathbf{y}') \in J_{N\beta}) \le 2^{-N(I(X;Y) - 3\beta)}.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof Sketch of the Joint Typicality Theorem)</span></p>

Parts 1 and 2 follow from the law of large numbers applied to the source coding theorem, with the pair $x, y$ playing the role of $x$ and $P(x, y)$ replacing $P(x)$.

For part 3:

$$P((\mathbf{x}', \mathbf{y}') \in J_{N\beta}) = \sum_{(\mathbf{x}, \mathbf{y}) \in J_{N\beta}} P(\mathbf{x}) P(\mathbf{y}) \le \lvert J_{N\beta} \rvert \cdot 2^{-N(H(X) - \beta)} \cdot 2^{-N(H(Y) - \beta)},$$

$$\le 2^{N(H(X,Y) + \beta) - N(H(X) + H(Y) - 2\beta)} = 2^{-N(I(X;Y) - 3\beta)}.$$

Intuitively, two independent typical vectors are jointly typical with probability $\simeq 2^{-NI(X;Y)}$ because the total number of independent typical pairs is $2^{NH(X)} \cdot 2^{NH(Y)}$, while the number of jointly-typical pairs is roughly $2^{NH(X,Y)}$, giving a ratio of $2^{NH(X,Y)} / 2^{N(H(X) + H(Y))} = 2^{-NI(X;Y)}$.

</div>

### 10.3 Proof of the Noisy-Channel Coding Theorem

The proof uses a technique analogous to proving that a baby weighing less than 10 kg exists in a class of 100: weigh all the babies at once and check the average. If the average is below 10 kg, at least one must weigh less than 10 kg.

Similarly, instead of constructing a specific good code and evaluating its error probability, Shannon calculated the **average probability of block error over all random codes** and showed it to be small. There must then exist individual codes with small error probability.

#### Random Coding and Typical-Set Decoding

Consider the following encoding--decoding system with rate $R'$:

1. Fix $P(x)$ and generate $S = 2^{NR'}$ codewords of an $(N, K)$ code $\mathcal{C}$ at random: $P(\mathbf{x}) = \prod_{n=1}^{N} P(x_n)$.
2. The code is known to both sender and receiver.
3. A message $s$ is chosen from $\lbrace 1, 2, \ldots, 2^{NR'} \rbrace$, and $\mathbf{x}^{(s)}$ is transmitted. The received signal is $\mathbf{y}$, with $P(\mathbf{y} \mid \mathbf{x}^{(s)}) = \prod_{n=1}^{N} P(y_n \mid x_n^{(s)})$.
4. The signal is decoded by **typical-set decoding**.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Typical-Set Decoding)</span></p>

Decode $\mathbf{y}$ as $\hat{s}$ if $(\mathbf{x}^{(\hat{s})}, \mathbf{y})$ are jointly typical **and** there is no other $s'$ such that $(\mathbf{x}^{(s')}, \mathbf{y})$ are jointly typical; **otherwise** declare a failure ($\hat{s} = 0$).

</div>

#### Three Levels of Error Probability

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Error Probability Hierarchy)</span></p>

Three error quantities are distinguished:

- **Block error probability** for a particular code $\mathcal{C}$: $p_\text{B}(\mathcal{C}) \equiv P(\hat{s} \ne s \mid \mathcal{C})$.
- **Average block error probability** over all random codes: $\langle p_\text{B} \rangle \equiv \sum_{\mathcal{C}} P(\hat{s} \ne s \mid \mathcal{C}) P(\mathcal{C})$.
- **Maximal block error probability**: $p_\text{BM}(\mathcal{C}) \equiv \max_s P(\hat{s} \ne s \mid s, \mathcal{C})$.

The strategy is: show $\langle p_\text{B} \rangle$ is small $\Rightarrow$ there exists a code $\mathcal{C}$ with $p_\text{B}(\mathcal{C})$ small $\Rightarrow$ modify it so $p_\text{BM}(\mathcal{C})$ is also small.

</div>

#### Bounding the Average Error Probability

There are two sources of error under typical-set decoding (assume WLOG that $s = 1$ was sent):

- **(a)** The output $\mathbf{y}$ is not jointly typical with the transmitted codeword $\mathbf{x}^{(1)}$. By the joint typicality theorem (part 1), this probability $\delta \to 0$ as $N \to \infty$.
- **(b)** Some other codeword $\mathbf{x}^{(s')}$ ($s' \ne 1$) is jointly typical with $\mathbf{y}$. By the joint typicality theorem (part 3), for a given $s' \ne 1$, this probability is $\le 2^{-N(I(X;Y) - 3\beta)}$.

By the **union bound**, with $(2^{NR'} - 1)$ rival codewords:

$$\langle p_\text{B} \rangle \le \delta + 2^{-N(I(X;Y) - R' - 3\beta)}.$$

This can be made $< 2\delta$ by increasing $N$ if:

$$R' < I(X; Y) - 3\beta.$$

#### Three Final Modifications

1. Choose $P(x)$ to be the **optimal input distribution**, so $I(X; Y) = C$ and the condition becomes $R' < C - 3\beta$.
2. Since $\langle p_\text{B} \rangle < 2\delta$, there must exist a code $\mathcal{C}$ with $p_\text{B}(\mathcal{C}) < 2\delta$.
3. **Expurgation**: throw away the worst 50% of codewords (those most likely to cause errors). The remaining codewords all have conditional error probability $< 4\delta$. The new code has $2^{NR' - 1}$ codewords, reducing the rate from $R'$ to $R' - 1/N$ (negligible for large $N$), and achieving $p_\text{BM} < 4\delta$.

In conclusion, we can construct a code of rate $R' - 1/N$ where $R' < C - 3\beta$, with maximal error probability $< 4\delta$. Setting $R' = (R + C)/2$, $\delta = \epsilon/4$, $\beta < (C - R')/3$, and $N$ sufficiently large completes the proof of part 1 of the theorem. $\square$

### 10.4 Communication (with Errors) above Capacity

Part 1 of the theorem establishes achievability for the portion of the $(R, p_\text{b})$ plane where $R < C$ and $p_\text{b} = 0$. Part 2 extends the right-hand boundary of the achievable region to non-zero error probabilities using **rate-distortion theory**.

The idea: since we can turn any noisy channel into a perfect channel with rate up to $C$, it suffices to consider communication with errors over a **noiseless** channel.

For a noiseless binary channel at rate $R > 1$ (forced to communicate faster than capacity), the best strategy is to communicate a fraction $1/R$ of the source bits and spread the corruption evenly. This achieves:

$$p_\text{b} = H_2^{-1}(1 - 1/R).$$

This is achieved by reusing the $(N, K)$ code for a BSC as a **lossy compressor**: take the signal, chop it into blocks of length $N$, pass each block through the *decoder* to get a shorter $K$-bit message, then communicate the $K$ bits over the noiseless channel. To reconstruct, pass the $K$ bits through the *encoder*. The bit error probability is $p_\text{b} = q$ (the BSC transition probability), and the rate of this lossy compressor is $R = N/K = 1/R' = 1/(1 - H_2(p_\text{b}))$.

Attaching this lossy compressor to the capacity-$C$ error-free communicator proves achievability up to the curve:

$$R = \frac{C}{1 - H_2(p_\text{b})}. \quad \square$$

### 10.5 The Non-Achievable Region (Part 3)

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Converse of the Noisy-Channel Coding Theorem)</span></p>

The source, encoder, noisy channel, and decoder define a Markov chain $s \to \mathbf{x} \to \mathbf{y} \to \hat{s}$:

$$P(s, \mathbf{x}, \mathbf{y}, \hat{s}) = P(s) P(\mathbf{x} \mid s) P(\mathbf{y} \mid \mathbf{x}) P(\hat{s} \mid \mathbf{y}).$$

By the data processing inequality, $I(s; \hat{s}) \le I(\mathbf{x}; \mathbf{y}) \le NC$.

If a system achieves rate $R$ and bit error probability $p_\text{b}$, then $I(s; \hat{s}) \ge NR(1 - H_2(p_\text{b}))$. But $I(s; \hat{s}) > NC$ leads to a contradiction, so:

$$R > \frac{C}{1 - H_2(p_\text{b})}$$

is **not achievable**. $\square$

</div>

### 10.6 Computing Capacity

To compute the capacity of a given discrete memoryless channel, we need to find the optimal input distribution $P(x)$ that maximizes $I(X; Y)$.

Since $I(X; Y)$ is a **concave** $\frown$ function of the input probability vector $\mathbf{p}$, any stationary distribution must be a global maximum. The stationarity condition is:

$$\frac{\partial I(X; Y)}{\partial p_i} = \lambda \quad \text{for all } i \text{ with } p_i > 0,$$

$$\frac{\partial I(X; Y)}{\partial p_i} \le \lambda \quad \text{for all } i \text{ with } p_i = 0,$$

where $\lambda$ is a Lagrange multiplier associated with the constraint $\sum_i p_i = 1$, and $C = \lambda + \log_2 e$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Ternary Confusion Channel)</span></p>

Consider a channel with $\mathcal{A}_X = \lbrace 0, ?, 1 \rbrace$ and $\mathcal{A}_Y = \lbrace 0, 1 \rbrace$. The transition probabilities are:

$$P(y=0 \mid x=0) = 1, \quad P(y=0 \mid x=?) = 1/2, \quad P(y=0 \mid x=1) = 0,$$

$$P(y=1 \mid x=0) = 0, \quad P(y=1 \mid x=?) = 1/2, \quad P(y=1 \mid x=1) = 1.$$

Whenever the input $?$ is used, the output is random; the other inputs are reliable. The maximum information rate of 1 bit is achieved by making no use of the input $?$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Helpful Results for Finding Optimal Input Distributions)</span></p>

- All outputs must be used.
- $I(X; Y)$ is a convex $\smile$ function of the channel parameters $Q(y \mid x)$.
- There may be several optimal input distributions, but they all produce the same output distribution.
- No output $y$ is unused by an optimal input distribution (unless it is unreachable: $Q(y \mid x) = 0$ for all $x$).

</div>

#### Symmetric Channels

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Symmetric Channel)</span></p>

A discrete memoryless channel is a **symmetric channel** if the set of outputs can be partitioned into subsets in such a way that for each subset the matrix of transition probabilities has the property that each row (if more than 1) is a permutation of each other row and each column is a permutation of each other column.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Symmetric Channel)</span></p>

The channel with $\mathcal{A}_X = \lbrace 0, 1 \rbrace$ and $\mathcal{A}_Y = \lbrace 0, ?, 1 \rbrace$ defined by:

$$P(y\!=\!0 \mid x\!=\!0) = 0.7, \quad P(y\!=\!? \mid x\!=\!0) = 0.2, \quad P(y\!=\!1 \mid x\!=\!0) = 0.1,$$

$$P(y\!=\!0 \mid x\!=\!1) = 0.1, \quad P(y\!=\!? \mid x\!=\!1) = 0.2, \quad P(y\!=\!1 \mid x\!=\!1) = 0.7,$$

is symmetric because the outputs can be partitioned into $\lbrace 0, 1 \rbrace$ and $\lbrace ? \rbrace$. Within each subset the matrix rows are permutations of each other and columns are permutations of each other.

</div>

Symmetry is useful because communication at capacity can be achieved over symmetric channels by **linear codes**. For a symmetric channel with any number of inputs, the **uniform distribution** over the inputs is an optimal input distribution.

### 10.7 Other Coding Theorems

The noisy-channel coding theorem proved above is general but not very specific: it only guarantees the existence of good codes for *sufficiently large* blocklength $N$, without saying how large $N$ must be.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Noisy-Channel Coding Theorem -- Explicit $N$-Dependence)</span></p>

For a discrete memoryless channel, a blocklength $N$, and a rate $R$, there exist block codes of length $N$ whose average probability of error satisfies:

$$p_\text{B} \le \exp[-N E_\text{r}(R)],$$

where $E_\text{r}(R)$ is the **random-coding exponent** (also known as the reliability function) of the channel -- a convex $\smile$, decreasing, positive function of $R$ for $0 \le R < C$.

By an expurgation argument, there also exist codes for which the *maximal* probability of error $p_\text{BM}$ is exponentially small in $N$.

</div>

The random-coding exponent $E_\text{r}(R) \to 0$ as $R \to C$. Computing $E_\text{r}(R)$ for interesting channels is a challenging task; even for the BSC there is no simple closed-form expression.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Lower Bound on Error Probability)</span></p>

For any code with blocklength $N$ on a discrete memoryless channel, assuming all source messages are used with equal probability:

$$p_\text{B} \gtrsim \exp[-N E_\text{sp}(R)],$$

where $E_\text{sp}(R)$, the **sphere-packing exponent**, is a convex $\smile$, decreasing, positive function of $R$ for $0 \le R < C$. Thus the error probability is bounded both above and below by exponentials in $N$.

</div>

### 10.8 Coding Theorems and Coding Practice

From a practical perspective, the customer who wants an error-correcting code need not know the exact functions $E_\text{r}(R)$ and $E_\text{sp}(R)$. The sensible approach (Berlekamp, 1980) is to design encoding--decoding systems and plot their performance on a variety of idealized channels as a function of the channel's noise level, allowing the customer to choose a system without knowing the exact channel.

A key practical caveat: while the theorem guarantees the *existence* of good random codes, the cost of implementing the encoder and decoder for a random code with large $N$ would be exponentially large in $N$. Finding structured codes that approach capacity with tractable encoding and decoding is the central challenge of coding theory.

## Chapter 11 -- Error-Correcting Codes and Real Channels

This chapter addresses two questions: (1) what can Shannon tell us about continuous (real-valued) channels? and (2) how are practical error-correcting codes made, and how do they compare to Shannon's theoretical limits?

### Prerequisites: The Gaussian Distribution

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Gaussian Distribution)</span></p>

A random variable $y$ with mean $\mu$ and variance $\sigma^2$ has a **Gaussian distribution**, written $y \sim \text{Normal}(\mu, \sigma^2)$, with density:

$$P(y \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left[-(y - \mu)^2 / 2\sigma^2\right].$$

The **inverse-variance** $\tau \equiv 1/\sigma^2$ is sometimes called the **precision**.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Multivariate Gaussian Distribution)</span></p>

If $\mathbf{y} = (y_1, y_2, \ldots, y_N)$ has a multivariate Gaussian distribution, then:

$$P(\mathbf{y} \mid \mathbf{x}, \mathbf{A}) = \frac{1}{Z(\mathbf{A})} \exp\!\left(-\frac{1}{2}(\mathbf{y} - \mathbf{x})^\mathsf{T} \mathbf{A} (\mathbf{y} - \mathbf{x})\right),$$

where $\mathbf{x}$ is the mean, $\mathbf{A}$ is the inverse of the variance--covariance matrix, and $Z(\mathbf{A}) = (\det(\mathbf{A}/2\pi))^{-1/2}$.

The covariance $\Sigma_{ij} \equiv \mathcal{E}[(y_i - \bar{y}_i)(y_j - \bar{y}_j)] = A_{ij}^{-1}$, where $\mathbf{A}^{-1}$ is the inverse of $\mathbf{A}$. The marginal and conditional distributions of any subset of components are also Gaussian.

</div>

### 11.1 The Gaussian Channel

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Gaussian Channel)</span></p>

The **Gaussian channel** (also called the additive white Gaussian noise (AWGN) channel) has a real input $x$ and a real output $y$. The conditional distribution of $y$ given $x$ is:

$$P(y \mid x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left[-(y - x)^2 / 2\sigma^2\right].$$

This channel has continuous input and output but is discrete in time.

</div>

#### Motivation from Continuous-Time Channels

A physical channel with continuous-time inputs and outputs transmits $x(t)$ and receives $y(t) = x(t) + n(t)$, where $n(t)$ is white Gaussian noise with spectral density $N_0$. The average power is constrained: $\int_0^T [x(t)]^2 \, \mathrm{d}t / T \le P$.

We can represent the signal using $N$ orthonormal basis functions $\phi_n(t)$:

$$x(t) = \sum_{n=1}^{N} x_n \phi_n(t),$$

where $\int_0^T \phi_n(t)\phi_m(t)\,\mathrm{d}t = \delta_{nm}$. The receiver computes:

$$y_n = \int_0^T \phi_n(t) y(t)\,\mathrm{d}t = x_n + n_n, \qquad n_n \sim \text{Normal}(0, N_0/2).$$

The power constraint becomes $\sum_n x_n^2 \le PT$, or equivalently $\overline{x_n^2} \le PT/N$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Bandwidth)</span></p>

The **bandwidth** of the continuous channel (in Hertz) is:

$$W = \frac{N^{\max}}{2T},$$

where $N^{\max}$ is the maximum number of orthonormal functions that can be produced in an interval of length $T$. By the Nyquist sampling theorem, $N^{\max} = 2WT$.

Thus a real continuous channel with bandwidth $W$, power $P$, and noise spectral density $N_0$ is equivalent to $N/T = 2W$ uses per second of a Gaussian channel with $\sigma^2 = N_0/2$ and signal power constraint $\overline{x_n^2} \le P/2W$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($E_\text{b}/N_0$: Rate-Compensated Signal-to-Noise Ratio)</span></p>

When comparing encoding systems that transmit binary source bits at rate $R$ bits per channel use with different signal powers $\overline{x_n^2}$, the standard measure is:

$$E_\text{b}/N_0 = \frac{\overline{x_n^2}}{2\sigma^2 R}.$$

This is the ratio of the power per source bit $E_\text{b} = \overline{x_n^2}/R$ to the noise spectral density $N_0$, and is usually reported in decibels ($10 \log_{10} E_\text{b}/N_0$).

</div>

### 11.2 Inferring the Input to a Real Channel

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Optimal Detection of Pulses)</span></p>

Consider differentiating between two pulse types $\mathbf{x}_0$ and $\mathbf{x}_1$ transmitted over a Gaussian channel with noise covariance inverse $\mathbf{A}$. The posterior probability ratio is:

$$\frac{P(s=1 \mid \mathbf{y})}{P(s=0 \mid \mathbf{y})} = \exp\!\left(\mathbf{y}^\mathsf{T}\mathbf{A}(\mathbf{x}_1 - \mathbf{x}_0) + \theta\right),$$

where $\theta = -\frac{1}{2}\mathbf{x}_1^\mathsf{T}\mathbf{A}\mathbf{x}_1 + \frac{1}{2}\mathbf{x}_0^\mathsf{T}\mathbf{A}\mathbf{x}_0 + \ln\frac{P(s=1)}{P(s=0)}$.

The optimal decision uses a **discriminant function** $a(\mathbf{y}) = \mathbf{w}^\mathsf{T}\mathbf{y} + \theta$, where $\mathbf{w} \equiv \mathbf{A}(\mathbf{x}_1 - \mathbf{x}_0)$: guess $s = 1$ if $a(\mathbf{y}) > 0$, guess $s = 0$ if $a(\mathbf{y}) < 0$. The decision boundary is a **linear** function of the received vector.

</div>

### 11.3 Capacity of the Gaussian Channel

#### The Problem of Infinite Inputs

Without constraints, one could transmit infinite information in a single use of a Gaussian channel by encoding arbitrarily many digits in one real number. To prevent this, we introduce a **cost function** $v(x) = x^2$ and constrain the average power: $\overline{x^2} = v$.

#### The Problem of Infinite Precision

Defining entropy for continuous variables by replacing sums with integrals is not well-defined: the entropy of a discretized continuous distribution diverges as the granularity decreases. However, the **mutual information** is well-behaved because the log argument is a ratio of two densities over the same space:

$$I(X; Y) = \int \mathrm{d}x\,\mathrm{d}y\; P(x, y) \log \frac{P(x, y)}{P(x)P(y)} = \int \mathrm{d}x\,\mathrm{d}y\; P(x) P(y \mid x) \log \frac{P(y \mid x)}{P(y)}.$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Capacity of the Gaussian Channel)</span></p>

The probability distribution $P(x)$ that maximizes the mutual information subject to the constraint $\overline{x^2} = v$ is a Gaussian distribution with mean zero and variance $v$. The resulting capacity is:

$$C = \frac{1}{2} \log\!\left(1 + \frac{v}{\sigma^2}\right) \quad \text{bits per channel use},$$

where $v/\sigma^2$ is the **signal-to-noise ratio**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proof Sketch)</span></p>

Given a Gaussian input $P(x) = \text{Normal}(x; 0, v)$ and $P(y \mid x) = \text{Normal}(y; x, \sigma^2)$, the output distribution is $P(y) = \text{Normal}(y; 0, v + \sigma^2)$. The mutual information is:

$$I(X; Y) = \frac{1}{2}\log\frac{1}{\sigma^2} - \frac{1}{2}\log\frac{1}{v + \sigma^2} = \frac{1}{2}\log\!\left(1 + \frac{v}{\sigma^2}\right).$$

That the Gaussian input is optimal can be shown via Lagrange multipliers: the stationarity condition forces $\ln[P(y)\sigma]$ to be quadratic in $y$, which means $P(y)$ must be Gaussian.

</div>

#### Inferences Given a Gaussian Input Distribution

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Bayesian Inference with Gaussians)</span></p>

If $P(x) = \text{Normal}(x; 0, v)$ and $P(y \mid x) = \text{Normal}(y; x, \sigma^2)$, then the posterior distribution of $x$ given $y$ is:

$$P(x \mid y) = \text{Normal}\!\left(x;\; \frac{v}{v + \sigma^2}\,y,\; \left(\frac{1}{v} + \frac{1}{\sigma^2}\right)^{\!-1}\right).$$

The posterior mean $\frac{v}{v+\sigma^2}\,y$ is a weighted combination of the data ($x = y$) and the prior ($x = 0$). The weights $1/\sigma^2$ and $1/v$ are the **precisions** of the likelihood and the prior respectively. The precision of the posterior is the **sum** of these two precisions -- when independent Gaussian sources contribute information, precisions add.

</div>

#### Geometrical View: Sphere Packing

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Sphere-Packing Argument for the Gaussian Channel)</span></p>

Consider a sequence $\mathbf{x} = (x_1, \ldots, x_N)$ of inputs and the corresponding output $\mathbf{y}$. For large $N$:

- The output $\mathbf{y}$ is likely to lie on the surface of a sphere of radius $\sqrt{N\sigma^2}$ centred on $\mathbf{x}$.
- If $\mathbf{x}$ is generated subject to $\overline{x^2} = v$, then $\mathbf{x}$ lies close to a sphere of radius $\sqrt{Nv}$ centred on the origin.
- The received signal $\mathbf{y}$ lies on the surface of a sphere of radius $\sqrt{N(v + \sigma^2)}$ centred on the origin.

The maximum number of non-confusable inputs is bounded by dividing the volume of the large sphere by the volume of each noise sphere:

$$S \le \left(\frac{\sqrt{N(v + \sigma^2)}}{\sqrt{N\sigma^2}}\right)^{\!N},$$

giving the capacity bound $C = \frac{1}{N}\log S \le \frac{1}{2}\log\!\left(1 + \frac{v}{\sigma^2}\right)$.

</div>

#### Back to the Continuous Channel

Substituting the Gaussian channel capacity into the continuous-channel setting ($2W$ uses per second, $\sigma^2 = N_0/2$, $\overline{x_n^2} \le P/2W$):

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Shannon--Hartley Theorem)</span></p>

The capacity of a continuous channel with bandwidth $W$, power $P$, and noise spectral density $N_0$ is:

$$C = W \log\!\left(1 + \frac{P}{N_0 W}\right) \quad \text{bits per second}.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Bandwidth--Power Tradeoff)</span></p>

Introducing $W_0 = P/N_0$ (the bandwidth for which the signal-to-noise ratio is 1), the capacity normalized by $W_0$ is $C/W_0 = (W/W_0)\log(1 + W_0/W)$. As $W \to \infty$, this approaches $W_0 \log e$. It is dramatically better (in terms of capacity for fixed power) to transmit at a low signal-to-noise ratio over a large bandwidth than at high signal-to-noise ratio in a narrow bandwidth. This motivates wideband communication methods such as spread-spectrum.

</div>

### 11.4 Capabilities of Practical Error-Correcting Codes

Nearly all codes are good (in the sense of Shannon's proof), but nearly all require exponential look-up tables. By a **practical** error-correcting code we mean one that can be encoded and decoded in time polynomial in the blocklength $N$ -- preferably linearly.

#### The Shannon Limit Is Not Achieved in Practice

Writing down an explicit practical encoder and decoder that are as good as promised by Shannon remains an unsolved problem. The gap between theory and practice motivates the following classification.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Code Quality Classification)</span></p>

- **Very good codes:** families of block codes achieving arbitrarily small error probability at any rate up to the channel capacity.
- **Good codes:** families achieving arbitrarily small error probability at non-zero rates up to some maximum that may be *less* than capacity.
- **Bad codes:** families that cannot achieve arbitrarily small error probability, or can only do so by decreasing the rate to zero. Repetition codes are an example.
- **Practical codes:** families that can be encoded and decoded in time and space polynomial in the blocklength.

</div>

#### Linear Codes

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Linear Block Code)</span></p>

A **linear** $(N, K)$ **block code** is one in which the codewords $\lbrace \mathbf{x}^{(s)} \rbrace$ make up a $K$-dimensional subspace of $\mathcal{A}_X^N$. The encoding operation can be represented by an $N \times K$ binary matrix $\mathbf{G}^\mathsf{T}$ such that the encoded signal is $\mathbf{t} = \mathbf{G}^\mathsf{T}\mathbf{s} \bmod 2$, where $\mathbf{s}$ is the $K$-bit source vector.

The codewords can equivalently be defined as the set of vectors satisfying $\mathbf{H}\mathbf{t} = \mathbf{0} \bmod 2$, where $\mathbf{H}$ is the **parity-check matrix** of the code.

</div>

Most established codes are generalizations of Hamming codes: Bose--Chaudhury--Hocquenhem codes, Reed--Muller codes, Reed--Solomon codes, and Goppa codes.

#### Convolutional Codes

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Convolutional Code)</span></p>

**Convolutional codes** are linear codes that do not divide the source stream into blocks but instead read and transmit bits continuously. The transmitted bits are a linear function of past source bits, generated by feeding each source bit into a linear-feedback shift-register of length $k$ and transmitting one or more linear functions of the state at each iteration. The resulting transmitted bit stream is the convolution of the source stream with a linear filter.

</div>

#### Concatenation and Interleaving

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Concatenated Codes)</span></p>

An encoder--channel--decoder system $\mathcal{C} \to Q \to \mathcal{D}$ defines a super-channel $Q'$ with smaller error probability. A second encoder $\mathcal{C}'$ and decoder $\mathcal{D}'$ can be designed for this super-channel. The combined code (outer code $\mathcal{C}'$ followed by inner code $\mathcal{C}$) is a **concatenated code**.

**Interleaving** is a technique used with concatenated codes: after encoding with the outer code $\mathcal{C}'$, bits within a block are reordered so that nearby bits are spread across different codewords of the inner code $\mathcal{C}$. This breaks up correlated error patterns from the inner decoder, allowing the outer code to treat errors as independent. A simple example is a **product code** (or rectangular code) where data is arranged in a $K_2 \times K_1$ block and encoded horizontally with one code and vertically with another.

</div>

#### Other Channel Models

**Burst-error channels** are channels where errors come in bursts. Reed--Solomon codes, which use Galois fields with large alphabets (e.g., $2^{16}$), provide natural protection against burst errors: even if 17 successive bits are corrupted, only 2 successive symbols in the Galois field representation are affected. Concatenation and interleaving provide further protection.

**Fading channels** are like Gaussian channels except that the received power varies with time, as in mobile phone communication where the signal is reflected off nearby objects.

### 11.5 The State of the Art

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Practical Code Families)</span></p>

- **Textbook convolutional codes**: the de facto standard for satellite communications; constraint length 7.
- **Concatenated convolutional codes**: a convolutional code as the inner code with a Reed--Solomon outer code (used in Voyager and deep space missions).
- **Turbo codes** (1993): two convolutional encoders fed the same source bits in different (randomly permuted) orders, with parity bits from both transmitted. Decoded by iterative message-passing (the **sum--product algorithm**) between the two constituent decoders.
- **Gallager's low-density parity-check (LDPC) codes** (1962, rediscovered 1995): the best block codes known for Gaussian channels. Like turbo codes, decoded by message-passing algorithms. Outstanding theoretical and practical properties.

</div>

### 11.6 Summary

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Summary of Practical Coding)</span></p>

- **Random codes** are good but require exponential resources to encode and decode.
- **Non-random codes** tend not to be as good as random codes. Even for simply-defined linear codes, the general decoding problem is NP-complete.
- **The best practical codes** (a) employ very large block sizes; (b) are based on semi-random code constructions; and (c) make use of probability-based decoding algorithms.

</div>

### 11.7 Nonlinear Codes

Not all practical codes are linear. Digital soundtracks encoded onto cinema film use a nonlinear $(8, 6)$ code consisting of 64 of the $\binom{8}{4}$ binary patterns of weight 4. This design ensures no codeword looks like all-1s or all-0s, making it easy to detect errors caused by dirt and scratches on the film.

### 11.8 Errors Other Than Noise

Another source of uncertainty for the receiver is uncertainty about the **timing** of the transmitted signal. In ordinary coding theory, the transmitter's time $t$ and the receiver's time $u$ are assumed to be perfectly synchronized. If $u$ is an imperfectly known function $u(t)$, then the capacity is reduced. The theory of channels with synchronization errors is incomplete -- not even the capacity is known in general.

## Part III -- Further Topics in Information Theory

## Chapter 12 -- Hash Codes: Codes for Efficient Information Retrieval

In Chapters 1--11, the focus was on two aspects of information theory and coding theory: **source coding** (compression) and **channel coding** (redundant encoding for error detection and correction). The prime criterion was the efficiency of the code in terms of channel resources. In this chapter the viewpoint shifts to *ease of information retrieval* as a primary goal. It turns out that the random codes which were theoretically useful in the study of channel coding are also useful for rapid information retrieval.

### 12.1 The Information-Retrieval Problem

The information-retrieval problem is formalized as follows. We are given a list of $S$ binary strings of length $N$, $\lbrace \mathbf{x}^{(1)}, \dots, \mathbf{x}^{(S)} \rbrace$, where $S$ is considerably smaller than the total number of possible strings $2^N$. The superscript $s$ in $\mathbf{x}^{(s)}$ is called the **record number** of the string.

The task is to construct the inverse of the mapping from $s$ to $\mathbf{x}^{(s)}$: given a string $\mathbf{x}$, return the value of $s$ such that $\mathbf{x} = \mathbf{x}^{(s)}$, or report that no such $s$ exists. The aim is to minimize both the amount of memory used and the time to compute this inverse mapping.

#### Some Standard Solutions

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Look-Up Table)</span></p>

A **look-up table** is a piece of memory of size $2^N \log_2 S$, with one entry for each of the $2^N$ possible strings. At each location $\mathbf{x}$ that corresponds to a stored string $\mathbf{x}^{(s)}$, the value $s$ is written; all other locations contain zero.

Retrieval is instant (one memory access), but for $N \simeq 200$ the memory required ($\sim 2^{200}$ entries) is astronomically infeasible.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Raw List)</span></p>

A **raw list** stores ordered pairs $(s, \mathbf{x}^{(s)})$, ordered by the value of $s$. The mapping from $\mathbf{x}$ to $s$ is achieved by searching through the list from the top and comparing the incoming string $\mathbf{x}$ with each record $\mathbf{x}^{(s)}$ until a match is found.

- **Memory:** about $SN$ bits.
- **Average search time:** about $S + N$ binary comparisons (since a comparison can be terminated at the first mismatch, on average about 2 binary comparisons are needed per incorrect string match, and the correct string is located on average halfway through the list).
- **Worst-case search time:** $SN$ binary comparisons.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Alphabetical List)</span></p>

An **alphabetical list** stores the strings $\lbrace \mathbf{x}^{(s)} \rbrace$ sorted into alphabetical order. By iterating a binary search (splitting-in-the-middle), we can identify the target string or establish its absence in $\lceil \log_2 S \rceil$ string comparisons.

- **Memory:** same as raw list ($SN$ bits).
- **Total binary comparisons:** at most $\lceil \log_2 S \rceil N$.
- **Insertion cost:** about $\lceil \log_2 S \rceil$ binary comparisons to find the correct location.

</div>

The task of information retrieval is essentially constructing a mapping $\mathbf{x} \to s$ from $N$ bits to $\log_2 S$ bits -- a pseudo-invertible mapping, since for any $\mathbf{x}$ that maps to a non-zero $s$, the customer database contains the pair $(s, \mathbf{x}^{(s)})$ that takes us back. This is analogous to source coding (mapping $N$ symbols to a label) and to channel coding (mapping $K$ bits to $N$ bits), except here we map from $N$ bits to $M$ bits where $M$ is *smaller* than $N$.

### 12.2 Hash Codes

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Hash Function)</span></p>

A **hash function** is a pseudo-random function $\mathbf{h}$ that maps the $N$-bit string $\mathbf{x}$ to an $M$-bit string $\mathbf{h}(\mathbf{x})$, where $M$ is smaller than $N$. The value $M$ is typically chosen such that the **table size** $T \simeq 2^M$ is a little bigger than $S$ (e.g., ten times bigger). The hash function should ideally be indistinguishable from a fixed random code and must be quick to compute.

</div>

Two simple examples of hash functions:

- **Division method:** the table size $T$ is a prime number (preferably not close to a power of 2). The hash value is the remainder when the integer $\mathbf{x}$ is divided by $T$.
- **Variable string addition method:** assumes $\mathbf{x}$ is a string of bytes and $T = 256$. The characters of $\mathbf{x}$ are added modulo 256. A defect is that it maps anagrams to the same hash. An improvement is the *variable string exclusive-or method*: the running total is put through a fixed pseudorandom permutation after each character is added.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Hash Table Encoding and Decoding)</span></p>

**Encoding.** A piece of memory called the **hash table** is created of size $2^M b$ memory units, where $b$ is the amount of memory needed to represent an integer between 0 and $S$. The table is initially set to zero. Each string $\mathbf{x}^{(s)}$ is put through the hash function, and at the location in the hash table corresponding to $\mathbf{h}^{(s)} = \mathbf{h}(\mathbf{x}^{(s)})$, the integer $s$ is written -- unless that entry is already occupied, in which case we have a **collision**.

**Decoding.** To retrieve the record number corresponding to a target vector $\mathbf{x}$:
1. Compute the hash $\mathbf{h}$ of $\mathbf{x}$ and look at the corresponding location in the hash table of size $2^M$.
2. If the entry is zero, then $\mathbf{x}$ is not in the database.
3. If there is a non-zero entry $s$, look up $\mathbf{x}^{(s)}$ in the original forward database and compare it bit by bit with $\mathbf{x}$. If it matches, report $s$ as the answer.

The total cost of a successful retrieval: one hash-function evaluation, one look-up in the table of size $2^M$, another look-up in a table of size $S$, and $N$ binary comparisons.

</div>

### 12.3 Collision Resolution

A **collision** occurs when two strings $\mathbf{x}^{(s)}$ and $\mathbf{x}^{(s')}$ happen to have the same hash code. Two standard methods for resolving collisions are:

#### Appending in Table

When encoding, if a collision occurs, continue down the hash table and write the value of $s$ into the next available location containing a zero (wrapping around from the bottom to the top if needed).

When decoding, if the $s$ contained in the table doesn't point to an $\mathbf{x}^{(s)}$ that matches the cue $\mathbf{x}$, continue down the hash table until we either find an $s$ whose $\mathbf{x}^{(s)}$ matches the cue (success), or encounter a zero (the cue is not in the database).

For this method, it is essential that the table be substantially bigger than $S$. If $2^M < S$, the encoding rule will become stuck.

#### Storing Elsewhere

A more robust and flexible method is to use **pointers** to additional pieces of memory (buckets) in which collided strings are stored. At each location $\mathbf{h}$ in the hash table, a pointer leads to a **bucket** -- a sorted list of all strings that have hash code $\mathbf{h}$. The decoder looks in the relevant bucket and checks the short list by a brief alphabetical search.

This method allows the hash table to be made quite small, since almost all strings are involved in collisions when the table is small, so all buckets contain a small number of strings.

### 12.4 Planning for Collisions: A Birthday Problem

The expected number of collisions when storing $S$ entries using a hash function whose output has $M$ bits (table size $T = 2^M$) is related to the birthday problem. The probability that one particular pair of entries collides under a random hash function is $1/T$. The number of pairs is $S(S-1)/2$. So the expected number of collisions between pairs is:

$$\frac{S(S-1)}{2T}.$$

For the expected number of collisions to be smaller than 1, we need $T > S(S-1)/2$, so:

$$M > 2 \log_2 S.$$

We need *twice as many* bits as the number of bits $\log_2 S$ that would be sufficient to give each entry a unique name. If we are happy to have occasional collisions involving a fraction $f$ of the names $S$, then we need $T > S/f$, so:

$$M > \log_2 S + \log_2 [1/f].$$

For $f \simeq 0.01$, that means an extra 7 bits above $\log_2 S$.

### 12.5 Other Roles for Hash Codes

#### Checking Arithmetic

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Casting Out Nines)</span></p>

**Casting out nines** is a method for checking addition by hand. One finds the sum, modulo nine, of all the *digits* of the numbers to be summed and compares it with the sum, modulo nine, of the digits of the putative answer.

For any addition expression of the form $a + b + c + \cdots$ where $a, b, c, \dots$ are decimal numbers, define:

$$h(a + b + c + \cdots) = \text{sum modulo nine of all digits in } a, b, c.$$

It is a nice property of decimal arithmetic that if $a + b + c + \cdots = m + n + o + \cdots$, then $h(a + b + c + \cdots)$ and $h(m + n + o + \cdots)$ are equal.

</div>

#### Error Detection Among Friends

Hash codes can be used to determine whether two files are the same without transferring either file. Alice and Bob each compute the hash of their respective files using the same $M$-bit hash function. If the hashes match, the two files are almost surely the same.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(False Negative Probability)</span></p>

What is the probability of a false negative, i.e., the probability that the two files do differ but the two hashes are nevertheless identical?

If we assume the hash function is random and the process causing the files to differ knows nothing about the hash function, the probability of a false negative is $2^{-M}$. A 32-bit hash gives a probability of false negative of about $10^{-10}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Cyclic Redundancy Check)</span></p>

It is common practice to use a linear hash function called a **32-bit cyclic redundancy check** to detect errors in files. A cyclic redundancy check is a set of 32 parity-check bits similar to the 3 parity-check bits of the $(7,4)$ Hamming code.

To have a false-negative rate smaller than one in a billion, $M = 32$ bits is plenty -- if the errors are produced by noise.

</div>

#### Tamper Detection

If the differences between two files are introduced by an adversary (a forger Fiona), rather than by noise, then a simple linear hash function provides no security: Fiona can make chosen modifications and use linear algebra to restore the hash to its original value. *Linear hash functions give no security against forgers.*

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(One-Way Hash Function)</span></p>

A **one-way hash function** is a hash function that is easy to compute but hard to invert -- that is, no one can construct a tampering that leaves the hash function unaffected. Finding such functions is one of the active research areas of cryptography. An example is **MD5**, which produces a 128-bit hash.

</div>

Even with a good one-way hash function, digital signatures are still vulnerable if the attacker has access to the hash function. If the hash has $M$ bits, the attacker can try about $2^M$ file modifications on average before finding a tampered file whose hash matches the original. To be secure against forgery, digital signatures must either have enough bits for such a random search to take too long, or the hash function itself must be kept secret.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Forgery by the Author)</span></p>

If Alice herself wishes to forge (e.g., place a bet without revealing it, then later change her bet), she has an advantage: she is the author of *both* files. She can search for a collision between two sets of files. By preparing $N_1$ versions of bet one and $N_2$ versions of bet two, the expected number of collisions between a Montague and a Capulet is:

$$N_1 N_2 \, 2^{-M}.$$

To minimize the total number of files hashed ($N_1 + N_2$), Alice should make $N_1$ and $N_2$ equal, and will need to hash about $2^{M/2}$ files until she finds two that match. This is the square root of the number of hashes Fiona (an external forger) had to make. Thus if Alice has $C = 10^6$ computers for $T = 10$ years with each hash taking $t = 1\text{ ns}$, the bet-communication system is secure against Alice's dishonesty only if $M \gg 2 \log_2 CT/t \simeq 160$ bits.

</div>

## Chapter 13 -- Binary Codes

This chapter focuses on the special case of channels with binary inputs, with the binary symmetric channel as the implicit first choice. The optimal decoder for a code on a binary symmetric channel finds the codeword that is closest to the received vector in **Hamming distance**. One of the key aims of the chapter is to contrast Shannon's goal of reliable communication over a noisy channel with the coding-theoretic obsession with maximizing the distance between codewords -- an emphasis that turns out to have only a tenuous relationship to Shannon's aim.

### 13.1 Distance Properties of a Code

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Hamming Distance and Minimum Distance)</span></p>

The **Hamming distance** between two binary vectors is the number of coordinates in which the two vectors differ. The **distance** (or **minimum distance**) of a code, often denoted $d$ or $d_{\min}$, is the smallest separation between any two of its codewords. The maximum number of errors a code with distance $d$ can *guarantee* to correct is $t = \lfloor (d-1)/2 \rfloor$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Weight Enumerator Function)</span></p>

The **weight enumerator function** (or **distance distribution**) of a code, $A(w)$, is defined to be the number of codewords in the code that have weight $w$. For a linear code, all codewords have identical distance properties, so we can summarize all the distances between codewords by counting the distances from the all-zero codeword.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Weight Enumerators of Known Codes)</span></p>

- The $(7, 4)$ Hamming code has distance $d = 3$. Its weight enumerator is: $A(0) = 1$, $A(3) = 7$, $A(4) = 7$, $A(7) = 1$; total 16 codewords.
- The $(30, 11)$ dodecahedron code has distance $d = 5$. Its weight enumerator has a broader distribution peaking around weight 15.

</div>

### 13.2 Obsession with Distance

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Bounded-Distance Decoder)</span></p>

A **bounded-distance decoder** returns the closest codeword to a received binary vector $\mathbf{r}$ if the distance from $\mathbf{r}$ to that codeword is less than or equal to $t$; otherwise it returns a failure message.

</div>

The rationale for not trying to decode when more than $t$ errors have occurred is a form of **worst-case-ism**: "we can't *guarantee* that we can correct more than $t$ errors, so we won't bother trying." This defeatist attitude is a widespread mental ailment. The fact is that bounded-distance decoders cannot reach the Shannon limit of the binary symmetric channel; only a decoder that often corrects more than $t$ errors can do this.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Good and Bad Distance)</span></p>

Given a family of codes of increasing blocklength $N$ with rates approaching a limit $R > 0$:

- A sequence of codes has **'good' distance** if $d/N$ tends to a constant greater than zero.
- A sequence of codes has **'bad' distance** if $d/N$ tends to zero.
- A sequence of codes has **'very bad' distance** if $d$ tends to a constant.

A *low-density generator-matrix code* (whose $K \times N$ generator matrix $\mathbf{G}$ has a small fixed number $d_0$ of 1s per row) has minimum distance at most $d_0$, so these codes have 'very bad' distance.

</div>

### 13.3 Perfect Codes

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Perfect Code)</span></p>

A **$t$-sphere** (sphere of radius $t$) in Hamming space, centred on a point $\mathbf{x}$, is the set of points whose Hamming distance from $\mathbf{x}$ is less than or equal to $t$. A code is a **perfect $t$-error-correcting code** if the set of $t$-spheres centred on the codewords fill the Hamming space without overlapping.

The number of points in a Hamming sphere of radius $t$ in $N$-dimensional space is:

$$\sum_{w=0}^{t} \binom{N}{w}.$$

For a perfect code with $S = 2^K$ codewords in a space of $2^N$ points:

$$2^K \sum_{w=0}^{t} \binom{N}{w} = 2^N, \qquad \text{or equivalently,} \qquad \sum_{w=0}^{t} \binom{N}{w} = 2^{N-K}.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Classification of Perfect Binary Codes)</span></p>

There are almost no nontrivial perfect binary codes. The only ones are:

1. **The Hamming codes**, which are perfect codes with $t = 1$ and blocklength $N = 2^M - 1$. The rate approaches 1 as $N$ increases.
2. **The repetition codes** of odd blocklength $N$, which are perfect codes with $t = (N-1)/2$. The rate goes to zero as $1/N$.
3. **The binary Golay code**, a remarkable 3-error-correcting code with $2^{12}$ codewords of blocklength $N = 23$. It satisfies $1 + \binom{23}{1} + \binom{23}{2} + \binom{23}{3} = 2^{11}$.

There are no other binary perfect codes. For most codes, whether good or bad, almost all the Hamming space is taken up by the space *between* $t$-spheres.

</div>

#### The Hamming Codes

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Generalized Hamming Code)</span></p>

The Hamming codes are single-error-correcting codes defined by picking a number of parity-check constraints $M$. The blocklength is $N = 2^M - 1$ and the parity-check matrix contains, as its columns, all the $N$ non-zero vectors of length $M$ bits. Since these $N$ vectors are all different, any single bit-flip produces a distinct syndrome, so all single-bit errors can be detected and corrected.

| Checks $M$ | $(N, K)$ | $R = K/N$ | Note |
| --- | --- | --- | --- |
| 2 | $(3, 1)$ | $1/3$ | repetition code $R_3$ |
| 3 | $(7, 4)$ | $4/7$ | $(7, 4)$ Hamming code |
| 4 | $(15, 11)$ | $11/15$ | |
| 5 | $(31, 26)$ | $26/31$ | |
| 6 | $(63, 57)$ | $57/63$ | |

</div>

### 13.4 Perfectness Is Unattainable -- First Proof

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(No Perfect Codes for $f > 1/3$)</span></p>

For a binary symmetric channel with noise level $f > 1/3$, there are *no perfect codes that can do this*. Suppose a large perfect $fN$-error-correcting code exists at rate $R \simeq 1 - H_2(f)$. Consider three codewords: the all-zero codeword, and two others that differ from the first in fractions $u + v$ and $u + w$ of their coordinates, respectively, and from each other in a fraction $v + w$. A fraction $x$ of coordinates have value zero in all three codewords.

Since the code is $fN$-error-correcting, the minimum distance must be greater than $2fN$, so $u + v > 2f$, $v + w > 2f$, and $u + w > 2f$. Summing these three inequalities and dividing by two:

$$u + v + w > 3f.$$

Since $u + v + w + x = 1$, we deduce $x < 1 - 3f$. If $f > 1/3$, then $x < 0$, which is impossible. Such a code cannot even have *three* codewords, let alone $2^{NR}$.

</div>

### 13.5 Weight Enumerator Function of Random Linear Codes

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Expected Weight Enumerator of Random Linear Codes)</span></p>

For a random linear code with $M \times N$ parity-check matrix $\mathbf{H}$ (entries chosen uniformly at random), the weight enumerator of one particular code with parity-check matrix $\mathbf{H}$ is:

$$A(w)_{\mathbf{H}} = \sum_{\mathbf{x}: |\mathbf{x}|=w} \mathbb{1}[\mathbf{H}\mathbf{x} = 0],$$

where the truth function $\mathbb{1}[\mathbf{H}\mathbf{x} = 0]$ equals one if $\mathbf{H}\mathbf{x} = 0$ and zero otherwise. The probability that the entire syndrome $\mathbf{H}\mathbf{x}$ is zero is $2^{-M}$, independent of $w$. Therefore:

$$\langle A(w) \rangle = \binom{N}{w} 2^{-M} \quad \text{for any } w > 0.$$

For large $N$, using $\log \binom{N}{w} \simeq N H_2(w/N)$ and $R \simeq 1 - M/N$:

$$\log_2 \langle A(w) \rangle \simeq N[H_2(w/N) - (1-R)] \quad \text{for any } w > 0.$$

</div>

### 13.5.1 Gilbert--Varshamov Distance

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Gilbert--Varshamov Distance)</span></p>

For weights $w$ such that $H_2(w/N) < (1-R)$, the expectation of $A(w)$ is smaller than 1; for weights such that $H_2(w/N) > (1-R)$, it is greater than 1. The minimum distance of a random linear code will thus be close to the **Gilbert--Varshamov distance** $d_{\text{GV}}$ defined by:

$$H_2(d_{\text{GV}} / N) = (1 - R).$$

The **Gilbert--Varshamov conjecture** asserts that (for large $N$) it is not possible to create binary codes with minimum distance significantly greater than $d_{\text{GV}}$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Gilbert--Varshamov Rate)</span></p>

The **Gilbert--Varshamov rate** $R_{\text{GV}}$ is the maximum rate at which you can reliably communicate with a bounded-distance decoder, assuming the Gilbert--Varshamov conjecture is true. If one uses a bounded-distance decoder, the maximum tolerable noise level $f_{\text{bd}} = \frac{1}{2} d_{\min}/N$. Assuming $d_{\min} = d_{\text{GV}}$:

$$H_2(2f_{\text{bd}}) = (1 - R_{\text{GV}}), \qquad R_{\text{GV}} = 1 - H_2(2f_{\text{bd}}).$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Bounded-Distance Decoders Lose Half the Noise Tolerance)</span></p>

Shannon's capacity for the BSC is $C = 1 - H_2(f)$, so for a given rate $R$, the maximum tolerable noise level according to Shannon satisfies $H_2(f) = (1-R)$. Comparing with the bounded-distance equation $H_2(2f_{\text{bd}}) = (1-R)$, we get:

$$f_{\text{bd}} = f/2.$$

Bounded-distance decoders can only ever cope with **half** the noise level that Shannon proved is tolerable. This is a fundamental limitation of the distance-obsessed approach.

</div>

### 13.6 Berlekamp's Bats

The relationship between minimum distance and error probability is illustrated by Berlekamp's analogy of a blind bat in a cave. The bat flies about the centre of the cave (corresponding to one codeword), and the boundaries of the cave are made up of stalactites (corresponding to boundaries between codewords).

- **Low noise (small friskiness):** the bat stays near the centre and only collides with the closest stalactites. Under low-noise conditions, decoding errors are rare and involve the lowest-weight codewords. The minimum distance is relevant to the (very small) probability of error.
- **Moderate noise:** the bat makes excursions beyond the safe distance $t$ but collides most frequently with more distant stalactites (owing to their greater number). There are only a tiny number of stalactites at the minimum distance, so they are relatively unlikely to cause errors. Errors depend on the *weight enumerator function*, not just $d_{\min}$.
- **High noise:** the bat is always far from the centre, and almost all collisions involve distant stalactites. The collision frequency has nothing to do with the distance to the closest stalactite.

### 13.7 Concatenation of Hamming Codes

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Concatenated Hamming Codes)</span></p>

We can create a concatenated code for a binary symmetric channel by encoding with several Hamming codes in succession. The Hamming codes have:

$$N = 2^M - 1, \quad K = N - M, \quad p_{\text{B}} = \frac{3}{N}\binom{N}{2}f^2 \text{ (to leading order)}.$$

If we make a product code by concatenating $C$ Hamming codes with parameters $\lbrace M_c \rbrace_{c=1}^{C}$, the rate of the product code is:

$$R_C = \prod_{c=1}^{C} \frac{N_c - M_c}{N_c},$$

which tends to a non-zero limit as $C$ increases (e.g., setting $M_1 = 2$, $M_2 = 3$, $M_3 = 4$, etc., gives an asymptotic rate of 0.093).

The minimum distance of the $C$th product is $3^C$, and the blocklength grows as $\sim 3^C$ too, so $d/N$ tends to zero -- concatenated Hamming codes have **'bad' distance**. Nevertheless, they are a 'good' code family: the bit error probability drops to zero while the rate tends to 0.093.

The take-home message: **distance isn't everything**.

</div>

### 13.8 Distance Isn't Everything

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Error Probability from a Single Low-Weight Codeword)</span></p>

Let a binary code have blocklength $N$ and just two codewords differing in $d$ places (assume $d$ even). The error probability when used on a BSC with noise level $f$ is dominated by the probability that $d/2$ of the differing bits are flipped:

$$P(\text{block error}) \simeq \binom{d}{d/2} f^{d/2}(1-f)^{d/2} \simeq [\beta(f)]^d,$$

where $\beta(f) = 2f^{1/2}(1-f)^{1/2}$ is called the **Bhattacharyya parameter** of the channel.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Practical Implications of Distance)</span></p>

A general linear code with distance $d$ has block error probability at least $\binom{d}{d/2} f^{d/2}(1-f)^{d/2}$, independent of the blocklength $N$. Therefore, a sequence of codes with constant $d$ ('very bad' distance) cannot have block error probability tending to zero on any BSC.

However, for practical purposes, it is not essential to have good distance. Codes of blocklength 10000 known to have many codewords of weight 32 can nevertheless correct errors of weight 320 with tiny error probability. For disk drives needing error probability below $10^{-18}$, a code with distance $d = 20$ at raw error rate 0.001 gives error probability below $10^{-24}$.

</div>

### 13.9 The Union Bound

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Union Bound)</span></p>

The error probability of a code on the binary symmetric channel can be bounded in terms of its weight enumerator function:

$$P(\text{block error}) \le \sum_{w > 0} A(w) [\beta(f)]^w,$$

where $\beta(f) = 2f^{1/2}(1-f)^{1/2}$ is the Bhattacharyya parameter. This **union bound** is accurate for low noise levels $f$ but inaccurate for high noise levels, because it overcounts errors that cause confusion with more than one codeword at a time.

</div>

### 13.10 Dual Codes

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Dual Code)</span></p>

The set of *all* vectors of length $N$ that are orthogonal to all codewords in a code $\mathcal{C}$ is called the **dual** of the code, $\mathcal{C}^{\perp}$.

- The **generator matrix** specifies $K$ vectors *from which* all codewords can be built.
- The **parity-check matrix** specifies $M$ vectors *to which* all codewords are orthogonal.
- The dual of a code is obtained by exchanging the generator matrix and the parity-check matrix.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Duals of Standard Codes)</span></p>

- **Repetition code $R_3$:** generator matrix $\mathbf{G} = [1\ 1\ 1]$, parity-check matrix $\mathbf{H} = \begin{bmatrix} 1 & 1 & 0 \\\\ 1 & 0 & 1 \end{bmatrix}$. The dual code has generator matrix $\mathbf{G}^{\perp} = \mathbf{H}$, which is the **simple parity code** $P_3$ (the code with one parity-check bit equal to the sum of the two source bits).
- **$(7,4)$ Hamming code:** the dual $\mathcal{H}_{(7,4)}^{\perp}$ is *contained within* $\mathcal{H}_{(7,4)}$ itself, i.e., $\mathcal{H}_{(7,4)}^{\perp} \subset \mathcal{H}_{(7,4)}$. Every word in the dual code is a codeword of the original.

In general, if a code has a systematic generator matrix $\mathbf{G} = [\mathbf{I}_K | \mathbf{P}^{\mathsf{T}}]$, then its parity-check matrix is $\mathbf{H} = [\mathbf{P} | \mathbf{I}_M]$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Goodness of Duals)</span></p>

If a sequence of codes is 'good', are their duals good too? All combinations can be constructed:
- Good codes with good duals (random linear codes).
- Bad codes with bad duals.
- Good codes with bad duals. This is especially important: many state-of-the-art codes (e.g., low-density parity-check codes) have the property that their duals are bad (the dual being a low-density generator-matrix code).

</div>

#### Self-Dual Codes

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Self-Orthogonal and Self-Dual Codes)</span></p>

A code is **self-orthogonal** if it is contained in its dual: $\mathcal{C} \subseteq \mathcal{C}^{\perp}$. The $(7, 4)$ Hamming code is self-orthogonal (the overlap between any pair of rows of $\mathbf{H}$ is even). Codes that contain their duals are important in quantum error-correction.

A code $\mathcal{C}$ is **self-dual** if $\mathcal{C}^{\perp} = \mathcal{C}$. Properties of self-dual codes:
1. The generator matrix is also a parity-check matrix.
2. The rate is $1/2$ (i.e., $M = K = N/2$).
3. All codewords have even weight.

If $\mathbf{G} = [\mathbf{I}_K | \mathbf{P}^{\mathsf{T}}]$ is self-dual, then $\mathbf{P}$ must be an orthogonal matrix (modulo 2): $\mathbf{P}^{\mathsf{T}}\mathbf{P} = \mathbf{I}_K$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Self-Dual Codes)</span></p>

- The repetition code $R_2$ with $\mathbf{G} = \mathbf{H} = [1\ 1]$ is a simple self-dual code.
- The smallest non-trivial self-dual code is the $(8, 4)$ code with generator matrix:

$$\mathbf{G} = [\mathbf{I}_4 | \mathbf{P}^{\mathsf{T}}] = \begin{bmatrix} 1 & 0 & 0 & 0 & 0 & 1 & 1 & 1 \\\\ 0 & 1 & 0 & 0 & 1 & 0 & 1 & 1 \\\\ 0 & 0 & 1 & 0 & 1 & 1 & 0 & 1 \\\\ 0 & 0 & 0 & 1 & 1 & 1 & 1 & 0 \end{bmatrix}.$$

This $(8, 4)$ code is obtained from the $(7, 4)$ Hamming code by appending an extra parity-check bit.

</div>

### 13.11 Generalizing Perfectness to Other Channels

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Maximum Distance Separable Code)</span></p>

A code that can restore any $u$ erased bits and never more than $u$ can be called a **perfect $u$-error-correcting code for the binary erasure channel**. The conventional term is a **maximum distance separable (MDS) code**. In such a code, the number of redundant bits must be $N - K = u$.

The $(7, 4)$ Hamming code is *not* an MDS code: it can recover *some* sets of 3 erased bits but not all. No MDS binary codes exist apart from the repetition codes and simple parity codes. For the $q$-ary erasure channel with $q > 2$, MDS block codes of any rate can be found (e.g., **Reed--Solomon codes**).

</div>

### 13.12 Summary

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Summary: Why Distance Has Little Relevance)</span></p>

Shannon's codes for the binary symmetric channel can almost always correct $fN$ errors, but they are not $fN$-error-correcting codes. Reasons why the distance of a code has little relevance:

1. The Shannon limit shows that the best codes must cope with a noise level **twice as big** as the maximum noise level for a bounded-distance decoder.
2. When the BSC has $f > 1/4$, no code with a bounded-distance decoder can communicate at all -- but Shannon says good codes exist for such channels.
3. Concatenation shows that we can get good performance even if the distance is bad.

The whole **weight enumerator function** is relevant to whether a code is good, not just the minimum distance.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Bitwise vs. Block Decoding Error Probability)</span></p>

If a binary linear code has minimum distance $d_{\min}$, then for any given channel, the codeword bit error probability of the optimal bitwise decoder $p_{\text{b}}$ and the block error probability of the maximum likelihood decoder $p_{\text{B}}$ are related by:

$$p_{\text{B}} \ge p_{\text{b}} \ge \frac{1}{2} \frac{d_{\min}}{N} p_{\text{B}}.$$

</div>

## Chapter 14 -- Very Good Linear Codes Exist

This chapter draws together several ideas from earlier chapters into a single, elegant proof. It simultaneously proves both Shannon's noisy-channel coding theorem (for symmetric binary channels) and his source coding theorem (for binary sources). The proof is more constructive than the one given in Chapter 10: rather than showing that almost any random code is very good, it shows that almost any *linear* code is very good. It makes use of typical sets (Chapters 4 and 10) and the weight enumerator function of random linear codes (Section 13.5). On the source coding side, the proof shows that *random linear hash functions* can compress compressible binary sources, connecting to Chapter 12.

### 14.1 A Simultaneous Proof of the Source Coding and Noisy-Channel Coding Theorems

We consider a linear error-correcting code with binary parity-check matrix $\mathbf{H}$. The matrix has $M$ rows and $N$ columns. We increase $N$ and $M$, keeping $M \propto N$. The rate of the code satisfies

$$R \ge 1 - \frac{M}{N}.$$

If all rows of $\mathbf{H}$ are independent then $R = 1 - M/N$. We assume equality holds throughout.

A codeword $\mathbf{t}$ is selected satisfying

$$\mathbf{H}\mathbf{t} = \mathbf{0} \bmod 2,$$

and a binary symmetric channel adds noise $\mathbf{x}$, giving the received signal

$$\mathbf{r} = \mathbf{t} + \mathbf{x} \bmod 2.$$

Here $\mathbf{x}$ denotes the noise added by the channel, not the input to the channel.

The receiver aims to infer both $\mathbf{t}$ and $\mathbf{x}$ from $\mathbf{r}$ using a **syndrome-decoding** approach. The receiver computes the syndrome

$$\mathbf{z} = \mathbf{H}\mathbf{r} \bmod 2 = \mathbf{H}\mathbf{t} + \mathbf{H}\mathbf{x} \bmod 2 = \mathbf{H}\mathbf{x} \bmod 2.$$

The syndrome depends only on the noise $\mathbf{x}$, and the decoding problem is to find the most probable $\mathbf{x}$ satisfying $\mathbf{H}\mathbf{x} = \mathbf{z} \bmod 2$. The best estimate $\hat{\mathbf{x}}$ is subtracted from $\mathbf{r}$ to give the best guess for $\mathbf{t}$.

Our aim is to show that, as long as $R < 1 - H(X) = 1 - H_2(f)$, where $f$ is the flip probability of the BSC, the optimal decoder has vanishing probability of error as $N$ increases, for random $\mathbf{H}$.

#### The Typical-Set Decoder

We prove this by studying a sub-optimal strategy: the *typical-set decoder*. It examines the typical set $T$ of noise vectors, the set of noise vectors $\mathbf{x}'$ that satisfy $\log 1/P(\mathbf{x}') \simeq NH(X)$, checking if any of those typical vectors $\mathbf{x}'$ satisfies the observed syndrome

$$\mathbf{H}\mathbf{x}' = \mathbf{z}.$$

If exactly one typical vector $\mathbf{x}'$ does so, the decoder reports that vector as the hypothesized noise. If no typical vector or more than one matches the syndrome, the decoder reports an error.

#### Error Analysis

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Decomposition of Error Probability)</span></p>

The probability of error of the typical-set decoder, for a given matrix $\mathbf{H}$, decomposes as

$$P_{\text{TS}|\mathbf{H}} = P^{(I)} + P^{(II)}_{\text{TS}|\mathbf{H}},$$

where:
- $P^{(I)}$ is the probability that the true noise vector $\mathbf{x}$ is itself not typical. This vanishes as $N$ increases.
- $P^{(II)}_{\text{TS}|\mathbf{H}}$ is the probability that the true $\mathbf{x}$ is typical and at least one other typical vector clashes with it.

</div>

We concentrate on the type-II error. We use the indicator function $\mathbb{1}[\mathbf{H}(\mathbf{x}' - \mathbf{x}) = 0]$, which is one if the statement is true and zero otherwise. The number of type-II errors when the noise is $\mathbf{x}$ is bounded by

$$[\text{Number of errors given } \mathbf{x} \text{ and } \mathbf{H}] \le \sum_{\mathbf{x}': \, \mathbf{x}' \in T, \, \mathbf{x}' \neq \mathbf{x}} \mathbb{1}[\mathbf{H}(\mathbf{x}' - \mathbf{x}) = 0].$$

This is a union bound. Averaging over $\mathbf{x}$:

$$P^{(II)}_{\text{TS}|\mathbf{H}} \le \sum_{\mathbf{x} \in T} P(\mathbf{x}) \sum_{\mathbf{x}': \, \mathbf{x}' \in T, \, \mathbf{x}' \neq \mathbf{x}} \mathbb{1}[\mathbf{H}(\mathbf{x}' - \mathbf{x}) = 0].$$

#### Averaging over All Linear Codes

We find the average of this probability of type-II error over all linear codes by averaging over $\mathbf{H}$. We denote this average by $\langle \ldots \rangle_{\mathbf{H}}$.

The quantity $\langle \mathbb{1}[\mathbf{H}(\mathbf{x}' - \mathbf{x}) = 0] \rangle_{\mathbf{H}}$ already appeared when calculating the expected weight enumerator function of random linear codes (Section 13.5): for any non-zero binary vector $\mathbf{v}$, the probability that $\mathbf{H}\mathbf{v} = 0$, averaging over all binary matrices $\mathbf{H}$, is $2^{-M}$. So

$$\bar{P}^{(II)}_{\text{TS}} = \left( \sum_{\mathbf{x} \in T} P(\mathbf{x}) \right) (|T| - 1) \, 2^{-M} \le |T| \, 2^{-M}.$$

Since there are roughly $2^{NH(X)}$ noise vectors in the typical set:

$$\bar{P}^{(II)}_{\text{TS}} \le 2^{NH(X)} 2^{-M}.$$

This bound either vanishes or grows exponentially as $N$ increases. It vanishes if

$$H(X) < M/N.$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Noisy-Channel Coding Theorem for Linear Codes)</span></p>

Substituting $R = 1 - M/N$, we establish the noisy-channel coding theorem for the binary symmetric channel: very good linear codes exist for any rate $R$ satisfying

$$R < 1 - H(X),$$

where $H(X)$ is the entropy of the channel noise, per bit. $\square$

</div>

### 14.2 Data Compression by Linear Hash Codes

The decoding game above can also be viewed as an *uncompression* game. The world produces a binary noise vector $\mathbf{x}$ from a source $P(\mathbf{x})$. The noise has redundancy (if the flip probability is not $0.5$). We compress it with a linear compressor that maps the $N$-bit input $\mathbf{x}$ (the noise) to the $M$-bit output $\mathbf{z}$ (the syndrome). The uncompression task is to recover the input $\mathbf{x}$ from the output $\mathbf{z}$. The rate of the compressor is

$$R_{\text{compressor}} \equiv M/N.$$

The result that the decoding problem can be solved, for almost any $\mathbf{H}$, with vanishing error probability as long as $H(X) < M/N$ instantly proves a source coding theorem:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Source Coding via Linear Hash Functions)</span></p>

Given a binary source $X$ of entropy $H(X)$, and a required compressed rate $R > H(X)$, there exists a linear compressor $\mathbf{x} \to \mathbf{z} = \mathbf{H}\mathbf{x} \bmod 2$ having rate $M/N$ equal to that required rate $R$, and an associated uncompressor, that is virtually lossless.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Generality of the Source Coding Theorem)</span></p>

This theorem is true not only for a source of independent identically distributed symbols but also for any source for which a typical set can be defined: sources with memory, and time-varying sources, for example; all that's required is that the source be ergodic.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Notes)</span></p>

This method for proving that codes are good can be applied to other linear codes, such as low-density parity-check codes. For each code we need an approximation of its expected weight enumerator function.

</div>

## Chapter 15 -- Further Exercises on Information Theory

This chapter collects additional exercises on information theory, ranging from refresher problems on source coding and noisy channels to more advanced topics such as communication of correlated sources, multiple access channels, and broadcast channels.

### Refresher Exercises on Source Coding and Noisy Channels

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 15.1</span></p>

Let $X$ be an ensemble with $\mathcal{A}_X = \{0, 1\}$ and $\mathcal{P}_X = \{0.995, 0.005\}$. Consider source coding using the block coding of $X^{100}$ where every $\mathbf{x} \in X^{100}$ containing 3 or fewer 1s is assigned a distinct codeword, while the other $\mathbf{x}$s are ignored.

- **(a)** If the assigned codewords are all of the same length, find the minimum length required to provide the above set with distinct codewords.
- **(b)** Calculate the probability of getting an $\mathbf{x}$ that will be ignored.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 15.2</span></p>

Let $X$ be an ensemble with $\mathcal{P}_X = \{0.1, 0.2, 0.3, 0.4\}$. The ensemble is encoded using the symbol code $\mathcal{C} = \{0001, 001, 01, 1\}$. Consider the codeword corresponding to $\mathbf{x} \in X^N$, where $N$ is large.

- **(a)** Compute the entropy of the fourth bit of transmission.
- **(b)** Compute the conditional entropy of the fourth bit given the third bit.
- **(c)** Estimate the entropy of the hundredth bit.
- **(d)** Estimate the conditional entropy of the hundredth bit given the ninety-ninth bit.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 15.3</span></p>

Two fair dice are rolled by Alice and the sum is recorded. Bob's task is to ask a sequence of questions with yes/no answers to find out this number. Devise in detail a strategy that achieves the minimum possible average number of questions.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 15.4</span></p>

How can you use a coin to draw straws among 3 people?

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 15.5</span></p>

In a magic trick, there are three participants: the magician, an assistant, and a volunteer. The assistant, who claims to have paranormal abilities, is in a soundproof room. The magician gives the volunteer six blank cards, five white and one blue. The volunteer writes a different integer from 1 to 100 on each card, as the magician is watching. The volunteer keeps the blue card. The magician arranges the five white cards in some order and passes them to the assistant. The assistant then announces the number on the blue card.

How does the trick work?

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 15.6</span></p>

How does *this* trick work?

'Here's an ordinary pack of cards, shuffled into random order. Please choose five cards from the pack, any that you wish. Don't let me see their faces. No, don't give them to me: pass them to my assistant Esmerelda. She can look at them.'

'Now, Esmerelda, show me four of the cards. Hmm... nine of spades, six of clubs, four of hearts, ten of diamonds. The hidden card, then, must be the queen of spades!'

The trick can be performed as described above for a pack of 52 cards. Use information theory to give an upper bound on the number of cards for which the trick can be performed.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 15.7</span></p>

Find a probability sequence $\mathbf{p} = (p_1, p_2, \ldots)$ such that $H(\mathbf{p}) = \infty$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 15.8</span></p>

Consider a discrete memoryless source with $\mathcal{A}_X = \{a, b, c, d\}$ and $\mathcal{P}_X = \{1/2, 1/4, 1/8, 1/8\}$. There are $4^8 = 65\,536$ eight-letter words that can be formed from the four letters. Find the total number of such words that are in the typical set $T_{N\beta}$ (equation 4.29) where $N = 8$ and $\beta = 0.1$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 15.9</span></p>

Consider the source $\mathcal{A}_S = \{a, b, c, d, e\}$, $\mathcal{P}_S = \{1/3, 1/3, 1/9, 1/9, 1/9\}$ and the channel whose transition probability matrix is

$$Q = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 2/3 & 0 \\ 0 & 1 & 0 & 1 \\ 0 & 0 & 1/3 & 0 \end{bmatrix}.$$

Note that the source alphabet has five symbols, but the channel alphabet $\mathcal{A}_X = \mathcal{A}_Y = \{0, 1, 2, 3\}$ has only four. Assume that the source produces symbols at exactly $3/4$ the rate that the channel accepts channel symbols. For a given (tiny) $\epsilon > 0$, explain how you would design a system for communicating the source's output over the channel with an average error probability per source symbol less than $\epsilon$. Be as explicit as possible. In particular, do *not* invoke Shannon's noisy-channel coding theorem.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 15.10</span></p>

Consider a binary symmetric channel and a code $C = \{0000, 0011, 1100, 1111\}$; assume that the four codewords are used with probabilities $\{1/2, 1/8, 1/8, 1/4\}$.

What is the decoding rule that minimizes the probability of decoding error? [The optimal decoding rule depends on the noise level $f$ of the binary symmetric channel. Give the decoding rule for each range of values of $f$, for $f$ between 0 and $1/2$.]

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 15.11</span></p>

Find the capacity and optimal input distribution for the three-input, three-output channel whose transition probabilities are:

$$Q = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 2/3 & 1/3 \\ 0 & 1/3 & 2/3 \end{bmatrix}.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 15.12</span></p>

The input to a channel $Q$ is a word of 8 bits. The output is also a word of 8 bits. Each time it is used, the channel flips *exactly one* of the transmitted bits, but the receiver does not know which one. The other seven bits are received without error. All 8 bits are equally likely to be the one that is flipped. Derive the capacity of this channel.

Show, by describing an *explicit* encoder and decoder that it is possible to communicate *reliably* (that is, with *zero* error probability) 5 bits per cycle over this channel.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 15.13</span></p>

A channel with input $x \in \{\mathtt{a}, \mathtt{b}, \mathtt{c}\}$ and output $y \in \{\mathtt{r}, \mathtt{s}, \mathtt{t}, \mathtt{u}\}$ has conditional probability matrix:

$$\mathbf{Q} = \begin{bmatrix} 1/2 & 0 & 0 \\ 1/2 & 1/2 & 0 \\ 0 & 1/2 & 1/2 \\ 0 & 0 & 1/2 \end{bmatrix}.$$

What is its capacity?

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 15.14</span></p>

The ten-digit number on the cover of a book known as the ISBN incorporates an error-detecting code. The number consists of nine source digits $x_1, x_2, \ldots, x_9$, satisfying $x_n \in \{0, 1, \ldots, 9\}$, and a tenth check digit whose value is given by

$$x_{10} = \left( \sum_{n=1}^{9} n x_n \right) \bmod 11.$$

Here $x_{10} \in \{0, 1, \ldots, 9, 10\}$. If $x_{10} = 10$ then the tenth digit is shown using the roman numeral X.

Show that a valid ISBN satisfies: $\left( \sum_{n=1}^{10} n x_n \right) \bmod 11 = 0.$

Show that this code can be used to detect (but not correct) all errors in which any one of the ten digits is modified. Show that this code can be used to detect all errors in which any two adjacent digits are transposed. What other transpositions of pairs of *non-adjacent* digits can be detected?

If the tenth digit were defined to be $x_{10} = \left( \sum_{n=1}^{9} n x_n \right) \bmod 10$, why would the code not work so well?

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 15.15</span></p>

A channel with input $x$ and output $y$ has transition probability matrix:

$$Q = \begin{bmatrix} 1-f & f & 0 & 0 \\ f & 1-f & 0 & 0 \\ 0 & 0 & 1-g & g \\ 0 & 0 & g & 1-g \end{bmatrix}.$$

Assuming an input distribution of the form $\mathcal{P}_X = \left\{ \frac{p}{2}, \frac{p}{2}, \frac{1-p}{2}, \frac{1-p}{2} \right\}$, write down the entropy of the output, $H(Y)$, and the conditional entropy of the output given the input, $H(Y \mid X)$.

Show that the optimal input distribution is given by

$$p = \frac{1}{1 + 2^{-H_2(g) + H_2(f)}},$$

where $H_2(f) = f \log_2 \frac{1}{f} + (1-f) \log_2 \frac{1}{1-f}$.

Write down the optimal input distribution and the capacity of the channel in the case $f = 1/2$, $g = 0$, and comment on your answer.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 15.16</span></p>

What are the differences in the redundancies needed in an error-detecting code (which can reliably detect that a block of data has been corrupted) and an error-correcting code (which can detect and correct errors)?

</div>

### Further Tales from Information Theory

The following exercises introduce some more surprising results of information theory.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 15.17</span><span class="math-callout__name">(Communication of Information from Correlated Sources)</span></p>

Imagine that we want to communicate data from two data sources $X^{(A)}$ and $X^{(B)}$ to a central location C via noise-free one-way communication channels. The signals $x^{(A)}$ and $x^{(B)}$ are strongly dependent, so their joint information content is only a little greater than the marginal information content of either of them. For example, C is a weather collator who wishes to receive a string of reports saying whether it is raining in Allerton ($x^{(A)}$) and whether it is raining in Bognor ($x^{(B)}$). The joint probability might be

| $P(x^{(A)}, x^{(B)})$ | $x^{(A)} = 0$ | $x^{(A)} = 1$ |
|---|---|---|
| $x^{(B)} = 0$ | 0.49 | 0.01 |
| $x^{(B)} = 1$ | 0.01 | 0.49 |

Assuming that $x^{(A)}$ and $x^{(B)}$ are generated repeatedly from this distribution, can they be encoded at rates $R_A$ and $R_B$ such that C can reconstruct all the variables, with the sum of information transmission rates on the two lines being less than two bits per cycle?

In the general case of two dependent sources $X^{(A)}$ and $X^{(B)}$, there exist codes for the two transmitters that can achieve reliable communication of both $X^{(A)}$ and $X^{(B)}$ to C, as long as: the information rate from $X^{(A)}$, $R_A$, exceeds $H(X^{(A)} \mid X^{(B)})$; the information rate from $X^{(B)}$, $R_B$, exceeds $H(X^{(B)} \mid X^{(A)})$; and the total information rate $R_A + R_B$ exceeds the joint entropy $H(X^{(A)}, X^{(B)})$ (Slepian and Wolf, 1973).

In the weather example, each transmitter must transmit at a rate greater than $H_2(0.02) = 0.14$ bits, and the total rate $R_A + R_B$ must be greater than 1.14 bits, for example $R_A = 0.6$, $R_B = 0.6$. Both strings can be conveyed without error even though $R_A < H(X^{(A)})$ and $R_B < H(X^{(B)})$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 15.18</span><span class="math-callout__name">(Multiple Access Channels)</span></p>

Consider a channel with two sets of inputs and one output -- for example, a shared telephone line. A simple model system has two binary inputs $x^{(A)}$ and $x^{(B)}$ and a ternary output $y$ equal to the arithmetic sum of the two inputs, that's $0$, $1$ or $2$. There is no noise. Users $A$ and $B$ cannot communicate with each other, and they cannot hear the output of the channel.

How should users $A$ and $B$ use this channel so that their messages can be deduced from the received signals? How fast can $A$ and $B$ communicate?

Clearly the total information rate from $A$ and $B$ to the receiver cannot be two bits. On the other hand it is easy to achieve a total information rate $R_A + R_B$ of one bit. Can reliable communication be achieved at rates $(R_A, R_B)$ such that $R_A + R_B > 1$?

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 15.19</span><span class="math-callout__name">(Broadcast Channels)</span></p>

A broadcast channel consists of a single transmitter and two or more receivers. The properties of the channel are defined by a conditional distribution $Q(y^{(A)}, y^{(B)} \mid x)$. The task is to add an encoder and two decoders to enable reliable communication of a common message at rate $R_0$ to both receivers, an individual message at rate $R_A$ to receiver $A$, and an individual message at rate $R_B$ to receiver $B$. The **capacity region** of the broadcast channel is the convex hull of the set of achievable rate triplets $(R_0, R_A, R_B)$.

An example of a broadcast channel consists of two binary symmetric channels with a common input. The two halves of the channel have flip probabilities $f_A$ and $f_B$, where $f_A < f_B < 1/2$, so $A$ has the better half-channel. A closely related channel is a 'degraded' broadcast channel, in which the conditional probabilities have the structure of a Markov chain $x \to y^{(A)} \to y^{(B)}$. In this case, whatever information gets through to receiver $B$ can also be recovered by receiver $A$, so the task reduces to finding the capacity region for the rate pair $(R_0, R_A)$, where $R_0$ is the rate of information reaching both $A$ and $B$, and $R_A$ is the rate of the extra information reaching $A$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 15.20</span><span class="math-callout__name">(Variable-Rate Error-Correcting Codes for Channels with Unknown Noise Level)</span></p>

In real life, channels may not be well characterized before the encoder is installed. As a model of this situation, imagine that a channel is known to be a binary symmetric channel with noise level either $f_A$ or $f_B$. Let $f_B > f_A$, and let the two capacities be $C_A$ and $C_B$.

A conservative approach would design the encoding system for the worst-case scenario, installing a code with rate $R_B \simeq C_B$. Is it possible to create a system that not only transmits reliably at some rate $R_0$ whatever the noise level, but also communicates some extra, 'lower-priority' bits if the noise level is low? Such a code communicates the high-priority bits reliably at all noise levels between $f_A$ and $f_B$, and communicates the low-priority bits also if the noise level is $f_A$ or below.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 15.21</span><span class="math-callout__name">(Multiterminal Information Networks)</span></p>

Consider the following example of a two-way binary channel: two people both wish to talk over the channel, and they both want to hear what the other person is saying; but you can hear the signal transmitted by the other person only if you are transmitting a zero.

What simultaneous information rates from $A$ to $B$ and from $B$ to $A$ can be achieved, and how? Obviously, we can achieve rates of $1/2$ in both directions by simple time-sharing. But can the two information rates be made larger?

Finding the capacity of a general two-way channel is still an open problem. However, for the simple binary channel described above, there exist codes that can achieve rates up to a boundary beyond the 'obviously achievable' time-sharing region.

</div>

## Chapter 16 -- Message Passing

One of the themes of this book is the idea of doing complicated calculations using simple distributed hardware. It turns out that quite a few interesting problems can be solved by *message-passing* algorithms, in which simple messages are passed locally among simple processors whose operations lead, after some time, to the solution of a global problem.

### 16.1 Counting

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Counting Soldiers in a Line)</span></p>

Consider a line of soldiers walking in the mist. The commander wishes to count them. A centralized solution requires global communication hardware and the ability to listen to all incoming messages. A message-passing solution requires only *local* communication: each soldier can communicate single integers with the two adjacent soldiers in the line, and can add one to a number. Each soldier follows these rules:

1. If you are the front soldier in the line, say the number 'one' to the soldier behind you.
2. If you are the rearmost soldier in the line, say the number 'one' to the soldier in front of you.
3. If a soldier ahead of or behind you says a number to you, add one to it, and say the new number to the soldier on the other side.

The commander can find the global number of soldiers by adding: the number said by the soldier in front (total number in front) + the number said by the soldier behind (number behind) + one (himself).

</div>

#### Separation

This trick exploits a profound property of the total number of soldiers: it can be written as the sum of the number of soldiers *in front of* a point and the number *behind* that point -- two quantities that can be computed *separately*, because the two groups are separated by the commander.

If the soldiers were not arranged in a line but in a *swarm* (a graph containing cycles), the above message-passing rule-set would not work, because it is not possible for a soldier in a cycle to separate the group into 'those in front' and 'those behind'.

A swarm of soldiers *can* be counted by a modified message-passing algorithm *if they are arranged in a graph that contains no cycles* -- i.e., a **tree** (cycle-free graph).

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Message-Passing Rule-Set B: Counting on a Tree)</span></p>

1. Count your number of neighbours, $N$.
2. Keep count of the number of messages you have received from your neighbours, $m$, and of the values $v_1, v_2, \ldots, v_N$ of each of those messages. Let $V$ be the running total of the messages you have received.
3. If the number of messages you have received, $m$, is equal to $N - 1$, then identify the neighbour who has not sent you a message and tell them the number $V + 1$.
4. If the number of messages you have received is equal to $N$, then:
   - (a) the number $V + 1$ is the required total.
   - (b) for each neighbour $n$: say to neighbour $n$ the number $V + 1 - v_n$.

</div>

### 16.2 Path-Counting

A more profound task is counting the number of paths through a grid, and finding how many paths pass through any given point. Consider a rectangular grid connecting points A and B. A valid path starts from A and proceeds to B by rightward and downward moves only.

#### The Sum--Product Algorithm (Forward Pass)

The computational breakthrough is to realize that to find the *number* of paths, we do not have to enumerate all paths explicitly. Pick a point P in the grid. Every path from A to P must come in to P through one of its upstream neighbours ('upstream' meaning above or to the left). So the number of paths from A to P can be found by adding up the number of paths from A to each of those neighbours.

We start by sending the '1' message from A. When any node has received messages from all its upstream neighbours, it sends the *sum* of them on to its downstream neighbours. At B, the total number of paths emerges. This message-passing algorithm is an example of the **sum--product algorithm**.

#### Probability of Passing through a Node

By making a **backward pass** as well as the forward pass, we can deduce how many of the paths go through each node; dividing by the total number of paths gives the probability that a randomly selected path passes through that node. By multiplying the forward-passing and backward-passing messages at a given vertex, we find the total number of paths passing through that vertex.

#### Random Path Sampling

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 16.1</span></p>

If one creates a 'random' path from A to B by flipping a fair coin at every junction where there is a choice of two directions, is the resulting path a uniform random sample from the set of all paths? [Hint: imagine trying it for a small grid.]

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 16.2</span></p>

Having run the forward and backward algorithms between points A and B on a grid, how can one draw one path from A to B *uniformly* at random?

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Solution to Exercise 16.2)</span></p>

To make a uniform random walk, each forward step of the walk should be chosen using a different biased coin at each junction, with the biases chosen in proportion to the *backward* messages emanating from the two options. For example, at the first choice after leaving A, if there is a '3' message coming from the East and a '2' coming from South, one should go East with probability $3/5$ and South with probability $2/5$.

</div>

### 16.3 Finding the Lowest-Cost Path

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Min--Sum Algorithm / Viterbi Algorithm)</span></p>

Given a directed acyclic graph with costs on edges, the **min--sum algorithm** (also called the **Viterbi algorithm**) finds the lowest-cost path from a source node A to a destination node B.

For each node $x$, define 'the cost of $x$' as the cost of the lowest-cost path from A to $x$. Each node can broadcast its cost to its descendants once it knows the costs of all its possible predecessors. The cost of A is zero. As the message passes along each edge, the cost of that edge is *added*. When competing messages arrive at a node, the algorithm sets the cost equal to the **minimum** of the options (the 'min'), and records which was the smallest-cost route into that node by retaining only the edge that achieved the minimum and pruning away the others.

We recover the lowest-cost path by backtracking from B, following the trail of surviving edges back to A.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Other Applications of the Min--Sum Algorithm)</span></p>

The critical path of a set of operations (the subset of operations holding up production) can be found using the min--sum algorithm. In Chapter 25, the min--sum algorithm will be used in the decoding of error-correcting codes.

</div>

### 16.4 Summary and Related Ideas

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Separability and Message Passing)</span></p>

Some global functions have a **separability property**. For example, the number of paths from A to P separates into the sum of the number of paths from A to M (the point to P's left) and the number of paths from A to N (the point above P). Such functions can be computed efficiently by message-passing. Other functions do *not* have such separability properties, for example:
1. the number of pairs of soldiers in a troop who share the same birthday;
2. the size of the largest group of soldiers who share a common height;
3. the length of the shortest tour that a travelling salesman could take that visits every soldier in a troop.

One of the challenges of machine learning is to find low-cost solutions to problems like these.

</div>

### 16.5 Further Exercises

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 16.3</span></p>

Describe the asymptotic properties of the probabilities of passing through each node, for a grid in a triangle of width and height $N$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 16.4</span></p>

In image processing, the **integral image** $I(x, y)$ obtained from an image $f(x, y)$ (where $x$ and $y$ are pixel coordinates) is defined by

$$I(x, y) \equiv \sum_{u=0}^{x} \sum_{v=0}^{y} f(u, v).$$

Show that the integral image $I(x, y)$ can be efficiently computed by message passing.

Show that, from the integral image, some simple functions of the image can be obtained. For example, give an expression for the sum of the image intensities $f(x, y)$ for all $(x, y)$ in a rectangular region extending from $(x_1, y_1)$ to $(x_2, y_2)$.

</div>

## Chapter 17 -- Communication over Constrained Noiseless Channels

This chapter studies the task of communicating efficiently over a **constrained noiseless channel** -- a constrained channel over which not all strings from the input alphabet may be transmitted. It makes use of the idea from Chapter 16 that *global properties of graphs can be computed by a local message-passing algorithm*.

### 17.1 Three Examples of Constrained Binary Channels

A constrained channel can be defined by rules that define which strings are permitted.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Channel A)</span></p>

In **Channel A** every 1 must be followed by at least one 0. The substring $11$ is forbidden. A valid string: $0010010100101000010$. A motivation: a channel where 1s represent pulses of electromagnetic energy, and the device requires a recovery time of one clock cycle after generating a pulse.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Channel B)</span></p>

**Channel B** has the rule that all 1s must come in groups of two or more, and all 0s must come in groups of two or more. The substrings $101$ and $010$ are forbidden. A valid string: $0011100111000011$. A motivation: a disk drive where isolated magnetic domains surrounded by domains of opposite orientation are unstable.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Channel C)</span></p>

**Channel C** has the rule that the largest permitted runlength is two -- each symbol can be repeated at most once. The substrings $111$ and $000$ are forbidden. A valid string: $1001001101100110100$. A motivation: a disk drive where the rate of rotation is not known accurately, making it difficult to distinguish between runs of two and three identical symbols.

</div>

All three are examples of **runlength-limited channels**. Their runlength constraints are:

| Channel | Runlength of 1s (min, max) | Runlength of 0s (min, max) |
|---|---|---|
| unconstrained | 1, $\infty$ | 1, $\infty$ |
| A | 1, 1 | 1, $\infty$ |
| B | 2, $\infty$ | 2, $\infty$ |
| C | 1, 2 | 1, 2 |

#### Some Codes for a Constrained Channel

Concentrating on Channel A (runs of 1s restricted to length one):

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Codes for Channel A)</span></p>

**Code $C_1$** is a $(2, 1)$ code that maps each source bit into two transmitted bits: $0 \to 00$, $1 \to 10$. This is a rate-$1/2$ code, so the capacity of Channel A is at least 0.5.

**Code $C_2$** is a variable-length code: $0 \to 0$, $1 \to 10$. If source symbols are used with equal frequency, the average transmitted length per source bit is $L = \frac{1}{2} \cdot 1 + \frac{1}{2} \cdot 2 = \frac{3}{2}$, so the average communication rate is $R = 2/3$, and the capacity of Channel A is at least $2/3$.

</div>

Starting from code $C_2$, we can reduce the average message length by decreasing the fraction of 1s that we transmit. Feeding into $C_2$ a stream of bits in which the frequency of 1s is $f$, the information rate $R$ achieved is the entropy of the source, $H_2(f)$, divided by the mean transmitted length,

$$L(f) = (1 - f) + 2f = 1 + f.$$

Thus

$$R(f) = \frac{H_2(f)}{L(f)} = \frac{H_2(f)}{1 + f}.$$

This has a maximum of about $0.69$ bits per channel use at $f \simeq 0.38$. By this argument, the capacity of Channel A is at least $\max_f R(f) = 0.69$.

### 17.2 The Capacity of a Constrained Noiseless Channel

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Capacity of a Constrained Noiseless Channel)</span></p>

We defined the capacity of a noisy channel in terms of the mutual information between its input and output, then proved that the capacity was related to the number of distinguishable messages $S(N)$ that could be reliably conveyed in $N$ uses of the channel:

$$C = \lim_{N \to \infty} \frac{1}{N} \log S(N).$$

For the constrained noiseless channel, we adopt this identity as our definition of capacity. We denote the number of distinguishable messages of length $N$ by $M_N$, and define:

$$C = \lim_{N \to \infty} \frac{1}{N} \log M_N.$$

</div>

### 17.3 Counting the Number of Possible Messages

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(State Diagram and Trellis)</span></p>

A **state diagram** represents the states of the transmitter as circles labelled with the name of the state. Directed edges from one state to another indicate that the transmitter is permitted to move from the first state to the second, and a label on that edge indicates the symbol emitted when that transition is made.

A **trellis section** shows two successive states in time at two successive horizontal locations. The state of the transmitter at time $n$ is called $s_n$. The set of possible state sequences can be represented by a **trellis**. A valid sequence corresponds to a path through the trellis, and the number of valid sequences is the number of paths.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Connection Matrix)</span></p>

The **connection matrix** $\mathbf{A}$ summarizes the trellis section: $A_{ss'} = 1$ if there is an edge from state $s$ to $s'$, and $A_{ss'} = 0$ otherwise. For Channel A with states $\{0, 1\}$:

$$\mathbf{A} = \begin{bmatrix} 1 & 1 \\ 1 & 0 \end{bmatrix}.$$

</div>

The number of paths through the trellis can be counted by message-passing. The count of the number of paths is a vector $\mathbf{c}^{(n)}$; we can obtain $\mathbf{c}^{(n+1)}$ from $\mathbf{c}^{(n)}$ using:

$$\mathbf{c}^{(n+1)} = \mathbf{A} \mathbf{c}^{(n)}.$$

So

$$\mathbf{c}^{(N)} = \mathbf{A}^N \mathbf{c}^{(0)},$$

where $\mathbf{c}^{(0)}$ is the initial state count. The total number of paths is $M_n = \sum_s c_s^{(n)} = \mathbf{c}^{(n)} \cdot \mathbf{n}$.

For Channel A, the sequence $M_n$ is the **Fibonacci series**: $2, 3, 5, 8, 13, 21, 34, 55, 89, \ldots$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Capacity from the Connection Matrix)</span></p>

In the limit, $\mathbf{c}^{(N)}$ becomes dominated by the principal right-eigenvector of $\mathbf{A}$:

$$\mathbf{c}^{(N)} \to \text{constant} \cdot \lambda_1^N \mathbf{e}_{\text{R}}^{(0)},$$

where $\lambda_1$ is the principal eigenvalue of $\mathbf{A}$. Thus, to within a constant factor, $M_N \sim \lambda_1^N$ as $N \to \infty$, and the capacity of any constrained channel is:

$$C = \log_2 \lambda_1.$$

For Channel A, the ratio of successive Fibonacci terms tends to the golden ratio $\gamma = \frac{1 + \sqrt{5}}{2} = 1.618$, so $C = \log_2 1.618 = 0.694$ bits per channel use.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Channels A, B, and C Have Equal Capacity)</span></p>

Comparing the trellises for Channels A, B, and C, they all have $M_N$ following the same Fibonacci-like growth. The principal eigenvalues of the three connection matrices are the same, so all three channels have the same capacity $C = 0.694$ bits per channel use. Indeed the channels are intimately related.

</div>

### 17.4 Back to Our Model Channels

#### Equivalence of Channels A and B

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Accumulator and Binary Differentiator)</span></p>

If we take any valid string $\mathbf{s}$ for Channel A and pass it through an **accumulator**:

$$t_1 = s_1, \qquad t_n = t_{n-1} + s_n \bmod 2 \quad \text{for } n \ge 2,$$

then the resulting string $\mathbf{t}$ is a valid string for Channel B, because there are no $11$s in $\mathbf{s}$, so there are no isolated digits in $\mathbf{t}$.

The accumulator is invertible; its inverse is the **binary differentiator**:

$$s_1 = t_1, \qquad s_n = t_n - t_{n-1} \bmod 2 \quad \text{for } n \ge 2.$$

Since $+$ and $-$ are equivalent in modulo 2 arithmetic, the differentiator is also a blurrer, convolving the source stream with the filter $(1, 1)$. Channel C is also intimately related to Channels A and B.

</div>

### 17.5 Practical Communication over Constrained Channels

Since all three channels are equivalent, we concentrate on Channel A.

#### Fixed-Length Solutions

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Fixed-Length Codes for Channel A)</span></p>

A runlength-limited code with $N = 5$ transmitted bits achieves a rate of $3/5 = 0.6$: there are 8 valid strings of length 5 ending in the zero state, so we can map 3 source bits (8 messages) to 5 transmitted bits.

With $N = 8$ transmitted bits, there are 34 valid strings ending in the zero state, so we can map 5 source bits (32 messages) to 8 transmitted bits and achieve rate $5/8 = 0.625$.

With $N = 16$ transmitted bits, the largest integer number of source bits that can be encoded is 10, so the maximum rate of a fixed-length code with $N = 16$ is $0.625$.

</div>

#### Optimal Variable-Length Solution

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Optimal Transition Probabilities)</span></p>

The optimal way to convey information over the constrained channel is to find the optimal transition probabilities for all points in the trellis, $Q_{s'|s}$, and make transitions with these probabilities. The optimal transition probabilities $\mathbf{Q}$ can be found as follows.

Find the principal right- and left-eigenvectors of $\mathbf{A}$, that is the solutions of $\mathbf{A}\mathbf{e}^{(R)} = \lambda \mathbf{e}^{(R)}$ and $\mathbf{e}^{(L)\mathsf{T}} \mathbf{A} = \lambda \mathbf{e}^{(L)\mathsf{T}}$ with largest eigenvalue $\lambda$. Then construct a matrix $\mathbf{Q}$ whose invariant distribution is proportional to $e_i^{(R)} e_i^{(L)}$, namely

$$Q_{s'|s} = \frac{e_{s'}^{(L)} A_{s's}}{\lambda e_s^{(L)}}.$$

When sequences are generated using this optimal transition probability matrix, the entropy of the resulting sequence is asymptotically $\log_2 \lambda$ per symbol.

</div>

### 17.6 Variable Symbol Durations

We can add a further frill to the task of communicating over constrained channels by assuming that the symbols we send have different *durations*, and that our aim is to communicate at the maximum possible rate per unit time.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Optimal Symbol Frequencies for Variable-Duration Channels)</span></p>

For an unconstrained noiseless channel whose symbols have durations $l_i$ (in some units of time), the optimal probability with which those symbols should be used is

$$p_i^* = 2^{-\beta l_i},$$

where $\beta$ is the capacity of the channel in bits per unit time. This is analogous to optimal symbol codes: when making a binary symbol code for a source with unequal probabilities $p_i$, the optimal message lengths are $l_i^* = \log_2 1/p_i$, so $p_i = 2^{-l_i^*}$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 17.13</span><span class="math-callout__name">(The Morse Channel)</span></p>

A classic example of a constrained channel with variable symbol durations is the 'Morse' channel, whose symbols are: the dot (d), the dash (D), the short space (s, used between letters in morse code), and the long space (S, used between words). The constraints are that spaces may only be followed by dots and dashes.

Find the capacity of this channel in bits per unit time assuming (a) that all four symbols have equal durations; or (b) that the symbol durations are 2, 4, 3 and 6 time units respectively.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 17.15</span><span class="math-callout__name">(How Difficult Is It to Get DNA into a Narrow Tube?)</span></p>

To an information theorist, the entropy associated with a constrained channel reveals how much information can be conveyed over it. In statistical physics, the same calculations are done for a different reason: to predict the thermodynamics of polymers.

Consider a polymer of length $N$ that can either sit in a constraining tube of width $L$, or in the open. In the open, the polymer adopts a state drawn at random from the set of one-dimensional random walks, with 3 possible directions per step, giving an entropy of $\log 3$ per step (total of $N \log 3$). In the tube, the polymer's walk can go in 3 directions unless the wall is in the way.

The *change* in entropy associated with the polymer entering the tube determines the force required to pull the DNA into the tube. The difference in capacity between two channels, one constrained and one unconstrained, is directly proportional to this force.

</div>

## Chapter 18 -- Crosswords and Codebreaking

This chapter is a random walk through topics related to language modelling, including the information-theoretic analysis of crossword puzzles, simple language models, units of information content, and the Banburismus method used for codebreaking at Bletchley Park.

### 18.1 Crosswords

Crossword-making can be thought of as defining a **constrained channel**. The fact that *many* valid crosswords can be made demonstrates that this constrained channel has a capacity greater than zero.

There are two archetypal crossword formats:
- **Type A (American):** Every row and column consists of a succession of words of length 2 or more separated by one or more spaces. Every letter lies in both a horizontal *and* a vertical word.
- **Type B (British):** Each row and column is a mixture of words and single characters, separated by one or more spaces. Only about half the letters lie in two words; the rest lie in one word only.

Type A crosswords are harder to *create* than type B because no single characters are permitted. Type B crosswords are generally harder to *solve* because there are fewer constraints per character.

#### Why Are Crosswords Possible?

If a language has no redundancy, then any letters written on a grid form a valid crossword. In a language with high redundancy, it is hard to make crosswords (except perhaps a small number of trivial ones). The possibility of making crosswords demonstrates a **bound on the redundancy** of the language.

Crosswords are written in 'word-English', the language consisting of strings of words from a dictionary, separated by spaces. We model word-English by Wenglish, using $W$ words all of length $L$. The entropy per character is:

$$H_W = \frac{\log_2 W}{L + 1}.$$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Counting Crosswords)</span></p>

Consider a large crossword of size $S$ squares. Let $f_w S$ be the number of words and $f_1 S$ be the number of letter-occupied squares. For typical crosswords of types A and B made of words of length $L$, the fractions $f_w$ and $f_1$ are:

| | A | B |
|---|---|---|
| $f_w$ | $\frac{2}{L+1}$ | $\frac{1}{L+1}$ |
| $f_1$ | $\frac{L}{L+1}$ | $\frac{3}{4} \frac{L}{L+1}$ |

The total number of *typical* fillings-in of the $f_1 S$ squares is $\lvert T \rvert = 2^{f_1 S H_0}$. The probability that one word of length $L$ is validly filled-in is $\beta = W / 2^{L H_0}$, and the probability the whole crossword is valid is approximately $\beta^{f_w S}$.

Arbitrarily many crosswords can be made only if there are enough words in the dictionary, that is:

$$H_W > \frac{(f_w L - f_1)}{f_w(L+1)} H_0.$$

For type A: $H_W > \frac{1}{2}\frac{L}{L+1} H_0$. For type B: $H_W > \frac{1}{4}\frac{L}{L+1} H_0$.

</div>

Setting $H_0 = 4.2$ bits and assuming $W = 4000$ words with length $L = 5$, the condition for type B crosswords is satisfied, but the condition for type A is only *just* satisfied. This fits with the experience that type A crosswords usually contain more obscure words.

### 18.2 Simple Language Models

#### The Zipf--Mandelbrot Distribution

The crudest model for a language is the **monogram model**, which asserts that each successive word is drawn independently from a distribution over words.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Zipf's Law)</span></p>

**Zipf's law** (Zipf, 1949) asserts that the probability of the $r$th most probable word in a language is approximately

$$P(r) = \frac{\kappa}{r^\alpha},$$

where the exponent $\alpha$ has a value close to 1 and $\kappa$ is a normalizing constant. A log--log plot of frequency versus word-rank should show a straight line with slope $-\alpha$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Zipf--Mandelbrot Distribution)</span></p>

Mandelbrot's (1982) modification of Zipf's law introduces a third parameter $v$:

$$P(r) = \frac{\kappa}{(r + v)^\alpha}.$$

This distribution fits certain documents (such as Jane Austen's *Emma*, with fitted parameters $\kappa = 0.56$, $v = 8.0$, $\alpha = 1.26$) well, but other documents show deviations from this model.

</div>

#### The Dirichlet Process

One difficulty in modelling a language is the **unboundedness of vocabulary**: the greater the sample of language, the greater the number of words encountered. A generative model for a language should satisfy two properties:
1. It must assign non-zero probability to *words never used before*.
2. It must satisfy **exchangeability**: the probability of finding a particular word should be the same everywhere in the stream.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Dirichlet Process)</span></p>

The **Dirichlet process** is a model for a stream of symbols that satisfies the exchangeability rule and allows the vocabulary to grow without limit. The model has one parameter $\alpha$. After seeing a stream of $F$ symbols with counts $\lbrace F_w \rbrace$:

- The probability that the next symbol is a **new** symbol, never seen before:

$$\frac{\alpha}{F + \alpha}.$$

- The probability that the next symbol is **symbol $w$**:

$$\frac{F_w}{F + \alpha}.$$

</div>

A Dirichlet process is not an adequate model for observed distributions that roughly obey Zipf's law. However, with a small tweak -- generating characters from a Dirichlet process with a small $\alpha$ (so about 27 frequent symbols), declaring one character to be a space, and identifying the strings between spaces as 'words' -- the resulting word frequencies produce nice Zipf plots.

### 18.3 Units of Information Content

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Information Content)</span></p>

The **information content** of an outcome $x$ with probability $P(x)$ is:

$$h(x) = \log \frac{1}{P(x)}.$$

The **entropy** of an ensemble is an average information content:

$$H(X) = \sum_x P(x) \log \frac{1}{P(x)}.$$

</div>

When comparing hypotheses in the light of data, we evaluate the **log evidence** for a hypothesis $\mathcal{H}_i$:

$$\text{`log evidence for } \mathcal{H}_i\text{'} = \log P(D \mid \mathcal{H}_i),$$

or, when comparing two hypotheses, the **log odds**:

$$\log \frac{P(D \mid \mathcal{H}_1)}{P(D \mid \mathcal{H}_2)}.$$

The log evidence for a hypothesis is the negative of the information content of the data $D$ given that hypothesis: if the data have large information content under a hypothesis, they are surprising to it; if some other hypothesis is not so surprised, that hypothesis becomes more probable. 'Information content', 'surprise value', and log likelihood or log evidence are the same thing.

The units depend on the choice of the base of the logarithm:

| Unit | Expression |
|---|---|
| bit | $\log_2 p$ |
| nat | $\log_e p$ |
| ban | $\log_{10} p$ |
| deciban (db) | $10 \log_{10} p$ |

The **bit** (also called the **shannon**) is the most common unit. A **byte** is 8 bits. The most interesting units historically are the **ban** and the **deciban**.

#### The History of the Ban

The term "ban" originates from codebreaking at Bletchley Park during World War II. When Alan Turing and the other codebreakers were breaking the daily Enigma code, their task was a huge inference problem: infer the day's settings of the Enigma machines from the cyphertexts. These inferences were conducted using Bayesian methods, and the chosen units were decibans or half-decibans -- the deciban being judged the smallest weight of evidence discernible to a human. The evidence was tallied using sheets printed in **Banbury**, so the inference task was known as **Banburismus**, and the units were called bans.

### 18.4 A Taste of Banburismus

The Enigma machine had about $8 \times 10^{12}$ possible settings. To deduce the full state required about 129 decibans of evidence. Banburismus was aimed not at deducing the entire state but at figuring out which wheels were in use; the logic-based bombes then cracked the wheel settings.

#### How to Detect That Two Messages Came from Machines with a Common State Sequence

The hypotheses are:
- $\mathcal{H}_0$: The machines are in *different* states and the two plain messages are unrelated.
- $\mathcal{H}_1$: The machines are in the *same* state and the two plain messages are unrelated.

The data are two cyphertexts $\mathbf{x}$ and $\mathbf{y}$ of length $T$ over an alphabet of size $A$ (26 in Enigma).

Under $\mathcal{H}_0$ (different states), each machine uses an unrelated time-varying permutation, so all cyphertexts are equally likely:

$$P(\mathbf{x}, \mathbf{y} \mid \mathcal{H}_0) = \left(\frac{1}{A}\right)^{2T}.$$

Under $\mathcal{H}_1$ (same state), a *single* time-varying permutation $c_t$ underlies both messages.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Match Probability)</span></p>

Assuming the source language is monogram-English with letter probabilities $\lbrace p_i \rbrace$, the **match probability** $m$ is:

$$m \equiv \sum_i p_i^2.$$

For both English and German, $m$ is about $2/26 \approx 0.076$, compared to $1/26 \approx 0.038$ for a completely random language.

</div>

Assuming $c_t$ is an ideal random permutation, the probability that two synchronized ciphertext characters are identical is:

$$P(x_t, y_t \mid \mathcal{H}_1) = \begin{cases} \frac{m}{A} & \text{if } x_t = y_t \\ \frac{(1-m)}{A(A-1)} & \text{if } x_t \neq y_t. \end{cases}$$

Given a pair of cyphertexts of length $T$ that match in $M$ places and do not match in $N$ places, the log evidence in favour of $\mathcal{H}_1$ is:

$$\log \frac{P(\mathbf{x}, \mathbf{y} \mid \mathcal{H}_1)}{P(\mathbf{x}, \mathbf{y} \mid \mathcal{H}_0)} = M \log mA + N \log \frac{(1-m)A}{A - 1}.$$

Every match contributes $\log mA$ in favour of $\mathcal{H}_1$; every non-match contributes $\log \frac{(1-m)A}{A-1}$ in favour of $\mathcal{H}_0$.

| Quantity | Formula | Value |
|---|---|---|
| Match probability (monogram-English) | $m$ | 0.076 |
| Coincidental match probability | $1/A$ | 0.037 |
| log-evidence for $\mathcal{H}_1$ per match | $10 \log_{10} mA$ | 3.1 db |
| log-evidence for $\mathcal{H}_0$ per non-match | $10 \log_{10} \frac{(1-m)A}{A-1}$ | $-0.18$ db |

For example, $M = 4$ matches and $N = 47$ non-matches in a pair of length $T = 51$ gives a weight of evidence of $+4$ decibans in favour of $\mathcal{H}_1$, or a likelihood ratio of about 2.5 to 1. The expected weight of evidence from a line of text of $T = 20$ characters is about 1.4 decibans if $\mathcal{H}_1$ is true, and $-1.1$ decibans if $\mathcal{H}_0$ is true. Roughly 400 characters need to be inspected to accumulate 20 decibans (a hundred to one) in favour of either hypothesis.

Furthermore, because consecutive characters in English are not independent and the bigram and trigram statistics are nonuniform, matches tend to occur in **bursts of consecutive matches**. Using better language models, the evidence contributed by runs of matches was more accurately computed by Turing and refined by Good.

## Chapter 19 -- Why Have Sex? Information Acquisition and Evolution

Evolution has been happening on Earth for about $10^9$ years. The entire blueprint of all organisms has emerged through natural selection: fitter individuals have more progeny, the fitness being defined by the local environment. The teaching signal is only a few bits per individual -- at most **one bit per offspring**.

The fitness of an organism is largely determined by its DNA. The information content of natural selection is fully contained in a specification of which offspring survived to have children.

### 19.1 The Model

We study a simple model of a reproducing population of $N$ individuals with a genome of size $G$ bits. Variation is produced by mutation or by recombination (sex), and truncation selection selects the $N$ fittest children at each generation.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Fitness Function)</span></p>

The genotype of each individual is a vector $\mathbf{x}$ of $G$ bits, each having a good state $x_g = 1$ and a bad state $x_g = 0$. The **fitness** is the sum of the bits:

$$F(\mathbf{x}) = \sum_{g=1}^{G} x_g.$$

The **normalized fitness** is $f(\mathbf{x}) \equiv F(\mathbf{x}) / G$.

</div>

The essential property is that fitness is locally a roughly **linear function** of the genome -- many possible changes each have a small effect, and these effects combine approximately linearly.

**Variation by mutation.** At each generation $t$, every individual produces two children. The children's genotypes differ from the parent's by random mutations: each bit is independently flipped with probability $m$. Natural selection selects the fittest $N$ progeny (known as **truncation selection**).

**Variation by recombination (sex).** The $N$ individuals are married into $M = N/2$ couples at random. Each couple has $C = 4$ children. Each child obtains its genotype $\mathbf{z}$ by random crossover of the parents' genotypes $\mathbf{x}$ and $\mathbf{y}$:

$$z_g = \begin{cases} x_g & \text{with probability } 1/2 \\ y_g & \text{with probability } 1/2. \end{cases}$$

The fittest $N$ progeny are then selected, and a new generation starts.

### 19.2 Rate of Increase of Fitness

#### Theory of Mutations

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Rate of Fitness Increase under Mutation)</span></p>

Assume an individual with normalized fitness $f = F/G$ is subject to mutations that flip bits with probability $m$. If the average normalized fitness $f > 1/2$, define the **excess normalized fitness** $\delta f \equiv f - 1/2$. Then:

The child's excess fitness has mean $(1 - 2m)\delta f$ and variance $\simeq m/G$. The mean fitness and variance at the next generation evolve as:

$$\delta f(t+1) = (1 - 2m)\delta f(t) + \alpha\sqrt{(1+\beta)} \sqrt{\frac{m}{G}},$$

$$\sigma^2(t+1) = \gamma(1+\beta)\frac{m}{G},$$

where $\alpha = \sqrt{2/\pi} \simeq 0.8$ is the mean deviation from the mean (in standard deviations) and $\gamma = (1 - 2/\pi) \simeq 0.36$ is the factor by which selection reduces the variance. For the Gaussian case, the factor $\alpha\sqrt{1+\beta} = 1$.

The rate of increase of normalized fitness is:

$$\frac{\mathrm{d}f}{\mathrm{d}t} \simeq -2m\,\delta f + \sqrt{\frac{m}{G}},$$

which is maximized for

$$m_{\text{opt}} = \frac{1}{16G(\delta f)^2},$$

yielding an optimal rate:

$$\left(\frac{\mathrm{d}f}{\mathrm{d}t}\right)_{\text{opt}} = \frac{1}{8G(\delta f)}.$$

The rate of increase of fitness $F = fG$ is at most

$$\frac{\mathrm{d}F}{\mathrm{d}t} = \frac{1}{8(\delta f)} \text{ per generation.}$$

</div>

For $\delta f > 0.125$, the rate of increase of fitness is smaller than one per generation. As fitness approaches $G$, the optimal mutation rate tends to $m = 1/(4G)$, and information is gained at about 0.5 bits per generation. It takes about $2G$ generations for all individuals to attain perfection.

#### Theory of Sex

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Rate of Fitness Increase under Sex)</span></p>

Under the assumptions that the gene-pool mixes sufficiently rapidly (so correlations between genes can be neglected) and **homogeneity** (the fraction $f_g$ of good bits is the same $f(t)$ for all $g$):

If two parents of fitness $F = fG$ mate, the variation produced by sex does not reduce the average fitness. The standard deviation of children's fitness scales as $\sqrt{Gf(1-f)}$. Since the increase in fitness after selection is proportional to this standard deviation, the **fitness increase per generation scales as $\sqrt{G}$**:

$$\frac{\mathrm{d}\bar{F}}{\mathrm{d}t} \simeq \eta\sqrt{f(t)(1-f(t))G},$$

where $\eta \equiv \sqrt{2/(\pi + 2)} \simeq 0.62$.

The solution of this differential equation is:

$$f(t) = \frac{1}{2}\left[1 + \sin\!\left(\frac{\eta}{\sqrt{G}}(t + c)\right)\right],$$

where $c = \sin^{-1}(2f(0) - 1)$. The system reaches $f = 1$ in finite time: $(\pi/\eta)\sqrt{G}$ generations.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why Sex Is Better Than No Sex)</span></p>

The key difference between the two models:
- **Mutation only:** Mutations *reduce* average fitness of children below that of parents. Selection then bumps it back up. The net gain is at most of order 1 bit per generation, regardless of genome size.
- **Sex (recombination):** Sex produces variation *without decreasing* average fitness. The typical amount of variation scales as $\sqrt{G}$, so after selection the average fitness rises by $O(\sqrt{G})$ per generation.

</div>

### 19.3 The Maximal Tolerable Mutation Rate

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Maximal Tolerable Mutation Rate)</span></p>

Combining the two models of variation (mutation and sex), the rate of increase of fitness is:

$$\frac{\mathrm{d}f}{\mathrm{d}t} \simeq -2m\,\delta f + \eta\sqrt{2}\sqrt{\frac{m + f(1-f)/2}{G}},$$

which is positive if the mutation rate satisfies:

$$m < \eta\sqrt{\frac{f(1-f)}{G}}.$$

Without sex, the maximum tolerable mutation rate is:

$$m < \frac{1}{G} \cdot \frac{1}{(2\,\delta f)^2}.$$

The tolerable mutation rate **with** sex is of order $\sqrt{G}$ times greater than without sex.

</div>

A parthenogenetic (non-sexual) species can try to compensate by increasing litter sizes. If mutation flips on average $mG$ bits, the probability that no bits are flipped is $e^{-mG}$, so a mother needs roughly $e^{mG}$ offspring to have one child with the same fitness.

The maximum tolerable mutation rate is pinned close to $1/G$ for a non-sexual species, whereas it is of order $1/\sqrt{G}$ for a species with recombination. The largest possible genome size for a given mutation rate $m$: for a parthenogenetic species it is of order $1/m$, and for a sexual species it is $1/m^2$. Taking $m = 10^{-8}$ per nucleotide per generation, we predict that **all species with more than $G = 10^9$ coding nucleotides make at least occasional use of recombination**.

### 19.4 Fitness Increase and Information Acquisition

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Information Acquired by a Species)</span></p>

The **information acquired** at an intermediate fitness is defined as the amount of selection (measured in bits) required to select the perfect state from the gene pool. Let a fraction $f_g$ of the population have $x_g = 1$. Then:

$$I = \sum_g \log_2 \frac{f_g}{1/2} \text{ bits.}$$

If all fractions $f_g$ equal $F/G$, then

$$I = G \log_2 \frac{2F}{G},$$

which is well approximated by

$$\tilde{I} \equiv 2(F - G/2).$$

The rate of information acquisition is thus roughly **two times** the rate of increase of fitness in the population.

</div>

### 19.5 Discussion

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Advantage of Sex)</span></p>

A population that reproduces by sex with recombination can acquire information from natural selection at a rate of order $\sqrt{G}$ times faster than a parthenogenetic population, and it can tolerate a mutation rate that is of order $\sqrt{G}$ times greater. For genomes of size $G \simeq 10^8$ coding nucleotides, this factor of $\sqrt{G}$ is substantial.

The 'cost of males' argument -- that a parthenogenetic mother could produce twice as many offspring -- does not outweigh this advantage when the mutation rate is sufficiently high. Simulations show that in a mixed population with both sexual and parthenogenetic individuals (where single mothers produce as many children as a sexual couple), sex dominates when $mG \geq 4$, whereas parthenogens can take over only when the population has reached its top fitness *and* the mutation rate is sufficiently low ($mG \simeq 1$).

In summary: why have sex? Because **sex is good for your bits**.

</div>

## About Part IV

The number of inference problems that can be tackled by Bayesian inference methods is enormous: decoding error-correcting codes, inferring clusters from data, interpolation through noisy data, and classifying patterns given labelled examples. Techniques for solving these problems fall into two categories:

- **Exact methods** compute the required quantities directly. Only a few interesting problems have a direct solution, but exact methods are important as tools for solving subtasks within larger problems.
- **Approximate methods** can be subdivided into:
  1. **Deterministic approximations**, which include maximum likelihood, Laplace's method, and variational methods.
  2. **Monte Carlo methods**, in which random numbers play an integral part.

A widespread misconception is that the aim of inference is to find *the most probable explanation* for some data. While the most probable hypothesis may be of interest, it is just the peak of a probability distribution, and it is the whole distribution that is of interest. The most probable outcome from a source is often not a *typical* outcome from that source.

## Chapter 20 -- An Example Inference Task: Clustering

Human brains are good at finding regularities in data. One way of expressing regularity is to put a set of objects into groups that are similar to each other -- this operation is called **clustering**. In this chapter we discuss ways to take a set of $N$ objects and group them into $K$ clusters.

There are several motivations for clustering:
1. A good clustering has **predictive power** -- cluster labels lead to a more efficient description of data and help us choose better actions. This type of clustering is sometimes called *mixture density modelling*, and the objective function that measures how well the predictive model is working is the information content of the data, $\log 1/P(\lbrace\mathbf{x}\rbrace)$.
2. Clusters can be a useful aid to **communication** because they allow lossy compression. In lossy image compression, we divide the image into $N$ small patches, find a close match to each patch in an alphabet of $K$ image-templates, and send the list of labels. This type of clustering is sometimes called *vector quantization*.
3. Failures of the cluster model may **highlight interesting objects** that deserve special attention.
4. Clustering algorithms may serve as **models of learning processes** in neural systems.

We can formalize a vector quantizer in terms of an *assignment rule* $\mathbf{x} \to k(\mathbf{x})$ for assigning datapoints $\mathbf{x}$ to one of $K$ codenames, and a *reconstruction rule* $k \to \mathbf{m}^{(k)}$, the aim being to minimize the **expected distortion**:

$$D = \sum_{\mathbf{x}} P(\mathbf{x}) \frac{1}{2} \left[\mathbf{m}^{(k(\mathbf{x}))} - \mathbf{x}\right]^2.$$

### 20.1 K-means Clustering

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(K-means Clustering)</span></p>

The K-means algorithm puts $N$ data points in an $I$-dimensional space into $K$ clusters. Each cluster is parameterized by a vector $\mathbf{m}^{(k)}$ called its mean. The data points are denoted by $\lbrace\mathbf{x}^{(n)}\rbrace$ where $n$ runs from 1 to $N$. The distance metric is:

$$d(\mathbf{x}, \mathbf{y}) = \frac{1}{2} \sum_i (x_i - y_i)^2.$$

**Initialization.** Set $K$ means $\lbrace\mathbf{m}^{(k)}\rbrace$ to random values.

**Assignment step.** Each data point $n$ is assigned to the nearest mean. The cluster assignment is:

$$\hat{k}^{(n)} = \operatorname*{argmin}_k \lbrace d(\mathbf{m}^{(k)}, \mathbf{x}^{(n)}) \rbrace.$$

Equivalently, using indicator variables (responsibilities) $r_k^{(n)}$:

$$r_k^{(n)} = \begin{cases} 1 & \text{if } \hat{k}^{(n)} = k \\ 0 & \text{if } \hat{k}^{(n)} \neq k. \end{cases}$$

In case of ties, $\hat{k}^{(n)}$ is set to the smallest of the winning $\lbrace k\rbrace$.

**Update step.** The means are adjusted to match the sample means of the data points they are responsible for:

$$\mathbf{m}^{(k)} = \frac{\sum_n r_k^{(n)} \mathbf{x}^{(n)}}{R^{(k)}},$$

where $R^{(k)} = \sum_n r_k^{(n)}$ is the total responsibility of mean $k$. If $R^{(k)} = 0$, the mean $\mathbf{m}^{(k)}$ is left where it is.

**Repeat** the assignment step and update step until the assignments do not change.

</div>

The K-means algorithm always converges to a fixed point. We can associate an "energy" with the state of the algorithm by connecting a spring between each point $\mathbf{x}^{(n)}$ and the mean that is responsible for it. The total energy of all the springs is a **Lyapunov function** for the algorithm: (a) the assignment step can only decrease the energy, (b) the update step can only decrease the energy, and (c) the energy is bounded below.

However, the outcome of the algorithm depends on the initial condition -- different initializations can lead to different final clusterings.

#### Limitations of K-means

The K-means algorithm has several *ad hoc* features and limitations:

- It takes account only of the distance between the means and the data points; it has **no representation of the weight or breadth** of each cluster. This can cause data points from a broad cluster to be incorrectly assigned to a narrower cluster.
- When data falls into **elongated clusters**, K-means slices them in half rather than finding the natural groupings, because it has no way of representing the shape of a cluster.
- It is a **hard** rather than a **soft** algorithm: points are assigned to exactly one cluster with equal weight. Points near the border between clusters should arguably play a partial role in determining the locations of multiple clusters.

### 20.2 Soft K-means Clustering

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Soft K-means Clustering)</span></p>

The soft K-means algorithm has one parameter, $\beta$, which we could term the **stiffness**.

**Assignment step.** Each data point $\mathbf{x}^{(n)}$ is given a soft "degree of assignment" to each of the means. The degree to which $\mathbf{x}^{(n)}$ is assigned to cluster $k$ is the responsibility $r_k^{(n)}$:

$$r_k^{(n)} = \frac{\exp\bigl(-\beta \, d(\mathbf{m}^{(k)}, \mathbf{x}^{(n)})\bigr)}{\sum_{k'} \exp\bigl(-\beta \, d(\mathbf{m}^{(k')}, \mathbf{x}^{(n)})\bigr)}.$$

The sum of the $K$ responsibilities for the $n$th point is 1.

**Update step.** The means are adjusted to match the sample means of the data points they are responsible for:

$$\mathbf{m}^{(k)} = \frac{\sum_n r_k^{(n)} \mathbf{x}^{(n)}}{R^{(k)}}, \qquad R^{(k)} = \sum_n r_k^{(n)}.$$

</div>

The update step is identical to hard K-means; the only difference is that the responsibilities $r_k^{(n)}$ can take values between 0 and 1. As the stiffness $\beta \to \infty$, the soft K-means algorithm becomes identical to the hard K-means algorithm.

Dimensionally, $\beta$ is an inverse-length-squared, so we can associate a lengthscale $\sigma \equiv 1/\sqrt{\beta}$ with it.

#### Bifurcation Analysis

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Stability of the Soft K-means Fixed Point)</span></p>

Consider $K = 2$ means initialized to $\mathbf{m}^{(1)} = (m, 0)$ and $\mathbf{m}^{(2)} = (-m, 0)$ with data drawn from a distribution $P(x_1)$ having variance $\sigma_1^2$. The fixed point $m = 0$ is **stable** if

$$\sigma_1^2 \leq 1/\beta,$$

and **unstable** otherwise. If $\sigma_1^2 > 1/\beta$, there is a bifurcation and two stable fixed points emerge at approximately:

$$m = \pm \beta^{-1/2} \frac{(\sigma_1^2 \beta - 1)^{1/2}}{\sigma_1^2 \beta}.$$

</div>

### 20.3 Conclusion

Adding one stiffness parameter $\beta$ does not resolve all problems with K-means -- the issues of how to set $\beta$, how to handle elongated clusters, and clusters of unequal weight and width remain. These questions will be addressed later through the mixture-density-modelling view of clustering.

## Chapter 21 -- Exact Inference by Complete Enumeration

We open our toolbox of methods for handling probabilities by discussing a brute-force inference method: complete enumeration of all hypotheses, and evaluation of their probabilities. This approach is an exact method, and the difficulty of carrying it out will motivate the smarter exact and approximate methods introduced in the following chapters.

### 21.1 The Burglar Alarm

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(The Burglar Alarm)</span></p>

Fred lives in Los Angeles and commutes 60 miles to work. Whilst at work, he receives a phone-call from his neighbour saying that Fred's burglar alarm is ringing. What is the probability that there was a burglar in his house today? While driving home to investigate, Fred hears on the radio that there was a small earthquake near his home. What is the probability that there was a burglar in his house?

Introduce variables: $b$ (burglar present), $a$ (alarm ringing), $p$ (phonecall from neighbour), $e$ (earthquake), and $r$ (radio report of earthquake). The joint probability factorizes as:

$$P(b, e, a, p, r) = P(b)P(e)P(a \mid b, e)P(p \mid a)P(r \mid e).$$

With plausible values:
- Burglar probability: $P(b{=}1) = \beta = 0.001$
- Earthquake probability: $P(e{=}1) = \epsilon = 0.001$
- Alarm reliability for burglars: $\alpha_b = 0.99$; for earthquakes: $\alpha_e = 0.01$; false alarm rate: $f = 0.001$
- Alarm probability given $b$ and $e$ follows a "noisy-or" model:

$$P(a{=}1 \mid b, e) = 1 - (1-f)(1-\alpha_b)^b(1-\alpha_e)^e.$$

**First inference (alarm is ringing, $p = 1$):** The posterior is

$$P(b, e \mid a{=}1) = \frac{P(a{=}1 \mid b, e)P(b)P(e)}{P(a{=}1)}.$$

After computing the normalizing constant $P(a{=}1) = 0.002$, we find:

$$P(b{=}0 \mid a{=}1) = 0.505, \quad P(b{=}1 \mid a{=}1) = 0.495.$$

There is nearly a 50% chance a burglar was present.

**Second inference (also learn earthquake happened, $e = 1$):**

$$P(b{=}0 \mid e{=}1, a{=}1) = 0.92, \quad P(b{=}1 \mid e{=}1, a{=}1) = 0.08.$$

There is now only an 8% chance of a burglar. Learning that an earthquake happened -- an alternative explanation for the alarm -- reduces the probability of a burglar.

</div>

#### Explaining Away

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Explaining Away)</span></p>

The phenomenon where one of the possible causes ($b{=}1$) of some data ($a{=}1$) becomes *less* probable when another of the causes ($e{=}1$) becomes more probable, even though the two causes were independent *a priori*, is known as **explaining away**. This is an important feature of correct inferences, and one that any artificial intelligence should replicate.

Note that the variables $b$ and $e$, which were independent a priori, become **dependent** in the posterior distribution (21.5). The posterior is not a separable function of $b$ and $e$.

</div>

### 21.2 Exact Inference for Continuous Hypothesis Spaces

Many of the hypothesis spaces we will consider are naturally thought of as continuous. In any practical computer implementation, such continuous spaces will necessarily be discretized, and so can, in principle, be enumerated.

#### A Two-Parameter Model

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Inferring Gaussian Parameters by Enumeration)</span></p>

The one-dimensional Gaussian distribution is parameterized by a mean $\mu$ and standard deviation $\sigma$:

$$P(x \mid \mu, \sigma) = \frac{1}{\sqrt{2\pi}\sigma} \exp\!\left(-\frac{(x - \mu)^2}{2\sigma^2}\right).$$

Given $N$ data points $x_n$, $n = 1, \ldots, N$, assumed to be drawn independently, we evaluate the posterior probability of each hypothesis $(\mu, \sigma)$ on a discretized grid. The likelihood values can be represented by line thickness in the hypothesis space. Subhypotheses having likelihood smaller than $e^{-8}$ times the maximum likelihood are deleted.

Using a finer grid, the likelihood can be represented as a surface plot or contour plot as a function of $\mu$ and $\sigma$.

</div>

#### A Five-Parameter Mixture Model

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Mixture of Two Gaussians)</span></p>

Consider a mixture of two Gaussians with five parameters $\mu_1, \sigma_1, \mu_2, \sigma_2, \pi_1$ (where $\pi_1 + \pi_2 = 1$, $\pi_i \geq 0$):

$$P(x \mid \mu_1, \sigma_1, \pi_1, \mu_2, \sigma_2, \pi_2) = \frac{\pi_1}{\sqrt{2\pi}\sigma_1}\exp\!\left(-\frac{(x-\mu_1)^2}{2\sigma_1^2}\right) + \frac{\pi_2}{\sqrt{2\pi}\sigma_2}\exp\!\left(-\frac{(x-\mu_2)^2}{2\sigma_2^2}\right).$$

To compare the one-Gaussian model with the mixture-of-two model, we evaluate the **marginal likelihood** (evidence) for each model $\mathcal{H}$:

$$P(\lbrace x\rbrace \mid \mathcal{H}) = \sum_{\boldsymbol{\theta}} P(\boldsymbol{\theta}) P(\lbrace x\rbrace \mid \boldsymbol{\theta}, \mathcal{H}),$$

where $P(\boldsymbol{\theta})$ is the prior distribution over the grid of parameter values. For the mixture of two Gaussians this is a five-dimensional integral. If each of $K$ parameters has its uncertainty reduced by a factor of ten, then brute-force integration requires a grid of at least $10^K$ points. This exponential growth of computation with model size is the reason why complete enumeration is rarely a feasible computational strategy.

</div>

## Chapter 22 -- Maximum Likelihood and Clustering

Rather than enumerate all hypotheses -- which may be exponential in number -- we can save a lot of time by homing in on one good hypothesis that fits the data well. This is the philosophy behind the **maximum likelihood** method, which identifies the setting of the parameter vector $\boldsymbol{\theta}$ that maximizes the likelihood, $P(\text{Data} \mid \boldsymbol{\theta}, \mathcal{H})$.

For any model, it is usually easiest to work with the *logarithm* of the likelihood rather than the likelihood, since likelihoods, being products of the probabilities of many data points, tend to be very small. Likelihoods multiply; log likelihoods add.

### 22.1 Maximum Likelihood for One Gaussian

We return to the Gaussian for our first examples. Assume we have data $\lbrace x_n\rbrace_{n=1}^N$. The log likelihood is:

$$\ln P(\lbrace x_n\rbrace_{n=1}^N \mid \mu, \sigma) = -N \ln(\sqrt{2\pi}\sigma) - \sum_n (x_n - \mu)^2 / (2\sigma^2).$$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Sufficient Statistics for a Gaussian)</span></p>

The likelihood can be expressed in terms of two functions of the data, the **sample mean**

$$\bar{x} \equiv \sum_{n=1}^N x_n / N,$$

and the **sum of square deviations**

$$S \equiv \sum_n (x_n - \bar{x})^2:$$

$$\ln P(\lbrace x_n\rbrace_{n=1}^N \mid \mu, \sigma) = -N \ln(\sqrt{2\pi}\sigma) - [N(\mu - \bar{x})^2 + S]/(2\sigma^2).$$

Because the likelihood depends on the data only through $\bar{x}$ and $S$, these two quantities are known as **sufficient statistics**.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Maximum Likelihood Estimators for a Gaussian)</span></p>

The values of $\mu$ and $\ln\sigma$ that jointly maximize the likelihood are:

$$\lbrace\mu, \sigma\rbrace_{\text{ML}} = \left\lbrace \bar{x},\; \sigma_N = \sqrt{S/N} \right\rbrace,$$

where

$$\sigma_N \equiv \sqrt{\frac{\sum_{n=1}^N (x_n - \bar{x})^2}{N}}.$$

The error bars on $\mu$ (derived from the curvature of the log likelihood) are:

$$\sigma_\mu = \frac{\sigma}{\sqrt{N}}.$$

The error bars on $\ln\sigma$ are:

$$\sigma_{\ln\sigma} = \frac{1}{\sqrt{2N}}.$$

</div>

### 22.2 Maximum Likelihood for a Mixture of Gaussians

We now derive an algorithm for fitting a mixture of Gaussians to one-dimensional data. Consider a *mixture of two Gaussians*:

$$P(x \mid \mu_1, \mu_2, \sigma) = \left[\sum_{k=1}^{2} p_k \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(x - \mu_k)^2}{2\sigma^2}\right)\right],$$

where the prior probability of class label $k$ is $\lbrace p_1 = 1/2,\; p_2 = 1/2\rbrace$; $\lbrace\mu_k\rbrace$ are the means and both Gaussians have standard deviation $\sigma$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Posterior Class Probabilities as Sigmoid)</span></p>

Assuming that the means $\lbrace\mu_k\rbrace$ and $\sigma$ are known, the posterior probability of the class label $k_n$ of the $n$th point can be written as:

$$P(k_n{=}1 \mid x_n, \boldsymbol{\theta}) = \frac{1}{1 + \exp[-(w_1 x_n + w_0)]},$$

$$P(k_n{=}2 \mid x_n, \boldsymbol{\theta}) = \frac{1}{1 + \exp[+(w_1 x_n + w_0)]},$$

where $w_1 = (\mu_1 - \mu_2)/\sigma^2$ and $w_0 = (\mu_2^2 - \mu_1^2)/(2\sigma^2)$.

</div>

Assuming the means are *not* known, and $\sigma$ is known, we derive an iterative algorithm for maximizing the likelihood $P(\lbrace x_n\rbrace_{n=1}^N \mid \lbrace\mu_k\rbrace, \sigma) = \prod_n P(x_n \mid \lbrace\mu_k\rbrace, \sigma)$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Maximum Likelihood via Newton--Raphson for Gaussian Mixtures)</span></p>

Let $L$ denote the natural log of the likelihood, and $p_{k|n} \equiv P(k_n{=}k \mid x_n, \boldsymbol{\theta})$. The derivative of the log likelihood with respect to $\mu_k$ is:

$$\frac{\partial}{\partial \mu_k} L = \sum_n p_{k|n} \frac{(x_n - \mu_k)}{\sigma^2}.$$

The second derivative is approximately:

$$\frac{\partial^2}{\partial \mu_k^2} L = -\sum_n p_{k|n} \frac{1}{\sigma^2}.$$

An approximate Newton--Raphson step updates the means to:

$$\mu_k' = \frac{\sum_n p_{k|n} x_n}{\sum_n p_{k|n}}.$$

This algorithm is identical to the soft K-means algorithm. The connection between clustering and mixture-density-modelling allows us to derive enhanced versions of K-means.

</div>

### 22.3 Enhancements to Soft K-means

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Soft K-means, Version 2 -- Spherical Gaussians)</span></p>

Each cluster is a spherical Gaussian with its own width $\sigma_k$ and weight $\pi_k$. Let $I$ denote the dimensionality of $\mathbf{x}$.

**Assignment step.** The responsibilities are:

$$r_k^{(n)} = \frac{\pi_k \frac{1}{(\sqrt{2\pi}\sigma_k)^I} \exp\!\left(-\frac{1}{\sigma_k^2} d(\mathbf{m}^{(k)}, \mathbf{x}^{(n)})\right)}{\sum_{k'} \pi_{k'} \frac{1}{(\sqrt{2\pi}\sigma_{k'})^I} \exp\!\left(-\frac{1}{\sigma_{k'}^2} d(\mathbf{m}^{(k')}, \mathbf{x}^{(n)})\right)}.$$

**Update step.** Each cluster's parameters $\mathbf{m}^{(k)}$, $\pi_k$, and $\sigma_k^2$ are adjusted to match the data points it is responsible for:

$$\mathbf{m}^{(k)} = \frac{\sum_n r_k^{(n)} \mathbf{x}^{(n)}}{R^{(k)}}, \qquad \sigma_k^2 = \frac{\sum_n r_k^{(n)} (\mathbf{x}^{(n)} - \mathbf{m}^{(k)})^2}{I \, R^{(k)}}, \qquad \pi_k = \frac{R^{(k)}}{\sum_k R^{(k)}},$$

where $R^{(k)} = \sum_n r_k^{(n)}$ is the total responsibility of mean $k$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Soft K-means, Version 3 -- Axis-Aligned Gaussians)</span></p>

To model elongated clusters, each cluster has its own variance along each coordinate axis $i$: $\sigma_i^{(k)}$. The assignment step becomes:

$$r_k^{(n)} = \frac{\pi_k \prod_{i=1}^{I} \frac{1}{\sqrt{2\pi}\sigma_i^{(k)}} \exp\!\left(-\sum_{i=1}^{I} (m_i^{(k)} - x_i^{(n)})^2 / (2(\sigma_i^{(k)})^2)\right)}{\sum_{k'} (\text{numerator, with } k' \text{ in place of } k)}.$$

The variance update rule becomes:

$$(\sigma_i^{(k)})^2 = \frac{\sum_n r_k^{(n)} (x_i^{(n)} - m_i^{(k)})^2}{R^{(k)}}.$$

</div>

### 22.4 A Fatal Flaw of Maximum Likelihood

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Overfitting in Maximum Likelihood)</span></p>

Soft K-means can blow up. Put one cluster exactly on one data point and let its variance go to zero -- you can obtain an arbitrarily large likelihood! Maximum likelihood methods can break down by finding highly tuned models that fit part of the data perfectly. This phenomenon is known as **overfitting**.

The reason we are not interested in these solutions with enormous likelihood is this: these parameter-settings may have enormous posterior probability *density*, but the density is large over only a very small *volume* of parameter space. So the probability *mass* associated with these likelihood spikes is usually tiny.

We conclude that maximum likelihood methods are not a satisfactory general solution to data-modelling problems: the likelihood may be infinitely large at certain parameter settings, the maximum of the likelihood is often unrepresentative in high-dimensional problems, and even in low-dimensional problems the maximum likelihood estimator $\sigma_N$ is a **biased** estimator.

</div>

#### The Maximum A Posteriori (MAP) Method

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Problems with MAP)</span></p>

A popular replacement for maximizing the likelihood is maximizing the Bayesian posterior probability density of the parameters. However, multiplying the likelihood by a prior and maximizing the posterior does not make all the problems go away: the posterior density often also has infinitely-large spikes, and the maximum of the posterior probability density is often unrepresentative of the whole posterior distribution.

A further reason for disliking the maximum *a posteriori* is that it is **basis-dependent**. If we make a nonlinear change of basis from the parameter $\theta$ to $u = f(\theta)$, then

$$P(u) = P(\theta) \left\lvert\frac{\partial \theta}{\partial u}\right\rvert.$$

The maximum of $P(u)$ will usually not coincide with the maximum of $P(\theta)$. It is undesirable to use a method whose answers change when we change representation.

</div>

## Chapter 23 -- Useful Probability Distributions

In Bayesian data modelling, there is a small collection of probability distributions that come up again and again. There is no need to memorize any of them, except perhaps the Gaussian; if a distribution is important enough, it will memorize itself.

### 23.1 Distributions over Integers

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Common Discrete Distributions)</span></p>

The **binomial distribution** for an integer $r$ with parameters $f$ (the bias, $f \in [0,1]$) and $N$ (the number of trials) is:

$$P(r \mid f, N) = \binom{N}{r} f^r (1-f)^{N-r}, \quad r \in \lbrace 0, 1, 2, \ldots, N\rbrace.$$

The **Poisson distribution** with parameter $\lambda > 0$ is:

$$P(r \mid \lambda) = e^{-\lambda} \frac{\lambda^r}{r!}, \quad r \in \lbrace 0, 1, 2, \ldots\rbrace.$$

The **exponential distribution on integers** with parameter $f$ is:

$$P(r \mid f) = f^r(1-f), \quad r \in (0, 1, 2, \ldots, \infty),$$

which may also be written as $P(r \mid f) = (1-f) e^{-\lambda r}$ where $\lambda = \ln(1/f)$. It arises in waiting problems.

</div>

### 23.2 Distributions over Unbounded Real Numbers

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Gaussian Distribution)</span></p>

The **Gaussian distribution** (or normal distribution) with mean $\mu$ and standard deviation $\sigma$ is:

$$P(x \mid \mu, \sigma) = \frac{1}{Z} \exp\!\left(-\frac{(x - \mu)^2}{2\sigma^2}\right), \quad x \in (-\infty, \infty),$$

where $Z = \sqrt{2\pi\sigma^2}$. The quantity $\tau \equiv 1/\sigma^2$ is called the **precision** parameter.

A sample $z$ from a standard univariate Gaussian can be generated by computing $z = \cos(2\pi u_1)\sqrt{2\ln(1/u_2)}$, where $u_1$ and $u_2$ are uniformly distributed in $(0, 1)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Caution about Gaussians)</span></p>

The Gaussian has very light tails: the log-probability-density decreases quadratically. The typical deviation of $x$ from $\mu$ is $\sigma$, but the respective probabilities that $x$ deviates from $\mu$ by more than $2\sigma$, $3\sigma$, $4\sigma$, and $5\sigma$ are 0.046, 0.003, $6 \times 10^{-5}$, and $6 \times 10^{-7}$. If a variable that is modelled with a Gaussian actually has a heavier-tailed distribution, the rest of the model will contort itself to reduce the deviations of the outliers, like a sheet of paper being crushed by a rubber band.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Student-t Distribution)</span></p>

An appropriately weighted mixture of an infinite number of Gaussians, all having mean $\mu$, gives the **Student-t distribution**:

$$P(x \mid \mu, s, n) = \frac{1}{Z} \frac{1}{(1 + (x - \mu)^2/(ns^2))^{(n+1)/2}},$$

where $Z = \sqrt{\pi n s^2}\;\Gamma(n/2)/\Gamma((n+1)/2)$ and $n$ is the number of degrees of freedom. If $n > 1$, the mean is $\mu$. If $n > 2$, the variance is $\sigma^2 = ns^2/(n-2)$. As $n \to \infty$, the Student distribution approaches the normal distribution. The special case $n = 1$ is the **Cauchy distribution**.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Biexponential and Inverse-Cosh Distributions)</span></p>

The **biexponential distribution** (Laplace distribution) has tails intermediate in heaviness between Student and Gaussian:

$$P(x \mid \mu, s) = \frac{1}{Z} \exp\!\left(-\frac{\lvert x - \mu\rvert}{s}\right), \quad x \in (-\infty, \infty), \quad Z = 2s.$$

The **inverse-cosh distribution**:

$$P(x \mid \beta) \propto \frac{1}{[\cosh(\beta x)]^{1/\beta}},$$

is a popular model in independent component analysis. In the limit of large $\beta$, it becomes a biexponential distribution. In the limit $\beta \to 0$, it approaches a Gaussian with mean zero and variance $1/\beta$.

</div>

### 23.3 Distributions over Positive Real Numbers

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Exponential Distribution)</span></p>

The **exponential distribution**:

$$P(x \mid s) = \frac{1}{Z} \exp\!\left(-\frac{x}{s}\right), \quad x \in (0, \infty), \quad Z = s,$$

arises in waiting problems. The probability of your wait $x$ for a bus arriving independently at random with one every $s$ minutes on average is exponential with mean $s$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Gamma Distribution)</span></p>

The **gamma distribution** is the product of a one-parameter exponential distribution with a polynomial $x^{c-1}$:

$$P(x \mid s, c) = \Gamma(x;\, s, c) = \frac{1}{Z}\left(\frac{x}{s}\right)^{c-1} \exp\!\left(-\frac{x}{s}\right), \quad 0 \leq x < \infty,$$

where $Z = \Gamma(c)\,s$. This is a simple peaked distribution with mean $sc$ and variance $s^2 c$.

It is often natural to represent a positive real variable $x$ in terms of its logarithm $l = \ln x$. The density of $l$ is:

$$P(l) = \frac{1}{Z_l}\left(\frac{x(l)}{s}\right)^c \exp\!\left(-\frac{x(l)}{s}\right), \quad Z_l = \Gamma(c).$$

In the limit $sc = 1, c \to 0$, we obtain the **noninformative prior** for a scale parameter, the $1/x$ prior. This improper prior is invariant under the reparameterization $x = mx$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Inverse-Gamma and Log-Normal Distributions)</span></p>

If $x$ has a gamma distribution and we work in terms of $v = 1/x$, we obtain the **inverse-gamma distribution**:

$$P(v \mid s, c) = \frac{1}{Z_v}\left(\frac{1}{sv}\right)^{c+1} \exp\!\left(-\frac{1}{sv}\right), \quad 0 \leq v < \infty, \quad Z_v = \Gamma(c)/s.$$

The **log-normal distribution** is the distribution that results when $l = \ln x$ has a normal distribution. Defining $m$ as the median and $s$ as the standard deviation of $\ln x$:

$$P(l \mid m, s) = \frac{1}{Z}\exp\!\left(-\frac{(l - \ln m)^2}{2s^2}\right), \quad Z = \sqrt{2\pi s^2},$$

which implies

$$P(x \mid m, s) = \frac{1}{x}\frac{1}{Z}\exp\!\left(-\frac{(\ln x - \ln m)^2}{2s^2}\right), \quad x \in (0, \infty).$$

</div>

### 23.4 Distributions over Periodic Variables

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Von Mises Distribution)</span></p>

A periodic variable $\theta \in [0, 2\pi]$ has the property that $\theta = 0$ and $\theta = 2\pi$ are equivalent. The **Von Mises distribution** plays for periodic variables the role of the Gaussian for real variables:

$$P(\theta \mid \mu, \beta) = \frac{1}{Z}\exp(\beta\cos(\theta - \mu)), \quad \theta \in (0, 2\pi),$$

where $Z = 2\pi I_0(\beta)$ and $I_0(x)$ is a modified Bessel function.

A related distribution arising from Brownian diffusion around the circle is the **wrapped Gaussian distribution**:

$$P(\theta \mid \mu, \sigma) = \sum_{n=-\infty}^{\infty} \text{Normal}(\theta;\, (\mu + 2\pi n), \sigma^2), \quad \theta \in (0, 2\pi).$$

</div>

### 23.5 Distributions over Probabilities

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Beta Distribution)</span></p>

The **beta distribution** is a probability density over a variable $p$ that is a probability, $p \in (0, 1)$:

$$P(p \mid u_1, u_2) = \frac{1}{Z(u_1, u_2)} p^{u_1 - 1}(1 - p)^{u_2 - 1},$$

where

$$Z(u_1, u_2) = \frac{\Gamma(u_1)\Gamma(u_2)}{\Gamma(u_1 + u_2)}.$$

Special cases include: the uniform distribution ($u_1 = 1, u_2 = 1$), the Jeffreys prior ($u_1 = 0.5, u_2 = 0.5$), and the improper Laplace prior ($u_1 = 0, u_2 = 0$).

If we transform to the logit $l \equiv \ln p/(1-p)$, the density is always a pleasant bell-shaped density over $l$, while the density over $p$ may have singularities at $p = 0$ and $p = 1$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Dirichlet Distribution)</span></p>

The **Dirichlet distribution** is a density over an $I$-dimensional vector $\mathbf{p}$ whose components are positive and sum to 1. The beta distribution is the special case $I = 2$. The Dirichlet distribution is parameterized by a measure $\mathbf{u} = \alpha\mathbf{m}$, where $\mathbf{m}$ is a normalized measure ($\sum m_i = 1$) and $\alpha$ is positive:

$$P(\mathbf{p} \mid \alpha\mathbf{m}) = \frac{1}{Z(\alpha\mathbf{m})} \prod_{i=1}^{I} p_i^{\alpha m_i - 1}\, \delta\!\left(\sum_i p_i - 1\right),$$

where

$$Z(\alpha\mathbf{m}) = \prod_i \Gamma(\alpha m_i) / \Gamma(\alpha).$$

The mean of the distribution is $\mathbf{m}$: $\int \text{Dirichlet}^{(I)}(\mathbf{p} \mid \alpha\mathbf{m})\,\mathbf{p}\;\mathrm{d}^I\mathbf{p} = \mathbf{m}$.

The parameter $\alpha$ controls sharpness: large $\alpha$ produces a distribution sharply peaked around $\mathbf{m}$, while small $\alpha$ produces distributions where one component $p_i$ receives an overwhelming share of the probability.

When working with a probability vector $\mathbf{p}$, it is often helpful to work in the **softmax basis**: $p_i = \frac{1}{Z}e^{a_i}$ where $Z = \sum_i e^{a_i}$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Entropic Distribution)</span></p>

The **entropic distribution** for a probability vector $\mathbf{p}$ is sometimes used in the "maximum entropy" image reconstruction community:

$$P(\mathbf{p} \mid \alpha, \mathbf{m}) = \frac{1}{Z(\alpha, \mathbf{m})} \exp[-\alpha D_{\text{KL}}(\mathbf{p} \| \mathbf{m})]\,\delta\!\left(\sum_i p_i - 1\right),$$

where $\mathbf{m}$ is a positive vector (the measure) and $D_{\text{KL}}(\mathbf{p} \| \mathbf{m}) = \sum_i p_i \log p_i/m_i$.

</div>

## Chapter 24 -- Exact Marginalization

How can we avoid the exponentially large cost of complete enumeration of all hypotheses? Before we stoop to approximate methods, we explore two approaches to exact marginalization: first, marginalization over continuous variables (nuisance parameters) by doing *integrals*; and second, summation over discrete variables by message-passing.

### 24.1 Inferring the Mean and Variance of a Gaussian Distribution

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Conjugate Priors for Gaussian Parameters)</span></p>

The **conjugate prior** for a mean $\mu$ is a Gaussian: $P(\mu \mid \mu_0, \sigma_\mu) = \text{Normal}(\mu;\, \mu_0, \sigma_\mu^2)$. In the limit $\mu_0 = 0$, $\sigma_\mu \to \infty$, we obtain the **noninformative prior** for a location parameter (the flat prior). This is *invariant* under $\mu' = \mu + c$. It is an *improper* prior (not normalizable).

The **conjugate prior** for a standard deviation $\sigma$ is a gamma distribution over the precision $\beta = 1/\sigma^2$, with parameters $b_\beta$ and $c_\beta$:

$$P(\beta) = \Gamma(\beta;\, b_\beta, c_\beta) = \frac{1}{\Gamma(c_\beta)} \frac{\beta^{c_\beta - 1}}{b_\beta^{c_\beta}} \exp\!\left(-\frac{\beta}{b_\beta}\right), \quad 0 \leq \beta < \infty.$$

In the limit $b_\beta c_\beta = 1, c_\beta \to 0$, we obtain the noninformative $1/\sigma$ prior for a scale parameter.

</div>

#### Maximum Likelihood and Marginalization: $\sigma_N$ and $\sigma_{N-1}$

Given data $D = \lbrace x_n\rbrace_{n=1}^N$, an estimator of $\mu$ is $\bar{x} \equiv \sum_{n=1}^N x_n / N$, and two estimators of $\sigma$ are:

$$\sigma_N \equiv \sqrt{\frac{\sum_{n=1}^N (x_n - \bar{x})^2}{N}} \quad \text{and} \quad \sigma_{N-1} \equiv \sqrt{\frac{\sum_{n=1}^N (x_n - \bar{x})^2}{N-1}}.$$

In sampling theory, $\bar{x}$ is an unbiased estimator of $\mu$ with smallest variance. The estimator $(\bar{x}, \sigma_N)$ is the maximum likelihood estimator. However, $\sigma_N$ is **biased**: its expectation, averaging over many imagined experiments, is not $\sigma$. The estimator $\sigma_{N-1}^2$ is an unbiased estimator of $\sigma^2$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Bayesian Marginalization for Gaussian Parameters)</span></p>

Using improper noninformative priors for $\mu$ and $\sigma$, the joint posterior probability of $\mu$ and $\sigma$ is proportional to the likelihood function:

$$P(\mu, \sigma \mid \lbrace x_n\rbrace_{n=1}^N) \propto \frac{1}{(2\pi\sigma^2)^{N/2}} \exp\!\left(-\frac{N(\mu - \bar{x})^2 + S}{2\sigma^2}\right) \frac{1}{\sigma_\mu} \frac{1}{\sigma},$$

where $S \equiv \sum_n (x_n - \bar{x})^2$.

**Conditional on $\sigma$:** The posterior probability of $\mu$ is:

$$P(\mu \mid \lbrace x_n\rbrace_{n=1}^N, \sigma) = \text{Normal}(\mu;\, \bar{x},\, \sigma^2/N).$$

**Marginalizing over $\mu$:** The evidence for $\sigma$ (the marginal likelihood) is:

$$\ln P(\lbrace x_n\rbrace_{n=1}^N \mid \sigma) = -N\ln(\sqrt{2\pi}\sigma) - \frac{S}{2\sigma^2} + \ln\frac{\sqrt{2\pi}\sigma/\sqrt{N}}{\sigma_\mu}.$$

The first two terms are the best-fit log likelihood (with $\mu = \bar{x}$). The last term is the log of the **Occam factor** which penalizes smaller values of $\sigma$. Differentiating with respect to $\ln\sigma$, the most probable $\sigma$ shifts from $\sigma_N$ to:

$$\sigma_{N-1} = \sqrt{S/(N-1)}.$$

Intuitively, the denominator $(N{-}1)$ counts the number of effective noise measurements contained in $S$: the sum contains $N$ residuals squared, but there are only $(N{-}1)$ effective noise measurements because the determination of $\mu$ from the data consumes one dimension of noise.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Posterior Distribution of $\mu$ after Marginalizing $\sigma$)</span></p>

Marginalizing over $\sigma$, the posterior marginal distribution of $\mu$ is a **Student-t distribution**:

$$P(\mu \mid D) \propto 1/\left(N(\mu - \bar{x})^2 + S\right)^{N/2}.$$

</div>

## Chapter 25 -- Exact Marginalization in Trellises

In this chapter we discuss a few exact methods that are used in probabilistic modelling. As an example we discuss the task of decoding a linear error-correcting code. Inferences can be conducted most efficiently by **message-passing algorithms**, which take advantage of the graphical structure of the problem to avoid unnecessary duplication of computations.

### 25.1 Decoding Problems

A codeword $\mathbf{t}$ is selected from a linear $(N, K)$ code $\mathcal{C}$, and it is transmitted over a noisy channel; the received signal is $\mathbf{y}$. Given an assumed channel model $P(\mathbf{y} \mid \mathbf{t})$, there are two decoding problems:

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Decoding Problems)</span></p>

**The codeword decoding problem** is the task of inferring which codeword $\mathbf{t}$ was transmitted given the received signal.

**The bitwise decoding problem** is the task of inferring for each transmitted bit $t_n$ how likely it is that that bit was a one rather than a zero.

**The MAP codeword decoding problem** is the task of identifying the *most probable codeword* $\mathbf{t}$ given the received signal. If the prior over codewords is uniform, this is identical to *maximum likelihood decoding*.

</div>

By Bayes' theorem, the posterior probability of the codeword $\mathbf{t}$ is:

$$P(\mathbf{t} \mid \mathbf{y}) = \frac{P(\mathbf{y} \mid \mathbf{t})P(\mathbf{t})}{P(\mathbf{y})}.$$

The **likelihood function** for any memoryless channel is a separable function:

$$P(\mathbf{y} \mid \mathbf{t}) = \prod_{n=1}^{N} P(y_n \mid t_n).$$

For a Gaussian channel with transmissions $\pm x$ and additive noise of standard deviation $\sigma$, the likelihood ratio is:

$$\frac{P(y_n \mid t_n{=}1)}{P(y_n \mid t_n{=}0)} = \exp\!\left(\frac{2xy_n}{\sigma^2}\right).$$

The exact solution of the bitwise decoding problem is obtained by **marginalizing** over the other bits:

$$P(t_n \mid \mathbf{y}) = \sum_{\lbrace t_{n'} :\, n' \neq n\rbrace} P(\mathbf{t} \mid \mathbf{y}).$$

Computing these marginal probabilities by explicit summation over all codewords $\mathbf{t}$ takes exponential time. But for certain codes, the bitwise decoding problem can be solved much more efficiently using the **forward--backward algorithm** (also known as the sum--product algorithm).

### 25.2 Codes and Trellises

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Trellis)</span></p>

A **trellis** is a graph consisting of **nodes** (also known as states or vertices) and **edges**. The nodes are grouped into vertical slices called **times**, and the times are ordered such that each edge connects a node in one time to a node in a neighbouring time. Every edge is labelled with a **symbol**. The leftmost and rightmost states contain only one node.

A trellis with $N+1$ times defines a code of blocklength $N$: a codeword is obtained by taking a path from left to right and reading out the symbols on the edges traversed. The **width** of the trellis at a given time is the number of nodes at that time.

A trellis is called a **linear trellis** if the code it defines is a linear code. A minimal trellis for a linear $(N, K)$ code cannot have width greater than $2^K$. If $M = N - K$, the minimal trellis's width is everywhere less than $2^M$.

</div>

### 25.3 Solving the Decoding Problems on a Trellis

#### The Min--Sum Algorithm (Viterbi Algorithm)

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Min--Sum / Viterbi Algorithm)</span></p>

The MAP codeword decoding problem can be solved using the **min--sum algorithm**. Each codeword of the code corresponds to a path across the trellis. The log likelihood of a codeword is the sum of the bitwise log likelihoods. We associate with each edge a cost $-\log P(y_n \mid t_n)$, where $t_n$ is the transmitted bit associated with that edge and $y_n$ is the received symbol.

The min--sum algorithm identifies the most probable codeword in a number of computer operations equal to the number of edges in the trellis. This algorithm is also known as the **Viterbi algorithm**.

</div>

#### The Sum--Product Algorithm (Forward--Backward Algorithm)

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Sum--Product / Forward--Backward Algorithm on a Trellis)</span></p>

To solve the bitwise decoding problem, we replace the costs on the edges by the likelihoods themselves, $P(y_n \mid t_n)$, and replace the min and sum operations by sum and product respectively.

Let $i$ run over nodes/states, $i = 0$ be the start state, $\mathcal{P}(i)$ denote the set of parent states, and $w_{ij}$ be the likelihood associated with the edge from node $j$ to node $i$. Define **forward-pass messages** $\alpha_i$:

$$\alpha_0 = 1, \qquad \alpha_i = \sum_{j \in \mathcal{P}(i)} w_{ij}\,\alpha_j.$$

These are computed sequentially from left to right. The message $\alpha_I$ at the end node is proportional to the marginal probability of the data.

Define **backward-pass messages** $\beta_i$. Let node $I$ be the end node:

$$\beta_I = 1, \qquad \beta_j = \sum_{i:\, j \in \mathcal{P}(i)} w_{ij}\,\beta_i.$$

These are computed sequentially from right to left. Finally, to find the probability that the $n$th bit was $t = 0$ or $1$, we compute for each value of $t$:

$$r_n^{(t)} = \sum_{i,j:\, j \in \mathcal{P}(i),\, t_{ij} = t} \alpha_j\, w_{ij}\, \beta_i.$$

The posterior probability that $t_n$ was $t = 0/1$ is:

$$P(t_n = t \mid \mathbf{y}) = \frac{1}{Z}\, r_n^{(t)},$$

where $Z = r_n^{(0)} + r_n^{(1)}$ should be identical to $\alpha_I$.

Other names for this algorithm include "the BCJR algorithm" and "belief propagation."

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(MAP vs. Bitwise Decoding)</span></p>

The MAP codeword (the most probable codeword) is not always the same as the bitwise decoding (selecting the most probable state for each bit using the posterior marginal distributions). The bitwise decoding may produce a string that is not even a valid codeword.

</div>

### 25.4 More on Trellises

The **span** of a codeword is the set of bits contained between the first non-zero bit and the last non-zero bit (inclusive). A generator matrix is in **trellis-oriented form** if the spans of its rows all start in different columns and all end in different columns.

To make a trellis from a generator matrix, first put it into trellis-oriented form by row-manipulations similar to Gaussian elimination. Each row defines an $(N, 1)$ subcode. The trellis is built incrementally by adding one subcode at a time: the vertices within the span of the new subcode are all duplicated, edge symbols in the original trellis are unchanged, and edge symbols in the second part are flipped wherever the new subcode has a 1.

Trellises can also be constructed from **parity-check matrices** via the syndrome. Each state in the trellis is a partial syndrome. The starting and ending states are both constrained to be the zero syndrome. The trellis is obtained as the intersection of two trees of possible syndrome sequences, one grown from each end.

An important observation is that **rearranging the order of the codeword bits can sometimes lead to smaller, simpler trellises**.

## Chapter 26 -- Exact Marginalization in Graphs

We now take a more general view of the tasks of inference and marginalization.

### 26.1 The General Problem

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Factor Graphs and Marginalization Problems)</span></p>

Assume that a function $P^*$ of a set of $N$ variables $\mathbf{x} \equiv \lbrace x_n\rbrace_{n=1}^N$ is defined as a product of $M$ **factors**:

$$P^*(\mathbf{x}) = \prod_{m=1}^{M} f_m(\mathbf{x}_m),$$

where each factor $f_m(\mathbf{x}_m)$ is a function of a subset $\mathbf{x}_m$ of the variables. If $P^*$ is a positive function, we may define a normalized function:

$$P(\mathbf{x}) \equiv \frac{1}{Z} P^*(\mathbf{x}) = \frac{1}{Z}\prod_{m=1}^{M} f_m(\mathbf{x}_m),$$

where $Z = \sum_{\mathbf{x}} \prod_{m=1}^{M} f_m(\mathbf{x}_m)$.

A function of the factored form can be depicted by a **factor graph**, in which the variables are depicted by circular nodes and the factors are depicted by square nodes. An edge is placed between variable node $n$ and factor node $m$ if $f_m(\mathbf{x}_m)$ depends on variable $x_n$.

The three key tasks are:
1. **Normalization:** compute $Z$.
2. **Marginalization:** compute $Z_n(x_n) = \sum_{\lbrace x_{n'}\rbrace,\, n' \neq n} P^*(\mathbf{x})$.
3. **Normalized marginalization:** compute $P_n(x_n) = Z_n(x_n)/Z$.

All these tasks are intractable in general -- the cost grows exponentially with $N$. But for certain functions $P^*$ the marginals can be computed efficiently by exploiting the factorization.

</div>

### 26.2 The Sum--Product Algorithm

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Sum--Product Algorithm on Factor Graphs)</span></p>

Let $\mathcal{N}(m)$ denote the set of variable indices that factor $m$ depends on, and $\mathcal{M}(n)$ the set of factors that variable $n$ participates in. The shorthand $\mathbf{x}_m \setminus n$ denotes the set of variables in $\mathbf{x}_m$ with $x_n$ excluded.

The algorithm involves messages of two types passing along the edges:

**From variable to factor:**

$$q_{n \to m}(x_n) = \prod_{m' \in \mathcal{M}(n) \setminus m} r_{m' \to n}(x_n).$$

**From factor to variable:**

$$r_{m \to n}(x_n) = \sum_{\mathbf{x}_m \setminus n} \left(f_m(\mathbf{x}_m) \prod_{n' \in \mathcal{N}(m) \setminus n} q_{n' \to m}(x_{n'})\right).$$

**Initialization (Method 1 -- for tree-like graphs):** Leaf factor nodes $m$ send $r_{m \to n}(x_n) = f_m(x_n)$. Leaf variable nodes $n$ send $q_{n \to m}(x_n) = 1$. Messages propagate inward; a message is created only when all messages it depends on have been received.

**Initialization (Method 2 -- also works for graphs with cycles):** Set all initial messages $q_{n \to m}(x_n) = 1$, then alternate factor and variable message updates. This "lazy" method can also be applied to graphs with cycles (loopy belief propagation), though convergence is not guaranteed.

**Reading out the answer:** The marginal function of $x_n$ is:

$$Z_n(x_n) = \prod_{m \in \mathcal{M}(n)} r_{m \to n}(x_n).$$

The normalizing constant $Z = \sum_{x_n} Z_n(x_n)$, and $P_n(x_n) = Z_n(x_n)/Z$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(On-the-fly Normalization)</span></p>

If we are interested only in the *normalized* marginals, an alternative version normalizes the variable-to-factor messages:

$$q_{n \to m}(x_n) = \alpha_{nm} \prod_{m' \in \mathcal{M}(n) \setminus m} r_{m' \to n}(x_n),$$

where $\alpha_{nm}$ is a scalar chosen such that $\sum_{x_n} q_{n \to m}(x_n) = 1$. This is useful because if $P^*$ is a product of many factors, its values are likely to be very large or very small.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Factorization View of the Sum--Product Algorithm)</span></p>

The sum--product algorithm reexpresses the original factored function $P^*(\mathbf{x}) = \prod_{m=1}^{M} f_m(\mathbf{x}_m)$ as another factored function with $M + N$ factors:

$$P^*(\mathbf{x}) = \prod_{m=1}^{M} \phi_m(\mathbf{x}_m) \prod_{n=1}^{N} \psi_n(x_n).$$

Initially $\phi_m(\mathbf{x}_m) = f_m(\mathbf{x}_m)$ and $\psi_n(x_n) = 1$. Each time a factor-to-variable message $r_{m \to n}(x_n)$ is sent, the factorization is updated:

$$\psi_n(x_n) = \prod_{m \in \mathcal{M}(n)} r_{m \to n}(x_n), \qquad \phi_m(\mathbf{x}_m) = \frac{f_m(\mathbf{x}_m)}{\prod_{n \in \mathcal{N}(m)} r_{m \to n}(x_n)}.$$

Eventually $\psi_n(x_n)$ becomes the marginal $Z_n(x_n)$. This viewpoint applies whether or not the graph is tree-like.

</div>

### 26.3 The Min--Sum Algorithm

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Min--Sum Algorithm on Factor Graphs)</span></p>

The sum--product algorithm solves the marginalization problem. The closely related **maximization problem** -- find the setting of $\mathbf{x}$ that maximizes $P^*(\mathbf{x})$ -- can be solved by replacing the two operations **add** and **multiply** in the sum--product algorithm by **max** and **multiply** (or equivalently **min** and **sum** in the negative log likelihood domain).

The resulting **max--product algorithm** computes $\max_{\mathbf{x}} P^*(\mathbf{x})$, from which the solution of the maximization problem can be deduced. Each "marginal" $Z_n(x_n)$ then lists the maximum value that $P^*(\mathbf{x})$ can attain for each value of $x_n$.

In the negative log likelihood domain, where max and product become min and sum, this is known as the **min--sum algorithm** (also the **Viterbi algorithm**).

</div>

### 26.4 The Junction Tree Algorithm

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Graphs with Cycles)</span></p>

When the factor graph is not a tree, there are several options:
- The most widely used exact method is the **junction tree algorithm**, which works by agglomerating variables together until the agglomerated graph has no cycles. The complexity of the marginalization grows exponentially with the number of agglomerated variables.
- Among approximate methods, the most notable is **loopy belief propagation**: apply the sum--product algorithm to the graph as if it were a tree, iterate, and hope for convergence. The algorithm does not necessarily converge, and certainly does not in general compute the correct marginal functions; but it is nevertheless of great practical importance, especially in the decoding of sparse-graph codes.
- Other approximate methods include Monte Carlo methods and variational methods.

</div>

## Chapter 27 -- Laplace's Method

The idea behind the Laplace approximation is simple: approximate an unnormalized probability density by a Gaussian fitted at its peak, then use the Gaussian's normalizing constant as an approximation to the true one.

### 27.1 One-Dimensional Case

Assume we have an unnormalized probability density $P^*(x)$ whose normalizing constant

$$Z_P \equiv \int P^*(x) \, \mathrm{d}x$$

is of interest, and that $P^*(x)$ has a peak at a point $x_0$. We Taylor-expand the logarithm of $P^*(x)$ around this peak:

$$\ln P^*(x) \simeq \ln P^*(x_0) - \frac{c}{2}(x - x_0)^2 + \cdots,$$

where

$$c = -\left.\frac{\partial^2}{\partial x^2} \ln P^*(x)\right|_{x=x_0}.$$

We then approximate $P^*(x)$ by an unnormalized Gaussian:

$$Q^*(x) \equiv P^*(x_0) \exp\!\left[-\frac{c}{2}(x - x_0)^2\right],$$

and we approximate the normalizing constant $Z_P$ by the normalizing constant of this Gaussian:

$$Z_Q = P^*(x_0)\sqrt{\frac{2\pi}{c}}.$$

### 27.2 Multi-Dimensional Generalization

We can generalize the integral approximation to a density $P^*(\mathbf{x})$ over a $K$-dimensional space $\mathbf{x}$. If the matrix of second derivatives of $-\ln P^*(\mathbf{x})$ at the maximum $\mathbf{x}_0$ is $\mathbf{A}$, defined by:

$$A_{ij} = -\left.\frac{\partial^2}{\partial x_i \partial x_j} \ln P^*(\mathbf{x})\right|_{\mathbf{x}=\mathbf{x}_0},$$

so that the expansion is generalized to

$$\ln P^*(\mathbf{x}) \simeq \ln P^*(\mathbf{x}_0) - \frac{1}{2}(\mathbf{x} - \mathbf{x}_0)^\mathsf{T} \mathbf{A} (\mathbf{x} - \mathbf{x}_0) + \cdots,$$

then the normalizing constant can be approximated by:

$$Z_P \simeq Z_Q = P^*(\mathbf{x}_0) \frac{1}{\sqrt{\det \frac{1}{2\pi}\mathbf{A}}} = P^*(\mathbf{x}_0)\sqrt{\frac{(2\pi)^K}{\det \mathbf{A}}}.$$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Gaussian Normalizing Constant)</span></p>

The fact that the normalizing constant of a Gaussian is given by

$$\int \mathrm{d}^K \mathbf{x} \exp\!\left[-\frac{1}{2}\mathbf{x}^\mathsf{T}\mathbf{A}\mathbf{x}\right] = \sqrt{\frac{(2\pi)^K}{\det \mathbf{A}}}$$

can be proved by making an orthogonal transformation into the basis $\mathbf{u}$ in which $\mathbf{A}$ is transformed into a diagonal matrix. The integral then separates into a product of one-dimensional integrals, each of the form

$$\int \mathrm{d}u_i \exp\!\left[-\frac{1}{2}\lambda_i u_i^2\right] = \sqrt{\frac{2\pi}{\lambda_i}}.$$

The product of the eigenvalues $\lambda_i$ is the determinant of $\mathbf{A}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Basis Dependence)</span></p>

The Laplace approximation is **basis-dependent**: if $x$ is transformed to a nonlinear function $u(x)$ and the density is transformed to $P(u) = P(x)\,|\mathrm{d}x/\mathrm{d}u|$, then in general the approximate normalizing constants $Z_Q$ will be different. This can be viewed as a defect -- since the true value $Z_P$ is basis-independent -- or as an opportunity, because we can hunt for a choice of basis in which the Laplace approximation is most accurate.

</div>

Predictions can be made using the approximation $Q$. Physicists also call this widely-used approximation the **saddle-point approximation**.

## Chapter 28 -- Model Comparison and Occam's Razor

### 28.1 Occam's Razor

Occam's razor is the principle that states a preference for simple theories: "Accept the simplest explanation that fits the data." But is this an ad hoc rule of thumb, or is there a convincing reason for believing in simpler models?

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Bayesian Occam's Razor)</span></p>

Coherent inference (as embodied by Bayesian probability) automatically embodies Occam's razor, quantitatively.

</div>

#### Model Comparison and Occam's Razor

We evaluate the plausibility of two alternative theories $\mathcal{H}_1$ and $\mathcal{H}_2$ in the light of data $D$ as follows: using Bayes' theorem, we relate the plausibility of model $\mathcal{H}_1$ given the data, $P(\mathcal{H}_1 \mid D)$, to the predictions made by the model about the data, $P(D \mid \mathcal{H}_1)$, and the prior plausibility of $\mathcal{H}_1$, $P(\mathcal{H}_1)$. This gives the following probability ratio between theory $\mathcal{H}_1$ and theory $\mathcal{H}_2$:

$$\frac{P(\mathcal{H}_1 \mid D)}{P(\mathcal{H}_2 \mid D)} = \frac{P(\mathcal{H}_1)}{P(\mathcal{H}_2)} \frac{P(D \mid \mathcal{H}_1)}{P(D \mid \mathcal{H}_2)}.$$

The first ratio on the right-hand side measures how much our initial beliefs favoured $\mathcal{H}_1$ over $\mathcal{H}_2$. The second ratio expresses how well the observed data were predicted by $\mathcal{H}_1$, compared to $\mathcal{H}_2$.

Simple models tend to make precise predictions. Complex models, by their nature, are capable of making a greater variety of predictions. So if $\mathcal{H}_2$ is a more complex model, it must spread its predictive probability $P(D \mid \mathcal{H}_2)$ more thinly over the data space than $\mathcal{H}_1$. Thus, in the case where the data are compatible with both theories, the simpler $\mathcal{H}_1$ will turn out more probable than $\mathcal{H}_2$, without our having to express any subjective dislike for complex models.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Arithmetic vs. Cubic: Sequence $-1, 3, 7, 11$)</span></p>

Consider the sequence $-1, 3, 7, 11$. Let:

- $\mathcal{H}_a$ -- the sequence is an **arithmetic** progression "add $n$", where $n$ is an integer.
- $\mathcal{H}_c$ -- the sequence is generated by a **cubic** function of the form $x \to cx^3 + dx^2 + e$, where $c, d$ and $e$ are fractions.

Assuming both numbers (the added integer $n$ and the first number) could be anywhere between $-50$ and $50$, only the pair $\{n=4, \text{first number} = -1\}$ gives rise to $D = (-1, 3, 7, 11)$:

$$P(D \mid \mathcal{H}_a) = \frac{1}{101}\frac{1}{101} = 0.00010.$$

For $\mathcal{H}_c$, a reasonable prior states that for each fraction the numerator could be any number between $-50$ and $50$, and the denominator is any number from $1$ to $50$. After counting the compatible parameter settings:

$$P(D \mid \mathcal{H}_c) = 0.0000000000025 = 2.5 \times 10^{-12}.$$

Even if our prior probabilities for $\mathcal{H}_a$ and $\mathcal{H}_c$ are equal, the odds $P(D \mid \mathcal{H}_a) : P(D \mid \mathcal{H}_c)$ in favour of $\mathcal{H}_a$ are about **forty million to one**.

</div>

The complex theory $\mathcal{H}_c$ always suffers an "Occam factor" because it has more parameters, and so can predict a greater variety of data sets.

#### Bayesian Methods and Data Analysis

Two levels of inference can often be distinguished in the process of data modelling:

1. At the first level of inference, we assume that a particular model is true, and we fit that model to the data, i.e., we infer what values its free parameters should plausibly take, given the data. The results are often summarized by the most probable parameter values, and error bars on those parameters.
2. The second level of inference is the task of **model comparison**. Here we wish to compare the models in the light of the data, and assign some sort of preference or ranking to the alternatives.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Inference vs. Decision Theory)</span></p>

Both levels of *inference* are distinct from *decision theory*. The goal of inference is, given a defined hypothesis space and a particular data set, to assign probabilities to hypotheses. Decision theory typically chooses between alternative *actions* on the basis of these probabilities so as to minimize the expectation of a "loss function". This chapter concerns inference alone and no loss functions are involved. When we discuss model comparison, this should not be construed as implying model *choice*. Ideal Bayesian predictions do not involve choice between models; rather, predictions are made by summing over all the alternative models, weighted by their probabilities.

</div>

#### The Mechanism of the Bayesian Razor: the Evidence and the Occam Factor

Each model $\mathcal{H}_i$ is assumed to have a vector of parameters $\mathbf{w}$. A model is defined by a collection of probability distributions: a "prior" distribution $P(\mathbf{w} \mid \mathcal{H}_i)$, which states what values the model's parameters might be expected to take; and a set of conditional distributions, one for each value of $\mathbf{w}$, defining the predictions $P(D \mid \mathbf{w}, \mathcal{H}_i)$ that the model makes about the data $D$.

**1. Model fitting.** At the first level of inference, we assume that one model, the $i$th say, is true, and we infer what the model's parameters $\mathbf{w}$ might be, given the data $D$. Using Bayes' theorem, the **posterior probability** of the parameters $\mathbf{w}$ is:

$$P(\mathbf{w} \mid D, \mathcal{H}_i) = \frac{P(D \mid \mathbf{w}, \mathcal{H}_i) P(\mathbf{w} \mid \mathcal{H}_i)}{P(D \mid \mathcal{H}_i)},$$

that is,

$$\text{Posterior} = \frac{\text{Likelihood} \times \text{Prior}}{\text{Evidence}}.$$

The normalizing constant $P(D \mid \mathcal{H}_i)$ is commonly ignored since it is irrelevant to the first level of inference, i.e., the inference of $\mathbf{w}$; but it becomes important in the second level of inference, and we name it the **evidence** for $\mathcal{H}_i$.

It is common practice to use gradient-based methods to find the maximum of the posterior, which defines the most probable value for the parameters, $\mathbf{w}_\text{MP}$; it is then usual to summarize the posterior distribution by the value of $\mathbf{w}_\text{MP}$, and error bars or confidence intervals on these best-fit parameters. Error bars can be obtained from the curvature of the posterior; evaluating the Hessian at $\mathbf{w}_\text{MP}$, $\mathbf{A} = -\nabla\nabla \ln P(\mathbf{w} \mid D, \mathcal{H}_i)\big|_{\mathbf{w}_\text{MP}}$, and Taylor-expanding the log posterior probability with $\Delta\mathbf{w} = \mathbf{w} - \mathbf{w}_\text{MP}$:

$$P(\mathbf{w} \mid D, \mathcal{H}_i) \simeq P(\mathbf{w}_\text{MP} \mid D, \mathcal{H}_i) \exp\!\left(-\tfrac{1}{2}\Delta\mathbf{w}^\mathsf{T}\mathbf{A}\Delta\mathbf{w}\right),$$

so the posterior can be locally approximated as a Gaussian with covariance matrix $\mathbf{A}^{-1}$.

**2. Model comparison.** At the second level of inference, we wish to infer which model is most plausible given the data. The posterior probability of each model is:

$$P(\mathcal{H}_i \mid D) \propto P(D \mid \mathcal{H}_i) P(\mathcal{H}_i).$$

Notice that the data-dependent term $P(D \mid \mathcal{H}_i)$ is the evidence for $\mathcal{H}_i$, which appeared as the normalizing constant in the model fitting step. The second term, $P(\mathcal{H}_i)$, is the subjective prior over our hypothesis space. Assuming that we choose to assign equal priors $P(\mathcal{H}_i)$ to the alternative models, *models $\mathcal{H}_i$ are ranked by evaluating the evidence*.

#### Evaluating the Evidence

The evidence is the normalizing constant for the posterior:

$$P(D \mid \mathcal{H}_i) = \int P(D \mid \mathbf{w}, \mathcal{H}_i) P(\mathbf{w} \mid \mathcal{H}_i) \, \mathrm{d}\mathbf{w}.$$

For many problems the posterior $P(\mathbf{w} \mid D, \mathcal{H}_i) \propto P(D \mid \mathbf{w}, \mathcal{H}_i) P(\mathbf{w} \mid \mathcal{H}_i)$ has a strong peak at the most probable parameters $\mathbf{w}_\text{MP}$. Then, taking for simplicity the one-dimensional case, the evidence can be approximated, using Laplace's method, by the height of the peak of the integrand $P(D \mid \mathbf{w}, \mathcal{H}_i) P(\mathbf{w} \mid \mathcal{H}_i)$ times its width, $\sigma_{w \mid D}$:

$$P(D \mid \mathcal{H}_i) \simeq \underbrace{P(D \mid \mathbf{w}_\text{MP}, \mathcal{H}_i)}_{\text{Best fit likelihood}} \times \underbrace{P(\mathbf{w}_\text{MP} \mid \mathcal{H}_i) \, \sigma_{w \mid D}}_{\text{Occam factor}}.$$

#### Interpretation of the Occam Factor

The quantity $\sigma_{w \mid D}$ is the posterior uncertainty in $\mathbf{w}$. Suppose for simplicity that the prior $P(\mathbf{w} \mid \mathcal{H}_i)$ is uniform on some large interval $\sigma_w$, representing the range of values of $\mathbf{w}$ that were possible *a priori*, according to $\mathcal{H}_i$. Then $P(\mathbf{w}_\text{MP} \mid \mathcal{H}_i) = 1/\sigma_w$, and

$$\text{Occam factor} = \frac{\sigma_{w \mid D}}{\sigma_w},$$

i.e., *the Occam factor is equal to the ratio of the posterior accessible volume of $\mathcal{H}_i$'s parameter space to the prior accessible volume*, or the factor by which $\mathcal{H}_i$'s hypothesis space collapses when the data arrive. The logarithm of the Occam factor is a measure of the amount of information we gain about the model's parameters when the data arrive.

A complex model having many parameters, each of which is free to vary over a large range $\sigma_w$, will typically be penalized by a stronger Occam factor than a simpler model. The Occam factor also penalizes models that have to be finely tuned to fit the data, favouring models for which the required precision of the parameters $\sigma_{w \mid D}$ is coarse.

#### Occam Factor for Several Parameters

If the posterior is well approximated by a Gaussian, then the Occam factor is obtained from the determinant of the corresponding covariance matrix (cf. Chapter 27):

$$P(D \mid \mathcal{H}_i) \simeq \underbrace{P(D \mid \mathbf{w}_\text{MP}, \mathcal{H}_i)}_{\text{Best fit likelihood}} \times \underbrace{P(\mathbf{w}_\text{MP} \mid \mathcal{H}_i) \det^{-1/2}\!\left(\mathbf{A}/2\pi\right)}_{\text{Occam factor}},$$

where $\mathbf{A} = -\nabla\nabla \ln P(\mathbf{w} \mid D, \mathcal{H}_i)$, the Hessian which we evaluated when we calculated the error bars on $\mathbf{w}_\text{MP}$. As the amount of data collected increases, this Gaussian approximation is expected to become increasingly accurate.

In summary, Bayesian model comparison is a simple extension of maximum likelihood model selection: *the evidence is obtained by multiplying the best-fit likelihood by the Occam factor*.

### 28.2 Example

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(How Many Boxes Behind the Tree?)</span></p>

Consider an image of an area around a tree trunk with a size of 50 pixels, the trunk 10 pixels wide, and 16 distinguishable colours for boxes.

The theory $\mathcal{H}_1$ (one box near the trunk) has four free parameters: three coordinates defining the top three edges of the box, and one parameter giving the box's colour.

The theory $\mathcal{H}_2$ (two boxes near the trunk) has eight free parameters (twice four), plus a ninth binary variable indicating which of the two boxes is closest to the viewer.

With uniform priors over the parameters in pixels:

$$P(D \mid \mathcal{H}_1) = \frac{1}{20}\frac{1}{20}\frac{1}{20}\frac{1}{16},$$

and (approximating unconstrained parameters as having half their values available):

$$P(D \mid \mathcal{H}_2) \simeq \frac{1}{20}\frac{1}{20}\frac{10}{20}\frac{1}{16}\frac{1}{20}\frac{1}{20}\frac{10}{20}\frac{1}{16}\frac{2}{2}.$$

The posterior probability ratio is (assuming equal prior probability):

$$\frac{P(D \mid \mathcal{H}_1)P(\mathcal{H}_1)}{P(D \mid \mathcal{H}_2)P(\mathcal{H}_2)} = 20 \times 2 \times 2 \times 16 \simeq 1000/1.$$

The four factors can be interpreted in terms of Occam factors. The more complex model pays two big Occam factors ($1/20$ and $1/16$) for the coincidences that the two box heights and colours match exactly, and two lesser Occam factors for the coincidence that both boxes happened to have one of their edges hidden behind a tree or behind each other.

</div>

### 28.3 Minimum Description Length (MDL)

A complementary view of Bayesian model comparison is obtained by replacing probabilities of events by the lengths in bits of messages that communicate the events without loss to a receiver. Message lengths $L(\mathbf{x})$ correspond to a probabilistic model over events $\mathbf{x}$ via the relations:

$$P(\mathbf{x}) = 2^{-L(\mathbf{x})}, \quad L(\mathbf{x}) = -\log_2 P(\mathbf{x}).$$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Minimum Description Length Principle)</span></p>

The **MDL principle** (Wallace and Boulton, 1968) states that one should prefer models that can communicate the data in the smallest number of bits. Consider a two-part message that states which model $\mathcal{H}$ is to be used, and then communicates the data $D$ within that model, to some pre-arranged precision $\delta D$. This produces a message of length $L(D, \mathcal{H}) = L(\mathcal{H}) + L(D \mid \mathcal{H})$. The lengths $L(\mathcal{H})$ for different $\mathcal{H}$ define an implicit prior $P(\mathcal{H})$ over the alternative models. Similarly $L(D \mid \mathcal{H})$ corresponds to a density $P(D \mid \mathcal{H})$. Thus, a procedure for assigning message lengths can be mapped onto posterior probabilities:

$$L(D, \mathcal{H}) = -\log P(\mathcal{H}) - \log (P(D \mid \mathcal{H}) \delta D) = -\log P(\mathcal{H} \mid D) + \text{const.}$$

</div>

Models with a small number of parameters have only a short parameter block but do not fit the data well, so the data message (a list of large residuals) is long. As the number of parameters increases, the parameter block lengthens, and the data message becomes shorter. There is an optimum model complexity for which the total message length is minimized.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(On-Line Learning and Cross-Validation)</span></p>

In cases where the data consist of a sequence of points $D = \mathbf{t}^{(1)}, \mathbf{t}^{(2)}, \ldots, \mathbf{t}^{(N)}$, the log evidence can be decomposed as a sum of "on-line" predictive performances:

$$\log P(D \mid \mathcal{H}) = \log P(\mathbf{t}^{(1)} \mid \mathcal{H}) + \log P(\mathbf{t}^{(2)} \mid \mathbf{t}^{(1)}, \mathcal{H}) + \cdots + \log P(\mathbf{t}^{(N)} \mid \mathbf{t}^{(1)} \ldots \mathbf{t}^{(N-1)}, \mathcal{H}).$$

This decomposition can be used to explain the difference between the evidence and "leave-one-out cross-validation" as measures of predictive ability. Cross-validation examines the average value of just the last term under random re-orderings of the data. The evidence, on the other hand, sums up how well the model predicted all the data, starting from scratch.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The "Bits-Back" Encoding Method)</span></p>

Another MDL thought experiment (Hinton and van Camp, 1993) involves incorporating random bits into our message. The data are communicated using a parameter block and a data block. The parameter vector sent is a random sample from the posterior, $P(\mathbf{w} \mid D, \mathcal{H}) = P(D \mid \mathbf{w}, \mathcal{H})P(\mathbf{w} \mid \mathcal{H})/P(D \mid \mathcal{H})$. This sample $\mathbf{w}$ is sent using a message length $L(\mathbf{w} \mid \mathcal{H}) = -\log[P(\mathbf{w} \mid \mathcal{H})\delta\mathbf{w}]$. The data are encoded relative to $\mathbf{w}$ with a message of length $L(D \mid \mathbf{w}, \mathcal{H}) = -\log[P(D \mid \mathbf{w}, \mathcal{H})\delta D]$. Once the data message has been received, the random bits used to generate the sample $\mathbf{w}$ from the posterior can be deduced by the receiver. The number of bits so recovered is $-\log[P(\mathbf{w} \mid D, \mathcal{H})\delta\mathbf{w}]$. The net description cost is therefore:

$$L(\mathbf{w} \mid \mathcal{H}) + L(D \mid \mathbf{w}, \mathcal{H}) - \text{"Bits back"} = -\log P(D \mid \mathcal{H}) - \log \delta D.$$

Thus this thought experiment has yielded the optimal description length.

</div>

## Chapter 29 -- Monte Carlo Methods

The last couple of chapters assumed that a Gaussian approximation to the probability distribution we are interested in is adequate. What if it is not? When the likelihood function is multimodal, or has nasty unboundedly-high spikes, maximizing the posterior and fitting a Gaussian is not always going to work. This difficulty with Laplace's method is one motivation for being interested in Monte Carlo methods. Monte Carlo methods provide a general-purpose set of tools with applications in Bayesian data modelling and many other fields.

### 29.1 The Problems to Be Solved

Monte Carlo methods are computational techniques that make use of random numbers. The aims of Monte Carlo methods are to solve one or both of the following problems.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The Two Problems of Monte Carlo)</span></p>

**Problem 1:** To generate samples $\{\mathbf{x}^{(r)}\}_{r=1}^{R}$ from a given probability distribution $P(\mathbf{x})$.

**Problem 2:** To estimate expectations of functions under this distribution, for example

$$\Phi = \langle \phi(\mathbf{x}) \rangle \equiv \int \mathrm{d}^N\mathbf{x}\; P(\mathbf{x})\phi(\mathbf{x}).$$

</div>

The probability distribution $P(\mathbf{x})$, which we call the **target density**, might be a distribution from statistical physics or a conditional distribution arising in data modelling -- for example, the posterior probability of a model's parameters given some observed data.

We concentrate on Problem 1 (sampling), because if we have solved it, then we can solve Problem 2 by using the random samples $\{\mathbf{x}^{(r)}\}_{r=1}^{R}$ to give the estimator

$$\hat{\Phi} \equiv \frac{1}{R} \sum_r \phi(\mathbf{x}^{(r)}).$$

If the vectors $\{\mathbf{x}^{(r)}\}_{r=1}^{R}$ are generated from $P(\mathbf{x})$ then the expectation of $\hat{\Phi}$ is $\Phi$. The variance of $\hat{\Phi}$ decreases as $\sigma^2/R$, where $\sigma^2$ is the variance of $\phi$:

$$\sigma^2 = \int \mathrm{d}^N\mathbf{x}\; P(\mathbf{x})(\phi(\mathbf{x}) - \Phi)^2.$$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Dimension-Independence of Monte Carlo Accuracy)</span></p>

The accuracy of the Monte Carlo estimate depends only on the variance of $\phi$, not on the dimensionality of the space sampled. The variance of $\hat{\Phi}$ goes as $\sigma^2/R$. So regardless of the dimensionality of $\mathbf{x}$, it may be that as few as a dozen independent samples $\{\mathbf{x}^{(r)}\}$ suffice to estimate $\Phi$ satisfactorily.

</div>

#### Why Is Sampling from $P(\mathbf{x})$ Hard?

We assume that the density from which we wish to draw samples, $P(\mathbf{x})$, can be evaluated, at least to within a multiplicative constant; that is, we can evaluate a function $P^*(\mathbf{x})$ such that

$$P(\mathbf{x}) = P^*(\mathbf{x})/Z.$$

There are two difficulties. First, we typically do not know the normalizing constant $Z = \int \mathrm{d}^N\mathbf{x}\; P^*(\mathbf{x})$. Second, even if we did know $Z$, drawing samples from $P(\mathbf{x})$ is still challenging in high-dimensional spaces, because there is no obvious way to sample from $P$ without enumerating most or all of the possible states.

#### Uniform Sampling Is Hopeless in High Dimensions

A high-dimensional distribution is often concentrated in a small region of the state space known as its **typical set** $T$, whose volume is given by $|T| \simeq 2^{H(\mathbf{X})}$, where $H(\mathbf{X})$ is the entropy of $P(\mathbf{x})$. Uniform sampling will only give a good estimate of $\Phi$ if we make the number of samples $R$ sufficiently large to hit the typical set at least once or twice.

For the Ising model on $N$ spins, the total size of the state space is $2^N$ and the typical set has size $2^H$. The number of samples required to hit the typical set once is of order $R_\text{min} \simeq 2^{N-H}$. At the critical temperature, $H \approx N/2$, so $R_\text{min} \simeq 2^{N/2}$, which for $N = 1000$ is about $10^{150}$ -- roughly the square of the number of particles in the universe. Thus uniform sampling is utterly useless for high-dimensional problems.

### 29.2 Importance Sampling

Importance sampling is not a method for generating samples from $P(\mathbf{x})$ (Problem 1); it is just a method for estimating the expectation of a function $\phi(\mathbf{x})$ (Problem 2). It can be viewed as a generalization of the uniform sampling method.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Importance Sampling)</span></p>

We assume we have a simpler **sampler density** $Q(x)$ from which we *can* generate samples and which we can evaluate to within a multiplicative constant. We generate $R$ samples $\{x^{(r)}\}_{r=1}^{R}$ from $Q(x)$. To account for the fact that we have sampled from the wrong distribution, we introduce **importance weights**:

$$w_r \equiv \frac{P^*(x^{(r)})}{Q^*(x^{(r)})},$$

which we use to adjust the "importance" of each point in our estimator:

$$\hat{\Phi} \equiv \frac{\sum_r w_r \phi(x^{(r)})}{\sum_r w_r}.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Pitfalls of Importance Sampling)</span></p>

A practical difficulty with importance sampling is that it is hard to estimate how reliable the estimator $\hat{\Phi}$ is. If the proposal density $Q(x)$ is small in a region where $|\phi(x)P^*(x)|$ is large, then even after many points $x^{(r)}$ have been generated, none of them will have fallen in that region. The estimate of $\hat{\Phi}$ would be drastically wrong, and there would be no indication in the *empirical* variance that the true variance of the estimator $\hat{\Phi}$ is large. An importance sampler should have **heavy tails**.

</div>

#### Importance Sampling in Many Dimensions

In high dimensions, importance sampling suffers from two difficulties. First, we need samples that lie in the typical set of $P$, which may take a long time unless $Q$ is a good approximation to $P$. Second, even if we obtain samples in the typical set, the weights associated with those samples are likely to vary by large factors: points in a typical set, although similar to each other, still differ by factors of order $\exp(\sqrt{N})$, so the weights will too, unless $Q$ is a near-perfect approximation to $P$.

The ratio of the largest weight to the median weight after one hundred samples will typically be in the ratio $w_r^\text{max}/w_r^\text{med} = \exp\!\left(\sqrt{2N}\right)$. In $N = 1000$ dimensions, the largest weight after one hundred samples is likely to be roughly $10^{19}$ times greater than the median weight.

### 29.3 Rejection Sampling

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Rejection Sampling)</span></p>

We assume a one-dimensional density $P(x) = P^*(x)/Z$ that is too complicated to sample from directly. We assume a simpler **proposal density** $Q(x)$ which we can evaluate (within a multiplicative factor $Z_Q$), and from which we can generate samples. We further assume that we know the value of a constant $c$ such that

$$c\,Q^*(x) > P^*(x), \quad \text{for all } x.$$

The algorithm proceeds as follows:

1. Generate $x$ from $Q(x)$.
2. Evaluate $c\,Q^*(x)$ and generate a uniformly distributed random variable $u$ from the interval $[0, c\,Q^*(x)]$.
3. Evaluate $P^*(x)$ and accept or reject: if $u > P^*(x)$ then $x$ is **rejected**; otherwise it is **accepted** and added to the set of samples $\{x^{(r)}\}$.

The accepted samples are **independent** samples from $P(x)$.

</div>

Rejection sampling will work best if $Q$ is a good approximation to $P$. If $Q$ is very different from $P$ then, for $cQ$ to exceed $P$ everywhere, $c$ will necessarily have to be large and the frequency of rejection will be large.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Rejection Sampling in Many Dimensions)</span></p>

In a high-dimensional problem it is very likely that the requirement that $c\,Q^*$ be an upper bound for $P^*$ will force $c$ to be so huge that acceptances will be very rare indeed. As a case study, consider a pair of $N$-dimensional Gaussian distributions with mean zero. If $\sigma_Q$ is $1\%$ larger than $\sigma_P$, we need to set

$$c = \frac{(2\pi\sigma_Q^2)^{N/2}}{(2\pi\sigma_P^2)^{N/2}} = \exp\!\left(N \ln \frac{\sigma_Q}{\sigma_P}\right).$$

With $N = 1000$ and $\sigma_Q/\sigma_P = 1.01$, we find $c = \exp(10) \simeq 20{,}000$, giving an acceptance rate of $1/c = 1/20{,}000$. In general, $c$ grows exponentially with $N$, so rejection sampling is not a practical technique for generating samples from high-dimensional distributions $P(\mathbf{x})$.

</div>

### 29.4 The Metropolis--Hastings Method

Importance sampling and rejection sampling work well only if the proposal density $Q(x)$ is similar to $P(x)$. In large and complex problems it is difficult to create a single density $Q(x)$ that has this property. The Metropolis--Hastings algorithm instead makes use of a proposal density $Q$ which **depends on the current state** $x^{(t)}$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Metropolis--Hastings)</span></p>

The density $Q(x'; x^{(t)})$ might be a simple distribution such as a Gaussian centred on the current $x^{(t)}$. A tentative new state $x'$ is generated from the proposal density $Q(x'; x^{(t)})$. To decide whether to accept the new state, we compute the quantity

$$a = \frac{P^*(x')}{P^*(x^{(t)})} \frac{Q(x^{(t)}; x')}{Q(x'; x^{(t)})}.$$

- If $a \geq 1$ then the new state is accepted.
- Otherwise, the new state is accepted with probability $a$.

If the step is accepted, we set $x^{(t+1)} = x'$. If the step is rejected, we set $x^{(t+1)} = x^{(t)}$.

</div>

Note the difference from rejection sampling: rejected points are not discarded. A rejection causes the current state to be written again onto the list. A Metropolis--Hastings simulation of $T$ iterations does not produce $T$ *independent* samples from the target distribution $P$. The samples are **dependent**.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Convergence of the Metropolis Method)</span></p>

It can be shown that for any positive $Q$ (that is, any $Q$ such that $Q(x'; x) > 0$ for all $x, x'$), as $t \to \infty$, the probability distribution of $x^{(t)}$ tends to $P(x) = P^*(x)/Z$. The Metropolis method is an example of a **Markov chain Monte Carlo** method (abbreviated MCMC). In contrast to rejection sampling, where the accepted points $\{x^{(r)}\}$ are independent samples from the desired distribution, Markov chain Monte Carlo methods involve a Markov process in which a sequence of states $\{x^{(t)}\}$ is generated, each sample $x^{(t)}$ having a probability distribution that depends on the previous value, $x^{(t-1)}$.

</div>

#### Random Walk Behaviour

Many implementations of the Metropolis method employ a proposal distribution with a length scale $\epsilon$ that is short relative to the longest length scale $L$ of the probable region. If $\epsilon$ is large, movement around the state space will be slow because transitions to low-probability states are unlikely to be accepted. If $\epsilon$ is small, the Metropolis method will explore by a **random walk**, which is also slow.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Rule of Thumb for Metropolis Iterations)</span></p>

If the largest length scale of the space of probable states is $L$, a Metropolis method whose proposal distribution generates a random walk with step size $\epsilon$ must be run for at least

$$T \simeq (L/\epsilon)^2$$

iterations to obtain an independent sample. If only a fraction $f$ of the steps are accepted on average, then this time is increased by a factor $1/f$.

</div>

#### Metropolis Method in High Dimensions

For a target distribution that is an $N$-dimensional Gaussian with standard deviations $\sigma^{\max}$ and $\sigma^{\min}$ along the longest and shortest lengthscales, and a spherical Gaussian proposal distribution of standard deviation $\epsilon$, the optimal $\epsilon$ must be similar to $\sigma^{\min}$. We will then need at least $T \simeq (\sigma^{\max}/\sigma^{\min})^2$ iterations to obtain an independent sample. Unlike rejection sampling and importance sampling, there is no catastrophic dependence on the dimensionality $N$ -- but the quadratic dependence on the lengthscale-ratio may still force us to make very lengthy simulations.

### 29.5 Gibbs Sampling

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Gibbs Sampling)</span></p>

**Gibbs sampling**, also known as the **heat bath method** or "Glauber dynamics", is a method for sampling from distributions over at least two dimensions. It can be viewed as a Metropolis method in which a sequence of proposal distributions $Q$ are defined in terms of the *conditional* distributions of the joint distribution $P(\mathbf{x})$. It is assumed that, whilst $P(\mathbf{x})$ is too complex to draw samples from directly, its conditional distributions $P(x_i \mid \{x_j\}_{j \neq i})$ are tractable to work with.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Gibbs Sampling for $K$ Variables)</span></p>

In the general case of a system with $K$ variables, a single iteration involves sampling one parameter at a time:

$$x_1^{(t+1)} \sim P(x_1 \mid x_2^{(t)}, x_3^{(t)}, \ldots, x_K^{(t)})$$

$$x_2^{(t+1)} \sim P(x_2 \mid x_1^{(t+1)}, x_3^{(t)}, \ldots, x_K^{(t)})$$

$$x_3^{(t+1)} \sim P(x_3 \mid x_1^{(t+1)}, x_2^{(t+1)}, \ldots, x_K^{(t)}), \quad \text{etc.}$$

Because Gibbs sampling is a Metropolis method (with every proposal always accepted), the probability distribution of $\mathbf{x}^{(t)}$ tends to $P(\mathbf{x})$ as $t \to \infty$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Properties of Gibbs Sampling)</span></p>

Gibbs sampling suffers from the same defect as simple Metropolis algorithms -- the state space is explored by a slow random walk, unless a fortuitous parameterization has been chosen that makes the probability distribution $P(\mathbf{x})$ separable. If two variables $x_1$ and $x_2$ are strongly correlated, having marginal densities of width $L$ and conditional densities of width $\epsilon$, then it will take at least about $(L/\epsilon)^2$ iterations to generate an independent sample from the target density.

However, Gibbs sampling involves no adjustable parameters, so it is an attractive strategy when one wants to get a model running quickly.

</div>

### 29.6 Terminology for Markov Chain Monte Carlo Methods

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Markov Chain)</span></p>

A **Markov chain** can be specified by an initial probability distribution $p^{(0)}(\mathbf{x})$ and a **transition probability** $T(\mathbf{x}'; \mathbf{x})$. The probability distribution of the state at the $(t+1)$th iteration of the Markov chain, $p^{(t+1)}(\mathbf{x})$, is given by

$$p^{(t+1)}(\mathbf{x}') = \int \mathrm{d}^N\mathbf{x}\; T(\mathbf{x}'; \mathbf{x}) p^{(t)}(\mathbf{x}).$$

</div>

When designing a Markov chain Monte Carlo method, we construct a chain with the following properties:

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Invariant Distribution and Ergodicity)</span></p>

1. The desired distribution $P(\mathbf{x})$ is an **invariant distribution** of the chain. A distribution $\pi(\mathbf{x})$ is an invariant distribution of the transition probability $T(\mathbf{x}'; \mathbf{x})$ if

   $$\pi(\mathbf{x}') = \int \mathrm{d}^N\mathbf{x}\; T(\mathbf{x}'; \mathbf{x})\pi(\mathbf{x}).$$

   An invariant distribution is an eigenvector of the transition probability matrix that has eigenvalue 1.

2. The chain must also be **ergodic**, that is,

   $$p^{(t)}(\mathbf{x}) \to \pi(\mathbf{x}) \text{ as } t \to \infty, \text{ for any } p^{(0)}(\mathbf{x}).$$

A chain might fail to be ergodic if its matrix is *reducible* (the state space contains two or more subsets of states that can never be reached from each other) or if it has a *periodic* set (for some initial conditions, $p^{(t)}(\mathbf{x})$ tends to a periodic limit-cycle instead of an invariant distribution).

</div>

#### Methods of Construction of Markov Chains

It is often convenient to construct $T$ by **mixing** or **concatenating** simple base transitions $B$ all of which satisfy

$$P(\mathbf{x}') = \int \mathrm{d}^N\mathbf{x}\; B(\mathbf{x}'; \mathbf{x}) P(\mathbf{x}),$$

for the desired density $P(\mathbf{x})$, i.e., they all have the desired density as an invariant distribution.

$T$ is a **mixture** of several base transitions $B_b(\mathbf{x}', \mathbf{x})$ if we make the transition by picking one of the base transitions at random:

$$T(\mathbf{x}', \mathbf{x}) = \sum_b p_b B_b(\mathbf{x}', \mathbf{x}).$$

$T$ is a **concatenation** of two base transitions $B_1(\mathbf{x}', \mathbf{x})$ and $B_2(\mathbf{x}', \mathbf{x})$ if we first make a transition to an intermediate state $\mathbf{x}''$ using $B_1$, and then a transition from $\mathbf{x}''$ to $\mathbf{x}'$ using $B_2$:

$$T(\mathbf{x}', \mathbf{x}) = \int \mathrm{d}^N\mathbf{x}''\; B_2(\mathbf{x}', \mathbf{x}'') B_1(\mathbf{x}'', \mathbf{x}).$$

#### Detailed Balance

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Detailed Balance)</span></p>

Many useful transition probabilities satisfy the **detailed balance** property:

$$T(\mathbf{x}_a; \mathbf{x}_b) P(\mathbf{x}_b) = T(\mathbf{x}_b; \mathbf{x}_a) P(\mathbf{x}_a), \quad \text{for all } \mathbf{x}_b \text{ and } \mathbf{x}_a.$$

This equation says that if we pick (by magic) a state from the target density $P$ and make a transition under $T$ to another state, it is just as likely that we will pick $\mathbf{x}_b$ and go from $\mathbf{x}_b$ to $\mathbf{x}_a$ as it is that we will pick $\mathbf{x}_a$ and go from $\mathbf{x}_a$ to $\mathbf{x}_b$. Markov chains that satisfy detailed balance are also called **reversible** Markov chains.

Detailed balance implies invariance of the distribution $P(\mathbf{x})$ under the Markov chain $T$, which is a necessary condition for the key property that we want from our MCMC simulation -- that the probability distribution of the chain should converge to $P(\mathbf{x})$.

</div>

The Metropolis method satisfies detailed balance. However, detailed balance is not an essential condition, and irreversible Markov chains can be useful in practice because they may have different random walk properties.

### 29.7 Slice Sampling

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Slice Sampling)</span></p>

**Slice sampling** (Neal, 1997a; Neal, 2003) is a Markov chain Monte Carlo method that has similarities to rejection sampling, Gibbs sampling and the Metropolis method. It can be applied wherever the target density $P^*(\mathbf{x})$ can be evaluated at any point $\mathbf{x}$; it has the advantage over simple Metropolis methods that it is more robust to the choice of parameters like step sizes.

A single transition $(x, u) \to (x', u')$ of a one-dimensional slice sampling algorithm:

1. Evaluate $P^*(x)$.
2. Draw a vertical coordinate $u' \sim \text{Uniform}(0, P^*(x))$.
3. Create a horizontal interval $(x_l, x_r)$ enclosing $x$.
4. Loop:
   - Draw $x' \sim \text{Uniform}(x_l, x_r)$.
   - Evaluate $P^*(x')$.
   - If $P^*(x') > u'$, break out of loop.
   - Else modify the interval $(x_l, x_r)$ by shrinking it towards $x$.

The interval $(x_l, x_r)$ is created by the **"stepping out" method**: step out in steps of length $w$ until we find endpoints $x_l$ and $x_r$ at which $P^*$ is smaller than $u'$. The interval is modified by the **"shrinking" method**: whenever a point $x'$ is rejected, shrink the interval so that one of the end points is $x'$, while ensuring the original point $x$ is still enclosed.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Properties of Slice Sampling)</span></p>

Like a standard Metropolis method, slice sampling gets around by a random walk, but whereas in the Metropolis method the choice of the step size is critical, in slice sampling the step size is **self-tuning**:

- If the initial interval size $w$ is too small by a factor $f$ compared with the width of the probable region, then the stepping-out procedure expands the interval. The cost of this stepping-out is only **linear** in $f$, whereas in the Metropolis method the computer-time scales as the **square** of $f$ if the step size is too small.
- If $w$ is too large by a factor $F$, the algorithm spends a time proportional to the logarithm of $F$ shrinking the interval down to the right size (the interval typically shrinks by a factor of 0.6 each time a point is rejected). In contrast, the Metropolis algorithm responds to a too-large step size by rejecting almost all proposals, so the rate of progress is exponentially bad in $F$.
- There are no rejections in slice sampling. The probability of staying in exactly the same place is very small.

</div>

An $N$-dimensional density $P(\mathbf{x}) \propto P^*(\mathbf{x})$ may be sampled with the help of the one-dimensional slice sampling method by picking a sequence of directions $\mathbf{y}^{(1)}, \mathbf{y}^{(2)}, \ldots$ and defining $\mathbf{x} = \mathbf{x}^{(t)} + x\mathbf{y}^{(t)}$. The directions may be chosen in various ways: for example, as in Gibbs sampling, the directions could be the coordinate axes; alternatively, the directions $\mathbf{y}^{(t)}$ may be selected at random in any manner such that the overall procedure satisfies detailed balance.

### 29.8 Practicalities

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Practical Issues for MCMC)</span></p>

**Can we predict how long a MCMC simulation will take to equilibrate?** By considering the random walks involved, we can obtain simple *lower bounds* on the time required for convergence. But predicting this time more precisely is a difficult problem, and most theoretical upper bounds are of little practical use.

**Can we diagnose or detect convergence in a running simulation?** This is also a difficult problem. There are a few practical tools available, but none of them is perfect.

**Can we speed up the convergence time and time between independent samples?** There is good news, as described in the next chapter, which describes the Hamiltonian Monte Carlo method, overrelaxation, and simulated annealing.

</div>

### 29.9 Further Practical Issues

#### Can the Normalizing Constant Be Evaluated?

If the target density $P(\mathbf{x})$ is given in the form of an unnormalized density $P^*(\mathbf{x})$ with $P(\mathbf{x}) = \frac{1}{Z}P^*(\mathbf{x})$, the value of $Z$ may well be of interest. Monte Carlo methods do not readily yield an estimate of this quantity. Techniques for evaluating $Z$ include:

1. Importance sampling and annealed importance sampling.
2. "Thermodynamic integration" during simulated annealing, the "acceptance ratio" method, and "umbrella sampling".
3. "Reversible jump Markov chain Monte Carlo".

#### How Many Samples Are Needed?

In many problems, we really only need about **twelve independent samples** from $P(\mathbf{x})$. If $\sigma^2$ is the variance of the cost $\phi(\mathbf{x})$, then the true cost is likely to differ by $\pm\sigma$ from the expectation $\Phi$. There is little point in knowing $\Phi$ to a precision finer than about $\sigma/3$. With $R = 12$ independent samples from $P(\mathbf{x})$, we can estimate $\Phi$ to a precision of $\sigma/\sqrt{12}$ -- which is smaller than $\sigma/3$.

#### Allocation of Resources

A typical Markov chain Monte Carlo experiment involves an initial period for adjusting control parameters (e.g., step sizes), followed by a "burn in" period during which the simulation converges to the desired distribution, and finally a sampling period where the state vector is recorded occasionally to create a list of states $\{\mathbf{x}^{(r)}\}_{r=1}^{R}$.

There are several possible strategies:

1. Make one long run, obtaining all $R$ samples from it.
2. Make a few medium-length runs with different initial conditions, obtaining some samples from each.
3. Make $R$ short runs, each starting from a different random initial condition, with only the final state being recorded.

The first strategy has the best chance of attaining convergence. The middle path is popular among MCMC experts because it avoids the inefficiency of discarding burn-in iterations in many runs, while still allowing one to detect problems with lack of convergence.

### 29.10 Summary

- Monte Carlo methods are a powerful tool that allow one to sample from any probability distribution that can be expressed in the form $P(\mathbf{x}) = \frac{1}{Z}P^*(\mathbf{x})$.
- Monte Carlo methods can answer virtually any query related to $P(\mathbf{x})$ by putting the query in the form $\int \phi(\mathbf{x})P(\mathbf{x}) \simeq \frac{1}{R}\sum_r \phi(\mathbf{x}^{(r)})$.
- In high-dimensional problems the only satisfactory methods are those based on Markov chains, such as the Metropolis method, Gibbs sampling and slice sampling. Gibbs sampling is an attractive method because it has no adjustable parameters but its use is restricted to cases where samples can be generated from the conditional distributions. Slice sampling is attractive because, whilst it has step-length parameters, its performance is not very sensitive to their values.
- Simple Metropolis algorithms and Gibbs sampling algorithms, although widely used, perform poorly because they explore the space by a slow random walk. Methods for speeding up Markov chain Monte Carlo simulations are discussed in the next chapter.
- Slice sampling does not avoid random walk behaviour, but it automatically chooses the largest appropriate step size, thus reducing the bad effects of the random walk compared with, say, a Metropolis method with a tiny step size.

## Chapter 30 -- Efficient Monte Carlo Methods

This chapter discusses several methods for reducing random walk behaviour in Metropolis methods. The aim is to reduce the time required to obtain effectively independent samples.

### 30.1 Hamiltonian Monte Carlo

The **Hamiltonian Monte Carlo** method is a Metropolis method, applicable to continuous state spaces, that makes use of gradient information to reduce random walk behaviour. (It was originally called hybrid Monte Carlo, for historical reasons.)

For many systems whose probability $P(\mathbf{x})$ can be written in the form

$$P(\mathbf{x}) = \frac{e^{-E(\mathbf{x})}}{Z},$$

not only $E(\mathbf{x})$ but also its gradient with respect to $\mathbf{x}$ can be readily evaluated. It seems wasteful to use a simple random-walk Metropolis method when this gradient is available -- the gradient indicates which direction one should go in to find states that have higher probability.

#### Overview of Hamiltonian Monte Carlo

In the Hamiltonian Monte Carlo method, the state space $\mathbf{x}$ is augmented by **momentum variables** $\mathbf{p}$, and there is an alternation of two types of proposal:

1. The first proposal randomizes the momentum variable, leaving the state $\mathbf{x}$ unchanged.
2. The second proposal changes both $\mathbf{x}$ and $\mathbf{p}$ using simulated Hamiltonian dynamics as defined by the Hamiltonian

$$H(\mathbf{x}, \mathbf{p}) = E(\mathbf{x}) + K(\mathbf{p}),$$

where $K(\mathbf{p})$ is a "kinetic energy" such as $K(\mathbf{p}) = \mathbf{p}^\top \mathbf{p}/2$. These two proposals are used to create (asymptotically) samples from the joint density

$$P_H(\mathbf{x}, \mathbf{p}) = \frac{1}{Z_H} \exp[-H(\mathbf{x}, \mathbf{p})] = \frac{1}{Z_H} \exp[-E(\mathbf{x})] \exp[-K(\mathbf{p})].$$

This density is separable, so the marginal distribution of $\mathbf{x}$ is the desired distribution $\exp[-E(\mathbf{x})]/Z$. Simply discarding the momentum variables, we obtain a sequence of samples $\lbrace \mathbf{x}^{(t)} \rbrace$ that asymptotically come from $P(\mathbf{x})$.

#### Details of Hamiltonian Monte Carlo

The first proposal, which can be viewed as a Gibbs sampling update, draws a new momentum from the Gaussian density $\exp[-K(\mathbf{p})]/Z_K$. This proposal is always accepted. During the second, dynamical proposal, the momentum variable determines where the state $\mathbf{x}$ goes, and the *gradient* of $E(\mathbf{x})$ determines how the momentum $\mathbf{p}$ changes, in accordance with the equations

$$\dot{\mathbf{x}} = \mathbf{p}, \qquad \dot{\mathbf{p}} = -\frac{\partial E(\mathbf{x})}{\partial \mathbf{x}}.$$

Because of the persistent motion of $\mathbf{x}$ in the direction of the momentum $\mathbf{p}$ during each dynamical proposal, the state of the system tends to move a distance that goes *linearly* with the computer time, rather than as the square root.

The second proposal is accepted in accordance with the Metropolis rule. If the simulation of the Hamiltonian dynamics is numerically perfect then the proposals are accepted every time, because the total energy $H(\mathbf{x}, \mathbf{p})$ is a constant of the motion and the acceptance ratio $a$ equals one. If the simulation is imperfect (because of finite step sizes for example), then some of the dynamical proposals will be rejected.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Hamiltonian Monte Carlo -- Leapfrog)</span></p>

The leapfrog algorithm for simulating Hamiltonian dynamics proceeds as follows:

1. Set the gradient $\mathbf{g} = \nabla E(\mathbf{x})$ and the energy $E = E(\mathbf{x})$ using the initial $\mathbf{x}$.
2. For each of $L$ iterations:
   - Draw a random momentum $\mathbf{p} \sim \text{Normal}(0, I)$.
   - Evaluate $H = \mathbf{p}^\top \mathbf{p}/2 + E$.
   - Make $\tau$ "leapfrog" steps, each consisting of:
     - Half-step in $\mathbf{p}$: $\mathbf{p} \leftarrow \mathbf{p} - \epsilon \, \mathbf{g}/2$.
     - Full step in $\mathbf{x}$: $\mathbf{x}_\text{new} \leftarrow \mathbf{x} + \epsilon \, \mathbf{p}$.
     - Recompute the gradient: $\mathbf{g}_\text{new} = \nabla E(\mathbf{x}_\text{new})$.
     - Half-step in $\mathbf{p}$: $\mathbf{p} \leftarrow \mathbf{p} - \epsilon \, \mathbf{g}_\text{new}/2$.
   - Compute $H_\text{new} = \mathbf{p}^\top \mathbf{p}/2 + E_\text{new}$ and $\Delta H = H_\text{new} - H$.
   - Accept the proposal if $\Delta H < 0$, or with probability $\exp(-\Delta H)$ otherwise.

</div>

### 30.2 Overrelaxation

The method of **overrelaxation** is a method for reducing random walk behaviour in Gibbs sampling. Overrelaxation was originally introduced for systems in which all the conditional distributions are Gaussian.

#### Overrelaxation for Gaussian Conditional Distributions

In ordinary Gibbs sampling, one draws the new value $x_i^{(t+1)}$ of the current variable $x_i$ from its conditional distribution, ignoring the old value $x_i^{(t)}$. The state makes lengthy random walks in cases where the variables are strongly correlated.

In Adler's (1981) overrelaxation method, one instead samples $x_i^{(t+1)}$ from a Gaussian that is biased to the *opposite* side of the conditional distribution. If the conditional distribution of $x_i$ is $\text{Normal}(\mu, \sigma^2)$ and the current value of $x_i$ is $x_i^{(t)}$, then Adler's method sets $x_i$ to

$$x_i^{(t+1)} = \mu + \alpha(x_i^{(t)} - \mu) + (1 - \alpha^2)^{1/2} \sigma \nu,$$

where $\nu \sim \text{Normal}(0, 1)$ and $\alpha$ is a parameter between $-1$ and $1$, usually set to a negative value. (If $\alpha$ is positive, then the method is called under-relaxation.)

A single iteration of Adler's overrelaxation, like one of Gibbs sampling, updates each variable in turn. The transition matrix $T(\mathbf{x}'; \mathbf{x})$ defined by a complete update of all variables in some fixed order does *not* satisfy detailed balance. Each individual transition for one coordinate does satisfy detailed balance -- so the overall chain gives a valid sampling strategy converging to $P(\mathbf{x})$ -- but when we form a chain by applying the individual transitions in a fixed sequence, the overall chain is not reversible. This temporal asymmetry is the key to why overrelaxation can be beneficial: positively correlated variables will evolve in a directed manner instead of by random walk.

#### Ordered Overrelaxation

The overrelaxation method has been generalized by Neal (1995) whose **ordered overrelaxation** method is applicable to *any* system where Gibbs sampling is used. In ordered overrelaxation, instead of taking one sample from the conditional distribution $P(x_i \mid \lbrace x_j \rbrace_{j \neq i})$, we create $K$ such samples $x_i^{(1)}, x_i^{(2)}, \ldots, x_i^{(K)}$, where $K$ might be set to ten or twenty. The points $\lbrace x_i^{(k)} \rbrace$ are then sorted numerically, and the current value of $x_i$ is inserted into the sorted list, giving a list of $K + 1$ points. If $\kappa$ is the rank of the current value of $x_i$ in the list, we set $x_i'$ to the value that is an equal distance from the other end of the list, that is, the value with rank $K - \kappa$. When $K = 1$, we obtain ordinary Gibbs sampling. For practical purposes Neal estimates that ordered overrelaxation may speed up a simulation by a factor of ten or twenty.

### 30.3 Simulated Annealing

A third technique for speeding convergence is **simulated annealing**. In simulated annealing, a "temperature" parameter is introduced which, when large, allows the system to make transitions that would be improbable at temperature 1. The temperature is set to a large value and gradually reduced to 1. This procedure is supposed to reduce the chance that the simulation gets stuck in an unrepresentative probability island.

We assume that we wish to sample from a distribution of the form $P(\mathbf{x}) = \frac{e^{-E(\mathbf{x})}}{Z}$ where $E(\mathbf{x})$ can be evaluated. In the simplest simulated annealing method, we instead sample from the distribution

$$P_T(\mathbf{x}) = \frac{1}{Z(T)} e^{-\frac{E(\mathbf{x})}{T}}$$

and decrease $T$ gradually to 1. Often the energy function can be separated into two terms,

$$E(\mathbf{x}) = E_0(\mathbf{x}) + E_1(\mathbf{x}),$$

of which the first term is "nice" (e.g., a separable function of $\mathbf{x}$) and the second is "nasty". In these cases, a better simulated annealing method might make use of the distribution

$$P_T'(\mathbf{x}) = \frac{1}{Z'(T)} e^{-E_0(\mathbf{x}) - E_1(\mathbf{x})/T}$$

with $T$ gradually decreasing to 1, so that at high temperatures the distribution reverts to a well-behaved distribution defined by $E_0$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Simulated Annealing as Optimization)</span></p>

Simulated annealing is often used as an **optimization** method, where the aim is to find an $\mathbf{x}$ that minimizes $E(\mathbf{x})$, in which case the temperature is decreased to zero rather than to 1.

As a Monte Carlo method, simulated annealing as described above doesn't sample exactly from the right distribution, because there is no guarantee that the probability of falling into one basin of the energy is equal to the total probability of all the states in that basin. The closely related "**simulated tempering**" method (Marinari and Parisi, 1992) corrects the biases introduced by the annealing process by making the temperature itself a random variable that is updated in Metropolis fashion during the simulation. Neal's (1998) "**annealed importance sampling**" method removes the biases introduced by annealing by computing importance weights for each generated point.

</div>

### 30.4 Skilling's Multi-State Leapfrog Method

A fourth method for speeding up Monte Carlo simulations, due to John Skilling, has a similar spirit to overrelaxation, but works in more dimensions. This method is applicable to sampling from a distribution over a continuous state space, and the sole requirement is that the energy $E(\mathbf{x})$ should be easy to evaluate. The gradient is not used. This leapfrog method is not intended to be used on its own but rather in sequence with other Monte Carlo operators.

Instead of moving just one state vector $\mathbf{x}$ around the state space, Skilling's leapfrog method simultaneously maintains a *set* of $S$ state vectors $\lbrace \mathbf{x}^{(s)} \rbrace$, where $S$ might be six or twelve. The aim is that all $S$ of these vectors will represent independent samples from the same distribution $P(\mathbf{x})$.

Skilling's leapfrog makes a proposal for the new state $\mathbf{x}^{(s)'}$, which is accepted or rejected in accordance with the Metropolis method, by leapfrogging the current state $\mathbf{x}^{(s)}$ over another state vector $\mathbf{x}^{(t)}$:

$$\mathbf{x}^{(s)'} = \mathbf{x}^{(t)} + (\mathbf{x}^{(t)} - \mathbf{x}^{(s)}) = 2\mathbf{x}^{(t)} - \mathbf{x}^{(s)}.$$

The acceptance probability depends only on the change in energy of $\mathbf{x}^{(s)}$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why the Leapfrog Is a Good Idea)</span></p>

Imagine that the target density $P(\mathbf{x})$ has strong correlations -- for example, the density might be a needle-like Gaussian with width $\epsilon$ and length $L\epsilon$, where $L \gg 1$. Motion around such a density by standard methods proceeds by a slow random walk.

Under Skilling's leapfrog method, a typical first move will take the point a little outside the current ball, perhaps doubling its distance from the centre of the ball. After all the points have had a chance to move, the ball will have increased in size by a factor of two or so in all dimensions. The typical distance travelled in the long dimension grows **exponentially** with the number of iterations. It will only take a number of iterations proportional to $\log L / \log(1.1)$ for the long dimension to be explored.

However, for a separable $N$-dimensional Gaussian at equilibrium, the typical change in energy when a leapfrog move is made is $+2N$. The probability of acceptance therefore scales as $e^{-2N}$, implying that Skilling's method is not effective in very high-dimensional problems. Nevertheless it has the impressive advantage that its convergence properties are independent of the strength of correlations between the variables.

</div>

### 30.5 Monte Carlo Algorithms as Communication Channels

It may be a helpful perspective, when thinking about speeding up Monte Carlo methods, to think about the information that is being communicated. Two communications take place when a sample from $P(\mathbf{x})$ is being generated:

1. The selection of a particular $\mathbf{x}$ from $P(\mathbf{x})$ necessarily requires that at least $\log 1/P(\mathbf{x})$ random bits be consumed.
2. The generation of a sample conveys information about $P(\mathbf{x})$ from the subroutine that is able to evaluate $P^*(\mathbf{x})$.

In a dumb Metropolis method, the proposals $Q(\mathbf{x}'; \mathbf{x})$ have nothing to do with $P(\mathbf{x})$. Properties of $P(\mathbf{x})$ are only involved in the algorithm at the acceptance step, when the ratio $P^*(\mathbf{x}')/P^*(\mathbf{x})$ is computed. The channel from the true distribution $P(\mathbf{x})$ to the user who is interested in computing properties of $P(\mathbf{x})$ thus passes through a bottleneck: all the information about $P$ is conveyed by the string of acceptances and rejections.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Optimal Acceptance Rate)</span></p>

The information-theoretic viewpoint offers a simple justification for the widely-adopted rule of thumb which states that the parameters of a dumb Metropolis method should be adjusted such that the acceptance rate is about one half. Let the acceptance history be the binary string of accept or reject decisions, $\mathbf{a}$. The information learned about $P(\mathbf{x})$ after the algorithm has run for $T$ steps is less than or equal to the information content of $\mathbf{a}$, since all information about $P$ is mediated by $\mathbf{a}$. The information content of $\mathbf{a}$ is upper-bounded by $T H_2(f)$, where $f$ is the acceptance rate. This bound on information acquired about $P$ is maximized by setting $f = 1/2$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Information-Theoretic Bound on Metropolis)</span></p>

The fastest progress that a standard Metropolis method can make, in information terms, is about **one bit per iteration**. This gives a strong motivation for reducing random-walk behaviour. The methods reviewed in this chapter -- Hamiltonian Monte Carlo, overrelaxation, Skilling's leapfrog -- also aim to speed up the rate at which information is acquired.

</div>

### 30.6 Multi-State Methods

In a multi-state method, multiple parameter vectors $\mathbf{x}$ are maintained; they evolve individually under moves such as Metropolis and Gibbs; there are also interactions among the vectors. The intention is either that eventually all the vectors $\mathbf{x}$ should be samples from $P(\mathbf{x})$ (as illustrated by Skilling's leapfrog method), or that information associated with the final vectors $\mathbf{x}$ should allow us to approximate expectations under $P(\mathbf{x})$, as in importance sampling.

#### Genetic Methods

Genetic algorithms can be viewed as Monte Carlo algorithms. A population of $R$ vectors is maintained with target $P^*(\lbrace \mathbf{x}^{(r)} \rbrace_1^R) = \prod P^*(\mathbf{x}^{(r)})$. A genetic algorithm involves moves of two or three types:

1. **Individual moves:** one state vector is perturbed, $\mathbf{x}^{(r)} \to \mathbf{x}^{(r)'}$, using any of the Monte Carlo methods previously mentioned.
2. **Crossover moves:** of the form $\mathbf{x}, \mathbf{y} \to \mathbf{x}', \mathbf{y}'$; the progeny $\mathbf{x}'$ receives half his state vector from one parent $\mathbf{x}$, and half from the other, $\mathbf{y}$. The crossover proposal is accepted or rejected using the Metropolis rule with the ratio $\frac{P^*(\mathbf{x}')P^*(\mathbf{y}')}{P^*(\mathbf{x})P^*(\mathbf{y})}$.

The secret of success in a genetic algorithm is that the parameter $\mathbf{x}$ must be encoded in such a way that the crossover of two independent states $\mathbf{x}$ and $\mathbf{y}$, both of which have good fitness $P^*$, should have a reasonably good chance of producing progeny who are equally fit. In practice, this constraint is hard to satisfy in many problems, which is why genetic algorithms are mainly talked about and hyped up, and rarely used by serious experts.

To obtain a speed-up compared with standard Monte Carlo methods, what's required is two things: multiplication and death; and at least one of these must operate *selectively*. Either we must kill off the less-fit state vectors, or we must allow the more-fit state vectors to give rise to more offspring.

#### Particle Filters

Particle filters, which are particularly popular in inference problems involving temporal tracking, are multistate methods that mix the ideas of importance sampling and Markov chain Monte Carlo.

### 30.7 Methods That Do Not Necessarily Help

It is common practice to use *many* initial conditions for a particular Markov chain. If you are worried about sampling well from a complicated density $P(\mathbf{x})$, can you ensure the states produced by the simulations are well distributed about the typical set of $P(\mathbf{x})$ by ensuring the initial points are "well distributed about the whole state space"?

The answer is, unfortunately, no. In hierarchical Bayesian models, for example, a large number of parameters $\lbrace x_n \rbrace$ may be coupled together via another parameter $\beta$ (a hyperparameter). If we distribute the points $\lbrace x_n \rbrace$ widely, what we are actually doing is favouring an initial value of the noise level $1/\beta$ that is *large*. The random walk of the parameter $\beta$ will thus tend, after the first drawing of $\beta$ from $P(\beta \mid x_n)$, always to start off from one end of the $\beta$-axis.

### 30.8 Summary

- **Hamiltonian Monte Carlo** uses gradient information and momentum variables to make large moves through state space, traversing distances that grow *linearly* with computation time rather than as the square root.
- **Overrelaxation** reduces random walk behaviour in Gibbs sampling by biasing updates to the opposite side of the conditional distribution. Ordered overrelaxation generalizes this to any system where Gibbs sampling is used and may speed up simulations by a factor of ten or twenty.
- **Simulated annealing** introduces a temperature parameter to help the simulation escape unrepresentative probability islands, though it does not sample exactly from the target distribution without corrections (simulated tempering, annealed importance sampling).
- **Skilling's multi-state leapfrog** maintains multiple state vectors and makes exponentially fast progress along correlated dimensions, but its acceptance probability degrades exponentially with dimension.
- The **information-theoretic** perspective suggests that a standard Metropolis method can acquire at most about one bit per iteration, motivating the use of smarter methods.
- **Multi-state methods** including genetic algorithms and particle filters maintain populations of state vectors with interactions, but obtaining genuine speed-ups requires selective multiplication and death.

## About Chapter 31

Some of the neural network models that we will encounter are related to Ising models, which are idealized magnetic systems. Ising models are also related to several other topics in information theory: exact tree-based computation methods (Chapter 25), crude models for binary images, and two-dimensional constrained channels (Chapter 17).

## Chapter 31 -- Ising Models

An Ising model is an array of spins (e.g., atoms that can take states $\pm 1$) that are magnetically coupled to each other. If one spin is, say, in the $+1$ state then it is energetically favourable for its immediate neighbours to be in the same state (ferromagnet) or in the opposite state (antiferromagnet).

### Setup and Energy Function

Let the state $\mathbf{x}$ of an Ising model with $N$ spins be a vector in which each component $x_n$ takes values $-1$ or $+1$. If two spins $m$ and $n$ are neighbours we write $(m, n) \in \mathcal{N}$. The coupling between neighbouring spins is $J$. We define $J_{mn} = J$ if $m$ and $n$ are neighbours and $J_{mn} = 0$ otherwise.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Ising Model Energy)</span></p>

The energy of a state $\mathbf{x}$ is

$$E(\mathbf{x}; J, H) = -\left[\frac{1}{2}\sum_{m,n} J_{mn} x_m x_n + \sum_n H x_n\right],$$

where $H$ is the applied field. If $J > 0$ the model is **ferromagnetic**, and if $J < 0$ it is **antiferromagnetic**. The factor of $1/2$ accounts for each pair being counted twice.

</div>

At equilibrium at temperature $T$, the probability that the state is $\mathbf{x}$ is

$$P(\mathbf{x} \mid \beta, J, H) = \frac{1}{Z(\beta, J, H)} \exp[-\beta E(\mathbf{x}; J, H)],$$

where $\beta = 1/k_\text{B} T$, $k_\text{B}$ is Boltzmann's constant, and

$$Z(\beta, J, H) \equiv \sum_{\mathbf{x}} \exp[-\beta E(\mathbf{x}; J, H)].$$

#### Relevance of Ising Models

Ising models are relevant for three reasons:

1. **Phase transitions:** Ising models are important as models of magnetic systems that have a phase transition. By the theory of universality, all systems with the same dimension and the same symmetries have equivalent critical properties.
2. **Spin glasses and neural networks:** If we generalize the energy function to allow non-constant couplings $J_{mn}$ and applied fields $h_n$, we obtain models known as "spin glasses" (to physicists) and "Hopfield networks" or "Boltzmann machines" (to the neural network community).
3. **Statistical model:** The Ising model is also useful as a statistical model in its own right.

### Some Remarkable Relationships in Statistical Physics

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Heat Capacity and Energy Fluctuations)</span></p>

The heat capacity of a system, defined as $C \equiv \frac{\partial}{\partial T}\bar{E}$, is intimately related to energy fluctuations at constant temperature. Starting from the partition function $Z = \sum_{\mathbf{x}} \exp(-\beta E(\mathbf{x}))$, we can derive:

- The mean energy: $\frac{\partial \ln Z}{\partial \beta} = -\bar{E}$.
- The variance of the energy: $\frac{\partial^2 \ln Z}{\partial \beta^2} = \langle E^2 \rangle - \bar{E}^2 = \text{var}(E)$.
- The heat capacity:

$$C = \frac{\text{var}(E)}{k_\text{B} T^2} = k_\text{B} \beta^2 \, \text{var}(E).$$

Thus if we can observe the variance of the energy of a system at equilibrium, we can estimate its heat capacity.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The $1/T^2$ Factor)</span></p>

The $1/T^2$ factor in the heat capacity can be viewed as an accident of history. If temperature scales had been defined using $\beta = \frac{1}{k_\text{B}T}$, then the definition of heat capacity would be $C^{(\beta)} \equiv \frac{\partial \bar{E}}{\partial \beta} = \text{var}(E)$, and heat capacity and fluctuations would be identical quantities.

</div>

### 31.1 Ising Models -- Monte Carlo Simulation

We study two-dimensional planar Ising models using a simple Gibbs-sampling method. Starting from some initial state, a spin $n$ is selected at random, and the probability that it should be $+1$ given the state of the other spins and the temperature is computed:

$$P(+1 \mid b_n) = \frac{1}{1 + \exp(-2\beta b_n)},$$

where $\beta = 1/k_\text{B}T$ and $b_n$ is the local field

$$b_n = \sum_{m:(m,n)\in\mathcal{N}} J x_m + H.$$

The factor of 2 appears because the two spin states are $\lbrace +1, -1 \rbrace$ rather than $\lbrace +1, 0 \rbrace$. Spin $n$ is set to $+1$ with that probability, and otherwise to $-1$. After sufficiently many iterations, this procedure converges to the equilibrium distribution.

An alternative to the Gibbs sampling formula is the Metropolis algorithm, in which we consider the change in energy that results from flipping the chosen spin from its current state $x_n$:

$$\Delta E = 2 x_n b_n,$$

and adopt this change with probability

$$P(\text{accept}; \Delta E, \beta) = \begin{cases} 1 & \Delta E \le 0 \\ \exp(-\beta \Delta E) & \Delta E > 0. \end{cases}$$

#### Rectangular Geometry

For a rectangular Ising model with periodic boundary conditions and $H = 0$:

- **At low temperatures:** The system is expected to be in a ground state. The rectangular Ising model with $J = 1$ has two ground states: the all $+1$ state and the all $-1$ state. The energy per spin of either ground state is $-2$.
- **At high temperatures:** The spins are independent, all states are equally probable, and the energy is expected to fluctuate around a mean of $0$ with a standard deviation proportional to $1/\sqrt{N}$.

The basic picture: the energy rises monotonically with temperature. As we increase the number of spins to 100, new details emerge. The fluctuations at large temperature decrease as $1/\sqrt{N}$, and the fluctuations at intermediate temperature become relatively *bigger*. This is the signature of a "collective phenomenon" -- a **phase transition**. Only systems with infinite $N$ show true phase transitions, but with $N = 100$ we already see a hint of the critical fluctuations.

#### Contrast with Schottky Anomaly

A peak in the heat capacity, as a function of temperature, occurs in any system that has a finite number of energy levels; a peak is not in itself evidence of a phase transition. Such peaks were viewed as anomalies in classical thermodynamics, since "normal" systems with infinite numbers of energy levels have heat capacities that are either constant or increasing functions of temperature.

For a two-level system with states $x = 0$ (energy 0) and $x = 1$ (energy $\epsilon$), the mean energy is

$$\bar{E}(\beta) = \epsilon \frac{\exp(-\beta \epsilon)}{1 + \exp(-\beta \epsilon)} = \epsilon \frac{1}{1 + \exp(\beta \epsilon)}$$

and the heat capacity is

$$C = \frac{\epsilon^2}{k_\text{B}T^2} \frac{\exp(\beta \epsilon)}{[1 + \exp(\beta \epsilon)]^2}.$$

The take-home message is that whilst Schottky anomalies do have a peak in the heat capacity, there is *no* peak in their fluctuations; the variance of the energy simply increases monotonically with temperature. It is a peak in the *fluctuations* that is interesting, rather than a peak in the heat capacity, and the Ising model has such a peak in its fluctuations.

#### Rectangular Ising Model with $J = -1$

The ground states of the antiferromagnetic rectangular Ising model ($J = -1$) are the two checkerboard patterns. The two systems ($J = +1$ and $J = -1$) are equivalent under a checkerboard symmetry operation: flipping all the spins on the black squares of an infinite checkerboard maps a $J = +1$ configuration to $J = -1$ with the same energy.

However, there is a subtlety for **finite grids with periodic boundary conditions**: if the size of the grid in any direction is *odd*, the checkerboard operation is no longer a symmetry relating $J = +1$ to $J = -1$ because the checkerboard doesn't match up at the boundaries. This means that for systems of odd size, the ground state of $J = -1$ will have degeneracy greater than 2, and we expect qualitative differences between $J = \pm 1$ in odd-sized systems.

#### Triangular Ising Model

For the triangular Ising model with $J = -1$, the case is radically different from the rectangular counterpart: *there is no unfrustrated ground state*. In any state, there *must* be frustrations -- pairs of neighbours who have the same sign as each other. Every set of three mutually neighbouring spins must be in a state of frustration. Thus we expect this system to have a non-zero entropy at absolute zero, violating the third law of thermodynamics.

## Chapter 32 -- Exact Monte Carlo Sampling

### 32.1 The Problem with Monte Carlo Methods

For high-dimensional problems, the most widely used random sampling methods are Markov chain Monte Carlo methods like the Metropolis method, Gibbs sampling, and slice sampling. The problem with all these methods is this: yes, a given algorithm can be guaranteed to produce samples from the target density $P(\mathbf{x})$ asymptotically, "once the chain has converged to the equilibrium distribution". But if one runs the chain for too short a time $T$, then the samples will come from some other distribution $P^{(T)}(\mathbf{x})$.

For how long must the Markov chain be run before it has "converged"? This question is usually very hard to answer. However, the pioneering work of Propp and Wilson (1996) allows one, for certain chains, to answer this very question; furthermore Propp and Wilson show how to obtain "exact" samples from the target density.

### 32.2 Exact Sampling Concepts

Propp and Wilson's **exact sampling method** (also known as "perfect simulation" or "coupling from the past") depends on three ideas.

#### Coalescence of Coupled Markov Chains

If several Markov chains starting from different initial conditions share a single random-number generator, then their trajectories in state space may *coalesce*; and, having coalesced, will not separate again. If *all* initial conditions lead to trajectories that coalesce into a single trajectory, then we can be sure that the Markov chain has "forgotten" its initial condition.

#### Coupling from the Past

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why Forward Sampling Fails)</span></p>

The state of the system at the moment when complete coalescence occurs is *not* a valid sample from the equilibrium distribution. For example, in a Metropolis chain on states $\lbrace 0, 1, \ldots, 20 \rbrace$, final coalescence always occurs when the state is against one of the two walls, because trajectories merge only at the walls. So sampling forward in time until coalescence occurs is not a valid method.

</div>

The second key idea of exact sampling is that we can obtain exact samples by sampling *from a time $T_0$ in the past, up to the present*. If coalescence has occurred, the present sample is an unbiased sample from the equilibrium distribution; if not, we restart the simulation from a time $T_0$ further into the past, *reusing the same random numbers*. The simulation is repeated at a sequence of ever more distant times $T_0$, with a doubling of $T_0$ from one run to the next being a convenient choice. When coalescence occurs at a time before "the present", we can record $x(0)$ as an **exact sample** from the equilibrium distribution of the Markov chain.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Reusing Random Numbers Is Essential)</span></p>

The method of coupling from the past requires that the random numbers used at each time $t$ are *identical* across all runs with different starting times $T_0$. The trajectories produced from $t = -50$ to $t = 0$ by a run that started from $T_0 = -100$ are identical to a *subset* of the trajectories in the first simulation with $T_0 = -50$. This ensures that once coalescence is detected at any starting time, all earlier starting times would also have coalesced to the same final point.

</div>

This method is important because it allows us to obtain exact samples from the equilibrium distribution; but, as described here, it is of little practical use, since we are obliged to simulate chains starting in *all* initial states. In realistic sampling problems there will be an utterly enormous number of states -- think of the $2^{1000}$ states of a system of 1000 binary spins. The whole point of introducing Monte Carlo methods was to try to avoid having to visit all the states of such a system!

#### Monotonicity

The third trick of Propp and Wilson, which makes the exact sampling method useful in practice, is the idea that for some Markov chains, it may be possible to detect coalescence of all trajectories *without simulating all those trajectories*. This property holds when the chain has the property that *two trajectories never cross*. So if we simply track the two trajectories starting from the leftmost and rightmost states, we will know that coalescence of *all* trajectories has occurred when those *two* trajectories coalesce.

### 32.3 Exact Sampling from Interesting Distributions

In the toy problem we studied, the states could be put in a one-dimensional order such that no two trajectories crossed. The states of many interesting state spaces can also be put into a **partial order** and coupled Markov chains can be found that respect this partial order.

As an example, consider the Gibbs sampling method applied to a ferromagnetic Ising spin system, with the partial ordering of states being defined thus: state $\mathbf{x}$ is "greater than or equal to" state $\mathbf{y}$ if $x_i \ge y_i$ for all spins $i$. The maximal and minimal states are the all-up and all-down states.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Gibbs Sampling Coupling for Ising Models)</span></p>

The Markov chains are coupled together by having all chains update the same spin $i$ at each time step and having all chains share a common sequence of random numbers $u$:

1. Compute the local field $a_i := \sum_j J_{ij} x_j$.
2. Draw $u$ from $\text{Uniform}(0, 1)$.
3. If $u < 1/(1 + e^{-2a_i})$, set $x_i := +1$; else set $x_i := -1$.

Propp and Wilson (1996) show that exact samples can be generated for this system, although the time to find exact samples is large if the Ising model is below its critical temperature, since the Gibbs sampling method itself is slowly-mixing under these conditions.

</div>

Propp and Wilson have improved on this method for the Ising model by using a Markov chain called the **single-bond heat bath algorithm** to sample from a related model called the **random cluster model**; they show that exact samples from the random cluster model can be obtained rapidly and can be converted into exact samples from the Ising model. Their ground-breaking paper includes an exact sample from a 16-million-spin Ising model at its critical temperature.

#### Generalization to Non-Attractive Distributions (Summary States)

The method of Propp and Wilson, as sketched above, can be applied only to probability distributions that are "attractive" -- for practical purposes, to spin systems in which all the couplings are positive, and to a few special spin systems with negative couplings. It cannot be applied to general spin systems in which some couplings are negative, because the trajectories followed by the all-up and all-down states are not guaranteed to be upper and lower bounds for the set of all trajectories.

The idea of the **summary state** version of exact sampling is that we keep track of bounds on the set of all trajectories, and detect when these bounds are equal, so as to find exact samples. Instead of simulating two trajectories, each of which moves in a state space $\lbrace -1, +1 \rbrace^N$, we simulate one *trajectory envelope* in an augmented state space $\lbrace -1, +1, ? \rbrace^N$, where the symbol $?$ denotes "either $-1$ or $+1$". We call the state of this augmented system the "summary state".

The update rule at each step takes a single spin, enumerates all possible states of the neighbouring spins that are compatible with the current summary state, and for each of these local scenarios, computes the new value ($+$ or $-$) of the spin using Gibbs sampling (coupled to a common random number $u$). If all these new values agree, then the new value of the updated spin in the summary state is set to the unanimous value; otherwise, the new value in the summary state is "?". The initial condition, at time $T_0$, is given by setting all the spins in the summary state to "?", which corresponds to considering all possible start configurations.

Coalescence is detected when all the "?" symbols have disappeared. The summary state method can be applied to general spin systems with any couplings, though the time for coalescence to be *detected* may be considerably larger than the actual time taken for the underlying Markov chain to coalesce.

#### Other Uses for Coupling

The idea of coupling together Markov chains by having them share a random number generator has other applications beyond exact sampling. The accuracy of estimates obtained from a Markov chain Monte Carlo simulation, using the estimator

$$\hat{\Phi}_P \equiv \frac{1}{T} \sum_t \phi(\mathbf{x}^{(t)}),$$

can be improved by coupling the chain of interest, which converges to $P$, to a second chain, which generates samples from a second, simpler distribution, $Q$. The coupling must be set up so that the states of the two chains are strongly correlated. If $\hat{\Phi}_Q$ is an overestimate then it is likely that $\hat{\Phi}_P$ will be an overestimate too. The difference $(\hat{\Phi}_Q - \Phi_Q)$ can thus be used to correct $\hat{\Phi}_P$.

## Chapter 33 -- Variational Methods

Variational methods are an important technique for the approximation of complicated probability distributions, having applications in statistical physics, data modelling and neural networks.

### 33.1 Variational Free Energy Minimization

One method for approximating a complex distribution in a physical system is **mean field theory**. Mean field theory is a special case of a general **variational free energy** approach of Feynman and Bogoliubov.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Relative Entropy / KL Divergence)</span></p>

The **relative entropy** between two probability distributions $Q(x)$ and $P(x)$ that are defined over the same alphabet $\mathcal{A}_X$ is

$$D_\text{KL}(Q \| P) = \sum_x Q(x) \log \frac{Q(x)}{P(x)}.$$

The relative entropy satisfies $D_\text{KL}(Q \| P) \ge 0$ (Gibbs' inequality) with equality only if $Q = P$. In general $D_\text{KL}(Q \| P) \neq D_\text{KL}(P \| Q)$.

</div>

In statistical physics one often encounters probability distributions of the form

$$P(\mathbf{x} \mid \beta, \mathbf{J}) = \frac{1}{Z(\beta, \mathbf{J})} \exp[-\beta E(\mathbf{x}; \mathbf{J})],$$

where for example the state vector is $\mathbf{x} \in \lbrace -1, +1 \rbrace^N$, and the energy function is $E(\mathbf{x}; \mathbf{J}) = -\frac{1}{2}\sum_{m,n} J_{mn} x_m x_n - \sum_n h_n x_n$. Evaluating the normalizing constant $Z(\beta, \mathbf{J})$ is difficult, and from $Z$ we can derive all the thermodynamic properties of the system.

**Variational free energy minimization** is a method for *approximating* the complex distribution $P(\mathbf{x})$ by a simpler ensemble $Q(\mathbf{x}; \boldsymbol{\theta})$ that is parameterized by adjustable parameters $\boldsymbol{\theta}$. We adjust these parameters so as to get $Q$ to best approximate $P$, in some sense. A by-product of this approximation is a lower bound on $Z(\beta, \mathbf{J})$.

#### The Variational Free Energy

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Variational Free Energy)</span></p>

The objective function chosen to measure the quality of the approximation is the **variational free energy**

$$\beta \tilde{F}(\boldsymbol{\theta}) = \sum_{\mathbf{x}} Q(\mathbf{x}; \boldsymbol{\theta}) \ln \frac{Q(\mathbf{x}; \boldsymbol{\theta})}{\exp[-\beta E(\mathbf{x}; \mathbf{J})]}.$$

This expression can be manipulated into two useful forms:

1. $\beta \tilde{F}(\boldsymbol{\theta}) = \beta \langle E(\mathbf{x}; \mathbf{J}) \rangle_Q - S_Q$, where $\langle E \rangle_Q$ is the average energy under $Q$ and $S_Q$ is the entropy of $Q$.

2. $\beta \tilde{F}(\boldsymbol{\theta}) = D_\text{KL}(Q \| P) + \beta F$, where $\beta F \equiv -\ln Z(\beta, \mathbf{J})$ is the true free energy.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Variational Bound on Free Energy)</span></p>

By Gibbs' inequality, the variational free energy $\tilde{F}(\boldsymbol{\theta})$ is bounded below by $F$ and attains this value only for $Q(\mathbf{x}; \boldsymbol{\theta}) = P(\mathbf{x} \mid \beta, \mathbf{J})$. Our strategy is thus to vary $\boldsymbol{\theta}$ so that $\beta \tilde{F}(\boldsymbol{\theta})$ is minimized. The approximating distribution then gives a simplified approximation to the true distribution, and the value of $\beta \tilde{F}(\boldsymbol{\theta})$ will be an upper bound for $\beta F$. Equivalently, $\tilde{Z} \equiv e^{-\beta \tilde{F}(\boldsymbol{\theta})}$ is a **lower bound** for $Z$.

</div>

### 33.2 Variational Free Energy Minimization for Spin Systems

An example of a tractable variational free energy is given by the spin system whose energy function is $E(\mathbf{x}; \mathbf{J}) = -\frac{1}{2}\sum_{m,n} J_{mn} x_m x_n - \sum_n h_n x_n$, which we can approximate with a *separable* approximating distribution,

$$Q(\mathbf{x}; \mathbf{a}) = \frac{1}{Z_Q} \exp\left(\sum_n a_n x_n\right).$$

The variational parameters $\boldsymbol{\theta}$ are the components of the vector $\mathbf{a}$. The entropy of the separable approximating distribution is the sum of the entropies of the individual spins:

$$S_Q = \sum_n H_2^{(e)}(q_n),$$

where $q_n$ is the probability that spin $n$ is $+1$:

$$q_n = \frac{e^{a_n}}{e^{a_n} + e^{-a_n}} = \frac{1}{1 + \exp(-2a_n)},$$

and $H_2^{(e)}(q) = q \ln \frac{1}{q} + (1-q) \ln \frac{1}{(1-q)}$. The mean value of $x_n$ is $\bar{x}_n = \tanh(a_n) = 2q_n - 1$.

The mean energy under $Q$ is

$$\langle E(\mathbf{x}; \mathbf{J}) \rangle_Q = -\frac{1}{2}\sum_{m,n} J_{mn} \bar{x}_m \bar{x}_n - \sum_n h_n \bar{x}_n.$$

So the variational free energy is

$$\beta \tilde{F}(\mathbf{a}) = \beta\left(-\frac{1}{2}\sum_{m,n} J_{mn} \bar{x}_m \bar{x}_n - \sum_n h_n \bar{x}_n\right) - \sum_n H_2^{(e)}(q_n).$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Mean Field Equations)</span></p>

Minimizing the variational free energy $\tilde{F}(\mathbf{a})$ with respect to the variational parameters, we find that $\tilde{F}$ is extremized at any point satisfying:

$$a_m = \beta\left(\sum_n J_{mn} \bar{x}_n + h_m\right), \qquad \bar{x}_n = \tanh(a_n).$$

These are the **mean field equations** for a spin system. The variational parameter $a_n$ may be thought of as the strength of a fictitious field applied to an isolated spin $n$. The equation $\bar{x}_n = \tanh(a_n)$ describes the mean response of spin $n$, and the equation for $a_m$ describes how the field $a_m$ is set in response to the mean state of all the other spins.

One way of solving these equations is to update each parameter $a_m$ and the corresponding $\bar{x}_m$ asynchronously, one at a time. This **asynchronous updating** of the parameters is guaranteed to decrease $\beta \tilde{F}(\mathbf{a})$.

</div>

The variational free energy derivation is a helpful viewpoint for mean field theory for two reasons:

1. It associates an objective function $\beta \tilde{F}$ with the mean field equations; such an objective function is useful because it can help identify alternative dynamical systems that minimize the same function.
2. The theory is readily generalized to other approximating distributions. One could introduce a more complex approximation $Q(\mathbf{x}; \boldsymbol{\theta})$ that captures correlations among the spins. The more degrees of freedom the approximating distribution has, the tighter the bound on the free energy becomes.

### 33.3 Example: Mean Field Theory for the Ferromagnetic Ising Model

In the simple Ising model studied in Chapter 31, every coupling $J_{mn}$ is equal to $J$ if $m$ and $n$ are neighbours and zero otherwise. There is an applied field $h_n = h$ that is the same for all spins. A very simple approximating distribution is one with just a single variational parameter $a$, which defines a separable distribution

$$Q(\mathbf{x}; a) = \frac{1}{Z_Q} \exp\left(\sum_n a x_n\right)$$

in which all spins are independent and have the same probability $q_n = \frac{1}{1 + \exp(-2a)}$ of being up. The mean magnetization is $\bar{x} = \tanh(a)$ and the mean field equation becomes

$$a = \beta(CJ\bar{x} + h),$$

where $C$ is the number of couplings that a spin is involved in -- $C = 4$ in the case of a rectangular two-dimensional Ising model.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Phase Transition in Mean Field Theory)</span></p>

When $h = 0$, there is a **pitchfork bifurcation** at a critical temperature $T_c^\text{mft}$:

- **Above** $T_c^\text{mft}$: there is only one minimum in the variational free energy, at $a = 0$ and $\bar{x} = 0$; this corresponds to an approximating distribution that is uniform over all states.
- **Below** $T_c^\text{mft}$: there are two minima corresponding to approximating distributions that are symmetry-broken, with all spins more likely to be up, or all spins more likely to be down. The state $\bar{x} = 0$ persists as a stationary point but is now a local *maximum*.

When $h > 0$, there is a global variational free energy minimum at any temperature for a positive value of $\bar{x}$. As long as $h < JC$, there is also a second local minimum in the free energy (a **saddle-node bifurcation**).

</div>

The variational free energy per spin is

$$\beta \tilde{F} = \beta\left(-\frac{C}{2}J\bar{x}^2 - h\bar{x}\right) - H_2^{(e)}\left(\frac{\bar{x}+1}{2}\right).$$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Inadequacies of Mean Field Theory)</span></p>

While the mean field theory reproduces the key qualitative properties of the real Ising system (critical temperature, long-range order, two macroscopic states), there are important quantitative inadequacies:

- The critical temperature $T_c^\text{mft} = 4$ is nearly a factor of 2 greater than the true critical temperature $T_c = 2.27$.
- The variational model has equivalent properties in *any* number of dimensions, including $d = 1$, where the true system does not have a phase transition.
- One of the biggest differences is in the fluctuations in energy: the real system has large fluctuations near the critical temperature, whereas the approximating distribution has no correlations among its spins and thus has an energy-variance which scales simply linearly with the number of spins.

</div>

### 33.4 Variational Methods in Inference and Data Modelling

In statistical data modelling we are interested in the posterior probability distribution of a parameter vector $\mathbf{w}$ given data $D$ and model assumptions $\mathcal{H}$:

$$P(\mathbf{w} \mid D, \mathcal{H}) = \frac{P(D \mid \mathbf{w}, \mathcal{H}) P(\mathbf{w} \mid \mathcal{H})}{P(D \mid \mathcal{H})}.$$

In a variational approach to inference, we introduce an approximating probability distribution over the parameters, $Q(\mathbf{w}; \boldsymbol{\theta})$, and optimize this distribution so that it approximates the posterior distribution $P(\mathbf{w} \mid D, \mathcal{H})$ well. The variational free energy is

$$\tilde{F}(\boldsymbol{\theta}) = \int \mathrm{d}^k \mathbf{w} \; Q(\mathbf{w}; \boldsymbol{\theta}) \ln \frac{Q(\mathbf{w}; \boldsymbol{\theta})}{P(D \mid \mathbf{w}, \mathcal{H}) P(\mathbf{w} \mid \mathcal{H})}.$$

$\tilde{F}(\boldsymbol{\theta})$ can be viewed as the sum of $-\ln P(D \mid \mathcal{H})$ and the relative entropy between $Q(\mathbf{w}; \boldsymbol{\theta})$ and $P(\mathbf{w} \mid D, \mathcal{H})$. $\tilde{F}(\boldsymbol{\theta})$ is bounded below by $-\ln P(D \mid \mathcal{H})$ and only attains this value for $Q(\mathbf{w}; \boldsymbol{\theta}) = P(\mathbf{w} \mid D, \mathcal{H})$.

The approximation of posterior probability distributions using variational free energy minimization provides a useful approach to approximating Bayesian inference in a number of fields ranging from neural networks to the decoding of error-correcting codes. The method is sometimes called **ensemble learning** to contrast it with traditional learning processes in which a single parameter vector is optimized. Another name for it is **variational Bayes**.

### 33.5 The Case of an Unknown Gaussian: Approximating the Posterior Distribution of $\mu$ and $\sigma$

We fit an approximating ensemble $Q(\mu, \sigma)$ to the posterior distribution $P(\mu, \sigma \mid \lbrace x_n \rbrace_{n=1}^N)$. We make the single assumption that the approximating ensemble is separable: $Q(\mu, \sigma) = Q_\mu(\mu) Q_\sigma(\sigma)$. No restrictions on the functional form of $Q_\mu(\mu)$ and $Q_\sigma(\sigma)$ are made.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Optimal Separable Approximation for Gaussian Posterior)</span></p>

Optimizing the variational free energy alternately over $Q_\mu$ and $Q_\sigma$:

- The optimal $Q_\mu$ for fixed $Q_\sigma$ is a Gaussian: $Q_\mu^\text{opt}(\mu) = \text{Normal}(\mu; \bar{x}, \sigma_{\mu|D}^2)$, where $\sigma_{\mu|D}^2 = 1/(N\bar{\beta})$ and $\bar{\beta} \equiv \int \mathrm{d}\sigma \, Q_\sigma(\sigma) / \sigma^2$.
- The optimal $Q_\sigma$ for fixed $Q_\mu$ is a gamma distribution (in $\beta = 1/\sigma^2$): $Q_\sigma^\text{opt}(\beta) = \Gamma(\beta; b', c')$ with $\frac{1}{b'} = \frac{1}{2}(N\sigma_{\mu|D}^2 + S)$ and $c' = \frac{N}{2}$, where $S = \sum_n (x_n - \bar{x})^2$.

These two update rules are applied alternately, starting from an arbitrary initial condition. The algorithm converges to the optimal approximating ensemble in a few iterations.

</div>

The direct solution for the joint optimum gives $1/\bar{\beta} = S/(N-1)$. This is similar to the true posterior distribution of $\sigma$, which is a gamma distribution with a mean value of $\beta$ satisfying $1/\bar{\beta} = S/(N-1)$; the only difference is that the approximating distribution's shape parameter $c'$ is too large by $1/2$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Compactness of Variational Approximations)</span></p>

The approximations given by variational free energy minimization always tend to be **more compact** than the true distribution. The approximate posterior distribution over $\beta$ is a gamma distribution with mean $\bar{\beta}$ corresponding to a variance of $\sigma^2 = S/(N-1) = \sigma_{N-1}^2$. The approximate posterior distribution over $\mu$ is a Gaussian with mean $\bar{x}$ and standard deviation $\sigma_{N-1}/\sqrt{N}$.

The variational free energy minimization approach has the nice property that it is parameterization-independent; it avoids the problem of basis-dependence from which MAP methods and Laplace's method suffer.

</div>

### 33.6 Interlude

One of MacKay's students asked: "How do you ever come up with a useful approximating distribution, given that the true distribution is so complex you can't compute it directly?"

One interpretation of the standard practice (MAP estimation) is that the full description of our knowledge about $\mathbf{x}$, $P(\mathbf{x} \mid D)$, is being approximated by a delta-function, a probability distribution concentrated on $\mathbf{x}^*$. From this perspective, *any* approximating distribution $Q(\mathbf{x}; \boldsymbol{\theta})$, no matter how crude, *has* to be an improvement on the spike produced by the standard method. So even if we use only a simple Gaussian approximation, we are doing well.

### 33.7 K-means Clustering and the Expectation--Maximization Algorithm as a Variational Method

K-means clustering is an example of an **expectation--maximization (EM)** algorithm, with the two steps called the "E-step" (assignment) and the "M-step" (update). Neal and Hinton (1998) give a more general view of K-means clustering in which the algorithm is shown to optimize a variational objective function.

Let the parameters of the mixture model be denoted by $\boldsymbol{\theta}$. For each data point, there is a missing variable (latent variable) $k_n$. We approximate the posterior distribution by a separable distribution

$$Q_k(\lbrace k_n \rbrace_1^N) \, Q_{\boldsymbol{\theta}}(\boldsymbol{\theta}),$$

and define a variational free energy

$$\tilde{F}(Q_k, Q_{\boldsymbol{\theta}}) = \sum_{\lbrace k_n \rbrace} \int \mathrm{d}^D \boldsymbol{\theta} \; Q_k(\lbrace k_n \rbrace_1^N) \, Q_{\boldsymbol{\theta}}(\boldsymbol{\theta}) \ln \frac{Q_k(\lbrace k_n \rbrace_1^N) \, Q_{\boldsymbol{\theta}}(\boldsymbol{\theta})}{P(\lbrace \mathbf{x}^{(n)}, k_n \rbrace_1^N, \boldsymbol{\theta} \mid \mathcal{H})}.$$

$\tilde{F}$ is bounded below by minus the evidence, $\ln P(\lbrace \mathbf{x}^{(n)} \rbrace_1^N \mid \mathcal{H})$. The iterative algorithm consists of:

- **Assignment step (E-step):** $Q_k(\lbrace k_n \rbrace_1^N)$ is adjusted to reduce $\tilde{F}$, for fixed $Q_{\boldsymbol{\theta}}$.
- **Update step (M-step):** $Q_{\boldsymbol{\theta}}(\boldsymbol{\theta})$ is adjusted to reduce $\tilde{F}$, for fixed $Q_k$.

If we wish to obtain exactly the soft K-means algorithm, we impose a further constraint: $Q_{\boldsymbol{\theta}}(\boldsymbol{\theta}) = \delta(\boldsymbol{\theta} - \boldsymbol{\theta}^*)$. The optimal $Q_k$ is then a separable distribution in which the probability that $k_n = k$ is given by the responsibility $r_k^{(n)}$, and the optimal $\boldsymbol{\theta}^*$ is obtained by the update step of the soft K-means algorithm.

### 33.8 Variational Methods Other Than Free Energy Minimization

There are other strategies for approximating a complicated distribution $P(\mathbf{x})$. One approach pioneered by Jaakkola and Jordan is to create adjustable upper and lower bounds $Q^U$ and $Q^L$ to $P$. These bounds (which are unnormalized densities) are parameterized by variational parameters which are adjusted in order to obtain the tightest possible fit. The lower bound can be adjusted to *maximize* $\sum_{\mathbf{x}} Q^L(\mathbf{x})$, and the upper bound can be adjusted to *minimize* $\sum_{\mathbf{x}} Q^U(\mathbf{x})$.

#### The Bethe and Kikuchi Free Energies

The sum--product algorithm (Chapter 26) for functions of the factor-graph form may also be applied to factor graphs that are not tree-like. If the algorithm converges to a fixed point, it has been shown that that fixed point is a stationary point (usually a minimum) of a function of the messages called the **Kikuchi free energy**. In the special case where all factors in the factor graph are functions of one or two variables, the Kikuchi free energy is called the **Bethe free energy**.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Compactness vs. Heavy-Tailedness)</span></p>

*Variational free energy minimization* typically leads to approximating distributions whose length scales match the **shortest length scale** of the target distribution. The approximating distribution might be viewed as *too compact*.

In contrast, if we use the objective function $G = \int \mathrm{d}\mathbf{x} \, P(\mathbf{x}) \ln \frac{P(\mathbf{x})}{Q(\mathbf{x}; \sigma^2)}$ (i.e., $D_\text{KL}(P \| Q)$), the optimal approximation matches the **mean variance** of the target distribution, yielding a distribution that is *too spread out*.

Thus: a good variational approximation is *more compact* than the true distribution, while a good sampler (proposal density for importance sampling) should be *more heavy-tailed* than the true distribution. An over-compact distribution would be a lousy sampler with a large variance.

</div>

## Chapter 34 -- Independent Component Analysis and Latent Variable Modelling

### 34.1 Latent Variable Models

Many statistical models are **generative models** that specify a full probability density over all variables in the situation, making use of **latent variables** to describe a probability distribution over observables. Examples include:

- **Mixture models** (Chapter 22), where observables come from a superposed mixture of simple probability distributions (the latent variables are the unknown class labels).
- **Hidden Markov models**, where the latent variables represent hidden states.
- **Factor analysis**, where continuous latent variables generate the observables through a linear mapping with additive noise.

The decoding problem for error-correcting codes can also be viewed as a latent variable model: the encoding matrix $\mathbf{G}$ is known in advance, whereas in latent variable modelling the parameters equivalent to $\mathbf{G}$ must be inferred from the data along with the latent variables $\mathbf{s}$.

Usually, the latent variables have a simple, often separable, distribution. Fitting a latent variable model is therefore finding a description of the data in terms of "independent components". **Independent component analysis** (ICA) corresponds to perhaps the simplest possible latent variable model with continuous latent variables.

### 34.2 The Generative Model for Independent Component Analysis

A set of $N$ observations $D = \lbrace \mathbf{x}^{(n)} \rbrace_{n=1}^N$ are assumed to be generated as follows. Each $J$-dimensional vector $\mathbf{x}$ is a linear mixture of $I$ underlying source signals $\mathbf{s}$:

$$\mathbf{x} = \mathbf{G}\mathbf{s},$$

where the matrix of mixing coefficients $\mathbf{G}$ is not known.

The simplest algorithm results if we assume $I = J$. Our aim is to recover the source variables $\mathbf{s}$ (within some multiplicative factors, and possibly permuted), i.e., to create the inverse of $\mathbf{G}$ (within a post-multiplicative factor) given only a set of examples $\lbrace \mathbf{x} \rbrace$. We assume the latent variables are independently distributed, with marginal distributions $P(s_i \mid \mathcal{H}) \equiv p_i(s_i)$.

The joint probability of the observables and the hidden variables, given $\mathbf{G}$ and $\mathcal{H}$, is:

$$P(\lbrace \mathbf{x}^{(n)}, \mathbf{s}^{(n)} \rbrace_{n=1}^N \mid \mathbf{G}, \mathcal{H}) = \prod_{n=1}^{N} \left[ \left( \prod_j \delta\!\left(x_j^{(n)} - \sum_i G_{ji} s_i^{(n)}\right) \right) \left( \prod_i p_i(s_i^{(n)}) \right) \right].$$

We assume the vector $\mathbf{x}$ is generated **without noise** -- this makes the inference problem simpler to solve.

#### The Likelihood Function

The likelihood function for learning $\mathbf{G}$ from data $D$ is:

$$P(D \mid \mathbf{G}, \mathcal{H}) = \prod_n P(\mathbf{x}^{(n)} \mid \mathbf{G}, \mathcal{H}).$$

A single factor in the likelihood is obtained by marginalizing over the latent variables:

$$P(\mathbf{x}^{(n)} \mid \mathbf{G}, \mathcal{H}) = \int \mathrm{d}^I \mathbf{s}^{(n)} \; P(\mathbf{x}^{(n)} \mid \mathbf{s}^{(n)}, \mathbf{G}, \mathcal{H}) \, P(\mathbf{s}^{(n)} \mid \mathcal{H}) = \frac{1}{\lvert \det \mathbf{G} \rvert} \prod_i p_i(G_{ij}^{-1} x_j).$$

Taking the logarithm:

$$\ln P(\mathbf{x}^{(n)} \mid \mathbf{G}, \mathcal{H}) = -\ln \lvert \det \mathbf{G} \rvert + \sum_i \ln p_i(G_{ij}^{-1} x_j).$$

Introducing $\mathbf{W} \equiv \mathbf{G}^{-1}$, the log likelihood contributed by a single example may be written:

$$\ln P(\mathbf{x}^{(n)} \mid \mathbf{G}, \mathcal{H}) = \ln \lvert \det \mathbf{W} \rvert + \sum_i \ln p_i(W_{ij} x_j).$$

#### Gradient of the Log Likelihood

We will need the following identities:

$$\frac{\partial}{\partial G_{ji}} \ln \det \mathbf{G} = G_{ij}^{-1} = W_{ij},$$

$$\frac{\partial}{\partial G_{ji}} G_{lm}^{-1} = -G_{lj}^{-1} G_{im}^{-1} = -W_{lj} W_{im}.$$

Let us define $a_i \equiv W_{ij} x_j$ and

$$\phi_i(a_i) \equiv \frac{\mathrm{d} \ln p_i(a_i)}{\mathrm{d} a_i},$$

and $z_i = \phi_i(a_i)$, which indicates in which direction $a_i$ needs to change to make the probability of the data greater. The gradient with respect to $G_{ji}$ is:

$$\frac{\partial}{\partial G_{ji}} \ln P(\mathbf{x}^{(n)} \mid \mathbf{G}, \mathcal{H}) = -W_{ij} - a_i z_{i'} W_{i'j}.$$

Or alternatively, with respect to $W_{ij}$:

$$\frac{\partial}{\partial W_{ij}} \ln P(\mathbf{x}^{(n)} \mid \mathbf{G}, \mathcal{H}) = G_{ji} + x_j z_i.$$

If we choose to change $\mathbf{W}$ so as to ascend this gradient, we obtain the learning rule:

$$\Delta \mathbf{W} \propto [\mathbf{W}^\top]^{-1} + \mathbf{z}\mathbf{x}^\top.$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(ICA -- Online Steepest Ascents Version)</span></p>

Repeat for each datapoint $\mathbf{x}$:

1. Put $\mathbf{x}$ through a linear mapping: $\mathbf{a} = \mathbf{W}\mathbf{x}$.
2. Put $\mathbf{a}$ through a nonlinear map: $z_i = \phi_i(a_i)$, where a popular choice is $\phi = -\tanh(a_i)$.
3. Adjust the weights in accordance with $\Delta \mathbf{W} \propto [\mathbf{W}^\top]^{-1} + \mathbf{z}\mathbf{x}^\top$.

</div>

#### Choices of $\phi$

The choice of the function $\phi$ defines the assumed prior distribution of the latent variable $s$.

- **Linear choice:** $\phi_i(a_i) = -\kappa a_i$ implicitly assumes a Gaussian distribution on the latent variables. Since the Gaussian is invariant under rotation of the latent variables, there is no evidence favouring any particular alignment -- the linear algorithm will never recover $\mathbf{G}$ or the original sources. Our only hope is thus that the sources are **non-Gaussian**.
- **Tanh nonlinearity:** $\phi_i(a_i) = -\tanh(a_i)$ implies $p_i(s_i) \propto 1/\cosh(s_i)$. This is a heavier-tailed distribution than the Gaussian.
- **Tanh with gain $\beta$:** $\phi_i(a_i) = -\tanh(\beta a_i)$, which implies $p_i(s_i) \propto 1/[\cosh(\beta s_i)]^{1/\beta}$. In the limit of large $\beta$, the nonlinearity becomes a step function and $p_i(s_i)$ becomes a biexponential distribution, $p_i(s_i) \propto \exp(-\lvert s \rvert)$. In the limit $\beta \to 0$, $p_i(s_i)$ approaches a Gaussian with mean zero and variance $1/\beta$.

### 34.3 A Covariant, Simpler, and Faster Learning Algorithm

The steepest ascents algorithm derived above does not work very quickly, even on toy data. The algorithm is ill-conditioned and illustrates the general advice that *while finding the gradient of an objective function is a splendid idea, ascending the gradient directly may not be*.

#### Covariant Optimization in General

The principle of covariance says that a consistent algorithm should give the same results independent of the units in which quantities are measured. The popular steepest descents rule

$$\Delta w_i = \eta \frac{\partial L}{\partial w_i}$$

is dimensionally inconsistent: the left-hand side has dimensions of $[w_i]$ and the right-hand side has dimensions $1/[w_i]$. A **covariant** algorithm would have the form

$$\Delta w_i = \eta \sum_{i'} M_{ii'} \frac{\partial L}{\partial w_{i'}},$$

where $\mathbf{M}$ is a positive-definite matrix whose $i, i'$ element has dimensions $[w_i w_{i'}]$.

#### Metrics and Curvatures

Two sources of such matrices are **metrics** and **curvatures**:

- If there is a natural metric that defines distances in the parameter space $\mathbf{w}$, then $\mathbf{M}$ can be obtained from the metric.
- Alternatively, defining $\mathbf{A} \equiv -\nabla \nabla L$ (the curvature of the objective function), the matrix $\mathbf{M} = \mathbf{A}^{-1}$ gives a covariant algorithm -- this is the **Newton algorithm**, which converges to the minimum in a single step if $L$ is quadratic.

#### Back to Independent Component Analysis

Steepest ascents in $\mathbf{W} = \mathbf{G}^{-1}$ is not covariant. Constructing a covariant algorithm with the help of the curvature of the log likelihood, and making several approximations (assuming homogeneous sources with $\boldsymbol{\Sigma}\mathbf{D} = 1$), we arrive at the covariant learning algorithm:

$$\Delta W_{ij} = \eta \left( W_{ij} + x_j' z_i \right),$$

where $x_j' = W_{i'j} a_{i'}$ is obtained by a single backward pass through the weights. The quantity $(W_{ij} + x_j' z_i)$ is sometimes called the **natural gradient**.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(ICA -- Covariant Version)</span></p>

Repeat for each datapoint $\mathbf{x}$:

1. Put $\mathbf{x}$ through a linear mapping: $\mathbf{a} = \mathbf{W}\mathbf{x}$.
2. Put $\mathbf{a}$ through a nonlinear map: $z_i = \phi_i(a_i)$, where a popular choice is $\phi = -\tanh(a_i)$.
3. Put $\mathbf{a}$ back through $\mathbf{W}$: $\mathbf{x}' = \mathbf{W}^\top \mathbf{a}$.
4. Adjust the weights in accordance with $\Delta \mathbf{W} \propto \mathbf{W} + \mathbf{z}\mathbf{x}'^\top$.

This covariant algorithm does not require any matrix inversion and only needs a single backward pass through the weights once $\mathbf{z}$ has been computed.

</div>

#### Infinite Models

While latent variable models with a finite number of latent variables are widely used, it is often the case that our beliefs about the situation would be most accurately captured by a very large number of latent variables.

Consider clustering: if we model speech recognition using a cluster model, how many clusters should we use? The number of possible words is unbounded, so we would really like a model in which it is always possible for new clusters to arise. Furthermore, clusters should themselves be composed of clusters -- a hierarchy of clusters within clusters.

Infinite mixture models for categorical data are presented in Neal (1991), along with a Monte Carlo method for simulating inferences and predictions. Infinite Gaussian mixture models with a flat hierarchical structure are presented in Rasmussen (2000). Neal (2001) shows how to use Dirichlet diffusion trees to define models of hierarchical clusters, building on the Dirichlet process.

### 34.4 Exercises

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Exercise 34.1 -- ICA with Noise)</span></p>

Repeat the derivation of the algorithm, but assume a small amount of noise in $\mathbf{x}$: $\mathbf{x} = \mathbf{G}\mathbf{s} + \mathbf{n}$, so the term $\delta\!\left(x_j^{(n)} - \sum_i G_{ji} s_i^{(n)}\right)$ is replaced by a probability distribution over $x_j^{(n)}$ with mean $\sum_i G_{ji} s_i^{(n)}$. Show that, if this noise distribution has sufficiently small standard deviation, the identical algorithm results.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Exercise 34.3 -- Factor Analysis)</span></p>

**Factor analysis** assumes that the observations $\mathbf{x}$ can be described in terms of independent latent variables $\lbrace s_k \rbrace$ and independent additive noise. The observable $\mathbf{x}$ is given by

$$\mathbf{x} = \mathbf{G}\mathbf{s} + \mathbf{n},$$

where $\mathbf{n}$ is a noise vector whose components have a separable probability distribution. In factor analysis, $\lbrace s_k \rbrace$ and $\lbrace n_i \rbrace$ are often assumed to be zero-mean Gaussians; the noise terms may have different variances $\sigma_i^2$.

Create algorithms appropriate for the situations: (a) $\mathbf{x}$ includes substantial Gaussian noise; (b) more measurements than latent variables ($J > I$); (c) fewer measurements than latent variables ($J < I$).

</div>

## Chapter 35 -- Random Inference Topics

### 35.1 What Do You Know If You Are Ignorant?

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Benford's Law)</span></p>

A real variable $x$ is measured in an accurate experiment (e.g., the half-life of the neutron, the wavelength of light emitted by a firefly, the depth of Lake Vostok). What is the probability that the value of $x$ starts with a '1'?

If you don't know the units that a quantity is measured in, the probability of the first digit must be proportional to the length of the corresponding piece of logarithmic scale. The probability that the first digit is $1$ is:

$$p_1 = \frac{\log 2 - \log 1}{\log 10 - \log 1} = \frac{\log 2}{\log 10}.$$

Since $2^{10} = 1024 \simeq 10^3$, we have $10 \log 2 \simeq 3 \log 10$ and

$$p_1 \simeq \frac{3}{10}.$$

More generally, the probability that the first digit is $d$ is:

$$(\log(d+1) - \log(d)) / (\log 10 - \log 1) = \log_{10}(1 + 1/d).$$

This observation about initial digits is known as **Benford's law**. Ignorance does not correspond to a uniform probability distribution over $d$.

</div>

### 35.2 The Luria--Delbruck Distribution

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Exercise 35.4 -- The Luria--Delbruck Distribution)</span></p>

In their landmark paper demonstrating that bacteria could mutate from virus sensitivity to virus resistance, Luria and Delbruck (1943) wanted to estimate the mutation rate in an exponentially-growing population from the total number of mutants found at the end of the experiment.

In each culture, a single bacterium that is *not resistant* gives rise, after $g$ generations, to $N = 2^g$ descendants. The final culture is exposed to a virus, and the number of resistant bacteria $n$ is measured. The mutation rate (per cell per generation) is $a \approx 10^{-8}$. The total number of opportunities to mutate is $N$, since $\sum_{i=0}^{g-1} 2^i \simeq 2^g = N$.

The number of mutations that occurred, $r$, is roughly Poisson:

$$P(r \mid a, N) = e^{-aN} \frac{(aN)^r}{r!}.$$

Each mutation gives rise to $n_i$ final mutant cells depending on its generation time. A smoothed version of this distribution that permits all integers to occur is:

$$P(n_i) = \frac{1}{Z} \frac{1}{n_i^2},$$

where $Z = \pi^2 / 6 = 1.645$. The observed number of mutants is $n = \sum_{i=1}^r n_i$.

Given $M$ experiments, each with colony size $N$, and measured numbers of resistant bacteria $\lbrace n_m \rbrace_{m=1}^M$, the mutation rate $a$ can be inferred via Bayesian inference by computing $P(n \mid a) = \sum_{r=0}^{N} P(n \mid r) P(r \mid a, N)$.

</div>

### 35.3 Inferring Causation

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Exercise 35.5 -- Inferring Causation)</span></p>

In the Bayesian graphical model community, the task of inferring which way the arrows point -- that is, which nodes are parents, and which children -- is one on which much has been written. Inferring causation is tricky because of **likelihood equivalence**: two graphical models are likelihood-equivalent if for any setting of the parameters of either, there exists a setting of the parameters of the other such that the two joint probability distributions of all observables are identical. An example of a pair of likelihood-equivalent models are $A \to B$ and $B \to A$.

Consider the toy problem where $A$ and $B$ are binary variables. The two models are $\mathcal{H}_{A \to B}$ and $\mathcal{H}_{B \to A}$. $\mathcal{H}_{A \to B}$ asserts that the marginal probability of $A$ comes from a Beta$(1,1)$ distribution (i.e., the uniform distribution) and that the two conditional distributions $P(b \mid a=0)$ and $P(b \mid a=1)$ also come independently from Beta$(1,1)$ distributions.

Given $F = 1000$ outcomes:

|  | $a=0$ | $a=1$ |  |
|---|---|---|---|
| $b=0$ | 760 | 5 | 765 |
| $b=1$ | 190 | 45 | 235 |
|  | 950 | 50 |  |

The posterior probability ratio is:

$$\frac{P(\mathcal{H}_{A \to B} \mid \text{Data})}{P(\mathcal{H}_{B \to A} \mid \text{Data})} = \frac{(765 + 1)(235 + 1)}{(950 + 1)(50 + 1)} = \frac{3.8}{1}.$$

There is modest evidence in favour of $\mathcal{H}_{A \to B}$ because the three probabilities inferred for that hypothesis (roughly 0.95, 0.8, and 0.1) are more typical of the prior than the three probabilities inferred for the other (0.24, 0.008, and 0.19).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(On Causation and Bayesian Inference)</span></p>

The topic of inferring causation is complex. The fact that Bayesian inference can sensibly be used to infer the directions of arrows in graphs seems to be a neglected view, but it is certainly not the whole story. See Pearl (2000) for discussion of many other aspects of causality.

Some statistical methods that use the likelihood alone are unable to distinguish between likelihood-equivalent models. In a Bayesian approach, two likelihood-equivalent models may nevertheless be somewhat distinguished in the light of data, since likelihood-equivalence does not force a Bayesian to use priors that assign equivalent densities over the two parameter spaces.

</div>

### 35.4 Further Exercises

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Exercise 35.6 -- Poisson Process Inference)</span></p>

Photons arriving at a photon detector are believed to be emitted as a Poisson process with a time-varying rate,

$$\lambda(t) = \exp(a + b \sin(\omega t + \phi)),$$

where the parameters $a$, $b$, $\omega$, and $\phi$ are known. Data are collected during the time $t = 0 \ldots T$. Given that $N$ photons arrived at times $\lbrace t_n \rbrace_{n=1}^N$, discuss the inference of $a$, $b$, $\omega$, and $\phi$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Exercise 35.8 -- Biexponential Distribution)</span></p>

In an experiment, the measured quantities $\lbrace x_n \rbrace$ come independently from a biexponential distribution with mean $\mu$,

$$P(x \mid \mu) = \frac{1}{Z} \exp(-\lvert x - \mu \rvert),$$

where $Z$ is the normalizing constant, $Z = 2$. The mean $\mu$ is not known. Assuming the four datapoints are $\lbrace x_n \rbrace = \lbrace 0, 0.9, 2, 6 \rbrace$, what do these data tell us about $\mu$? Include detailed sketches in your answer.

</div>

## Chapter 36 -- Decision Theory

Decision theory is trivial, apart from computational details (just like playing chess!). You have a choice of various actions, $a$. The world may be in one of many states $\mathbf{x}$; which one occurs may be influenced by your action. The world's state has a probability distribution $P(\mathbf{x} \mid a)$. Finally, there is a utility function $U(\mathbf{x}, a)$ which specifies the payoff you receive when the world is in state $\mathbf{x}$ and you chose action $a$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Decision Theory)</span></p>

The task of decision theory is to select the action that maximizes the **expected utility**:

$$\mathcal{E}[U \mid a] = \int \mathrm{d}^K \mathbf{x} \; U(\mathbf{x}, a) \, P(\mathbf{x} \mid a).$$

The computational problem is to maximize $\mathcal{E}[U \mid a]$ over $a$. Pessimists may prefer to define a **loss function** $L$ instead of a utility function $U$ and minimize the expected loss.

</div>

In a real problem, the choice of an appropriate utility function may be quite difficult. Furthermore, when a sequence of actions is to be taken, with each action providing information about $\mathbf{x}$, we have to take into account the effect that this anticipated information may have on our subsequent actions. The resulting mixture of forward probability and inverse probability computations in a decision problem is distinctive.

### 36.1 Rational Prospecting

Suppose you have the task of choosing the site for a Tanzanite mine. Your final action will be to select the site from a list of $N$ sites. The $n$th site has a net value called the return $x_n$ which is initially unknown, and will be found out exactly only after site $n$ has been chosen. Before you take your final action you have the opportunity to do some **prospecting**. Prospecting at the $n$th site has a cost $c_n$ and yields data $d_n$ which reduce the uncertainty about $x_n$.

The utility function is:

$$U = x_{n_a}$$

if no prospecting is done (where $n_a$ is the chosen "action" site), and

$$U = -c_{n_p} + x_{n_a}$$

if prospecting is done at site $n_p$.

We assume Gaussian distributions throughout:

- Prior distribution of site return: $P(x_n) = \text{Normal}(x_n; \mu_n, \sigma_n^2)$.
- If you prospect at site $n$, the datum $d_n$ is a noisy version of $x_n$: $P(d_n \mid x_n) = \text{Normal}(d_n; x_n, \sigma^2)$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Gaussian Updating for Prospecting)</span></p>

The prior probability distribution of the datum $d_n$ is:

$$P(d_n) = \text{Normal}(d_n; \mu_n, \sigma^2 + \sigma_n^2)$$

(mnemonic: when independent variables add, variances add), and the posterior distribution of $x_n$ given $d_n$ is:

$$P(x_n \mid d_n) = \text{Normal}\!\left(x_n; \mu_n', \sigma_n^{2\prime}\right),$$

where

$$\mu_n' = \frac{d_n / \sigma^2 + \mu_n / \sigma_n^2}{1/\sigma^2 + 1/\sigma_n^2} \qquad \text{and} \qquad \frac{1}{\sigma_n^{2\prime}} = \frac{1}{\sigma^2} + \frac{1}{\sigma_n^2}$$

(mnemonic: when Gaussians multiply, precisions add).

</div>

#### Expected Utility Without Prospecting

Without prospecting, the optimal action is to select the site with the biggest mean:

$$n_a = \operatorname{argmax}_n \mu_n,$$

and the expected utility of this action is:

$$\mathcal{E}[U \mid \text{optimal } n] = \max_n \mu_n.$$

#### Expected Utility With Prospecting

The probability distribution of the new mean $\mu_n'$ is Gaussian with mean $\mu_n$ and variance

$$s^2 \equiv \sigma_n^2 \frac{\sigma_n^2}{\sigma^2 + \sigma_n^2}.$$

Let the biggest mean of the other sites be $\mu_1$. Consider prospecting at site $n$. When we obtain the new value of the mean, $\mu_n'$, we will choose site $n$ and get an expected return of $\mu_n'$ if $\mu_n' > \mu_1$, and we will choose site 1 and get an expected return of $\mu_1$ if $\mu_n' < \mu_1$. The expected utility of prospecting at site $n$, then picking the best site, is:

$$\mathcal{E}[U \mid \text{prospect at } n] = -c_n + P(\mu_n' < \mu_1) \, \mu_1 + \int_{\mu_1}^{\infty} \mathrm{d}\mu_n' \; \mu_n' \, \text{Normal}(\mu_n'; \mu_n, s^2).$$

The difference in utility between prospecting and not prospecting depends on what we would have done without the experimental information. This difference depends on whether $\mu_1$ is bigger than $\mu_n$: prospecting is valuable precisely when the outcome $d_n$ may alter the action from the one that we would have taken in the absence of the experimental information.

### 36.2 Further Reading

If the world in which we act is more complicated than the prospecting problem -- for example, if multiple iterations of prospecting are possible, and the cost of prospecting is uncertain -- then finding the optimal balance between **exploration and exploitation** becomes a much harder computational problem. **Reinforcement learning** addresses approximate methods for this problem.

### 36.3 Further Exercises

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Exercise 36.4 -- The Four Doors Problem)</span></p>

A new game show uses rules similar to those of the three doors (exercise 3.8), but there are four doors. The host explains: "First you will point to one of the doors, and then I will open one of the other doors, guaranteeing to choose a non-winner. Then you decide whether to stick with your original pick or switch to one of the remaining doors. Then I will open another non-winner (but never the current pick). You will then make your final decision by sticking with the door picked on the previous decision or by switching to the only other remaining door."

What is the optimal strategy? Should you switch on the first opportunity? Should you switch on the second opportunity?

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Exercise 36.5 -- The Allais Paradox)</span></p>

One of the challenges of decision theory is figuring out exactly what the utility function is. The utility of money, for example, is notoriously nonlinear for most people. The behaviour of many people cannot be captured by a coherent utility function, as illustrated by the *Allais paradox*:

Which of these choices do you find most attractive?
- **A.** &pound;1 million guaranteed.
- **B.** 89% chance of &pound;1 million; 10% chance of &pound;2.5 million; 1% chance of nothing.

Now consider these choices:
- **C.** 89% chance of nothing; 11% chance of &pound;1 million.
- **D.** 90% chance of nothing; 10% chance of &pound;2.5 million.

Many people prefer A to B, and at the same time, D to C. Prove that these preferences are inconsistent with any utility function $U(x)$ for money.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Exercise 36.6 -- Optimal Stopping)</span></p>

A large queue of $N$ potential partners is waiting at your door, all asking to marry you. They have arrived in random order. As you meet each partner, you have to decide on the spot whether to marry them or say no. Each potential partner has a desirability $d_n$. You must marry one of them, but you are not allowed to go back to anyone you have said no to.

- **(a)** Assuming your aim is to maximize the desirability $d_n$, what strategy should you use?
- **(b)** Assuming you wish to marry *the most desirable* person (utility is 1 if you achieve that, and zero otherwise), what strategy should you use?
- **(c)** Assuming the "strategy $M$" approach: meet the first $M$ partners and say no to all of them; memorize the maximum desirability $d_{\max}$ among them; then meet the others in sequence, waiting until a partner with $d_n > d_{\max}$ comes along, and marry them. If none more desirable comes along, marry the final $N$th partner. What is the optimal value of $M$?

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Exercise 36.7 -- Regret as an Objective Function)</span></p>

Many people working in learning theory and decision theory use "minimizing the maximal possible regret" as an objective function. Consider a lottery ticket that is a winner with probability $1/100$ and worth &pound;10 if it wins. Fred offers to sell you the ticket for &pound;1.

| Outcome | Buy | Don't buy |
|---|---|---|
| No win | $-1$ | $0$ |
| Wins | $+9$ | $0$ |

The four possible regret outcomes are: if you buy and it doesn't win, regret is &pound;1; if you don't buy and it wins, regret is &pound;9. The action that minimizes the maximum possible regret is thus to buy the ticket. Discuss whether this use of regret to choose actions can be philosophically justified.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Exercise 36.8 -- Gambling Oddities / Cover's Aim)</span></p>

A horse race involving $I$ horses occurs repeatedly, and you are obliged to bet all your money each time. Your bet at time $t$ can be represented by a normalized probability vector $\mathbf{b}$ multiplied by your money $m(t)$. The odds offered by the bookies are such that if horse $i$ wins then your return is $m(t+1) = b_i o_i m(t)$. Assuming the bookies' odds are "fair" ($\sum_i 1/o_i = 1$) and the probability that horse $i$ wins is $p_i$, work out the optimal betting strategy if your aim is **Cover's aim**, namely to maximize the *expected value of* $\log m(T)$.

Show that the optimal strategy sets $\mathbf{b}$ equal to $\mathbf{p}$, independent of the bookies' odds $\mathbf{o}$. Show that when this strategy is used, the money is expected to grow exponentially as $2^{nW(\mathbf{b}, \mathbf{p})}$, where $W = \sum_i p_i \log b_i o_i$.

</div>

## Chapter 37 -- Bayesian Inference and Sampling Theory

There are two schools of statistics. **Sampling theorists** concentrate on having methods guaranteed to work most of the time, given minimal assumptions. **Bayesians** try to make inferences that take into account all available information and answer the question of interest given the particular data set.

Sampling theory is the widely used approach to statistics, and most papers in most journals report their experiments using quantities like confidence intervals, significance levels, and $p$-values. A **$p$-value** (e.g. $p = 0.05$) is the probability, given a null hypothesis for the probability distribution of the data, that the outcome would be as extreme as, or more extreme than, the observed outcome. Untrained readers -- and perhaps, more worryingly, the authors of many papers -- usually interpret such a $p$-value as if it is a Bayesian probability (for example, the posterior probability of the null hypothesis), an interpretation that both sampling theorists and Bayesians would agree is incorrect.

### 37.1 A Medical Example

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Vaccination Trial)</span></p>

We are trying to reduce the incidence of an unpleasant disease called *microsoftus*. Two vaccinations, $A$ and $B$, are tested on a group of volunteers. Vaccination $B$ is a control treatment, a placebo. Of the 40 subjects, 30 are randomly assigned to treatment $A$ and 10 to control treatment $B$. After one year: of the 30 in group $A$, one contracts *microsoftus*; of the 10 in group $B$, three contract *microsoftus*. Is treatment $A$ better than treatment $B$?

</div>

#### Sampling Theory Has a Go

The standard sampling theory approach is to construct a **statistical test**. The test compares a hypothesis $\mathcal{H}_1$: "A and B have different effectivenesses" with a null hypothesis $\mathcal{H}_0$: "A and B have exactly the same effectivenesses as each other."

One chooses a **statistic** which measures how much a data set deviates from the null hypothesis. Here the standard statistic is $\chi^2$ (chi-squared):

$$\chi^2 = \sum_i \frac{(F_i - \langle F_i \rangle)^2}{\langle F_i \rangle},$$

where $F_i$ are the four data measurements ($F_{A+}$, $F_{A-}$, $F_{B+}$, $F_{B-}$) and $\langle F_i \rangle$ are the expected counts under the null hypothesis. With Yates's correction:

$$\chi^2 = \sum_i \frac{(\lvert F_i - \langle F_i \rangle \rvert - 0.5)^2}{\langle F_i \rangle}.$$

The estimated values of the null hypothesis parameters are $\hat{f}_+ = (F_{A+} + F_{B+})/(N_A + N_B)$ and $\hat{f}_- = (F_{A-} + F_{B-})/(N_A + N_B)$, giving $f_+ = 1/10$ and $f_- = 9/10$.

The expected values of the four measurements are $\langle F_{A+} \rangle = 3$, $\langle F_{A-} \rangle = 27$, $\langle F_{B+} \rangle = 1$, $\langle F_{B-} \rangle = 9$, and $\chi^2 = 5.93$.

Since this exceeds $\chi^2_{0.05} = 3.84$ (critical value for one degree of freedom), we reject the null hypothesis at the 0.05 significance level. However, with Yates's correction, $\chi^2 = 3.33$, and therefore we *accept* the null hypothesis.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Limitations of the $p$-value)</span></p>

Notice that the sampling theory answer does not say how much more effective $A$ is than $B$. It simply says that $A$ is "significantly different" from $B$. And here, "significant" means only "statistically significant", not practically significant.

The man in the street, reading "$p = 0.07$", might conclude "there is a 93% chance that the treatments differ in effectiveness". But what "$p = 0.07$" actually means is: "if you did this experiment many times, and the two treatments *had* equal effectiveness, then 7% of the time you would find a value of $\chi^2$ more extreme than the one that happened here." This has almost nothing to do with what we want to know, which is how likely it is that treatment A is better than B.

</div>

#### The Bayesian Approach

We scrap the hypothesis that the two treatments have exactly equal effectivenesses, since we do not believe it. There are two unknown parameters, $p_{A+}$ and $p_{B+}$, which are the probabilities that people given treatments $A$ and $B$, respectively, contract the disease. The posterior distribution is:

$$P(p_{A+}, p_{B+} \mid \lbrace F_i \rbrace) = \frac{P(\lbrace F_i \rbrace \mid p_{A+}, p_{B+}) \, P(p_{A+}, p_{B+})}{P(\lbrace F_i \rbrace)}.$$

The likelihood function is:

$$P(\lbrace F_i \rbrace \mid p_{A+}, p_{B+}) = \binom{30}{1} p_{A+}^1 p_{A-}^{29} \binom{10}{3} p_{B+}^3 p_{B-}^7.$$

Using a uniform prior $P(p_{A+}, p_{B+}) = 1$, the posterior is separable:

$$P(p_{A+}, p_{B+} \mid \lbrace F_i \rbrace) = P(p_{A+} \mid F_{A+}, F_{A-}) \, P(p_{B+} \mid F_{B+}, F_{B-}).$$

If we want to know "how probable is it that $p_{A+}$ is smaller than $p_{B+}$?", we compute:

$$P(p_{A+} < p_{B+} \mid \text{Data}),$$

which is the integral of the joint posterior probability over the region in which $p_{A+} < p_{B+}$. A straightforward numerical integration of the likelihood function gives:

$$P(p_{A+} < p_{B+} \mid \text{Data}) = 0.990.$$

Thus there is a **99% chance**, given the data and our prior assumptions, that treatment $A$ is superior to treatment $B$.

#### Model Comparison

If there were a situation in which we really did want to compare the two hypotheses $\mathcal{H}_0$: $p_{A+} = p_{B+}$ and $\mathcal{H}_1$: $p_{A+} \neq p_{B+}$, we can of course do this directly with Bayesian methods.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Bayesian Model Comparison -- Minimal Data)</span></p>

Consider the data set: one subject, given treatment $A$, subsequently contracted *microsoftus*; one subject, given treatment $B$, did not.

Under $\mathcal{H}_0$ (single parameter $p$, uniform prior):

$$P(D \mid \mathcal{H}_0) = \int \mathrm{d}p \; p(1-p) \times 1 = 1/6.$$

Under $\mathcal{H}_1$ (two parameters $p_{A+}$, $p_{B+}$, uniform priors):

$$P(D \mid \mathcal{H}_1) = \int \mathrm{d}p_{A+} \; p_{A+} \int \mathrm{d}p_{B+} \; (1 - p_{B+}) = 1/2 \times 1/2 = 1/4.$$

The evidence ratio is:

$$\frac{P(D \mid \mathcal{H}_1)}{P(D \mid \mathcal{H}_0)} = \frac{1/4}{1/6} = \frac{3}{2} = \frac{0.6}{0.4}.$$

So if the prior probability over the two hypotheses was 50:50, the posterior probability is 60:40 in favour of $\mathcal{H}_1$ -- the data give *weak* evidence that the effectivenesses are unequal. The sampling theory answer to the same question would be "not significant".

</div>

### 37.2 Dependence of $p$-values on Irrelevant Information

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Dr. Bloggs's Coin -- Stopping Rule Dependence)</span></p>

In an expensive laboratory, Dr. Bloggs tosses a coin labelled $a$ and $b$ twelve times and the outcome is the string $aaabaaaabaab$, which contains three $b$s and nine $a$s. What evidence do these data give that the coin is biased in favour of $a$?

Dr. Bloggs consults his sampling theory friend who says: "let $r$ be the number of $b$s and $n = 12$ be the total number of tosses; I view $r$ as the random variable and find the probability of $r$ taking on the value $r = 3$ or a more extreme value, assuming the null hypothesis $p_a = 0.5$ to be true." He computes:

$$P(r \le 3 \mid n=12, \mathcal{H}_0) = \sum_{r=0}^{3} \binom{12}{r} 1/2^{12} = 0.07,$$

and reports "there is not significant evidence of bias in favour of $a$" (at the significance level of 5%).

But Dr. Bloggs reveals: "the random variable in the experiment was not $r$; I decided before running the experiment that I would keep tossing the coin until I saw three $b$s; the random variable is thus $n$." According to sampling theory, a different calculation is required. The probability distribution of $n$ given $\mathcal{H}_0$ is:

$$P(n \mid \mathcal{H}_0, r) = \binom{n-1}{r-1} 1/2^n.$$

The sampling theorist now computes $P(n \ge 12 \mid r=3, \mathcal{H}_0) = 0.03$ and reports: "the $p$-value is 3% -- there *is* significant evidence of bias after all!"

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Stopping Rule Is Irrelevant to Bayesians)</span></p>

The $p$-values of sampling theory *do* depend on the stopping rule. But the Bayesian inferences about $p_a$ do **not** depend on the stopping rule. The likelihood function -- and therefore the posterior -- depends only on the data actually observed ($r = 3$, $n = 12$), not on how the experiment was designed. This is consistent with the **likelihood principle**: the stopping rule is not relevant to what we have learned about $p_a$.

As a thought experiment, consider spies observing Dr. Bloggs's experiments: each time he tosses the coin, the spies update the values of $r$ and $n$ and eagerly make inferences from the data as each new result occurs. Should the spies' beliefs about the bias of the coin depend on Dr. Bloggs's intentions regarding the continuation of the experiment? Clearly not.

</div>

### 37.3 Confidence Intervals

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Confidence Interval)</span></p>

In an experiment in which data $D$ are obtained from a system with an unknown parameter $\theta$, a **confidence interval** is a function $(\theta_{\min}(D), \theta_{\max}(D))$ of the data set $D$. The **confidence level** of the confidence interval is a property that we can compute before the data arrive: we imagine generating many data sets from a particular true value of $\theta$, calculating the interval, and checking whether the true value lies in that interval. If, averaging over all imagined repetitions of the experiment, the true value of $\theta$ lies in the confidence interval a fraction $f$ of the time (for all true values of $\theta$), then the confidence level is $f$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Confidence Interval Paradox)</span></p>

Let the parameter $\theta$ be an integer, and let the data be a pair of points $x_1, x_2$, drawn independently from:

$$P(x \mid \theta) = \begin{cases} 1/2 & x = \theta \\ 1/2 & x = \theta + 1 \\ 0 & \text{otherwise}. \end{cases}$$

Consider the confidence interval $[\theta_{\min}(D), \theta_{\max}(D)] = [\min(x_1, x_2), \min(x_1, x_2)]$, which has a confidence level of 75%.

If the data are $(x_1, x_2) = (29, 29)$, the confidence interval is $[29, 29]$, and its associated confidence level is 75%. But intuitively (or by Bayes's theorem), it is clear that $\theta$ could either be 29 or 28, and both possibilities are equally likely. The posterior probability of $\theta$ is 50% on 29 and 50% on 28.

If the data are $(x_1, x_2) = (29, 30)$, the confidence interval is still $[29, 29]$, and its associated confidence level is 75%. But in this case, by Bayes's theorem or common sense, we are 100% sure that $\theta$ is 29.

In neither case is the probability that $\theta$ lies in the "75% confidence interval" equal to 75%! Thus:

1. The way in which many people interpret the confidence levels of sampling theory is *incorrect*.
2. Given some data, what people usually want to know (whether they know it or not) is a Bayesian **posterior probability distribution**.

</div>

### 37.4 Some Compromise Positions

Many sampling theorists are pragmatic -- they are happy to choose from a selection of statistical methods, choosing whichever has the "best" long-run properties. In contrast, the Bayesian view is that there is only *one* answer to a well-posed problem; but it is not essential to convert sampling theorists -- instead, we can offer them Bayesian estimators and Bayesian confidence intervals, and request that the sampling theoretical properties of these methods be evaluated.

Another piece of common ground: while most well-posed inference problems have a unique correct answer (which can be found by Bayesian methods), not all problems are well-posed. A common question arising in data modelling is "am I using an appropriate model?" **Model criticism** -- hunting for defects in a current model -- is a task that may be aided by sampling theory tests, in which the null hypothesis ("the current model is correct") is well defined, but the alternative model is not specified. One could use sampling theory measures such as $p$-values to guide one's search for the aspects of the model most in need of scrutiny.

### 37.5 Further Exercises

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Exercise 37.2 -- Traffic Survey)</span></p>

A traffic survey records traffic on two successive days. On Friday morning, there are 12 vehicles in one hour. On Saturday morning, there are 9 vehicles in half an hour. Assuming the vehicles are Poisson distributed with rates $\lambda_F$ and $\lambda_S$ (in vehicles per hour) respectively:

- (a) Is $\lambda_S$ greater than $\lambda_F$?
- (b) By what factor is $\lambda_S$ bigger or smaller than $\lambda_F$?

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Exercise 37.3 -- Comparing Treatments Program)</span></p>

Write a program to compare treatments $A$ and $B$ given data $F_{A+}$, $F_{A-}$, $F_{B+}$, $F_{B-}$ as described in section 37.1. The outputs of the program should be: (a) the probability that treatment $A$ is more effective than treatment $B$; (b) the probability that $p_{A+} < 10 \, p_{B+}$; (c) the probability that $p_{B+} < 10 \, p_{A+}$.

</div>


## Part V: Neural Networks

## Chapter 38: Introduction to Neural Networks

In the field of neural networks, we study the properties of networks of idealized 'neurons'. Three motivations underlie work in this broad and interdisciplinary field:

- **Biology**: Understanding how the brain works -- how computation and memory are performed by brains.
- **Engineering**: Creating machines that can 'learn', perform 'pattern recognition' or 'discover patterns in data'.
- **Complex systems**: Neural networks are complex adaptive systems whose properties are interesting in their own right.

### 38.1 Memories

In conventional digital computation, a memory (e.g., a string of 5000 bits describing a person's name and face) is stored at an *address*. To retrieve the memory, you need to know the address. This scheme has several limitations:

1. **Address-based memory is not associative.** You cannot retrieve a memory from partial content without knowing the address.
2. **Address-based memory is not robust or fault-tolerant.** A one-bit mistake in the address retrieves a completely different memory. If one bit of the memory itself is flipped, the error persists.
3. **Address-based memory is not distributed.** Only a tiny fraction of devices participate in any given memory recall -- the rest sit idle.

Biological memory systems are fundamentally different:

1. **Biological memory is associative.** Memory recall is *content-addressable*. Given a person's name, we can often recall their face, and *vice versa*.
2. **Biological memory recall is error-tolerant and robust.** Errors in cues can be corrected (e.g., recalling "president Bush" from a cue containing a factual error). Hardware faults (cell damage) can also be tolerated.
3. **Biological memory is parallel and distributed.** Many neurons participate in the storage of multiple memories.

These properties of biological memory systems motivate the study of 'artificial neural networks' -- parallel distributed computational systems consisting of many interacting simple elements.

### 38.2 Terminology

Each time we describe a neural network algorithm we will typically specify three things:

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Neural Network Components)</span></p>

- **Architecture**: Specifies what variables are involved in the network and their topological relationships -- for example, the *weights* of the connections between the neurons, along with the *activities* of the neurons.
- **Activity rule**: Local rules that define how the *activities* of the neurons change in response to each other. Typically the activity rule depends on the *weights* (the parameters) in the network. This operates on a short time scale.
- **Learning rule**: Specifies the way in which the neural network's *weights* change with time. This learning operates on a longer time scale than the activity rule. Usually the learning rule depends on the *activities* of the neurons, and may also depend on *target* values supplied by a *teacher* and on the current value of the weights.

</div>

Activity rules and learning rules may be invented by researchers, or alternatively *derived* from carefully chosen *objective functions*.

Neural network algorithms can be roughly divided into two classes:

- **Supervised neural networks** are given data in the form of *inputs* and *targets*, the targets being a *teacher*'s specification of what the neural network's response to the input should be.
- **Unsupervised neural networks** are given data in an undivided form -- simply a set of examples $\lbrace \mathbf{x} \rbrace$. Some learning algorithms memorize these data for later recall; others are intended to 'generalize', discover 'patterns', or extract underlying 'features'.

## Chapter 39: The Single Neuron as a Classifier

We study a single neuron for two reasons: (1) many neural network models are built from single neurons, so they must be understood in detail; (2) a single neuron is itself capable of 'learning' -- various standard statistical methods can be viewed in terms of single neurons -- so it serves as a first example of a **supervised neural network**.

### 39.1 The Single Neuron

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Single Neuron)</span></p>

**Architecture.** A single neuron has a number $I$ of *inputs* $x_i$ and one *output* $y$. Associated with each input is a *weight* $w_i$ ($i = 1, \ldots, I$). There may be an additional parameter $w_0$ called a *bias*, which we may view as the weight associated with an input $x_0$ permanently set to 1. The single neuron is a *feedforward* device -- the connections are directed from the inputs to the output.

**Activity rule.** The activity rule has two steps:

1. Compute the *activation* of the neuron:

$$a = \sum_i w_i x_i,$$

where the sum is over $i = 0, \ldots, I$ if there is a bias and $i = 1, \ldots, I$ otherwise.

2. The *output* $y$ is set as a function $f(a)$ of the activation. The output is also called the *activity* of the neuron (not to be confused with the activation $a$).

</div>

#### Activation Functions

**(a) Deterministic activation functions:**

| Function | Formula | Range |
| --- | --- | --- |
| Linear | $y(a) = a$ | $(-\infty, \infty)$ |
| Sigmoid (logistic) | $y(a) = \dfrac{1}{1 + e^{-a}}$ | $(0, 1)$ |
| Sigmoid (tanh) | $y(a) = \tanh(a)$ | $(-1, 1)$ |
| Threshold | $y(a) = \Theta(a) \equiv \begin{cases} 1 & a > 0 \\ -1 & a \le 0 \end{cases}$ | $\lbrace -1, 1 \rbrace$ |

**(b) Stochastic activation functions** ($y$ is stochastically selected from $\pm 1$):

| Function | Rule |
| --- | --- |
| Heat bath | $y(a) = 1$ with probability $\dfrac{1}{1 + e^{-a}}$, else $-1$ |
| Metropolis | Compute $\Delta = ay$. If $\Delta \le 0$, flip $y$; else flip $y$ with probability $e^{-\Delta}$. |

### 39.2 Basic Neural Network Concepts

A neural network implements a function $y(\mathbf{x}; \mathbf{w})$; the 'output' $y$ is a nonlinear function of the 'inputs' $\mathbf{x}$, parameterized by 'weights' $\mathbf{w}$. For a single neuron with a sigmoid activation:

$$y(\mathbf{x}; \mathbf{w}) = \frac{1}{1 + e^{-\mathbf{w} \cdot \mathbf{x}}}.$$

#### Input Space and Weight Space

For a two-dimensional case with $\mathbf{x} = (x_1, x_2)$ and $\mathbf{w} = (w_1, w_2)$:

$$y(\mathbf{x}; \mathbf{w}) = \frac{1}{1 + e^{-(w_1 x_1 + w_2 x_2)}}.$$

On any line perpendicular to $\mathbf{w}$, the output is constant; along a line in the direction of $\mathbf{w}$, the output is a sigmoid function. The gain of the sigmoid (the gradient of the ramp) increases as the magnitude of $\mathbf{w}$ increases.

The **weight space** is the parameter space of the network. Each *point* in weight space corresponds to a *function* of $\mathbf{x}$.

The central idea of supervised neural networks: given *examples* of a relationship between input $\mathbf{x}$ and target $t$, we hope to make the network 'learn' a model of the relationship. *Training* the network involves searching in weight space for a value of $\mathbf{w}$ that produces a function fitting the training data well.

An **objective function** or **error function** $G(\mathbf{w})$ measures how well the network with weights $\mathbf{w}$ solves the task. Training is an exercise in *function minimization* -- adjusting $\mathbf{w}$ to minimize $G(\mathbf{w})$. For general feedforward neural networks, the *backpropagation* algorithm efficiently evaluates the gradient of the output $y$ with respect to $\mathbf{w}$.

### 39.3 Training the Single Neuron as a Binary Classifier

Assume we have a data set of inputs $\lbrace \mathbf{x}^{(n)} \rbrace_{n=1}^{N}$ with binary labels $\lbrace t^{(n)} \rbrace_{n=1}^{N}$, and a neuron whose output $y(\mathbf{x}; \mathbf{w})$ is bounded between 0 and 1.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Cross-Entropy Error Function)</span></p>

The error function for training the single neuron as a binary classifier is:

$$G(\mathbf{w}) = -\sum_n \left[ t^{(n)} \ln y(\mathbf{x}^{(n)}; \mathbf{w}) + (1 - t^{(n)}) \ln(1 - y(\mathbf{x}^{(n)}; \mathbf{w})) \right].$$

Each term can be recognized as the *information content* of one outcome, or equivalently as the relative entropy between the empirical probability distribution $(t^{(n)}, 1 - t^{(n)})$ and the probability distribution implied by the neuron $(y, 1-y)$. This objective is bounded below by zero and attains this value only if $y(\mathbf{x}^{(n)}; \mathbf{w}) = t^{(n)}$ for all $n$.

</div>

Differentiating with respect to $\mathbf{w}$, the gradient is:

$$g_j = \frac{\partial G}{\partial w_j} = \sum_{n=1}^{N} -(t^{(n)} - y^{(n)}) x_j^{(n)}.$$

The quantity $e^{(n)} \equiv t^{(n)} - y^{(n)}$ is the *error* on example $n$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(On-Line Gradient-Descent Learning)</span></p>

**Architecture.** A single neuron with $I$ inputs $x_i$ and one output $y$. Each input has a weight $w_i$ ($i = 1, \ldots, I$).

**Activity rule.**
1. Compute the activation: $a = \sum_i w_i x_i$.
2. Compute the output: $y(a) = \dfrac{1}{1 + e^{-a}}$.

This output states the probability (according to the neuron) that the input is in class 1 rather than class 0.

**Learning rule.** The teacher supplies a target value $t \in \lbrace 0, 1 \rbrace$. Compute the error signal:

$$e = t - y,$$

then adjust the weights:

$$\Delta w_i = \eta e x_i,$$

where $\eta$ is the *learning rate*. The activity rule and learning rule are repeated for each input/target pair $(\mathbf{x}, t)$.

</div>

#### Batch Learning vs. On-Line Learning

In the **on-line** learning algorithm, a change in the weights is made after every example is presented (this is *stochastic gradient descent*). In **batch** learning, we go through all examples, accumulate the gradient, and make a single update at the end:

$$\Delta w_i = -\eta \sum_n g_i^{(n)}, \quad \text{where} \quad g_i^{(n)} = -e^{(n)} x_i^{(n)}.$$

This is a **gradient descent algorithm**.

### 39.4 Beyond Descent on the Error Function: Regularization

If the examples are *linearly separable*, the neuron finds this linear separation and its weights diverge to ever-larger values -- an example of **overfitting**, where a model fits the training data so well that its generalization performance is adversely affected.

Two approaches to combat overfitting:

1. **Early stopping**: Use an algorithm intended to minimize $G(\mathbf{w})$, then halt at some point before convergence.

2. **Regularization**: Modify the objective function to incorporate a bias against undesirable solutions.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Regularized Objective Function)</span></p>

We modify the objective function to:

$$M(\mathbf{w}) = G(\mathbf{w}) + \alpha E_W(\mathbf{w}),$$

where the simplest choice of regularizer is the **weight decay** regularizer:

$$E_W(\mathbf{w}) = \frac{1}{2} \sum_i w_i^2.$$

The **regularization constant** $\alpha$ is called the *weight decay rate*. This additional term favours small values of $\mathbf{w}$ and decreases the tendency of a model to overfit fine details of the training data. The quantity $\alpha$ is known as a *hyperparameter* -- it plays a role in the learning algorithm but not in the activity rule.

</div>

As the weight decay rate $\alpha$ is increased, the solution becomes biased towards broader sigmoid functions with decision boundaries closer to the origin.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Optimization Methods)</span></p>

Gradient descent with a fixed step size $\eta$ is in general *not* the most efficient way to minimize a function. A modification known as *momentum*, while improving convergence, is also not recommended. Most neural network experts use more advanced optimizers such as **conjugate gradient algorithms**.

</div>

## Chapter 40: Capacity of a Single Neuron

### 40.1 Neural Network Learning as Communication

Many neural network models involve adapting a set of weights $\mathbf{w}$ in response to a set of $N$ target values $D_N = \lbrace t_n \rbrace_{n=1}^{N}$ at given locations $\lbrace \mathbf{x}_n \rbrace_{n=1}^{N}$. This process can be viewed as a **communication process**: the sender examines the data $D_N$ and creates a message $\mathbf{w}$ that depends on those data. The receiver then uses $\mathbf{w}$ to reconstruct the data. The adapted network weights $\mathbf{w}$ therefore play the role of a communication channel, conveying information about the training data.

The central question: **what is the capacity of this channel?** That is, how much information can be stored by training a neural network?

### 40.2 The Capacity of a Single Neuron

We look at the simplest case: a single binary threshold neuron. The capacity of such a neuron is **two bits per weight**. A neuron with $K$ inputs can store $2K$ bits of information.

The receiver is constrained to observe the output of the neuron at the same fixed set of $N$ points $\lbrace \mathbf{x}_n \rbrace$ that were in the training set. The question becomes: how many different binary labellings of $N$ points can a linear threshold function produce?

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(General Position)</span></p>

A set of points $\lbrace \mathbf{x}_n \rbrace$ in $K$-dimensional space are in **general position** if any subset of size $\le K$ is linearly independent, and no $K + 1$ of them lie in a $(K-1)$-dimensional plane.

In $K = 3$ dimensions, for example, a set of points are in general position if no three points are colinear and no four points are coplanar. Intuitively, points in general position are like random points in the space.

</div>

The neuron performs the function:

$$y = f\!\left(\sum_{k=1}^{K} w_k x_k\right), \quad \text{where} \quad f(a) = \begin{cases} 1 & a > 0 \\ 0 & a \le 0. \end{cases}$$

We will not have a bias $w_0$; the capacity for a neuron with a bias can be obtained by replacing $K$ by $K+1$.

### 40.3 Counting Threshold Functions

Let $T(N, K)$ denote the number of distinct threshold functions on $N$ points in general position in $K$ dimensions.

#### Base Cases

- **$K = 1$ dimension, any $N$:** The $N$ points lie on a line. By changing the sign of $w_1$ we can label all points on one side of the origin 1 and the others 0, or *vice versa*. Thus $T(N, 1) = 2$.
- **$N = 1$ point, any $K$:** We can realize both possible labellings by setting $\mathbf{w} = \pm \mathbf{x}^{(1)}$. Thus $T(1, K) = 2$.
- **$K = 2$ dimensions:** Spinning the separating line around the origin, each time the line passes over a point we obtain a new function. In one revolution, every point is passed over twice. Thus $T(N, 2) = 2N$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Unrealizable Functions)</span></p>

For $N \ge 3$ and $K = 2$, not all binary functions can be realized by a linear threshold function. One famous example with $N = 4$ and $K = 2$ is the **exclusive-or** function on the points $\mathbf{x} = (\pm 1, \pm 1)$. This function remains unrealizable even if the points are perturbed into general position.

</div>

#### Counting via Weight Space

Each data point $\mathbf{x}^{(n)}$ defines a hyperplane $\mathbf{x}^{(n)} \cdot \mathbf{w} = 0$ in weight space. The set of weight vectors on one side of this hyperplane ($\mathbf{x}^{(n)} \cdot \mathbf{w} > 0$) classify that point as 1, and those on the other side classify it as 0. With $N$ data points, weight space is divided into regions by $N$ hyperplanes, and the number of regions equals $T(N, K)$.

#### Recurrence Relation

When we add the $N$th hyperplane in $K$ dimensions, it bisects $T(N-1, K-1)$ of the $T(N-1, K)$ regions created by the previous $N-1$ hyperplanes. This gives:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Recurrence for Threshold Functions)</span></p>

$$T(N, K) = T(N-1, K) + T(N-1, K-1),$$

with boundary conditions $T(N, 1) = 2$ and $T(1, K) = 2$.

</div>

This recurrence is the same as that of Pascal's triangle. By comparing boundary conditions, the solution is:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Number of Threshold Functions)</span></p>

$$T(N, K) = 2 \sum_{k=0}^{K-1} \binom{N-1}{k}.$$

Using the fact that the $N$th row of Pascal's triangle sums to $2^N$, this simplifies to:

$$T(N, K) = \begin{cases} 2^N & K \ge N \\ 2 \sum_{k=0}^{K-1} \binom{N-1}{k} & K < N. \end{cases}$$

</div>

#### Interpretation

The ratio $T(N, K) / 2^N$ tells us the probability that an arbitrary labelling $\lbrace t_n \rbrace_{n=1}^{N}$ can be memorized by our neuron:

- For all $N \le K$, the two functions are equal: $T(N, K) = 2^N$. Every labelling can be realized.
- The line $N = K$ defines the maximum number of points on which *any* arbitrary labelling can be realized.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Vapnik--Chervonenkis Dimension)</span></p>

The **Vapnik--Chervonenkis (VC) dimension** of a class of functions is the maximum number of points on which *any* arbitrary labelling can be realized. The VC dimension of a binary threshold function on $K$ dimensions is thus $K$.

</div>

For large $K$, the sum in $T(N,K)$ is well approximated by the error function:

$$\sum_0^{K} \binom{N}{k} \simeq 2^N \, \Phi\!\left(\frac{K - (N/2)}{\sqrt{N/2}}\right),$$

where $\Phi(z) \equiv \int_{-\infty}^{z} \exp(-z^2/2) / \sqrt{2\pi} \, \mathrm{d}z$. For $N < 2K$ the ratio $T(N,K)/2^N$ is still greater than $1/2$, and for large $K$ the ratio is very close to 1. There is a catastrophic drop to zero at $N = 2K$: for $N > 2K$, only a tiny fraction of the binary labellings can be realized.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Capacity of a Threshold Neuron)</span></p>

The capacity of a linear threshold neuron, for large $K$, is **2 bits per weight**. A single neuron can almost certainly memorize up to $N = 2K$ random binary labels perfectly, but will almost certainly fail to memorize more.

</div>

## Chapter 41: Learning as Inference

### 41.1 Neural Network Learning as Inference

In Chapter 39, we trained a simple neural network as a classifier by minimizing an objective function

$$M(\mathbf{w}) = G(\mathbf{w}) + \alpha E_W(\mathbf{w}),$$

made up of the error function

$$G(\mathbf{w}) = -\sum_n \left[ t^{(n)} \ln y(\mathbf{x}^{(n)}; \mathbf{w}) + (1 - t^{(n)}) \ln(1 - y(\mathbf{x}^{(n)}; \mathbf{w})) \right]$$

and a regularizer $E_W(\mathbf{w}) = \frac{1}{2} \sum_i w_i^2$.

This learning process can be given a probabilistic interpretation. The output $y(\mathbf{x}; \mathbf{w})$ is interpreted as the probability that an input $\mathbf{x}$ belongs to class $t = 1$:

$$y(\mathbf{x}; \mathbf{w}) \equiv P(t = 1 \mid \mathbf{x}, \mathbf{w}).$$

The likelihood of a single data point can be written as:

$$P(t \mid \mathbf{w}, \mathbf{x}) = y^t (1-y)^{1-t},$$

so the error function $G$ is minus the log likelihood:

$$P(D \mid \mathbf{w}) = \exp[-G(\mathbf{w})].$$

Similarly, the regularizer corresponds to a log prior probability distribution over the parameters:

$$P(\mathbf{w} \mid \alpha) = \frac{1}{Z_W(\alpha)} \exp(-\alpha E_W).$$

If $E_W$ is quadratic, the prior is a Gaussian with variance $\sigma_W^2 = 1/\alpha$, and the normalizing constant is $1/Z_W(\alpha) = (\alpha / 2\pi)^{K/2}$, where $K$ is the number of parameters.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Posterior over Weights)</span></p>

The objective function $M(\mathbf{w})$ corresponds to the *inference* of the parameters $\mathbf{w}$ given the data:

$$P(\mathbf{w} \mid D, \alpha) = \frac{P(D \mid \mathbf{w}) P(\mathbf{w} \mid \alpha)}{P(D \mid \alpha)} = \frac{1}{Z_M} \exp(-M(\mathbf{w})).$$

The $\mathbf{w}$ found by (locally) minimizing $M(\mathbf{w})$ can be interpreted as the *(locally) most probable parameter vector*, $\mathbf{w}^* \equiv \mathbf{w}_{\mathrm{MP}}$.

</div>

Error functions are naturally interpreted as *log* probabilities because they are additive ($G$ is a *sum* of information contents, $E_W$ is a *sum* of squared weights), while probabilities are multiplicative.

### 41.2 Illustration for a Neuron with Two Weights

For a neuron with two inputs and no bias,

$$y(\mathbf{x}; \mathbf{w}) = \frac{1}{1 + e^{-(w_1 x_1 + w_2 x_2)}},$$

we can plot the posterior probability $P(\mathbf{w} \mid D, \alpha) \propto \exp(-M(\mathbf{w}))$. The likelihood function $\exp(-G(\mathbf{w}))$ is shown as a function of $\mathbf{w}$, multiplied by a broad Gaussian prior. As the amount of data increases, the posterior ensemble becomes increasingly concentrated around the most probable value $\mathbf{w}^*$.

In the traditional view, learning produces a single point $\mathbf{w}^*$ in weight space. In the **Bayesian view**, the product of learning is an *ensemble* of plausible parameter values -- we do not choose one particular $\mathbf{w}$; rather we evaluate their posterior probabilities.

### 41.3 Beyond Optimization: Making Predictions

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Over-Confidence of Point Estimates)</span></p>

The best-fit parameters $\mathbf{w}_{\mathrm{MP}}$ often give **over-confident predictions**. Consider new data arriving at points A (near the training data) and B (far from it). The best-fit model assigns both examples the same probability of being in class 1, because they have the same value of $\mathbf{w}_{\mathrm{MP}} \cdot \mathbf{x}$. But intuitively, predictions at B should be less confident since it is far from the training data. The Bayesian approach accounts for this by taking into account the whole posterior ensemble.

</div>

The Bayesian prediction of a new target $t^{(N+1)}$ at location $\mathbf{x}^{(N+1)}$ involves **marginalizing** over the parameters:

$$P(t^{(N+1)} \mid \mathbf{x}^{(N+1)}, D, \alpha) = \int \mathrm{d}^K \mathbf{w} \; P(t^{(N+1)} \mid \mathbf{x}^{(N+1)}, \mathbf{w}, \alpha) \, P(\mathbf{w} \mid D, \alpha),$$

where $K$ is the dimensionality of $\mathbf{w}$. The predictions are obtained by weighting the prediction for each possible $\mathbf{w}$ by its posterior probability:

$$P(t^{(N+1)} = 1 \mid \mathbf{x}^{(N+1)}, D, \alpha) = \int \mathrm{d}^K \mathbf{w} \; y(\mathbf{x}^{(N+1)}; \mathbf{w}) \, \frac{1}{Z_M} \exp(-M(\mathbf{w})),$$

which is the average of the output of the neuron at $\mathbf{x}^{(N+1)}$ under the posterior distribution of $\mathbf{w}$.

### 41.4 Monte Carlo Implementation of a Single Neuron

The integral above can be evaluated using Monte Carlo methods, treating $y(\mathbf{x}^{(N+1)}; \mathbf{w})$ as a function $f$ of $\mathbf{w}$ whose mean we compute:

$$\langle f(\mathbf{w}) \rangle \simeq \frac{1}{R} \sum_r f(\mathbf{w}^{(r)}),$$

where $\lbrace \mathbf{w}^{(r)} \rbrace$ are samples from the posterior distribution $\frac{1}{Z_M} \exp(-M(\mathbf{w}))$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Langevin Monte Carlo Method)</span></p>

The Langevin method can be summarized as 'gradient descent with added noise'. A noise vector $\mathbf{p}$ is generated from a Gaussian with unit variance. The gradient $\mathbf{g}$ is computed, and a step in $\mathbf{w}$ is made:

$$\Delta \mathbf{w} = -\tfrac{1}{2} \epsilon^2 \mathbf{g} + \epsilon \mathbf{p}.$$

If the $\epsilon \mathbf{p}$ term were omitted this would simply be gradient descent with learning rate $\eta = \frac{1}{2} \epsilon^2$. The step is accepted or rejected depending on the change in $M(\mathbf{w})$ and the change in gradient, with a probability of acceptance such that detailed balance holds.

The method has one free parameter, $\epsilon$, which controls the typical step size.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Optimization vs. Typicality)</span></p>

During Monte Carlo sampling, $G(\mathbf{w})$ fluctuates around the value of $G(\mathbf{w}_{\mathrm{MP}})$, but $M(\mathbf{w})$ does not fluctuate *around* $M(\mathbf{w}_{\mathrm{MP}})$ -- since $M$ is minimized at $\mathbf{w}_{\mathrm{MP}}$, it can only go up. The **typical set** of $\mathbf{w}$ has different properties from the most probable state $\mathbf{w}_{\mathrm{MP}}$.

A general lesson: one should be cautious about making use of *optimized* parameters, as the properties of optimized parameters may be unrepresentative of the properties of typical, plausible parameters; and the predictions obtained using optimized parameters alone will often be unreasonably over-confident.

</div>

The **Hamiltonian Monte Carlo** method is a more efficient variant where each proposal uses multiple gradient evaluations along a dynamical trajectory in $(\mathbf{w}, \mathbf{p})$ space. Compared to the Langevin method, the autocorrelation of the Hamiltonian Monte Carlo simulation falls much more rapidly with simulation time -- it is at least ten times more efficient in its use of computer time.

### 41.5 Implementing Inference with Gaussian Approximations

An alternative to Monte Carlo is to make a **Gaussian approximation** to the posterior probability. We Taylor-expand $M(\mathbf{w})$ around its minimum $\mathbf{w}_{\mathrm{MP}}$:

$$M(\mathbf{w}) \simeq M(\mathbf{w}_{\mathrm{MP}}) + \frac{1}{2} (\mathbf{w} - \mathbf{w}_{\mathrm{MP}})^\mathsf{T} \mathbf{A} (\mathbf{w} - \mathbf{w}_{\mathrm{MP}}),$$

where $\mathbf{A}$ is the **Hessian** matrix of second derivatives:

$$A_{ij} \equiv \frac{\partial^2}{\partial w_i \, \partial w_j} M(\mathbf{w}) \bigg|_{\mathbf{w} = \mathbf{w}_{\mathrm{MP}}}.$$

The Gaussian approximation to the posterior is then:

$$Q(\mathbf{w}; \mathbf{w}_{\mathrm{MP}}, \mathbf{A}) = [\det(\mathbf{A}/2\pi)]^{1/2} \exp\!\left[ -\frac{1}{2} (\mathbf{w} - \mathbf{w}_{\mathrm{MP}})^\mathsf{T} \mathbf{A} (\mathbf{w} - \mathbf{w}_{\mathrm{MP}}) \right].$$

The matrix $\mathbf{A}$ defines **error bars** on $\mathbf{w}$: the variance--covariance matrix of $Q$ is $\mathbf{A}^{-1}$.

The Hessian for the single neuron classifier is:

$$\frac{\partial^2}{\partial w_i \, \partial w_j} M(\mathbf{w}) = \sum_{n=1}^{N} f'(a^{(n)}) x_i^{(n)} x_j^{(n)} + \alpha \delta_{ij},$$

where $f'(a) = f(a)(1 - f(a))$ and $a^{(n)} = \sum_j w_j x_j^{(n)}$.

#### Calculating the Marginalized Probability

The output $y(\mathbf{x}; \mathbf{w})$ depends on $\mathbf{w}$ only through the scalar $a(\mathbf{x}; \mathbf{w})$. Under the Gaussian posterior, the activation $a$ is itself Gaussian-distributed:

$$P(a \mid \mathbf{x}, D, \alpha) = \mathrm{Normal}(a_{\mathrm{MP}}, s^2), \quad \text{where } a_{\mathrm{MP}} = a(\mathbf{x}; \mathbf{w}_{\mathrm{MP}}) \text{ and } s^2 = \mathbf{x}^\mathsf{T} \mathbf{A}^{-1} \mathbf{x}.$$

The marginalized output is:

$$P(t = 1 \mid \mathbf{x}, D, \alpha) = \psi(a_{\mathrm{MP}}, s^2) \equiv \int \mathrm{d}a \; f(a) \, \mathrm{Normal}(a_{\mathrm{MP}}, s^2).$$

This integral of a sigmoid times a Gaussian can be approximated by:

$$\psi(a_{\mathrm{MP}}, s^2) \simeq \phi(a_{\mathrm{MP}}, s^2) \equiv f(\kappa(s) \, a_{\mathrm{MP}}),$$

with $\kappa = 1 / \sqrt{1 + \pi s^2 / 8}$. This is to be contrasted with $y(\mathbf{x}; \mathbf{w}_{\mathrm{MP}}) = f(a_{\mathrm{MP}})$, the output of the most probable network: the Gaussian approximation moderates the predictions, especially where $s^2$ is large (i.e., far from the training data).

## Chapter 42: Hopfield Networks

Having studied the single neuron, we now connect multiple neurons together. Neural networks can be divided into two classes based on their connectivity:

- **Feedforward networks**: All connections are directed such that the network forms a directed acyclic graph.
- **Feedback networks**: Any network that is not feedforward. Every neuron's output is an input to all the other neurons.

The **Hopfield network** is a fully connected feedback network with *symmetric* weights ($w_{ij} = w_{ji}$). It has two applications: (1) acting as **associative memories**, and (2) solving **optimization problems**.

### 42.1 Hebbian Learning

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Hebbian Learning Rule)</span></p>

A simple model due to Donald Hebb (1949) captures the idea of associative memory: the weights between neurons whose activities are *positively correlated* are *increased*:

$$\frac{\mathrm{d} w_{ij}}{\mathrm{d} t} \sim \mathrm{Correlation}(x_i, x_j).$$

When two stimuli co-occur in the environment, the Hebbian learning rule increases the weights between the corresponding neurons, so that later, when one stimulus occurs in isolation, the other neuron is also activated. This produces **pattern completion** -- an unsupervised, local learning algorithm that spontaneously produces associative memory.

</div>

### 42.2 Definition of the Binary Hopfield Network

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Binary Hopfield Network)</span></p>

**Architecture.** A Hopfield network consists of $I$ neurons. They are fully connected through *symmetric*, *bidirectional* connections with weights $w_{ij} = w_{ji}$. There are no self-connections, so $w_{ii} = 0$ for all $i$. Biases $w_{i0}$ may be included. The activity of neuron $i$ (its output) is denoted $x_i$.

**Activity rule.** Each neuron updates its state as if it were a single neuron with a threshold activation function:

$$x(a) = \Theta(a) \equiv \begin{cases} 1 & a \ge 0 \\ -1 & a < 0. \end{cases}$$

The updates may be:
- **Synchronous** -- all neurons compute their activations $a_i = \sum_j w_{ij} x_j$ and update simultaneously to $x_i = \Theta(a_i)$.
- **Asynchronous** -- one neuron at a time computes its activation and updates its state (in a fixed or random sequence).

**Learning rule.** The learning rule is intended to make a set of desired *memories* $\lbrace \mathbf{x}^{(n)} \rbrace$ be *stable states* of the activity rule. Each memory is a binary pattern with $x_i \in \lbrace -1, 1 \rbrace$. The weights are set using the **sum of outer products** or **Hebb rule**:

$$w_{ij} = \eta \sum_n x_i^{(n)} x_j^{(n)},$$

where $\eta$ is an unimportant constant (often $\eta = 1/N$).

</div>

### 42.3 Definition of the Continuous Hopfield Network

Using the identical architecture and learning rule, we can define a Hopfield network whose activities are real numbers between $-1$ and $1$.

**Activity rule.** Each neuron updates as if it were a single neuron with a sigmoid activation function:

$$a_i = \sum_j w_{ij} x_j, \qquad x_i = \tanh(a_i).$$

A *gain* $\beta \in (0, \infty)$ may be introduced into the activation function: $x_i = \tanh(\beta a_i)$.

### 42.4 Convergence of the Hopfield Network

The continuous Hopfield network's activity rules (if implemented asynchronously) have a **Lyapunov function**. Recall from the discussion of variational methods (section 33.2) that the mean-field equations for a spin system with energy $E(\mathbf{x}; \mathbf{J}) = -\frac{1}{2} \sum_{m,n} J_{mn} x_m x_n - \sum_n h_n x_n$ are identical to the Hopfield network's equations if we replace $J$ by $w$, $\bar{x}$ by $x$, and $h_n$ by $w_{i0}$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Lyapunov Function for Hopfield Networks)</span></p>

The continuous Hopfield network with asynchronous updates has the **variational free energy** as its Lyapunov function:

$$\beta \tilde{F}(\mathbf{x}) = -\beta \frac{1}{2} \mathbf{x}^\mathsf{T} \mathbf{W} \mathbf{x} - \sum_i H_2^{(e)}[(1 + x_i)/2],$$

where $H_2^{(e)}$ is the binary entropy function. This function decreases under the dynamical evolution and is bounded below. Therefore, the Hopfield network's dynamics are bound to settle to a **fixed point** (local minimum), a **limit cycle** (along which the function is constant), or converge. Chaotic behaviour is not possible.

This convergence proof depends crucially on the weights being *symmetric* and the updates being *asynchronous*.

</div>

### 42.5 The Associative Memory in Action

A 25-unit binary Hopfield network trained on four patterns by Hebbian learning demonstrates associative memory. For an initial condition randomly perturbed from a memory, it often takes only one iteration for all the errors to be corrected.

The network has additional stable states beyond the four desired memories: the inverse of any stable state is also a stable state, and there are several stable states that can be interpreted as mixtures of the memories.

**Brain damage**: The network can be severely damaged and still function. If 50 or even 100 of the 300 weights are randomly set to zero, the desired memories remain as attracting stable states -- unlike a digital computer, which would fail catastrophically if 20% of its components were destroyed.

**Overloading**: If we try to store too many patterns, the associative memory fails catastrophically. When a sixth pattern is added to a 25-unit network that previously stored five patterns, only one of the patterns is stable; the others all flow into spurious stable states.

### 42.6 The Continuous-Time Continuous Hopfield Network

We can define a continuous-time version where each neuron's activity $x_i(t)$ is a continuous function of time, with activations computed instantaneously:

$$a_i(t) = \sum_j w_{ij} x_j(t).$$

The neuron's response to its activation is mediated by the differential equation:

$$\frac{\mathrm{d}}{\mathrm{d}t} x_i(t) = -\frac{1}{\tau}(x_i(t) - f(a_i)),$$

where $f(a) = \tanh(a)$. For a steady activation $a_i$, the activity $x_i(t)$ relaxes exponentially to $f(a_i)$ with time-constant $\tau$. As long as the weight matrix is symmetric, this system has the variational free energy $\tilde{F}$ as its Lyapunov function.

### 42.7 The Capacity of the Hopfield Network

The Hopfield associative memory can be viewed as a communication channel. A list of desired memories is encoded into a set of weights $\mathbf{W}$; the receiver finds the stable states of the network and interprets them as the original memories. This communication system can fail in several ways:

1. Individual bits in some memories might be **corrupted** (stable state displaced from desired memory).
2. Entire memories might be **absent** from the list of attractors, or have a negligibly small basin of attraction.
3. **Spurious** additional memories unrelated to the desired memories might be present.
4. Spurious memories derived from desired memories by operations such as **mixing and inversion** may also be present.

#### Capacity -- Stringent Definition

We consider the stability of a single bit of one desired pattern $\mathbf{x}^{(n)}$ when the network is in that state. The activation of neuron $i$ is:

$$a_i = \sum_{j \ne i} x_i^{(n)} x_j^{(n)} x_j^{(n)} + \sum_{j \ne i} \sum_{m \ne n} x_i^{(m)} x_j^{(m)} x_j^{(n)}.$$

The first term is the 'signal' $(I-1) x_i^{(n)}$ reinforcing the desired memory. The second term is 'noise' -- a sum of $(I-1)(N-1)$ random quantities with mean 0 and variance 1. The activation $a_i$ is approximately Gaussian-distributed with mean $I x_i^{(n)}$ and variance $IN$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Bit-Flip Probability in Hopfield Network)</span></p>

The probability that bit $i$ will flip on the first iteration of the Hopfield network's dynamics is:

$$P(i \text{ unstable}) = \Phi\!\left(-\frac{I}{\sqrt{IN}}\right) = \Phi\!\left(-\frac{1}{\sqrt{N/I}}\right),$$

where $\Phi(z) \equiv \int_{-\infty}^{z} \frac{1}{\sqrt{2\pi}} e^{-z^2/2} \, \mathrm{d}z$. If we store $N \simeq 0.18I$ patterns, there is about a 1% chance that a specified bit will be unstable on the first iteration.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Stringent Capacity of Hopfield Network)</span></p>

If we require all desired patterns to be completely stable (no bit flips at all, with total error probability less than $\epsilon$), the maximum number of patterns that can be stored is:

$$N_{\max} \simeq \frac{I}{4 \ln I + 2 \ln(1/\epsilon)}.$$

</div>

#### The Statistical Physicists' Capacity

If we allow a small amount of corruption, the number of storable patterns increases. Amit *et al.* (1985) found numerically a sharp discontinuity at:

$$N_{\mathrm{crit}} = 0.138 I.$$

Below this value, there is likely to be a stable state near every desired memory. Above it, the system has only spurious stable states known as **spin glass states**, uncorrelated with any of the desired memories. Just below the critical value, the fraction of bits flipped is 1.6%.

Phase behaviour as a function of $N/I$:

- **For all $N/I$:** stable spin glass states exist, uncorrelated with desired memories.
- **$N/I > 0.138$:** spin glass states are the only stable states.
- **$N/I \in (0, 0.138)$:** stable states close to the desired memories exist.
- **$N/I \in (0, 0.05)$:** desired-memory states have lower energy than spin glass states.
- **$N/I \in (0.05, 0.138)$:** spin glass states dominate (lower energy than desired memories).
- **$N/I \in (0, 0.03)$:** additional *mixture* states (combinations of several memories) also exist.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Hopfield Network Capacity in Bits)</span></p>

The capacity of the Hopfield network (defined by the critical transition) is $0.138I$ random binary patterns, each of length $I$, each received with 1.6% of its bits flipped. In bits:

$$0.138 I^2 \times (1 - H_2(0.016)) = 0.122 \, I^2 \text{ bits}.$$

Since there are $I^2/2$ weights in the network, this can also be expressed as **0.24 bits per weight**.

</div>

### 42.8 Improving on the Capacity of the Hebb Rule

We can do better than the Hebb rule by defining an objective function that measures how well the network stores all the memories and minimizing it. For each neuron $i$, every pattern $\mathbf{x}^{(n)}$ defines an input/target pair -- exactly what we wanted the single neuron of Chapter 39 to do. We define:

$$G(\mathbf{W}) = -\sum_i \sum_n t_i^{(n)} \ln y_i^{(n)} + (1 - t_i^{(n)}) \ln(1 - y_i^{(n)}),$$

where $t_i^{(n)} = 1$ if $x_i^{(n)} = 1$ and $t_i^{(n)} = 0$ if $x_i^{(n)} = -1$, and $y_i^{(n)} = 1/(1 + \exp(-a_i^{(n)}))$ with $a_i^{(n)} = \sum_j w_{ij} x_j^{(n)}$.

This gradient-based learning algorithm does a better job than the one-shot Hebb rule. For example, six patterns that cannot be memorized by the Hebb rule all become stable states when this algorithm is used.

### 42.9 Hopfield Networks for Optimization Problems

Since a Hopfield network's dynamics minimize an energy function, it is natural to map interesting optimization problems onto Hopfield networks.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Travelling Salesman Problem)</span></p>

A set of $K$ cities is given with $K(K-1)/2$ distances between them. The task is to find a closed tour visiting each city once with the smallest total distance. This is NP-complete.

The method by Hopfield and Tank (1985) represents a tentative solution by the state of a network with $I = K^2$ neurons arranged in a square, with each neuron representing the hypothesis that a particular city comes at a particular point in the tour.

The weights in the Hopfield network play two roles:
1. **Validity constraints**: Large negative weights between neurons in the same row or column enforce that valid states look like permutation matrices.
2. **Distance objective**: Negative weights proportional to $-d_{BD}$ between nodes in adjacent columns encode the total distance.

A difficulty arises from the conflict between the validity-enforcing weights and the distance weights. Aiyer (1991) resolved this using a **deterministic annealing** approach, gradually increasing the gain $\beta$ from 0 to $\infty$, confining dynamics to a 'valid subspace'.

</div>

## Chapter 43: Boltzmann Machines

### 43.1 From Hopfield Networks to Boltzmann Machines

The binary Hopfield network minimizes an energy function $E(\mathbf{x}) = -\frac{1}{2} \mathbf{x}^\mathsf{T} \mathbf{W} \mathbf{x}$, and the continuous Hopfield network with activation function $x_n = \tanh(a_n)$ can be viewed as *approximating* the probability distribution associated with that energy function:

$$P(\mathbf{x} \mid \mathbf{W}) = \frac{1}{Z(\mathbf{W})} \exp\!\left[\frac{1}{2} \mathbf{x}^\mathsf{T} \mathbf{W} \mathbf{x}\right].$$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Boltzmann Machine)</span></p>

The **stochastic Hopfield network** or **Boltzmann machine** (Hinton and Sejnowski, 1986) has the following activity rule: after computing the activation $a_i = \sum_j w_{ij} x_j$,

$$\text{set } x_i = +1 \text{ with probability } \frac{1}{1 + e^{-2a_i}}, \quad \text{else set } x_i = -1.$$

This rule implements **Gibbs sampling** for the probability distribution $P(\mathbf{x} \mid \mathbf{W})$.

</div>

#### Boltzmann Machine Learning

Given examples $\lbrace \mathbf{x}^{(n)} \rbrace_1^N$, we want to adjust $\mathbf{W}$ so that the generative model $P(\mathbf{x} \mid \mathbf{W})$ matches the data. Differentiating the log likelihood:

$$\frac{\partial}{\partial w_{ij}} \ln P(\lbrace \mathbf{x}^{(n)} \rbrace_1^N \mid \mathbf{W}) = N \left[ \langle x_i x_j \rangle_{\mathrm{Data}} - \langle x_i x_j \rangle_{P(\mathbf{x} \mid \mathbf{W})} \right],$$

where the first term is the **empirical correlation**:

$$\langle x_i x_j \rangle_{\mathrm{Data}} \equiv \frac{1}{N} \sum_{n=1}^{N} x_i^{(n)} x_j^{(n)},$$

and the second term is the **model correlation** $\langle x_i x_j \rangle_{P(\mathbf{x} \mid \mathbf{W})} = \sum_{\mathbf{x}} x_i x_j P(\mathbf{x} \mid \mathbf{W})$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Waking and Sleeping Interpretation)</span></p>

The two terms in the gradient can be viewed as 'waking' and 'sleeping' rules. While the network is 'awake', it measures the correlation between $x_i$ and $x_j$ in the real world, and weights are *increased* in proportion. While the network is 'asleep', it 'dreams' about the world using the generative model, measures the model correlations, and these determine a proportional *decrease* in the weights. At equilibrium, the dream-world correlations match the real-world correlations and the weights do not change.

</div>

Note that when starting from $\mathbf{W} = 0$, a single gradient step recovers exactly the Hebb rule $w_{ij} = \eta \sum_n x_i^{(n)} x_j^{(n)}$.

#### Criticism of Simple Boltzmann Machines

Simple Boltzmann machines where all neurons correspond to *visible* variables can only capture the **second-order statistics** $\langle x_i x_j \rangle$ of the environment. The real world often has higher-order correlations that must be included for an effective description. Second-order statistics only capture whether two pixels are likely to be in the same state -- they are insufficient for describing complex ensembles such as images of chairs.

### 43.2 Boltzmann Machine with Hidden Units

The key innovation (Hinton and Sejnowski, 1986) is to add **hidden neurons** (also called *latent variables*) that do not correspond to observed variables. The idea is that high-order correlations among visible variables are described by including extra hidden variables while keeping only second-order interactions between variables. The hidden units effectively perform **feature extraction**.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Boltzmann Machine with Hidden Units)</span></p>

We denote visible units by $\mathbf{x}$, hidden units by $\mathbf{h}$, and the generic state of a neuron by $y_i$, with $\mathbf{y} \equiv (\mathbf{x}, \mathbf{h})$. The likelihood of $\mathbf{W}$ given a single data example $\mathbf{x}^{(n)}$ is:

$$P(\mathbf{x}^{(n)} \mid \mathbf{W}) = \sum_{\mathbf{h}} \frac{1}{Z(\mathbf{W})} \exp\!\left[\frac{1}{2} [\mathbf{y}^{(n)}]^\mathsf{T} \mathbf{W} \mathbf{y}^{(n)}\right] = \frac{Z_{\mathbf{x}^{(n)}}(\mathbf{W})}{Z(\mathbf{W})},$$

where $Z_{\mathbf{x}^{(n)}}(\mathbf{W}) = \sum_{\mathbf{h}} \exp\!\left[\frac{1}{2} [\mathbf{y}^{(n)}]^\mathsf{T} \mathbf{W} \mathbf{y}^{(n)}\right]$.

</div>

The derivative of the log likelihood with respect to any weight $w_{ij}$ is again the difference between a 'waking' term and a 'sleeping' term:

$$\frac{\partial}{\partial w_{ij}} \ln P(\lbrace \mathbf{x}^{(n)} \rbrace_1^N \mid \mathbf{W}) = \sum_n \left\lbrace \langle y_i y_j \rangle_{P(\mathbf{h} \mid \mathbf{x}^{(n)}, \mathbf{W})} - \langle y_i y_j \rangle_{P(\mathbf{x}, \mathbf{h} \mid \mathbf{W})} \right\rbrace.$$

The first term $\langle y_i y_j \rangle_{P(\mathbf{h} \mid \mathbf{x}^{(n)}, \mathbf{W})}$ is the correlation with visible variables clamped to $\mathbf{x}^{(n)}$ and hidden variables freely sampling. The second term is the correlation when the Boltzmann machine generates samples from its full model distribution. Both require Monte Carlo methods to evaluate, making the Boltzmann machine time-consuming to simulate.

## Chapter 44: Supervised Learning in Multilayer Networks

### 44.1 Multilayer Perceptrons

The **multilayer perceptron** is a feedforward network with input neurons, hidden neurons, and output neurons arranged in a sequence of layers. The most common multilayer perceptrons have a single hidden layer and are known as 'two-layer' networks (counting layers of neurons, not including the inputs).

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Two-Layer Feedforward Network)</span></p>

A feedforward network defines a nonlinear parameterized mapping from an input $\mathbf{x}$ to an output $\mathbf{y} = \mathbf{y}(\mathbf{x}; \mathbf{w}, \mathcal{A})$. For a network with one hidden layer:

**Hidden layer:**

$$a_j^{(1)} = \sum_l w_{jl}^{(1)} x_l + \theta_j^{(1)}; \quad h_j = f^{(1)}(a_j^{(1)})$$

**Output layer:**

$$a_i^{(2)} = \sum_j w_{ij}^{(2)} h_j + \theta_i^{(2)}; \quad y_i = f^{(2)}(a_i^{(2)})$$

where $l$ runs over the inputs $x_1, \ldots, x_L$, $j$ runs over the hidden units, and $i$ runs over the outputs. For regression, typically $f^{(1)}(a) = \tanh(a)$ and $f^{(2)}(a) = a$ (linear output). The 'weights' $w$ and 'biases' $\theta$ together make up the parameter vector $\mathbf{w}$.

</div>

#### Properties of Random Networks

The sort of functions produced by random weights depend on the standard deviations $\sigma_{\mathrm{bias}}$, $\sigma_{\mathrm{in}}$, and $\sigma_{\mathrm{out}}$ of the biases, input-to-hidden weights, and output weights respectively:

- The vertical scale of a typical function is of order $\sqrt{H} \, \sigma_{\mathrm{out}}$.
- The horizontal range in which the function varies significantly is of order $\sigma_{\mathrm{bias}} / \sigma_{\mathrm{in}}$.
- The shortest horizontal length scale is of order $1/\sigma_{\mathrm{in}}$.

Radford Neal (1996) showed that in the limit $H \to \infty$, the statistical properties of the functions generated by randomizing the weights are independent of the number of hidden units. The complexity of the functions is determined by the characteristic magnitude of the weights -- controlling this magnitude is the key to controlling the complexity of the fitted function.

### 44.2 How a Regression Network is Traditionally Trained

The network is trained using a data set $D = \lbrace \mathbf{x}^{(n)}, \mathbf{t}^{(n)} \rbrace$ by adjusting $\mathbf{w}$ to minimize an error function:

$$E_D(\mathbf{w}) = \frac{1}{2} \sum_n \sum_i \left( t_i^{(n)} - y_i(\mathbf{x}^{(n)}; \mathbf{w}) \right)^2.$$

This minimization uses the **backpropagation** algorithm (Rumelhart *et al.*, 1986), which uses the chain rule to efficiently compute the gradient of $E_D$.

Often, regularization (weight decay) is included, modifying the objective function to:

$$M(\mathbf{w}) = \beta E_D + \alpha E_W,$$

where $E_W = \frac{1}{2} \sum_i w_i^2$.

### 44.3 Neural Network Learning as Inference

The learning process can be given a probabilistic interpretation, generalizing the discussion of Chapter 41.

The error function $\beta E_D$ is the negative log likelihood:

$$P(D \mid \mathbf{w}, \beta, \mathcal{H}) = \frac{1}{Z_D(\beta)} \exp(-\beta E_D).$$

The use of the sum-squared error corresponds to an assumption of **Gaussian noise** on the target variables, with noise level $\sigma_\nu^2 = 1/\beta$.

The regularizer corresponds to a log prior:

$$P(\mathbf{w} \mid \alpha, \mathcal{H}) = \frac{1}{Z_W(\alpha)} \exp(-\alpha E_W).$$

If $E_W$ is quadratic, the prior is a Gaussian with variance $\sigma_W^2 = 1/\alpha$.

The objective function $M(\mathbf{w})$ then corresponds to the **inference** of the parameters:

$$P(\mathbf{w} \mid D, \alpha, \beta, \mathcal{H}) = \frac{1}{Z_M} \exp(-M(\mathbf{w})).$$

The $\mathbf{w}$ found by minimizing $M(\mathbf{w})$ is the most probable parameter vector $\mathbf{w}_{\mathrm{MP}}$.

#### Binary Classification Networks

For binary classification with output $y(\mathbf{x}; \mathbf{w}, \mathcal{A}) \in (0,1)$ interpreted as $P(t = 1 \mid \mathbf{x}, \mathbf{w}, \mathcal{A})$, the error function $\beta E_D$ is replaced by the negative log likelihood:

$$G(\mathbf{w}) = -\left[ \sum_n t^{(n)} \ln y(\mathbf{x}^{(n)}; \mathbf{w}) + (1 - t^{(n)}) \ln(1 - y(\mathbf{x}^{(n)}; \mathbf{w})) \right].$$

The total objective is $M = G + \alpha E_W$ (no parameter $\beta$ since there is no Gaussian noise).

#### Multi-Class Classification Networks

For multi-class classification, the targets are represented by a vector $\mathbf{t}$ with a single element set to 1 (indicating the correct class) and all others 0. A **softmax** output layer is used:

$$y_i = \frac{e^{a_i}}{\sum_{i'} e^{a_{i'}}},$$

with negative log likelihood:

$$G(\mathbf{w}) = -\sum_n \sum_i t_i^{(n)} \ln y_i(\mathbf{x}^{(n)}; \mathbf{w}).$$

### 44.4 Benefits of the Bayesian Approach to Supervised Feedforward Neural Networks

From the statistical perspective, supervised neural networks are nonlinear curve-fitting devices. The effective complexity of the interpolating model is of crucial importance.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Overfitting Problem)</span></p>

As a control parameter (e.g., regularization constant $\alpha$) is varied to increase model complexity, the best fit to the *training* data improves, but the *test* error first decreases then increases again. An over-complex model **overfits** the data and generalizes poorly. The Bayesian approach solves this by evaluating the **evidence** $P(\mathrm{Data} \mid \mathrm{Control \; Parameters})$ for alternative values of the control parameters.

</div>

Bayesian optimization of model control parameters has four important advantages:

1. **No test/validation set needed** -- all available training data can be used for both model fitting and model comparison.
2. **On-line optimization** -- regularization constants can be optimized simultaneously with the ordinary model parameters.
3. **Smooth objective** -- the Bayesian objective function is not noisy, in contrast to a cross-validation measure.
4. **Gradient available** -- the gradient of the evidence with respect to control parameters can be evaluated, enabling simultaneous optimization of a large number of control parameters.

Probabilistic modelling also handles **uncertainty** naturally through *marginalization* -- incorporating uncertainty about parameters into predictions. Error bars on the predictions of a trained neural network become larger where the data are sparse.

Within the Bayesian framework, one can define models with multiple hyperparameters that capture the idea of uncertain input variable relevance -- these models can infer automatically from the data which are the relevant input variables for a problem.

## Chapter 45: Gaussian Processes

Feedforward neural networks such as multilayer perceptrons are popular tools for nonlinear regression and classification problems. From a Bayesian perspective, a choice of a neural network model can be viewed as defining a prior probability distribution over nonlinear functions, and the neural network's learning process can be interpreted in terms of the posterior probability distribution over the unknown function. Neal (1996) showed that the prior distribution over nonlinear functions implied by a Bayesian neural network falls in a class of probability distributions known as **Gaussian processes** in the limit of large but otherwise standard networks. This motivates discarding parameterized networks and working directly with Gaussian processes.

### The Bayesian Framework for Regression

In the Bayesian interpretation, a nonlinear function $y(\mathbf{x})$ parameterized by parameters $\mathbf{w}$ is assumed to underlie the data $\lbrace \mathbf{x}^{(n)}, t_n \rbrace_{n=1}^N$. Denote the set of input vectors by $\mathbf{X}_N \equiv \lbrace \mathbf{x}^{(n)} \rbrace_{n=1}^N$ and the set of corresponding target values by $\mathbf{t}_N \equiv \lbrace t_n \rbrace_{n=1}^N$. The inference of $y(\mathbf{x})$ is described by the posterior probability distribution:

$$P(y(\mathbf{x}) \mid \mathbf{t}_N, \mathbf{X}_N) = \frac{P(\mathbf{t}_N \mid y(\mathbf{x}), \mathbf{X}_N) P(y(\mathbf{x}))}{P(\mathbf{t}_N \mid \mathbf{X}_N)}.$$

The first term $P(\mathbf{t}_N \mid y(\mathbf{x}), \mathbf{X}_N)$ is the likelihood of the target values given the function, often assumed to be a separable Gaussian distribution. The second term $P(y(\mathbf{x}))$ is the prior distribution on functions assumed by the model. The idea of Gaussian process modelling is to place a prior $P(y(\mathbf{x}))$ directly on the space of functions, without parameterizing $y(\mathbf{x})$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Gaussian Process)</span></p>

A **Gaussian process** is a generalization of a Gaussian distribution over a finite vector space to a function space of infinite dimension. Just as a Gaussian distribution is fully specified by its mean and covariance matrix, a Gaussian process is specified by a **mean function** and a **covariance function** $C(\mathbf{x}, \mathbf{x}')$ that expresses the expected covariance between the values of the function $y$ at points $\mathbf{x}$ and $\mathbf{x}'$.

The probability distribution of a function $y(\mathbf{x})$ is a Gaussian process if for any finite selection of points $\mathbf{x}^{(1)}, \mathbf{x}^{(2)}, \ldots, \mathbf{x}^{(N)}$, the density $P(y(\mathbf{x}^{(1)}), y(\mathbf{x}^{(2)}), \ldots, y(\mathbf{x}^{(N)}))$ is a Gaussian.

</div>

### 45.1 Standard Methods for Nonlinear Regression

We are given $N$ data points $\mathbf{X}_N, \mathbf{t}_N = \lbrace \mathbf{x}^{(n)}, t_n \rbrace_{n=1}^N$. The inputs $\mathbf{x}$ are vectors of some fixed input dimension $I$. The targets $t$ are either real numbers (regression) or categorical variables, e.g. $t \in \lbrace 0, 1 \rbrace$ (classification). We concentrate on regression.

#### Parametric Approaches

In a parametric approach we express the unknown function $y(\mathbf{x})$ in terms of a nonlinear function $y(\mathbf{x}; \mathbf{w})$ parameterized by parameters $\mathbf{w}$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Fixed Basis Functions)</span></p>

Using a set of basis functions $\lbrace \phi_h(\mathbf{x}) \rbrace_{h=1}^H$, we can write:

$$y(\mathbf{x}; \mathbf{w}) = \sum_{h=1}^{H} w_h \phi_h(\mathbf{x}).$$

If the basis functions are nonlinear functions of $\mathbf{x}$ such as radial basis functions centred at fixed points $\lbrace \mathbf{c}_h \rbrace_{h=1}^H$,

$$\phi_h(\mathbf{x}) = \exp\left[ -\frac{(\mathbf{x} - \mathbf{c}_h)^2}{2r^2} \right],$$

then $y(\mathbf{x}; \mathbf{w})$ is a nonlinear function of $\mathbf{x}$; however, since the dependence of $y$ on the parameters $\mathbf{w}$ is linear, we call this a 'linear' model.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Adaptive Basis Functions -- Neural Network)</span></p>

In a two-layer feedforward neural network with nonlinear hidden units and a linear output, the function can be written:

$$y(\mathbf{x}; \mathbf{w}) = \sum_{h=1}^{H} w_h^{(2)} \tanh\left( \sum_{i=1}^{I} w_{hi}^{(1)} x_i + w_{h0}^{(1)} \right) + w_0^{(2)},$$

where $I$ is the input dimensionality and $\mathbf{w}$ consists of the input weights $\lbrace w_{hi}^{(1)} \rbrace$, hidden unit biases $\lbrace w_{h0}^{(1)} \rbrace$, output weights $\lbrace w_h^{(2)} \rbrace$ and output bias $w_0^{(2)}$. In this model, the dependence of $y$ on $\mathbf{w}$ is nonlinear.

</div>

Having chosen the parameterization, we infer the function $y(\mathbf{x}; \mathbf{w})$ by inferring the parameters $\mathbf{w}$. The posterior probability of the parameters is:

$$P(\mathbf{w} \mid \mathbf{t}_N, \mathbf{X}_N) = \frac{P(\mathbf{t}_N \mid \mathbf{w}, \mathbf{X}_N) P(\mathbf{w})}{P(\mathbf{t}_N \mid \mathbf{X}_N)}.$$

In the Laplace method, we minimize an objective function:

$$M(\mathbf{w}) = -\ln [P(\mathbf{t}_N \mid \mathbf{w}, \mathbf{X}_N) P(\mathbf{w})]$$

with respect to $\mathbf{w}$, locating the locally most probable parameters, then use the curvature of $M$, $\partial^2 M(\mathbf{w}) / \partial w_i \partial w_j$, to define error bars on $\mathbf{w}$. Predictions are made by marginalizing over the parameters:

$$P(t_{N+1} \mid \mathbf{t}_N, \mathbf{X}_{N+1}) = \int \mathrm{d}^H \mathbf{w} \; P(t_{N+1} \mid \mathbf{w}, \mathbf{x}^{(N+1)}) P(\mathbf{w} \mid \mathbf{t}_N, \mathbf{X}_N).$$

#### Nonparametric Approaches -- Splines

In nonparametric methods, predictions are obtained without explicitly parameterizing the unknown function $y(\mathbf{x})$; it lives in the infinite-dimensional space of all continuous functions. The **spline smoothing method** defines the estimator $\hat{y}(\mathbf{x})$ to be the function that minimizes:

$$M(y(x)) = \frac{1}{2} \beta \sum_{n=1}^{N} (y(x^{(n)}) - t_n)^2 + \frac{1}{2} \alpha \int \mathrm{d}x \, [y^{(p)}(x)]^2,$$

where $y^{(p)}$ is the $p$th derivative of $y$. If $p = 2$ then the resulting function $\hat{y}(\mathbf{x})$ is a piecewise cubic spline with 'knots' (discontinuities in its second derivative) at the data points.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Bayesian Interpretation of Splines)</span></p>

The spline estimation method can be interpreted as a Bayesian MAP estimate by identifying the prior for the function $y(x)$ as:

$$\ln P(y(x) \mid \alpha) = -\frac{1}{2} \alpha \int \mathrm{d}x \, [y^{(p)}(x)]^2 + \text{const},$$

and the data probability as:

$$\ln P(\mathbf{t}_N \mid y(x), \beta) = -\frac{1}{2} \beta \sum_{n=1}^{N} (y(x^{(n)}) - t_n)^2 + \text{const}.$$

Then $M(y(x))$ equals minus the log of the posterior probability $P(y(x) \mid \mathbf{t}_N, \alpha, \beta)$, up to an additive constant. The Bayesian perspective additionally allows us to put error bars on the splines estimate, draw typical samples from the posterior distribution, and gives an automatic method for inferring the hyperparameters $\alpha$ and $\beta$.

</div>

The prior distribution defined above is our first example of a Gaussian process. A Gaussian process can be defined as a probability distribution on a space of functions $y(x)$ that can be written in the form:

$$P(y(x) \mid \mu(x), A) = \frac{1}{Z} \exp\left[ -\frac{1}{2} (y(x) - \mu(x))^\mathsf{T} A (y(x) - \mu(x)) \right],$$

where $\mu(x)$ is the mean function and $A$ is a linear operator (with the inner product of two functions defined by $\int \mathrm{d}x \, y(x) z(x)$).

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Representation is Irrelevant for Prediction)</span></p>

From the point of view of prediction, there are two objects of interest: the conditional distribution $P(t_{N+1} \mid \mathbf{t}_N, \mathbf{X}_{N+1})$ and the evidence $P(\mathbf{t}_N \mid \mathbf{X}_N)$. Neither of these quantities makes any reference to the representation of the unknown function $y(x)$. For standard models with fixed basis functions and Gaussian distributions on the unknown parameters, the evidence $P(\mathbf{t}_N \mid \mathbf{X}_N)$ is a multivariate Gaussian with mean zero and a covariance matrix determined by the basis functions. Standard parametric models are therefore simple examples of Gaussian processes.

</div>

### 45.2 From Parametric Models to Gaussian Processes

#### Linear Models

Consider a regression problem using $H$ fixed basis functions (e.g. one-dimensional radial basis functions). Given $N$ input points $\lbrace \mathbf{x}^{(n)} \rbrace$, define the $N \times H$ matrix $\mathbf{R}$ with entries:

$$R_{nh} \equiv \phi_h(\mathbf{x}^{(n)}).$$

The vector $\mathbf{y}_N$ of values of $y(\mathbf{x})$ at the $N$ points is:

$$y_n \equiv \sum_h R_{nh} w_h.$$

If the prior distribution of $\mathbf{w}$ is Gaussian with zero mean, $P(\mathbf{w}) = \text{Normal}(\mathbf{w}; \mathbf{0}, \sigma_w^2 \mathbf{I})$, then $\mathbf{y}$, being a linear function of $\mathbf{w}$, is also Gaussian distributed with mean zero. The covariance matrix of $\mathbf{y}$ is:

$$\mathbf{Q} = \langle \mathbf{y} \mathbf{y}^\mathsf{T} \rangle = \sigma_w^2 \mathbf{R} \mathbf{R}^\mathsf{T}.$$

So the prior distribution of $\mathbf{y}$ is:

$$P(\mathbf{y}) = \text{Normal}(\mathbf{y}; \mathbf{0}, \sigma_w^2 \mathbf{R} \mathbf{R}^\mathsf{T}).$$

This result, that the vector of $N$ function values $\mathbf{y}$ has a Gaussian distribution, is true for any selected points $\mathbf{X}_N$ -- this is the defining property of a Gaussian process.

If each target $t_n$ is assumed to differ from the underlying function value $y_n$ by additive Gaussian noise of variance $\sigma_\nu^2$, then $\mathbf{t}$ also has a Gaussian prior distribution:

$$P(\mathbf{t}) = \text{Normal}(\mathbf{t}; \mathbf{0}, \mathbf{Q} + \sigma_\nu^2 \mathbf{I}).$$

We denote the covariance matrix of $\mathbf{t}$ by $\mathbf{C}$:

$$\mathbf{C} = \mathbf{Q} + \sigma_\nu^2 \mathbf{I} = \sigma_w^2 \mathbf{R} \mathbf{R}^\mathsf{T} + \sigma_\nu^2 \mathbf{I}.$$

The $(n, n')$ entry of $\mathbf{Q}$ is:

$$Q_{nn'} = \sigma_w^2 \sum_h \phi_h(\mathbf{x}^{(n)}) \phi_h(\mathbf{x}^{(n')}),$$

and the $(n, n')$ entry of $\mathbf{C}$ is:

$$C_{nn'} = \sigma_w^2 \sum_h \phi_h(\mathbf{x}^{(n)}) \phi_h(\mathbf{x}^{(n')}) + \delta_{nn'} \sigma_\nu^2.$$

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Radial Basis Functions in 1D)</span></p>

Consider one-dimensional radial basis functions, uniformly spaced with the basis function labelled $h$ centred on $x = h$. Taking the limit $H \to \infty$, the sum over $h$ becomes an integral. To avoid a divergent covariance, we scale $\sigma_w^2$ as $S / (\Delta H)$. Then:

$$Q_{nn'} = S \int_{h_{\min}}^{h_{\max}} \mathrm{d}h \; \exp\left[ -\frac{(x^{(n)} - h)^2}{2r^2} \right] \exp\left[ -\frac{(x^{(n')} - h)^2}{2r^2} \right].$$

Letting the limits of integration be $\pm \infty$, this evaluates to:

$$Q_{nn'} = \sqrt{\pi r^2} \, S \exp\left[ -\frac{(x^{(n')} - x^{(n)})^2}{4r^2} \right].$$

Instead of specifying the prior on functions in terms of basis functions and priors on parameters, the prior can be summarized simply by a **covariance function**:

$$C(x^{(n)}, x^{(n')}) \equiv \theta_1 \exp\left[ -\frac{(x^{(n')} - x^{(n)})^2}{4r^2} \right].$$

</div>

Generalizing from this example, given any valid covariance function $C(\mathbf{x}, \mathbf{x}')$, we can define the covariance matrix for $N$ function values at locations $\mathbf{X}_N$ as:

$$Q_{nn'} = C(\mathbf{x}^{(n)}, \mathbf{x}^{(n')}),$$

and the covariance matrix for $N$ corresponding target values, assuming Gaussian noise, as:

$$C_{nn'} = C(\mathbf{x}^{(n)}, \mathbf{x}^{(n')}) + \sigma_\nu^2 \delta_{nn'}.$$

The prior probability of the $N$ target values $\mathbf{t}$ in the data set is:

$$P(\mathbf{t}) = \text{Normal}(\mathbf{t}; \mathbf{0}, \mathbf{C}) = \frac{1}{Z} e^{-\frac{1}{2} \mathbf{t}^\mathsf{T} \mathbf{C}^{-1} \mathbf{t}}.$$

#### Multilayer Neural Networks and Gaussian Processes

Neal (1996) showed that the properties of a neural network with one hidden layer (as in the adaptive basis functions example) converge to those of a Gaussian process as the number of hidden neurons tends to infinity, if standard 'weight decay' priors are assumed. The covariance function of this Gaussian process depends on the details of the priors assumed for the weights and the activation functions of the hidden units.

### 45.3 Using a Given Gaussian Process Model in Regression

Having formed the covariance matrix $\mathbf{C}$, our task is to infer $t_{N+1}$ given the observed vector $\mathbf{t}_N$. The joint density $P(t_{N+1}, \mathbf{t}_N)$ is a Gaussian, so the conditional distribution:

$$P(t_{N+1} \mid \mathbf{t}_N) = \frac{P(t_{N+1}, \mathbf{t}_N)}{P(\mathbf{t}_N)}$$

is also a Gaussian. We distinguish between different sizes of covariance matrix $\mathbf{C}$ with a subscript, so that $\mathbf{C}_{N+1}$ is the $(N+1) \times (N+1)$ covariance matrix for the vector $\mathbf{t}_{N+1} \equiv (t_1, \ldots, t_{N+1})^\mathsf{T}$. We define submatrices of $\mathbf{C}_{N+1}$:

$$\mathbf{C}_{N+1} \equiv \begin{bmatrix} \mathbf{C}_N & \mathbf{k} \\ \mathbf{k}^\mathsf{T} & \kappa \end{bmatrix}.$$

The posterior distribution is:

$$P(t_{N+1} \mid \mathbf{t}_N) \propto \exp\left[ -\frac{1}{2} \begin{bmatrix} \mathbf{t}_N \\ t_{N+1} \end{bmatrix}^\mathsf{T} \mathbf{C}_{N+1}^{-1} \begin{bmatrix} \mathbf{t}_N \\ t_{N+1} \end{bmatrix} \right].$$

Using the partitioned inverse equations, $\mathbf{C}_{N+1}^{-1}$ can be written in terms of $\mathbf{C}_N^{-1}$:

$$\mathbf{C}_{N+1}^{-1} = \begin{bmatrix} \mathbf{M} & \mathbf{m} \\ \mathbf{m}^\mathsf{T} & m \end{bmatrix},$$

where:

$$m = (\kappa - \mathbf{k}^\mathsf{T} \mathbf{C}_N^{-1} \mathbf{k})^{-1}, \qquad \mathbf{m} = -m \, \mathbf{C}_N^{-1} \mathbf{k}, \qquad \mathbf{M} = \mathbf{C}_N^{-1} + \frac{1}{m} \mathbf{m} \mathbf{m}^\mathsf{T}.$$

Substituting this into the posterior, we find:

$$P(t_{N+1} \mid \mathbf{t}_N) = \frac{1}{Z} \exp\left[ -\frac{(t_{N+1} - \hat{t}_{N+1})^2}{2 \sigma_{\hat{t}_{N+1}}^2} \right],$$

where:

$$\hat{t}_{N+1} = \mathbf{k}^\mathsf{T} \mathbf{C}_N^{-1} \mathbf{t}_N, \qquad \sigma_{\hat{t}_{N+1}}^2 = \kappa - \mathbf{k}^\mathsf{T} \mathbf{C}_N^{-1} \mathbf{k}.$$

The predictive mean $\hat{t}_{N+1}$ and standard deviation $\sigma_{\hat{t}_{N+1}}$ define the error bars on the prediction. Only $\mathbf{C}_N$ needs to be inverted (not $\mathbf{C}_{N+1}$). The computational requirement is of order $N^3$, independent of the number of basis functions $H$.

### 45.4 Examples of Covariance Functions

The only constraint on our choice of covariance function is that it must generate a non-negative-definite covariance matrix for any set of points $\lbrace \mathbf{x}_n \rbrace_{n=1}^N$. We denote the parameters of a covariance function by $\boldsymbol{\theta}$. The covariance matrix of $\mathbf{t}$ has entries:

$$C_{mn} = C(\mathbf{x}^{(m)}, \mathbf{x}^{(n)}; \boldsymbol{\theta}) + \delta_{mn} \mathcal{N}(\mathbf{x}^{(n)}; \boldsymbol{\theta}),$$

where $C$ is the covariance function and $\mathcal{N}$ is a noise model which might be stationary or spatially varying, for example:

$$\mathcal{N}(\mathbf{x}; \boldsymbol{\theta}) = \begin{cases} \theta_3 & \text{for input-independent noise} \\ \exp\left( \sum_{j=1}^{J} \beta_j \phi_j(\mathbf{x}) \right) & \text{for input-dependent noise.} \end{cases}$$

#### Stationary Covariance Functions

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Stationary Covariance Function)</span></p>

A **stationary** covariance function is one that is translation invariant:

$$C(\mathbf{x}, \mathbf{x}'; \boldsymbol{\theta}) = D(\mathbf{x} - \mathbf{x}'; \boldsymbol{\theta}),$$

for some function $D$ (autocovariance function). If additionally $C$ depends only on the *magnitude* of the distance between $\mathbf{x}$ and $\mathbf{x}'$, then it is said to be **homogeneous**. Stationary covariance functions may also be described in terms of the Fourier transform of $D$, known as the **power spectrum** of the Gaussian process. This Fourier transform is necessarily a positive function of frequency.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Gaussian Covariance Function)</span></p>

Let the power spectrum be a Gaussian function of frequency. Since the Fourier transform of a Gaussian is a Gaussian, the autocovariance function is also a Gaussian function of separation. This rederives the covariance function:

$$C(x^{(n)}, x^{(n')}) = \theta_1 \exp\left[ -\frac{(x^{(n')} - x^{(n)})^2}{4r^2} \right].$$

</div>

A popular form for $C$ with hyperparameters $\boldsymbol{\theta} = (\theta_1, \theta_2, \lbrace r_i \rbrace)$ is:

$$C(\mathbf{x}, \mathbf{x}'; \boldsymbol{\theta}) = \theta_1 \exp\left[ -\frac{1}{2} \sum_{i=1}^{I} \frac{(x_i - x_i')^2}{r_i^2} \right] + \theta_2.$$

Here $r_i$ is a lengthscale associated with input $x_i$ -- a very large lengthscale means $y$ is expected to be essentially a constant function of that input (automatic relevance determination). The $\theta_1$ hyperparameter defines the vertical scale of variations, and $\theta_2$ allows the function to be offset from zero by an unknown constant.

Another stationary covariance function is:

$$C(x, x') = \exp(-|x - x'|^\nu), \qquad 0 < \nu \le 2.$$

For $\nu = 2$ this is a special case of the Gaussian covariance function. For $\nu \in (1, 2)$, typical functions are smooth but not analytic. For $\nu \le 1$ typical functions are continuous but not smooth.

A covariance function that models a function periodic with known period $\lambda_i$ in the $i$th input direction is:

$$C(\mathbf{x}, \mathbf{x}'; \boldsymbol{\theta}) = \theta_1 \exp\left[ -\frac{1}{2} \sum_i \left( \frac{\sin\left( \frac{\pi}{\lambda_i} (x_i - x_i') \right)}{r_i} \right)^2 \right].$$

#### Nonstationary Covariance Functions

The simplest nonstationary covariance function corresponds to a linear trend. If the plane $y(\mathbf{x}) = \sum_i w_i x_i + c$ has Gaussian parameters with variances $\sigma_w^2$ and $\sigma_c^2$, then the covariance function is:

$$C_{\text{lin}}(\mathbf{x}, \mathbf{x}'; \lbrace \sigma_w, \sigma_c \rbrace) = \sum_{i=1}^{I} \sigma_w^2 x_i x_i' + \sigma_c^2.$$

### 45.5 Adaptation of Gaussian Process Models

Assuming a form of covariance function with undetermined hyperparameters $\boldsymbol{\theta}$, we would like to learn these hyperparameters from the data. Ideally we integrate over them:

$$P(t_{N+1} \mid \mathbf{x}_{N+1}, \mathcal{D}) = \int P(t_{N+1} \mid \mathbf{x}_{N+1}, \boldsymbol{\theta}, \mathcal{D}) P(\boldsymbol{\theta} \mid \mathcal{D}) \, \mathrm{d}\boldsymbol{\theta}.$$

This integral is usually intractable. Two approaches:

1. **MAP approximation:** Use the most probable values of the hyperparameters: $P(t_{N+1} \mid \mathbf{x}_{N+1}, \mathcal{D}) \simeq P(t_{N+1} \mid \mathbf{x}_{N+1}, \mathcal{D}, \boldsymbol{\theta}_{\text{MP}})$.
2. **Monte Carlo integration** over $\boldsymbol{\theta}$ numerically.

#### Gradient of the Evidence

The posterior probability of $\boldsymbol{\theta}$ is:

$$P(\boldsymbol{\theta} \mid \mathcal{D}) \propto P(\mathbf{t}_N \mid \mathbf{X}_N, \boldsymbol{\theta}) P(\boldsymbol{\theta}).$$

The log of the evidence is:

$$\ln P(\mathbf{t}_N \mid \mathbf{X}_N, \boldsymbol{\theta}) = -\frac{1}{2} \ln \det \mathbf{C}_N - \frac{1}{2} \mathbf{t}_N^\mathsf{T} \mathbf{C}_N^{-1} \mathbf{t}_N - \frac{N}{2} \ln 2\pi,$$

and its derivative with respect to a hyperparameter $\theta$ is:

$$\frac{\partial}{\partial \theta} \ln P(\mathbf{t}_N \mid \mathbf{X}_N, \boldsymbol{\theta}) = -\frac{1}{2} \text{Trace}\left( \mathbf{C}_N^{-1} \frac{\partial \mathbf{C}_N}{\partial \theta} \right) + \frac{1}{2} \mathbf{t}_N^\mathsf{T} \mathbf{C}_N^{-1} \frac{\partial \mathbf{C}_N}{\partial \theta} \mathbf{C}_N^{-1} \mathbf{t}_N.$$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Computational Challenges)</span></p>

Two problems arise when optimizing hyperparameters. First, the evidence may be **multimodal** -- suitable priors and sensible optimization strategies often eliminate poor optima. Second, the evaluation of the gradient of the log likelihood requires computing $\mathbf{C}_N^{-1}$. Any exact inversion method (Cholesky decomposition, LU decomposition, or Gauss--Jordan elimination) has a cost of order $N^3$, so calculating gradients becomes time consuming for large training data sets. Approximate methods based on iterative techniques can achieve cost $\mathcal{O}(N^2)$.

</div>

### 45.6 Classification

Gaussian processes can be integrated into classification modelling once we identify a variable that can sensibly be given a Gaussian process prior. In a binary classification problem, we define $a_n \equiv a(\mathbf{x}^{(n)})$ such that:

$$P(t_n = 1 \mid a_n) = \frac{1}{1 + e^{-a_n}}.$$

Large positive values of $a$ correspond to probabilities close to one; large negative values to probabilities close to zero. The prior belief that $P(t_n = 1)$ should be a smoothly varying function of $\mathbf{x}$ is embodied by defining $a(\mathbf{x})$ to have a Gaussian process prior.

Since the likelihood function is not a Gaussian function of $a_n$, the posterior distribution of $\mathbf{a}$ given observations $\mathbf{t}$ is not Gaussian, and the normalization constant $P(\mathbf{t}_N \mid \mathbf{X}_N)$ cannot be written down analytically. Implementation approaches include:

- **Laplace approximations** (Barber and Williams, 1997)
- **Monte Carlo methods** (Neal, 1997b)
- **Variational Gaussian process classifier** (Gibbs and MacKay, 2000): obtain tractable upper and lower bounds for the unnormalized posterior density over $\mathbf{a}$, $P(\mathbf{t}_N \mid \mathbf{a}) P(\mathbf{a})$, parameterized by variational parameters adjusted for the tightest fit.

### 45.7 Discussion

Gaussian processes are moderately simple to implement and use. Because very few parameters need to be determined by hand (generally only the priors on the hyperparameters), they are useful tools for automated tasks where fine tuning is not possible, without sacrificing performance for this simplicity.

It is easy to construct Gaussian processes with particular desired properties, for example, straightforward automatic relevance determination models.

One obvious problem with Gaussian processes is the computational cost associated with inverting an $N \times N$ matrix. The cost of direct methods of inversion becomes prohibitive when $N$ is greater than about 1000.

Gaussian processes and **support vector machines** have a lot in common -- both are kernel-based predictors, the kernel being another name for the covariance function. A Bayesian version of support vectors exploits this connection.

## Chapter 46: Deconvolution

### 46.1 Traditional Image Reconstruction Methods

#### Optimal Linear Filters

In many imaging problems, the data measurements $\lbrace d_n \rbrace$ are linearly related to the underlying image $\mathbf{f}$:

$$d_n = \sum_k R_{nk} f_k + n_n,$$

where $\mathbf{n}$ denotes the noise corrupting the data. In the case of a camera producing a blurred picture, $\mathbf{f}$ is the true image, $\mathbf{d}$ is the blurred and noisy picture, and the linear operator $\mathbf{R}$ is a convolution defined by the point spread function.

#### Bayesian Derivation

We assume that the noise $\mathbf{n}$ is Gaussian and independent, with known standard deviation $\sigma_\nu$:

$$P(\mathbf{d} \mid \mathbf{f}, \sigma_\nu, \mathcal{H}) = \frac{1}{(2\pi \sigma_\nu^2)^{N/2}} \exp\left( -\sum_n (d_n - \sum_k R_{nk} f_k)^2 \Big/ (2\sigma_\nu^2) \right).$$

We assume the prior probability of the image is also Gaussian, with a scale parameter $\sigma_f$:

$$P(\mathbf{f} \mid \sigma_f, \mathcal{H}) = \frac{\det^{-\frac{1}{2}} \mathbf{C}}{(2\pi \sigma_f^2)^{K/2}} \exp\left( -\sum_{k,k'} f_k C_{kk'} f_{k'} \Big/ (2\sigma_f^2) \right),$$

where $\mathbf{C}$ is a symmetric, full rank matrix. If we assume no correlations among the pixels, $\mathbf{C} = \mathbf{I}$. The more sophisticated 'intrinsic correlation function' model uses $\mathbf{C} = [\mathbf{G} \mathbf{G}^\mathsf{T}]^{-1}$, where $\mathbf{G}$ is a convolution that takes us from an uncorrelated 'hidden' image to the real correlated image.

The posterior probability of the image $\mathbf{f}$ given the data $\mathbf{d}$ is:

$$P(\mathbf{f} \mid \mathbf{d}, \sigma_\nu, \sigma_f, \mathcal{H}) = \frac{P(\mathbf{d} \mid \mathbf{f}, \sigma_\nu, \mathcal{H}) P(\mathbf{f} \mid \sigma_f, \mathcal{H})}{P(\mathbf{d} \mid \sigma_\nu, \sigma_f, \mathcal{H})}.$$

Since the posterior is the product of two Gaussian functions of $\mathbf{f}$, it is also a Gaussian, and can be summarized by its mean (the **most probable image** $\mathbf{f}_\text{MP}$) and its covariance matrix:

$$\boldsymbol{\Sigma}_{\mathbf{f}|\mathbf{d}} \equiv [-\nabla \nabla \log P(\mathbf{f} \mid \mathbf{d}, \sigma_\nu, \sigma_f, \mathcal{H})]^{-1}.$$

We can find $\mathbf{f}_\text{MP}$ by differentiating the log of the posterior and solving for the derivative being zero:

$$\mathbf{f}_\text{MP} = \left[ \mathbf{R}^\mathsf{T} \mathbf{R} + \frac{\sigma_\nu^2}{\sigma_f^2} \mathbf{C} \right]^{-1} \mathbf{R}^\mathsf{T} \mathbf{d}.$$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Optimal Linear Filter)</span></p>

The operator $\left[ \mathbf{R}^\mathsf{T} \mathbf{R} + \frac{\sigma_\nu^2}{\sigma_f^2} \mathbf{C} \right]^{-1} \mathbf{R}^\mathsf{T}$ is called the **optimal linear filter**. When $\frac{\sigma_\nu^2}{\sigma_f^2} \mathbf{C}$ can be neglected, the optimal linear filter is the pseudoinverse $[\mathbf{R}^\mathsf{T} \mathbf{R}]^{-1} \mathbf{R}^\mathsf{T}$. The term $\frac{\sigma_\nu^2}{\sigma_f^2} \mathbf{C}$ regularizes this ill-conditioned inverse. The optimal linear filter can also be expressed as:

$$\text{Optimal linear filter} = \mathbf{C}^{-1} \mathbf{R}^\mathsf{T} \left[ \mathbf{R} \mathbf{C}^{-1} \mathbf{R}^\mathsf{T} + \frac{\sigma_\nu^2}{\sigma_f^2} \mathbf{I} \right]^{-1}.$$

</div>

#### Minimum Square Error Derivation

The non-Bayesian derivation of the optimal linear filter assumes that we estimate the true image $\mathbf{f}$ by a linear function of the data: $\hat{\mathbf{f}} = \mathbf{W} \mathbf{d}$. The operator $\mathbf{W}$ is 'optimized' by minimizing the expected sum-squared error between $\hat{\mathbf{f}}$ and the true image. Introducing $\mathbf{F} \equiv \langle f_{j'} f_j \rangle$ (cf. $\sigma_f^2 \mathbf{C}^{-1}$ in the Bayesian derivation), the optimal linear filter is:

$$\mathbf{W}_\text{opt} = \mathbf{F} \mathbf{R}^\mathsf{T} \left[ \mathbf{R} \mathbf{F} \mathbf{R}^\mathsf{T} + \sigma_\nu^2 \mathbf{I} \right]^{-1}.$$

If we identify $\mathbf{F} = \sigma_f^2 \mathbf{C}^{-1}$, this reproduces the Bayesian optimal linear filter. The ad hoc assumptions in the non-Bayesian derivation were the choice of a quadratic error measure and the decision to use a linear estimator.

#### Other Image Models

The Gaussian models leading to the optimal linear filter are poorly matched to the real world -- for example, they fail to enforce positivity of pixel intensities. The **maximum entropy** model for image deconvolution assigns an entropic prior:

$$P(\mathbf{f} \mid \alpha, \mathbf{m}, \mathcal{H}_\text{Classic}) = \exp(\alpha S(\mathbf{f}, \mathbf{m})) / Z,$$

where:

$$S(\mathbf{f}, \mathbf{m}) = \sum_i (f_i \ln(m_i / f_i) + f_i - m_i).$$

This model enforces positivity; the parameter $\alpha$ defines a characteristic dynamic range.

The 'intrinsic-correlation-function maximum-entropy' model introduces spatial correlations by writing $\mathbf{f} = \mathbf{G} \mathbf{h}$, where $\mathbf{G}$ is a convolution with an intrinsic correlation function, and putting a classic maxent prior on the underlying hidden image $\mathbf{h}$.

#### Probabilistic Movies

Having found the most probable image $\mathbf{f}_\text{MP}$ and its covariance $\boldsymbol{\Sigma}_{\mathbf{f}|\mathbf{d}}$, error bars can be visualized using a correlated random walk around the posterior distribution. For a Gaussian posterior, we create a correlated sequence of unit normal random vectors $\mathbf{n}$ using:

$$\mathbf{n}^{(t+1)} = c \mathbf{n}^{(t)} + s \mathbf{z},$$

where $\mathbf{z}$ is a unit normal random vector and $c^2 + s^2 = 1$ ($c$ controls persistence). We then render the image sequence:

$$\mathbf{f}^{(t)} = \mathbf{f}_\text{MP} + \boldsymbol{\Sigma}_{\mathbf{f}|\mathbf{d}}^{1/2} \mathbf{n}^{(t)},$$

where $\boldsymbol{\Sigma}_{\mathbf{f}|\mathbf{d}}^{1/2}$ is the Cholesky decomposition of $\boldsymbol{\Sigma}_{\mathbf{f}|\mathbf{d}}$.

### 46.2 Supervised Neural Networks for Image Deconvolution

Neural network researchers exploit the following strategy: interpret the computations performed by a standard algorithm as a parameterized mapping from input to output, and call this mapping a neural network; then adapt the parameters to data so as to produce another mapping that solves the task better. This data-driven adaptation can only improve performance.

Standard algorithms can be bettered this way because:

1. **Discriminative vs. generative training:** Algorithms are often not designed to optimize the real objective function. For example, in speech recognition, a hidden Markov model maximizes the generative probability, but the real objective is to *discriminate* between different words. Discriminative training of hidden Markov models improves performance.
2. **Greater flexibility:** Some adaptive parameters might have been viewed as fixed features by the original designers. A flexible network can find properties in the data that were not included in the original model.

### 46.3 Deconvolution in Humans

A huge fraction of our brain is devoted to vision. A remarkable and neglected feature is that the raw image falling on the retina is severely blurred: while most people can see with a resolution of about 1 arcminute, *the image on our retina is blurred through a point spread function of width as large as 5 arcminutes*. We are able to resolve pixels twenty-five times smaller in area than the blob produced on our retina by any point source.

Our cornea and lens, like a lens made of ordinary glass, refract blue light more strongly than red (**chromatic aberration**). Typically our eyes focus correctly for green light, so a single white dot produces a sharply focussed green dot surrounded by a broader red blob superposed on an even broader blue blob on the retina.

One of the main functions of early visual processing must be to **deconvolve** this chromatic aberration. The centre-surround receptive fields of retinal ganglion cells and cells in the lateral geniculate nucleus likely serve this purpose, rather than 'feature extraction' or 'edge detection'.

## Part VI: Sparse Graph Codes

The central problem of communication theory is to construct an encoding and a decoding system that make it possible to communicate reliably over a noisy channel. During the 1990s, remarkable progress was made towards the Shannon limit, using codes that are defined in terms of sparse random graphs and decoded by a simple probability-based message-passing algorithm.

In a **sparse-graph code**, the nodes in the graph represent the transmitted bits and the constraints they satisfy. For a linear code with a codeword length $N$ and rate $R = K/N$, the number of constraints is of order $M = N - K$. What makes a sparse-graph code special is that each constraint involves only a small number of variables in the graph, so the number of edges scales roughly linearly with $N$, rather than quadratically.

Four families of sparse-graph codes are covered: **low-density parity-check codes**, **turbo codes**, and **repeat--accumulate codes** (excellent for error-correction); and **digital fountain codes** (outstanding for erasure-correction). All these codes can be decoded by a local message-passing algorithm on the graph -- the **sum--product algorithm** -- and while this algorithm is not a perfect maximum likelihood decoder, the empirical results are record-breaking.

## Chapter 47: Low-Density Parity-Check Codes

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Low-Density Parity-Check Code)</span></p>

A **low-density parity-check code** (or **Gallager code**) is a block code that has a parity-check matrix $\mathbf{H}$, every row and column of which is 'sparse'.

A **regular** Gallager code is a low-density parity-check code in which every column of $\mathbf{H}$ has the same weight $j$ and every row has the same weight $k$; regular Gallager codes are constructed at random subject to these constraints.

</div>

### 47.1 Theoretical Properties

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Properties of Gallager Codes)</span></p>

Low-density parity-check codes, in spite of their simple construction, are good codes, *given an optimal decoder* (in the sense of section 11.4). Furthermore, they have good distance (in the sense of section 13.2). These two results hold for any column weight $j \ge 3$. Furthermore, there are sequences of low-density parity-check codes in which $j$ increases gradually with $N$, in such a way that $j/N$ still goes to zero, that are *very good*, and that have very good distance.

However, we don't have an optimal decoder, and decoding low-density parity-check codes is an NP-complete problem.

</div>

### 47.2 Practical Decoding

Given a channel output $\mathbf{r}$, we wish to find the codeword $\mathbf{t}$ whose likelihood $P(\mathbf{r} \mid \mathbf{t})$ is biggest. All the effective decoding strategies for low-density parity-check codes are message-passing algorithms. The best algorithm is the **sum--product algorithm**, also known as iterative probabilistic decoding or belief propagation.

For any memoryless channel, there are two approaches to the decoding problem, both of which lead to the generic problem 'find the $\mathbf{x}$ that maximizes

$$P^*(\mathbf{x}) = P(\mathbf{x}) \, \mathbb{1}[\mathbf{H}\mathbf{x} = \mathbf{z}]',$$

where $P(\mathbf{x})$ is a separable distribution on a binary vector $\mathbf{x}$, and $\mathbf{z}$ is another binary vector.

#### The Codeword Decoding Viewpoint

The prior distribution over codewords,

$$P(\mathbf{t}) \propto \mathbb{1}[\mathbf{H}\mathbf{t} = \mathbf{0} \bmod 2],$$

can be represented by a factor graph, with the factorization:

$$P(\mathbf{t}) \propto \prod_m \mathbb{1}\!\left[ \sum_{n \in \mathcal{N}(m)} t_n = 0 \bmod 2 \right].$$

The posterior distribution over codewords is given by multiplying this prior by the likelihood:

$$P(\mathbf{t} \mid \mathbf{r}) \propto \prod_m \mathbb{1}\!\left[ \sum_{n \in \mathcal{N}(m)} t_n = 0 \right] \prod_n P(r_n \mid t_n).$$

#### The Syndrome Decoding Viewpoint

Alternatively, we can view the channel output in terms of a binary received vector $\mathbf{r}$ and a noise vector $\mathbf{n}$, with a probability distribution $P(\mathbf{n})$. For a binary symmetric channel, the noise is $\mathbf{r} = \mathbf{t} + \mathbf{n}$, the syndrome $\mathbf{z} = \mathbf{H}\mathbf{r}$, and noise model $P(n_n = 1) = f$. The joint probability of the noise $\mathbf{n}$ and syndrome $\mathbf{z} = \mathbf{H}\mathbf{n}$ can be factored as:

$$P(\mathbf{n}, \mathbf{z}) = \prod_n P(n_n) \prod_m \mathbb{1}\!\left[ z_m = \sum_{n \in \mathcal{N}(m)} n_n \right].$$

Both decoding viewpoints involve essentially the same graph. Either version can be expressed as the generic decoding problem 'find the $\mathbf{x}$ that maximizes $P^*(\mathbf{x}) = P(\mathbf{x}) \, \mathbb{1}[\mathbf{H}\mathbf{x} = \mathbf{z}]$': in the codeword viewpoint, $\mathbf{x}$ is the codeword $\mathbf{t}$ and $\mathbf{z}$ is $\mathbf{0}$; in the syndrome viewpoint, $\mathbf{x}$ is the noise $\mathbf{n}$ and $\mathbf{z}$ is the syndrome.

### 47.3 Decoding with the Sum--Product Algorithm

We aim, given the observed checks, to compute the marginal posterior probabilities $P(x_n = 1 \mid \mathbf{z}, \mathbf{H})$ for each $n$. It is hard to compute these exactly because the graph contains many cycles. However, it is interesting to implement the decoding algorithm that would be appropriate if there were no cycles, on the assumption that the errors introduced might be relatively small.

The graph connecting the checks and bits is a **bipartite graph**: bits connect only to checks, and vice versa. On each iteration, a probability ratio is propagated along each edge in the graph, and each bit node $x_n$ updates its probability that it should be in state 1.

We denote the set of bits $n$ that participate in check $m$ by $\mathcal{N}(m) \equiv \lbrace n : H_{mn} = 1 \rbrace$. Similarly $\mathcal{M}(n) \equiv \lbrace m : H_{mn} = 1 \rbrace$. We denote $\mathcal{N}(m)$ with bit $n$ excluded by $\mathcal{N}(m) \setminus n$.

The algorithm has two alternating parts, in which quantities $q_{mn}$ and $r_{mn}$ associated with each edge in the graph are iteratively updated:

- $q_{mn}^x$ is meant to be the probability that bit $n$ of $\mathbf{x}$ has the value $x$, given the information obtained via checks other than check $m$.
- $r_{mn}^x$ is meant to be the probability of check $m$ being satisfied if bit $n$ of $\mathbf{x}$ is considered fixed at $x$ and the other bits have a separable distribution given by the probabilities $\lbrace q_{mn'} : n' \in \mathcal{N}(m) \setminus n \rbrace$.

The algorithm would produce the exact posterior probabilities of all the bits after a fixed number of iterations if the bipartite graph defined by $\mathbf{H}$ contained no cycles.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Sum--Product Decoding for LDPC Codes)</span></p>

**Initialization.** Let $p_n^0 = P(x_n = 0)$ (the prior probability that bit $x_n$ is 0), and let $p_n^1 = P(x_n = 1) = 1 - p_n^0$. For every $(n, m)$ such that $H_{mn} = 1$, the variables $q_{mn}^0$ and $q_{mn}^1$ are initialized to $p_n^0$ and $p_n^1$ respectively.

**Horizontal step.** Run through the checks $m$ and compute for each $n \in \mathcal{N}(m)$ two probabilities: $r_{mn}^0$, the probability of the observed value of $z_m$ arising when $x_n = 0$; and $r_{mn}^1$, the probability when $x_n = 1$:

$$r_{mn}^0 = \sum_{\lbrace x_{n'}: n' \in \mathcal{N}(m) \setminus n \rbrace} P\!\left(z_m \mid x_n = 0, \lbrace x_{n'} \rbrace \right) \prod_{n' \in \mathcal{N}(m) \setminus n} q_{mn'}^{x_{n'}}.$$

A convenient implementation uses the differences $\delta q_{mn} \equiv q_{mn}^0 - q_{mn}^1$ and $\delta r_{mn} \equiv r_{mn}^0 - r_{mn}^1$, linked by the identity:

$$\delta r_{mn} = (-1)^{z_m} \prod_{n' \in \mathcal{N}(m) \setminus n} \delta q_{mn'}.$$

We recover $r_{mn}^0 = \frac{1}{2}(1 + \delta r_{mn})$ and $r_{mn}^1 = \frac{1}{2}(1 - \delta r_{mn})$.

**Vertical step.** Take the computed values of $r_{mn}^0$ and $r_{mn}^1$ and update the probabilities $q_{mn}^0$ and $q_{mn}^1$. For each $n$:

$$q_{mn}^0 = \alpha_{mn} \, p_n^0 \prod_{m' \in \mathcal{M}(n) \setminus m} r_{m'n}^0, \qquad q_{mn}^1 = \alpha_{mn} \, p_n^1 \prod_{m' \in \mathcal{M}(n) \setminus m} r_{m'n}^1,$$

where $\alpha_{mn}$ is chosen such that $q_{mn}^0 + q_{mn}^1 = 1$.

We also compute the 'pseudoposterior probabilities' $q_n^0$ and $q_n^1$:

$$q_n^0 = \alpha_n \, p_n^0 \prod_{m \in \mathcal{M}(n)} r_{mn}^0, \qquad q_n^1 = \alpha_n \, p_n^1 \prod_{m \in \mathcal{M}(n)} r_{mn}^1.$$

These are used to create a tentative decoding $\hat{\mathbf{x}}$, the consistency of which is used to decide whether the algorithm can halt. (Halt if $\mathbf{H}\hat{\mathbf{x}} = \mathbf{z}$.)

**Termination.** Set $\hat{x}_n$ to 1 if $q_n^1 > 0.5$ and see if the checks $\mathbf{H}\hat{\mathbf{x}} = \mathbf{z} \bmod 2$ are all satisfied, halting when they are, and declaring a failure if some maximum number of iterations (e.g. 200 or 1000) occurs without successful decoding.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Detected vs. Undetected Errors)</span></p>

In our stop-when-it's-done procedure, 'undetected' errors occur if the decoder finds an $\hat{\mathbf{x}}$ satisfying $\mathbf{H}\hat{\mathbf{x}} = \mathbf{z} \bmod 2$ that is not equal to the true $\mathbf{x}$. 'Detected' errors occur if the algorithm runs for the maximum number of iterations without finding a valid decoding. Undetected errors are of scientific interest because they reveal distance properties of a code. In engineering practice, blocks known to contain detected errors should preferably be labelled as such.

</div>

#### Computational Cost

In a brute-force approach, the time to create the generator matrix scales as $N^3$. Encoding involves only binary arithmetic, so it takes considerably less time than the simulation of the Gaussian channel. Decoding involves approximately $6Nj$ floating-point multiplies per iteration, so the total number of operations per decoded bit (assuming 20 iterations) is about $120t/R$, independent of blocklength. For typical codes, this is about 800 operations.

### 47.4 Pictorial Demonstration of Gallager Codes

#### Encoding

The encoding operation for a Gallager code whose parity-check matrix is a $10\,000 \times 20\,000$ matrix with three 1s per column creates transmitted vectors consisting of $10\,000$ source bits and $10\,000$ parity-check bits. The high density of the *generator* matrix is revealed by the fact that changing one of the $10\,000$ source bits alters about half of the parity-check bits.

#### Iterative Decoding

The transmission is sent over a channel with noise level $f = 7.5\%$ and the iterative probabilistic decoding process progressively refines the best guess, bit by bit. The decoder halts after the 13th iteration when the best guess violates no parity checks, yielding an error-free decoding. For this code and a channel with $f = 7.5\%$, failures happen about once in every $100\,000$ transmissions.

#### Variation of Performance with Code Parameters

As Shannon would predict, increasing the blocklength $N$ leads to improved performance. The dependence on $j$ follows a different pattern. Given an *optimal* decoder, the best performance would be obtained for the codes closest to random codes, i.e. those with largest $j$. However, the sum--product decoder makes poor progress in dense graphs, so the best performance is obtained for a small value of $j$. Among the values of $j$ tested, $j = 3$ is the best for a blocklength of 816, down to a block error probability of $10^{-5}$.

### 47.5 Density Evolution

One way to study the decoding algorithm is to imagine it running on an infinite tree-like graph with the same local topology as the Gallager code's graph. The larger the matrix $\mathbf{H}$, the closer its decoding properties should approach those of the infinite graph.

Imagine an infinite belief network with no loops, in which every bit $x_n$ connects to $j$ checks and every check $z_m$ connects to $k$ bits. We consider the iterative flow of information in this network, and examine the average entropy of one bit as a function of number of iterations. Successful decoding will occur only if the average entropy of a bit decreases to zero as the number of iterations increases.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Decoding Threshold)</span></p>

The boundary between successful decoding (entropy collapses to zero) and failure (entropy remains nonzero) is called the **threshold** of the decoding algorithm. For the binary symmetric channel, Monte Carlo simulation shows that the threshold for regular $(j, k) = (4, 8)$ codes is about $f = 0.075$. Richardson and Urbanke (2001a) have derived thresholds for regular codes by direct analytic methods. Some thresholds for rate-$1/2$ codes:

| $(j, k)$ | $f_{\max}$ |
| --- | --- |
| $(3, 6)$ | 0.084 |
| $(4, 8)$ | 0.076 |
| $(5, 10)$ | 0.068 |

The Shannon limit for rate-$1/2$ codes is $f_{\max} = 0.11$.

</div>

For practical purposes, the computational cost of density evolution can be reduced by making Gaussian approximations to the probability distributions over the messages, updating only the parameters of these approximations. These techniques produce diagrams known as **EXIT charts**.

### 47.6 Improving Gallager Codes

Two methods have been found for enhancing the performance of Gallager codes.

#### Clump Bits and Checks Together

We can make Gallager codes in which the variable nodes are grouped together into metavariables (e.g. 3 binary variables), and the check nodes similarly grouped into metachecks. One way to set the wiring is to work in a finite field $GF(q)$ such as $GF(4)$ or $GF(8)$, define low-density parity-check matrices using elements of $GF(q)$, and translate binary messages into $GF(q)$ using a mapping. When messages are passed during decoding, they are probabilities and likelihoods over *conjunctions* of binary variables.

With carefully optimized constructions, the resulting codes over $GF(8)$ and $GF(16)$ perform nearly one decibel better than comparable binary Gallager codes. The computational cost for decoding in $GF(q)$ scales as $q \log q$ if the appropriate Fourier transform is used in the check nodes.

#### Make the Graph Irregular

The second method, introduced by Luby *et al.* (2001b), is to make the graphs **irregular**. Instead of giving all variable nodes the same degree $j$, we can have some nodes with degree 2, some 3, some 4, and a few with degree 20. Check nodes can also be given unequal degrees -- this helps improve performance on erasure channels, but for the Gaussian channel the best graphs have regular check degrees.

Making the binary code irregular gives a win of about 0.4 dB; switching from $GF(2)$ to $GF(16)$ gives about 0.6 dB; combining both features gives about 0.9 dB over the regular binary Gallager code. Methods for optimizing the *profile* of a Gallager code (its number of rows and columns of each degree) have led to low-density parity-check codes whose performance, when decoded by the sum--product algorithm, is within a hair's breadth of the Shannon limit.

#### Algebraic Constructions

The performance of regular Gallager codes can also be enhanced by designing the code to have **redundant sparse constraints**. A *difference-set cyclic code*, for example, has $N = 273$ and $K = 191$, but the code satisfies not $M = 82$ but $N$, i.e. 273 low-weight constraints. It is impossible to make random Gallager codes that have anywhere near this much redundancy among their checks. The difference-set cyclic code performs about 0.7 dB better than an equivalent random Gallager code.

### 47.7 Fast Encoding of Low-Density Parity-Check Codes

The standard encoding method finds a generator matrix $\mathbf{G}$ by Gaussian elimination (cost of order $M^3$) and then encodes each block by multiplying it by $\mathbf{G}$ (cost of order $M^2$). Faster methods exist.

#### Staircase Codes

Certain low-density parity-check matrices with $M$ columns of weight 2 or less can be encoded easily in linear time. If the matrix has a *staircase* structure, and if the data $\mathbf{s}$ are loaded into the first $K$ bits, then the $M$ parity bits $\mathbf{p}$ can be computed from left to right in linear time:

$$p_1 = \sum_{n=1}^{K} H_{1n} s_n, \quad p_2 = p_1 + \sum_{n=1}^{K} H_{2n} s_n, \quad \ldots, \quad p_M = p_{M-1} + \sum_{n=1}^{K} H_{Mn} s_n.$$

If we call two parts of the $\mathbf{H}$ matrix $[\mathbf{H}_s | \mathbf{H}_p]$, we can describe the encoding in two steps: first compute an intermediate parity vector $\mathbf{v} = \mathbf{H}_s \mathbf{s}$; then pass $\mathbf{v}$ through an accumulator to create $\mathbf{p}$. The cost of this encoding method is linear if the sparsity of $\mathbf{H}$ is exploited.

#### Fast Encoding of General Low-Density Parity-Check Codes

Richardson and Urbanke (2001b) demonstrated an elegant method by which the encoding cost of any low-density parity-check code can be reduced from the straightforward method's $M^2$ to a cost of $N + g^2$, where $g$, the *gap*, is hopefully a small constant scaling as a small fraction of $N$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Richardson--Urbanke Fast Encoding)</span></p>

The parity-check matrix is rearranged into **approximate lower-triangular form**:

$$\mathbf{H} = \begin{bmatrix} \mathbf{A} & \mathbf{B} & \mathbf{T} \\ \mathbf{C} & \mathbf{D} & \mathbf{E} \end{bmatrix},$$

where $\mathbf{T}$ is lower triangular with 1s on the diagonal. The source vector $\mathbf{s}$ of length $K = N - M$ is encoded into a transmission $\mathbf{t} = [\mathbf{s}, \mathbf{p}_1, \mathbf{p}_2]$ as follows:

1. Compute the upper syndrome of the source vector: $\mathbf{z}_A = \mathbf{A}\mathbf{s}$. (Linear time.)
2. Find a setting of the second parity bits, $\mathbf{p}_2^A$, such that the upper syndrome is zero: $\mathbf{p}_2^A = -\mathbf{T}^{-1} \mathbf{z}_A$. (Linear time by back-substitution.)
3. Compute the lower syndrome of the vector $[\mathbf{s}, \mathbf{0}, \mathbf{p}_2^A]$: $\mathbf{z}_B = \mathbf{C}\mathbf{s} - \mathbf{E}\mathbf{p}_2^A$. (Linear time.)
4. Define the matrix $\mathbf{F} \equiv -\mathbf{E}\mathbf{T}^{-1}\mathbf{B} + \mathbf{D}$ and find its inverse $\mathbf{F}^{-1}$ (a dense $g \times g$ matrix, computed once, cost $g^3$). Set the first parity bits: $\mathbf{p}_1 = -\mathbf{F}^{-1} \mathbf{z}_B$. (Cost $g^2$.)
5. Discard the tentative $\mathbf{p}_2^A$ and find the new upper syndrome: $\mathbf{z}_C = \mathbf{z}_A + \mathbf{B}\mathbf{p}_1$. (Linear time.)
6. Find a setting of the second parity bits: $\mathbf{p}_2 = -\mathbf{T}^{-1} \mathbf{z}_C$. (Linear time by back-substitution.)

</div>

## Chapter 48: Convolutional Codes and Turbo Codes

This chapter follows tightly on from Chapter 25. It makes use of the ideas of codes and trellises and the forward--backward algorithm.

When we studied linear block codes, we described them in three ways:

1. The **generator matrix** describes how to turn a string of $K$ arbitrary source bits into a transmission of $N$ bits.
2. The **parity-check matrix** specifies the $M = N - K$ parity-check constraints that a valid codeword satisfies.
3. The **trellis** of the code describes its valid codewords in terms of paths through a trellis with labelled edges.

We now describe convolutional codes in two ways: first, in terms of mechanisms for generating transmissions $\mathbf{t}$ from source bits $\mathbf{s}$; and second, in terms of trellises that describe the constraints satisfied by valid transmissions.

### 48.1 Linear-Feedback Shift-Registers

We generate a transmission with a convolutional code by putting a source stream through a linear filter. This filter makes use of a shift register, linear output functions, and, possibly, linear feedback.

The rectangular box surrounding the bits $z_1 \ldots z_7$ indicates the *memory* of the filter, also known as its *state*. All three filters have one input and two outputs. On each clock cycle, the source supplies one bit, and the filter outputs two bits $t^{(a)}$ and $t^{(b)}$. By concatenating together these bits we can obtain from our source stream $s_1 s_2 s_3 \ldots$ a transmission stream $t_1^{(a)} t_1^{(b)} t_2^{(a)} t_2^{(b)} t_3^{(a)} t_3^{(b)} \ldots$ Because there are two transmitted bits for every source bit, the codes have rate $1/2$. Because these filters require $k = 7$ bits of memory, the codes they define are known as a **constraint-length 7 codes**.

Convolutional codes come in three flavours:

#### Systematic Nonrecursive

The filter has no feedback. One of the output bits, $t^{(a)}$, is identical to the source bit $s$. This encoder is called **systematic**, because the source bits are reproduced transparently in the transmitted stream, and **nonrecursive**, because it has no feedback. The other transmitted bit $t^{(b)}$ is a linear function of the state of the filter. One way of describing that function is as a dot product (modulo 2) between two binary vectors of length $k + 1$: a binary vector $\mathbf{g}^{(b)} = (1, 1, 1, 0, 1, 0, 1, 1)$ and the state vector $\mathbf{z} = (z_k, z_{k-1}, \ldots, z_1, z_0)$. The vector $\mathbf{g}^{(b)}$ has $g_\kappa^{(b)} = 1$ for every $\kappa$ where there is a tap (a downward pointing arrow) from state bit $z_\kappa$ into the transmitted bit $t^{(b)}$.

A convenient way to describe these binary tap vectors is in octal. Thus, this filter makes use of the tap vector $353_8$.

#### Nonsystematic Nonrecursive

The filter has no feedback, but it is not systematic. It makes use of two tap vectors $\mathbf{g}^{(a)}$ and $\mathbf{g}^{(b)}$ to create its two transmitted bits. This encoder is thus **nonsystematic** and **nonrecursive**. Because of their added complexity, nonsystematic codes can have error-correcting abilities superior to those of systematic nonrecursive codes with the same constraint length.

#### Systematic Recursive

The filter uses the taps that formerly made up $\mathbf{g}^{(a)}$ to make a linear signal that is fed back into the shift register along with the source bit. The output $t^{(b)}$ is a linear function of the state vector as before. The other output is $t^{(a)} = s$, so this filter is systematic. A recursive code is conventionally identified by an octal ratio, e.g., figure 48.1c's code is denoted by $(247/371)_8$.

### 48.2 Equivalence of Systematic Recursive and Nonsystematic Nonrecursive Codes

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Code Equivalence)</span></p>

The two filters (nonsystematic nonrecursive and systematic recursive with the same taps) are **code-equivalent** in that the *sets* of codewords that they define are identical. For every codeword of the nonsystematic nonrecursive code we can choose a source stream for the other encoder such that its output is identical (and *vice versa*).

</div>

To prove this, denote by $p$ the quantity $\sum_{\kappa=1}^{k} g_\kappa^{(a)} z_\kappa$. In the nonrecursive case we have

$$t^{(a)} = p \oplus z_0^{\text{nonrecursive}},$$

whereas in the recursive case we have

$$z_0^{\text{recursive}} = t^{(a)} \oplus p.$$

Substituting for $t^{(a)}$, and using $p \oplus p = 0$ we immediately find

$$z_0^{\text{recursive}} = z_0^{\text{nonrecursive}}.$$

Thus, any codeword of a nonsystematic nonrecursive code is a codeword of a systematic recursive code with the same taps -- the same taps in the sense that there are vertical arrows in all the same places, though one of the arrows points up instead of down.

Now, while these two codes are equivalent, the two encoders behave differently. The nonrecursive encoder has a **finite impulse response**: if one puts in a string that is all zeroes except for a single one, the resulting output stream contains a finite number of ones. Once the one bit has passed through all the states of the memory, the delay line returns to the all-zero state. The recursive encoder, in contrast, has an **infinite impulse response**. The response settles into a periodic state with period equal to three clock cycles.

In general a linear-feedback shift-register with $k$ bits of memory has an impulse response that is periodic with a period that is at most $2^k - 1$, corresponding to the filter visiting every non-zero state in its state space.

### 48.3 Decoding Convolutional Codes

The receiver receives a bit stream, and wishes to infer the state sequence and thence the source stream. The posterior probability of each bit can be found by the **sum--product algorithm** (also known as the forward--backward or BCJR algorithm), which was introduced in section 25.3. The most probable state sequence can be found using the **min--sum algorithm** of section 25.3 (also known as the Viterbi algorithm).

The min--sum algorithm seeks the path through the trellis that uses as many solid lines as possible; more precisely, it minimizes the cost of the path, where the cost is zero for a solid line, one for a thick dotted line, and two for a thin dotted line.

#### Unequal Protection

A defect of the convolutional codes presented thus far is that they offer unequal protection to the source bits. The last source bit is less well protected than the other source bits. This unequal protection of bits motivates the **termination** of the trellis.

A terminated trellis ensures that when any codeword is completed, the filter state is $0000$. Termination slightly reduces the number of source bits used per codeword. Here, four source bits are turned into parity bits because the $k = 4$ memory bits must be returned to zero.

### 48.4 Turbo Codes

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Turbo Code)</span></p>

An $(N, K)$ **turbo code** is defined by a number of constituent convolutional encoders (often, two) and an equal number of **interleavers** which are $K \times K$ permutation matrices. Without loss of generality, we take the first interleaver to be the identity matrix. A string of $K$ source bits is encoded by feeding them into each constituent encoder in the order defined by the associated interleaver, and transmitting the bits that come out of each constituent encoder.

</div>

Often the first constituent encoder is chosen to be a systematic encoder, and the second is a non-systematic one of rate 1 that emits parity bits only. The transmitted codeword then consists of $K$ source bits followed by $M_1$ parity bits generated by the first convolutional code and $M_2$ parity bits from the second. The resulting turbo code has rate $1/3$.

The turbo code can be represented by a factor graph in which the two trellises are represented by two large rectangular nodes; the $K$ source bits and the first $M_1$ parity bits participate in the first trellis and the $K$ source bits and the last $M_2$ parity bits participate in the second trellis. Each codeword bit participates in either one or two trellises, depending on whether it is a parity bit or a source bit.

If a turbo code of smaller rate such as $1/2$ is required, a standard modification to the rate-$1/3$ code is to **puncture** some of the parity bits.

#### Decoding Turbo Codes

Turbo codes are decoded using the sum--product algorithm described in Chapter 26. On the first iteration, each trellis receives the channel likelihoods, and runs the forward--backward algorithm to compute, for each bit, the relative likelihood of its being 1 or 0, given the information about the other bits. These likelihoods are then passed across from each trellis to the other, and multiplied by the channel likelihoods on the way. We are then ready for the second iteration: the forward--backward algorithm is run again in each trellis using the updated probabilities. After about ten or twenty such iterations, it's hoped that the correct decoding will be found.

As a stopping criterion, for each time-step in each trellis, we identify the most probable edge, according to the local messages. If these most probable edges join up into two valid paths, one in each trellis, and if these two paths are consistent with each other, it is reasonable to stop. If a maximum number of iterations is reached without this stopping criterion being satisfied, a decoding error can be reported.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Error Floor of Turbo Codes)</span></p>

Turbo codes as described here have excellent performance down to decoded error probabilities of about $10^{-5}$, but randomly-constructed turbo codes tend to have an **error floor** starting at that level. This error floor is caused by low-weight codewords. To reduce the height of the error floor, one can attempt to modify the random construction to increase the weight of these low-weight codewords. The tweaking of turbo codes is a black art, and it never succeeds in totally eliminating low-weight codewords; the low-weight codewords can be eliminated only by sacrificing the turbo code's excellent performance. In contrast, low-density parity-check codes rarely have error floors, as long as their number of weight-2 columns is not too large.

</div>

### 48.5 Parity-Check Matrices of Convolutional Codes and Turbo Codes

We close by discussing the parity-check matrix of a rate-$1/2$ convolutional code viewed as a linear block code. We adopt the convention that the $N$ bits of one block are made up of the $N/2$ bits $t^{(a)}$ followed by the $N/2$ bits $t^{(b)}$.

A convolutional code has a **low-density parity-check matrix**. The parity-check matrix of a turbo code can be written down by listing the constraints satisfied by the two constituent trellises. So turbo codes are also special cases of low-density parity-check codes. If a turbo code is punctured, it no longer necessarily has a low-density parity-check matrix, but it always has a **generalized parity-check matrix** that is sparse, as explained in the next chapter.

## Chapter 49: Repeat--Accumulate Codes

In Chapter 1 we discussed a very simple and not very effective method for communicating over a noisy channel: the repetition code. We now discuss a code that is almost as simple, and whose performance is outstandingly good. *Repeat--accumulate codes* were studied by Divsalar *et al.* (1998) for theoretical purposes, as simple turbo-like codes that might be more amenable to analysis than messy turbo codes. Their practical performance turned out to be just as good as other sparse-graph codes.

### 49.1 The Encoder

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Repeat--Accumulate Encoder)</span></p>

1. Take $K$ source bits: $s_1 s_2 s_3 \ldots s_K$.
2. Repeat each bit three times, giving $N = 3K$ bits: $s_1 s_1 s_1 s_2 s_2 s_2 s_3 s_3 s_3 \ldots s_K s_K s_K$.
3. Permute these $N$ bits using a random permutation (a fixed random permutation -- the same one for every codeword). Call the permuted string $\mathbf{u}$: $u_1 u_2 u_3 u_4 u_5 u_6 u_7 u_8 u_9 \ldots u_N$.
4. Transmit the **accumulated sum**:

$$t_1 = u_1, \quad t_2 = t_1 + u_2 \pmod{2}, \quad \ldots \quad t_n = t_{n-1} + u_n \pmod{2}, \quad t_N = t_{N-1} + u_N \pmod{2}.$$

5. That's it!

</div>

### 49.2 Graph

The graph of a repeat--accumulate code uses four types of node: equality constraints, intermediate binary variables (black circles), parity constraints, and the transmitted bits (white circles). The source sets the values of the black bits at the bottom, three at a time, and the accumulator computes the transmitted bits along the top.

This graph is a factor graph for the prior probability over codewords, with the circles being binary variable nodes, and the squares representing two types of factor nodes. Each parity constraint contributes a factor of the form $\mathbf{1}[\sum x = 0 \mod 2]$; each equality constraint contributes a factor of the form $\mathbf{1}[x_1 = x_2 = x_3]$.

### 49.3 Decoding

The repeat--accumulate code is normally decoded using the sum--product algorithm on the factor graph. The top box represents the trellis of the accumulator, including the channel likelihoods. In the first half of each iteration, the top trellis receives likelihoods for every transition in the trellis, and runs the forward--backward algorithm so as to produce likelihoods for each variable node. In the second half of the iteration, these likelihoods are multiplied together at the equality constraint nodes to produce new likelihood messages to send back to the trellis.

As with Gallager codes and turbo codes, the stop-when-it's-done decoding method can be applied, so it is possible to distinguish between undetected errors (which are caused by low-weight codewords in the code) and detected errors (where the decoder gets stuck and knows that it has failed to find a valid answer).

The performance of six randomly-constructed repeat--accumulate codes on the Gaussian channel is staggeringly good for such a simple code. If one does not mind the error floor which kicks in at about a block error probability of $10^{-4}$, the performance is impressive.

### 49.4 Empirical Distribution of Decoding Times

It is interesting to study the number of iterations $\tau$ of the sum--product algorithm required to decode a sparse-graph code. Given one code and a set of channel conditions, the decoding time varies randomly from trial to trial. The histogram of decoding times follows a power law, $P(\tau) \propto \tau^{-p}$, for large $\tau$. The power $p$ depends on the signal-to-noise ratio and becomes smaller (so that the distribution is more heavy-tailed) as the signal-to-noise ratio decreases. These power laws have been observed in repeat--accumulate codes and in irregular and regular Gallager codes.

### 49.5 Generalized Parity-Check Matrices

Forney (2001) introduced the idea of a *normal graph* in which the only nodes are check nodes and equality nodes, and all variable nodes have degree one or two; variable nodes with degree two can be represented on edges that connect a check node to an equality node. The **generalized parity-check matrix** is a graphical way of representing normal graphs.

In a parity-check matrix, the columns are transmitted bits, and the rows are linear constraints. In a generalized parity-check matrix, additional columns may be included, which represent state variables that are not transmitted. One way of thinking of these state variables is that they are punctured from the code before transmission. State variables are indicated by a horizontal line above the corresponding columns.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Generalized Parity-Check Matrix)</span></p>

A **generalized parity-check matrix** is a pair $\lbrace \mathbf{A}, \mathbf{p} \rbrace$, where $\mathbf{A}$ is a binary matrix and $\mathbf{p}$ is a list of the punctured bits. The matrix defines a set of *valid* vectors $\mathbf{x}$, satisfying

$$\mathbf{A}\mathbf{x} = 0;$$

for each valid vector there is a codeword $\mathbf{t}(\mathbf{x})$ that is obtained by puncturing from $\mathbf{x}$ the bits indicated by $\mathbf{p}$. For any one code there are many generalized parity-check matrices.

</div>

The **rate** of a code with generalized parity-check matrix $\lbrace \mathbf{A}, \mathbf{p} \rbrace$ can be estimated as follows. If $\mathbf{A}$ is $L \times M'$, and $\mathbf{p}$ punctures $S$ bits and selects $N$ bits for transmission ($L = N + S$), then the effective number of constraints on the codeword, $M$, is

$$M = M' - S,$$

the number of source bits is

$$K = N - M = L - M',$$

and the rate is greater than or equal to

$$R = 1 - \frac{M}{N} = 1 - \frac{M' - S}{L - S}.$$

#### Examples

**Repetition code.** The generator matrix, parity-check matrix, and generalized parity-check matrix of a simple rate-$1/3$ repetition code can be represented schematically.

**Systematic low-density generator-matrix code.** In an $(N, K)$ systematic low-density generator-matrix code, there are no state variables. A transmitted codeword $\mathbf{t}$ of length $N$ is given by $\mathbf{t} = \mathbf{G}^{\mathrm{T}} \mathbf{s}$, where $\mathbf{G}^{\mathrm{T}} = \begin{bmatrix} \mathbf{I}_K \\ \mathbf{P} \end{bmatrix}$, with $\mathbf{I}_K$ denoting the $K \times K$ identity matrix, and $\mathbf{P}$ being a very sparse $M \times K$ matrix, where $M = N - K$. The parity-check matrix of this code is $\mathbf{H} = [\mathbf{P} | \mathbf{I}_M]$.

**Non-systematic low-density generator-matrix code.** In an $(N, K)$ non-systematic low-density generator-matrix code, a transmitted codeword $\mathbf{t}$ of length $N$ is given by $\mathbf{t} = \mathbf{G}^{\mathrm{T}} \mathbf{s}$, where $\mathbf{G}^{\mathrm{T}}$ is a very sparse $N \times K$ matrix. The generalized parity-check matrix of this code is $\mathbf{A} = [\overline{\mathbf{G}^{\mathrm{T}}} | \mathbf{I}_N]$, and the corresponding generalized parity-check equation is $\mathbf{A}\mathbf{x} = 0$, where $\mathbf{x} = \begin{bmatrix} \mathbf{s} \\ \mathbf{t} \end{bmatrix}$. Whereas the parity-check matrix of this simple code is typically a complex, dense matrix, the generalized parity-check matrix retains the underlying simplicity of the code.

**Convolutional codes.** In a non-systematic, non-recursive convolutional code, the source bits, which play the role of state bits, are fed into a delay-line and two linear functions of the delay-line are transmitted.

**Concatenation.** 'Parallel concatenation' of two codes is represented by aligning the matrices of two codes in such a way that the 'source bits' line up, and by adding blocks of zero-entries to the matrix such that the state bits and parity bits of the two codes occupy separate columns. In 'serial concatenation', the columns corresponding to the *transmitted* bits of the first code are aligned with the columns corresponding to the *source* bits of the second code.

**Turbo codes.** A turbo code is the parallel concatenation of two convolutional codes.

**Repeat--accumulate codes.** The generalized parity-check matrices of a rate-$1/3$ repeat--accumulate code are shown schematically. Repeat-accumulate codes are equivalent to staircase codes.

**Intersection.** The generalized parity-check matrix of the intersection of two codes is made by stacking their generalized parity-check matrices on top of each other in such a way that all the transmitted bits' columns are correctly aligned, and any punctured bits associated with the two component codes occupy separate columns.

## Chapter 50: Digital Fountain Codes

Digital fountain codes are record-breaking sparse-graph codes for channels with erasures.

Channels with erasures are of great importance. For example, files sent over the internet are chopped into packets, and each packet is either received without error or not received. A simple channel model describing this situation is a **$q$-ary erasure channel**, which has (for all inputs in the input alphabet $\lbrace 0, 1, 2, \ldots, q-1 \rbrace$) a probability $1 - f$ of transmitting the input without error, and probability $f$ of delivering the output '?'. The alphabet size $q$ is $2^l$, where $l$ is the number of bits in a packet.

Common methods for communicating over such channels employ a feedback channel from receiver to sender to control the retransmission of erased packets. These simple retransmission protocols work regardless of the erasure probability $f$, but purists who have learned their Shannon theory will feel they are wasteful.

According to Shannon, there is no need for the feedback channel: the capacity of the forward channel is $(1 - f)l$ bits, whether or not we have feedback. The wastefulness of the simple retransmission protocols is especially evident in the case of a **broadcast channel with erasures** -- channels where one sender broadcasts to many receivers, and each receiver receives a random fraction $(1-f)$ of the packets.

### 50.1 Reed--Solomon Codes and Their Limitations

The classic block codes for erasure correction are called **Reed--Solomon codes**. An $(N, K)$ Reed--Solomon code (over an alphabet of size $q = 2^l$) has the ideal property that if any $K$ of the $N$ transmitted symbols are received then the original $K$ source symbols can be recovered. But Reed--Solomon codes have the disadvantage that they are practical only for small $K$, $N$, and $q$: standard implementations of encoding and decoding have a cost of order $K(N-K) \log_2 N$ packet operations. Furthermore, with a Reed--Solomon code, as with any block code, one must estimate the erasure probability $f$ and choose the code rate $R = K/N$ *before* transmission.

### 50.2 LT Codes

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(LT Code -- Digital Fountain Code)</span></p>

**LT codes** (Luby Transform codes), invented by Luby in 1998, are digital fountain codes. The encoder is a fountain that produces an endless supply of water drops (encoded packets); the original source file has a size of $Kl$ bits, and each drop contains $l$ encoded bits. Anyone who wishes to receive the encoded file holds a bucket under the fountain and collects drops until the number of drops in the bucket is a little larger than $K$. They can then recover the original file.

Digital fountain codes are **rateless** in the sense that the number of encoded packets that can be generated from the source message is potentially limitless; and the number of encoded packets generated can be determined on the fly. The source data can be decoded from any set of $K'$ encoded packets, for $K'$ slightly larger than $K$ (in practice, about 5% larger).

</div>

Digital fountain codes also have fantastically small encoding and decoding complexities. With probability $1 - \delta$, $K$ packets can be communicated with average encoding and decoding costs both of order $K \ln(K/\delta)$ packet operations.

Luby calls these codes **universal** because they are simultaneously near-optimal for every erasure channel, and they are very efficient as the file length $K$ grows. The overhead $K' - K$ is of order $\sqrt{K}(\ln(K/\delta))^2$.

### 50.3 A Digital Fountain's Encoder

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Digital Fountain Encoder)</span></p>

Each encoded packet $t_n$ is produced from the source file $s_1 s_2 s_3 \ldots s_K$ as follows:

1. Randomly choose the degree $d_n$ of the packet from a degree distribution $\rho(d)$; the appropriate choice of $\rho$ depends on the source file size $K$.
2. Choose, uniformly at random, $d_n$ distinct input packets, and set $t_n$ equal to the bitwise sum, modulo 2, of those $d_n$ packets. This sum can be done by successively exclusive-or-ing the packets together.

</div>

This encoding operation defines a graph connecting encoded packets to source packets. If the mean degree $\bar{d}$ is significantly smaller than $K$ then the graph is sparse. We can think of the resulting code as an irregular low-density generator-matrix code.

### 50.4 The Decoder

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Digital Fountain Decoder)</span></p>

Decoding a sparse-graph code is especially easy in the case of an erasure channel. The decoder's task is to recover $\mathbf{s}$ from $\mathbf{t} = \mathbf{G}\mathbf{s}$, where $\mathbf{G}$ is the matrix associated with the graph. The decoding algorithm uses message-passing where all messages are either *completely uncertain* messages or *completely certain* messages:

1. Find a check node $t_n$ that is connected to *only one* source packet $s_k$. (If there is no such check node, the algorithm halts and fails to recover all the source packets.)
   - (a) Set $s_k = t_n$.
   - (b) Add $s_k$ to all checks $t_{n'}$ that are connected to $s_k$: $t_{n'} := t_{n'} + s_k$ for all $n'$ such that $G_{n'k} = 1$.
   - (c) Remove all the edges connected to the source packet $s_k$.
2. Repeat (1) until all $\lbrace s_k \rbrace$ are determined.

</div>

### 50.5 Designing the Degree Distribution

The probability distribution $\rho(d)$ of the degree is a critical part of the design: occasional encoded packets must have high degree (i.e., $d$ similar to $K$) in order to ensure that there are not some source packets that are connected to no-one. Many packets must have low degree, so that the decoding process can get started, and keep going, and so that the total number of addition operations involved in the encoding and decoding is kept small.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Ideal Soliton Distribution)</span></p>

The **ideal soliton distribution** is defined by:

$$\rho(1) = 1/K, \qquad \rho(d) = \frac{1}{d(d-1)} \quad \text{for } d = 2, 3, \ldots, K.$$

The expected degree under this distribution is roughly $\ln K$.

</div>

*In expectation*, this distribution achieves the ideal behaviour where just one check node has degree one at each iteration. However, this degree distribution works poorly in practice, because fluctuations around the expected behaviour make it very likely that at some point in the decoding process there will be no degree-one check nodes; and, furthermore, a few source nodes will receive no connections at all.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Robust Soliton Distribution)</span></p>

The **robust soliton distribution** has two extra parameters, $c$ and $\delta$; it is designed to ensure that the expected number of degree-one checks is about

$$S \equiv c \ln(K/\delta) \sqrt{K},$$

rather than 1, throughout the decoding process. The parameter $\delta$ is a bound on the probability that the decoding fails to run to completion after a certain number $K'$ of packets have been received. We define a positive function

$$\tau(d) = \begin{cases} \frac{S}{K} \frac{1}{d} & \text{for } d = 1, 2, \ldots (K/S) - 1 \\ \frac{S}{K} \ln(S/\delta) & \text{for } d = K/S \\ 0 & \text{for } d > K/S \end{cases}$$

then add the ideal soliton distribution $\rho$ to $\tau$ and normalize to obtain the robust soliton distribution, $\mu$:

$$\mu(d) = \frac{\rho(d) + \tau(d)}{Z},$$

where $Z = \sum_d \rho(d) + \tau(d)$. The number of encoded packets required at the receiving end to ensure that the decoding can run to completion, with probability at least $1 - \delta$, is $K' = KZ$.

</div>

Luby's key result is that (for an appropriate value of the constant $c$) receiving $K' = K + 2\ln(S/\delta)S$ checks ensures that all packets can be recovered with probability at least $1 - \delta$.

In practice, LT codes can be tuned so that a file of original size $K \simeq 10\,000$ packets is recovered with an overhead of about 5%.

### 50.6 Applications

Digital fountain codes are an excellent solution in a wide variety of situations.

#### Storage

You wish to make a backup of a large file, but are aware that your magnetic tapes and hard drives are all unreliable in the sense that catastrophic failures, in which some stored packets are permanently lost within one device, occur at a rate of something like $10^{-3}$ per day. A digital fountain can be used to spray encoded packets all over the place, on every storage device available. Then to recover the backup file, whose size was $K$ packets, one simply needs to find $K' \simeq K$ packets from anywhere. Corrupted packets do not matter; we simply skip over them and find more packets elsewhere.

This method of storage also has advantages in terms of speed of file recovery: if files were stored using the digital fountain principle, with the digital drops stored in one or more consecutive sectors on the drive, then one would never need to endure the delay of re-reading a packet; packet loss would become less important, and the hard drive could consequently be operated faster.

#### Broadcast

For the broadcast scenario with ten thousand subscribers wishing to receive a digital movie from a broadcaster, the broadcaster can send the movie in packets encoded by a digital fountain. Each subscriber collects drops until they have $K' \simeq K$ packets, then decodes. No feedback channel is needed, and the broadcaster does not need to know the erasure probability of any subscriber's channel.

## Appendix A: Notation

A summary of notation used throughout the book.

- **$P(A \mid B, C)$** is pronounced 'the probability that $A$ is true *given that* $B$ is true *and* $C$ is true'. Or, more briefly, 'the probability of $A$ given $B$ and $C$'.
- **$\log x$** means the base-two logarithm, $\log_2 x$; **$\ln x$** means the natural logarithm, $\log_e x$.
- **$\hat{s}$** -- a 'hat' over a variable denotes a guess or estimator. So $\hat{s}$ is a guess at the value of $s$.
- **Integrals.** There is no difference between $\int f(u) \, \mathrm{d}u$ and $\int \mathrm{d}u \, f(u)$. The integrand is $f(u)$ in both cases.

### Products and Combinations

$$\prod_{n=1}^{N} n = 1 \times 2 \times 3 \times \cdots \times N = N! = \exp\!\left[\sum_{n=1}^{N} \ln n\right].$$

The **binomial coefficient** $\binom{N}{n}$ is pronounced '$N$ choose $n$', and it is the number of ways of selecting an unordered set of $n$ objects from a set of size $N$:

$$\binom{N}{n} = \frac{N!}{(N-n)!\, n!}.$$

### The Gamma Function

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Gamma Function)</span></p>

The **gamma function** is defined by $\Gamma(x) \equiv \int_0^\infty \mathrm{d}u \; u^{x-1} e^{-u}$, for $x > 0$. It is an extension of the factorial function to real number arguments. In general, $\Gamma(x+1) = x\Gamma(x)$, and for integer arguments, $\Gamma(x+1) = x!$. The **digamma function** is defined by $\Psi(x) \equiv \frac{\mathrm{d}}{\mathrm{d}x} \ln \Gamma(x)$.

For large $x$ (for practical purposes, $0.1 \le x \le \infty$):

$$\ln \Gamma(x) \simeq \left(x - \tfrac{1}{2}\right) \ln(x) - x + \tfrac{1}{2} \ln 2\pi + O(1/x).$$

For small $x$ (for practical purposes, $0 \le x \le 0.5$):

$$\ln \Gamma(x) \simeq \ln \frac{1}{x} - \gamma_e x + O(x^2),$$

where $\gamma_e$ is Euler's constant.

</div>

### Inverse Functions, Derivatives, and the Error Function

- **$H_2^{-1}(1 - R/C)$**: Just as $\sin^{-1}(s)$ denotes the inverse function to $s = \sin(x)$, so $H_2^{-1}(h)$ is the inverse function to $h = H_2(x)$.
- **$f'(x)$**: Often, a 'prime' denotes differentiation: $f'(x) \equiv \frac{\mathrm{d}}{\mathrm{d}x} f(x)$. A dot denotes differentiation with respect to time: $\dot{x} \equiv \frac{\mathrm{d}}{\mathrm{d}t} x$. However, the prime is also a useful indicator for 'another variable', for example $x'$ might denote 'the new value of $x$'. The rule is: if a prime occurs in an expression that could be a function, such as $f'(x)$ or $h'(y)$, then it denotes differentiation; otherwise it indicates 'another variable'.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Error Function)</span></p>

The **error function** $\Phi(z)$ is the cumulative probability of a standard (variance $= 1$) normal distribution:

$$\Phi(z) \equiv \int_{-\infty}^{z} \exp(-z^2/2) / \sqrt{2\pi} \;\mathrm{d}z.$$

</div>

### Matrices and Linear Algebra

- **$\mathcal{E}(r)$** or **$\mathcal{E}[r]$** is pronounced 'the expected value of $r$' or 'the expectation of $r$', and it is the mean value of $r$. Another symbol for 'expected value' is the pair of angle-brackets, $\langle r \rangle$.
- **$\lvert x \rvert$**: The vertical bars have two meanings. If $\mathcal{A}$ is a set, then $\lvert \mathcal{A} \rvert$ denotes the number of elements in the set; if $x$ is a number, then $\lvert x \rvert$ is the absolute value of $x$.
- **$[\mathbf{A} | \mathbf{P}]$**: Here, $\mathbf{A}$ and $\mathbf{P}$ are matrices with the same number of rows. $[\mathbf{A} | \mathbf{P}]$ denotes the double-width matrix obtained by putting $\mathbf{A}$ alongside $\mathbf{P}$. The vertical bar is used to avoid confusion with the product $\mathbf{AP}$.
- **$\mathbf{x}^{\mathrm{T}}$**: The superscript $\mathrm{T}$ is pronounced 'transpose'. Transposing a row-vector turns it into a column vector: $(1, 2, 3)^{\mathrm{T}} = \begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix}$. Vectors indicated by bold face type ($\mathbf{x}$) are column vectors. If $M_{ij}$ is the entry in row $i$ and column $j$ of matrix $\mathbf{M}$, and $\mathbf{N} = \mathbf{M}^{\mathrm{T}}$, then $N_{ji} = M_{ij}$.
- **Trace $\mathbf{M}$** and **$\det \mathbf{M}$**: The trace of a matrix is the sum of its diagonal elements, $\text{Trace}\,\mathbf{M} = \sum_i M_{ii}$. The determinant of $\mathbf{M}$ is denoted $\det \mathbf{M}$.
- **$\delta_{mn}$**: The Kronecker delta is the identity matrix: $\delta_{mn} = \begin{cases} 1 & \text{if } m = n \\ 0 & \text{if } m \neq n. \end{cases}$ Another name for the identity matrix is $\mathbf{I}$ or $\mathbf{1}$. Sometimes a subscript indicates the size: $\mathbf{1}_K$ ($K \times K$).

### Special Functions and Symbols

- **$\delta(x)$**: The delta function has the property $\int \mathrm{d}x \; f(x)\delta(x) = f(0)$.
- **$\mathbf{1}[S]$**: The truth function, which is 1 if the proposition $S$ is true and 0 otherwise. For example, the number of positive numbers in the set $T = \lbrace -2, 1, 3 \rbrace$ can be written $\sum_{x \in T} \mathbf{1}[x > 0]$.
- **':=' vs '='**: In an algorithm, $x := y$ means that the variable $x$ is updated by assigning it the value of $y$. In contrast, $x = y$ is a proposition, a statement that $x$ is equal to $y$.

## Appendix B: Some Physics

### B.1 About Phase Transitions

A system with states $\mathbf{x}$ in contact with a heat bath at temperature $T = 1/\beta$ has probability distribution

$$P(\mathbf{x} \mid \beta) = \frac{1}{Z(\beta)} \exp(-\beta E(\mathbf{x})).$$

The partition function is

$$Z(\beta) = \sum_{\mathbf{x}} \exp(-\beta E(\mathbf{x})).$$

The inverse temperature $\beta$ can be interpreted as defining an exchange rate between entropy and energy. $(1/\beta)$ is the amount of energy that must be given to a heat bath to increase its entropy by one nat.

For any system with a finite number of states, the function $Z(\beta)$ is evidently a continuous function of $\beta$, since it is simply a sum of exponentials. Moreover, all the derivatives of $Z(\beta)$ with respect to $\beta$ are continuous too. Phase transitions correspond to values of $\beta$ and $V$ (called critical points) at which the derivatives of $Z$ have discontinuities or divergences. Therefore:

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Necessary Conditions for Phase Transitions)</span></p>

Only systems with an **infinite number of states** can show phase transitions. Furthermore, only systems with **long-range correlations** show phase transitions. Long-range correlations do not require long-range energetic couplings; for example, a magnet has only short-range couplings (between adjacent spins) but these are sufficient to create long-range order.

</div>

If we make the system large by simply grouping together $N$ independent systems whose partition function is $Z_{(1)}(\beta)$, then nothing interesting happens. The partition function for $N$ independent identical systems is simply $Z_{(N)}(\beta) = [Z_{(1)}(\beta)]^N$. The natural way to look at the partition function is in the logarithm: $\ln Z_{(N)}(\beta) = N \ln Z_{(1)}(\beta)$. Duplicating the original system $N$ times simply scales up all properties like the energy and heat capacity of the system by a factor of $N$.

#### Why Are Points at Which Derivatives Diverge Interesting?

The derivatives of $\ln Z$ describe properties like the heat capacity of the system (the second derivative) or its fluctuations in energy. If the second derivative of $\ln Z$ diverges at a temperature $1/\beta$, then the heat capacity of the system diverges there, which means it can absorb or release energy without changing temperature (think of ice melting in ice water); when the system is at equilibrium at that temperature, its energy fluctuates a lot, in contrast to the normal law-of-large-numbers behaviour, where the energy only varies by one part in $\sqrt{N}$.

#### A Toy System That Shows a Phase Transition

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Coupled Spins Phase Transition)</span></p>

Imagine a collection of $N$ coupled spins that have the following energy as a function of their state $\mathbf{x} \in \lbrace 0, 1 \rbrace^N$:

$$E(\mathbf{x}) = \begin{cases} -N\epsilon & \mathbf{x} = (0, 0, 0, \ldots, 0) \\ 0 & \text{otherwise}. \end{cases}$$

This energy function describes a ground state in which all the spins are aligned in the zero direction; the energy per spin in this state is $-\epsilon$. If any spin changes state then the energy is zero. This model is like an extreme version of a magnetic interaction, which encourages pairs of spins to be aligned.

The partition function of the coupled-spin system is

$$Z(\beta) = e^{\beta N \epsilon} + 2^N - 1.$$

The function $\ln Z(\beta) = \ln\!\left(e^{\beta N \epsilon} + 2^N - 1\right)$ has low temperature behaviour $\ln Z(\beta) \simeq N\beta\epsilon$ ($\beta \to \infty$) and high temperature behaviour $\ln Z(\beta) \simeq N \ln 2$ ($\beta \to 0$). The two asymptotes intersect at the critical point

$$\beta_c = \frac{\ln 2}{\epsilon}.$$

In the limit $N \to \infty$, the graph of $\ln Z(\beta)$ becomes more and more sharply bent at this point. The second derivative of $\ln Z$, which describes the variance of the energy of the system, has a peak value at $\beta = \ln 2/\epsilon$ roughly equal to $N^2 \epsilon^2 / 4$, which corresponds to the system spending half of its time in the ground state and half its time in the other states. At this critical point, the heat capacity per spin is proportional to $N$, which, for infinite $N$, is infinite.

</div>

#### Types of Phase Transitions

Phase transitions can be categorized into **first-order** and **continuous** transitions. In a first-order phase transition, there is a discontinuous change of one or more order-parameters; in a continuous transition, all order-parameters change continuously. [An order-parameter is a scalar function of the state of the system; or, to be precise, the expectation of such a function.]

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Phase Transitions and Typicality)</span></p>

In the vicinity of a critical point, the concept of 'typicality' defined in Chapter 4 does not hold. For example, our toy system, at its critical point, has a 50% chance of being in a state with energy $-N\epsilon$, and roughly a $1/2^{N+1}$ chance of being in each of the other states that have energy zero. It is thus not the case that $\ln 1/P(\mathbf{x})$ is very likely to be close to the entropy of the system at this point, unlike a system with $N$ i.i.d. components.

Remember that information content ($\ln 1/P(\mathbf{x})$) and energy are very closely related. If typicality holds, then the system's energy has negligible fluctuations, and *vice versa*.

</div>

## Appendix C: Some Mathematics

### C.1 Finite Field Theory

*Most linear codes are expressed in the language of Galois theory.*

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Field)</span></p>

A **field** $F$ is a set $F = \lbrace 0, F' \rbrace$ such that:

1. $F$ forms an Abelian group under an addition operation '$+$', with $0$ being the identity. (Abelian means all elements commute, i.e., $a + b = b + a$.)
2. $F'$ forms an Abelian group under a multiplication operation '$\cdot$'; multiplication of any element by 0 yields 0.
3. These operations satisfy the distributive rule $(a + b) \cdot c = a \cdot c + b \cdot c$.

For example, the real numbers form a field, with '$+$' and '$\cdot$' denoting ordinary addition and multiplication.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Galois Field)</span></p>

A **Galois field** $GF(q)$ is a field with a finite number of elements $q$. A unique Galois field exists for any $q = p^m$, where $p$ is a prime number and $m$ is a positive integer; there are no other finite fields.

</div>

**$GF(2)$.** The addition and multiplication tables for $GF(2)$ are the rules of addition and multiplication modulo 2.

**$GF(p)$.** For any prime number $p$, the addition and multiplication rules are those for ordinary addition and multiplication, modulo $p$.

**$GF(4)$.** The rules for $GF(p^m)$, with $m > 1$, are *not* those of ordinary addition and multiplication modulo $p^m$. The elements can be related to *polynomials*: polynomial functions of $x$ of degree 1 and with coefficients that are elements of $GF(2)$. The polynomials obey the addition and multiplication rules of $GF(4)$ *if* addition and multiplication are modulo the polynomial $x^2 + x + 1$, and the coefficients of the polynomials are from $GF(2)$.

| Element | Polynomial | Bit pattern |
|---|---|---|
| 0 | 0 | 00 |
| 1 | 1 | 01 |
| $A$ | $x$ | 10 |
| $B$ | $x + 1$ | 11 |

**$GF(8)$.** We can denote the elements of $GF(8)$ by $\lbrace 0, 1, A, B, C, D, E, F \rbrace$. Each element can be mapped onto a polynomial over $GF(2)$. The multiplication and addition operations are given by multiplication and addition of the polynomials, modulo $x^3 + x + 1$.

| Element | Polynomial | Binary |
|---|---|---|
| 0 | 0 | 000 |
| 1 | 1 | 001 |
| $A$ | $x$ | 010 |
| $B$ | $x + 1$ | 011 |
| $C$ | $x^2$ | 100 |
| $D$ | $x^2 + 1$ | 101 |
| $E$ | $x^2 + x$ | 110 |
| $F$ | $x^2 + x + 1$ | 111 |

Galois fields are relevant to linear codes because when generalizing a binary generator matrix $\mathbf{G}$ and binary vector $\mathbf{s}$ to a matrix and vector with elements from a larger set, it would be convenient if, for random $\mathbf{s}$, the product $\mathbf{Gs}$ produced all elements in the enlarged set with equal probability. This uniform distribution is easiest to guarantee if these elements form a group under both addition and multiplication. Galois fields, by their definition, avoid symmetry-breaking effects.

### C.2 Eigenvectors and Eigenvalues

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Eigenvectors)</span></p>

A **right-eigenvector** of a square matrix $\mathbf{A}$ is a non-zero vector $\mathbf{e}_{\mathrm{R}}$ that satisfies

$$\mathbf{A}\mathbf{e}_{\mathrm{R}} = \lambda \mathbf{e}_{\mathrm{R}},$$

where $\lambda$ is the eigenvalue associated with that eigenvector. The eigenvalue may be a real number or complex number and it may be zero. Eigenvectors may be real or complex.

A **left-eigenvector** of a matrix $\mathbf{A}$ is a vector $\mathbf{e}_{\mathrm{L}}$ that satisfies

$$\mathbf{e}_{\mathrm{L}}^{\mathrm{T}} \mathbf{A} = \lambda \mathbf{e}_{\mathrm{L}}^{\mathrm{T}}.$$

</div>

Key properties:

- If a matrix has two or more linearly independent right-eigenvectors with the same eigenvalue then that eigenvalue is called a **degenerate eigenvalue** of the matrix, or a repeated eigenvalue. Any linear combination of those eigenvectors is another right-eigenvector with the same eigenvalue.
- The **principal right-eigenvector** of a matrix is, by definition, the right-eigenvector with the largest associated eigenvalue.
- If a real matrix has a right-eigenvector with complex eigenvalue $\lambda = x + yi$ then it also has a right-eigenvector with the conjugate eigenvalue $\lambda^* = x - yi$.

#### Symmetric Matrices

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Properties of Symmetric Matrices)</span></p>

If $\mathbf{A}$ is a real symmetric $N \times N$ matrix then:

1. All the eigenvalues and eigenvectors of $\mathbf{A}$ are real.
2. Every left-eigenvector of $\mathbf{A}$ is also a right-eigenvector of $\mathbf{A}$ with the same eigenvalue, and *vice versa*.
3. A set of $N$ eigenvectors and eigenvalues $\lbrace \mathbf{e}^{(a)}, \lambda_a \rbrace_{a=1}^N$ can be found that are orthonormal, that is, $\mathbf{e}^{(a)} \cdot \mathbf{e}^{(b)} = \delta_{ab}$; the matrix can be expressed as a weighted sum of outer products of the eigenvectors:

$$\mathbf{A} = \sum_{a=1}^{N} \lambda_a [\mathbf{e}^{(a)}][\mathbf{e}^{(a)}]^{\mathrm{T}}.$$

</div>

#### General Square Matrices

An $N \times N$ matrix can have up to $N$ distinct eigenvalues. Generically, there are $N$ eigenvalues, all distinct, and each has one left-eigenvector and one right-eigenvector. In cases where two or more eigenvalues coincide, for each distinct eigenvalue that is non-zero there is at least one left-eigenvector and one right-eigenvector.

Left- and right-eigenvectors that have different eigenvalue are orthogonal, that is:

$$\text{if } \lambda_a \neq \lambda_b \text{ then } \mathbf{e}_{\mathrm{L}}^{(a)} \cdot \mathbf{e}_{\mathrm{R}}^{(b)} = 0.$$

#### Non-Negative Matrices

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Non-Negative Matrix)</span></p>

If all the elements of a non-zero matrix $\mathbf{C}$ satisfy $C_{mn} \ge 0$ then $\mathbf{C}$ is a **non-negative matrix**. Similarly, if all the elements of a non-zero vector $\mathbf{c}$ satisfy $c_n \ge 0$ then $\mathbf{c}$ is a non-negative vector.

</div>

A non-negative matrix has a principal eigenvector that is non-negative. If the principal eigenvalue of a non-negative matrix is not degenerate, then the matrix has only one principal eigenvector $\mathbf{e}^{(1)}$, and it is non-negative. Generically, all the other eigenvalues are smaller in absolute magnitude.

#### Transition Probability Matrices

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Transition Probability Matrix)</span></p>

A **transition probability matrix** $\mathbf{Q}$ has columns that are probability vectors, that is, it satisfies $\mathbf{Q} \ge 0$ and

$$\sum_i Q_{ij} = 1 \quad \text{for all } j.$$

</div>

This property can be rewritten in terms of the all-ones vector $\mathbf{n} = (1, 1, \ldots, 1)^{\mathrm{T}}$:

$$\mathbf{n}^{\mathrm{T}} \mathbf{Q} = \mathbf{n}^{\mathrm{T}}.$$

So $\mathbf{n}$ is the principal left-eigenvector of $\mathbf{Q}$ with eigenvalue $\lambda_1 = 1$: $\mathbf{e}_{\mathrm{L}}^{(1)} = \mathbf{n}$.

Because it is a non-negative matrix, $\mathbf{Q}$ has a principal right-eigenvector that is non-negative, $\mathbf{e}_{\mathrm{R}}^{(1)}$. Generically, for Markov processes that are ergodic, this eigenvector is the only right-eigenvector with eigenvalue of magnitude 1. This vector, if we normalize it such that $\mathbf{e}_{\mathrm{R}}^{(1)} \cdot \mathbf{n} = 1$, is called the **invariant distribution** of the transition probability matrix. It is the probability density that is left unchanged under $\mathbf{Q}$.

The matrix may have up to $N - 1$ other right-eigenvectors all of which are orthogonal to the left-eigenvector $\mathbf{n}$, that is, they are zero-sum vectors.

### C.3 Perturbation Theory

Perturbation theory is useful for finding the eigenvectors and eigenvalues of square, *not necessarily symmetric*, matrices. We assume that we have an $N \times N$ matrix $\mathbf{H}$ that is a function $\mathbf{H}(\epsilon)$ of a real parameter $\epsilon$, with $\epsilon = 0$ being our starting point. We assume that a Taylor expansion of $\mathbf{H}(\epsilon)$ is appropriate:

$$\mathbf{H}(\epsilon) = \mathbf{H}(0) + \epsilon \mathbf{V} + \cdots$$

where $\mathbf{V} \equiv \frac{\partial \mathbf{H}}{\partial \epsilon}$.

We write the eigenvectors and eigenvalues as:

$$\mathbf{H}(\epsilon)\mathbf{e}_{\mathrm{R}}^{(a)}(\epsilon) = \lambda^{(a)}(\epsilon)\mathbf{e}_{\mathrm{R}}^{(a)}(\epsilon),$$

and Taylor-expand:

$$\lambda^{(a)}(\epsilon) = \lambda^{(a)}(0) + \epsilon \mu^{(a)} + \cdots, \qquad \mathbf{e}_{\mathrm{R}}^{(a)}(\epsilon) = \mathbf{e}_{\mathrm{R}}^{(a)}(0) + \epsilon \mathbf{f}_{\mathrm{R}}^{(a)} + \cdots$$

with $\mu^{(a)} \equiv \frac{\partial \lambda^{(a)}(\epsilon)}{\partial \epsilon}$ and $\mathbf{f}_{\mathrm{R}}^{(a)} \equiv \frac{\partial \mathbf{e}_{\mathrm{R}}^{(a)}}{\partial \epsilon}$.

We constrain the inner products with: $\mathbf{e}_{\mathrm{L}}^{(a)}(\epsilon) \mathbf{e}_{\mathrm{R}}^{(a)}(\epsilon) = 1$ for all $a$.

#### First-Order Perturbation Theory

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(First-Order Perturbation Theory)</span></p>

Expanding the defining equation to first order in $\epsilon$ and identifying the terms of order $\epsilon$, we obtain:

**First-order eigenvalue correction:**

$$\mu^{(a)} = \mathbf{e}_{\mathrm{L}}^{(a)}(0) \mathbf{V} \mathbf{e}_{\mathrm{R}}^{(a)}(0).$$

**First-order eigenvector correction:**

$$\mathbf{f}_{\mathrm{R}}^{(a)} = \sum_{b \neq a} \frac{\mathbf{e}_{\mathrm{L}}^{(b)}(0) \mathbf{V} \mathbf{e}_{\mathrm{R}}^{(a)}(0)}{\lambda^{(a)}(0) - \lambda^{(b)}(0)} \mathbf{e}_{\mathrm{R}}^{(b)}(0).$$

</div>

#### Second-Order Perturbation Theory

If we expand the eigenvector equation to second order in $\epsilon$, and assume that $\mathbf{H}(\epsilon) = \mathbf{H}(0) + \epsilon \mathbf{V}$ is exact (i.e., $\mathbf{H}$ is a purely linear function of $\epsilon$), we obtain the second derivative of the eigenvalue:

$$\frac{1}{2}\nu^{(a)} = \mathbf{e}_{\mathrm{L}}^{(a)}(0) \mathbf{V} \mathbf{f}_{\mathrm{R}}^{(a)},$$

so the second derivative of the eigenvalue with respect to $\epsilon$ is given by

$$\frac{1}{2}\nu^{(a)} = \sum_{b \neq a} \frac{[\mathbf{e}_{\mathrm{L}}^{(b)}(0) \mathbf{V} \mathbf{e}_{\mathrm{R}}^{(a)}(0)][\mathbf{e}_{\mathrm{L}}^{(a)}(0) \mathbf{V} \mathbf{e}_{\mathrm{R}}^{(b)}(0)]}{\lambda^{(a)}(0) - \lambda^{(b)}(0)}.$$

#### Summary

If we introduce the abbreviation $V_{ba}$ for $\mathbf{e}_{\mathrm{L}}^{(b)}(0) \mathbf{V} \mathbf{e}_{\mathrm{R}}^{(a)}(0)$, we can write the eigenvectors of $\mathbf{H}(\epsilon) = \mathbf{H}(0) + \epsilon \mathbf{V}$ to first order as

$$\mathbf{e}_{\mathrm{R}}^{(a)}(\epsilon) = \mathbf{e}_{\mathrm{R}}^{(a)}(0) + \epsilon \sum_{b \neq a} \frac{V_{ba}}{\lambda^{(a)}(0) - \lambda^{(b)}(0)} \mathbf{e}_{\mathrm{R}}^{(b)}(0) + \cdots$$

and the eigenvalues to second order as

$$\lambda^{(a)}(\epsilon) = \lambda^{(a)}(0) + \epsilon V_{aa} + \epsilon^2 \sum_{b \neq a} \frac{V_{ba} V_{ab}}{\lambda^{(a)}(0) - \lambda^{(b)}(0)} + \cdots$$
