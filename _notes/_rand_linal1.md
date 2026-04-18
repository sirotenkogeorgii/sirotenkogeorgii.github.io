## 5.3 Trace Estimation

Many scientific computing and machine learning applications require estimating the trace of a square linear operator $\boldsymbol{A}$ that is represented implicitly. Randomized methods are especially effective for such problems.

### 5.3.1 Trace Estimation by Sampling

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Girard-Hutchinson Estimator)</span></p>

Let $\boldsymbol{A}$ be $n \times n$ and $\lbrace \boldsymbol{e}_1, \ldots, \boldsymbol{e}_n \rbrace$ be the standard basis vectors in $\mathbb{R}^n$. One can compute $\operatorname{tr}(\boldsymbol{A})$ exactly with $n$ matrix-vector products:

$$\operatorname{tr}(\boldsymbol{A}) = \sum_{i=1}^n \boldsymbol{e}_i^*\boldsymbol{A}\boldsymbol{e}_i.$$

Randomization allows estimating this quantity using $m \ll n$ matrix-vector multiplications. If $\boldsymbol{\omega} \sim \mathcal{D}$ is a random vector satisfying $\mathbb{E}[\boldsymbol{\omega}\boldsymbol{\omega}^\ast] = \boldsymbol{I}\_n$, then 

$$\operatorname{tr}(\boldsymbol{A}) = \mathbb{E}[\boldsymbol{\omega}^\ast\boldsymbol{A}\boldsymbol{\omega}]$$

Drawing $m$ independent vectors $\boldsymbol{\omega}_i \sim \mathcal{D}$, the **Girard-Hutchinson estimator** is

$$\operatorname{tr}(\boldsymbol{A}) \approx \frac{1}{m}\sum_{i=1}^m \boldsymbol{\omega}_i^*\boldsymbol{A}\boldsymbol{\omega}_i.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof 1 (trace of scalar)</summary>

We use two wonderful tricks that for the scalar $x$ or equivalently a $1\times 1$ matrix $x$ holds $\text{tr}(x) = x$. Another trick is a cycle property of the trace: for the matrices $A,B$ and $C$ the following holds: $\text{tr}(ABC)=\text{tr}(BCA)=\text{tr}(CAB)$.

$$\mathbb{E}[w^\top A w] = $$

$$\mathbb{E}[\text{tr}(w^\top A w)] = $$

$$\mathbb{E}[\text{tr}(A w w^\top )] = $$

$$\text{tr}(\mathbb{E}[A w w^\top]) = $$

$$\text{tr}(A\mathbb{E}[w w^\top]) = $$

$$\text{tr}(AI) = $$

$$\text{tr}(A).$$

We used:

$$(i) \quad \text{tr}(w^\top A w) = \text{tr}(A w w^\top)$$

$$(ii) \quad \mathbb{E}[\text{tr}(A)] = \text{tr}(\mathbb{E}[A])$$

</details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof 2 (element-wise)</summary>

$$\mathbb{E}[w^\top A w] = $$

$$\mathbb{E}[\sum_i w_i \sum_j A_{ij} w_j] =$$

$$\mathbb{E}[\sum_i \sum_j A_{ij} w_i w_j] =$$

$$\sum_i \sum_j \mathbb{E}[A_{ij} w_i w_j] =$$

$$\sum_i \sum_j A_{ij} \mathbb{E}[w_i w_j] =$$

$$\sum_i \sum_j A_{ij} \mathbb{E}[w w^\top]_{ij} =$$

$$\sum_i \sum_j A_{ij} I_{ij} =$$

$$\text{tr}(A).$$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Isotropic random vectors)</span></p>

The set of vectors $D is said to be in **isotropic position**, if for every vector $w \in D$ holds $\mathbb{E}[w w^\top] = I$.

We say the random vectors $w \in D$ are **isotropic**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Choice of Distribution for Trace Estimation)</span></p>

The idea for this method goes back to 1987 with work by Girard [Gir87], who proposed that $\mathcal{D}$ be the uniform distribution over the $\ell_2$ hypersphere with radius $\sqrt{n}$. Shortly thereafter, Hutchinson proposed taking $\mathcal{D}$ as a distribution over Rademacher random vectors [Hut90].

- **Hutchinson's choice** of $\mathcal{D}$ minimizes the variance of the estimator when $\boldsymbol{A}$ is fixed.
- **Girard's choice** minimizes the worst-case variance over sets of matrices that are closed under conjugation by unitary matrices: $A\in S \implies UAU^\ast \in S$.

Such estimators require $m \in \Omega(1/\epsilon^2)$ samples to approximate $\operatorname{tr}(\boldsymbol{A})$ to within $\epsilon$ error for some constant failure probability.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Girard-Hutchinson estimator is unbiased)</span></p>

Provided the random vectors are **isotropic**: $\mathbb{E}[w_i w_i^\top] = I$

$$\mathbb{E}\Bigl[ \frac{1}{m}\sum_{i=1}^m \boldsymbol{\omega}_i^\top\boldsymbol{A}\boldsymbol{\omega}_i \Bigr] = \frac{1}{m}\sum_{i=1}^m \mathbb{E}\Bigl[ \boldsymbol{\omega}_i^\top\boldsymbol{A}\boldsymbol{\omega}_i \Bigr] = \frac{1}{m}\sum_{i=1}^m \cdot m \cdot \text{tr}(A) = \text{tr}(A)$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Hutchinson vs. Hutch++ variance of estimation)</span></p>

Continue to assume that the $\omega_i$‘s are isotropic and now assume that $\omega_1,\ldots,\omega_m$ are iid. By independence, the variance can be written as

$$\text{Var}(\hat{\text{tr}}) = \frac{1}{m^2}\sum_{i=1}^m\text{Var}(w^{\ast}_i A w_i) = \frac{1}{m}\text{Var}(w^{\ast} A w)$$

The variance decreases like $1/m$, which is characteristic of Monte Carlo-type algorithms. Since $\hat{\tr}$ is unbiased, this means that the mean square error decays like $1/m$ so the average error (more precisely root-mean-square error) decays like

$$\sqrt{\mathbb{E}[(\hat{\text{tr}} - \text{tr})^2]} \approx \sqrt{\frac{\text{const}}{m}} =  \frac{\text{const}}{\sqrt{m}}$$

**This type of convergence is very slow.** If I want to decrease the error by a factor of $10$, I must do $100\times$ the work!

Variance-reduced trace estimators like Hutch++ and new trace estimator XTrace improve the rate of convergence substantially. Even in the worst case, Hutch++ and XTrace reduce the variance at a rate $1/m^2$ and (root-mean-square) error at rates $1/m$.

</div>

### 5.3.2 Trace Estimation with Help from Low-Rank Approximation

#### Compress and Trace

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Compress-and-Trace Method)</span></p>

In [SAI17], Saibaba, Alexanderian, and Ipsen propose two randomized algorithms for estimating the trace of a psd linear operator $\boldsymbol{A}$. When $\boldsymbol{A}$ is **accessible by matrix-vector products**, the proposed method begins with a rangefinder step to find a column-orthonormal $n \times m$ matrix $\boldsymbol{Q}$ where $\boldsymbol{Q}\boldsymbol{Q}^\ast\boldsymbol{A}\boldsymbol{Q}\boldsymbol{Q}^\ast \approx \boldsymbol{A}$. The method then approximates

$$\operatorname{tr}(\boldsymbol{A}) \approx \operatorname{tr}(\boldsymbol{Q}\boldsymbol{Q}^*\boldsymbol{A}\boldsymbol{Q}\boldsymbol{Q}^*) = \operatorname{tr}(\boldsymbol{Q}^*\boldsymbol{A}\boldsymbol{Q}).$$

This can provide better relative error bounds than a Girard-Hutchinson estimator if $\boldsymbol{A}$'s spectral decay is sufficiently fast and $\boldsymbol{Q}$ is obtained by power iteration.

Trace estimation is especially challenging when **matrix-vector products with $\boldsymbol{A}$ are expensive**. In practice, $\boldsymbol{A}$ is often the image of another matrix $\boldsymbol{B}$ under a matrix function. For the case where $\boldsymbol{A} = \log(\boldsymbol{I} + \boldsymbol{B})$ for a psd matrix $\boldsymbol{B}$, one finds $\boldsymbol{Q}$ so that $\boldsymbol{Q}\boldsymbol{Q}^\ast\boldsymbol{B}\boldsymbol{Q}\boldsymbol{Q}^\ast$ is a good low-rank approximation of $\boldsymbol{B}$, and then

$$\operatorname{tr}(\boldsymbol{A}) \approx \sum_{i=1}^m \log\!\left(1 + \lambda_i(\boldsymbol{Q}^*\boldsymbol{B}\boldsymbol{Q})\right),$$

where $\lambda_i(\cdot)$ return the $i$-th largest eigenvalue of the given matrix.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Trace of log PSD matrix)</span></p>

Let M be PSD, then

$$\text{tr}(\log M) \log(\det M)$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

**Expand RHS**

$$\log(\det M) = \log(\det(Q\Lambda Q^\ast)) = \log(\det Q) + \log(\det \Lambda) + \log(\det Q^\ast)$$

$$= \log(\det \Lambda) = \log(\prod \lambda_i) = \sum \log \lambda_i = \text{tr}(\log \Lambda)$$

**Expand LHS**

$$\log M = \log(Q\Lambda Q^\ast) = Q\log(\Lambda) Q^\ast$$

$$\implies \text{tr}(\log M) = \text{tr}(Q\log(\Lambda) Q^\ast) = \text{tr}(\log(\Lambda) Q^\ast Q) = \text{tr}(\log(\Lambda))$$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Matrix Compression)</span></p>

We build a tall orthonormal matrix $Q\in\mathbb{R}^{n\times m}$, with $m\ll n$, whose columns approximately span the dominant eigenspace of $A$ or $B$.

Then:

$$QQ^*AQ Q^*$$

is a rank-$m$ approximation of $A$, and

$$Q^*AQ$$

is the small $m\times m$ compressed matrix.

Because trace is cyclic,

$$\operatorname{tr}(QQ^*AQ Q^*)=\operatorname{tr}(Q^*AQ)$$

So instead of working with a huge matrix, we work with the much smaller matrix $Q^*AQ$.

</div>

#### Split, Trace, and Approximate

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Hutch++)</span></p>

In [MMM+21], Meyer et al. combined ideas from low-rank approximation with the Girard-Hutchinson estimator to obtain **Hutch++**. The algorithm proceeds as follows:

1. Sample a matrix $\boldsymbol{Q}$ uniformly at random from the set of $n \times m$ column-orthonormal matrices.
2. Define the low-rank approximation $\hat{\boldsymbol{A}} = \boldsymbol{Q}\boldsymbol{Q}^\ast\boldsymbol{A}\boldsymbol{Q}\boldsymbol{Q}^\ast$ and compute 
   
  $$\operatorname{tr}(\hat{\boldsymbol{A}}) = \operatorname{tr}(\boldsymbol{Q}^*\boldsymbol{A}\boldsymbol{Q})$$

3. Apply Girard-Hutchinson to the **deflated matrix** 
   
  $$\boldsymbol{\Delta} = (\boldsymbol{I} - \boldsymbol{Q}\boldsymbol{Q}^*)\boldsymbol{A}(\boldsymbol{I} - \boldsymbol{Q}\boldsymbol{Q}^*)$$

1. $\operatorname{tr}(\boldsymbol{A}) = \operatorname{tr}(\hat{\boldsymbol{A}}) + \operatorname{tr}(\boldsymbol{\Delta})$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Intuition)</span></p>

$$\Delta = (I-QQ^*)A(I-QQ^*)$$

is **not usually the matrix error itself**, but it is the **deflated residual part of $A$**, and for trace purposes it behaves like the error term.

Think of the total trace as split into two pieces:

1. the contribution captured by the low-rank subspace:
   
   $$\operatorname{tr}(\hat A)=\operatorname{tr}(Q^*AQ)$$
   
2. the leftover contribution outside that subspace:
   
   $$\operatorname{tr}(\Delta)$$

So $\Delta$ is the “what remains after removing the important directions” part.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>More explanation and derivation</summary>

#### Step 1: what is $QQ^*$?

If $Q\in\mathbb{R}^{n\times m}$ has orthonormal columns, then

$$P := QQ^*$$

is the orthogonal projector onto the subspace spanned by the columns of $Q$.

Then

$$I-P = I-QQ^*$$

is the projector onto the orthogonal complement.

So:

* $P$ keeps the part in the $Q$-subspace,
* $I-P$ keeps the part orthogonal to that subspace.

#### Step 2: what is $\hat A$?

They define the low-rank approximation

$$\hat A = QQ^*AQQ^* = PAP$$

This means:

* project into the $Q$-subspace,
* apply $A$,
* project back into the same subspace.

So $\hat A$ is the part of $A$ seen **inside** the important low-dimensional subspace.

#### Step 3: what is $\Delta$?

They define

$$\Delta = (I-QQ^*)A(I-QQ^*) = (I-P)A(I-P)$$

This means:

* keep only the component orthogonal to $Q$,
* apply $A$,
* then again discard anything that leaks back into the $Q$-subspace.

So $\Delta$ is the part of $A$ that lives **entirely outside** the span of $Q$.

That is why it is called the **deflated matrix**: the dominant subspace captured by $Q$ has been removed.

#### Is it the error term? Not exactly as a matrix

The actual matrix difference is

$$A-\hat A = A-PAP$$

But

$$\Delta = (I-P)A(I-P)=A-PA-AP+PAP$$

These are generally **not equal**.

So in general,

$$\Delta \neq A-\hat A$$

#### Then why do they use $\Delta$?

Because although the matrices are different, their **traces are the same**:

$$\operatorname{tr}(A-\hat A)=\operatorname{tr}(\Delta)$$

That is the key fact.

Since Hutch++ only wants the **trace**, it is enough to estimate $\operatorname{tr}(\Delta)$ instead of $\operatorname{tr}(A-\hat A)$.

So the right interpretation is:

> $\Delta$ is not the exact matrix error, but it is a **trace-equivalent residual**.

#### Why do the traces match?

Let $P=QQ^\ast$. Expand:

$$\Delta=(I-P)A(I-P)=A-PA-AP+PAP$$

Taking trace,

$$\operatorname{tr}(\Delta) = \operatorname{tr}(A)-\operatorname{tr}(PA)-\operatorname{tr}(AP)+\operatorname{tr}(PAP).$$

Now use cyclicity of trace:

$$
\operatorname{tr}(PA)=\operatorname{tr}(AP),
\qquad
\operatorname{tr}(PAP)=\operatorname{tr}(AP).
$$

So

$$\operatorname{tr}(\Delta) = \operatorname{tr}(A)-\operatorname{tr}(AP)$$

Also,

$$\operatorname{tr}(\hat A)=\operatorname{tr}(PAP)=\operatorname{tr}(AP)$$

Therefore

$$\operatorname{tr}(A)-\operatorname{tr}(\hat A) = \operatorname{tr}(\Delta)$$

That is exactly why the excerpt says

$$\operatorname{tr}(A)=\operatorname{tr}(\hat A)+\operatorname{tr}(\Delta)$$

#### Why is this useful?

Because if $Q$ captures the dominant eigendirections of $A$, then $\hat A$ already explains most of the trace, and $\Delta$ is much smaller / less variable.

That makes randomized trace estimation easier.

This is the whole Hutch++ idea:

* estimate the big important part deterministically through $Q^*AQ$,
* only use Hutchinson on the smaller leftover $\Delta$.

That reduces variance a lot.

#### Best-case picture

Suppose $A$ is PSD and $Q$ spans the top eigenvectors exactly.

Then $\hat A$ contains the large eigenvalues, and $\Delta$ contains only the tail eigenvalues.

So $\Delta$ really is the “remaining spectrum.”

In that ideal case it is very natural to think of it as the residual error after low-rank approximation.

#### Bottom line

$$\Delta=(I-QQ^*)A(I-QQ^*)$$

is the **deflated residual matrix**.

* It is **not generally equal** to $A-\hat A$ as a matrix.
* But it **has the same trace** as $A-\hat A$.
* So for trace estimation, it plays the role of the **leftover error term**.

A good mental model is:

> $\hat A$ = part of $A$ captured by the chosen low-rank subspace
> $\Delta$ = leftover trace contribution after that subspace is removed

I can also show this with a small $2\times 2$ or $3\times 3$ example if you want.

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Hutch++ vs. Hutch)</span></p>

As a splitting and deflation approach, Hutch++ method is very eﬀective in reducing the variance of the Girard–Hutchinson estimator. 

For PSD matrices, Hutch++ can (with some small fixed failure probability) compute $\operatorname{tr}(\boldsymbol{A})$ to within $\epsilon$ relative error using only $O(1/\epsilon)$ matrix-vector products — a substantial improvement over the $O(1/\epsilon^2)$ products required by plain Girard-Hutchinson estimators. In fact, this sample complexity cannot be improved when considering a large class of algorithms [MMM+21, Theorems 4.1 and 4.2].

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Hutch++ Extensions)</span></p>

- Persson, Cortinovis, and Kressner extended Hutch++ so that it can proceed adaptively, only terminating once some error tolerance has been achieved (up to a *controllable* failure probability) [PCK21].
- The modified Hutch++ method notably accommodates **symmetric indefinite matrices** $\boldsymbol{A}$. The accuracy guarantees for indefinite matrices cannot be as strong as those for positive definite matrices — relative error guarantees are essentially impossible when $\operatorname{tr}(\boldsymbol{A}) = 0$. Persson et al. therefore provide *additive* error guarantees in this setting.

</div>
