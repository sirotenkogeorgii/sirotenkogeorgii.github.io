**Table of Contents**
- TOC
{:toc}

## Abstract

Solovay reducibility is studied intensively as a tool to compare the approximability and the degree of randomness of left-c.e. reals. By definition, a real is left-c.e. if it has a left-c.e. approximation, that is, it is the limit of an effective nondecreasing sequence of rationals. If reals $\alpha$ and $\beta$ have left-c.e. approximations $a_0,a_1,\ldots$ and $b_0,b_1,\ldots$, respectively, such that the approximation ratios

$$\frac{\alpha-a_n}{\beta-b_n}$$

are bounded from above by a constant, the real $\alpha$ is Solovay reducible to $\beta$. The latter is the case for any such $\alpha$ and $\beta$ and their left-c.e. approximations whenever $\beta$ is Martin-Löf random by the Kučera-Slaman Theorem. This result was substantially strengthened by Barmpalias and Lewis-Pye, who demonstrated that, under the given assumptions, the approximation ratios are not only bounded but actually converge to a limit, which does not depend on the considered left-c.e. approximations. Outside the realm of left-c.e. reals, Solovay reducibility is viewed as badly behaved [5, Section 9.2], and is thus rarely used. Accordingly, there is a quest for a suitable extension of Solovay reducibility to the class of all reals: one that coincides with Solovay reducibility on the left-c.e. reals but is better behaved when applied to reals in general. Promising candidates include S2a-reducibility on the set of computably approximable reals by Zheng and Rettinger and monotone Solovay reducibility by Titov. For the latter, Titov demonstrated that the theorems of Kučera and Slaman and of Barmpalias and Lewis-Pye extend to all reals. He further conjectured [14, Conjecture 3.2] that similar extensions hold for S2a-reducibility in terms of its functional characterization by Kumabe, Miyabe, and Suzuki.

In this work, we refute this conjecture by proving that the analogue of the Barmpalias-Lewis-Pye Limit Theorem does not hold for S2a-reducibility.

## Introduction and background

* The main objects of interest of computable analysis are computable real numbers and computable real functions, 
  * i.e., real numbers and real-valued functions of a real argument that can be computed (in an appropriate way) by a Turing functional. 
  
#TODO: what is a turing functional? how does turing functional differ from turing machine

### Representation of reals and computable approximations

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name"></span></p>

* A **computable approximation** is a computable Cauchy sequence of rationals $a_0,a_1,\ldots$. 
* A **left-c.e.** and a **right-c.e.** approximation is a strictly increasing and a strictly decreasing computable approximation, respectively. 
* A **d.c.e.** approximation is an approximation of the form

  $$a_0-b_0,a_1-b_1,\ldots,$$

  where $a_0,a_1,\ldots$ and $b_0,b_1,\ldots$ are left-c.e. approximations. A Cauchy name is a sequence of rationals $a_0,a_1,\ldots$ such that

  $$|a_m-a_n|\le 2^{-n}\qquad\text{for all }m\ge n.$$

* **Computably approximable (or c.a.)** reals are limit points of computable approximations. 
* Left-c.e., right-c.e., and d.c.e. reals are limit points of left-c.e., right-c.e., and d.c.e. approximations, respectively. 
* **Computable reals** are limit points of computable Cauchy names.

</div>

### Computable functions on rationals and on reals

* In theoretical computer science, computability notions differ for functions $\mathbb Q\to\mathbb Q$ and $\mathbb R\to\mathbb R$ since **real numbers in general cannot be encoded finitely**.
* In the paper, by computability of a real function we mean the existence of a Turing functional
  * given any Cauchy name of $x$, returns a Cauchy name of $f(x)$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name"></span></p>

* A function $f:\mathbb R\to\mathbb R$ is computable (as a real function) on a set $A$ if there exists a Turing functional with one oracle tape $M$
  * for every Cauchy name $(q_0,q_1,\ldots)$ of a real $x\in A$ given as oracle, $M$ returns a Cauchy name of $f(x)$. 
* In the latter case, we write

  $$f_n^{(q_0,q_1,\ldots)}$$

  for the $n$th element (if defined) on the output tape of $M$ with the oracle $(q_0,q_1,\ldots)$ and call it the evaluation of $f(x)$ with precision $n$.

</div>

### Differentiability and algorithmic randomness

* (Lebesgue, 1904): functions of **bounded variation** on $\mathbb R$ (Lebesgue, 1904) and **Lipschitz continuous functions** on $\mathbb R^n$ (Rademacher, 1919) are differentiable almost everywhere.
* (Demuth, 1975): in the first **effective version** of these analytic results was showed that every computable function of bounded variation is **differentiable at all Martin-Löf random points**.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name"></span></p>

A function $f$ is of bounded variation if there exists a constant $C$ such that, for every finite ordered subset

$$x_0<x_1<\cdots<x_n$$

of $\operatorname{dom}(f)$,

$$\sum_{i=1}^n |f(x_i)-f(x_{i-1})|\le C.$$


</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Can the definition use countable ordered subsets instead?</summary>

Yes, if “countable” includes finite sets and a countably infinite ordered subset is given by an increasing sequence

$$x_0<x_1<x_2<\cdots.$$

Then one may require

$$\sum_{i=1}^{\infty}|f(x_i)-f(x_{i-1})|\le C.$$

This is equivalent to the finite-subset definition. If the finite condition holds, every partial sum is at most $C$, so the infinite sum is at most $C$. Conversely, every finite ordered subset is itself a finite sequence, so the countable-sequence condition includes the original condition.

For an arbitrary countable linearly ordered subset, consecutive elements need not exist. The standard formulation using all finite ordered subsets avoids this issue; equivalently, the variation is the supremum of the corresponding finite sums.

</details>
</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name"></span></p>

A function $f$ is Lipschitz continuous if it satisfies

$$|f(x)-f(y)|\le L|x-y|$$

for some constant $L$ and all $x,y\in\operatorname{dom}(f)$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name"></span></p>

Let $I$ be a compact interval.

* Every **Lipschitz continuous function** $f$ defined on $I$ is of bounded variation.
* Every **nondecreasing function** defined on $I$ is of bounded variation.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name"></span></p>

* A **Martin-Löf test** is a uniformly computable sequence of open sets, or layers, $(L_n)\_{n\in\mathbb N}$ such that for the Lebesgue measure $\mu$,

  $$\mu(L_n)\le 2^{-n}\qquad\text{for all }n\in\mathbb N.$$

  A real $\alpha$ fails a Martin-Löf test $(L_n)\_{n\in\mathbb N}$ if

  $$\alpha\in\bigcap_{n\in\mathbb N}L_n.$$

* A real is **Martin-Löf nonrandom** if it fails some Martin-Löf test, and Martin-Löf random otherwise.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Uniformly</summary>

**“Uniformly” means that one algorithm handles all layers $L_n$, with $n$ supplied as input.**

For open subsets of $\mathbb R$, an effectively open set is normally represented as a computably enumerable union of rational open intervals:

$$L_n=\bigcup_{k} (a_{n,k},b_{n,k}), \qquad a_{n,k},b_{n,k}\in\mathbb Q.$$

Uniformity means there is a single algorithm which, given $n$, enumerates the intervals forming $L_n$. Equivalently, the set

$$\lbrace (n,a,b): (a,b)\text{ is enumerated into }L_n\rbrace$$

is computably enumerable.

So it is not enough that, for every $n$, some separate algorithm enumerates $L_n$. The algorithms must be obtainable effectively and uniformly from $n$.

The paper defines a Martin-Löf test as such a uniform sequence $(L_n)$, together with the measure condition

$$\mu(L_n)\le 2^{-n}.$$

</details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Are the open sets represented by cylidners?</summary>

That depends on the underlying space.

* In **Cantor space** $2^{\mathbb N}$, yes: effectively open sets are c.e. unions of cylinders
  
  $$[\sigma]=\lbrace \in2^{\mathbb N}:\sigma\prec X\rbrace,$$
  
  where $\sigma$ is a finite binary string.

* In **$\mathbb R$**, as in this paper, they are usually represented by c.e. unions of rational open intervals.

These are closely related: after encoding a real by its binary expansion, rational dyadic intervals play essentially the role of cylinders. But strictly speaking, the paper’s tests are open subsets of $\mathbb R$, so the natural basic objects are rational intervals, not cylinders.

</details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Is enumeration of ML test layers done via diagonalization, meaning that we do not enumerate the test layer by layer sequentially since the layer could be a union of infinite number of intervals?</summary>

Yes. **To enumerate all layers simultaneously**, one typically uses **dovetailing** (sometimes informally called diagonalization), because each layer may require infinitely many enumeration steps.

Suppose

$$L_n=\bigcup_k I_{n,k},$$

where each $I_{n,k}$ is a rational open interval. Uniformity means there is one computable procedure producing the pairs

$$(n,I_{n,k}).$$

Equivalently, the collection

$$\lbrace (n,k): I_{n,k}\text{ is enumerated into }L_n\rbrace$$

is computably enumerable.

**How the enumeration runs.** At global stage $s$, run the enumerators for layers

$$L_0,L_1,\dots,L_s$$

for $s$ computation steps each. Thus every layer receives arbitrarily much computation time:

$$
\begin{array}{c|cccc}
\text{stage }0 & L_0\\
\text{stage }1 & L_0 & L_1\\
\text{stage }2 & L_0 & L_1 & L_2\\
\vdots
\end{array}
$$

So we do **not** finish enumerating $L_0$, then start $L_1$: $L_0$ may never finish. Instead, the enumerations are interleaved.

But this dovetailing is an implementation technique, not an extra condition in the definition. Formally, “uniformly effectively open” simply means that one algorithm, given $n$, enumerates the intervals forming $L_n$. The paper then adds the measure bound $\mu(L_n)\le 2^{-n}$. 

Also, intervals can be enumerated in any order and with repetitions; what matters is that their union equals $L_n$.

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Martin-Löf)</span></p>

There exists a Martin-Löf test failed by all Martin-Löf nonrandom reals.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Demuth)</span></p>

Every computable function of bounded variation $f$ on the reals is differentiable at every Martin-Löf random point.

</div>

#TODO: are the function at ML random points somehow special?

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Titov extended Demuth)</span></p>

* The result of Demuth was extended by Titov in 2025 to a wider class of functions. 
* Titov proved that, if a real $\beta$ is Martin-Löf random, then any function of finite variation that is computable at every argument $x<\beta$ (but not necessarily at $\beta$) has a left derivative at $\beta$. 
* This result is remarkable from the viewpoint of relative randomness, since it can be applied to some classes of translation functions in terms of Solovay reducibility, which will be explained in the next section.

</div>

#TODO: did not get the last point of the remark

## Solovay reducibility and its versions

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Translation function)</span></p>

A **translation function** from a real $\beta$ to a real $\alpha$ is a computable function

$$f:\subseteq\mathbb Q\to\mathbb Q$$

that is defined on the left cut of $\beta$ and fulfils

$$\lim_{q\nearrow\beta} f(q)=\alpha.$$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Solovay reducibility)</span></p>

A real $\alpha$ is **Solovay reducible** to a real $\beta$, written $\alpha\le_S\beta$, if there exist a constant $c$ and a translation function $f$ from $\beta$ to $\alpha$ that satisfies

$$\alpha-f(q)<c(\beta-q)\qquad\text{for all }q<\beta.$$

</div>

### Solovay reducibility on left-c.e. reals

On the set of left-c.e. reals, Solovay reducibility was characterized by Calude, Hertling, Khoussainov, and Wang in 1998 as a measure of convergence speed of left-c.e. approximations.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Calude et al.)</span></p>

A left-c.e. real $\alpha$ is Solovay reducible to a left-c.e. real $\beta$ iff there exist two left-c.e. approximations

$$a_0,a_1,\ldots\nearrow\alpha,\qquadb_0,b_1,\ldots\nearrow\beta,$$

and a constant $c$ such that

$$\alpha-a_n<c(\beta-b_n)\qquad\text{for all }n. \tag{1}$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

* In 2001, Kučera and Slaman showed that, on the set of left-c.e. reals, Martin-Löf random reals form the highest Solovay degree. 
* In 2017, Barmpalias and Lewis-Pye strengthened the Kučera-Slaman theorem by proving that, for any two left-c.e. approximations $a_0,a_1,\ldots$ and $b_0,b_1,\ldots$, the ratio $\frac{\alpha-a_n}{\beta-b_n}$ is not only bounded but also converges to a real number called by Miller the derivative of $\alpha$ relative to $\beta$, which does not depend on the choice of the two left-c.e. approximations.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Barmpalias, Lewis-Pye)</span></p>

Let $\alpha$ be a left-c.e. real and $\beta$ be a Martin-Löf random left-c.e. real. Then there exists a real $d$ such that, for any two left-c.e. approximations

$$a_0,a_1,\ldots\nearrow\alpha,\qquadb_0,b_1,\ldots\nearrow\beta,$$

$$d=\lim_{n\to\infty}\frac{\alpha-a_n}{\beta-b_n}. \tag{2}$$

Moreover, $\alpha$ is Martin-Löf random iff $d\ne 0$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof idea</summary>

The proof of the latter result can be split into two logically independent clauses, the interdiction of unbounded growth and the interdiction of infinite oscillation:

1. The unboundedness of the ratio $(\alpha-a_n)/(\beta-b_n)$ as $n\to\infty$ would imply the existence of a Martin-Löf test that $\beta$ fails, contradicting its Martin-Löf randomness.
2. The existence of two constants $c<d$ such that $(\alpha-a_n)/(\beta-b_n)<c$ for infinitely many $n$ and $(\alpha-a_n)/(\beta-b_n)>d$ for infinitely many $n$ would also imply the existence of a Martin-Löf test that $\beta$ fails, again contradicting its Martin-Löf randomness.

</details>
</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Problem</span><span class="math-callout__name">(Solovay reducibility is bad-behaved outside of left-c.e. reals)</span></p>

Outside the class of left-c.e. reals, the original notion of Solovay reducibility (Definition 8) is not widespread because it does not induce any meaningful degree structure on larger classes of reals.

</div>

### Versions of Solovay reducibility outside of left-c.e. reals

A modification of Solovay reducibility extending it to the computably approximable reals was introduced by Zheng and Rettinger in 2004. Today it is considered by some authors as the standard Solovay reducibility on c.a. reals.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(S2a-reducibility)</span></p>

A real $\alpha$ is S2a-reducible to a real $\beta$, written $\alpha\le_S^{2a}\beta$, if there exist two computable approximations $a_0,a_1,\ldots$ and $b_0,b_1,\ldots$ of $\alpha$ and $\beta$, respectively, and a constant $c$ such that

$$|\alpha-a_n|<c\bigl(|\beta-b_n|+2^{-n}\bigr)\qquad\text{for all }n. \tag{3}$$

</div>

#TODO: take a look at the proof

Rettinger and Zheng also showed that, on the set of d.c.e. reals, Martin-Löf random left-c.e. and right-c.e. reals form a highest degree; this result was strengthened by Miller: in the same way as in the Barmpalias-Lewis-Pye Limit Theorem, Miller showed that the ratio $\frac{\lvert\alpha-a_n\rvert}{\lvert \beta-b_n\rvert}$ is not only bounded but also convergent.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Miller)</span></p>

Let $\alpha$ be a d.c.e. real and $\beta$ be a Martin-Löf random d.c.e. real. Then there exists a real $d$ such that, for any two d.c.e. approximations

$$a_0,a_1,\ldots\to\alpha,\qquadb_0,b_1,\ldots\to\beta,$$

$$d=\lim_{n\to\infty}\frac{|\alpha-a_n|}{|\beta-b_n|}. \tag{4}$$

Moreover, $\alpha$ is Martin-Löf random iff $d\ne 0$.

</div>

---

---

<!-- <div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(\mathbb{R}-translation function)</span></p>

An $\mathbb R$-translation function from a real $\beta$ to a real $\alpha$ is a computable function

$$f\subseteq\mathbb R\to\mathbb R$$

that is defined on $[-\infty,\beta)$ and fulfils

$$\lim_{x\nearrow\beta}f(x)=\alpha.$$

A real $\alpha$ is cl-open-reducible to a real $\beta$, written $\alpha\le_{cL}^{\mathrm{open}}\beta$, if there exists a Lipschitz continuous $\mathbb R$-translation function from $\beta$ to $\alpha$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Titov)</span></p>

Let $\alpha$ be a real and $\beta$ be a Martin-Löf random real. Then there exists a real $d$ such that, for every $\mathbb Q$- or $\mathbb R$-translation function of bounded variation $f$ from $\beta$ to $\alpha$ (if any exists),

$$d=\lim_{x\nearrow\beta}\frac{\alpha-f(x)}{\beta-x}. \tag{5}$$

In particular, if an $\mathbb R$-translation function of bounded variation from $\beta$ to $\alpha$ exists, then $\alpha\le_{cL}^{\mathrm{open}}\beta$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Bounded variation of translation function is crucial)</span></p>

The requirement that $f$ have bounded variation is crucial: by [14, Proposition 1.11], for every real $\beta$, there exists a $\mathbb Q$-translation function of unbounded variation from $\beta$ to itself that does not fulfil (5).

</div>

---

--- -->

### Semicomputability and translation function intervals

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Upper semicomputable and Lower semicomputable function)</span></p>

A function $f:\mathbb R\to\mathbb R$ is **lower semicomputable** on a set $A$ if there exists a Turing functional with one oracle tape $M$ such that, for every Cauchy name $(q_0,q_1,\ldots)$ of a real $x\in A$ given as oracle, $M$ returns an increasing sequence converging to $f(x)$, and **upper semicomputable** if $-f$ is lower semicomputable.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Semicomputable functions)</span></p>

Let $I$ be an open interval and $f$ a real function defined on $I$.

* $f$ is lower semicomputable on $I$ $\iff$ the set $\lbrace x\in I(x)<q\rbrace$ is c.e. uniformly in $q\in\mathbb Q$.
* $f$ is computable $\iff$ $f$ is both lower and upper semicomputable.
* If $f$ is computable $\implies$ $f$ is continuous.
* If $f$ is lower (resp. upper) semicomputable $\implies$ $f$ is lower (resp. upper) semicontinuous.

</div>

#TODO: prove it

A **functional characterization of S2a-reducibility** via a function interval consisting of two Lipschitz continuous functions was found by Kumabe, Miyabe, and Suzuki.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Translation function interval)</span></p>

A **translation function interval** $(f,h)$ from a real $\beta$ to a real $\alpha$ is a pair of functions $f,h:\mathbb R\to\mathbb R$ such that

* $f(x)\le h(x)$ for all $x\in\mathbb R$;
* $f$ is lower semicomputable;
* $h$ is upper semicomputable;
* $f(\beta)=h(\beta)=\alpha$.

</div>

#TODO: what does translation function interval give us? What do we need it?

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name"></span></p>

In the definition of translation function interval, requirements (i)-(iv) imply altogether that

$$\lim_{x\to\beta}f(x)=f(\beta)=\lim_{x\to\beta}h(x)=h(\beta)=\alpha.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Since $f$ is lower semicomputable and $h$ is upper semicomputable, $f$ is lower semicontinuous and $h$ is upper semicontinuous by Proposition 17(iv), so

$$f(\beta)\le\liminf_{x\to\beta}f(x)\qquad\text{and}\qquad\limsup_{x\to\beta}h(x)\le h(\beta).$$

By requirement (i), $f(x)\le h(x)$ for all $x\in\mathbb R$, hence

$$\liminf_{x\to\beta}f(x)\le\liminf_{x\to\beta}h(x)\qquad\text{and}\qquad\limsup_{x\to\beta}f(x)\le\limsup_{x\to\beta}h(x).$$

By combining these inequalities with requirement (iv), we obtain

$$\alpha=f(\beta)\le\liminf_{x\to\beta}f(x)\le\limsup_{x\to\beta}f(x)\le\limsup_{x\to\beta}h(x)\le h(\beta)=\alpha$$

and

$$\alpha=f(\beta)\le\liminf_{x\to\beta}f(x)\le\liminf_{x\to\beta}h(x)\le\limsup_{x\to\beta}h(x)\le h(\beta)=\alpha.$$

Thus,

$$\liminf_{x\to\beta}f(x)=\limsup_{x\to\beta}f(x)=\alpha,$$

hence we obtain $\lim_{x\to\beta}f(x)=\alpha$, and similarly for $h$.

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name"></span></p>

If $(f,h)$ is a translation function interval from $\beta$ to $\alpha$, then $g=h-f$ is an upper semicomputable function that satisfies

$$\lim_{x\to\beta}g(x)=g(\beta)=0.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

The function $g(x)=h(x)-f(x)$ is upper semicomputable as the sum of two upper semicomputable functions $-f$ and $h$. By Proposition 19,

$$\lim_{x\to\beta}g(x)=\lim_{x\to\beta}(h(x)-f(x))=\lim_{x\to\beta}h(x)-\lim_{x\to\beta}f(x)=\alpha-\alpha=0,$$

and the equality $g(\beta)=h(\beta)-f(\beta)=\alpha-\alpha=0$ is immediate from (iv).

</details>
</div>

The next proposition shows that the set of computably approximable reals is closed downwards under the relation **"there exists a translation function from one real to another"**.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name"></span></p>

If $\beta$ is a c.a. real, and there exists a translation function interval from $\beta$ to $\alpha$, then $\alpha$ is c.a. as well.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name"></span></p>

If $(f,h)$ is a translation function interval from a c.a. real $\beta$ to another real $\alpha$, then there exists a computable approximation

$$b'_0,b'_1,\ldots\to\beta$$

such that

$$h(b'_i)-f(b'_i)<2^{-i}$$

for all $i$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of the Proposition</summary>

> The proof relies on the lemma above, whose proof is technical and is moved to the appendix. #TODO: prove it

Let $(f,h)$ be a translation function interval from a c.a. real $\beta$ to another real $\alpha$, and let $b'_0,b'_1,\ldots$ be a computable approximation of $\beta$ as guaranteed by Lemma 22, where $h(b'_i)-f(b'_i)<2^{-i}$ for every $i$.

We construct a computable approximation of $\alpha$ by defining

$$a_n=\widetilde f(b'_n), \tag{6}$$

where $\widetilde f(b'_n)$ is the value of $f(b'_n)$ approximated from below with accuracy $2^{-n}$ (which is possible by simultaneously approximating $f(b'_n)$ from below and $h(b'_n)$ from above, since $h(b'_n)-f(b'n)<2^{-n}$). The sequence $a_0,a_1,\ldots$ is infinite, computable, and has limit $\alpha$ since $\lim{x\to\beta}f(x)=\alpha$ by Proposition 19. Thus, $\alpha$ is computably approximable.

</details>
</div>

Kumabe, Miyabe, and Suzuki demonstrated in 2025 that S2a-reducibility can be equivalently characterized via the existence of a translation function interval consisting of two Lipschitz continuous functions.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name"></span></p>

A c.a. real $\alpha$ is S2a-reducible to a c.a. real $\beta$ iff there exists a translation function interval $(f,h)$ from $\beta$ to $\alpha$ such that $f$ and $h$ are Lipschitz continuous.

</div>

#TODO: prove it

In the context of randomness, by Remark 15, it makes sense to consider only translation function intervals $(f,h)$ where both $f$ and $h$ have bounded variation. In particular, by Proposition 4, nondecreasing or Lipschitz continuous functions automatically have bounded variation.

In 2025, Titov conjectured that the interdiction of infinite oscillation, which is the second logical clause of Theorem 14, also holds for function intervals.

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Conjecture</span><span class="math-callout__name">(Titov)</span></p>

Let $\alpha$ be a c.a. real and $\beta$ be a Martin-Löf random c.a. real that fulfils $\alpha\le_S^{2a}\beta$ via a function interval $(f,h)$. 

Then there exists a constant $d$ such that

$$f'(\beta)=h'(\beta)=d,$$

where $d$ does not depend on the choice of the function interval witnessing the reducibility $\alpha\le_S^{2a}\beta$. 

Moreover, $d=0$ $\iff$ $\alpha$ is not Martin-Löf random.

</div>

In the next section, we prove that the conjecture is incorrect. In fact, both interdictions of unbounded growth and of infinite oscillation, which are the two clauses of the Barmpalias-Lewis-Pye Limit Theorem, fail for function intervals.

## The Barmpalias–Lewis-Pye Limit Theorem does not hold for S2a-reducibility

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(25)</span></p>

There exist a c.a. Martin-Löf random real $\beta$, a function interval $(-h,h)$ from $\beta$ to $0$, and a function interval $(\widetilde f,\widetilde h)$ from $\beta$ to $\beta$ such that $h$ is Lipschitz continuous, and $\widetilde f$ and $\widetilde h$ are nondecreasing,

$$
\begin{aligned} \liminf_{x\nearrow\beta}\left|D_\beta^h(x)\right|
&=\liminf_{x\searrow\beta}\left|D_\beta^h(x)\right| =0<1 =\limsup_{x\nearrow\beta}\left|D_\beta^h(x)\right| =\limsup_{x\searrow\beta}\left|D_\beta^h(x)\right|,\\
\limsup_{x\nearrow\beta}\left|D_\beta^{\widetilde f}(x)\right| 
&=\infty =\limsup_{x\searrow\beta}\left|D_\beta^{\widetilde h}(x)\right|.
\end{aligned}$$

</div>

## Conclusion and future work
