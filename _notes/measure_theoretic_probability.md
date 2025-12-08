---
layout: default
title: Measure-theoretic probability
date: 2024-11-01
# excerpt: Connecting differential equations, stability analysis, and attractor theory to the training dynamics of modern machine learning models.
# tags:
#   - dynamical-systems
#   - machine-learning
#   - theory
---

# Measure-theoretic probability

These notes present the measure-theoretic basis of modern probability.

Everyone with a basic notion of mathematics and probability would understand what is meant by $f(x)$ and $\mathbb{P}(A)$. The similarity in notation suggests a shared idea: $\mathbb{P}$ can be treated as a function too — specifically, a special kind of **measure** defined on a carefully chosen collection of sets that satisfy certain properties. That collection is a **$\sigma$-algebra**, and a measure is a map on it satisfying certain axioms. This leads to the abstract notion of a measure space $(S, \Sigma, \mu)$. In probability, by convention and because of some distinctive features, the same structure is written as a probability space $(\Omega, \mathcal{F}, \mathbb{P})$.

Within this framework, one develops integrals of functions on $S$. In probabilistic terms, these integrals represent **expectations**. The setup has two main benefits:

1. It removes the need to treat discrete and continuous random variables separately when defining expectations. Sums and Riemann integrals become special cases of the more general Lebesgue integral.
2. It provides strong **convergence theorems** from analysis, which translate into results about convergence of expectations of random variables.

## 1. $\sigma$-algebras and measures

### 1.1 $\sigma$-algebras

$\textbf{Definition (Algebra of sets):}$ Let $S$ be a nonempty set. A collection $\Sigma_0 \subset 2^S$ is called an **algebra on $S$** if:
1. $S \in \Sigma_0$,
2. $E \in \Sigma_0 \implies E^c \in \Sigma_0$,
3. $E, F \in \Sigma_0 \implies E \cup F \in \Sigma_0$.

From these properties it follows that:
* $\varnothing \in \Sigma_0$ because $\varnothing = S^c$,
* closure under unions automatically extends to **finite** unions, 
* it is also closed under intersections since $E \cap F = (E^c \cup F^c)^c$,
* and under set differences since $E\setminus F = E \cap F^c$.

> **Remark**: finite number unions doesn’t automatically imply countable number of unions.

$\textbf{Definition } (\sigma\textbf{-algebra}):$ Let $S$ be a nonempty set. A collection $\Sigma \subset 2^S$ is a **$\sigma$-algebra on $S$** if:
* it is an algebra, and additionally
* it is closed under **countable unions**: $E_n \in \Sigma (n=1,2,\dots) \implies \bigcup_{n=1}^\infty E_n \in \Sigma$.

If $\Sigma$ is a $\sigma$-algebra on $S$, then $(S,\Sigma)$ is called a **measurable space**, and the sets in $\Sigma$ are called **measurable sets**.

> **Remark**: Yes, a sigma-algebra is essentially an algebra (or field of sets) that adds the crucial property of being closed under **countable** unions and intersections.

> **Remark**: In a sigma algebra, the Greek letter sigma ($\sigma$) signifies closure under countable unions, coming from the German word "Summe" (sum), representing the idea that you can "sum" or combine an unlimited, but countable, number of sets within the collection. It distinguishes ($\sigma$)-algebras from regular algebras, which only require closure under finite unions, indicating a more powerful structure for defining probabilities and measures. 

**Why $\sigma$-algebra is so important compared basic algebra or why countable unions matter in probability:**
<div class="accordion">
  <details markdown="1">
    <summary>Why $\sigma$-algebra is so important compared basic algebra or why countable unions matter in probability.</summary>

### Why **countable** unions matter (not just finite)

Probability constantly deals with **limits** of events.
And limits usually produce **countable** operations.

#### 1. “Limit of events” naturally means countable unions/intersections

If you have events $E_1, E_2, \dots$, you often define things like:

* **“Eventually happens”**:
  
  $$\limsup_{n\to\infty} E_n = \bigcap_{n=1}^\infty \bigcup_{k\ge n} E_k$$
  
* **“Happens infinitely often”** is the same idea.
* **“Happens for all sufficiently large $n$”** uses similar forms.

These objects are built from **countable unions and intersections**.
So you need your collection of measurable sets to be closed under them.

#### 2. Convergence of random variables uses countable event operations

Example:

$$X_n \to X \text{ a.s.} \quad \Longleftrightarrow \quad \mathbb P\Big(\bigcap_{m=1}^\infty \bigcup_{n\ge m}{|X_n-X|>1/m}\Big)=0$$


Again: *countable* operations appear automatically.

#### 3. Continuity properties of probability depend on this

A probability measure (\mathbb P) satisfies:

* If $E_n \uparrow E$, then $\mathbb P(E_n)\uparrow \mathbb P(E)$
  
* If $E_n \downarrow E$, then $\mathbb P(E_n)\downarrow \mathbb P(E)$
  
These are essential for proofs and only make sense cleanly when the limit set $E$ stays inside your event class — which is guaranteed by a $\sigma$-algebra.

#### 4. You need Borel sets for real-valued random variables

Even defining: $\lbrace X \le x\rbrace$
for all real $x$, and more complicated events like $\lbrace X \in B\rbrace$, pushes you toward the **Borel $\sigma$-algebra**, which is generated using countable operations.

### One-line summary

Finite closure (an algebra) is good for simple set manipulations;
**countable** closure (a $\sigma$-algebra) is what makes probability stable under limits — and modern probability is full of limit arguments.

If you want, I can give a tiny concrete example showing an algebra that *fails* to contain a natural limit event, illustrating exactly what breaks.

  </details>
</div>

$\textbf{Definition (Generated $\sigma$-algebra)}:$ Given any collection $\mathcal C$ of subsets of $S$, the notation $\sigma(\mathcal C)$ means:
* the **smallest $\sigma$-algebra containing $\mathcal C$**,
* equivalently, the **intersection of all $\sigma$-algebras** that contain $\mathcal C$.

If $\Sigma = \sigma(\mathcal C)$, we say that $\mathcal C$ **generates** $\Sigma$.

> **Remark**:
> * the union of two $\sigma$-algebras $\Sigma_1 \cup \Sigma_2$ is generally **not** a $\sigma$-algebra.
> * We write $\Sigma_1 \vee \Sigma_2 := \sigma(\Sigma_1 \cup \Sigma_2)$.
  
*Important example*: Borel $\sigma$-algebra

A central example in probability is the **Borel $\sigma$-algebra on $\mathbb{R}$**: $\mathcal{B} = \mathcal{B}(\mathbb{R})$.

$\textbf{Example (Borel $\sigma$-algebra):}$ Let $\mathcal{O}$ be the collection of all open subsets of $\mathbb{R}$ (in the usual topology, where intervals $(a,b)$ are open). Then:

$$\mathcal B := \sigma(\mathcal O).$$

Similarly, one defines Borel sets in $\mathbb{R}^d$, and more generally for any topological space $(S,\mathcal O)$:

$$\text{Borel sets} = \sigma(\mathcal O).$$


Borel sets can be quite complicated, but it helps to remember they are generated from simple building blocks like open sets.