---
title: "Cheatsheet — Lecture 2: Multi-Armed Bandits"
layout: default
noindex: true
math: true
tags:
  - reinforcement-learning
  - multi-armed-bandits
  - exploration-exploitation
  - ucb
  - gradient-bandits
  - cheatsheet
---

# Cheatsheet — Lecture 2: Multi-Armed Bandits

*Exam recap: formulas and key results only. Full explanations live in the [main notes]({{ '/subpages/reinforcement_learning_hd/' | relative_url }}#lecture-2-multi-armed-bandits).*

## The $k$-armed bandit

* $k$ actions, each with unknown **value** $q_\ast(a)=\mathbb{E}[R_t\mid A_t=a]$. Goal: maximise total reward. **Stationary** = the $q_\ast(a)$ don't change over time.
* **Evaluative** feedback (*how good* the taken action was) — **not instructive** (which action was best). This is what **forces exploration**: you only learn about actions you actually try.
* The whole problem = **explore vs exploit**, with **no state and no delayed reward** (RL stripped to one step).

## Value estimation

* **Sample average** (average the rewards seen for $a$): $\;Q\_n(a)=\dfrac{1}{n-1}\sum_{i=1}^{n-1}R_i \xrightarrow{\text{LLN}} q_\ast(a)$.
* **Incremental update** (no need to store history):

$$\boxed{\,Q_{n+1}=Q_n+\tfrac{1}{n}\,(R_n-Q_n)\,}$$

* **Universal RL update template:** $\;\text{new} \leftarrow \text{old} + \text{step}\cdot(\text{target}-\text{old})$.

## Non-stationarity → constant step size

$$\boxed{\,Q_{n+1}=Q_n+\alpha\,(R_n-Q_n)\,},\qquad \alpha\in(0,1].$$

* Unrolls to an **exponential recency-weighted average**:

$$Q_n=(1-\alpha)^{n-1}Q_1+\sum_{i=1}^{n-1}\alpha(1-\alpha)^{\,n-1-i}R_i,$$

  weights sum to $1$; recent rewards count more (older ones decay geometrically).
* **Sample average** ($\alpha=\tfrac1n$) = *uniform memory* → best for **stationary**; **constant $\alpha$** *tracks a moving target* → best for **non-stationary**.
* **Robbins–Monro convergence:** $\sum_n\alpha_n=\infty$ **and** $\sum_n\alpha_n^2<\infty$. Sample-average satisfies both (converges); constant $\alpha$ satisfies only the first (keeps adapting — *by design* for non-stationary).

## Exploration strategies (value-based)

* **Greedy:** $A\_t=\arg\max_a Q_t(a)$ — pure exploitation; can lock onto a suboptimal arm forever.
* **$\varepsilon$-greedy:** greedy w.p. $1-\varepsilon$, else uniform random. Every action sampled infinitely often ⇒ $Q\to q_\ast$ — but exploration is **blind** (wastes pulls on clearly-bad arms).
* **Optimistic initial values:** set $Q\_1(a)$ high ⇒ every action "disappoints" ⇒ all tried early. *Implicit, transient* exploration only — fades once explored, useless when non-stationary.
* **UCB (optimism under uncertainty):**

$$\boxed{\,A_t=\arg\max_a\left[\,Q_t(a)+c\,\sqrt{\tfrac{\ln t}{N_t(a)}}\,\right]}$$

  Bonus = **uncertainty**: grows with $t$ (arm untried for a while), shrinks with visit count $N\_t(a)$. Directs exploration to arms that are *promising **or** under-explored* — smarter than $\varepsilon$-greedy's uniform noise.

## Gradient bandits (policy-based)

Learn **preferences** $H\_t(a)$ (not values); act by **softmax**:

$$\pi_t(a)=\frac{e^{H_t(a)}}{\sum_{b=1}^{k} e^{H_t(b)}}.$$

Stochastic **gradient ascent** on expected reward $J(H)=\mathbb{E}[R_t]$:

$$\boxed{\,H_{t+1}(a)=H_t(a)+\alpha\,(R_t-\bar R_t)\bigl(\mathbf{1}\lbrace A_t=a\rbrace-\pi_t(a)\bigr)\,}$$

* **Baseline** $\bar R\_t$ (running average reward): reduces variance, **does not bias** the gradient.
* Effect: if $R\_t>\bar R\_t$, push the chosen action's preference **up** and the rest **down** (reverse if below). Learns a **policy directly** — the precursor to policy-gradient methods.

## Contextual bandits (associative search)

* Observe a **context** $x\_t$ before acting ⇒ learn a **policy** $a=\pi(x)$ (best action depends on context).
* A bridge toward full RL — but actions still **don't change the next context**: no state transitions, no **delayed consequences**.

## Method map

| Idea | Fixes | Type |
| :-- | :-- | :-- |
| $\varepsilon$-greedy | greedy never explores | value, blind exploration |
| Optimistic init | needs early exploration | value, transient |
| **UCB** | blind exploration | value, uncertainty-directed |
| **Gradient bandit** | learns values, not a policy | policy-based (softmax) |
| Contextual | ignores context | associative (state, no transitions) |

**Punchline:** bandits isolate **evaluative feedback + exploration** with no state and no delayed reward — every later RL idea (value estimation, step-sizes, exploration, policy gradients) shows up here in its simplest form.
