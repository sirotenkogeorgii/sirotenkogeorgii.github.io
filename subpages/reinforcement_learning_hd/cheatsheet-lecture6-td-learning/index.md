---
title: "Cheatsheet — Lecture 6: Temporal-Difference Learning"
layout: default
noindex: true
math: true
tags:
  - reinforcement-learning
  - temporal-difference-learning
  - sarsa
  - q-learning
  - cheatsheet
---

# Cheatsheet — Lecture 6: Temporal-Difference Learning

*Exam recap: formulas and key results only. Full explanations live in the [main notes]({{ '/subpages/reinforcement_learning_hd/' | relative_url }}#lecture-6-temporal-difference-learning).*

## The idea — sampling **and** bootstrapping

* **TD** learns from **experience** (like MC — no model) **and** **bootstraps** (like DP — updates a guess from other guesses). Updates **online, every step**, before the episode ends.
* Taxonomy: **DP** = bootstrap + *expected* backup (needs a model); **MC** = *sample* + no bootstrap; **TD** = *sample* + bootstrap.

## TD(0) prediction

$$\boxed{\,V(S_t)\;\leftarrow\;V(S_t)+\alpha\bigl[\,R_{t+1}+\gamma V(S_{t+1})-V(S_t)\,\bigr]\,}$$

* **TD error:** $\;\delta_t=R_{t+1}+\gamma V(S_{t+1})-V(S_t)$.
* The **TD target** $R_{t+1}+\gamma V(S_{t+1})$ is the **sampled Bellman expectation equation** — *two* approximations: sample instead of the full expectation, and **bootstrap** ($V$ in place of $v\_\pi$).
* If $V$ is held fixed, the **MC error is the sum of discounted TD errors**: $G_t-V(S_t)=\sum_{k=t}^{T-1}\gamma^{k-t}\delta_k$.

## TD vs MC

| | **Monte Carlo** | **TD(0)** |
| :-- | :-- | :-- |
| Target | $G\_t$ (unbiased) | $R_{t+1}+\gamma V(S_{t+1})$ (biased) |
| Variance | high | low |
| Timing | offline (wait for episode end) | **online, incremental** |
| Tasks | episodic only | continuing **and** episodic |

* **Batch fixed points differ:** batch MC minimizes **MSE on the data**; batch TD gives the **certainty-equivalence** estimate — the value of the **maximum-likelihood MDP** fit to the data. TD exploits the **Markov** structure, so it's usually faster / more accurate there.

## Control (GPI with TD) — all share $Q\leftarrow Q+\alpha[\,\text{target}-Q\,]$

**Sarsa** (on-policy — uses the *actually taken* next action $A\_{t+1}$):

$$\boxed{\,Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha\bigl[R_{t+1}+\gamma Q(S_{t+1},A_{t+1})-Q(S_t,A_t)\bigr]\,}$$

**Q-learning** (off-policy — target uses the **greedy** action regardless of behaviour):

$$\boxed{\,Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha\bigl[R_{t+1}+\gamma\max_a Q(S_{t+1},a)-Q(S_t,A_t)\bigr]\,}$$

**Expected Sarsa** (expectation over the next action ⇒ lower variance, no sampled $A\_{t+1}$):

$$Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha\Bigl[R_{t+1}+\gamma\textstyle\sum_a\pi(a\mid S_{t+1})Q(S_{t+1},a)-Q(S_t,A_t)\Bigr].$$

* **Sarsa** learns the value of the **exploratory policy it follows** (converges to optimal under **GLIE**); **Q-learning** learns $q\_\ast$ **directly** while behaving exploratorily; **Expected Sarsa** with a greedy target $=$ Q-learning.
* **Cliff walking:** Q-learning finds the **optimal cliff-edge** path but gets worse *online* returns (ε-exploration falls off); **Sarsa** learns a **safer, longer** route because it accounts for exploration.

## Maximization bias & Double Q-learning

* **Max bias:** using the same estimates to **select and evaluate** the greedy action overestimates —

$$\mathbb{E}\bigl[\max_a Q(s',a)\bigr]\;\ge\;\max_a q_\ast(s',a).$$

* **Double Q-learning:** keep two tables; **select with one, evaluate with the other** (swap $Q\_1\leftrightarrow Q\_2$ w.p. $\tfrac12$):

$$\boxed{\,Q_1(S,A)\leftarrow Q_1(S,A)+\alpha\Bigl[R+\gamma\,Q_2\bigl(S',\arg\max_a Q_1(S',a)\bigr)-Q_1(S,A)\Bigr]\,}$$

  Decoupling gives $\mathbb{E}[Q_2(s',A^\ast)]=\mathbb{E}[q_\ast(s',A^\ast)]\le\max_a q_\ast(s',a)$, removing the upward bias (not perfectly unbiased, but much closer).

## Method map

| Method | Policy | Next-state target |
| :-- | :-- | :-- |
| **TD(0)** | prediction | $R+\gamma V(S')$ |
| **Sarsa** | on-policy | $R+\gamma Q(S',A')$ (sampled) |
| **Expected Sarsa** | on/off-policy | $R+\gamma\sum_a\pi(a\mid S')Q(S',a)$ |
| **Q-learning** | off-policy | $R+\gamma\max_a Q(S',a)$ |
| **Double Q** | off-policy | decoupled max (bias-corrected) |

**Punchline:** every method is $Q\leftarrow Q+\alpha[\,\text{target}-Q\,]$ — they differ **only** in how the target treats the next action (**sampled** / **expected** / **max** / **decoupled max**).
