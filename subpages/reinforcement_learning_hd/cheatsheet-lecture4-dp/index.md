---
title: "Cheatsheet — Lecture 4: Dynamic Programming"
layout: default
noindex: true
math: true
tags:
  - reinforcement-learning
  - dynamic-programming
  - policy-iteration
  - value-iteration
  - cheatsheet
---

# Cheatsheet — Lecture 4: Dynamic Programming

*Exam recap: formulas and key results only. Full explanations live in the [main notes]({{ '/subpages/reinforcement_learning_hd/' | relative_url }}#lecture-4-dynamic-programming).*

## Setup

* **DP = planning with a known model** $p(s',r\mid s,a)$. No sampling — every expectation is an *exact* finite sum.
* Three primitives: **evaluation** → **improvement** → **iterate to a fixed point**.
* PI and VI differ only in *how much evaluation* is done between improvements.

## Policy evaluation (prediction)

**Iterative update** (Bellman *expectation* backup; converges to $v\_\pi$ for $\gamma<1$):

$$\boxed{\,v_{k+1}(s)=\sum_a \pi(a\mid s)\sum_{s',r}p(s',r\mid s,a)\bigl[r+\gamma\,v_k(s')\bigr]\,}$$

* **Expected update** = bootstrapping (use current $v\_k(s')$) + full backup (whole successor distribution).
* **Two-array (sync)** vs **in-place (async)** — same fixed point; in-place is cheaper and usually faster.

**Linear-algebra view** (a fixed $\pi$ turns the MDP into a Markov chain):

$$v_\pi = r_\pi + \gamma P_\pi v_\pi \;\Longleftrightarrow\; (I-\gamma P_\pi)\,v_\pi = r_\pi \;\Longrightarrow\; v_\pi=(I-\gamma P_\pi)^{-1}r_\pi,$$

$$r_\pi(s_i)=\sum_a\pi(a\mid s_i)\sum_{s',r}p(s',r\mid s_i,a)\,r,\qquad (P_\pi)_{ij}=\sum_a\pi(a\mid s_i)\,p(s_j\mid s_i,a).$$

* $(I-\gamma P\_\pi)$ **invertible**: $P\_\pi$ row-stochastic $\Rightarrow \lvert\lambda\rvert\le1 \Rightarrow 1-\gamma\lambda\ne0$.
* Direct solve $O(\lvert\mathcal{S}\rvert^3)$; iteration preferred (sparse, cheap, survives into model-free RL).

**Bellman expectation operator:** $\;T\_\pi v \doteq r\_\pi+\gamma P\_\pi v$ (affine); update is $v\_{k+1}=T\_\pi v\_k$; $v\_\pi$ is its unique fixed point $v\_\pi=T\_\pi v\_\pi$.

## Contraction & convergence

$$\boxed{\,\lVert T_\pi u - T_\pi v\rVert_\infty \le \gamma\lVert u-v\rVert_\infty\,}\qquad\text{(same for optimality operator }T\text{)}$$

* Geometric convergence, rate $\gamma$: $\;\lVert v\_{k+1}-v\_\pi\rVert\_\infty\le\gamma\lVert v\_k-v\_\pi\rVert\_\infty$.
* **A-posteriori bound:** $\;\lVert V-v\_\pi\rVert\_\infty\le \Delta/(1-\gamma)$ where $\Delta$ = max change in a sweep.
* **Stopping rule:** iterate until $\lVert v\_{k+1}-v\_k\rVert\_\infty < \dfrac{\varepsilon(1-\gamma)}{\gamma}\;\Rightarrow\; \lVert v\_k-v\_\ast\rVert\_\infty<\varepsilon$.

## Policy improvement

**One-step lookahead value** (take $a$ once, then follow $\pi$):

$$q_\pi(s,a)=\sum_{s',r}p(s',r\mid s,a)\bigl[r+\gamma\,v_\pi(s')\bigr].$$

**Policy Improvement Theorem** — if for *every* $s$: $\;q\_\pi(s,\pi'(s))\ge v\_\pi(s)$, then $v\_{\pi'}(s)\ge v\_\pi(s)\;\forall s$ (strict where the premise is strict).

* Punchline: a **local one-step** improvement is automatically a **global** improvement.

**Greedy policy** (guaranteed $\ge\pi$; strictly better unless $\pi$ already greedy):

$$\pi'(s)\in\arg\max_a q_\pi(s,a)=\arg\max_a\sum_{s',r}p(s',r\mid s,a)\bigl[r+\gamma\,v_\pi(s')\bigr].$$

* Memory aid: **evaluation changes values; improvement changes the policy.**

## Policy iteration

$$\pi_0 \xrightarrow{E} v_{\pi_0} \xrightarrow{I} \pi_1 \xrightarrow{E} v_{\pi_1} \xrightarrow{I} \cdots \xrightarrow{I} \pi_\ast.$$

1. Evaluate $v\_\pi$ (to tolerance $\theta$). 2. Improve greedily. 3. Stop when the policy is unchanged.

* **Stable ⇒ optimal:** if greedy improvement leaves $\pi$ unchanged, $v\_\pi$ satisfies the Bellman optimality equation, so $v\_\pi=v\_\ast$.
* **Terminates in finitely many steps:** each change is a *strict* improvement, and there are $\le\lvert\mathcal{A}\rvert^{\lvert\mathcal{S}\rvert}$ deterministic policies (no policy revisited). Usually only a handful of outer iterations.

## Value iteration

**Merge evaluation + improvement into one max backup** (Bellman *optimality* update):

$$\boxed{\,v_{k+1}(s)=\max_a\sum_{s',r}p(s',r\mid s,a)\bigl[r+\gamma\,v_k(s')\bigr]\,}$$

* Extract $\pi^\ast$ once at the end: $\;\pi^\ast(s)\in\arg\max\_a\sum\_{s',r}p(s',r\mid s,a)[r+\gamma\,V(s')]$.
* **Convergence:** $\;\lVert v\_k-v\_\ast\rVert\_\infty\le\gamma^k\lVert v\_0-v\_\ast\rVert\_\infty$.
* **Iteration count** to reach $\varepsilon$: $\;k^\ast=\left\lceil \dfrac{\log(\varepsilon/\lVert v\_0-v\_\ast\rVert\_\infty)}{\log\gamma}\right\rceil$. E.g. $\gamma=0.9,\varepsilon=0.01,\lVert v\_0-v\_\ast\rVert\_\infty\le10\Rightarrow k^\ast=66$. As $\gamma\to1$: $-\log\gamma\sim(1-\gamma)$ ⇒ **slow**.
* View: VI = policy iteration with evaluation **truncated to one sweep**.

## PI vs VI

| | Policy iteration | Value iteration |
| :-- | :-- | :-- |
| Objects | policy $+$ value fn | value fn only |
| Evaluation | many sweeps | one sweep |
| Improvement | explicit greedy step | built into the max |
| Intermediate $v$ | true values of *real* policies | estimates of $v\_\ast$ (no policy) |
| Mantra | evaluate, then improve | improve during every backup |

## Generalised Policy Iteration (GPI)

* Any interleaving of "make $V$ closer to $v\_\pi$" and "make $\pi$ greedy w.r.t. $V$" — neither need be exact.
* PI (full evaluation) and VI (one sweep) are the two extremes; **asynchronous DP** updates a subset of states per sweep (every state visited infinitely often).
* Every later method (MC, SARSA, Q-learning, actor–critic) = GPI with one arrow replaced by a sample-based / model-free surrogate.
