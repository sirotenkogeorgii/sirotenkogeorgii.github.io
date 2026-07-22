---
title: "Cheatsheet — Lecture 3: Finite MDPs"
layout: default
noindex: true
math: true
tags:
  - reinforcement-learning
  - markov-decision-process
  - bellman-equations
  - cheatsheet
---

# Cheatsheet — Lecture 3: Finite MDPs

*Exam recap: formulas and key results only. Full explanations live in the [main notes]({{ '/subpages/reinforcement_learning_hd/' | relative_url }}#lecture-3-finite-markov-decision-processes).*

## MDP objects

* **Finite MDP** = tuple $(\mathcal{S},\, \mathcal{A},\, p,\, \gamma)$; $\mathcal{S},\mathcal{A}$ finite, rewards bounded, $\gamma\in[0,1]$.
* **One-step dynamics** (contains *everything*):

$$p(s', r \mid s, a) \doteq \Pr\lbrace S_{t+1}=s',\, R_{t+1}=r \mid S_t=s,\, A_t=a\rbrace,\qquad \sum_{s',r}p(s',r\mid s,a)=1.$$

* **Derived quantities:**

$$p(s'\mid s,a)=\sum_r p(s',r\mid s,a),\qquad r(s,a)=\sum_{s',r}r\,p(s',r\mid s,a).$$

* **Markov property** — future depends on past only through the present:

$$\Pr(S_{t+1},R_{t+1}\mid S_0,A_0,\dots,S_t,A_t)=\Pr(S_{t+1},R_{t+1}\mid S_t,A_t).$$

  Punchline: Markov is a constraint on **state design**, not on the world — if history matters, the state is too small.

## Return & discounting

* **Episodic:** $\;G\_t = R\_{t+1}+R\_{t+2}+\cdots+R\_T$, with $G\_T=0$.
* **Discounted (continuing):**

$$G_t=\sum_{k=0}^{\infty}\gamma^k R_{t+k+1},\qquad 0\le\gamma<1.$$

* **Recursive identity** (seed of every Bellman equation):

$$\boxed{\,G_t = R_{t+1}+\gamma\, G_{t+1}\,}$$

* **Bounded return:** $\lvert R\rvert\le R\_{\max}\Rightarrow \lvert G\_t\rvert\le \dfrac{R\_{\max}}{1-\gamma}$. Small $\gamma$ = short-sighted, $\gamma\to1$ = far-sighted.

## Policy & value functions

* **Policy:** $\pi(a\mid s)\doteq\Pr(A\_t=a\mid S\_t=s)$.
* **State value / action value** (fix $\pi$):

$$v_\pi(s)=\mathbb{E}_\pi[G_t\mid S_t=s],\qquad q_\pi(s,a)=\mathbb{E}_\pi[G_t\mid S_t=s,A_t=a].$$

* **Link:** $\;v\_\pi(s)=\sum\_a \pi(a\mid s)\,q\_\pi(s,a)$.
* **Two problems:** *prediction* = compute $v\_\pi,q\_\pi$ for fixed $\pi$; *control* = find $\pi^\ast\in\arg\max\_\pi v\_\pi(s)\;\forall s$.

## Bellman expectation equations (fixed $\pi$)

$$\boxed{\,v_\pi(s)=\sum_{a}\pi(a\mid s)\sum_{s',r}p(s',r\mid s,a)\bigl[r+\gamma\,v_\pi(s')\bigr]\,}$$

$$\boxed{\,q_\pi(s,a)=\sum_{s',r}p(s',r\mid s,a)\Bigl[r+\gamma\sum_{a'}\pi(a'\mid s')\,q_\pi(s',a')\Bigr]\,}$$

* Read as a **backup**: average over action ($\pi$) → average over environment ($p$) → plug in successor value.
* One equation per state ⇒ a **linear system** in $\lvert\mathcal{S}\rvert$ unknowns.

## Optimality

* **Policy order:** $\pi\succeq\pi' \iff v\_\pi(s)\ge v\_{\pi'}(s)\;\forall s$. $\pi^\ast$ optimal if it dominates all.
* **Optimal values:** $v\_\ast(s)=\max\_\pi v\_\pi(s)$, $\;q\_\ast(s,a)=\max\_\pi q\_\pi(s,a)$, with $\;v\_\ast(s)=\max\_a q\_\ast(s,a)$.
* **Facts:** optimal *values* are **unique**; optimal *policies* need not be; a **deterministic** optimal policy always exists.

## Bellman optimality equations

$$\boxed{\,v_\ast(s)=\max_{a}\sum_{s',r}p(s',r\mid s,a)\bigl[r+\gamma\,v_\ast(s')\bigr]\,}$$

$$\boxed{\,q_\ast(s,a)=\sum_{s',r}p(s',r\mid s,a)\Bigl[r+\gamma\max_{a'}q_\ast(s',a')\Bigr]\,}$$

* Difference from expectation eqs: **average over actions $\to$ max over actions** (that single change = control). Environment branching is never max'd away.

## Greedy extraction (the punchline)

$$\pi_{\text{greedy}}(s)\in\arg\max_a\sum_{s',r}p(s',r\mid s,a)\bigl[r+\gamma\,v_\ast(s')\bigr],\qquad \pi^\ast(s)\in\arg\max_a q_\ast(s,a).$$

$$\boxed{\text{Solve the Bellman optimality equations} \Rightarrow \text{optimal control is a one-step }\arg\max\text{ per state.}}$$

* With $v\_\ast$ you need the model $p$ to act greedily; with $q\_\ast$ you do **not** — hence control methods learn $q$.

## One-glance table

| Object | Meaning | Backup |
| :-- | :-- | :-- |
| $v\_\pi(s)$ | value of $s$ under $\pi$ | expectation over action *and* environment |
| $q\_\pi(s,a)$ | value of $a$ then $\pi$ | environment, then policy |
| $v\_\ast(s)$ | best state value | **max** over actions |
| $q\_\ast(s,a)$ | best value after forcing $(s,a)$ | transition + max over future |
