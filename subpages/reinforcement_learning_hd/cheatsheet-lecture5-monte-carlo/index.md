---
title: "Cheatsheet — Lecture 5: Monte Carlo Methods"
layout: default
noindex: true
math: true
tags:
  - reinforcement-learning
  - monte-carlo
  - importance-sampling
  - off-policy
  - cheatsheet
---

# Cheatsheet — Lecture 5: Monte Carlo Methods

*Exam recap: formulas and key results only. Full explanations live in the [main notes]({{ '/subpages/reinforcement_learning_hd/' | relative_url }}#lecture-5-monte-carlo-methods).*

## The idea

* **MC** = model-free prediction **and** control from **sampled complete episodes** — just **average the actual returns** $G\_t$.
* **No model** (unlike DP), **no bootstrapping** (unlike DP/TD): each estimate uses a *full* return, not other estimates.
* **Episodic tasks only** — you must wait for an episode to terminate before updating.

## MC prediction

$$v_\pi(s)=\mathbb{E}_\pi[G_t\mid S_t=s]\;\approx\;\text{average of returns following visits to }s.$$

* **First-visit MC:** average returns after the **first** visit to $s$ in each episode — i.i.d., **unbiased**.
* **Every-visit MC:** average after **every** visit — biased but **consistent** (both $\to v\_\pi$).
* **Incremental update:**

$$V(S_t)\;\leftarrow\;V(S_t)+\tfrac{1}{N(S_t)}\bigl[G_t-V(S_t)\bigr]$$

  (or a constant step $\alpha$ for non-stationary tracking).

## Why action values (control needs $q$)

* Without a model you **can't act greedily from $v\_\pi$** (that needs a one-step look-ahead). So estimate $q\_\pi(s,a)$ directly ⇒ greedy $=\arg\max_a Q(s,a)$.
* **Exploration problem:** under a deterministic policy many $(s,a)$ are never visited ⇒ can't improve them. Fixes: **exploring starts** (any $(s,a)$ can begin an episode) or **$\varepsilon$-soft** policies.

## On-policy MC control ($\varepsilon$-soft)

* **$\varepsilon$-soft:** $\pi(a\mid s)\ge\dfrac{\varepsilon}{\lvert\mathcal{A}(s)\rvert}$ for all $a$ (every action keeps probability).
* **$\varepsilon$-greedy** (simplest $\varepsilon$-soft):

$$\pi(a\mid s)=\begin{cases}1-\varepsilon+\dfrac{\varepsilon}{\lvert\mathcal{A}(s)\rvert}, & a=\arg\max_{a'}Q(s,a'),\\[4pt]\dfrac{\varepsilon}{\lvert\mathcal{A}(s)\rvert}, & \text{otherwise.}\end{cases}$$

* **GPI:** evaluate $q\_\pi$ by MC averaging, then improve **$\varepsilon$-greedily**. $\varepsilon$-greedy improvement is **monotone within the $\varepsilon$-soft class** ⇒ converges to the best $\varepsilon$-soft policy (not the true optimum unless $\varepsilon\to0$).

## Off-policy MC (behaviour $b\ne$ target $\pi$)

Learn about **target** $\pi$ while following **behaviour** $b$. **Coverage:** $\pi(a\mid s)>0\Rightarrow b(a\mid s)>0$.

**Importance-sampling ratio** (dynamics cancel — depends only on the policies):

$$\boxed{\,\rho_{t:T-1}=\prod_{k=t}^{T-1}\frac{\pi(A_k\mid S_k)}{b(A_k\mid S_k)}\,}$$

* **Ordinary IS** — $V(s)=\dfrac{1}{\lvert\mathcal{T}(s)\rvert}\displaystyle\sum_{t\in\mathcal{T}(s)}\rho_{t:T-1}\,G_t$: **unbiased**, but **high (possibly infinite) variance**.
* **Weighted IS** — $V(s)=\dfrac{\sum_{t}\rho_{t:T-1}\,G_t}{\sum_{t}\rho_{t:T-1}}$: **biased** (bias $\to 0$), **much lower variance** — preferred in practice.
* **Off-policy MC control:** target $\pi$ greedy, behaviour $b$ exploratory (e.g. $\varepsilon$-soft). Weakness: it can only learn from an episode's **tail** (the suffix where $\pi$ and $b$ still agree) ⇒ slow.

## Where MC sits

| | **DP** | **MC** | **TD** |
| :-- | :--: | :--: | :--: |
| Model | needs | **free** | **free** |
| Bootstrap | yes | **no** | yes |
| Update from | full expected backup | **full episode return** | one-step sample |
| Tasks | any | **episodic only** | any |

**Punchline:** MC samples **complete returns** and averages them — no model, no bootstrap, so it's unbiased but must **wait for episodes to end** and suffers **high variance**. TD (next lecture) fixes both by bootstrapping a one-step estimate.
