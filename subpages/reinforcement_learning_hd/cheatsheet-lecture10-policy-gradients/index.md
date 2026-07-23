---
title: "Cheatsheet — Lecture 10: Policy Gradient Methods"
layout: default
noindex: true
math: true
tags:
  - reinforcement-learning
  - policy-gradient
  - reinforce
  - actor-critic
  - cheatsheet
---

# Cheatsheet — Lecture 10: Policy Gradient Methods

*Exam recap: formulas and key results only. Full explanations live in the [main notes]({{ '/subpages/reinforcement_learning_hd/' | relative_url }}#lecture-10-policy-gradient-methods).*

## Why parameterize the policy

Learn a policy $\pi(a\mid s,\theta)$ **directly** by gradient ascent on performance $J(\theta)$ — no $\arg\max$ over values. Three reasons:

* can represent **stochastic** optimal policies (partial observability, need to randomize);
* natural for **large / continuous** action spaces (no max over actions);
* the policy changes **smoothly** with $\theta$ ⇒ better convergence than a value-greedy policy that can jump; can approach determinism.

## Soft-max policy (discrete actions)

$$\pi(a\mid s,\theta)=\frac{e^{h(s,a,\theta)}}{\sum_b e^{h(s,b,\theta)}},$$

with **action preferences** $h(s,a,\theta)$ (e.g. linear $\theta^\top x(s,a)$). Unlike $\varepsilon$-greedy it moves smoothly, keeps every action possible, can tend to deterministic, and is **not tied to value estimates**.

## Objective + Policy Gradient Theorem

Episodic performance $J(\theta)=v_{\pi_\theta}(s_0)$. The **policy gradient theorem** avoids differentiating the state distribution:

$$\nabla J(\theta)\;\propto\;\sum_s \mu(s)\sum_a q_\pi(s,a)\,\nabla\pi(a\mid s,\theta).$$

Using the **log-likelihood trick** $\nabla\pi=\pi\,\nabla\log\pi$:

$$\boxed{\,\nabla J(\theta)\;\propto\;\mathbb{E}_{S\sim\mu,\,A\sim\pi}\bigl[\,q_\pi(S,A)\,\nabla\log\pi(A\mid S,\theta)\,\bigr]\,}$$

* Key win: the gradient is an **expectation under the on-policy distribution** — $\nabla\mu$ (how $\theta$ shifts state visitation) **drops out**, so it's estimable from samples.

## REINFORCE (Monte-Carlo policy gradient)

Replace $q_\pi(S_t,A_t)$ by the sampled return $G\_t$:

$$\boxed{\,\theta_{t+1}=\theta_t+\alpha\,\gamma^{t}\,G_t\,\nabla\ln\pi(A_t\mid S_t,\theta_t)\,}$$

* $\nabla\ln\pi$ = **score / eligibility vector** (direction that most raises the probability of the taken action). Soft-max linear: $\nabla\ln\pi(a\mid s,\theta)=x(s,a)-\sum_b\pi(b\mid s,\theta)\,x(s,b)$.
* **Unbiased** gradient (true SGD on $J$) but **high variance**, slow; the $\gamma^{t}$ discounts later-step updates.

## REINFORCE with baseline

Subtract a state-dependent baseline $b(S\_t)$ — **does not change** the expected gradient, **cuts variance**:

$$\theta_{t+1}=\theta_t+\alpha\,\gamma^{t}\bigl(G_t-b(S_t)\bigr)\nabla\ln\pi(A_t\mid S_t,\theta_t).$$

* Natural choice $b(S\_t)=\hat v(S\_t,w)$ (a learned value); then $G\_t-\hat v(S\_t,w)$ is a Monte-Carlo **advantage** estimate.

## Actor–Critic

Bootstrap the return with a **critic** $\hat v(S\_t,w)$ ⇒ the one-step advantage is the **TD error**:

$$\delta_t=R_{t+1}+\gamma\,\hat v(S_{t+1},w)-\hat v(S_t,w),$$

$$\boxed{\,\theta_{t+1}=\theta_t+\alpha^{\theta}\gamma^{t}\,\delta_t\,\nabla\ln\pi(A_t\mid S_t,\theta_t)\,},\qquad w\leftarrow w+\alpha^{w}\,\delta_t\,\nabla\hat v(S_t,w).$$

* **Actor** = policy ($\theta$); **critic** = value ($w$). Bootstrapping ⇒ **online, lower variance, slightly biased** (vs REINFORCE's unbiased-but-noisy MC). The critic serves as both **baseline and bootstrap**.
* **Continuous actions:** Gaussian policy $\pi(a\mid s,\theta)=\mathcal N\!\bigl(a;\,\mu(s,\theta),\,\sigma(s,\theta)^2\bigr)$ — learn mean (and spread) as functions of state; $\nabla\ln\pi$ is closed-form.

## Value-based vs policy-based

| | **Value-based** (Q-learning, Sarsa) | **Policy-based** (PG) |
| :-- | :-- | :-- |
| Learns | action values → greedy | the **policy** directly |
| Policy | deterministic ($\varepsilon$-greedy) | **stochastic, smooth** |
| Actions | discrete | discrete **+ continuous** |
| Convergence | can oscillate near argmax | smoother (to a **local** optimum) |

**Actor–Critic = both** — learn a value (critic) *and* a policy (actor).

## Context: deep-RL stability & AlphaGo

* **Deadly triad** — *bootstrapping* $+$ *function approximation* $+$ *off-policy* training can **diverge**. Mitigations: target networks, experience replay, gradient clipping. (On-policy PG sidesteps the off-policy leg.)
* **AlphaGo** — combines a **policy network** (supervised + RL policy gradient), a **value network** (critic), and **MCTS** search: learned policy/value guide the tree, search sharpens the policy.

**Punchline:** policy gradients optimize $\mathbb{E}[\,q_\pi\,\nabla\ln\pi\,]$; the whole family is *how you estimate the return weight* — full MC return (REINFORCE), MC minus a baseline, or a bootstrapped TD error (actor–critic) — trading variance for bias.
