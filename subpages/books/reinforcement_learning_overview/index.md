---
layout: default
title: "Reinforcement Learning: An Overview"
date: 2025-03-21
excerpt: Notes on the RL overview by Kevin Murphy, covering sequential decision making, canonical models, and high-level RL methods.
tags:
  - reinforcement-learning
  - machine-learning
---

**Table of Contents**
- TOC
{:toc}

# Chapter 1: Introduction

## Sequential Decision Making

**Reinforcement learning** (RL) is a class of methods for solving various kinds of sequential decision making tasks. The agent maintains an internal state $z_t$, which it passes to its **policy** $\pi$ to choose an action $a_t = \pi(z_t)$. The environment responds by sending back an observation $o_{t+1}$, which the agent uses to update its internal state using the state-update function $z_{t+1} = SU(z_t, a_t, o_{t+1})$.

To simplify things, we often assume that the environment is a Markovian process with hidden internal world state $w_t$, from which the observations $o_t$ are derived (this is called a POMDP). We often simplify further by assuming that the observation $o_t$ reveals the hidden environment state; in this case $s_t = o_t = w_t = z_t$ (this is called an MDP).

### Maximum Expected Utility Principle

The goal of the agent is to choose a policy $\pi$ so as to maximize the sum of expected rewards:

$$
V_\pi(s_0) = \mathbb{E}_{p(a_0, s_1, a_1, \ldots, a_T, s_T | s_0, \pi)} \left[ \sum_{t=0}^{T} R(s_t, a_t) | s_0 \right]
$$

where $s_0$ is the agent's initial state, $R(s_t, a_t)$ is the **reward function**, and $V_\pi(s_0)$ is the **value function** for policy $\pi$ evaluated at $s_0$. The expectation is with respect to

$$
p(a_0, s_1, a_1, \ldots, a_T, s_T | s_0, \pi) = \pi(a_0|s_0) p_{\text{env}}(o_1|a_0) \delta(s_1 = U(s_0, a_0, o_1)) \times \pi(a_1|s_1) p_{\text{env}}(o_2|a_1, o_1) \delta(s_2 = U(s_1, a_1, o_2)) \times \cdots
$$

where $p_{\text{env}}$ is the environment's distribution over observations (usually unknown).

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Optimal Policy)</span></p>

The **optimal policy** is defined as

$$
\pi^* = \arg\max_\pi \mathbb{E}_{p_0(s_0)} \left[ V_\pi(s_0) \right]
$$

Picking a policy to maximize the sum of expected rewards is an instance of the **maximum expected utility** principle.

</div>

### Episodic vs Continual Tasks

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Continual and Episodic Tasks)</span></p>

* A **continual task** is one where the agent can potentially interact with the environment forever. The value function is defined using the **average reward**.
* An **episodic task** terminates once the system enters a **terminal state** (or **absorbing state**), which transitions to itself with $0$ reward. After entering a terminal state, a new **episode** begins from a new initial world state $z_0 \sim p_0$.
* If the trajectory length $T$ is fixed and known, it is called a **finite horizon problem**.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Return and Discount Factor)</span></p>

The **return** for a state at time $t$ is the sum of expected rewards obtained going forwards, where each reward is multiplied by a **discount factor** $\gamma \in [0,1]$:

$$
G_t \triangleq r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots + \gamma^{T-t-1} r_{T-1} = \sum_{k=0}^{T-t-1} \gamma^k r_{t+k}
$$

where $r_t = R(s_t, a_t)$ is the reward, and $G_t$ is the **reward-to-go**. For episodic tasks that terminate at time $T$, we define $G_t = 0$ for $t \ge T$. The return satisfies the recursive relationship:

$$
G_t = r_t + \gamma G_{t+1}
$$

The value function is defined as the expected reward-to-go:

$$
V_\pi(s_t) = \mathbb{E}\left[G_t | \pi\right]
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Role of the Discount Factor)</span></p>

The discount factor $\gamma$ plays two roles:

1. It ensures the return is finite even if $T = \infty$ (infinite horizon), provided $\gamma < 1$ and the rewards $r_t$ are bounded.
2. It puts more weight on short-term rewards, encouraging the agent to achieve its goals more quickly.

If $\gamma = 0$, the agent is completely **myopic** and only maximizes its immediate reward. If $\gamma = 1 - \frac{1}{T}$, the agent expects to live on the order of $T$ steps. For finite horizon problems where $T$ is known, we can set $\gamma = 1$.

</div>

### Universal Model

A generic representation for sequential decision making problems assumes the environment can be modeled by a **controlled Markov process** with hidden state $w_t$, updated at each step in response to the agent's action $a_t$:

$$
w_{t+1} = M(w_t, a_t, \epsilon_t^w)
$$

where $M$ is the environment's state transition function (usually unknown to the agent) and $\epsilon_t^w$ is random system noise. The agent does not see $w_t$ directly, but instead sees a potentially noisy and/or partial observation $o_{t+1} = O(w_{t+1}, \epsilon_{t+1}^o)$ at each step, where $\epsilon_{t+1}^o$ is random observation noise.

Any given observation could correspond to many different locations in the world --- this is called **perceptual aliasing**.

The agent needs to maintain an internal **belief state** about the world, denoted by $z$. This gets updated using the state update function:

$$
z_{t+1} = SU(z_t, a_t, o_{t+1})
$$

We can break the state-update function into two parts. First the agent predicts its own next state, $z_{t+1|t} = P(z_t, a_t)$, using a **prediction function** $P$, and then it updates this prediction given the observation using an **update function** $U$, to give $z_{t+1} = U(z_{t+1|t}, o_{t+1})$. Thus:

$$
z_{t+1} = U(P(z_t, a_t), o_{t+1})
$$

If observations are high-dimensional (e.g., images), the agent may encode them into a low-dimensional embedding $e_{t+1} = E(o_{t+1})$, so the state update becomes $z_{t+1} = U(P(z_t, a_t), E(o_{t+1}))$.

The agent needs to learn the action policy $\pi_t(z_t) = \pi(z_t; \boldsymbol{\theta}_t)$. We can update the policy parameters using a **learning algorithm**:

$$
\boldsymbol{\theta}_t = \mathcal{A}(o_{1:t}, a_{1:t}, r_{1:t}) = \mathcal{A}(\boldsymbol{\theta}_{t-1}, a_t, z_t, r_t)
$$

In general, there are three interacting stochastic processes to deal with: the environment's states $w_t$; the agent's internal states $z_t$ (beliefs about the environment); and the agent's policy parameters $\boldsymbol{\theta}_t$.

## Canonical Models

### Partially Observed MDPs

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(POMDP)</span></p>

A **partially observable Markov decision process** (POMDP, pronounced "pom-dee-pee") is the general model where the environment's dynamics are represented by a stochastic transition function rather than a deterministic function with noise as input. The stochastic transition function is:

$$
p(w_{t+1}|w_t, a_t) = \mathbb{E}_{\epsilon_t^w}\left[\mathbb{I}(w_{t+1} = W(w_t, a_t, \epsilon_t^w))\right]
$$

The stochastic observation function is:

$$
p(o_{t+1}|w_{t+1}) = \mathbb{E}_{\epsilon_{t+1}^o}\left[\mathbb{I}(o_{t+1} = O(w_{t+1}, \epsilon_{t+1}^o))\right]
$$

</div>

If the world model (both $p(o|w)$ and $p(w'|w,a)$) is known, then we can --- in principle --- solve for the optimal policy. The method requires that the agent's internal state correspond to the **belief state** $s_t = \boldsymbol{b}_t = p(w_t | \boldsymbol{h}_t)$, where $\boldsymbol{h}_t = (o_{1:t}, a_{1:t-1})$ is the observation history. The belief state can be updated recursively using Bayes' rule and forms a sufficient statistic for the optimal policy. Unfortunately, computing the belief state and the resulting optimal policy is wildly intractable.

### Markov Decision Processes (MDPs)

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Markov Decision Process)</span></p>

A **Markov decision process** (MDP) is a special case of a POMDP in which the environment states are observed, so $w_t = o_t = s_t$. An MDP is defined in terms of the state transition matrix induced by the world model:

$$
p_S(s_{t+1}|s_t, a_t) = \mathbb{E}_{\epsilon_t^s}\left[\mathbb{I}(s_{t+1} = W(s_t, a_t, \epsilon_t^s))\right]
$$

In lieu of an observation model, the environment sends out a reward signal, sampled from $p_R(r_t | s_t, a_t, s_{t+1})$. The expected reward is:

$$
R(s_t, a_t, s_{t+1}) = \sum_r r \; p_R(r|s_t, a_t, s_{t+1})
$$

$$
R(s_t, a_t) = \sum_{s_{t+1}} p_S(s_{t+1}|s_t, a_t) R(s_t, a_t, s_{t+1})
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Control Theory Terminology)</span></p>

The field of control theory uses slightly different terminology: the environment is the **plant**, the agent is the **controller**, states are denoted by $\boldsymbol{x}_t \in \mathcal{X} \subseteq \mathbb{R}^D$, actions by $\boldsymbol{u}_t \in \mathcal{U} \subseteq \mathbb{R}^K$, and rewards are replaced by costs $c_t \in \mathbb{R}$.

</div>

When both state and action sets are finite, we can represent functions as lookup tables; this is known as a **tabular representation**. We can represent the MDP as a **finite state machine**, where nodes correspond to states and edges to actions and resulting rewards/next states.

Given a stochastic policy $\pi(a_t|s_t)$, each step is a **transition** $(s_t, a_t, r_t, s_{t+1})$. Under policy $\pi$, the probability of generating a **trajectory** of length $T$, $\boldsymbol{\tau} = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots, s_T)$, is:

$$
p(\boldsymbol{\tau}) = p_0(s_0) \prod_{t=0}^{T-1} \pi(a_t|s_t) p_S(s_{t+1}|s_t, a_t) p_R(r_t|s_t, a_t, s_{t+1})
$$

### Goal-Conditioned MDPs

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Goal-Conditioned MDP)</span></p>

A **goal-conditioned MDP** is one in which the reward is defined as $R(s,a|g) = 1$ iff the goal state is achieved, i.e., $R(s,a|s) = \mathbb{I}(s = g)$. We can also define a dense reward signal using a state abstraction function $\phi$, by defining $R(s,a|g) = \text{sim}(s,g)$, where $\text{sim}$ is some similarity metric. For example, using cosine similarity:

$$
\text{sim}(s,g) = \frac{\phi(s)^\top \psi(g)}{\|\phi(s)\| \; \|\psi(g)\|}
$$

where $\phi(s)$ is an embedding of the state and $\psi(g)$ is an embedding of the goal.

A goal-conditioned policy $\pi(a|s,g)$ is sometimes called a **universal policy**.

</div>

### Contextual MDPs

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Contextual MDP)</span></p>

A **Contextual MDP** is an MDP where the dynamics and rewards depend on a hidden static parameter referred to as the **context**. A simple example is a video game where each level is **procedurally generated**, meaning it is randomly generated each time the agent starts a new episode. The agent must solve a sequence of related MDPs drawn from a common distribution, requiring the agent to **generalize** across multiple MDPs rather than overfitting to a specific environment.

A contextual MDP is a special kind of POMDP where the hidden variable corresponds to the unknown parameters of the model. This is also called an **epistemic POMDP**.

</div>

### Contextual Bandits

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Contextual Bandit)</span></p>

A **contextual bandit** is a special case of a POMDP where the world state transition function is independent of the action and the previous state, i.e., $p(w_t|w_{t-1}, a_t) = p(w_t)$. The world states are called "contexts" and are observable ($o_t = w_t$). Since the world state distribution is independent of the agent's actions, the agent has no effect on the external environment. However, its actions do affect the rewards.

A special case with no context ($s_t$ is a fixed constant) and a finite number of possible actions $\mathcal{A} = \lbrace a_1, \ldots, a_K \rbrace$ is called a **multi-armed bandit**. The reward model has the form $R(a) = f(\boldsymbol{w}_a)$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Online Advertising)</span></p>

In an **online advertising system**, the state $s_t$ represents features of the web page the user is currently looking at, and the action $a_t$ is the identity of the ad to show. The reward function $R(s_t, a_t)$ depends on the relevance of the ad to the page, so the problem is contextual. The goal is to maximize the expected number of times people click on ads, known as the **click through rate** (CTR).

</div>

### Belief State MDPs

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Belief State MDP)</span></p>

A **belief state MDP** is a kind of MDP where the state represents a probability distribution, known as a **belief state** or **information state**, which is updated by the agent as it receives information from the environment. The agent approximates the unknown reward by a function $R(o,a) = f(o, a; \boldsymbol{w})$. Let $\boldsymbol{b}_t = p(\boldsymbol{w}|\boldsymbol{h}_t)$ denote the posterior over the unknown parameters, where $\boldsymbol{h}_t = \lbrace o_{1:t}, a_{1:t}, r_{1:t} \rbrace$ is the history. This belief state can be updated deterministically using Bayes' rule:

$$
\boldsymbol{b}_{t+1} = \text{BayesRule}(\boldsymbol{b}_t, o_{t+1}, a_{t+1}, r_{t+1})
$$

The transition dynamics are deterministic:

$$
p(\boldsymbol{b}_{t+1}|\boldsymbol{b}_t, o_{t+1}, a_{t+1}, r_{t+1}) = \mathbb{I}(\boldsymbol{b}_{t+1} = \text{BayesRule}(\boldsymbol{b}_t, o_{t+1}, a_{t+1}, r_{t+1}))
$$

and the reward function is:

$$
p(r_t|o_t, a_t, \boldsymbol{b}_t) = \int p_R(r_t|o_t, a_t; \boldsymbol{w}) p(\boldsymbol{w}|\boldsymbol{b}_t) d\boldsymbol{w}
$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Bernoulli Bandit)</span></p>

Consider a context-free **Bernoulli bandit**, where $p_R(r|a) = \text{Ber}(r|\mu_a)$, and $\mu_a = p_R(r=1|a) = R(a)$ is the expected reward. The only unknown parameters are $\boldsymbol{w} = \mu_{1:A}$. With a factored Beta prior $p_0(\boldsymbol{w}) = \prod_a \text{Beta}(\mu_a | \alpha_0^a, \beta_0^a)$, the posterior in closed form is:

$$
p(\boldsymbol{w}|\mathcal{D}_t) = \prod_a \text{Beta}(\mu_a | \alpha_0^a + N_t^0(a), \beta_0^a + N_t^1(a))
$$

where $N_t^r(a) = \sum_{i=1}^{t-1} \mathbb{I}(a_i = a, r_i = r)$ counts how many times action $a$ yielded reward $r$.

</div>

We can use a similar method for a **Gaussian bandit** ($p_R(r|a) = \mathcal{N}(r|\mu_a, \sigma_a^2)$), a **linear regression bandit** ($p_R(r|s,a;\boldsymbol{w}) = \mathcal{N}(r|\phi(s,a)^\top \boldsymbol{w}, \sigma^2)$) with Bayesian linear regression in closed form, a **logistic regression bandit** requiring approximate Bayesian methods, or a **neural bandit** ($p_R(r|s,a;\boldsymbol{w}) = \mathcal{N}(r|f(s,a;\boldsymbol{w}))$) where posterior inference is equivalent to inference in Bayesian neural networks.

Once the belief state is computed, we can derive a policy with optimal regret using methods like UCB or Thompson sampling.

### Optimization Problems as Decision Problems

The bandit problem is an example where the agent interacts with the world to collect information but does not otherwise affect the environment. The agent's internal belief state changes over time, but the environment state does not. Such problems commonly arise when trying to optimize a fixed but unknown function $R$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Best-Arm Identification)</span></p>

In some cases, the goal is to determine the best arm given a fixed budget of $T$ trials; this is **best-arm identification**. Formally, this corresponds to optimizing the **final reward** criterion:

$$
V_{\pi, \pi_T} = \mathbb{E}_{p(a_{1:T}, r_{1:T}|s_0, \pi)}\left[R(\hat{a})\right]
$$

where $\hat{a} = \pi_T(a_{1:T}, r_{1:T})$ is the estimated optimal arm as computed by the **terminal policy** $\pi_T$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Bayesian Optimization)</span></p>

**Bayesian optimization** (BayesOpt) is a gradient-free approach to optimizing expensive blackbox functions. The goal is to find $\boldsymbol{w}^* = \arg\max_{\boldsymbol{w}} R(\boldsymbol{w})$ for some unknown function $R$, where $\boldsymbol{w} \in \mathbb{R}^N$, using as few function evaluations as possible. The agent's state $s_t$ is a belief state over the unknown function, $s_t = p(R|\boldsymbol{h}_t)$. A common way to represent this distribution is to use Gaussian processes, combined with heuristics like expected improvement, knowledge gradient, or Thompson sampling.

</div>

**Active learning** is similar to BayesOpt, but instead of finding the point at which $R$ is largest (i.e., $\boldsymbol{w}^*$), we try to learn the whole function $R$ by querying it at different points. The optimal strategy requires maintaining a belief state over the unknown function.

**Stochastic Gradient Descent (SGD)** can be interpreted as a sequential decision making process. The action space consists of querying $R$ at locations $\boldsymbol{a}_t = \boldsymbol{w}_t$, observing the function value $r_t = R(\boldsymbol{w}_t)$ and the gradient $\boldsymbol{g}_t = \nabla_{\boldsymbol{w}} R(\boldsymbol{w})|_{\boldsymbol{w}_t}$. The update rule for vanilla SGD is $\boldsymbol{w}_{t+1} = \boldsymbol{w}_t + \alpha_t \boldsymbol{g}_t$, where the stepsize $\alpha_t$ is chosen by the policy. The terminal policy is $\pi(s_T) = \boldsymbol{w}_T$.

## Reinforcement Learning: A High-Level Summary

We can categorize RL methods along multiple dimensions:

* **What does the agent learn?** Options include the value function, the policy, the model, or some combination.
* **How does the agent represent its unknown functions?** The two main choices are non-parametric or **tabular representations**, or parametric representations based on function approximation. If based on neural networks with many layers, this is called **deep RL**.
* **How are the actions selected?** **On-policy** methods require actions selected by the agent's current policy; **off-policy** methods allow actions selected by any kind of policy, including human demonstrations.

| Approach | Method | Functions Learned | On/Off |
| --- | --- | --- | --- |
| Value-based | SARSA | $Q(s,a)$ | On |
| Value-based | Q-learning | $Q(s,a)$ | Off |
| Policy-based | REINFORCE | $\pi(a\|s)$ | On |
| Policy-based | A2C | $\pi(a\|s), V(s)$ | On |
| Policy-based | TRPO/PPO | $\pi(a\|s), \text{Adv}(s,a)$ | On |
| Policy-based | DDPG | $a = \pi(s), Q(s,a)$ | Off |
| Policy-based | Soft actor-critic | $\pi(a\|s), Q(s,a)$ | Off |
| Model-based | MBRL | $p(s'\|s,a)$ | Off |

### Value-Based RL

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Value Function and Bellman Equation)</span></p>

The value function for policy $\pi$ is:

$$
V_\pi(s) \triangleq \mathbb{E}_\pi\left[G_0 | s_0 = s\right] = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t r_t | s_0 = s\right]
$$

The value function for the optimal policy $\pi^*$ satisfies **Bellman's equation**:

$$
V^*(s) = \max_a R(s,a) + \gamma \mathbb{E}_{p_S(s'|s,a)}\left[V^*(s')\right]
$$

This follows from the principle of **dynamic programming**, which computes the optimal solution to a problem by combining the optimal solutions of subproblems.

</div>

This can be used to derive the **Temporal Difference** (TD) learning rule:

$$
V(s) \leftarrow V(s) + \eta\left[r + \gamma V(s') - V(s)\right]
$$

where $s' \sim p_S(\cdot|s,a)$ is the next state and $r = R(s,a)$ is the observed reward.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Q Function)</span></p>

The **Q function** assigns a value to a state-action pair:

$$
Q_\pi(s,a) \triangleq \mathbb{E}_\pi\left[G_0 | s_0 = s, a_0 = a\right] = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t r_t | s_0 = s, a_0 = a\right]
$$

The Q function for the optimal policy satisfies the modified Bellman equation:

$$
Q^*(s,a) = R(s,a) + \gamma \mathbb{E}_{p_S(s'|s,a)}\left[\max_{a'} Q^*(s', a')\right]
$$

</div>

This gives rise to the **Q learning** TD update rule:

$$
Q(s,a) \leftarrow r + \gamma \max_{a'} Q(s', a') - Q(s,a)
$$

The action at each step is chosen from the implicit policy $a = \arg\max_{a'} Q(s, a')$.

### Policy-Based RL

In policy-based methods, we try to directly maximize $J(\pi_{\boldsymbol{\theta}}) = \mathbb{E}_{p_0(s_0)}\left[V_{\pi}(s_0)\right]$ with respect to the parameter $\boldsymbol{\theta}$; this is called **policy search**. If $J(\pi_{\boldsymbol{\theta}})$ is differentiable with respect to $\boldsymbol{\theta}$, we can use stochastic gradient ascent, known as **policy gradient**.

Policy gradient methods provably converge to a local optimum for many common policy classes, whereas Q-learning may diverge with approximation. They can also handle continuous action spaces easily, since they do not need $\arg\max_a Q(s,a)$. However, the score function estimator for $\nabla_{\boldsymbol{\theta}} J(\pi_{\boldsymbol{\theta}})$ can have very high variance.

One way to reduce variance is to learn an approximate value function $V_{\boldsymbol{w}}(s)$ and use it as a baseline. Alternatively, we can learn an advantage function $A_{\boldsymbol{w}}(s,a)$. These variants are called **actor-critic** methods, where the actor refers to the policy $\pi_{\boldsymbol{\theta}}$ and the critic refers to $V_{\boldsymbol{w}}$ or $A_{\boldsymbol{w}}$.

### Model-Based RL

Value-based and policy search methods can be very **sample inefficient**, requiring many interactions with the environment. In **model-based RL**, we first learn the MDP (the transition $p_S(s'|s,a)$ and reward $R(s,a)$ functions), then compute the policy using approximate dynamic programming or lookahead search. In practice, we often interleave the model learning and planning phases, using the partially learned policy to decide what data to collect.

### State Uncertainty (Partial Observability)

In many problems, the observation $o_t$ only gives partial information about the underlying state (e.g., a rodent navigating a maze). This is called **partial observability**. A policy of the form $a_t = \pi(o_t)$ is suboptimal; instead we need $a_t = \pi(\boldsymbol{h}_t)$, where $\boldsymbol{h}_t = (a_1, o_1, \ldots, a_{t-1}, o_t)$ is the entire past history.

**Optimal solution:** If we know the true latent structure ($p(o|z)$ and $p(z'|z,a)$), we can compute a belief state $\boldsymbol{b}_t = p(w_t|\boldsymbol{h}_t)$ and use POMDP solution methods. However, this is computationally very difficult.

**Predictive state representation (PSR):** We can marginalize out the POMDP latent state $w_t$ to derive a prediction over the next observable state, $p(\boldsymbol{o}_{t+1}|\boldsymbol{h}_t, \boldsymbol{a}_t)$, providing a learning target without explicitly invoking latent state.

**Finite observation history:** Define the state as the last $k$ observations, $s_t = \boldsymbol{h}_{t-k:t}$; when observations are images, this is called **frame stacking**. This cannot capture long-range dependencies.

**Stateful (recurrent) policies:** Represent the policy by an RNN whose hidden state $w_t$ implicitly summarizes the past observations $\boldsymbol{h}_t$, and can be used in lieu of the state $s_t$ in any standard RL algorithm (e.g., **R2D2**). RNN policies are widely used but typically will not plan to perform information-gathering actions, since there is no explicit notion of belief state or uncertainty.

### Model Uncertainty (Exploration-Exploitation Tradeoff)

In RL problems, the underlying transition and reward models are typically unknown. We need to explore the environment to collect enough data to figure out what to do. This is the **exploration-exploitation tradeoff**.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Exploration Strategies)</span></p>

* **Greedy policy:** $a_t = \arg\max_a Q(s,a)$ --- exploits current knowledge without exploration.
* **$\epsilon$-greedy policy:** Pick the greedy action with probability $1-\epsilon$ and a random action with probability $\epsilon$. Suboptimal since it explores every action with at least probability $\epsilon/|\mathcal{A}|$, but can be improved by annealing $\epsilon$ to $0$.
* **$\epsilon z$-greedy policy:** With probability $1-\epsilon$ exploit, and with probability $\epsilon$ repeat the sampled action for $n \sim z()$ steps. This helps escape local minima.
* **Boltzmann exploration:** Uses the policy $\pi_\tau(a|s) = \frac{\exp(\hat{R}_t(s_t, a)/\tau)}{\sum_{a'} \exp(\hat{R}_t(s_t, a')/\tau)}$, where $\tau > 0$ is a temperature parameter. As $\tau \to 0$, this becomes greedy; higher $\tau$ encourages more exploration.
* **Exploration bonus:** Add an **intrinsic reward** $R^b_t(s,a)$ (large if state-action is rarely visited) to the regular reward, biasing behavior toward information-gathering.

</div>

### Reward Functions

#### The Reward Hypothesis

The **reward hypothesis** states that "all of what we mean by goals and purposes can be well thought of as maximization of the expected value of the cumulative sum of a received scalar signal (reward)." Whether this is true depends on what one means by "goals and purposes."

#### Non-Markovian Rewards

Most of the literature assumes the reward can be defined in terms of the current state and action, $R(s,a)$, or the most recent state transition, $R(s,a,s')$. In general, the reward function may need to be non-Markovian. For example, in the **MineDojo** paper, a reward model of the form $R(o(t-K:t), g)$ was pre-trained, where $o(t-K:t)$ are the last $K$ frames and $g$ is the goal. This model, known as **MineCLIP**, was trained using contrastive learning on a large corpus of video-text pairs.

#### Reward Hacking

The reward function may be misspecified. An optimal agent may maximize the reward in unintended ways. In the **AI alignment** community, this is the **paperclip maximizer problem** (due to Nick Bostrom). This is an instance of **reward hacking**.

#### Sparse Reward

Many problems suffer from **sparse reward**, where $R(s,a) = 0$ for almost all states and actions. This requires **deep exploration** to find the rewarding states.

#### Reward Shaping

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Potential-Based Reward Shaping)</span></p>

In **reward shaping**, we define a new reward function $r' = r + F$ to combat sparse reward. If the shaping function has the form

$$
F(s,a,s') = \gamma \Phi(s') - \Phi(s)
$$

where $\Phi: \mathcal{S} \to \mathbb{R}$ is a **potential function**, then the sum of shaped rewards will match the sum of original rewards plus a constant. This is called **Potential-Based Reward Shaping**. In the tabular case, this is equivalent to initializing the value function to $V(s) = \Phi(s)$.

More generally, a potential of the form $F(s,a,s',a') = \gamma \Phi(s',a') - \Phi(s,a)$ is also valid and more expressive. We can also introduce a reward shaping function $z$ to down-weight or up-weight the shaping function:

$$
r'(s,a) = r(s,a) + z_\phi(s,a) F(s,a)
$$

</div>

#### Intrinsic Reward

**Intrinsic reward** is a set of methods for encouraging agent behavior without any external reward signal, for example, encouraging agents to explore their environment just to "figure things out."

### Best Practices for Experimental Work in RL

Implementing RL algorithms is much trickier than supervised learning or generative methods. RL experiments can have very high variance, making it hard to draw valid conclusions. Recommended practices include reporting the **interquartile mean** (IQM) of the performance metric (the mean of samples between the 0.25 and 0.75 percentiles). We can estimate uncertainty in this estimate using bootstrap resampling or a Gaussian approximation (standard error of the mean: $\frac{\hat{\sigma}}{\sqrt{n}}$).

# Chapter 2: Value-Based RL

## Basic Concepts

### Value Functions

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Value Functions)</span></p>

Let $\pi$ be a given policy. The **state-value function** (or **value function**) is:

$$
V_\pi(s) \triangleq \mathbb{E}_\pi\left[G_0 | s_0 = s\right] = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t r_t | s_0 = s\right]
$$

This is the expected return obtained if we start in state $s$ and follow $\pi$ to choose actions in a continuing task ($T = \infty$).

The **state-action value function** (or **Q-function**) is:

$$
Q_\pi(s,a) \triangleq \mathbb{E}_\pi\left[G_0 | s_0 = s, a_0 = a\right] = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t r_t | s_0 = s, a_0 = a\right]
$$

This represents the expected return obtained if we start by taking action $a$ in state $s$, and then follow $\pi$ thereafter.

The **advantage function** is:

$$
\text{Adv}_\pi(s,a) \triangleq Q_\pi(s,a) - V_\pi(s)
$$

This tells us the benefit of picking action $a$ in state $s$ then switching to policy $\pi$, relative to the baseline return of always following $\pi$. Note that $\text{Adv}_\pi(s,a)$ can be both positive and negative, and $\mathbb{E}_{\pi(a|s)}\left[\text{Adv}_\pi(s,a)\right] = 0$ due to $V_\pi(s) = \mathbb{E}_{\pi(a|s)}\left[Q_\pi(s,a)\right]$.

</div>

### Bellman's Equations

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Bellman's Optimality Equations)</span></p>

Suppose $\pi^*$ is a policy such that $V_{\pi^*} \ge V_\pi$ for all $s \in \mathcal{S}$ and all policies $\pi$; then it is an **optimal policy**. There can be multiple optimal policies, but their value functions must be the same, denoted $V^*$ and $Q^*$. Any finite MDP must have at least one deterministic optimal policy.

The optimal value functions satisfy **Bellman's optimality equations**:

$$
V^*(s) = \max_a R(s,a) + \gamma \mathbb{E}_{p_S(s'|s,a)}\left[V^*(s')\right]
$$

$$
Q^*(s,a) = R(s,a) + \gamma \mathbb{E}_{p_S(s'|s,a)}\left[\max_{a'} Q^*(s',a')\right]
$$

The optimal value functions are the only solutions that satisfy these equations. Although the value function is defined as the expectation of a sum of infinitely many rewards, it can be characterized by a recursive equation involving only one-step transition and reward models.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Bellman Operator and Backup)</span></p>

Given a value function ($V$ or $Q$), the discrepancy between the right- and left-hand sides of the Bellman optimality equations is called the **Bellman error** or **Bellman residual**. We can define the **Bellman operator** $\mathcal{B}$ given an MDP $M = (R, T)$ and policy $\pi$ as a function that takes a value function $V$ and derives a new value function $V'$:

$$
V'(s) = \mathcal{B}_M^\pi V(s) \triangleq \mathbb{E}_{\pi(a|s)}\left[R(s,a) + \gamma \mathbb{E}_{T(s'|s,a)}\left[V(s')\right]\right]
$$

This reduces the Bellman error. Applying the Bellman operator to a state is called a **Bellman backup**. If we iterate this process, we will converge to the optimal value function $V_*$.

</div>

Given the optimal value function, we can derive an optimal policy using:

$$
\pi^*(s) = \arg\max_a Q^*(s,a) = \arg\max_a \left[R(s,a) + \gamma \mathbb{E}_{p_S(s'|s,a)}\left[V^*(s')\right]\right]
$$

The maximizing action is called the **greedy action** with respect to the value functions $Q^*$ or $V^*$.

The problem of solving for $V^*$, $Q^*$ or $\pi^*$ is called **policy optimization**. Solving for $V_\pi$ or $Q_\pi$ for a given policy $\pi$ is called **policy evaluation**. For policy evaluation, we have similar Bellman equations which simply replace $\max_a\lbrace\cdot\rbrace$ with $\mathbb{E}_{\pi(a|s)}[\cdot]$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(1D Grid World)</span></p>

Consider a 1D **grid world** with 5 possible states. States $S_{T1}$ and $S_{T2}$ are absorbing states. There are 2 actions ($\uparrow$ and $\downarrow$). The reward function is zero everywhere except at the goal state $S_{T2}$, which gives a reward of 1.

* $\gamma = 0$: $Q^*(s_3, \downarrow) = 1.0$ and $Q^*(s,a) = 0$ for all other state-action pairs. This is completely myopic and ignores the future.
* $\gamma = 1$: $Q^*(s,a) = 1$ for all state-action pairs, since the agent can always reach the goal eventually. This is infinitely far-sighted but gives no short-term guidance.
* $\gamma = 0.9$: Reflects a preference for near-term rewards while also taking future reward into account. This encourages the agent to seek the shortest path to the goal.

</div>

## Solving for the Optimal Policy in a Known World Model

When the MDP model is known, computing the optimal value function is the **prediction problem** and computing the optimal policy is the **control problem**. The algorithms are based on **dynamic programming** (DP) and **linear programming** (LP).

Exact calculation of optimal policies often depends polynomially on the sizes of $\mathcal{S}$ and $\mathcal{A}$, and is intractable when the state space is a Cartesian product of several finite sets. This is the **curse of dimensionality**. Approximations are typically needed, leading to **approximate dynamic programming** (ADP) and **approximate linear programming** (ALP).

### Value Iteration

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Value Iteration)</span></p>

**Value iteration** (VI) is a popular DP method for solving an MDP. Starting from an initial value function estimate $V_0$, the algorithm iteratively updates:

$$
V_{k+1}(s) = \max_a \left[R(s,a) + \gamma \sum_{s'} p(s'|s,a) V_k(s')\right]
$$

The update rule (a **Bellman backup**) is exactly the right-hand side of the Bellman optimality equation with the unknown $V^*$ replaced by the current estimate $V_k$. A fundamental property is that the update is a **contraction**:

$$
\max_s |V_{k+1}(s) - V^*(s)| \le \gamma \max_s |V_k(s) - V^*(s)|
$$

Every iteration reduces the maximum value function error by a constant factor. $V_k$ will converge to $V^*$, after which an optimal policy can be extracted using the greedy action.

</div>

### Real-Time Dynamic Programming (RTDP)

In value iteration, we compute $V^*(s)$ and $\pi^*(s)$ for all possible states $s$ at each iteration. However, for some problems we may only be interested in the value for certain starting states (e.g., **shortest path problems**).

**Real-time dynamic programming** (RTDP) efficiently computes an **optimal partial policy**, which only specifies what to do for the reachable states. At each step, it performs a Bellman backup for the current state, picks an action (often with some exploration), reaches a next state, and repeats. This is a form of **asynchronous value iteration**, focusing computational effort on parts of the state space more likely to be reachable from the current state.

### Policy Iteration

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Policy Iteration)</span></p>

**Policy iteration** (PI) is an iterative algorithm that searches in the space of deterministic policies until converging to an optimal policy. Each iteration consists of two steps: **policy evaluation** and **policy improvement**.

**Policy Evaluation:** Compute the value function for the current policy $\pi$. Let $\boldsymbol{v}(s) = V_\pi(s)$, $r(s) = \sum_a \pi(a|s)R(s,a)$, and $\mathbf{T}(s'|s) = \sum_a \pi(a|s)p(s'|s,a)$. Bellman's equation in matrix-vector form is:

$$
\boldsymbol{v} = \boldsymbol{r} + \gamma \mathbf{T} \boldsymbol{v}
$$

This can be solved by matrix inversion: $\boldsymbol{v} = (\mathbf{I} - \gamma \mathbf{T})^{-1} \boldsymbol{r}$, or iteratively by computing $\boldsymbol{v}_{t+1} = \boldsymbol{r} + \gamma \mathbf{T} \boldsymbol{v}_t$.

**Policy Improvement:** Compute a deterministic policy $\pi'$ that acts greedily with respect to $V_\pi$:

$$
\pi'(s) = \arg\max_a \lbrace R(s,a) + \gamma \mathbb{E}\left[V_\pi(s')\right] \rbrace
$$

The **policy improvement theorem** guarantees that $V_{\pi'} \ge V_\pi$.

The algorithm alternates between evaluation ($E$) and improvement ($I$):

$$
\pi_0 \xrightarrow{E} V_{\pi_0} \xrightarrow{I} \pi_1 \xrightarrow{E} V_{\pi_1} \cdots \xrightarrow{I} \pi^* \xrightarrow{E} V^*
$$

Since there are at most $|\mathcal{A}|^{|\mathcal{S}|}$ deterministic policies, and every iteration strictly improves the policy, the algorithm must converge after finite iterations.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(PI vs VI)</span></p>

In PI, we alternate between policy evaluation (which involves multiple iterations until convergence of $V_\pi$) and policy improvement. In VI, we alternate between one iteration of policy evaluation followed by one iteration of policy improvement (the "max" operator in the update rule). We are in fact free to intermix any number of these steps in any order. The process will converge once the policy is greedy with respect to its own value function.

Note that policy evaluation computes $V_\pi$ whereas value iteration computes $V^*$. In a **backup diagram**, the root node represents any state $s$, nodes at the next level represent state-action combinations, and nodes at the leaves represent the set of possible resulting next states. In PE, we average over all actions according to the policy, whereas in VI, we take the maximum over all actions.

</div>

## Value Function Learning Using Samples from the World Model

In the rest of this chapter, the agent only has access to samples from the environment, $(s', r) \sim p(s', r|s, a)$. We show how to use these samples to estimate the optimal value function and Q-function, even without explicitly knowing the MDP dynamics. This is "learning" as opposed to "planning."

### Monte Carlo Estimation

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Monte Carlo Estimation)</span></p>

Recall that $V_\pi(s) = \mathbb{E}\left[G_t | s_t = s\right]$ is the sum of expected (discounted) returns from state $s$ under policy $\pi$. **Monte Carlo estimation** estimates this by rolling out the policy and computing the average sum of discounted rewards. The update rule is:

$$
V(s_t) \leftarrow V(s_t) + \eta\left[G_t - V(s_t)\right]
$$

where $\eta$ is the learning rate and $G_t$ is the sampled return.

We can use MC estimation of $Q$, together with policy iteration, to learn an optimal policy. At iteration $k$, compute $\pi_{k+1}(s) = \arg\max_a Q_k(s,a)$, where $Q_k$ is approximated using MC estimation. This is called **Monte Carlo control**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(MC Convergence)</span></p>

To ensure MC control converges to the optimal policy, every (state, action) pair must be visited. Since this is an on-policy algorithm, the resulting method converges to the optimal $\epsilon$-soft policy, as opposed to the optimal policy. One can use importance sampling to estimate the value function for the optimal policy even if actions are chosen according to the $\epsilon$-greedy policy, or just gradually reduce $\epsilon$.

</div>

### Temporal Difference (TD) Learning

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(TD Learning)</span></p>

The MC method has very high variance (returns are sums of many random rewards) and is limited to episodic tasks (must unroll to the end of the episode). **Temporal difference** (TD) learning is a more efficient technique that incrementally reduces the Bellman error based on transitions instead of full trajectories.

Given a state transition $(s_t, a_t, r_t, s_{t+1})$ where $a_t \sim \pi(s_t)$, we change the estimate $V(s_t)$ so that it moves towards the **target value** $y_t = r_t + \gamma V(s_{t+1}) \approx G_{t:t+1}$:

$$
V(s_t) \leftarrow V(s_t) + \eta\left[\underbrace{r_t + \gamma V(s_{t+1}) - V(s_t)}_{\delta_t}\right]
$$

The term $\delta_t = y_t - V(s_t)$ is the **TD error**. A more general form for parametric value function representations is:

$$
\boldsymbol{w} \leftarrow \boldsymbol{w} + \eta\left[r_t + \gamma V_{\boldsymbol{w}}(s_{t+1}) - V_{\boldsymbol{w}}(s_t)\right] \nabla_{\boldsymbol{w}} V_{\boldsymbol{w}}(s_t)
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(TD as Semi-Gradient and Bootstrapping)</span></p>

TD learning in the tabular case converges to the correct value function under proper conditions. However, it may diverge when using nonlinear function approximators. The reason is that the update is a "**semi-gradient**": we only take the gradient with respect to the value function, $\nabla_{\boldsymbol{w}} V(s_t, \boldsymbol{w}_t)$, treating the target $U_t$ as constant.

The potential divergence arises because the update does not correspond to a gradient update on any objective function, despite having a form similar to SGD. Instead, TD learning is an example of **bootstrapping**: the estimate $V_{\boldsymbol{w}}(s_t)$ is updated to approach a target $r_t + \gamma V_{\boldsymbol{w}}(s_{t+1})$ that is itself defined by the value function estimate. MC estimation avoids this issue by using complete trajectory returns, but is often much less efficient.

</div>

### Combining TD and MC Learning Using TD($\lambda$)

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(N-Step Returns and TD($\lambda$))</span></p>

TD estimates the return by one-step lookahead, $G_{t:t+1} = r_t + \gamma V(s_{t+1})$. MC waits until the end of the episode and uses $G_{t:T} = r_t + \gamma r_{t+1} + \cdots + \gamma^{T-t-1} r_{T-1}$. We can interpolate by using the **$n$-step return**:

$$
G_{t:t+n} = r_t + \gamma r_{t+1} + \cdots + \gamma^{n-1} r_{t+n-1} + \gamma^n V(s_{t+n})
$$

Rather than picking a specific $n$, we can take a weighted average of all possible values with a single parameter $\lambda \in [0,1]$, giving the **lambda return**:

$$
G_t^\lambda \triangleq (1 - \lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_{t:t+n}
$$

The coefficients sum to one (since $\sum_{t=0}^{\infty} (1-\lambda)\lambda^t = 1$ for $\lambda < 1$), so the return is a convex combination of $n$-step returns. Using $G_t^\lambda$ inside the TD update instead of $G_{t:t+n}$ is called **TD($\lambda$)**.

* If $\lambda = 1$, the $\lambda$-return equals the regular MC return $G_t$.
* If $\lambda = 0$, the $\lambda$-return equals the one-step return $G_{t:t+1}$ (standard TD, also called **TD(0)**).

For episodic tasks, the $\lambda$-return satisfies the recursive equation:

$$
G_t^\lambda = r_t + \gamma\left[(1-\lambda) v_{t+1} + \lambda G_{t+1}^\lambda\right]
$$

initialized with $G_T = v_t$.

</div>

### Eligibility Traces

The TD($\lambda$) update requires summing over future rewards, which cannot be done in an online way. It is possible to derive a backwards-looking version that is fully online using **eligibility traces**, which are a weighted sum of the gradients of the value function:

$$
\boldsymbol{z}_t = \gamma \lambda \boldsymbol{z}_{t-1} + \nabla_{\boldsymbol{w}} V_{\boldsymbol{w}}(s_t)
$$

(This trace is reset to $0$ at the start of each episode.) The online TD($\lambda$) update rule becomes:

$$
\boldsymbol{w}_{t+1} = \boldsymbol{w}_t + \eta \delta_t \boldsymbol{z}_t
$$

## SARSA: On-Policy TD Policy Learning

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(SARSA)</span></p>

TD learning is for policy evaluation (it computes $V^\pi$). To find an optimal policy $\pi^*$, we can use it as a building block inside generalized policy iteration. It is more convenient to work with the action-value function $Q$ and a policy $\pi$ that is greedy with respect to $Q$. The agent follows $\pi$ in every step, and upon a transition $(s, a, r, s')$ the TD update rule is:

$$
Q(s,a) \leftarrow Q(s,a) + \eta\left[r + \gamma Q(s', a') - Q(s,a)\right]
$$

where $a' \sim \pi(s')$ is the action the agent will take in state $s'$. This converges to $Q^\pi$. After $Q$ is updated, $\pi$ also changes to be greedy with respect to $Q$ (policy improvement). This algorithm is called **SARSA** because its update rule involves the augmented transition $(s, a, r, s', a')$.

</div>

### Convergence

For SARSA to converge to $Q^*$, every state-action pair must be visited infinitely often (at least in the tabular case). One way to ensure this is to use a "greedy in the limit with infinite exploration" (**GLIE**) policy, such as the $\epsilon$-greedy policy with $\epsilon$ vanishing to $0$ gradually. SARSA with a GLIE policy will converge to $Q^*$ and $\pi^*$.

### Sarsa($\lambda$)

It is possible to apply the eligibility trace idea to SARSA. We compute an eligibility for each state-action pair, denoted $N(s,a)$, representing the visit count:

$$
Q(s,a) \leftarrow Q(s,a) + \eta \delta N(s,a)
$$

where $\delta = r + \gamma Q(s', a') - Q(s,a)$. Then we decay all the visit counts (traces):

$$
N(s,a) \leftarrow \gamma \lambda N(s,a)
$$

This is called **Sarsa($\lambda$)**.

## Q-Learning: Off-Policy TD Policy Learning

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Q-Learning)</span></p>

SARSA is an on-policy algorithm: it learns the Q-function for the policy it is currently using, which is typically not the optimal policy due to exploration. With a simple modification, we can convert this to an off-policy algorithm that learns $Q^*$, even if a suboptimal or exploratory policy is used to choose actions.

We modify SARSA by replacing the sampled next action $a' \sim \pi(s')$ with a greedy action $a' = \arg\max_b Q(s', b)$:

$$
Q(s,a) \leftarrow Q(s,a) + \eta\left[r + \gamma \max_{a'} Q(s', a') - Q(s,a)\right]
$$

This is the update rule of **Q-learning**. Since it is off-policy, the method can use $(s, a, r, s')$ triples coming from any data source (older versions of the policy, log data, etc.). If every state-action pair is visited infinitely often, the algorithm provably converges to $Q^*$ in the tabular case with properly decayed learning rates.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Tabular Q-Learning with $\epsilon$-Greedy Exploration)</span></p>

1. Initialize value function $Q$
2. **repeat**
3. &emsp; Sample starting state $s$ of new episode
4. &emsp; **repeat**
5. &emsp;&emsp; Sample action $a = \begin{cases} \arg\max_b Q(s,b), & \text{with probability } 1-\epsilon \\ \text{random action}, & \text{with probability } \epsilon \end{cases}$
6. &emsp;&emsp; $(s', r) = \text{env.step}(a)$
7. &emsp;&emsp; Compute TD error: $\delta = r + \gamma \max_{a'} Q(s', a') - Q(s,a)$
8. &emsp;&emsp; $Q(s,a) \leftarrow Q(s,a) + \eta \delta$
9. &emsp;&emsp; $s \leftarrow s'$
10. &emsp; **until** state $s$ is terminal
11. **until** converged

For terminal states $s \in \mathcal{S}^+$, we know $Q(s,a) = 0$ for all actions. When performing online learning, we use the modified update rule:

$$
Q(s,a) \leftarrow Q(s,a) + \eta\left[r + (1 - \text{done}(s')) \gamma \max_{a'} Q(s',a') - Q(s,a)\right]
$$

where $\text{done}(s')$ is a binary indicator telling us if $s'$ is terminal.

</div>

### Q Learning with Function Approximation

To make Q learning work with high-dimensional state spaces, we replace the tabular representation with a parametric approximation $Q_{\boldsymbol{w}}(s,a)$. We update this function using one or more steps of SGD on the loss function:

$$
\mathcal{L}(\boldsymbol{w}|\boldsymbol{s}, a, r, \boldsymbol{s}') = \left((r + \gamma \max_{a'} Q_{\boldsymbol{w}}(\boldsymbol{s}', a')) - Q_{\boldsymbol{w}}(\boldsymbol{s}, a)\right)^2
$$

Since nonlinear functions need to be trained on minibatches, we compute the average loss over randomly sampled experience tuples:

$$
\mathcal{L}(\boldsymbol{w}) = \mathbb{E}_{(\boldsymbol{s},a,r,\boldsymbol{s}') \sim U(\mathcal{D})}\left[\mathcal{L}(\boldsymbol{w}|\boldsymbol{s}, a, r, \boldsymbol{s}')\right]
$$

#### Neural Fitted Q

The first approach of this kind is known as **fitted Q evaluation** (FQE), which was extended to use neural networks as **Neural fitted Q**. This corresponds to fully optimizing $\mathcal{L}(\boldsymbol{w})$ at each iteration (equivalent to using $G = \infty$ gradient steps).

#### DQN

The influential **deep Q-network** (DQN) paper also used neural nets to represent the Q function, but performed a smaller number of gradient updates per iteration. Furthermore, they proposed to modify the target value when fitting the Q function to avoid instabilities during training (see The Deadly Triad below). DQN became famous for training agents that can outperform humans on various Atari games from the **ALE** (Atari Learning Environment) benchmark.

#### Experience Replay

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Experience Replay)</span></p>

Since Q learning is an off-policy method, we can update the Q function using any data source. An **experience replay** buffer stores $(s, a, r, s')$ transition tuples into a buffer. This modification has two advantages:

1. It improves data efficiency as every transition can be used multiple times.
2. It improves stability in training, by reducing the correlation of the data samples (training tuples do not have to come from adjacent moments in time).

Experience replay requires the use of off-policy learning methods, since the training data is sampled from older versions of the policy, not the current policy.

</div>

#### Prioritized Experience Replay

It is possible to replace the uniform sampling from the buffer with one that favors more important transition tuples. **Prioritized sweeping** iterates over all state-action pairs $(s^-, a^-)$ that can immediately transition into $s$ and increases their priority based on $\mathcal{T}(s|s^-, a^-) \times |V(s) - V^{\text{old}}(s)|$.

In **prioritized experience replay**, the priority of the $i$'th tuple $\tau_i$ is based on the TD error:

$$
\delta_i = r_i + \gamma \max_{a'} Q_{\overline{\boldsymbol{w}}}(s_i', a') - Q_{\boldsymbol{w}}(s_i, a_i)
$$

$$
p_i = (\delta_i + \epsilon)^\alpha
$$

where $\alpha \ge 0$ determines the degree of prioritization ($\alpha = 0$ is uniform sampling). The probability of sampling $i$ is $P(i) = \frac{p_i}{\sum_k p_k}$.

#### The Deadly Triad

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Deadly Triad)</span></p>

An RL algorithm can become unstable when it has these three components simultaneously: **function approximation** (such as neural networks), **bootstrapped value function estimation** (using TD-like methods instead of MC), and **off-policy learning** (where the actions are sampled from a different distribution than the policy being optimized). This combination is known as **the deadly triad**.

The root cause is that bootstrapping methods typically are not minimizing a fixed objective function. Rather, they create a learning target using their own estimates, thus potentially creating a self-reinforcing loop that pushes estimates to infinity. The contraction property of the tabular case may no longer hold when $V$ is approximated by $V_{\boldsymbol{w}}$.

A classic illustration is **Baird's counter example**: a simple MDP with 7 states where off-policy TD(0) with linear approximation causes the parameters to diverge to $\infty$, even though the true value function is $V_\pi(s) = 0$ for all states and can be exactly represented by setting $\boldsymbol{w} = \mathbf{0}$.

</div>

#### Target Networks

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Target Networks)</span></p>

A heuristic solution to the deadly triad is to use a frozen **target network** computed at an earlier iteration to define the target value for updates. We maintain an extra copy of the Q-network, $Q_{\overline{\boldsymbol{w}}}$, with the same structure as $Q_{\boldsymbol{w}}$. This new Q-network computes bootstrapping targets:

$$
y(r, \boldsymbol{s}'; \overline{\boldsymbol{w}}) = r + \gamma \max_{a'} Q_{\overline{\boldsymbol{w}}}(\boldsymbol{s}', a')
$$

We can periodically set $\overline{\boldsymbol{w}} \leftarrow \text{sg}(\boldsymbol{w})$ (a **detached target**), or use an **exponential moving average** (EMA):

$$
\overline{\boldsymbol{w}}_k = \rho \overline{\boldsymbol{w}}_{k-1} + (1 - \rho) \boldsymbol{w}_k
$$

where $\rho \approx 0.999$ ensures $Q_{\overline{\boldsymbol{w}}}$ slowly catches up with $Q_{\boldsymbol{w}}$. The final loss is:

$$
\mathcal{L}(\boldsymbol{w}|\boldsymbol{s}, a, r, \boldsymbol{s}') = (y(r, \boldsymbol{s}'; \overline{\boldsymbol{w}}) - Q_{\boldsymbol{w}}(\boldsymbol{s}, a))^2
$$

</div>

#### Other Stability Methods

* **Gradient TD methods:** Construct an objective function whose minimization leads to a good value function approximation, ensuring convergence in off-policy learning.
* **Two time-scale methods:** Update the target value in the TD update more quickly than the value function itself.
* **Layer norm:** Nonlinear TD learning can be made to converge even in the off-policy setting if three conditions on the critic ($Q$ network) are satisfied: the final layer weights are bounded (e.g., using $\ell_2$ normalization or AdamW), the penultimate layer is sufficiently wide, and the input to the critic has bounded norm (e.g., using LayerNorm or RMSNorm). Since $\|\text{LayerNorm}(f(s,a;\boldsymbol{\theta}))\| \le 1$, the magnitude of the output is always bounded, preventing catastrophic overestimation.

### Maximization Bias

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Maximization Bias)</span></p>

Standard Q-learning suffers from the **optimizer's curse** (or **maximization bias**): $\mathbb{E}\left[\max_a X_a\right] \ge \max_a \mathbb{E}\left[X_a\right]$ for random variables $\lbrace X_a \rbrace$. If we pick actions greedily according to noisy scores, we might pick a wrong action just because random noise makes it look appealing.

</div>

#### Double Q-Learning

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Double Q-Learning)</span></p>

One solution to maximization bias is to use two separate Q-functions, $Q_1$ and $Q_2$, one for selecting the greedy action and the other for estimating the corresponding Q-value. Upon a transition $(s, a, r, s')$, for $i = 1:2$:

$$
Q_i(s,a) \leftarrow Q_i(s,a) + \eta(y_i(s,a) - Q_i(s,a))
$$

$$
y_i(s,a) = r + \gamma Q_i(s', \arg\max_{a'} Q_{-i}(s', a'))
$$

So $Q_1$ uses $Q_2$ to choose the best action but uses $Q_1$ to evaluate it, and vice versa. This is called **double Q-learning**.

</div>

**Double DQN** combines double Q learning with deep Q networks. The training target uses the current network for action proposals but the target network for action evaluation:

$$
y(r, \boldsymbol{s}'; \boldsymbol{w}, \overline{\boldsymbol{w}}) = r + \gamma Q_{\overline{\boldsymbol{w}}}(\boldsymbol{s}', \arg\max_{a'} Q_{\boldsymbol{w}}(\boldsymbol{s}', a'))
$$

**Clipped double DQN** uses two Q networks and their frozen copies:

$$
y(r, \boldsymbol{s}'; \boldsymbol{w}_{1:2}, \overline{\boldsymbol{w}}_{1:2}) = r + \gamma \min_{i=1,2} Q_{\overline{\boldsymbol{w}}_i}(\boldsymbol{s}', \arg\max_{a'} Q_{\boldsymbol{w}_i}(\boldsymbol{s}', a'))
$$

**REDQ** (randomized ensembled double Q learning) extends this with an ensemble of $N > 2$ Q-networks. At each step, it draws a random sample of $M \le N$ networks and takes the minimum over them when computing the target value:

$$
y(r, \boldsymbol{s}'; \boldsymbol{w}_{1:N}, \overline{\boldsymbol{w}}_{1:N}) = r + \gamma \max_{a'} \min_{i \in \mathcal{M}} Q_{\overline{\boldsymbol{w}}_i}(\boldsymbol{s}', a')
$$

The ensemble reduces variance, and the minimum reduces overestimation bias.

### DQN Extensions

#### Q Learning for Continuous Actions

Q learning is not directly applicable to continuous actions due to the need to compute $\arg\max$ over actions. Solutions include: learning a policy to predict the argmax (basis of the **DDPG** algorithm), using gradient-free optimizers such as the cross-entropy method (**QT-Opt**), treating the action vector as a sequence and optimizing one dimension at a time, using mixed integer programming to solve the argmax problem (**CAQL**), or quantizing each action dimension and solving using multi-agent RL methods.

#### Dueling DQN

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Dueling DQN)</span></p>

**Dueling DQN** learns a value function and an advantage function, and derives the Q function rather than learning it directly. This is helpful when there are many actions with similar Q-values. We define a network with $|\mathcal{A}|+1$ output heads, computing $A_{\boldsymbol{w}}(\boldsymbol{s}, a)$ for each action $a$ and $V_{\boldsymbol{w}}(\boldsymbol{s})$. The naive composition $Q_{\boldsymbol{w}}(\boldsymbol{s},a) = V_{\boldsymbol{w}}(\boldsymbol{s}) + A_{\boldsymbol{w}}(\boldsymbol{s},a)$ ignores the constraint $\mathbb{E}_{\pi(a|s)}[A^\pi(s,a)] = 0$. To satisfy this, we subtract off $\max_a A(\boldsymbol{s},a)$ from the advantage head:

$$
Q_{\boldsymbol{w}}(\boldsymbol{s},a) = V_{\boldsymbol{w}}(\boldsymbol{s}) + A_{\boldsymbol{w}}(\boldsymbol{s},a) - \max_{a'} A_{\boldsymbol{w}}(\boldsymbol{s}, a')
$$

In practice, the max is replaced by an average, which works better empirically.

</div>

#### Noisy Nets and Exploration

Standard DQN relies on $\epsilon$-greedy for exploration, which explores equally in all states. **Noisy nets** add random noise to the network weights to encourage exploration that reflects **epistemic uncertainty** (uncertainty due to lack of knowledge rather than aleatoric or irreducible uncertainty). The noise encourages exploration which is temporally consistent within episodes.

#### Multi-Step DQN

We can reduce the bias introduced by bootstrapping by replacing TD(1) updates with TD($n$) updates, where we unroll the value computation for $n$ MC steps and then plug in the value function at the end:

$$
y(s_0, a_0) = \sum_{t=1}^{n} \gamma^{t-1} r_t + \gamma^n \max_{a_n} Q_{\boldsymbol{w}}(s_n, a_n)
$$

Theoretically this requires all intermediate actions $a_{2:n-1}$ to be sampled from the current optimal policy, but in practice restricting sampling to recent samples from the replay buffer makes the method approximately on-policy.

#### Q($\lambda$)

Instead of using a fixed $n$, use a weighted combination of returns (the $Q(\lambda)$ algorithm), relying on eligibility traces. This is more complicated than the SARSA case since Q learning is off-policy, but the eligibility traces backpropagate information obtained by the exploration policy.

#### Rainbow

The **Rainbow** method combined 6 improvements to the vanilla DQN method:

* Double DQN
* Prioritized experience replay
* Categorical DQN (C51) distributional RL
* $n$-step returns (with $n = 3$)
* Dueling DQN
* Noisy nets

The "Beyond the Rainbow" paper proposed further extensions including a larger CNN with residual connections (Impala network with spectral normalization), Implicit Quantile Networks replacing C51, and **Munchausen RL** (adding an entropy-like penalty to the Q learning update rule).

#### Bigger, Better, Faster (BBF)

The **BBF** algorithm achieved SOTA on the 100k sample-efficient Atari benchmark using (in order of decreasing importance):

* A larger CNN with residual connections (modified Impala network)
* Increased **update-to-data** (UTD) ratio
* Periodic soft reset of network weights (following the **SR-SPR** method) to avoid loss of elasticity
* $n$-step returns with gradual annealing from $n=10$ to $n=3$
* Weight decay
* Self-predictive representation loss
* Gradual discount factor increase from $\gamma = 0.97$ to $\gamma = 0.997$
* Dueling DQN and distributional DQN

### Q-Learning for GCRL Using Hindsight Relabeling

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Hindsight Experience Replay)</span></p>

To learn a goal-conditioned policy, we collect trajectories from $s_0$ to some terminal state $s_T$, then define the goal as $g = s_T$; this trajectory serves as a demonstration of how to achieve this goal. This is called **hindsight experience relabeling** (HER) or just **hindsight relabeling**. We relabel the trajectories in the replay buffer: if we have $(s, a, R(s|g), s', g)$ tuples, we replace them with $(s, a, R(s|g'), g')$ where $g' = s_T$. We can then use any off-policy RL method to learn $\pi(a|s,g)$.

HER can be viewed as a special case of maximum-entropy inverse RL, since it is estimating the reward for which the corresponding trajectory was optimal. One limitation is that it only works when the reward is Markovian ($R(s,a) = 1$ iff $s = g$).

</div>

# Chapter 3: Policy-Based RL

In Chapter 2, we considered methods that estimate $Q(s,a)$, from which we derive a policy. However, these methods have several disadvantages: (1) they can be difficult to apply to continuous action spaces; (2) they may diverge if function approximation is used (the deadly triad); (3) the training of $Q$ via TD-style updates is not directly related to the expected return; (4) they learn deterministic policies, whereas in stochastic and partially observed environments, stochastic policies are provably better.

In this chapter, we discuss **policy search** methods, which directly optimize the parameters of the policy so as to maximize its expected return. We mostly focus on **policy gradient** methods, that use the gradient of the loss to guide the search. These methods often benefit from estimating a value or advantage function to reduce variance, so they also use techniques from Chapter 2.

The parametric policy is denoted by $\pi_{\boldsymbol{\theta}}(a|s)$, which is usually some form of neural network. For discrete actions, the final layer uses softmax to produce a categorical distribution. For continuous actions, a Gaussian output layer is typically used (potentially clipped to $[-1,1]$), although more expressive distributions such as diffusion models can also be used (a **diffusion policy**).

## Policy Gradient Methods

### Likelihood Ratio Estimate

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Policy Gradient via Likelihood Ratio)</span></p>

We define the value of a policy as:

$$
J(\pi_{\boldsymbol{\theta}}) = J(\boldsymbol{\theta}) = \mathbb{E}_{p_{\boldsymbol{\theta}}(\boldsymbol{\tau})}\left[R(\boldsymbol{\tau})\right]
$$

where $R(\boldsymbol{\tau}) = \gamma^0 r_0 + \gamma^1 r_1 + \ldots$ is the return along the trajectory, and $p_{\boldsymbol{\theta}}(\boldsymbol{\tau})$ is the distribution over trajectories induced by the policy (and world model):

$$
p_{\boldsymbol{\theta}}(\boldsymbol{\tau}) = p(s_1) \prod_{k=1}^{T} \mathcal{T}(s_{k+1}|s_k, a_k) \pi_{\boldsymbol{\theta}}(a_k|s_k)
$$

The gradient of the policy value is given by the **likelihood ratio estimator**:

$$
\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \int \nabla_{\boldsymbol{\theta}} p_{\boldsymbol{\theta}}(\boldsymbol{\tau}) R(\boldsymbol{\tau}) d\boldsymbol{\tau} = \mathbb{E}_{\boldsymbol{\tau}}\left[\frac{\nabla_{\boldsymbol{\theta}} p_{\boldsymbol{\theta}}(\boldsymbol{\tau})}{p_{\boldsymbol{\theta}}(\boldsymbol{\tau})} R(\boldsymbol{\tau})\right]
$$

Using the **log derivative trick** ($\nabla \log \pi = \frac{\nabla \pi}{\pi}$), we can rewrite this as:

$$
\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \mathbb{E}_{\boldsymbol{\tau}}\left[\nabla_{\boldsymbol{\theta}} \log p_{\boldsymbol{\theta}}(\boldsymbol{\tau}) R(\boldsymbol{\tau})\right]
$$

Since $\nabla_{\boldsymbol{\theta}} \log p_{\boldsymbol{\theta}}(\boldsymbol{\tau}) = \sum_{k=1}^{T} \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_k|s_k)$, we get:

$$
\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \mathbb{E}_{\boldsymbol{\tau}}\left[\left(\sum_{k=1}^{T} \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_k|s_k)\right) R(\boldsymbol{\tau})\right]
$$

The term $\nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a|s)$ is called the (Fisher) **score function**, so this is sometimes called the **score function estimator** (SFE). The expectations can be estimated using Monte Carlo sampling.

</div>

### Variance Reduction Using Reward-to-Go

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Reward-to-Go Reduces Variance)</span></p>

The likelihood ratio estimator can have high variance since we are sampling entire trajectories. We can reduce the variance using the temporal/causal structure: the reward at step $k$ cannot depend on actions at future time steps. This gives:

$$
\nabla J(\boldsymbol{\theta}) = \mathbb{E}_{\boldsymbol{\tau}}\left[\sum_{k=1}^{T} \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_k|s_k) \gamma^{k-1} G_k\right]
$$

where $G_k \triangleq r_k + \gamma r_{k+1} + \cdots + \gamma^{T-k-1} r_{T-1} = \sum_{l=k}^{T-1} \gamma^{l-k} r_l$ is the **reward-to-go** (or return). The reward-to-go of a state-action pair $(s,a)$ can be considered as a single sample approximation of the state-action value function $Q_{\boldsymbol{\theta}}(s,a)$.

</div>

### REINFORCE

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(REINFORCE)</span></p>

The **REINFORCE** algorithm uses the gradient estimate of the policy value together with SGD to fit a policy:

$$
\boldsymbol{\theta}_{j+1} := \boldsymbol{\theta}_j + \eta \sum_{k=1}^{T} \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}_j}(a_k|s_k) \gamma^{k-1} G_k
$$

where $j$ is the SGD iteration number, and we draw a single trajectory at each step. The update can be interpreted as follows: we compute the sum of discounted future rewards induced by a trajectory, and if this is positive, we increase $\boldsymbol{\theta}$ to make this trajectory more likely; otherwise we decrease $\boldsymbol{\theta}$. Thus, we reinforce good behaviors and reduce the chances of generating bad ones.

1. Initialize policy parameters $\boldsymbol{\theta}$
2. **repeat**
3. &emsp; Sample an episode $\boldsymbol{\tau} = (s_1, a_1, r_1, s_2, \ldots, s_T)$ using $\pi_{\boldsymbol{\theta}}$
4. &emsp; **for** $k = 1, \ldots, T$ **do**
5. &emsp;&emsp; $G_k = \sum_{l=k}^{T} \gamma^{l-k} R_l$
6. &emsp;&emsp; $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \eta_{\boldsymbol{\theta}} \gamma^{k-1} G_k \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_k|s_k)$
7. **until** converged

</div>

### The Policy Gradient Theorem

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Policy Gradient Theorem)</span></p>

We define the **discounted state visitation measure**:

$$
\rho_\pi^\gamma(s) \triangleq \gamma^0 P(s_0 = s|\pi) + \gamma P(s_1 = s|\pi) + \gamma^2 P(s_2 = s|\pi) + \cdots = \sum_{t=0}^{\infty} \gamma^t p_t^\pi(s)
$$

where $p_t^\pi(s)$ is the marginal probability of being in state $s$ at time $t$. The **normalized discounted state visitation distribution** is $p_\pi^\gamma(s) = (1-\gamma) \rho_\pi^\gamma(s)$.

The **policy gradient theorem** states:

$$
\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \mathbb{E}_{\rho_\pi^\gamma(s) \pi_{\boldsymbol{\theta}}(a|s)}\left[Q^{\pi_{\boldsymbol{\theta}}}(s,a) \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a|s)\right]
$$

$$
= \frac{1}{1-\gamma} \mathbb{E}_{p_\pi^\gamma(s) \pi_{\boldsymbol{\theta}}(a|s)}\left[Q^{\pi_{\boldsymbol{\theta}}}(s,a) \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a|s)\right]
$$

This rewrites the gradient in terms of expectations over states rather than over trajectories.

</div>

### Variance Reduction Using a Baseline

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Baseline for Variance Reduction)</span></p>

A **baseline** function $b(s)$ can be used for variance reduction:

$$
\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \mathbb{E}_{\rho_{\boldsymbol{\theta}}(s) \pi_{\boldsymbol{\theta}}(a|s)}\left[(Q_{\pi_{\boldsymbol{\theta}}}(s,a) - b(s)) \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a|s)\right]
$$

Any function that satisfies $\mathbb{E}\left[\nabla_{\boldsymbol{\theta}} b(s)\right] = 0$ is a valid baseline. This follows since $\sum_a \nabla_{\boldsymbol{\theta}} \pi_{\boldsymbol{\theta}}(a|s)(Q(s,a) - b(s)) = \nabla_{\boldsymbol{\theta}} \sum_a \pi_{\boldsymbol{\theta}}(a|s) Q(s,a) - 0$.

A common choice is $b(s) = V(s)$, which is valid if we use an old (frozen) version of the policy that is independent of $\boldsymbol{\theta}$. Since $Q(s,a) - V(s) = A(s,a)$ is the advantage function, we get:

$$
\nabla J(\boldsymbol{\theta}) = \mathbb{E}_{\boldsymbol{\tau}}\left[\sum_{k=1}^{T} \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_k|s_k) \gamma^{k-1} A_{\boldsymbol{\theta}}(s_k, a_k)\right]
$$

Using the reward-to-go formulation with baseline:

$$
\nabla J(\boldsymbol{\theta}) = \mathbb{E}_{\boldsymbol{\tau}}\left[\sum_{k=1}^{T} \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_k|s_k) \gamma^{k-1}(G_k - b(s_k))\right]
$$

</div>

### REINFORCE with Baseline

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(REINFORCE with Value Function Baseline)</span></p>

1. Initialize policy parameters $\boldsymbol{\theta}$, baseline parameters $\boldsymbol{w}$
2. **repeat**
3. &emsp; Sample an episode $\boldsymbol{\tau} = (s_1, a_1, r_1, s_2, \ldots, s_T)$ using $\pi_{\boldsymbol{\theta}}$
4. &emsp; **for** $k = 1, \ldots, T$ **do**
5. &emsp;&emsp; $G_k = \sum_{l=k}^{T} \gamma^{l-k} R_l$
6. &emsp;&emsp; $\delta_k = G_k - V_{\boldsymbol{w}}(s_k)$
7. &emsp;&emsp; $\boldsymbol{w} \leftarrow \boldsymbol{w} - \eta_w \delta_k \nabla_{\boldsymbol{w}} V_{\boldsymbol{w}}(s_k)$
8. &emsp;&emsp; $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \eta_{\boldsymbol{\theta}} \gamma^{k-1} \delta_k \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_k|s_k)$
9. **until** converged

</div>

## Actor-Critic Methods

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Actor-Critic)</span></p>

An **actor-critic** method uses the policy gradient method, but the expected return $G_t$ is estimated using temporal difference learning of a value function instead of MC rollouts. The "actor" refers to the policy and the "critic" refers to the value function. The use of bootstrapping in TD updates allows more efficient learning and further reduces variance. It also enables a fully online, incremental algorithm that does not need to wait until the end of the trajectory.

</div>

### Advantage Actor Critic (A2C)

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(A2C --- Advantage Actor Critic)</span></p>

We replace the MC return $G_t$ with the one-step TD estimate $G_{t:t+1} = r_t + \gamma V_{\boldsymbol{w}}(s_{t+1})$. Using $V_{\boldsymbol{w}}(s_t)$ as a baseline, the REINFORCE update becomes:

$$
\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \eta \sum_{t=0}^{T-1} \gamma^t \left(r_t + \gamma V_{\boldsymbol{w}}(s_{t+1}) - V_{\boldsymbol{w}}(s_t)\right) \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t|s_t)
$$

Note that $\delta_t = r_{t+1} + \gamma V_{\boldsymbol{w}}(s_{t+1}) - V_{\boldsymbol{w}}(s_t)$ is a single sample approximation to the advantage function $\text{Adv}(s_t, a_t) = Q(s_t, a_t) - V(s_t)$. This is called **advantage actor critic** or **A2C**.

This is an on-policy algorithm where we update the value function $V_{\boldsymbol{w}}^\pi$ to reflect the value of the current policy $\pi$.

In practice, it is common to use a shared network with separate value and policy heads, trained with a single loss function:

$$
\mathcal{L}(\boldsymbol{\phi}; \tau) = \frac{1}{T} \sum_{t=1}^{T}\left[\lambda_{TD} \mathcal{L}_{TD}(s_t, a_t, r_t, s_{t+1}) - \lambda_{PG} J_{PG}(s_t, a_t, r_t, s_{t+1}) - \lambda_{ent} J_{ent}(s_t)\right]
$$

where $\mathcal{L}_{TD} = (\text{sg}(y_t) - V_\phi(s))^2$, $J_{PG} = (\text{sg}(y_t - V_\phi(s_t))) \log \pi_\phi(a_t|s_t)$, and $J_{ent} = -\sum_a \pi_\phi(a|s_t) \log \pi_\phi(a|s_t)$ is an entropy regularizer that encourages the policy to remain stochastic.

The **PopArt** method can be used to dynamically normalize the TD and policy gradient losses to allow for a fixed set of hyper-parameter values $\lambda_i$ even as the range of losses changes over time.

</div>

### Generalized Advantage Estimation (GAE)

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Generalized Advantage Estimation)</span></p>

In A2C, we replaced the high-variance, unbiased MC return $G_t$ with the low-variance, but biased, one-step bootstrap return. More generally, we can compute the $n$-step advantage estimate:

$$
A_{\boldsymbol{w}}^{(n)}(s_t, a_t) = G_{t:t+n} - V_{\boldsymbol{w}}(s_t)
$$

where $A_t^{(1)} = r_t + \gamma v_{t+1} - v_t$ (high bias, low variance) and $A_t^{(\infty)} = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots - v_t$ (unbiased, high variance).

Instead of a single $n$, take a weighted average with $w_n = \lambda^{n-1}$:

$$
A_t = \frac{\sum_{n=1}^{T} w_n A_t^{(n)}}{\sum_{n=1}^{T} w_n}
$$

This gives a simple recursive calculation:

$$
\delta_t = r_t + \gamma v_{t+1} - v_t
$$

$$
A_t = \delta_t + \gamma \lambda A_{t+1}
$$

Here $\lambda \in [0,1]$ controls the bias-variance tradeoff: larger values decrease the bias but increase the variance. This is called **generalized advantage estimation** (GAE).

</div>

The gradient estimator $\nabla J(\boldsymbol{\theta}) = \mathbb{E}\left[\sum_{t=0}^{\infty} \Psi_t \nabla \log \pi_{\boldsymbol{\theta}}(a_t|s_t)\right]$ admits several choices for $\Psi_t$:

| $\Psi_t$ | Description |
| --- | --- |
| $\sum_{i=t}^{\infty} \gamma^i r_i$ | Monte Carlo target |
| $\sum_{i=t}^{\infty} \gamma^i r_i - V_{\boldsymbol{w}}(s_t)$ | MC with baseline |
| $A_{\boldsymbol{w}}(s_t, a_t)$ | Advantage function |
| $Q_{\boldsymbol{w}}(s_t, a_t)$ | Q function |
| $r_t + \gamma V_{\boldsymbol{w}}(s_{t+1}) - V_{\boldsymbol{w}}(s_t)$ | TD residual |

### Two-Time Scale Actor Critic Algorithms

In standard AC, we update the actor and critic in parallel. However, it is better to let the critic $V_{\boldsymbol{w}}$ learn using a faster learning rate (or more updates) so that it reflects the value of the current policy $\pi_{\boldsymbol{\theta}}$ more accurately. This is known as **two timescale learning** or **bilevel optimization**.

An alternative is to alternate between updating the policy and the value function rather than updating them simultaneously. This is called **phasic policy gradient**.

### Natural Policy Gradient Methods

#### Natural Gradient Descent

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Natural Gradient Descent)</span></p>

**Natural gradient descent** (NGD) is a second-order method for optimizing the parameters of probability distributions such as policies $\pi_{\boldsymbol{\theta}}(\boldsymbol{a}|\boldsymbol{s})$. Standard SGD uses a **proximal update**: $\boldsymbol{\theta}_{k+1} = \arg\min_{\boldsymbol{\theta}} \hat{\mathcal{L}}_k(\boldsymbol{\theta})$ s.t. $\|\boldsymbol{\theta} - \boldsymbol{\theta}_k\|_2^2 \le \epsilon$. However, Euclidean distance in parameter space does not make sense for probabilistic models (changes in $\mu$ matter more when $\sigma$ is small).

The key idea of NGD is to measure distance between distributions using the **KL divergence**, approximated by the **Fisher information matrix** (FIM):

$$
D_{KL}(p_{\boldsymbol{\theta}}(\boldsymbol{y}|\boldsymbol{x}) \| p_{\boldsymbol{\theta}+\boldsymbol{\delta}}(\boldsymbol{y}|\boldsymbol{x})) \approx \frac{1}{2} \boldsymbol{\delta}^\top \mathbf{F}_{\boldsymbol{x}} \boldsymbol{\delta}
$$

where $\mathbf{F}_{\boldsymbol{x}}(\boldsymbol{\theta}) = \mathbb{E}_{p_{\boldsymbol{\theta}}(\boldsymbol{y}|\boldsymbol{x})}\left[(\nabla \log p_{\boldsymbol{\theta}}(\boldsymbol{y}|\boldsymbol{x}))(\nabla \log p_{\boldsymbol{\theta}}(\boldsymbol{y}|\boldsymbol{x}))^\top\right]$.

Replacing the Euclidean constraint with $\boldsymbol{\delta}^\top \mathbf{F}_k \boldsymbol{\delta} \le \epsilon$ and solving gives the **natural gradient** update:

$$
\boldsymbol{\delta} = -\eta_k \mathbf{F}_k^{-1} \boldsymbol{g}_k
$$

This is equivalent to a preconditioned gradient update using the inverse FIM. The adaptive learning rate is $\eta_k = \sqrt{\frac{\epsilon}{\boldsymbol{g}_k^\top \mathbf{F}_k^{-1} \boldsymbol{g}_k}}$.

The FIM can be approximated using the **empirical Fisher**: $\mathbf{F}(\boldsymbol{\theta}) \approx \frac{1}{|\mathcal{D}|} \sum_{(\boldsymbol{x},\boldsymbol{y}) \in \mathcal{D}} \nabla \log p(\boldsymbol{y}|\boldsymbol{x}, \boldsymbol{\theta}) \nabla \log p(\boldsymbol{y}|\boldsymbol{x}, \boldsymbol{\theta})^\top$.

</div>

#### Natural Actor Critic

To apply NGD to RL, define $g_{kt} = \nabla_{\boldsymbol{\theta}_k} A_t \log \pi_{\boldsymbol{\theta}}(\boldsymbol{a}_t|\boldsymbol{s}_t)$ and compute $\boldsymbol{g}_k = \frac{1}{T} \sum_{t=1}^{T} g_{kt}$, $\mathbf{F}_k = \frac{1}{T} \sum_{t=1}^{T} g_{kt} g_{kt}^\top$, then $\boldsymbol{\delta}_{k+1} = -\eta_k \mathbf{F}_k^{-1} \boldsymbol{g}_k$. This is called **natural policy gradient**.

We can compute $\mathbf{F}_k^{-1} \boldsymbol{g}_k$ without having to invert $\mathbf{F}_k$ by using the conjugate gradient method (**Hessian free optimization**). The **KFAC** method approximates the FIM of a DNN as a block diagonal matrix, where each block is a Kronecker product of two small matrices.

### Architectural Issues

It is common to use a single neural network for both the actor and critic, but using different output heads: a scalar output for the value function and a vector output for the policy. The **Amago** method uses a transformer backbone. However, it can sometimes be better to use different networks for the actor and critic, since they need to extract different kinds of features.

### Deterministic Policy Gradient Methods

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Deterministic Policy Gradient)</span></p>

An actor-critic method that uses a deterministic policy $a_t = \mu_{\boldsymbol{\theta}}(s_t)$ rather than $a_t \sim \pi_{\boldsymbol{\theta}}(s_t)$. This can be thought of as a version of DQN designed for continuous actions.

The **deterministic policy gradient theorem** states:

$$
\nabla_{\boldsymbol{\theta}} J(\mu_{\boldsymbol{\theta}}) = \mathbb{E}_{\rho_{\mu_{\boldsymbol{\theta}}}(s)}\left[\nabla_{\boldsymbol{\theta}} \mu_{\boldsymbol{\theta}}(s) \nabla_a Q_{\mu_{\boldsymbol{\theta}}}(s,a)|_{a=\mu_{\boldsymbol{\theta}}(s)}\right]
$$

The gradient integrates over states but not over actions (reducing variance). Since the deterministic policy does no exploration, we use an off-policy method with data collected from a stochastic behavior policy $\pi_b$. We learn both a state-action critic $Q_{\boldsymbol{w}}$ and an actor $\mu_{\boldsymbol{\theta}}$ using:

$$
\boldsymbol{w}_{t+1} \leftarrow \boldsymbol{w}_t + \eta_w \delta \nabla_{\boldsymbol{w}} Q_{\boldsymbol{w}}(s_t, a_t)
$$

$$
\boldsymbol{\theta}_{t+1} \leftarrow \boldsymbol{\theta}_t + \eta_{\boldsymbol{\theta}} \nabla_{\boldsymbol{\theta}} \mu_{\boldsymbol{\theta}}(s_t) \nabla_a Q_{\boldsymbol{w}}(s_t, a)|_{a=\mu_{\boldsymbol{\theta}}(s_t)}
$$

where $\delta = r_t + \gamma Q_{\boldsymbol{w}}(s_{t+1}, \mu_{\boldsymbol{\theta}}(s_{t+1})) - Q_{\boldsymbol{w}}(s_t, a_t)$.

</div>

#### DDPG

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(DDPG)</span></p>

The **DDPG** (deep deterministic policy gradient) algorithm uses the DQN method to learn the Q function (with a target network and replay buffer), and then uses this to evaluate and improve the policy. The actor tries to minimize the output of the critic:

$$
\mathcal{L}_{\boldsymbol{\theta}}(s) = Q_{\boldsymbol{w}}(s, \mu_{\boldsymbol{\theta}}(s))
$$

The critic tries to minimize the 1-step TD loss:

$$
\mathcal{L}_w(s, a, r, s') = \left[Q_{\boldsymbol{w}}(s,a) - (r + \gamma Q_{\overline{\boldsymbol{w}}}(s', \mu_{\boldsymbol{\theta}}(s')))\right]^2
$$

where $Q_{\overline{\boldsymbol{w}}}$ is the target critic network. The **D4PG** algorithm extends DDPG to handle distributed training and distributional RL.

</div>

#### Twin Delayed DDPG (TD3)

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(TD3 --- Twin Delayed DDPG)</span></p>

**TD3** extends DDPG in 3 main ways:

1. **Target policy smoothing:** Noise is added to the action to encourage generalization: $\tilde{a} = \mu_{\boldsymbol{\theta}}(\boldsymbol{s}) + \text{noise} = \pi_{\boldsymbol{\theta}}(\boldsymbol{s})$

2. **Clipped double Q learning:** Target values are defined using the minimum of two target critic networks to avoid over-estimation bias:

$$
y(r, \boldsymbol{s}'; \overline{\boldsymbol{w}}_{1:2}, \overline{\boldsymbol{\theta}}) = r + \gamma \min_{i=1,2} Q_{\overline{\boldsymbol{w}}_i}(\boldsymbol{s}', \pi_{\overline{\boldsymbol{\theta}}}(\boldsymbol{s}'))
$$

3. **Delayed policy updates:** The policy is only updated after the value function has stabilized.

</div>

#### Wasserstein Policy Optimization (WPO)

**WPO** approximates **Wasserstein gradient flows** over the space of all parametric policies, arriving at an update similar to DPG but for general stochastic policies:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t + \mathcal{F}^{-1} \mathbb{E}_{\boldsymbol{s} \sim \mathcal{D}, \boldsymbol{a} \sim \pi_{\boldsymbol{\theta}}(\cdot|\boldsymbol{s})}\left[\nabla_{\boldsymbol{\theta}} \left(\nabla_{\boldsymbol{a}} \log \pi_{\boldsymbol{\theta}}(\boldsymbol{a}|\boldsymbol{s})^\top\right) \nabla_{\boldsymbol{a}} Q^{\pi_{\boldsymbol{\theta}}}(\boldsymbol{s}, \boldsymbol{a})\right]
$$

where $\mathcal{F}$ is the Fisher information matrix. If we ignore the FIM preconditioner, WPO reduces to the DPG theorem except that $\nabla_{\boldsymbol{\theta}} \mu_{\boldsymbol{\theta}}(\boldsymbol{s})$ is replaced by $\nabla_{\boldsymbol{\theta}}(\nabla_{\boldsymbol{a}} \log \pi_{\boldsymbol{\theta}}(\boldsymbol{a}|\boldsymbol{s}))^\top$, capturing the change in **probability flow** over the action space. The FIM preconditioner keeps the update closer to the true gradient flow and avoids numerical issues when the policy converges to a deterministic one.

## Policy Improvement Methods

These methods try to monotonically improve the performance of the policy at each step, rather than just following the gradient (which can result in high-variance performance fluctuations).

### Policy Improvement Lower Bound

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Policy Improvement Lower Bound)</span></p>

Let $\pi_k$ be the current policy and $\pi$ be any other policy. Let $A^{\pi_k}(s,a) = Q^{\pi_k}(s,a) - V^{\pi_k}(s)$ be the advantage function. Then:

$$
J(\pi) - J(\pi_k) \ge \frac{1}{1-\gamma} \underbrace{\mathbb{E}_{p_{\pi_k}^\gamma(s) \pi_k(a|s)}\left[\frac{\pi(a|s)}{\pi_k(a|s)} A^{\pi_k}(s,a)\right]}_{L(\pi, \pi_k)} - \frac{2\gamma C^{\pi, \pi_k}}{(1-\gamma)^2} \mathbb{E}_{p_{\pi_k}^\gamma(s)}\left[\text{TV}(\pi(\cdot|s), \pi_k(\cdot|s))\right]
$$

where $C^{\pi, \pi_k} = \max_s |\mathbb{E}_{\pi(a|s)}[A^{\pi_k}(s,a)]|$, $L(\pi, \pi_k)$ is a surrogate objective, and the second term is a penalty based on the total variation distance $\text{TV}(p,q) = \frac{1}{2}\|\boldsymbol{p} - \boldsymbol{q}\|_1$.

If we can optimize this lower bound (or a stochastic approximation), we can guarantee monotonic policy improvement (in expectation) at each step.

</div>

We replace this with a trust-region update:

$$
\pi_{k+1} = \arg\max_\pi L(\pi, \pi_k) \quad \text{s.t.} \quad \mathbb{E}_{p_{\pi_k}^\gamma(s)}\left[\text{TV}(\pi, \pi_k)(s)\right] \le \epsilon
$$

### Trust Region Policy Optimization (TRPO)

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(TRPO)</span></p>

**Trust region policy optimization** (TRPO) implements the policy improvement lower bound approximately. It leverages the fact that if $\mathbb{E}_{p_{\pi_k}^\gamma(s)}[D_{KL}(\pi_k \| \pi)(s)] \le \delta$, then $\pi$ also satisfies the TV constraint with $\delta = \frac{\epsilon^2}{2}$.

It considers a first-order expansion of the surrogate objective:

$$
L(\pi, \pi_k) = \mathbb{E}_{p_{\pi_k}^\gamma(s) \pi_k(a|s)}\left[\frac{\pi(a|s)}{\pi_k(a|s)} A^{\pi_k}(s,a)\right] \approx \boldsymbol{g}_k^\top (\boldsymbol{\theta} - \boldsymbol{\theta}_k)
$$

and a second-order expansion of the KL term:

$$
\mathbb{E}_{p_{\pi_k}^\gamma(s)}[D_{KL}(\pi_k \| \pi)(s)] \approx \frac{1}{2}(\boldsymbol{\theta} - \boldsymbol{\theta}_k)^\top \mathbf{F}_k (\boldsymbol{\theta} - \boldsymbol{\theta}_k)
$$

Then the update is $\boldsymbol{\theta}_{k+1} = \boldsymbol{\theta}_k + \eta_k \boldsymbol{v}_k$ where $\boldsymbol{v}_k = \mathbf{F}_k^{-1} \boldsymbol{g}_k$ is the natural gradient, and $\eta_k = \sqrt{\frac{2\delta}{\boldsymbol{v}_k^\top \mathbf{F}_k \boldsymbol{v}_k}}$. We use a backtracking line search to ensure the trust region is satisfied.

</div>

### Proximal Policy Optimization (PPO)

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(PPO --- Proximal Policy Optimization)</span></p>

**PPO** is a simplification of TRPO. Let $\rho_k(s,a) = \frac{\pi(a|s)}{\pi_k(a|s)}$ be the likelihood ratio and $\tilde{\rho}_k(s,a) = \text{clip}(\rho_k(s,a), 1-\epsilon, 1+\epsilon)$. The policy update is:

$$
\pi_{k+1} = \arg\max_\pi \mathbb{E}_{(s,a) \sim p_{\pi_k}^\gamma}\left[\min\left(\rho_k(s,a) A^{\pi_k}(s,a), \; \tilde{\rho}_k(s,a) A^{\pi_k}(s,a)\right)\right]
$$

**Simplified form of the clipping term:**

$$
L(s, a, \theta_k, \theta) = \min(\rho A, \; g(\epsilon, A))
$$

where $g(\epsilon, A) = \begin{cases} (1+\epsilon)A & \text{if } A \ge 0 \\ (1-\epsilon)A & \text{if } A < 0 \end{cases}$

If $A > 0$ (action better than expected), we want to increase $\rho$, but $\text{clip}(\rho)A$ restricts the increase to at most $(1+\epsilon)A$. If $A < 0$ (action worse than expected), we want to decrease $\rho$, but the clip restricts the decrease to at most $(1-\epsilon)A$. This prevents the new policy from straying too far from the old one.

PPO pseudocode with GAE is essentially identical to the AC code, except the policy loss uses $\min(\rho_t A_t, \tilde{\rho}_t A_t)$ instead of $A_t \log \pi_\phi(a_t|s_t)$, and we perform multiple policy updates per rollout for increased sample efficiency.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(PPO for Diffusion Policies)</span></p>

PPO relies on computing the likelihood of a candidate action, which is difficult for diffusion policies. **DPPO** (PPO for diffusion models) treats each step of the diffusion process as a step of an "inner" MDP, nested inside the outer (main) MDP, and applies PPO to this combined system. **NCDPO** (noise-conditioned diffusion policy optimization) fixes the noise sequence for all diffusion steps and deterministically backpropagates gradients through the entire denoising chain.

**Simple Policy Optimization** (SPO) improves upon ratio clipping, offering stronger theoretical properties and better constraining the probability ratio within the trust region.

</div>

### Variational Maximum a Posteriori Policy Optimization (VMPO)

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(VMPO)</span></p>

**VMPO** is an on-policy extension of the off-policy MPO algorithm. It leverages the fact that if $\mathbb{E}_{p_{\pi_k}^\gamma(s)}[D_{KL}(\pi \| \pi_k)(s)] \le \delta$, then $\pi$ also satisfies the TV constraint (note the reversed KL compared to TRPO, which encourages mode-covering and improved robustness).

VMPO adopts an EM-type approach:

**E step:** Compute a non-parametric target distribution $\psi_{k+1}(s,a) = p_{\pi_k}^\gamma(s) \pi_k(a|s) w(s,a)$ where

$$
w(s,a) = \frac{\exp(A^{\pi_k}(s,a)/\lambda^*)}{ Z(\lambda^*)}, \quad \lambda^* = \arg\min_{\lambda \ge 0} \lambda \delta + \lambda \log Z(\lambda)
$$

**M step:** Project this target distribution back onto the space of parametric policies while satisfying the KL trust region constraint:

$$
\pi_{k+1} = \arg\max_\pi \mathbb{E}_{(s,a) \sim p_{\pi_k}^\gamma}\left[w(s,a) \log \pi(a|s)\right] \quad \text{s.t.} \quad \mathbb{E}_{p_{\pi_k}^\gamma}\left[D_{KL}(\psi_k \| \psi)(s)\right] \le \delta
$$

</div>

## Off-Policy Methods

In many cases, it is useful to train a policy using data collected from a distinct **behavior policy** $\pi_b(a|s)$ that is not the same as the **target policy** $\pi(a|s)$ being learned. This could be data from earlier trials, parallel workers, a **replay buffer**, or **demonstration data** from human experts. This is called **off-policy RL**, and can be much more sample efficient since it can use data from multiple sources.

The basic difficulty is that the target policy may want to try an action in a state that has not been experienced before, so there is no way to predict the outcome. We tackle this by assuming the target policy is not too different from the behavior policy (so that $\pi(a|s)/\pi_b(a|s)$ is bounded), using methods based on **importance sampling**.

### Policy Evaluation Using Importance Sampling

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Off-Policy Policy Evaluation)</span></p>

Given a dataset $\mathcal{D} = \lbrace \boldsymbol{\tau}^{(i)} \rbrace_{1 \le i \le n}$ of trajectories sampled according to a behavior policy $\pi_b$, we want to evaluate the performance of a target policy $\pi$; this is called **off-policy policy evaluation** (OPE). Using **importance sampling** (IS) to correct for the distributional mismatch:

$$
\hat{J}_{\text{IS}}(\pi) \triangleq \frac{1}{n} \sum_{i=1}^{n} \frac{p(\boldsymbol{\tau}^{(i)}|\pi)}{p(\boldsymbol{\tau}^{(i)}|\pi_b)} \sum_{t=0}^{T-1} \gamma^t r_t^{(i)}
$$

The basic difficulty is that the target policy may want to try an action in a state that has not been experienced before. We assume the target policy is not too different from the behavior policy so that the ratio $\pi(a|s)/\pi_b(a|s)$ is bounded.

</div>

The IS estimate is **unbiased**, provided that $p(\boldsymbol{\tau}|\pi_b) > 0$ whenever $p(\boldsymbol{\tau}|\pi) > 0$. The **importance ratio** simplifies because the transition dynamics and reward model cancel:

$$
\frac{p(\boldsymbol{\tau}|\pi)}{p(\boldsymbol{\tau}|\pi_b)} = \prod_{t=0}^{T-1} \frac{\pi(a_t|s_t)}{\pi_b(a_t|s_t)}
$$

Define the **per-step importance ratio** $\rho_t(\boldsymbol{\tau}) \triangleq \pi(a_t|s_t)/\pi_b(a_t|s_t)$. We can reduce the variance by noting that the reward $r_t$ is independent of the trajectory beyond time $t$, leading to a **per-decision importance sampling** variant:

$$
\hat{J}_{\text{PDIS}}(\pi) \triangleq \frac{1}{n} \sum_{i=1}^{n} \sum_{t=0}^{T-1} \prod_{t' \le t} \rho_{t'}(\boldsymbol{\tau}^{(i)}) \gamma^t r_t^{(i)}
$$

### Off-Policy Actor Critic Methods

#### Learning the Critic Using V-Trace

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(V-Trace)</span></p>

**V-trace** is a practical method for estimating the value function for a target policy using off-policy data. It extends the on-policy $n$-step target value by using truncated IS weights to bound the variance.

Define the truncated IS weights:

$$
c_t = \min\left(\bar{c}, \frac{\pi(a_t|s_t)}{\pi_b(a_t|s_t)}\right), \quad \rho_t = \min\left(\bar{\rho}, \frac{\pi(a_t|s_t)}{\pi_b(a_t|s_t)}\right)
$$

where $\bar{c}$ and $\bar{\rho}$ are hyperparameters. The V-trace target value for $V(s_i)$ is:

$$
v_i = V(s_i) + \sum_{t=i}^{i+n-1} \gamma^{t-i} \left(\prod_{t'=i}^{t-1} c_{t'}\right) \rho_t \delta_t
$$

where $\delta_t = (r_t + \gamma V(s_{t+1}) - V(s_t))$ is the TD error. These targets can be computed recursively:

$$
v_i = V(s_i) + \rho_i \delta_i + \gamma c_i (v_{i+1} - V(s_{i+1}))
$$

The product $c_i \ldots c_{t-1}$ (the "trace") measures how much a temporal difference $\delta_t$ at time $t$ impacts the update of the value function at earlier time $i$. The truncation parameter $\bar{c}$ reduces variance. For $\bar{\rho} = \infty$, we converge to $V^\pi$; for $\bar{\rho} \to 0$, we converge to $V^{\pi_b}$. In practice, $\bar{c} = 1$ and $\bar{\rho} = 1$ work best.

</div>

#### Learning the Actor

The **off-policy policy-gradient** uses a one-step IS correction ratio:

$$
\nabla_{\boldsymbol{\theta}} J_{\pi_b}(\pi_{\boldsymbol{\theta}}) \approx \mathbb{E}_{p_{\pi_b}^\gamma(s), \pi_b(a|s)}\left[\frac{\pi_{\boldsymbol{\theta}}(a|s)}{\pi_b(a|s)} \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a|s) Q_\pi(s,a)\right]
$$

In practice, we approximate $Q_\pi(s_t, a_t)$ by $q_t = r_t + \gamma v_{t+1}$, where $v_{t+1}$ is the V-trace estimate for state $s_{t+1}$. Using $V(s_t)$ as a baseline to reduce variance:

$$
\nabla J(\boldsymbol{\theta}) = \mathbb{E}_{t \sim \mathcal{D}}\left[\rho_t \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t|s_t)(r_t + \gamma v_t - V_{\boldsymbol{w}}(s_t))\right]
$$

We can also replace the 1-step IS-weighted TD error with an IS-weighted GAE value by modifying GAE to replace $A_t$ with $\rho_t A_t$.

#### IMPALA

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(IMPALA)</span></p>

**IMPALA** ("Importance Weighted Actor-Learning Architecture") uses shared parameters for the policy and value function (with different output heads), and adds an entropy bonus to ensure the policy remains stochastic. The objective is very similar to the on-policy actor-critic:

$$
\mathcal{L}(\boldsymbol{\phi}) = \mathbb{E}_{t \sim \mathcal{D}}\left[\lambda_{TD}(V_\phi(s_t) - v_t)^2 - \lambda_{PG} \rho_t A_t \log \pi_\phi(a_t|s_t) - \lambda_{ent} \mathbb{H}(\pi_\phi(\cdot|s_t))\right]
$$

The only difference from standard A2C is that we also store $\pi_b(a_t|s_t)$ in the dataset to compute $\rho_t$. IMPALA was able to train a single agent (using a shared CNN and LSTM) to play all 57 Atari games at a high level, outperforming the A3C method thanks to its off-policy corrections.

</div>

#### Off-Policy Learning with Deterministic Policies

Actor-critic methods that learn deterministic policies for continuous actions, based on the deterministic policy gradient (DPG) method, can work off-policy without the need for importance sampling correction.

#### PGQL: Combining Off-Policy Q-Learning with Policy Gradient

**PGQL** uses Q learning to learn from off-policy data in a replay buffer, and uses this to regularize the on-policy actor-critic learner.

### Off-Policy Policy Improvement Methods

Policy improvement methods (such as PPO) are often preferred to AC methods since they monotonically improve the objective. The key insight is that we can generalize the policy improvement lower bound to any reference policy:

$$
J(\pi) - J(\pi_k) \ge \frac{1}{1-\gamma} \mathbb{E}_{p_{\pi_\text{ref}}^\gamma(s) \pi_k(a|s)}\left[\frac{\pi(a|s)}{\pi_\text{ref}(a|s)} A^{\pi_k}(s,a)\right] - \frac{2\gamma C^{\pi, \pi_k}}{(1-\gamma)^2} \mathbb{E}_{p_{\pi_\text{ref}}^\gamma(s)}\left[\text{TV}(\pi(\cdot|s), \pi_\text{ref}(\cdot|s))\right]
$$

The reference policy can be any previous policy, or a convex combination $\pi_\text{ref} = \sum_{i=1}^{M} \nu_i \pi_{k-i}$, which allows sampling from the replay buffer.

To compute the advantage $A^{\pi_k}$ from off-policy data, we adapt V-trace:

$$
A_\text{trace}^{\pi_k}(s_t, a_t) = \delta_t + \sum_{j=0}^{n-1} \gamma^j \left(\prod_{m=1}^{j} c_{t+m}\right) \delta_{t+j}
$$

where $c_t = \min\left(\bar{c}, \frac{\pi_k(a_t|s_t)}{\pi_{k-i}(a_t|s_t)}\right)$ is the truncated importance sampling ratio.

We can derive **Generalized PPO** (an off-policy version of PPO):

$$
\pi_{k+1} = \arg\max_\pi \mathbb{E}_{t \sim \nu}\left[\mathbb{E}_{(s,a) \sim p_{\pi_{k-i}}^\gamma}\left[\min(\rho_{k-i}(s,a) A^{\pi_k}(s,a), \; \tilde{\rho}_{k-i}(s,a) A^{\pi_k}(s,a))\right]\right]
$$

where $\rho_{k-i}(s,a) = \frac{\pi(a|s)}{\pi_{k-i}(a|s)}$ and $\tilde{\rho}_{k-i}$ is its clipped version.

## Gradient-Free Policy Optimization

So far, we have focused on fitting differentiable parametric policies using methods based on the policy gradient theorem. However, gradient-based methods can get stuck in poor local optima and cannot be applied to non-differentiable policies (such as programs or functions with discrete latent variables).

We can consider other kinds of methods for policy learning based on **blackbox optimization**, aka **derivative-free optimization**. This includes techniques such as the **cross-entropy method** and **evolutionary strategies**. For example, **EGGROLL** ("Evolution Guided General Optimization via Low-rank Learning") provides a way to scale evolutionary strategies to very large models.

## RL as Inference

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Control as Inference)</span></p>

**Control as inference** (or **RL as inference**) is an approach to policy optimization that reduces it to probabilistic inference. The primary advantage is that it enables policy learning using off-policy data while avoiding the need for (potentially high-variance) importance sampling corrections. It also enables us to use the large toolkit of methods for probabilistic modeling and inference.

The core idea is based on a probabilistic model that augments an MDP with **optimality variables** $\mathcal{O}_t$, indicating whether the action at time $t$ is optimal or not:

$$
p(\mathcal{O}_t = 1 | s_t, a_t) \propto \exp(\eta^{-1} G(s_t, a_t))
$$

where $\eta > 0$ is a temperature parameter and $G(s,a)$ is some quality function such as $G(s,a) = R(s,a)$, $G(s,a) = Q(s,a)$, or $G(s,a) = A(s,a)$.

</div>

### Deterministic Case (Planning/Control as Inference)

In the deterministic case, where $p(s_{t+1}|s_t, a_t)$ is either 1 or 0, rather than learning a policy $\pi$ that maps states to actions, we just need to learn a plan (a specific sequence of actions $\boldsymbol{a}_{1:T}$). We want to maximize:

$$
p(\boldsymbol{\tau}|\mathcal{O}=1, \boldsymbol{a}_{1:T}) \propto \left[\prod_{t=1}^{T-1} p(s_{t+1}|s_t, a_t)\right]\left[\exp\left(\sum_{t=1}^{T} R(s_t, a_t)\right)\right]
$$

The MAP sequence of actions is the optimal **open loop plan**. Computing this trajectory is known as the **control as inference** problem and can be solved using model predictive control methods.

### Stochastic Case (Policy Learning as Variational Inference)

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Policy Learning as Variational Inference)</span></p>

In the stochastic case, we want to learn a policy $\pi$ that generates a distribution over optimal trajectories. We define the objective as $\log p(\mathcal{O} = 1|\pi) = \log \int p_\pi(\boldsymbol{\tau}) p(\mathcal{O}=1|\boldsymbol{\tau}) d\boldsymbol{\tau}$.

We introduce a variational distribution $q(\boldsymbol{\tau}) = p(s_1) \prod_t p(s_{t+1}|s_t, a_t) \pi_q(a_t|s_t)$ (note that we only introduce the variational distribution for the actions $\pi_q$, not for the dynamics, to avoid **optimism bias**).

This gives us the **evidence lower bound** (ELBO):

$$
\log p(\mathcal{O}=1|\pi) \ge J(p_\pi, q) \triangleq \underbrace{\log p_\pi(\mathcal{O}=1|\boldsymbol{\tau})}_\text{} - D_{KL}(q(\boldsymbol{\tau}) \| p_\pi(\boldsymbol{\tau}))
$$

$$
J(\pi_p, \pi_q) = \sum_{t=1}^{T} \mathbb{E}_q\left[\eta^{-1} G(s_t, a_t) - D_{KL}(\pi_q(\cdot|s_t) \| \pi_p(\cdot|s_t))\right]
$$

We define the policy learning task as maximizing the ELBO, subject to the constraints that $\pi_p$ and $\pi_q$ are distributions that integrate to 1 across actions for all states.

</div>

There are two main ways to optimize this: **EM control** and **KL control**.

### EM Control

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(EM Control)</span></p>

We optimize the ELBO using the **Expectation Maximization** (EM) algorithm, also called a **bound optimization** or MM (majorize/maximize) method.

**E step:** Maximize $J$ with respect to a non-parametric representation of the variational posterior $\pi_q$, while holding the parametric prior $\pi_p = \pi_{\theta_p}^{k-1}$ fixed. The optimal (non-parametric) solution is:

$$
\pi_q^k(a|s) = Z(s)^{-1} \pi_{\theta_p}^{k-1}(a|s) \exp(\eta^{-1} G(s,a))
$$

where $Z(s) = \int \pi_{\theta_p}^{k-1}(a|s) \exp(\eta^{-1} G(s,a)) da$ is the partition function.

**M step:** Maximize $J$ with respect to $\pi_{\theta_p}$, holding $\pi_q^k$ fixed. This reduces to a weighted maximum likelihood problem:

$$
J(\pi_q^k, \pi_{\theta_p}) = \mathbb{E}_{d_q(s) \pi_q^k(a|s)}\left[\log \pi_{\theta_p}(a|s)\right]
$$

</div>

### KL Control (Maximum Entropy RL)

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Maximum Entropy RL)</span></p>

In **KL control**, we only optimize the variational posterior $\pi_q$, holding the prior $\pi_p$ fixed (thus only an E step). We represent $\pi_q$ parametrically as $\pi_{\theta_q}$. If the prior $\pi_p$ is uniform and $G(s_t, a_t) = R(s_t, a_t)$, the ELBO becomes:

$$
\eta J(\pi_p, \pi_q) = \sum_{t=1}^{T} \mathbb{E}_q\left[R(s_t, a_t) - \eta H(\pi_q(\cdot|s_t))\right]
$$

where $-H(q) = D_{KL}(q \| \text{unif})$ is the negative entropy function. This is called the **maximum entropy RL** objective. It differs from the standard RL objective by the addition of the entropy regularizer on the policy, which provides a lower bound on the sum of expected rewards.

</div>

### Maximum a Posteriori Policy Optimization (MPO)

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(MPO)</span></p>

**MPO** is an instance of EM control, where $G(s,a) = Q(s,a)$, estimated using the retrace algorithm or a single-step Bellman update.

**E step:** Compute $q^k(a|s) = \frac{1}{Z(s)} \pi_{\theta_p}^{k-1}(a|s) \exp(\eta^{-1} G(s,a))$, where $Z(s)$ is approximated with Monte Carlo: $Z(s) \approx \frac{1}{M} \sum_{j=1}^{M} \exp(\eta^{-1} G(s, a_j))$ with $a_j \sim \pi_{\theta_p}^{k-1}(\cdot|s)$. The temperature $\eta$ is solved for by minimizing the dual of the constrained problem.

**M step:** Augment the weighted MLE objective with a prior centered at the previous parameters (a Gaussian with covariance $\lambda \mathbf{F}_k$, where $\mathbf{F}_k$ is the Fisher information matrix), giving:

$$
\max_{\theta_p} E_{d_q(s)}\left[E_{q(a|s)} \log \pi(a|s, \theta_p) - \lambda D_{KL}(\pi(a|s, \theta_k) \| \pi(a|s, \theta_p))\right]
$$

This can also be rewritten as a constrained optimization problem with a KL trust region: $E_{d_q(s)}[D_{KL}(\pi(a|s, \theta_k) \| \pi(a|s, \theta_p))] \le \epsilon_m$.

</div>

### SMC-PO

**SMC-PO** is a model-based version of MPO which uses Sequential Monte Carlo (SMC) to perform approximate inference in the E step. It samples from a distribution over optimal future trajectories starting from the current state $s_t$, using the current policy $\pi_{\theta_p}$ and dynamics model $\mathcal{T}(s'|s,a)$, to derive a non-parametric distribution over optimal actions.

### AWR and AWAC

The **Advantage Weighted Regression** (AWR) and **Advantage Weighted Actor Critic** (AWAC) are both EM control methods where $G(s,a) = A(s,a)$ (the advantage function). AWR estimates $V(s)$ using TD($\lambda$) and the value for the average of all previous policies, $\tilde{\pi}_{p^k} = \frac{1}{k} \sum_{j=0}^{k-1} \pi_{\theta_p}^j$. AWAC instead uses $G(s,a) = Q(s,a)$, estimated by TD(0).

### Soft Actor Critic (SAC)

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Soft Actor Critic)</span></p>

The **soft actor-critic** (SAC) algorithm is an off-policy actor-critic method based on the maximum entropy RL (KL control) framework. The variational posterior policy $\pi_q = \pi_{\theta_q}$ is parameterized, but the prior policy $\pi_p$ is fixed to the uniform distribution. SAC uses $G(s,a) = Q^\text{soft}(s,a)$, where the soft-Q function is defined below. SAC only has an E step (implemented with SGD), no M step.

**SAC objective:**

$$
J^\text{SAC}(\boldsymbol{\theta}) \triangleq \mathbb{E}_{p_{\pi_{\boldsymbol{\theta}}}^\gamma(s) \pi_{\boldsymbol{\theta}}(a|s)}\left[R(s,a) + \alpha \mathbb{H}(\pi_{\boldsymbol{\theta}}(\cdot|s))\right]
$$

The entropy term encourages exploration and makes the objective easier to optimize.

</div>

#### Policy Evaluation (Tabular Case)

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Soft Bellman Backup)</span></p>

Policy evaluation is performed by repeatedly applying a modified Bellman backup operator $\mathcal{T}^\pi$:

$$
\mathcal{T}^\pi Q(\boldsymbol{s}_t, \boldsymbol{a}_t) = r(\boldsymbol{s}_t, \boldsymbol{a}_t) + \gamma \mathbb{E}_{\boldsymbol{s}_{t+1} \sim p}\left[V(\boldsymbol{s}_{t+1})\right]
$$

where the **soft value function** is:

$$
V(\boldsymbol{s}_t) = \mathbb{E}_{\boldsymbol{a}_t \sim \pi}\left[Q(\boldsymbol{s}_t, \boldsymbol{a}_t) - \alpha \log \pi(\boldsymbol{a}_t|\boldsymbol{s}_t)\right]
$$

In the tabular case, the optimal policy is the softmax over Q-values:

$$
\pi^*(a|s) = \frac{\exp\left(\frac{Q^*(s,a)}{\alpha}\right)}{\sum_{a'} \exp\left(\frac{Q^*(s,a')}{\alpha}\right)}
$$

The optimal soft value function is then $V^*(s) = \alpha \log \sum_a \exp\left(\frac{Q^*(s,a)}{\alpha}\right)$.

</div>

#### Policy Evaluation (General Case)

For the non-tabular case, we hold the policy parameters $\pi$ fixed and optimize the Q function parameters $\boldsymbol{w}$ by minimizing:

$$
J_Q(\boldsymbol{w}) = \mathbb{E}_{(\boldsymbol{s}_t, \boldsymbol{a}_t, r_{t+1}, \boldsymbol{s}_{t+1}) \sim \mathcal{D}}\left[\frac{1}{2}(Q_{\boldsymbol{w}}(\boldsymbol{s}_t, \boldsymbol{a}_t) - y(r_{t+1}, \boldsymbol{s}_{t+1}))^2\right]
$$

where $y(r_{t+1}, \boldsymbol{s}_{t+1}) = r_{t+1} + \gamma V_{\overline{\boldsymbol{w}}}^\pi(\boldsymbol{s}_{t+1})$ is the frozen target value, with $V_{\overline{\boldsymbol{w}}}(\boldsymbol{s}_t) = \mathbb{E}_{\pi(\boldsymbol{a}_t|\boldsymbol{s}_t)}[Q_{\overline{\boldsymbol{w}}}(\boldsymbol{s}_t, \boldsymbol{a}_t) - \alpha \log \pi(\boldsymbol{a}_t|\boldsymbol{s}_t)]$.

To avoid overestimation bias, SAC fits two soft Q functions (inspired by clipped double Q learning in TD3) with targets:

$$
y(r_{t+1}, \boldsymbol{s}_{t+1}; \overline{\boldsymbol{w}}_{1:2}, \boldsymbol{\theta}) = r_{t+1} + \gamma\left[\min_{i=1,2} Q_{\overline{\boldsymbol{w}}_i}(\boldsymbol{s}_{t+1}, \tilde{\boldsymbol{a}}_{t+1}) - \alpha \log \pi_{\boldsymbol{\theta}}(\tilde{\boldsymbol{a}}_{t+1}|\boldsymbol{s}_{t+1})\right]
$$

where $\tilde{\boldsymbol{a}}_{t+1} \sim \pi_{\boldsymbol{\theta}}(\boldsymbol{s}_{t+1})$ is a sampled next action.

#### Policy Improvement

In the policy improvement step, we derive the new policy based on the soft Q function by softmaxing over possible actions, then project onto the policy class $\Pi$:

$$
\pi_\text{new} = \arg\min_{\pi' \in \Pi} D_{KL}\left(\pi'(\cdot|\boldsymbol{s}_t) \;\Big\|\; \frac{\exp(\frac{1}{\alpha} Q^{\pi_\text{old}}(\boldsymbol{s}_t, \cdot))}{Z^{\pi_\text{old}}(\boldsymbol{s}_t)}\right)
$$

For the non-tabular case, we optimize the parameters $\boldsymbol{\theta}$ by minimizing:

$$
J_\pi(\boldsymbol{\theta}) = \mathbb{E}_{\boldsymbol{s}_t \sim \mathcal{D}}\left[\mathbb{E}_{\boldsymbol{a}_t \sim \pi_{\boldsymbol{\theta}}}\left[\alpha \log \pi_{\boldsymbol{\theta}}(\boldsymbol{a}_t|\boldsymbol{s}_t) - Q_{\boldsymbol{w}}(\boldsymbol{s}_t, \boldsymbol{a}_t)\right]\right]
$$

We use the **reparameterization trick** (writing $\boldsymbol{a}_t = \boldsymbol{\mu}(\boldsymbol{s}_t) + \sigma^2 \boldsymbol{\epsilon}_t$ with $\boldsymbol{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$) instead of the REINFORCE estimator, since it has much lower variance. For discrete actions, we use the gumbel-softmax reparameterization or **SAC-Discrete**.

#### Adjusting the Temperature

The temperature parameter $\alpha$ can be automatically adjusted by optimizing:

$$
J(\alpha) = \mathbb{E}_{\boldsymbol{s}_t \sim \mathcal{D}, \boldsymbol{a}_t \sim \pi_{\boldsymbol{\theta}}}\left[-\alpha(\log \pi_{\boldsymbol{\theta}}(\boldsymbol{a}_t|\boldsymbol{s}_t) + \overline{H})\right]
$$

where $\overline{H}$ is the target entropy (a hyper-parameter). This is approximated by sampling actions from the replay buffer.

### Active Inference

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Active Inference)</span></p>

**Active inference** is closely related to control as inference. It was developed in the neuroscience community and is based on the **free energy principle** (FEP). The FEP is equivalent to using variational inference to perform state estimation (perception) and parameter estimation (learning) in a latent variable model $p(\boldsymbol{z}, \boldsymbol{o}|\boldsymbol{\theta})$ with hidden states $\boldsymbol{z}$, observations $\boldsymbol{o}$, and parameters $\boldsymbol{\theta}$.

The **variational free energy** (VFE) is:

$$
\mathcal{F}(\boldsymbol{o}|\boldsymbol{q}, \boldsymbol{\theta}) = D_{KL}(q(\boldsymbol{z}|\boldsymbol{o}, \boldsymbol{\theta}) \| p(\boldsymbol{z}|\boldsymbol{o}, \boldsymbol{\theta})) - \log p(\boldsymbol{o}|\boldsymbol{\theta}) \ge -\log p(\boldsymbol{o}|\boldsymbol{\theta})
$$

State estimation (perception) corresponds to $\min_q \mathcal{F}$, and parameter estimation (model fitting) corresponds to $\min_{\boldsymbol{\theta}} \mathcal{F}$, just as in the EM algorithm.

To extend this to decision making, we define the **expected free energy** (EFE):

$$
\mathcal{G}(\boldsymbol{a}) = \underbrace{\mathbb{E}_{q(\boldsymbol{o}|\boldsymbol{a})}\left[D_{KL}(q(\boldsymbol{z}|\boldsymbol{o}) \| p(\boldsymbol{z}|\boldsymbol{o}))\right]}_{\mathcal{G}_\text{epistemic}(\boldsymbol{a})} - \underbrace{\mathbb{E}_{q(\boldsymbol{o}|\boldsymbol{a})}\left[\log p(\boldsymbol{o}|\boldsymbol{\theta})\right]}_{\mathcal{G}_\text{extrinsic}(\boldsymbol{a})}
$$

The EFE decomposes into two terms:

* The **intrinsic value** (or **epistemic drive**): minimizing this encourages the agent to choose actions that maximize the mutual information between observations $\boldsymbol{o}$ and hidden states $\boldsymbol{z}$, reducing uncertainty (**epistemic foraging**).
* The **extrinsic value** (or **exploitation term**): maximizing this encourages the agent to choose actions that result in observations matching its prior. This prior can be related to a reward function by defining $p(\boldsymbol{o}) \propto e^{R(\boldsymbol{o})}$, encouraging goal-directed behavior.

The agent picks actions to reduce its uncertainty about hidden states (improving its predictive model $p_{\boldsymbol{\theta}}$), which in turn helps minimize the VFE of future observations. The agent acts so it becomes less surprised by what it sees --- achieving **homeostasis** with its environment. Training a policy network $\pi(\boldsymbol{a}|\boldsymbol{h}) = \arg\min_{\boldsymbol{a}} \mathcal{G}(\boldsymbol{a}|\boldsymbol{h})$ to amortize the cost of solving for the optimal action at each step is called **deep active inference**.

</div>

# Chapter 4: Model-Based RL

## Introduction

Model-free approaches to RL typically need a lot of interactions with the environment to achieve good performance. A promising approach to greater sample efficiency is **model-based RL** (MBRL). In the simplest approach, we first learn the state transition or dynamics model $p_S(s'|s,a)$ --- also called a **world model** --- and the reward function $R(s,a)$, using some offline trajectory data, and then use these models to compute a policy (e.g., using dynamic programming or model-free policy learning on simulated data).

The two-stage approach (learn model, then plan) can suffer from the usual problems of offline RL: the policy may query the model at states for which no data has been collected, so predictions can be unreliable. To get better results, we have to interleave the model learning and policy learning.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Two Approaches to MBRL)</span></p>

There are two main ways to perform MBRL:

1. **Decision-time planning** (or **model predictive control**): Use the model to choose the next action by searching over possible future trajectories. Score each trajectory, pick the action corresponding to the best one, take a step in the environment, and repeat. Discussed in Section 4.2.
2. **Background planning**: Use the current model and policy to rollout imaginary trajectories, and use this data (optionally in addition to empirical data) to improve the policy using model-free RL. Discussed in Section 4.3.

The advantage of decision-time planning is that it allows training a world model on reward-free data and then using any reward function at test time. The downside is that it is much slower. The two methods can be combined.

</div>

## Decision-Time (Online) Planning

In this section, we discuss how to choose the best action at each step based on planning forward from the current state using a known (or learned) world model. This is called **decision time planning** or "planning in the now."

### Receding Horizon Control

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Receding Horizon Control)</span></p>

In **receding horizon control** (RHC), we plan from the current state $s_t$ to a maximum fixed depth (horizon) of $d$. We then take the first action $a_t$ based on this future planning, observe the new state $s_{t+1}$, and then replan. This approach can be slow (requires search/optimization at each step), but it can give good results since it chooses an action tailored to the current state.

</div>

#### Forward Search

In **forward search**, we examine all possible transitions up to depth $d$ by starting from the current state and considering all possible actions, then all possible next states, etc. The resulting **search tree** has the reward associated with each edge. At the leaves, we compute the remaining reward-to-go using a value function $V(s)$. This takes $O((|\mathcal{S}| \times |\mathcal{A}|)^d)$ time.

#### Branch and Bound

In **branch and bound**, we prune suboptimal paths using a lower bound on the value function $\underline{V}(s)$ and an upper bound on the action value function $\overline{Q}(s,a)$. At each state node $s$, we examine actions in decreasing order of their upper bound. If $\overline{Q}(s,a)$ is less than the current best lower bound, we prune this branch. This can be significantly faster than forward search.

#### Sparse Sampling

**Sparse sampling** speeds up forward search by sampling a subset of $m$ possible next states for each action, giving complexity $O((m \times |\mathcal{A}|)^d)$, which is independent of $|\mathcal{S}|$.

#### Heuristic Search

In **heuristic search**, we start with a heuristic function $\overline{V}(s)$ to initialize $V(s)$. We perform $m$ Monte Carlo rollouts starting from the root node $s$, greedily picking actions and sampling next states, updating $V(s) = \max_a R(s,a) + \gamma \sum_{s'} p(s'|s,a)V(s')$ at each node. If the heuristic function is an upper bound on the optimal value function, it is called an **admissible heuristic**, and heuristic search is guaranteed to converge to the optimal value.

### Monte Carlo Tree Search (MCTS)

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Monte Carlo Tree Search)</span></p>

**Monte Carlo tree search** (MCTS) is a receding horizon control procedure. Given the root node $s_t$, we perform $m$ Monte Carlo rollouts to estimate $Q(s_t, a)$, then return the best action $\arg\max_a Q(s_t, a)$ or the action distribution $\text{softmax}(Q(s_t, a))$. A rollout from state $s$ proceeds as follows:

1. **Action selection:** If $s$ is unvisited, initialize $N(s,a) = 0$ and $Q(s,a) = 0$, and return $U(s)$ as the value. Otherwise, use the **Upper Confidence Tree** (**UCT**) heuristic (based on UCB) to select actions:

$$
a = \arg\max_{a \in \mathcal{A}(s)} Q(s,a) + c\sqrt{\frac{\log N(s)}{N(s,a)}}
$$

where $N(s) = \sum_a N(s,a)$ is the total visit count to $s$, and $c$ is an exploration bonus scaling term. If we have a predictor or prior $P(s,a)$, we can use: $a = \arg\max_{a \in \mathcal{A}(s)} Q(s,a) + c\left(P(s,a) \frac{\sqrt{N(s)}}{1+N(s,a)}\right)$

2. **Expansion:** Sample the next state $s' \sim p(s'|s,a)$.

3. **Rollout:** Recursively estimate $u = U(s')$ using MCTS from that node. At some depth, stop and use the value function: $u = r + \gamma v(s')$.

4. **Backup:** Update the Q function using a running average: $Q(s,a) \leftarrow Q(s,a) + \frac{1}{N(s,a)}(u - Q(s,a))$

</div>

#### MCTS for 2-Player Zero-Sum Games: AlphaGo, AlphaGoZero, and AlphaZero

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(AlphaGo and AlphaZero)</span></p>

MCTS can be applied to games. In two-player, zero-sum symmetric games, the agent models the opponent using its own policy with roles reversed (**self-play**).

**AlphaGo** used MCTS (with self-play) combined with a neural network computing $(v_s, \boldsymbol{\pi}^s) = f(s; \boldsymbol{\theta})$, where $v_s$ is the expected game outcome and $\boldsymbol{\pi}_s$ is the policy. It was the first AI to beat a human grandmaster at Go. **AlphaGoZero** was trained entirely using RL and self-play (no human data) and significantly outperformed AlphaGo. **AlphaZero** generalized this to play expert-level Go, chess, and shogi without any domain knowledge.

The policy/value network is trained by optimizing the actor-critic loss:

$$
\mathcal{L}(\boldsymbol{\theta}) = \mathbb{E}_{(s, \boldsymbol{\pi}_s^\text{MCTS}, V^\text{MCTS}(s)) \sim \mathcal{D}}\left[(V^\text{MCTS}(s) - V_{\boldsymbol{\theta}}(s))^2 - \sum_a \boldsymbol{\pi}_s^\text{MCTS}(a) \log \boldsymbol{\pi}_{\boldsymbol{\theta}}(a|s)\right]
$$

where $\boldsymbol{\pi}_s^\text{MCTS}(a) = [N(s,a)/(\sum_b N(s,b))]^{1/\tau}$ is the MCTS-derived action distribution.

</div>

#### MCTS with Learned World Model: MuZero and EfficientZero

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(MuZero)</span></p>

**MuZero** learns a world model by training a latent representation (embedding function) $\boldsymbol{z}_t = e_\phi(\boldsymbol{o}_t)$ and a corresponding latent dynamics (and reward) model $(\boldsymbol{z}_t, r_t) = M_w(\boldsymbol{z}_t, a_t)$. The model is trained to predict the immediate reward, future reward (value), and the optimal policy, where the optimal policy is computed using MCTS. The loss augments the actor-critic loss with a reward prediction term:

$$
\mathcal{L}(\boldsymbol{\theta}, \boldsymbol{w}, \boldsymbol{\phi}) = (V^\text{MCTS}(z) - V_{\boldsymbol{\theta}}(e_\phi(\boldsymbol{o})))^2 - \sum_a \boldsymbol{\pi}_z^\text{MCTS}(a) \log \boldsymbol{\pi}_{\boldsymbol{\theta}}(a|e_\phi(\boldsymbol{o})) + (r - M_w^r(e_\phi(\boldsymbol{o}), a_t))^2
$$

MuZero was applied to Go, Chess, Shogi, and Atari.

</div>

**Stochastic MuZero** extends MuZero to stochastic environments (e.g., 2048 and Backgammon). **Sampled MuZero** allows for large and/or continuous action spaces. **Gumbel MuZero** proposes a better policy improvement algorithm based on sampling actions without replacement. **MuZero Unplugged** applies the **Reanalyse** algorithm (MCTS-based policy and value improvement) to offline trajectories with a learned world model.

**Efficient Zero** extends MuZero by adding a self-prediction loss $(\boldsymbol{z}_{t+1} - M_w^s(\boldsymbol{z}_t, a_t))^2$ to help train the world model, and replaces the empirical sum of rewards with an LSTM model that predicts the value prefix. **Efficient Zero V2** extends this to work with continuous actions using sampling-based Gumbel search.

#### MCTS in Belief Space

**BetaZero** performs MCTS in belief space. The current state is represented by a belief state $b_t$, which is passed to the network to generate an initial policy proposal $\pi_{\boldsymbol{\theta}}(a|b)$ and value function $v_{\boldsymbol{\theta}}(b)$. Rollouts expand nodes by sampling hidden states $s \sim b$, next hidden states $s' \sim T(s'|s,a)$, observations $o \sim O(s')$, rewards $r \sim R(s,a,s')$, and new belief states $b' = \text{Update}(b,a,o)$ using e.g. a particle filter.

### Sequential Monte Carlo (SMC) for Online Planning

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(SMC Planning)</span></p>

**SMC-PO** (Sequential Monte Carlo Policy Optimisation) is based on the "RL as inference" framework. The goal is to sample trajectories that are likely to be high scoring from the distribution:

$$
q(\tau) \propto d_q(s_0) \prod_{t \ge 0} \mathcal{T}(s_{t+1}|s_t, a_t) \pi(a_t|s_t, \boldsymbol{\theta}_i) \exp\left(\frac{A(s_t, a_t)}{\eta}\right)
$$

where $A(s_t, a_t) = Q(s_t, a_t) - V(s_t) \approx r_t + V(s_{t+1}) - V(s_t)$ is the advantage function and $\eta$ is a temperature. We sample trajectories using **SMC** (a generalization of **particle filtering**): at each step we propose a new particle according to $\beta_i(\tau_t | \tau_{1:t-1}) \propto \hat{\mathcal{T}}(s_{t+1}|s_t, a_t)\pi(a_t|s_t, \boldsymbol{\theta}_i)$, then weight it according to $w(\tau_{1:t}) \propto w(\tau_{1:t-1}) \cdot \exp(A(s_t, a_t)/\eta)$. Optionally resample particles when effective sample size becomes too small. The resulting empirical distribution over actions can be used to estimate the next best action.

This framework is a special case of **twisted SMC**, where the advantage function plays the role of a "twist" function summarizing expected future rewards.

</div>

### Model Predictive Control (MPC), aka Open Loop Planning

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Model Predictive Control)</span></p>

**Model predictive control** (MPC) is an **open loop** version of receding horizon control. At each step, it solves for the sequence of subsequent actions that is most likely to achieve high expected reward:

$$
\boldsymbol{a}_{t:t+d}^* = \arg\max_{\boldsymbol{a}_{t:t+d}} \mathbb{E}_{s_{t+1:t+d} \sim \mathcal{T}(\cdot|s_t, \boldsymbol{a}_{t:t+d})}\left[\sum_{h=0}^{d} R(s_{t+h}, a_{t+h}) + \hat{V}(s_{t+d+1})\right]
$$

where $\mathcal{T}$ is the dynamics model. It then returns $a_t^*$ as the best action, takes a step, and replans. The future actions are chosen without knowing the future states ("open loop"). This can be suboptimal in stochastic environments but is fast and popular for continuous control with deterministic dynamics.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Suboptimality of Open-Loop Planning)</span></p>

Open-loop planning can be suboptimal in stochastic environments. For example, consider 9 states, 2 actions (up/down), planning horizon $d=2$, with a stochastic transition from $s_1$. All four open-loop plans (up-up, up-down, down-up, down-down) achieve expected reward of at most 20. However, closed-loop planning (which senses the resulting state after each action) can guarantee a reward of 30 by adapting the second action based on the observed state.

</div>

#### Trajectory Optimization

If the dynamics are deterministic, MPC becomes a **trajectory optimization** problem:

$$
\max_{a_{1:d}, s_{2:d}} \sum_{t=1}^{d} \gamma^t R(s_t, a_t) \quad \text{s.t.} \quad s_{t+1} = \mathcal{T}(s_t, a_t)
$$

#### LQR

If the system dynamics are linear and the reward function is quadratic, the optimal action sequence can be computed exactly using the **linear quadratic regulator** (LQR). If the model is nonlinear, we can use **differential dynamic programming** (DDP), which linearizes the system dynamics around states on a reference trajectory and forms a locally quadratic approximation of the reward function.

#### Random Shooting

For general nonlinear models (such as neural networks), **random shooting** picks a sequence of random actions, evaluates the reward for each trajectory, and picks the best.

#### CEM

The **cross-entropy method** (CEM) improves upon random shooting. We start with a multivariate Gaussian $\mathcal{N}(\boldsymbol{\mu}_0, \boldsymbol{\Sigma}_0)$ representing a distribution over possible actions, sample from it, evaluate all proposals, pick the top $K$, refit the Gaussian to these top $K$, and repeat.

#### MPPI

The **model predictive path integral** (MPPI) approach is a version of CEM where the initial mean of the Gaussian at step $t$ is computed based on shifting $\hat{\boldsymbol{\mu}}_{t-1}$ forward by one step (the previous reference trajectory). It has been applied to robot control by training the model using the **Dagger** method (alternating between supervised learning and deployment).

#### GP-MPC

**GP-MPC** combines a Gaussian process dynamics model with MPC. It computes a Gaussian approximation to the future state trajectory given a candidate action trajectory, by moment matching, and uses this to deterministically compute the expected reward and its gradient. This can be seen as a generalization of the **LQG** method. An earlier method, **PILCO**, learns a policy by maximizing the expected reward from rollouts but is less data-efficient since it only updates the GP model at the end of a trajectory.

## Background (Offline) Planning

Decision-time planning can be slow. We can amortize the planning process into a reactive policy by using the model to generate synthetic trajectories "in the background" (while executing the current policy), and using this imaginary data to train the policy. This is called **background planning**.

### A Game-Theoretic Perspective on MBRL

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(MBRL as a Game)</span></p>

We define the value of a policy $\pi$ when rolled out in some model $M'$ as $J(\pi, M') = \mathbb{E}_{M', \pi}\left[\sum_{t=0}^{\infty} \gamma^t R(s_t)\right]$. The loss of a model $\hat{M}$ given a state-action distribution $\mu(s,a)$ is $\ell(\hat{M}, \mu) = \mathbb{E}_{(s,a) \sim \mu}\left[D_{KL}(M_\text{env}(\cdot|s,a) \| \hat{M}(\cdot|s,a))\right]$.

MBRL can be defined as a two-player general-sum game:

$$
\underbrace{\max_\pi J(\pi, \hat{M})}_\text{policy player}, \quad \underbrace{\min_{\hat{M}} \ell(\hat{M}, \mu_{M_\text{env}}^{\pi})}_\text{model player}
$$

A **Nash equilibrium** $(\pi, \hat{M})$ satisfies that the model is accurate when predicting the rollouts from $\pi$, and $\pi$ cannot be improved when evaluated in $\hat{M}$. The Nash equilibrium policy $\pi$ is near optimal with respect to the real world.

</div>

Two approaches to finding the equilibrium:

* **Policy as leader (PAL):** First fit the model $\hat{M}_{k+1} = \arg\min_{\hat{M}} \ell(\hat{M}, \mu_{M_\text{env}}^{\pi_k})$ on data from $\pi_k$, then improve the policy using conservative updates (e.g., natural actor-critic, TRPO) on "imaginary" model rollouts from $\hat{M}_{k+1}$.
* **Model as leader (MAL):** First fully optimize the policy $\pi_{k+1} = \arg\max_\pi J(\pi, \hat{M}_k)$, then update the model using a gradient step on data from $\pi_{k+1}$ in the real world. A conservative model update by mixing data from earlier models (**data aggregation**) helps build a more global model.

### Dyna

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Dyna)</span></p>

**Dyna** trains a policy and world model in parallel, with the key difference from pure MBRL being that the policy is also trained on real data, not just imaginary data. At each step:

$$
\pi_{k+1} = \pi_k + \eta_\pi \nabla_\pi J(\pi_k, \hat{D}_k \cup \mathcal{D}_k)
$$

where $\mathcal{D}_k$ is data from the real environment and $\hat{D}_k = \text{rollout}(\pi_k, M_k)$ is imaginary data from the model. This makes Dyna a hybrid model-free and model-based RL method.

At each step: (1) collect new data from the environment and add to a real replay buffer; (2) do an off-policy update on real data; (3) update the world model on real data; (4) simulate imaginary data (starting from a previously visited state, rolling out the current policy in the learned model); (5) add imaginary data to an imaginary replay buffer and do an on-policy update.

</div>

**Tabular Dyna** (Dyna-Q) assumes a deterministic tabular world model $s' = M(s,a)$. Sampling a single step from a previously visited state is then equivalent to experience replay (we can think of ER as a kind of non-parametric world model).

**Dyna with function approximation** extends this using the MBRL agent code, training the policy on both real and imaginary data. **MBPO** (model-based policy optimization) uses Dyna with the off-policy SAC method and an **ensemble of DNNs** for the world model (from the **PETS** approach). One should gradually increase the fraction of real data to avoid suboptimal performance due to model limitations.

## World Models

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Types of World Models)</span></p>

World models can be trained to predict future observations (generative WMs) or just future rewards/values and/or future latent embeddings (non-generative / non-reconstructive WMs). Once trained, models can be used for decision-time planning, background planning, or as an auxiliary signal (e.g., intrinsic curiosity).

| | Background planning | Online planning | Exploration |
| --- | --- | --- | --- |
| **Observation prediction** | Dyna, DreamerV3, IRIS, Delta-IRIS, Diamond | CEM: PlaNet, Rnd shooting: TDM | SPR |
| **Value + self prediction** | DreamingV2, AIS | MCTS: MuZero, EfficientZero, CEM: TD-MPC | BYOL-Explore |

</div>

### World Models Trained to Predict Observation Targets

These models learn $\mathcal{T}(\boldsymbol{s}'|\boldsymbol{s}, a)$ to generate imaginary trajectories:

$$
p(\boldsymbol{s}_{t+1:T}, \boldsymbol{r}_{t+1:T}, \boldsymbol{a}_{t:T-1}|\boldsymbol{s}_t) = \prod_{i=t}^{T-1} \pi(\boldsymbol{a}_i|\boldsymbol{s}_i) \mathcal{T}(\boldsymbol{s}_{i+1}|\boldsymbol{s}_i, \boldsymbol{a}_i) R(r_{i+1}|\boldsymbol{s}_i, \boldsymbol{a}_i)
$$

If the state space is high-dimensional (e.g., images, denoted $\boldsymbol{o}$), we can learn $\mathcal{T}(\boldsymbol{o}'|\boldsymbol{o}, a)$ using standard techniques for conditional image generation such as diffusion. This kind of world model is equivalent to an action-conditional version of a **video generative model** (such as **Sora**, **Veo-3**, **seedance**, etc.). Examples include the **Diamond** method, **Genie2**, and **GAIA-1**.

Note that these methods are trained to predict the entire observation vector, even if we use latent variables. (This is what we mean by "generative world model".) One big disadvantage is that observations may contain irrelevant or distractor variables not necessary for task performance. In addition, such models are often slow to use, and there may be a distribution shift in the observation process between train and test time.

#### Generative World Models without Latent Variables

The simplest approach is to define $\mathcal{T}(\boldsymbol{o}'|\boldsymbol{o}, a)$ as a conditional generative model over states. If the observed states are low-dimensional vectors (e.g., proprioceptive states), we can use transformers (see e.g., the **Transformer Dynamics Model**).

In some cases, the dimensions of the state vector $\boldsymbol{s}$ represent distinct variables, and the joint Markov transition matrix $p(\boldsymbol{s}'|\boldsymbol{s}, a)$ has conditional independence properties which can be represented as a sparse graphical model. This is called a **factored MDP**.

#### Generative World Models with Latent Variables

Some methods use latent variables as part of their world model. This can improve the speed of generating imaginary futures and can provide a compact latent space as input to a policy.

Let $\boldsymbol{z}_t$ denote the latent (or hidden) state at time $t$; this can be a discrete or continuous variable (or vector of variables). The generative model has the form of a controlled HMM:

$$
p(\boldsymbol{o}_{t+1:T}, \boldsymbol{z}_{t+1:T}, \boldsymbol{r}_{t+1:T}, \boldsymbol{a}_{t:T-1}|\boldsymbol{z}_t) = \prod_{i=t}^{T-1} \pi(\boldsymbol{a}_i|\boldsymbol{z}_i) \mathcal{M}(\boldsymbol{z}_{i+1}|\boldsymbol{z}_i, \boldsymbol{a}_i) R(r_i|\boldsymbol{z}_i) D(\boldsymbol{o}_i|\boldsymbol{z}_i)
$$

where $p(\boldsymbol{o}_t|\boldsymbol{z}_t) = D(\boldsymbol{o}_t|\boldsymbol{z}_t)$ is the decoder or likelihood function, $\mathcal{M}(\boldsymbol{z}'|\boldsymbol{z}, \boldsymbol{a})$ is the dynamics in latent space, and $\pi(\boldsymbol{a}_t|\boldsymbol{z}_t)$ is the policy in latent space.

The world model is usually trained by maximizing the marginal likelihood of the observed outputs given an action sequence. Computing the marginal likelihood requires marginalizing over the hidden variables $\boldsymbol{z}_{t+1:T}$. To make this computationally tractable, it is common to use amortized variational inference, in which we train an encoder network $p(\boldsymbol{z}_t|\boldsymbol{o}_t) = E(\boldsymbol{z}_t|\boldsymbol{o}_t)$ to approximate the posterior over the latents.

#### Example: Dreamer

The **Dreamer** approach and its extensions (DreamerV2, DreamerV3, DreamerV4) are all based on background planning, in which the policy is trained on imaginary trajectories generated by a latent variable world model. (Dreamer is based on an earlier approach called **PlaNet**, which used MPC instead of background planning.)

In Dreamer, the stochastic dynamic latent variables are replaced by deterministic dynamic latent variables $\boldsymbol{h}_t$, since this makes the model easier to train. ($\boldsymbol{h}_t$ acts like the posterior over the hidden state at time $t-1$; this is also the prior predictive belief state before we see $\boldsymbol{o}_t$.) A static stochastic variable $\boldsymbol{z}_t$ is generated for each time step and acts like a "random effect" to help generate the observations, without relying on $\boldsymbol{h}_t$ to store all the necessary information. Dreamer defines the following functions:

* A hidden dynamics (sequence) model: $\boldsymbol{h}_{t+1} = \mathcal{U}(\boldsymbol{h}_t, \boldsymbol{a}_t, \boldsymbol{z}_t)$
* A latent state conditional prior: $\hat{\boldsymbol{z}}_t \sim P(\hat{\boldsymbol{z}}_t|\boldsymbol{h}_t)$
* A latent state decoder (observation predictor): $\hat{\boldsymbol{o}}_t \sim D(\hat{\boldsymbol{o}}_t|\boldsymbol{h}_t, \hat{\boldsymbol{z}}_t)$
* A reward predictor: $\hat{r}_t \sim R(\hat{r}_t|\boldsymbol{h}_t, \hat{\boldsymbol{z}}_t)$
* A latent state encoder: $\boldsymbol{z}_t \sim E(\boldsymbol{z}_t|\boldsymbol{h}_t, \boldsymbol{o}_t)$
* A policy function: $\boldsymbol{a}_t \sim \pi(\boldsymbol{a}_t|\boldsymbol{h}_t)$

The world model loss has the form

$$
\mathcal{L}^{\text{WM}} = \mathbb{E}_{q(\boldsymbol{z}_{1:T})} \left[ \sum_{t=1}^{T} \beta_o \mathcal{L}^o(\boldsymbol{o}_t, \hat{\boldsymbol{o}}_t) + \beta_z \mathcal{L}^z(\boldsymbol{z}_t, \hat{\boldsymbol{z}}_t) \right]
$$

where the $\beta$ terms are different weights for each loss, and $q$ is the posterior over the latents:

$$
q(\boldsymbol{z}_{1:T}|\boldsymbol{h}_0, \boldsymbol{o}_{1:T}, \boldsymbol{a}_{1:T}) = \prod_{t=1}^{T} E(\boldsymbol{z}_t|\boldsymbol{h}_t, \boldsymbol{o}_t) \delta(\boldsymbol{h}_t - \mathcal{U}(\boldsymbol{h}_{t-1}, \boldsymbol{a}_{t-1}, \boldsymbol{z}_{t-1}))
$$

The loss terms correspond to the observation prediction cross entropy and the posterior-to-prior KL penalty:

$$
\mathcal{L}^o = -\ln D(\boldsymbol{o}_t|\boldsymbol{z}_t, \boldsymbol{h}_t)
$$

$$
\mathcal{L}^z = D_{\text{KL}}(E(\boldsymbol{z}_t|\boldsymbol{h}_t, \boldsymbol{o}_t) \| P(\boldsymbol{z}_t|\boldsymbol{h}_t))
$$

In addition to the world model loss, the actor-critic losses are:

$$
\mathcal{L}^{\text{critic}} = \sum_{t=1}^{T} (V(\boldsymbol{h}_t) - \text{sg}(G_t^\lambda))^2
$$

$$
\mathcal{L}^{\text{actor}} = -\sum_{t=1}^{T} \text{sg}((G_t^\lambda - V(\boldsymbol{h}_t))) \log \pi(\boldsymbol{a}_t|\boldsymbol{h}_t)
$$

where $G_t^\lambda$ is the GAE estimate of the reward to go:

$$
G_t^\lambda = r_t + \gamma \left( (1-\lambda) V(\boldsymbol{h}_t) + \lambda G_{t+1}^\lambda \right)
$$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Dreamer Extensions)</span></p>

* **DreamerV2** adds categorical (discrete) latents and KL balancing between prior and posterior estimates. It was the first imagination-based agent to outperform humans in Atari games.
* **DayDreamer** applies DreamerV2 to real robots.
* **DreamerV3** builds upon DreamerV2 using various tricks (such as symlog encodings for the reward, critic, and decoder) to enable more stable optimization and domain-independent choice of hyper-parameters. It was the first method to create diamonds in Minecraft without requiring human demonstration data.
* **DreamerV4** uses a standard conditional latent video diffusion model $p(\boldsymbol{z}_t|\boldsymbol{z}_{t-c:t-1}, a_t)$ with a transformer backbone, combined with an autoencoder. The WM is trained offline on 2500 hours of reward-free but action-labeled videos. The key difference from prior Dreamer models is that the world model is more powerful and trained offline on diverse human-collected data.

Other variants include **TransDreamer** and **STORM** (replacing the RNN with transformers), the **S4WM** method (using S4 models), and **DreamingV2** (replacing the generative loss with a non-generative self-prediction loss).

</div>

#### Example: IRIS

The **IRIS** method ("Imagination with auto-Regression over an Inner Speech") follows the MBRL paradigm: (1) learn a world model using real data $D_r$ and then generate imaginary rollouts $D_i$ using the WM, and (2) learn the policy given $D_i$ and collect new data $D_r'$ for learning. In the model learning stage, IRIS learns a discrete latent encoding using the VQ-VAE method, and then fits a transformer dynamics model to the latent codes. In the policy learning stage, it uses actor-critic methods. The **Delta-IRIS** method extends this by training the model to only predict the delta between neighboring frames.

#### Code World Models

It has become popular to represent the world model $p(s'|s, a)$ using code, such as Python. This is called a **code world model**. It is possible to learn such models from trajectory data using LLMs.

#### Partial Observation Prediction

Predicting all pixels in an image may waste capacity and may distract the agent from the important bits. A natural alternative is to just predict some function of the observations, rather than the entire observation vector. This is known as a **partial world model**. One way to implement this is to impose an information bottleneck between the latent state and the observed variables, to prevent the agent focusing on irrelevant observational details (see e.g., the **denoised MDP** method).

### World Models Trained to Predict Other Targets

These are models that are not necessarily able to predict all the future observations. They are often still (conditional) generative models, but they are **lossy** models because they do not capture all the details of the data.

#### The Objective Mismatch Problem

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Small Agent, Big World)</span></p>

A simple agent may not be able to capture the full complexity of the true environment; this is called the **"small agent, big world"** problem. When the agent's model is misspecified (i.e., it cannot represent the true world model), the agent will train its model to reduce state (or observation) prediction error. However, not all features of the state are useful for planning. For example, a dynamics model with limited representational capacity may choose to focus on predicting background pixels rather than control-relevant features like small moving objects. This is called **"objective mismatch"** --- the discrepancy between the way a model is usually trained (to predict the observations) vs the way its representation is used for control.

</div>

#### Observation Prediction

We consider a modeling paradigm where we learn an encoder $\boldsymbol{z} = \phi(\boldsymbol{o})$; a dynamics model in latent space $\boldsymbol{z}' = \mathcal{M}(\boldsymbol{z}, \boldsymbol{a})$, for future prediction; and an update model in latent space $\boldsymbol{z}' = \mathcal{U}(\boldsymbol{z}, a, \boldsymbol{o})$, for belief state tracking.

A representation $\phi$ satisfies the **OP** (observation prediction) criterion if:

$$
\exists D \;\text{s.t.}\; p^*(\boldsymbol{o}'|\boldsymbol{h}, a) = D(\boldsymbol{o}'|\phi(\boldsymbol{h}), a) \;\forall \boldsymbol{h}, a
$$

We must also satisfy the recurrent encoder condition (**Rec**):

$$
\exists \mathcal{U} \;\text{s.t.}\; \phi(\boldsymbol{h}') = \mathcal{U}(\phi(\boldsymbol{h}), a, \boldsymbol{o}') \;\forall \boldsymbol{h}, a, \boldsymbol{o}'
$$

where $\mathcal{U}$ is the update operator. Belief state updates (as in a POMDP) satisfy this property. Furthermore, belief states are a sufficient statistic to satisfy the OP condition. The drawback is that in general it is very hard to predict future observations in high-dimensional settings like images.

#### Reward Prediction

We can also train the latent encoder to predict the reward. The **RP** (reward prediction) condition is:

$$
\exists R \;\text{s.t.}\; \mathbb{E}_{R^*}[r|\boldsymbol{h}, a] = \mathbb{E}_R[r|\phi(\boldsymbol{h}), a] \;\forall \boldsymbol{h}, a
$$

A representation that satisfies ZP and RP is enough to satisfy value equivalence (sufficiency for $Q^*$).

#### Value Prediction

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Value Equivalence and State Abstraction)</span></p>

Let $\boldsymbol{h}_t = (\boldsymbol{h}_{t-1}, \boldsymbol{a}_{t-1}, r_{t-1}, \boldsymbol{o}_t)$ be all the visible data (history) at time $t$, and let $\boldsymbol{z}_t = \phi(\boldsymbol{h}_t)$ be a latent representation (compressed encoding) of this history, where $\phi$ is called an encoder or a **state abstraction** function. We train the policy $\boldsymbol{a}_t = \pi(\boldsymbol{z}_t)$ in the usual way.

An optimal representation $\boldsymbol{z}_t = \phi(\boldsymbol{h}_t)$ is a sufficient statistic for the optimal action-value function $Q^*$. It satisfies the **value equivalence** principle: two states $s_1$ and $s_2$ are value equivalent (given a policy) if $V^\pi(s_1) = V^\pi(s_2)$. In particular, if the representation is optimal, it will satisfy value equivalence w.r.t. the optimal policy, i.e., if $\phi(\boldsymbol{h}_i) = \phi(\boldsymbol{h}_j)$ then $Q^*(\boldsymbol{h}_i, a) = Q^*(\boldsymbol{h}_j, a)$.

We can train such a representation function by using its output $\boldsymbol{z} = \phi(\boldsymbol{h})$ as input to the Q function or to the policy. (We call such a loss **VP**, for value prediction.)

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Bisimulation)</span></p>

There is a stronger property than value equivalence called **bisimulation**: two states $s_1$ and $s_2$ are bisimilar if $P(s'|s_1, a) \approx P(s'|s_2, a)$ and $R(s_1, a) = R(s_2, a)$. From this, we can derive a continuous measure called the **bisimulation metric**. This has the advantage (compared to value equivalence) of being policy independent, but the disadvantage that it can be harder to compute. Recent progress on computationally efficient methods includes **MICo** and **KSMe**.

</div>

#### Policy Prediction

The value function and reward losses may be too sparse to learn efficiently. Although self-prediction loss can help somewhat, it does not use any extra information from the environment as feedback. When using MCTS, it is possible to compute what the policy should be for a given state, and this can be used as a prediction target for the reactive policy $a_t = \pi(\boldsymbol{z}_t)$, which in turn can be used as a feedback signal for the latent state. This method is used by MuZero and EfficientZero.

#### Self Prediction (Self Distillation)

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Expected $\boldsymbol{z}$ Prediction)</span></p>

In problems with sparse reward, predicting the value or policy may not provide enough of a feedback signal to learn quickly. We can augment the training with a **self-prediction** loss where we train $\phi$ to ensure the following condition holds:

$$
\exists M \;\text{s.t.}\; \mathbb{E}_{M^*}[\boldsymbol{z}'|\boldsymbol{h}, a] = \mathbb{E}_M[\boldsymbol{z}'|\phi(\boldsymbol{h}), a] \;\forall \boldsymbol{h}, a
$$

where the LHS is the predicted mean of the next latent state under the true model, and the RHS is the predicted mean under the learned dynamics model. We call this the **EZP** (expected $\boldsymbol{z}$ prediction).

</div>

#### Avoiding Self-Prediction Collapse Using Frozen Targets

A trivial way to minimize the self-prediction loss is to learn an embedding that maps everything to a constant vector, say $E(\boldsymbol{h}) = \boldsymbol{0}$, in which case $\boldsymbol{z}_{t+1}$ will be trivial for the dynamics model $M$ to predict. This is called **representational collapse**. We can provably prevent collapse (at least for linear encoders) by using a frozen target network. The auxiliary loss is:

$$
\mathcal{L}_{\text{EZP}}(\boldsymbol{\phi}, \boldsymbol{\theta}; \boldsymbol{h}, a, \boldsymbol{h}') = \|M_\phi(E_\phi(\boldsymbol{h}, a)) - \text{sg}(E_{\overline{\boldsymbol{\phi}}}(\boldsymbol{h}'))\|_2^2
$$

where

$$
\overline{\boldsymbol{\phi}}_t = \rho \boldsymbol{\phi}_t + (1 - \rho) \overline{\boldsymbol{\phi}}_{t-1}
$$

is the exponential moving average (EMA) of the encoder weights $\boldsymbol{\phi}$. (If we use a frozen (old) copy of the weights instead, this is called a detached network.) This approach means the "goalposts" (the target representations) evolve slowly and consistently over time, guided by the progress of the encoder and predictor.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Self-Supervised Learning Methods Using Frozen Targets)</span></p>

The above approach is used in many papers, such as **BYOL** (Bootstrap Your Own Latent), **SimSiam**, **DinoV2**, **JEPA** (Joint Embedding Prediction Architecture), **I-JEPA**, **V-JEPA**, **Image World Models**, **Predictron**, **Value Prediction Networks**, **Self Predictive Representations** (SPR), **Efficient Zero**, etc.

An alternative to predicting the encoding of the next frame is to mask out the current frame in a random way and use this as the target. These are **self-supervised learning** (SSL) methods.

Minimizing the self-prediction objective (with the stop-gradient term) has been proven to be theoretically sound for the case of linear encoders with a fixed policy (called BYOL-$\pi$): the encoder converges to learn the singular vectors of the transition matrix induced by the policy.

</div>

#### Avoiding Self-Prediction Collapse Using Information-Theoretic Regularization

An alternative way to avoid the latent collapse problem is to add regularization terms that try to maximize the information content in $\boldsymbol{z}_t$ and $\boldsymbol{z}_{t+1}$, while also minimizing the prediction error:

$$
J(\boldsymbol{\phi}) = E_{\boldsymbol{o}_t, \boldsymbol{a}_t, \boldsymbol{o}_{t+1}, \epsilon_t} \left( \|\boldsymbol{z}_{t+1} - \hat{\boldsymbol{z}}_{t+1}\|_2^2 - \lambda I(\boldsymbol{z}_t) - \lambda I(\boldsymbol{z}_{t+1}) \right)
$$

where $\boldsymbol{z}_t = E(\boldsymbol{o}_t; \boldsymbol{\phi})$, $\boldsymbol{z}_{t+1} = E(\boldsymbol{o}_{t+1}; \boldsymbol{\phi})$, $\hat{\boldsymbol{z}}_{t+1} = \mathcal{M}(\boldsymbol{z}_t, \boldsymbol{a}_t, \epsilon_t; \boldsymbol{\theta})$.

Various methods have been proposed to approximate the information content $I(\boldsymbol{z}_t)$, mostly based on some function of the outer product matrix $\sum_t \boldsymbol{z}_t \boldsymbol{z}_t^\top$, which captures second-order moments.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Barlow Twins and VICReg)</span></p>

The **Barlow Twins** method aims to make the cross-correlation matrix between the representations of two embeddings as close to the identity matrix as possible. This simultaneously encourages similar representations for similar inputs while decorrelating the features within the representation.

The **VICReg** method (Variance-Invariance-Covariance Regularization) uses three loss terms: one to maintain variance in the representations (avoid collapse), one to decorrelate the different variables in the latent vector (reduce redundancy), and one to make them invariant to data augmentations (by bringing different views closer together in embedding space). The VICReg approach can be thought of as constrastive across dimensions, whereas standard contrastive methods like SimCLR are contrastive across samples.

One can also use information-theoretic regularizers when training generative models that predict future observables. The idea is to create an **information bottleneck** that remembers as little about the inputs (past) as possible, while still being able to predict the future.

</div>

#### Preventing Self-Prediction Collapse Using Game-Theoretic Approaches

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(CTRL: Closed-Loop Transcription)</span></p>

An arguably more theoretically sound approach to self-supervised learning, known as **CTRL** (closed-loop transcription), tries to solve a minmax game (similar to a GAN), rather than optimize a (regularized) objective. There are two players: the encoder $E$, and the generator or decoder $G$. The generator minimizes distortion measured in embedding space:

$$
D(\mathbf{Z}, \mathbf{Z}') = \|\mathbf{Z} - \mathbf{Z}'\|_F^2
$$

where $\mathbf{Z} = E(\mathbf{X})$ is the embedding of a batch of inputs and $\mathbf{Z}' = E(G(E(\mathbf{X})))$ is the embedding of their reconstructions. The encoder should also try to maximize its own **rate** (information content):

$$
R(\mathbf{Z}) = \log \det(C(\mathbf{Z}))
$$

where $C(\mathbf{Z}) = \mathbf{Z}^\top \mathbf{Z}$ is the covariance matrix. Minimizing this single term achieves decorrelation (forcing off-diagonal elements of the covariance to zero) and variance preservation (encouraging non-zero diagonal elements), very similar in spirit to VICReg.

The CTRL game is:

$$
\min_G D(\mathbf{Z}, \mathbf{Z}'; E, G)
$$

$$
\max_E R(\mathbf{Z}; E) - \lambda D(\mathbf{Z}, \mathbf{Z}'; E, G)
$$

The encoder maximizes the **rate reduction** $\Delta R(\mathbf{Z}, \mathbf{Z}'; E, G) = R(\mathbf{Z}; E) - R(\mathbf{Z}'; E, G) \ge 0$, and the new game becomes:

$$
\min_G \Delta R(\mathbf{Z}, \mathbf{Z}'; E, G) + \lambda D(\mathbf{Z}, \mathbf{Z}'; E, G)
$$

$$
\max_E \Delta R(\mathbf{Z}, \mathbf{Z}'; E, G) - \lambda D(\mathbf{Z}, \mathbf{Z}'; E, G)
$$

</div>

#### Example: JEPA

The **JEPA** (Joint Embedding Prediction Architecture) approach to world modeling jointly embeds the current and following observations, computing $\boldsymbol{z}_t = E(\boldsymbol{o}_t)$ and $\boldsymbol{z}_{t+1} = E(\boldsymbol{o}_{t+1})$, and then comparing the actual latent embedding $\boldsymbol{z}_{t+1}$ to its prediction $\boldsymbol{z}_{t+1}' = \mathcal{M}(\boldsymbol{z}_t, a_t; \epsilon_t)$, where $\epsilon_t$ is a random noise source and $M$ is the deterministic world model. The encoder is then trained to minimize the difference between $\boldsymbol{z}_t$ and $\boldsymbol{z}_t'$.

To prevent collapse, two classes of methods have been considered: (1) using a frozen EMA version of the encoder, and (2) adding regularization terms that try to maximize the information content in $\boldsymbol{z}_t$ and $\boldsymbol{z}_{t+1}$ while minimizing prediction error. These can also be combined.

JEPA also leverages the fact that the encoder is a low-dimensional embedding of the input, and the predictor is a shallow network, to create an information bottleneck. The **I-JEPA** method is designed for images, and the **V-JEPA** method is designed for videos (also training on masked versions of the inputs to create a harder learning problem).

#### Example: DinoWM

In the case where the observations are high-dimensional (such as images), it is natural to use a pre-trained representation $\boldsymbol{z}_t = \phi(\boldsymbol{o}_t)$ as input to the world model (or policy). The representation function $\phi$ can be pretrained on a large dataset using a non-reconstructive loss, such as the **DINOv2** method. Although this can sometimes give gains (as in the **DinoWM** and **Dino-World** papers), in other cases better results are obtained by training the representation from scratch. The performance is highly dependent on the similarity or differences between the pretraining distribution and the agent's distribution.

#### Example: TD-MPC

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(TD-MPC2)</span></p>

**TD-MPC2** is an extension of **TD-MPC** that learns the following functions:

* Encoder: $\boldsymbol{e}_t = E(\boldsymbol{o}_t)$
* Latent dynamics (for rollouts): $\boldsymbol{z}_t' = \mathcal{M}(\boldsymbol{z}_{t-1}, \boldsymbol{a}_t)$
* Latent update (after each observation): $\boldsymbol{z}_t = \mathcal{U}(\boldsymbol{z}_{t-1}, \boldsymbol{e}_t, \boldsymbol{a}_t) = \boldsymbol{e}_t$
* Reward: $\hat{r}_t = R(\boldsymbol{z}_t, \boldsymbol{a}_t)$
* Value: $\hat{q}_t = Q(\boldsymbol{z}_t, \boldsymbol{a}_t)$
* Policy prior: $\hat{\boldsymbol{a}}_t = \pi_{\text{prior}}(\boldsymbol{z}_{t-1})$

The model is trained using the following VP+ZP loss applied to trajectories sampled from the replay buffer:

$$
\mathcal{L}(\boldsymbol{\theta}) = \mathbb{E}_{(\boldsymbol{o}, \boldsymbol{a}, r, \boldsymbol{o}')_{0:H} \sim \mathcal{B}} \left[ \sum_{t=0}^{H} \lambda^t \left( \|\boldsymbol{z}_t' - \text{sg}(E(\boldsymbol{o}_t'))\|_2^2 + \text{CE}(\hat{r}_t, r_t) + \text{CE}(\hat{q}_t, q_t) \right) \right]
$$

We use cross-entropy loss on a discretized representation of the reward and Q value in a log-transformed space. The target value for the Q function update is:

$$
q_t = r_t + \gamma \overline{Q}(\boldsymbol{z}_t', \pi_{\text{prior}}(\boldsymbol{z}_t'))
$$

where $\overline{Q}$ is the EMA for the Q function.

The policy is trained using the SAC objective on imaginary rollouts in latent space:

$$
\mathcal{L}_\pi(\boldsymbol{\theta}) = \mathbb{E}_{(\boldsymbol{o}, \boldsymbol{a})_{0:H} \sim \mathcal{B}} \left[ \sum_{t=0}^{H} \lambda^t \left[ \alpha Q(\boldsymbol{z}_t, \pi_{\text{prior}}(\boldsymbol{z}_t)) - \beta \,\mathbb{H}(\pi_{\text{prior}}(\cdot|\boldsymbol{z}_t)) \right] \right], \; \boldsymbol{z}_{t+1} = \mathcal{M}(\boldsymbol{z}_t, \boldsymbol{a}_t), \; \boldsymbol{z}_0 = E(\boldsymbol{o}_0)
$$

This policy is used as a proposal (prior), in conjunction with the MPPI trajectory planning method to select actions at run time.

</div>

#### Example: BYOL

In **BYOL** (Build Your Own Latents), they use the ZP and VP loss. The computation graph is slightly simpler than Dreamer due to the lack of stochastic latents.

**BYOL-Explore** extends BYOL by using the self-prediction error to define an intrinsic reward. This encourages the agent to explore states where the model is uncertain.

#### Example: Imagination-Augmented Agents

**Imagination-augmented agents** train a model to predict future states and rewards, and then use the hidden states of this model as additional context for a policy-based learning method. This can help overcome partial observability.

### World Models Trained to Help Planning

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Differentiable Planning)</span></p>

One solution to the objective mismatch problem is to use **differentiable planning**, in which we combine model learning and policy learning together, and train them jointly end-to-end, rather than in an alternating fashion. In particular, we can solve:

$$
\min_{\hat{M}, Q} \mathbb{E}_{(s,a,s') \sim \mathcal{D}} \left[ (R(s,a) + \gamma V(s') - Q(s,a))^2 \right]
$$

where $s' = \hat{M}(s, a)$ is the learned dynamics model, subject to the constraint that the value function is derived from the model using

$$
V(s) = \text{argmax}_{a(0:K)} \mathbb{E}_{\hat{M}} \left[ \sum_{k=0}^{K-1} \gamma^k R(s_k, a_k) + \gamma^K V(s_K) | S_0 = s \right]
$$

This bilevel optimization problem was first proposed in the **Value Iteration Network** paper, and extended in the **TreeQN** paper. The **D-TSN** (differentiable tree search network) is similar to TreeQN but constructs a best-first search tree rather than a fixed depth tree.

</div>

### Dealing with Model Errors and Uncertainty

The model-as-leader approach (which trains a new policy in imagination at each inner iteration while gradually improving the model in the outer loop) will converge to the optimal policy, provided the model converges to the true model (or one that is value equivalent to it). Nevertheless, models will inevitably have errors, and it can be useful for the policy learning to be aware of this.

#### Avoiding Compounding Errors in Rollouts

In MBRL, we have to rollout imaginary trajectories to use for training the policy. It makes intuitive sense to start from a previously visited real-world state, since the model will likely be reliable there. We should start rollouts from different points along each real trajectory, to ensure good state coverage. However, if we roll out too far from a previously seen state, the trajectories are likely to become less realistic, due to **compounding errors** from the model.

The **MBPO** method uses short rollouts (inside Dyna) to prevent compounding error. Another approach is to learn a trajectory-level dynamics model instead of a single-step model, e.g., using diffusion to train $p(s_{t+1:t+H}|s_t, a_{t:t+H-1})$, and using this inside an MPC loop.

If the model is able to predict a reliable distribution over future states, then we can leverage this uncertainty estimate to compute an estimate of the expected reward. For example, **PILCO** uses Gaussian processes as the world model and analytically derives the expected reward over trajectories as a function of policy parameters. One can also combine the MPO algorithm for continuous control with **uncertainty sets** on the dynamics to learn a policy that optimizes for a worst-case expected return objective.

#### Unified Model and Planning Variational Lower Bound

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Mismatched No More)</span></p>

The **Mismatched No More** (MNM) method solves the objective mismatch problem. They define an optimality variable $p(O=1|\tau) = R(\tau) = \sum_{t=1}^{\infty} \gamma^t R(s_t, a_t)$. This gives rise to the following variational lower bound on the log probability of optimality:

$$
\log p(O=1) \ge \mathbb{E}_{Q(\tau)} [\log R(\tau) + \log P(\tau) - \log Q(\tau)]
$$

where $P(\tau)$ is the distribution over trajectories induced by policy applied to the true world model, $P(\tau) = \mu(s_0) \prod_{t=0}^{\infty} M(s_{t+1}|s_t, a_t) \pi(a_t|s_t)$, and $Q(\tau)$ is the distribution over trajectories using the estimated world model, $Q(\tau) = \mu(s_0) \prod_{t=0}^{\infty} \hat{M}(s_{t+1}|s_t, a_t) \pi(a_t|s_t)$. They then maximize this bound w.r.t. $\pi$ and $\hat{M}$.

The method is extended in **Aligned Latent Models** by learning a latent encoder $\hat{E}(\boldsymbol{z}_t|\boldsymbol{o}_t)$ as well as latent dynamics $\hat{M}(\boldsymbol{z}_{t+1}|\boldsymbol{z}_t, a_t)$, similar to other self-predictive methods.

</div>

#### Dynamically Switching between MFRL and MBRL

One problem with the model-based approach is that if the model is of limited capacity, or if it learns to model "irrelevant" aspects of the environment, then any MBRL method may be dominated by a MFRL method that directly optimizes the true expected reward. A safer approach is to use a model-based policy only when the agent is confident it is better, but otherwise to fall back to a model-free policy. This is the strategy proposed in the **Unified RL** method.

### Exploration for Learning World Models

When using MBRL, the need for diverse data becomes even more important, to ensure we learn the correct underlying world model (which is then used to train the policy, or for online planning).

One popular approach is to use posterior sampling RL, which applies Thompson sampling to the MDP parameters (i.e., the world model). If we are in the reward-free setting, we can view the problem of learning a world model as similar to the scientist's job of trying to create a **causal model** of the world, which can explain the effects of actions (interventions). This requires designing and carrying out experiments to collect informative trajectories for model fitting. Recently it has become popular to use LLMs to help with this problem; this can be thought of as an **AI scientist**.

## Beyond One-Step Models: Predictive Representations

The "world models" described in Section 4.4 are **one-step models** of the form $p(s'|s, a)$, or $p(z'|z, a)$ for $z = \phi(s)$, where $\phi$ is a state-abstraction function. However, such models are problematic when it comes to predicting many kinds of future events, such as "will a car pull in front of me?" or "when will it start raining?", since it is hard to predict exactly when these events will occur, and these events may correspond to many different "ground states".

In principle we can roll out many possible long-term futures, and apply some abstraction function to the resulting generated trajectories to extract features of interest, and thus derive a predictive model of the form $p(t', \phi(s_{t+1:t'})|s_t, \pi)$, where $t'$ is the random duration of the sampled trajectory, and $\phi$ maps from state trajectories to features. However, it would be more efficient if we could directly predict this distribution without having to know the value of $t'$, and without having to predict all the details of all the intermediate future states. These are called **predictive representations**, and are a compromise between standard model-based RL and model-free RL.

### General Value Functions

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(General Value Function)</span></p>

The value function is based on predicting the sum of expected discounted future rewards. But the reward is just one possible signal of interest we can extract from the environment. We can generalize this by considering a **cumulant** $C_t \in \mathbb{R}$, which is some scalar of interest derived from the state or observation (e.g., did a loud bang just occur? is there a tree visible in the image?). The **general value function** or **GVF** is defined as:

$$
V^{\pi, C, \gamma}(s) = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t C(s_{t+1}) | s_0 = s, a_{0:\infty} \sim \pi \right]
$$

If $C(s_{t+1}) = R_{t+1}$, this reduces to the value function. If we define the cumulant to be the observation vector, then the GVF will learn to predict future observations at multiple time scales; this is called **nexting**.

</div>

Predicting the GVFs for multiple cumulants can be useful as an auxiliary task while solving the main task (e.g., as a form of auxiliary input to the policy, or just to "densify" the training signal). One can use an approach (based on meta-gradients) to learn which cumulants are worth predicting. In the inner loop, the model $f$ predicts the policy $\pi_t$ and value function $V_t$, as usual, and also predicts the GVFs $\boldsymbol{y}_t$ for the specified cumulants. In the outer loop, the model $g$ learns to extract the cumulants and their discounts given future observations; this is called the question network, denoted by $(\boldsymbol{c}_t, \boldsymbol{\gamma}_t) = g_{\boldsymbol{\eta}}(\boldsymbol{o}_{t+1:t+j})$.

### Successor Representations

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Successor Representation)</span></p>

Consider a variant of GVF where the cumulant corresponds to a state occupancy vector $C_{\bar{s}}(s_{t+1}) = \mathbb{I}(s_{t+1} = \bar{s})$, which provides a dense feedback signal. Computing this for each possible state $\bar{s}$ gives us the **successor representation** or **SR**:

$$
M^\pi(s, \bar{s}) = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t \mathbb{I}(s_{t+1} = \bar{s}) | S_0 = s \right]
$$

If we define the policy-dependent state-transition matrix by

$$
T^\pi(s, s') = \sum_a \pi(a|s) T(s'|s, a)
$$

then the SR matrix can be rewritten as

$$
\mathbf{M}^\pi = \sum_{t=0}^{\infty} \gamma^t [\mathbf{T}^\pi]^{t+1} = \mathbf{T}^\pi (\mathbf{I} - \gamma \mathbf{T}^\pi)^{-1}
$$

Thus the SR replaces information about individual transitions with their cumulants, just as the value function replaces individual rewards with the reward-to-go.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Successor Representation)</span></p>

Like the value function, the SR obeys a Bellman equation:

$$
M^\pi(s, \bar{s}) = \sum_a \pi(a|s) \sum_{s'} T(s'|s, a) \left(\mathbb{I}(s' = \bar{s}) + \gamma M^\pi(s', \bar{s})\right) = \mathbb{E}\left[\mathbb{I}(s' = \bar{s}) + \gamma M^\pi(s', \bar{s})\right]
$$

Hence we can learn an SR using a TD update of the form

$$
M^\pi(s, \bar{s}) \leftarrow M^\pi(s, \bar{s}) + \eta \underbrace{\left(\mathbb{I}(s' = \bar{s}) + \gamma M^\pi(s', \bar{s}) - M^\pi(s, \bar{s})\right)}_{\delta}
$$

where $s'$ is the next state sampled from $T(s'|s, a)$. Compare this to the value-function TD update: $V^\pi(s) \leftarrow V^\pi(s) + \eta \underbrace{(R(s') + \gamma V^\pi(s') - V^\pi(s))}_{\delta}$.

With an SR, we can easily compute the value function for any reward function (given a fixed policy):

$$
V^{R, \pi} = \sum_{\bar{s}} M^\pi(s, \bar{s}) R(\bar{s})
$$

</div>

We can also make a version of SR that depends on the action as well as the state:

$$
M^\pi(s, a, \bar{s}) = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t \mathbb{I}(s_{t+1} = \bar{s}) | s_0 = s, a_0 = a, a_{1:\infty} \sim \pi \right]
$$

$$
= \mathbb{E}\left[\mathbb{I}(s' = \bar{s}) + \gamma M^\pi(s', a', \bar{s}) | s_0 = s, a_0 = a, a_{1:\infty} \sim \pi \right]
$$

This gives rise to a TD update of the form

$$
M^\pi(s, a, \bar{s}) \leftarrow M^\pi(s, a, \bar{s}) + \eta \underbrace{\left(\mathbb{I}(s' = \bar{s}) + \gamma M^\pi(s', a', \bar{s}) - M^\pi(s, a, \bar{s})\right)}_{\delta}
$$

where $s'$ is the next state sampled from $T(s'|s, a)$ and $a'$ is the next action sampled from $\pi(s')$. Compare this to the (on-policy) SARSA update: $Q^\pi(s,a) \leftarrow Q^\pi(s,a) + \eta \underbrace{(R(s') + \gamma Q^\pi(s', a') - Q^\pi(s,a))}_{\delta}$.

From an SR, we can compute the state-action value function for any reward function:

$$
Q^{R, \pi}(s, a) = \sum_{\bar{s}} M^\pi(s, a, \bar{s}) R(\bar{s})
$$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Advantages and Limitations of SR)</span></p>

The SR has the computational advantages of model-free RL (no need to do explicit planning or rollouts in order to compute the optimal action), but also the flexibility of model-based RL (we can easily change the reward function without having to learn a new value function). This latter property makes SR particularly well suited to problems that use intrinsic reward, which often changes depending on the information state of the agent.

Unfortunately, the SR is limited in two key ways: (1) it assumes a finite, discrete state space; (2) it depends on a given policy.

</div>

### Successor Features

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Successor Features)</span></p>

SRs require defining expectations or distributions over the entire future state vector, which can be problematic in high-dimensional and continuous spaces. **Successor features** (SFs) generalize SRs by working with features $\boldsymbol{\phi}(s)$ instead of primitive states. If we define the cumulant to be $C(s_{t+1}) = \boldsymbol{\phi}(s_{t+1})$, we get:

$$
\boldsymbol{\psi}^{\pi, \boldsymbol{\phi}}(s) = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t \boldsymbol{\phi}(s_{t+1}) | s_0 = s, a_{0:\infty} \sim \pi \right]
$$

In matrix form (by analogy to the SR matrix):

$$
\boldsymbol{\Psi}^\pi = \sum_{t=0}^{\infty} (\gamma \mathbf{T}^\pi)^t \boldsymbol{\Phi} = (\mathbf{I} - \gamma \mathbf{T}^\pi)^{-1} \boldsymbol{\Phi}
$$

where $\boldsymbol{\Phi}$ is the $S \times D$ matrix of features for each state. SFs also obey a Bellman equation:

$$
\boldsymbol{\psi}(s) = \mathbb{E}\left[\boldsymbol{\phi}(s') + \gamma \boldsymbol{\psi}(s')\right]
$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Value Function from Successor Features)</span></p>

If we assume the reward function can be written as $R(s, \boldsymbol{w}) = \boldsymbol{\phi}(s)^\top \boldsymbol{w}$, then we can derive the value function for any reward as follows:

$$
V^{\pi, \boldsymbol{w}}(s) = \mathbb{E}\left[R(s_1) + \gamma R(s_2) + \cdots | s_0 = s \right] = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t \boldsymbol{\phi}(s_{t+1}) | s_0 = s \right]^\top \boldsymbol{w} = \boldsymbol{\psi}^\pi(s)^\top \boldsymbol{w}
$$

Similarly, we can define an action-conditioned version of SF:

$$
\boldsymbol{\psi}^{\pi, \boldsymbol{\phi}}(s, a) = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t \boldsymbol{\phi}(s_{t+1}) | s_0 = s, a_0 = a, a_{1:\infty} \sim \pi \right] = \mathbb{E}\left[\boldsymbol{\phi}(s') + \gamma \boldsymbol{\psi}(s', a')\right]
$$

We can learn this using a TD rule:

$$
\boldsymbol{\psi}^\pi(s, a) \leftarrow \boldsymbol{\psi}^\pi(s, a) + \eta \underbrace{\left(\boldsymbol{\phi}(s') + \gamma \boldsymbol{\psi}^\pi(s', a') - \boldsymbol{\psi}^\pi(s, a)\right)}_{\delta}
$$

And we can use it to derive a state-action value function: $Q^{\pi, \boldsymbol{w}}(s) = \boldsymbol{\psi}^\pi(s, a)^\top \boldsymbol{w}$.

This allows us to define multiple $Q$ functions (and hence policies) just by changing the weight vector $\boldsymbol{w}$.

</div>

#### Generalized Policy Improvement

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Generalized Policy Improvement)</span></p>

So far, we have discussed how to compute the value function for a new reward function but using the SFs from an existing known policy. **Generalized Policy Improvement** or **GPI** discusses how to create a new policy that is better than an existing set of policies.

Suppose we have learned a set of $N$ (potentially optimal) policies $\pi_i$ and their corresponding SFs $\boldsymbol{\psi}^{\pi_i}$ for maximizing rewards defined by $\boldsymbol{w}_i$. When presented with a new task $\boldsymbol{w}_{\text{new}}$, we can compute a new policy using GPI as follows:

$$
a^*(s; \boldsymbol{w}_{\text{new}}) = \text{argmax}_a \max_i Q^{\pi_i}(s, a, \boldsymbol{w}_{\text{new}}) = \text{argmax}_a \max_i \boldsymbol{\psi}^{\pi_i}(s, a)^\top \boldsymbol{w}_{\text{new}}
$$

If $\boldsymbol{w}_{\text{new}}$ is in the span of the training tasks (i.e., there exist weights $\alpha_i$ such that $\boldsymbol{w}_{\text{new}} = \sum_i \alpha_i \boldsymbol{w}_i$), then the GPI theorem states that $\pi(a|s) = \mathbb{I}(a = a^*(s, \boldsymbol{w}_{\text{new}}))$ will perform at least as well as any of the existing policies.

</div>

Note that GPI is a model-free approach to computing a new policy, based on an existing library of policies. One can also leverage a (possibly approximate) world model to learn better policies that can outperform the library of existing policies by performing more decision-time search.

#### Option Keyboard

One limitation of GPI is that it requires that the reward function, and the resulting policy, be defined in terms of a fixed weight vector $\boldsymbol{w}_{\text{new}}$, where the preference over features is constant over time. However, for some tasks we might want to initially avoid a feature or state and then later move towards it. To solve this, the **option keyboard** was introduced, in which the weight vector for a task can be computed dynamically in a state-dependent way, using $\boldsymbol{w}_s = g(s, \boldsymbol{w}_{\text{new}})$. Actions can then be chosen as:

$$
a^*(s; \boldsymbol{w}_{\text{new}}) = \text{argmax}_a \max_i \boldsymbol{\psi}^{\pi_i}(s, a)^\top \boldsymbol{w}_s
$$

Thus $\boldsymbol{w}_s$ induces a set of policies that are active for a period of time, similar to playing a chord on a piano.

#### Learning SFs

A key question when using SFs is how to learn the cumulants or state-features $\boldsymbol{\phi}(s)$. Various approaches have been suggested, including leveraging meta-gradients, image reconstruction, maximizing the mutual information between task encodings and the cumulants that an agent experiences when pursuing that task, and reward prediction methods. The cumulants are encouraged to satisfy the linear reward constraint by minimizing

$$
\mathcal{L}_r = \|r - \boldsymbol{\phi}_\theta(s)^\top \boldsymbol{w}\|_2^2
$$

Once the cumulant function is known, we have to learn the corresponding SF. The standard approach learns a different SF for every policy, which is limiting. **Universal Successor Feature Approximators** (USFAs) take as input a policy encoding $\boldsymbol{z}_w$, representing a policy $\pi_w$ (typically we set $\boldsymbol{z}_w = \boldsymbol{w}$):

$$
\boldsymbol{\psi}^{\pi_w}(s, a) = \boldsymbol{\psi}_\theta(s, a, \boldsymbol{z}_w)
$$

The GPI update then becomes

$$
a^*(s; \boldsymbol{w}_{\text{new}}) = \text{argmax}_a \max_{\boldsymbol{z}_w} \boldsymbol{\psi}_\theta(s, a, \boldsymbol{z}_w)^\top \boldsymbol{w}_{\text{new}}
$$

so we replace the discrete over a finite number of policies, $\max_i$, with a continuous optimization problem $\max_{\boldsymbol{z}_w}$, to be solved per state.

If we want to learn the policies and SFs at the same time, we can optimize the following losses in parallel:

$$
\mathcal{L}_Q = \|\boldsymbol{\psi}_\theta(s, a, \boldsymbol{z}_w)^\top \boldsymbol{w} - \boldsymbol{y}_Q\|, \quad \boldsymbol{y}_Q = R(s'; w) + \gamma \boldsymbol{\psi}_\theta(s', a^*, \boldsymbol{z}_w)^\top \boldsymbol{w}
$$

$$
\mathcal{L}_\psi = \|\boldsymbol{\psi}_\theta(s, a, \boldsymbol{z}_w) - \boldsymbol{y}_\psi\|, \quad \boldsymbol{y}_\psi = \boldsymbol{\phi}(s') + \gamma \boldsymbol{\psi}_\theta(s', a^*, \boldsymbol{z}_w)
$$

where $a^* = \text{argmax}_{a'} \boldsymbol{\psi}_\theta(s', a', \boldsymbol{z}_w)^\top \boldsymbol{w}$. The **Successor Features Keyboard** can learn the policy, the SFs, and the task encoding $\boldsymbol{z}_w$ all simultaneously.

#### Choosing the Tasks

A key advantage of SFs is that they provide a way to compute a value function and policy for any given reward, as specified by a task-specific weight vector $\boldsymbol{w}$. But how do we choose these tasks? One approach is to sample $\boldsymbol{w}$ from a distribution at the start of each task, to encourage the agent to learn to explore different parts of the state space (as specified by the feature function $\boldsymbol{\phi}$). This can be extended by adding an intrinsic reward that favors exploring parts of the state space that are surprising (i.e., which induce high entropy).

### Successor Measures

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Successor Measure)</span></p>

The **successor model** (also called a $\gamma$-model, or **geometric horizon model**) is a probabilistic extension of SR. Rather than just working with expectations, we can simulate future state trajectories by sampling. This allows us to generalize SR to work with continuous states and actions.

The basic idea is to define the cumulant as the $k$-step conditional distribution $C(s_{k+1}) = P(s_{k+1} = \bar{s} | s_0 = s, \pi)$. (Compare this to the SR cumulant, which is $C(s_{k+1}) = \mathbb{I}(s_{k+1} = \bar{s})$.) The **successor measure** (SM) is then defined as

$$
\boldsymbol{\mu}^\pi(\bar{s}|s) = (1-\gamma) \sum_{t=0}^{\infty} \gamma^t P(s_{t+1} = \bar{s} | s_0 = s)
$$

where the $1-\gamma$ term ensures that $\boldsymbol{\mu}^\pi$ integrates to 1. (Recall that $\sum_{t=0}^{\infty} \gamma^t = \frac{1}{1-\gamma}$ for $\gamma < 1$.) In the tabular setting, the SM is just the normalized SR, since $\boldsymbol{\mu}^\pi(\bar{s}|s) = (1-\gamma) M^\pi(s, \bar{s})$.

$\boldsymbol{\mu}^\pi(\bar{s}|s)$ tells us the probability that $\bar{s}$ can be reached from $s$ within a horizon determined by $\gamma$ when following $\pi$, even though we don't know exactly when we will reach $\bar{s}$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Successor Measure)</span></p>

SMs obey a Bellman-like recursion:

$$
\boldsymbol{\mu}^\pi(\bar{s}|s) = \mathbb{E}\left[(1-\gamma) T(\bar{s}|s, a) + \gamma \boldsymbol{\mu}^\pi(\bar{s}|s')\right]
$$

We can use this to perform policy evaluation by computing

$$
V^\pi(s) = \frac{1}{1-\gamma} \mathbb{E}_{\boldsymbol{\mu}^\pi(\bar{s}|s)}\left[R(\bar{s})\right]
$$

We can also define an action-conditioned SM:

$$
\boldsymbol{\mu}^\pi(\bar{s}|s, a) = (1-\gamma) \sum_{t=0}^{\infty} \gamma^t P(s_{t+1} = \bar{s} | s_0 = s, a_0 = a) = (1-\gamma) T(\bar{s}|s, a) + \gamma \mathbb{E}\left[\boldsymbol{\mu}^\pi(\bar{s}|s', a', \pi)\right]
$$

Hence we can learn an SM using a TD update:

$$
\boldsymbol{\mu}^\pi(\bar{s}|s, a) \leftarrow \boldsymbol{\mu}^\pi(\bar{s}|s, a) + \eta \underbrace{\left((1-\gamma)T(s'|s, a) + \gamma \boldsymbol{\mu}^\pi(\bar{s}|s', a') - \boldsymbol{\mu}^\pi(\bar{s}|s, a)\right)}_{\delta}
$$

With an SM, we can compute the state-action value for any reward:

$$
Q^{R, \pi}(s, a) = \frac{1}{1-\gamma} \mathbb{E}_{\boldsymbol{\mu}^\pi(\bar{s}|s, a)}\left[R(\bar{s})\right]
$$

</div>

#### Learning SMs

Although we can learn SMs using the TD update, this requires evaluating $T(s'|s, a)$ to compute the target update $\delta$, and this one-step transition model is typically unknown. Instead, since $\boldsymbol{\mu}^\pi$ is a conditional density model, we can optimize the cross-entropy TD loss:

$$
\mathcal{L}_\mu = \mathbb{E}_{(s,a) \sim p(s,a), \bar{s} \sim (T^\pi \boldsymbol{\mu}^\pi)(\cdot|s,a)} \left[\log \boldsymbol{\mu}_\theta(\bar{s}|s, a)\right]
$$

where $(T^\pi \boldsymbol{\mu}^\pi)(\cdot|s, a)$ is the Bellman operator applied to $\boldsymbol{\mu}^\pi$ and then evaluated at $(s, a)$:

$$
(T^\pi \boldsymbol{\mu}^\pi)(\bar{s}|s, a) = (1-\gamma) T(s'|s, a) + \gamma \sum_{s'} T(\bar{s}|s, a) \sum_{a'} \pi(a'|s') \boldsymbol{\mu}^\pi(\bar{s}|s', a')
$$

We can sample from this as follows: first sample $s' \sim T(s'|s, a)$ from the environment (or an offline replay buffer), and then with probability $1-\gamma$ set $\bar{s} = s'$ and terminate. Otherwise sample $a' \sim \pi(a'|s')$ and then create a bootstrap sample from the SM using $\bar{s} \sim \boldsymbol{\mu}^\pi(\bar{s}|s', a')$.

There are many possible density models we can use for $\boldsymbol{\mu}^\pi$: VAEs, autoregressive transformers applied to discrete latent tokens (learned using VQ-VAE or a non-reconstructive self-supervised loss, called **Video Occupancy Models**), and diffusion (flow matching).

An alternative approach to learning SMs that avoids fitting a normalized density model over states is to use contrastive learning to estimate how likely $\bar{s}$ is to occur after some number of steps, given $(s, a)$, compared to some randomly sampled negative state.

#### Jumpy Models Using Geometric Policy Composition

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Geometric Policy Composition)</span></p>

**Geometric policy composition** (GPC) is a way to learn a new policy by sequencing together a set of $N$ policies, as opposed to taking $N$ primitive actions in a row. This can be thought of as a **jumpy model**, since it predicts multiple steps into the future, instead of one step at a time.

In more detail, in GPC, the agent picks a sequence of $n$ policies $\pi_i$ for $i = 1 : n$, and then samples states according to their corresponding SMs: starting with $(s_0, a_0)$, we sample $s_1 \sim \boldsymbol{\mu}_{\gamma'}^{\pi_1}(\cdot|s_0, a_0)$, then $a_1 \sim \pi_1(\cdot|s_1)$, then $s_2 \sim \boldsymbol{\mu}_{\gamma'}^{\pi_2}(\cdot|s_1, a_1)$, etc. Finally we sample $s_n \sim \boldsymbol{\mu}_{\gamma'}^{\pi_n}(\cdot|s_{n-1}, a_{n-1})$, where $\gamma' > \gamma$ represents a longer horizon SM. The reward estimates computed along this sampled path can then be combined to compute the value of each candidate policy sequence.

</div>

#### Other Related Work

**Proto-value networks** introduce a way to define auxiliary tasks based on successor measures. The **forwards-backwards** representations provide a general framework for learning SMs.

### Connection between Options and Successor Representations

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Options and the Graph Laplacian)</span></p>

**Options** are temporally extended actions that are closely related to the successor representation. Indeed, the eigenvectors of the SR matrix $\mathbf{M}^\pi$ correspond to the eigenvectors of the **graph Laplacian**, defined as

$$
\mathbf{L} = \mathbf{D}^{-\frac{1}{2}} (\mathbf{D} - \mathbf{A}) \mathbf{D}^{-\frac{1}{2}}
$$

where $\mathbf{A}$ is the adjacency matrix corresponding to $T^\pi$, and $\mathbf{D}$ is a diagonal matrix whose entries are the row-sums of $\mathbf{A}$. These eigenvectors are known as **proto value functions**, or **eigen-options**.

We can use this connection to define a "universal" form of successor representation that is independent of a specific policy. Rather than constructing an option for every eigenvector of the graph Laplacian, a single option based on the second eigenvector is sufficient. This is called a **covering option**, since it minimizes the cover time of the underlying MDP, which loosely refers to how long it takes for a random high-level policy to visit all states.

</div>

# Chapter 5: Multi-Agent RL

**Multi-agent RL** (**MARL**) is closely related to game theory and multi-agent systems design. In the **game theory** community, the rules of the game (i.e., the environment dynamics, and the reward function, aka **payoff function**) are usually assumed known, and the focus is on computing **strategies** (i.e., policies) for each **player** (i.e., agent), whereas in MARL, we usually assume the environment is unknown and the agents have to learn just by interacting with it.

## Games

Multi-agent environments are often called **games**, even if they represent "real-world" problems such as multi-robot coordination or agent-based trading.

### Normal-Form Games

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Normal-Form Game)</span></p>

A **normal-form game** defines a single interaction between $n \ge 2$ agents. We have a finite set of agents $\mathcal{I} = \lbrace 1, \ldots, n \rbrace$. For each agent $i \in \mathcal{I}$ we have a finite set of actions $\mathcal{A}_i$ and a reward function $\mathcal{R}_i : \mathcal{A}_{1:n} \to \mathbb{R}$, where $\mathcal{A}_{1:n} = \mathcal{A}_1 \times \cdots \times \mathcal{A}_n$. Each agent samples an action $a_i \in \mathcal{A}_i$ with probability $\pi_i(a_i)$, then the resulting **joint action** $\boldsymbol{a} = (a_1, \ldots, a_n)$ is taken and the reward $\boldsymbol{r} = (r_1, \ldots, r_m)$ is given to each player, where $r_i = \mathcal{R}_i(\boldsymbol{a})$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Types of Games by Reward Structure)</span></p>

Games can be classified based on the type of rewards they contain:

* In **zero-sum games**, we have $\sum_i \mathcal{R}_i(\boldsymbol{a}) = 0$ for all $\boldsymbol{a}$. (For a two-player zero-sum game, **2p0s**, we must have $R_1(\boldsymbol{a}) = -R_2(\boldsymbol{a})$.)
* In **common-payoff games** (aka common-reward games), we have $\mathcal{R}_i(\boldsymbol{a}) = \mathcal{R}_j(\boldsymbol{a})$ for all $\boldsymbol{a}$.
* In **general-sum games**, there are no restrictions on the rewards.

In zero-sum games, agents must compete. In common-reward games, agents generally must cooperate (although the **credit assignment** problem --- disentangling each agent's contribution --- can be challenging). In general-sum games, there can be a mix of cooperation and competition.

</div>

Normal-form games with 2 agents are called **matrix games** because they can be defined by a 2d reward matrix. Well-known examples include:

* **Rock-paper-scissors**: a zero-sum game.
* **Battle of the sexes**: a **coordination game** where both players prefer to do the same activity but have different individual preferences.
* **Prisoner's dilemma**: a general-sum game where both players have an incentive to defect, even though they would be better off if they both cooperated.

A **repeated matrix game** is the multi-agent analog of a multi-armed bandit problem. The policy has the form $\pi_i(a_t^i|\boldsymbol{h}_t)$, where $\boldsymbol{h}_t = (\boldsymbol{a}_0, \ldots, \boldsymbol{a}_{t-1})$ is the history of joint-actions. For example, in the **tit-for-tat** strategy in the prisoner's dilemma, the policy for agent $i$ at step $t$ is to do the same action that agent $-i$ did at step $t-1$, which can lead to the evolution of cooperative behavior, even in selfish agents.

### Stochastic Games

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Stochastic Game)</span></p>

A **stochastic game** is a multi-agent version of an MDP. It is defined by a finite set of agents $\mathcal{I} = \lbrace 1, \ldots, n \rbrace$; a finite set of states $\mathcal{S}$, of which a subset $\overline{\mathcal{S}} \subset \mathcal{S}$ are terminal; a finite action set $\mathcal{A}_i$ for each agent $i \in \mathcal{I}$; a reward function $\mathcal{R}_i(s, a, s')$ for each agent $i$; a state transition distribution $\mathcal{T}(s_{t+1}|s_{1:t}, \boldsymbol{a}_t) \in [0, 1]$; and an initial state distribution $\mu(s_0) \in [0, 1]$.

Typically the transition distribution is Markovian, i.e., $\mathcal{T}(s_{t+1}|s_{1:t}, \boldsymbol{a}_t) = \mathcal{T}(s_{t+1}|s_t, \boldsymbol{a}_t)$, in which case this is called a **Markov game**.

The policy for each agent has the form $\pi_i(a_t^i|\boldsymbol{h}_t)$ where $\boldsymbol{h}_t = (s_0, \boldsymbol{a}_1, \ldots, s_t)$ is the state-action history. The overall **joint policy** is denoted by $\boldsymbol{\pi} = (\pi_1, \ldots, \pi_n)$; if the agents make their decisions independently:

$$
\pi(\boldsymbol{a}_t|\boldsymbol{h}_t) = \prod_i \pi_i(a_t^i|\boldsymbol{h}_t)
$$

</div>

From the perspective of each agent $i$, the environment transition function has the form

$$
\mathcal{T}_i(s_{t+1}|s_t, a_t^i) = \sum_{\boldsymbol{a}_t^{-i}} \mathcal{T}(s_{t+1}|s_t, (a_t^i, \boldsymbol{a}_t^{-i})) \prod_{j \ne i} \pi_j(a_t^j|s_t)
$$

Thus $\mathcal{T}_i$ depends on the policies of the other players, which are often changing, which makes these local/agent-centric transition matrices non-stationary.

### Partially Observed Stochastic Games (POSG)

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(POSG)</span></p>

A **Partially Observed Stochastic Game** or **POSG** is a multi-agent version of a POMDP. We augment the stochastic game with the observation distributions $\mathcal{O}_i(o_{i+1}^i|s_{t+1}, \boldsymbol{a}_t) \in [0, 1]$ for each agent $i$. Let $\boldsymbol{o}_t = (o_t^1, \ldots, o_t^n)$ be the **joint observation** generated by the product distribution $\mathcal{O}_{1:n}(\boldsymbol{o}_t|s_t, \boldsymbol{a}_{t-1})$. The policy for each agent has the form $\pi_i(a_t^i|\boldsymbol{h}_t^i)$ where $\boldsymbol{h}_t^i = (o_0^i, a_0^i, o_1^i, a_1^i, \ldots, o_t^i)$ is the **action observation history** for agent $i$.

A **Decentralized POMDP** or **Dec-POMDP** is a special case of a POSG where the reward function is the same for all agents (thus it can only capture cooperative behavior).

</div>

From the perspective of agent $i$, it just observes a sequence of observations generated by a "sensor stream distribution" (which is non-Markovian):

$$
p_i(o_{t+1}^i|\boldsymbol{h}_t^i, a_t^i) = \sum_{s_{t+1}} \sum_{\boldsymbol{a}_t^{-i}} \hat{\mathcal{O}}_i(o_{t+1}^i|s_{t+1}, \boldsymbol{a}_t) p_i(\boldsymbol{a}_t^{-i}|\boldsymbol{h}_t^i) p_i(s_{t+1}|\boldsymbol{h}_t^i, \boldsymbol{a}_t)
$$

where $p_i(\boldsymbol{a}_t^{-i}|\boldsymbol{h}_t^i) = \prod_{j \ne i} \hat{\pi}_j^i(a_t^j|\boldsymbol{h}_t^i)$ uses $i$'s estimate of $j$'s policy, and $b_i(s_t|\boldsymbol{h}_t^i)$ is $i$'s **belief state** (posterior distribution over the underlying latent state). The agent can either learn a policy given this "collapsed" representation, or explicitly try to learn the true joint world model $\mathcal{T}$, local observation model $\mathcal{O}_i$, and other agent policies $\pi_j^i$, so it can reason about the other agents.

#### Factored Observation Stochastic Games (FOSG)

**Factored Observation Stochastic Games** or **FOSG** extend POSGs by partitioning the observation for each player into public and private. Information is public if it is visible to all players, and all players know this; thus it is a form of **common knowledge**. Explicitly distinguishing these two kinds of information is important in order to tractably solve certain kinds of games, like Poker or Hanabi.

### Extensive Form Games (EFG)

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Extensive Form Game)</span></p>

In the game theory literature, it is common to use the **extensive form game** representation. Rather than representing a sequence of world states that evolve over time, we represent a tree of possible choices or actions taken by each player (and optionally a **chance player**, if the game is stochastic). Each node represents a unique sequence (history) of actions leading up to that point.

* If all the nodes are observed (including chance nodes), the game has perfect and complete information.
* If the moves of some players are not visible and/or the state of the game is not fully known (e.g., poker), the game has **imperfect information**. We define an **information set** as the set of nodes that an agent cannot distinguish between. This is analogous to having a distribution over the hidden states in a POSG.
* If an agent does not know the other player's type or payoff function, the game has **incomplete information**. The agent should maintain a Bayesian belief state about the unknown factors.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Kuhn Poker as EFG)</span></p>

**Kuhn Poker** is a form of (two player) poker where the deck includes only three cards: Jack, Queen, and King. First, each player places one chip into the pot as the initial forced bet (ante). Each player is then privately dealt one card (the last card isn't revealed). This is followed by a betting phase. The game ends either when one player folds (forfeiting all bets made so far to their opponent) or there is a showdown, where the private cards are revealed and the higher card's owner receives the bets. At the start of the betting, player one can either check/pass or raise/bet (one chip). If they check, player two can also check/pass (leading to a showdown) or bet. If one of the players bets, their opponent must either call (betting one chip to match the opponent's bet), followed by a showdown, or fold.

</div>

We can convert an FOSG into an EFG by "unrolling" it. First we define the information set for a given information state as the set of consistent world state trajectories. By applying the policy of each agent to the world model, we can derive a tree of possible world states (trajectories) and corresponding information sets for each agent.

## Solution Concepts

In the multi-agent setting the definition of "optimality" is much more complex than in the single agent setting. There are multiple **solution concepts**.

### Notation and Definitions

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Expected Return and Best Response)</span></p>

Let $\hat{\boldsymbol{h}}_t = \lbrace (s_k, \boldsymbol{o}_k, \boldsymbol{a}_k)_{k=1}^{t-1}, s_t, \boldsymbol{o}_t \rbrace$ be the **full history**, containing all the past states, joint observations, and joint actions. We define the expected return for agent $i$ under joint policy $\boldsymbol{\pi}$ by

$$
U_i(\boldsymbol{\pi}) = \sum_{\hat{\boldsymbol{h}}_t} p(\hat{\boldsymbol{h}}_t|\boldsymbol{\pi}) u_i(\hat{\boldsymbol{h}}_t)
$$

where $u_i(\hat{\boldsymbol{h}}_t) = \sum_{k=0}^{t-1} \gamma^k \mathcal{R}_i(s_k, \boldsymbol{a}_k, s_{k+1})$ is the discounted actual return for agent $i$ in a given full history.

We can derive Bellman-like equations:

$$
V_i^\pi(\hat{\boldsymbol{h}}) = \sum_{\boldsymbol{a}} \pi(\boldsymbol{a}|\sigma(\hat{\boldsymbol{h}})) Q_i^\pi(\hat{\boldsymbol{h}}, \boldsymbol{a})
$$

$$
Q_i^\pi(\hat{\boldsymbol{h}}, \boldsymbol{a}) = \sum_{s'} \mathcal{T}(s'|s(\hat{\boldsymbol{h}}), \boldsymbol{a}) \left[\mathcal{R}_u(s(\hat{\boldsymbol{h}}), \boldsymbol{a}, s') + \gamma \sum_{\boldsymbol{o}'} \mathcal{O}_{1:n}(\boldsymbol{o}'|\boldsymbol{a}, s') V_i^\pi((\hat{\boldsymbol{h}}, \boldsymbol{a}, s', \boldsymbol{o}'))\right]
$$

The **best response policy** for agent $i$ is the one that maximizes the expected return for agent $i$ against a given set of policies for all the other agents, $\boldsymbol{\pi}_{-i} = (\pi_1, \ldots, \pi_{i-1}, \pi_{i+1}, \ldots, \pi_n)$:

$$
\text{BR}_i(\boldsymbol{\pi}_{-i}) = \text{argmax}_{\pi_i} U_i((\pi_i, \boldsymbol{\pi}_{-i}))
$$

</div>

### Minimax

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Minimax Solution)</span></p>

The **minimax** solution is defined for two-agent zero-sum games. Its existence for normal-form games was first proven by John von Neumann in 1928. We say that joint policy $\boldsymbol{\pi} = (\pi_i, \pi_j)$ is a minimax solution if

$$
U_i(\boldsymbol{\pi}) = \max_{\pi_i'} \min_{\pi_j'} U_i(\pi_i', \pi_j') = \min_{\pi_j'} \max_{\pi_i'} U_i(\pi_i', \pi_j') = -U_j(\boldsymbol{\pi})
$$

In other words, $\boldsymbol{\pi}$ is a minimax solution iff $\pi_i \in \text{BR}_i(\pi_j)$ and $\pi_j \in \text{BR}_j(\pi_i)$. We can solve for the minimax solution using linear programming.

Minimax solutions also exist for two-player zero-sum stochastic games with finite episode lengths, such as chess and Go. In the case of perfect information games, dynamic programming can be used; in general, minimax search (a depth-limited version of DP that requires a heuristic function) can be used.

</div>

### Exploitability

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Exploitability)</span></p>

In the case of a 2-player zero-sum game, we can measure how close we are to a minimax solution by computing the **exploitability** score:

$$
\text{exploitability}(\boldsymbol{\pi}) = \frac{1}{2} \left[\max_{\pi_1'} J(\pi_1', \pi_2) - \min_{\pi_2'} J(\pi_1, \pi_2')\right]
$$

where $J(\boldsymbol{\pi})$ is the expected reward for player 1 (which is the loss for player 2). Exploitability is the expected return of $\pi_i$ playing against a best response to $\pi_i$, averaged over both players $i \in 1, 2$. Joint policies with exploitability zero are Nash equilibria.

</div>

### Nash Equilibrium

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Nash Equilibrium)</span></p>

The **Nash equilibrium** (NE) generalizes the idea of mutual best response to general-sum games with two or more agents. We say that $\boldsymbol{\pi}$ is a Nash equilibrium if no agent $i$ can improve its expected returns by changing its policy $\pi_i$, assuming the other agents' policies remain fixed:

$$
\forall i, \pi_i'. \; U_i(\pi_i', \boldsymbol{\pi}_{-i}) \le U_i(\boldsymbol{\pi})
$$

John Nash proved the existence of such a solution for general-sum non-repeated normal-form games in 1950.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Nash Equilibria in Matrix Games)</span></p>

* **Rock-paper-scissors**: the only NE is the **mixed strategy** (stochastic policy) where each agent chooses actions uniformly at random, $\pi_i = (1/3, 1/3, 1/3)$. This yields an expected return of 0.
* **Battle of the sexes**: there are two pure strategy NEs --- (Opera, Opera) and (Football, Football) --- and also a mixed strategy equilibrium with lower expected payoffs.
* **Prisoner's Dilemma**: the only NE is the pure strategy of (D,D), yielding $(-3, -3)$. Note this is worse than the maximum possible expected return of $(-1, -1)$ given by (C,C), but (C,C) is not an NE since each player could improve its return by unilaterally defecting.

</div>

Computing a Nash equilibrium is a famously hard problem for general games, falling into a complexity class called **PPAD-complete**, which suggests that there is no efficient, universal algorithm for finding one.

### Approximate Nash Equilibrium

It is possible to relax the definition by defining an $\epsilon$-Nash equilibrium as a joint policy that satisfies

$$
\forall i, \pi_i'. \; U_i(\pi_i', \boldsymbol{\pi}_{-i}) - \epsilon \le U_i(\boldsymbol{\pi})
$$

Unfortunately, the expected return from an $\epsilon$-NE can be very different from a true NE. We can measure the rate of convergence to a NE by defining

$$
\text{NashConv}(\boldsymbol{\pi}) = \sum_i \delta_i(\boldsymbol{\pi})
$$

where $\delta_i(\boldsymbol{\pi}) = u_i(\pi_i^b, \boldsymbol{\pi}_{-i}) - u_i(\boldsymbol{\pi}), \; \pi_i^b \in \text{BR}(\boldsymbol{\pi}_{-i})$ is the amount of incentive that $i$ has to deviate to one of its best responses.

### Entropy Regularized Nash Equilibria (Quantal Response Equilibria)

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Quantal Response Equilibrium)</span></p>

**Quantal response equilibria** or **QRE** are like Nash equilibria except the best response policy is a "soft" entropy-regularized policy. This kind of equilibrium reflects the fact that players may not always choose the best response with certainty, but instead make choices based on a probability distribution over actions, based on the relative expected utility of each action. It is a Bayesian equilibrium, useful for modeling human behavior in **behavioral game theory**.

For single agent problems with a single state (i.e., bandit problems), a policy $\pi$ is $\alpha$-soft optimal in the normal sense if it satisfies

$$
\pi = \text{argmax}_{\pi' \in \Delta(A)} \mathbb{E}_{A \sim \pi'} q(A) + \alpha \,\mathbb{H}(\pi')
$$

For two-player zero-sum NFGs, a policy is a QRE if each player's policy is soft optimal conditioned on the other player not changing its policy. For two-player zero-sum games with multiple states (i.e., EFGs), we say that a policy is an **agent QRE** if each player's policy is soft optimal in the behavioral sense conditioned on the other player not changing its policy.

</div>

### Correlated Equilibrium

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Correlated Equilibrium)</span></p>

A Nash equilibrium assumes the policies are independent, which can limit the expected returns. A **correlated equilibrium** (CE) allows for correlated policies. Specifically, we assume there is a central policy $\boldsymbol{\pi}_c$ that defines a distribution over joint actions. Agents can follow this recommended policy, or can choose to deviate from it by using an action modifier $\xi_i : \mathcal{A}_i \to \mathcal{A}_i$. We then say that $\boldsymbol{\pi}_c$ is a CE if for all $i$ and $\xi_i$ we have

$$
\sum_{\boldsymbol{a}} \boldsymbol{\pi}_c(\boldsymbol{a}) \mathcal{R}_i((\xi_i(a^i), \boldsymbol{a}^{-i})) \le \sum_{\boldsymbol{a}} \boldsymbol{\pi}_c(\boldsymbol{a}) \mathcal{R}_i(\boldsymbol{a})
$$

That is, player $i$ has no incentive to deviate from the recommendation, after receiving it. The set of correlated equilibria contains the set of Nash equilibria. The correlated equilibrium solution can be computed via linear programming.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Chicken Game)</span></p>

In the **Chicken game**, two agents are driving towards each other. Each can either stay on course (S) or leave (L). The payoff matrix is:

| | S | L |
| --- | --- | --- |
| **S** | 0, 0 | 7, 2 |
| **L** | 2, 7 | 6, 6 |

There are 3 uncorrelated NEs: $\boldsymbol{\pi} = (1, 0)$ with return $(7, 2)$; $\boldsymbol{\pi} = (0, 1)$ with return $(2, 7)$; and $\boldsymbol{\pi} = (\frac{1}{2}, \frac{1}{2})$ with return $(4.66, 4.66)$. There is 1 CE, namely $\boldsymbol{\pi}_c(L,L) = \boldsymbol{\pi}_c(S,L) = \boldsymbol{\pi}_c(L,S) = \frac{1}{3}$ and $\boldsymbol{\pi}_c(S,S) = 0$. The central policy has an expected return of $7 \cdot \frac{1}{3} + 2 \cdot \frac{1}{3} + 6 \cdot \frac{1}{3} = 5$, which is higher than the mixed NE of 4.66. This is because it avoids the deadly joint (S,S) action.

</div>

### Limitations of Equilibrium Solutions

Equilibrium solutions have several limitations. First, they do not always maximize expected returns (e.g., in Prisoner's Dilemma, (D,D) is Nash but (C,C) yields higher returns). Second, there can be multiple (even infinitely many) equilibria, each with different expected returns. Third, equilibria for sequential games don't specify what to do if the history deviates from the equilibrium path.

### Pareto Optimality

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Pareto Optimality)</span></p>

We say a joint policy $\boldsymbol{\pi}$ is **Pareto optimal** if it is not **Pareto dominated** by any other joint policy $\boldsymbol{\pi}'$. We say that $\boldsymbol{\pi}$ is Pareto dominated by $\boldsymbol{\pi}'$ if $\boldsymbol{\pi}'$ improves the expected return for at least one agent:

$$
\forall i. \; U_i(\boldsymbol{\pi}') \ge U_i(\boldsymbol{\pi}) \quad \text{and} \quad \exists i. \; U_i(\boldsymbol{\pi}') > U_i(\boldsymbol{\pi})
$$

and if it does not decrease the payoff for any agents.

</div>

### Social Welfare and Fairness

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Welfare and Fairness Optimality)</span></p>

We define **welfare optimality** as

$$
W(\boldsymbol{\pi}) = \sum_i U_i(\boldsymbol{\pi})
$$

A joint policy is welfare-optimal if $\boldsymbol{\pi} \in \text{argmax}_{\boldsymbol{\pi}'} W(\boldsymbol{\pi}')$. One can show that welfare optimality implies Pareto optimality, but not (in general) vice versa.

Similarly, we define **fairness optimality** as

$$
F(\boldsymbol{\pi}) = \prod_i U_i(\boldsymbol{\pi})
$$

A joint policy is fairness-optimal if $\boldsymbol{\pi} \in \text{argmax}_{\boldsymbol{\pi}'} F(\boldsymbol{\pi}')$.

</div>

### No Regret

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Regret)</span></p>

The quantity known as **regret** measures the difference between the rewards an agent received versus the maximum rewards it could have received if it had chosen a different action. For a non-repeated normal-form general-sum game, played over $E$ episodes:

$$
\text{Regret}_i^E = \max_{a^i} \sum_{e=1}^{E} [\mathcal{R}_i((a^i, \boldsymbol{a}_e^{-i})) - \mathcal{R}_i(\boldsymbol{a}_e)]
$$

We can generalize to stochastic games and POSGs by defining the regret over policies instead of actions:

$$
\text{Regret}_i^E = \max_{\pi^i} \sum_{e=1}^{E} [U_i((\pi^i, \boldsymbol{\pi}_e^{-i})) - U_i(\boldsymbol{\pi}_e)]
$$

An agent is said to have **no-regret** if $\forall i. \; \lim_{E \to \infty} \frac{1}{E} \text{Regret}_i^E \le 0$.

</div>

### Shapley Values

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Shapley Value)</span></p>

The **Shapley value** allows one to estimate the marginal contribution of a single agent to a common reward (cooperative) game, which can help ameliorate the **credit assignment problem**. Specifically, suppose there are $N$ players, and let $S$ be a subset of $\le N$ players that form a team. Let $v(S)$ be the expected value obtained by that team, and $v(S \setminus \lbrace i \rbrace)$ be the value of the team when $i$ is absent. Then we define $i$'s Shapley value as

$$
\phi(i) = \sum_{S \subseteq N \setminus \lbrace i \rbrace} w(S) [v(S \cup \lbrace i \rbrace) - v(S)]
$$

where the weighting term is

$$
w(S) = \frac{|S|!(|N| - |S| - 1)!}{|N|!} = \frac{1}{|N| \cdot \binom{|N|-1}{|S|}}
$$

which represents the probability that, for a random ordering of all players, the players in coalition $S$ come before player $i$.

</div>

An interesting application of Shapley values arises in the "explainable AI" literature, where one of the goals is to estimate the importance of individual predictors to an overall prediction. This can be done using the **SHAP** (SHapley Additive exPlanations) framework.

### Stackelberg Equilibrium

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Stackelberg Equilibrium)</span></p>

In **sequential games**, where the players take turns, the concept of **Stackelberg equilibrium** becomes relevant. One player, called the "leader", makes their decision first. The other player, the "follower", observes the leader's choice and then makes their own decision. This sequential structure gives the leader a significant advantage: the leader can choose a strategy that maximizes their own payoff, anticipating the follower's subsequent best response.

Stackelberg equilibria can be easier to compute than Nash equilibria. To find the Stackelberg equilibrium, you can use **backward induction**: determine the follower's best response for every possible action the leader could take, then the leader simply chooses the action that will lead to the best outcome for themselves.

The concept is useful for analysing **setter-solver** problems, which arise in unsupervised environment design / curriculum learning.

</div>

## Algorithms

### Centralized Learning

The simplest way to solve a MARL problem is to reduce it to a single agent RL (SARL) problem. In **centralized learning**, we learn a single joint policy over the joint action space. This requires that we can transform the joint reward $\boldsymbol{r}_t = (r_t^1, \ldots, r_t^n)$ into a scalar $r_t$. This is easy to do in common-reward games, but for general-sum games, it may be impossible to define a single scalar reward across all agents, and the method may not scale well with the number of agents.

### Independent Learning

In **independent learning**, each agent treats all other agents as part of the environment, and then uses any standard single-agent RL algorithm for training. This is done in parallel across all agents.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Independent Q Learning - IQL)</span></p>

**Independent Q Learning** (IQL) runs DQN for each agent independently:

1. Initialize $n$ value networks with random parameters $\theta_1, \ldots, \theta_n$; initialize $n$ target networks $\bar{\theta}_i = \theta_i$; initialize a replay buffer $D_i$ for each agent.
2. For each time step $t$: Collect observations $o_t^1, \ldots, o_t^n$; for each agent $i$, with probability $\epsilon$ choose random action, otherwise $a_t^i \in \text{argmax}_{a_i} Q(h_t^i, a_i; \theta_i)$.
3. Apply joint actions; collect rewards $r_t^1, \ldots, r_t^n$ and next observations.
4. For each agent $i$: store transition in $D_i$; sample mini-batch; compute targets $y_k^i = r_k^i + \gamma \max_{a_i'} Q(h_{k+1}^i, a_i'; \bar{\theta}_i)$ (or $y_k^i = r_k^i$ if terminal); minimize loss $\mathcal{L}(\theta_i) = \frac{1}{B}\sum_{k=1}^{B}(y_k^i - Q(h_k^i, a_k^i; \theta_i))^2$; periodically update $\bar{\theta}_i$.

</div>

#### Independent Actor Critic

The multi-agent version of the policy gradient theorem is:

$$
\nabla_\theta J(\boldsymbol{\theta}_{1:n}) \propto \mathbb{E}_{\hat{\boldsymbol{h}} \sim p(\hat{\boldsymbol{h}}|\pi), a^i \sim \pi_i, \boldsymbol{a}^{-i} \sim \boldsymbol{\pi}^{-i}} \left[Q_i^\pi(\hat{\boldsymbol{h}}, (a^i, \boldsymbol{a}^{-i})) \nabla_{\theta_i} \log \pi(a_i | h_i = \sigma_i(\hat{\boldsymbol{h}}); \theta_i)\right]
$$

In practice, we use the advantage $\text{Adv}_i^\pi(\hat{\boldsymbol{h}}, \boldsymbol{a}) = Q_i^\pi(\hat{\boldsymbol{h}}, (a^i, \boldsymbol{a}^{-i})) - V_i^\pi(\hat{\boldsymbol{h}})$ as the baseline to reduce variance. This can be used inside a multi-agent version of A2C (known as **MAA2C**). An independent version of PPO is known as **IPPO**.

#### Learning Dynamics of Multi-Agent Policy Gradient Methods

Applying policy gradient methods to multiple agents in parallel may not result in convergence. This is known as **infinitesimal gradient ascent** or **IGA**. IGA does not always converge, but the method **Win or Learn Fast** or **WoLF** ensures that IGA policies always converge to a NE for two-agent two-action normal-form games. The trick is to learn slow (by using smaller $\kappa$) when winning, and to learn fast (by using larger $\kappa$) when losing. The resulting method is called **WoLF-PHC** (WoLF with Policy Hill Climbing).

### Centralized Training of Decentralized Policies (CTDE)

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(CTDE)</span></p>

We can improve performance beyond independent learning by using a paradigm known as **Centralized Training and Decentralized Execution** (**CTDE**), in which the learning algorithm has access to all the information (from all agents) at training time, but at test time, agents only observe their own local observations. The **central information** can contain the joint action taken by all agents, and/or the joint observation vector, even such joint information is not available at execution time.

We can modify the multi-agent A2C algorithm to exploit this by writing the $i$'th value/advantage function as $V(h_t^i, c_t; \boldsymbol{w}_i) / \text{Adv}(h_t^i, c_t, a_t^i)$, where $c_t$ is the shared central information. This is known as **centralized critics** with **decentralized actors**. A CTDE version of PPO is known as **MAPPO**.

</div>

#### Application to Diplomacy (Cicero)

The **Cicero** system achieved human-level performance in the complex natural language 7-player strategy game called **Diplomacy**, which requires both cooperative and competitive behavior. Cicero used CTDE, combining an LLM for generating and interpreting dialog with a mix of self-play RL, imitation learning, opponent modeling, and policy generation using regret minimization. The system uses imitation learning on human games to warm-start the initial policy and language model, and then is refined using RL with self-play. The system uses explicit belief state modeling over the opponents' intents and plans.

### Value Decomposition Methods for Common-Reward Games

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Value Decomposition and IGM)</span></p>

To ensure that the per-agent policy can be implemented using only locally available information, we need to use **value decomposition** methods, which assume that the global value function can be decomposed into separate value functions, one per agent. This decomposition is valid provided the separate value functions satisfy the **individual global max** or **IGM** property:

$$
\forall \boldsymbol{a}. \; \boldsymbol{a} \in A^*(\boldsymbol{h}, c; \boldsymbol{\theta}) \Leftrightarrow \forall i. \; a_i \in A_i^*(h^i; \boldsymbol{\theta}_i)
$$

where $A^*(\boldsymbol{h}, c; \boldsymbol{\theta}) = \text{argmax}_{\boldsymbol{a}} Q(\boldsymbol{h}, c, \boldsymbol{a}; \boldsymbol{\theta})$ and $A_i^*(h^i; \boldsymbol{\theta}_i) = \text{argmax}_a Q(h^i, a^i; \boldsymbol{\theta}_i)$. This ensures that picking actions locally for each agent will also be optimal globally.

</div>

#### Value Decomposition Network (VDN)

The **VDN** method assumes a linear decomposition: $Q(\boldsymbol{h}_t, c_t, \boldsymbol{a}_t; \boldsymbol{\theta}) = \sum_i Q(h_t^i, a_t^i; \boldsymbol{\theta}_i)$. This clearly satisfies IGM.

#### QMIX

**QMIX** assumes $Q(\boldsymbol{h}_t, c_t, \boldsymbol{a}_t; \boldsymbol{\theta}) = f_{\text{mix}}(Q(h_t^1, a_t^1; \boldsymbol{\theta}_1), \ldots, Q(h_t^n, a_t^n; \boldsymbol{\theta}_n))$, where $f_{\text{mix}}$ is a neural network constructed so that it is monotonically increasing in each of its arguments (ensured by requiring all weights of the mixing network to be non-negative; the weights themselves are predicted by another "hyper-network" conditioned on the state $h_t^i$). This satisfies IGM since

$$
\max_{\boldsymbol{a}} Q(\boldsymbol{h}_{t+1}, c_{t+1}, \boldsymbol{a}; \bar{\boldsymbol{\theta}}) = f_{\text{mix}}\left(\max_{a^1} Q(h_{t+1}^1, a^1; \bar{\theta}_1), \ldots, \max_{a^n} Q(h_{t+1}^n, a^n; \bar{\theta}_n)\right)
$$

### Policy Learning with Self-Play

For symmetric zero-sum games, where $r_i(s) = -r_j(s)$, we can assume that each player uses the same policy, modulo rearrangement of the input state: $\pi_j(\cdot|s) = \pi_i(\cdot|\psi(s))$, where $\psi(s)$ deterministically modifies the state to reflect the symmetry. The result is a single-agent problem in which we treat player $j$ as part of the environment:

$$
p^\pi(s'|s, a^i) = \sum_{a^j} \pi(a^j|\psi(s)) p(s'|s, a^i, a^j)
$$

This lets us learn $\pi$ using standard single-agent policy learning methods. This is known as **self-play**. Self-play is used by AlphaZero to learn to play Chess and Go at super-human level. For perfect information, zero-sum games, self-play can be proved to converge to a Nash equilibrium. Unfortunately, for imperfect information games (e.g., Poker or Hanabi) or general-sum games, self-play can lead to oscillating strategies or cyclical behavior.

### Policy Learning with Learned Opponent Models

Instead of using self-play, we can learn an **opponent model**. In the CTDE paradigm, where each agent sees the other agents' actions, agent $i$ can use supervised learning to predict the actions of agent $j$ given $i$'s observations. For example, we can train an encoder-decoder network to predict the actions of other agents via a bottleneck, and then pass this bottleneck embedding to the policy as side information: let $m_t^i = f^e(h_t^i; \psi_i^e)$ be the encoding, which is then passed to the decoder $f^d$ to predict the other agents' actions $\hat{\boldsymbol{\pi}}_{-i}^{i,t} = f^d(m_t^i; \psi_i^d)$, and also to the policy $a_t^i \sim \pi(\cdot|h_t^i, m_t^i; \theta_i)$.

### Best Response

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Best Response with Opponent Modeling)</span></p>

MARL algorithms that can provably converge to a Nash equilibrium (even for zero-sum, imperfect-information games) are built on the concept of a best response. Let $h^i$ be the information state for agent $i$. We compute its expected state-action value, given the joint policy:

$$
AV_i^\pi(h^i, a^i) = \sum_{\boldsymbol{a}^{-i}} Q_i(\boldsymbol{h}^i, (a^i, \boldsymbol{a}^{-i})) \prod_{j \ne i} \pi_j(a^j|h^i; \theta_j^i)
$$

The best response is then $\text{BR}_i(h^i) = \text{argmax}_{a^i} AV_i^\pi(h^i, a^i)$.

</div>

#### Fictitious Play

In **fictitious play**, each agent $i$ estimates the policies of the other players, based on their past actions. It then computes a best response. For non-repeated normal-form games (stateless), we estimate the policies by counting and averaging:

$$
\hat{\pi}_j^t(a^j) = \frac{C_j^t(a^j)}{\sum_{a'} C_j^t(a')}
$$

where $C_j^t(a^j)$ is the number of times agent $j$ chose action $a^j$ in episodes up to step $t$. For two-player zero-sum finite games, the exploitability of the average $\overline{\pi}_t$ generated by FP converges to zero as $t$ grows large.

#### Neural Fictitious Self Play (NFSP)

We can extend FP to the partially observed, non-tabular setting. Agent $i$ will learn a model of $j$'s policy, given $i$'s state (history), which we denote by $\overline{\pi}_{j|i}^t(a^j|h^i)$. We fit this by minimizing the cross-entropy loss:

$$
\mathcal{L}(\overline{\pi}_{j|i}^t) = \mathbb{E}_{k \sim U(1,t), (h_k^i, a_k^j) \in \mathcal{D}_i} \left[-\log \overline{\pi}_{j|i}^t(a_k^j|h_k^i)\right]
$$

We then use DQN to learn $Q_i(h^i, \boldsymbol{a})$ for each agent, and use this learned average policy plus the Q functions to compute $AV_i$ and hence the best response. In the zero-sum two-player case, we can use self-play. This is called **fictitious self play**, extended to neural nets as **neural fictitious self play**. If the Q function converges to the optimal function, the process converges to a NE.

### Population-Based Training

In self-play, we model the opponent as using the same policy as the agent itself. To avoid overfitting, we typically train against multiple versions of the agent's own policy. This concept can be generalized to work with general-sum games with two or more players, by training against a population of different policies. This is called **population-based training**.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(PSRO)</span></p>

The **policy space response oracle** or **PSRO** method is a game-theoretic instance of population-based training, which can compute policies that satisfy various solution concepts for any kind of stochastic game, including partially observed, general-sum games.

At generation $k$, each agent $i$ has a finite set of policies it can use, denoted $\Pi_i^k$. We define a normal-form **meta-game** $M^k$ from this by letting each agent choose one of its policies, where the reward is $\mathcal{R}_i(\boldsymbol{\pi}) = U_i(\boldsymbol{\pi})$. Once we have the reward matrix, we solve for some kind of equilibrium solution using a **meta-strategy solver**, extracting the probability distributions over policies $\sigma_i^k$ for each agent.

We expand the set of policies by using an oracle to compute a new policy $\pi_i'$, e.g., a best response:

$$
\pi_i' \in \text{argmax}_{\pi_i} \mathbb{E}_{\boldsymbol{\pi}_{-i} \sim \sigma_{-i}^k} [U_i((\pi_i, \boldsymbol{\pi}_{-i}))]
$$

and adding it to $\Pi_i^k$ to create $\Pi_i^{k+1}$. If PSRO uses exact Nash equilibria for the meta-game, and if the oracle computes exact best-response policies, then the distributions $\lbrace \sigma_i^k \rbrace_{i \in \mathcal{I}}$ converge to a Nash equilibrium of the underlying game $G$.

</div>

The **AlphaStar** system used a PSRO-like method, combined with the (single agent) A2C RL algorithm, to achieve grandmaster status in the challenging real-time strategy game **StarCraft II**. It used the following steps: build a pool of agents that represent different playstyles and skill levels (known as a league); compute best responses to existing strategies; update a meta-strategy to mix agents in a way that approximates a Nash equilibrium; select opponents from the Nash mixture to ensure robustness; and train a new agent against the weighted mixture of past opponents.

### Counterfactual Regret Minimization (CFR)

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Counterfactual Regret Minimization)</span></p>

**Counterfactual Regret Minimization** (**CFR**) is an algorithm for imperfect information, two-player, zero-sum games. When using this procedure, the average policies converge to an $\epsilon$-Nash.

Let $\boldsymbol{\tau} = (s_0, \ldots, s_t)$ be a trajectory. Let $\eta^\pi(\boldsymbol{\tau})$ be the probability of this trajectory under the joint policy, decomposed as $\eta^\pi(\boldsymbol{\tau}) = \eta_i^\pi(\boldsymbol{\tau}) \eta_{-i}^\pi(\boldsymbol{\tau})$. Define the **counterfactual state-action value** for an information state as

$$
q_{\boldsymbol{\pi},i}^c(\boldsymbol{h}^i, a^i) = \sum_{(\boldsymbol{\tau}, \boldsymbol{z}) \in Z(\boldsymbol{h}^i)} \eta_{-i}^\pi(\boldsymbol{\tau}) \eta^\pi(\boldsymbol{\tau} a^i, \boldsymbol{z}) u_i(\boldsymbol{z})
$$

The instantaneous **counterfactual regret** for player $i$ at iteration $k$ is

$$
r_i^k(\boldsymbol{h}^i, a^i) = q_{\boldsymbol{\pi}^k,i}^c(\boldsymbol{h}^i, a^i) - v_{\boldsymbol{\pi},i}^c(\boldsymbol{h}^i)
$$

(this is the counterfactual version of an advantage function). The cumulative counterfactual regret is $R_i^k(\boldsymbol{h}^i, a^i) = \sum_{j=0}^{k} r_i^j(\boldsymbol{h}^i, a^i)$.

CFR starts with a uniform random joint policy $\boldsymbol{\pi}^0$ and then updates it at each iteration by performing **regret matching**:

$$
\pi_i^{k+1}(\boldsymbol{h}^i, a^i) = \begin{cases} \frac{R_i^{k,+}(\boldsymbol{h}^i, a^i)}{\sum_{a \in \mathcal{A}_i(\boldsymbol{h}^i)} R_i^{k,+}(\boldsymbol{h}^i, a)} & \text{if denominator is positive} \\ \frac{1}{|\mathcal{A}_i(\boldsymbol{h}^i)|} & \text{otherwise} \end{cases}
$$

where $x^+ = \max(x, 0)$. This converges to an $\epsilon$-Nash equilibrium where $\epsilon = O(\max_i |\mathcal{H}_i| \sqrt{|\mathcal{A}_i|} / \sqrt{t})$.

</div>

**Deep CFR** approximates the expectations over trajectories using Monte Carlo sampling and approximates the tabular $q$, $v$, and $r$ terms with neural networks, building on the earlier Regression CFR method.

The first known combination of CFR with neural networks was **DeepStack**, one of the first systems to beat professional players at **heads-up no-limit Texas hold'em**. Another system was the **Libratus** method based on regret matching, later extended to make the **Pluribus** method, which could beat human players at the six-player version. **Student of Games** is a version of AlphaZero where CFR is the policy improvement operator, applied to Chess, Go, Poker, and Scotland Yard.

### Regularized Policy Gradient Methods

#### Magnetic Mirror Descent (MMD)

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Magnetic Mirror Descent)</span></p>

The **Magnetic Mirror Descent** or **MMD** algorithm is designed for two-player zero-sum games (but also works well for single player games). MMD is a modification of policy gradient that adds additional regularizers to ensure it converges. In the tabular case, applied at each decision point (state) $s$ and for each agent $i$ separately:

$$
\pi_{k+1} = \text{argmax}_\pi \langle \pi, q_k \rangle - \alpha \, D_{\text{KL}}(\pi, \rho) - \frac{1}{\eta} D_{\text{KL}}(\pi, \pi_k)
$$

where $q_k(a) = q_k(s, A)$ is the value of action $a$ in state $s$, $\rho$ is a magnet policy (designed to prevent oscillation), $\alpha$ is a regularization term (corresponding to entropy penalty if $\rho$ is uniform), and $\eta$ is a stepsize. For discrete actions, the optimal solution is

$$
\pi_{k+1} \propto [\pi_k \rho^{\alpha\eta} e^{\eta q_k}]^{\frac{1}{1+\alpha\eta}}
$$

If we drop the magnet term by setting $\alpha = 0$, the method is equivalent to the **mirror descent policy optimization** or **MDPO** algorithm: $\pi_{k+1} \propto [\pi_k e^{\eta q_k}]$ (the exponentiated gradient algorithm).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(MMD vs PPO)</span></p>

MMD is very similar to the KL-penalized version of PPO. In particular, the KL-penalized PPO uses:

$$
\mathbb{E}_{s_t, a_t} \left[\frac{\pi(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)} A_{\text{old}}(s_t, a_t) + \alpha \,\mathbb{H}(\pi(\cdot|s_t)) - \beta \, D_{\text{KL}}(\pi_{\text{old}}(\cdot|s_t), \pi(\cdot|s_t))\right]
$$

The main difference is just the use of a reverse KL instead of forwards KL. Despite the similarities, PPO has been shown to perform worse than MMD on various 2p0s games. Properly tuned policy gradient methods (including both PPO and MMD) can be competitive with or superior to model-free deep RL approaches based on fictitious play, population-based training, or counterfactual regret minimization in two-player zero-sum imperfect-information games ("Policy Gradient Hypothesis").

</div>

### Decision-Time Planning Methods

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Magnetic Mirror Descent Search - MMDS)</span></p>

Decision-time planning (DTP) methods improve upon a base policy (known as a **blueprint** policy) by doing some kind of forward search from the current state in a world model. The **update-equivalent DTP** method makes a connection between DTP and other policy update algorithms.

**Magnetic mirror descent search** (**MMDS**) generalizes this idea to the multi-agent setting by using the MMD algorithm as the update operator. The local policy update (for player $i$) has the form

$$
\pi_{\text{new}} = U(\pi, q) = \text{argmax}_{\pi' \in \Delta(A)} \langle \pi', q \rangle - \alpha \, D_{\text{KL}}(\pi', \rho) - \frac{1}{\eta} D_{\text{KL}}(\pi', \pi)
$$

where $\pi$ is the previous local (**blueprint**) policy and $\rho$ is the local magnet policy (which can be taken as uniform). If $h_t^i$ is the current state and the actions are discrete, we can equivalently perform an SGD step on the following parametric policy loss:

$$
\mathcal{L}(\boldsymbol{\theta}) = \sum_a \left[\bar{\pi}_\theta(a|h_t^i) q(h_t^i, a) - \alpha \bar{\pi}_\theta(a|h_t^i) \log \frac{\bar{\pi}_\theta(a|h_t^i)}{\rho(a)} - \frac{1}{\eta} \bar{\pi}_\theta(a|h_t^i) \log \frac{\bar{\pi}_\theta(a|h_t^i)}{\pi_{\text{old}}(a|h_t^i)}\right]
$$

Note that if we use a uniform magnet, this is equivalent to adding an entropy regularizer. For common-payoff games, we can drop the magnet term, giving rise to the simpler **mirror descent search** method.

</div>

#### Belief State Approximations

To implement MMDS, we need to sample from $P_\pi(s_t|h_t^i)$, the distribution over world states given agent $i$'s local history. One approach is particle filtering. Another is to train a **belief model** to predict the other player's private information, and the underlying environment state, given the current player's history. The **learned belief search** (LBS) method (designed for Hanabi, which is a Dec-POMDP) learns to predict private information (card hands) for each agent as a sequence of tokens.

### MARL for LLM Agents

A recently growing trend is to use LLMs as agents, which can be made to interact with each other via protocols such as **A2A** (agent-to-agent). It is possible to apply MARL techniques to optimize such systems, and to use self-play to train LLMs.

# Chapter 6: LLMs and RL

## Introduction

This chapter discusses connections between RL and **foundation models**, also called **large language models** or **LLMs**. LLMs are generative models (usually based on auto-regressive transformers) which are trained on large amounts of web data. When also trained on visual data, LLMs are sometimes called **vision language models** or **VLMs**, and when trained on action data, they are called **vision language action** models or **VLAs**.

## RL for LLMs

### RL Fine Tuning (RLFT)

LLMs are usually trained with behavior cloning (i.e., MLE on a fixed dataset, such as a large text corpus scraped from the web). This is called **pre-training**. We can then improve their performance using various **post-training** methods, which are designed to improve their capabilities and **alignment** with human preferences. A simple way to perform post-training is to use **instruction fine tuning**, also called **supervised fine-tuning** (or **SFT**), in which we collect human demonstrations of (prompt, response) pairs, and fine-tune the model on them.

An alternative to demonstrating good behaviors is to use RL to train the model using a suitable reward function. This is called **reinforcement learning fine-tuning** or **RLFT**.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why RLFT over SFT)</span></p>

RLFT can be preferable to SFT for several reasons:

1. Verification is easier than generation (e.g., it is easier to ask people which answer they prefer rather than to ask them to generate good answers).
2. RL can be used to learn a set of "thinking actions", which are created in response to the question before generating the answer. For complex problems (e.g., in math), this tends to work much better than trying to directly learn an input-output mapping.
3. RL opens the path to super-human performance, going beyond whatever supervised examples humans can create.

</div>

### Reward Models

#### RL with Verifiable Rewards (RLVR)

For problems such as math and coding, it can be easy to determine if an answer is correct, by checking equality between the generated answer and the true answer (for math), or checking if a set of unit tests pass (for code). This allows us to define a binary reward signal. Using RL with such a reward is called **"RL with verifiable rewards"** or **RLVR**.

#### Process vs Outcome Reward Models

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(PRM and ORM)</span></p>

If the reward function $R(s_t)$ is defined on partial trajectories, it is called a **process reward model** or **PRM**. This provides a form of dense feedback. If the reward is just defined on the final sequence $R(s_T) = R(s_0, \boldsymbol{a}_{1:T})$, it is called an **outcome reward model** or **ORM**, and corresponds to a sparse reward.

For example, suppose we are solving a math problem using a thinking model: if we just check the final answer, we have an ORM, but if we also check correctness of the intermediate proof steps, we have a PRM. Note that a PRM is related to a value function (that models expected future reward), and is typically harder to learn than an ORM.

</div>

#### Learning the Reward Model from Human Feedback (RLHF)

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(RLHF and the Bradley-Terry Model)</span></p>

To train LLMs to do well in general tasks (such as text summarization or poetry writing), it is common to use **reinforcement learning from human feedback** or **RLHF**, which refers to learning a reward model from human data, and then using RL to train the LLM to maximize this.

We generate a large number of (context, answer1, answer2) tuples. We then ask human raters if they prefer answer 1 or answer 2. Let $x$ be the prompt (context), $y_w$ be the winning (preferred) output, and $y_l$ be the losing output. Let $r_\theta(x, y)$ be the reward assigned to output $y$. (This model is typically a shallow MLP on top of the last layer of a pretrained LLM.) We train the reward model by maximizing the likelihood of the observed preference data using the **Bradley-Terry choice model**:

$$
p_\theta(y_w \succ y_l) = \frac{\exp(r_\theta(x, y_w))}{\exp(r_\theta(x, y_w)) + \exp(r_\theta(x, y_l))}
$$

Equivalently we can minimize

$$
\mathcal{L}(\boldsymbol{\theta}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[\log\left(\sigma(r_\theta(x, y_w) - r_\theta(x, y_l))\right)\right]
$$

</div>

#### Learning the Reward Model from AI Feedback (RLAIF)

Instead of asking humans their preferences, we can ask an LLM to predict the preference. This is called **LLM as judge**. We can then fit the reward model to this synthetically labeled data (just as in RLHF). Alternatively, we can just ask the LLM to predict the reward directly. This is called **RLAIF**, which stands for RL from AI feedback.

Anthropic created a technique called **constitutional AI**, where the prompt is viewed as a "constitution", which specifies what kinds of responses are desirable or undesirable. With this method, the system can critique its own outputs, and thus self-improve.

#### Generative Reward Models (GRM)

A **generative reward model** or **GRM** predicts the reward for a given response, but also returns its chain of thought, thus providing richer textual feedback. In addition to passing the scalar reward to an RL algorithm, the textual feedback can be parsed by the LLM itself to decide how to improve the policy. For example, the **GEPA** algorithm uses an evolutionary algorithm to optimize prompts for a frozen LLM, by mutating them given textual feedback from a GRM.

### Agents which "Think"

#### Chain of Thought Prompting

The quality of the output from an LLM can be improved by prompting it to "show its work" before presenting the final answer. These intermediate tokens are called a **"Chain of Thought"** (CoT). Models that act in this way are often said to be doing **"reasoning"** or **"thinking"**, although in less anthropomorphic terms, we can think of them as just policies with dynamically unrolled computational graphs. This is motivated by various theoretical results that show that such CoT can significantly improve the expressive power of transformers.

#### Training a Thinking Model Using RL

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(DeepSeek-R1-Zero and Reasoning Models)</span></p>

Rather than just relying on prompting, we can explicitly train a model to think by letting it generate a variable number of tokens "in its head" before generating the final answer. Only the final outcome is evaluated, using a known reward function (as in the case of math and coding problems).

This approach was recently demonstrated by the **DeepSeek-R1-Zero** system (released January 2025). They started with a strong LLM base model (**DeepSeek-V3-Base**), pre-trained on a large variety of data (including Chains of Thought). They then used a variant of PPO, known as **GRPO**, to do RLFT, using a set of math and coding benchmarks where the ground truth answer is known. The closed-source models **ChatGPT-o1** and **ChatGPT-o3** from OpenAI and the **Gemini 2.0 Flash Thinking** model from Google Deepmind are believed to follow similar principles.

We can view training a thinking model as equivalent to maximizing the marginal likelihood $p(y|x) = \sum_z p(y, z|x)$, where $z$ are the latent thoughts.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Can RL Bootstrap Thinking from Scratch?)</span></p>

The claim that RL "caused" emergent abilities (such as self-reflection) has been disputed. The general consensus is that the base model itself was already trained on datasets that contained some COT-style reasoning patterns. Applying RL to a base model that had not been pre-trained on reasoning patterns did not result in a final model that could exhibit such behaviors. However, RL can "expose" or "amplify" such abilities in a base model if they are already present to a certain extent.

The **Absolute Zero Reasoner** showed it is possible to automatically generate a curriculum of (programs, inputs, outputs), which is used to improve the math and coding abilities of the LLM using RL: the LLM is trained to perform induction, deduction, and abduction.

</div>

#### Agentic AI

**Agentic AI** systems consist of a set of interacting LLMs, often called "agents", which are essentially different prompts reflecting different roles or personas. Typically these prompts, and the way the different agents interact, are hand-designed --- this is called a **workflow** or **scaffolding**. Such agents may process the input, some may access or process memory, and some may call tools such as web search. Note, however, that unlike the true multi-agent setup of Chapter 5, these "agents" do not maximize their own reward functions, and are really just a set of modules inside a single larger agent. These workflows are usually hand-engineered but they can be improved using RL.

### Algorithms for Single-Turn RL

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(LLM as an MDP)</span></p>

In the single-turn setup, there is just a single state, namely the input prompt $s$; the action is a sequence of tokens generated by the policy in response, and then the game ends. This is equivalent to a contextual bandit problem, with sequence-valued input (context) and output (action). Formally, the goal is to maximize

$$
J(\boldsymbol{\theta}) = \mathbb{E}_{s \sim \mathcal{D}, \boldsymbol{a} \sim \pi_\theta(\boldsymbol{a}|s)} [R(s, \boldsymbol{a})]
$$

where $\pi_\theta(\boldsymbol{a}|s) = \prod_{t=1}^{T} \pi_\theta(a_t|\boldsymbol{a}_{1:t-1}, s)$ and $T = |\boldsymbol{a}|$ is the length of the generated output (terminated by generating an $\langle\text{eos}\rangle$ token).

We can convert this into an MDP by defining the deterministic state transition $p(s_t|s_{t-1}, a_t) = \delta(s_t = \text{concat}(s_{t-1}, a_t))$, with initial distribution $\delta(s_0 = s)$. Thus the state $s_t$ is just the set of tokens from the initial prompt $s$ plus the generated tokens up until time $t$. The reward is sparse:

$$
R(s_t, a_t) = \begin{cases} 0 & \text{if } t < T \\ R(s_T, a_T = \text{eos}) = R(s, a_1, \ldots, a_T) & \text{if } t = T \end{cases}
$$

In practice, we usually regularize the problem to ensure the policy $\pi_\theta$ remains close to the base pre-trained LLM $\pi_{\text{ref}}$, by adding a penalty $-\beta D_{\text{KL}}(\pi_\theta(a_t|s_t) \| \pi_{\text{ref}}(a_t|s_t))$ to the per-token reward.

</div>

#### PPO for LLMs

A natural approach is to use PPO. In the bandit case:

$$
J_{\text{ppo}}(\boldsymbol{\theta}) = \mathbb{E}_{s_n \sim D} \mathbb{E}_{\boldsymbol{a}_n \sim \pi_{\text{old}}(\cdot|s_n)} \min\left(\rho_n(\boldsymbol{\theta}) \cdot A_n, \text{clip}(\rho_n(\boldsymbol{\theta}), \cdot A_n)\right)
$$

where $\rho_n(\boldsymbol{\theta}) = \frac{\pi_\theta(\boldsymbol{a}_n|s_n)}{\pi_{\text{old}}(\boldsymbol{a}_n|s_n)}$ is the likelihood ratio, $A_n = R_n - b_n$ is the advantage, $R_n = R(s_n + \boldsymbol{a}_n)$ is the trajectory-level reward, and $b_n = V(s_n + \boldsymbol{a}_n)$ is the baseline.

#### GRPO

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Group Relative Policy Optimization)</span></p>

The **Group Relative PPO** or **GRPO** algorithm (used to train DeepSeek-R1-Zero) is a variant of PPO which replaces the critic network with a Monte Carlo estimate of the value function. For each prompt $s_n$, we generate $J$ answers $\boldsymbol{a}_n^j \sim \pi_{\text{old}}(\cdot|s_n)$ (called a **group**, often of size $J \sim 8$) which give final rewards $R_n^j$. We compute the advantage by subtracting the group average and dividing by the group standard deviation:

$$
\hat{A}_n^j = \frac{R_n^j - \mu_n}{\sigma_n}
$$

where $\mu_n = \text{mean}(R_n^j)$ and $\sigma_n = \text{std}(R_n^j)$. The use of the normalization term ensures the rewards are **calibrated**. This is related to the use of **reward centering** in continual RL.

Since the policy generates a sequence, we expand out the loss for each sequence into a sum of per-token losses. In GRPO, the step-level advantage is set to equal the normalized trajectory-level advantage, $\hat{A}_{nt}^j = \hat{A}_n^j$:

$$
J_{\text{GRPO}}(\boldsymbol{\theta}) = \frac{1}{N} \sum_{n=1}^{N} \frac{1}{J} \sum_{j=1}^{J} \frac{1}{|\boldsymbol{a}_n^j|} \sum_{t=1}^{|\boldsymbol{a}_n^j|} \min\left(\rho_{nt}^j(\boldsymbol{\theta}) \hat{A}_{nt}^j, \text{clip}(\rho_{nt}(\boldsymbol{\theta}), 1-\epsilon, 1+\epsilon) \hat{A}_{nt}^j\right)
$$

where $\rho_{nt}^j(\boldsymbol{\theta}) = \frac{\pi_\theta(a_{nt}^j|a_{n,<t}, s_n)}{\pi_{\text{old}}(a_{nt}^j|a_{n,<t}, s_n)}$ is the per-token likelihood ratio.

</div>

#### DAPO

The **DAPO** method suggests an asymmetric clipping of the likelihood ratio term: $\text{clip}(\rho_{nt}(\boldsymbol{\theta}), 1 - \epsilon_{\text{low}}, 1 + \epsilon_{\text{high}}) \hat{A}_{nt}^j$, with $\epsilon_{\text{high}} = 0.28 > \epsilon_{\text{low}} = 0.2$, so that actions which are low-probability under the previous model (and which therefore get large likelihood ratio $\rho_{it}$) are not clipped as much, which would otherwise suppress exploration and result in "entropy collapse".

#### GSPO

The **GSPO** (Group Sequence Policy Optimization) method points out a flaw with GRPO: the importance sampling correction $\rho_{nt}^j$ is applied to each token, even though the reward is evaluated at the trajectory level, which can result in unstable training. They therefore propose to use a sequence-level importance ratio:

$$
\rho_n^j(\boldsymbol{\theta}) = \left(\frac{\pi_\theta(\boldsymbol{a}_n^j|s_n)}{\pi_{\text{old}}(\boldsymbol{a}_n^j|s_n)}\right)^{1/|\boldsymbol{a}_n^j|} = \exp\left(\frac{1}{|\boldsymbol{a}_n^j|} \sum_{t=1}^{|\boldsymbol{a}_n^j|} \log \frac{\pi_\theta(a_{nt}^j|\boldsymbol{a}_{n,<t}^j, s_n)}{\pi_{\text{old}}(a_{nt}^j|\boldsymbol{a}_{n,<t}^j, s_n)}\right)
$$

#### RLOO

The division by the standard deviation used by GRPO when normalizing the advantage terms induces a **difficulty bias** (a very easy or hard prompt $s_n$ may have a low group-level standard deviation $\sigma_n$, and dividing by a small $\sigma_n$ can result in unstable gradients). The **Dr GRPO** (GRPO Done Right) method simply drops the denominator:

$$
\hat{A}_{nj}^{\text{DrGRPO}} = R_{nj} - \mu_n, \quad \mu_n = \frac{1}{J}\sum_{j=1}^{J} R_{nj}
$$

The **RLOO** (Reinforce Leave-One-Out) method proposes the baseline:

$$
\hat{A}_{nj}^{\text{RLOO}} = R_{nj} - \mu_n, \quad \mu_n = \frac{1}{J-1} \sum_{j=1, j \ne i}^{J} R_{nj}
$$

which is the average reward for all samples in the batch, excluding the current sample. RLOO is identical (up to a scaling factor) to Dr GRPO.

#### REINFORCE++

The **REINFORCE++** algorithm computes the mean and standard deviation globally across the batch (rather than per-prompt), which is unbiased and improves training stability:

$$
\hat{A}_n^j = \frac{R_n^j - \mu}{\sigma}, \quad \mu = \text{mean}(R_n^j : n = 1:N, j = 1:J), \quad \sigma = \text{std}(R_n^j : n = 1:N, j = 1:J)
$$

#### VinePPO

GRPO and related methods compute an unbiased estimate of the value of each token based on rolling out multiple trajectories from the same starting state (prompt). All intermediate states (action tokens) are treated equally. For long chains of thought, this can be a poor estimate of the true RTG of an intermediate token. **VinePPO** exploits the fact that in language-based environments, it is possible to reset directly to any intermediate state simply by refeeding the partial context; this enables multiple MC rollouts from any state in the trajectory, thus providing a more accurate value estimate without fitting a value function network. Unfortunately this technique is slow and not general purpose.

#### Adding a KL Regularizer

It is common to add a KL penalty to the per-step reward, to prevent the policy from deviating too far from the base (SFT) LLM:

$$
\hat{R}_{n,t}^{J} = R_n^j - \beta D_{\text{KL}}\left(\pi_{\text{old}}(a_{nt}^j|s_n, \boldsymbol{a}_{n,<t}^j) \| \pi_{\text{ref}}(a_{nt}^j|s_n, \boldsymbol{a}_{n,<t}^j)\right)
$$

In GRPO (and many other papers), they use a low-variance MC estimator of KL divergence. The naive estimator of $\text{KL}(q, p)$ is $k_1 = \log(r) = \log q(a) - \log p(a)$ where $a \sim q$ and $r(a) = q(a)/p(a)$. A biased but lower-variance estimator is $k_2 = \frac{1}{2}(\log(r))^2$. An unbiased estimator with low variance is $k_3 = (r-1) - \log(r)$, although its gradient is biased, so $k_2$ is often recommended in practice.

#### DPO

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Direct Preference Optimization)</span></p>

Rather than first fitting a reward model from preference data using the Bradley-Terry model, and then optimizing the policy to maximize this, it is possible to optimize the preferences directly, using the **DPO** (Direct Preference Optimization) method. (This is sometimes called **direct alignment**.)

The objective for KL-regularized policy learning is to maximize

$$
J(\pi) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi(y|x)} \left[R(x, y) - \beta \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}\right]
$$

The optimal solution is

$$
\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta} R(x, y)\right)
$$

from which we can derive the optimal reward as $R^*(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$.

Plugging this into the Bradley-Terry model, the $Z(x)$ terms cancel, and we can fit the policy by minimizing

$$
\mathcal{L}(\boldsymbol{\theta}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[\log \sigma \left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]
$$

The main downside of DPO is that it is limited to learning from preference data, whereas the other policy gradient methods can work with any reward function, including verifiable (non-learned) rewards.

</div>

#### Inference-Time Scaling Using Posterior Sampling

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Inference-Time Scaling)</span></p>

We can view the problem of generating samples from an LLM that maximizes some reward as equivalent to posterior sampling from a **tilted distribution**, that combines the prior $\pi_{\text{ref}}(y|x)$ with a likelihood $p(O=1|x,y) = \exp(R(x,y)/\beta)$. The target posterior is:

$$
\pi^*(y|x) \propto e^{\frac{1}{\beta}R(x,y)} \pi_{\text{ref}}(y|x)
$$

This is the optimal solution to the KL-regularized RL problem: $\max_{\pi(\cdot|x)} \mathbb{E}_{\pi(y|x)} R(x,y) - \beta D_{\text{KL}}(\pi(\cdot|x) \| \pi_{\text{ref}}(\cdot|x))$.

Note that if we set $\beta = 1$ and define $R(x,y) = \log(\pi_{\text{ref}}(y|x)^{\alpha-1})$, the optimal solution becomes the **tempered distribution** or **power distribution**: $\pi^*(y|x) = \pi_{\text{ref}}(y|x)^\alpha$, which can be flatter if $\alpha < 1$ or sharper if $\alpha > 1$.

Sampling from such distributions can be done using various methods:
* **Best-of-N** sampling: generate $N$ trajectories, pick the best. This is equivalent to ancestral sampling from the forwards model, and then weighting by the final likelihood (a soft version of rejection sampling). However, performance can decrease when $N$ increases, due to deviating too far from the base model.
* **Twisted SMC** (sometimes called **SMC steering**): combines particle filtering with a "twist" function which predicts the future reward given the current state, analogous to a value function.

The posterior sampling approach provides another kind of scaling law, known as **test time compute** or **inference time scaling**, besides just improving the size (and training time) of the base model.

</div>

#### RLFT as Amortized Posterior Sampling

The disadvantage of decision-time planning (online posterior sampling) is that it can be slow. Hence it can be desirable to "amortize" the cost by fine-tuning the base LLM so it matches the tilted posterior. We can do this by minimizing $J(\boldsymbol{\theta}) = D_{\text{KL}}(\pi_\theta \| \pi^*)$. This is equivalent to maximizing the ELBO:

$$
J'(\boldsymbol{\theta}) = \mathbb{E}_{\pi_\theta(y)} [\log \Phi(y)] - D_{\text{KL}}(\pi_\theta(y) \| \pi_{\text{ref}}(y)) \propto \mathbb{E}_{\pi_\theta(y)} [R(y)] - \beta D_{\text{KL}}(\pi_\theta(y) \| \pi_{\text{ref}}(y))
$$

which is exactly the KL-regularized RL objective used in DPO.

The advantage of this probabilistic (distribution matching) perspective over the RL (reward maximizing) perspective is that it suggests natural alternatives, such as optimizing the inclusive or forwards KL, $D_{\text{KL}}(\pi^* \| \pi_\theta)$, which is "mode covering" rather than "mode seeking". This can prevent "catastrophic forgetting", in which the tuned policy loses diversity as well as some of its original capabilities. (This is similar to the advantage of **reweighted wake sleep** training compared to amortized VI training for latent variable models.)

Note, however, that these offline approaches to LLM finetuning have the disadvantage, compared to the online (decision-time) approach to inference, that they cannot easily handle hard constraints, since they only train policies that respect the constraints on average.

### Algorithms for Multi-Turn RL

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Multi-Turn RL for LLMs)</span></p>

So far, we have considered the bandit setting, in which a single prompt $s$ is presented, and then the agent optionally generates some thinking tokens, followed by the answer tokens, and then immediately gets a reward. In the multi-turn setting, we discuss how to train agents that can interact with an external environment, which enables **tool use** or **dialog agents**. The difference from the standard contextual bandit LLM reasoning setting is that the effect of an action on the external environment is typically unknown, and may be stochastic, and the reward may be delayed. In addition, the external environment is often stateful, so actions may be irreversible. Training agents in this setting requires true **multi-turn RL** methods.

</div>

#### Example: RAGEN

The **RAGEN** system illustrates multi-turn RL. The policy generates a set of thinking tokens, followed by a set of action tokens, until the STOP token is generated (representing the end of action). The policy is used to generate $N$ trajectories, from which the advantage can be estimated using a Monte Carlo estimate. They call this algorithm **StarPO** (State-Thinking-Action-Reward Policy Optimization).

#### Dealing with Invalid Actions

When interacting with the external environment, some trajectories may result in an **invalid action** during generation. Several ways to deal with such **partial rollouts** include: (1) truncate the sequence to the step where the error occurred; (2) start a new rollout from the failure state; (3) replace the invalid action with a random legal action and continue the rollout.

#### Turn-Level Training

Methods like RAGEN do not work well in long-horizon tasks for at least two reasons: (1) the context length (which is input to the policy) can grow very large; and (2) the REINFORCE estimator can be very high variance once the number of steps needed to reach a terminal reward becomes large. We can tackle (1) by truncating (or summarizing, using an LLM) the history of previous states and actions. We can tackle (2) by learning a value function or critic, and then using the Generalized Advantage Estimator to compute $A_t^n$, instead of using the Monte Carlo estimate. The loss (for the on-policy case) becomes

$$
\mathcal{L}_{A2C}(\theta) = \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} A_t^n \log \pi_\theta(a_t^n|s_t^n)
$$

An alternative is to replace the GAE estimate with return batch normalization (**ReBN**), which is similar to GRPO but defined for the multi-turn setting: $A_t^n = \frac{G_t^n - \mu}{\sigma}$, $\mu = \text{mean}(G_{1:T}^{1:N})$, $\sigma = \text{std}(G_{1:T}^{1:N})$.

#### Self-Play for LLM Training

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Red-Teaming and SPIRAL)</span></p>

Although RL is mostly used to train LLMs to act in single-agent environments, it is also possible to train them to act in multi-agent environments. One application is **red-teaming**: training an LLM adversary to attack an LLM defender in a two-player zero-sum adversarial game setup to improve LLM safety. The goal is to find the Nash equilibrium:

$$
\min_{x \sim \pi_{\text{Adv}}} \max_{y \sim \pi_{\text{Def}}} r_\theta(x, y)
$$

At the Nash equilibrium $(\pi_A^*, \pi_D^*)$, we have $r_\theta(x, \pi_D^*(x)) \ge 0$ for any prompt $x \sim \pi_A^*$, meaning the response is safe.

The **SPIRAL** algorithm trains a single LLM to play various zero-sum two-player games (Tic-Tac-Toe, Kuhn Poker, and Simple Negotiation) using self-play. They show that the resulting system is better able to solve math and reasoning problems, which share some commonalities with game play. Training uses REINFORCE with **Role-conditioned Advantage Estimation** (**RAE**), which is an EMA estimator: $b_{G,p} = \alpha b_{G,p} + (1-\alpha) R_p(\tau)$, $A_{G,p}(\tau) = R_p(\tau) - b_{G,p}$.

</div>

### Alignment and the Assistance Game

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Reward Misspecification and the Assistance Game)</span></p>

Encouraging an agent to behave in a way that satisfies one or more human preferences is called **alignment**. We can use RL for this, by creating suitable reward functions. However, any objective-maximizing agent may engage in reward hacking, in which it finds ways to maximize the specified reward but which humans consider undesirable. This is due to **reward misspecification**, or simply, the law of unintended consequences (classically illustrated by *The Sorcerer's Apprentice* and the **cobra effect**). This is summarized in **Goodhart's Law**: "When a measure becomes a target, it ceases to be a good measure."

One proposed solution is the **assistance game**, where the human and machine are both treated as agents in a two-player, partially observed cooperative game (an instance of a Dec-POMDP). The machine's goal is to maximize the user's utility (reward) function, which is inferred based on the human's behavior using inverse RL. That is, instead of trying to learn the reward function using RLHF and then optimizing that, we treat the reward function as an unknown part of the environment. If we adopt a Bayesian perspective, the machine maintains a posterior belief over the model parameters, which will incentivize the agent to perform information-gathering actions.

</div>

## LLMs for RL

In this section, we discuss how to use LLMs to help create agents that themselves may or may not use language. The LLMs can be used for their prior knowledge, their ability to generate code, their "reasoning" ability, and their ability to perform **in-context learning**. The literature groups into four main categories: LLMs for pre-processing the inputs, LLMs for rewards, LLMs for world models, and LLMs for decision making or policies.

### LLMs for Pre-Processing the Input

If the input observations $\boldsymbol{o}_t$ sent to the agent are in natural language (or some other textual representation, such as JSON), it is natural to use an LLM to process them, in order to compute a more compact representation, $\boldsymbol{s}_t = \phi(\boldsymbol{o}_t)$, where $\phi$ can be the hidden state of the last layer of an LLM. This encoder can either be frozen, or fine-tuned with the policy network. We can also pass in the entire past observation history, $\boldsymbol{o}_{1:t}$, as well as static "side information", such as instruction manuals or human hints; these can all be concatenated to form the LLM prompt.

#### Example: AlphaProof

The **AlphaProof** system uses an LLM (called the "formalizer network") to translate an informal specification of a math problem into the formal Lean representation, which is then passed to an agent (called the "solver network") which is trained, using the AlphaZero method, to generate proofs inside the Lean theorem proving environment. The reward is 0 or 1 (proof is correct or not), the state space is a structured set of previously proved facts and the current goal, and the action space is a set of proof tactics. The agent itself is a separate transformer policy network (distinct from the formalizer) that is a pre-trained LLM fine-tuned on math, Lean and code, and then further trained using RL.

#### VLMs for Parsing Images into Structured Data

If the observations are images, it is traditional to use a CNN to process the input, so $\boldsymbol{s}_t \in \mathbb{R}^N$ would be an embedding vector. However, we could alternatively use a VLM to compute a structured representation, where $\boldsymbol{s}_t$ might be a set of tokens describing the scene at a high level, or potentially a JSON dictionary. We can then pass this symbolic representation to the policy function. We can also fine tune the VLM with RL.

#### Active Control of LLM Sensor/Preprocessor

The information that is extracted will heavily depend on the prompt that is used. Thus we should think of an LLM/VLM as an **active sensor** that we can control via prompts. Choosing how to control this sensor requires expanding the action space of the agent to include computational actions. Note also that these kinds of "sensors" are very expensive to invoke, so an agent with some limits on its time and compute will need to reason about the value of information and the cost of computation. This is called **metareasoning**. Devising good ways to train agents to perform both computational actions (e.g., invoking an LLM or VLM) and environment actions (e.g., taking a step in the environment or calling a tool) is an open research problem.

### LLMs for Rewards

It is difficult to design a reward function to cause an agent to exhibit some desired behavior. Fortunately LLMs can often help with this task, especially when using goal-conditioned RL.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(LLM-Based Reward Design)</span></p>

Several approaches exist for using LLMs to design rewards:

* The **Motif** system uses an LLM in lieu of a human to provide preference judgements to an RLHF system. A pre-trained policy is used to collect trajectories, from which pairs of states $(\boldsymbol{o}, \boldsymbol{o}')$ are selected at random. The LLM is then asked which state is preferable, generating $(\boldsymbol{o}, \boldsymbol{o}', y)$ tuples from which a binary classifier (reward model) is extracted. The learned reward model is then used in lieu of the environment reward, or as a shaping function.
* The **Eureka** system learns the reward using bilevel optimization, with RL on the inner loop and LLM-powered evolutionary search on the outer loop. In the inner loop, given a candidate reward function $R_i$, we use PPO to train a policy, and then return a scalar quality score $S_i = S(R_i)$. In the outer loop, we ask an LLM to generate a new set of reward functions $R_i'$, given a population of old reward functions and their scores. Each generated reward function $R_i$ is represented as a Python function that has access to the ground truth state of the underlying robot simulator.
* **Code as reward**: prompt a VLM with an initial and goal image, ask it to describe the corresponding sequence of subtasks needed to reach the goal, then ask the LLM to synthesize code that checks for completion of each subtask. These reward functions are then "verified" by applying them to an offline set of expert and random trajectories.

</div>

### LLMs for World Models

We can use LLMs to create world models of the form $p(s'|s, a)$. We can either do this by treating the LLM itself as a WM (which is then updated using in-context learning), or asking the LLM to generate another artefact, such as some python code, that represents the WM. The advantage of the latter approach is that the resulting WM will be much faster to run, and may be more interpretable.

#### LLMs as World Models

In principle it is possible to treat a pre-trained LLM (or other kind of foundation model) as an implicit model of the form $p(s'|s, a)$ by sampling responses to a suitable prompt, which encodes $s$ and $a$. This rarely works out of the box, but it can be made to work by suitable pre-training.

For example, **UniSim** is an action-conditioned video diffusion model trained on large amounts of robotics and visual navigation data. Combined with a VLM reward model, this can be used for decision-time planning: sample a candidate action sequence, generate the corresponding images, feed them to the reward model, score the rollouts, and then pick the best action from this set. This is just standard model-predictive control in image space with a diffusion WM and a random shooting planning algorithm.

#### LLMs for Generating Code World Models

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Code World Models)</span></p>

Calling the LLM at every step to sample from the WM $p(s'|s, a)$ is very slow, so an alternative is to use the LLM to generate code that represents the world model. This is called a **code world model** (CWM).

* **GIF-MCTS** (Generate, Improve and Fix with Monte Carlo Tree Search) learns CWMs given a natural language description of the task, and a fixed offline dataset of trajectories. The method maintains a representation of the posterior over the WM, $M = p(s'|s, a)$, as a tree of partial programs. At each step, a node is chosen from the tree using the UCT formula, then expanded in one of three ways: (G) the LLM generates code to solve the task; (I) the LLM improves the current code so it passes more unit tests; (F) the LLM fixes execution bugs in the current code.
* **WorldCoder** learns a CWM in an online fashion by interacting with the environment and prompting an LLM. It maintains a sample-based representation of the posterior of $p(M|\mathcal{D}(1:t))$, where the weight for each sampled program $\rho$ is represented by a Beta distribution. At each step, it samples one of these models from this weighted posterior, and then uses it inside of a planning algorithm, similar to Thompson sampling. The agent then executes this in the environment, and passes back failed predictions to the LLM, asking it to improve the WM, or to fix bugs if it does not run. This refinement step is similar to the I and F steps of GIF-MCTS.
* **FunSearch** (recently rebranded as **AlphaEvolve**) uses the LLM as a mutation operator inside of an evolutionary search algorithm, where the goal is to search over program space to find code that minimizes some objective, such as prediction errors on a given dataset.

</div>

#### LLMs for Generating Partial Code World Models

This approach can be extended to create a **PoE World Model**, defined in terms of a product of experts. Each term in the product is a distribution over a single element of the state space, and the terms themselves are defined as a product of deterministic experts, each of which is learned using code synthesis. This approach allows parts of the world model to be learned independently, although it does not capture constraints or correlations between the parts.

### LLMs for Policies

We can use LLMs for creating policies. We can either do this by treating the LLM itself as a policy (which is then updated using in-context learning), or asking the LLM to generate some code that represents the policy.

#### LLMs for Generating Actions

We can sample an action from a policy $\pi(a_t|o_t, h_{t-1})$ by using an LLM, where the input context contains the past data $(o_t, h_{t-1})$, and then output token is interpreted as the action. For this to work, the model must be pre-trained on state-action sequences using behavior cloning. See e.g., **Gato**, **RT-2**, and **RoboCat**.

An alternative approach is to enumerate all possible discrete actions, and use the LLM to score them in terms of their likelihoods given the goal, and their suitability given a learned value function applied to the current state: $\pi(a_t = k|g, p_t, o_t, h_t) \propto \text{LLM}(w_k|g_t, p_t, h_t) V_k(o_t)$, where $g_t$ is the current goal, $w_k$ is a text description of action $k$, and $V_k$ is the value function for action $k$. This is the approach used in the robotics **SayCan** approach, where the primitive actions $a_k$ are separately trained goal-conditioned policies.

Alternatively, we can use a general purpose pre-trained LLM, combined with a suitable prompt chosen by the human, to request the LLM to generate the right kind of output. This approach is used by the **ReAct** paper which works by prompting the LLM to do some Chain of Thought reasoning before acting. This approach can be extended by giving feedback on earlier actions, a technique called **Reflexion**. We can also prompt the LLM to first retrieve relevant past examples from an external "memory", rather than explicitly storing the entire history $h_t$ in the context (this is called **retrieval augmented generation** or **RAG**).

Note that no explicit learning (in the form of parametric updates) is performed in any of these systems; instead they rely entirely on in-context learning and **prompt engineering** / **context engineering**.

#### LLMs for Generating Code Policies

Calling the LLM at every step is very slow, so an alternative is to use the LLM to generate code that represents (parts of) the policy. This is called a **code policy**.

For example, the **Voyager** system builds up a reusable skill library (represented as Python functions), by alternating between environment exploration and prompting the (frozen) LLM to generate new tasks and skills, given the feedback (environment trajectories) collected so far.

We can also use the LLM as a mutation operator inside of an evolutionary search algorithm, as in the **FunSearch** system, where the objective is to maximize performance of the generated policy when deployed in one or more environments.

#### LLMs for Generating Code Actions

An alternative to asking the LLM to generate a code policy is to ask the LLM to generate code for the current action (see e.g., the **CodeAct** system). This is often better than asking it to call a tool multiple times, since the generated code can represent this "action chunk" with a for-loop, and add extra logic. This is different to a policy, since it is not a function mapping all states to actions. Instead it is generating a (potentially closed-loop) plan to be executed from the current state.

#### In-Context RL

Large LLMs have shown a surprising property, known as **In-Context Learning** or **ICL**, in which they can be "taught" to do function approximation just by being given $(x, y)$ pairs in their context (prompt), and then being asked to predict the output for a novel $x$. This can be used to train LLMs without needing to do any gradient updates to the underlying parameters.

### Speeding Up LLMs

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Computational Complexity of Transformer Policies)</span></p>

Most LLMs are based on transformers. These work well, but can be very slow, particularly for long trajectories and/or settings in which each observation takes many tokens to encode (e.g., an image). LLM policies are non-Markovian models of the form $p(a_t|o_{1:t}, a_{1:t-1})$ which need $O(t)$ time at step $t$ to generate an action, so generating a trajectory of length $T$ takes $O(T^2)$ time and $O(T)$ memory. This is problematic for **lifelong learning agents**, as well as problems with real-time constraints.

**Time-to-first-token** (TTFT) latency --- the time to generate the first output token $y_1$ given $x_{1:N}$ (the **prefill phase**) --- grows as $O(N^2)$ due to the use of full cross attention on $x$.

To generate $M$ subsequent tokens, the **time-to-iterative-token** generation (the **decode phase**) is $O(NM + M^2)$. With **KV caching** (avoiding recomputing previous keys and values), when we generate a new token at step $t$, the model only needs to compute the Query for the single token it just produced, then compares to all $N_t = N + t$ previously computed keys. Thus the first decoding step takes $O(N+1)$ time, the second $O(N+2)$ time, all the way up to the $M$'th step which takes $O(N+M)$ time, for a total decoding cost of $O(NM + M^2)$.

When using an LLM as a policy, we are basically always in the decode phase. The first action takes $O(1)$ time, the second action takes $O(2)$ time, the $t$'th action takes $O(t)$ time.

</div>

#### Modern RNNs

To reduce the per-step complexity to $O(1)$, while still maintaining non-Markovianity, it is possible to use RNN-type policies. Traditional RNNs are slow to train, since they have a serial dependency between the latent states. More "modern" RNNs --- such as **xLSTMs** and linear **state space models** (SSMs) such as **Mamba** --- enjoy fast parallel offline training, like transformers, but still support constant time inference.

Although these linear RNNs are fast, they are not as accurate as transformers. However, it is possible to create hybrid RNN/transformer models, such as **Griffin**, which combines recurrent connections with local attention, in alternating layers. One can also develop RNN-type methods based on **test-time training** (TTT), such as **ATLAS** and **MesaNet**, which perform an inner iterative (or closed-form) optimization of the state after each step.

These "modern" RNNs can be used for world models (e.g., using Mamba), for policies (e.g., using xLSTM or Mamba), or for representing any other kind of non-Markovian function.

## Implementation Details

In addition to the algorithmic issues discussed above, RL for LLMs (which are very large models) requires a lot of engineering effort, to ensure things run efficiently and stably.

### Policy Gradient Using Tinker

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Multi-Turn RL Implementation)</span></p>

We can implement a multi-step LLM agent using policy gradient descent with the Tinker library, which performs asynchronous rollouts (sampling) and gradient-based computation in the cloud. At each step of policy updating, it rolls out some episodes, computes the advantages, converts the data into a format suitable for Tinker, computes the gradient of the loss, and then updates the parameters.

Each step of a trajectory contains a sequence of $N_s$ tokens representing the state $\boldsymbol{s}_t^e$, and a sequence of $N_a$ tokens representing the action $\boldsymbol{a}_t^e$. The importance sampling loss is given by

$$
\mathcal{L}(\boldsymbol{\theta}) \approx -\frac{1}{N} \sum_{n=1}^{N} \frac{\pi_\theta(\boldsymbol{a}^n|\boldsymbol{s}^n)}{q(\boldsymbol{a}^n|\boldsymbol{s}^n)} A(\boldsymbol{s}^n, \boldsymbol{a}^n)
$$

The token-level loss uses a mask variable $m_k^n$ (which is 0 for state tokens and 1 for action tokens, so the loss only comes from the action tokens):

$$
\mathcal{L}(\boldsymbol{\theta}) = -\frac{1}{N} \sum_{n=1}^{N} \sum_{k=1}^{K} m_k^n \frac{\pi_\theta(x_k^n|\boldsymbol{x}_{1:k-1}^n)}{q(x_k^n|\boldsymbol{x}_{1:k-1}^n)} A(x_k^n)
$$

Note that we average over transitions, not over elements of the action sequence. This avoids a bias towards longer thinking traces. Tinker also supports a clipped version of the above loss (as in PPO), and computes gradients with respect to the LoRA (low-rank adaptation) parameters $\mathbf{A}$ and $\mathbf{B}$ of each layer, where $\mathbf{W}' = \mathbf{W} + \mathbf{AB}^\top$.

</div>

### Computing the Advantages

To compute the advantages, we first compute the return-to-go: $G_t^e = r_t^e + \gamma G_{t+1}^e$. We then use the return-batch normalization (ReBN) method to convert these into advantages:

$$
A_t^e = \frac{G_t^e - \mu}{\sigma}
$$

where $\mu = \text{mean}(\lbrace G_t^e : t = 1:T_e, e = 1:E \rbrace)$ and $\sigma = \text{std}(\lbrace G_t^e : t = 1:T_e, e = 1:E \rbrace)$.

### Computing Token-Level Loss

The transitions are converted into a set of "training datums", each of which contains the input tokens $\boldsymbol{x}^n = (\boldsymbol{s}^n, \boldsymbol{a}^n[:-1])$, the target tokens $\boldsymbol{y}^n = (\boldsymbol{s}^n[1:], \boldsymbol{a}^n)$, the masked sampling log probabilities $\log q(x_k^n)$, and the masked token advantages $A_k^n$, which we set to be equal to the turn-level advantages $R^n$ (derived from the turn-level rewards). The masking ensures the state tokens contribute 0 to the loss.

We can optionally add a KL penalty to the policy to ensure it does not deviate too far from the base model. It is not correct to add this to the advantage itself; instead, we modify the token-level reward to be:

$$
R_k^n = R^n + \beta D_{\text{KL}}\left(\pi_\theta(a_k^n|\boldsymbol{a}_{1:k-1}^n, \boldsymbol{s}^n) \| \pi_0(a_k^n|\boldsymbol{a}_{1:k-1}^n, \boldsymbol{s}^n)\right)
$$

where $\pi_0$ is the reference prior and $R^n$ is the reward from the environment after transition $n$.

### Computing Metrics Related to Training Stability

In practice it is important to log various metrics to monitor the training process. For example, we might want to compute $D_{\text{KL}}(q \| p)$, where $q$ is the sampling distribution (used to rollout the episodes), and $p$ is the training distribution (used to compute gradients). We can estimate the KL using the $k_1$ or $k_2$ estimators. If the KL exceeds 0.01, it means that learning is very off-policy, and results might be unstable. In addition, we can compute the entropy of the policy: we want to ensure this is initially not too small (to enable exploration), but that it does not blow up over time.

# Chapter 7: Other Topics in RL

## Regret Minimization

### Regret for Static MDPs

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Regret in MDPs)</span></p>

In the MDP case, the **regret** of a policy $\pi$ is defined as the difference between its expected return and the expected return of an optimal policy $\pi^*$:

$$
\text{Regret}_T(\pi; M, \Pi) = \mathbb{E}_{s_t \sim M(\cdot|s_{t-1}, a_t), a_t \sim \pi(\cdot|s_t), a_t^* \sim \pi^*(\cdot|s_t)} \left[\sum_{t=1}^{T} (r_t(s_t, a_t^*) - r_t(s_t, a_t))\right]
$$

where $\pi^* = \text{argmax}_{\pi \in \Pi} \mathbb{E}_{s_0 \sim M} [V^\pi(s_0|M)]$ is the **best policy in hindsight**.

Since the true MDP $M$ is usually unknown, we can define the **maximum regret** of a policy as its worst case regret w.r.t. some class of models $\mathcal{M}$:

$$
\text{MaxRegret}_T(\pi; \mathcal{M}, \Pi) = \max_{M \in \mathcal{M}} \text{Regret}_T(\pi|M, \Pi)
$$

The **minimax optimal policy** minimizes the maximum regret:

$$
\pi_{MM}^*(\mathcal{M}, \Pi) = \text{argmin}_{\pi \in \Pi} \max_{M \in \mathcal{M}} \text{Regret}_T(\pi|\pi^*(M), M)
$$

In the case of a tabular episodic MDP, the optimal minimax regret is $O(\sqrt{HSAT})$ (ignoring logarithmic factors), where $H$ is the horizon length, $S$ is the number of states, $A$ is the number of actions, and $T$ is the total number of steps.

</div>

### Regret for Non-Stationary MDPs

When the world can change, there may be no single optimal policy $\pi^*$ we can compare to. Instead, the **dynamic regret** (aka **adaptive regret**) compares to a sequence of optimal policies:

$$
\text{DynamicRegret}_T(\pi_{1:T}|M_{1:T}, \Pi) = \mathbb{E}\left[\sum_{t=1}^{T} (r(s_t, a_t^*) - r(s_t, a_t))\right]
$$

where $\pi_t^* = \text{argmax}_{\pi \in \Pi} V^\pi(s_t|M_t)$ is the optimal policy at that moment in time. To compute bounds on the optimal dynamic regret, we need assumptions about how often and how much the world changes. This is called a **variational budget**:

$$
\text{VB}_T = \sum_{t=2}^{T} \text{dist}(\mathcal{M}_t, \mathcal{M}_{t-1})
$$

### Minimizing Regret vs Maximizing Expected Utility

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Bayesian vs Frequentist RL)</span></p>

The **Bayes optimal** agent maximizes its expected utility (minimizes its risk), where we take expectations not only over the sequence of observations and rewards, but also over the unknown environment $M$ itself, rather than assuming it is known:

$$
\mathcal{A}_{\text{Bayes}}^*(P_0) = \text{argmax}_{\mathcal{A}} U_T(\mathcal{A}|P_0)
$$

$$
U_T(\mathcal{A}|P_0) = \mathbb{E}_{M \sim P_0(\mathcal{M})} \left[\mathbb{E}_{s_t \sim M, a_t \sim \pi_t, \pi_t = \mathcal{A}(s_{1:t-1}, a_{1:t-1}, r_{1:t-1})} \left[\sum_{t=1}^{T} r(s_t, a_t)\right]\right]
$$

Note that the uncertainty over models automatically encourages the optimal amount of exploration. The optimal algorithm is uniquely determined by the prior $P_0$.

By contrast, the regret-minimizing policy is the one that minimizes the maximum regret. Unlike the Bayesian case, we must now manually design the exploration algorithm (e.g., using Thompson sampling or UCB).

| Aspect | Bayes-Optimal (BAMDP) | Regret-Minimizing (Minimax) |
| --- | --- | --- |
| Knowledge | Requires a known prior over MDPs | No prior; judged against best policy in hindsight |
| Objective | Maximize expected return under the prior | Minimize regret w.r.t. optimal policy in the true MDP |
| Exploration | Performs optimal Bayesian exploration | Often uses optimism or randomness (e.g., UCB, TS) |
| Adaptation | Fully adaptive via posterior updates | May use confidence bounds, resets, or pessimism |

</div>

## Exploration-Exploitation Tradeoff

### Optimal (Bayesian) Approach

#### Bandit Case (Gittins Indices)

In the special case of context-free bandits with a finite number of arms, the optimal policy of the belief state MDP can be computed using dynamic programming. For a Bernoulli bandit with $n$ arms, let the belief state be $b = (w_1, l_1, \ldots, w_n, l_n)$, where $w_a$ is the number of times arm $a$ has won and $l_a$ is the number of times arm $a$ has lost. Using Bellman's equation:

$$
V^*(b) = \max_a Q^*(b, a)
$$

$$
Q^*(b, a) = \frac{w_a + 1}{w_a + l_a + 2}(1 + V^*(\cdots, w_a + 1, l_a, \cdots)) + \left(1 - \frac{w_a + 1}{w_a + l_a + 2}\right) V^*(\cdots, w_a, l_a + 1, \cdots)
$$

In the finite horizon case with $h$ steps, we can compute $Q^*$ using dynamic programming. Unfortunately, the number of belief states is $O(h^{2n})$, rendering it intractable. For the infinite horizon discounted case, the problem can be solved efficiently using **Gittins indices**. However, these optimal methods do not extend to contextual bandits.

#### MDP Case (Bayes-Adaptive MDPs)

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(BAMDP)</span></p>

We can extend the Bayesian approach to the MDP case by constructing a **BAMDP** (Bayes-Adaptive MDP). The basic idea is to define an MDP with an augmented state-space, consisting of the original state $s$ and the belief state $b$, representing a distribution over the model parameters. The transition function is given by

$$
\mathcal{T}'(s', b'|s, b, a) = \delta(b' = BU(s, b, a, s')) P(s'|s, b, a)
$$

where $BU$ is the (deterministic) Bayes updating procedure (e.g., incrementing the pseudo counts of the Dirichlet distribution, in the case of a discrete MDP), and the second term is the posterior predictive distribution over states:

$$
P(s'|s, b, a) = \int b(\boldsymbol{\theta}) \mathcal{T}(s'|s, a; \boldsymbol{\theta}) d\boldsymbol{\theta}
$$

Thus Bellman's equation gives us

$$
V^*(s, b) = \max_a \left(R(s, a) + \gamma \sum_{s'} P(s'|s, b, a) V^*(s', BU(s, b, a, s'))\right)
$$

Unfortunately, this is computationally intractable to solve exactly.

</div>

As a more computationally efficient alternative, it is also possible to maintain a posterior over policies or $Q$ functions instead of over world models: **bootstrap DQN** for a simple implementation, **epistemic neural networks** for an implementation based on epistemic neural networks, and **epistemic value estimation** for an implementation based on Laplace approximation. Another approach is to use successor features, where the $Q$ function is assumed to have the form $Q^\pi(s, a) = \boldsymbol{\psi}^\pi(s, a)^\top \boldsymbol{w}$. **Successor Uncertainties** models the uncertainty over $\boldsymbol{w}$ as a Gaussian, $p(\boldsymbol{w}) = \mathcal{N}(\boldsymbol{\mu}_w, \boldsymbol{\Sigma}_w)$.

### Thompson Sampling

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Thompson Sampling)</span></p>

**Thompson sampling** (also called **probability matching**) is a common approximation to the fully Bayesian approach. In the bandit case, we define the policy at step $t$ to be $\pi_t(a|s_t, \boldsymbol{h}_t) = p_a$, where $p_a$ is the probability that $a$ is the optimal action:

$$
p_a = \Pr(a = a_*|s_t, \boldsymbol{h}_t) = \int \mathbb{I}\left(a = \text{argmax}_{a'} R(s_t, a'; \boldsymbol{\theta})\right) p(\boldsymbol{\theta}|\boldsymbol{h}_t) d\boldsymbol{\theta}
$$

To implement this, we can use a single Monte Carlo sample $\tilde{\theta}_t \sim p(\boldsymbol{\theta}|\boldsymbol{h}_t)$. We then plug in this parameter into our reward model, and greedily pick the best action:

$$
a_t = \text{argmax}_{a'} R(s_t, a'; \tilde{\boldsymbol{\theta}}_t)
$$

If the posterior is uncertain, the agent will sample many different actions, automatically resulting in exploration. As the uncertainty decreases, it will start to exploit its knowledge.

</div>

In the (episodic) MDP case, we can generalize Thompson sampling by maintaining a posterior over all the model parameters (reward function and transition model), sampling an MDP from this belief state at the start of each episode, solving for the optimal policy corresponding to the sampled MDP, using the resulting policy to collect new data, and then updating the belief state at the end of the episode. This is called **posterior sampling RL**.

### Upper Confidence Bounds (UCBs)

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Upper Confidence Bound)</span></p>

The optimal solution to explore-exploit is intractable. An intuitively sensible approach is based on the principle known as **"optimism in the face of uncertainty"** (OFU). The principle selects actions greedily, but based on optimistic estimates of their rewards (this approach is optimal in the regret minimization sense, as proved in the **R-Max** paper).

The most common implementation is based on the notion of an **upper confidence bound** or **UCB**. The agent maintains an optimistic reward function estimate $\tilde{R}_t$, so that $\tilde{R}_t(s_t, a) \ge R(s_t, a)$ for all $a$ with high probability, and then chooses the greedy action accordingly:

$$
a_t = \text{argmax}_a \tilde{R}_t(s_t, a)
$$

UCB can be viewed as a form of **exploration bonus**, where the optimistic estimate encourages exploration. Typically, the amount of optimism, $\tilde{R}_t - R$, decreases over time so that the agent gradually reduces exploration.

</div>

#### Bandit Case: Frequentist Approach

A frequentist approach to computing a confidence bound can be based on a **concentration inequality** to derive a high-probability upper bound of the estimation error: $|\hat{R}_t(s, a) - R_t(s, a)| \le \delta_t(s, a)$. For a Bernoulli bandit, the MLE is $\hat{\mu}_t(a) = \frac{N_t^1(a)}{N_t(a)}$, and the **Chernoff-Hoeffding inequality** leads to $\delta_t(a) = c/\sqrt{N_t(a)}$ for some constant $c$, so

$$
\tilde{R}_t(a) = \hat{\mu}_t(a) + \frac{c}{\sqrt{N_t(a)}}
$$

#### Bandit Case: Bayesian Approach

We can also derive an upper confidence about using Bayesian inference. The posterior mean is $\hat{\mu}_t(a) = \mathbb{E}[\mu(a)|\boldsymbol{h}_t]$ and the posterior standard deviation is approximately $\hat{\sigma}_t(a) \approx \sqrt{\hat{\mu}_t(a)(1-\hat{\mu}_t(a))/N_t(a)}$. We then define the optimistic reward estimate as

$$
\tilde{R}_t(a) = \hat{\mu}_t(a) + c\hat{\sigma}_t(a)
$$

for some constant $c$ that controls how greedy the policy is.

#### MDP Case

The UCB idea can be extended to the MDP case by combining UCB with Q learning:

$$
\pi(a|s) = \mathbb{I}\left(a = \text{argmax}_{a'} Q(s, a') + c\sqrt{\log(t)/N_t(s, a')}\right)
$$

The more sophisticated **UCRL2** algorithm computes confidence intervals on all the MDP model parameters at the start of each episode; it then computes the resulting **optimistic MDP** and solves for the optimal policy, which it uses to collect more data.

## Distributional RL

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Distributional RL)</span></p>

The **distributional RL** approach predicts the distribution of (discounted) returns, not just the expected return. Let $Z_t^\pi = \sum_{k=0}^{T-t} \gamma^k R(s_{t+k}, a_{t+k})$ be a random variable representing the (discounted) reward-to-go from step $t$. The standard value function is the expectation of this variable: $V^\pi(s) = \mathbb{E}[Z_0^\pi|s_0 = s]$. In DRL, we instead attempt to learn the full distribution, $p(Z_0^\pi|s_0 = s)$, when training the critic. We then compute the expectation of this distribution when training the actor.

</div>

### Quantile Regression Methods

An alternative to predicting a full distribution is to predict a fixed set of quantiles. This is called quantile regression, and has been used with DQN to get **QR-DQN**, and with SAC to get **QR-SAC**. (The latter was used in Sony's **GTSophy** Gran Turismo AI racing agent.)

### Replacing Regression with Classification

An alternative to quantile regression is to approximate the distribution over returns using a histogram, and then fit it using cross-entropy loss. This approach was first suggested by **categorical DQN**, also called **C51** (using 51 discrete categories or atoms).

An even simpler approach is to replace the distributional target with the standard scalar target (representing the mean), and then discretize this target and use cross-entropy loss instead of squared error. Several encoding schemes have been proposed:

* The **two-hot** transform: a lossless encoding of the target based on putting appropriate weight on the nearest two bins.
* The **HL-Gauss** histogram loss: convolves the target value $y$ with a Gaussian, and then discretizes the resulting continuous distribution. This is more symmetric than two-hot encoding.

Regardless of how the discrete target is chosen, predictions are made using $\hat{y}(s; \boldsymbol{\theta}) = \sum_k p_k(s) b_k$, where $p_k(s)$ is the probability of bin $k$, and $b_k$ is the bin center. The HL-Gauss trick works much better than MSE, two-hot, and C51 across a variety of problems, because cross entropy is more robust to noisy targets and nonstationary targets, and because it reduces overfitting by having a softer (more entropic) target distribution.

Recently, the **Fourier head** was proposed as an alternative, in which the linear output layer is replaced by a Fourier transform before discretizing. This is compatible with standard transformer training and gives improved results when generating continuous outputs.

## Intrinsic Motivation for Reward-Free RL

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Intrinsic Motivation)</span></p>

When the extrinsic reward is sparse, or does not exist at all ("unsupervised RL"), it can be useful to reward the agent for solving "generally useful" tasks, such as learning about the world, or developing a set of skills. This is called **intrinsically motivated RL**. We can classify these methods into two main types:

1. **Knowledge-based intrinsic motivation**, or **artificial curiosity**, where the agent is rewarded for learning about its environment (focused on reducing prediction error).
2. **Competence-based intrinsic motivation**, where the agent is rewarded for achieving novel goals or mastering new skills (focused on control).

</div>

### Knowledge-Based Intrinsic Motivation

#### Exploration Bonuses

One simple approach is to create an intrinsic **exploration bonus** $R_t^i(s_t)$ which is high when the agent visits novel states. For tabular environments, we can just count the number of visits to each state, $N_t(s)$, and define $R_t^i(s) = 1/N_t(s)$ or $R_t^i(s) = 1/\sqrt{N_t(s)}$, which is similar to the UCB heuristic used in bandits. We can extend exploration bonuses to high-dimensional states (e.g., images) using density models. Alternatively, we can use the $\ell_1$ norm of the successor feature representation as an alternative to the visitation count: $R^i(s) = 1/\|\boldsymbol{\psi}^\pi(s)\|_1$. This can be combined with **predecessor** representations, which encode retrospective information about the previous state, encouraging exploration towards bottleneck states.

#### Random Network Distillation (RND)

The **Random Network Distillation** or **RND** method uses a fixed random neural network feature extractor $\boldsymbol{z}_t = f(\boldsymbol{s}_t; \boldsymbol{\theta}^*)$ to define a target, and then trains a predictor $\hat{\boldsymbol{z}}_t = f(\boldsymbol{s}_t; \hat{\boldsymbol{\theta}}_t)$ to predict these targets. If $s_t$ is similar to previously seen states, then the trained model will have low prediction error. We can thus define the intrinsic reward as proportional to $\|\hat{\boldsymbol{z}}_t - \boldsymbol{z}_t\|_2^2$. The **BYOL-Explore** method goes beyond RND by learning the target representation (for the next state), rather than using a fixed random projection, but is still based on prediction error.

#### Information-Theoretic Measures

We can define an intrinsic reward in terms of the information-theoretic **surprise** of the next state given the current one: $R(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{s}') = -\log q(\boldsymbol{s}'|\boldsymbol{s}, \boldsymbol{a})$. Unfortunately such methods can suffer from the **noisy TV problem** (also called a **stochastic trap**), in which an agent is attracted to states which are intrinsically difficult to predict. To help filter out such random noise, the **Intrinsic Curiosity Module** first learns an **inverse dynamics model** of the form $a = f(\boldsymbol{s}, \boldsymbol{s}')$, which tries to predict which action was used. The classifier focuses on parts of the state that the agent can control. Then the agent learns a forwards dynamics model in $\boldsymbol{z}$-space and defines the intrinsic reward as

$$
R(\boldsymbol{s}, \boldsymbol{a}, \boldsymbol{s}') = -\log q(\phi(\boldsymbol{s}')|\phi(\boldsymbol{s}), a)
$$

Another solution is to replace the cross entropy with the KL divergence, $R(\boldsymbol{s}, \boldsymbol{a}) = D_{\text{KL}}(p\|q) = \mathbb{H}_{ce}(p, q) - \mathbb{H}(p)$, which goes to zero once the learned model matches the true model, even for unpredictable states. This encourages exploration towards states with epistemic uncertainty (reducible noise) but not aleatoric uncertainty (irreducible noise).

A related idea is to use the **information gain** as a reward: $R_t(s_t, a_t) = D_{\text{KL}}(q(\boldsymbol{s}_t|\boldsymbol{h}_t, a_t, \theta_t) \| q(\boldsymbol{s}_t|\boldsymbol{h}_t, a_t, \theta_{t-1}))$, where $\theta_t = \text{update}(\theta_{t-1}, h_t, a_t, s_t)$ are the new model parameters. This is closely related to the BALD (Bayesian Active Learning by Disagreement) criterion, and has the advantage of being easier to compute since it does not reference the true distribution $p$.

### Competence-Based Intrinsic Motivation

Another way to explore the environment is to use goal-conditioned RL, where the agent creates its own goals; this is known as an **autotelic agent**. Intuitively it is desirable to choose goals that cover the set of states.

#### Empowerment

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Empowerment)</span></p>

One approach to choosing goals is to use the concept of **empowerment**, which is defined as the mutual information between the goal $G$ (or past action $A$) and the future state $S$:

$$
I(G, S) = H(S) - H(S|G)
$$

Thus we can maximize empowerment by maximizing the entropy of the states (a form of diversity) while minimizing the conditional entropy of the states given the goal (ensuring that the goal is predictable in its effects).

</div>

#### Curriculum Design

Since the space of possible goals is usually too vast to explore, it is important to choose useful goals for the agent to learn from. A good goal is often defined as one that is not too hard or too easy to learn, since this maximizes **learning progress**, also called the **"zone of proximal development"**. Choosing the best order in which to tackle various goals is an example of **automatic curriculum** design; similar methods can also be used to automatically design new environments, a process which is sometimes called **open-ended learning**. It is also possible to train one agent to design an environment that another other agent finds challenging to solve; this is known as **asymmetric self-play**.

#### Using an LLM to Choose Goals

Another approach to goal generation is to use suitably prompted LLMs. This can leverage the prior knowledge of LLMs to not only propose novel goals, but also ones that are plausibly useful to humans.

#### Go-Explore

The **Go-Explore** algorithm proposes to first follow a goal-conditioned policy to reach (or to reset the environment state to) a goal state, which is chosen from an archive of "interestingly new" previously visited states, and then switch to an exploration policy (using random actions) to expand the coverage of the state space. **Intelligent Go-Explore** uses an LLM to decide what is an interesting goal to return to, to decide what exploratory actions to take after reaching the goal, and to decide whether to add any newly explored states to the archive.

## Hierarchical RL

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Hierarchical RL and Options)</span></p>

**Hierarchical RL** or **HRL** extends standard RL to consider actions that operate at multiple levels of **temporal abstraction**. Formally, an **option** $\omega = (I, \pi, \beta)$ is a form of temporally extended action, consisting of:

* The **subpolicy** (aka **intra-option policy**, or **action policy**) $\pi_\omega(a|s) \in [0, 1]$.
* The **termination probability** $\beta_\omega(s) \in [0, 1]$, which gives the probability of finishing in state $s$. This induces a geometric distribution over option durations, $\tau \sim \beta_\omega$.
* The **initiation set** $I_\omega \subset S$, which is the set of states this option can start from. Alternatively, $I_\omega(s) \in [0, 1]$ as the probability that $\omega$ can be started from $s$ and achieve its goal. (The **affordances** of a state, $A(s) = \lbrace \omega : I_\omega(s) > \epsilon \rbrace$, is the set of options that can be initiated from $s$.)

Executing an option at step $t$ entails choosing an action using $a_t = \pi_\omega(s_t)$ and then deciding whether to terminate at step $t+1$ with probability $1 - \beta_\omega(s_{t+1})$ or to continue. This is an example of a **semi-Markov decision process**. If we define $\pi_\omega(s) = a$ and $\beta_\omega(s) = 0$ for all $s$, then this option corresponds to primitive action $a$ that terminates in one step. But with options we can expand the repertoire of actions to include those that take many steps to finish.

</div>

Note that goal-conditioned RL can be considered a special case of options where each option corresponds to a different goal. The reward function for each option has the form $R_\omega(s) = \text{sim}(s, \omega)$, the termination function is $\beta_\omega(s) = \text{sim}(s, \omega) > \text{thresh}$, and the initiation set is the entire state space (a **global option**).

To create a semi-MDP with options, we need to define the reward function and dynamics model. The reward is $R(s, \omega) = \mathbb{E}[R_1 + \gamma R^2 + \cdots + \gamma^{\tau-1}R_\tau | S_0 = s, A_{0:\tau-1} \sim \pi_\omega, \tau \sim \beta_\omega]$, and the dynamics model is $T_\gamma(s'|s, \omega) = \sum_{k=1}^{\infty} \gamma^k \Pr(S_k = s', \beta_\omega(s_k)|S_0 = s, A_{0:k-1} \sim \pi_\omega)$. Note that a dynamics model that can predict multiple steps ahead is sometimes called a **jumpy model**. The value function for a hierarchical policy using a generalized Bellman equation:

$$
V_\pi(s) = \sum_{\omega \in \Omega(s)} \pi(\omega|s) \left[R(s, \omega) + \sum_{s'} T_\gamma(s'|s, \omega) V_\pi(s')\right]
$$

We can compute this using value iteration, policy iteration, or a policy gradient method.

### Option Hierarchies

We can define the HRL problem in terms of a nested set of options. Let $\Omega_l$ be the set of options or subtasks at level $l$ of the hierarchy, where $\Omega_1 = \mathcal{A}$ is the set of primitive actions. Let $\Gamma$ denote the top of the hierarchy, corresponding to the main task. The **hierarchical policy** is $\pi_{\text{hier}} = \pi_1 \odot \cdots \odot \pi_\Gamma$, which maps states to primitive actions by successively invoking policies at descending levels.

$$
\Omega_{\text{hier}}^*, \pi_{\text{hier}}^* = \text{argmax}_{\Omega_{\text{hier}}} \text{argmax}_{\pi_{\text{hier}}|\Omega_{\text{hier}}} Q^{\text{hier}}(s, a)
$$

The problem breaks down into two parts: learning the hierarchical policy $\pi_{\text{hier}}$ given a fixed hierarchy of tasks $\Omega_{\text{hier}}$, and learning the hierarchy itself. The latter is called the **subtask discovery** problem.

### Methods for Learning with Options

**Hierarchical Q learning** learns the option policy using Q-learning by regressing towards targets that incorporate the discounted reward across the option's duration: $y_t = \sum_{t'=t}^{\tau} \gamma^{t'-t} r_{t'} + \gamma^{\tau-t} Q(s_{t+\tau}, o_{t+\tau}^*)$.

In the **MAXQ** approach, the core MDP is decomposed into smaller sub-MDP components. Each sub-MDP is associated to a subtask whose policy can be learned separately. However, the resulting policy will only be recursively optimal, rather than globally optimal.

If the set of options is unknown, we can learn them by segmenting the trajectories into sub-trajectories, which correspond to the latent options. This can be done using the EM algorithm (**option learning using EM**).

**Skill chaining** establishes an initial skill $\omega$ with the primary goal of the overall task as its objective, learns its initiation set $I_\omega$, and then creates a new skill $\omega'$ whose objective is to reach $I_\omega$. This process repeats, creating a chain of interconnected skills. The **DemoStart** system relies on a human demonstration that reaches the goal, and then initializes the agent in a state that is near the end of this trajectory.

The **option-critic** architecture proposes end-to-end training where the options and their policies are jointly learned online using policy gradient methods designed for semi-MDPs. The **double actor critic** (DAC) approach defines two parallel **augmented MDPs**, where the state space of each MDP is the cross-product of the original state space and the set of options. For a two level hierarchy, the manager learns a policy over options, and the worker learns a policy over states for each option.

To avoid excessive option switching, a regularizer called the **deliberation cost** can be added, in which the higher level policy is penalized whenever it switches options. To avoid insufficient option switching (where the higher level policy selects a single option for the entire task duration), the **Interest Option Critic** learns the initiation condition $I_\omega$ so that the option is selected only in certain states of interest.

### HRL Using Feudal Hierarchies

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Feudal RL vs Options)</span></p>

The other main framework for HRL is **feudal RL**. In this approach, the policy at level $l$ (known as a **manager**) chooses a goal from some goal space (equal to the state space, or some abstraction thereof), and passes that down to the level below (known as a **worker**). Thus rather than having a finite number of options to choose from, we can have a nested set of parameterized (universal) policies, $\pi_l(g_{l-1}|s, g_l)$, for each manager level $l$, and $\pi_1(a|s, g_1)$ for the worker level. The value for a policy at a given level is the expected reward until the policy finishes, where the reward is defined intrinsically in terms of reaching the specified goal. Thus only the top level manager gets to see the external (environment) reward, a principle known as **reward hiding**.

| Feature | Options Framework | Feudal RL |
| --- | --- | --- |
| Hierarchical Structure | Flat controller + multiple sub-policies (options) | Manager $\to$ Worker (two-level policy hierarchy) |
| Control Flow | Top-level policy chooses discrete option | Manager emits subgoals; worker acts toward them |
| Termination Handling | Explicit termination function $\beta(s)$ | Often fixed horizon or implicit (e.g., $N$ steps) |
| Goal Communication | Top policy selects an option (index) | Manager gives a vectorial subgoal (e.g., in state space) |
| Training Paradigm | Semi-MDP; option-critic and variants | Two-agent structure; manager learns to guide worker |

Although the feudal approach is somewhat easier to learn (due to the locality/modularity of subgoals), the resulting hierarchical policy may be suboptimal compared to the optimal flat policy, since learning is performed at each level w.r.t. local goals. By contrast, option-based HRL can match the optimality of a flat policy, and can have options with richer termination conditions.

</div>

A major difficulty in HRL when training multiple levels of policies simultaneously is that the resulting data distribution is **non-stationary**. **HIRO** (Hierarchical Reinforcement Learning with Off-policy Correction) tackles this using hindsight relabeling. **HAC** (Hierarchical Actor-Critic) extends this idea further.

In the **hierarchical actor critic** (HAC) approach, the output subgoal in the higher level data, and the input subgoal in the lower-level data, are replaced with the actual state that was achieved in hindsight. This allows the training of each level of the hierarchy independently, by assuming the lower level policies are already optimal (since they achieved the specified goal). As a result, the distribution of $(s, a, s')$ tuples experienced by a higher level will be stable, providing a stationary learning target.

#### Learning the Goal Space and Policy

In the previous approaches, the subgoals are defined in terms of the states that were achieved at the end of each trajectory, $g' = s_T$. This can be generalized by using a state abstraction function to get $g' = \phi(s_T)$.

**Feudal Networks** learn a two level hierarchy. The manager samples subgoals in a learned latent subgoal space. The worker uses distance to this subgoal as a reward, and is trained in the usual way. The manager uses the "transition gradient" as a reward, which is derived from the task reward as well as the distance between the subgoal and the actual state transition made by the worker. This reward signal is used to learn the manager policy and the latent subgoal space.

### Subtask Discovery

In this section, we discuss ways of learning hierarchical structure in an environment (given a dataset), independent of the specific task that needs to be solved. This can be thought of as a "pre-training" phase. The resulting subtasks can then be used to define a hierarchical structure, for which a hierarchical policy can be trained.

#### Discovery of Subgoals

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Subgoal Discovery Methods)</span></p>

One way to define subtasks is in terms of subgoals that might be worth achieving. These are often chosen to be **bottleneck states**, through which many paths (in state space) must pass when going from different starting states to different goal states. These states are often identified by performing various graph-theoretic analyses of the graph $G$ derived from the state transition diagram (e.g., the **diverse density** metric or betweenness centrality).

A notable drawback of existing subgoal discovery methods is their reliance on a discrete subgoal space. **Hierarchical Self Play** (HSP) is a method for learning a continuous embedding of subgoals through an unsupervised pre-training technique called **asymmetric self-play** (a **setter-solver** pair):

1. Starting from an initial state $s_0$, Alice's policy $\pi_A$ acts for $T_A$ steps, arriving at a final state $s^* = s_{T_A}$.
2. The environment is then reset to $s_0$, and $s^*$ is assigned as a target for Bob's policy $\pi_B$.
3. As Bob executes its policy, a learned encoder $E$ generates a low-dimensional subgoal embedding at each timestep $t$: $g_t = E(s_t^B, s^*)$.
4. Bob's policy selects actions based on its current state and this subgoal embedding: $a_t^B = \pi_B(s_t^B, g_t)$.
5. Bob has $T_B$ steps to reach the target state $s^*$. A reward of $R_B = 1$ is given for success and 0 for failure.
6. Alice receives an opposing reward, $R_A = 1 - R_B$.

This setup creates a dynamic where Bob learns to reach goals set by Alice, while Alice is incentivized to discover novel states that are currently challenging for Bob.

</div>

#### Discovery of Skills

Approaches centered on subgoal discovery are inadequate for identifying subtasks that do not have a specific, concrete objective (e.g., "navigating through traffic"). Instead, we focus on methods for learning a varied collection of **skills**, where we define a skill to be a policy for subtask, encapsulating the agent's ability to perform a certain action.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Skill Discovery via Mutual Information)</span></p>

A prominent strategy for discovering diverse skills involves maximizing the Mutual Information (MI) between a given skill $\omega$ and the resulting states or trajectories produced when that skill is executed. This is typically implemented by conditioning a universal policy on a latent skill vector $z_\omega$, creating a skill-specific policy $\pi_\omega(s) = \pi(s, z_\omega)$.

* The **Variational Intrinsic Control** (VIC) method focuses on maximizing the mutual information between the skill vector $z_\omega$ and the terminal state $s_T$, given an initial state $s_0$.
* The **Diversity Is All You Need** (DIAYN) method aims to discover skills by maximizing the mutual information between $z_\omega$ and every state visited within the trajectory generated by $\pi(s, z_\omega)$, without considering the order of the states. (This approach is also called **Mutual Information Skill Learning** or **MISL**.)
* The **VALOR** (Variational Autoencoding Learning of Options by Reinforcement) method uses a VAE-like method to learn skills from trajectories, respecting the order of the states.
* The **SeCTAR** (Self-Consistent Trajectory Autoencoder) method also uses an autoencoder, but based on an LSTM encoder and decoder. They combine this with an exploration mechanism to generate diverse trajectories, so that the continuous latent space represents a diverse set of skills.

</div>

## Imitation Learning

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Imitation Learning)</span></p>

**Imitation learning** (IL), also known as **apprenticeship learning** and **learning from demonstration** (LfD), is a different setting in which the agent does not observe rewards, but has access to a collection $\mathcal{D}_{\text{exp}}$ of trajectories generated by an expert policy $\pi_{\text{exp}}$; that is, $\boldsymbol{\tau} = (s_0, a_0, s_1, a_1, \ldots, s_T)$ and $a_t \sim \pi_{\text{exp}}(s_t)$ for $\boldsymbol{\tau} \in \mathcal{D}_{\text{exp}}$. The goal is to learn a good policy by imitating the expert, in the absence of reward signals. IL finds many applications in scenarios where we have demonstrations of experts (often humans) but designing a good reward function is not easy, such as car driving and conversational systems.

</div>

### Imitation Learning by Behavior Cloning

**Behavior cloning** reduces IL to supervised learning. It interprets a policy as a classifier that maps states (inputs) to actions (labels), and finds a policy by minimizing the imitation error:

$$
\min_\pi \mathbb{E}_{p_{\pi_{\text{exp}}}^\gamma(s)} \left[D_{\text{KL}}(\pi_{\text{exp}}(s) \| \pi(s))\right]
$$

A challenge with this method is that the loss does not consider the sequential nature of IL: future state distribution is not fixed but instead depends on earlier actions. Therefore, if we learn a policy $\hat{\pi}$ that has a low imitation error under distribution $p_{\pi_{\text{exp}}}^\gamma$, it may still incur a large error under distribution $p_{\hat{\pi}}^\gamma$ (when the policy $\hat{\pi}$ is actually run). This problem has been tackled by the offline RL literature.

### Imitation Learning by Inverse Reinforcement Learning

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Inverse RL)</span></p>

An effective approach to IL is **inverse reinforcement learning** (IRL) or **inverse optimal control** (IOC). Here, we first infer a reward function that "explains" the observed expert trajectories, and then compute a (near-)optimal policy against this learned reward using any standard RL algorithm. The key step of reward learning (from expert trajectories) is the opposite of standard RL, thus called inverse RL.

There are infinitely many reward functions for which the expert policy is optimal. To address this challenge, we can follow the maximum entropy principle, and use an energy-based probability model to capture how expert trajectories are generated:

$$
p(\boldsymbol{\tau}) \propto \exp\left(\sum_{t=0}^{T-1} R_\theta(s_t, a_t)\right)
$$

This model assigns exponentially small probabilities to trajectories with lower cumulative rewards. We may infer $\boldsymbol{\theta}$ by maximizing the likelihood $p(\mathcal{D}_{\text{exp}}|\boldsymbol{\theta})$, or equivalently, minimizing the negative log-likelihood loss:

$$
\mathcal{L}(\boldsymbol{\theta}) = -\frac{1}{|\mathcal{D}_{\text{exp}}|} \sum_{\boldsymbol{\tau} \in \mathcal{D}_{\text{exp}}} R_\theta(\boldsymbol{\tau}) + \log \frac{1}{|\mathcal{D}|} \sum_{\boldsymbol{\tau} \in \mathcal{D}} \frac{\exp(R_\theta(\boldsymbol{\tau}))}{q(\boldsymbol{\tau})}
$$

The term inside the log is an importance sampling estimate of the partition function $Z_\theta$. This process produces both the inferred reward $R_\theta$ and an approximate optimal policy $\hat{\pi}$. This approach is used by **guided cost learning** and has been found effective in robotics applications.

</div>

### Imitation Learning by Divergence Minimization

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(GAIL and Adversarial IL)</span></p>

A different approach to IL is based on the observation that if we can find a policy $\pi$ so that $p_\pi^\gamma(s, a)$ and $p_{\pi_{\text{exp}}}^\gamma(s, a)$ are close, then $\pi$ receives similar long-term reward as $\pi_{\text{exp}}$, and is a good imitation. A number of IL algorithms find $\pi$ by minimizing the $f$-divergence $D_f(p_{\pi_{\text{exp}}}^\gamma \| p_\pi^\gamma)$. Using a variational approximation, we can solve the following optimization problem for $\pi$:

$$
\min_\pi \max_{\boldsymbol{w}} \mathbb{E}_{p_{\pi_{\text{exp}}}^\gamma(s,a)} [T_{\boldsymbol{w}}(s, a)] - \mathbb{E}_{p_\pi^\gamma(s,a)} [f^*(T_{\boldsymbol{w}}(s, a))]
$$

where $f^*$ is the convex conjugate of $f$, and $T_{\boldsymbol{w}} : \mathcal{S} \times \mathcal{A} \to \mathbb{R}$ is some function parameterized by $\boldsymbol{w}$. We can think of $\pi$ as a generator (of actions) and $T_{\boldsymbol{w}}$ as an adversarial critic that is used to compare the generated $(s, a)$ pairs to the real ones. With different choices of the convex function $f$, we can obtain many existing IL algorithms, such as **generative adversarial imitation learning** (**GAIL**) and **adversarial inverse RL** (**AIRL**).

</div>

## Offline RL

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Offline RL)</span></p>

**Offline reinforcement learning** (also called **batch reinforcement learning**) is concerned with learning a reward-maximizing policy from a fixed, static dataset, collected by some existing policy, known as the **behavior policy**. Thus no interaction with the environment is allowed. This makes policy learning harder than the online case, since we do not know the consequences of actions that were not taken in a given state, and cannot test any such "counterfactual" predictions by trying them. (This is the same problem as in off-policy RL.) In addition, the policy will be deployed on new states that it may not have seen, requiring that the policy generalize out-of-distribution, which is the main bottleneck for current offline RL methods.

</div>

A very simple and widely used offline RL method is **behavior cloning** (BC), which amounts to training a policy to predict the observed output action $a_t$ associated with each observed state $s_t$, as in supervised learning. This assumes the offline dataset was created by an expert. By contrast, offline RL methods can leverage suboptimal data.

### Offline Model-Free RL

In principle, we can tackle offline RL using the off-policy methods: use some form of importance sampling, based on $\pi(a|s)/\pi_b(a|s)$, to reweight the data in the replay buffer $D$, which was collected by the behavior policy, towards the current policy (the one being evaluated/learned). Unfortunately, such methods only work well if the behavior policy is close to the new policy.

#### Policy Constraint Methods

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Policy Constraint Methods for Offline RL)</span></p>

In the **policy constraint** method, we use a modified form of actor-critic, where the policy improvement step is constrained to stay close to the behavior policy:

$$
\pi_{k+1} \leftarrow \text{argmax}_\pi \mathbb{E}_{s \sim D} \left[\mathbb{E}_{\pi(a|s)} \left[Q_{k+1}^\pi(s, a)\right]\right] \quad \text{s.t.} \; D(\pi(\cdot|s), \pi_b(\cdot|s)) \le \epsilon
$$

Alternatively, we can add a penalty of $\alpha D(\pi(\cdot|s), \pi_b(\cdot|s'))$ to the target $Q$ value and the actor objective. In the case of KL divergence, this can be enforced implicitly, as in the **advantage weighted regression** (AWR), **reward weighted regression**, **advantage weighted actor critic** (AWAC), and **advantage weighted behavior model** (ABM) methods. In this approach, we first solve (nonparametrically) for the new policy:

$$
\bar{\pi}_{k+1}(a|s) \leftarrow \frac{1}{Z} \pi_b(a|s) \exp\left(\frac{1}{\alpha} Q_k^\pi(s, a)\right)
$$

and then project it into the required policy function class: $\pi_{k+1} \leftarrow \text{argmin}_\pi D_{\text{KL}}(\bar{\pi}_{k+1} \| \pi)$.

</div>

#### Behavior-Constrained Policy Gradient Methods

A class of methods first learns a baseline policy $\pi(a|s)$ (using BC) and a $Q$ function (using Bellman minimization) on the offline data, and then updates the policy to pick actions that have high expected value according to $Q$ and which are also likely under the BC prior. An early example is the $Q^f$ algorithm. The **DDPG+BC** method optimizes $\max_\pi J(\pi) = \mathbb{E}_{(s,a) \sim \mathcal{D}} [Q(s, \mu^\pi(s)) + \alpha \log \pi(a|s)]$. The **DQL** method optimizes a diffusion policy using $\min_\pi \mathcal{L}(\pi) = \mathcal{L}_{\text{diffusion}}(\pi) - \alpha \mathbb{E}_{s \sim D, a \sim \pi(\cdot|s)} [Q(s, a)]$.

#### Uncertainty Penalties

An alternative way to avoid picking out-of-distribution actions is to add a penalty term to the $Q$ function based on the estimated epistemic uncertainty, given the dataset $\mathcal{D}$. For example, we can use a deep ensemble to represent the distribution over $Q$ functions, and use the variance of $Q(s, a)$ across ensemble members as a measure of uncertainty:

$$
\pi_{k+1} \leftarrow \text{argmax}_\pi \mathbb{E}_{s \sim D} \left[\mathbb{E}_{\pi(a|s)} \left[\mathbb{E}_{P_D(Q_{k+1}^\pi)} [Q_{k+1}^\pi(s, a)]\right] - \alpha \text{Unc}(P_D(Q_{k+1}^\pi))\right]
$$

#### Conservative Q-Learning

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Conservative Q Learning - CQL)</span></p>

An alternative to explicitly estimating uncertainty is to add a **conservative penalty** directly to the Q-learning error term. In the **conservative Q learning** or **CQL** method, we minimize:

$$
\overline{\mathcal{E}}(\mathcal{B}, \boldsymbol{w}) = \alpha \mathcal{C}(\mathcal{B}, \boldsymbol{w}) + \mathcal{E}(\mathcal{B}, \boldsymbol{w})
$$

where $\mathcal{E}$ is the usual loss for Q-learning, and $\mathcal{C}$ is a conservative penalty:

$$
\mathcal{C}(\mathcal{B}, \boldsymbol{w}) = \mathbb{E}_{s \sim \mathcal{D}} \left[\mathbb{E}_{a \sim \mu(\cdot|s)} [Q_{\boldsymbol{w}}(s, a)] - \mathbb{E}_{a \sim \pi_b(\cdot|s)} [Q_{\boldsymbol{w}}(s, a)]\right] + R(\mu)
$$

where $\mu$ is the new policy derived from $Q$, and $R(\mu) = -D_{\text{KL}}(\mu \| \rho)$ is the action prior. Since we are minimizing $\mathcal{C}$ (in addition to $\mathcal{E}$), we see that we are simultaneously maximizing the Q values for actions that are drawn from the behavior policy while minimizing the Q values for actions sampled from $\mu$. This is to combat the optimism bias of Q-learning (hence the term "conservative").

The optimal solution has the form $\mu(a|s) = \frac{1}{Z}\rho(a|s)\exp(Q(s, a))$. If we set $\rho(a|s)$ to be the previous policy, we can approximate the first term in the penalty using importance sampling. Alternatively, if $\rho(a|s)$ is uniform (as in maxent RL), we should replace the value function with the soft value function: $\mathbb{E}_a[Q_{\text{soft}}(s, a)] = V_{\text{soft}}(s) = \log \sum_a \exp(Q(s, a))$.

</div>

### Offline Model-Based RL

In model-based offline RL, we can train a dynamics model given a fixed dataset, and then use this to generate synthetic data to evaluate and optimize different possible policies. However, if the model is wrong, the method may learn a suboptimal policy. This problem is particularly severe in the offline RL case, since we cannot recover from any errors by collecting more data. Therefore various conservative MBRL algorithms have been developed, to avoid exploiting model errors.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(MOREL and MOPO)</span></p>

* The **MOREL** algorithm modifies the MDP so that the agent enters an absorbing state with a low reward when the model uncertainty $u(s, a)$ is sufficiently large.
* The **MOPO** algorithm defines a conservative reward using $\bar{R}(s, a) = R(s, a) - \lambda u(s, a)$.

In both cases, it is possible to prove that the model-based estimate of the policy's performance under the modified reward or dynamics is a lower bound of the performance of the policy's true performance in the real MDP, provided that the uncertainty function $u$ is an error oracle, which means that it satisfies $D(M_\theta(s'|s, a), M^*(s'|s, a)) \le u(s, a)$, where $M^*$ is the true dynamics and $M_\theta$ is the estimated dynamics.

</div>

### Offline RL Using Reward-Conditioned Sequence Modeling

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Decision Transformers and Diffusers)</span></p>

Recently an approach to offline RL based on sequence modeling has become very popular. The basic idea --- known as **upside down RL** or **RvS** (RL via Supervised learning) --- is to train a generative model over future states and/or actions conditioned on the observed reward, rather than predicting the reward given a state-action trajectory. At test time, the conditioning is changed to represent the desired reward, and futures are sampled from the model.

* The **trajectory transformer** learns a joint model of the form $p(\boldsymbol{s}_{1:T}, \boldsymbol{a}_{1:T}, \boldsymbol{r}_{1:T})$ using a transformer, and then samples from this using beam search, selecting the ones with high reward (similar to MPC).
* The **decision transformer** just generates action sequences, and conditions on the past observations and the future reward-to-go: $\text{argmax}_\theta \mathbb{E}_{p_\mathcal{D}} [\log \pi_\theta(a_t|s_{0:t}, a_{0:t-1}, \text{RTG}_{0:t})]$, where $\text{RTG}_t = \sum_{k=t}^{T} r_k$. At run time, $\text{RTG}_0$ is set to some desired high value. The **Q-learning Decision Transformer** (QDT) conditions on a Q value (learned using Q learning) instead of RTG.
* The **diffuser** method is a diffusion version of trajectory transformer, so it fits $p(\boldsymbol{s}_{1:T}, \boldsymbol{a}_{1:T}, \boldsymbol{r}_{1:T})$ using diffusion, where the action space is assumed to be continuous. The **decision diffuser** extends diffuser by using classifier-free guidance conditioned on the reward-to-go; however, unlike diffuser, the decision diffuser just models the future state trajectories, and infers the actions using an **inverse dynamics model** $a_t = \pi(s_t, s_{t+1})$.
* The **latent plan transformer** replaces conditioning on the reward-to-go with conditioning on a latent "plan", $\boldsymbol{z} \in \mathbb{R}^D$. It fits the following latent variable sequence model using MC-EM: $p(\boldsymbol{z})p(\tau|\boldsymbol{z})p(y|\boldsymbol{z})$, where $\tau$ is the state-action trajectory and $y$ is the observed (trajectory-level) reward. The latent variables provide a way to "stitch together" individual (high performing) trajectories. During decision time, they infer $\hat{\boldsymbol{z}} = \text{argmax} \, p(\boldsymbol{z}|y = y_{\max})$ using gradient ascent, and then autoregressively generate actions.

One problem with all these approaches is that conditioning on a desired return and taking the predicted action can fail dramatically in stochastic environments, since trajectories that result in a return may have only achieved that return due to chance (this is the optimism bias problem in the control-as-inference approach).

</div>

### Offline-to-Online Methods

Despite the progress in offline RL, it is fundamentally more limited in what it can learn compared to online RL, because the agent cannot explore the consequences of its own actions. Therefore, there is a lot of interest in pre-training offline, and then using online finetuning. This is called the **offline-to-online** (O2O) paradigm.

#### Calibrated Q Learning

**Calibrated Q learning** suggests pre-training with CQL followed by online finetuning. Naively this does not work that well, because CQL can be too conservative, requiring the online learning to waste some time at the beginning fixing the pessimism. Calibrated Q learning simply prevents CQL from being too conservative, by replacing the CQL regularizer with a slightly modified expression.

#### Dagger

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(DAgger)</span></p>

The **DAgger** algorithm (Dataset Aggregation) iteratively trains the policy on expert-provided data. We start with an initial dataset $\mathcal{D}$ (e.g., empty) and an initial policy $\pi_1$ (e.g., random). At iteration $t$, we run the current policy $\pi_t$ in the environment to collect states $\lbrace s_i \rbrace$. We then ask an expert policy for the correct actions $a_i^* = \pi^*(s_i)$. We aggregate the data to compute $\mathcal{D} = \mathcal{D} \cup \lbrace (s_i, a_i^*) \rbrace$, and train the new policy $\pi_{t+1}$ on $\mathcal{D}$.

The key idea is to not train passively on expert trajectories as in BC, but to train on the states that the policy actually visits. This avoids overfitting to idealized data and improves robustness (avoids compounding error), since the policy is learning the effects of its own causal interventions.

</div>

## General RL, AIXI and Universal AGI

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(AIXI)</span></p>

The term **"general RL"** refers to the setup in which an agent receives a stream of observations $o_1, o_2, \ldots$ and rewards $r_1, r_2, \ldots$, and performs a sequence of actions $a_1, a_2, \ldots$, but where we do not make any Markovian (or even stationarity) assumptions about the environment. Instead, we assume that the environment is a computable function or program $p^*$, which generated the observations $o_{1:t}$ and $r_{1:t}$ seen so far in response to the actions taken, $a_{1:t-1}$. If we use the receeding horizon control strategy, the optimal action at each step is the one that maximizes the posterior expected reward-to-go (out to some horizon $m$ steps into the future). If we assume the agent represents the unknown environment as a program $p \in \mathcal{M}$, then the optimal action is given by the following **expectimax** formula:

$$
a_t = \text{argmax}_{a_t} \sum_{o_t, r_t} \cdots \max_{a_m} \sum_{o_m, r_m} [r_t + \cdots + r_m] \sum_{p: U(p, \boldsymbol{a}_{1:m}) = (o_1 r_1 \cdots o_m r_m)} \Pr(p)
$$

where $\Pr(p)$ is the prior probability of $p$, and we assume the likelihood is 1 if $p$ can generate the observations given the actions, and is 0 otherwise.

Marcus Hutter proposed to apply the idea of **Solomonoff induction** to the case of an online decision making agent, using the prior $\Pr(p) = 2^{-\ell(p)}$, where $\ell(p)$ is the length of program $p$. This prior favors shorter programs, and the likelihood filters out programs that cannot explain the data. The resulting agent is known as **AIXI**, where "AI" stands for "Artificial Intelligence" and "XI" refers to the Greek letter $\xi$ used in Solomonoff induction. The AIXI agent has been called the "most intelligent general-purpose agent possible", and can be viewed as the theoretical foundation of (universal) **artificial general intelligence** or **AGI**.

Unfortunately, the AIXI agent is intractable to compute, for two main reasons: (1) it relies on Solomonoff induction and Kolmogorov complexity, both of which are intractable; and (2) the expectimax computation is intractable. Various tractable approximations have been devised. For the expectimax computation, we can use MCTS. It is also possible to use meta learning to train a generic sequence predictor (such as a transformer or LSTM) on data generated by random Turing machines, so that the transformer learns to approximate a universal predictor. Another approach is to learn a policy using TD-learning; the weighting term in the policy mixture requires that the agent predict its own future actions, so this approach is known as **self-AIXI**.

**Capacity-Limited Bayesian RL** (CBRL) extends the above Bayesian framework while also taking into account the data budget (due to limited environment interactions) that real agents must contend with (which prohibits modeling the entire environment or finding the optimal action). This approach combines Bayesian inference, RL, and rate distortion theory, and can be seen as a normative theoretical foundation for computationally bounded rational agents.

</div>
