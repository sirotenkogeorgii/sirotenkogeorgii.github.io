---
layout: default
title: "Reinforcement Learning (HD): Problem Sheet 5"
tags:
  - reinforcement-learning
  - machine-learning
  - temporal-difference-learning
  - markov-processes
---


## Exercise 1 (Temporal Difference Prediction)

Let $\pi$ be a fixed policy on a finite MDP with discount $\gamma \in [0,1)$ and bounded rewards $\lvert r\rvert \leq R_{\max}$. The tabular TD(0) algorithm updates the value estimate after every transition $(S_t, R_{t+1}, S_{t+1})$ as

$$V(S_t) \leftarrow V(S_t) + \alpha \delta_t, \qquad \delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t),$$

where $\alpha \in (0,1]$ is the step size and $\delta_t$ is the **TD error**.


<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Task 1</span><span class="math-callout__name"></span></p>

Start from the Bellman expectation equation

$$
v_\pi(s)
=
\mathbb{E}_\pi
\left[
R_{t+1} + \gamma v_\pi(S_{t+1})
\,\middle|\,
S_t = s
\right]
$$

and explain the two approximations that turn this identity into the TD(0) target $R_{t+1} + \gamma V(S_{t+1})$. Why is the TD(0) target a **biased** estimator of

$$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

(the true return), yet an \emph{unbiased} estimator of $R_{t+1} + \gamma v_\pi(S_{t+1})$ when conditioned on $S_t$?

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

Start from the Bellman expectation equation:

$$
v_\pi(s) = \mathbb E_\pi!\left[
R_{t+1}+\gamma v_\pi(S_{t+1})
\mid S_t=s
\right].
$$

**The first approximation is sampling the expectation.**

The Bellman equation averages over all possible next rewards and next states. TD(0) does not know or use the transition model; it observes one actual transition $(S_t,R_{t+1},S_{t+1})$ and replaces the conditional expectation by this single sample. So the exact Bellman RHS

$$\mathbb E_\pi[ R_{t+1}+\gamma v_\pi(S_{t+1}) \mid S_t=s]$$

is replaced by the sampled $R_{t+1}+\gamma v_\pi(S_{t+1}).$

**The second approximation is bootstrapping.**

The true value $v_\pi$ is unknown, so TD(0) replaces it by the current estimate $V$:

$$ R_{t+1}+\gamma v_\pi(S_{t+1}) \quad\leadsto\quad R_{t+1}+\gamma V(S_{t+1}).$$

Thus the TD(0) target is $Y_t(V):=R_{t+1}+\gamma V(S_{t+1})$. We will reuse $Y_t(V)$ later.

This is exactly the "prediction from another prediction" idea: the estimate of $V(S_t)$ is updated using the current estimate of $V(S_{t+1})$.

**Bias point.**

The TD target is usually **biased as an estimator of the full return**

$$G_t=\sum_{k=0}^\infty \gamma^k R_{t+k+1},$$

because it replaces the random future return $G_{t+1}$ by the current estimate $V(S_{t+1})$. In general,

$$\mathbb E_\pi[ R_{t+1}+\gamma V(S_{t+1}) \mid S_t=s] = (T_\pi V)(s),$$

which is not necessarily equal to

$$v_\pi(s) = \mathbb E_\pi[G_t\mid S_t=s].$$

It becomes unbiased for the Bellman target if $V=v_\pi$. More precisely,

$$R_{t+1}+\gamma v_\pi(S_{t+1})$$

is an unbiased one-sample estimator of

$$\mathbb E_\pi[ R_{t+1}+\gamma v_\pi(S_{t+1}) \mid S_t=s].$$

So the sampling step is unbiased; the bootstrap substitution $v_\pi\rightsquigarrow V$ is where bias enters.

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Task 2</span><span class="math-callout__name"></span></p>

Show that if $V = v_\pi$ is the true value function, then the TD error $\delta_t$ has zero mean conditioned on $S_t$:

$$\mathbb{E}_\pi[\delta_t \mid S_t = s] = 0 \qquad \text{for all } s.$$

Use this to argue informally why $V = v_\pi$ is a fixed point of the expected TD(0) updates.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

$$\delta_t = R_{t+1}+\gamma v_\pi(S_{t+1})-v_\pi(S_t).$$

Then

$$
\begin{aligned}
\mathbb E_\pi[\delta_t\mid S_t=s]
&= \mathbb E_\pi[
R_{t+1}+\gamma v_\pi(S_{t+1})-v_\pi(S_t) \mid S_t=s] \
&=
\mathbb E_\pi[
R_{t+1}+\gamma v_\pi(S_{t+1}) \mid S_t=s] - v_\pi(s).
\end{aligned}
$$

By the Bellman expectation equation,

$$\mathbb E_\pi[ R_{t+1}+\gamma v_\pi(S_{t+1}) \mid S_t=s] = v_\pi(s).$$

Therefore $\mathbb E_\pi[\delta_t\mid S_t=s]=0$.

**Fixed-point interpretation.**

The TD(0) update is

$$V(S_t)\leftarrow V(S_t)+\alpha\delta_t.$$

If $V=v_\pi$, then the expected update at every state is zero:

$$\mathbb E_\pi[ V_{\text{new}}(s)-V(s) \mid S_t=s] =\alpha\mathbb E_\pi[\delta_t\mid S_t=s] = 0.$$

Thus $v_\pi$ is a fixed point of the expected TD(0) update.

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Task 3</span><span class="math-callout__name"></span></p>

Consider an episode

$$S_0, R_1, S_1, R_2, \ldots, R_T, S_T$$

with $V(S_T) = 0$. Show that the sum of discounted TD errors along the episode telescopes into the MC error:

$$\sum_{t=0}^{T-1} \gamma^t \delta_t = G_0 - V(S_0),$$

where

$$G_0 = \sum_{k=0}^{T-1} \gamma^k R_{k+1}$$

is the full return, assuming $V$ does not change during the episode. Interpret this result: what does it say about the relationship between the TD and MC updates?

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

Assume $V$ is held fixed during the episode and $V(S_T)=0$. For $t=0,\dots,T-1$,

$$\delta_t = R_{t+1}+\gamma V(S_{t+1})-V(S_t).$$

Multiply by $\gamma^t$ and sum:

$$\sum_{t=0}^{T-1}\gamma^t\delta_t = \sum_{t=0}^{T-1}\gamma^t R_{t+1} + \sum_{t=0}^{T-1}\gamma^{t+1}V(S_{t+1}) - \sum_{t=0}^{T-1}\gamma^t V(S_t).$$

The first term is the Monte Carlo return:

$$\sum_{t=0}^{T-1}\gamma^t R_{t+1} = G_0.$$

For the value terms, reindex the middle sum:

$$\sum_{t=0}^{T-1}\gamma^{t+1}V(S_{t+1}) = \sum_{j=1}^{T}\gamma^jV(S_j).$$

Thus

$$
\begin{aligned}
\sum_{t=0}^{T-1}\gamma^t\delta_t
&=
G_0 + \sum_{j=1}^{T}\gamma^jV(S_j) - \sum_{j=0}^{T-1}\gamma^jV(S_j) \
&=
G_0 - V(S_0) + \gamma^T V(S_T).
\end{aligned}
$$

Since $V(S_T)=0$,

$$\sum_{t=0}^{T-1}\gamma^t\delta_t = G_0-V(S_0).$$

**Interpretation.**

The Monte Carlo error $G_0-V(S_0)$ is exactly the discounted sum of all one-step TD errors along the episode. So TD and MC are not unrelated. MC waits until the whole return $G_0$ is known and then uses the full error. TD breaks that same total error into local one-step prediction errors and can update immediately after each transition. This is why the notes emphasize that MC must wait for the complete return, while TD can update after one transition. 

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Task 4</span><span class="math-callout__name"></span></p>

Explain qualitatively why TD(0) typically has **lower variance** but **higher bias** than constant-$\alpha$ MC for value prediction. In what kind of environments---long episodes, continuing tasks, non-stationary dynamics---is each advantage most beneficial?

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

**TD(0) usually has lower variance.**

The MC target is

$$G_t=R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\cdots.$$

It contains randomness from the whole future trajectory: all future rewards, all future transitions, and all future actions. For long episodes or $\gamma\approx 1$, this can have large variance.

The TD target is only

$$R_{t+1}+\gamma V(S_{t+1}).$$

It samples only one reward and one next state, then replaces the random future tail by the current estimate $V(S_{t+1})$. This removes much of the future randomness. Hence lower variance.

**TD(0) usually has higher bias.**

The cost is that $V(S_{t+1})$ may be wrong. TD updates using its own current estimate as part of the target. Therefore early in learning,

$$R_{t+1}+\gamma V(S_{t+1})$$

may systematically point in the wrong direction. This is the bootstrapping bias.

MC does not bootstrap. It uses the actually realized return $G_t$. For a fixed policy in an episodic stationary environment,

$$\mathbb E_\pi[G_t\mid S_t=s]=v_\pi(s),$$

so MC has an unbiased target for $v_\pi(s)$, but often a noisy one.

**Where TD is especially useful.**

TD is most beneficial in:

$$
\text{long episodes},\qquad
\text{continuing tasks},\qquad
\gamma\approx 1,
\qquad
\text{online learning}.
$$

In these cases MC either has to wait too long or suffers high variance. TD can update after each transition.

TD is also useful in non-stationary environments when combined with a constant step size $\alpha$, because it can keep tracking changing values online. The notes make the same point for constant step-size updates: the goal in non-stationary problems is tracking rather than convergence to a fixed number. 

**Where MC is especially useful.**

MC is most attractive in:

$$
\text{short episodic tasks},\qquad
\text{when full returns are cheap to observe},\qquad
\text{when avoiding bootstrap bias matters}.
$$

It is conceptually simpler: wait until the episode ends, compute $G_t$, and update toward it.

</details>
</div>