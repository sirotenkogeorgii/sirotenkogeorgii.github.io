---
layout: default
title: "Reinforcement Learning (HD): Problem Sheet 6"
tags:
  - reinforcement-learning
  - machine-learning
  - temporal-difference-learning
  - markov-processes
---


## Exercise 1 (n-step TD Prediction)

...Description goes here...

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Task 1</span><span class="math-callout__name"></span></p>

Show that the n-step return interpolates between one-step TD ($n=1$) and Monte Carlo ($n\geq T−t$). Specifically, write out $G_{t:t+1}$ and $G_{t:T}$ explicitly and identify them as the usual TD target and the full episodic return $G_t$, respectively. Then argue why n-step methods are still TD methods: they update an **earlier** estimate based on a **later** estimate, just $n$ steps later.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

For $n=1$, the definition gives

$$G_{t:t+1}=R_{t+1}+\gamma V_t(S_{t+1}),$$

which is exactly the usual one-step TD target. The corresponding update is

$$V(S_t)\leftarrow V(S_t)+\alpha\bigl(R_{t+1}+\gamma V(S_{t+1})-V(S_t)\bigr).$$

If the backup reaches the end of the episode, i.e. $n\geq T-t$, the bootstrap term disappears because there is no value beyond the terminal state. Thus

$$G_{t:T} = R_{t+1}+\gamma R_{t+2}+\cdots+\gamma^{T-t-1}R_T = G_t,$$

the full Monte Carlo return.

The $n$-step method therefore interpolates between TD and Monte Carlo:

$$
\underbrace{R_{t+1}+\gamma V(S_{t+1})}_{\text{one-step TD}}
\quad \to \quad
\underbrace{R_{t+1}+\cdots+\gamma^{n-1}R_{t+n}
  +\gamma^n V(S_{t+n})}_{n\text{-step TD}}
\quad \to \quad
\underbrace{R_{t+1}+\cdots+\gamma^{T-t-1}R_T}_{\text{Monte Carlo}}.
$$

It is still a TD method whenever the bootstrap term is present: the estimate of $S_t$ is updated using another estimate, namely the later estimate $V(S_{t+n})$. The only difference from TD(0) is that the later estimate is used after $n$ real rewards instead of after one real reward.

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Task 2</span><span class="math-callout__name"></span></p>

Prove the **error-reduction property**: if $V$ is any value estimate, then

$$\max_s \Bigl\lvert \mathbb{E}_\pi [G_{t:t+n} \mid S_t=s] - v_\pi(s) \Bigr\rvert \leq \gamma^n\max_s \Bigl\lvert V(s)−v_\pi(s)\Bigr\rvert.$$

*Hint:* Write $G_{t:t+n}−v_\pi(S_t)$ by adding and subtracting $\gamma^n v_\pi(S_{t+n})$, then bound the bootstrap error. 

What does this imply for the convergence of n-step TD?

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

We show the error-reduction property

$$
\max_s\left|
\mathbb E_\pi[G_{t:t+n}\mid S_t=s]-v_\pi(s)
\right|
\leq
\gamma^n \max_s |V(s)-v_\pi(s)|.
$$

The key point is that the first $n$ rewards together with the true value $v_\pi(S_{t+n})$ would already be an unbiased representation of $v_\pi(S_t)$. Indeed, by the Bellman equation iterated $n$ times,

$$
v_\pi(S_t)
=
\mathbb E_\pi\left[
R_{t+1}+\gamma R_{t+2}+\cdots+\gamma^{n-1}R_{t+n}
+\gamma^n v_\pi(S_{t+n})
\mid S_t
\right].
$$

Subtract this identity from the expected $n$-step return:

$$
\begin{aligned}
&\mathbb E_\pi[G_{t:t+n}\mid S_t=s]-v_\pi(s)\\
&=
\mathbb E_\pi\left[
\gamma^n\bigl(V(S_{t+n})-v_\pi(S_{t+n})\bigr)
\mid S_t=s
\right].
\end{aligned}
$$

Therefore

$$
\begin{aligned}
\left|
\mathbb E_\pi[G_{t:t+n}\mid S_t=s]-v_\pi(s)
\right|
&\leq
\gamma^n
\mathbb E_\pi\left[
|V(S_{t+n})-v_\pi(S_{t+n})|
\mid S_t=s
\right]\\
&\leq
\gamma^n \max_{s'} |V(s')-v_\pi(s')|.
\end{aligned}
$$

Taking the maximum over $s$ gives the claim.

Since $\gamma^n<1$, the expected $n$-step target is closer to $v_\pi$ than the current estimate in sup norm. Thus $n$-step TD is a genuine contraction method in expectation. With the usual stochastic approximation conditions on the step sizes and sufficient state visitation, this is the reason $n$-step TD converges to $v_\pi$.

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Task 3</span><span class="math-callout__name"></span></p>

Compare the bias and variance of the n-step return $G_{t:t+n}$ as a function of $n$.

* (a) Why does the bias **decrease** as $n$ increases (assuming $V \neq v_\pi$)?
* (b) Why does the variance **increase** as $n$ increases?
* (c) The empirical results on the k-state random walk show that an intermediate $n$ minimises RMS error, and that larger $n$ requires a smaller step size $\alpha$. Explain both observations in terms of the bias–variance trade-off.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

Small $n$ uses a short sampled prefix and then relies heavily on the bootstrap $V(S_{t+n})$. If $V\neq v_\pi$, this introduces bias, because the target contains the error $V(S_{t+n})-v_\pi(S_{t+n})$. The bound above shows that the bootstrap bias is damped by $\gamma^n$, so increasing $n$ decreases the bias.

On the other hand, larger $n$ includes more sampled rewards. Random rewards and random transitions accumulate, so the target becomes more sensitive to the particular trajectory. Thus variance increases with $n$. Monte Carlo has the smallest bootstrap bias but usually the largest variance.

The random-walk experiment illustrates exactly this trade-off. For $n=1$, reward information propagates slowly and the method can be biased by poor early estimates. For very large $n$, the target is close to a Monte Carlo return and becomes noisy. An intermediate value, often such as $n=4$, balances these effects. Since larger $n$ gives higher-variance targets, the same step size $\alpha$ produces more unstable updates; therefore larger $n$ usually requires a smaller $\alpha$.

</details>
</div>

## Exercise 2 (Off-policy Learning: Importance Sampling and Control Variates)

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Task 1</span><span class="math-callout__name"></span></p>

Interpret the importance-sampling ratio $\rho_{t:t+n−1}$ in terms of relative likelihoods.
* (a) Show that $\mathbb{E}\_b[\rho_{t:t+n−1}\mid S_t] = 1$.
* (b) What happens to the update when $\pi(A_k\mid S_k) = 0$ for some $k \in \lbrace t,\dots,t+ n−1 \rbrace$? Argue that this is the correct behaviour.
* (c) What happens when $\pi = b$? Verify that the update reduces to the on-policy case.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>


</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Task 2</span><span class="math-callout__name"></span></p>


</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>


</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Task 3</span><span class="math-callout__name"></span></p>

Discuss the practical challenges of importance sampling for large $n$.

* (a) The product $\rho_{t:t+n−1}$ involves $n$ ratios. What happens to its variance as $n$ grows if $\pi$ and $b$ differ substantially? How does this affect the required step size $\alpha$?
* (b) Explain informally why the problem of high-variance importance sampling is particularly severe for off-policy methods but absent for on-policy methods.
* (c) Name one structural advantage of the control-variate approach over the naive importance-weighted return in reducing this variance problem.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>


</details>
</div>
