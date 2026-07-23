---
title: "Cheatsheet — Lecture 7: n-step Bootstrapping"
layout: default
noindex: true
math: true
tags:
  - reinforcement-learning
  - n-step-bootstrapping
  - temporal-difference-learning
  - off-policy
  - cheatsheet
---

# Cheatsheet — Lecture 7: $n$-step Bootstrapping

*Exam recap: formulas and key results only. Full explanations live in the [main notes]({{ '/subpages/reinforcement_learning_hd/' | relative_url }}#lecture-7-n-step-bootstrapping).*

## The idea

* $n$-step methods span the continuum between **one-step TD** ($n=1$) and **Monte Carlo** ($n=\infty$): take $n$ *real* rewards, then **bootstrap**. Still TD methods throughout.
* $n$ **decouples** *how often you act* from *how far you look before bootstrapping*; intermediate $n$ is usually best.

## $n$-step TD (prediction)

**$n$-step return** ($n$ real rewards + a bootstrap; $=G\_t$ once $t+n\ge T$):

$$\boxed{\,G_{t:t+n}=R_{t+1}+\gamma R_{t+2}+\cdots+\gamma^{n-1}R_{t+n}+\gamma^{n}V_{t+n-1}(S_{t+n})\,}$$

**Update** (applied at time $t+n$):

$$V_{t+n}(S_t)\;\leftarrow\;V_{t+n-1}(S_t)+\alpha\bigl[G_{t:t+n}-V_{t+n-1}(S_t)\bigr].$$

**Error-reduction property** (soundness of the whole family):

$$\boxed{\,\max_s\bigl|\mathbb{E}_\pi[G_{t:t+n}\mid S_t=s]-v_\pi(s)\bigr|\;\le\;\gamma^{n}\max_s\bigl|V_{t+n-1}(s)-v_\pi(s)\bigr|\,}$$

$\gamma^{n}<1$ ⇒ the expected $n$-step backup is a **contraction** toward $v\_\pi$ ⇒ $n$-step TD converges.

## Bias–variance: choosing $n$

* $n=1$: maximal bootstrapping — **low variance, biased**. $n=\infty$ (MC): **unbiased, high variance**.
* $n$ is the **bias–variance dial**; on most problems an **intermediate $n$ wins**.

## $n$-step Sarsa (control)

Switch to action values; the last node is a **sampled action**:

$$G_{t:t+n}=R_{t+1}+\cdots+\gamma^{n-1}R_{t+n}+\gamma^{n}Q_{t+n-1}(S_{t+n},A_{t+n}),$$

$$Q_{t+n}(S_t,A_t)\;\leftarrow\;Q_{t+n-1}(S_t,A_t)+\alpha\bigl[G_{t:t+n}-Q_{t+n-1}(S_t,A_t)\bigr].$$

* Propagates credit **back $n$ steps per episode** ⇒ a single trajectory teaches far more than one-step Sarsa.

## $n$-step Expected Sarsa

Last node = **expectation over the target policy** (lower variance than a sampled last step):

$$G_{t:t+n}=R_{t+1}+\cdots+\gamma^{n-1}R_{t+n}+\gamma^{n}\bar V_{t+n-1}(S_{t+n}),\qquad \bar V(s)=\sum_a\pi(a\mid s)\,Q(s,a).$$

* **One-step** Expected Sarsa with a **greedy** target policy $=$ **Q-learning** (the average collapses to the max). (At $n>1$ that equivalence breaks — the honest off-policy version needs importance sampling / Tree Backup.)

## Off-policy $n$-step (importance sampling)

Reweight returns by the **importance-sampling ratio** (target $\pi$ over behaviour $b$):

$$\rho_{t:h}=\prod_{k=t}^{\min(h,\,T-1)}\frac{\pi(A_k\mid S_k)}{b(A_k\mid S_k)}.$$

$$V_{t+n}(S_t)\leftarrow V_{t+n-1}(S_t)+\alpha\,\rho_{t:t+n-1}\bigl[G_{t:t+n}-V_{t+n-1}(S_t)\bigr]\quad\text{(state values)},$$

$$Q_{t+n}(S_t,A_t)\leftarrow Q_{t+n-1}(S_t,A_t)+\alpha\,\rho_{t+1:t+n}\bigl[G_{t:t+n}-Q_{t+n-1}(S_t,A_t)\bigr]\quad\text{(action values)}.$$

* Action-value ratio starts at $t+1$: the pair $(S\_t,A\_t)$ being updated is **given**, so only *later* actions need correcting.
* $\rho=0$ if $\pi$ would never take an action (return **ignored**); $\rho>1$ if $\pi$ prefers it (**up-weighted**); $\rho\equiv1$ on-policy.

## Control variates (tame off-policy variance)

Fold in a **zero-mean correction** so a $\rho=0$ step falls back to the current estimate instead of collapsing to $0$ (unbiased, lower variance):

$$G_{t:h}=\rho_t\bigl(R_{t+1}+\gamma\,G_{t+1:h}\bigr)+(1-\rho_t)\,V_{h-1}(S_t)\quad\text{(state values)},$$

$$G_{t:h}=R_{t+1}+\gamma\Bigl(\bar V_{h-1}(S_{t+1})+\rho_{t+1}\bigl[G_{t+1:h}-Q_{h-1}(S_{t+1},A_{t+1})\bigr]\Bigr)\quad\text{(action values)}.$$

* Because $\mathbb{E}\_b[1-\rho\_t\mid S\_t]=0$, the control variate **does not bias** the update — it only cuts variance (importance-sample *only the correction*, add back the target-policy expectation $\bar V$).

## Map of the $n$-step family

| Method | On/off-policy | Last node |
| :-- | :-- | :-- |
| **$n$-step TD** | prediction | bootstrap $V_{t+n-1}(S_{t+n})$ |
| **$n$-step Sarsa** | on-policy | sampled $Q_{t+n-1}(S_{t+n},A_{t+n})$ |
| **$n$-step Expected Sarsa** | on/off | expectation $\bar V_{t+n-1}(S_{t+n})$ |
| **Off-policy $n$-step** | off-policy | IS-weighted ($+$ control variates) |

**Punchline:** every row is the *same skeleton* — $n$ real rewards then a bootstrap — differing only in **how the last node is handled**; the $\gamma^{n}$ error-reduction property guarantees soundness for all of them.
