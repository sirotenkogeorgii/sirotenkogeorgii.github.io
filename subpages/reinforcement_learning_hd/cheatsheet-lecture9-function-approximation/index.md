---
title: "Cheatsheet — Lecture 9: On-Policy Prediction with Function Approximation"
layout: default
noindex: true
math: true
tags:
  - reinforcement-learning
  - function-approximation
  - semi-gradient-td
  - linear-methods
  - cheatsheet
---

# Cheatsheet — Lecture 9: On-Policy Prediction with Function Approximation

*Exam recap: formulas and key results only. Full explanations live in the [main notes]({{ '/subpages/reinforcement_learning_hd/' | relative_url }}#lecture-9-on-policy-prediction-with-function-approximation).*

## Why function approximation

* **Tabular breaks** when $\lvert\mathcal{S}\rvert$ is huge or continuous — can't store one value per state, and there's **no generalization**.
* Approximate $\hat v(s,w)$ with $\dim(w)\ll\lvert\mathcal{S}\rvert$; updating one state now **generalizes** to similar states.
* Forms: **linear** $\hat v(s,w)=w^\top x(s)$ (most theory) or **neural net** $\hat v(s,w)$ — anything **differentiable in $w$**.

## The objective — mean-squared value error

$$\boxed{\,\overline{\text{VE}}(w)=\sum_s \mu(s)\,\bigl[v_\pi(s)-\hat v(s,w)\bigr]^2\,}$$

* $\mu$ = **on-policy state distribution** (how often the policy visits each state) — it decides *which errors matter*.
* Fewer params than states ⇒ can't be $0$ everywhere ⇒ must **trade off** errors; changing the **start distribution** changes $\mu$ and hence the whole $\overline{\text{VE}}$ landscape.

## SGD and the general update

Sampling $S\_t\sim\mu$ and using $-\tfrac12\nabla_w\overline{\text{VE}}(w)\approx[v_\pi(S_t)-\hat v(S_t,w)]\nabla_w\hat v(S_t,w)$, replace the unknown $v_\pi(S_t)$ by a **target** $U\_t$:

$$\boxed{\,w_{t+1}=w_t+\alpha\,\bigl[U_t-\hat v(S_t,w_t)\bigr]\,\nabla_w\hat v(S_t,w_t)\,}$$

* **True gradient** (⇒ converges to a local min of $\overline{\text{VE}}$ under Robbins–Monro $\alpha\_t$) **iff** $U\_t$ is an **unbiased** target *not* depending on $w$.

## Targets: Monte Carlo vs TD

* **Gradient Monte Carlo** — $U\_t=G\_t$ (**unbiased**) ⇒ **true SGD** on $\overline{\text{VE}}$ ⇒ local (linear: **global**) optimum. *Sound but slow, high variance.*
* **Semi-gradient TD(0)** — bootstrapped target $U\_t=R_{t+1}+\gamma\hat v(S_{t+1},w)$ (**biased**):

$$\boxed{\,w_{t+1}=w_t+\alpha\,\bigl[R_{t+1}+\gamma\,\hat v(S_{t+1},w_t)-\hat v(S_t,w_t)\bigr]\,\nabla_w\hat v(S_t,w_t)\,}$$

  *"Semi"* = differentiate the **prediction** only, treat the target as constant (the target *does* depend on $w$, so this is **not** a true gradient). *Fast, low-variance, biased.*

| | **Gradient MC** | **Semi-gradient TD** |
| :-- | :-- | :-- |
| Target $U\_t$ | $G\_t$ (unbiased) | $R_{t+1}+\gamma\hat v(S_{t+1},w)$ (biased) |
| True gradient? | **yes** (SGD on $\overline{\text{VE}}$) | **no** (semi-gradient) |
| Speed / variance | slow, noisy | fast, low-variance |
| Converges (linear, on-policy) | $\overline{\text{VE}}$ optimum | **TD fixed point** |

## Linear function approximation

$\hat v(s,w)=w^\top x(s)\Rightarrow \nabla_w\hat v(s,w)=x(s)$; the objective is **convex** (one optimum). Update:

$$w \leftarrow w + \alpha\bigl[R+\gamma\, w^\top x(S')-w^\top x(S)\bigr]x(S).$$

**Expected TD as a linear system** $\bar g(w)=b-Aw$, with $D=\operatorname{diag}(\mu)$, feature matrix $X$:

$$A=X^\top D(I-\gamma P_\pi)X,\qquad b=X^\top D\,r_\pi.$$

* **TD fixed point:** $\bar g(w)=0\Rightarrow \boxed{Aw=b}\Rightarrow w_{\mathrm{TD}}=A^{-1}b$; equivalently the **projected Bellman equation** $\;Xw=\Pi_D\,T_\pi(Xw)$.
* **Stability:** expected update $w\_{k+1}=w\_k+\alpha(b-Aw\_k)$ converges (for SPD $A$) iff $0<\alpha<\tfrac{2}{\lambda\_{\max}(A)}$.
* **TD is *not* the $\overline{\text{VE}}$ minimizer** (extra $I-\gamma P\_\pi$ tilts the projection), but is bounded:

$$\boxed{\,\overline{\text{VE}}(w_{\mathrm{TD}})\;\le\;\frac{1}{1-\gamma}\,\min_w\overline{\text{VE}}(w)\,}$$

  factor $\tfrac{1}{1-\gamma}$ blows up as $\gamma\to1$ ("bounded" ≠ "tight").

**$n$-step semi-gradient TD:** target $G_{t:t+n}=R_{t+1}+\cdots+\gamma^{n-1}R_{t+n}+\gamma^{n}\hat v(S_{t+n},w)$; $n$ interpolates TD ($n{=}1$) ↔ MC ($n{=}\infty$) — the usual bias–variance dial.

## Feature construction (usually matters more than the algorithm)

* **State aggregation** — group states ⇒ piecewise-constant value (simplest linear method).
* **Polynomial / Fourier cosine basis** — global smooth bases.
* **Coarse coding / RBFs** — overlapping receptive fields; **width** trades *fast early generalization* vs *fine final resolution*.
* **Tile coding** — several offset partitions (**tilings**) ⇒ efficient, structured coarse coding; resolution set by number of tilings.

## Nonlinear (neural networks)

* $\hat v(s,w)$ **learns features** automatically (no hand design) but **loses convexity + convergence theory** ⇒ needs stability tricks. Same semi-gradient machinery, with $\nabla_w\hat v$ from backprop.

## Convergence at a glance (on-policy prediction)

| Method | Tabular | Linear | Nonlinear |
| :-- | :--: | :--: | :--: |
| Gradient MC | ✓ | ✓ (global) | ✓ (local) |
| Semi-gradient TD | ✓ | ✓ (TD fixed pt) | ✗ (can diverge) |

**Punchline:** with function approximation you give up *exact* values for **generalization**; on-policy linear TD stays safe (bounded $\overline{\text{VE}}$, TD fixed point), but leaving that corner (off-policy, nonlinear, bootstrapping) is where stability guarantees end.
