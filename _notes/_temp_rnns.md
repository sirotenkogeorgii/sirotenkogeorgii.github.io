
## Recurrent Neural Networks (RNNs)

### Introducing RNNs

Some historical remarks on recurrent neural networks:
- **Elman networks** were introduced to study language
- **Hopfield networks** for "associative memory"
- **Long-Short-Term-Memory (LSTM)**

#### Architecture of RNN

In a feedforward neural network (FNN), signals flow strictly from input layer through hidden layers to the output layer. An RNN differs by introducing recurrent connections: the output units feed back into the network, allowing the hidden state to depend on both the current input and the previous hidden state.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Recurrent Neural Network)</span></p>

Consider an RNN with $M$ units. The state of unit $i$ at time $t$ is given by

$$x_{i,t} = \phi\!\left(\sum_{j=1}^{M} w_{ij}\, x_{j,t-1} + b_i + \sum_{k=1}^{K} c_{ik}\, s_{k,t}\right), \quad i = 1,\dots,M$$

where:
- $x_{j,t-1}$ are the previous hidden states (recurrent connections with weights $w_{ij}$),
- $s_{k,t}$ are the external inputs (with input weights $c_{ik}$),
- $b_i$ is a bias term,
- $\phi$ is a (elementwise) activation function.

In **matrix notation**, let $\mathbf{x}_t \in \mathbb{R}^{M \times 1}$, $\mathbf{s}_t \in \mathbb{R}^{K \times 1}$, $W \in \mathbb{R}^{M \times M}$, $\mathbf{b} \in \mathbb{R}^{M \times 1}$, and $C \in \mathbb{R}^{M \times K}$. Then:

$$\mathbf{x}_t = \phi\!\left(W\,\mathbf{x}_{t-1} + \mathbf{b} + C\,\mathbf{s}_t\right)$$

</div>

#### Activation Functions

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Activation Functions)</span></p>

**1) Sigmoid:**

$$\phi(u) = \frac{1}{1 + e^{-u}} \in [0, 1]$$

- Maximum slope is at $u = 0$.
- The bias $b_i$ shifts the sigmoid along the input axis.
- Suffers from **saturation effects**: for large $\lvert u \rvert$, the derivative $\phi'(u) \approx 0$, which is problematic for gradient-based methods.

**2) Hyperbolic tangent (tanh):**

$$\phi(u) = \tanh(u) \in [-1, 1]$$

- Similar saturation problems as the sigmoid.

**3) Rectified Linear Units (ReLU):**

$$\phi(u) = \max(u, 0) \in [0, \infty)$$

- No saturation for positive inputs.
- Variant: **Leaky ReLU** allows a small positive gradient for negative inputs.

</div>

#### Illustration of RNN Dynamics

Consider a single-unit RNN with sigmoid activation:

$$x_t = \phi(w\,x_{t-1} + b + s_t) = \frac{1}{1 + e^{-(w\,x_{t-1} + b + s_t)}}$$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(RNN Dynamics and Bifurcations)</span></p>

- When $w$ is **very small**, the map $x_{t+1}$ vs. $x_t$ is nearly flat — there is a single stable fixed point and the system is weakly responsive to inputs.
- When $w$ is **large**, the sigmoid becomes steep and the map can exhibit **bistability**: two stable fixed points separated by an unstable one. An external input $s_t$ can push the state from one attractor to another, and the state persists after the input is removed.
- This transition from one to two stable fixed points is a **pitchfork bifurcation** in the parameter $w$.
- RNNs with standard activation functions are **universal function approximators** for a broad class of dynamical systems.

</div>

#### Data and Tasks

There are two main learning settings:

1. **Supervised learning:** We are given explicit target outputs $\lbrace \tilde{x}_t^{(p)} \rbrace$ for each input sequence (e.g., class labels, desired output sequences).
2. **Unsupervised setting:** We are given sequences $\lbrace x_t \rbrace$ and the learning signal is derived from the data itself (e.g., predicting the next time step).

**Supervised tasks:**
- Given a set of input patterns over time $\lbrace s_T^{(p)} \rbrace_{T \in \mathcal{T}}$, $p = 1,\dots,P$ (patterns), and a set of desired outputs $\lbrace \tilde{x}_t^{(p)} \rbrace_{t \in \mathcal{T}'}$.
- Training data are pairs: $\left(\lbrace s_t^{(p)} \rbrace,\; \lbrace \tilde{x}_t^{(p)} \rbrace\right)$, $p = 1,\dots,P$.
- Examples: speech-to-text, machine translation, topic classification.

**Unsupervised tasks:**
- Data: $\lbrace x_t \rbrace$, $t = 1,\dots,T$. We want to characterize statistical or dynamical structure.
- Examples: time-series forecasting, mechanistic models / data-generating processes.

#### Loss Function

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Loss Function)</span></p>

Let:
- $x_t^{(p)}$ be the RNN output/prediction,
- $\tilde{x}_t^{(p)}$ be the desired target,
- $\theta$ be the collection of all parameters,
- $\ell_t^{(p)}$ be the per-time-step loss.

The **total loss function** is:

$$\mathcal{L}(\theta) := \frac{1}{P} \sum_{p=1}^{P} \sum_{t=1}^{T} \ell_t^{(p)}$$

For **mean squared error (MSE)** loss:

$$\ell_t^{(p)} = \left(\tilde{x}_t^{(p)} - x_t^{(p)}\right)^\top \left(\tilde{x}_t^{(p)} - x_t^{(p)}\right)$$

In the unsupervised case, $\tilde{x}_t^{(p)} = x_t^{\text{data}}$. For classification tasks, one could use **cross-entropy** loss instead.

</div>

### Training: Gradient Descent and Backpropagation Through Time (BPTT)

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Gradient Descent for RNNs)</span></p>

**Goal:** $\min_{\theta} \mathcal{L}(\theta)$

**GD update algorithm:**
1. Initialize $\theta_0$.
2. Repeat until $\lvert \Delta \mathcal{L}(\theta_n) \rvert < \varepsilon$:

$$\theta_n \leftarrow \theta_{n-1} - \gamma\,\nabla_\theta\,\mathcal{L}(\theta_{n-1})$$

$$n \leftarrow n + 1$$

</div>

#### Idea of BPTT

The RNN recurrence is $\mathbf{x}_t = g(W\,\mathbf{x}_{t-1} + \mathbf{b})$ with $W, \mathbf{b} \in \theta$. The key idea is to **unfold the network in time** and treat it as a feedforward network:

$$\mathbf{x}_1 \xrightarrow{\theta} \mathbf{x}_2 \xrightarrow{\theta} \mathbf{x}_3 \to \cdots \to \mathbf{x}_T$$

The unfolded network has **shared weights** across all time steps, and the loss is computed at each time step.

The gradient of the total loss for a single parameter $w_{ij}$ is:

$$\frac{\partial \mathcal{L}}{\partial w_{ij}} = \frac{\partial}{\partial w_{ij}} \sum_{t=1}^{T} \ell_t = \sum_{t=1}^{T} \frac{\partial \ell_t}{\partial w_{ij}}$$

Each $\ell_t$ depends on $\mathbf{x}_t$, which in turn is a nested function $\mathbf{x}_t = g(g(\cdots g(\mathbf{x}_0)\cdots)) = g^t(\cdots)$. By the chain rule:

$$\frac{\partial \ell_t}{\partial w_{ij}} = \frac{\partial \ell_t}{\partial \mathbf{x}_t}\frac{\partial \mathbf{x}_t}{\partial \mathbf{x}_1}\frac{\partial \mathbf{x}_1}{\partial w_{ij}} + \frac{\partial \ell_t}{\partial \mathbf{x}_t}\frac{\partial \mathbf{x}_t}{\partial \mathbf{x}_2}\frac{\partial \mathbf{x}_2}{\partial w_{ij}} + \cdots$$

This can be written compactly as:

$$\frac{\partial \ell_t}{\partial w_{ij}} = \sum_{\tau=1}^{t} \underbrace{\frac{\partial \ell_t}{\partial \mathbf{x}_t^{(p)}}}_{\in \mathbb{R}^{1 \times M}} \underbrace{\frac{\partial \mathbf{x}_t}{\partial \mathbf{x}_\tau}}_{\in \mathbb{R}^{M \times M}} \underbrace{\frac{\partial \mathbf{x}_\tau}{\partial w_{ij}}}_{\in \mathbb{R}^{M \times 1}}$$

where $\frac{\partial \mathbf{x}_t}{\partial \mathbf{x}_\tau}$ is the **Jacobian product**:

$$\frac{\partial \mathbf{x}_t}{\partial \mathbf{x}_\tau} = \frac{\partial \mathbf{x}_t}{\partial \mathbf{x}_{t-1}} \cdot \frac{\partial \mathbf{x}_{t-1}}{\partial \mathbf{x}_{t-2}} \cdots \frac{\partial \mathbf{x}_{\tau+1}}{\partial \mathbf{x}_\tau} = \prod_{u=\tau+1}^{t} \frac{\partial \mathbf{x}_u}{\partial \mathbf{x}_{u-1}}$$

#### Jacobians and the Exploding/Vanishing Gradient Problem

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Jacobians Between Consecutive Time Steps)</span></p>

Given $\mathbf{x}_u = g(W\,\mathbf{x}_{u-1} + \mathbf{b})$, the Jacobian between consecutive steps is:

$$J_u := \frac{\partial \mathbf{x}_u}{\partial \mathbf{x}_{u-1}} = \mathrm{diag}\!\left(g'(W\,\mathbf{x}_{u-1} + \mathbf{b})\right) \cdot W$$

For long sequences, we get products of Jacobians:

$$\frac{\partial \mathbf{x}_t}{\partial \mathbf{x}_\tau} = J_t \cdot J_{t-1} \cdots J_{\tau+1}$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Exploding and Vanishing Gradient Problem — EVGP)</span></p>

The behavior of the Jacobian product $\frac{\partial \mathbf{x}_t}{\partial \mathbf{x}_\tau}$ is governed by the spectral radius of the Jacobians $J_u$:

- If $\max \lvert \mathrm{eig}(J_u) \rvert > 1$ (typical spectral radius $> 1$): gradients **explode** $\Rightarrow$ unstable training.
- If $\max \lvert \mathrm{eig}(J_u) \rvert < 1$ (typical spectral radius $< 1$): gradients **vanish** $\Rightarrow$ forgetting of long-range dependencies.

This is the **exploding and vanishing gradients problem (EVGP)**, a core difficulty in training RNNs.

</div>

The total loss derivative combines all these terms:

$$\frac{\partial \mathcal{L}}{\partial w_{ij}} = \sum_{p=1}^{P} \sum_{t=1}^{T} \frac{\partial \ell_t^{(p)}}{\partial w_{ij}} = \sum_{p=1}^{P} \sum_{t=1}^{T} \sum_{\tau=1}^{t} \frac{\partial \ell_t^{(p)}}{\partial \mathbf{x}_t^{(p)}} \frac{\partial \mathbf{x}_t^{(p)}}{\partial \mathbf{x}_\tau^{(p)}} \frac{\partial \mathbf{x}_\tau^{(p)}}{\partial w_{ij}}$$

where $\frac{\partial \mathbf{x}_t^{(p)}}{\partial \mathbf{x}_\tau^{(p)}}$ is the Jacobian product and $\frac{\partial \mathbf{x}_\tau^{(p)}}{\partial w_{ij}}$ is the local derivative.

#### Remarks on BPTT

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Full BPTT vs. Truncated BPTT)</span></p>

- **Full BPTT** requires storing states for all $t = 1,\dots,T$ (memory-intensive).
- **Truncated BPTT (TBPTT)** backpropagates only through a window of length $K$: backprop through $t - K$ to $t$ time steps, trading accuracy for memory efficiency.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Optimizers)</span></p>

- In practice, we use **mini-batch SGD** (use a subset of patterns to train on).
- **Adaptive optimizers** such as momentum, Adam, etc.

</div>

### Capturing Long-Term Temporal Dependencies (Solutions to EVGP)

**Recap:** In vanilla RNNs, $\mathbf{x}_t = \phi(W\,\mathbf{x}_{t-1} + C\,\mathbf{s}_t + \mathbf{b})$, and the influence of the past is a Jacobian product:

$$\frac{\partial \mathbf{x}_t}{\partial \mathbf{x}_{t-k}} = J_t \cdot J_{t-1} \cdots, \quad J_t = \mathrm{diag}(\phi'(\cdot)) \cdot W$$

**EVGP summary:**
- If $\lVert J_j \rVert < 1$: typically vanishing gradients $\Rightarrow$ forgetting.
- If $\lVert J_j \rVert > 1$: exploding gradients $\Rightarrow$ unstable training.

**Key question:** How can we design an RNN whose dynamics can retain information over long time scales, while keeping gradient flow stable?

**Solutions:**
1. **Architectural:** Build explicit memory paths.
2. **Loss function:** Regularize.
3. **Training:** Control gradients.
4. **Data:** How to show and use data.

#### Architecture

##### (a) Long-Short-Term-Memory (LSTM)

Introduced by Hochreiter & Schmidhuber (1997).

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(LSTM Cell)</span></p>

The LSTM introduces a **cell state** $c_t$ that acts as a memory conveyor belt, controlled by three **gates** (all in $[0, 1]$ via the sigmoid $\sigma$):

**Gates:**

$$i_t = \sigma(W_i\,\mathbf{z}_{t-1} + U_i\,\mathbf{s}_t + b_i) \quad \text{(input gate)}$$

$$f_t = \sigma(W_f\,\mathbf{z}_{t-1} + U_f\,\mathbf{s}_t + b_f) \quad \text{(forget gate)}$$

$$o_t = \sigma(W_o\,\mathbf{z}_{t-1} + U_o\,\mathbf{s}_t + b_o) \quad \text{(output gate)}$$

**Candidate cell state:**

$$\tilde{c}_t = \tanh(U_c\,\mathbf{z}_{t-1} + V_c\,\mathbf{s}_t + b_c)$$

**Cell state update:**

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

**Memory / hidden state:**

$$\mathbf{z}_t = o_t \odot \tanh(c_t)$$

The model can learn to set $f_t$ close to $1$ to circumvent the vanishing gradient problem, since $\frac{\partial c_t}{\partial c_{t-1}} = f_t$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(LSTM)</span></p>

- **Downside:** Very difficult to understand mechanistically.
- **Extension:** Beck, Hochreiter (2024) introduced **xLSTM** ("extended LSTM").

</div>

##### (b) Gated Recurrent Units (GRUs)

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(GRU)</span></p>

The GRU simplifies the LSTM by using only two gates and no explicit memory cell:

**Reset gate:**

$$r_t = \sigma(W_r\,\mathbf{x}_t + U_r\,\mathbf{z}_{t-1} + b_r)$$

**Update gate:**

$$u_t = \sigma(W_u\,\mathbf{x}_t + U_u\,\mathbf{z}_{t-1} + b_u)$$

**Candidate hidden state:**

$$\tilde{\mathbf{z}}_t = \tanh(W_z\,\mathbf{x}_t + U_z\,(r_t \odot \mathbf{z}_{t-1}) + b_z)$$

**Hidden state update:**

$$\mathbf{z}_t = (1 - u_t) \odot \mathbf{z}_{t-1} + u_t \odot \tilde{\mathbf{z}}_t$$

GRUs have 2 gates (vs. 3 for LSTM) and no explicit memory cell.

</div>

##### (c) Role of Activation Functions

Another architectural solution: use **ReLU** instead of sigmoid or tanh to mitigate saturation-induced vanishing gradients.

#### Loss Function / Regularization

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Manifold Attractor Regularization)</span></p>

Consider an RNN of the form:

$$\mathbf{z}_t = A\,\mathbf{z}_{t-1} + W\,\phi(\mathbf{z}_{t-1}) + C\,\mathbf{s}_t + \mathbf{h} + \boldsymbol{\varepsilon}_t, \quad \mathbf{z}_t \in \mathbb{R}^N$$

where $A$ is diagonal with entries $a_{ii}$, $W$ contains nonlinear coupling weights, and $\mathbf{h}$ is a bias vector.

The idea is to **force coefficients to specific values** through the loss function. For a subset of units $i \leq P$, $P \leq N$:
- $a_{ii} \to 1$ (preserve memory along linear channels),
- $w_{ij} \to 0$ (suppress nonlinear feedback),
- $h_i \to 0$ (remove bias).

The regularized loss becomes:

$$\mathcal{L} = \mathcal{L}_{\text{old}} + \mathcal{L}_{\text{reg}}$$

$$\mathcal{L}_{\text{reg}} = \tau_A \sum_{i=1}^{P} (A_{ii} - 1)^2 + \tau_W \sum_{i=1}^{P} \sum_{j=1}^{N} w_{ij}^2 + \tau_h \sum_{i=1}^{P} h_i^2$$

where $\tau_A, \tau_W, \tau_h > 0$ are hyperparameters.

</div>

#### Training / Optimization Solutions

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Gradient Clipping)</span></p>

To prevent exploding gradients, the gradient is clipped:

$$\nabla_\theta \mathcal{L} \leftarrow \begin{cases} \nabla_\theta \mathcal{L} & \text{if } \lVert \nabla_\theta \mathcal{L} \rVert \leq c \\\\ c \cdot \dfrac{\nabla_\theta \mathcal{L}}{\lVert \nabla_\theta \mathcal{L} \rVert} & \text{otherwise} \end{cases}$$

for a threshold $c > 0$ (hyperparameter).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Second-Order Methods)</span></p>

An alternative approach is to use curvature information to rescale gradients (second-order optimization methods).

</div>

#### Data / Training

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Curriculum Learning / Stage-Wise Training)</span></p>

The idea is to train the network from easy to hard — starting with **short sequences** and progressively increasing to **long sequences**.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Generalized Teacher Forcing — GTF)</span></p>

In the unsupervised learning setting, let $\mathbf{z}_t = f_\theta(\mathbf{z}_{t-1}, s_t)$ and define $J_t := \frac{\partial \mathbf{z}_t}{\partial \mathbf{z}_{t-1}}$. In BPTT, $\frac{\partial \mathbf{z}_t}{\partial \mathbf{z}_\tau} = J_t\,J_{t-1}\cdots J_{\tau+1}$.

**GTF state interpolation:** Assume we have a "teacher signal" $\bar{\mathbf{z}}_t$ inferred from the data. Define an interpolated state:

$$\widehat{\mathbf{z}}_t := (1 - \alpha)\,\mathbf{z}_t + \alpha\,\bar{\mathbf{z}}_t, \quad 0 \leq \alpha \leq 1$$

where $\alpha$ is a hyperparameter. The network then evolves as $\mathbf{z}_t = f_\theta(\widehat{\mathbf{z}}_{t-1}, s_t)$.

The Jacobians change to:

$$J_t' = \frac{\partial \mathbf{z}_t}{\partial \mathbf{z}_{t-1}} = \frac{\partial \mathbf{z}_t}{\partial \widehat{\mathbf{z}}_{t-1}} \cdot \frac{\partial \widehat{\mathbf{z}}_{t-1}}{\partial \mathbf{z}_{t-1}} = \partial f_\theta(\widehat{\mathbf{z}}_{t-1}, s_t) \cdot (1 - \alpha)\,\mathbb{I} = (1 - \alpha)\,\tilde{J}_t$$

Thus:

$$\frac{\partial \mathbf{z}_t}{\partial \mathbf{z}_\tau} = (1 - \alpha)^{t-\tau} \prod_{k=0}^{t-\tau-1} \tilde{J}_{t-k}$$

**Sufficient choice of $\alpha$ for bounded gradients:** Let $\tilde{\sigma}_{\max} := \sup_t \lVert \tilde{J}_t \rVert_2$ (largest singular value across trajectories). Then choosing

$$\alpha^* := 1 - \frac{1}{\tilde{\sigma}_{\max}} \quad (\text{if } \tilde{\sigma}_{\max} \geq 1)$$

ensures bounded gradient flow.

</div>
