## Lecture 10

This lecture explores the intersection of dynamical systems theory and machine learning, specifically focusing on the reconstruction of underlying models from observed time series data. We address the challenges of training recurrent architectures, examine a specific solution in the Piecewise Linear Recurrent Neural Network (PLRNN), develop formal frameworks for assessing reconstruction quality, and study advanced training techniques such as sparse teacher forcing and generalized teacher forcing.

### Foundations of Dynamical Systems Reconstruction (DSR)

Dynamical Systems Reconstruction (DSR) is the process of uncovering the hidden mathematical model that generates an observed sequence of data points over time.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Dynamical Systems Reconstruction)</span></p>

The objective of **DSR** is to estimate the underlying dynamics of a system from an observed time series $X = \lbrace x_1, x_2, \dots, x_T \rbrace$. We assume the existence of a latent state $z_t$ governed by a flow operator and an observation function.

* **Data Generating System:** An unknown form that generates the data.
* **Observation Function ($g$):** A function that maps the latent state $z_t$ to the observed value $x_t$.
* **Flow Operator ($f$):** A parameterized function $f_\lambda$ used to approximate the underlying dynamics (the "flow") of the system.

</div>

<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/properties_dsr.png' | relative_url }}" alt="Newton–Raphson iteration animation" loading="lazy">
</figure>

#### Mathematical Framework

We describe the system using a recursive formulation, typically implemented as a Recurrent Neural Network (RNN):

$$z_t = f_\lambda(z_{t-1}, s_t), \qquad x_t = g_\lambda(z_t)$$

where:

* $z_t \in \mathbb{R}^M$ represents the latent states.
* $s_t$ represents potential external inputs.
* $x_t$ are the observations.
* $f_\lambda$ is the flow operator parameterized by $\lambda$.
* $g_\lambda$ is the observation function (or decoder).

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Loss Function)</span></p>

To estimate the parameters $\lambda$, we minimize a loss function, typically the Mean Squared Error (MSE) between the observed values $x_t$ and the predicted values $\hat{x}_t$:

$$\mathcal{L}_{\text{MSE}} = \sum_{t=1}^{T} \| x_t - \hat{x}_t \|^2 = \sum_{t=1}^{T} \| x_t - g_\lambda(z_t) \|^2$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Gradient Problem)</span></p>

When training these systems using gradient descent, we encounter the **Exploding/Vanishing Gradient Problem**. Due to the chain rule, the gradient of the loss with respect to parameters involves a product of Jacobians:

$$\frac{\partial z_t}{\partial z_{t-n}} = \prod_{i=t-n+1}^{t} \frac{\partial z_i}{\partial z_{i-1}}$$

If the eigenvalues of these Jacobians are significantly larger or smaller than one, the gradients will either explode or vanish as the time horizon increases, making it impossible for the network to learn long-term dependencies.

</div>

### Piecewise Linear Recurrent Neural Networks (PLRNN)

The PLRNN is a class of recurrent networks designed to mitigate gradient issues while remaining capable of approximating complex flows.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(PLRNN Architecture)</span></p>

The latent state transition in a **PLRNN** is defined by the following map:

$$z_t = A z_{t-1} + W \phi(z_{t-1}) + h + s_t$$

where:
* $A$ is a (usually diagonal) weight matrix.
* $W$ is the recurrence weight matrix.
* $h$ is a bias vector.
* $s_t$ are external inputs.
* $\phi(z)$ is the Rectified Linear Unit (ReLU) nonlinearity: $\phi(z) = \max(0, z)$.

</div>

<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/PLRNN.png' | relative_url }}" alt="Newton–Raphson iteration animation" loading="lazy">
</figure>

<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/IllustrativePLRNNdynamicsIn2D.png' | relative_url }}" alt="Newton–Raphson iteration animation" loading="lazy">
  <figcaption>Hidden-state dynamics are linear within each region of state space, but switch between regions when units cross thresholds, usually because the nonlinearity is a ReLU or thresholded ReLU.</figcaption>
</figure>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Versatility of the Form)</span></p>

While the formulation places the nonlinearity outside the weight multiplication for the $W$ term, any standard RNN where the affine mapping is inside the nonlinearity can be rewritten into this piecewise linear form through a substitution of variables:

$$\phi(Wx + h)$$

could be rewritten as

$$W\phi(x) + h$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proof</span><span class="math-callout__name">(Jacobian of the PLRNN)</span></p>

To understand how PLRNNs handle gradients, we examine the derivative of the transition function $f$ with respect to the latent state $z$:

1. Given $z_t = A z_{t-1} + W \phi(z_{t-1}) + h$.
2. The Jacobian $\frac{\partial z_t}{\partial z_{t-1}}$ is calculated as:

   $$\frac{\partial z_t}{\partial z_{t-1}} = A + W \operatorname{diag}(\phi’(z_{t-1}))$$

3. Since $\phi$ is the ReLU function, its derivative $\phi’(z)$ is:

   $$\phi’(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z < 0 \end{cases}$$

4. Therefore, the Jacobian is a piecewise constant matrix that depends on whether the components of the previous state were positive or negative.

</div>

### Solving Gradient Problems via Subspace Regularization

The core idea for solving the exploding/vanishing gradient problem in PLRNNs is to force a portion of the network to behave like an identity mapping, effectively "transporting" gradients through time without decay or explosion.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Identity Solution)</span></p>

If we set $A = I$ (the identity matrix), $W = 0$, and $h = 0$, then $z_t = z_{t-1}$. In this state, the Jacobian is exactly the identity matrix, and the gradient problem is solved. However, such a network performs no computation. The solution is to apply this "identity" behavior only to a subspace of the latent variables.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Subspace Splitting)</span></p>

We split the latent state vector $z_t \in \mathbb{R}^M$ into two sets of units:

1. **Regularized Units** ($M_{\text{reg}}$): Pushed toward identity dynamics to store memory.
2. **Free Units:** Allowed to vary freely to capture the complex nonlinearities of the data.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Regularization Loss)</span></p>

To induce this behavior, we add a regularization term $\mathcal{L}_{\text{reg}}$ to the MSE loss, controlled by a hyperparameter $\lambda$:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE}} + \lambda \, \mathcal{L}_{\text{reg}}$$

The regularization term targets the coefficients of the regularized subspace:

$$\mathcal{L}_{\text{reg}} = \sum_{j=1}^{M_{\text{reg}}} (A_{jj} - 1)^2 + \sum_{j=1}^{M_{\text{reg}}} \sum_{i=1}^{M} (W_{ji})^2 + \sum_{j=1}^{M_{\text{reg}}} (h_j)^2$$

This forces the regularized units to approximate $z_{t,j} \approx z_{t-1,j}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Intuition</span><span class="math-callout__name">(Manifold Attractors and Time Constants)</span></p>

This regularization has two profound effects on the system’s dynamics:

* **Line/Manifold Attractors:** By setting $A \approx I$, we create a continuous sheet of fixed points. The system can store a value indefinitely along this manifold, similar to how LSTMs function.
* **Slow Flow/Time Constants:** If $A_{jj}$ is slightly less than 1 (e.g., 0.999), the system "forgets" the state very slowly. By adjusting these values, the network can learn different time constants suitable for the specific temporal scales of the input data.

</div>

### Benchmarks in Long-Term Dependency

To evaluate the effectiveness of the PLRNN and its regularization in solving the gradient problem, researchers use specific machine learning benchmarks.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(The Addition Problem)</span></p>

The Addition Problem is a classical benchmark used to test a network’s ability to maintain information over long time delays.

* **Setup:** The network receives two sequences as input:
  1. A sequence of random numbers in the interval $[0, 1]$.
  2. An "indicator bit" sequence (mostly 0s, with 1s marking the numbers to be added).
* **Task:** The network must remember the numbers marked by the indicator bits — which may be separated by hundreds or thousands of time steps — and output their sum at the end of the sequence.
* **Significance:** Success in this task demonstrates that the architecture has successfully overcome the vanishing gradient problem and can maintain a stable "memory" across the time axis.

</div>

### Dynamical Systems Reconstruction: Theory and Assessment

In the study of Recurrent Neural Networks (RNNs) and related models, we often move beyond standard machine learning benchmarks to the more rigorous task of Dynamical Systems Reconstruction. This task involves recreating an underlying system based solely on observed time series data.

#### Objectives of System Recreation

Unlike typical machine learning tasks where the goal might be minimizing a specific error metric on a test set, the primary interest here is in the recreation of the governing mechanism. We assume we have observed a system through a set of time series observations and wish to establish a model that captures the true underlying dynamics.

#### Properties of a Valid Reconstruction

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Valid Reconstruction)</span></p>

To consider a model a successful reconstruction of a dynamical system, it must exhibit specific behaviors that demonstrate it has captured more than just a sequence of numbers:

* **Attractor Extent Reconstruction:** From a finite and potentially short trajectory, the model must be able to reconstruct the full extent of the underlying system’s attractor (e.g., the chaotic Lorenz attractor).
* **Generalization to Initial Conditions:** The model should generalize to nearby initial conditions that were not explicitly present in the training trajectory.
* **Handling Disparate Time Scales:** Real-world systems often contain dynamics operating on vastly different scales. A robust model must capture both fast oscillations and slow underlying drifts.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Bursting Neurons and ECG Data)</span></p>

In models of bursting neurons, the system exhibits very fast spikes separated by long quiescent periods. Similarly, human Electrocardiogram (ECG) data displays disparate time scales. In specialized architectures like the regularized PLRNN, it has been observed that regularized units tend to learn the slow dynamics, while non-regularized units capture the fast dynamics.

</div>

#### Formal Framework: Topological Conjugacy

Establishing whether a model has successfully reproduced a dynamical system requires a formal mathematical basis. The most rigorous standard for reconstruction is **Topological Conjugacy**. If a model’s flow is topologically conjugate to the original system’s flow, they are qualitatively identical in terms of their dynamics.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Dynamical Systems Reconstruction via Topological Conjugacy)</span></p>

Let $R \subseteq \mathbb{R}^m$ and $R^\ast \subseteq \mathbb{R}^{m^\ast}$ be open sets representing subspaces of Euclidean space. Define two dynamical systems:

1. $D = (T, R, \phi)$, where $\phi$ is the flow operator of the original system.
2. $D^\ast = (T, R^\ast, \phi^\ast)$, where $\phi^\ast$ is the flow operator of the reconstructed system.

Let $A$ be an attractor of $\phi$ surrounded by a basin of attraction $B \subseteq R$.

We call the system $D^\ast$ a **dynamical systems reconstruction** of $D$ on the domain $B$ if $\phi^\ast$ is topologically conjugate to $\phi$ on $B$. This implies there exists a homeomorphism $G: B \to V^\ast$ (where $V^\ast \subseteq R^\ast$) such that for every initial condition $x_0 \in B$:

$$\phi^*(t, G(x_0)) = G(\phi(t, x_0))$$

This conjugacy ensures that the trajectory $x^\ast(t)$ produced by $\phi^\ast$ is topologically equivalent to the trajectory $x(t)$ produced by $\phi$, preserving the parameterization by time.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Intuition</span><span class="math-callout__name">(Topological Conjugacy)</span></p>

Topological conjugacy is a high bar. It essentially means that there is a continuous, invertible mapping (a "stretching" or "bending" of the space) that can perfectly align the trajectories of the model with the trajectories of the real system.

</div>

#### Quantitative Measures of Similarity

While topological conjugacy provides a theoretical ideal, empirical evaluation requires quantitative measures to assess how closely the geometry of the reconstructed attractor matches the original.

In a machine learning context, statistical performance is often tracked using:

* **Mean Squared Error (MSE):** Often plotted on a logarithmic axis to identify the point where the model’s performance "breaks down."
* **Sequence Length Breakdown:** Determining the maximum sequence length at which the model can maintain accurate predictions.

To assess the similarity between the geometries of two attractors (the "true" observed attractor and the "generated" model attractor), we treat them as probability distributions over the state space.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Kullback-Leibler Divergence, State Space)</span></p>

The **KL divergence** assesses the overlap between the true probability distribution of observations $P_{\text{true}}(x)$ and the generated probability distribution $P_{\text{gen}}(x)$ produced by the model (often conditioned on latent states). It is defined as:

$$D_{\text{KL}}(P_{\text{true}} \| P_{\text{gen}}) = \int_D P_{\text{true}}(x) \log \left( \frac{P_{\text{true}}(x)}{P_{\text{gen}}(x \mid \text{latent})} \right) dx$$

where the integral is taken over the domain $D$ of the state space.

* **Interpretation:** If the distributions precisely overlap, $D_{\text{KL}} = 0$. As the disagreement between distributions increases, $D_{\text{KL}}$ becomes larger than zero.
* **Divergence:** If there are regions where one distribution has values and the other is zero, the measure may diverge to infinity.

</div>

#### Practical Implementation and Estimation

Evaluating these measures on empirical data introduces several challenges, particularly regarding the dimensionality of the state space.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(State Space KL Estimator)</span></p>

The simplest way to estimate the KL divergence integral is through discretization, similar to the box-counting method used for calculating dimensions. By partitioning the state space into $K$ discrete bins, we can approximate the probability distributions based on the relative frequencies of data points falling into each bin:

$$\hat{D}_{\text{KL}} \approx \sum_{i=1}^{K} \hat{P}_{\text{true}}(x_i) \log \left( \frac{\hat{P}_{\text{true}}(x_i)}{\hat{P}_{\text{gen}}(x_i)} \right)$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Empirical Data Preparation)</span></p>

Before performing these comparisons on empirical data, one must ensure the data is represented in the correct state space. This often requires performing an optimal delay embedding first to reconstruct the state space from a single time series.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(High-Dimensional Challenges and Gaussian Mixture Models)</span></p>

The binning approach suffers from the "curse of dimensionality"; as the number of dimensions increases, the number of bins required grows exponentially, making it difficult to get reliable frequency estimates.

* **Simulated vs. Empirical Data:** For simulated data, we have access to the ground truth governing equations, allowing for precise evaluation. For empirical data, we must rely on estimators.
* **Gaussian Mixture Models (GMM):** Instead of rigid bins, one can define $\epsilon$-neighborhoods along a trajectory and model these neighborhoods using Gaussians. A Gaussian Mixture Model can then act as a smoother estimate for the probability distributions, which is often more robust in higher dimensions and avoids issues with zero-valued bins.

</div>

### Evaluation Measures for Dynamical Reconstructions

When reconstructing a dynamical system — particularly chaotic ones — traditional statistical metrics often fail to capture the qualitative and quantitative essence of the underlying dynamics. We require measures that assess the overlap in geometry, temporal structure, and complexity.

#### The Failure of Standard Metrics

In classical machine learning, the Mean Squared Error (MSE) is the standard for time series prediction. However, in the context of dynamical systems, MSE is often an inadequate and even misleading statistic.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Inadequacy of MSE for Chaotic Systems)</span></p>

Consider two trajectories generated by the Lorenz 63 system using the exact same parameters but infinitesimally different initial conditions. Because the system is chaotic, these trajectories will eventually diverge. Once divergence occurs, the point-to-point MSE becomes extremely high, even though both trajectories belong to the same attractor and represent the same system.

</div>

<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/InadequacyOfMSEforChaoticSystems.png' | relative_url }}" alt="Newton–Raphson iteration animation" loading="lazy">
</figure>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(The MSE Illusion)</span></p>

In an experiment comparing two Recurrent Neural Network (RNN) reconstructions of the Lorenz system:

* **Model A:** Captures one major oscillation period correctly but fails to represent the attractor geometry. It may yield a relatively low MSE initially.
* **Model B:** Perfectly captures the Lorenz attractor’s geometry and chaotic nature. Because it is chaotic, its trajectory diverges from the ground truth quickly, resulting in a higher MSE than Model A.

Despite the higher MSE, Model B is the superior reconstruction because it correctly identifies the system’s underlying dynamics, whereas Model A provides a "wrong illusion" of quality.

</div>

#### Geometric and Probabilistic Measures

To assess whether a model has captured the "shape" of the system in state space, we look toward probabilistic descriptions and invariant measures.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Invariant Measures)</span></p>

In dynamical systems theory, chaotic attractors are often described in terms of **invariant measures**, which characterize the distribution of states over the attractor as $t \to \infty$. This is particularly useful for real-world data, which is inherently noisy.

Measures of geometric overlap include:

* **Wasserstein Distance:** A measure used to assess the distance between two probability distributions, providing a metric for how well the reconstructed geometry overlaps with the original.
* **Kullback-Leibler (KL) Divergence:** Used to quantify the difference between the state-space distributions of the real and reconstructed systems.

</div>

#### Temporal Domain Measures

Since precise point-by-point temporal agreement is impossible for chaotic systems, we assess the general temporal structure using frequency-domain analysis.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Hellinger Distance)</span></p>

The **Hellinger distance** is used to quantify the similarity between the power spectra of the original signal and the reconstructed signal. Assuming the power spectra are normalized such that the area under the curve is one:

$$\int P(\omega)\, d\omega = 1$$

The Hellinger distance $H$ is defined as:

$$H = \sqrt{1 - \int_{-\infty}^{+\infty} \sqrt{S_{\text{real}}(\omega)\, S_{\text{recon}}(\omega)}\, d\omega}$$

where:

* $S_{\text{real}}(\omega)$ is the power spectrum of the real signal.
* $S_{\text{recon}}(\omega)$ is the power spectrum of the reconstructed signal.
* $\omega$ represents frequency.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Hellinger Distance Interpretation)</span></p>

The Hellinger distance lives within the range $[0, 1]$. When the correlation between power spectra is very high, the integral term approaches $1$, and the distance becomes $0$. If there is no overlap, the distance approaches $1$. This makes it a normalized, robust measure of temporal similarity.

</div>

<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/An-illustration-of-the-behavior-of-the-Squared-Hellinger-distance-Euclidean-distance.ppm.png' | relative_url }}" alt="Newton–Raphson iteration animation" loading="lazy">
</figure>

#### Complexity and Chaos Measures

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Good reconstruction should preserve the complexity of the original system)</span></p>

A successful reconstruction should preserve the "complexity" of the original system, which can be quantified using the measures previously discussed in the context of dynamical systems analysis.

* **Lyapunov Exponents:** One can estimate the Maximum Lyapunov Exponent (MLE) from both the real data (via delay embedding) and the RNN. For an RNN where the equations are known, the full Lyapunov Spectrum can be computed directly from the Jacobians.
* **Fractal Dimensionality:**
  * **Box-counting Dimension:** Suitable for lower-dimensional systems.
  * **Correlation Dimension:** Used for higher-dimensional systems to describe behavior within an $\epsilon$-ball around data points along trajectories.
  * **Kaplan-Yorke Dimension:** An estimator for fractal dimensionality derived from the Lyapunov spectrum.

</div>

### The Gradient-Lyapunov Connection in RNNs

The training of RNNs is fraught with the "exploding and vanishing gradient" problem. In the context of dynamical systems, this is not merely a numerical issue but is deeply connected to the system’s stability and its Lyapunov spectrum.

#### Formal RNN Framework

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Generic Recurrent Neural Network)</span></p>

We define a **generic RNN** as a recursive map $f$ with parameters $\theta$:

$$z_t = f_\theta(z_{t-1}, u_t)$$

where:

* $z_t \in \mathbb{R}^m$ represents the state at time $t$.
* $u_t$ represents potential external inputs.
* $f$ can represent any recursive structure, such as LSTMs or Piecewise Linear RNNs (PLRNNs).

</div>

#### The Lyapunov Spectrum

The Lyapunov spectrum characterizes the rates of exponential divergence or convergence of nearby trajectories in state space.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Maximum Lyapunov Exponent Along a Trajectory)</span></p>

The **Maximum Lyapunov Exponent** is derived from the geometric mean of the product of Jacobians **along a trajectory**. In the limit $T \to \infty$:

$$\lambda_{\max} = \lim_{T \to \infty} \frac{1}{T} \ln \left\| \prod_{r=0}^{T-2} J(z_r) \right\|$$

where $J(z_r)$ is the Jacobian of the system at state $z_r$, defined as the derivative of the map with respect to the state:

$$J(z_r) = \frac{\partial z_{r+1}}{\partial z_r}$$

</div>

#### Loss Gradients and Jacobian Products

The difficulty in training RNNs arises because the mathematical structure of the loss gradient mirrors the mathematical structure of the Lyapunov exponent.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proof</span><span class="math-callout__name">(Derivation of the Loss Gradient)</span></p>

Let the total loss $L$ be the sum of losses at each time step: $L = \sum_{t=1}^T L_t$. We wish to find the gradient of the loss with respect to a parameter $\theta$.

1. Using the chain rule, the derivative for a single time step $t$ is:

   $$\frac{\partial L_t}{\partial \theta} = \sum_{r=1}^t \frac{\partial L_t}{\partial z_t} \frac{\partial z_t}{\partial z_r} \frac{\partial z_r}{\partial \theta}$$

2. The middle term, $\frac{\partial z_t}{\partial z_r}$, represents the dependency of the state at time $t$ on the state at an earlier time $r$. This can be expanded via the chain rule as a product series of Jacobians:

   $$\frac{\partial z_t}{\partial z_r} = \prod_{k=0}^{t-r-1} J(z_{t-k-1}) = \frac{\partial z_t}{\partial z_{t-1}} \frac{\partial z_{t-1}}{\partial z_{t-2}} \cdots \frac{\partial z_{r+1}}{\partial z_r}$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Gradient-Lyapunov Structural Identity)</span></p>

Observe the structural identity between the definition of the Lyapunov exponent and the loss gradient. Both rely on a long product of Jacobians.

* If the system is **chaotic** (positive Lyapunov exponent), the product of Jacobians grows exponentially, leading to **exploding gradients**.
* If the system is **highly stable** (negative Lyapunov exponent), the product of Jacobians shrinks exponentially, leading to **vanishing gradients**.

This deep connection implies that the very dynamics we aim to reconstruct (like chaos) inherently make the optimization of the model parameters ($\theta$) challenging. Successful Dynamical Systems Reconstruction requires balancing these gradient dynamics to capture the true nature of the target system.

</div>

### Training RNNs on Chaotic Dynamical Systems

In the context of dynamical systems reconstruction, we often employ Recurrent Neural Networks (RNNs) to approximate an underlying flow or map. However, when the target system exhibits chaotic behavior, a fundamental training difficulty arises concerning the stability of gradients.

#### The Challenge of Chaotic Dynamics and Exploding Gradients

When we train an RNN defined by a mechanism $f$ to capture the behavior of a chaotic system, we must consider the recursion of the latent states and the resulting product series of Jacobians.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Gradient Explosion in Chaotic Reconstructions)</span></p>

If a recurrent neural network $f$ successfully captures a chaotic system, the loss gradients will inevitably explode.

</div>

<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/ChaoticDynamicsAndLossGradients.png' | relative_url }}" alt="Newton–Raphson iteration animation" loading="lazy">
</figure>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proof</span><span class="math-callout__name">(Gradient Explosion in Chaotic Reconstructions)</span></p>

1. Recall the recursion relationship for the latent states $z_t$ in an RNN. The loss gradient involves a product series of Jacobians of the form: $\prod_{i} \frac{\partial z_{i+1}}{\partial z_i}$.
2. By definition, a chaotic system is characterized by a maximum Lyapunov exponent $\lambda_{\max} > 0$.
3. The Lyapunov exponent describes the exponential rate of divergence of nearby trajectories. In terms of the Jacobian product series, a positive $\lambda_{\max}$ implies that the singular values of this product series have an absolute value larger than $1$.
4. As the product series grows with the time horizon, the terms within the gradient calculation grow exponentially.
5. Therefore, if the RNN $f$ effectively recreates the chaotic properties of the target system (such as the Lorenz attractor), the loss gradients will explode.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Chaos-Training Paradox)</span></p>

This presents a paradox: to accurately reconstruct a dynamical system, the model must exhibit the same chaotic properties as the data. However, those very properties make the model nearly impossible to train using standard backpropagation because the sensitivity to initial conditions (chaos) manifests as exploding gradients in the parameter space.

</div>

#### Sparse Teacher Forcing

To address the exploding gradient problem in chaotic systems, we utilize a technique known as Teacher Forcing. While classical teacher forcing has existed in the literature since the late 1980s (e.g., Williams and Zipser, 1989), modern dynamical systems reconstruction requires a more nuanced approach. In classical teacher forcing, every predicted value during training is replaced by the actual observed value from the data. However, for dynamical systems, we use Sparse Teacher Forcing (STF).

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Sparse Teacher Forcing)</span></p>

**STF** is a control-theoretic technique where the forward-propagated latent states are replaced by estimates derived from observed data at specific, well-chosen temporal intervals, rather than at every time step.

Assume we have an observed time series of length $T$, where observations $x_t \in \mathbb{R}^n$ are coupled to latent states $z_t \in \mathbb{R}^m$ via an observation function (decoder) $g$:

$$x_t = g(z_t)$$

To implement STF, we perform the following steps:

1. **Invert the Decoder:** Obtain an estimate of the latent state $\tilde{z}_t$ from the observed data $x_t$. If $g$ is a linear mapping, we use the pseudo-inverse: 
   
   $$\tilde{z}_t = g^+(x_t)$$
   
   Alternatively, this can be framed as a regression problem to find the latent state that best maps to the observation.

2. **Define the Forcing Interval:** Let $\tau \in \mathbb{N}^+$ be the forcing interval. We define a set of forcing times: 
   
   $$T_{\text{force}} = \lbrace n \cdot \tau \mid n \in \mathbb{N} \rbrace$$

3. **Compute the Latent Trajectory:** During training, the next latent state $z_{t+1}$ is calculated as:

   $$z_{t+1} = \begin{cases} f(\tilde{z}_t) & \text{if } t \in T_{\text{force}} \\ f(z_t) & \text{otherwise} \end{cases}$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Sparse Teacher Forcing)</span></p>

By forcing the trajectory back onto the "true" latent path every $\tau$ steps, we prevent the model’s trajectory from diverging too far from the data. Crucially, because $\tilde{z}\_t$ is derived directly from the data and is not a function of the previous hidden state $z_{t-1}$ in the computational graph, the gradient $\frac{\partial \tilde{z}\_t}{\partial z_{t-1}}$ is zero. This effectively "cuts" the gradient chain every $\tau$ steps, preventing the product series from exploding.

</div>

<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/SparseTeachingForcing.png' | relative_url }}" alt="Newton–Raphson iteration animation" loading="lazy">
</figure>

#### Optimizing the Forcing Interval

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Forcing Interval Choice Trade-off)</span></p>

The choice of the forcing interval $\tau$ is critical for successful reconstruction.

* If $\tau$ is **too small:** The system only learns one-step-ahead predictions. It loses the ability to capture long-term properties, such as the overall geometry of the attractor or temporal agreement.
* If $\tau$ is **too large:** The system runs back into the exploding gradient problem as the chaotic divergence dominates the product series.

The optimal $\tau$ can be determined by the physical properties of the system being studied.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Predictability Time)</span></p>

The **predictability time** provides an estimate of how long a trajectory remains "close" to its initial path before chaotic divergence becomes severe. It is inversely proportional to the maximum Lyapunov exponent $\lambda_{\max}$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Optimal Forcing Interval)</span></p>

The optimal choice for the forcing interval $\tau$ is approximately given by the predictability time:

$$\tau \approx \frac{\ln(2)}{\lambda_{\max}}$$

</div>

<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/OptimalForcingInterval.png' | relative_url }}" alt="Newton–Raphson iteration animation" loading="lazy">
</figure>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Applications of Sparse Teacher Forcing)</span></p>

* **Brain Dynamics:** In reconstructions of human fMRI or EEG data, utilizing STF with an optimized $\tau$ allows the model to generate patterns that overlap with real data in both power spectra and state space.
* **Empirical Minimization:** Empirical tests show that measures like the Hellinger distance or state space divergence reach a minimum when $\tau$ is chosen near the predictability time.

</div>

#### Alternative Approaches: Multiple Shooting

A related technique for handling long-term trajectories in dynamical systems is Multiple Shooting, a term adapted from the physical sciences literature (e.g., Voss et al., 2004).

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Multiple Shooting)</span></p>

**Multiple Shooting** involves segmenting a long time series into $N$ shorter segments. For each segment $n$, we estimate a unique initial condition $m_0^{(n)}$ as a trainable parameter.

The goal is to minimize a squared error loss function across all segments and all time steps within those segments, while maintaining temporal contingency:

$$\min_{\theta, m_0^{(n)}} \sum_{n} \sum_{t} \| x_{n,t} - \hat{x}_{n,t}(\theta, m_0^{(n)}) \|^2$$

</div>

<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/MultipleShooting.png' | relative_url }}" alt="Newton–Raphson iteration animation" loading="lazy">
</figure>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Property</span><span class="math-callout__name">(Multiple Shooting vs. Teacher Forcing)</span></p>

Similar to Sparse Teacher Forcing, Multiple Shooting breaks the long-term dependency into manageable pieces. The primary challenge is ensuring "temporal contingency" — meaning that the end of one segment should logically connect to the beginning of the next, despite the segments being treated as independent initial-value problems during optimization.

</div>

### Advanced Training Techniques

When dealing with complex temporal sequences, standard gradient descent often fails due to exploding gradients or highly non-convex loss landscapes. We investigate strategies such as multiple shooting with continuity regularization, control-theoretic approaches, and generalized teacher forcing to stabilize training and ensure model fidelity.

#### Multiple Shooting and Continuity Regularization

In many reconstruction tasks, it is beneficial to break a long time series into smaller segments or "batches." However, this introduces the problem of maintaining continuity across these artificial boundaries.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Multiple Shooting Constraint)</span></p>

In a **multiple shooting framework**, the initial condition of a segment $n+1$ must be contingent upon the value forward-propagated from the preceding segment $n$. If we denote the estimate at segment $n+1$ as $z_{n+1}$, we require that it matches the result of the map $f_\theta$ applied to the final state of the previous segment.

</div>

To enforce this continuity without strictly constraining the optimizer, we introduce a regularization term to the standard loss function.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Continuity Regularization Loss)</span></p>

The **Continuity Regularization Loss** $L_{\text{total}}$ is formulated by adding a regularization term $L_{\text{reg}}$ to the primary loss $L_{\text{usual}}$, weighted by a parameter $\lambda$:

$$L_{\text{total}} = L_{\text{usual}} + \lambda \sum_{n} \| z_{n+1} - f_\theta^T(z_n) \|^2$$

where $f_\theta^T(z_n)$ represents the state after $T$ time steps (the length of the segment) starting from the initial condition estimate $m_0$ of the previous segment.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Intuition</span><span class="math-callout__name">(Continuity Regularization)</span></p>

By adding this term, we "knit" the individual temporal intervals together. This allows the system to explore the state space more effectively while ensuring that the resulting trajectory is physically plausible and continuous across the entire observed duration.

</div>

#### Control Theory in Dynamical Systems

The challenge of training dynamical models can be viewed through the lens of Control Theory. Control theory is the study of finding optimal signals to steer a system toward a desired behavior, such as pushing a chaotic system onto a specific limit cycle within the state space. In the context of machine learning, techniques like Teacher Forcing are effectively control-theoretic methods where we use loss gradients to guide the model’s behavior.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Biophysical Neuron Model)</span></p>

Consider a biophysical model of a spiking neuron defined by a voltage signal and gating variables. To estimate the parameters of this system from real voltage recordings, we can add a control term to the differential equations:

$$\dot{V} = g(V, \dots) + \kappa(V_{\text{observed}} - V_{\text{model}})$$

where $\kappa$ is a gain parameter. Key observations:

* If $\kappa$ is sufficiently large, the model is "pushed" toward the real signal.
* This process actually smooths the loss function, making it more convex and easier to optimize.
* However, just as with sparse teacher forcing, a scheme is required to regulate $\kappa$ during the training process to ensure the model eventually learns to generate the signal autonomously.

</div>

#### Generalized Teacher Forcing (GTF)

Generalized Teacher Forcing (GTF) is a sophisticated training technique (introduced by Florian Hess et al., 2023) that builds upon the work of Kenji Doya (1992). Unlike sparse teacher forcing, which simply replaces model states with data, GTF uses a graded approach.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(GTF State Update)</span></p>

In GTF, we define a hybrid state $\tilde{z}_t$ as a weighted average of the true state (from data) and the model’s forward-propagated estimate:

$$\tilde{z}_t = (1 - \alpha)\, f_\theta(\tilde{z}_{t-1}) + \alpha\, \hat{z}_t$$

where:

* $\alpha \in [0, 1]$ is the weighting parameter.
* $f_\theta(\tilde{z}_{t-1})$ is the forward-propagated state from the previous step.
* $\hat{z}_t$ is the estimate obtained from the real data (typically via an inverted decoder model applied to the time series $X_t$).

</div>

#### Gradient Stability and Jacobian Analysis

The primary motivation for GTF is to solve the exploding gradient problem by regulating the product series of Jacobians.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proof</span><span class="math-callout__name">(Derivation of the GTF Jacobian)</span></p>

We wish to find the Jacobian of the state $\tilde{z}\_t$ with respect to the state at a previous time step $\tilde{z}\_{t-1}$.

1. Start with the GTF definition: 
   
   $$\tilde{z}_t = (1 - \alpha)\, f_\theta(\tilde{z}_{t-1}) + \alpha\, \hat{z}_t$$

2. Differentiate $\tilde{z}\_t$ with respect to $\tilde{z}\_{t-1}$. Note that $\hat{z}\_t$ (the data estimate) is independent of the model’s previous state $\tilde{z}\_{t-1}$, so its derivative is zero.
3. Apply the chain rule to the first term:

   $$\frac{\partial \tilde{z}_t}{\partial \tilde{z}_{t-1}} = (1 - \alpha) \cdot \frac{\partial f_\theta(\tilde{z}_{t-1})}{\partial \tilde{z}_{t-1}}$$

4. Let $G_t$ be the Jacobian of the map $f_\theta$ at time $t$. The single-step Jacobian is: 
   
   $$J_t = (1 - \alpha)\, G_t$$

5. For a sequence of $T$ steps, the product series of Jacobians becomes:

   $$\prod_{k=0}^{T-1} (1 - \alpha)\, G_{T-k}$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Optimal Alpha Selection)</span></p>

To prevent gradients from exploding, the maximum singular value of the product series should be kept near unity. To regulate the gradients such that they hover around $1$, $\alpha$ should be chosen based on the maximum singular value ($\sigma_{\max}$) of the model’s Jacobian $G_t$:

$$\alpha \approx 1 - \frac{1}{\sigma_{\max}(G_t)}$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Property</span><span class="math-callout__name">(Practical Considerations for GTF)</span></p>

By adapting $\alpha$ at each time step (or every few steps), we can automatically ensure that the gradients remain stable. While calculating the Singular Value Decomposition (SVD) at every step is computationally expensive, one can use effective proxies to adapt $\alpha$ efficiently. GTF strikes an optimal balance between the true data trajectory and the model’s predicted trajectory. By choosing $\alpha$ correctly, the loss landscape becomes almost convex, facilitating efficient convergence in complex dynamical reconstruction tasks.

</div>

