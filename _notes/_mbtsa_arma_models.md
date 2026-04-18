## Autoregressive Moving Average (ARMA) Models

ARMA models are a fundamental class of models for analyzing stationary time series. They are built on the principle that the current value of a series can be explained by a combination of its own past values and past random shocks.

### Motivation and Components

If the residuals of a regression model on time series data are found to be autocorrelated or cross-correlated, it implies that the model is missing important temporal structure. ARMA models are designed to capture this very structure.

  * **Autoregressive (AR) Part:** This component regresses the time series on its own past values. It captures the "memory" or persistence in the series.
  * **Moving Average (MA) Part:** This component models the current value as a function of past random perturbations or "shocks". It can be thought of as a sequence of weighted random shocks.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Autoregressive model (AR))</span></p>

An **Autoregressive model of order $p$**, denoted **AR($p$)**, is defined as:

$$X_t = a_0 + \sum_{i=1}^p a_i X_{t-i} + \epsilon_t$$

where $\epsilon_t$ is a white noise process, typically $\epsilon_t \sim \mathcal{WN}(0, \sigma^2)$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Moving Average Model (MA))</span></p>

A **Moving Average model of order $q$**, denoted **MA($q$)**, is defined as:

$$X_t = b_0 + \sum_{j=1}^q b_j \epsilon_{t-j} + \epsilon_t$$

Note that $X_t$ depends on past **error terms**, not past values of $X$ itself.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(ARMA Model)</span></p>

Combining these two components gives the **ARMA($p$,$q$)** model:

$$X_t = c + \sum_{i=1}^p a_i X_{t-i} + \sum_{j=1}^q b_j \epsilon_{t-j} + \epsilon_t$$

This can also be extended to include external inputs $u_t$. The full set of model parameters to be estimated is 

$$\theta = \lbrace c, a_1, \dots, a_p, b_1, \dots, b_q, \sigma^2 \rbrace$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(What's the point of considering errors in predicting a new state? If they are errors, what's their predictive power?)</span></p>

The trick is to stop thinking of these as "mistakes" and start thinking of them as **shocks** or **innovations**.

**The "Impulse" Idea**

In an MA model, $\epsilon_t$ represents a new piece of information that enters the system at time $t$ that was **not** explained by the past.

Imagine you are modeling the water level in a harbor.

* **The Shock ($\epsilon$):** A giant ship enters the harbor, creating a displacement wave.
* **The Persistence:** That wave doesn't disappear instantly. It ripples and bounces for a while.
* **The Prediction:** Even if you can't predict *when* the next ship will arrive, if you know a ship arrived one minute ago ($\epsilon_{t-1}$), you can predict that the water will still be choppy *now* ($X_t$).

**The "error" has predictive power because its effects linger.**

**Information vs. Noise**

In econometrics, we often call these terms **innovations**.

* **White Noise:** Purely random and unpredictable *at the moment it happens*.
* **The Model's Job:** To capture how much of that "random shock" stays in the system for the next step.

If the coefficient $\theta$ is 0.8, it means 80% of yesterday's unexpected shock is still influencing today's value. We aren't predicting the *error itself*; we are predicting the **observed value** based on the fact that a specific shock recently occurred.

**Error vs. Residual**

There is a subtle but vital distinction here:

1. **The Theoretical Error ($\epsilon$):** This is the "shock." We assume it happened.
2. **The Residual ($\hat{\epsilon}$):** This is what we calculate after the fact.

When we "predict" using an MA model, we use the **residuals** from previous steps. If our model predicted the value would be 100, but it turned out to be 110, we know there was a  "shock." Since the MA model says shocks linger, we add a fraction of that  to our prediction for the next step.

**Why not just use past values of X (Autoregression)?**

You might ask: *"Why not just use the previous water level ($X_{t-1}$) instead of the previous shock ($\epsilon_{t-1}$)?*

* **AR models (using $X$):** Assume the *entire* past value influences the future. This creates a "long memory" where effects decay slowly.
* **MA models (using $\epsilon$):** Assume only the *random shocks* influence the future. This creates a "short memory." After  steps, the shock is completely gone from the system.

**Summary: The "Pothole" Analogy**

Think of driving a car with bad shocks:

* **AR Model:** The car’s height right now depends on its height a second ago (the car stays bouncy).
* **MA Model:** The car’s height right now depends on the fact that you hit a **pothole** (the error/shock) three seconds ago.

The "point" of the MA model is to capture those temporary, lingering effects of specific events without assuming the entire history of the variable matters.

</div>

### Duality and Stationarity

There is a fundamental duality between AR and MA processes. Under certain stability conditions, any finite-order AR process can be represented as an infinite-order MA process, and vice-versa.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Finite AR $\iff$ Infinite MA)</span></p>

Any finite-order AR process can be represented as an infinite-order MA process, and vice-versa:

$$X_t = a_0 + a_1 X_{t-1} + \epsilon_t = a_0 \sum_{k=0}^{\infty} a_1^k + \sum_{k=0}^{\infty} a_1^k \epsilon_{t-k}$$

**This infinite expansion is only valid if the series converges.**

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>Let's examine this with a simple AR(1) process:</p>
    $$X_t = a_0 + a_1 X_{t-1} + \epsilon_t$$
    <p>We can recursively expand this expression:</p>
    $$
    \begin{aligned}
    X_t &= a_0 + a_1(a_0 + a_1 X_{t-2} + \epsilon_{t-1}) + \epsilon_t \\
    &= a_0 + a_1 a_0 + a_1^2 X_{t-2} + a_1 \epsilon_{t-1} + \epsilon_t \\
    &= a_0(1 + a_1) + a_1^2 (a_0 + a_1 X_{t-3} + \epsilon_{t-2}) + a_1 \epsilon_{t-1} + \epsilon_t \\
    &= a_0(1 + a_1 + a_1^2) + a_1^3 X_{t-3} + a_1^2 \epsilon_{t-2} + a_1 \epsilon_{t-1} + \epsilon_t \\
    &\dots \\
    &= a_0 \sum_{k=0}^{\infty} a_1^k + \sum_{k=0}^{\infty} a_1^k \epsilon_{t-k}
    \end{aligned}
    $$
  </details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Stationarity in the Mean for AR(1))</span></p>

* **For the process to be stationary in the mean, its expected value must be constant and finite.**
* The condition for stationarity of an AR(1) process is $\lvert a_1 \rvert < 1$.

$$\mathbb{E}[X_t] = \frac{a_0}{1-a_1} \quad \text{if } \lvert a_1 \rvert < 1$$

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>Taking the expectation of the expanded form:</p>
    $$\mathbb{E}[X_t] = \mathbb{E}\left[ a_0 \sum_{k=0}^{\infty} a_1^k + \sum_{k=0}^{\infty} a_1^k \epsilon_{t-k} \right]$$
    $$\mathbb{E}[X_t] = a_0 \sum_{k=0}^{\infty} a_1^k + \sum_{k=0}^{\infty} a_1^k \mathbb{E}[\epsilon_{t-k}]$$
    <p>Since $\mathbb{E}[\epsilon_{t-k}]=0$, the second term vanishes. The first term is a geometric series which converges if and only if $\lvert a_1 \rvert < 1$.</p>
    $$\mathbb{E}[X_t] = \frac{a_0}{1-a_1} \quad \text{if } \lvert a_1 \rvert < 1$$
    <p>Therefore, the condition for stationarity of an AR(1) process is $\lvert a_1 \rvert < 1$.</p>
  </details>
</div>

<figure>
  <img src="{{ '/assets/images/notes/model-based-time-series-analysis/ar_1_different_a1.png' | relative_url }}" alt="Newton–Raphson iteration animation" loading="lazy">
  <figcaption>AR(1) time series with different $\rvert a_1\rvert$.</figcaption>
</figure>

#### State-Space Representation and Stability

A powerful technique for analyzing AR models is to write them in a **state-space (or vector) form**. 

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(scalar AR($p$) $\implies$ $p$-variate VAR(1))</span></p>

Any scalar AR($p$) process can be represented as a $p$-variate VAR(1) process:

$$X_t = a_0 + \sum_{i=1}^p a_i X_{t-i} + \epsilon_t \quad\implies\quad \mathbf{X}_t = \mathbf{a} + A \mathbf{X}_{t-1} + \mathbf{\epsilon}_t$$

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>Consider an AR($p$) process</p>
    $$X_t = a_0 + \sum_{i=1}^p a_i X_{t-i} + \epsilon_t$$
    <p>We can define a $p$-dimensional state vector $\mathbf{X}_t$:</p>
    $$\mathbf{X}_t = \begin{pmatrix} X_t \\ X_{t-1} \\ \vdots \\ X_{t-p+1} \end{pmatrix}$$
    <p>The process can then be written in the form</p>
    $$\mathbf{X}_t = \mathbf{a} + A \mathbf{X}_{t-1} + \mathbf{\epsilon}_t$$
    <p>where:</p>
    $$
    \mathbf{a} = \begin{pmatrix} a_0 \\ 0 \\ \vdots \\ 0 \end{pmatrix}, \quad
    A = \begin{pmatrix}
    a_1 & a_2 & \dots & a_p \\
    1 & 0 & \dots & 0 \\
    \vdots & \ddots & & \vdots \\
    0 & \dots & 1 & 0
    \end{pmatrix}, \quad
    \mathbf{\epsilon}_t = \begin{pmatrix} \epsilon_t \\ 0 \\ \vdots \\ 0 \end{pmatrix}
    $$
  </details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Stationarity $\implies$ spectral radius of $A$ is less than 1)</span></p>

* The stability and stationarity of the entire process can then be assessed by examining the eigenvalues of the companion matrix $A$. 
* For the process to be stationary, the spectral radius of $A$ must be less than 1.

$$\max_i \lvert \lambda_i(A) \rvert < 1$$

where $\lambda_i(A)$ are the eigenvalues of $A$.

</div>

<div class="accordion">
  <details>
    <summary>Why eigenvalues (not singular values) determine stationarity</summary>
    <p>
      This is a profound question that touches on the subtle difference between <strong>transient behavior</strong> (short-term) and <strong>asymptotic behavior</strong> (long-term).
    </p>
    <p>
      You are absolutely correct that Singular Value Decomposition (SVD) gives a clearer picture of how a matrix distorts space in the immediate, orthogonal sense. However, for stationarity, we rely on eigenvalues.
    </p>
    <hr>
    <h4>1. The Time Horizon: "Eventually" vs. "Immediately"</h4>
    <ul>
      <li><strong>Stationarity</strong> asks: "If I run this process for infinite time ($t \to \infty$), does it blow up?"</li>
      <li><strong>Singular values</strong> ask: "What is the maximum possible stretch this matrix causes in a <em>single</em> step?"</li>
    </ul>
    <p>A system can be stationary even if it stretches vectors significantly in the short run, provided that it eventually shrinks them back down.</p>
    <h4>2. The Math of Iteration: $A^k$</h4>
    <p>Stationary conditions usually involve iterating a transition matrix $A$. We look at the state vector evolving over time: $x_t = A^t x_0$.</p>
    <p>Let's look at what happens to the powers of the matrix using both decompositions.</p>
    <h5>The Eigendecomposition (Spectral Analysis)</h5>
    <p>
      If $A$ is diagonalizable, write $A = P \Lambda P^{-1}$. Then:
    </p>
    $$
    A^t = (P \Lambda P^{-1})(P \Lambda P^{-1}) \dots = P \Lambda^t P^{-1}
    $$
    <ul>
      <li>If $\lvert \lambda_i \rvert < 1$ for all $i$, then $\lambda_i^t \to 0$ as $t \to \infty$.</li>
      <li>Therefore, $A^t \to 0$ and the system is stable/stationary.</li>
    </ul>
    <h5>The SVD (Singular Values)</h5>
    <p>If we use SVD, $A = U \Sigma V^T$, then</p>
    $$
    A^t = (U \Sigma V^T)(U \Sigma V^T) \dots
    $$
    <ul>
      <li>$V^T U$ is <em>not</em> the identity (unless $A$ is normal/symmetric), so the rotations do not cancel.</li>
      <li>Therefore, $A^t \neq U \Sigma^t V^T$ and singular values cannot be simply raised to $t$ to predict the future.</li>
    </ul>
    <blockquote><strong>Key takeaway:</strong> Eigenvalues dictate the "fate" of the system because they survive repeated multiplication. Singular values describe the matrix right now, but that description gets scrambled during iteration.</blockquote>
    <hr>
    <h4>3. The "Strictness" Trap: Sufficient vs. Necessary</h4>
    <p>Forcing all singular values below 1 is a sufficient but not necessary condition.</p>
    <ul>
      <li><strong>$\sigma_{\text{max}} &lt; 1$</strong>: the system contracts in Euclidean length at every step (monotonic decay).</li>
      <li><strong>$\lvert \lambda_{\text{max}} \rvert &lt; 1$</strong>: the system contracts eventually; it may expand transiently before it dies out.</li>
    </ul>
    <p>If we required $\sigma_{\text{max}} &lt; 1$, we would reject many valid stationary models that merely experience transient growth.</p>
    <hr>
    <h4>4. A Concrete Counter-Example (Non-Normal Matrix)</h4>
    <p>Consider a shear matrix that is stable overall but stretches space heavily in the short term:</p>
    $$
    A = \begin{bmatrix} 0.5 & 100 \\ 0 & 0.5 \end{bmatrix}
    $$
    <p><strong>Eigenvalues ($\lambda$):</strong> Since it is upper triangular, $\lambda_1 = 0.5, \lambda_2 = 0.5$. Because $0.5 &lt; 1$, the system is stationary; $A^t \to 0$ as $t \to \infty$.</p>
    <p><strong>Singular values ($\sigma$):</strong> $\sigma_1 \approx 100$ and $\sigma_2 \approx 0.0025$, so $\sigma_{\max} \gg 1$.</p>
    <p>If we enforced $\sigma_{\max} &lt; 1$, we would incorrectly label this stable system as unstable. The matrix can grow vectors 100x in step 1, but by step 50 the $0.5^{50}$ factor dominates and the vector collapses.</p>
    <h4>Conclusion</h4>
    <p>Stationarity is an asymptotic property ($t \to \infty$). Eigenvalues track that long-run fate; singular values measure single-step gain. Singular values are great for understanding numerical stability and transient spikes, but eigenvalues are the gatekeepers of whether a process explodes or stabilizes over time.</p>
  </details>
</div>

<div class="accordion">
  <details>
    <summary>Transient growth (stable eigenvalues, large singular values)</summary>
    <p>
      Here is a Python demonstration of <strong>transient growth</strong>: a system that is asymptotically stable (eigenvalues &lt; 1) but effectively unstable in the short term (singular values &gt;&gt; 1). Energy humps upward before eventually decaying.
    </p>
    <h4>The setup</h4>
    <p>Use a non-normal shear matrix:</p>
    $$
    A = \begin{bmatrix} 0.9 & 5 \\ 0 & 0.9 \end{bmatrix}
    $$
    <ul>
      <li><strong>Eigenvalues:</strong> $\lambda = 0.9$ (eventually decays to 0).</li>
      <li><strong>Singular values:</strong> $\sigma_{\max} \approx 5.3$ (can grow $5\times$ in one step).</li>
    </ul>
    <p>
      <img src="{{ '/assets/images/notes/model-based-time-series-analysis/transient_growth.png' | relative_url }}" alt="Transient growth demo: shear matrix with stable eigenvalues and large singular values" loading="lazy">
    </p>
    <h4>What the graph shows</h4>
    <h5>Phase 1: the SVD phase (steps 0–~20)</h5>
    <p>The line shoots upward; magnitude grows from about 1.0 to nearly 15.0 even though long-term decay is 0.9.</p>
    <ul>
      <li><strong>Why?</strong> The shear term (5) dominates the diagonal decay (0.9), stretching the vector into a high-gain direction.</li>
    </ul>
    <h5>Phase 2: the eigen phase (steps 20+)</h5>
    <p>The curve peaks and crashes toward zero.</p>
    <ul>
      <li><strong>Why?</strong> Once the transient shear is exhausted, repeated multiplication by 0.9 takes over; the eigenvalue constraint wins and the system stabilizes.</li>
    </ul>
    <h4>The geometric reason: non-orthogonal eigenvectors</h4>
    <p>
      In a normal (symmetric) matrix, eigenvectors are orthogonal. In this shear matrix, they are nearly parallel. To represent the initial state, you subtract two large, nearly collinear eigenvector components; as time evolves one decays slightly faster, the cancellation breaks, and the large magnitude is revealed before it decays. Singular values flag this transient spike, while eigenvalues certify eventual stability.
    </p>
  </details>
</div>

<div class="accordion">
  <details>
    <summary>How non-normal shear squeezes eigenvectors</summary>
    <p>This happens because <strong>Non-Normal</strong> matrices (matrices that do not commute with their transpose, $A^T A \neq A A^T$) contain <em>shear</em>.</p>
    <p>In a Symmetric (Normal) matrix, the transformation is pure stretching along orthogonal axes. In a Non-Normal matrix, the transformation includes a sliding or shearing motion that corrupts orthogonality. Here is the geometric and mathematical reason why this squeezes the eigenvectors together.</p>
    <h4>1. The Geometric Intuition: Stretching vs. Shearing</h4>
    <p>Imagine painting a grid on a rubber sheet and applying a matrix transformation.</p>
    <ul>
      <li><strong>Symmetric Matrix (Stretch):</strong> You pull the sheet north/south and squash it east/west. The grid lines remain at 90 degrees to each other. These lines are your eigenvectors; they are orthogonal.</li>
      <li><strong>Non-Normal Matrix (Shear):</strong> You place your hand on the top of the sheet and slide it to the right while holding the bottom fixed. The vertical lines tilt over while the horizontal lines stay horizontal.
        <ul>
          <li>One eigenvector is still horizontal.</li>
          <li>The other eigenvector (which used to be vertical) tilts to chase the shear.</li>
          <li><strong>Result:</strong> The two eigenvectors are no longer at 90 degrees; they are squeezed toward each other.</li>
        </ul>
      </li>
    </ul>
    <h4>2. The $2 \times 2$ Proof</h4>
    <p>Consider a simple upper triangular matrix and its eigenvectors:</p>
    $$
    A = \begin{bmatrix} 1 & k \\ 0 & 2 \end{bmatrix}
    $$
    <p>Here, $k$ represents the shear (non-normality).</p>
    <ul>
      <li><strong>Eigenvalues:</strong> The diagonal entries, $\lambda_1 = 1$ and $\lambda_2 = 2$.</li>
      <li><strong>Eigenvector 1 ($v_1$):</strong> Associated with $\lambda_1 = 1$, $v_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$ (horizontal).</li>
      <li><strong>Eigenvector 2 ($v_2$):</strong> Associated with $\lambda_2 = 2$. Solving $(A - 2I)v = 0$ gives $v_2 = \begin{bmatrix} k \\ 1 \end{bmatrix}$.</li>
    </ul>
    <p><strong>Observe the angle:</strong></p>
    <ul>
      <li>$v_1$ points East $(1, 0)$.</li>
      <li>$v_2$ points North-East $(k, 1)$.</li>
    </ul>
    <p>As the shear $k$ grows (or as the difference between eigenvalues shrinks), $v_2$ tilts toward the horizontal axis.</p>
    <ul>
      <li>If $k = 100$, $v_2 = (100, 1)$, almost parallel to $v_1 = (1, 0)$.</li>
      <li>The angle between them is nearly zero—they are squeezed.</li>
    </ul>
    <h4>3. Why is this ill-conditioned? (The cancellation problem)</h4>
    <p>This squeezing creates a numerical nightmare called <strong>ill-conditioning</strong>. To represent a simple state vector, like vertical <em>Up</em> $\begin{bmatrix} 0 \\ 1 \end{bmatrix}$, using these eigenvectors, you must use a linear combination: $x = c_1 v_1 + c_2 v_2$.</p>
    <p>If $v_1$ and $v_2$ are nearly parallel (e.g., $v_1 = [1, 0]$ and $v_2 = [1, 0.01]$), you need massive coefficients to describe a small vertical vector:</p>
    $$
    \begin{bmatrix} 0 \\ 1 \end{bmatrix} = -100 \cdot \begin{bmatrix} 1 \\ 0 \end{bmatrix} + 100 \cdot \begin{bmatrix} 1 \\ 0.01 \end{bmatrix}
    $$
    <ul>
      <li><strong>Physically:</strong> Hidden energy; the system looks quiet but internally fights with massive opposing modes.</li>
      <li><strong>Numerically:</strong> Unstable; tiny rounding errors ruin the cancellation, causing transient growth.</li>
    </ul>
    <p>The condition number of the eigenvector matrix $\kappa(V)$ measures this. If eigenvectors are orthogonal, $\kappa(V) = 1$. As they squeeze together, $\kappa(V) \to \infty$.</p>
    <h4>Summary</h4>
    <p>Eigenvectors are directions that do not change orientation. When you apply shear (non-normality), you tilt the space. The eigenvectors tilt with it, losing their orthogonality and clamping together like a closing pair of scissors.</p>
  </details>
</div>

<div class="accordion">
  <details>
    <summary>Why non-normal weights trigger exploding gradients (deep nets)</summary>
    <p>This is the exact mathematical reason why exploding gradients are so dangerous in deep learning, especially in recurrent neural networks (RNNs). In deep learning, <em>depth</em> plays the role of <em>time</em>: layer 1, layer 2, layer 3 are steps $t=1, t=2, t=3$.</p>
    <p>During backpropagation, the gradient is multiplied by the weight matrix $W$ at every layer. If $W$ acts like a shear matrix, gradients follow the same transient-growth curve: they explode in middle layers even if the network is theoretically stable.</p>
    <h4>1. The finite-time trap</h4>
    <p>Control theory cares about $t \to \infty$; deep nets care about $t \approx 50$ or $100$ (network depth).</p>
    <ul>
      <li>At step 100: values are near 0 because stable eigenvalues eventually dominate.</li>
      <li>At step 10: values are huge because unstable singular values dominate.</li>
      <li>If a network is only 10 layers deep, it lives inside that spike; it never reaches the safe asymptotic zone, so gradients blow up and updates can become NaN.</li>
    </ul>
    <h4>2. The mechanics of backpropagation</h4>
    <p>Backpropagation forms a long product of Jacobian matrices. Even if $W$ is initialized so eigenvalues are small (e.g., $\lambda = 0.9$), a non-normal $W$ can have large singular values.</p>
    <ul>
      <li>The gradient aligns with the top singular vector.</li>
      <li>The gradient grows by $\sigma_{\max}$ at every layer.</li>
      <li>Example: if $\sigma_{\max} = 5$, by layer 5 the gradient is $5^5 = 3125$ times larger, overwhelming earlier layers.</li>
    </ul>
    <h4>3. Why weight matrices are "sheared"</h4>
    <p>Randomly initialized high-dimensional matrices are rarely normal; they are typically highly non-normal, which means their eigenvectors are squeezed and ill-conditioned—perfect conditions for transient growth.</p>
    <p>This motivates <strong>orthogonal initialization</strong> ($W^T W = I$):</p>
    <ul>
      <li>For orthogonal matrices, singular values equal eigenvalues and both equal 1.</li>
      <li>No hidden shear, no transient growth; signals travel deep without exploding or vanishing.</li>
    </ul>
    <h4>Summary</h4>
    <table>
      <thead>
        <tr>
          <th>Concept</th>
          <th>In Control Theory</th>
          <th>In Deep Learning</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Goal</td>
          <td>Stability as $t \to \infty$</td>
          <td>Stability at finite depth $L$</td>
        </tr>
        <tr>
          <td>Danger</td>
          <td>Unstable eigenvalues ($|\lambda| &gt; 1$)</td>
          <td>Non-normal weights with large $\sigma_{\max}$ causing transient spikes</td>
        </tr>
        <tr>
          <td>The Trap</td>
          <td>System looks unstable initially but settles</td>
          <td>Gradient explodes before reaching the input</td>
        </tr>
        <tr>
          <td>The Fix</td>
          <td>Wait longer</td>
          <td>Gradient clipping or orthogonal init</td>
        </tr>
      </tbody>
    </table>
    <p>Want a quick visual? See “Vanishing AND Exploding Gradient Problem Explained” (video) for animations of how gradients shrink or blow up as they flow backward.</p>
    <p>Would you like an extra note here explaining gradient clipping—the brute-force way to chop off that transient spike?</p>
  </details>
</div>

### Model Identification Using Autocorrelation

* A key step in ARMA modeling is identifying the orders $p$ and $q$. 
* **The Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) are the primary tools for this task.**

#### Autocorrelation in AR(1) Processes

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Autocovariance and autocorrelation for zero-mean AR(1))</span></p>

For a zero-mean ($a_0=0$) AR(1) process 

$$X_t = a_1 X_{t-1} + \epsilon_t$$

The **autocovariance** at lag $k$ ($\gamma(k)$) and **autocorrelation** at lag $k$ ($\rho(k)$):

$$\gamma(k) = a_1^k \gamma(0) \qquad \rho(k) = a_1^k$$

The ACF of an AR(1) process **decays exponentially** to zero.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>Consider a zero-mean ($a_0=0$) AR(1) process:</p>
    $$X_t = a_1 X_{t-1} + \epsilon_t$$
    <p>The autocovariance at lag $k$, $\gamma(k)$, can be calculated.</p>
    <ul>
      <li><strong>Lag 1:</strong></li>
    </ul>
    $$\mathbb{E}[X_t X_{t-1}] = \mathbb{E}[(a_1 X_{t-1} + \epsilon_t)X_{t-1}] = a_1 \mathbb{E}[X_{t-1}^2] + \mathbb{E}[\epsilon_t X_{t-1}]$$
    <p>$\mathbb{E}[\epsilon_t X_{t-1}]=0$ since $\epsilon_t$ is uncorrelated with past values of $X$. Thus, $\gamma(1) = a_1 \gamma(0)$.</p>
    <ul>
      <li><strong>Lag 2:</strong></li>
    </ul>
    $$\mathbb{E}[X_t X_{t-2}] = \mathbb{E}[(a_1 X_{t-1} + \epsilon_t)X_{t-2}] = a_1 \mathbb{E}[X_{t-1} X_{t-2}] = a_1 \gamma(1) = a_1^2 \gamma(0)$$
    <ul>
      <li><strong>General Lag $k$:</strong></li>
    </ul>
    $$\gamma(k) = a_1^k \gamma(0)$$
    <p>The autocorrelation function, $\rho(k) = \gamma(k)/\gamma(0)$, is therefore $\rho(k) = a_1^k$. The ACF of an AR(1) process <strong>decays exponentially</strong> to zero.</p>
  </details>
</div>

#### Autocorrelation in MA($q$) Processes

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Autocorrelation in MA($q$) Processes)</span></p>

For zero-mean MA($q$) process 

$$X_t = \epsilon_t + \sum_{j=1}^q b_j \epsilon_{t-j}$$

$$\text{ACF}(k) = 0 \quad \text{for all } k > q$$

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>Consider a zero-mean MA($q$) process: $X_t = \epsilon_t + \sum_{j=1}^q b_j \epsilon_{t-j}$. Let's calculate the autocovariance at lag $k > q$.</p>
    <p>Because the error terms are white noise, $\mathbb{E}[\epsilon_i \epsilon_j] = \sigma^2$ if $i=j$ and 0 otherwise. For the expectation $\mathbb{E}[X_t X_{t-k}]$ to be non-zero, there must be at least one pair of matching indices in the sums. If we consider a lag $k>q$, it is impossible to satisfy the matching index condition. Therefore, for any $k>q$, all cross-product terms have an expectation of zero.</p>
    $$\text{ACF}(k) = 0 \quad \text{for all } k > q$$
  </details>
</div>

<figure>
  <img src="{{ '/assets/images/notes/model-based-time-series-analysis/ACF_MA.png' | relative_url }}" alt="Filtering Smoothing Schema" loading="lazy">
  <figcaption>Autocorrelation in MA($q$) Process.</figcaption>
</figure>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(ACF of an MA($q$) process)</span></p>

This provides a clear signature: the ACF of an MA($q$) process **sharply cuts off** to zero after lag $q$.

</div>

#### The Partial Autocorrelation Function (PACF)

The PACF at lag $k$ measures the correlation between $X_t$ and $X_{t-k}$ after removing the linear dependence on the intervening variables ($X_{t-1}, X_{t-2}, \dots, X_{t-k+1}$). A key property of the PACF for an AR($p$) process is:

$$\text{PACF}(k) = 0 \quad \text{for all } k > p$$

This is because in an AR($p$) model, the direct relationship between $X_t$ and $X_{t-k}$ (for $k>p$) is fully mediated by the first $p$ lags.

### Modeling with ARMA

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Parameter Estimation for AR and ARMA)</span></p>

**For a pure AR($p$) model, parameter estimation is equivalent to a linear regression problem.**

$$
y = \begin{pmatrix} X_T \\ X_{T-1} \\ \vdots \\ X_{p+1} \end{pmatrix} \quad
X = \begin{pmatrix}
1 & X_{T-1} & \dots & X_{T-p} \\
1 & X_{T-2} & \dots & X_{T-p-1} \\
\vdots & \vdots & \ddots & \vdots \\
1 & X_p & \dots & X_1
\end{pmatrix}
$$

* The parameters can then be estimated using ordinary least squares. 
* **For ARMA models with an MA component, estimation is more complex and typically requires numerical optimization methods like maximum likelihood estimation.**

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Goals of ARMA Modeling)</span></p>

Once an ARMA model is fitted, it can be used for:

  * **Goodness-of-Fit:** Assess how well the model describes the temporal structure of the process.
  * **Stationarity Analysis:** Determine if the process properties are stable over time.
  * **Memory and Dependence:** The orders $p$ and $q$ define a "memory horizon."
  * **Hypothesis Testing:** Test the significance of specific coefficients (e.g., $H_0: a_i = 0$).
  * **Forecasting:** Predict future values of the time series.
  * **Control:** Understand how to steer the system towards a desired state.

</div>
