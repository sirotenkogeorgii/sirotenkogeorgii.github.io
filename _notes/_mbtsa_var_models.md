## Vector Autoregressive (VAR) Models

### Introduction to Multivariate Time Series

In contrast to univariate analysis, multivariate time series analysis considers datasets where multiple variables are recorded simultaneously over time.

- **Multivariate Time Series Data:** A vector $X_t$ representing observations at time $t$. $X_t = (X_{1t}, \dots, X_{Nt})^\top \in \mathbb{R}^N$. Here, $N$ is the number of simultaneously recorded variables.
- **New Phenomena of Interest:** The primary advantage of the multivariate approach is the ability to model interactions between time series. A key phenomenon is the cross-correlation between the different component series.

<figure>
  <img src="{{ '/assets/images/notes/model-based-time-series-analysis/var3.png' | relative_url }}" alt="Newton–Raphson iteration animation" loading="lazy">
  <figcaption>VAR(3)</figcaption>
</figure>

### The VAR($p$) Model Architecture

The Vector Autoregressive (VAR) model is a natural extension of the univariate autoregressive (AR) model to multivariate time series. A VAR($p$) model expresses each variable as a linear function of its own past values, the past values of all other variables in the system, and a random error term.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(VAR($p$) Model)</span></p>

The general form of a **VAR($p$) model** is:

$$X_t = c + \sum_{i=1}^{p} A_i X_{t-i} + \varepsilon_t$$

- **Intercept:** $c \in \mathbb{R}^N$ is the intercept vector.
- **Coefficient Matrices:** $A_i \in \mathbb{R}^{N \times N}$ are the coefficient matrices for each lag $i=1, \dots, p$.
- **Error Term:** $\varepsilon_t$ is a vector of white noise error terms, typically assumed to be multivariate normal: $\varepsilon_t \sim \mathcal{N}(0, \Sigma_\varepsilon)$.

The **covariance matrix of the error** term, $\Sigma_\varepsilon$, is given by:

$$
\Sigma_\varepsilon = \mathbb{E}[\varepsilon_t \varepsilon_t^\top] =
\begin{pmatrix}
  \mathbb{E}[\varepsilon_{1t}^2] & \mathbb{E}[\varepsilon_{1t}\varepsilon_{2t}] & \dots \\
  \vdots & \ddots & \vdots \\
  \mathbb{E}[\varepsilon_{Nt}\varepsilon_{1t}] & \dots & \mathbb{E}[\varepsilon_{Nt}^2]
\end{pmatrix}
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Empirical Covariance)</span></p>

For the VAR($p$) process, the **empirical covariance** based on the path of the length $T$ is 

$$\hat{\Sigma_\varepsilon} = \frac{1}{T - (p + 1)} \sum_{t=p+2}^T \epsilon_t \epsilon_t^\top$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Off-diagonal elements of $\Sigma_\varepsilon$)</span></p>

Crucially, the off-diagonal elements of $\Sigma_\varepsilon$ are allowed to be non-zero, meaning the contemporaneous error terms for different variables can be correlated.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Structure of Coefficient Matrices)</span></p>

Each coefficient matrix $A_i$ captures the influence of variables at lag $i$ on the current state of the system.

$$
A_i = \begin{pmatrix}
a_{11}^{(i)} & \dots & a_{1N}^{(i)} \\
\vdots & \ddots & \vdots \\
a_{N1}^{(i)} & \dots & a_{NN}^{(i)}
\end{pmatrix}
$$

- **Diagonal Entries ($a_{jj}^{(i)}$):** 
  * These entries relate a variable to its own past. 
  * They capture the internal time constants and autoregressive properties of each individual series.
- **Off-Diagonal Entries ($a_{jk}^{(i)}$ for $j \neq k$):** 
  * These entries quantify how the past of variable $k$ influences the present of variable $j$. 
  * They are the key to understanding the cross-series dynamics and interactions. 
  * This is the basis for concepts like Granger causality.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Equivalence and Companion Form)</span></p>

1. Any **scalar AR($p$) process** can be written as a **$p$-variate VAR(1) process**.
2. Any **VAR($p$) process in $K$ variables** can be written as a **$Kp$-variate VAR(1) process**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Significance of the equivalence)</span></p>

* A significant theoretical result is that any VAR($p$) process can be rewritten as a VAR(1) process. **This is extremely useful for analysis, particularly for assessing model stability.**
* This transformation allows us to study the stability and properties of a high-order model by analyzing a **single, larger coefficient matrix corresponding to the VAR(1) representation.**

</div>

### Stationarity of VAR Processes

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Stationarity $\implies$ spectral radius of $A$ is less than 1)</span></p>

* The stability of a VAR process is determined by the properties of its coefficient matrices. 
* For a VAR(1) process, the condition for stationarity is based on the eigenvalues of the coefficient matrix.
* For the VAR(1) process $X_t = c + AX_{t-1} + \varepsilon_t$, the necessary and sufficient condition for stationarity

$$\max_i \lvert \lambda_i(A) \rvert < 1$$

where $\lambda_i(A)$ are the eigenvalues of $A$.

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <h4>Proof Sketch for Stationarity</h4>
    <ol>
      <li><strong>Iterative Substitution:</strong> Consider the process without the intercept and noise terms for simplicity: $X_t = AX_{t-1}$. By iterating backwards, we can express $X_t$ in terms of an initial state $X_0$:</li>
    </ol>
    $$
    X_t = A X_{t-1} = A (A X_{t-2}) = A^2 X_{t-2} = \dots = A^t X_0
    $$
    <ol>
      <li><strong>Eigendecomposition:</strong> We can decompose the matrix $A$ into its eigenvalues and eigenvectors: $A = V \Lambda V^{-1}$ where $\Lambda$ is a diagonal matrix containing the eigenvalues $\lambda_i$, and $V$ is the matrix of corresponding eigenvectors.</li>
      <li><strong>Power of $A$:</strong> Using the eigendecomposition, the $t$-th power of $A$ is:</li>
    </ol>
    $$
    A^t = (V \Lambda V^{-1})^t = (V \Lambda V^{-1})(V \Lambda V^{-1}) \dots = V \Lambda^t V^{-1}
    $$
    <ol>
      <li><strong>System Evolution:</strong> Substituting this back into the expression for $X_t$:</li>
    </ol>
    $$
    X_t = V \Lambda^t V^{-1} X_0
    $$
    <ol>
      <li><strong>Condition for Stability:</strong> The system is stable (i.e., stationary) if $X_t \to 0$ as $t \to \infty$. This requires that $A^t \to 0$. This, in turn, depends on the behavior of $\Lambda^t$, which is a diagonal matrix with entries $\lambda_i^t$.</li>
    </ol>
    <ul>
      <li>If $\max_i(\lvert\lambda_i\rvert) < 1$, then all $\lambda_i^t \to 0$ as $t \to \infty$. Consequently, $\Lambda^t \to 0$, $A^t \to 0$, and the process is stable and stationary.</li>
      <li>If $\max_i(\lvert\lambda_i\rvert) > 1$, at least one eigenvalue has a modulus greater than 1. Its corresponding term $\lambda_i^t$ will grow exponentially, causing $X_t$ to explode along the direction of the corresponding eigenvector. The process is non-stationary (divergent).</li>
      <li>If $\max_i(\lvert\lambda_i\rvert) = 1$, the system is marginally stable. This can lead to behaviors like a random walk.</li>
    </ul>
  </details>
</div>

### Parameter Estimation

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Parameter Estimation for VAR)</span></p>

The parameters of a VAR($p$) model $(c, A_1, \dots, A_p, \Sigma_\varepsilon)$ can be estimated using Maximum Likelihood Estimation (MLE).

Given a VAR($p$) model:

$$X_t = c + \sum_{i=1}^{p} A_i X_{t-i} + \varepsilon_t$$

The log-likelihood for a sequence of observations $X_1, \dots, X_T$ is then:

$$\ell(\theta) = \sum_{t=p+1}^{T} \log p(X_t \mid X_{t-1}, \dots, X_{t-p}, \theta)$$

Maximizing this log-likelihood function provides the parameter estimates.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Gaussian errors)</span></p>

In the case of Gaussian errors, MLI is equivalent to multivariate least squares.

Given a VAR($p$) model:

$$X_t = c + \sum_{i=1}^{p} A_i X_{t-i} + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, \Sigma_\varepsilon)$$

$$p(X_t \mid X_{t-1}, \dots, X_{t-p}, \theta) = \mathcal{N}\left(c + \sum_{i=1}^{p} A_i X_{t-i}, \Sigma_\varepsilon\right)$$

where $\theta$ represents all model parameters.

The log-likelihood for a sequence of observations $X_1, \dots, X_T$ is then:

$$\ell(\theta) = \sum_{t=p+1}^{T} \log p(X_t \mid X_{t-1}, \dots, X_{t-p}, \theta)$$

For Gaussian errors, this can be framed as a multivariate linear regression problem.

</div>
