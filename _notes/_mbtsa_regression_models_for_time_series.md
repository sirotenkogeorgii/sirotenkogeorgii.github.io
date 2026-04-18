## Regression Models for Time Series

### Linear Regression

#### Model Components

A complete statistical model generally consists of four key components:

1.  **Model Architecture:** The mathematical form of the relationship between variables.
2.  **Loss Function:** A function that quantifies the error between model predictions and actual data.
3.  **Training Algorithm:** An optimization procedure to find the parameters that minimize the loss function.
4.  **Data:** The observations used to train and evaluate the model.

#### Model Architecture

The simple linear regression model assumes a linear relationship between the predictors and the response, corrupted by additive Gaussian noise.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Univariate Linear Regression Model)</span></p>

The response variable $y_t$ is modeled as a linear combination of the predictor variables $x_t$ plus an error term $\epsilon_t$.

$$y_t = x_t^\top \beta + \epsilon_t$$

where:

* $y_t \in \mathbb{R}$ is the response variable at time $t$.
* $x_t \in \mathbb{R}^{p+1}$ is the vector of predictor variables at time $t$ (including a constant term for the intercept).
* $\beta \in \mathbb{R}^{p+1}$ is the vector of model parameters or coefficients.
* $\epsilon_t$ is the error term, assumed to be independent and identically distributed (i.i.d.) Gaussian noise: $\epsilon_t \sim \mathcal{N}(0, \sigma^2)$.

This implies that the conditional distribution of the response variable is also Gaussian:

$$y_t \mid x_t, \beta, \sigma^2 \sim \mathcal{N}(x_t^T \beta, \sigma^2)$$

For the entire dataset, we can express the model in a vectorized form:

$$Y = X\beta + E$$

where:

* $Y = (y_1, \dots, y_T)^\top$ is the $T \times 1$ vector of responses.
* $X$ is the $T \times (p+1)$ design matrix, where each row is $x_t^\top$.
* $\beta = (\beta_0, \dots, \beta_p)^T$ is the $(p+1) \times 1$ parameter vector.
* $E = (\epsilon_1, \dots, \epsilon_T)^T$ is the $T \times 1$ vector of errors.
</div>

#### Loss Function

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Least Squares Error or Sum of Squared Errors)</span></p>

$$\text{LSE}(\beta) = \sum_{t=1}^T (y_t - \hat{y}_t)^2 = \sum_{t=1}^T (y_t - x_t^\top \beta)^2$$

In vector notation, this is:

$$\text{LSE}(\beta) = (y - X\beta)^\top(y - X\beta)$$

</div>

The objective is to find the parameter vector $\beta$ that minimizes it.

#### Training Algorithm

The most common method for estimating the parameters $\beta$ in a linear regression model is the **method of ordinary least squares (LSE)**. This involves minimizing the sum of the squared differences between the observed responses and the responses predicted by the model. This is achieved by taking the derivative of the loss function with respect to $\beta$ and setting it to zero.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(OLS Estimator)</span></p>

The solution for $\hat{\beta}$, known as the **ordinary least squares (OLS) estimator**, is:

$$\hat{\beta} = (X^\top X)^{-1}X^\top y$$

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>Expanding the LSE expression:</p>
    $$\text{LSE}(\beta) = y^\top y - y^\top X\beta - \beta^\top X^\top y + \beta^\top X^\top X\beta$$
    <p>Since $y^\top X\beta$ is a scalar, it is equal to its transpose $(\beta^\top X^\top y)$. Therefore:</p>
    $$\text{LSE}(\beta) = y^\top y - 2\beta^\top X^\top y + \beta^\top X^\top X\beta$$
    <p>Now, we take the derivative with respect to the vector $\beta$:</p>
    $$\frac{\partial \text{LSE}(\beta)}{\partial \beta} = \frac{\partial}{\partial \beta} (y^\top y - 2\beta^\top X^\top y + \beta^\top X^\top X\beta)$$
    <p>The derivative is:</p>
    $$\frac{\partial \text{LSE}(\beta)}{\partial \beta} = -2X^\top y + (X^\top X + (X^\top X)^\top)\beta$$
    <p>Since $X^\top X$ is a symmetric matrix, $(X^\top X)^\top = X^\top X$. The derivative simplifies to:</p>
    $$\frac{\partial \text{LSE}(\beta)}{\partial \beta} = -2X^\top y + 2X^\top X\beta$$
    <p>Setting the derivative to zero to find the minimum:</p>
    $$-2X^\top y + 2X^\top X\hat{\beta} = 0$$
    $$2X^\top X\hat{\beta} = 2X^\top y$$
    $$X^\top X\hat{\beta} = X^\top y$$
    $$\hat{\beta} = (X^\top X)^{-1}X^\top y \qquad\square$$
  </details>
</div>

#### Parameter Estimation: Maximum Likelihood

An alternative framework for parameter estimation is **Maximum Likelihood Estimation (MLE)**. This approach finds the parameter values that maximize the likelihood of observing the given data.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Coincidence of MLE and LSE solutions)</span></p>

For the linear regression model with the assumption of i.i.d. Gaussian errors, the Least Squares Estimator (LSE) and the Maximum Likelihood Estimator (MLE) for the regression coefficients $\beta$ are identical:

$$\hat{\beta}_{\text{LS}} = \hat{\beta}_{\text{MLE}}$$

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>Given the model assumption $y_t \sim \mathcal{N}(x_t^\top \beta, \sigma^2)$, the likelihood of observing the entire dataset $D$ is the product of the probability densities for each observation:</p>
    $$
    \mathcal{L}(\beta, \sigma^2) = p(Y \mid X, \beta, \sigma^2) = \prod_{t=1}^T p(y_t \mid x_t, \beta, \sigma^2)
    $$
    $$
    \mathcal{L}(\beta, \sigma^2) = \prod_{t=1}^T \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_t - x_t^\top \beta)^2}{2\sigma^2}\right)
    $$
    <p>It is often more convenient to work with the <strong>log-likelihood function</strong>:</p>
    $$
    l(\beta, \sigma^2) = \log \mathcal{L}(\beta, \sigma^2) = \sum_{t=1}^T \left[ -\frac{1}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}(y_t - x_t^\top \beta)^2 \right]
    $$
    <p>To find the MLE for $\beta$, we maximize $l(\beta, \sigma^2)$ with respect to $\beta$. Notice that the terms involving $\sigma^2$ and $2\pi$ are constant with respect to $\beta$. Therefore, maximizing the log-likelihood is equivalent to minimizing the sum of squared errors:</p>
    $$
    \arg\max_\beta l(\beta, \sigma^2) \equiv \arg\min_\beta \sum_{t=1}^T (y_t - x_t^\top \beta)^2
    $$
    $$\hat{\beta}_{\text{LS}} = \hat{\beta}_{\text{MLE}}$$
  </details>
</div>

#### Model Diagnostics

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Examining the residuals, Assumptions of LRM on errors)</span></p>

After fitting a model, it is crucial to assess its validity by examining the **residuals**, which are the differences between the observed and fitted values.

The model decomposes the observed data $x_t$ into a fitted signal $\hat{y}_t$ and a residual component $res_t$:

$$y_t = \hat{y}_t + res_t$$

where $\hat{y}_t = x_t^\top \hat{\beta}$ is the predicted value.

The core assumptions of the linear regression model:
* **linearity, normality of errors**
* **constant variance**
* **independence of errors** 

should be checked by analyzing the residuals.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Common Diagnostic Visualizations)</span></p>

  * **Error Histogram:** A histogram of the residuals $res_t$ can provide a first check on the assumption that the errors are normally distributed.
  * **Q-Q (Quantile-Quantile) Plot:** This is a more rigorous way to check for normality. It plots the sorted quantiles of the residuals against the theoretical quantiles of a standard normal distribution. If the residuals are normally distributed, the points on the Q-Q plot will lie close to a straight diagonal line.
  * **ACF Plot of Residuals:** The Autocorrelation Function (ACF) plot of the residuals is used to check the assumption of independence. For time series data, it is critical to ensure there is no remaining temporal structure (autocorrelation) in the residuals. Significant spikes in the ACF plot suggest that the model has failed to capture the temporal dynamics in the data.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Sources of "Bad Residuals" (Model Misspecification))</span></p>

  * **Wrong assumption on modality:** The errors may not be normally distributed.
  * **Wrong assumption on relationship:** The true relationship between predictors and response may be non-linear.
  * **Missing external predictors:** Important explanatory variables may have been omitted from the model.
  * **Missing temporal dynamics:** The model fails to account for dependencies between observations over time.

</div>

### Handling Non-Linearity: Basis Expansion

One way to address a non-linear relationship between predictors and the response is through **basis expansion**. This technique extends the linear model by including non-linear transformations of the original predictors as additional regressors.

Suppose we have a set of predictors $x_t = (x_{1t}, \dots, x_{pt})^\top$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Basis Expansion Model)</span></p>

The model architecture is extended by introducing a set of $K$ **basis functions**, $\phi_k(\cdot)$:

$$y_t = \beta_0 + \sum_{k=1}^K \beta_k \phi_k(x_t) + \epsilon_t$$

where $\phi_k(\cdot)$ are chosen functions that transform the original predictor vector $x_t$. This model is still linear in the parameters $\beta_k$, so the standard LSE and MLE solutions still apply, but with a new design matrix whose columns are the transformed predictors $\phi_k(x_t)$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Common Choices for Basis Functions)</span></p>

* **Polynomials:** $\phi_k(u_j) = u_{jt}^k$ (e.g., $u_t, u_t^2, u_t^3, \dots$).
* **Interaction Terms:** $\phi(u_i, u_j) = u_{it} \cdot u_{jt}$.
* **Radial Basis Functions:** Functions that depend on the distance from a center point.
* **Fourier Basis:** Sines and cosines to model periodic patterns, e.g., $\sin(\omega t)$ and $\cos(\omega t)$.

</div>

### Multivariate Linear Regression

In many scenarios, we want to predict multiple response variables simultaneously. This leads to **multivariate linear regression**.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Multivariate Linear Regression Model)</span></p>

**Data:** At each time $t=1, \dots, T$:

* **Response vector:** $y_t = (y_{1t}, \dots, y_{qt})^\top \in \mathbb{R}^q$.
* **Predictor vector:** $x_t = (x_{1t}, \dots, x_{pt})^\top \in \mathbb{R}^p$.

**Model:** The model is a direct extension of the univariate case, written in matrix form:

$$Y = XB + E$$

where:

  * $Y$ is the $T \times q$ matrix of response variables.
  * $X$ is the $T \times (p+1)$ design matrix.
  * $B$ is the $(p+1) \times q$ matrix of parameters, where each column corresponds to a response variable.
  * $E$ is the $T \times q$ matrix of errors.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Estimation of LSE/MLE for Multivariate Linear Regression Model)</span></p>

The solution for the parameter matrix $B$ is analogous to the univariate case:

$$
\hat{B} = (X^\top X)^{-1} X^\top Y
$$

This is equivalent to performing $q$ separate univariate linear regressions, one for each response variable.

</div>

### Generalized Linear Models (GLMs)

* Linear regression assumes a normally distributed response variable. **Generalized Linear Models (GLMs)** provide a framework to handle response variables with other distributions (e.g., binary, count data).
* Parameters in a GLM are typically estimated using MLE.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Generalized Linear Models (GLMs))</span></p>

GLMs are composed of three components:

1.  **Random Component:** The response variable $y_t$ follows a probability distribution from the exponential family (e.g., Bernoulli, Poisson, Gamma).
2.  **Systematic Component:** A **linear predictor**, $\eta_t$, is constructed as a linear combination of the predictors:

    $$\eta_t = x_t^\top \beta$$

3.  **Link Function:** A function $g(\cdot)$ that links the expected value of the response, $\mu_t = \mathbb{E}[y_t]$, to the linear predictor:

    $$g(\mu_t) = \eta_t$$

The inverse of the link function, $g^{-1}(\cdot)$, maps the linear predictor back to the mean of the response: $\mu_t = g^{-1}(\eta_t)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(LRM is a spacial case GLM)</span></p>

The standard linear regression model is a special case of a GLM.

  * **Random Component:** $y_t \sim \mathcal{N}(\mu_t, \sigma^2)$.
  * **Systematic Component:** $\eta_t = x_t^\top \beta$.
  * **Link Function:** The identity link, $g(\mu_t) = \mu_t$. Therefore, $\eta_t = \mu_t$, which gives us the familiar $\mathbb{E}[y_t] = x_t^\top \beta$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Logistic Regression)</span></p>

Logistic regression is a GLM used for modeling **binary response variables**.

**Model Architecture:**

  * **Data:** Observed binary response variable $y_t \in \lbrace 0, 1 \rbrace$ for $t=1, \dots, T$, with predictor vector $x_t$.
  * **Random Component:** The response is assumed to follow a Bernoulli distribution:
    
    $$y_t \sim \text{Bernoulli}(\pi_t)$$
    
    where $\pi_t = P(y_t=1 \mid x_t)$ is the "success" probability.
  * **Systematic Component:** The linear predictor is $\eta_t = x_t^\top \beta$.
  * **Link Function:** The **logit** link function is used, which is the natural logarithm of the odds:
    
    $$g(\pi_t) = \log\left(\frac{\pi_t}{1-\pi_t}\right) = \eta_t$$
    
    The inverse link function is the **sigmoid** (or logistic) function, which maps the linear predictor to a probability between 0 and 1:
    
    $$\pi_t = g^{-1}(\eta_t) = \frac{\exp(\eta_t)}{1+\exp(\eta_t)} = \frac{1}{1+\exp(-\eta_t)} = \sigma(\eta_t)$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Maximum Likelihood Estimation for Logistic Regression)</span></p>

The following statements are true for logistic regression:
* Does not have close-form solution (unlike linear regression)
* $\nabla_\beta \text{log-likelihood}(\beta) = \sum_{t=1}^T (y_t - \pi_t) x_t$

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p><strong>Loss Function (Negative Log-Likelihood):</strong> The probability mass function for a single Bernoulli observation is</p>
    $$p(y_t \mid \pi_t) = \pi_t^{y_t}(1-\pi_t)^{1-y_t}$$
    <p>The likelihood for the entire dataset is:</p>
    $$\mathcal{L}(\beta) = \prod_{t=1}^T p(y_t \mid x_t, \beta) = \prod_{t=1}^T \pi_t^{y_t} (1-\pi_t)^{1-y_t}$$
    <p>The log-likelihood is:</p>
    $$l(\beta) = \sum_{t=1}^T \left[ x_t \log(\pi_t) + (1-x_t)\log(1-\pi_t) \right]$$
    <p>Substituting $\pi_t = \sigma(x_t^\top \beta)$, we can express the log-likelihood in a form common to the exponential family:</p>
    $$l(\beta) = \sum_{t=1}^T \left[ y_t (x_t^\top \beta) - \log(1+\exp(x_t^\top \beta)) \right]$$
    <p><strong>Training Algorithm:</strong> We maximize the log-likelihood by taking its derivative with respect to $\beta$ and setting it to zero.</p>
    $$\nabla_\beta l(\beta) = \nabla_\beta \sum_{t=1}^T \left[ y_t \log(\sigma(x_t^\top\beta)) + (1-y_t)\log(1-\sigma(x_t^\top\beta)) \right]$$
    <p>The derivative of the log-likelihood for a single observation with respect to $\beta$ is:</p>
    $$\nabla_\beta l_t(\beta) = \left(y_t - \sigma(x_t^\top\beta)\right)x_t = (y_t - \pi_t)x_t$$
    <p>Summing over all observations gives the full gradient:</p>
    $$\nabla_\beta l(\beta) = \sum_{t=1}^T (y_t - \pi_t) x_t$$
    <p>Unlike in linear regression, setting this equation to zero does not yield a closed-form solution for $\beta$. Therefore, iterative optimization algorithms like <strong>gradient ascent</strong> are used.</p>
    <p>The gradient ascent update rule is:</p>
    $$\beta_{\text{new}} = \beta_{\text{old}} + \alpha \nabla_\beta l(\beta_{\text{old}})$$
    <p>where $\alpha$ is the learning rate. For a single data point (stochastic gradient ascent), the rule is:</p>
    $$\beta_{\text{new}} = \beta_{\text{old}} + \alpha (y_t - \pi_t) x_t$$
  </details>
</div>

### Modeling Complex Data Structures

#### Multimodal Regression

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Multivariate Linear Regression Model)</span></p>

This approach models datasets where each observation consists of multiple types of data, or **modalities**.

**Setup:**

* We have multiple observed modalities, for example, a binary variable $y_{1t} \in \lbrace 0,1 \rbrace$ and a continuous variable $y_{2t} \in \mathbb{R}$.
* These are collected into a response vector $y_t = (y_{1t}, y_{2t})^T$.
* The full dataset is $D = \lbrace(y_t, x_t)\rbrace_{t=1}^T$.

**Model:** We model each modality separately, conditional on the predictors $x_t$. This typically involves different parameter vectors ($\beta_1, \beta_2$) and distributions for each modality.

* Linear predictor for modality 1: $\eta_{1t} = \beta_1^\top x_t$
* Linear predictor for modality 2: $\eta_{2t} = \beta_2^\top x_t$
* Model for modality 1: $y_{1t} \mid x_t \sim \text{Bernoulli}(\pi_t)$, where $\pi_t = \sigma(\eta_{1t})$.
* Model for modality 2: $y_{2t} \mid x_t \sim \mathcal{N}(\mu_t, \sigma^2)$, where $\mu_t = \eta_{2t}$.

**Key assumption:**

* **conditional independence**: the different modalities are independent of each other, given the predictors.

$$p(y_{1t}, y_{2t} \mid x_t) = p(y_{1t} \mid x_t) p(y_{2t} \mid x_t)$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Loss Function (MLE) for Multimodal Regression Above)</span></p>

Due to the conditional independence assumption, the total log-likelihood is the sum of the log-likelihoods for each modality:

$$l(\beta_1, \beta_2, \sigma^2) = \sum_{t=1}^T \log p(y_{1t} \mid x_t) + \sum_{t=1}^T \log p(y_{2t} \mid x_t)$$

$$
l(\beta_1, \beta_2, \sigma^2) = \sum_{t=1}^T [y_{1t}\eta_{1t} - \log(1+\exp(\eta_{1t}))] + \sum_{t=1}^T \left[-\frac{1}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}(y_{2t} - \eta_{2t})^2 \right]
$$

This loss function can be maximized jointly to find the parameters for both models.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(What this calculation actually proves)</span></p>

* If the assumption of conditional independence between modalities is not reasonable, a common approach is to introduce a shared **latent variable** $\epsilon_t$ that **captures common dynamics or unobserved factors influencing all modalities**. 
* Conditional on both the predictors $x_t$ and the latent variable $\epsilon_t$, the modalities are then assumed to be independent.

</div>
