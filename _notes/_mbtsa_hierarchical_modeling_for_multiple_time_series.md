## Hierarchical Modeling for Multiple Time Series

When analyzing data from multiple subjects or independent sources, we often face a modeling choice between two extremes. This section introduces hierarchical modeling as a powerful intermediate approach that balances individual specificity with group-level stability.

<div class="accordion">
  <details>
    <summary>Framework of Bayesian hierarchical model</summary>
    <p>
      Let $y_j$ be an observation and $\theta_j$ a parameter governing the data generating process for $y_j$. Assume further that the parameters $\theta_1, \theta_2, \dots, \theta_j$ are generated exchangeably from a common population, with distribution governed by a hyperparameter $\phi$.
    </p>
    <p>The Bayesian hierarchical model contains the following stages:</p>
    <p><strong>Stage I:</strong> $y_j \mid \theta_j, \phi \sim P(y_j \mid \theta_j, \phi)$</p>
    <p><strong>Stage II:</strong> $\theta_j \mid \phi \sim P(\theta_j \mid \phi)$</p>
    <p><strong>Stage III:</strong> $\phi \sim P(\phi)$</p>
    <p>
      The likelihood, as seen in stage I is $P(y_j \mid \theta_j, \phi)$, with $P(\theta_j, \phi)$ as its prior distribution. Note that the likelihood depends on $\phi$ only through $\theta_j$.
    </p>
    <p>The prior distribution from stage I can be broken down into:</p>
    $$P(\theta_j, \phi) = P(\theta_j \mid \phi)P(\phi) \quad \text{from the definition of conditional probability}$$
    <p>With $\phi$ as its hyperparameter with hyperprior distribution, $P(\phi)$.</p>
    <p>Thus, the posterior distribution is proportional to:</p>
    $$P(\phi, \theta_j \mid y) \propto P(y_j \mid \theta_j, \phi)P(\theta_j, \phi) \quad \text{using Bayes' Theorem}$$
    $$P(\phi, \theta_j \mid y) \propto P(y_j \mid \theta_j)P(\theta_j \mid \phi)P(\phi)$$
  </details>
</div>

<div class="accordion">
  <details>
    <summary>Example calculation of Bayesian hierarchical modeling in two variants</summary>
    <h4>Example calculation</h4>
    <p>
      As an example, a teacher wants to estimate how well a student did on the SAT. The teacher uses the current grade point average (GPA) of the student for an estimate. Their current GPA, denoted by $Y$, has a likelihood given by some probability function with parameter $\theta$, i.e. $Y \mid \theta \sim P(Y \mid \theta)$. This parameter $\theta$ is the SAT score of the student. The SAT score is viewed as a sample coming from a common population distribution indexed by another parameter $\phi$, which is the high school grade of the student (freshman, sophomore, junior or senior).$^{[14]}$ That is, $\theta \mid \phi \sim P(\theta \mid \phi)$. Moreover, the hyperparameter $\phi$ follows its own distribution given by $P(\phi)$, a hyperprior.
    </p>
    <p>These relationships can be used to calculate the likelihood of a specific SAT score relative to a particular GPA:</p>
    $$P(\theta, \phi \mid Y) \propto P(Y \mid \theta, \phi)P(\theta, \phi)$$
    $$P(\theta, \phi \mid Y) \propto P(Y \mid \theta)P(\theta \mid \phi)P(\phi)$$
    <p>
      All information in the problem will be used to solve for the posterior distribution. Instead of solving only using the prior distribution and the likelihood function, using hyperpriors allows a more nuanced distinction of relationships between given variables.$^{[15]}$
    </p>
    <h4>2-stage hierarchical model</h4>
    <p>In general, the joint posterior distribution of interest in 2-stage hierarchical models is:</p>
    $$P(\theta, \phi \mid Y) = \frac{P(Y \mid \theta, \phi)P(\theta, \phi)}{P(Y)} = \frac{P(Y \mid \theta)P(\theta \mid \phi)P(\phi)}{P(Y)}$$
    $$P(\theta, \phi \mid Y) \propto P(Y \mid \theta)P(\theta \mid \phi)P(\phi)^{[15]}$$
    <h4>3-stage hierarchical model</h4>
    <p>For 3-stage hierarchical models, the posterior distribution is given by:</p>
    $$P(\theta, \phi, X \mid Y) = \frac{P(Y \mid \theta)P(\theta \mid \phi)P(\phi \mid X)P(X)}{P(Y)}$$
    $$P(\theta, \phi, X \mid Y) \propto P(Y \mid \theta)P(\theta \mid \phi)P(\phi \mid X)P(X)^{[15]}$$
  </details>
</div>

### Hierarchical Modeling

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Problem</span><span class="math-callout__name">(Multiple Datasets)</span></p>

Consider a scenario with $N$ subjects, where for each subject $i=1, \dots, N$, we have a time series dataset $D_i$.

  * The data for subject $i$ consists of $T_i$ time steps: $D_i = \lbrace x_{i1}, x_{i2}, \dots, x_{iT_i} \rbrace$.
  * For simplicity, we can assume all time series have the same length, $T_i = T$.
  * Each observation may also be associated with inputs: $W_i = \lbrace w_{i1}, w_{i2}, \dots, w_{iT_i} \rbrace$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposed Solutions</span><span class="math-callout__name">(Two Extreme Modeling Strategies)</span></p>

1.  **Separate Models:** Fit a completely independent model with parameters $\theta_i$ to each dataset $D_i$.

      * **Advantage:** This approach is excellent for capturing individual differences, as each model is tailored specifically to its own data.
      * **Disadvantage:** If the number of time steps $T$ is small and the model is complex, the estimates for $\theta_i$ can be noisy and highly prone to overfitting. There is no sharing of information across subjects.

1.  **Fully Pooled Model:** Union all data into a single large dataset: $D_{\text{pooled}} = \lbrace D_1, D_2, \dots, D_N \rbrace$. Fit a single parameter vector $\theta$ to all the data.

      * **Advantage:** This method yields more stable parameter estimates because it leverages the entire data pool.
      * **Disadvantage:** It completely ignores inter-individual differences, assuming all subjects are governed by the exact same process.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Partial pooling)</span></p>

We want to find a middle ground that finds a compromise between these extremes – **partial pooling**. This brings us to Bayesian hierarchical modeling, also known as multilevel modeling.

</div>

#### The Hierarchical Approach

Hierarchical modeling provides a principled compromise between these two extremes.

**Core Idea:** We introduce subject-specific parameters $\theta_1, \theta_2, \dots, \theta_N$, but we assume they are not arbitrary. Instead, they are drawn from a common parent distribution, which is itself described by hyperparameters.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Bayesian hierarchical modeling)</span></p>

**Bayesian hierarchical modeling** makes use of two important concepts in deriving the posterior distribution, namely:

* **Hyperparameters:** parameters of the prior distribution
* **Hyperpriors:** distributions of Hyperparameters

Each subject's parameter $\theta_i$ is drawn from a **parent distribution** parameterized by $\beta$:

$$\theta_i \sim p(\theta \mid \beta)$$

We then place a prior distribution, known as a **hyperprior**, on the hyperparameters $\beta$:

$$\beta \sim p(\beta)$$

**Hierarchical chain of dependencies**:

$$\beta \rightarrow \theta_i \rightarrow D_i \quad \text{for } i=1, \dots, N$$

The **joint distribution** over all data ($D_{1:N}$), parameters ($\theta_{1:N}$), and hyperparameters ($\beta$) is given by:

$$p(\beta, \lbrace\theta_i\rbrace_{i=1}^N, \lbrace D_i\rbrace_{i=1}^N) = p(\beta) \prod_{i=1}^N \left[ p(\theta_i \mid \beta) p(D_i \mid \theta_i) \right]$$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Hierarchical Bayesian inference)</span></p>

The **goal of hierarchical Bayesian inference** is to compute the joint posterior distribution of all subject-level parameters and group-level hyperparameters given the observed data from all subjects.

$$
p(\beta, \lbrace\theta_i\rbrace_{i=1}^N \mid \lbrace D_i\rbrace_{i=1}^N) = \dfrac{p(\lbrace D_i\rbrace_{i=1}^N \mid \lbrace\theta_i\rbrace_{i=1}^N)p(\lbrace\theta_i\rbrace_{i=1}^N \mid \beta)p(\beta)}{p(\lbrace D_i\rbrace_{i=1}^N)}
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Information Sharing in Bayesian hierarchical modeling)</span></p>

* Our primary target is the posterior distribution $p(\theta_1, \dots, \theta_N, \beta \mid D_1, \dots, D_N)$. 
* This structure allows the model to "borrow statistical strength" across subjects. 
* The data from subject $j$ informs the posterior of $\theta_j$, 
* which in turn informs the posterior of the group hyperparameter $\beta$. 
* This updated knowledge about $\beta$ then helps to regularize and improve the estimates for all other subjects' parameters $\theta_i$ where $i \neq j$.

</div>

<div class="gd-grid">
  <figure>
    <img src="{{ '/assets/images/notes/model-based-time-series-analysis/hier_model1.png' | relative_url }}" alt="hier_model1" loading="lazy">
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/model-based-time-series-analysis/hier_model2.jpg' | relative_url }}" alt="hier_model2" loading="lazy">
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/model-based-time-series-analysis/hier_model3.png' | relative_url }}" alt="hier_model3" loading="lazy">
  </figure>
</div>

<div class="accordion">
  <details>
    <summary>Example: Hierarchical Delay Discounting</summary>
    <h3>Example: Hierarchical Delay Discounting</h3>
    <p>Let's consider a concrete example of hierarchical modeling applied to a delay discounting task in cognitive science.</p>
    <h4>Data and Task Structure</h4>
    <p>We have data from $N$ individuals. Subject $i$ performs $T_i$ trials, indexed by $t=1, \dots, T_i$. In each trial, the subject chooses between two options:</p>
    <ol>
      <li>An immediate reward of value $A_{I,it}$.</li>
      <li>A delayed reward of value $A_{D,it}$ available after a delay $D_{it}$.</li>
    </ol>
    <p>The observed data for each trial is a binary choice $y_{it}$:</p>
    $$y_{it} = \begin{cases} 1 & \text{if delayed choice is selected} \\ 0 & \text{if immediate choice is selected} \end{cases}$$
    <p>The full dataset for subject $i$ is $Y_i = \lbrace y_{it}\rbrace_{t=1}^{T_i}$.</p>
    <h4>Model Architecture: Hyperbolic Delay Discounting</h4>
    <p>The subjective value of the options is modeled as follows:</p>
    <ul>
      <li><strong>Value of Immediate Option:</strong> $V_{I,it} = A_{I,it}$</li>
      <li><strong>Value of Delayed Option:</strong> The value of the delayed reward is discounted hyperbolically based on the delay duration.</li>
    </ul>
    $$V_{D,it} = \frac{A_{D,it}}{1 + k_i D_{it}}$$
    <p>Here, $k_i > 0$ is the subject-specific <strong>discount rate</strong>. A higher $k_i$ means the subject devalues future rewards more steeply.</p>
    <p>The choice is then modeled based on the utility difference, $\Delta V_{it} = V_{D,it} - V_{I,it}$. A positive $\Delta V_{it}$ indicates a preference for the delayed option.</p>
    <p>The probability of choosing the delayed option is modeled using a logistic function (i.e., logistic regression):</p>
    $$p_{it} := P(y_{it} = 1 \mid k_i, \beta_i) = \sigma(\beta_i \Delta V_{it}) = \frac{1}{1 + \exp(-\beta_i \Delta V_{it})}$$
    <p>where $\beta_i$ is a subject-specific "softmax" temperature parameter controlling the steepness of the decision boundary. The likelihood for the observed choices of a single individual $i$ follows a Bernoulli distribution:</p>
    $$\mathcal{L}(k_i, \beta_i \mid Y_i) = \prod_{t=1}^{T_i} p_{it}^{y_{it}} (1 - p_{it})^{1 - y_{it}}$$
    <h4>Hierarchical Priors</h4>
    <p>Instead of treating each $k_i$ and $\beta_i$ as independent, we impose a group-level structure. Since $k_i$ and $\beta_i$ must be positive, we model their logarithms as being drawn from a Normal distribution.</p>
    <p><strong>Individual Parameters:</strong> $k = \lbrace k_i\rbrace_{i=1}^N$ and $\beta = \lbrace \beta_i\rbrace_{i=1}^N$.</p>
    $$\log(k_i) \sim \mathcal{N}(\mu_k, \sigma_k^2) \quad \log(\beta_i) \sim \mathcal{N}(\mu_\beta, \sigma_\beta^2)$$
    <p><strong>Group-level Hyperparameters:</strong> The means $(\mu_k, \mu_\beta)$ and variances $(\sigma_k^2, \sigma_\beta^2)$ of the parent distributions.</p>
    <p><strong>Hyperpriors:</strong> We place priors on the hyperparameters themselves.</p>
    <p><em>Priors on Means:</em> $\mu_k \sim \mathcal{N}(\mu_{k0}, \sigma_{k0}^2)$ and $\mu_\beta \sim \mathcal{N}(\mu_{\beta 0}, \sigma_{\beta 0}^2)$.</p>
    <p><em>Priors on Variances:</em> $\sigma_k^2 \sim \text{Inverse-Gamma}(a_k, b_k)$ and $\sigma_\beta^2 \sim \text{Inverse-Gamma}(a_\beta, b_\beta)$.</p>
    <h4>Full Bayesian Formulation</h4>
    <p>The goal is to obtain the full posterior distribution over all parameters and hyperparameters given the data $Y$:</p>
    $$P(k, \beta, \mu_k, \sigma_k^2, \mu_\beta, \sigma_\beta^2 \mid Y)$$
    <p>By Bayes' theorem, this is proportional to the product of the likelihood and the priors:</p>
    $$\propto P(Y \mid k, \beta) P(k, \beta \mid \mu_k, \sigma_k^2, \mu_\beta, \sigma_\beta^2) P(\mu_k, \sigma_k^2, \mu_\beta, \sigma_\beta^2)$$
    <ul>
      <li><strong>Joint Likelihood:</strong> Assuming independence across subjects given their parameters:</li>
    </ul>
    $$P(Y \mid k, \beta) = \prod_{i=1}^N P(Y_i \mid k_i, \beta_i)$$
    <ul>
      <li><strong>Joint Prior on Individual Parameters:</strong></li>
    </ul>
    $$P(k, \beta \mid \dots) = \left[ \prod_{i=1}^N P(\log(k_i) \mid \mu_k, \sigma_k^2) \right] \left[ \prod_{i=1}^N P(\log(\beta_i) \mid \mu_\beta, \sigma_\beta^2) \right]$$
    <ul>
      <li><strong>Joint Hyperprior:</strong> Assuming independence of the hyperparameters:</li>
    </ul>
    $$P(\mu_k, \mu_\beta, \sigma_k^2, \sigma_\beta^2) = P(\mu_k) P(\mu_\beta) P(\sigma_k^2) P(\sigma_\beta^2)$$
    <p>By combining these expressions, we define the full model. Inference is typically performed using sampling techniques like Markov Chain Monte Carlo (MCMC).</p>
  </details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Alternative: Hierarchical Modeling with a Parent Matrix)</span></p>

An alternative parametrization for hierarchical models, particularly useful in high-dimensional settings, involves a shared parent matrix.

  * **Naive Parametrization:** Each subject $i$ has a parameter vector $\theta_i \in \mathbb{R}^p$. The total number of parameters to estimate for the subject level is $p \times N$.
  * **Parent Matrix Parametrization:** We introduce a shared parent matrix $W \in \mathbb{R}^{p \times k}$ (with $k < p$) and individual subject vectors $h_i \in \mathbb{R}^k$. The subject-specific parameter vector $\theta_i$ is then constructed as a linear combination of the columns of $W$:
    
    $$\theta_i = W h_i$$

    The total number of parameters is now $(p \times k) + (k \times N)$, which can be significantly smaller than $p \times N$ if $k$ is chosen well. This acts as a form of dimensionality reduction.
  * **Hierarchical Chain:** The dependency structure becomes: $W, h_i \rightarrow \theta_i \rightarrow D_i$.

</div>
