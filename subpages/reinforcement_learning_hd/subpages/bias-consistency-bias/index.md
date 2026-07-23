
## Estimators and Bias

An **estimator** is a rule for computing an estimate of a given value, often an expectation of some random value(s).

For example, we might estimate the *mean* of a random variable by sampling a value according to its probability distribution.

**Bias** of an estimator is the difference between the expected value of the estimator and the true value being estimated:

$$\text{estimator bias} \stackrel{\mathrm{def}}{=} \mathbb{E}_{\text{estimator}}[\text{estimate}] - \text{true estimated value}.$$

If the bias is zero, we call the estimator **unbiased**; otherwise, we call it **biased**.

As an example, consider estimating $\mathbb{E}\_P[f(x)]$ by generating a single sample $x$ from $P$ and returning $f(x)$. Such an estimate is unbiased, because

$$\mathbb{E}[\text{estimate}] = \mathbb{E}_P[f(x)],$$

which is exactly the true estimated value.

### Consistent Estimator

**Consistency** is a statistical property indicating that as your sample size grows infinitely large, the estimator's value converges to the true population parameter. A consistent estimator gets closer and closer to the exact target value as you gather more data, ultimately guaranteeing near-perfect accuracy with infinite data.

If we have a sequence of estimates, it might also happen that the bias converges to zero.

Consider the well-known sample estimate of variance. Given independent and identically distributed random variables $x_1,\ldots,x_n$, we might estimate the mean and variance as

$$\hat{\mu} = \frac{1}{n}\sum_i x_i, \qquad \hat{\sigma}^2 = \frac{1}{n}\sum_i (x_i-\hat{\mu})^2.$$

Such an estimate is biased, because

$$\mathbb{E}[\hat{\sigma}^2] = \left(1-\frac{1}{n}\right)\sigma^2,$$

but the bias converges to zero with increasing $n$.

Consistency is a statistical property indicating that as your sample size grows infinitely large, the estimator's value converges to the true population parameter. A consistent estimator gets closer and closer to the exact target value as you gather more data, ultimately guaranteeing near-perfect accuracy with infinite data. [1, 2, 3] 
## How Consistency Works
Consistency is an asymptotic property, meaning it only concerns the behavior of the estimator as the number of data points (n) approaches infinity. [1] 
Mathematically, an estimator $\hat{\theta}_n$ (based on n observations) is considered consistent for a true parameter θ if it converges in probability to θ. This is written as:
$\hat{\theta}_n \xrightarrow{p} \theta$ [3] 
This means that for any small, arbitrary margin ε > 0, the probability that the estimate deviates from the true value by more than ε goes to zero as the sample size increases:
$\lim_{n \to \infty} P(\vert{}\hat{\theta}_n - \theta\vert{} > \epsilon) = 0$ [3] 
## Weak vs. Strong Consistency

* Weak Consistency: The estimate converges to the true parameter in probability (as described in the mathematical formula above).
* Strong Consistency: A stricter condition where the sequence of estimates converges almost surely (or with probability 1) to the true parameter. [4] 

## Consistency vs. Unbiasedness
It is common to confuse these two properties, but they represent entirely different concepts: [5] 

* Consistency is a long-term, large-sample property about where the estimate lands as n → ∞.
* Unbiasedness is a property for any given sample size; it means the expected value of the estimator equals the true parameter value (i.e., it is "on target" on average). [1, 6, 7] 

An estimator can be biased but consistent (if the bias shrinks to zero as data grows), or unbiased but inconsistent (if the estimate never narrows in on the true value). [1] 
## Examples of Consistent Estimators
Most commonly used estimators in statistics and econometrics are consistent: [1, 3, 8, 9] 

* Sample Mean: A consistent estimator for the population expected value.
* Sample Variance: A consistent estimator for the population variance.
* Ordinary Least Squares (OLS): Provides consistent estimates for linear regression coefficients (provided the zero conditional mean assumption is met).
* Maximum Likelihood Estimators (MLE): Generally consistent under mild regularity conditions. [8, 10, 11, 12, 13] 

You can check out [Wikipedia](https://en.wikipedia.org/wiki/Consistent_estimator) for a deeper look at the formal mathematical proofs.
If you have a specific statistical model or estimator in mind, I can help you evaluate whether it is consistent or how to mathematically prove its consistency.

An estimator is biased but consistent when its expected value does not exactly equal the true parameter for finite samples, but its bias shrinks to exactly zero as the sample size approaches infinity. [1, 2, 3, 4, 5] 
Here are the most common examples used in statistics:
## 1. Maximum Likelihood Estimator (MLE) for Variance
The standard MLE formula for estimating the population variance σ² from a sample $X_1, X_2, \dots, X_n$ divides the sum of squared deviations by n rather than n-1:
$$\hat{\sigma}^2_{MLE} = \frac{1}{n} \sum_{i=1}^{n} (X_i - \bar{X})^2$$ 

* Why it is biased: For any finite sample size, its expected value is $E[\hat{\sigma}^2_{MLE}] = \frac{n-1}{n}\sigma^2$, which systematically underestimates the true variance.
* Why it is consistent: As n → ∞, the fraction $\frac{n-1}{n} \to 1$, meaning the bias disappears completely in large samples. [6] 

## 2. Standard Deviation Estimators
Even when using the unbiased sample variance (dividing by n-1), taking the square root to find the standard deviation introduces bias due to Jensen's Inequality (since the square root function is concave): [7, 8, 9, 10, 11] 
$$S = \sqrt{\frac{1}{n-1} \sum_{i=1}^{n} (X_i - \bar{X})^2}$$ 

* Why it is biased: $E[S] < \sigma$ for all finite sample sizes.
* Why it is consistent: By the Continuous Mapping Theorem, because the sample variance converges in probability to σ², its square root must converge to σ. [12] 

## 3. Ridge Regression Coefficients
In machine learning and regularization, Ridge regression adds a penalty (λ) to the size of the coefficients to prevent overfitting: [13, 14] 
$$\hat{\beta}_{Ridge} = (X^T X + \lambda I)^{-1} X^T Y$$ 

* Why it is biased: The penalty intentionally pulls the coefficient estimates toward zero, making $E[\hat{\beta}_{Ridge}] \neq \beta$.
* Why it is consistent: If the penalty λ is scaled relative to the sample size such that $\frac{\lambda}{n} \to 0$ as n → ∞, the estimator converges to the true β. [15] 

## 4. Plug-In Estimator for the Odds Ratio
When calculating the odds ratio from a contingency table, the natural approach is to plug in the sample proportions directly.

* Why it is biased: The odds ratio formula involves dividing by random variables in the denominator. Nonlinear transformations of unbiased proportions result in biased estimates for small samples. [16] 
* Why it is consistent: The sample proportions converge exactly to the population probabilities, forcing the sample odds ratio to converge to the true population odds ratio. [17] 

## 5. Ratio Estimators
When estimating the ratio of two population means ($R = \frac{\mu_y}{\mu_x}$) using the ratio of sample means ($\hat{R} = \frac{\bar{Y}}{\bar{X}}$): [18] 

* Why it is biased: The expectation of a quotient is not equal to the quotient of the expectations ($E[\frac{\bar{Y}}{\bar{X}}] \neq \frac{E[\bar{Y}]}{E[\bar{X}]}$).
* Why it is consistent: By Slutsky's Theorem, since $\bar{Y} \xrightarrow{p} \mu_y$ and $\bar{X} \xrightarrow{p} \mu_x$, the ratio $\frac{\bar{Y}}{\bar{X}} \xrightarrow{p} \frac{\mu_y}{\mu_x}$. [19, 20, 21] 

------------------------------
## Summary Comparison Table

| Estimator Name | Finite-Sample Bias | Asymptotic Behavior (n → ∞) |
|---|---|---|
| MLE Variance (σ̂²) | Downward bias (underestimates) | Bias shrinks to 0 |
| Sample Standard Deviation (S) | Downward bias via Jensen's Inequality | Converges perfectly to σ |
| Ridge Regression Coefficients | Shrunk toward zero | Converges to true β if penalty shrinks |
| Sample Odds Ratio | Small-sample distortion | Converges to true population odds |

## ✅ Conclusion
Biased but consistent estimators are highly practical. In many scenarios, accepting a small amount of bias in a small sample allows for a dramatic reduction in variance, leading to a lower overall Mean Squared Error (MSE). [22, 23, 24, 25, 26] 
If you want, let me know:

* Do you need to see the mathematical proof for one of these?
* Are you studying this for a specific context like econometrics or machine learning?
* Would you like an example of an unbiased but inconsistent estimator?

An unbiased but inconsistent estimator is "on target" on average for any sample size, but it never narrows in on the true population parameter as you collect more data. Adding more data points does not reduce its variance or make it more precise. [1, 2, 3] 
Here are the most common examples of unbiased but inconsistent estimators:
## 1. The "First Observation" Estimator
Imagine trying to estimate the mean (μ) of a population. Your estimator is simply the very first data point you collect, completely ignoring all subsequent data. [4] 
$$\hat{\mu} = X_1$$ 

* Why it is unbiased: The expected value of any single random draw from the population is the population mean: $E[X_1] = \mu$.
* Why it is inconsistent: As the sample size (n) grows to infinity, this estimator never changes. It does not utilize the new data, so its variance remains constant at σ² instead of shrinking to zero. [5, 6, 7, 8, 9] 

## 2. The Last k Observations Mean
Instead of using just one data point, you decide to estimate the population mean (μ) by averaging only the last 5 observations in your dataset, regardless of how large the dataset gets. [10, 11] 
$$\hat{\mu} = \frac{X_{n-4} + X_{n-3} + X_{n-2} + X_{n-1} + X_{n}}{5}$$ 

* Why it is unbiased: The expected value of each individual observation is μ, so the average of these 5 observations is exactly μ.
* Why it is inconsistent: Whether your dataset has 100 points or 10,000,000 points, you are still only averaging 5 points. The variance of this estimator is permanently stuck at $\frac{\sigma^2}{5}$ and never converges to zero. [12, 13, 14] 

## 3. Randomly Selected Sample Point
You estimate the population mean by randomly picking exactly one observation from your sample of size n, giving each data point an equal probability (1/n) of being chosen. [15] 

* Why it is unbiased: Because every point has an expected value of μ, any single point chosen at random will also have an expected value of μ. [16] 
* Why it is inconsistent: Even with an infinite sample size, your estimate is still just a single data point. The probability distribution of your estimate remains identical to the population distribution, meaning it never collapses to a single certain value. [17, 18, 19] 

## 4. Omitted Variable Bias in a Specific Setting (Pure Noise Case)
Suppose you are running a regression to find an intercept β₀, and the true model is just a constant plus random noise: $Y_i = \beta_0 + \epsilon_i$. If you mistakenly add a completely irrelevant, independent random noise variable $Z_i$ to your regression: [20, 21] 
$$Y_i = \hat{\beta}_0 + \hat{\beta}_1 Z_i + \epsilon_i$$ 
Under standard OLS assumptions, your estimate β̂₁ will be unbiased because the true relationship is zero ($E[\hat{\beta}_1] = 0$). However, if the variance of $Z_i$ grows proportionally with n due to a flawed experimental design, β̂₁ can remain highly volatile and fail to converge to 0. [22] 
------------------------------
## Summary Comparison Table

| Estimator | Unbiased? | Variance as n → ∞ | Consistent? |
|---|---|---|---|
| Sample Mean (X̄) | Yes | Shrinks to 0 | Yes |
| First Observation (X₁) | Yes | Stays at σ² | No |
| Last 5 Observations Average | Yes | Stays at $\frac{\sigma^2}{5}$ | No |

## 💡 The Takeaway
For an estimator to be consistent, it must pool information from the entire dataset so that the variance of the estimate shrinks toward zero as n grows. Unbiased estimators that throw away data or freeze their sample sizes will always fail the consistency test. [23, 24] 
If you are interested, I can:

* Show you how this looks on a sampling distribution graph
* Help you check if a specific estimator in your homework or project is consistent
* Explain Mean Squared Error (MSE), which combines both bias and consistency/variance [25, 26, 27, 28] 






### Sometimes Biased or Consistent Estimator is Preferable

Also, an unbiased estimator does not necessarily have a small variance—in some cases, it can have a large variance, so a biased estimator with a smaller variance might be preferred.
