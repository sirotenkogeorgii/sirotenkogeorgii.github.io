## Nonlinear Dynamical Systems

This chapter introduces the fundamental concepts of nonlinear dynamical systems, moving beyond the linear framework of VAR models to explore more complex behaviors like fixed points, cycles, and chaos.

### Motivation: From Linear to Nonlinear Systems

A VAR(1) model, 

$$X_t = c + AX_{t-1}$$

is a **Linear Dynamical System (LDS)**. We can generalize this by replacing the linear function with a nonlinear function $F(\cdot)$, such as a recurrent neural network (RNN).

$$X_t = F(X_{t-1})$$

This defines a nonlinear dynamical system. Understanding the behavior of simple nonlinear systems provides the foundation for analyzing more complex models.

### Analysis of 1D Systems

Let's start with a simple first-order nonlinear difference equation: $x_{t+1} = f(x_t)$.

#### Fixed Points

A central concept is the fixed point (FP), a state where **the system remains unchanged over time**.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Fixed point of map)</span></p>

A point $x^{\ast}$ is a **fixed point of the map** $f$ if it satisfies the equation:
  
$$x^{\ast} = f(x^{\ast})$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Fixed point of Linear AR(1))</span></p>

For $f(x) = \alpha x + c$, the fixed point equation is 

$$x^{\ast} = \alpha x^{\ast} + c$$

Solving for $x^{\ast}: x^{\ast}(1-\alpha) = c$, which gives 

$$x^{\ast} = \dfrac{c}{1-\alpha}$$ 

provided $\alpha \neq 1$.

</div>

#### Stability of Fixed Points

The behavior of the system near a fixed point determines its stability.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Stable FP = Attractor, Unstable FP = Repeller)</span></p>

* A **stable fixed point**, or attractor, is a point such that trajectories starting in its vicinity converge to it:
  * $x^{\ast}$ is an attractor if there exists a set $C$, called the basin of attraction, such that for any initial condition $x_0 \in C$, $\lim_{t \to \infty} x_t = x^{\ast}$
* An **unstable fixed point**, or repeller, is a point from which nearby trajectories diverge.
  * $x^{\ast}$ is a repeller if there exists a set $C$, such that for any initial condition $x_0 \in C$, $\lim_{t \to -\infty} x_t = x^{\ast}$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Stability of FP of Linear System)</span></p>

For the linear system $x_{t+1} = \alpha x_t + c$, the fixed point $x^{\ast}$ is:

- **Stable** if $\lvert\alpha\rvert < 1$. Trajectories converge to $x^{\ast}$.
- **Unstable** if $\lvert\alpha\rvert > 1$. Trajectories diverge from $x^{\ast}$.
- **Neutrally** stable if $\lvert\alpha\rvert = 1$.

</div>

#### Cycles

A system may not settle on a single point but may instead visit a set of points periodically.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($p$-cycle)</span></p>

A set of $p$ distinct points $\lbrace x_1^{\ast}, \dots, x_p^{\ast}\rbrace$ that the system visits in sequence: 

$$f(x_1^{\ast}) = x_2^{\ast}, \dots, f(x_p^{\ast}) = x_1^{\ast}$$

A point on a **$p$-cycle is a fixed point of the $p$-th iterated map** $f^p(\cdot)$. That is, 

$$x^{\ast} = f^p(x^{\ast})$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(2-cycle and neutrally stable cycle)</span></p>

For the map 

$$x_{t+1} = -x_t + c$$ 

if we start at $x_0$, then 

$$x_1 = -x_0+c$$

$$x_2 = -x_1+c = -(-x_0+c)+c = x_0$$

Every point is part of a 2-cycle. This is a **neutrally stable cycle**.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(1D Linear System)</span></p>

1. **Convergence** to a solitary, stable fixed point (attractor) for $\lvert\alpha\rvert < 1$.
2. **Divergence** from an isolated, unstable fixed point for $\lvert\alpha\rvert > 1$.
3. An infinite set of **neutrally stable fixed points** (e.g., a line) if $\alpha=1$, $c=0$.
4. **No fixed point or cycle** (linear drift) if $\alpha=1$, $c \neq 0$.
5. An infinite set of **neutrally stable cycles** if $\alpha=-1$.

</div>

<div class="pmf-grid">
  <figure>
    <img src="{{ '/assets/images/notes/model-based-time-series-analysis/lin_map_case1.png' | relative_url }}" alt="Binomial PMF" loading="lazy">
    <!-- <figcaption>Chi-squared distribution for different $k$.</figcaption> -->
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/model-based-time-series-analysis/lin_map_case2.png' | relative_url }}" alt="Binomial PMF" loading="lazy">
    <!-- <figcaption>Chi-squared distribution for different $k$.</figcaption> -->
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/model-based-time-series-analysis/lin_map_case3.png' | relative_url }}" alt="Binomial PMF" loading="lazy">
    <!-- <figcaption>Chi-squared distribution for different $k$.</figcaption> -->
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/model-based-time-series-analysis/lin_map_case4.png' | relative_url }}" alt="Binomial PMF" loading="lazy">
    <!-- <figcaption>Chi-squared distribution for different $k$.</figcaption> -->
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/model-based-time-series-analysis/lin_map_case5.png' | relative_url }}" alt="Binomial PMF" loading="lazy">
    <!-- <figcaption>Chi-squared distribution for different $k$.</figcaption> -->
  </figure>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Multivariate Extension (Linear Case))</span></p>

For a **multivariate linear system** 

$$X_t = AX_{t-1} + c$$

where $X_t \in \mathbb{R}^N$:

- **Fixed Point:** 
  
  $$X^{\ast} = AX^{\ast} + c$$
  
  $$(I-A)X^{\ast} = c$$
  
  $$X^{\ast} = (I-A)^{-1}c$$ 
  
  provided $(I-A)$ is invertible.
- **Stability:** The stability of $X^{\ast}$ is determined by the eigenvalues of the Jacobian matrix, which is simply $A$.
  1. $X^{\ast}$ is a **stable fixed point** if $\max_i(\lvert\lambda_i(A)\rvert) < 1$.
  2. $X^{\ast}$ is **unstable** if $\max_i(\lvert\lambda_i(A)\rvert) > 1$.
  3. $X^{\ast}$ is **neutrally stable** if $\max_i(\lvert\lambda_i(A)\rvert) = 1$.

</div>


### The Logistic Map: A Case Study in Nonlinearity


<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Logistic Map)</span></p>

The **logistic map** is a simple, archetypal example of a nonlinear system that exhibits complex behavior, including chaos.

- **Equation:**
  
  $$x_{t+1} = f(x_t) = \alpha x_t (1 - x_t)$$

- **Constraints:** We consider initial conditions $x_0 \in [0, 1]$ and the parameter $\alpha \in [0, 4]$. These constraints ensure that if $x_t$ is in $[0, 1]$, then $x_{t+1}$ will also be in $[0, 1]$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Fixed Points of the Logistic Map)</span></p>

The logistic map has two fixed points:

1. $x_1^{\ast} = 0$
2. $x_2^{\ast} = \dfrac{\alpha - 1}{\alpha}$

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    $$x^{\ast} = f(x^{\ast}) = \alpha x^{\ast} (1 - x^{\ast})$$
    $$x^{\ast} - \alpha x^{\ast} + \alpha (x^{\ast})^2 = 0 \implies x^{\ast}(1 - \alpha + \alpha x^{\ast}) = 0$$
    <p>This gives two fixed points:</p>
    <ol>
      <li>FP1: $x_1^{\ast} = 0$.</li>
      <li>FP2: $1 - \alpha + \alpha x^{\ast} = 0 \implies x_2^{\ast} = \dfrac{\alpha - 1}{\alpha}$. This fixed point is only physically relevant (i.e., in our state space $[0,1]$) when $\alpha \ge 1$.</li>
    </ol>
  </details>
</div>

#### Formal Stability Analysis via Linearization

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Stability Analysis</span><span class="math-callout__name">(Univariate Linearization)</span></p>

For a univariate system $x_t = f(x_{t-1})$ with fixed point $x^{\ast} = f(x^{\ast})$, the stability is determined by linearizing around $x^{\ast}$. The **evolution of a small perturbation** $\varepsilon_t$ is **governed by the derivative** $f'$ of $f$ evaluated at $x^{\ast}$.

$$\varepsilon_{t+1} \approx f'(x^{\ast}) \varepsilon_t$$

The fixed point $x^{\ast}$ is **stable if absolute value of $f'(x^{\ast})$ is less than 1**: 

$$\lvert f'(x^{\ast})\rvert < 1$$

</div>

<div class="accordion">
  <details>
    <summary>proof</summary>
    <p>To analyze the stability of a fixed point $x^{\ast}$ for a general nonlinear map $f(x)$, we consider a small perturbation $\varepsilon_t$ around the fixed point:</p>
    $$x_t = x^{\ast} + \varepsilon_t$$
    $$x_{t+1} = x^{\ast} + \varepsilon_{t+1} = f(x^{\ast} + \varepsilon_t)$$
    <p>Using a first-order Taylor expansion of $f(x)$ around $x^{\ast}$:</p>
    $$f(x^{\ast} + \varepsilon_t) \approx f(x^{\ast}) + f'(x^{\ast}) \cdot \varepsilon_t$$
    <p>Since $f(x^{\ast}) = x^{\ast}$, this simplifies to:</p>
    $$x^{\ast} + \varepsilon_{t+1} \approx x^{\ast} + f'(x^{\ast}) \cdot \varepsilon_t \implies \varepsilon_{t+1} \approx f'(x^{\ast}) \cdot \varepsilon_t$$
    <p>This is a linear difference equation for the perturbation $\varepsilon_t$. The perturbation will decay to zero (i.e., the FP is stable) if $\lvert f'(x^{\ast})\rvert < 1$.</p>
  </details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Stability Analysis of Logistic Map)</span></p>

- **Applying to the Logistic Map:** The derivative is $f'(x) = \alpha(1-2x)$.
  1. At FP1 $x_1^{\ast}=0$: 
   
    $$f'(0) = \alpha(1-0) = \alpha$$

    - The fixed point at $0$ is stable if $\lvert f'(0)\rvert < 1$, which means $0 \le \alpha < 1$.
  2. At FP2 $x_2^{\ast} = (\alpha-1)/\alpha$: 
 
    $$f'(x_2^{\ast}) = \alpha\left(1 - 2\frac{\alpha-1}{\alpha}\right) = \alpha\left(\frac{\alpha - 2\alpha + 2}{\alpha}\right) = 2-\alpha$$
    
    - This fixed point is stable if $\lvert f'(x_2^{\ast})\rvert = \lvert 2-\alpha\rvert < 1$. This inequality holds for $1 < \alpha < 3$.

- **Stability Summary:**
  - For $0 \le \alpha < 1$: One stable FP at $x^{\ast}=0$.
  - For $1 < \alpha < 3$: The FP at $x^{\ast}=0$ becomes unstable, and a new stable FP appears at $x^{\ast}=(\alpha-1)/\alpha$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Stability Analysis</span><span class="math-callout__name">(Multivariate Linearization)</span></p>

For a multivariate system $X_t = F(X_{t-1})$ with fixed point $X^{\ast} = F(X^{\ast})$, the stability is determined by linearizing around $X^{\ast}$. The **evolution of a small perturbation** $\mathcal{E}_t$ is **governed by the Jacobian matrix** $J$ of $F$ evaluated at $X^{\ast}$.

$$\mathcal{E}_{t+1} \approx J(X^{\ast}) \mathcal{E}_t$$

The fixed point $X^{\ast}$ is **stable if all eigenvalues of the Jacobian matrix have a modulus less than 1**: 

$$\max_i(\lvert\lambda_i(J(X^{\ast}))\rvert) < 1$$

</div>

<div class="pmf-grid">
  <figure>
    <img src="{{ '/assets/images/notes/model-based-time-series-analysis/nonlin_map_case1.png' | relative_url }}" alt="Binomial PMF" loading="lazy">
    <!-- <figcaption>Chi-squared distribution for different $k$.</figcaption> -->
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/model-based-time-series-analysis/nonlin_map_case2.png' | relative_url }}" alt="Binomial PMF" loading="lazy">
    <!-- <figcaption>Chi-squared distribution for different $k$.</figcaption> -->
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/model-based-time-series-analysis/nonlin_map_case3.png' | relative_url }}" alt="Binomial PMF" loading="lazy">
    <!-- <figcaption>Chi-squared distribution for different $k$.</figcaption> -->
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/model-based-time-series-analysis/nonlin_map_case4.png' | relative_url }}" alt="Binomial PMF" loading="lazy">
    <!-- <figcaption>Chi-squared distribution for different $k$.</figcaption> -->
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/model-based-time-series-analysis/nonlin_map_case5.png' | relative_url }}" alt="Binomial PMF" loading="lazy">
    <!-- <figcaption>Chi-squared distribution for different $k$.</figcaption> -->
  </figure>
</div>

### Bifurcation and Chaos

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Observation</span><span class="math-callout__name">(Bifurcation, Period-Doubling and Chaos)</span></p>

As the parameter $\alpha$ increases beyond $3$ in the logistic map, the system's behavior undergoes a series of qualitative changes known as **bifurcations**.

- **Period-Doubling:** At $\alpha=3$, the fixed point $x_2^{\ast}$ becomes unstable, and a stable 2-cycle emerges. As $\alpha$ increases further, this 2-cycle becomes unstable and bifurcates into a stable 4-cycle, then an 8-cycle, and so on. This cascade is known as the "period-doubling route to chaos."
- **Chaos:** For larger values of $\alpha$ (e.g., $\alpha \gtrsim 3.57$), the system's behavior becomes chaotic.
  - The trajectory is aperiodic and appears irregular or random, but it is still **fully deterministic**.
  - A key feature is sensitivity to initial conditions: two trajectories starting arbitrarily close to each other will diverge exponentially fast.

</div>

<div class="pmf-grid">
  <figure>
    <img src="{{ '/assets/images/notes/model-based-time-series-analysis/bifurcation_diagram.png' | relative_url }}" alt="Binomial PMF" loading="lazy">
    <!-- <figcaption>Chi-squared distribution for different $k$.</figcaption> -->
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/model-based-time-series-analysis/bifurcation_graph.png' | relative_url }}" alt="Binomial PMF" loading="lazy">
    <!-- <figcaption>Chi-squared distribution for different $k$.</figcaption> -->
  </figure>
</div>

#### Chaotic Attractors and the Lyapunov Exponent

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Chaotic Attractor)</span></p>

A set $A$ is a **chaotic attractor** if trajectories starting within its basin of attraction converge to $A$, and within $A$, the dynamics are chaotic and sensitive to initial conditions.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Lyapunov Exponent)</span></p>

**Lyapunov Exponent** $\lambda$ quantifies the rate of separation of infinitesimally close trajectories. For a 1D map $f(x)$, it is defined as:
  
$$\lambda(x_0) = \lim_{T \to \infty} \frac{1}{T} \sum_{t=0}^{T-1} \ln\lvert f'(x_t)\rvert$$

- $\lambda > 0$: **Exponential divergence**, a signature of chaos.
- $\lambda < 0$: **Exponential convergence**, corresponding to a stable fixed point or cycle.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Implications for Prediction and Modeling)</span></p>

The existence of chaotic dynamics has profound implications for time series modeling:

1. **Prediction Horizon:** Sensitivity to initial conditions makes long-term prediction fundamentally impossible, as any tiny error in measuring the current state will be exponentially amplified.
2. **Chaos vs. Noise:** It can be extremely difficult to distinguish between a deterministic chaotic process and a stochastic (noisy) process based on observed data alone.
3. **Loss Functions:** Traditional loss functions like Mean Squared Error (MSE) may be problematic for evaluating models of chaotic systems, as even a perfect model will produce trajectories that diverge from the data due to initial condition uncertainty.
4. **Parameter Estimation:** The loss landscapes for models of chaotic systems can be highly non-convex and irregular, making optimization and parameter estimation very challenging.

</div>
