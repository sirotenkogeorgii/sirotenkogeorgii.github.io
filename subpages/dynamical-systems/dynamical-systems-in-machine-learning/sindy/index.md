---
title: SINDy
layout: default
noindex: true
---

# SINDy: Sparse Identification of Nonlinear Dynamics

## Core Hypothesis/Problem

The central technical problem is the data-driven discovery of governing nonlinear differential equations for a dynamical system from time-series measurements alone. **The core hypothesis is that for many physical systems, the governing equations are sparse in a high-dimensional space of possible functions**, given an *appropriately selected coordinate system* and *quality training data*, which permits the use of sparse regression to identify the few active terms that define the dynamics. 

## Mathematical & Theoretical Framework

### Continuous-Time Formulation

The objective is to identify the function $f$ for a system of ordinary differential equations of the form:

$$\dot{\mathbf{x}}(t) = \mathbf{f}(\mathbf{x}(t))$$

where:
* $\mathbf{x}(t) = [x_1(t), x_2(t), \dots, x_n(t)]^T \in \mathbb{R}^n$ is the state of the system at time $t$.
* $\mathbf{f}(\mathbf{x}(t))$ is the nonlinear function defining the system's dynamics.

**Data Matrix Construction:** Time-series data is collected at $m$ time instances $(t_1, t_2, \dots, t_m)$ and arranged into two matrices:  

$$
\mathbf{X} = \begin{bmatrix} \mathbf{x}^T(t_1) \\ \mathbf{x}^T(t_2) \\ \vdots \\ \mathbf{x}^T(t_m) \end{bmatrix} = \begin{bmatrix} x_1(t_1) & x_2(t_1) & \dots & x_n(t_1) \\ x_1(t_2) & x_2(t_2) & \dots & x_n(t_2) \\ \vdots & \vdots & \ddots & \vdots \\ x_1(t_m) & x_2(t_m) & \dots & x_n(t_m) \end{bmatrix}
$$   

$$
\dot{\mathbf{X}} = \begin{bmatrix} \dot{\mathbf{x}}^T(t_1) \\ \dot{\mathbf{x}}^T(t_2) \\ \vdots \\ \dot{\mathbf{x}}^T(t_m) \end{bmatrix} = \begin{bmatrix} \dot{x}_1(t_1) & \dot{x}_2(t_1) & \dots & \dot{x}_n(t_1) \\ \dot{x}_1(t_2) & \dot{x}_2(t_2) & \dots & \dot{x}_n(t_2) \\ \vdots & \vdots & \ddots & \vdots \\ \dot{x}_1(t_m) & \dot{x}_2(t_m) & \dots & \dot{x}_n(t_m) \end{bmatrix}
$$

**Candidate Function Library:** A library matrix $\mathbf{\Theta}(\mathbf{X}) \in \mathbb{R}^{m \times p}$ is constructed, where each of the $p$ columns is a candidate nonlinear function evaluated on the states in $\mathbf{X}$.

$$
\mathbf{\Theta}(\mathbf{X}) = \begin{bmatrix} | & | & | & | & | & | \\ \mathbf{1} & \mathbf{X} & \mathbf{X}^{P_2} & \mathbf{X}^{P_3} & \dots & \sin(\mathbf{X}) & \dots \\ | & | & | & | & | & | \end{bmatrix}
$$

For instance, the block for quadratic nonlinearities, $\mathbf{X}^{P_2}$, is structured as:  

$$
\mathbf{X}^{P_2} = \begin{bmatrix} x_1^2(t_1) & x_1(t_1)x_2(t_1) & \dots & x_n^2(t_1) \\ x_1^2(t_2) & x_1(t_2)x_2(t_2) & \dots & x_n^2(t_2) \\ \vdots & \vdots & \ddots & \vdots \\ x_1^2(t_m) & x_1(t_m)x_2(t_m) & \dots & x_n^2(t_m) \end{bmatrix}
$$

Sparse Regression Problem: The problem is cast as a linear system, seeking a sparse coefficient matrix $\mathbf{\Xi} \in \mathbb{R}^{p \times n}$ that relates the library to the measured derivatives.  

$$\dot{\mathbf{X}} = \mathbf{\Theta}(\mathbf{X})\mathbf{\Xi}$$

The matrix $\mathbf{\Xi} = [\boldsymbol{\xi}_1, \boldsymbol{\xi}_2, \dots, \boldsymbol{\xi}_n]$ contains column vectors $\boldsymbol{\xi}_k$, where each sparse vector defines the active terms for the $k$-th state variable's dynamics:

$$\dot{x}_k = f_k(\mathbf{x}) = \mathbf{\Theta}(\mathbf{x}^T)\boldsymbol{\xi}_k$$

The complete identified model is then:  

$$
\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x}) = \mathbf{\Xi}^T (\mathbf{\Theta}(\mathbf{x}^T))^T
$$

Handling Noisy Data: When derivatives $\dot{\mathbf{X}}$ are contaminated with noise, the model becomes:

$$\dot{\mathbf{X}} = \mathbf{\Theta}(\mathbf{X})\mathbf{\Xi} + \eta\mathbf{Z} \quad (10)$$

where $\mathbf{Z}$ is a matrix of i.i.d. Gaussian entries and $\eta$ is the noise magnitude. This overdetermined system is solved via sparse regression. One method is LASSO, which adds an $L_1$ regularization term:  

$$
\boldsymbol{\xi} = \underset{\boldsymbol{\xi}'}{\arg\min} \ |\mathbf{\Theta}\boldsymbol{\xi}' - \mathbf{y}|_2 + \lambda|\boldsymbol{\xi}'|_1
$$

where $\mathbf{y}$ is a column of $\dot{\mathbf{X}}$ and $\lambda$ is the sparsity-promoting parameter.

### Extensions

#### Discrete-Time Systems: For systems of the form 

$$\mathbf{x}_{k+1} = \mathbf{f}(\mathbf{x}_k)$$

the data matrices are constructed as:

$$
\mathbf{X}_{1}^{m-1} = \begin{bmatrix} \mathbf{x}_{2}^{m} \end{bmatrix} =\mathbf{\Theta}\!\left(\mathbf{X}_{1}^{m-1}\right)\mathbf{\Xi}
$$

For a linear library $\mathbf{\Theta}(\mathbf{x}) = \mathbf{x}$, this reduces to $\mathbf{X}_{2}^{m} = \mathbf{X}_{1}^{m-1} \mathbf{\Xi}$, which is equivalent to the Dynamic Mode Decomposition (DMD) formulation.

#### High-Dimensional Systems (PDEs)

For high-dimensional state vectors, dimensionality reduction is applied first. Using the Singular Value Decomposition (SVD), 

$$\mathbf{X}^T = \mathbf{\Psi}\mathbf{\Sigma}\mathbf{V}^*$$

the state can be approximated in a low-rank basis of $r$ modes:  

$$\mathbf{x} \approx \mathbf{\Psi}_r \mathbf{a}$$ 

where $\mathbf{\Psi}_r$ contains the first $r$ columns of $\mathbf{\Psi}$ (POD modes), and $\mathbf{a}$ is the $r$-dimensional vector of mode coefficients. The SINDy framework is then applied to find the Galerkin projected dynamics in the reduced coordinates:

$$\dot{\mathbf{a}} = \mathbf{f}_P(\mathbf{a})$$

#### Parameterized and Forced Systems

Bifurcation parameters $\mu$ and external forcing $u(t)$ can be incorporated by augmenting the state vector.

* For a parameter 
  $$\mu:  \dot{\mathbf{x}} = \mathbf{f}(\mathbf{x}; \mu) \\  \dot{\mu} = 0$$
* For time-dependence or forcing $u(t)$:  $$\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x}, u(t), t)  \\ \dot{t} = 1$$ 
    The library $\mathbf{\Theta}$ is then constructed from the augmented state vector (e.g., $[\mathbf{x}, \mu]$).

## Methodology/Algorithm

The primary algorithm employed is Sequential Thresholded Least-Squares, which is computationally efficient.

1. **Data Preparation:** Collect time-series measurement data for the state vector $\mathbf{x}(t)$. Numerically differentiate the data to obtain $\dot{\mathbf{x}}(t)$, potentially using a noise-robust method such as total variation regularized differentiation. Assemble the data into matrices $\mathbf{X}$ and $\dot{\mathbf{X}}$.
2. **Library Construction:** Construct the library matrix $\mathbf{\Theta}(\mathbf{X})$ containing a comprehensive set of candidate nonlinear functions of the state variables (e.g., polynomials, trigonometric functions).
3. **Sparse Regression:** Solve for each column $\boldsymbol{\xi}_k$ of the coefficient matrix $\mathbf{\Xi}$ independently using the following iterative procedure:
  * a. Initial Guess: Compute an initial full solution for the coefficients using standard least-squares: Xi = Theta\dXdt
  * b. Iterative Thresholding and Refitting (Loop for k=1:10):
    * i. Thresholding: Identify coefficients in Xi with a magnitude less than a predefined threshold $\lambda$ and set them to zero. smallinds = (abs(Xi)<lambda) Xi(smallinds)=0
    * ii. Refitting: For each state dimension ind, identify the remaining non-zero coefficients (biginds). Perform a new least-squares regression of dXdt(:,ind) onto only the columns of Theta corresponding to these biginds. Update Xi with this new, smaller set of coefficients. Xi(biginds,ind) = Theta(:,biginds)\dXdt(:,ind)
    * iii. Repeat: Continue this process until the set of non-zero coefficients in $X_i$ converges.
4. **Model Selection:** The sparsification parameter $\lambda$ is a critical knob. It is determined using cross-validation on held-out test data. An optimal $\lambda$ is chosen from the "elbow" of the Pareto front, which plots model accuracy versus model complexity (number of non-zero terms).

## Key Results (Quantified)

| System           | Key Findings |
|------------------|--------------|
| **Lorenz System** | The algorithm correctly identified the seven true terms $(x, y, z, xy, xz)$ in the dynamics from a library of polynomials up to 5th order.<br>- With sensor noise of η = 1.0 applied to derivatives, the recovered coefficients were close to the true values ($\sigma$ = 10, $\rho$ = 28, $\beta$ = 8/3 ≈ 2.667):<br> &nbsp;&nbsp;• **ẋ:** x term = -9.9996 (true: -10), y term = 9.9998 (true: 10)<br> &nbsp;&nbsp;• **ẏ:** x term = 27.9980 (true: 28), y term = -0.9997 (true: -1), xz term = -0.9999 (true: -1)<br> &nbsp;&nbsp;• **ż:** xy term = 1.0000 (true: 1), z term = -2.6665 (true: -2.6667)<br>- In low-noise scenarios, coefficients were identified to within **0.03%** of their true values. |
| **Fluid Wake**    | - Applied to data from a direct numerical simulation (state dimension: 292,500) of flow past a cylinder, after dimensionality reduction to a 3-mode system.<br>- Correctly identified a model with **quadratic nonlinearities**, consistent with the Navier–Stokes equations.<br>- Avoided incorrectly identifying an approximate **cubic Hopf normal form**, which is only valid on the system’s slow manifold. |
| **Hopf Normal Form** | - Correctly identified the structure of the Hopf normal form, including dependence on the bifurcation parameter μ, from noisy trajectory data.<br>- True dynamics: ẋ = μx − ωy − A x(x² + y²).<br>- Noisy training data caused ~8% error in cubic-term coefficients. Recovered coefficients for ẋ included: xxx = -0.9208 and xyy = -0.9211, where both should equal -A. |
| **Logistic Map**  | - Identified the discrete-time parameterized dynamics \(x_{k+1} = \mu x_k (1 - x_k)\) from stochastically forced data.<br>- Recovered model: x_{k+1} = 0.9993\,\mu x_k - 0.9989\,\mu x_k^2\, demonstrating high accuracy in structure and parameter recovery. |


## Logical Justification

The effectiveness of the SINDy framework is rooted in the physical assumption of parsimony: many fundamental governing laws are described by a small number of functional terms. This sparsity allows the system identification problem, which would otherwise require an intractable combinatorial search through all possible functions, to be reformulated as a computationally efficient sparse regression problem. By leveraging convex optimization methods (or efficient iterative approximations like sequential thresholding), the algorithm can effectively search a very large library of candidate functions and select only the most relevant terms. This inherent promotion of sparsity acts as a form of regularization, naturally balancing model complexity and descriptive accuracy, thereby preventing overfitting.
