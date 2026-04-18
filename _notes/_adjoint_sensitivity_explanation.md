# The Adjoint Sensitivity Method: Complete Derivation

## The Setup

We have:

- A state $z(t) \in \mathbb{R}^n$ evolving by $\frac{dz}{dt} = f_\theta(z, t)$, where $\theta \in \mathbb{R}^p$ are learnable parameters.
- A loss $L(z(t_1))$: any differentiable function of the final state.
- Goal: compute $\frac{dL}{d\theta} \in \mathbb{R}^p$.

The chain of dependence is: $\theta \to f_\theta \to z(t) \to z(t_1) \to L$. Since $\theta$ affects $L$ only through the trajectory, the derivative $\frac{dL}{d\theta}$ must account for how $\theta$ influences $z$ at every instant from $t_0$ to $t_1$.

---

## Part 1: The Adjoint ODE

### Definition

$$a(t) \;=\; \frac{\partial L}{\partial z(t)} \;\in\; \mathbb{R}^n$$

This is the sensitivity of the loss to the state at time $t$. At the final time we can compute it directly:

$$a(t_1) = \frac{\partial L}{\partial z(t_1)}$$

For example, if $L = \|z(t_1) - z_{\text{target}}\|^2$, then $a(t_1) = 2(z(t_1) - z_{\text{target}})$.

### Derivation of the ODE governing $a(t)$

We want to find $a(t)$ for $t < t_1$. Consider two nearby times $t$ and $t + \epsilon$. By the chain rule:

$$a(t) = \frac{\partial L}{\partial z(t)} = \frac{\partial L}{\partial z(t+\epsilon)} \cdot \frac{\partial z(t+\epsilon)}{\partial z(t)}$$

$$\tag{1} a(t) = a(t+\epsilon) \cdot \frac{\partial z(t+\epsilon)}{\partial z(t)}$$

We need $\frac{\partial z(t+\epsilon)}{\partial z(t)}$. The ODE gives us:

$$z(t+\epsilon) = z(t) + \int_t^{t+\epsilon} f_\theta(z(s), s)\, ds$$

For small $\epsilon$, Taylor-expand the integral:

$$z(t+\epsilon) = z(t) + \epsilon \, f_\theta(z(t), t) + O(\epsilon^2)$$

Differentiate both sides with respect to $z(t)$:

$$\frac{\partial z(t+\epsilon)}{\partial z(t)} = I + \epsilon \frac{\partial f_\theta(z(t), t)}{\partial z} + O(\epsilon^2)$$

where $I$ is the $n \times n$ identity and $\frac{\partial f_\theta}{\partial z} \in \mathbb{R}^{n \times n}$ is the Jacobian of $f_\theta$ with respect to the state.

Substitute into $(1)$:

$$a(t) = a(t+\epsilon) \left(I + \epsilon \frac{\partial f_\theta}{\partial z}\right) + O(\epsilon^2)$$

Expand:

$$a(t) = a(t+\epsilon) + \epsilon \, a(t+\epsilon) \frac{\partial f_\theta}{\partial z} + O(\epsilon^2)$$

Move $a(t+\epsilon)$ to the left:

$$a(t) - a(t+\epsilon) = \epsilon \, a(t+\epsilon) \frac{\partial f_\theta}{\partial z} + O(\epsilon^2)$$

Divide by $\epsilon$:

$$\frac{a(t) - a(t+\epsilon)}{\epsilon} = a(t+\epsilon) \frac{\partial f_\theta}{\partial z} + O(\epsilon)$$

Multiply both sides by $-1$:

$$\frac{a(t+\epsilon) - a(t)}{\epsilon} = -a(t+\epsilon) \frac{\partial f_\theta}{\partial z} + O(\epsilon)$$

Take $\epsilon \to 0$. The left side becomes $\frac{da}{dt}$. On the right, $a(t+\epsilon) \to a(t)$:

$$\tag{2} \boxed{\frac{da(t)}{dt} = -a(t)^\top \frac{\partial f_\theta(z(t), t)}{\partial z}}$$

This is a linear ODE in $a$ with time-varying coefficients (since $\frac{\partial f_\theta}{\partial z}$ depends on $z(t)$, which changes along the trajectory).

**What we have so far:** Given $a(t_1) = \frac{\partial L}{\partial z(t_1)}$, we can solve equation $(2)$ backward from $t_1$ to $t_0$ to obtain $a(t)$ at any time $t$.

**What we still need:** A formula for $\frac{dL}{d\theta}$.

---

## Part 2: Deriving the Parameter Gradient via the Augmented System

### The idea

We proved equation $(2)$ for the system $\frac{dz}{dt} = f_\theta(z,t)$. But the proof only used the chain rule and Taylor expansion — it works for **any** ODE system $\frac{dy}{dt} = g(y,t)$ with **any** adjoint $\frac{\partial L}{\partial y(t)}$. We will exploit this generality.

### Construct the augmented system

Define a new, larger state vector that includes $\theta$:

$$\tilde{z} = \begin{bmatrix} z \\ \theta \end{bmatrix} \in \mathbb{R}^{n+p}$$

Its dynamics are:

$$\frac{d\tilde{z}}{dt} = \underbrace{\begin{bmatrix} f_\theta(z, t) \\ 0 \end{bmatrix}}_{\tilde{f}(\tilde{z}, t)}$$

The second component is $0$ because $\theta$ is constant during the forward pass. This is the same physical system — nothing has changed except notation.

### The augmented adjoint

The adjoint of the augmented system is:

$$\tilde{a}(t) = \frac{\partial L}{\partial \tilde{z}(t)} = \begin{bmatrix} \frac{\partial L}{\partial z(t)} \\ \frac{\partial L}{\partial \theta} \end{bmatrix}^\top = \begin{bmatrix} a(t) & a_\theta(t) \end{bmatrix}$$

where:

- $a(t) = \frac{\partial L}{\partial z(t)} \in \mathbb{R}^n$ is the original adjoint (known).
- $a_\theta(t) = \frac{\partial L}{\partial \theta} \in \mathbb{R}^p$ is the parameter gradient we want.

Note: $a_\theta$ is our target quantity **by definition** — it is the $\theta$-component of $\frac{\partial L}{\partial \tilde{z}}$.

### Compute the augmented Jacobian

The Jacobian of $\tilde{f}$ with respect to $\tilde{z} = [z, \theta]$ is:

$$\frac{\partial \tilde{f}}{\partial \tilde{z}} = \begin{pmatrix} \frac{\partial f_\theta}{\partial z} & \frac{\partial f_\theta}{\partial \theta} \\ \frac{\partial (0)}{\partial z} & \frac{\partial (0)}{\partial \theta} \end{pmatrix} = \begin{pmatrix} \frac{\partial f_\theta}{\partial z} & \frac{\partial f_\theta}{\partial \theta} \\ 0 & 0 \end{pmatrix}$$

where $\frac{\partial f_\theta}{\partial z} \in \mathbb{R}^{n \times n}$ and $\frac{\partial f_\theta}{\partial \theta} \in \mathbb{R}^{n \times p}$.

### Apply the adjoint ODE (equation 2) to the augmented system

Since equation $(2)$ holds for any ODE system, we can apply it to $\tilde{z}$:

$$\tag{3} \frac{d\tilde{a}}{dt} = -\tilde{a}^\top \frac{\partial \tilde{f}}{\partial \tilde{z}}$$

Write this out explicitly:

$$\frac{d}{dt}\begin{pmatrix} a & a_\theta \end{pmatrix} = -\begin{pmatrix} a & a_\theta \end{pmatrix} \begin{pmatrix} \frac{\partial f_\theta}{\partial z} & \frac{\partial f_\theta}{\partial \theta} \\ 0 & 0 \end{pmatrix}$$

### Perform the matrix multiplication

The right-hand side is a row vector $\in \mathbb{R}^{1 \times (n+p)}$ multiplied by a matrix $\in \mathbb{R}^{(n+p) \times (n+p)}$, producing a row vector $\in \mathbb{R}^{1 \times (n+p)}$.

**Left block** (first $n$ columns — the $z$-component):

$$-\left(a \cdot \frac{\partial f_\theta}{\partial z} + a_\theta \cdot 0\right) = -a^\top \frac{\partial f_\theta}{\partial z}$$

This gives:

$$\frac{da}{dt} = -a^\top \frac{\partial f_\theta}{\partial z}$$

This is equation $(2)$ again. Nothing new.

**Right block** (last $p$ columns — the $\theta$-component):

$$-\left(a \cdot \frac{\partial f_\theta}{\partial \theta} + a_\theta \cdot 0\right) = -a^\top \frac{\partial f_\theta}{\partial \theta}$$

This gives:

$$\tag{4} \boxed{\frac{da_\theta(t)}{dt} = -a(t)^\top \frac{\partial f_\theta(z(t), t)}{\partial \theta}}$$

This is a new ODE. It says: the rate of change of the parameter gradient $a_\theta$ at time $t$ equals $-a(t)^\top \frac{\partial f_\theta}{\partial \theta}$.

Note that $\frac{\partial f_\theta(z(t),t)}{\partial \theta} \in \mathbb{R}^{n \times p}$ depends on time through $z(t)$ — it is NOT constant.

---

## Part 3: Integrating to obtain $\frac{dL}{d\theta}$

### What we integrate

Equation $(4)$ is an ODE: $\frac{da_\theta}{dt} = -a(t)^\top \frac{\partial f_\theta}{\partial \theta}$.

By the fundamental theorem of calculus, integrating both sides from $t_1$ to $t_0$:

$$\int_{t_1}^{t_0} \frac{da_\theta(t)}{dt}\, dt = \int_{t_1}^{t_0} \left(-a(t)^\top \frac{\partial f_\theta(z(t), t)}{\partial \theta}\right) dt$$

The left side evaluates directly:

$$\int_{t_1}^{t_0} \frac{da_\theta(t)}{dt}\, dt = a_\theta(t_0) - a_\theta(t_1)$$

So:

$$\tag{5} a_\theta(t_0) - a_\theta(t_1) = -\int_{t_1}^{t_0} a(t)^\top \frac{\partial f_\theta(z(t), t)}{\partial \theta}\, dt$$

### Why $a_\theta(t_1) = 0$

Recall $a_\theta(t) = \frac{\partial L}{\partial \theta}$ evaluated considering how $\theta$ affects the trajectory from time $t$ onward.

At $t = t_1$, the trajectory is over. The loss $L(z(t_1))$ depends on $z(t_1)$, which is already determined — perturbing $\theta$ at the very end of the trajectory changes nothing because there is no remaining integration for $\theta$ to influence. The effect of $\theta$ on $L$ comes entirely from shaping $f_\theta$ during $[t_0, t_1]$, not from a direct dependence at the endpoint.

Therefore $a_\theta(t_1) = 0$.

### The result

Substituting $a_\theta(t_1) = 0$ into equation $(5)$:

$$a_\theta(t_0) - 0 = -\int_{t_1}^{t_0} a(t)^\top \frac{\partial f_\theta(z(t), t)}{\partial \theta}\, dt$$

Since $a_\theta(t_0)$ has accumulated the sensitivity of $L$ to $\theta$ over the entire trajectory $[t_0, t_1]$, it equals the total derivative:

$$\tag{6} \boxed{\frac{dL}{d\theta} = -\int_{t_1}^{t_0} a(t)^\top \frac{\partial f_\theta(z(t), t)}{\partial \theta}\, dt}$$

### What each piece of the integrand means

At each time $t$, the integrand $a(t)^\top \frac{\partial f_\theta(z(t), t)}{\partial \theta}$ has a concrete meaning:

- $\frac{\partial f_\theta(z(t), t)}{\partial \theta} \in \mathbb{R}^{n \times p}$: how a small change in $\theta$ would change the vector field at state $z(t)$ at time $t$. This changes with $t$ because $z(t)$ changes.
- $a(t)^\top \in \mathbb{R}^{1 \times n}$: how much the loss cares about the state at time $t$.
- Their product $a(t)^\top \frac{\partial f_\theta}{\partial \theta} \in \mathbb{R}^{1 \times p}$: how much the loss cares about a change in $\theta$ **at this specific instant** $t$ — the instantaneous contribution to the total gradient.

The integral sums these instantaneous contributions over the entire trajectory. A perturbation to $\theta$ changes the vector field at every time point; the integral aggregates the effect of all these changes on $L$.

### Why we integrate backward (from $t_1$ to $t_0$)

The limits of integration $\int_{t_1}^{t_0}$ go backward because we are solving the adjoint ODE backward — $a(t)$ is only available if we start from $a(t_1)$ and propagate to earlier times using equation $(2)$. We cannot compute $a(t)$ forward because its initial condition is at $t_1$, not $t_0$.

In practice, we solve equations $(2)$ and $(4)$ simultaneously backward, and the state $z(t)$ is reconstructed backward at the same time (by solving $\frac{dz}{dt} = f_\theta(z,t)$ in reverse from $z(t_1)$).

---

## Part 4: The Complete Algorithm

### Forward pass ($t_0 \to t_1$)

Solve $\frac{dz}{dt} = f_\theta(z, t)$ forward. Store only $z(t_1)$. Compute $L(z(t_1))$.

### Backward pass ($t_1 \to t_0$)

Set initial conditions:

$$z(t_1) = \text{(from forward pass)}, \qquad a(t_1) = \frac{\partial L}{\partial z(t_1)}, \qquad a_\theta(t_1) = 0$$

Solve three coupled ODEs backward simultaneously:

$$\frac{dz}{dt} = f_\theta(z, t)$$

$$\frac{da}{dt} = -a(t)^\top \frac{\partial f_\theta(z(t), t)}{\partial z}$$

$$\frac{da_\theta}{dt} = -a(t)^\top \frac{\partial f_\theta(z(t), t)}{\partial \theta}$$

At each backward step:
1. The current $z(t)$ is available (reconstructed by the first ODE).
2. Using $z(t)$, evaluate $\frac{\partial f_\theta}{\partial z}$ and $\frac{\partial f_\theta}{\partial \theta}$ — both depend on $z(t)$.
3. Using $a(t)$, compute the right-hand sides of the second and third ODEs.
4. Advance all three backward by one step.

### Output

At $t_0$: read $a_\theta(t_0) = \frac{dL}{d\theta}$. Use it for gradient descent: $\theta \leftarrow \theta - \eta \frac{dL}{d\theta}$.

### Memory comparison

| | BPTT (naive) | Adjoint method |
|---|---|---|
| Forward | Store $z$ at all $N$ solver steps | Store only $z(t_1)$ |
| Backward | Backprop through each stored step | Solve 3 ODEs, reconstruct $z$ on the fly |
| Memory | $O(N)$ | $O(1)$ w.r.t. solver steps |

The saving is in memory: $z(t)$ is reconstructed during the backward pass instead of being retrieved from storage.
