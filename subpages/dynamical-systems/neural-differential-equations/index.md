---
title: Neural Differential Equations
layout: default
noindex: true
---

# Neural Differential Equations



## Symbolic regression

Deep learning models (including neural differential equations) are often accurate but hard to interpret, and in science understanding the model can be as important as performance. That motivates seeking **symbolic expressions** — compact formulas built from simple operations like (+), (\times), exponentiation, etc.

**Symbolic regression** aims to automatically discover such expressions from data. But it’s difficult because the space of possible expressions is not smoothly differentiable (so search often relies on *genetic algorithms*) and it is **combinatorially huge** — the number of possible expression trees grows extremely fast with size: $\frac{(2n)!}{(n+1)!n!}$ binary trees with $n$ vertices and so as a rough approximation we may expect there to be a similar number of possible expressions to consider. 

As a result, symbolic regression tends to work best for simpler problems; beyond a certain complexity, the search becomes impractical.

### Symbolic regression for dynamical systems

$\text{Example (SINDy)}:$ Assume we have pairs  $(\frac{dy}{dt}(t), y(t))$, and that they satisfy an equation of the form

$$\frac{dy}{dt}(t) = f(y(t))$$

In this setting, [SINDy](/subpages/dynamical-systems/dynamical-systems-in-machine-learning/sindy) tries to find a symbolic form for $f$ by first choosing a set of candidate features $f_i$. It then represents the function as

$$f(y) = \sum_{i=1}^N \theta_i f_i(y)$$

and fits this model by regressing $\frac{dy}{dt}(t)$ onto the feature values ${f_i(y(t))}_{i=1}^N$. A sparsity-inducing penalty, such as $L^1$ regularization, is applied to $\theta$ so that the final expression contains only a small number of terms. 

SINDy is widely regarded as a leading approach for symbolic regression in dynamical systems, with many extensions and applications reported in the literature.

However, SINDy relies on two strong assumptions:
* that matched measurements of both $y$ and $\frac{dy}{dt}(t)$ are available, and
* $f$ can be expressed as a shallow formula—specifically, a sparse linear combination of a preselected set of candidate functions.

> SINDy sometimes works around the first assumption by approximating $\frac{dy}{dt(t)}$ using finite diﬀerences. However this requires densely-packed observations.

Next, we will see how **neural differential equations (NDEs)** can help relax both of these assumptions introduced in this example.

###TODO: finish the chapter