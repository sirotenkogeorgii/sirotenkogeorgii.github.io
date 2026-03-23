---
title: Problems from the DSTML course
layout: default
noindex: true
---

# Selected DSTML Course Problems

**Table of Contents**
- TOC
{:toc}

## 1 Higher-order ODE systems

Show that any $n$-th order ODE of the form

$$f!\left(t, x, \frac{dx}{dt}, \frac{d^2x}{dt^2}, \ldots, \frac{d^n x}{dt^n}\right)=0$$

can always be transformed into a system of $n$ first order ODEs.

## 2 ODEs on the real line

Explain/show why one-dimensional ODEs on the real line can never give rise to oscillatory behavior.

## 3 Flow field of linear ODEs

Assume a linear ODE system, defined by

$$
\frac{d\mathbf{x}}{dt} = \mathbf{A}\cdot \mathbf{x},
\qquad
\text{where }
\mathbf{A}=
\begin{pmatrix}
0.1 & -0.3\\
0.2 & -0.3
\end{pmatrix}
\text{ and }
\mathbf{x}=
\begin{pmatrix}
x_1\\
x_2
\end{pmatrix}
$$

1. Determine the fixed points of this system by hand.

2. Plot the time evolution of the system for an initial vector of
   
   $$
   \mathbf{x}_0=
   \begin{pmatrix}
   1\\
   0
   \end{pmatrix}
   $$

3. Plot the flow field of the system in an appropriate range. What do you observe?

4. In the lecture you learned about several different types of dynamics (see also figure 5.2.8 in the Strogatz book) that can be obtained from a two-dimensional linear dynamical system (unstable node, stable node, saddle point, line attractor, degenerate node, stable spiral, unstable spiral and center). Define system parameters of your choosing that satisfy the respective requirements, and plot the corresponding trajectories and flow fields.

## 4 The Lotka-Volterra System

In this Exercise we are going to develop a small toolbox that will help you to inspect dynamical systems visually. This is supposed to help you develop an intuition for the phenomena that we are going to look at. Therefore we are going to analyze the Lotka-Volterra system of the form:

$$\frac{dx}{dt} = xa - bxy \qquad \text{(I)}$$

$$\frac{dy}{dt} = cxy - ey \qquad \text{(II)}$$

where $a>0,; b>0,; c>0,; e>0$

The Lotka-Volterra system is part of the larger class of Komoglorov-models which are used in biology and ecology to destibe predator-prey interactions. Here consider that $x$ represents prey and $y$ stands for predator.

1. Determine the fix-points of the system. How many are there? Describe in words what do these fixed-points mean in the light of predator prey dynamics.

2. Determine the the stability of the fixed-points. Hint: You need to compute the Jacobian of the system.

3. Plot the vector field of the Lotka-Volterra system. Form here consider $b = 1-a$ and $e = 1-c$. Plot the four fields for $a = 0.8, 0.2$ and $c = 0.8, 0.2$. Describe the vector field. Hint: use `meshgrid` and `quiver`.

4. Plot the fixed points and their stability to the plots for the respective parameter configurations from part 3 $(a=0.8, 0.2 \text{ and } c=0.8, 0.2)$.

5. Add the respective nullclines to the plots.

6. Add a subplot that visualizes the trajectory over time from a random initial condition.

7. Plot the trajectory form the origin $(x,y)=(0,0)$ and plot the trajectory for a small perturbation form the origin. What does this mean in the light of predator-prey dynamics.

8. What dynamical behavior do you expect if the population of one of the two species is zero $(x=0 \text{ and } y>0 \text{ or } x>0 \text{ and } y=0)$.

9. How does the mean over one period of the system evolves in time? (Please provide a plot)

If you want, I can also clean up the obvious OCR-like typos and give you a polished version.

Here is the parsed text from the two new images.

## 5 Linearization Fail

Note that for any $a \in \mathbb{R}$ the following system has a fixed point at $(0,0)$:

$$\dot{x} = -y + (ax-y)(x^2+y^2)$$

$$\dot{y} = x + (x+ay)(x^2+y^2)$$

1. Show that the linearization technique predicts $(0,0)$ to be a center.

2. To prove that it is not always a true center, transfer the system into polar coordinates. This means that you have to find equivalent differential equations for
   
   $$
   r=\sqrt{x^2+y^2}
   \quad\text{and}\quad
   \theta=\tan^{-1}!\left(\frac{y}{x}\right).
   $$

3. Find $a \in \mathbb{R}$ such that $(0,0)$ is a stable spiral, unstable spiral, or center.

4. For each of these cases, find the stable set $\Omega^s$ and the unstable set $\Omega^u$.

5. Explain the Hartman-Grobman Theorem and why it does not apply to this particular case.

## 6 Limit cycles

Consider the system

$$\dot{x} = -x + ay + x^2y$$

$$\dot{y} = b - ay - x^2y$$

where $a,b>0$. This is a model of glycolysis, the transformation of sugar $(y)$ to ADP $(x)$, which fuels cells. This is why we only consider $x,y \ge 0$.

1. Find equations for the nullclines and draw a phase portrait. For each of the subregions of $\mathbb{R}^2$ that are created by the nullclines, determine the signs of the flow.

2. Find conditions on $a,b$ such that the (single) fixed point is unstable.

3. Let $C$ be the closed set bordered by the pentagon defined by (1) the $x$ axis, (2) the $y$ axis, (3) $y=b/a$, (4) $y=b/a+b-x$ and (5) $x=z$ where $z$ is defined as the intersection of $y=b/a+b-x$ and the $x$ nullcline (no need to determine this value exactly, it’s ugly). Show that there are no trajectories moving out of $C$ and none moving along its boundary.

4. Argue why, if the fixed point is unstable, there must be an attracting limit cycle inside of $C$ around the fixed point.

## 7 Example Systems

For each of the following conditions, provide an example of a dynamical system defined in $\mathbb{R}^2$ which fulfills them (with proof that it does, the proof may be graphical). The examples have to be nontrivial, i.e., no derivative can be 0 everywhere. Use either cartesian, polar, or cylinder coordinates. If you think that no such system exists, explain why.

1. A stable fixed point at $(-1,1)$ and a saddle node at $(1,1)$. No other fixed points.

2. A linear system with an isolated fixed point other than the origin.

3. Two saddle nodes, a stable and an unstable node. The system may have more fixed points.

4. A half-stable ”saddle” cycle around the origin.

5. Four saddle nodes at $(\pm\sqrt{2}, \pm\sqrt{2})$ and a half stable cycle with radius $2$ around the $(0,\sqrt{2})$. No other fixed points.

6. Infinitely many fixed points, but no line attractor.

## 8 The Logistic Map

Consider the map:

$$x_{n+1} = r x_n (1-x_n)$$

For $0 \le x_n \le 1$ and $0 \le r \le 4$.

1. Plot a cobweb plot for $r = 0.5, 1.5, 2.5, 3.5$ and (3.9) with 30 steps. Describe the behavior for different $r$ in words.

2. Find all fix points of the logistic map and determine their stability. Give your result as function of $r$. Differentiate between $r < 1$, $r = 1$ and $r > 1$. What happens around $r = 1$? How is this phenomenon called?

3. Show that the logistic map has a 2-cycle for $r > 3$. Use that a 2-cycle requires $f(q) = p$ and $f(p) = q$.

4. What is stability of the 2-cycle from $3$? Does the stability change for some $r > 3$? Describe your findings in words.

5. Programming Exercise: For each $r \in \lbrace 0.001, 0.002, 0.003, \ldots, 3.998, 3.999\rbrace$ produce $1000$ trajectories from random initial conditions with 100 steps with the logistic map. Plot the endpoints of the simulated trajectories according to their respective $r$ in a 2D-scatter plot. What’s the name of such a diagram? (**Hint:** Since you only need to plot the last points of the trajectories, don’t store the trajectories in order to save memory.)


## 9 Poincaré Maps

The Poincaré Map can be used to find closed orbits and classify their stability. When you have an $n$-dimensional system $\dot{x}=f(x)$, a Surface of Section $S$ is an $(n-1)$-dimensional subspace, chosen such that for a trajectory of interest, there are $t_0,t_1,t_2,\ldots$ with $x(t_0),x(t_1),x(t_2),\ldots \in S$, but for all other $t$, $x(t)\notin S$. Also, $S$ is not tangent to the trajectory. That means, that the trajectory crosses the Surface of Section, but does not just touch it or move within it. A Poincaré Map $P$ for the Surface of Section $S$ and the dynamical system $\dot{x}=f(x)$ is then defined via $P(x(t_i)) = x(t_{i+1})$, i.e. the map “samples” from a trajectory at the times it crosses $S$.

1. Let $\dot{x}=f(x)$ be a dynamical system and $P$ a corresponding Poincaré map with surface of section $S$. If there exists $y \in S$ such that $P(y)=y$, then there exists a closed cycle in the system. Is that also a necessary condition? Why/why not?

2. Consider the following system:

$$\dot{r} = r(1-r)$$

$$\dot{\theta} = 2$$

Explicitly find a Poincaré Map for the surface of section

$$S=\lbrace (r,\theta): r>0,\ \theta=0\rbrace$$

Use it to prove that there exists a closed cycle at $r=1$. Is it stable, unstable or half-stable? (Hint: to find the map, take an initial condition $(r_1,0)\in S$. After one revolution, the trajectory intersects $S$ at $(r_2,0)$. To get $r_2$, note that you can read off the time $T$ the revolution takes from the equations.)

## 10 Coupled Oscillators

Consider the coupled oscillators:

$$\dot{\varphi}_1 = \omega_1 + C_1 \sin(\varphi_2-\varphi_1)$$

$$\dot{\varphi}_2 = \omega_2 + C_2 \sin(\varphi_1-\varphi_2)$$

1. Transform it into a 1D system that represents the oscillators’ phase difference by setting $\theta=\varphi_1-\varphi_2$ and write down the differential equation.

2. Show that the system undergoes a saddle-node bifurcation at $\|\omega_1-\omega_2\|=C_1+C_2$. What does this mean qualitatively for the coupled oscillators?


## 11 Sustainable Fishing Bifurcation

In the idyllic Bernese Oberland lies the particularly idyllic Oeschinen Lake, home to a variety of fish.[1] Let $x$ be proportional to the number of fish happily swimming in the lake. They reproduce at rate $r$, but the more fish there are, the more the limited resources available in the small lake are strained, so we model their reproduction rate as

$$\dot{x} = rx(1-x)$$

Recently, the people living in a nearby village have discovered that these fish are very tasty, which is why they now catch them at rate $c$. This leads to the total fish population being described by the equation

$$\dot{x} = rx(1-x) - cx$$

As the village’s only expert on dynamical systems and bifurcation theory, the local council has commissioned you as a consultant. Using quantitative bifurcation-theoretic arguments, what advice would you give the council regarding the village’s fishing activities – particularly the catching rate $c$ – assuming the villagers want to continue eating fresh, tasty fish?

[1] **Disclaimer:** The background information and mathematical modeling in this exercise may contain inaccuracies. Consider it as an exercise in dynamical systems theory rather than mathematical biology.


## 12 Hopf Bifurcation

In the 1800s, Australia decided to combat the invasive rabbit population $(R)$ by introducing another foreign species: foxes $(F)$. The following system of population growth rates of both species shows why this idea was not so ingenious after all:

$$
\dot{R} = (1-R)R - \frac{RF}{0.3+\alpha R},
\qquad
\dot{F} = -0.5F + \frac{RF}{0.3+\alpha R}
$$

Show that the system undergoes two supercritical Hopf bifurcations at $\alpha = 0.5$ and $\alpha = 1.2$. Note that the bifurcation point is not the origin. Draw a bifurcation diagram (by hand or computer). Also classify the origin’s stability.

**Hint:** It is recommended to use Python’s symbolic computing library, SymPy. No calculations by hand are required for this exercise.

## 13 Lorenz System

The Lorenz system, described by equations

$$\dot{x} = \sigma(y-x)$$

$$\dot{y} = \rho x - y - xz$$

$$\dot{z} = xy - \beta z,$$

with parameters $\sigma,\rho,\beta \ge 0$, was originally developed as a model for atmospheric convection.[2] Nowadays, it serves as an important toy model in the study of bifurcations and chaos. It is also often used as a benchmark for dynamical systems reconstruction.[3] The following exercises are meant to provide some first insights.

1. Find three fixed points. Some of them exist only for certain parameter configurations, which are to be determined.

2. Classify the fixed points’ stability depending on parameter configurations. What kind of bifurcation(s) do they undergo? Draw a bifurcation diagram (by hand or computer).
   Note: Determining stability only means to distinguish between stable, unstable or half-stable points. Their exact shape doesn’t matter.

3. Show that there can be no cycles for $\rho < 1$. (Hint: The function
   
   $$V(x,y,z) = \frac{x^2}{\sigma} + y^2 + z^2$$
   
   might be of help.)

## 14 Lorenz Map

Consider the Lorenz system given by the following system of ODEs:

$$\dot{x}=\sigma(y-x)$$

$$\dot{y}=x(\rho-z)-y$$

$$\dot{z}=xy-\beta z$$

Consider its chaotic regime by choosing $\sigma=10$, $\beta=\frac{8}{3}$ and $\rho=28$.

1. Choose a reasonable initial condition and draw a trajectory of the system. Visualize 3D state space. The trajectory should be long enough such that you can identify the butterfly-shaped chaotic attractor.

2. In the lecture, you learned that Edward Lorenz used consecutive maxima of the $z$ coordinate to quantify the chaotic nature of the system. Plot the time trace of the $z$ coordinate of the trajectory drawn before. Find all consecutive maxima of the $z$ coordinate, $\lbrace z_n\rbrace$, and plot the first-order return plot of the resulting series of peaks. Overlay the bisectrix. By inspection of the return plot, verify that $\forall n:\ \|F'(z_n)\|>1$.
   **Hint:** To find the peaks of the signal, you can use `Peaks.jl` in Julia or `scipy.signal.find_peaks` in Python.

3. Now, given the Lorenz map $z_{n+1}=F(z_n)$ and assuming $\|F'(z_n)\|>1\ \forall n$, show that all closed orbits of $F$ are unstable.
   **Hint:** Consider the fate of a small perturbation $\varepsilon_n$ at step $n$ under repeated application of $F$ and recall that for a $k$-cycle we need to have $z_{n+k}=z_n$.

## 15 Cantor Set as a Bernoulli Process

In the lecture we discussed the Cantor set. You can model it through a Bernoulli process by drawing from a Bernoulli distribution with $p=\frac12$ and defining the following mapping:

$$\text{if } q=0:\quad x_{n+1}=\frac{x_n}{3}$$

$$\text{if } q=1:\quad x_{n+1}=\frac{x_n+2}{3}$$

Simulate this mapping for an appropriate number of iterations $N$ for different initial conditions in the interval between $0$ and $1$ and plot the resulting set of points. Investigate the self-similar behavior of the set by plotting different slices of the x-axis.

## 16 Cantor Set and the Tent Map

There is an interesting relationship between the Cantor set and the tent map, which is defined by

$$
x_{n+1}=f_\mu(x_n)=
\begin{cases}
\mu x_n & \text{for } x_n<\frac12 \
\mu(1-x_n) & \text{for } \frac12 \le x_n
\end{cases}
$$

with $\mu$ a positive real parameter.

This map can be interpreted as a simple piecewise discrete-time dynamical system, which will play a crucial role in the rest of the lecture. Generate trajectories from the tent map for $\mu=3$, starting from different initial conditions. When do you observe divergent behavior? How can you make use of the results of the previous exercise to find initial states that do not lead to divergences?
