## Lecture 4

#### Clarification: The Nature of Manifolds

There was a subtle but critical point regarding the definition of manifolds, specifically whether they are open or closed sets. The answer depends on the context in which you view the manifold.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

Let us clarify the nature of a manifold with respect to its own dimensionality versus its embedding in a higher-dimensional ambient space.

* **A Manifold is Locally an Open Set:** When we say a manifold is an "open set," we are speaking locally and with respect to its own dimension. At any point $p$ on a $k$-dimensional manifold, one can always find a $k$-dimensional neighborhood around $p$ that contains only other points from the manifold. This neighborhood is homeomorphic to a $k$-dimensional open Euclidean ball.
  * **Example:** Consider a 1D limit cycle (a closed orbit) living in a 2D state space. If you pick any point on this circular manifold, you can always find a small 1D line segment (an open interval) around it that is entirely contained within the manifold.
* **A Manifold is a Closed Set in its Ambient Space:** When viewed from the perspective of the higher-dimensional space it resides in (the "ambient space"), the manifold is a closed set.
  * **Example:** For the same 1D limit cycle in 2D space, you cannot draw a 2D open ball around any point on the cycle that is completely contained within the manifold. Any such 2D ball will inevitably contain points from the surrounding 2D space that are not on the 1D limit cycle. Therefore, with respect to the 2D topology, the limit cycle is a closed set.

This distinction is crucial for a precise understanding of the geometric structures that govern system dynamics.

</div>

#### Clarification: Equilibria as Temporal Anchors

We also discussed an example of a mapping that failed to be a homeomorphism because of its behavior at an equilibrium point. This highlights a fundamental principle: the nature of equilibria profoundly impacts the global topology of a phase portrait.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

A system with a stable equilibrium and a system with an unstable equilibrium are not topologically equivalent. The intuition behind this is that equilibria serve as temporal anchors for the dynamics.

* An equilibrium point fixes where trajectories go as $t \to \infty$ (for a stable equilibrium) or where they came from as $t \to -\infty$ (for an unstable one). It imposes a specific and preferred temporal orientation on the flow in its vicinity.
* If you were to remove the equilibrium point from the state space, the defining "anchor" of the flow would be gone. In this modified space (without the equilibrium), it might be possible to find a homeomorphism between two different flows.
* However, once the equilibrium is included, its role as a temporal anchor breaks the equivalence. A continuous mapping (a homeomorphism) cannot reconcile the fundamentally different long-term behaviors of convergence versus divergence.

Think of equilibria as providing the ultimate destinations or origins for trajectories, and this function cannot be smoothly mapped away.

</div>

### Multistability and the Wilson-Cowan Model

We now move to a fascinating and ubiquitous phenomenon in nonlinear systems: multistability. This is the capacity for a system to possess more than one stable state (e.g., multiple stable equilibria) for a single set of parameters.

#### Introduction to Multistability

When a system is multistable, its long-term behavior depends entirely on its initial conditions. The state space is partitioned into distinct regions, known as basins of attraction. If a trajectory starts within a particular basin, it will inevitably converge to the stable state (or attractor) associated with that basin. The boundaries between these basins are called separatrices.

To explore these concepts, we will analyze a classic and influential model from computational neuroscience.

#### The Wilson-Cowan Model: A Neurodynamic System

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

The Wilson-Cowan model, introduced in the 1970s, was designed to explain the origin of oscillatory activity in the brain, such as the brain waves measured by an electroencephalogram (EEG). It is a foundational model that continues to inspire modern work in both neuroscience and machine learning (e.g., Neural ODEs).

The model simplifies the complexity of the brain by considering only two large, interacting populations of neurons:

1. A population of excitatory neurons, which tend to increase the activity of other neurons.
2. A population of inhibitory neurons, which tend to decrease the activity of other neurons.

The core mechanism for generating complex dynamics, including oscillations, arises from their interaction via feedback loops:
* Positive Feedback: The excitatory population excites itself.
* Interacting Feedback: The excitatory population excites the inhibitory one, which in turn sends back inhibition to the excitatory population, creating a negative feedback loop.

This model is an example of a mean-field approach, where the collective activity of a large, assumedly homogeneous group of cells is described by a single continuous variable.

</div>

#### Mathematical Formulation

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Wilson-Cowan system)</span></p>

The state of the **Wilson-Cowan system** is described by the average firing rates of the excitatory and inhibitory populations, denoted by $\nu_e(t)$ and $\nu_i(t)$, respectively. The dynamics are governed by the following system of nonlinear ordinary differential equations:

$$\tau_e \frac{d\nu_e}{dt} = -\nu_e + f_e(w_{ee}\nu_e - w_{ie}\nu_i - \theta_e)$$

$$\tau_i \frac{d\nu_i}{dt} = -\nu_i + f_i(w_{ei}\nu_e - \theta_i)$$

Where:

* $\tau_e, \tau_i$ are the time constants for each population.
* $w_{ab}$ represents the synaptic weight from population b to population a.
* $\theta_e, \theta_i$ are activation thresholds.
* $f(\cdot)$ is a nonlinear sigmoid activation function, given by: $f(x) = \frac{1}{1 + e^{-\beta x}} = (1 + e^{-\beta x})^{-1}$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

The sigmoid function is a crucial component, modeling the input-output relationship of the neuron populations. It has two key properties:

1. It is bounded between $0$ and $1$, representing a firing rate that cannot be negative or infinitely high.
2. Its steepness is controlled by the slope parameter $\beta$. A larger $\beta$ results in a sharper, more switch-like transition from "off" $0$ to "on" $1$.

The term inside the sigmoid function, such as $w_{ee}\nu_e - w_{ie}\nu_i - \theta_e$, represents the total input current to the population.

</div>

#### Qualitative Analysis of the Vector Field

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

The nullclines are powerful analytical tools because they segregate the state space into distinct regions of flow. The sign of a variable's derivative must flip every time a trajectory crosses that variable's nullcline.

Let's consider the flow directions in the ($\nu_e, \nu_i$) plane:

* Across the $\nu_i$-nullcline (the sigmoid curve):
  * For points above this curve, the linear term $-\nu_i$ dominates, so $\frac{d\nu_i}{dt} < 0$. The flow is downward.
  * For points below this curve, $\frac{d\nu_i}{dt} > 0$. The flow is upward.
* Across the $\nu_e$-nullcline (the $N$-shaped curve):
  * For points to the right of this curve, $\frac{d\nu_e}{dt} < 0$. The flow is to the left.
  * For points to the left of this curve, $\frac{d\nu_e}{dt} > 0$. The flow is to the right.

By sketching these general directions in each region bounded by the nullclines, we can build a qualitative picture of the system's phase portrait.

</div>

#### Stability and Saddle Manifolds

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

Combining the qualitative flow analysis with the location of the equilibria, we can infer their stability properties. In the three-equilibrium case:

* The two outer equilibria are stable fixed points. Trajectories starting nearby will spiral or move directly into them.
* The middle equilibrium is a saddle point. It is unstable, attracting trajectories along one direction (its stable manifold) and repelling them along another (its unstable manifold).

These stability assignments can be formally proven by calculating the Jacobian matrix of the system at each fixed point and analyzing its eigenvalues.

The saddle point and its manifolds play a crucial structural role. The stable manifold of the saddle is a particularly important curve. Any trajectory initiated exactly on this manifold will flow directly into the saddle point. More importantly, this manifold acts as the separatrix dividing the basins of attraction of the two stable equilibria.

This leads to a critical question: what happens if we start a trajectory just slightly to one side of the saddle's stable manifold?

</div>

### Attractors, Basins, and Limit Sets

In the study of dynamical systems, a central goal is to understand the long-term behavior of trajectories. Where do they originate, and where do they ultimately lead? This chapter introduces the foundational concepts of attractors, the regions from which they draw trajectories (their basins), and the mathematical tools used to formalize these ideas, namely alpha and omega limit sets.

#### Bistability, Basins of Attraction, and the Separatrix

Consider a simple one-dimensional system with two stable fixed points (equilibria) and one unstable fixed point between them.

* Point Attractors: The two stable equilibria are called point attractors. Trajectories that start near them will converge to them as time progresses.
* Basin of Attraction: Each point attractor has a surrounding neighborhood from which all trajectories converge to it. This neighborhood is called the basin of attraction.
* Separatrix: In this system, the unstable fixed point acts as a boundary. If an initial condition is to the left of this point, its trajectory will converge to the left attractor. If it starts to the right, it will converge to the right attractor. This boundary, which separates the basins of attraction, is known as a separatrix.

A system exhibiting this property of having two distinct attractors, each with its own basin of attraction, is said to possess bistability. This is a common and important feature in many natural and engineered systems.

#### Alpha and Omega Limit Sets

To more rigorously describe the long-term behavior of trajectories, we introduce the concepts of $\omega$-limit sets (for forward time) and $\alpha$-limit sets (for backward time).

##### Intuitive Understanding

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

An omega $\omega$ limit set is the set of points that a trajectory approaches as time goes to positive infinity $t \to \infty$. It describes where the system "ends up." For example, in a system with a single point attractor, the $\omega$-limit set for any point in its basin of attraction is the attractor itself.

An alpha $\alpha$ limit set is the set of points that a trajectory approaches as time goes to negative infinity $t \to -\infty$. It describes where the system "came from."

Consider a heteroclinic orbit, which is a trajectory that connects two different equilibrium points. For instance, an orbit that starts at an unstable saddle point and flows into a stable node.

* The $\omega$-limit set of this trajectory is the stable node.
* The $\alpha$-limit set of this trajectory is the saddle point.

For a trajectory in the basin of attraction of a point attractor that does not lie on a specific manifold, its $\alpha$-limit set might be at infinity, depending on whether the system's state space is bounded.

</div>

##### Formal Definition

Let a dynamical system be defined by a state space $E \subseteq \mathbb{R}^m$ and a flow operator $\phi_t(x_0)$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Omega Limit Set)</span></p>

The **omega limit set** of a point $x_0 \in E$, denoted $\omega(x_0)$, is the set of points that the trajectory through $x_0$ approaches as $t \to \infty$. It is formally defined as the intersection of the closures of the trajectory's future paths:

$$\omega(x_0) = \bigcap_{s \in \mathbb{R}} \overline{\bigcup_{t > s} \phi_t(x_0)} = \bigcap_{s \in \mathbb{R}} \overline{ \lbrace \phi_t(x_0)\mid t > s\rbrace }$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Chaos would needs a complicated $\omega$-limit set)</span></p>

A chaotic trajectory does **not** settle to:
* a single equilibrium, or
* a single periodic orbit.

Instead, it keeps wandering forever in a complicated recurrent way. So its $\omega$-limit set must be something much richer: a set containing infinitely many recurrent motions, typically a strange attractor.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">($\omega$-limit set and stability)</span></p>

* if the solution tends to a stable equilibrium, then the **$\omega$-limit set is just that one point**;
* if it approaches a stable periodic orbit, then the **$\omega$-limit set is the whole closed orbit**.

</div>

<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/omega_limit_set_visualization.png' | relative_url }}" alt="a" loading="lazy">
</figure>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

This definition works by considering the entire future path of the trajectory starting from some time $s$. As we let $s$ increase towards infinity, we take the intersection of all these future paths. This process "trims away" the transient parts of the trajectory, leaving only the set of points that the system visits infinitely often as $t \to \infty$. The closure (denoted by the overline) ensures that the limit points themselves are included in the set.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Alpha Limit Set)</span></p>

The **alpha limit set** of a point $x_0 \in E$, denoted $A(x_0)$, is defined analogously for reverse time $t \to -\infty$:

$$A(x_0) = \bigcap_{s \in \mathbb{R}} \overline{\bigcup_{t < s} \phi_t(x_0)} = \bigcap_{s \in \mathbb{R}} \overline{ \lbrace \phi_t(x_0)\mid t < s\rbrace }$$

</div>

#### Formal Definition of an Attractor

Using the concepts above, we can now provide a rigorous definition of an attractor and its associated basin of attraction.

##### Defining Properties

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Attractor)</span></p>

Given a dynamical system $(\mathbb{R} \times E, \phi)$ where $E \subseteq \mathbb{R}^m$ is the state space, a set $A \subseteq E$ is an **attractor** if it satisfies the following three properties:

1. **Invariance:** The set $A$ is invariant under the flow. This means that if a trajectory starts in $A$, it remains in $A$ for all time.
  
   $$\phi_t(A) = A \quad \forall t \in \mathbb{R}$$

2. **Attraction:** There exists a neighborhood of $A$, called the basin of attraction $B$ (where $A \subset B \subseteq E$), such that all trajectories starting in $B$ converge to $A$ as $t \to \infty$.
   
   $$\forall x_0 \in B, \quad \lim_{t \to \infty} d(\phi_t(x_0), A) = 0$$
  
   Here, $d(p, S)$ can be defined as the minimum distance from a point $p$ to any point in the set 
  
   $$S: d(p, S) = \inf_{y \in S} \|p-y\|$$

3. **Minimality:** $A$ is a minimal set with the first two properties. There is no smaller, proper subset of $A$ that is also an attractor with the same basin of attraction. If such a smaller set existed, the other points in $A$ would simply be part of the basin for that smaller attractor.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

An attractor is not necessarily a single point. It can be a more complex set, such as a closed orbit or even a fractal structure (a strange attractor). The key properties are that it is a self-contained "destination" for trajectories and that it draws in all trajectories from a surrounding region.

</div>

##### The Basin of Attraction

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Basin of Attraction)</span></p>

The **basin of attraction** $B$ for an attractor $A$ is defined as the largest set of initial conditions whose trajectories converge to $A$. It is the maximal set that satisfies the attraction property defined above.

</div>

### Periodic Behavior: Closed Orbits and Limit Cycles

While fixed points describe states of equilibrium, many systems exhibit sustained, periodic behavior. This corresponds to trajectories that form closed loops in the state space. This chapter explores these closed orbits, distinguishes between different types, and introduces the important concept of a limit cycle.

#### Closed Orbits in State Space

A closed orbit is a trajectory that returns to its starting point after a finite time, tracing a closed loop in the state space. Not all closed orbits are attractors. To understand a crucial distinction, we first analyze a specific nonlinear system.

#### Case Study: The Lotka-Volterra System as a Nonlinear Center

##### System Definition and Equilibria

Let us revisit the Lotka-Volterra system, a model of predator-prey dynamics:

$$\begin{aligned} \dot{x} &= \alpha x - \beta xy \\ \dot{y} &= \gamma xy - \lambda y \end{aligned}$$

We will consider the specific parameter set: $\alpha = 3, \beta = 1, \gamma = 0.5$, and $\lambda = 1$.

This system has two equilibria:

1. A trivial equilibrium at $(0, 0)$. For these parameters, this point is a saddle.
2. A co-existence equilibrium at $(\lambda/\gamma, \alpha/\beta)$, which for our parameters is at $(2, 3)$.

##### Linearization and Non-Hyperbolic Systems

To analyze the stability of the equilibrium at $(2, 3)$, we compute the Jacobian matrix at this point:  

$$J(2, 3) = \begin{pmatrix} 0 & -2 \\ 3/2 & 0 \end{pmatrix}$$  

The eigenvalues of this matrix are purely imaginary:  

$$\lambda_{1,2} = \pm i\sqrt{3} \approx \pm 1.73i$$ 

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

In linear systems, purely imaginary eigenvalues correspond to a center, around which trajectories form closed, neutrally stable orbits. However, we must be cautious when applying this intuition to a nonlinear system.

The equilibrium at $(2, 3)$ is non-hyperbolic because its eigenvalues have a real part equal to zero. For such systems, the Hartman-Grobman theorem does not necessarily hold. This theorem guarantees that the behavior of a nonlinear system near a hyperbolic equilibrium is qualitatively the same as its linearization. Since the theorem does not apply here, we cannot be certain of the system's behavior based on the first-order Taylor expansion (the linearization) alone; higher-order terms could fundamentally change the dynamics.

</div>

##### Conservative Systems and Neutrally Stable Orbits

In this particular case, the linearization does correctly predict the qualitative behavior. The system exhibits a dense set of closed, neutrally stable orbits around the equilibrium at $(2,3)$. This structure is called a nonlinear center.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name"></span></p>

The term "neutrally stable" means that if the system is on one closed orbit and is perturbed slightly, it does not return to the original orbit nor does it spiral away. Instead, it simply settles onto a new, nearby closed orbit.

This behavior is not typical for a randomly chosen nonlinear system. It arises here because the Lotka-Volterra system is a conservative system. This means it possesses a quantity that is conserved (remains constant) along any given trajectory. Such systems belong to a special class known as Hamiltonian systems. The existence of this conserved quantity is what enforces the structure of a continuous family of closed orbits.

</div>

#### Introduction to Limit Cycles

The nonlinear center seen in the Lotka-Volterra system is a special case. A more common and structurally robust form of periodic behavior is the limit cycle.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Limit Cycle)</span></p>

A **limit cycle** is a closed orbit that is isolated. The key distinction is "isolated." Unlike the dense family of orbits in a nonlinear center, a limit cycle has no other closed orbits in its immediate vicinity. Trajectories near a stable limit cycle will spiral towards it, making it an attractor. Trajectories near an unstable limit cycle will spiral away from it. This property of being isolated makes limit cycles far more robust to perturbations than the neutrally stable orbits of a center.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Isolation, Limit Cycle vs. Center)</span></p>

Consider a **stable limit cycle** being inside of the **unstable limit cycle**. They are neighbouring, but both are limit cycles.

The word **isolated** does **not** mean "there are no other closed orbits anywhere nearby in the phase plane." It means:

> for a given closed orbit, there exists some small neighborhood of that orbit that contains no other closed orbits.

So if you have:

* one **stable** closed orbit,
* and a larger **unstable** closed orbit around it,

then each one can still be isolated **individually**, provided there is a little band around each cycle with no other closed orbits inside that band.

What is forbidden is a **continuous family** of closed orbits packed arbitrarily close together, like in a center. In a center, every small neighborhood of one closed orbit contains more closed orbits, so none of them is isolated.

This is completely possible. The two cycles are separate closed orbits, and each may have its own **empty "buffer zone"** with no other periodic orbit right next to it.

<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/two_limit_cycles_phase_portrait.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>Inner stable limit cycle inside an outer unstable limit cycle</figcaption>
</figure>

The real contrast is:

* **center**: infinitely many nearby closed orbits, not isolated
* **limit cycle**: a particular closed orbit stands alone locally

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Is Limit Cycle an attracting or repelling Center?)</span></p>

No.

A **center** and a **limit cycle** are different kinds of objects.

A **center** is:

* an **equilibrium point** in the middle, and
* around it there is a whole family of closed orbits.

A **limit cycle** is:

* not an equilibrium point,
* but one particular **periodic orbit**,
* and it is **isolated** from other periodic orbits.

So the analogy is tempting, but not correct.

More precisely:

* a **center** has nearby trajectories that are also closed curves
* a **stable focus** has nearby trajectories spiraling into an equilibrium
* a **stable limit cycle** has nearby trajectories spiraling into a **closed orbit**
* an **unstable limit cycle** has nearby trajectories spiraling away from a **closed orbit**

So a stable or unstable limit cycle is much closer in spirit to a **focus**, not to a center.

Why “attracting center” is not really a thing:

* if trajectories spiral into the middle point, that point is a **stable focus** or node, not a center
* if trajectories spiral away, it is an **unstable focus** or node, not a center
* for a true center, nearby trajectories stay on closed orbits and neither approach nor leave the equilibrium asymptotically

So:

* **center** = equilibrium + non-isolated family of periodic orbits + neutral behavior
* **limit cycle** = isolated periodic orbit + can be attracting, repelling, or semistable

The main difference is not just attraction/repulsion.
The crucial difference is **equilibrium point vs isolated closed orbit**.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Limit Cycle is inherently a nonlinear phenomenon)</span></p>

Limit cycles are inherently nonlinear phenomena; they can’t occur in linear systems. 

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Closed orbits could happen in linear DS, but not limit cycles)</span></p>

Of course, a linear system 

$$\dot x = x A$$

can have closed orbits, but **they won’t be isolated**; if $x(t)$ is a periodic solution, then so is $cx(t)$ for any constant $c\neq 0$. Hence $x(t)$ is surrounded by a one-parameter family of closed orbits (see Figure).

<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/one_parameter_family_of_closed_orbits.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>One-parameter family of closed orbits</figcaption>
</figure>

Consequently, the amplitude of a linear oscillation is set entirely by its initial conditions; **any slight disturbance to the amplitude will persist forever**. In contrast, limit cycle oscillations are determined by the structure of the system itself.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Importance of Limit Cycles)</span></p>

Stable limit cycles are very important scientifically—they model systems that exhibit self-sustained oscillations. In other words, **these systems oscillate even in the absence of external periodic forcing**. Of the countless examples that could be given, we mention only a few: 

* the beating of a heart; 
* the periodic firing of a pacemaker neuron; 
* daily rhythms in human body temperature and hormone secretion; 
* chemical reactions that oscillate spontaneously;
* dangerous self-excited vibrations in bridges and airplane wings. 

In each case, there is a standard oscillation of some preferred period, waveform, and amplitude. If the system is perturbed slightly, it always returns to the standard cycle.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Simple Limit Cycle)</span></p>

**It’s straightforward to construct examples of limit cycles if we use polar coordinates.**

Consider the system

$$\dot r = r(1 - r^2)$$

$$\dot \theta = 1$$

where $r\geq 0$. The radial and angular dynamics are uncoupled and so can be analyzed separately.

<div class="pmf-grid">
<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/simple_limit_cycle1.png' | relative_url }}" alt="a" loading="lazy">
</figure>
<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/simple_limit_cycle2.png' | relative_url }}" alt="a" loading="lazy">
</figure>
<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/simple_limit_cycle3.png' | relative_url }}" alt="a" loading="lazy">
</figure>
</div>

**As expected, the solution settles down to a sinusoidal oscillation of constant amplitude, corresponding to the limit cycle solution.**

</div>

### Limit Cycles and Nonlinear Oscillations

In our previous analysis, we focused on systems whose long-term behavior converges to a single point in state space—an equilibrium or fixed point. However, many systems in nature, from the firing of neurons to the orbits of planets, exhibit sustained, stable oscillations. These phenomena cannot be explained by fixed points alone. This chapter introduces a new type of attractor: the limit cycle, which provides the mathematical framework for understanding stable, nonlinear oscillations.

#### From Stable Points to Stable Orbits: An Example

Let us revisit the Wilson-Cowan model of excitatory and inhibitory neuron populations. The dynamics are described by a set of differential equations, and their equilibria are found at the intersections of the nullclines. By adjusting the system's parameters, we can fundamentally change its behavior.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

Consider a scenario where we adjust the parameters of the Wilson-Cowan model, specifically the $\beta$ parameter that governs the slope of the inhibitory nullcline's sigmoid function. If we make this slope sufficiently steep, a significant change occurs. The system's single fixed point can transform from a stable spiral, which draws all trajectories inward, into an unstable spiral.

When the fixed point is an unstable spiral, trajectories starting near it are pushed outwards. However, due to the bounded nature of the sigmoid functions in the equations, trajectories starting very far from the origin are pushed back inwards. This creates a "push-pull" dynamic: trajectories are repelled from the center and corralled from the periphery. The result is that the system's state does not fly to infinity or settle at a point; instead, it converges to a closed orbit that encircles the unstable fixed point. This isolated, closed orbit is what we call a limit cycle.

This behavior represents a nonlinear oscillation. If we were to plot one of the system's variables (e.g., the firing rate of the excitatory population) against time, we would observe a stable, repeating wave. Unlike a simple sine wave from a linear system, the shape of this oscillation can be quite complex, reflecting the underlying nonlinear dynamics.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name"></span></p>

Limit cycles are not merely a mathematical curiosity; they appear in numerous models across science and engineering.

* **Van der Pol Oscillator:** A classic 2D textbook example that describes oscillations in a vacuum tube circuit.
* **Duffing Oscillator:** Another famous 2D example used to illustrate nonlinear oscillations and chaotic behavior.

A compelling biological application is modeling short-term or working memory.

* **The Model:** Consider a task where a subject is shown a stimulus, which is then removed. After a delay, the subject must make a choice based on the remembered stimulus.
* **Neural Correlate:** During the delay period, populations of neurons in frontal brain regions are observed to jump to a high-firing state, often called an "upstate," and maintain this activity until it is no longer needed. They then "hop down" to a baseline state.
* **Dynamical Systems Interpretation:** This phenomenon can be modeled as a bistable system with two point attractors (a "downstate" and an "upstate"). The stimulus effectively "kicks" the system into the basin of attraction of the upstate, where it remains, thus "holding" the information online. This concept can be extended where different attractors correspond to different memories. The central idea is that information is maintained through the stable states of a dynamical system.

</div>

#### Damped Oscillations

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Damped Oscillations)</span></p>

A **damped oscillator** is an oscillating system in which a force or term acts to reduce the motion over time by removing energy.

In differential-equation form, a basic damped oscillator is

$$m\ddot x + c\dot x + kx = 0,$$

where

* $x(t)$ is the displacement,
* $m\ddot x$ is inertia,
* $kx$ is the restoring force,
* $c\dot x$ is the **damping term**.

The damping term opposes the motion and causes the amplitude of oscillation to decrease with time.

So a compact definition is:

> A damped oscillator is an oscillator whose motion is accompanied by energy dissipation, usually modeled by a term depending on velocity, so that the oscillations decay over time.

In nonlinear systems, “damping” can depend on $x$ or $\dot x$, and then it may not always reduce motion everywhere. That is why nonlinear damping can even produce a limit cycle. But the basic meaning is still: **a mechanism that changes the oscillation by dissipating or regulating energy**.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Damping Term)</span></p>

It usually means a term proportional to the **velocity** that either removes or injects energy.

For a second-order oscillator, you often see something like

$$x'' + cx' + f(x)=0$$

or, in the van der Pol form,

$$x''-\mu(1-x^2)x' + x = 0.$$

Here the term involving $x'$ is called the **damping term**.

What it does:

* if the coefficient of $x'$ is **positive**, it opposes motion and **dissipates energy**
* if the coefficient is **negative**, it effectively **adds energy**
* if the coefficient depends on $x$ or amplitude, the damping is **nonlinear**

Why this matters for a limit cycle:

A stable limit cycle often appears when the system behaves like this:

* for **small oscillations**, damping is negative, so the motion gets amplified
* for **large oscillations**, damping is positive, so the motion gets reduced

So the trajectory neither dies out nor blows up forever. It settles to one preferred amplitude: that is the **stable limit cycle**.

Example: van der Pol oscillator

$$x''-\mu(1-x^2)x' + x = 0$$

Rewrite the damping part as

$$-\mu(1-x^2)x' = \mu(x^2-1)x'.$$

Now look at the sign:

* when $\lvert x\rvert <1$, $x^2-1<0$: this acts like **negative damping**, so energy is pumped in
* when $\lvert x\rvert >1$, $x^2-1>0$: this acts like **positive damping**, so energy is removed

That balance creates a stable periodic oscillation.

So the damping term is not some special “limit-cycle term.” It is a term controlling **amplitude growth or decay**, and in nonlinear systems it can be exactly what produces a limit cycle.

A useful intuition:

* **ordinary positive damping**: oscillation shrinks to an equilibrium
* **nonlinear damping with sign change**: oscillation is pushed toward a specific finite size, giving a limit cycle

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Damped Oscialltions in Physics)</span></p>

<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/damped_oscillation_physics.jpg' | relative_url }}" alt="a" loading="lazy">
  <!-- <figcaption>One-parameter family of closed orbits</figcaption> -->
</figure>

<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/Damped_spring.gif' | relative_url }}" alt="a" loading="lazy">
  <figcaption>Underdamped spring–mass system</figcaption>
</figure>

</div>

<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/VanDerPol_randomly_chosen_initial_conditions_are_attracted_to_stable_orbit.gif' | relative_url }}" alt="a" loading="lazy">
  <figcaption>Van Der Pol Oscillator. Randomly chosen initial conditions are attracted to stable orbit</figcaption>
</figure>

<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/VanDerPolPhaseSpace.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>Stable limit cycle (shown in bold) in phase space for the Van der Pol oscillator</figcaption>
</figure>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Generality of Dynamical Systems Theory)</span></p>

One might question if these simple models are "expressive enough" to capture the complexity of natural phenomena. A core strength of dynamical systems theory is its generality. It describes the evolution of a system over time purely from the perspective of its dynamics. Whether the underlying substrate is a set of neurons, a physical circuit, or an ecological population, the system is described by differential equations. The principles of attractors, repellors, and orbits provide a universal language for understanding the emergent behavior, regardless of the system's specific implementation.

</div>

#### Formal Definition of a Limit Cycle

We can now state the formal mathematical definition of a limit cycle.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Limit Cycle)</span></p>

A **limit cycle**, $\Gamma$, is an isolated closed orbit in the state space of a dynamical system.

A trajectory $\mathbf{x}(t) = \phi(t, \mathbf{x}_0)$ is a closed orbit if there exists a period $T > 0$ such that $\mathbf{x}(t+T) = \mathbf{x}(t)$ for all $t$. For any point $\mathbf{x}_0$ on the limit cycle $\Gamma$, its trajectory must satisfy:

$$\phi(t+T, \mathbf{x}_0) = \phi(t, \mathbf{x}_0)$$

where $T$ is the smallest positive number for which this relation holds. The term "isolated" means that there are no other closed orbits in the immediate neighborhood of $\Gamma$.

</div>

#### Stability of Limit Cycles

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Stability of Limit Cycles)</span></p>

Just like fixed points, limit cycles can be classified by their stability. The stability determines whether trajectories near the cycle converge to it, diverge from it, or exhibit a combination of behaviors.

* **Stable Limit Cycle:** Trajectories starting in a neighborhood of the cycle, both inside and outside, converge to the limit cycle as $t \to \infty$. This limit cycle is an attractor. The Wilson-Cowan example described above features a stable limit cycle.
* **Unstable Limit Cycle:** Trajectories starting in a neighborhood of the cycle are repelled from it as $t \to \infty$. An unstable cycle acts as a repellor or a boundary between basins of attraction.
* **Half-Stable (or Saddle) Cycle:** Trajectories approach the cycle from one direction (e.g., from the inside) but are repelled from it in another direction (e.g., from the outside).

</div>

<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/stable_unstable_halfstable_limit_cycle.png' | relative_url }}" alt="a" loading="lazy">
  <!-- <figcaption>One-parameter family of closed orbits</figcaption> -->
</figure>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Half-Stable "Saddle" Cycle Around the Origin)</span></p>

Consider the following system in 

**In Polar:**

$$
\begin{cases}
\dot{r} = r(r^2 - 1)^2 \\
\dot{\theta} = 1
\end{cases}
$$

**In Cartesian:**

$$
\begin{cases}
\dot{x} = (x^2 + y^2 - 1)^2 x - y \\
\dot{y} = (x^2 + y^2 - 1)^2 y + x
\end{cases}
$$

**Properties:**
- **$r = 1$ is a cycle**: $\dot{r}\lvert_{r=1} = 1 \cdot 0 = 0$ and $\dot{\theta} = 1 > 0$ (counter-clockwise rotation).
- **Half-stable**: For $r < 1$: $(r^2-1)^2 > 0$, so $\dot{r} > 0$ (approach from inside). For $r > 1$: $(r^2-1)^2 > 0$, so $\dot{r} > 0$ (repel from outside).
- **Origin is the only FP**: At $(0,0)$, $\dot{x} = 0, \dot{y} = 0$. The origin has $J = [[1,-1],[1,1]]$, eigenvalues $1 \pm i$ (unstable spiral).

> The cycle is **semistable**: attracting from inside, repelling from outside.

<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/half_stable_cycle.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>Half-Stable "Saddle" Cycle Around the Origin</figcaption>
</figure>

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

The formal definition of stability for a limit cycle is analogous to the "Lyapunov-like" definition used for equilibria. For a cycle to be stable, any trajectory starting within a small neighborhood ($\epsilon$) of the cycle must remain within that neighborhood for all future time. For it to be asymptotically stable, the trajectory must also converge to the cycle as $t \to \infty$.

</div>

#### Methods for Detecting Limit Cycles

Proving the existence of a limit cycle is generally more complex than finding a fixed point (which only requires solving $\dot{\mathbf{x}} = 0$). We will introduce several powerful concepts for analyzing systems in the 2D plane.

##### Lyapunov Functions

Even in systems that are not related to mechanics, it is sometimes possible to define an energy-like quantity that decreases as the system evolves. Such a quantity is called a **Lyapunov function**. If a Lyapunov function exists, then the system cannot have closed or periodic orbits.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Lyapunov function)</span></p>

Consider a system

$$\dot{\mathbf{x}}=\mathbf{f}(\mathbf{x})$$

with a fixed point at $\mathbf{x}^\ast$. Suppose there exists a continuously differentiable real-valued function $V(\mathbf{x})$ called **Lyapunov function** such that:

1. $V(\mathbf{x})>0$ for every $\mathbf{x}\neq \mathbf{x}^\ast$, and $V(\mathbf{x}^\ast)=0$.
   * In this case, $V$ is called **positive definite**.

2. $\dot V < 0$ for every $\mathbf{x}\neq \mathbf{x}^\ast$.
   * This means that along every trajectory, the value of $V$ strictly decreases, so the motion always goes “downhill” toward $\mathbf{x}^\ast$.

<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/lyapunov_function.png' | relative_url }}" alt="a" loading="lazy">
  <!-- <figcaption>One-parameter family of closed orbits</figcaption> -->
</figure>

The intuition is that all trajectories move steadily downward on the surface defined by $V(\mathbf{x})$, eventually reaching the minimum at $\mathbf{x}^\ast$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Fixed point stability using Lyapunov function)</span></p>

Under these conditions, the fixed point $\mathbf{x}^\ast$ is **globally asymptotically stable**: no matter where the system starts, the solution $\mathbf{x}(t)$ approaches $\mathbf{x}^\ast$ as $t\to\infty$. In particular, the system cannot contain any closed orbits.

A solution cannot get stuck anywhere else. If it did, then $V$ would stop changing there. But this is impossible because, by assumption, $\dot V<0$ everywhere except at $\mathbf{x}^\ast$.

* if $\dot V(x)\le 0$, the equilibrium is **stable**
* if $\dot V(x)<0$ for all $x\neq 0$, the equilibrium is **asymptotically stable**

</div>

<div class="gd-grid">
  <figure>
    <img src="{{ '/assets/images/notes/dynamical-systems/lyapunov-functions-1.png' | relative_url }}" alt="a" loading="lazy">
    <!-- <figcaption>Vector field on the line</figcaption> -->
  </figure>
  
  <figure>
    <img src="{{ '/assets/images/notes/dynamical-systems/Geometric-meaning-of-the-Lyapunov-function.png' | relative_url }}" alt="a" loading="lazy">
    <!-- <figcaption>Vector field on the circle</figcaption> -->
  </figure>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Drawback of Lyapunov functions method)</span></p>

* Unfortunately, **there is no general procedure for finding Lyapunov functions**. 
* In practice, constructing one often requires creativity or insight, although sometimes it is possible to work backward from the desired properties. In some cases, sums of squares provide useful candidates.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Lyapunov Function Application)</span></p>

Use a Lyapunov function to show that the system

$$\dot x = -x + 4y, \qquad \dot y = -x - y^3$$

has no closed orbits.

**Solution:**

Consider the candidate Lyapunov function

$$V(x,y)=x^2 + ay^2,$$

where $a$ is a constant to be chosen.

To check whether this works, compute its derivative along trajectories:

$$\dot V = 2x\dot x + 2ay\dot y.$$

Substituting the system equations gives

$$\dot V = 2x(-x+4y) + 2ay(-x-y^3).$$

Expanding,

$$\dot V = -2x^2 + (8-2a)xy - 2ay^4.$$

Now choose $a=4$. Then the mixed term $xy$ disappears, and we obtain

$$\dot V = -2x^2 - 8y^4$$

With this choice,

$$V(x,y)=x^2+4y^2$$

which is positive for every $(x,y)\neq(0,0)$, and $V(0,0)=0$. Also,

$$\dot V < 0 \qquad \text{for all } (x,y)\neq (0,0)$$

Therefore $V$ is a Lyapunov function. It follows that the system has no closed orbits. In fact, every trajectory approaches the origin as $t\to\infty$.

<div class="pmf-grid">
<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/lyapunov_function_example1.png' | relative_url }}" alt="a" loading="lazy">
</figure>
<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/lyapunov_function_example2.png' | relative_url }}" alt="a" loading="lazy">
</figure>
<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/lyapunov_function_example3.png' | relative_url }}" alt="a" loading="lazy">
</figure>
</div>

</div>

##### The Poincaré-Bendixson Theorem and Trapping Regions

Before, we learned how to show that closed orbits do **not** exist. Now we look at the opposite question: how can we prove that closed orbits **do** exist in some systems? One of the main tools for this is the **Poincaré–Bendixson Theorem**. It is very important in planar nonlinear dynamics because it also shows that chaos cannot occur in two-dimensional phase planes.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Poincaré-Bendixson Theorem Idea)</span></p>

Assume that:

1. $R$ is a closed and bounded region in the plane.
2. $\dot{\mathbf{x}}=\mathbf{f}(\mathbf{x})$ is a continuously differentiable vector field on an open set containing $R$.
3. $R$ contains no equilibrium points.
4. There is a trajectory $C$ that starts in $R$ and remains in $R$ for all future time.

Then one of two things must happen:

* either $C$ itself is a closed orbit,
* or $C$ approaches a closed orbit as $t \to \infty$.

So in either case, the **region $R$ contains at least one closed orbit**.

<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/PoincareBendixsonTheoremIdea.png' | relative_url }}" alt="a" loading="lazy">
</figure>

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Trapping Region)</span></p>

A **trapping region** is a closed set in the phase space such that any trajectory that starts inside the region remains inside for all future time. Critically, the vector field on the boundary of this region must point inwards everywhere.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Trapping Region Construction)</span></p>

When using the Poincaré–Bendixson theorem, conditions (1)–(3) are usually easy to check. The difficult part is condition (4): how do we know there is a trajectory $C$ that stays inside the region forever?

A common method is to build a **trapping region** $R$. This is a closed, connected region whose boundary has the property that the vector field points **inward** at every point. Because of this, any trajectory that enters $R$ cannot leave it, so trajectories in (R) remain confined there for all future time.

If we can also make sure that $R$ contains **no equilibrium points**, then the Poincaré–Bendixson theorem tells us that $R$ must contain a **closed orbit**.

In practice, the theorem can still be hard to use. It becomes much easier in special cases, for example when the system can be written simply in **polar coordinates**.

<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/TrappingRegion.png' | relative_url }}" alt="a" loading="lazy">
</figure>

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Poincaré Map is better or more systematic analysis of limit cycles)</span></p>

For a more systematic analysis of limit cycles, especially their stability, one can use the Poincaré map. The core idea is to convert the continuous flow around the cycle into a discrete map. By analyzing the properties of this map, one can deduce properties of the limit cycle itself. This topic will be revisited in greater detail later.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Poincaré-Bendixson theorem)</span></p>

Suppose you have a system of continuously differentiable differential equations in the plane ($\mathbb{R}^2$):

$$\dot{x} = f(x,y)$$

$$\dot{y} = g(x,y)$$

Let $R$ be a closed, bounded subset of the plane. If a trajectory $C$ is confined to $R$ for all time $t \ge 0$, then the long-term behavior of $C$ (its $\omega$-limit set) must be exactly one of the following three things:
* **A fixed point** (equilibrium point).
* **A closed orbit** (a periodic trajectory or limit cycle).
* **A network of fixed points joined by trajectories** (homoclinic or heteroclinic orbits).

</div>

### Topological Tools: Index Theory

Another powerful tool for analyzing 2D vector fields is index theory. It uses topological properties to constrain the types and numbers of fixed points that can exist within a closed curve.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Winding Number)</span></p>

The core concept is the winding number. Imagine walking along a closed curve $C$. As you walk, consider a vector pointing from your position to a fixed point inside the curve. The winding number is the total number of full counter-clockwise rotations this vector makes during your complete circuit. Because the curve is closed, this must be an integer.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Index of a Fixed Point)</span></p>

The **index of a fixed point** is a property of the vector field surrounding it. To calculate it, we draw a small closed curve $C$ around the fixed point and traverse it once counter-clockwise. We observe the direction of the vectors of the vector field $\dot{\mathbf{x}}$ at each point on $C$. The index is the total number of counter-clockwise revolutions that the vector field itself makes.

Let's calculate the index for different types of fixed points:

* **Stable Node:** The vector field points inwards from all directions. As we move counter-clockwise around the curve $C$, the vector field also rotates counter-clockwise by one full turn.
  * Index = $+1$
* **Unstable Node:** The vector field points outwards in all directions. As we move counter-clockwise around $C$, the vector field again rotates counter-clockwise by one full turn.
  * Index = $+1$
* **Saddle Point:** The flow moves inwards along the stable manifold and outwards along the unstable manifold. Let's trace the vector's rotation as we move around the curve $C$:
  * As we start on the right and move up (counter-clockwise), the vector field points mostly down and left. As we cross the stable manifold, the vector points... (The analysis from the source context is incomplete here).

</div>

### Topological Properties of Orbits in the Plane

In the study of two-dimensional dynamical systems, we often encounter closed orbits known as limit cycles. The behavior of the vector field around these orbits and the equilibria they enclose are not arbitrary; they are governed by strict topological rules. This chapter introduces the concept of the index of a closed curve and a fundamental theorem that constrains the types of equilibria a limit cycle can contain.

#### The Index of a Closed Curve

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Index of a Closed Curve)</span></p>

The **index of a closed curve** (also known as the Poincaré index) quantifies the total rotation of the vector field as one traverses the curve. By convention, we traverse the closed curve in a counter-clockwise direction. The index is an integer value representing the number of net rotations the vectors make.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

Imagine walking along a closed path on a field of arrows (the vector field). As you walk, you keep track of the direction the nearest arrow is pointing. The index tells you how many full circles the arrow's direction has turned by the time you return to your starting point. A positive index means the vector field rotates in the same direction as your path (counter-clockwise), while a negative index means it rotates in the opposite direction (clockwise).

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name"></span></p>

* Index of $+1$: If we draw a closed curve around a stable or unstable equilibrium point (like a node or a focus), the vectors of the field will make one full counter-clockwise rotation as we move counter-clockwise along the curve. This configuration has an index of $+1$.
* Index of $-1$: Consider a saddle point. If we trace a closed curve around it in a counter-clockwise direction, the vector field will appear to rotate one full turn in the clockwise direction. By convention, this gives the curve an index of $-1$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Additivity of Indices)</span></p>

For a closed curve, often called a Jordan curve, that encloses multiple regions, its total index is the sum of the indices of its subcurves.

</div>

#### A Theorem on Limit Cycles and Equilibria

The concept of the index leads to a powerful and restrictive theorem concerning limit cycles in the plane.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(The Poincaré–Bendixson Index Theorem (variant))</span></p>

For any limit cycle in a two-dimensional (2D) continuous dynamical system, the sum of the indices of all equilibria (fixed points) contained within the limit cycle must be exactly $+1$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

This theorem provides a profound insight into the "topological relationships between different objects in state space." It acts as a strict rule for what configurations are possible within a planar system.

* A limit cycle can enclose a single stable or unstable equilibrium (node or focus), as these have an index of $+1$.
* A limit cycle cannot enclose only a single saddle point, because a saddle has an index of $-1$, which violates the theorem. This explains why we do not see stable orbits circulating around a solitary saddle point.
* A limit cycle can enclose more complex configurations, as long as the indices sum to $+1$. For example, a limit cycle can contain two stable points (each with index $+1$) and one saddle point (with index $-1$), because the total index is $(+1) + (+1) + (-1) = +1$.

</div>

#### Topological Constraints in System Reconstruction

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Practical Implications)</span></p>

These topological relationships are not merely mathematical curiosities; they have significant practical importance, especially when attempting to reconstruct dynamical systems from observational data. In real-world scenarios, we may not have observed every component or state of a system. Knowledge of these underlying topological rules provides powerful constraints on what the complete system can look like.

If we observe an oscillation that appears to be a limit cycle, this theorem immediately tells us what kind of fixed-point structures must lie inside it. We know, for instance, that there cannot be just a saddle point inside. This a priori knowledge helps guide the modeling process and allows us to infer the existence of unobserved features. While this specific theorem is for 2D systems, similar topological principles and constraints exist in higher-dimensional systems as well, making them a crucial tool for understanding complex dynamics.

</div>

### Timescale Separation and Bifurcation Analysis

This chapter explores a powerful technique for analyzing complex dynamical systems that evolve on different timescales. By separating the system's variables into slow and fast categories, we can gain profound insights into its behavior, including phenomena like bursting oscillations. The primary tool for this analysis is the bifurcation graph.

#### The Method of Timescale Separation

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

Many complex systems in nature, from neural networks to climate models, feature variables that change at vastly different rates. The core idea behind timescale separation is to simplify the analysis by first understanding the behavior of the fast-moving variables while treating the slow-moving variables as temporarily constant.

The general technique involves the following steps:

1. Segregate the system variables into distinct slow and fast groups.
2. Analyze the fast subsystem: Consider the dynamics of the fast variables alone, treating the slow variables as fixed parameters.
3. Construct a bifurcation graph: Plot the stable and unstable solutions (or "objects") of the fast subsystem as a function of the slow "parameter."
4. Synthesize: Understand the behavior of the full system by envisioning it as a point that moves along this bifurcation graph as the slow variable evolves according to its own dynamics.

</div>

#### Introduction to Bifurcation Graphs

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Bifurcation Graph)</span></p>

A **bifurcation graph** is a diagram that plots the state of a system's stable and unstable objects (such as fixed points and limit cycles) as a function of a chosen system parameter.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Homoclinic Orbit)</span></p>

A **homoclinic orbit** (or **homoclinic connection**) is a trajectory in a dynamical system that joins a saddle equilibrium point to itself. The event where the expanding limit cycle collides with the saddle point creates such an orbit. The bifurcation that occurs at this point is known as a homoclinic bifurcation.

</div>

