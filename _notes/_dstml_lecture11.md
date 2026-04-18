## Lecture 11

This lecture provides a comprehensive study of how Recurrent Neural Networks (RNNs) can be used as surrogate models for dynamical systems. We cover the goals and evaluation of dynamical systems reconstruction, specialized training methodologies for chaotic systems, the analytically tractable Piecewise Linear RNN (PLRNN), bifurcation phenomena during training, flow operator properties, Reservoir Computing, Autoencoders, and the integration of SINDy for latent dynamics discovery.

### Dynamical Systems Reconstruction

The primary objective in reconstructing dynamical systems using RNNs is to identify an approximate flow map, $\phi^*$, that effectively models the underlying dynamics of a system.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Dynamical System Reconstruction)</span></p>

A **dynamical system reconstruction** aims to find an approximate flow map $\phi^*$, typically modeled through an RNN, that is **topologically conjugate** to the underlying dynamical system described by a flow operator $\phi$ on a domain $D$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Beyond Prediction)</span></p>

In a scientific context, we do not view these RNNs merely as black-box prediction models. Instead, we treat them as **surrogate models**. The goal is to capture the dynamical properties of the system to gain insight into the underlying mechanisms governing the observed data.

</div>

#### Assessing Reconstruction Quality

When evaluating how well an RNN has reconstructed a dynamical system, we look beyond simple error metrics and focus on geometrical, temporal, and dynamical properties.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Key Performance Measures)</span></p>

* **Geometrical Properties:** We assess the overlap in state space. One common measure is the Kullback-Leibler (KL) divergence applied to the distribution of states in the state space.
* **Temporal Properties:** To quantify how well the long-term temporal properties are matched, we use the Hellinger distance defined on the power spectra of the true and generated signals.
* **Dynamical Properties:** We calculate and compare the Lyapunov exponents (specifically the maximal Lyapunov exponent, $\lambda_{\max}$) of both the original system and the reconstructed model.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Attractor Localization)</span></p>

In an ideal reconstruction, the RNN can detect features of the underlying system that were not explicitly present in the training trajectories. For instance, a model trained only on trajectories residing on an attractor might still accurately localize the system’s equilibria (fixed points).

</div>

### Training Methodologies for Chaotic Systems

Training RNNs on chaotic systems presents significant challenges, most notably the exploding and vanishing gradient problem, which is often inevitable when dealing with underlying chaotic dynamics. To mitigate these issues, specialized training techniques are employed. These techniques build on the teacher forcing and generalized teacher forcing methods discussed in Lecture 10.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Teacher Forcing)</span></p>

**Teacher Forcing** is a training technique where the latent state $z_t$ of the RNN is replaced by an estimate $\hat{z}_t$ derived from the actual data.

* The estimate $\hat{z}_t$ is obtained by inverting (or pseudo-inverting) the decoder/observation function $G$.
* This replacement occurs every $\tau$ time steps.
* The interval $\tau$ is chosen based on the Lyapunov spectrum or the maximal Lyapunov exponent of the system.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Generalized Teacher Forcing)</span></p>

**Generalized Teacher Forcing** is a refinement where, instead of a total replacement, a weighted average is used to update the state:

$$z_t^{\text{updated}} = \alpha \, z_t^{\text{forward}} + (1 - \alpha)\,\hat{z}_t^{\text{data}}$$

where:
* $z_t^{\text{forward}}$ is the state predicted by the RNN.
* $\hat{z}_t^{\text{data}}$ is the estimate inferred from the data.
* $\alpha$ is a weighting factor.

</div>

<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/GeneralizedTeacherForcingTrainingIdea.png' | relative_url }}" alt="Newton–Raphson iteration animation" loading="lazy">
</figure>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Optimization of $\alpha$)</span></p>

The parameter $\alpha$ can be adjusted optimally by considering the Singular Value Decomposition (SVD) of the underlying Jacobian matrix of the system. Recall from Lecture 10 that the optimal choice is 

$$\alpha_t = 1 - \frac{1}{\sigma_{\max}(G_t)}$$

which keeps gradient magnitudes controlled.

</div>

<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/GeneralizedTeacherForcing.png' | relative_url }}" alt="Newton–Raphson iteration animation" loading="lazy">
</figure>

### Piecewise Linear Recurrent Neural Networks (PLRNN)

To ensure mathematical tractability and interpretability, we often utilize Piecewise Linear Recurrent Neural Networks (PLRNN).

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(PLRNN State Equations)</span></p>

The latent variables $z_t$ in a **PLRNN** evolve according to the following multivariate map:

$$z_t = A z_{t-1} + W \phi(z_{t-1}) + h$$

where:
* $A$ is a weight matrix (often diagonal).
* $W$ is the weight matrix for the non-linear term.
* $h$ is a bias term.
* $\phi$ is the Rectified Linear Unit (ReLU) activation function, defined as $\phi(z) = \max(0, z)$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Observation Function)</span></p>

The latent states are linked to the actual observations $x_t$ through an **observation function** $G$:

$$x_t = G(z_t;\, \lambda)$$

where $\lambda$ represents trainable parameters.

</div>

#### Mathematical Analysis of Trained Models

The piecewise linear nature of the PLRNN allows us to reformulate the system into a more analytically accessible form.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Matrix Representation of ReLU)</span></p>

The $\max$ operator in the PLRNN can be rewritten as a time-dependent diagonal matrix $D_{t-1}$, allowing the system to be expressed as an affine mapping:

$$z_t = (A + W D_{t-1})\, z_{t-1} + h$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proof</span><span class="math-callout__name">(Construction of the Indicator Matrix)</span></p>

1. Let $\phi(z) = \max(0, z)$ be the ReLU activation function.
2. Define a diagonal matrix $D_t$ such that the $i$-th element on the diagonal corresponds to the $i$-th component of the state vector $z_t$.
3. Set the diagonal entries as:

   $$D_{ii} = \begin{cases} 1 & \text{if } z_i > 0 \\ 0 & \text{if } z_i \leq 0 \end{cases}$$

4. Substituting this into the state equation: 
   
   $$W\phi(z_{t-1}) = W D_{t-1} z_{t-1}$$

5. The full state equation becomes:

$$z_t = A z_{t-1} + W D_{t-1} z_{t-1} + h = (A + W D_{t-1})\, z_{t-1} + h$$

This confirms that for any given state $z$, the system behaves as a linear mapping specific to the "quadrant" or region of state space defined by the signs of the components of $z$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Fixed Points of the Map)</span></p>

A **fixed point $z^\ast$ of the map** is a state that remains constant under the iteration of the map, such that

$$z_t = z_{t-1} = z^\ast$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proof</span><span class="math-callout__name">(Analytical Extraction of Fixed Points)</span></p>

To find the fixed point $z^\ast$, we assume the system has settled into a specific linear region defined by $D^\ast$:

1. Start with the steady-state equation: 
   
   $$z^* = A z^* + W D^* z^* + h$$

2. Group the terms involving $z^*$:
   
   $$z^* - A z^* - W D^* z^* = h$$

3. Factor out $z^\ast$: 
   
   $$(I - A - W D^*)\, z^* = h$$

4. Solve for $z^\ast$:

   $$z^* = (I - A - W D^*)^{-1} h$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Consistency Constraint)</span></p>

While the above formula provides a candidate for a fixed point, it is not purely analytical. One must verify that the resulting $z^\ast$ is **consistent** with the matrix $D^\ast$. That is, the signs of the components of the calculated $z^\ast$ must actually produce the diagonal entries of $D^\ast$ used in the calculation.

The reason is that $D^\ast$ is **not a free parameter** — it is determined by $z^\ast$ itself. Recall that $D^\ast_{ii} = 1$ if $z^\ast_i > 0$ and $D^\ast_{ii} = 0$ if $z^\ast_i \leq 0$, which creates a circular dependency: to solve for $z^\ast$ we need $D^\ast$, but to know $D^\ast$ we need $z^\ast$.

The formula $(I - A - WD^\ast)^{-1} h$ resolves this by **guessing** a region (i.e., a particular $D^\ast$) and solving the corresponding affine system. However, the PLRNN partitions state space into $2^M$ regions, each defined by which components of $z$ are positive or negative, and in each region the system is governed by a **different** affine map. The candidate $z^\ast$ is only a true fixed point if it lies inside the region where the assumed map actually applies.

For instance, if we assume $D^\ast_{ii} = 0$ (i.e., $z^\ast_i \leq 0$), but the solution yields $z^\ast_i > 0$, then in the real system $\max(0, z^\ast_i) = z^\ast_i \neq 0$, meaning the ReLU is active for component $i$. The actual dynamics at that point use $D_{ii} = 1$, not $0$, so the equation we solved does not govern the system at the computed $z^\ast$. Iterating the true map from this candidate would not return $z^\ast$ — it is a fixed point of a neighboring affine map, not of the PLRNN itself.

</div>

### Fixed Points and Periodic Orbits in RNNs

In the study of RNNs as dynamical systems, identifying the long-term behavior of the system—specifically its fixed points and periodic orbits (cycles)—is essential for understanding the model’s computational properties and its validity as a surrogate for real-world systems.

#### The Consistency Problem in Fixed Point Localization

To find an exact solution for a fixed point $Z^*$, we must ensure that the state of the system is consistent with the activation of its units. In many RNN architectures, the transition is governed by a diagonal matrix $D$ that represents the "on/off" state of the neurons (often associated with rectified linear units or similar activations).

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Consistency Condition)</span></p>

A candidate fixed point $Z^\ast$ and its associated configuration matrix $D^*$ are considered **consistent** if and only if:

$$D_{ii}^* = 1 \iff Z_i^* > 0$$

and $D_{ii}^* = 0$ otherwise.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Combinatorial Complexity)</span></p>

Finding a fixed point is fundamentally a combinatorial problem. Because each unit in a hidden layer of dimension $m$ can be either active or inactive, there are $2^m$ possible configurations for the matrix $D^*$. In low-dimensional spaces, one could exhaustively check every configuration, but this becomes computationally intractable as $m$ increases.

</div>

#### Mathematical Formulation of $k$-Cycles

Beyond individual fixed points, we are interested in cycles—sets of points that the system visits in a repeating sequence.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($k$-Cycle)</span></p>

A **$k$-cycle** is a set of $k$ distinct points $\lbrace Z_1^\ast, Z_2^\ast, \dots, Z_k^\ast \rbrace$ such that each point is a fixed point of the $k$-times iterated map $f^{(k)}$. That is:

$$Z_{m}^* = f^{(k)}(Z_{m}^*) \quad \text{for } m \in \lbrace 1, \dots, k \rbrace$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(The Iterated Map of an RNN)</span></p>

For a system defined by an affine transition of the form 

$$Z_t = (A + W D_{t-1})\, Z_{t-1} + h$$

the two-time iterated map is expressed as:

$$Z_t = (A + W D_{t-1}) \left[ (A + W D_{t-2})\, Z_{t-2} + h \right] + h$$

Expanding this, we obtain:

$$Z_t = (A + W D_{t-1})(A + W D_{t-2})\, Z_{t-2} + (A + W D_{t-1})\, h + h$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proof</span><span class="math-callout__name">(General $k$-Iteration)</span></p>

To derive the general form for a $k$-times iterated map, we apply the recursive rule repeatedly:

1. Let the Jacobian of the map be defined as 
   
   $$J_t = (A + W D_t)$$

2. For a $k$-step iteration from $Z_{t-k}$ to $Z_t$, the state is:

   $$Z_t = \left( \prod_{j=1}^{k} J_{t-j} \right) Z_{t-k} + \sum_{i=1}^{k-1} \left( \prod_{j=1}^{i} J_{t-j} \right) h + h$$

3. This resulting expression remains an affine map of the initial state $Z_{t-k}$. Thus, finding a $k$-cycle reduces to solving a linear system, provided the sequence of matrices $\lbrace D_{t-1}, \dots, D_{t-k} \rbrace$ is known and consistent with the resulting states.

</div>

#### Search Algorithms for Fixed Points and Cycles

Given the $2^m$ complexity of the combinatorial search, efficient numerical procedures are required to locate these points exactly rather than relying on approximations.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(The "Virtual Point" Heuristic)</span></p>

A highly efficient heuristic for finding fixed points and cycles involves the following steps:

1. **Initialization:** Start with an initial configuration matrix $D$.
2. **Candidate Computation:** Solve the affine equation to find a candidate solution $Z^*$ (a "virtual" fixed point).
3. **Consistency Check:** Verify if the signs of $Z^*$ match the configuration $D$.
4. **Update:** If inconsistent, use the configuration $D$ derived from the current $Z^*$ to initialize the next round.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Intuition Behind the Heuristic)</span></p>

The problem we had before is: finding a fixed point requires knowing $D^\ast$, which requires knowing $z^\ast$, which requires knowing $D^\ast$ — a chicken-and-egg problem. And brute-forcing all $2^m$ configurations is intractable.

The heuristic resolves this by turning it into an iterative refinement:

1. Guess any $D$ (e.g., all ones, or random).
2. Solve $(I - A - WD)^{-1}h = z^\ast$. This $z^\ast$ is called "virtual" because it's almost certainly not a true fixed point — it lives in the wrong region of state space (its signs don't match the $D$ we assumed).
3. Read off the signs of $z^\ast$ to get a new $D'$: set $D'_{ii} = 1$ where $z^\ast_i > 0$, and $0$ otherwise.
4. Repeat with $D'$: solve again, check signs again.

If at any step the signs of $z^\ast$ match the $D$ used to compute it, you've found a true fixed point. The "virtual" points along the way are not fixed points — they're guideposts that tell you which region to look in next.

The intuition is: even though $z^\ast$ is wrong, it "knows something" about where the true fixed point is. Its signs point toward the correct region, so each iteration corrects the configuration. Instead of blindly searching $2^m$ regions, the system steers itself.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Termination Condition)</span></p>

The algorithm does not run for a fixed number of iterations. It repeats until one of two outcomes:

- **Convergence:** the signs of the computed $z^\ast$ match the configuration $D$ used to produce it, meaning a true fixed point has been found.
- **Failure:** the algorithm cycles between configurations without settling, or a maximum iteration limit is reached.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Fixed Points vs. $k$-Cycles)</span></p>

As stated, the algorithm solves $(I - A - WD)^{-1} h$ for a candidate $z^\ast$, which is the fixed point equation for the **single-step** map. This finds **1-cycles** (fixed points) only. To find a $k$-cycle, one must apply the same heuristic to the $k$-times iterated map, which involves guessing a **sequence** of $k$ configuration matrices $(D_{t-1}, \dots, D_{t-k})$ rather than a single $D$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Refinement for $k$-Cycles)</span></p>

Extending the heuristic to $k$-cycles raises the question of how to refine all $k$ matrices simultaneously, since solving the $k$-iterated fixed point equation yields only **one** cycle point $Z_1^\ast$. The answer is to **forward-propagate** through the single-step map to recover all intermediate states:

$$Z_2^\ast = (A + W D_1)\, Z_1^\ast + h, \quad Z_3^\ast = (A + W D_2)\, Z_2^\ast + h, \quad \dots, \quad Z_1^\ast = (A + W D_k)\, Z_k^\ast + h$$

With all $k$ states in hand, each matrix is updated from its corresponding state:

$$D_j \leftarrow \operatorname{signs}(Z_j^\ast) \quad \text{for } j = 1, \dots, k$$

The procedure then repeats: solve the $k$-iterated fixed point equation with the updated sequence $(D_1, \dots, D_k)$, forward-propagate to obtain all cycle points, check consistency for every pair $(Z_j^\ast, D_j)$, and update if needed.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Efficiency of the Virtual Point Heuristic)</span></p>

While the theoretical worst-case remains combinatorial, this heuristic often behaves linearly in time relative to the dimension. For certain matrix conditions, it can even be proven to converge in at most linear time. This efficiency is crucial for making RNNs "tractable" or "interpretable," allowing researchers to use them as surrogate systems to analyze underlying real-world data.

</div>

### Training Dynamics and Bifurcation Analysis

The process of training a neural network is itself a dynamical system. When we update parameters using gradient descent, we are moving through a parameter space that can fundamentally change the qualitative behavior of the network’s internal dynamics.

#### Optimization as a Dynamical System

Consider a standard gradient descent update rule:

$$\theta_{\text{next}} = \theta_{\text{prev}} - \eta \nabla L(\theta_{\text{prev}})$$

where $\theta$ represents the parameters (weights $W$, biases $h$) and $L$ is the loss function.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Dynamical Phenomena in Training)</span></p>

Because the training process is a recursive update, it is subject to all dynamical phenomena:

* **Oscillations:** The parameters may bounce around an optimum.
* **Chaos:** The training path may become unpredictable.
* **Attractors:** The system may converge to stable fixed points.
* **Information Loss:** If the system converges too strongly to a fixed point, it may lose the gradient information required to learn the underlying system.

</div>

#### Case Study: The Single-Unit RNN

To understand how parameter changes affect system behavior, we examine a one-unit RNN with a sigmoid activation function $\phi$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(1-Unit Scalar System)</span></p>

Let $W$ and $Z$ be scalars. The system is defined by:

$$Z_t = \phi(W Z_{t-1} + h)$$

As we vary the parameters $W$ (weight) and $h$ (bias), the system undergoes various bifurcations:

* **Varying $h$:** Changing the bias shifts the sigmoid function along the $Z_{t-1}$ axis.
* **Varying $W$:** Changing the weight alters the slope (steepness) of the function.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Pitchfork Bifurcation in RNNs)</span></p>

If the sigmoid function $\phi$ is perfectly symmetric around its inflection point, increasing the weight $W$ can lead to a **pitchfork bifurcation**.

* Initially, the system has a single stable fixed point.
* As $W$ increases and the slope becomes steeper, two new stable fixed points simultaneously appear while the original fixed point becomes unstable.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Bifurcation Visualization)</span></p>

This is visualized in a bifurcation graph where the stable states are plotted as a function of $W$. Understanding these transitions is vital because moving through a bifurcation during training can radically change the loss landscape and the network’s ability to represent the target system.

</div>

<div id="rb-wrap" style="margin:2em auto;max-width:1120px;">
  <h4 style="text-align:center;margin:0 0 .2em;">Interactive: Pitchfork Bifurcation in a 1-Unit RNN</h4>
  <p style="text-align:center;color:#888;font-size:.82em;margin:0 0 .5em;">
    \(Z_t = \sigma(W Z_{t-1} + h)\) with \(\sigma(x)=\frac{1}{1+e^{-x}}\). With \(h=-W/2\) the fixed point sits at the sigmoid's inflection point (\(z^*=0.5\)). Pitchfork at \(W=4\) where slope \(W\sigma'(0)=W/4=1\). Uncheck "Symmetric" and set \(h\) to break symmetry.
  </p>
  <div style="display:flex;justify-content:center;gap:14px;">
    <div style="text-align:center;">
      <div style="font-size:.85em;font-weight:600;margin-bottom:3px;">Sigmoid vs bisectrix</div>
      <canvas id="rb2cw" width="520" height="520" style="border:1px solid #ddd;border-radius:3px;"></canvas>
    </div>
    <div style="text-align:center;">
      <div style="font-size:.85em;font-weight:600;margin-bottom:3px;">Bifurcation diagram (z* vs W)</div>
      <canvas id="rb2bd" width="520" height="520" style="border:1px solid #ddd;border-radius:3px;cursor:col-resize;"></canvas>
    </div>
  </div>
  <div style="display:flex;align-items:center;justify-content:center;gap:8px;margin-top:10px;flex-wrap:wrap;">
    <span style="font-size:.85em;font-family:serif;">W =</span>
    <input type="range" id="rb2w" min="0.5" max="8" step="0.01" value="3" style="width:260px;">
    <span id="rb2wv" style="font-size:.85em;font-family:serif;min-width:35px;">3.00</span>
    <label style="margin-left:14px;font-size:.82em;font-family:serif;cursor:pointer;">
      <input type="checkbox" id="rb2sym" checked> Symmetric (h = −W/2)
    </label>
    <span style="font-size:.85em;font-family:serif;margin-left:8px;">h =</span>
    <input type="range" id="rb2h" min="-4" max="1" step="0.01" value="-1.5" style="width:140px;" disabled>
    <span id="rb2hv" style="font-size:.85em;font-family:serif;min-width:40px;">-1.50</span>
  </div>
  <div id="rb2info" style="text-align:center;font-size:.82em;margin-top:.4em;font-family:serif;color:#555;"></div>
</div>

<script>
(function(){
  var S=520;
  var Wp=3,hp=-1.5,sym=true,dragBif=false;

  var cwEl=document.getElementById('rb2cw'),bdEl=document.getElementById('rb2bd');
  var CW=cwEl.getContext('2d'),BD=bdEl.getContext('2d');
  var wS=document.getElementById('rb2w'),wV=document.getElementById('rb2wv');
  var hS=document.getElementById('rb2h'),hV=document.getElementById('rb2hv');
  var symCb=document.getElementById('rb2sym');

  function getH(){return sym?-Wp/2:hp;}
  function sigma(x){if(x>500)return 1;if(x< -500)return 0;return 1/(1+Math.exp(-x));}
  function fmap(z,w,h){return sigma(w*z+h);}
  function fmapD(z,w,h){var s=sigma(w*z+h);return w*s*(1-s);}

  function findFP(w,h){
    var fps=[],N2=800,zL=-0.15,zH=1.15;
    if(Math.abs(fmap(0.5,w,h)-0.5)<1e-8)fps.push({z:0.5,slope:fmapD(0.5,w,h)});
    var prev=fmap(zL,w,h)-zL;
    for(var i=1;i<=N2;i++){
      var z=zL+(zH-zL)*i/N2,val=fmap(z,w,h)-z;
      if(prev*val<0){
        var zn=zL+(zH-zL)*(i-.5)/N2;
        for(var k=0;k<40;k++){var fv=fmap(zn,w,h)-zn,fd=fmapD(zn,w,h)-1;if(Math.abs(fd)<1e-14)break;zn-=fv/fd;if(Math.abs(fv)<1e-13)break;}
        var dup=false;fps.forEach(function(fp){if(Math.abs(fp.z-zn)<1e-4)dup=true;});
        if(!dup&&zn>=-0.1&&zn<=1.1)fps.push({z:zn,slope:fmapD(zn,w,h)});
      }
      prev=val;
    }
    fps.forEach(function(fp){fp.stable=Math.abs(fp.slope)<1-1e-6;});
    fps.sort(function(a,b){return a.z-b.z;});
    return fps;
  }

  function drawCW(){
    var h=getH();
    CW.clearRect(0,0,S,S);CW.fillStyle='#fff';CW.fillRect(0,0,S,S);
    var PL=50,PR2=15,PT=25,PB=45,W2=S-PL-PR2,H2=S-PT-PB;
    var zL=-0.05,zH=1.05,zR=zH-zL;
    function cx(v){return PL+(v-zL)/zR*W2;}
    function cy(v){return PT+(zH-v)/zR*H2;}

    CW.strokeStyle='#eee';CW.lineWidth=.5;
    [.2,.4,.6,.8].forEach(function(v){CW.beginPath();CW.moveTo(cx(v),PT);CW.lineTo(cx(v),PT+H2);CW.stroke();CW.beginPath();CW.moveTo(PL,cy(v));CW.lineTo(PL+W2,cy(v));CW.stroke();});
    CW.strokeStyle='#81D4FA';CW.lineWidth=1;
    CW.beginPath();CW.moveTo(PL,cy(0));CW.lineTo(PL+W2,cy(0));CW.stroke();
    CW.beginPath();CW.moveTo(cx(0),PT);CW.lineTo(cx(0),PT+H2);CW.stroke();

    // Bisectrix
    CW.strokeStyle='#555';CW.lineWidth=1.5;
    CW.beginPath();CW.moveTo(cx(zL),cy(zL));CW.lineTo(cx(zH),cy(zH));CW.stroke();
    // Sigmoid
    CW.strokeStyle='#1976D2';CW.lineWidth=3;CW.beginPath();
    for(var z=zL;z<=zH;z+=.002){var y=fmap(z,Wp,h);if(z<=zL+.005)CW.moveTo(cx(z),cy(y));else CW.lineTo(cx(z),cy(y));}
    CW.stroke();
    // Fixed points
    var fps=findFP(Wp,h);
    fps.forEach(function(fp){
      var px=cx(fp.z),py=cy(fp.z);
      CW.fillStyle=fp.stable?'#4CAF50':'#F44336';CW.beginPath();CW.arc(px,py,8,0,Math.PI*2);CW.fill();
      CW.strokeStyle=fp.stable?'#1B5E20':'#B71C1C';CW.lineWidth=2;CW.beginPath();CW.arc(px,py,8,0,Math.PI*2);CW.stroke();
    });
    // Cobweb from z0=0.2
    var z=0.2,nIter=60;
    CW.strokeStyle='rgba(100,100,100,0.25)';CW.lineWidth=1;CW.beginPath();CW.moveTo(cx(z),cy(0));
    for(var i=0;i<nIter;i++){var y=fmap(z,Wp,h);if(y<-1||y>2)break;CW.lineTo(cx(z),cy(y));CW.lineTo(cx(y),cy(y));z=y;}
    CW.stroke();
    // Labels
    CW.font='14px "Times New Roman",serif';CW.fillStyle='#666';
    CW.fillText('x',S-20,cy(0)+18);CW.fillText('\u03C3(Wx+h)',PL+5,PT-6);
    CW.font='10px sans-serif';CW.fillStyle='#aaa';
    [0,.2,.4,.6,.8,1].forEach(function(v){CW.fillText(v.toFixed(1),cx(v)-8,cy(zL)+16);CW.fillText(v.toFixed(1),4,cy(v)+4);});
    CW.font='12px "Times New Roman",serif';
    CW.fillStyle='#1976D2';CW.fillText('\u03C3(Wx+h)',PL+8,PT+16);
    CW.fillStyle='#555';CW.fillText('y = x',PL+8,PT+32);
    CW.fillStyle='#4CAF50';CW.fillText('\u25CF stable',PL+8,S-PB-6);
    CW.fillStyle='#F44336';CW.fillText('\u25CF unstable',PL+70,S-PB-6);
  }

  function drawBD(){
    BD.clearRect(0,0,S,S);BD.fillStyle='#fff';BD.fillRect(0,0,S,S);
    var PL=50,PR2=15,PT=25,PB=45,W2=S-PL-PR2,H2=S-PT-PB;
    var wLo=0.5,wHi=8,zL2=-0.05,zH2=1.05;
    function bx(w){return PL+(w-wLo)/(wHi-wLo)*W2;}
    function by(z){return PT+(zH2-z)/(zH2-zL2)*H2;}

    BD.strokeStyle='#eee';BD.lineWidth=.5;
    [2,4,6].forEach(function(v){BD.beginPath();BD.moveTo(bx(v),PT);BD.lineTo(bx(v),PT+H2);BD.stroke();});
    [.2,.4,.6,.8].forEach(function(v){BD.beginPath();BD.moveTo(PL,by(v));BD.lineTo(PL+W2,by(v));BD.stroke();});
    BD.strokeStyle='#81D4FA';BD.lineWidth=1;
    BD.beginPath();BD.moveTo(PL,by(0));BD.lineTo(PL+W2,by(0));BD.stroke();
    BD.beginPath();BD.moveTo(PL,PT);BD.lineTo(PL,PT+H2);BD.stroke();
    // Scan
    for(var pw=0;pw<=W2;pw+=1){
      var w=wLo+(pw/W2)*(wHi-wLo);var hh=sym?-w/2:hp;
      var fps=findFP(w,hh);
      fps.forEach(function(fp){BD.fillStyle=fp.stable?'rgba(76,175,80,0.5)':'rgba(244,67,54,0.4)';BD.fillRect(PL+pw-.5,by(fp.z)-.5,1.5,1.5);});
    }
    // W indicator
    BD.setLineDash([4,3]);BD.strokeStyle='#7B1FA2';BD.lineWidth=2;
    BD.beginPath();BD.moveTo(bx(Wp),PT);BD.lineTo(bx(Wp),PT+H2);BD.stroke();BD.setLineDash([]);
    if(sym){BD.setLineDash([2,4]);BD.strokeStyle='#FF9800';BD.lineWidth=1;BD.beginPath();BD.moveTo(bx(4),PT);BD.lineTo(bx(4),PT+H2);BD.stroke();BD.setLineDash([]);BD.font='11px "Times New Roman",serif';BD.fillStyle='#FF9800';BD.fillText('W=4',bx(4)+4,PT+14);}
    BD.font='14px "Times New Roman",serif';BD.fillStyle='#666';BD.fillText('W',S-18,by(0.5)+18);BD.fillText('z*',PL+5,PT-6);
    BD.font='10px sans-serif';BD.fillStyle='#aaa';
    [1,2,3,4,5,6,7,8].forEach(function(v){BD.fillText(v,bx(v)-4,by(zL2)+16);});
    [0,.2,.4,.6,.8,1].forEach(function(v){BD.fillText(v.toFixed(1),4,by(v)+4);});
    BD.font='13px "Times New Roman",serif';BD.fillStyle='#7B1FA2';BD.fillText('W = '+Wp.toFixed(2),PL+5,PT+14);
    BD.font='10px sans-serif';BD.fillStyle='#4CAF50';BD.fillText('\u25A0 stable',PL+5,S-PB-6);BD.fillStyle='#F44336';BD.fillText('\u25A0 unstable',PL+65,S-PB-6);
  }

  function updInfo(){
    var el=document.getElementById('rb2info'),h=getH(),fps=findFP(Wp,h);
    var t='W='+Wp.toFixed(2)+' &nbsp;|&nbsp; h='+h.toFixed(2)+' &nbsp;|&nbsp; slope@0.5='+(Wp/4).toFixed(3)+' &nbsp;|&nbsp; ';
    fps.forEach(function(fp,i){
      t+='<span style="color:'+(fp.stable?'#4CAF50':'#F44336')+'">z*='+fp.z.toFixed(4)+' ('+(fp.stable?'stable':'unstable')+')</span>';
      if(i<fps.length-1)t+=' &nbsp; ';
    });
    if(fps.length>=3)t+=' &nbsp;<b style="color:#7B1FA2">PITCHFORK</b>';
    el.innerHTML=t;
  }
  function redraw(){drawCW();drawBD();updInfo();}

  function getWfromBD(e){var rect=bdEl.getBoundingClientRect(),PL2=50,W2=S-PL2-15;var cx2=(e.clientX-rect.left)*(S/rect.width);return Math.max(0.5,Math.min(8,0.5+(cx2-PL2)/W2*7.5));}
  bdEl.addEventListener('mousedown',function(e){dragBif=true;Wp=getWfromBD(e);wS.value=Wp;wV.textContent=Wp.toFixed(2);if(sym){hp=-Wp/2;hS.value=hp;hV.textContent=hp.toFixed(2);}redraw();});
  bdEl.addEventListener('mousemove',function(e){if(!dragBif)return;Wp=getWfromBD(e);wS.value=Wp;wV.textContent=Wp.toFixed(2);if(sym){hp=-Wp/2;hS.value=hp;hV.textContent=hp.toFixed(2);}redraw();});
  window.addEventListener('mouseup',function(){dragBif=false;});
  bdEl.addEventListener('touchstart',function(e){e.preventDefault();dragBif=true;Wp=getWfromBD(e.touches[0]);wS.value=Wp;wV.textContent=Wp.toFixed(2);if(sym){hp=-Wp/2;hS.value=hp;hV.textContent=hp.toFixed(2);}redraw();},{passive:false});
  bdEl.addEventListener('touchmove',function(e){e.preventDefault();if(!dragBif)return;Wp=getWfromBD(e.touches[0]);wS.value=Wp;wV.textContent=Wp.toFixed(2);if(sym){hp=-Wp/2;hS.value=hp;hV.textContent=hp.toFixed(2);}redraw();},{passive:false});
  bdEl.addEventListener('touchend',function(){dragBif=false;});

  wS.addEventListener('input',function(){Wp=parseFloat(this.value);wV.textContent=Wp.toFixed(2);if(sym){hp=-Wp/2;hS.value=hp;hV.textContent=hp.toFixed(2);}redraw();});
  hS.addEventListener('input',function(){hp=parseFloat(this.value);hV.textContent=hp.toFixed(2);redraw();});
  symCb.addEventListener('change',function(){sym=this.checked;hS.disabled=sym;if(sym){hp=-Wp/2;hS.value=hp;hV.textContent=hp.toFixed(2);}redraw();});
  redraw();
})();
</script>

### Bifurcations in Neural Dynamics

In high-dimensional recurrent networks, the system’s behavior is governed by its fixed points and their stability. As we adjust parameters during training, the system may undergo qualitative changes in its topological structure, known as **bifurcations**. This connects to the general bifurcation theory introduced in Part I.

#### The Saddle-Node Bifurcation in Sigmoidal Units

Consider a single sigmoidal unit within a network where we adjust a parameter $H$. As $H$ varies, the intersection between the state update function and the bisectrix changes.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Saddle-Node Bifurcation Mechanism)</span></p>

As we shift the curve defined by $H$, it will eventually touch the bisectrix. This contact point gives rise to a **saddle-node bifurcation**. Before this point, we might have two stable fixed points (one at a lower value, one at an upper value). As we move through the bifurcation point, these fixed points can merge and disappear, or a single stable point may remain.

</div>

#### Bifurcation Graphs and Parameter Sensitivity

We can visualize these transitions by plotting the fixed point $z^*$ against the parameter $H$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Bifurcation Graph)</span></p>

A **bifurcation graph** represents the location and stability of fixed points $z^*$ as a function of a system parameter $H$. For a sigmoidal system undergoing a saddle-node bifurcation, the graph typically displays a characteristic "arc" or "fold" where stable and unstable branches meet.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(The Learning Barrier — Kenji Doya, 1998)</span></p>

In a 1998 paper, Kenji Doya illustrated why certain configurations are unlearnable. If a desired state for a network is located at an unstable node surrounded by a cycle, gradient descent will fail to stabilize the system at that point. Because the target is unstable, the system will naturally drift away, preventing the network from ever reaching the desired output.

</div>

<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/bIFURCATIONS_iN_tHE_lEARNING_oF_rECURRENT_nEURAL_nETWORKS.png' | relative_url }}" alt="Newton–Raphson iteration animation" loading="lazy">
</figure>

#### Impact on Training: The Loss Landscape

Training an RNN via gradient descent involves iteratively readjusting parameters like $H$ to minimize a loss function. However, bifurcations create significant obstacles for this process.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The "Jump" Phenomenon)</span></p>

Imagine the system is in a regime with two stable fixed points (lower and upper arcs). If the target $z$ requires increasing $H$, the state will move along the current arc. Upon reaching the bifurcation point, the current stable equilibrium disappears, forcing the state to "jump" abruptly to the other arc. This transition causes a steep, discontinuous jump in the loss function.

</div>

<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/BifurcationAsLossJump.png' | relative_url }}" alt="Newton–Raphson iteration animation" loading="lazy">
</figure>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Gradient Behavior at Bifurcations)</span></p>

For certain types of bifurcations and dynamical systems, it can be formally proven that at the bifurcation point, the gradients will either:

1. **Diverge/Explode:** Tend toward infinity.
2. **Abruptly vanish:** Instantly go to zero.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Empirical Loss Landscapes)</span></p>

Research (e.g., Eisman et al.) using algorithms like PyDSTool to locate bifurcation curves has shown that huge jumps in the loss landscape coincide exactly with these curves. When plotting parameter trajectories during training, the loss spikes precisely as the trajectory crosses a bifurcation boundary.

</div>

#### Avoiding Bifurcations with Generalized Teacher Forcing

Standard backpropagation through time (BPTT) is highly susceptible to the instabilities caused by bifurcations. However, specific algorithms can mitigate this.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Smooth Loss via Alignment)</span></p>

Generalized Teacher Forcing (GTF), as defined earlier, is an algorithm that can formally be shown to avoid certain bifurcations. By aligning the system with the observed data at each time point, GTF "smooths out" the loss function. It effectively pushes the system into the correct dynamical regime without requiring it to cross the discontinuous "cliffs" in the loss landscape found in straightforward BPTT.

</div>

<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/GeneralizedTeacherForcingPreventsBifurcationsInTraining.png' | relative_url }}" alt="Newton–Raphson iteration animation" loading="lazy">
</figure>

### Flow Operators and Continuous-Time RNNs

While RNNs are often defined in discrete time, they are frequently used to approximate systems that exist in continuous time. To do this accurately, the network must behave like a true flow operator.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Flow Operator)</span></p>

A **flow operator** maps an initial state $x_0$ to a future state $x_t$ after a duration $t$. For a true flow operator, we expect:

1. The ability to continuously vary $\Delta t$ and obtain valid outputs.
2. Adherence to the semi-group property.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Semi-Group Property)</span></p>

If we advance a system by time steps $s$ and $t$ in succession, the resulting state must be the same as advancing the system by a single step of $(s + t)$:

$$\Phi_{s+t} = \Phi_s \circ \Phi_t = \Phi_t \circ \Phi_s$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Path Independence)</span></p>

In a training context, if we have data $x_0$ and we want to reach a state at a future time, the result must be identical regardless of the path taken:

* **Path A:** Move from $x_0$ to $x_1$ (time $\tau_1$) then to $x_2$ (time $\tau_2$).
* **Path B:** Move directly from $x_0$ to $x_2$ (time $\tau_1 + \tau_2$).
* **Path C:** Move by $\tau_2$ first, then $\tau_1$.

All these paths must yield the same result for the system to be a mathematically consistent flow.

</div>

#### Recursive Descriptions and Neutral Elements

To enforce these properties, we can define the RNN using a recursive structure, as suggested by Chen and Wu (2003).

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Recursive Flow Approximation)</span></p>

A system can be defined by the following recursive description:

$$z_t = z_{t - \Delta\tau} + \Delta\tau \cdot \sigma(z_{t - \Delta\tau},\, \Delta\tau)$$

where $\sigma$ is an activation function (which could itself be a deep feed-forward neural network) and $\Delta\tau$ is the time step.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proof</span><span class="math-callout__name">(Convergence to the Neutral Element)</span></p>

We demand that if the time step $\Delta t$ is zero, the state remains unchanged (the neutral element). Using the recursive definition:

$$z_t = \lim_{\Delta\tau \to 0} \left[ z_{t - \Delta\tau} + \Delta\tau \cdot \sigma(\dots) \right]$$

As $\Delta\tau \to 0$:

$$z_t = z_t + 0 \cdot \sigma(\dots) = z_t$$

This demonstrates that the recursive formulation automatically satisfies the neutral element property of a flow operator.

</div>

#### Enforcing Flow Properties through Loss Function Design

Standard RNN training does not guarantee the semi-group properties. To ensure the learned model behaves like a physical flow, we must use regularization terms in the loss function.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The Flow Operator Composition Law)</span></p>

A true flow operator $f$ acting on a state $z_t$ with time steps $\tau_1$ and $\tau_2$ must satisfy the following composition law:

$$f(z_t,\, \tau_1 + \tau_2) = f(f(z_t,\, \tau_1),\, \tau_2)$$

This implies that advancing the system by $\tau_1$ and then by $\tau_2$ must be equivalent to advancing it by the combined step $\tau_1 + \tau_2$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Regularization as a Constraint Mechanism)</span></p>

While we can attempt to design architectures that inherently respect these properties, a more flexible approach is to enforce them through the loss function. By adding a regularization term to the standard objective, we penalize the model when it deviates from the requirements of a flow operator. This principle can be extended to other physical constraints, such as Hamiltonian conservation laws.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(The Regularized Loss Function)</span></p>

The complete training loss $\mathcal{L}$ is constructed as the sum of a standard Mean Squared Error (MSE) and a regularization term $\lambda$ that enforces the flow properties:

$$\mathcal{L} = \text{MSE} + \lambda \sum \text{Deviations}$$

where the deviations are defined as:

1. **Composition Error:** The difference between a single step of $(\tau_1 + \tau_2)$ and the sequential application of $\tau_1$ and $\tau_2$:

   $$\lVert f(z_t,\, \tau_1 + \tau_2) - f(f(z_t,\, \tau_1),\, \tau_2) \rVert^2$$

2. **Commutativity Error:** The difference resulting from swapping the order of $\tau_1$ and $\tau_2$:

   $$\lVert f(f(z_t,\, \tau_1),\, \tau_2) - f(f(z_t,\, \tau_2),\, \tau_1) \rVert^2$$

</div>

### Reservoir Computing (Echo State Machines)

Training RNNs is notoriously difficult and computationally expensive. Techniques like generalized teacher forcing offer improvements, but **Reservoir Computing** (also known as Echo State Machines) approaches the problem from an entirely different perspective.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Reservoir Computing)</span></p>

First introduced by Jaeger and Haas (2004) in *Science*, **Reservoir Computing** is a type of RNN where the internal connectivity is fixed and only the output layer is trained. It aims to maintain the simplicity of linear regression while retaining the ability to approximate complex, nonlinear dynamical systems.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Core Idea)</span></p>

The core concept is to project an input into a high-dimensional "pool" or reservoir of complex dynamics. Instead of meticulously training every connection in the network, we use a large, fixed reservoir that expresses a wide variety of dynamical behaviors. We then "shape" or "read out" these dynamics through a simple linear layer to match our observations.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(MP Note: Echo state networks)</span></p>

**Echo state networks (ESNs)** have a large recurrent layer at the input, which has weights initialized randomly, and these are not trained in any way. Typically, this layer is implemented as a single $k \times m$ matrix, where $k=n+m$, $m$ is the size of the recurrent layer, and $n$ is the number of inputs. The network works by receiving vectors of length $n$ as input and appending its internal state of length $m$ to them. It applies its recurrent layer to it and thus gets a new internal state. The recurrent layer typically contains some activation function to make it non-linear (it is applied to all activations - the results of multiplying by a random matrix). After the recurrent layer, traditional ESNs contain only one layer, which, similarly to RBF networks, can be trained using linear regression or using the gradient method.

ESN may look strange at first glance, why should a random matrix give reasonable results? An important observation is that due to the size of the internal state (which is often larger than the number of inputs), the matrix actually randomly transforms the information from the input into a higher dimensional space. Other layers then actually try to decode this information and get the necessary information from it. Note also that the internal state of the network depends on all previous inputs, not just the last one, and thus contains information about all of them.

</div>

#### Architecture and Dynamics of Reservoir Systems

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Reservoir System Variables)</span></p>

* $x_t \in \mathbb{R}^N$: The true observed state (e.g., a time series from a Lorenz system or temperature data).
* $\hat{x}_t \in \mathbb{R}^N$: The network’s predicted state.
* $z_t \in \mathbb{R}^M$: The reservoir state (or latent state).
* **Dimensionality Constraint:** Crucially, $M \gg N$. The reservoir must be high-dimensional to provide a sufficiently rich "pool" of possibilities.

</div>

The reservoir state $z_t$ evolves according to a nonlinear function, typically utilizing a sigmoid activation:

$$z_t = \sigma(W z_{t-1} + h + W_{\text{in}}\, s_t)$$

where:
* $W$: The internal reservoir connectivity matrix. This is **fixed** and not changed during training.
* $W_{\text{in}}$: Input weights, also fixed.
* $s_t$: External inputs or forced true states $x_t$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Reservoir Connectivity Properties)</span></p>

To prevent the reservoir from exhibiting "boring" dynamics (such as immediately collapsing to a fixed point), the matrix $W$ is carefully initialized:

* **Sparse Connectivity:** Connections are often sparse.
* **Spectral Norm:** The eigenvalue spectrum is typically scaled so the spectral norm is close to 1. This ensures the reservoir is at the "edge of chaos"—neither exploding nor decaying too rapidly.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Reservoir Systems Prediction)</span></p>

The **prediction** $\hat{x}_t$ is generated via a linear mapping from the reservoir state:

$$\hat{x}_t = B z_t$$

In some cases, a basis expansion of $z_t$ is performed to improve performance. For example, concatenating $z_t$ with its squared terms: $\hat{x}_t = B [z_t;\, z_t^2]$. Importantly, the system remains linear in the parameters $B$.

</div>

#### Training Reservoir Computers via Linear Regression

Because the internal weights $W$ are fixed, the training process does not require backpropagation through time. Instead, it boils down to a simple regression problem.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Training vs. Inference)</span></p>

* **Training (Entrainment):** The reservoir is forced with the true states $x_t$. We record the resulting reservoir states $z_t$.
* **Test Time (Inference):** The true input $x_t$ is replaced by the network’s own previous prediction $\hat{x}_t$, letting the system run recursively.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Closed-Form Solution for Readout Weights)</span></p>

The optimal readout matrix $B$ can be solved analytically by minimizing the Mean Squared Error:

$$\mathcal{L} = \frac{1}{T} \sum_{t=1}^T \lVert x_t - B z_t \rVert^2$$

Setting the derivative with respect to $B$ to zero:

$$\frac{\partial \mathcal{L}}{\partial B} = \sum 2(x_t - B z_t)\, z_t^\top = 0$$

This yields the closed-form solution:

$$B = \left( \sum_{t=1}^T x_t\, z_t^\top \right) \left( \sum_{t=1}^T z_t\, z_t^\top \right)^{-1}$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proof</span><span class="math-callout__name">(Derivation of the Readout Matrix)</span></p>

1. Represent the loss in terms of the $L_2$ norm:
   
   $$\mathcal{L} = \frac{1}{T} \sum_{t=1}^T (x_t - B z_t)^\top (x_t - B z_t) \propto (x_t - B z_t)^\top (x_t - B z_t)$$

2. Expand the product:
   
   $$x_t^\top x_t - z_t^\top B^\top x_t - x_t^\top B z_t + z_t^\top B^\top B z_t$$

3. Differentiate with respect to $B$:
   
   $$\frac{\partial (-2\, x_t^\top B z_t)}{\partial B} = -2\, x_t\, z_t^\top$$
   
   $$\frac{\partial (z_t^\top B^\top B z_t)}{\partial B} = 2\, B z_t\, z_t^\top$$

4. Equate to zero:
   
   $$\sum_{t=1}^{T} x_t\, z_t^\top = B \sum_{t=1}^{T} z_t\, z_t^\top$$

5. Isolate $B$ by multiplying by the inverse of the covariance-like matrix of reservoir states:

   $$B = \left(\sum_{t=1}^{T} x_t\, z_t^\top\right) \left(\sum_{t=1}^{T} z_t\, z_t^\top\right)^{-1}$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Practical Application)</span></p>

In practice, training a reservoir computer is "one line of Python code." Once the reservoir is entrained with the training data and the matrix $B$ is calculated, the model can predict complex sequences (like the Lorenz system) with surprising accuracy, provided the reservoir properties (spectral norm, sparsity) are correctly tuned.

</div>

#### Refining Reservoir Computing for Topology Preservation

While standard Reservoir Computing provides a computationally efficient framework for time-series prediction, it often fails to capture the true limiting dynamics of a system—the behavior as $t \to \infty$. To reconstruct a dynamical system properly, the model must do more than minimize immediate error; it must replicate the system’s invariant properties.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Reservoir Weight Constraints)</span></p>

To ensure the reservoir remains stable and possesses the "Echo State Property," the internal weight matrix $W$ is typically constrained. A common condition is that the maximum singular value $\sigma_{\max}$ of $W$ is constrained:

$$\sigma_{\max}(W) \leq 1$$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Multi-Step Loss Function)</span></p>

Instead of minimizing the error for just $t+1$, we minimize the squared deviations over a window $U$:

$$\mathcal{L}_{\text{multi}} = \sum_{u=0}^{U} \sum_{t=u + 1}^{T} \lVert B \cdot f^u(z_{t-u}) - x_t \rVert^2$$

where:
* $B$ is the readout matrix.
* $f^u$ represents the $u$-th composition of the reservoir’s recurrent transition function.
* $z_{t-u}$ is the latent state at time $t-u$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Multi-Step Prediction)</span></p>

By forcing the network to predict multiple steps into the future using its own previous predictions (recursive mode), we encourage the system to obey longer-term dynamics. This prevents the "drift" often seen in models trained only on single-step transitions.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Statistical Regularization)</span></p>

The total loss $\mathcal{L}$ can be augmented by a penalty term that measures the deviation of invariant statistics:

$$\mathcal{L} = \text{MSE} + \lambda \lVert C_{\text{data}} - C_{\text{model}} \rVert$$

where $C$ represents a dynamical invariant, some long-term statistics such as:

* **Maximum Lyapunov Exponent:** The rate of exponential separation of nearby trajectories.
* **Lyapunov Spectrum:** The full set of exponents characterizing the system’s stability.
* **Fractal Dimensionality:** Measures like the Correlation Dimension or the Kaplan-Yorke Dimension.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Physics-Based Training)</span></p>

This approach explicitly builds the "physics" or the "limiting dynamics" into the training process. If the true system is chaotic with a specific fractal dimension, we penalize the neural network if its autonomous behavior produces a different dimensionality.

</div>

### Autoencoders: Nonlinear Dimensionality Reduction

A central challenge in dynamical systems reconstruction is dimensionality. While a system might be observed in a high-dimensional space $X \in \mathbb{R}^D$, its true degrees of freedom often live on a much lower-dimensional manifold. An **Autoencoder** (AE) is a feed-forward neural network designed to learn a compressed representation of the input data. Unlike PCA, which is limited to linear projections, Autoencoders can capture nonlinear manifolds.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Autoencoder Structure)</span></p>

An **Autoencoder** consists of two primary components:

1. **Encoder** ($\phi$): Maps the high-dimensional input $x_t$ to a low-dimensional latent state $z_t$.
2. **Decoder** ($\phi^{-1}$): Maps the latent state $z_t$ back to the reconstructed input $\hat{x}_t$.

The architecture follows an "hourglass" shape, where the inner layer (the bottleneck) has significantly fewer units than the input/output layers.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The Autoencoder Objective)</span></p>

The **objective of the Autoencoder** is to approximate the identity function through a bottleneck. We minimize the Mean Squared Error (MSE):

$$\mathcal{L}_{\text{AE}} = \sum_{t} \lVert x_t - \hat{x}_t \rVert^2 = \sum_{t} \lVert x_t - \phi^{-1}(\phi(x_t)) \rVert^2$$

Successful training implies that:

$$x_t \approx \phi^{-1}(\phi(x_t))$$

This suggests that $\phi^{-1}$ acts as an approximate inverse of $\phi$, and $z_t = \phi(x_t)$ captures the essential information required to reconstruct $x_t$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Deep Autoencoder: Layer-wise Formulation)</span></p>

Modern Autoencoders are implemented as Deep Neural Networks, alternating affine mappings with nonlinear activation functions. The latent representation $z_t$ for a three-layer encoder can be written as:

$$z_t = \sigma(W_3\, \sigma(W_2\, \sigma(W_1\, x_t + h_1) + h_2) + h_3)$$

where $\sigma$ is a nonlinearity like the ReLU ($\max(0, x)$), $W_i$ are weight matrices, and $h_i$ are bias vectors. The decoder follows a symmetric structure to expand $z_t$ back to the original dimensionality.

</div>

### Joint Manifold Discovery and SINDy Reconstruction

> RNNs work because they do something like implicit delay embedding. If our latent dimensionaly is greater than out observation dimensionality. RNNs learn those underlying latent dynamics. (or in opposite)

The ultimate goal in modern reconstruction (e.g., Champion and Brunton, 2019) is to combine the dimensionality reduction of Autoencoders with the interpretability of structural models like SINDy (Sparse Identification of Non-linear Dynamics). This builds upon the SINDy framework introduced in earlier lectures.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Latent Dynamics Assumption)</span></p>

We assume that while our measurements $x_t$ are high-dimensional and correlated, they are governed by a low-dimensional latent dynamic $\dot{z} = f(z)$. By training an Autoencoder and a dynamical model simultaneously, we identify the coordinate system ($z$) in which the dynamics are most "sparse" or "simple."

</div>

#### SINDy in Latent Space

Once the Autoencoder projects the data into the latent space $z_t$, we apply the SINDy framework to identify the governing differential equations.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(SINDy in Latent Space)</span></p>

We represent the dynamics of the latent state $z$ using a library of candidate basis functions $\Theta(z)$:

$$\dot{z} = \Theta(z)\,\Xi$$

where:
* $\Theta(z)$ contains functions like polynomials ($z_1, z_2, z_1^2, z_1 z_2, \dots$) or trigonometric functions.
* $\Xi$ is a sparse matrix of coefficients that determines which terms in the library actually contribute to the dynamics.

</div>

#### The Autoencoder-SINDy Architecture

To move from high-dimensional observations $x$ to a low-dimensional latent space $z$, we utilize an Autoencoder structure combined with a dynamics model:

* **Encoder** ($\phi$): A function that projects the input observations into a latent space: $z = \phi(x)$.
* **Decoder/Approximate Inverse** ($s$): A function that maps the latent state back to the observation space: $\hat{x} = s(z)$. Note that $s$ is an approximation of $\phi^{-1}$.
* **Latent Dynamics:** Within the latent space, the dynamics are governed by a function $\Theta(z)\,\xi$, representing the identified vector field.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Parsimonious Coordinates)</span></p>

This configuration allows the model to learn a coordinate transformation where the dynamics become parsimonious (sparse). Instead of modeling complex, high-dimensional noise, the system identifies the "true" underlying degrees of freedom.

</div>

#### Mathematical Formulation of the Loss Function

The training of this integrated system relies on a multi-term loss function designed to satisfy reconstruction accuracy, dynamical consistency, and sparsity.

**I. Reconstruction Loss**

The first term ensures the autoencoder can accurately reconstruct the input signal:

$$L_{\text{rec}} = \lVert x_t - s(\phi(x_t)) \rVert^2$$

**II. Latent Space Derivative Loss**

To ensure the dynamics in the latent space match the observed temporal evolution of the data, we must relate the latent derivatives $\dot{z}$ to the empirical observation derivatives $\dot{x}$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Derivation</span><span class="math-callout__name">(Latent Derivatives via the Chain Rule)</span></p>

Given the relationship $z = \phi(x)$, we find the temporal derivative $\dot{z}$ by applying the chain rule:

$$\dot{z} = \frac{d}{dt} \phi(x) = \nabla_x \phi(x) \cdot \frac{dx}{dt}$$

In the context of our learned dynamics $\Theta(z)\,\xi$, we require:

$$\nabla_x \phi(x) \cdot \dot{x} \approx \Theta(z)\,\xi$$

Thus, the latent derivative loss is defined as:

$$L_{\dot{z}} = \lVert \nabla_x \phi(x) \cdot \dot{x} - \Theta(z)\,\xi \rVert^2$$

where $\dot{x}$ is an empirical estimate (e.g., first-order temporal differences).

</div>

**III. Observation Space Derivative Loss**

We also enforce that the derivatives reconstructed from the latent space match the empirical derivatives in the original observation space.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Derivation</span><span class="math-callout__name">(Observation Derivatives)</span></p>

Starting from $x \approx s(z)$, the temporal derivative is:

$$\dot{x} \approx \frac{d}{dt} s(z) = \nabla_z s(z) \cdot \dot{z}$$

Substituting the latent dynamics $\dot{z} = \Theta(z)\,\xi$:

$$\dot{x} \approx \nabla_z s(z) \cdot (\Theta(z)\,\xi)$$

The resulting loss term is:

$$L_{\dot{x}} = \lVert \dot{x} - \nabla_z s(z) \cdot (\Theta(z)\,\xi) \rVert^2$$

</div>

**IV. Regularization Loss**

To enforce the sparsity of the identified dynamics (the SINDy principle), we apply an $L_1$ penalty to the parameters $\xi$:

$$L_{\text{reg}} = \lVert \xi \rVert_1$$

**V. Total Integrated Loss**

The complete objective function is a weighted sum of these terms:

$$L_{\text{total}} = L_{\text{rec}} + \lambda_1 L_{\dot{z}} + \lambda_2 L_{\dot{x}} + \lambda_3 L_{\text{reg}}$$

where $\lambda_1, \lambda_2, \lambda_3$ are hyper-parameters that weigh the importance of reconstruction, dynamical fidelity, and sparsity.

#### Empirical Validation: The Lorenz Attractor

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(High-Dimensional Lorenz Projection)</span></p>

To evaluate the performance of this approach, it is tested on the Lorenz Attractor:

1. **System Dynamics:** The Lorenz system is defined in 3D latent space $(z_1, z_2, z_3)$:
   * $\dot{z}_1 = \sigma(z_2 - z_1)$
   * $\dot{z}_2 = z_1(R - z_3) - z_2$
   * $\dot{z}_3 = z_1 z_2 - V z_3$
2. **Projection:** The 3D system is projected into a high-dimensional space (e.g., 128 dimensions) using non-linear basis functions $U_1, \dots, U_6$.
3. **Task:** The model must take the 128D empirical data, project it back to a 3D latent manifold, and correctly identify the Lorenz equations.
4. **Result:** The framework successfully retrieves the underlying 3D structure and the sparse coefficients of the vector field from the high-dimensional embedding.

</div>

#### Supplemental Insights: Reservoir Computing and Predictability

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Lyapunov Exponents and Predictability)</span></p>

Including Lyapunov exponents in the training criterion significantly improves the "mean valid prediction time." Models trained with these dynamical constraints can predict the future state of a chaotic system for a much longer duration compared to those trained solely on standard error metrics.

</div>

