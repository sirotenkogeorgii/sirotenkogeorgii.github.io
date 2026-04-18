## Lecture 9

### Introduction to Universal Approximators for Dynamical Systems

#### Beyond Pre-defined Libraries: The Need for Universal Approximators

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Motivation for Universal Approximators)</span></p>

In previous discussions, we explored methods for inferring dynamical systems from data, such as SINDy. A notable characteristic of such approaches is their reliance on a pre-defined library of functions that must be specified a priori. This requirement, while powerful in certain contexts, can be a limitation.

The focus of our study now shifts to a class of methods that do not have this caveat: deep learning methods. These models are known as universal approximators, capable of learning complex functions directly from data without the need to manually define a basis or library of candidate functions. This chapter will introduce a foundational deep learning architecture for modeling time-dependent systems.

</div>

#### Deep Learning and Recurrent Neural Networks (RNNs)

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Universal Approximation of Dynamical Systems)</span></p>

All the methods discussed henceforth are **universal approximators of functions in general, and of dynamical systems in particular**. A key architecture in this domain is the Recurrent Neural Network (RNN).

An RNN can be formally shown to be a universal approximator of dynamical systems. While we will not delve into the formal proofs of the theorems that establish this property, we will build a comprehensive understanding of their structure, function, and application. These models were, and in many domains remain, state-of-the-art for time series prediction and the modeling of dynamical systems.

</div>

### The Architecture and Motivation of Recurrent Neural Networks

#### Neuroscientific Origins and Core Concepts

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Neuroscientific Origins)</span></p>

Like many foundational neural network architectures, RNNs have their roots in neuroscience and psychology, where they were initially introduced as abstract models of the brain. The core idea is to model a system of interconnected processing units, or neurons, that influence each other's activity over time.

The key components of this model are:

* **Units (Neurons):** These are the nodes of the network, each possessing an activation value at a given point in time. We can denote the activation of unit $i$ at time $t$ as $x_i^t$.
* **Synaptic Connections (Weights):** The units are coupled through connections, each having an associated weight, denoted $w_{ij}$, which represents the strength of the connection from unit $j$ to unit $i$. These weights are adjustable parameters learned from data.
* **Recurrent Connections:** The defining feature of RNNs is the presence of feedback connections. Unlike feed-forward architectures (like many Convolutional Neural Networks) where information flows in a single direction, RNNs can have connections that form cycles. This allows for both forward and backward connections between units, enabling the network to maintain an internal state or "memory" of past events. This is what makes the network recurrent.
* **External Inputs:** Some or all units may receive input from the external world. This input is represented as a time series, $S_t$.
* **Outputs:** Similarly, some or all units can produce outputs that are sent back to the external world.

</div>

#### Mathematical Formulation of an RNN

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(RNN Activation Dynamics)</span></p>

The **activation** of a specific unit $i$ at time $t$, denoted $x_i^t$, is determined by an activation function, $\phi$. This function processes a weighted sum of inputs from other units at the previous time step ($t-1$), any external inputs at the current time step ($t$), and a unit-specific bias term.

The general formulation for the activation of unit $i$ is:

$$x_i^t = \phi \left( \sum_j w_{ij} x_j^{t-1} + h_i + \sum_k c_{ik} S_k^t \right)$$

Where:

* $x_i^t$ is the activation of unit $i$ at time $t$.
* $\phi$ is a non-linear activation function.
* $w_{ij}$ is the connection weight from unit $j$ to unit $i$.
* $x_j^{t-1}$ is the activation of unit $j$ at the previous time step, $t-1$.
* $h_i$ is a unit-specific, learnable bias term.
* $c_{ik}$ is the weight for the $k$-th external input to unit $i$.
* $S_k^t$ is the value of the $k$-th external input at time $t$.

The learnable parameters of the model, which are adjusted during training, include the connection weights ($w_{ij}$), the input weights ($c_{ik}$), and the bias terms ($h_i$).

</div>

#### Historical Context

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Historical Context)</span></p>

The foundational concepts and training algorithms for RNNs were first developed in the late 1980s and early 1990s. Key figures associated with their invention include Jeff Elman and Paul Werbos (referred to in the source as "Bar Palmer or Zipa").

For a significant period, RNNs were not widely popular in the broader machine learning community due to challenges in training them effectively. However, with advancements in algorithms and computational power, they have become indispensable tools, particularly for sequence and time-series data. We will return to the topic of training challenges and modern solutions later in the course.

</div>

### Formalizing Recurrent Neural Networks

#### From Scalar Operations to Vector Notation

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(From Scalars to Vectors)</span></p>

Recurrent Neural Networks (RNNs) are an inherently natural architectural choice for modeling time series and dynamical systems. Their structure, which processes information sequentially and maintains an internal state that evolves over time, mirrors the fundamental nature of such systems. This stands in contrast to other architectures, such as transformers, which may be adapted for these tasks but lack the intrinsic recursive formulation of an RNN.

To analyze these systems rigorously, we move from a component-wise description of individual network units to a more compact and powerful matrix notation. This allows us to treat the entire network's state as a single vector and its evolution as a unified vector-valued map.

</div>

#TODO: sigmoid: stronger weights -> steeper the ? what moves it?

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The RNN in Vector Notation)</span></p>

A recurrent neural network's state at a discrete time step $t$ can be described by a vector of activation values, $z_t$. The evolution of this state from one time step to the next is governed by a recursive map.

The components of this map are:

* **State Vector:** $z_t \in \mathbb{R}^M$. This is a vector containing the activation values of the $M$ units in the network at time $t$. These are also referred to as latent states, as they represent an internal, unobserved configuration of the system.
* **Weight Matrix:** $W \in \mathbb{R}^{M \times M}$. A square matrix containing the weights of the connections between the network's units. This matrix does not change with time.
* **Bias Vector:** $h \in \mathbb{R}^M$. Also known as a bias term, this vector applies a constant offset to the pre-activation of each unit, biasing it towards a particular activity regime.
* **External Input Vector:** $s_t \in \mathbb{R}^K$. An optional vector representing $K$ external inputs to the system at time $t$. The dimensionality $K$ does not need to equal the internal state dimensionality $M$.
* **Input Weight Matrix:** $C \in \mathbb{R}^{M \times K}$. This matrix maps the $K$-dimensional external input space to the $M$-dimensional latent state space.
* **Nonlinear Activation Function:** $\phi(\cdot)$. A scalar function (e.g., a sigmoid) that is applied element-wise to the pre-activation vector.

The state update equation, which describes the evolution of the network, can be written as:

$$z_t = \phi(W z_{t-1} + C s_t + h)$$

More generally, we can express the dynamics of an RNN as a function $f$ parameterized by a set of parameters $\theta$, which includes $W$, $C$, and $h$.

$$z_t = f(z_{t-1}, s_t; \theta)$$

This formulation makes it explicit that the state at time $t$ is a function of the state at the previous time step, $t-1$, and any external inputs at time $t$.

</div>

#### The RNN as a Discrete-Time Dynamical System

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(RNNs as Dynamical Systems)</span></p>

The recursive formulation $z_t = f(z_{t-1}, s_t; \theta)$ is of profound importance. It reveals that a recurrent neural network is, in essence, a discrete-time, multi-dimensional recursive map. This directly parallels the discrete-time maps, such as the logistic map, that are central to the study of dynamical systems.

This connection is not merely an analogy; it has direct and critical consequences. Because an RNN is a discrete dynamical system, it is subject to the full range of complex behaviors that these systems can exhibit. Specifically, depending on the parameters ($\theta$) and initial conditions ($z_0$), an RNN can:

* Converge to different fixed points.
* Exhibit periodic behavior (cycles).
* Undergo bifurcations as its parameters are changed.
* Display chaotic dynamics.

Understanding these potential behaviors is crucial for both analyzing and effectively training these networks.

</div>

### Training Recurrent Neural Networks

#### The Gradient Descent Paradigm

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why Gradient Descent)</span></p>

While numerous techniques exist for training machine learning models, the field is overwhelmingly dominated by methods based on gradient descent. This is not because gradient descent is the most powerful optimization technique available -- other methods may yield more accurate parameter estimates. Rather, its dominance stems from its effectiveness and scalability. Gradient descent-based techniques are generally well-understood, straightforward to implement, and scale favorably with the size of the dataset. For these reasons, it is the primary method for training RNNs.

The objective of training is to find a set of model parameters that minimizes a given loss function. **Since RNNs are highly nonlinear devices, it is impossible to find an analytical, closed-form solution for the optimal parameters.** We must therefore rely on iterative numerical optimization algorithms like gradient descent.

</div>

#### Defining the Core Components for Training

To train an RNN, we require three fundamental components: a dataset, a model architecture that links latent states to observable outputs, and a loss function to quantify performance.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The Dataset)</span></p>

The training data consists of a set of $P$ patterns or sequences. For each pattern $p \in \lbrace 1, \dots, P \rbrace$, the **dataset** provides:

* **Inputs:** A sequence of input vectors $\lbrace s_t^{(p)} \rbrace_{t=1}^{T_p}$, where $s_t^{(p)} \in \mathbb{R}^K$. These are optional, depending on the task.
* **Desired Outputs (Targets):** A sequence of target vectors $\lbrace x_t^{(p)} \rbrace_{t=1}^{T_p}$, where $x_t^{(p)} \in \mathbb{R}^N$. These are the ground-truth values the model should aim to produce.

Here, $T_p$ is the length of the $p$-th sequence.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The Model Architecture: State Dynamics and Observation)</span></p>

The complete model consists of two parts:

1. **The Recursive Core (State Equation):** This is the RNN itself, which describes the evolution of the latent states $z_t$.

   $$z_t = f(z_{t-1}, s_t; \theta)$$

2. **The Decoder (Observation Model):** This is a function, $g$, that maps the latent state $z_t$ to a predicted output $\hat{x}_t$. This is necessary because the latent states are not directly observed; the decoder must learn to translate them into the space of the target outputs.

   $$\hat{x}_t = g(z_t; \lambda)$$

   The decoder has its own set of parameters, denoted by $\lambda$.

This two-part structure is analogous to concepts in dynamical systems where an unobservable internal state generates observable measurements. Models of this form are sometimes referred to as State-Space Models, though this term often implies additional probabilistic assumptions.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Linear Decoder)</span></p>

A simple and common choice for the decoder $g$ is a linear mapping, also known as a linear layer in neural network terminology:

$$\hat{x}_t = B z_t$$

Here, the parameter set $\lambda$ is simply the matrix $B \in \mathbb{R}^{N \times M}$, which maps the $M$-dimensional latent space to the $N$-dimensional output space. The dimensionality of the latent space, $M$, is a design choice and does not need to be equal to the input or output dimensions.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(MSE loss function for sequences)</span></p>

The **loss function**, $L$, quantifies the discrepancy between the model's predicted outputs and the true target outputs. It is a function of the model's parameters ($\theta$ and $\lambda$). The goal of training is to minimize this function. A common and straightforward choice is the Sum of Squared Errors (SSE) loss, which is calculated by summing the squared deviations over all time steps and all patterns in the dataset.

Given the observed output $x_t^{(p)}$ and the predicted output $\hat{x}_t^{(p)}$, the SSE loss is:

$$L(\theta, \lambda) = \sum_{p=1}^{P} \sum_{t=1}^{T_p} \| x_t^{(p)} - \hat{x}_t^{(p)} \|^2$$

While SSE is used here for concreteness, any differentiable loss function (e.g., likelihood functions) can be used within the gradient descent framework. The fundamental goal remains the same: to adjust the network's parameters to make the predicted output as close as possible to the observed output.

</div>

#### The Gradient Descent Algorithm

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Gradient Descent Overview)</span></p>

Gradient descent is an iterative algorithm that seeks to find a minimum of the loss function. The process begins with an initial guess for the parameters and repeatedly adjusts them in the direction that most steeply decreases the loss.

**Algorithm Outline:**

1. **Initialization:** Start with an initial guess for the parameters, $\theta_0$ and $\lambda_0$. A common practice is to draw these initial values from a probability distribution, such as a Gaussian distribution with zero mean.

   $$\theta_0, \lambda_0 \sim \mathcal{N}(0, \sigma^2 I)$$

2. **Iteration:** Initialize an iteration counter, e.g., $k=1$. Begin a loop that continues until a stopping criterion is met. In each step of the loop, the parameters are updated based on the gradient of the loss function. (The process of calculating the gradient and performing the update will be detailed subsequently.)

</div>

### Gradient Descent-Based Training

#### The Core Idea: Minimizing the Loss Function

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Intuition for Gradient Descent)</span></p>

The central goal of training a model is to find a set of parameters, which we'll denote by $\theta$, that minimizes a loss function, $L(\theta)$. This function quantifies how poorly our model is performing on a given dataset; a lower loss value corresponds to a better model.

The idea behind gradient descent is intuitive: we start with an initial guess for our parameters $\theta$ and iteratively update them by taking small steps in the direction that most steeply decreases the loss. The gradient of the loss function, $\nabla L(\theta)$, points in the direction of the steepest ascent. Therefore, to minimize the loss, we must move in the opposite direction of the gradient.

Imagine a hilly landscape where the altitude represents the loss value for any given parameter set $\theta$. Our goal is to find the lowest valley.

* If we are on a slope where the gradient is positive, we need to move in the negative direction (downhill).
* If we are on a slope where the gradient is negative, we need to move in the positive direction (also downhill).

In both cases, we "go against the gradient." This iterative process continues until we reach a point where the loss is sufficiently low, ideally a minimum.

It is important to note that in classical machine learning, the aim was often to find the global optimum -- the single best parameter set that corresponds to the absolute lowest point in the loss landscape. However, in modern practice, finding the global optimum is often not feasible, nor is it always desirable. As we will discuss later, forcing a model to the global minimum on the training data can lead to a phenomenon known as overfitting.

</div>

#### The Gradient Descent Algorithm

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Gradient Descent)</span></p>

**Gradient Descent** is an iterative optimization algorithm used to find a local minimum of a differentiable function. The parameters are updated at each step $n$ according to the following rule:

$$\theta_n = \theta_{n-1} - \gamma \nabla L(\theta_{n-1})$$

Where:

* $\theta_n$ is the vector of parameters at iteration $n$.
* $\theta_{n-1}$ is the vector of parameters from the previous iteration.
* $\gamma$ is a positive scalar known as the learning rate, which controls the size of the step taken at each iteration.
* $\nabla L(\theta_{n-1})$ is the gradient of the loss function $L$ evaluated at the parameters $\theta_{n-1}$.

The gradient $\nabla L(\theta)$ is a vector of partial derivatives:

$$\nabla L(\theta) = \left[ \frac{\partial L}{\partial \theta_1}, \frac{\partial L}{\partial \theta_2}, \dots, \frac{\partial L}{\partial \theta_L} \right]$$

where $\theta_1, \dots, \theta_L$ are the individual parameters of the model.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(A Simple Gradient Descent Loop)</span></p>

A simple implementation of this algorithm can be formulated as a while loop, which continues as long as the improvement in loss is significant and a maximum number of iterations has not been reached.

**Algorithm:**

1. Initialize parameters $\theta_0$ and counter $n = 0$.
2. Set a learning rate $\gamma > 0$, a minimum loss change threshold $\epsilon$, and a maximum number of iterations $N_{\max}$.
3. **while** $\Delta L(\theta) > \epsilon$ **and** $n < N_{\max}$:
   * Calculate the gradient: $g = \nabla L(\theta_n)$
   * Update the parameters: $\theta_{n+1} = \theta_n - \gamma g$
   * Increment the counter: $n = n + 1$

This core idea forms the basis for the most common optimization procedures in machine learning. While more sophisticated versions are implemented in standard toolboxes, they are fundamentally built upon this principle. The same concept can be used for maximization (e.g., in Maximum Likelihood Estimation) by simply moving with the gradient (i.e., using a $+$ sign instead of a $-$), which is equivalent to minimizing the negative of the function.

</div>

#### The Training Algorithm as a Dynamical System

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Training as a Dynamical System)</span></p>

This is a fundamentally important point. The iterative update rule of gradient descent:

$$\theta_n = \theta_{n-1} - \gamma \nabla L(\theta_{n-1})$$

is a recursive procedure. It defines the state of the parameters at step $n$ based on their state at step $n-1$. This structure is precisely what defines a discrete-time dynamical system.

The implications of this are profound:

* The entire toolset of dynamical systems theory can be applied to analyze the training process itself.
* Just like the dynamical systems we have studied, the training process can exhibit complex behaviors. The parameter updates can:
  * **Converge to a fixed point:** This is often the desired outcome, as a fixed point where the gradient is zero corresponds to a local minimum (or a saddle point). Hopfield networks are an example where fixed points are defined to be local minima.
  * **Become oscillatory:** The parameters might not settle down, but instead cycle through a set of values.
  * **Exhibit chaos:** The parameter updates could be chaotic, never converging or repeating in a predictable pattern.

This perspective reveals that training a neural network is not just a simple optimization problem but a dynamical process with its own stability properties and potential complexities. We will see the consequences of this in the next section.

</div>

### Challenges in Gradient-Based Optimization

While powerful, the gradient descent algorithm is not without its challenges. The nature of the loss landscape -- the high-dimensional surface defined by $L(\theta)$ -- can introduce significant difficulties for the optimization process.

#### Local Minima and Saddle Points

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Local Minima and Saddle Points)</span></p>

A primary issue in gradient-based optimization is that the algorithm can get "stuck." The update rule relies on the gradient to determine the direction of movement. At any point where the gradient is zero ($\nabla L(\theta) = 0$), the update step becomes zero, and the algorithm halts.

These points can be:

* **Local Minima:** These are valley bottoms in the loss landscape that are not the single lowest point (the global minimum). If the algorithm converges to a local minimum, the resulting model may be suboptimal.
* **Saddle Points:** These are points that are a minimum along one dimension but a maximum along another. The gradient is also zero here, causing the algorithm to stall.

In either case, the optimizer may fail to find a better solution, even if one exists elsewhere in the parameter space.

</div>

#### Widely Differing Loss Slopes and Learning Rate Selection

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Learning Rate Dilemma)</span></p>

The geometry of the loss landscape presents another major challenge related to selecting an appropriate learning rate ($\gamma$). Loss functions for complex models are rarely smooth, uniform bowls. They often contain regions of vastly different curvature: some areas might be extremely steep "valleys," while others are wide, flat "plateaus."

This creates a dilemma for choosing $\gamma$:

* **If $\gamma$ is too small:** In flat regions of the loss landscape, the gradients will be very small. A small learning rate will result in minuscule updates, and the algorithm will take an extremely long time to converge, if it converges at all.
* **If $\gamma$ is too large:** In very steep regions, a large learning rate can cause the algorithm to overshoot the minimum. The update step may be so large that it jumps completely across the valley to a point where the loss is even higher. This can lead to oscillations where the parameters bounce back and forth, failing to converge, and may even cause the algorithm to diverge entirely.

The ideal learning rate would be adaptive: large in flat regions to speed up progress and small in steep regions to ensure careful convergence. This challenge has motivated the development of more advanced optimization algorithms beyond simple gradient descent.

</div>

#### The Impact of System Dynamics on the Loss Landscape

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Dynamics Shape the Loss Landscape)</span></p>

The dynamics of the model being trained have direct implications for the structure of its loss function. While the loss function is defined over the parameter space, not the state space of the model, the behavior of the model's dynamics shapes the landscape.

Consider training a Recurrent Neural Network.

* If the RNN is operating in a chaotic regime, its output can be extremely sensitive to small changes in its parameters.
* This sensitivity translates to the loss function. The resulting loss landscape can be incredibly complex and may even be fractal.
* Trying to perform gradient descent on such a landscape is exceptionally difficult, as the gradient can change dramatically and unpredictably with tiny steps.

In practice, the loss landscape for large systems (with potentially hundreds, thousands, or even billions of parameters) is extremely high-dimensional. While we cannot visualize it directly, we can analyze its properties by plotting cross-sections in subspaces or by observing the behavior of the training process itself.

</div>

### Classical Remedies and Modern Approaches

Over the years, researchers and practitioners have developed numerous techniques to mitigate the challenges of gradient-based optimization. Here, we survey a few key ideas.

#### Addressing Local Minima

##### Multiple Initial Conditions

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Multiple Initial Conditions)</span></p>

A straightforward, classical approach to increase the chances of finding a good minimum is to run the entire optimization process multiple times from different, randomly chosen initial parameter values ($\theta_0$).

If the loss landscape contains many local minima, starting from different points explores different regions of the space. After all the runs are complete, one simply chooses the model that achieved the lowest final loss value. While simple, this can be computationally expensive. This technique is not as commonly used today in its basic form for large-scale deep learning, but the principle of exploration remains important.

</div>

##### Overparameterization: Double Descent and the Lottery Ticket Hypothesis

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Double Descent and the Lottery Ticket Hypothesis)</span></p>

A more modern and perhaps counter-intuitive approach involves using strongly overparameterized models -- that is, using many more parameters than one might think are necessary to represent the data.

This is related to a phenomenon known as **double descent**. Classical statistical theory suggests that as you increase model complexity (number of parameters), the test loss (error on unseen data) will first decrease (good) and then increase as the model begins to overfit the training data (bad). This creates a U-shaped curve.

However, a surprising observation, highlighted in a notable 2019 paper by Belkin, Hsu, Ma, and Mandal (related to the Franklin and Carbone paper mentioned in the lecture), is that if you continue to increase the number of parameters far beyond the point of overfitting, the test loss can decrease again. This second drop is the "double descent."

This leads to the **Lottery Ticket Hypothesis**. The idea is that a very large, overparameterized network is like a lottery containing many tickets. Within this massive network, there exists a smaller, optimal sub-network (the "winning ticket") that is perfectly suited for the given task. The gradient descent training process, in this view, doesn't just tune all the parameters, but effectively carves out this winning sub-network from the larger structure by pruning unnecessary connections.

Note: This is an active area of research, and while powerful, this approach does not always work.

</div>

##### Stochasticity: Adding Noise to Gradients

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Adding Noise to Gradients)</span></p>

Another strategy is to intentionally introduce randomness into the optimization process. By adding a small amount of noise, $\epsilon$, to the gradient calculation at each step, the parameter updates become probabilistic:

$$\theta_{n+1} = \theta_n - \gamma (\nabla L(\theta_n) + \epsilon)$$

where $\epsilon$ is drawn from some probability distribution.

The purpose of this noise is to provide a chance for the parameters to "jump out" of a local minimum. If the algorithm is stuck in a shallow valley, a random nudge from the noise term might be enough to push it over the hill and into a deeper, better region of the loss landscape.

This principle is exploited by entire classes of models, such as Boltzmann Machines, which use thermal noise in a principled way to probabilistically find the global optimum of the system. This topic, however, goes beyond the scope of our current discussion.

</div>

### Advanced Optimization for Neural Network Training

#### Mitigating Local Minima: Stochastic Gradient Descent (SGD)

A primary challenge in gradient-based optimization is the risk of the algorithm converging to a local minimum in the loss function rather than the desired global minimum. One of the most common and effective procedures to address this is not to inject artificial noise into the gradient updates, but rather to leverage the noise inherent in the data itself.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Stochastic Gradient Descent: SGD)</span></p>

**Stochastic Gradient Descent (SGD)** is an optimization algorithm where, at each gradient step, the update is calculated based on a randomly drawn subsample (a "mini-batch") of the full training dataset, rather than the entire dataset.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Intuition for SGD)</span></p>

The core idea is that data is inherently noisy. By randomly drawing a different subset of data for each step, we introduce noise into the gradient calculation. This stochasticity can help the optimization process "jump out" of shallow local minima and continue its search for a better solution in the broader parameter space. The effect is conceptually similar to injecting noise directly into the gradient updates.

</div>

##### A Note on Time Series Data

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Handling Autocorrelations in Temporal Data)</span></p>

When applying SGD or related subsampling techniques to time series and dynamical systems, special care must be taken. These data types are characterized by significant autocorrelations, where the value of a point depends on previous points.

* **Problem:** Randomly sampling individual data points from a time series will destroy its temporal structure and, consequently, the very dynamics the model is intended to learn.
* **Solution:** To preserve the temporal integrity, one must sample consecutive blocks or segments of the time series for each gradient update. This ensures that the essential dynamic relationships within the data are maintained during training.

</div>

#### Addressing Varying Slopes: Adaptive Learning Rates

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Adaptive Learning Rates)</span></p>

Another significant challenge in training deep networks is the presence of "ravines" or "valleys" in the loss landscape, where the slope is very steep in one direction and very shallow in another. A fixed learning rate can cause oscillations across the steep direction while making painfully slow progress along the shallow one.

The naive but effective approach to this problem is to make the learning rate adaptive. Instead of a fixed scalar $\gamma$, we can use a learning rate $\gamma_n$ that changes at each step $n$. This adaptation can be based on the history of the gradients, their variance, or other principles designed to accelerate progress in flat directions and dampen updates in steep directions.

</div>

##### Common Adaptive Rate Algorithms

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(A Toolbox of Optimizers)</span></p>

Modern machine learning frameworks provide a host of optimizers that implement adaptive learning rate schemes. While a deep dive into each is beyond our current scope, it is essential to be aware of the most prominent examples:

* **Adagrad:** Adapts the learning rate based on the historical sum of squared gradients for each parameter.
* **Momentum:** Aims to accelerate descent by adding a fraction of the previous update vector to the current one, helping to build "velocity" in a consistent direction. It provides an implicit estimate of the slope.
* **Adam (Adaptive Moment Estimation):** A highly popular algorithm that combines the ideas of momentum and adaptive scaling of gradients (similar to RMSprop).
* **RAdam (Rectified Adam):** An enhancement to Adam that seeks to correct for the high variance of adaptive learning rates in the early stages of training.

In contemporary practice, a significant portion of the field has settled on using Adam or RAdam as default, robust optimizers for a wide range of problems. All these techniques function by adjusting learning rates in an intelligent manner during the gradient descent procedure.

</div>

#### Incorporating Curvature: Second-Order Algorithms

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Second-Order Methods Intuition)</span></p>

While first-order methods like gradient descent only use the gradient (first derivative) of the loss function, second-order algorithms incorporate additional information about the curvature of the loss surface.

The virtue of second-order methods is that they are often superior to standard gradient descent because they possess a more detailed "map" of the loss landscape. By considering how the gradient itself is changing (i.e., the second derivative), they can make more informed steps. The idea is that if the first derivative is small, the change in the derivative is also likely to be small. These methods weigh the gradient update by the magnitude of the second derivatives.

</div>

##### The Hessian and the Newton-Raphson Procedure

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Gradient update using local curvature)</span></p>

The Hessian is the matrix of second-order partial derivatives of the loss function. It describes the local curvature of the function at a given point.

A **naive second-order update rule** modifies the parameters $\theta$ not just with the gradient of the loss $\nabla_{\theta} L$, but by pre-multiplying it with the inverse of the Hessian, $H^{-1}$:

$$\theta_{n+1} = \theta_n - \gamma [H(\theta_n)]^{-1} \nabla_{\theta} L(\theta_n)$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Relation to Newton-Raphson)</span></p>

In its strict formulation, this update rule gives rise to the Newton-Raphson procedure, a well-known root-finding algorithm from statistics and numerical analysis.

</div>

##### Challenges and Adjustments

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Caveats of Second-Order Methods)</span></p>

Despite their theoretical advantages, pure second-order methods are rarely used in modern large-scale machine learning for two primary reasons:

1. **Computational Demand:** Calculating, storing, and inverting the Hessian matrix is computationally prohibitive. For a network with $N$ parameters, the Hessian is an $N \times N$ matrix, which quickly becomes intractable.
2. **Inclination and Saddle Points:** The naive update rule can get stuck in inclination points or saddle points where both the first and second derivatives vanish. Furthermore, it does not distinguish between local minima and local maxima, which is problematic as we only wish to find minima.

To make these methods viable, adjustments are necessary. A notable proposal by Pascanu and Bengio (c. 2014) involves modifying the Hessian to ensure updates always point towards a minimum.

* A Singular Value Decomposition (SVD) of the Hessian is performed.
* All singular values are set to be positive. Conceptually, taking an absolute value, denoted here as 

  $$|H| = Q|\Lambda| Q^\top$$

* This procedure ensures that the second derivatives cannot change sign at the same time as the first derivative vanishes, preventing convergence to maxima.

The adjusted update rule can be conceptualized as:

$$\theta_{n+1} = \theta_n - \gamma (|H(\theta_n)|)^{-1} \nabla_{\theta} L(\theta_n)$$

</div>

##### Quasi-Newton Methods and Their Relevance

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Quasi-Newton Methods)</span></p>

**Quasi-Newton methods** are a class of algorithms that seek to capture the benefits of second-order information without the prohibitive cost of computing the full Hessian. They do so by building an efficient numerical approximation of the inverse Hessian at each step.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Recursive Least Squares)</span></p>

Recursive Least Squares (RLS) is an algorithm formerly used for updating recurrent networks that falls into the family of quasi-Newton methods.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Enduring Value of Second-Order Thinking)</span></p>

While most large-scale applications have moved away from these methods, they should not be forgotten. For scientific applications with smaller datasets, the precision offered by incorporating curvature information can be extremely valuable. Furthermore, concepts in machine learning have a tendency to resurface, and a solid understanding of these powerful techniques remains a significant asset.

</div>

#### A Specialized Algorithm for RNNs: Backpropagation Through Time (BPTT)

We now turn to a very specific, time-efficient gradient descent algorithm tailored for Recurrent Neural Networks (RNNs): Backpropagation Through Time (BPTT).

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Backpropagation Through Time: BPTT)</span></p>

**BPTT** is the standard algorithm for training RNNs. It is an adaptation of the general backpropagation algorithm that applies gradient descent to an RNN by first "unwrapping" or "unrolling" the network through its time steps.

</div>

##### Historical Context

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Historical Context of BPTT)</span></p>

BPTT was introduced and refined by several researchers over the years, with key contributions from:

* Paul Werbos (1988)
* Ronald Williams and David Zipser
* David Rumelhart and others, with a famous paper in 1995.

</div>

##### The Core Idea: Unwrapping in Time

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(From Recurrence to Depth)</span></p>

The foundational insight of BPTT is to train a recurrent network in the exact same way as a standard feed-forward network. This is achieved by conceptually transforming the RNN's temporal recursion into a spatial deep structure.

Consider a simple RNN with two units, whose activations are $x_1$ and $x_2$, and a weight matrix $W$ that includes recurrent couplings like $W_{11}$, $W_{12}$, $W_{21}$, and $W_{22}$.

To train this network on a time series of length $T$, we perform the following "unwrapping" procedure:

1. **Create a Layer for Each Time Step:** The RNN is converted into a deep feed-forward network where each time step, from $t=1$ to $t=T$, becomes a distinct layer.
2. **Propagate Activations:** The state of the network at time $t$ becomes the input to the layer representing time $t+1$. An activation $x_1(t-1)$ propagates to influence $x_1(t)$, $x_2(t)$, and so on.
3. **Share Weights Across Layers:** This is the crucial feature that distinguishes an unwrapped RNN from a standard deep network. The same set of weights ($W_{11}$, $W_{12}$, $\dots$) is used at every layer (i.e., at every time step). The weights are effectively copied and pasted across the entire time-unrolled structure.

The following illustrates this transformation from a recurrent graph to a deep, feed-forward graph:

* **At Time $t=1$:** The network has units with activations $x_1(1)$ and $x_2(1)$.
* **At Time $t=2$:** This forms the next layer. The connection from $x_1(1)$ to $x_1(2)$ is governed by weight $W_{11}$. The connection from $x_2(1)$ to $x_1(2)$ is governed by $W_{12}$, and so on.
* ...and so on, until **Time $t=T$:** The final layer corresponds to the final time step, with activations $x_1(T)$ and $x_2(T)$.

This unwrapped structure is simply another way to write down the recursive update procedure of the RNN. Instead of updating a single network state recursively, we can think of it as propagating activity through a deep network where each layer corresponds to a moment in time.

</div>

##### The Backpropagation Procedure

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(The Backpropagation Procedure)</span></p>

BPTT is a specific, algorithmically efficient implementation of gradient descent.

1. **Forward Pass:** Propagate activity forward through the unrolled network, from $t=1$ to $t=T$.
2. **Calculate Errors:** At the output layer(s), calculate the error, which is the deviation between the network's prediction and the target value.
3. **Backward Pass:** Propagate these error signals backward through the network, from layer $T$ down to layer $1$. At each layer (time step), update the shared weight parameters based on the propagated error.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Algorithmic Efficiency)</span></p>

BPTT is a highly storage-efficient procedure. At each step of the backward pass, it only needs to account for the values present at that particular time step, as it leverages the already-computed values from the subsequent step. The complexity is linear in time, as it proceeds layer by layer.

</div>

##### Input and Output Configurations

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Input and Output Configurations)</span></p>

The specific structure of the unwrapped network depends on the task at hand.

* **External Inputs:** The network can receive external inputs at any or all time steps. For example, in sentence processing, each word could be an input at a sequential time step.
* **Target Outputs:** The network can be trained to produce a target output at any or all time steps.
  * **Sequence-to-Sequence (e.g., Time Series Modeling):** If we want an RNN to reproduce a temperature time series, we would have a target output (the desired temperature) at each time step.
  * **Sequence-to-Value (e.g., Classification):** If we want to perform sentiment classification on a sentence, we might provide word inputs at each time step but only have a single target output at the final time step ($t=T$), representing the overall sentiment.

</div>

##### Formalism for Training

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Simplifications for Derivation)</span></p>

For clarity in the following derivation, we will:

1. Neglect External Inputs: These do not fundamentally change the derivation of the gradient updates.
2. Consider a Single Data Pattern: The logic extends trivially to multiple patterns by summing or averaging the loss.

Let our RNN be given by the recursive form, and let our set of trainable parameters (weights and biases) be denoted by $\theta$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Loss Function for RNN)</span></p>

A typical **loss function $L(\theta)$ for an RNN** trained on a sequence of length $T$ is the mean squared error, averaged over time:

$$L(\theta) = \frac{1}{T} \sum_{t=1}^{T} \sum_{k=1}^{N} (x_k(t) - x_k^*(t))^2$$

where $N$ is the number of units, $x_k(t)$ is the activation of unit $k$ at time $t$, and $x_k^*(t)$ is the desired or target output for that unit at that time.

</div>

### Training Recurrent Networks: Backpropagation Through Time (Detailed Derivation)

#### The Optimization Problem: Minimizing a Loss Function

To train any neural network, we must first define an objective. This objective is typically formulated as the minimization of a loss function, $L$, which measures the discrepancy between the network's predictions and the observed data. For a single observed time series, we can define a total loss over the entire sequence.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Total toss as a sum of steps)</span></p>

The total loss, $L$, for a given time series is the **sum of the losses incurred at each individual time step**. If we denote the loss at a specific time step $t$ as $l_t$, the total loss is given by:

$$L = \sum_t l_t$$

This decomposition is possible due to the linearity of gradients, which allows us to consider the contribution of each time step to the total parameter gradient independently.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Squared Error Loss)</span></p>

For concreteness, a common choice for the loss function is the squared error loss. Let $x_{\text{obs}}(t)$ be the observed value at time $t$ and $x(t)$ be the value predicted by our model. The loss at that time step, $l_t$, would be:

$$l_t = (x_{\text{obs}}(t) - x(t))^2$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Generality of the Loss)</span></p>

While we use the squared error for this example, the derivations that follow are general. You can substitute any differentiable loss function $l_t$ without changing the core mechanics of the backpropagation algorithm. The loss $l_t$ is a function of both the system parameters, which we'll call $\theta$, and the network's state or activation at that time, $x(t)$.

</div>

#### Gradient Calculation for Recurrent Architectures

Our goal is to adjust the model's parameters, $\theta_i$, to minimize the total loss $L$. We achieve this using gradient descent, which requires computing the derivative of the loss with respect to each parameter, $\frac{\partial L}{\partial \theta_i}$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proof</span><span class="math-callout__name">(The Total Loss Gradient)</span></p>

By the linearity of the gradient operator, the derivative of the total loss is the sum of the derivatives of the per-time-step losses:

$$\frac{\partial L}{\partial \theta_i} = \frac{\partial}{\partial \theta_i} \sum_t l_t = \sum_t \frac{\partial l_t}{\partial \theta_i}$$

Now, we must analyze the term $\frac{\partial l_t}{\partial \theta_i}$. In a standard feedforward network, a parameter only affects the loss at the output layer. In an RNN, however, the situation is more complex. The parameters (e.g., the weight matrix $W$) are reused at every time step. This means a parameter $\theta_i$ at an early time step $\tau$ influences the state $x_t$ at a later time step $t$.

Consequently, to calculate the gradient of the loss at time $t$, we must sum over the influence of the parameter $\theta_i$ as it appears at all preceding time steps $\tau$ from $1$ to $t$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proof</span><span class="math-callout__name">(Decomposing the Gradient with the Chain Rule)</span></p>

Applying the chain rule to the term $\frac{\partial l_t}{\partial \theta_i}$ reveals this dependency. The loss $l_t$ is an explicit function of the state $x_t$. The state $x_t$, in turn, is a function of all previous states, including $x_\tau$, where the parameter $\theta_i$ has an effect. This creates a recursive dependency that we must unroll.

The full expression for the gradient of the loss at time $t$ with respect to a parameter $\theta_i$ is a sum over all previous time steps $\tau \le t$ where that parameter appears:

$$\frac{\partial l_t}{\partial \theta_i} = \sum_{\tau=1}^{t} \frac{\partial l_t}{\partial x_t} \frac{\partial x_t}{\partial x_\tau} \frac{\partial x_\tau}{\partial \theta_i}$$

Let's break down the components of this expression:

1. $\frac{\partial l_t}{\partial x_t}$: This is the local gradient of the loss at time $t$ with respect to the network's output/state at that same time. It measures how the final error at step $t$ changes with respect to the output at step $t$.
2. $\frac{\partial x_t}{\partial x_\tau}$: This is the **temporal Jacobian matrix**. It measures how the state at a later time $t$ is influenced by the state at an earlier time $\tau$. This term is the crux of BPTT, as it carries the gradient information backward through the unrolled network.
3. $\frac{\partial x_\tau}{\partial \theta_i}$: This term measures the direct influence of the parameter $\theta_i$ on the state $x_\tau$ at the time step $\tau$ where the parameter is applied.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Jacobian Matrix Dimensions)</span></p>

To better understand the mathematical objects we are manipulating, consider an RNN with an $m$-dimensional state vector $x \in \mathbb{R}^m$. The dimensions of the terms in the chain rule are as follows:

* $\frac{\partial l_t}{\partial x_t}$: A $1 \times m$ row vector (the gradient of the scalar loss w.r.t. the state vector).
* $\frac{\partial x_t}{\partial x_\tau}$: An $m \times m$ matrix, representing the Jacobian of the state at time $t$ with respect to the state at time $\tau$. Each element $(j, k)$ of this matrix is $\frac{\partial x_j(t)}{\partial x_k(\tau)}$.
* $\frac{\partial x_\tau}{\partial \theta_i}$: If $\theta_i$ is a scalar parameter, this is an $m \times 1$ column vector.

</div>

#### The Recursive Chain Rule and Temporal Dependencies

The most important and complex term in our gradient expression is the temporal Jacobian, $\frac{\partial x_t}{\partial x_\tau}$. This term quantifies the long-range dependencies in the sequence. We can decompose it further by another application of the chain rule.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proof</span><span class="math-callout__name">(Unrolling the Temporal Jacobian)</span></p>

The state $x_t$ is a direct function of $x_{t-1}$, which is a function of $x_{t-2}$, and so on. We can express the derivative of $x_t$ with respect to a distant past state $x_\tau$ as a product of intermediate, single-step Jacobians:

$$\frac{\partial x_t}{\partial x_\tau} = \frac{\partial x_t}{\partial x_{t-1}} \frac{\partial x_{t-1}}{\partial x_{t-2}} \cdots \frac{\partial x_{\tau+1}}{\partial x_\tau}$$

This can be written more compactly using product notation:

$$\frac{\partial x_t}{\partial x_\tau} = \prod_{u=\tau+1}^{t} \frac{\partial x_u}{\partial x_{u-1}}$$

Each term in this product, $\frac{\partial x_u}{\partial x_{u-1}}$, is the Jacobian of the state transition function at a single time step.

</div>

#### The Exploding and Vanishing Gradient Problem

Let's now investigate the structure of the single-step Jacobian, $\frac{\partial x_u}{\partial x_{u-1}}$, to understand the long-term behavior of this product.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(A Simple RNN Update Rule)</span></p>

Consider a standard RNN where the state $x_t$ is updated according to the following rule:

$$x_t = \phi(W x_{t-1} + \dots)$$

Here, $W$ is the recurrent weight matrix and $\phi$ is a non-linear, element-wise activation function. To find the Jacobian $\frac{\partial x_t}{\partial x_{t-1}}$, we apply the chain rule (outer derivative times inner derivative):

* The derivative of the inner part, $W x_{t-1}$, with respect to $x_{t-1}$ is simply the matrix $W$.
* The derivative of the outer element-wise function $\phi$ results in a diagonal matrix containing the derivatives of $\phi$ evaluated at each input component. Let's denote this $\text{diag}(\phi'(\dots))$.

Therefore, the single-step Jacobian is:

$$\frac{\partial x_t}{\partial x_{t-1}} = \text{diag}(\phi'(W x_{t-1} + \dots)) \cdot W$$

Substituting this back into our product expression for the temporal Jacobian, we get:

$$\frac{\partial x_t}{\partial x_\tau} = \prod_{u=\tau+1}^{t} \left( \text{diag}(\phi'(W x_{u-1} + \dots)) \cdot W \right)$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Exploding and Vanishing Gradient Problem)</span></p>

This product of matrices is the source of a fundamental instability in training RNNs. The expression involves repeatedly multiplying the weight matrix $W$, effectively raising it to the power of the time difference, $t-\tau$. The behavior of this matrix power is governed by the eigenvalues of the matrices in the product.

* **Exploding Gradients:** If the magnitudes of the leading eigenvalues of the Jacobian matrices are, on average, greater than $1$, their product will grow exponentially as the time gap $t-\tau$ increases. The gradients will "explode" to enormous values, leading to unstable training and divergent weight updates.
* **Vanishing Gradients:** If the magnitudes of the leading eigenvalues are, on average, less than $1$, their product will shrink exponentially towards zero as $t-\tau$ increases. The gradients will "vanish." This is also highly problematic, as it means the influence of early time steps on the loss at later time steps is effectively erased. The network becomes incapable of learning long-range dependencies, as the information required to update the parameters is lost during backpropagation.

This exploding and vanishing gradient problem is not merely a numerical inconvenience; it is a fundamental obstacle to learning long-term structure in sequential data with simple RNNs. The dynamics of this process are deeply connected to concepts from dynamical systems theory, such as the calculation of Lyapunov exponents and the stability analysis of linear systems converging to fixed points. **The repeated matrix multiplication is precisely the process used to determine the stability of a linear dynamical system.** This insight highlights why these "vanilla" training approaches are no longer standard practice and motivated the development of more sophisticated architectures.

</div>

### Long Short-Term Memory (LSTM) Networks

Long Short-Term Memory (LSTM) networks are a specialized type of recurrent neural network (RNN) architecture designed to handle long-term dependencies in sequential data. This section details the precise equations governing the LSTM cell and explores the core principles that make it effective.

#### The Complete LSTM Architecture

The power of an LSTM lies in its internal structure, which is composed of a memory cell and several gates that regulate the flow of information. These components work in concert to decide what information to store, what to discard, and what to output at each time step.

##### The Memory Cell Update

The most important component of the LSTM is the memory cell, which carries information through time. Its state, denoted by $c_t$, is updated at each time step $t$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Memory Cell Update)</span></p>

The state of the memory cell $c_t$ at time step $t$ is updated according to the following equation:

$$c_t = (f_t \odot c_{t-1}) + (i_t \odot \tanh(z_{t-1} + h_c))$$

Where:

* $c_{t-1}$ is the state of the memory cell from the previous time step.
* $f_t$ is the forget gate's activation vector.
* $i_t$ is the input gate's activation vector.
* $z_{t-1}$ is the total output from the previous time step.
* $h_c$ is a bias term for the candidate memory content.
* $\odot$ denotes the pointwise multiplication (Hadamard product) of vectors.
* $\tanh$ is the hyperbolic tangent activation function.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Intuition for the Memory Cell Update)</span></p>

This update equation has two primary parts:

1. **Forgetting:** The term $f_t \odot c_{t-1}$ determines which parts of the old memory cell state $c_{t-1}$ should be preserved or discarded. The forget gate $f_t$ acts as a filter; if an element of $f_t$ is close to $0$, the corresponding information in $c_{t-1}$ is forgotten. If it is close to $1$, the information is kept.
2. **Inputting:** The term $i_t \odot \tanh(z_{t-1} + h_c)$ determines what new information should be added to the cell state. A candidate memory content is first computed (the $\tanh$ part), and the input gate $i_t$ decides which parts of this new information are relevant enough to be stored in $c_t$.

The combination of these two operations allows the LSTM to selectively update its memory, preserving crucial long-term information while incorporating new, relevant inputs.

</div>

##### The Gating Mechanisms

The flow of information into and out of the memory cell is controlled by three gates: the input gate ($i_t$), the forget gate ($f_t$), and the output gate ($o_t$). These gates are implemented using a sigmoid activation function, which outputs values between $0$ and $1$, representing the degree to which information is allowed to pass.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Gate Equations)</span></p>

The activation of each gate at time step $t$ is calculated as follows:

* **Forget Gate** ($f_t$):

$$f_t = \sigma(W_f z_{t-1} + h_f)$$

* **Input Gate** ($i_t$):

$$i_t = \sigma(W_i z_{t-1} + h_i)$$

* **Output Gate** ($o_t$):

$$o_t = \sigma(W_o c_{t-1} + h_o)$$

Where:

* $\sigma$ is the sigmoid function, defined as:

$$\sigma(y) = \frac{1}{1 + e^{-y}}$$

* $W_f$, $W_i$, and $W_o$ are weight matrices for the respective gates.
* $h_f$, $h_i$, and $h_o$ are the bias vectors for the respective gates.
* $z_{t-1}$ is the output from the previous time step.
* $c_{t-1}$ is the memory cell state from the previous time step.

</div>

<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/lstm_straka_deep_learning.png' | relative_url }}" alt="Newton–Raphson iteration animation" loading="lazy">
</figure>

#### Fundamental Design Principles of LSTMs

The specific architectural choices in the LSTM are not arbitrary; they embody two crucial principles for processing sequential data: linearity and gating.

##### The Power of Linearity

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Power of Linearity)</span></p>

A key feature of the LSTM is the linear nature of the memory update through the forget gate. The operation $f_t \odot c_{t-1}$ is a linear interaction. This is critically important for preserving information over long time horizons.

* **Information Preservation:** If the forget gate $f_t$ is set to $1$, the previous memory content $c_{t-1}$ is passed through to the next step unmodified. This allows the network to "literally copy and paste the previous content" and "rescue it across long periods of time."
* **Control:** This introduces a form of control that is difficult to achieve in purely nonlinear systems. By managing the forget gate, the network can learn to maintain a stable memory state when needed. As stated in the lecture, "Linearity is important. Linearity allows a certain type of control that you don't that easily have in nonlinear systems."

</div>

##### The Concept of Gating

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Concept of Gating)</span></p>

The second foundational concept is gating, where information flow is modulated by multiplicative units (the gates). This idea of using multiplicative interactions to control pathways in a neural network is powerful and has proven influential. This principle is not unique to LSTMs and can be found in other modern architectures, such as Mamba.

</div>

<figure>
  <img src="{{ '/assets/images/notes/dynamical-systems/input_output_forget_gates.JPG' | relative_url }}" alt="Newton–Raphson iteration animation" loading="lazy">
</figure>

#### Variants and Simplifications: GRUs

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Gated Recurrent Units (GRUs))</span></p>

The LSTM architecture, while powerful, is also complex. This has led to the development of several simplified variants. One of the most prominent is the Gated Recurrent Unit (GRU).

* **Origin:** The GRU was introduced in a 2014 formulation by Cho, et al., in collaboration with Yoshua Bengio.
* **Purpose:** GRUs aim to capture the essence of gated RNNs with a simpler architecture, often combining the forget and input gates into a single "update gate."
* **Availability:** You will find GRUs, along with many other LSTM variants, implemented in virtually any standard machine learning toolbox.

</div>

