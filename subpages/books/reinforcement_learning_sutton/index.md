---
title: Reinforcement Learning
layout: default
noindex: true
tags:
  - machine-learning
  - reinforcement-learning
  - markov-decision-process
  - dynamic-programming
  - temporal-difference-learning
  - monte-carlo
  - multi-armed-bandits
---

# Reinforcement Learning

## Chapter 2: Multi-armed Bandits

The most important feature distinguishing reinforcement learning from other types of learning is that it uses training information that *evaluates* the actions taken rather than *instructs* by giving correct actions. This creates the need for active exploration. Evaluative feedback indicates how good the action taken was, but not whether it was the best or the worst action possible. Instructive feedback, on the other hand, indicates the correct action to take independently of the action actually taken.

### A $k$-armed Bandit Problem

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($k$-armed Bandit Problem)</span></p>

You are faced repeatedly with a choice among $k$ different options, or actions. After each choice you receive a numerical reward chosen from a stationary probability distribution that depends on the action you selected. The objective is to maximize the expected total reward over some time period.

Each of the $k$ actions has an expected or mean reward given that that action is selected; we call this the **value** of that action. The action selected on time step $t$ is denoted $A_t$, and the corresponding reward is $R_t$. The value of an arbitrary action $a$, denoted $q_*(a)$, is the expected reward given that $a$ is selected:

$$q_*(a) \doteq \mathbb{E}[R_t \mid A_t = a].$$

</div>

If we knew the value of each action, the problem would be trivial: always select the action with highest value. We assume that we do not know the action values with certainty, although we may have estimates. We denote the estimated value of action $a$ at time step $t$ as $Q_t(a)$, and we would like $Q_t(a)$ to be close to $q_*(a)$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Greedy and Exploring Actions)</span></p>

* **Greedy actions:** Actions whose estimated value is greatest. Selecting a greedy action means you are **exploiting** your current knowledge of the values.
* **Exploring actions:** Selecting a nongreedy action means you are **exploring**, because this enables you to improve your estimate of the nongreedy action's value.

Exploitation maximizes the expected reward on the one step, but exploration may produce greater total reward in the long run. It is not possible to both explore and exploit with any single action selection, creating a fundamental **conflict** between exploration and exploitation.

</div>

### Action-value Methods

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Sample-Average Method)</span></p>

One natural way to estimate action values is by averaging the rewards actually received:

$$Q_t(a) \doteq \frac{\text{sum of rewards when } a \text{ taken prior to } t}{\text{number of times } a \text{ taken prior to } t} = \frac{\sum_{i=1}^{t-1} R_i \cdot \mathbb{1}_{A_i = a}}{\sum_{i=1}^{t-1} \mathbb{1}_{A_i = a}},$$

where $\mathbb{1}_{predicate}$ denotes the indicator that is 1 if *predicate* is true and 0 otherwise. If the denominator is zero, we define $Q_t(a)$ as some default value (e.g. 0). By the law of large numbers, as the denominator goes to infinity, $Q_t(a)$ converges to $q_*(a)$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($\varepsilon$-Greedy Action Selection)</span></p>

The **greedy** action selection method selects the action with the highest estimated value:

$$A_t \doteq \argmax_a Q_t(a).$$

A simple alternative is the **$\varepsilon$-greedy** method: behave greedily most of the time, but with small probability $\varepsilon$, select randomly from among all actions with equal probability. An advantage is that, in the limit, every action will be sampled infinitely often, so all $Q_t(a)$ converge to $q_*(a)$. The probability of selecting the optimal action converges to greater than $1 - \varepsilon$.

</div>

### The 10-armed Testbed

The 10-armed testbed is a suite of 2000 randomly generated $k$-armed bandit problems with $k = 10$. For each bandit problem, the action values $q_*(a)$ for $a = 1, \dots, 10$ were selected according to a normal distribution with mean 0 and variance 1. The actual reward $R_t$ was selected from a normal distribution with mean $q_*(A_t)$ and variance 1.

Key findings from the 10-armed testbed:

* The greedy method improved slightly faster at the beginning but leveled off at about a reward of 1 (compared to a best possible of about 1.54). It got stuck performing suboptimal actions.
* The $\varepsilon = 0.1$ method explored more and found the optimal action earlier, but never selected it more than 91% of the time.
* The $\varepsilon = 0.01$ method improved more slowly, but eventually would outperform $\varepsilon = 0.1$ on both performance measures.

### Incremental Implementation

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Incremental Update Rule)</span></p>

Given $Q_n$ (the estimate after $n-1$ rewards) and the $n$th reward $R_n$, the new average of all $n$ rewards can be computed incrementally:

$$Q_{n+1} = Q_n + \frac{1}{n}\bigl[R_n - Q_n\bigr].$$

This holds even for $n = 1$, obtaining $Q_2 = R_1$ for arbitrary $Q_1$. This implementation requires memory only for $Q_n$ and $n$.

The general form of this update rule is:

$$\text{NewEstimate} \leftarrow \text{OldEstimate} + \text{StepSize}\bigl[\text{Target} - \text{OldEstimate}\bigr].$$

The expression $[\text{Target} - \text{OldEstimate}]$ is an *error* in the estimate. It is reduced by taking a step toward the "Target."

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(A Simple Bandit Algorithm)</span></p>

Initialize, for $a = 1$ to $k$:
* $Q(a) \leftarrow 0$
* $N(a) \leftarrow 0$

Loop forever:
* $A \leftarrow \begin{cases} \argmax_a Q(a) & \text{with probability } 1 - \varepsilon \\ \text{a random action} & \text{with probability } \varepsilon \end{cases}$ (breaking ties randomly)
* $R \leftarrow bandit(A)$
* $N(A) \leftarrow N(A) + 1$
* $Q(A) \leftarrow Q(A) + \frac{1}{N(A)}\bigl[R - Q(A)\bigr]$

</div>

### Tracking a Nonstationary Problem

For nonstationary problems (where reward probabilities change over time), it makes sense to give more weight to recent rewards. This is done using a constant step-size parameter $\alpha \in (0, 1]$:

$$Q_{n+1} \doteq Q_n + \alpha\bigl[R_n - Q_n\bigr].$$

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Exponential Recency-Weighted Average)</span></p>

The constant step-size update results in $Q_{n+1}$ being a weighted average of past rewards and the initial estimate $Q_1$:

$$Q_{n+1} = (1 - \alpha)^n Q_1 + \sum_{i=1}^{n} \alpha (1-\alpha)^{n-i} R_i.$$

This is called an **exponential recency-weighted average** because the weight $\alpha(1-\alpha)^{n-i}$ given to reward $R_i$ decreases exponentially according to the exponent on $1-\alpha$ as the number of intervening rewards increases.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Convergence Conditions for Step-Size Parameters)</span></p>

Let $\alpha_n(a)$ denote the step-size parameter used to process the reward received after the $n$th selection of action $a$. The conditions required to assure convergence with probability 1 are:

$$\sum_{n=1}^{\infty} \alpha_n(a) = \infty \qquad \text{and} \qquad \sum_{n=1}^{\infty} \alpha_n^2(a) < \infty.$$

The first condition guarantees that the steps are large enough to eventually overcome any initial conditions or random fluctuations. The second condition guarantees that eventually the steps become small enough to assure convergence.

* The sample-average case $\alpha_n(a) = \frac{1}{n}$ meets both conditions.
* The constant step-size case $\alpha_n(a) = \alpha$ does not meet the second condition, meaning estimates never completely converge but continue to vary in response to the most recently received rewards. This is actually desirable in a nonstationary environment.

</div>

### Optimistic Initial Values

All methods discussed so far are dependent on the initial action-value estimates $Q_1(a)$. For sample-average methods, the bias disappears once all actions have been selected at least once. For methods with constant $\alpha$, the bias is permanent, though decreasing over time.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Optimistic Initial Values)</span></p>

Setting initial action values optimistically (e.g. $Q_1(a) = +5$ when the true values are drawn from $\mathcal{N}(0, 1)$) encourages action-value methods to explore. Whichever actions are initially selected, the reward is less than the starting estimates; the learner switches to other actions, being "disappointed" with the rewards it is receiving. All actions are tried several times before the value estimates converge.

This technique is called **optimistic initial values**. It is a simple trick effective on stationary problems, but it is not well suited to nonstationary problems because its drive for exploration is inherently temporary.

</div>

### Upper-Confidence-Bound Action Selection

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Upper-Confidence-Bound (UCB) Action Selection)</span></p>

$\varepsilon$-greedy action selection forces exploration indiscriminately, with no preference for actions that are nearly greedy or particularly uncertain. A better approach is to select actions according to their potential for actually being optimal:

$$A_t \doteq \argmax_a \left[ Q_t(a) + c\sqrt{\frac{\ln t}{N_t(a)}} \right],$$

where $N_t(a)$ denotes the number of times action $a$ has been selected prior to time $t$, $c > 0$ controls the degree of exploration, and if $N_t(a) = 0$, then $a$ is considered to be a maximizing action.

The square-root term is a measure of the uncertainty or variance in the estimate of $a$'s value. The quantity being max'ed over is a sort of upper bound on the possible true value of action $a$, with $c$ determining the confidence level. Each time $a$ is selected, the uncertainty is reduced ($N_t(a)$ increments), and each time another action is selected, $t$ increases but $N_t(a)$ does not, so the uncertainty estimate increases.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(UCB Limitations)</span></p>

UCB often performs well but is more difficult than $\varepsilon$-greedy to extend beyond bandits to the more general reinforcement learning settings. Difficulties include dealing with nonstationary problems and dealing with large state spaces (particularly when using function approximation).

</div>

### Gradient Bandit Algorithms

Instead of estimating action values, we can learn a numerical **preference** for each action $a$, denoted $H_t(a) \in \mathbb{R}$. The larger the preference, the more often that action is taken, but the preference has no interpretation in terms of reward. Only the relative preference matters.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Soft-max Action Probabilities)</span></p>

The action probabilities are determined according to a **soft-max distribution** (Gibbs or Boltzmann distribution):

$$\Pr\lbrace A_t = a \rbrace \doteq \frac{e^{H_t(a)}}{\sum_{b=1}^{k} e^{H_t(b)}} \doteq \pi_t(a),$$

where $\pi_t(a)$ is the probability of taking action $a$ at time $t$. Initially all action preferences are the same (e.g. $H_1(a) = 0$ for all $a$) so that all actions have equal probability of being selected.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Gradient Bandit Algorithm)</span></p>

On each step, after selecting action $A_t$ and receiving reward $R_t$, the action preferences are updated by:

$$H_{t+1}(A_t) \doteq H_t(A_t) + \alpha\bigl(R_t - \bar{R}_t\bigr)\bigl(1 - \pi_t(A_t)\bigr),$$

$$H_{t+1}(a) \doteq H_t(a) - \alpha\bigl(R_t - \bar{R}_t\bigr)\pi_t(a), \qquad \text{for all } a \neq A_t,$$

where $\alpha > 0$ is a step-size parameter, and $\bar{R}_t \in \mathbb{R}$ is the average of all rewards up to but not including time $t$ (computed incrementally). The $\bar{R}_t$ term serves as a **baseline**: if the reward is higher than the baseline, the probability of taking $A_t$ in the future is increased, and if the reward is below baseline, the probability is decreased. The non-selected actions move in the opposite direction.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Gradient Bandit as Stochastic Gradient Ascent)</span></p>

The gradient bandit algorithm is an instance of **stochastic gradient ascent**. In exact gradient ascent, each action preference $H_t(a)$ would be incremented in proportion to the increment's effect on performance:

$$H_{t+1}(a) \doteq H_t(a) + \alpha \frac{\partial \mathbb{E}[R_t]}{\partial H_t(a)},$$

where the performance measure is the expected reward $\mathbb{E}[R_t] = \sum_x \pi_t(x) q_*(x)$. It can be shown that the expected update of the gradient bandit algorithm (2.12) is equal to this gradient, using the identity:

$$\frac{\partial \pi_t(x)}{\partial H_t(a)} = \pi_t(x)\bigl(\mathbb{1}_{a=x} - \pi_t(a)\bigr).$$

The choice of the baseline does not affect the expected update of the algorithm, but it does affect the variance of the update and thus the rate of convergence.

</div>

### Associative Search (Contextual Bandits)

So far we have considered only nonassociative tasks, where there is no need to associate different actions with different situations. In a general reinforcement learning task there is more than one situation, and the goal is to learn a **policy**: a mapping from situations to the actions that are best in those situations.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Associative Search / Contextual Bandits)</span></p>

An **associative search** task involves both trial-and-error learning to *search* for the best actions, and *association* of these actions with the situations in which they are best. Associative search tasks are often called **contextual bandits**. They are intermediate between the $k$-armed bandit problem and the full reinforcement learning problem:

* Like the full RL problem, they involve learning a policy.
* Like the $k$-armed bandit, each action affects only the immediate reward.

If actions are allowed to affect the *next situation* as well as the reward, then we have the full reinforcement learning problem.

</div>

### Summary

| Method | Key idea | Parameter | Strengths | Weaknesses |
| --- | --- | --- | --- | --- |
| $\varepsilon$-greedy | Random exploration with probability $\varepsilon$ | $\varepsilon$ | Simple, effective | Indiscriminate exploration |
| UCB | Select by value + confidence bound | $c$ | Favors uncertain actions | Hard to extend beyond bandits |
| Gradient bandit | Learn preferences via soft-max | $\alpha$ | Probabilistic, graded selection | More complex analysis |
| Optimistic init. | High initial values drive exploration | $Q_0$ | Simple trick, effective on stationary | Temporary, not suited for nonstationary |

All algorithms have a parameter; to get a meaningful comparison, their performance must be considered as a function of their parameter. All algorithms perform best at an intermediate value of their parameter, neither too large nor too small.

## Chapter 3: Finite Markov Decision Processes

Finite Markov decision processes (finite MDPs) involve evaluative feedback, as in bandits, but also an associative aspect — choosing different actions in different situations. MDPs are a classical formalization of sequential decision making, where actions influence not just immediate rewards, but also subsequent situations, or states, and through those future rewards. This introduces the need to trade off immediate and delayed reward.

### The Agent–Environment Interface

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Markov Decision Process)</span></p>

The agent and environment interact at each of a sequence of discrete time steps, $t = 0, 1, 2, 3, \dots$ At each time step $t$:

1. The agent receives a representation of the environment's **state**, $S_t \in \mathcal{S}$.
2. On that basis selects an **action**, $A_t \in \mathcal{A}(s)$.
3. One time step later, receives a numerical **reward**, $R_{t+1} \in \mathcal{R} \subset \mathbb{R}$, and finds itself in a new state, $S_{t+1}$.

The MDP and agent together give rise to a **trajectory**: $S_0, A_0, R_1, S_1, A_1, R_2, S_2, A_2, R_3, \dots$

In a **finite** MDP, the sets of states, actions, and rewards ($\mathcal{S}$, $\mathcal{A}$, and $\mathcal{R}$) all have a finite number of elements.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Dynamics Function)</span></p>

In a finite MDP, $R_t$ and $S_t$ have well defined discrete probability distributions dependent only on the preceding state and action. The **dynamics** of the MDP is defined by the four-argument function $p$:

$$p(s', r \mid s, a) \doteq \Pr\lbrace S_t = s', R_t = r \mid S_{t-1} = s, A_{t-1} = a \rbrace,$$

for all $s', s \in \mathcal{S}$, $r \in \mathcal{R}$, and $a \in \mathcal{A}(s)$. The function $p: \mathcal{S} \times \mathcal{R} \times \mathcal{S} \times \mathcal{A} \to [0, 1]$ is an ordinary deterministic function of four arguments that completely characterizes the environment's dynamics:

$$\sum_{s' \in \mathcal{S}} \sum_{r \in \mathcal{R}} p(s', r \mid s, a) = 1, \quad \text{for all } s \in \mathcal{S},\; a \in \mathcal{A}(s).$$

</div>

The fact that the probability of each possible value for $S_t$ and $R_t$ depends only on the immediately preceding state and action, $S_{t-1}$ and $A_{t-1}$, and not at all on earlier states and actions, is the **Markov property**. The state must include information about all aspects of the past agent–environment interaction that make a difference for the future.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Derived Quantities from the Dynamics Function)</span></p>

From the four-argument dynamics function $p$, one can compute:

* **State-transition probabilities** (three-argument):

$$p(s' \mid s, a) \doteq \Pr\lbrace S_t = s' \mid S_{t-1} = s, A_{t-1} = a \rbrace = \sum_{r \in \mathcal{R}} p(s', r \mid s, a).$$

* **Expected rewards** for state–action pairs (two-argument):

$$r(s, a) \doteq \mathbb{E}[R_t \mid S_{t-1} = s, A_{t-1} = a] = \sum_{r \in \mathcal{R}} r \sum_{s' \in \mathcal{S}} p(s', r \mid s, a).$$

* **Expected rewards** for state–action–next-state triples (three-argument):

$$r(s, a, s') \doteq \mathbb{E}[R_t \mid S_{t-1} = s, A_{t-1} = a, S_t = s'] = \sum_{r \in \mathcal{R}} r \frac{p(s', r \mid s, a)}{p(s' \mid s, a)}.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Agent–Environment Boundary)</span></p>

The boundary between agent and environment is typically not the same as the physical boundary of the agent's body. The boundary is drawn closer to the agent: the motors, mechanical linkages, and sensing hardware are usually considered parts of the environment. The general rule is that anything that cannot be changed arbitrarily by the agent is considered part of the environment. The agent–environment boundary represents the limit of the agent's **absolute control**, not of its knowledge.

</div>

### Goals and Rewards

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Reward Hypothesis)</span></p>

In reinforcement learning, the purpose or goal of the agent is formalized in terms of a special signal, the **reward**, $R_t \in \mathbb{R}$, passing from the environment to the agent. The agent's goal is to maximize the total amount of reward it receives over the long run. The **reward hypothesis** states:

> That all of what we mean by goals and purposes can be well thought of as the maximization of the expected value of the cumulative sum of a received scalar signal (called reward).

The reward signal is your way of communicating to the agent *what* you want achieved, not *how* you want it achieved.

</div>

### Returns and Episodes

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Return)</span></p>

The agent seeks to maximize the **expected return**, $G_t$, defined as some specific function of the reward sequence. In the simplest case, the return is the sum of the rewards:

$$G_t \doteq R_{t+1} + R_{t+2} + R_{t+3} + \cdots + R_T,$$

where $T$ is a final time step. This formulation makes sense for applications that have a natural notion of final time step — we call these **episodes**, and tasks with episodes are called **episodic tasks**. Each episode ends in a special state called the **terminal state**.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Discounted Return)</span></p>

For **continuing tasks** (no natural terminal state), the return could easily be infinite. We introduce **discounting**: the agent tries to maximize the expected **discounted return**:

$$G_t \doteq R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1},$$

where $\gamma \in [0, 1]$ is the **discount rate**.

* If $\gamma = 0$, the agent is "myopic": it only maximizes $R_{t+1}$.
* As $\gamma$ approaches 1, the agent becomes more farsighted, taking future rewards into account more strongly.
* If $\gamma < 1$ and the reward sequence $\lbrace R_k \rbrace$ is bounded, the infinite sum has a finite value.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Recursive Relationship of Returns)</span></p>

Returns at successive time steps are related in a way that is fundamental to reinforcement learning:

$$G_t = R_{t+1} + \gamma G_{t+1}.$$

This works for all time steps $t < T$, even if termination occurs at $t+1$, provided we define $G_T = 0$.

</div>

### Unified Notation for Episodic and Continuing Tasks

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Unified Return Notation)</span></p>

To cover both episodic and continuing tasks with a single notation, we can consider episode termination to be the entering of a special **absorbing state** that transitions only to itself and generates only rewards of zero. This allows us to write:

$$G_t \doteq \sum_{k=t+1}^{T} \gamma^{k-t-1} R_k,$$

including the possibility that $T = \infty$ or $\gamma = 1$ (but not both).

</div>

### Policies and Value Functions

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Policy)</span></p>

A **policy** is a mapping from states to probabilities of selecting each possible action. If the agent is following policy $\pi$ at time $t$, then $\pi(a \mid s)$ is the probability that $A_t = a$ if $S_t = s$. Like $p$, $\pi$ is an ordinary function; the "$\mid$" in the middle of $\pi(a \mid s)$ merely reminds us that it defines a probability distribution over $a \in \mathcal{A}(s)$ for each $s \in \mathcal{S}$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(State-Value Function)</span></p>

The **state-value function** of a state $s$ under a policy $\pi$, denoted $v_\pi(s)$, is the expected return when starting in $s$ and following $\pi$ thereafter:

$$v_\pi(s) \doteq \mathbb{E}_\pi[G_t \mid S_t = s] = \mathbb{E}_\pi\!\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \;\middle|\; S_t = s\right], \quad \text{for all } s \in \mathcal{S}.$$

The value of the terminal state, if any, is always zero.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Action-Value Function)</span></p>

The **action-value function** of taking action $a$ in state $s$ under a policy $\pi$, denoted $q_\pi(s, a)$, is the expected return starting from $s$, taking action $a$, and thereafter following $\pi$:

$$q_\pi(s, a) \doteq \mathbb{E}_\pi[G_t \mid S_t = s, A_t = a] = \mathbb{E}_\pi\!\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \;\middle|\; S_t = s, A_t = a\right].$$

The functions $v_\pi$ and $q_\pi$ can be estimated from experience. If an agent follows policy $\pi$ and maintains an average of the actual returns for each state encountered, the average will converge to $v_\pi(s)$. We call estimation methods of this kind **Monte Carlo methods**.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Bellman Equation for $v_\pi$)</span></p>

A fundamental property of value functions is that they satisfy recursive relationships. For any policy $\pi$ and any state $s$, the following consistency condition holds between the value of $s$ and the value of its possible successor states:

$$v_\pi(s) = \sum_a \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a) \bigl[r + \gamma v_\pi(s')\bigr], \quad \text{for all } s \in \mathcal{S}.$$

This is the **Bellman equation** for $v_\pi$. It expresses a relationship between the value of a state and the values of its successor states: the value of the start state must equal the (discounted) value of the expected next state, plus the reward expected along the way. The value function $v_\pi$ is the unique solution to its Bellman equation.

</div>

### Optimal Policies and Optimal Value Functions

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Optimal Policy and Optimal Value Functions)</span></p>

A policy $\pi$ is defined to be better than or equal to a policy $\pi'$ if its expected return is greater than or equal to that of $\pi'$ for all states: $\pi \geq \pi'$ if and only if $v_\pi(s) \geq v_{\pi'}(s)$ for all $s \in \mathcal{S}$. There is always at least one policy that is better than or equal to all other policies — this is an **optimal policy**, denoted $\pi_*$.

* The **optimal state-value function**: $v_*(s) \doteq \max_\pi v_\pi(s), \quad \text{for all } s \in \mathcal{S}.$

* The **optimal action-value function**: $q_*(s, a) \doteq \max_\pi q_\pi(s, a), \quad \text{for all } s \in \mathcal{S} \text{ and } a \in \mathcal{A}(s).$

We can write $q_*$ in terms of $v_*$:

$$q_*(s, a) = \mathbb{E}\bigl[R_{t+1} + \gamma v_*(S_{t+1}) \mid S_t = s, A_t = a\bigr].$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Bellman Optimality Equation for $v_*$)</span></p>

Because $v_*$ is the value function for a policy, it must satisfy the Bellman equation. Because it is the *optimal* value function, its consistency condition can be written in a special form without reference to any specific policy:

$$v_*(s) = \max_{a \in \mathcal{A}(s)} \sum_{s', r} p(s', r \mid s, a)\bigl[r + \gamma v_*(s')\bigr].$$

The Bellman optimality equation expresses the fact that the value of a state under an optimal policy must equal the expected return for the best action from that state.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Bellman Optimality Equation for $q_*$)</span></p>

$$q_*(s, a) = \sum_{s', r} p(s', r \mid s, a)\bigl[r + \gamma \max_{a'} q_*(s', a')\bigr].$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Optimal Policies from Optimal Value Functions)</span></p>

* For finite MDPs, the Bellman optimality equation for $v_*$ has a unique solution (it is a system of $n$ equations in $n$ unknowns, one for each state).
* Once one has $v_*$, it is relatively easy to determine an optimal policy: for each state $s$, any policy that assigns nonzero probability only to the actions at which the maximum is obtained in the Bellman optimality equation is an optimal policy. Any policy that is **greedy** with respect to $v_*$ is an optimal policy.
* With $q_*$, the agent does not even have to do a one-step-ahead search: for any state $s$, it can simply find any action that maximizes $q_*(s, a)$. The action-value function effectively caches the results of all one-step-ahead searches.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Practical Limitations of Bellman Optimality Equations)</span></p>

Explicitly solving the Bellman optimality equation is rarely directly useful. It relies on at least three assumptions that are rarely true in practice:

1. The dynamics of the environment are accurately known.
2. Computational resources are sufficient to complete the calculation.
3. The states have the Markov property.

For large state spaces (e.g., backgammon has about $10^{20}$ states), exact solutions are infeasible. Many reinforcement learning methods can be understood as approximately solving the Bellman optimality equation, using actual experienced transitions in place of knowledge of the expected transitions.

</div>

## Chapter 4: Dynamic Programming

The term dynamic programming (DP) refers to a collection of algorithms that can be used to compute optimal policies given a perfect model of the environment as a Markov decision process (MDP). Classical DP algorithms are of limited utility in reinforcement learning due to their assumption of a perfect model and their computational expense, but they are important theoretically — DP provides an essential foundation for understanding all subsequent methods. In fact, all later methods can be viewed as attempts to achieve the same effect as DP, only with less computation and without assuming a perfect model.

DP algorithms are obtained by turning Bellman equations into assignments, that is, into update rules for improving approximations of the desired value functions.

### Policy Evaluation (Prediction)

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Policy Evaluation / Prediction Problem)</span></p>

**Policy evaluation** is the computation of the state-value function $v_\pi$ for an arbitrary policy $\pi$. This is also called the **prediction problem**. Recall the Bellman equation for $v_\pi$:

$$v_\pi(s) = \sum_a \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a)\bigl[r + \gamma v_\pi(s')\bigr], \quad \text{for all } s \in \mathcal{S}.$$

If the environment's dynamics are completely known, this is a system of $\lvert\mathcal{S}\rvert$ simultaneous linear equations in $\lvert\mathcal{S}\rvert$ unknowns.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Iterative Policy Evaluation)</span></p>

Consider a sequence of approximate value functions $v_0, v_1, v_2, \dots$, where $v_0$ is chosen arbitrarily (except that the terminal state must be given value 0). Each successive approximation is obtained by using the Bellman equation for $v_\pi$ as an update rule:

$$v_{k+1}(s) \doteq \sum_a \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a)\bigl[r + \gamma v_k(s')\bigr], \quad \text{for all } s \in \mathcal{S}.$$

The sequence $\lbrace v_k \rbrace$ converges to $v_\pi$ as $k \to \infty$. This algorithm is called **iterative policy evaluation**. Each iteration updates the value of every state by replacing the old value with a new value obtained from the old values of the successor states and the expected immediate rewards.

**In-place version** (pseudocode):

* Input: $\pi$, the policy to be evaluated; threshold $\theta > 0$
* Initialize $V(s)$ arbitrarily for $s \in \mathcal{S}$, and $V(terminal) = 0$
* Loop:
  * $\Delta \leftarrow 0$
  * For each $s \in \mathcal{S}$:
    * $v \leftarrow V(s)$
    * $V(s) \leftarrow \sum_a \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a)[r + \gamma V(s')]$
    * $\Delta \leftarrow \max(\Delta, \lvert v - V(s) \rvert)$
  * until $\Delta < \theta$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Expected Updates and In-Place Updates)</span></p>

All updates done in DP are called **expected updates** because they are based on an expectation over all possible next states rather than on a sample next state. The in-place version (using a single array, overwriting old values immediately) usually converges faster than the two-array version because it uses new data as soon as they are available. We think of the updates as being done in a **sweep** through the state space.

</div>

### Policy Improvement

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Policy Improvement Theorem)</span></p>

Let $\pi$ and $\pi'$ be any pair of deterministic policies such that, for all $s \in \mathcal{S}$,

$$q_\pi(s, \pi'(s)) \geq v_\pi(s).$$

Then the policy $\pi'$ must be as good as, or better than, $\pi$. That is, it must obtain greater or equal expected return from all states $s \in \mathcal{S}$:

$$v_{\pi'}(s) \geq v_\pi(s).$$

Moreover, if there is strict inequality at any state, then there must be strict inequality at that state in the conclusion as well.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Policy Improvement / Greedy Policy)</span></p>

**Policy improvement** is the process of making a new policy that improves on an original policy, by making it greedy with respect to the value function of the original policy. The new **greedy policy** $\pi'$ is given by:

$$\pi'(s) \doteq \argmax_a q_\pi(s, a) = \argmax_a \sum_{s', r} p(s', r \mid s, a)\bigl[r + \gamma v_\pi(s')\bigr].$$

By construction, the greedy policy meets the conditions of the policy improvement theorem, so it is as good as, or better than, the original policy.

If $v_\pi = v_{\pi'}$ (the new greedy policy is as good as, but not better than, the old policy), then $v_{\pi'}$ must satisfy the Bellman optimality equation, and therefore both $\pi$ and $\pi'$ must be optimal policies.

</div>

### Policy Iteration

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Policy Iteration)</span></p>

Once a policy $\pi$ has been improved using $v_\pi$ to yield a better policy $\pi'$, we can then compute $v_{\pi'}$ and improve it again to yield an even better $\pi''$. We thus obtain a sequence of monotonically improving policies and value functions:

$$\pi_0 \xrightarrow{\text{E}} v_{\pi_0} \xrightarrow{\text{I}} \pi_1 \xrightarrow{\text{E}} v_{\pi_1} \xrightarrow{\text{I}} \pi_2 \xrightarrow{\text{E}} \cdots \xrightarrow{\text{I}} \pi_* \xrightarrow{\text{E}} v_*,$$

where $\xrightarrow{\text{E}}$ denotes a policy *evaluation* and $\xrightarrow{\text{I}}$ denotes a policy *improvement*. Each policy is guaranteed to be a strict improvement over the previous one (unless it is already optimal). Because a finite MDP has only a finite number of deterministic policies, this process must converge to an optimal policy in a finite number of iterations.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Policy Iteration)</span></p>

**1. Initialization:**
* $V(s) \in \mathbb{R}$ and $\pi(s) \in \mathcal{A}(s)$ arbitrarily for all $s \in \mathcal{S}$; $V(terminal) \doteq 0$

**2. Policy Evaluation:**
* Loop:
  * $\Delta \leftarrow 0$
  * For each $s \in \mathcal{S}$:
    * $v \leftarrow V(s)$
    * $V(s) \leftarrow \sum_{s', r} p(s', r \mid s, \pi(s))[r + \gamma V(s')]$
    * $\Delta \leftarrow \max(\Delta, \lvert v - V(s) \rvert)$
  * until $\Delta < \theta$

**3. Policy Improvement:**
* $\textit{policy-stable} \leftarrow \textit{true}$
* For each $s \in \mathcal{S}$:
  * $\textit{old-action} \leftarrow \pi(s)$
  * $\pi(s) \leftarrow \argmax_a \sum_{s', r} p(s', r \mid s, a)[r + \gamma V(s')]$
  * If $\textit{old-action} \neq \pi(s)$, then $\textit{policy-stable} \leftarrow \textit{false}$
* If $\textit{policy-stable}$, then stop and return $V \approx v_*$ and $\pi \approx \pi_*$; else go to 2

</div>

### Value Iteration

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Value Iteration)</span></p>

One drawback of policy iteration is that each of its iterations involves policy evaluation, which may itself be a protracted iterative computation. **Value iteration** is the special case when policy evaluation is stopped after just one sweep (one update of each state). It combines the policy improvement and truncated policy evaluation steps into a single update:

$$v_{k+1}(s) \doteq \max_a \sum_{s', r} p(s', r \mid s, a)\bigl[r + \gamma v_k(s')\bigr], \quad \text{for all } s \in \mathcal{S}.$$

For arbitrary $v_0$, the sequence $\lbrace v_k \rbrace$ converges to $v_*$ under the same conditions that guarantee the existence of $v_*$. Value iteration is obtained simply by turning the Bellman optimality equation into an update rule.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Value Iteration)</span></p>

Algorithm parameter: threshold $\theta > 0$
* Initialize $V(s)$ for all $s \in \mathcal{S}^+$, arbitrarily except that $V(terminal) = 0$

* Loop:
  * $\Delta \leftarrow 0$
  * For each $s \in \mathcal{S}$:
    * $v \leftarrow V(s)$
    * $V(s) \leftarrow \max_a \sum_{s', r} p(s', r \mid s, a)[r + \gamma V(s')]$
    * $\Delta \leftarrow \max(\Delta, \lvert v - V(s) \rvert)$
  * until $\Delta < \theta$

* Output a deterministic policy $\pi \approx \pi_*$, such that: $\pi(s) = \argmax_a \sum_{s', r} p(s', r \mid s, a)[r + \gamma V(s')]$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Relationship Between Policy Iteration and Value Iteration)</span></p>

Value iteration effectively combines, in each of its sweeps, one sweep of policy evaluation and one sweep of policy improvement. Faster convergence is often achieved by interposing multiple policy evaluation sweeps between each policy improvement sweep. The entire class of truncated policy iteration algorithms can be thought of as sequences of sweeps, some using policy evaluation updates and some using value iteration updates. All converge to an optimal policy for discounted finite MDPs.

</div>

### Asynchronous Dynamic Programming

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Asynchronous DP)</span></p>

**Asynchronous DP** algorithms are in-place iterative DP algorithms that are not organized in terms of systematic sweeps of the state set. They update the values of states in any order whatsoever, using whatever values of other states happen to be available. The values of some states may be updated several times before others are updated once.

To converge correctly, an asynchronous algorithm must continue to update all states — it cannot ignore any state after some point. Asynchronous DP algorithms allow great flexibility in selecting states to update and make it possible to:

* Intermix policy evaluation and value iteration updates.
* Run an iterative DP algorithm *at the same time that an agent is actually experiencing the MDP*, focusing updates on states most relevant to the agent.

</div>

### Generalized Policy Iteration

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Generalized Policy Iteration (GPI))</span></p>

**Generalized policy iteration** (GPI) refers to the general idea of letting policy-evaluation and policy-improvement processes interact, independent of the granularity and other details of the two processes. Almost all reinforcement learning methods are well described as GPI: the policy is always being improved with respect to the value function, and the value function is always being driven toward the value function for the policy.

The two processes can be viewed as both competing and cooperating:

* **Competing:** Making the policy greedy with respect to the value function typically makes the value function incorrect for the changed policy; making the value function consistent with the policy typically causes the policy to no longer be greedy.
* **Cooperating:** In the long run, the two processes interact to find a single joint solution: the optimal value function and an optimal policy.

Both processes stabilize only when a policy has been found that is greedy with respect to its own evaluation function — this implies that the Bellman optimality equation holds, and the policy and value function are optimal.

</div>

### Efficiency of Dynamic Programming

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Computational Efficiency of DP)</span></p>

* DP methods find an optimal policy in time **polynomial** in the number of states $n$ and actions $k$, even though the total number of deterministic policies is $k^n$. DP is thus exponentially faster than any direct search in policy space.
* Linear programming methods can also solve MDPs, and sometimes have better worst-case convergence guarantees, but they become impractical at a much smaller number of states than DP methods (by a factor of about 100).
* The **curse of dimensionality** — the number of states growing exponentially with the number of state variables — is an inherent difficulty of the problem, not of DP as a solution method. DP is comparatively better suited to handling large state spaces than competing methods.
* In practice, DP can solve MDPs with millions of states. Both policy iteration and value iteration are widely used, and they usually converge much faster than their theoretical worst-case run times.

</div>

### Summary

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Bootstrapping)</span></p>

A special property of DP methods is that all of them update estimates of the values of states based on estimates of the values of successor states — they update estimates on the basis of other estimates. This general idea is called **bootstrapping**. Many reinforcement learning methods perform bootstrapping, even those that do not require a complete and accurate model of the environment as DP requires.

</div>

| Method | Bellman equation used | Full sweeps? | Finds |
| --- | --- | --- | --- |
| Iterative policy evaluation | $v_\pi$ Bellman equation | Yes | $v_\pi$ |
| Policy iteration | Alternates evaluation + improvement | Yes | $\pi_*, v_*$ |
| Value iteration | Bellman optimality equation | Yes | $v_*, \pi_*$ |
| Asynchronous DP | Any Bellman equation | No (flexible order) | $v_*, \pi_*$ |

## Chapter 5: Monte Carlo Methods

Monte Carlo (MC) methods are ways of solving the reinforcement learning problem based on averaging sample returns. Unlike DP, Monte Carlo methods require only *experience* — sample sequences of states, actions, and rewards from actual or simulated interaction with an environment. They do not assume complete knowledge of the environment's dynamics. MC methods are defined only for episodic tasks: experience is divided into episodes that all eventually terminate.

Key advantages of Monte Carlo methods over DP:

* They can learn from actual experience with no model of the environment's dynamics.
* They can learn from simulated experience (the model need only generate sample transitions, not the complete probability distributions).
* The estimate for one state does not build upon the estimate of any other state — MC methods do **not bootstrap**.
* The computational expense of estimating the value of a single state is independent of the number of states.

### Monte Carlo Prediction

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(First-visit vs. Every-visit MC)</span></p>

Each occurrence of state $s$ in an episode is called a **visit** to $s$. The first time $s$ is visited in an episode is the **first visit**.

* The **first-visit MC method** estimates $v_\pi(s)$ as the average of the returns following first visits to $s$.
* The **every-visit MC method** averages the returns following all visits to $s$.

Both converge to $v_\pi(s)$ as the number of visits (or first visits) to $s$ goes to infinity. First-visit MC produces independent, unbiased estimates with standard deviation falling as $1/\sqrt{n}$. Every-visit MC estimates also converge quadratically.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(First-visit MC Prediction, for estimating $V \approx v_\pi$)</span></p>

* Input: a policy $\pi$ to be evaluated
* Initialize: $V(s) \in \mathbb{R}$ arbitrarily, $Returns(s) \leftarrow$ empty list, for all $s \in \mathcal{S}$

* Loop forever (for each episode):
  * Generate an episode following $\pi$: $S_0, A_0, R_1, S_1, A_1, R_2, \dots, S_{T-1}, A_{T-1}, R_T$
  * $G \leftarrow 0$
  * Loop for each step of episode, $t = T{-}1, T{-}2, \dots, 0$:
    * $G \leftarrow \gamma G + R_{t+1}$
    * Unless $S_t$ appears in $S_0, S_1, \dots, S_{t-1}$:
      * Append $G$ to $Returns(S_t)$
      * $V(S_t) \leftarrow \text{average}(Returns(S_t))$

</div>

### Monte Carlo Estimation of Action Values

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Action-Value Estimation and Exploring Starts)</span></p>

Without a model, state values alone are not sufficient to determine a policy — one must explicitly estimate the value of each action. Thus, one of the primary goals for MC methods is to estimate $q_\pi(s, a)$.

A state–action pair $s, a$ is said to be **visited** in an episode if state $s$ is visited and action $a$ is taken in it. The complication is that many state–action pairs may never be visited under a deterministic policy $\pi$. To maintain exploration, we can assume **exploring starts**: that episodes start in a state–action pair, and that every pair has a nonzero probability of being selected as the start. This guarantees that all state–action pairs will be visited infinitely often.

</div>

### Monte Carlo Control

MC control follows the pattern of generalized policy iteration (GPI): alternating between policy evaluation and policy improvement.

$$\pi_0 \xrightarrow{\text{E}} q_{\pi_0} \xrightarrow{\text{I}} \pi_1 \xrightarrow{\text{E}} q_{\pi_1} \xrightarrow{\text{I}} \pi_2 \xrightarrow{\text{E}} \cdots \xrightarrow{\text{I}} \pi_* \xrightarrow{\text{E}} q_*$$

Policy improvement is done by making the policy greedy with respect to the current action-value function. Since we have an action-value function, no model is needed:

$$\pi(s) \doteq \argmax_a q(s, a).$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Monte Carlo ES (Exploring Starts))</span></p>

* Initialize:
  * $\pi(s) \in \mathcal{A}(s)$ arbitrarily, for all $s \in \mathcal{S}$
  * $Q(s, a) \in \mathbb{R}$ arbitrarily, for all $s \in \mathcal{S}$, $a \in \mathcal{A}(s)$
  * $Returns(s, a) \leftarrow$ empty list, for all $s \in \mathcal{S}$, $a \in \mathcal{A}(s)$

* Loop forever (for each episode):
  * Choose $S_0 \in \mathcal{S}$, $A_0 \in \mathcal{A}(S_0)$ randomly such that all pairs have probability $> 0$
  * Generate an episode from $S_0, A_0$ following $\pi$: $S_0, A_0, R_1, \dots, S_{T-1}, A_{T-1}, R_T$
  * $G \leftarrow 0$
  * Loop for each step of episode, $t = T{-}1, T{-}2, \dots, 0$:
    * $G \leftarrow \gamma G + R_{t+1}$
    * Unless the pair $S_t, A_t$ appears in $S_0, A_0, S_1, A_1, \dots, S_{t-1}, A_{t-1}$:
      * Append $G$ to $Returns(S_t, A_t)$
      * $Q(S_t, A_t) \leftarrow \text{average}(Returns(S_t, A_t))$
      * $\pi(S_t) \leftarrow \argmax_a Q(S_t, a)$

</div>

### Monte Carlo Control without Exploring Starts

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(On-policy vs. Off-policy Methods)</span></p>

There are two approaches to ensuring continued exploration without exploring starts:

* **On-policy methods** evaluate or improve the policy that is used to make decisions. The policy is generally *soft* — $\pi(a \mid s) > 0$ for all $s \in \mathcal{S}$ and all $a \in \mathcal{A}(s)$ — but gradually shifted closer to a deterministic optimal policy.
* **Off-policy methods** evaluate or improve a policy different from the one used to generate the data. The policy being learned about is called the **target policy**, and the policy used to generate behavior is called the **behavior policy**.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($\varepsilon$-Greedy and $\varepsilon$-Soft Policies)</span></p>

An **$\varepsilon$-soft** policy is one for which $\pi(a \mid s) \geq \frac{\varepsilon}{\lvert\mathcal{A}(s)\rvert}$ for all states and actions, for some $\varepsilon > 0$. Among $\varepsilon$-soft policies, **$\varepsilon$-greedy** policies are in some sense those closest to greedy:

$$\pi(a \mid S_t) = \begin{cases} 1 - \varepsilon + \varepsilon / \lvert\mathcal{A}(S_t)\rvert & \text{if } a = A^* \\ \varepsilon / \lvert\mathcal{A}(S_t)\rvert & \text{if } a \neq A^* \end{cases}$$

where $A^* = \argmax_a Q(S_t, a)$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(On-policy First-visit MC Control, for $\varepsilon$-soft policies)</span></p>

* Algorithm parameter: small $\varepsilon > 0$
* Initialize: $\pi \leftarrow$ an arbitrary $\varepsilon$-soft policy; $Q(s, a) \in \mathbb{R}$ arbitrarily; $Returns(s, a) \leftarrow$ empty list

* Repeat forever (for each episode):
  * Generate an episode following $\pi$: $S_0, A_0, R_1, \dots, S_{T-1}, A_{T-1}, R_T$
  * $G \leftarrow 0$
  * Loop for each step of episode, $t = T{-}1, T{-}2, \dots, 0$:
    * $G \leftarrow \gamma G + R_{t+1}$
    * Unless the pair $S_t, A_t$ appears in $S_0, A_0, \dots, S_{t-1}, A_{t-1}$:
      * Append $G$ to $Returns(S_t, A_t)$
      * $Q(S_t, A_t) \leftarrow \text{average}(Returns(S_t, A_t))$
      * $A^* \leftarrow \argmax_a Q(S_t, a)$
      * For all $a \in \mathcal{A}(S_t)$: $\;\pi(a \mid S_t) \leftarrow \begin{cases} 1 - \varepsilon + \varepsilon/\lvert\mathcal{A}(S_t)\rvert & \text{if } a = A^* \\ \varepsilon/\lvert\mathcal{A}(S_t)\rvert & \text{if } a \neq A^* \end{cases}$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Limitation of On-policy Methods)</span></p>

On-policy methods are a compromise: they learn action values not for the optimal policy, but for a near-optimal policy that still explores. We only achieve the best policy among the $\varepsilon$-soft policies, but we have eliminated the assumption of exploring starts.

</div>

### Off-policy Prediction via Importance Sampling

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Coverage Assumption)</span></p>

In off-policy methods, we wish to estimate $v_\pi$ or $q_\pi$ from episodes following a different behavior policy $b$. We require the assumption of **coverage**: every action taken under $\pi$ must also be taken, at least occasionally, under $b$. That is, $\pi(a \mid s) > 0$ implies $b(a \mid s) > 0$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Importance-Sampling Ratio)</span></p>

Almost all off-policy methods utilize **importance sampling**, weighting returns according to the relative probability of their trajectories under the target and behavior policies. The **importance-sampling ratio** for a trajectory starting at time $t$ is:

$$\rho_{t:T-1} \doteq \prod_{k=t}^{T-1} \frac{\pi(A_k \mid S_k)}{b(A_k \mid S_k)}.$$

Note that the MDP's transition probabilities, $p(s' \mid s, a)$, appear identically in numerator and denominator and cancel. The ratio depends only on the two policies and the sequence, not on the MDP.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Ordinary vs. Weighted Importance Sampling)</span></p>

Let $\mathcal{T}(s)$ be the set of all time steps in which state $s$ is visited. Two forms of importance sampling:

* **Ordinary importance sampling** — a simple average of scaled returns:

$$V(s) \doteq \frac{\sum_{t \in \mathcal{T}(s)} \rho_{t:T(t)-1}\, G_t}{\lvert\mathcal{T}(s)\rvert}.$$

* **Weighted importance sampling** — a weighted average:

$$V(s) \doteq \frac{\sum_{t \in \mathcal{T}(s)} \rho_{t:T(t)-1}\, G_t}{\sum_{t \in \mathcal{T}(s)} \rho_{t:T(t)-1}}.$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Ordinary vs. Weighted Importance Sampling)</span></p>

| Property | Ordinary IS | Weighted IS |
| --- | --- | --- |
| Bias | Unbiased | Biased (bias $\to 0$ asymptotically) |
| Variance | Can be unbounded (even infinite) | Bounded; dramatically lower in practice |
| First-visit, single return | Unbiased, high variance | Equals $v_b(s)$, not $v_\pi(s)$ |

In practice, the weighted estimator usually has dramatically lower variance and is strongly preferred. The every-visit methods for both are biased, but the bias falls asymptotically to zero.

</div>

### Incremental Implementation

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Incremental Weighted Importance Sampling Update)</span></p>

For off-policy MC with weighted importance sampling, given a sequence of returns $G_1, G_2, \dots, G_{n-1}$ with corresponding weights $W_k = \rho_{t_k:T(t_k)-1}$, the estimate can be maintained incrementally:

$$V_{n+1} \doteq V_n + \frac{W_n}{C_n}\bigl[G_n - V_n\bigr], \qquad n \geq 1,$$

where $C_{n+1} \doteq C_n + W_{n+1}$, with $C_0 \doteq 0$. This applies to both on-policy (where $W$ is always 1) and off-policy cases.

</div>

### Off-policy Monte Carlo Control

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Off-policy MC Control)</span></p>

* Initialize, for all $s \in \mathcal{S}$, $a \in \mathcal{A}(s)$:
  * $Q(s, a) \in \mathbb{R}$ arbitrarily
  * $C(s, a) \leftarrow 0$
  * $\pi(s) \leftarrow \argmax_a Q(s, a)$ (deterministic greedy target policy)

* Loop forever (for each episode):
  * $b \leftarrow$ any soft policy with coverage of $\pi$
  * Generate an episode following $b$: $S_0, A_0, R_1, \dots, S_{T-1}, A_{T-1}, R_T$
  * $G \leftarrow 0$, $\;W \leftarrow 1$
  * Loop for each step of episode, $t = T{-}1, T{-}2, \dots, 0$, while $W \neq 0$:
    * $G \leftarrow \gamma G + R_{t+1}$
    * $C(S_t, A_t) \leftarrow C(S_t, A_t) + W$
    * $Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \frac{W}{C(S_t, A_t)}[G - Q(S_t, A_t)]$
    * $\pi(S_t) \leftarrow \argmax_a Q(S_t, a)$
    * If $A_t \neq \pi(S_t)$, then exit inner loop (proceed to next episode)
    * $W \leftarrow W \cdot \frac{1}{b(A_t \mid S_t)}$

The target policy $\pi$ converges to optimal at all encountered states even though actions are selected according to a different soft policy $b$.

</div>

### Summary

| Aspect | DP | Monte Carlo |
| --- | --- | --- |
| Model required? | Yes (complete) | No (only sample episodes) |
| Bootstraps? | Yes | No |
| Handles episodic tasks? | Yes | Yes (MC requires it) |
| Handles continuing tasks? | Yes | Not directly |
| Update basis | Expected updates (all successors) | Sample updates (single episode) |
| Cost per state | Depends on number of states | Independent of number of states |

## Chapter 6: Temporal-Difference Learning

If one had to identify one idea as central and novel to reinforcement learning, it would undoubtedly be *temporal-difference* (TD) learning. TD learning is a combination of Monte Carlo ideas and dynamic programming (DP) ideas. Like Monte Carlo methods, TD methods can learn directly from raw experience without a model of the environment's dynamics. Like DP, TD methods update estimates based in part on other learned estimates, without waiting for a final outcome (they bootstrap).

The chapter focuses first on the **prediction** problem (estimating $v_\pi$ for a given policy $\pi$), then on the **control** problem (finding an optimal policy). For control, DP, TD, and Monte Carlo methods all use some variation of generalized policy iteration (GPI). The differences in the methods are primarily differences in their approaches to the prediction problem.

### 6.1 TD Prediction

Both TD and Monte Carlo methods use experience to solve the prediction problem. Given some experience following a policy $\pi$, both methods update their estimate $V$ of $v_\pi$ for the nonterminal states $S_t$ occurring in that experience. Monte Carlo methods wait until the return following the visit is known, then use that return as a target for $V(S_t)$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Constant-$\alpha$ MC Update)</span></p>

A simple every-visit Monte Carlo method suitable for nonstationary environments is:

$$V(S_t) \leftarrow V(S_t) + \alpha\bigl[G_t - V(S_t)\bigr],$$

where $G_t$ is the actual return following time $t$, and $\alpha$ is a constant step-size parameter. Monte Carlo methods must wait until the end of the episode to determine the increment to $V(S_t)$ (only then is $G_t$ known).

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(TD(0) Update)</span></p>

TD methods need to wait only until the next time step. At time $t+1$ they immediately form a target and make a useful update using the observed reward $R_{t+1}$ and the estimate $V(S_{t+1})$. The simplest TD method, called **TD(0)** or **one-step TD**, makes the update:

$$V(S_t) \leftarrow V(S_t) + \alpha\bigl[R_{t+1} + \gamma V(S_{t+1}) - V(S_t)\bigr].$$

The target for the Monte Carlo update is $G_t$, whereas the target for the TD update is $R_{t+1} + \gamma V(S_{t+1})$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Tabular TD(0) for Estimating $v_\pi$)</span></p>

**Input:** the policy $\pi$ to be evaluated

**Algorithm parameter:** step size $\alpha \in (0, 1]$

Initialize $V(s)$, for all $s \in \mathcal{S}^+$, arbitrarily except that $V(terminal) = 0$

Loop for each episode:
* Initialize $S$
* Loop for each step of episode:
  * $A \leftarrow$ action given by $\pi$ for $S$
  * Take action $A$, observe $R$, $S'$
  * $V(S) \leftarrow V(S) + \alpha\bigl[R + \gamma V(S') - V(S)\bigr]$
  * $S \leftarrow S'$
* until $S$ is terminal

</div>

Because TD(0) bases its update in part on an existing estimate, we say that it is a **bootstrapping** method, like DP. We know from Chapter 3 that:

$$v_\pi(s) \doteq \mathbb{E}_\pi[G_t \mid S_t = s] = \mathbb{E}_\pi[R_{t+1} + \gamma G_{t+1} \mid S_t = s] = \mathbb{E}_\pi[R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s].$$

Roughly speaking, Monte Carlo methods use an estimate of the first expression as a target, whereas DP methods use an estimate of the last expression as a target. The Monte Carlo target is an estimate because the expected value is not known; a sample return is used in place of the real expected return. The DP target is an estimate not because of the expected values, but because $v_\pi(S_{t+1})$ is not known and the current estimate $V(S_{t+1})$ is used instead. The TD target is an estimate for both reasons: it samples the expected values *and* it uses the current estimate $V$ instead of the true $v_\pi$. Thus, TD methods combine the sampling of Monte Carlo with the bootstrapping of DP.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Sample Updates)</span></p>

TD and Monte Carlo updates are referred to as **sample updates** because they involve looking ahead to a sample successor state (or state–action pair), using the value of the successor and the reward along the way to compute a backed-up value, and then updating the value of the original state (or state–action pair) accordingly. Sample updates differ from the *expected* updates of DP methods in that they are based on a single sample successor rather than on a complete distribution of all possible successors.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(TD Error)</span></p>

The quantity in brackets in the TD(0) update is a sort of error, measuring the difference between the estimated value of $S_t$ and the better estimate $R_{t+1} + \gamma V(S_{t+1})$. This quantity, called the **TD error**, arises in various forms throughout reinforcement learning:

$$\delta_t \doteq R_{t+1} + \gamma V(S_{t+1}) - V(S_t).$$

The TD error at each time is the error in the estimate *made at that time*. Because the TD error depends on the next state and next reward, it is not actually available until one time step later. That is, $\delta_t$ is the error in $V(S_t)$, available at time $t+1$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Monte Carlo Error as Sum of TD Errors)</span></p>

If the array $V$ does not change during the episode (as it does not in Monte Carlo methods), then the Monte Carlo error can be written as a sum of TD errors:

$$G_t - V(S_t) = \sum_{k=t}^{T-1} \gamma^{k-t} \delta_k.$$

This identity is not exact if $V$ is updated during the episode (as it is in TD(0)), but if the step size is small then it may still hold approximately. Generalizations of this identity play an important role in the theory and algorithms of temporal-difference learning.

</div>

### 6.1.1 Example: Driving Home

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Driving Home)</span></p>

Each day as you drive home from work, you try to predict how long it will take to get home. You note the time, the day of week, the weather, and anything else that might be relevant. The sequence of states, elapsed times, and predictions is:

| State | Elapsed Time (min) | Predicted Time to Go | Predicted Total Time |
| --- | --- | --- | --- |
| leaving office, friday at 6 | 0 | 30 | 30 |
| reach car, raining | 5 | 35 | 40 |
| exiting highway | 20 | 15 | 35 |
| 2ndary road, behind truck | 30 | 10 | 40 |
| entering home street | 40 | 3 | 43 |
| arrive home | 43 | 0 | 43 |

With Monte Carlo methods, you must wait until you get home before increasing your estimate for the initial state. With a TD approach, you would learn immediately, shifting your initial estimate from 30 minutes toward 50 upon discovering the traffic jam. Each estimate would be shifted toward the estimate that immediately follows it — this is proportional to the change over time of the prediction, that is, to the *temporal differences* in predictions.

</div>

### 6.2 Advantages of TD Prediction Methods

TD methods have several advantages:

* **Over DP methods:** TD methods do not require a model of the environment, of its reward and next-state probability distributions.
* **Over Monte Carlo methods:** TD methods are naturally implemented in an online, fully incremental fashion. With Monte Carlo methods one must wait until the end of an episode. This is critical when episodes are very long, when there are continuing tasks with no episodes, or when some Monte Carlo methods must ignore or discount experimental episodes.
* **Online learning:** TD methods learn from each transition regardless of what subsequent actions are taken, making them much less susceptible to the problems caused by long episodes or continuing tasks.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Convergence of TD(0))</span></p>

For any fixed policy $\pi$, TD(0) has been proved to converge to $v_\pi$, in the mean for a constant step-size parameter if it is sufficiently small, and with probability 1 if the step-size parameter decreases according to the usual stochastic approximation conditions. Most convergence proofs apply only to the table-based case of the algorithm, but some also apply to the case of general linear function approximation.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Random Walk)</span></p>

A **Markov reward process** (MRP) is a Markov decision process without actions. Consider the following MRP with five states A through E:

All episodes start in the center state C, then proceed either left or right by one state on each step, with equal probability. Episodes terminate on the extreme left or right. A reward of $+1$ occurs only when terminating on the right; all other rewards are zero. Because this task is undiscounted, the true value of each state is the probability of terminating on the right. The true values of states A through E are $\frac{1}{6}, \frac{2}{6}, \frac{3}{6}, \frac{4}{6}, \frac{5}{6}$.

On this task, the TD method was consistently better than the MC method. The estimates after 100 episodes are about as close as they ever come to the true values with a constant step-size parameter ($\alpha = 0.1$).

</div>

### 6.3 Optimality of TD(0)

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Batch Updating)</span></p>

Suppose there is available only a finite amount of experience, say 10 episodes or 100 time steps. A common approach with incremental learning methods is to present the experience repeatedly until the method converges upon an answer. Given an approximate value function $V$, the increments specified by TD(0) or MC are computed for every time step $t$ at which a nonterminal state is visited, but the value function is changed only once, by the sum of all the increments. Then all the available experience is processed again with the new value function to produce a new overall increment, and so on, until the value function converges. We call this **batch updating**.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Batch TD(0) vs Batch Monte Carlo)</span></p>

Under batch updating:
* **Batch constant-$\alpha$ MC** converges to values $V(s)$ that are sample averages of the actual returns experienced after visiting each state $s$. These estimates minimize the mean square error on the training set.
* **Batch TD(0)** converges to the estimates that would be exactly correct for the maximum-likelihood model of the Markov process. It always finds the estimates that would be exactly correct if the model were exactly correct.

The batch TD method was consistently better than the batch Monte Carlo method on the random walk task, even though MC gives minimum squared error on the training data.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Certainty-Equivalence Estimate)</span></p>

The **maximum-likelihood estimate** of a parameter is the parameter value whose probability of generating the data is greatest. For a Markov process, the estimated transition probability from $i$ to $j$ is the fraction of observed transitions from $i$ that went to $j$, and the associated expected reward is the average of the rewards observed on those transitions. Given this model, we can compute the estimate of the value function that would be exactly correct if the model were exactly correct. This is called the **certainty-equivalence estimate** because it is equivalent to assuming that the estimate of the underlying process was known with certainty rather than being approximated. In general, batch TD(0) converges to the certainty-equivalence estimate.

This helps explain why TD methods converge more quickly than Monte Carlo methods: in batch form, TD(0) is faster because it computes the true certainty-equivalence estimate. Although the nonbatch methods do not achieve either the certainty-equivalence or the minimum squared error estimates, they can be understood as moving roughly in these directions.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(You Are the Predictor)</span></p>

Suppose you observe the following eight episodes of an unknown Markov reward process:

A, 0, B, 0 $\quad$ B, 1 $\quad$ B, 1 $\quad$ B, 1 $\quad$ B, 1 $\quad$ B, 1 $\quad$ B, 0 $\quad$ B, 1

The optimal value for $V(\text{B})$ is $\frac{3}{4}$ (six out of eight times in B the process terminated with reward 1). For $V(\text{A})$ there are two reasonable answers:
* **Batch TD(0) answer:** $V(\text{A}) = \frac{3}{4}$. Since 100% of the time state A transitioned to B, and $V(\text{B}) = \frac{3}{4}$, then $V(\text{A}) = \frac{3}{4}$ as well. This is based on first modeling the Markov process, then computing values from the model.
* **Batch MC answer:** $V(\text{A}) = 0$. We have seen A once and the return that followed was 0. This gives minimum squared error on the training data.

If the process is Markov, the first answer will produce lower error on *future* data, even though the MC answer is better on the existing data.

</div>

### 6.4 Sarsa: On-policy TD Control

We turn now to the use of TD prediction methods for the control problem. As usual, we follow the pattern of generalized policy iteration (GPI), only this time using TD methods for the evaluation or prediction part.

The first step is to learn an action-value function rather than a state-value function. For an on-policy method we must estimate $q_\pi(s, a)$ for the current behavior policy $\pi$ and for all states $s$ and actions $a$. An episode consists of an alternating sequence of states and state–action pairs:

$$\dots, S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1}, R_{t+2}, S_{t+2}, A_{t+2}, \dots$$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Sarsa Update)</span></p>

We consider transitions from state–action pair to state–action pair, and learn the values of state–action pairs. The update rule for action values under TD(0) is:

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha\bigl[R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)\bigr].$$

This update is done after every transition from a nonterminal state $S_t$. If $S_{t+1}$ is terminal, then $Q(S_{t+1}, A_{t+1})$ is defined as zero. This rule uses every element of the quintuple of events $(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$ that make up a transition from one state–action pair to the next. This quintuple gives rise to the name **Sarsa**.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Sarsa: On-policy TD Control for Estimating $Q \approx q_*$)</span></p>

**Algorithm parameters:** step size $\alpha \in (0, 1]$, small $\varepsilon > 0$

Initialize $Q(s, a)$, for all $s \in \mathcal{S}^+$, $a \in \mathcal{A}(s)$, arbitrarily except that $Q(terminal, \cdot) = 0$

Loop for each episode:
* Initialize $S$
* Choose $A$ from $S$ using policy derived from $Q$ (e.g., $\varepsilon$-greedy)
* Loop for each step of episode:
  * Take action $A$, observe $R$, $S'$
  * Choose $A'$ from $S'$ using policy derived from $Q$ (e.g., $\varepsilon$-greedy)
  * $Q(S, A) \leftarrow Q(S, A) + \alpha\bigl[R + \gamma Q(S', A') - Q(S, A)\bigr]$
  * $S \leftarrow S'$; $A \leftarrow A'$
* until $S$ is terminal

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Convergence of Sarsa)</span></p>

Sarsa converges with probability 1 to an optimal policy and action-value function, under the usual conditions on the step sizes and as long as all state–action pairs are visited an infinite number of times and the policy converges in the limit to the greedy policy (which can be arranged, for example, with $\varepsilon$-greedy policies by setting $\varepsilon = 1/t$).

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Windy Gridworld)</span></p>

A standard gridworld with start and goal states, but with a crosswind running upward through the middle of the grid. The actions are the standard four — up, down, right, and left — but in the middle region the resultant next states are shifted upward by a "wind" whose strength varies from column to column. This is an undiscounted episodic task, with constant rewards of $-1$ until the goal state is reached. Applying $\varepsilon$-greedy Sarsa ($\varepsilon = 0.1$, $\alpha = 0.5$, initial $Q(s, a) = 0$), the goal was reached more quickly over time — by 8000 time steps the greedy policy was long since optimal.

Note that Monte Carlo methods cannot easily be used here because termination is not guaranteed for all policies. If a policy was ever found that caused the agent to stay in the same state, the next episode would never end. Online learning methods such as Sarsa do not have this problem because they quickly learn *during the episode* that such policies are poor.

</div>

### 6.5 Q-learning: Off-policy TD Control

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Q-learning)</span></p>

One of the early breakthroughs in reinforcement learning was the development of an off-policy TD control algorithm known as **Q-learning** (Watkins, 1989), defined by:

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha\bigl[R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t)\bigr].$$

The learned action-value function $Q$ directly approximates $q_*$, the optimal action-value function, independent of the policy being followed. The policy still has an effect in that it determines which state–action pairs are visited and updated. However, all that is required for correct convergence is that all pairs continue to be updated.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Q-learning: Off-policy TD Control for Estimating $\pi \approx \pi_*$)</span></p>

**Algorithm parameters:** step size $\alpha \in (0, 1]$, small $\varepsilon > 0$

Initialize $Q(s, a)$, for all $s \in \mathcal{S}^+$, $a \in \mathcal{A}(s)$, arbitrarily except that $Q(terminal, \cdot) = 0$

Loop for each episode:
* Initialize $S$
* Loop for each step of episode:
  * Choose $A$ from $S$ using policy derived from $Q$ (e.g., $\varepsilon$-greedy)
  * Take action $A$, observe $R$, $S'$
  * $Q(S, A) \leftarrow Q(S, A) + \alpha\bigl[R + \gamma \max_a Q(S', a) - Q(S, A)\bigr]$
  * $S \leftarrow S'$
* until $S$ is terminal

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Q-learning is Off-policy)</span></p>

Q-learning is considered an off-policy method because the update rule uses $\max_a Q(S_{t+1}, a)$, which directly estimates the value under the *greedy* (target) policy, regardless of the (behavior) policy actually used to select actions. The backup diagram for Q-learning has a state–action pair at the top (filled action node) with the update going over all action nodes in the next state, maximizing over them (indicated by an arc).

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Cliff Walking)</span></p>

A standard undiscounted, episodic gridworld with start (S) and goal (G) states. The usual actions cause movement up, down, right, and left. Reward is $-1$ on all transitions except those into a region marked "The Cliff" — stepping into this region incurs a reward of $-100$ and sends the agent instantly back to the start.

With $\varepsilon$-greedy action selection ($\varepsilon = 0.1$):
* **Q-learning** learns the values of the *optimal* policy (the path right along the edge of the cliff), but its online performance is worse because the $\varepsilon$-greedy exploration occasionally steps off the cliff.
* **Sarsa** takes the action selection into account and learns the longer but *safer* path through the upper part of the grid, resulting in better online performance.

Although Q-learning learns the optimal policy values, its online performance is worse than Sarsa, which learns the roundabout policy. If $\varepsilon$ were gradually reduced, both methods would asymptotically converge to the optimal policy.

</div>

### 6.6 Expected Sarsa

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Expected Sarsa)</span></p>

**Expected Sarsa** is just like Q-learning except that instead of the maximum over next state–action pairs it uses the expected value, taking into account how likely each action is under the current policy. The update rule is:

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha\bigl[R_{t+1} + \gamma \mathbb{E}_\pi[Q(S_{t+1}, A_{t+1}) \mid S_{t+1}] - Q(S_t, A_t)\bigr]$$

$$= Q(S_t, A_t) + \alpha\Bigl[R_{t+1} + \gamma \sum_a \pi(a \mid S_{t+1}) Q(S_{t+1}, a) - Q(S_t, A_t)\Bigr].$$

Given the next state $S_{t+1}$, this algorithm moves *deterministically* in the same direction as Sarsa moves *in expectation*, and accordingly it is called Expected Sarsa.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Expected Sarsa)</span></p>

* Expected Sarsa is more complex computationally than Sarsa but, in return, it eliminates the variance due to the random selection of $A_{t+1}$.
* Given the same amount of experience we might expect it to perform slightly better than Sarsa, and indeed it generally does.
* On the cliff walking task, Expected Sarsa retains the significant advantage of Sarsa over Q-learning, and in addition shows a significant improvement over Sarsa over a wide range of values for $\alpha$.
* In cliff walking the state transitions are all deterministic and all randomness comes from the policy. Expected Sarsa can safely set $\alpha = 1$ without suffering any degradation of asymptotic performance, whereas Sarsa can only perform well in the long run at a small value of $\alpha$.
* If $\pi$ is the greedy policy while behavior is more exploratory, then Expected Sarsa is exactly Q-learning. In this sense Expected Sarsa subsumes and generalizes Q-learning while reliably improving over Sarsa.

</div>

### 6.7 Maximization Bias and Double Learning

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Maximization Bias)</span></p>

All the control algorithms discussed so far involve maximization in the construction of their target policies. In these algorithms, a maximum over estimated values is used implicitly as an estimate of the maximum value, which can lead to a significant positive bias. Consider a single state $s$ where there are many actions $a$ whose true values $q(s, a)$ are all zero but whose estimated values $Q(s, a)$ are uncertain and thus distributed some above and some below zero. The maximum of the true values is zero, but the maximum of the estimates is positive — a positive bias. We call this **maximization bias**.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Double Learning)</span></p>

The root cause of maximization bias is that the same samples (plays) are used both to determine the maximizing action *and* to estimate its value. **Double learning** avoids this by dividing the plays into two sets and using them to learn two independent estimates, $Q_1(a)$ and $Q_2(a)$, each an estimate of the true value $q(a)$ for all $a \in \mathcal{A}$.

One estimate, say $Q_1$, is used to determine the maximizing action $A^* = \argmax_a Q_1(a)$, and the other, $Q_2$, is used to provide the estimate of its value, $Q_2(A^*) = Q_2(\argmax_a Q_1(a))$. This estimate will be unbiased in the sense that $\mathbb{E}[Q_2(A^*)] = q(A^*)$. The roles of the two estimates can also be reversed. Although two estimates are learned, only one estimate is updated on each play; double learning doubles the memory requirements but does not increase the amount of computation per step.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Double Q-learning for Estimating $Q_1 \approx Q_2 \approx q_*$)</span></p>

**Algorithm parameters:** step size $\alpha \in (0, 1]$, small $\varepsilon > 0$

Initialize $Q_1(s, a)$ and $Q_2(s, a)$, for all $s \in \mathcal{S}^+$, $a \in \mathcal{A}(s)$, such that $Q(terminal, \cdot) = 0$

Loop for each episode:
* Initialize $S$
* Loop for each step of episode:
  * Choose $A$ from $S$ using the policy $\varepsilon$-greedy in $Q_1 + Q_2$
  * Take action $A$, observe $R$, $S'$
  * With 0.5 probability:
    * $Q_1(S, A) \leftarrow Q_1(S, A) + \alpha\bigl(R + \gamma Q_2(S', \argmax_a Q_1(S', a)) - Q_1(S, A)\bigr)$
  * else:
    * $Q_2(S, A) \leftarrow Q_2(S, A) + \alpha\bigl(R + \gamma Q_1(S', \argmax_a Q_2(S', a)) - Q_2(S, A)\bigr)$
  * $S \leftarrow S'$
* until $S$ is terminal

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Maximization Bias Example)</span></p>

Consider an MDP with two non-terminal states A and B. Episodes always start in A with a choice between **left** and **right**. The **right** action transitions immediately to the terminal state with reward and return of zero. The **left** action transitions to B with reward zero, from which there are many possible actions all causing immediate termination with a reward drawn from $\mathcal{N}(-0.1, 1.0)$. Thus the expected return for any trajectory starting with **left** is $-0.1$, so taking **left** in state A is always a mistake.

Q-learning with $\varepsilon$-greedy action selection initially learns to strongly favor **left** because of maximization bias making B appear to have a positive value. Even at asymptote, Q-learning takes **left** about 5% more often than optimal. Double Q-learning, in contrast, is essentially unaffected by maximization bias.

</div>

### 6.8 Games, Afterstates, and Other Special Cases

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Afterstates and Afterstate Value Functions)</span></p>

In some tasks, such as games, we have knowledge of an initial part of the environment's dynamics but not necessarily of the full dynamics. For example, in games we typically know the immediate effects of our moves but not how our opponent will reply. The board positions *after* the agent has made its move are called **afterstates**, and value functions over these are called **afterstate value functions**.

Afterstates are useful because many different position–move pairs can produce the same resulting position. A conventional action-value function would have to separately assess both pairs, whereas an afterstate value function would immediately assess them equally. Any learning about the position–move pair on the left would immediately transfer to the pair on the right.

Afterstate methods are still aptly described in terms of generalized policy iteration, with a policy and (afterstate) value function interacting in essentially the same way.

</div>

### 6.9 Summary

| Aspect | Monte Carlo | TD(0) | Sarsa | Q-learning | Expected Sarsa |
| --- | --- | --- | --- | --- | --- |
| Model required? | No | No | No | No | No |
| Bootstraps? | No | Yes | Yes | Yes | Yes |
| On/Off-policy | Both | On-policy | On-policy | Off-policy | Both |
| Update target | $G_t$ | $R_{t+1} + \gamma V(S_{t+1})$ | $R_{t+1} + \gamma Q(S_{t+1}, A_{t+1})$ | $R_{t+1} + \gamma \max_a Q(S_{t+1}, a)$ | $R_{t+1} + \gamma \sum_a \pi(a \mid S_{t+1})Q(S_{t+1}, a)$ |
| Learns | $v_\pi$ or $q_\pi$ | $v_\pi$ | $q_\pi$ | $q_*$ | $q_\pi$ or $q_*$ |

Key takeaways from Chapter 6:

* TD methods combine the sampling of Monte Carlo with the bootstrapping of DP.
* All TD control methods use some form of GPI, differing mainly in their prediction (evaluation) component.
* **Sarsa** is on-policy: it learns the value of the policy it is following, including exploration. This leads to safer but potentially suboptimal learned policies.
* **Q-learning** is off-policy: it learns the optimal action-value function directly, but online performance may suffer due to exploratory actions.
* **Expected Sarsa** generalizes both Sarsa and Q-learning, eliminating variance from the random next-action selection while retaining the advantages of on-policy learning.
* **Maximization bias** is a general problem in control algorithms that use maximization; **double learning** is an effective solution.
* The special cases of TD methods introduced in this chapter are *one-step, tabular, model-free* TD methods.

## Chapter 7: $n$-step Bootstrapping

In this chapter we unify the Monte Carlo (MC) methods and the one-step temporal-difference (TD) methods. Neither MC methods nor one-step TD methods are always the best. $n$-step TD methods generalize both methods so that one can shift from one to the other smoothly as needed to meet the demands of a particular task. $n$-step methods span a spectrum with MC methods at one end and one-step TD methods at the other. The best methods are often intermediate between the two extremes.

A key benefit of $n$-step methods is that they free you from the tyranny of the time step. With one-step TD methods the same time step determines how often the action can be changed and the time interval over which bootstrapping is done. $n$-step methods enable bootstrapping to occur over multiple steps, freeing us from the tyranny of the single time step.

### 7.1 $n$-step TD Prediction

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($n$-step Return)</span></p>

Consider the state–reward sequence $S_t, R_{t+1}, S_{t+1}, R_{t+2}, \dots, R_T, S_T$ (omitting the actions). In Monte Carlo updates the target is the complete return:

$$G_t \doteq R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots + \gamma^{T-t-1} R_T.$$

In one-step updates the target is the **one-step return**:

$$G_{t:t+1} \doteq R_{t+1} + \gamma V_t(S_{t+1}),$$

where $V_t : \mathcal{S} \to \mathbb{R}$ is the estimate at time $t$ of $v_\pi$. The target for a two-step update is the **two-step return**:

$$G_{t:t+2} \doteq R_{t+1} + \gamma R_{t+2} + \gamma^2 V_{t+1}(S_{t+2}).$$

Similarly, the target for an arbitrary $n$-step update is the **$n$-step return**:

$$G_{t:t+n} \doteq R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n V_{t+n-1}(S_{t+n}),$$

for all $n, t$ such that $n \geq 1$ and $0 \leq t < T - n$. All $n$-step returns can be considered approximations to the full return, truncated after $n$ steps and then corrected for the remaining missing terms by $V_{t+n-1}(S_{t+n})$. If $t + n \geq T$ (the $n$-step return extends to or beyond termination), then all the missing terms are taken as zero, and the $n$-step return defined to be equal to the ordinary full return ($G_{t:t+n} \doteq G_t$ if $t + n \geq T$).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">($n$-step Returns Are Still TD Methods)</span></p>

The methods that use $n$-step updates are still TD methods because they still change an earlier estimate based on how it differs from a later estimate. Now the later estimate is not one step later, but $n$ steps later. Methods in which the temporal difference extends over $n$ steps are called **$n$-step TD methods**. The TD methods introduced in Chapter 6 all used one-step updates, which is why we called them one-step TD methods.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($n$-step TD Update)</span></p>

The natural state-value learning algorithm for using $n$-step returns is:

$$V_{t+n}(S_t) \doteq V_{t+n-1}(S_t) + \alpha\bigl[G_{t:t+n} - V_{t+n-1}(S_t)\bigr], \qquad 0 \leq t < T,$$

while the values of all other states remain unchanged: $V_{t+n}(s) = V_{t+n-1}(s)$, for all $s \neq S_t$. We call this algorithm **$n$-step TD**. Note that no changes at all are made during the first $n - 1$ steps of each episode. To make up for that, an equal number of additional updates are made at the end of the episode, after termination and before starting the next episode.

Note that $n$-step returns for $n > 1$ involve future rewards and states that are not available at the time of transition from $t$ to $t + 1$. No real algorithm can use the $n$-step return until after it has seen $R_{t+n}$ and computed $V_{t+n-1}$. The first time these are available is $t + n$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">($n$-step TD for Estimating $V \approx v_\pi$)</span></p>

**Input:** a policy $\pi$

**Algorithm parameters:** step size $\alpha \in (0, 1]$, a positive integer $n$

Initialize $V(s)$ arbitrarily, for all $s \in \mathcal{S}$

All store and access operations (for $S_t$ and $R_t$) can take their index mod $n + 1$

Loop for each episode:
* Initialize and store $S_0 \neq terminal$
* $T \leftarrow \infty$
* Loop for $t = 0, 1, 2, \dots$:
  * If $t < T$, then:
    * Take an action according to $\pi(\cdot \mid S_t)$
    * Observe and store the next reward as $R_{t+1}$ and the next state as $S_{t+1}$
    * If $S_{t+1}$ is terminal, then $T \leftarrow t + 1$
  * $\tau \leftarrow t - n + 1$ &ensp; ($\tau$ is the time whose state's estimate is being updated)
  * If $\tau \geq 0$:
    * $G \leftarrow \sum_{i=\tau+1}^{\min(\tau+n, T)} \gamma^{i-\tau-1} R_i$
    * If $\tau + n < T$, then: $G \leftarrow G + \gamma^n V(S_{\tau+n})$ &ensp; ($G_{\tau:\tau+n}$)
    * $V(S_\tau) \leftarrow V(S_\tau) + \alpha\bigl[G - V(S_\tau)\bigr]$
* Until $\tau = T - 1$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Error Reduction Property of $n$-step Returns)</span></p>

An important property of $n$-step returns is that their expectation is guaranteed to be a better estimate of $v_\pi$ than $V_{t+n-1}$ is, in a worst-state sense. That is, the worst error of the expected $n$-step return is guaranteed to be less than or equal to $\gamma^n$ times the worst error under $V_{t+n-1}$:

$$\max_s \bigl\lvert \mathbb{E}_\pi[G_{t:t+n} \mid S_t = s] - v_\pi(s) \bigr\rvert \;\leq\; \gamma^n \max_s \bigl\lvert V_{t+n-1}(s) - v_\pi(s) \bigr\rvert,$$

for all $n \geq 1$. This is called the **error reduction property** of $n$-step returns. Because of the error reduction property, one can show formally that all $n$-step TD methods converge to the correct predictions under appropriate technical conditions.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">($n$-step TD Methods on the Random Walk)</span></p>

Consider using $n$-step TD methods on a 19-state random walk process (with a $-1$ outcome on the left, all values initialized to 0). If the first episode progressed directly from the center state C to the right, through D and E, and terminated with a return of 1:

* A **one-step** method would change only the estimate for the last state, $V(\text{E})$, incrementing it toward 1.
* A **two-step** method would increment $V(\text{D})$ and $V(\text{E})$ both toward 1.
* An **$n$-step** method for $n > 2$ would increment the values of all three visited states toward 1, all by the same amount.

Empirical results show that methods with an **intermediate value of $n$** worked best. This illustrates how the generalization of TD and Monte Carlo methods to $n$-step methods can potentially perform better than either of the two extreme methods.

</div>

### 7.2 $n$-step Sarsa

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($n$-step Sarsa)</span></p>

The $n$-step version of Sarsa (with the original version henceforth called **one-step Sarsa** or **Sarsa(0)**) simply switches states for actions (state–action pairs) and then uses an $\varepsilon$-greedy policy. We redefine $n$-step returns (update targets) in terms of estimated action values:

$$G_{t:t+n} \doteq R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n Q_{t+n-1}(S_{t+n}, A_{t+n}), \quad n \geq 1, \; 0 \leq t < T - n,$$

with $G_{t:t+n} \doteq G_t$ if $t + n \geq T$. The natural algorithm is then:

$$Q_{t+n}(S_t, A_t) \doteq Q_{t+n-1}(S_t, A_t) + \alpha\bigl[G_{t:t+n} - Q_{t+n-1}(S_t, A_t)\bigr], \qquad 0 \leq t < T,$$

while the values of all other state–action pairs remain unchanged.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">($n$-step Sarsa for Estimating $Q \approx q_*$ or $q_\pi$)</span></p>

Initialize $Q(s, a)$ arbitrarily, for all $s \in \mathcal{S}$, $a \in \mathcal{A}$

Initialize $\pi$ to be $\varepsilon$-greedy with respect to $Q$, or to a fixed given policy

**Algorithm parameters:** step size $\alpha \in (0, 1]$, small $\varepsilon > 0$, a positive integer $n$

All store and access operations (for $S_t$, $A_t$, and $R_t$) can take their index mod $n + 1$

Loop for each episode:
* Initialize and store $S_0 \neq terminal$
* Select and store an action $A_0 \sim \pi(\cdot \mid S_0)$
* $T \leftarrow \infty$
* Loop for $t = 0, 1, 2, \dots$:
  * If $t < T$, then:
    * Take action $A_t$
    * Observe and store the next reward as $R_{t+1}$ and the next state as $S_{t+1}$
    * If $S_{t+1}$ is terminal, then: $T \leftarrow t + 1$
    * else: Select and store an action $A_{t+1} \sim \pi(\cdot \mid S_{t+1})$
  * $\tau \leftarrow t - n + 1$ &ensp; ($\tau$ is the time whose estimate is being updated)
  * If $\tau \geq 0$:
    * $G \leftarrow \sum_{i=\tau+1}^{\min(\tau+n, T)} \gamma^{i-\tau-1} R_i$
    * If $\tau + n < T$, then: $G \leftarrow G + \gamma^n Q(S_{\tau+n}, A_{\tau+n})$ &ensp; ($G_{\tau:\tau+n}$)
    * $Q(S_\tau, A_\tau) \leftarrow Q(S_\tau, A_\tau) + \alpha\bigl[G - Q(S_\tau, A_\tau)\bigr]$
    * If $\pi$ is being learned, then ensure that $\pi(\cdot \mid S_\tau)$ is $\varepsilon$-greedy wrt $Q$
* Until $\tau = T - 1$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Speedup from $n$-step Methods)</span></p>

In a gridworld example, the one-step method strengthens only the last action of the sequence of actions that led to the high reward, whereas the $n$-step method strengthens the last $n$ actions of the sequence, so that much more is learned from the one episode. This illustrates how $n$-step methods can speed up policy learning.

</div>

#### $n$-step Expected Sarsa

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($n$-step Expected Sarsa)</span></p>

The backup diagram for $n$-step Expected Sarsa consists of a linear string of sample actions and states, just as in $n$-step Sarsa, except that its last element is a branch over all action possibilities weighted by their probability under $\pi$. The $n$-step return is redefined as:

$$G_{t:t+n} \doteq R_{t+1} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n \bar{V}_{t+n-1}(S_{t+n}), \qquad t + n < T,$$

(with $G_{t:t+n} \doteq G_t$ for $t + n \geq T$) where $\bar{V}_t(s)$ is the **expected approximate value** of state $s$, using the estimated action values at time $t$, under the target policy:

$$\bar{V}_t(s) \doteq \sum_a \pi(a \mid s) Q_t(s, a), \qquad \text{for all } s \in \mathcal{S}.$$

If $s$ is terminal, then its expected approximate value is defined to be 0. Expected approximate values are used in developing many of the action-value methods in the rest of the book.

</div>

### 7.3 $n$-step Off-policy Learning

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($n$-step Off-policy TD with Importance Sampling)</span></p>

Recall that off-policy learning is learning the value function for one policy, $\pi$, while following another policy, $b$. In $n$-step methods, returns are constructed over $n$ steps, so we are interested in the relative probability of just those $n$ actions. The update for time $t$ (actually made at time $t + n$) can simply be weighted by the **importance sampling ratio** $\rho_{t:t+n-1}$:

$$V_{t+n}(S_t) \doteq V_{t+n-1}(S_t) + \alpha \rho_{t:t+n-1}\bigl[G_{t:t+n} - V_{t+n-1}(S_t)\bigr], \qquad 0 \leq t < T,$$

where the importance sampling ratio is the relative probability under the two policies of taking the $n$ actions from $A_t$ to $A_{t+n-1}$:

$$\rho_{t:h} \doteq \prod_{k=t}^{\min(h, T-1)} \frac{\pi(A_k \mid S_k)}{b(A_k \mid S_k)}.$$

If any one of the actions would never be taken by $\pi$ (i.e., $\pi(A_k \mid S_k) = 0$) then the $n$-step return should be given zero weight and be totally ignored. On the other hand, if an action is taken that $\pi$ would take with much greater probability than $b$, this will increase the weight given to the return. If the two policies are actually the same (the on-policy case) then the importance sampling ratio is always 1.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($n$-step Off-policy Sarsa)</span></p>

The $n$-step Sarsa update can be completely replaced by a simple off-policy form:

$$Q_{t+n}(S_t, A_t) \doteq Q_{t+n-1}(S_t, A_t) + \alpha \rho_{t+1:t+n}\bigl[G_{t:t+n} - Q_{t+n-1}(S_t, A_t)\bigr],$$

for $0 \leq t < T$. Note that the importance sampling ratio here starts and ends one step later than for $n$-step TD. This is because we are updating a state–action pair. We do not have to care how likely we were to select the action; now that we have selected it we want to learn fully from what happens, with importance sampling only for subsequent actions.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Off-policy $n$-step Sarsa for Estimating $Q \approx q_*$ or $q_\pi$)</span></p>

**Input:** an arbitrary behavior policy $b$ such that $b(a \mid s) > 0$, for all $s \in \mathcal{S}$, $a \in \mathcal{A}$

Initialize $Q(s, a)$ arbitrarily, for all $s \in \mathcal{S}$, $a \in \mathcal{A}$

Initialize $\pi$ to be greedy with respect to $Q$, or as a fixed given policy

**Algorithm parameters:** step size $\alpha \in (0, 1]$, a positive integer $n$

All store and access operations (for $S_t$, $A_t$, and $R_t$) can take their index mod $n + 1$

Loop for each episode:
* Initialize and store $S_0 \neq terminal$
* Select and store an action $A_0 \sim b(\cdot \mid S_0)$
* $T \leftarrow \infty$
* Loop for $t = 0, 1, 2, \dots$:
  * If $t < T$, then:
    * Take action $A_t$
    * Observe and store the next reward as $R_{t+1}$ and the next state as $S_{t+1}$
    * If $S_{t+1}$ is terminal, then: $T \leftarrow t + 1$
    * else: Select and store an action $A_{t+1} \sim b(\cdot \mid S_{t+1})$
  * $\tau \leftarrow t - n + 1$ &ensp; ($\tau$ is the time whose estimate is being updated)
  * If $\tau \geq 0$:
    * $\rho \leftarrow \prod_{k=\tau+1}^{\min(\tau+n, T-1)} \frac{\pi(A_k \mid S_k)}{b(A_k \mid S_k)}$ &ensp; ($\rho_{\tau+1:\tau+n}$)
    * $G \leftarrow \sum_{i=\tau+1}^{\min(\tau+n, T)} \gamma^{i-\tau-1} R_i$
    * If $\tau + n < T$, then: $G \leftarrow G + \gamma^n Q(S_{\tau+n}, A_{\tau+n})$ &ensp; ($G_{\tau:\tau+n}$)
    * $Q(S_\tau, A_\tau) \leftarrow Q(S_\tau, A_\tau) + \alpha \rho\bigl[G - Q(S_\tau, A_\tau)\bigr]$
    * If $\pi$ is being learned, then ensure that $\pi(\cdot \mid S_\tau)$ is greedy wrt $Q$
* Until $\tau = T - 1$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">($n$-step Off-policy Expected Sarsa)</span></p>

The off-policy version of $n$-step Expected Sarsa would use the same update as above for $n$-step Sarsa except that the importance sampling ratio would have one less factor: $\rho_{t+1:t+n-1}$ instead of $\rho_{t+1:t+n}$, and it would use the Expected Sarsa version of the $n$-step return. This is because in Expected Sarsa all possible actions are taken into account in the last state; the one actually taken has no effect and does not have to be corrected for.

</div>

### 7.4 Per-decision Methods with Control Variates

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Off-policy $n$-step Return with Control Variates)</span></p>

The multi-step off-policy methods with plain importance sampling are simple and conceptually clear, but are probably not the most efficient. A more sophisticated approach uses per-decision importance sampling. The ordinary $n$-step return, like all returns, can be written recursively. For the $n$ steps ending at horizon $h$:

$$G_{t:h} = R_{t+1} + \gamma G_{t+1:h}, \qquad t < h < T.$$

Now consider following a behavior policy $b$ that is not the same as the target policy $\pi$. The importance sampling ratio for time $t$ is $\rho_t = \frac{\pi(A_t \mid S_t)}{b(A_t \mid S_t)}$. An alternate, off-policy definition of the $n$-step return ending at horizon $h$ is:

$$G_{t:h} \doteq \rho_t\bigl(R_{t+1} + \gamma G_{t+1:h}\bigr) + (1 - \rho_t) V_{h-1}(S_t), \qquad t < h < T.$$

In this approach, if $\rho_t$ is zero, then instead of the target being zero and causing the estimate to shrink, the target is the same as the estimate and causes no change. The second, additional term $(1 - \rho_t) V_{h-1}(S_t)$ is called a **control variate**. The control variate does not change the expected update; the importance sampling ratio has expected value one and is uncorrelated with the estimate, so the expected value of the control variate is zero.

Note that this off-policy definition is a strict generalization of the earlier on-policy definition of the $n$-step return, as the two are identical in the on-policy case (where $\rho_t$ is always 1).

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Off-policy $n$-step Return for Action Values with Control Variates)</span></p>

For action values, the off-policy definition of the $n$-step return is a little different because the first action does not play a role in the importance sampling (it has been taken and now full unit weight must be given to the reward and state that follows it). An off-policy form with control variates is:

$$G_{t:h} \doteq R_{t+1} + \gamma \rho_{t+1}\bigl(G_{t+1:h} - Q_{h-1}(S_{t+1}, A_{t+1})\bigr) + \gamma \bar{V}_{h-1}(S_{t+1}), \qquad t < h \leq T.$$

If $h < T$, then the recursion ends with $G_{h:h} \doteq Q_{h-1}(S_h, A_h)$, whereas, if $h \geq T$, the recursion ends with $G_{T-1:h} \doteq R_T$. The resultant prediction algorithm (after combining with the $n$-step Sarsa update) is analogous to Expected Sarsa.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Variance in Off-policy Learning)</span></p>

The importance sampling used in $n$-step off-policy methods enables sound off-policy learning, but also results in high variance updates, forcing the use of a small step-size parameter and thereby causing learning to be slow. It is probably inevitable that off-policy training is slower than on-policy training — after all, the data is less relevant to what is being learned. However, control variates are one way of reducing the variance. Another is to rapidly adapt the step sizes to the observed variance, as in the Autostep method.

</div>

### 7.5 Off-policy Learning Without Importance Sampling: The $n$-step Tree Backup Algorithm

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Tree Backup Algorithm)</span></p>

Is off-policy learning possible without importance sampling? Q-learning and Expected Sarsa do this for the one-step case, but is there a corresponding multi-step algorithm? The **tree-backup algorithm** is just such an $n$-step method.

The idea is suggested by a tree-shaped backup diagram. Down the central spine are sample states and rewards, and sample actions. Hanging off to the sides of each state are the actions that were *not* selected. Because we have no sample data for the unselected actions, we bootstrap and use the estimates of their values in forming the target for the update.

Each leaf node (non-selected action) contributes to the target with a weight proportional to its probability of occurring under the target policy $\pi$. Each first-level action $a$ contributes with a weight of $\pi(a \mid S_{t+1})$, except that the action actually taken, $A_{t+1}$, does not contribute at all. Its probability, $\pi(A_{t+1} \mid S_{t+1})$, is used to weight all the second-level action values. It is as if each arrow to an action node in the diagram is weighted by the action's probability of being selected under the target policy.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Tree Backup $n$-step Return)</span></p>

The one-step return (target) is the same as that of Expected Sarsa:

$$G_{t:t+1} \doteq R_{t+1} + \gamma \sum_a \pi(a \mid S_{t+1}) Q_t(S_{t+1}, a),$$

for $t < T - 1$. The general recursive definition of the tree-backup $n$-step return is:

$$G_{t:t+n} \doteq R_{t+1} + \gamma \sum_{a \neq A_{t+1}} \pi(a \mid S_{t+1}) Q_{t+n-1}(S_{t+1}, a) \;+\; \gamma \pi(A_{t+1} \mid S_{t+1}) G_{t+1:t+n},$$

for $t < T - 1$, $n \geq 2$, with the $n = 1$ case handled by the one-step return above and $G_{T-1:t+n} \doteq R_T$. This target is then used with the usual action-value update rule from $n$-step Sarsa:

$$Q_{t+n}(S_t, A_t) \leftarrow Q_{t+n-1}(S_t, A_t) + \alpha\bigl[G_{t:t+n} - Q_{t+n-1}(S_t, A_t)\bigr].$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">($n$-step Tree Backup for Estimating $Q \approx q_*$ or $q_\pi$)</span></p>

Initialize $Q(s, a)$ arbitrarily, for all $s \in \mathcal{S}$, $a \in \mathcal{A}$

Initialize $\pi$ to be greedy with respect to $Q$, or as a fixed given policy

**Algorithm parameters:** step size $\alpha \in (0, 1]$, a positive integer $n$

All store and access operations can take their index mod $n + 1$

Loop for each episode:
* Initialize and store $S_0 \neq terminal$
* Choose an action $A_0$ arbitrarily as a function of $S_0$; Store $A_0$
* $T \leftarrow \infty$
* Loop for $t = 0, 1, 2, \dots$:
  * If $t < T$:
    * Take action $A_t$; observe and store the next reward and state as $R_{t+1}$, $S_{t+1}$
    * If $S_{t+1}$ is terminal: $T \leftarrow t + 1$
    * else: Choose an action $A_{t+1}$ arbitrarily as a function of $S_{t+1}$; Store $A_{t+1}$
  * $\tau \leftarrow t + 1 - n$ &ensp; ($\tau$ is the time whose estimate is being updated)
  * If $\tau \geq 0$:
    * If $t + 1 \geq T$: $G \leftarrow R_T$
    * else: $G \leftarrow R_{t+1} + \gamma \sum_a \pi(a \mid S_{t+1}) Q(S_{t+1}, a)$
    * Loop for $k = \min(t, T - 1)$ down through $\tau + 1$:
      * $G \leftarrow R_k + \gamma \sum_{a \neq A_k} \pi(a \mid S_k) Q(S_k, a) + \gamma \pi(A_k \mid S_k) G$
    * $Q(S_\tau, A_\tau) \leftarrow Q(S_\tau, A_\tau) + \alpha\bigl[G - Q(S_\tau, A_\tau)\bigr]$
    * If $\pi$ is being learned, then ensure that $\pi(\cdot \mid S_\tau)$ is greedy wrt $Q$
* Until $\tau = T - 1$

</div>

### 7.6 A Unifying Algorithm: $n$-step $Q(\sigma)$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($n$-step $Q(\sigma)$)</span></p>

So far in this chapter we have considered three different kinds of action-value algorithms: $n$-step Sarsa has all sample transitions, the tree-backup algorithm has all state-to-action transitions fully branched without sampling, and $n$-step Expected Sarsa has all sample transitions except for the last state-to-action one, which is fully branched with an expected value.

The unification idea is that one might decide on a step-by-step basis whether one wanted to take the action as a sample (as in Sarsa), or consider the expectation over all actions instead (as in the tree-backup update). Let $\sigma_t \in [0, 1]$ denote the degree of sampling on step $t$, with $\sigma = 1$ denoting full sampling and $\sigma = 0$ denoting a pure expectation with no sampling. The random variable $\sigma_t$ might be set as a function of the state, action, or state–action pair at time $t$. We call this proposed new algorithm **$n$-step $Q(\sigma)$**.

The $n$-step return for $Q(\sigma)$ slides linearly between the two cases:

$$G_{t:h} \doteq R_{t+1} + \gamma\Bigl(\sigma_{t+1} \rho_{t+1} + (1 - \sigma_{t+1})\pi(A_{t+1} \mid S_{t+1})\Bigr)\Bigl(G_{t+1:h} - Q_{h-1}(S_{t+1}, A_{t+1})\Bigr) + \gamma \bar{V}_{h-1}(S_{t+1}),$$

for $t < h \leq T$. The recursion ends with $G_{h:h} \doteq Q_{h-1}(S_h, A_h)$ if $h < T$, or with $G_{T-1:T} \doteq R_T$ if $h = T$.

Special cases:
* If one always chooses to sample ($\sigma_t = 1$ for all $t$), one obtains $n$-step Sarsa (with importance sampling ratios).
* If one never samples ($\sigma_t = 0$ for all $t$), one gets the tree-backup algorithm.
* Expected Sarsa would be the case where one chooses to sample for all steps except for the last one.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Off-policy $n$-step $Q(\sigma)$ for Estimating $Q \approx q_*$ or $q_\pi$)</span></p>

**Input:** an arbitrary behavior policy $b$ such that $b(a \mid s) > 0$, for all $s \in \mathcal{S}$, $a \in \mathcal{A}$

Initialize $Q(s, a)$ arbitrarily, for all $s \in \mathcal{S}$, $a \in \mathcal{A}$

Initialize $\pi$ to be greedy with respect to $Q$, or else it is a fixed given policy

**Algorithm parameters:** step size $\alpha \in (0, 1]$, a positive integer $n$

All store and access operations can take their index mod $n + 1$

Loop for each episode:
* Initialize and store $S_0 \neq terminal$
* Choose and store an action $A_0 \sim b(\cdot \mid S_0)$
* $T \leftarrow \infty$
* Loop for $t = 0, 1, 2, \dots$:
  * If $t < T$:
    * Take action $A_t$; observe and store the next reward and state as $R_{t+1}$, $S_{t+1}$
    * If $S_{t+1}$ is terminal: $T \leftarrow t + 1$
    * else:
      * Choose and store an action $A_{t+1} \sim b(\cdot \mid S_{t+1})$
      * Select and store $\sigma_{t+1}$
      * Store $\frac{\pi(A_{t+1} \mid S_{t+1})}{b(A_{t+1} \mid S_{t+1})}$ as $\rho_{t+1}$
  * $\tau \leftarrow t - n + 1$ &ensp; ($\tau$ is the time whose estimate is being updated)
  * If $\tau \geq 0$:
    * If $t + 1 < T$: $G \leftarrow Q(S_{t+1}, A_{t+1})$
    * Loop for $k = \min(t + 1, T)$ down through $\tau + 1$:
      * if $k = T$: $G \leftarrow R_T$
      * else:
        * $\bar{V} \leftarrow \sum_a \pi(a \mid S_k) Q(S_k, a)$
        * $G \leftarrow R_k + \gamma\bigl(\sigma_k \rho_k + (1 - \sigma_k)\pi(A_k \mid S_k)\bigr)\bigl(G - Q(S_k, A_k)\bigr) + \gamma \bar{V}$
    * $Q(S_\tau, A_\tau) \leftarrow Q(S_\tau, A_\tau) + \alpha\bigl[G - Q(S_\tau, A_\tau)\bigr]$
    * If $\pi$ is being learned, then ensure that $\pi(\cdot \mid S_\tau)$ is greedy wrt $Q$
* Until $\tau = T - 1$

</div>

### 7.7 Summary

| Method | Sampling | Importance Sampling Needed? | Key Idea |
| --- | --- | --- | --- |
| $n$-step Sarsa | All transitions sampled | Yes (off-policy) | Extends one-step Sarsa to $n$ steps |
| $n$-step Tree Backup | No sampling (expectations) | No | Weights by $\pi$ at every branching point |
| $n$-step Expected Sarsa | All sampled except last | Yes, with one less factor | Last transition uses expectation |
| $n$-step $Q(\sigma)$ | Per-step choice ($\sigma_t$) | Incorporated in return | Unifies all three above |

Key takeaways from Chapter 7:

* $n$-step methods span a spectrum between one-step TD methods and Monte Carlo methods. Methods that involve an intermediate amount of bootstrapping are important because they will typically perform better than either extreme.
* The state-value update is for $n$-step TD with importance sampling, and the action-value update is for $n$-step $Q(\sigma)$, which generalizes Expected Sarsa and Q-learning.
* All $n$-step methods involve a delay of $n$ time steps before updating, as only then are all the required future events known.
* Compared to one-step methods, $n$-step methods require more memory to record the states, actions, rewards, and sometimes other variables over the last $n$ time steps, and involve more computation per time step.
* **Importance sampling** is conceptually simple but can be of high variance. **Tree-backup** updates involve no importance sampling but, if the target and behavior policies are substantially different, the bootstrapping may span only a few steps even if $n$ is large.
* The **$Q(\sigma)$ algorithm** unifies all the $n$-step action-value methods by choosing on a state-by-state basis whether to sample ($\sigma_t = 1$) or not ($\sigma_t = 0$).

## Chapter 8: Planning and Learning with Tabular Methods

This chapter develops a unified view of reinforcement learning methods that require a model of the environment (*model-based* methods, relying on *planning*) and methods that can be used without a model (*model-free* methods, relying on *learning*). The heart of both kinds of methods is the computation of value functions. All the methods are based on looking ahead to future events, computing a backed-up value, and then using it as an update target for an approximate value function.

### 8.1 Models and Planning

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Model of the Environment)</span></p>

A **model** of the environment is anything that an agent can use to predict how the environment will respond to its actions. Given a state and an action, a model produces a prediction of the resultant next state and next reward.

* **Distribution models** produce a description of all possibilities and their probabilities. The dynamics function $p(s', r \mid s, a)$ is a distribution model.
* **Sample models** produce just one of the possibilities, sampled according to the probabilities.

Distribution models are stronger than sample models in that they can always be used to produce samples. However, in many applications it is much easier to obtain sample models than distribution models.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Planning)</span></p>

We use the term **planning** to refer to any computational process that takes a model as input and produces or improves a policy for interacting with the modeled environment:

$$\text{model} \xrightarrow{\text{planning}} \text{policy}$$

**State-space planning** searches through the state space for an optimal policy or an optimal path to a goal. Actions cause transitions from state to state, and value functions are computed over states. The unified view presented in this chapter is that all state-space planning methods share a common structure:

$$\text{model} \longrightarrow \text{simulated experience} \xrightarrow{\text{backups}} \text{values} \longrightarrow \text{policy}$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Learning and Planning Commonality)</span></p>

The heart of both learning and planning methods is the estimation of value functions by backing-up update operations. The difference is that whereas planning uses simulated experience generated by a model, learning methods use real experience generated by the environment. The common structure means that many ideas and algorithms can be transferred between planning and learning. In many cases a learning algorithm can be substituted for the key update step of a planning method.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Random-Sample One-Step Tabular Q-Planning)</span></p>

Loop forever:
1. Select a state, $S \in \mathcal{S}$, and an action, $A \in \mathcal{A}(S)$, at random
2. Send $S, A$ to a sample model, and obtain a sample next reward, $R$, and a sample next state, $S'$
3. Apply one-step tabular Q-learning to $S, A, R, S'$:

$$Q(S, A) \leftarrow Q(S, A) + \alpha\bigl[R + \gamma \max_a Q(S', a) - Q(S, A)\bigr]$$

This method converges to the optimal policy for the model under the same conditions that one-step tabular Q-learning converges to the optimal policy for the real environment (each state–action pair must be selected an infinite number of times in Step 1, and $\alpha$ must decrease appropriately over time).

</div>

### 8.2 Dyna: Integrated Planning, Acting, and Learning

Within a planning agent, there are at least two roles for real experience: it can be used to improve the model (to make it more accurately match the real environment) and it can be used to directly improve the value function and policy using the kinds of reinforcement learning methods we have discussed.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Model Learning and Direct RL)</span></p>

* **Model learning:** Using real experience to improve the model of the environment. This is sometimes called **indirect reinforcement learning**.
* **Direct reinforcement learning (direct RL):** Using real experience to directly improve the value function and policy.

Indirect methods often make fuller use of a limited amount of experience and thus achieve a better policy with fewer environmental interactions. Direct methods are much simpler and are not affected by biases in the design of the model.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Dyna Architecture)</span></p>

**Dyna-Q** is a simple architecture integrating the major functions needed in an online planning agent. It includes:

* **Acting:** Interacting with the real environment
* **Model learning:** Updating the model from real experience
* **Direct RL:** Updating values/policy directly from real experience
* **Planning:** Updating values/policy from simulated experience generated by the model

Conceptually, planning, acting, model-learning, and direct RL occur simultaneously and in parallel. The reinforcement learning method is the "final common path" for both learning and planning, differing only in the source of their experience. We use the term **search control** to refer to the process that selects the starting states and actions for the simulated experiences generated by the model.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Tabular Dyna-Q)</span></p>

Initialize $Q(s, a)$ and $Model(s, a)$ for all $s \in \mathcal{S}$ and $a \in \mathcal{A}(s)$

Loop forever:
* (a) $S \leftarrow$ current (nonterminal) state
* (b) $A \leftarrow \varepsilon\text{-greedy}(S, Q)$
* (c) Take action $A$; observe resultant reward, $R$, and state, $S'$
* (d) $Q(S, A) \leftarrow Q(S, A) + \alpha\bigl[R + \gamma \max_a Q(S', a) - Q(S, A)\bigr]$
* (e) $Model(S, A) \leftarrow R, S'$ &ensp; (assuming deterministic environment)
* (f) Loop repeat $n$ times:
  * $S \leftarrow$ random previously observed state
  * $A \leftarrow$ random action previously taken in $S$
  * $R, S' \leftarrow Model(S, A)$
  * $Q(S, A) \leftarrow Q(S, A) + \alpha\bigl[R + \gamma \max_a Q(S', a) - Q(S, A)\bigr]$

Steps (d), (e), and (f) implement direct reinforcement learning, model-learning, and planning, respectively. If steps (e) and (f) were omitted, the remaining algorithm would be one-step tabular Q-learning.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(8.1: Dyna Maze)</span></p>

Consider a simple maze with 47 states and four actions (up, down, right, left). Movement is deterministic except when blocked by an obstacle or the maze edge, in which case the agent remains in place. Reward is zero on all transitions except those into the goal state, where it is $+1$. After reaching the goal state (G), the agent returns to the start state (S). This is a discounted, episodic task with $\gamma = 0.95$.

Dyna-Q agents with $\alpha = 0.1$, $\varepsilon = 0.1$, and varying numbers of planning steps $n$ were tested:

* The $n = 0$ agent (no planning, direct RL only) was the slowest, taking about 25 episodes to reach $\varepsilon$-optimal performance.
* The $n = 5$ agent took about five episodes.
* The $n = 50$ agent took only three episodes.

The planning agents found the solution much faster because the planning process builds an extensive policy during each episode while the agent is still wandering.

</div>

### 8.3 When the Model Is Wrong

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Incorrect Models)</span></p>

Models may be incorrect because the environment is stochastic and only a limited number of samples have been observed, or because the model was learned using function approximation that has generalized imperfectly, or simply because the environment has changed and its new behavior has not yet been observed. When the model is incorrect, the planning process is likely to compute a suboptimal policy.

In some cases, the suboptimal policy computed by planning quickly leads to the discovery and correction of the modeling error — this tends to happen when the model is optimistic. Greater difficulties arise when the environment changes to become *better* than before, and yet the formerly correct policy does not reveal the improvement. These cases may not be detected for a long time, if ever.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(8.2: Blocking Maze)</span></p>

Initially, there is a short path from start to goal, to the right of a barrier. After 1000 time steps, the short path is "blocked," and a longer path is opened up along the left-hand side. The graph shows that both Dyna-Q and an enhanced Dyna-Q+ agent found the short path within 1000 steps. When the environment changed, the graphs become flat (no reward), indicating a period of wandering behind the barrier. After a while, however, both agents were able to find the new opening and the new optimal behavior.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(8.3: Shortcut Maze)</span></p>

Initially, the optimal path is to go around the left side of a barrier. After 3000 steps, a shorter path is opened up along the right side. The regular Dyna-Q agent never switched to the shortcut — it never realized it existed. Its model said there was no shortcut, so the more it planned, the less likely it was to step to the right and discover it. Even with an $\varepsilon$-greedy policy, it is very unlikely that an agent will take so many exploratory actions as to discover the shortcut.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Dyna-Q+ Exploration Bonus)</span></p>

The **Dyna-Q+** agent keeps track for each state–action pair of how many time steps have elapsed since the pair was last tried in a real interaction with the environment. The more time that has elapsed, the greater the chance that the dynamics of this pair has changed and that the model of it is incorrect. To encourage behavior that tests long-untried actions, a special "bonus reward" is given on simulated experiences involving these actions.

If the modeled reward for a transition is $r$, and the transition has not been tried in $\tau$ time steps, then planning updates are done as if that transition produced a reward of $r + \kappa\sqrt{\tau}$, for some small $\kappa$. This encourages the agent to keep testing all accessible state transitions and even to find long sequences of actions in order to carry out such tests.

</div>

### 8.4 Prioritized Sweeping

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Backward Focusing)</span></p>

In the Dyna agents, simulated transitions are started in state–action pairs selected uniformly at random from all previously experienced pairs. A uniform selection is usually not the best; planning can be much more efficient if simulated transitions and updates are focused on particular state–action pairs.

The idea of **backward focusing** is to work backward from arbitrary states that have changed in value. When the agent discovers a change in the environment and changes the estimated value of one state, the only useful one-step updates are those of actions that lead directly into that state. If the values of these actions are updated, then the values of predecessor states may change in turn. One can work backward, either performing useful updates or terminating the propagation.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Prioritized Sweeping)</span></p>

**Prioritized sweeping** maintains a queue of every state–action pair whose estimated value would change nontrivially if updated, prioritized by the size of the change. When the top pair in the queue is updated, the effect on each of its predecessor pairs is computed. If the effect is greater than some small threshold $\theta$, then the pair is inserted in the queue with the new priority. In this way the effects of changes are efficiently propagated backward until quiescence.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Prioritized Sweeping for a Deterministic Environment)</span></p>

Initialize $Q(s, a)$, $Model(s, a)$, for all $s, a$, and $PQueue$ to empty

Loop forever:
* (a) $S \leftarrow$ current (nonterminal) state
* (b) $A \leftarrow \text{policy}(S, Q)$
* (c) Take action $A$; observe reward, $R$, and state, $S'$
* (d) $Model(S, A) \leftarrow R, S'$
* (e) $P \leftarrow \lvert R + \gamma \max_a Q(S', a) - Q(S, A) \rvert$
* (f) if $P > \theta$, then insert $S, A$ into $PQueue$ with priority $P$
* (g) Loop repeat $n$ times, while $PQueue$ is not empty:
  * $S, A \leftarrow first(PQueue)$
  * $R, S' \leftarrow Model(S, A)$
  * $Q(S, A) \leftarrow Q(S, A) + \alpha\bigl[R + \gamma \max_a Q(S', a) - Q(S, A)\bigr]$
  * Loop for all $\bar{S}, \bar{A}$ predicted to lead to $S$:
    * $\bar{R} \leftarrow$ predicted reward for $\bar{S}, \bar{A}, S$
    * $P \leftarrow \lvert \bar{R} + \gamma \max_a Q(S, a) - Q(\bar{S}, \bar{A}) \rvert$
    * if $P > \theta$ then insert $\bar{S}, \bar{A}$ into $PQueue$ with priority $P$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(8.4: Prioritized Sweeping on Mazes)</span></p>

Prioritized sweeping has been found to dramatically increase the speed at which optimal solutions are found in maze tasks, often by a factor of 5 to 10. For a sequence of maze tasks with the same structure but varying grid resolution, prioritized sweeping maintained a decisive advantage over unprioritized Dyna-Q. Both systems made at most $n = 5$ updates per environmental interaction.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Extensions of Prioritized Sweeping)</span></p>

Extensions to stochastic environments are straightforward. The model is maintained by keeping counts of the number of times each state–action pair has been experienced and of what the next states were. It is natural then to update each pair not with a sample update, but with an expected update, taking into account all possible next states and their probabilities of occurring.

One limitation of prioritized sweeping is that it uses *expected* updates, which in stochastic environments may waste lots of computation on low-probability transitions.

</div>

### 8.5 Expected vs. Sample Updates

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Three Dimensions of Value-Function Updates)</span></p>

Focusing on one-step updates, they vary primarily along three binary dimensions:

1. Whether they update state values or action values
2. Whether they estimate the value for the optimal policy or for an arbitrary given policy
3. Whether the updates are *expected* updates (considering all possible events) or *sample* updates (considering a single sample)

These three dimensions give rise to eight cases, seven of which correspond to specific algorithms (the eighth — $v_\pi$ expected update that is not policy evaluation — does not correspond to a useful update). The seven algorithms are: policy evaluation, TD(0), value iteration, q-policy evaluation, Sarsa, q-value iteration, and Q-learning.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Expected vs. Sample Updates for $q_*$)</span></p>

The **expected update** for a state–action pair, $s, a$, using estimated dynamics $\hat{p}(s', r \mid s, a)$, is:

$$Q(s, a) \leftarrow \sum_{s', r} \hat{p}(s', r \mid s, a)\bigl[r + \gamma \max_{a'} Q(s', a')\bigr].$$

The corresponding **sample update** for $s, a$, given a sample next state and reward, $S'$ and $R$ (from the model), is:

$$Q(s, a) \leftarrow Q(s, a) + \alpha\bigl[R + \gamma \max_{a'} Q(S', a') - Q(s, a)\bigr].$$

The difference between these is significant to the extent that the environment is stochastic. If only one next state is possible, the expected and sample updates are identical (taking $\alpha = 1$).

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Branching Factor)</span></p>

For a particular starting pair, $s, a$, let $b$ be the **branching factor**, i.e., the number of possible next states $s'$ for which $\hat{p}(s' \mid s, a) > 0$. An expected update of this pair requires roughly $b$ times as much computation as a sample update.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Expected vs. Sample Update Efficiency)</span></p>

* If there is enough time to complete an expected update, then the resulting estimate is generally better than that of $b$ sample updates because of the absence of sampling error.
* If there is insufficient time to complete an expected update, then sample updates are always preferable because they at least make some improvement in the value estimate with fewer than $b$ updates.
* For moderately large $b$, the error from sample updates falls dramatically with a tiny fraction of $b$ updates. Many state–action pairs could have their values improved dramatically, to within a few percent of the effect of an expected update, in the same time that a single state–action pair could undergo an expected update.
* Sample updates are likely to be superior to expected updates on problems with large stochastic branching factors and too many states to be solved exactly.

</div>

### 8.6 Trajectory Sampling

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Trajectory Sampling)</span></p>

There are two main ways of distributing updates:

1. **Exhaustive sweeps:** The classical approach from dynamic programming. Perform sweeps through the entire state (or state–action) space, updating each state (or state–action pair) once per sweep.
2. **Trajectory sampling:** Sample from the state or state–action space according to the **on-policy distribution**, that is, the distribution observed when following the current policy. One simulates explicit individual trajectories and performs updates at the state or state–action pairs encountered along the way.

Trajectory sampling is appealing because the on-policy distribution is easily generated by simply interacting with the model following the current policy. In an episodic task, one starts in a start state and simulates until the terminal state; in a continuing task, one starts anywhere and just keeps simulating.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Advantages of On-Policy Trajectory Sampling)</span></p>

Experiments on randomly generated episodic tasks comparing uniform (exhaustive sweeps) and on-policy (trajectory sampling) update distributions show:

* In the short term, sampling according to the on-policy distribution resulted in faster planning initially by focusing on states that are near descendants of the start state.
* The effect was stronger, and the initial period of faster planning was longer, with smaller branching factors.
* The advantage of on-policy focusing is large and long-lasting for problems in which a small subset of the state–action space is visited under the on-policy distribution.
* In the long run, focusing on the on-policy distribution may hurt for small problems because the commonly occurring states already have correct values, and sampling them is useless. This is why the exhaustive, unfocused approach can do better in the long run for small problems.

</div>

### 8.7 Real-time Dynamic Programming

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Real-time Dynamic Programming (RTDP))</span></p>

**Real-time dynamic programming (RTDP)** is an on-policy trajectory-sampling version of the value-iteration algorithm of dynamic programming. At every step of a real or simulated trajectory, RTDP selects a greedy action (breaking ties randomly) and applies the expected value-iteration update operation to the current state. It can also update the values of an arbitrary collection of other states at each step.

RTDP is an example of an **asynchronous** DP algorithm: asynchronous DP algorithms are not organized in terms of systematic sweeps of the state set; they update state values in any order whatsoever, using whatever values of other states happen to be available. In RTDP, the update order is dictated by the order states are visited in real or simulated trajectories.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Optimal Partial Policy and Irrelevant States)</span></p>

On-policy trajectory sampling allows the algorithm to completely skip states that cannot be reached by the given policy from any of the start states: such states are **irrelevant** to the prediction problem. What is needed is an **optimal partial policy**, meaning a policy that is optimal for the relevant states but can specify arbitrary actions, or even be undefined, for the irrelevant states.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(RTDP Convergence)</span></p>

For certain types of problems satisfying reasonable conditions, RTDP is guaranteed to find a policy that is optimal on the relevant states without visiting every state infinitely often, or even without visiting some states at all. This can be a great advantage for problems with very large state sets, where even a single sweep may not be feasible.

The result holds for undiscounted episodic tasks for MDPs with absorbing goal states that generate zero rewards. At every step, RTDP applies the expected value-iteration update. For each episode beginning in a state randomly chosen from the set of start states and ending at a goal state, RTDP converges with probability one to a policy that is optimal for all the relevant states provided:

1. The initial value of every goal state is zero.
2. There exists at least one policy that guarantees that a goal state will be reached with probability one from any start state.
3. All rewards for transitions from non-goal states are strictly negative.
4. All the initial values are equal to, or greater than, their optimal values (which can be satisfied by simply setting the initial values of all states to zero).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Stochastic Optimal Path Problems)</span></p>

Tasks having the properties required for RTDP convergence are examples of **stochastic optimal path problems**, which are usually stated in terms of cost minimization instead of reward maximization. Maximizing the negative returns in our version is equivalent to minimizing the costs of paths from a start state to a goal state. Examples include minimum-time control tasks, where each time step required to reach a goal produces a reward of $-1$.

</div>

### 8.8 Summary

| Concept | Key Idea |
| --- | --- |
| Models | Distribution models give $p(s', r \mid s, a)$; sample models produce individual transitions |
| Planning | Any computational process that takes a model and produces/improves a policy |
| Dyna-Q | Integrates acting, model-learning, direct RL, and planning in one architecture |
| Dyna-Q+ | Adds exploration bonus $\kappa\sqrt{\tau}$ to encourage revisiting long-untried transitions |
| Prioritized sweeping | Focuses planning on state–action pairs whose values would change most, using a priority queue |
| Expected vs. sample updates | Expected updates are more accurate but cost $b$ times more; sample updates win for large branching factors |
| Trajectory sampling | On-policy distribution focuses updates on relevant states; faster initially, especially for large problems |
| RTDP | On-policy trajectory-sampling value iteration; can find optimal partial policies without visiting all states |

Key takeaways from Chapter 8:

* Planning and learning are deeply related: both estimate value functions by backing-up update operations. They differ only in whether simulated or real experience is used.
* Planning in small, incremental steps enables it to be interrupted or redirected at any time with little wasted computation, which is key for efficiently intermixing planning with acting and learning.
* When the model is wrong, planning may compute a suboptimal policy. Exploration bonuses (as in Dyna-Q+) can help detect and correct model errors by encouraging the agent to try long-untried actions.
* Prioritized sweeping can dramatically improve planning efficiency by focusing updates where they will have the largest effect, propagating value changes backward through the state space.
* For problems with large stochastic branching factors, many sample updates distributed across the state space outperform fewer expected updates concentrated on individual state–action pairs.
* Trajectory sampling according to the on-policy distribution provides a natural and effective way to focus planning on the most relevant parts of the state space.
* RTDP shows that for certain problem classes, an optimal policy can be found without ever visiting large portions of the state space.

## Part II: Approximate Solution Methods

In the second part of the book we extend the tabular methods presented in the first part to apply to problems with arbitrarily large state spaces. In such cases we cannot expect to find an optimal policy or the optimal value function even in the limit of infinite time and data; our goal instead is to find a good approximate solution using limited computational resources.

The problem with large state spaces is not just the memory needed for large tables, but the time and data needed to fill them accurately. In many of our target tasks, almost every state encountered will never have been seen before. The key issue is that of *generalization*: how can experience with a limited subset of the state space be usefully generalized to produce a good approximation over a much larger subset?

The kind of generalization we require is often called **function approximation** because it takes examples from a desired function (e.g., a value function) and attempts to generalize from them to construct an approximation of the entire function. Function approximation is an instance of *supervised learning*.

## Chapter 9: On-policy Prediction with Approximation

In this chapter, we begin our study of function approximation in reinforcement learning by considering its use in estimating the state-value function from on-policy data, that is, in approximating $v_\pi$ from experience generated using a known policy $\pi$. The novelty in this chapter is that the approximate value function is represented not as a table but as a parameterized functional form with weight vector $\mathbf{w} \in \mathbb{R}^d$. We will write $\hat{v}(s, \mathbf{w}) \approx v_\pi(s)$ for the approximate value of state $s$ given weight vector $\mathbf{w}$.

Typically, the number of weights (the dimensionality of $\mathbf{w}$) is much less than the number of states ($d \ll \lvert \mathcal{S} \rvert$), and changing one weight changes the estimated value of many states. Consequently, when a single state is updated, the change generalizes from that state to affect the values of many other states.

### 9.1 Value-function Approximation

All of the prediction methods covered in this book have been described as updates to an estimated value function that shift its value at particular states toward a "backed-up value," or *update target*, for that state. Let us refer to an individual update by the notation $s \mapsto u$, where $s$ is the state updated and $u$ is the update target that $s$'s estimated value is shifted toward.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Update Targets for Various Methods)</span></p>

* The **Monte Carlo** update for value prediction is $S_t \mapsto G_t$.
* The **TD(0)** update is $S_t \mapsto R_{t+1} + \gamma \hat{v}(S_{t+1}, \mathbf{w}_t)$.
* The **$n$-step TD** update is $S_t \mapsto G_{t:t+n}$.
* The **DP** (dynamic programming) policy-evaluation update is $s \mapsto \mathbb{E}_\pi[R_{t+1} + \gamma \hat{v}(S_{t+1}, \mathbf{w}_t) \mid S_t = s]$.

Each update $s \mapsto u$ can be interpreted as specifying an example of the desired input–output behavior of the value function. When the outputs are numbers, this process is called **function approximation**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Requirements for RL Function Approximation)</span></p>

Not all function approximation methods are equally well suited for use in reinforcement learning:

* Methods must be able to learn **online**, while the agent interacts with its environment or with a model of its environment.
* Methods must handle **nonstationary** target functions (target functions that change over time), as arises in control methods based on GPI where $q_\pi$ changes as $\pi$ changes.
* Methods that assume a static training set over which multiple passes are made are less suitable for reinforcement learning.

</div>

### 9.2 The Prediction Objective ($\overline{\text{VE}}$)

With genuine approximation, an update at one state affects many others, and it is not possible to get the values of all states exactly correct. We are obligated then to say which states we care most about.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Mean Square Value Error)</span></p>

We must specify a state distribution $\mu(s) \geq 0$, $\sum_s \mu(s) = 1$, representing how much we care about the error in each state $s$. By the error in a state $s$ we mean the square of the difference between the approximate value $\hat{v}(s, \mathbf{w})$ and the true value $v_\pi(s)$. Weighting this over the state space by $\mu$, we obtain the **mean square value error**, denoted $\overline{\text{VE}}$:

$$\overline{\text{VE}}(\mathbf{w}) \doteq \sum_{s \in \mathcal{S}} \mu(s) \bigl[v_\pi(s) - \hat{v}(s, \mathbf{w})\bigr]^2.$$

The square root of $\overline{\text{VE}}$ gives a rough measure of how much the approximate values differ from the true values and is often used in plots.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(On-policy Distribution)</span></p>

Often $\mu(s)$ is chosen to be the fraction of time spent in $s$. Under on-policy training this is called the **on-policy distribution**. In continuing tasks, the on-policy distribution is the stationary distribution under $\pi$.

In an episodic task, the on-policy distribution depends on how the initial states of episodes are chosen. Let $h(s)$ denote the probability that an episode begins in state $s$, and let $\eta(s)$ denote the expected number of time steps spent, on average, in state $s$ in a single episode:

$$\eta(s) = h(s) + \sum_{\bar{s}} \eta(\bar{s}) \sum_a \pi(a \mid \bar{s}) p(s \mid \bar{s}, a), \quad \text{for all } s \in \mathcal{S}.$$

The on-policy distribution is then:

$$\mu(s) = \frac{\eta(s)}{\sum_{s'} \eta(s')}, \quad \text{for all } s \in \mathcal{S}.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Global vs Local Optima of $\overline{\text{VE}}$)</span></p>

An ideal goal in terms of $\overline{\text{VE}}$ would be to find a *global optimum*, a weight vector $\mathbf{w}^*$ for which $\overline{\text{VE}}(\mathbf{w}^*) \leq \overline{\text{VE}}(\mathbf{w})$ for all possible $\mathbf{w}$. Reaching this goal is sometimes possible for simple function approximators such as linear ones, but is rarely possible for complex function approximators such as artificial neural networks and decision trees. Complex function approximators may instead seek to converge to a *local optimum*, a weight vector $\mathbf{w}^*$ for which $\overline{\text{VE}}(\mathbf{w}^*) \leq \overline{\text{VE}}(\mathbf{w})$ for all $\mathbf{w}$ in some neighborhood of $\mathbf{w}^*$.

</div>

### 9.3 Stochastic-gradient and Semi-gradient Methods

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Stochastic Gradient Descent)</span></p>

In **stochastic gradient-descent** (SGD) methods, the weight vector is a column vector with a fixed number of real valued components, $\mathbf{w} \doteq (w_1, w_2, \dots, w_d)^\top$, and the approximate value function $\hat{v}(s, \mathbf{w})$ is a differentiable function of $\mathbf{w}$ for all $s \in \mathcal{S}$.

On each step, we observe a new example $S_t \mapsto v_\pi(S_t)$. SGD methods adjust the weight vector by a small amount in the direction that would most reduce the error on that example:

$$\mathbf{w}_{t+1} \doteq \mathbf{w}_t + \alpha\bigl[v_\pi(S_t) - \hat{v}(S_t, \mathbf{w}_t)\bigr] \nabla \hat{v}(S_t, \mathbf{w}_t),$$

where $\alpha$ is a positive step-size parameter, and $\nabla f(\mathbf{w})$ denotes the **gradient** of a scalar expression $f(\mathbf{w})$ with respect to $\mathbf{w}$:

$$\nabla f(\mathbf{w}) \doteq \left(\frac{\partial f(\mathbf{w})}{\partial w_1}, \frac{\partial f(\mathbf{w})}{\partial w_2}, \dots, \frac{\partial f(\mathbf{w})}{\partial w_d}\right)^\top.$$

This update moves $\mathbf{w}_t$ in the direction of the negative gradient of the example's squared error. Over many examples, the overall effect is to minimize an average performance measure such as $\overline{\text{VE}}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why SGD Takes Small Steps)</span></p>

SGD takes only a small step in the direction of the gradient rather than moving all the way to eliminate the error on the example. We do not seek or expect to find a value function that has zero error for all states, but only an approximation that balances the errors in different states. If we completely corrected each example in one step, we would not find such a balance. If $\alpha$ decreases over time satisfying the standard stochastic approximation conditions, then the SGD method is guaranteed to converge to a local optimum.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(General SGD for State-value Prediction)</span></p>

In practice, $v_\pi(S_t)$ is unknown, so we substitute a target $U_t$ in place of $v_\pi(S_t)$:

$$\mathbf{w}_{t+1} \doteq \mathbf{w}_t + \alpha\bigl[U_t - \hat{v}(S_t, \mathbf{w}_t)\bigr] \nabla \hat{v}(S_t, \mathbf{w}_t).$$

If $U_t$ is an *unbiased* estimate, i.e., $\mathbb{E}[U_t \mid S_t = s] = v_\pi(s)$ for each $t$, then $\mathbf{w}_t$ is guaranteed to converge to a local optimum under the usual stochastic approximation conditions for decreasing $\alpha$. For example, the Monte Carlo target $U_t \doteq G_t$ is by definition an unbiased estimate of $v_\pi(S_t)$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Gradient Monte Carlo Algorithm for Estimating $\hat{v} \approx v_\pi$)</span></p>

**Input:** the policy $\pi$ to be evaluated

**Input:** a differentiable function $\hat{v} : \mathcal{S} \times \mathbb{R}^d \to \mathbb{R}$

**Algorithm parameter:** step size $\alpha > 0$

Initialize value-function weights $\mathbf{w} \in \mathbb{R}^d$ arbitrarily (e.g., $\mathbf{w} = \mathbf{0}$)

Loop forever (for each episode):
* Generate an episode $S_0, A_0, R_1, S_1, A_1, \dots, R_T, S_T$ using $\pi$
* Loop for each step of episode, $t = 0, 1, \dots, T - 1$:
  * $\mathbf{w} \leftarrow \mathbf{w} + \alpha\bigl[G_t - \hat{v}(S_t, \mathbf{w})\bigr] \nabla \hat{v}(S_t, \mathbf{w})$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Semi-gradient Methods)</span></p>

Bootstrapping targets such as $n$-step returns $G_{t:t+n}$ or the DP target 

$$\sum_{a,s',r} \pi(a \mid S_t) \sum_{s',r} p(s',r \mid S_t, a)[r + \gamma \hat{v}(s', \mathbf{w}_t)]$$

all depend on the current value of the weight vector $\mathbf{w}_t$, which implies that they will be biased and that they will not produce a true gradient-descent method. Methods that take into account the effect of changing the weight vector $\mathbf{w}_t$ on the estimate, but ignore its effect on the target, are called **semi-gradient methods**.

Although semi-gradient (bootstrapping) methods do not converge as robustly as gradient methods, they do converge reliably in important cases such as the linear case. Moreover, they offer important advantages: they typically enable significantly faster learning and they enable learning to be continual and online, without waiting for the end of an episode.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Semi-gradient TD(0) for Estimating $\hat{v} \approx v_\pi$)</span></p>

**Input:** the policy $\pi$ to be evaluated

**Input:** a differentiable function $\hat{v} : \mathcal{S}^+ \times \mathbb{R}^d \to \mathbb{R}$ such that $\hat{v}(\text{terminal}, \cdot) = 0$

**Algorithm parameter:** step size $\alpha > 0$

Initialize value-function weights $\mathbf{w} \in \mathbb{R}^d$ arbitrarily (e.g., $\mathbf{w} = \mathbf{0}$)

Loop for each episode:
* Initialize $S$
* Loop for each step of episode:
  * Choose $A \sim \pi(\cdot \mid S)$
  * Take action $A$, observe $R, S'$
  * $\mathbf{w} \leftarrow \mathbf{w} + \alpha\bigl[R + \gamma \hat{v}(S', \mathbf{w}) - \hat{v}(S, \mathbf{w})\bigr] \nabla \hat{v}(S, \mathbf{w})$
  * $S \leftarrow S'$
* until $S$ is terminal

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(State Aggregation)</span></p>

**State aggregation** is a simple form of generalizing function approximation in which states are grouped together, with one estimated value (one component of the weight vector $\mathbf{w}$) for each group. The value of a state is estimated as its group's component, and when the state is updated, that component alone is updated. State aggregation is a special case of SGD in which the gradient, $\nabla \hat{v}(S_t, \mathbf{w}_t)$, is 1 for $S_t$'s group's component and 0 for the other components.

</div>

### 9.4 Linear Methods

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Linear Function Approximation)</span></p>

One of the most important special cases of function approximation is that in which the approximate function, $\hat{v}(\cdot, \mathbf{w})$, is a linear function of the weight vector $\mathbf{w}$. Corresponding to every state $s$, there is a real-valued vector $\mathbf{x}(s) \doteq (x_1(s), x_2(s), \dots, x_d(s))^\top$, with the same number of components as $\mathbf{w}$. Linear methods approximate the state-value function by the inner product between $\mathbf{w}$ and $\mathbf{x}(s)$:

$$\hat{v}(s, \mathbf{w}) \doteq \mathbf{w}^\top \mathbf{x}(s) \doteq \sum_{i=1}^{d} w_i x_i(s).$$

The vector $\mathbf{x}(s)$ is called a **feature vector** representing state $s$. Each component $x_i(s)$ is the value of a function $x_i : \mathcal{S} \to \mathbb{R}$. We think of a feature as the entirety of one of these functions, and we call its value for a state $s$ a **feature** of $s$. For linear methods, features are **basis functions** because they form a linear basis for the set of approximate functions.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Linear SGD Update)</span></p>

In the linear case the gradient of the approximate value function is simply:

$$\nabla \hat{v}(s, \mathbf{w}) = \mathbf{x}(s).$$

Thus, the general SGD update reduces to a particularly simple form:

$$\mathbf{w}_{t+1} \doteq \mathbf{w}_t + \alpha\bigl[U_t - \hat{v}(S_t, \mathbf{w}_t)\bigr] \mathbf{x}(S_t).$$

The linear SGD case is one of the most favorable for mathematical analysis. In particular, in the linear case there is only one optimum (or, in degenerate cases, one set of equally good optima), and thus any method that is guaranteed to converge to or near a local optimum is automatically guaranteed to converge to or near the global optimum.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(TD Fixed Point)</span></p>

The semi-gradient TD(0) algorithm also converges under linear function approximation, but not to the global optimum. The update at each time $t$ is:

$$\mathbf{w}_{t+1} \doteq \mathbf{w}_t + \alpha\bigl(R_{t+1} + \gamma \mathbf{w}_t^\top \mathbf{x}_{t+1} - \mathbf{w}_t^\top \mathbf{x}_t\bigr)\mathbf{x}_t,$$

where $\mathbf{x}_t = \mathbf{x}(S_t)$. Once the system has reached steady state, for any given $\mathbf{w}_t$, the expected next weight vector can be written:

$$\mathbb{E}[\mathbf{w}_{t+1} \mid \mathbf{w}_t] = \mathbf{w}_t + \alpha(\mathbf{b} - \mathbf{A}\mathbf{w}_t),$$

where $\mathbf{b} \doteq \mathbb{E}[R_{t+1}\mathbf{x}_t] \in \mathbb{R}^d$ and $\mathbf{A} \doteq \mathbb{E}\bigl[\mathbf{x}_t(\mathbf{x}_t - \gamma \mathbf{x}_{t+1})^\top\bigr] \in \mathbb{R}^{d \times d}$. If the system converges, it must converge to the weight vector $\mathbf{w}_{\text{TD}}$ at which:

$$\mathbf{w}_{\text{TD}} \doteq \mathbf{A}^{-1}\mathbf{b}.$$

This quantity is called the **TD fixed point**. In fact linear semi-gradient TD(0) converges to this point.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Convergence of Linear TD(0))</span></p>

The key matrix $\mathbf{A}$ can be written as $\mathbf{A} = \mathbf{X}^\top \mathbf{D}(\mathbf{I} - \gamma \mathbf{P})\mathbf{X}$, where $\mathbf{D}$ is the $\lvert \mathcal{S} \rvert \times \lvert \mathcal{S} \rvert$ diagonal matrix with $\mu(s)$ on its diagonal, $\mathbf{P}$ is the $\lvert \mathcal{S} \rvert \times \lvert \mathcal{S} \rvert$ matrix of transition probabilities under $\pi$, and $\mathbf{X}$ is the $\lvert \mathcal{S} \rvert \times d$ matrix with $\mathbf{x}(s)$ as its rows.

The matrix $\mathbf{A}$ is positive definite (for $\gamma < 1$ in the continuing case), which ensures that the inverse $\mathbf{A}^{-1}$ exists and that linear semi-gradient TD(0) is stable. The proof uses the fact that $\mathbf{D}(\mathbf{I} - \gamma \mathbf{P})$ has positive column sums (since $\mathbf{P}$ is stochastic, $\gamma < 1$, and $\mu$ is the stationary distribution).

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(TD Error Bound)</span></p>

At the TD fixed point, it has been proven (in the continuing case) that the $\overline{\text{VE}}$ is within a bounded expansion of the lowest possible error:

$$\overline{\text{VE}}(\mathbf{w}_{\text{TD}}) \;\leq\; \frac{1}{1 - \gamma} \min_{\mathbf{w}} \overline{\text{VE}}(\mathbf{w}).$$

That is, the asymptotic error of the TD method is no more than $\frac{1}{1-\gamma}$ times the smallest possible error, that attained in the limit by the Monte Carlo method. Because $\gamma$ is often near one, this expansion factor can be quite large, so there is substantial potential loss in asymptotic performance with the TD method. On the other hand, TD methods are often of vastly reduced variance compared to Monte Carlo methods, and thus faster.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">($n$-step Semi-gradient TD for Estimating $\hat{v} \approx v_\pi$)</span></p>

**Input:** the policy $\pi$ to be evaluated

**Input:** a differentiable function $\hat{v} : \mathcal{S}^+ \times \mathbb{R}^d \to \mathbb{R}$ such that $\hat{v}(\text{terminal}, \cdot) = 0$

**Algorithm parameters:** step size $\alpha > 0$, a positive integer $n$

Initialize value-function weights $\mathbf{w} \in \mathbb{R}^d$ arbitrarily (e.g., $\mathbf{w} = \mathbf{0}$)

All store and access operations ($S_t$ and $R_t$) can take their index mod $n + 1$

Loop for each episode:
* Initialize and store $S_0 \neq \text{terminal}$
* $T \leftarrow \infty$
* Loop for $t = 0, 1, 2, \dots$:
  * If $t < T$, then:
    * Take an action according to $\pi(\cdot \mid S_t)$
    * Observe and store the next reward as $R_{t+1}$ and the next state as $S_{t+1}$
    * If $S_{t+1}$ is terminal, then $T \leftarrow t + 1$
  * $\tau \leftarrow t - n + 1$ &ensp; ($\tau$ is the time whose state's estimate is being updated)
  * If $\tau \geq 0$:
    * $G \leftarrow \sum_{i=\tau+1}^{\min(\tau+n, T)} \gamma^{i-\tau-1} R_i$
    * If $\tau + n < T$, then: $G \leftarrow G + \gamma^n \hat{v}(S_{\tau+n}, \mathbf{w})$ &ensp; ($G_{\tau:\tau+n}$)
    * $\mathbf{w} \leftarrow \mathbf{w} + \alpha\bigl[G - \hat{v}(S_\tau, \mathbf{w})\bigr] \nabla \hat{v}(S_\tau, \mathbf{w})$
* Until $\tau = T - 1$

The key equation of this algorithm is:

$$\mathbf{w}_{t+n} \doteq \mathbf{w}_{t+n-1} + \alpha\bigl[G_{t:t+n} - \hat{v}(S_t, \mathbf{w}_{t+n-1})\bigr] \nabla \hat{v}(S_t, \mathbf{w}_{t+n-1}), \quad 0 \leq t < T,$$

where the $n$-step return is generalized from the tabular case to:

$$G_{t:t+n} \doteq R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n \hat{v}(S_{t+n}, \mathbf{w}_{t+n-1}), \quad 0 \leq t \leq T - n.$$

</div>

### 9.5 Feature Construction for Linear Methods

Linear methods are interesting because of their convergence guarantees, but also because in practice they can be very efficient in terms of both data and computation. Whether or not this is so depends critically on how the states are represented in terms of features. Choosing features appropriate to the task is an important way of adding prior domain knowledge to reinforcement learning systems.

A limitation of the linear form is that it cannot take into account any interactions between features. It needs instead features for combinations of the two underlying state dimensions.

#### 9.5.1 Polynomials

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Polynomial Basis Features)</span></p>

Suppose each state $s$ corresponds to $k$ numbers, $s_1, s_2, \dots, s_k$, with each $s_i \in \mathbb{R}$. For this $k$-dimensional state space, each order-$n$ polynomial-basis feature $x_i$ can be written as:

$$x_i(s) = \prod_{j=1}^{k} s_j^{c_{i,j}},$$

where each $c_{i,j}$ is an integer in the set $\lbrace 0, 1, \dots, n \rbrace$ for an integer $n \geq 0$. These features make up the order-$n$ polynomial basis for dimension $k$, which contains $(n+1)^k$ different features.

For example, with two state dimensions $s_1, s_2$ and order $n = 2$, the four-dimensional feature vector $\mathbf{x}(s) = (1, s_1, s_2, s_1 s_2)^\top$ enables representation of affine functions and interactions.

</div>

#### 9.5.2 Fourier Basis

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Fourier Cosine Basis)</span></p>

The one-dimensional order-$n$ Fourier cosine basis consists of $n + 1$ features:

$$x_i(s) = \cos(i\pi s), \quad s \in [0, 1],$$

for $i = 0, \dots, n$. The feature $x_0$ is a constant function.

In the multi-dimensional case, suppose each state $s$ corresponds to a vector of $k$ numbers, $\mathbf{s} = (s_1, s_2, \dots, s_k)^\top$, with each $s_i \in [0, 1]$. The $i$th feature in the order-$n$ Fourier cosine basis can then be written:

$$x_i(s) = \cos\bigl(\pi \mathbf{s}^\top \mathbf{c}^i\bigr),$$

where $\mathbf{c}^i = (c_1^i, \dots, c_k^i)^\top$, with $c_j^i \in \lbrace 0, \dots, n \rbrace$ for $j = 1, \dots, k$ and $i = 1, \dots, (n+1)^k$. This defines a feature for each of the $(n+1)^k$ possible integer vectors $\mathbf{c}^i$. The inner product $\mathbf{s}^\top \mathbf{c}^i$ assigns an integer in $\lbrace 0, \dots, n \rbrace$ to each dimension, which determines the feature's frequency along that dimension.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Fourier Basis Properties)</span></p>

* Fourier cosine features with Sarsa can produce good performance compared to several other collections of basis functions, including polynomial and radial basis functions.
* Fourier features have trouble with discontinuities because it is difficult to avoid "ringing" around points of discontinuity unless very high frequency basis functions are included.
* The number of features in the order-$n$ Fourier basis grows exponentially with the dimension of the state space. For high dimension state spaces, it is necessary to select a subset of features.
* Because Fourier features are non-zero over the entire state space (with few exceptions), they represent global properties of states, which can make it difficult to find good ways to represent local properties.
* When using Fourier cosine features with a learning algorithm, it may be helpful to use a different step-size parameter for each feature $x_i$: $\alpha_i = \alpha / \sqrt{(c_1^i)^2 + \cdots + (c_k^i)^2}$ (except when each $c_j^i = 0$, in which case $\alpha_i = \alpha$).

</div>

#### 9.5.3 Coarse Coding

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Coarse Coding)</span></p>

Consider a task in which the natural representation of the state set is a continuous two-dimensional space. One kind of representation for this case is made up of features corresponding to circles (or receptive fields) in state space. If the state is inside a circle, then the corresponding feature has the value 1 and is said to be **present**; otherwise the feature has the value 0 and is said to be **absent**. This kind of 1–0-valued feature is called a **binary feature**. Given a state, the binary features that are present indicate the circles in which the state lies, and these coarsely code for the state's location. Representing a state with features that overlap in this way (but are not the same as) is known as **coarse coding**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Effect of Receptive Field Size and Shape)</span></p>

Suppose we are doing on-line learning using the linear gradient-descent method. The initial generalization from one point to another is determined by the number of features that the two points have in common:

* **Small receptive fields:** Fine discrimination, less generalization. The learned value function can be more complex and accurate, but requires more experience to train.
* **Large receptive fields:** Broad generalization, coarser resolution. Initial generalization will be over a broader area, and the final value function will be smoother, less complex.
* **Shape matters:** Features with large shapes in one dimension and small shapes in another will generalize broadly along the first dimension and narrowly along the second, producing a final value function that discriminates more along the second dimension.

</div>

#### 9.5.4 Tile Coding

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Tile Coding)</span></p>

**Tile coding** is a form of coarse coding that is particularly computationally efficient and flexible. In tile coding the receptive fields of the features are grouped into partitions of the state space called **tilings**. Each element of a partition is called a **tile**. Each tiling partitions the entire state space with just one tile per point, so a single tiling provides only a coarse, non-overlapping coding. To get finer resolution, multiple tilings are used, each offset from the others by a fraction of a tile width.

A simple case is a one-dimensional continuous state space, covered by multiple overlapping tilings, each with regularly spaced tiles. With $n$ tilings, each offset by $1/n$ of a tile width, the resolution is $n$ times finer than the tile width. Every state activates exactly one tile per tiling, thus exactly $n$ features are present (have value 1) at any time.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Tile Coding)</span></p>

* With tile coding and the linear SGD learning rule, the step-size parameter is usually set to $\alpha = 1/n$, where $n$ is the number of tilings.
* An immediate practical advantage of tile coding is that the overall number of features present at one time (and thus the computation) is the same for any state space: exactly $n$ (one per tiling) regardless of how many dimensions or how finely we partition.
* Tile shapes can be chosen to be appropriate to the task at hand: elongated tiles along one dimension generalize broadly along that dimension.
* For spaces of high dimensionality, one can use **hashing** — a consistent pseudo-random collapsing of a large tiling into a much smaller set of tiles. Hashing produces tiles of irregular shape but still ensures that nearby points share more tiles than distant points.

</div>

#### 9.5.5 Radial Basis Functions

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Radial Basis Functions)</span></p>

**Radial basis functions** (RBFs) are the natural generalization of coarse coding to continuous-valued features. Rather than each feature being either 0 or 1, it can be anything in the interval $[0, 1]$, reflecting varying degrees to which the state "belongs" to the feature. A typical RBF feature $x_i$ has a Gaussian (bell-shaped) response $x_i(s)$ dependent only on the distance between the state $s$ and the feature's prototypical or center state $c_i$, and relative to the feature's width $\sigma_i$:

$$x_i(s) \doteq \exp\!\left(-\frac{\lVert s - c_i \rVert^2}{2\sigma_i^2}\right).$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(RBFs vs Binary Features)</span></p>

An advantage of RBFs over binary features is that they produce approximate value functions that vary smoothly and are differentiable. The main disadvantage of RBFs compared to simpler binary features like tile coding is greater computational complexity. In high-dimensional state spaces, tilings offer far better computational properties with little loss of approximation quality. An RBF *network* is a linear function approximator using RBFs as features, which constitutes a parameterized function approximator.

</div>

### 9.6 Selecting Step-Size Parameters Manually

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Selecting $\alpha$ for Linear SGD)</span></p>

For linear SGD methods, a good rule of thumb for setting the step-size parameter is:

$$\alpha \doteq \left(\tau \;\mathbb{E}[\mathbf{x}^\top \mathbf{x}]\right)^{-1},$$

where $\tau$ is the number of experiences within which you would like to learn, and $\mathbb{E}[\mathbf{x}^\top \mathbf{x}]$ is an estimate of the expected value of $\mathbf{x}^\top \mathbf{x}$. With tile coding this simplifies to $\alpha = 1/\tau n$, where $n$ is the number of tilings.

For example, if you want to largely learn within about 10 experiences with tile coding of $n = 8$ tilings, a good $\alpha$ would be $\alpha = 1/(10 \cdot 8) = 0.0125$.

</div>

### 9.7 Nonlinear Function Approximation: Artificial Neural Networks

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Artificial Neural Networks)</span></p>

**Artificial neural networks** (ANNs) are widely used as nonlinear function approximators in reinforcement learning. A network is typically composed of interconnected units (loosely modeling neurons) organized in layers. Each unit computes a weighted sum of its inputs plus a bias, then passes the result through a nonlinear **activation function** (such as the sigmoid $\sigma(x) = 1/(1 + e^{-x})$ or the rectifier $\max(0, x)$, producing a **rectified linear unit** or ReLU).

ANNs are trained by **backpropagation** — an efficient method for computing the gradient of the network's overall error with respect to all the weights, based on the chain rule. When combined with SGD (or its variants), this process adjusts the weights to minimize the error.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Deep Reinforcement Learning)</span></p>

Using deep neural networks (many hidden layers) as function approximators in RL is known as **deep reinforcement learning**. A breakthrough was the deep Q-network (DQN) by Mnih et al. (2015), which learned to play many Atari 2600 video games at a human level using only the raw pixel images as input.

Key techniques for stabilizing deep RL include:
* **Experience replay:** Storing experience in a replay buffer and sampling random mini-batches for training, which breaks the correlation between successive samples.
* **Target network:** Using a separate, slowly-updated copy of the network to compute update targets, which stabilizes learning.
* **Batch normalization:** Normalizing inputs to each layer to reduce internal covariate shift.

</div>

### 9.8 Least-Squares TD

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(LSTD — Least-Squares TD)</span></p>

The TD fixed point $\mathbf{w}_{\text{TD}} = \mathbf{A}^{-1}\mathbf{b}$ can be computed directly rather than iteratively. **Least-Squares TD** (LSTD) does this by forming estimates of $\mathbf{A}$ and $\mathbf{b}$ from the observed data:

$$\hat{\mathbf{A}}_t \doteq \sum_{k=0}^{t-1} \mathbf{x}_k (\mathbf{x}_k - \gamma \mathbf{x}_{k+1})^\top + \varepsilon \mathbf{I}, \qquad \hat{\mathbf{b}}_t \doteq \sum_{k=0}^{t-1} R_{k+1} \mathbf{x}_k,$$

where $\varepsilon \mathbf{I}$ (for $\varepsilon > 0$ small) ensures $\hat{\mathbf{A}}_t$ is always invertible. The LSTD estimate is then:

$$\mathbf{w}_t \doteq \hat{\mathbf{A}}_t^{-1} \hat{\mathbf{b}}_t.$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(LSTD)</span></p>

* LSTD is the most **data efficient** form of linear TD prediction. It makes the best possible use of the data but at the expense of more computation.
* Naively computing $\hat{\mathbf{A}}_t^{-1}$ at each step requires $O(d^3)$ time. Using the **Sherman–Morrison formula**, the inverse can be updated incrementally in $O(d^2)$ per step:

$$\hat{\mathbf{A}}_t^{-1} = \hat{\mathbf{A}}_{t-1}^{-1} - \frac{\hat{\mathbf{A}}_{t-1}^{-1} \mathbf{x}_t (\mathbf{x}_t - \gamma \mathbf{x}_{t+1})^\top \hat{\mathbf{A}}_{t-1}^{-1}}{1 + (\mathbf{x}_t - \gamma \mathbf{x}_{t+1})^\top \hat{\mathbf{A}}_{t-1}^{-1} \mathbf{x}_t}.$$

* The $O(d^2)$ per-step cost is still significantly more than the $O(d)$ cost of semi-gradient TD. Whether the greater data efficiency of LSTD is worth the computational expense depends on how large $d$ is, how important it is to learn quickly, and the expense of other parts of the system.
* LSTD has no step-size parameter $\alpha$, which is a major advantage: it never requires tuning of $\alpha$ and is never subject to step-size-related instabilities.
* The lack of a step-size parameter means LSTD never "forgets," which makes it inappropriate for nonstationary settings without additional mechanisms.

</div>

### 9.9 Memory-based Function Approximation

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Memory-based Function Approximation)</span></p>

**Memory-based** (or **lazy learning**) function approximation methods simply save training examples in memory as they arrive, and then, whenever a query state's value estimate is needed, retrieve a set of examples from memory and use them to compute a value estimate for the query state. This approach is sometimes called **local-learning** because approximations are local to the query state.

Examples include:
* **Nearest neighbor:** Return the value of the example in memory whose state is closest to the query state.
* **Weighted average:** Return a weighted average of several nearest neighbors' values, with weight decreasing with distance.
* **Locally weighted regression:** Fit a surface (e.g., a line or hyperplane) to the values of a set of nearest examples, weighted by distance, and evaluate the fitted function at the query state.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Advantages of Memory-based Methods)</span></p>

Memory-based methods are **nonparametric**: they do not restrict approximations to fixed parameterized classes. Instead, the approximation becomes more complex as more data is gathered. A significant advantage is that they naturally handle the locality issue: they do not worry about the global approximation — local approximations are built on the fly for each query. This avoids the potentially complex global approximation problem and its possible interference effects.

Another advantage is that memory-based methods allow an agent's experience to have an immediate effect on its behavior, without the need for iterative gradient-descent procedures.

</div>

### 9.10 Kernel-based Function Approximation

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Kernel-based Function Approximation)</span></p>

Memory-based methods such as weighted average and locally weighted regression depend on assigning weights to examples $s' \mapsto g$ in memory by some measure of distance between $s'$ and a query state $s$. The function $k : \mathcal{S} \times \mathcal{S} \to \mathbb{R}$ that assigns the weights is called a **kernel function**, or simply a **kernel**. A **kernel regression** estimate is:

$$\hat{v}(s, \mathcal{D}) = \sum_{s' \in \mathcal{D}} k(s, s') g(s'),$$

where $\mathcal{D}$ is the set of stored examples and $g(s')$ is the target for state $s'$. The common Gaussian RBF is an example of a kernel.

</div>

### 9.11 Looking Deeper at On-policy Learning: Interest and Emphasis

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Interest and Emphasis)</span></p>

In many applications, the on-policy distribution is not the ideal weighting of states for the learning objective. A uniform weighting over all states might not be desirable — for example, we may care more about some states than others. To accommodate this, two new concepts are introduced:

* A nonnegative scalar measure, the **interest** $I_t$, indicating the degree to which we are interested in accurately valuing the state (or state–action pair) at time $t$. If we care more about certain states, we give those higher interest.
* A nonnegative scalar random variable, the **emphasis** $M_t$, which multiplies the learning update and is defined as:

$$M_t = I_t + \gamma^n M_{t-n},$$

for general $n$-step learning. The emphasis determines how much a given update influences the weight vector. It captures the cascading effect of interest, distributing it back to the preceding states according to $\gamma^n$.

The update using emphasis is:

$$\mathbf{w}_{t+n} \doteq \mathbf{w}_{t+n-1} + \alpha M_t \bigl[G_{t:t+n} - \hat{v}(S_t, \mathbf{w}_{t+n-1})\bigr] \nabla \hat{v}(S_t, \mathbf{w}_{t+n-1}), \quad 0 \leq t < T.$$

</div>

### 9.12 Summary

| Method | Type | Convergence | Step-size | Computation | Key Property |
| --- | --- | --- | --- | --- | --- |
| Gradient MC | True gradient | To global optimum (linear) | Requires $\alpha$ | $O(d)$ | Unbiased targets |
| Semi-gradient TD(0) | Semi-gradient | To TD fixed point (linear) | Requires $\alpha$ | $O(d)$ | Bootstraps, faster learning |
| $n$-step semi-gradient TD | Semi-gradient | To TD-like fixed point | Requires $\alpha$ | $O(d)$ | Interpolates MC and TD(0) |
| LSTD | Direct solve | To TD fixed point | No $\alpha$ | $O(d^2)$ | Most data efficient (linear) |
| Nonlinear (ANN) | True gradient | Local optimum | Requires $\alpha$ | Varies | Universal approximation |

Key takeaways from Chapter 9:

* Extending reinforcement learning to function approximation makes the methods applicable to problems with arbitrarily large state spaces, but also introduces new issues such as a need for an explicit objective function ($\overline{\text{VE}}$) and potential divergence.
* **SGD methods** are the foundation: they adjust weights after each example in the direction that would most reduce the error on that example.
* **Semi-gradient methods** (used with bootstrapping targets) are not true gradient methods but converge reliably under linear function approximation. They offer the advantage of online, incremental learning.
* **Linear methods** have strong convergence guarantees: gradient MC converges to the global optimum of $\overline{\text{VE}}$, while semi-gradient TD(0) converges to a fixed point within a bounded expansion of the minimum.
* **Feature construction** is crucial for linear methods. Polynomials, Fourier bases, coarse coding, tile coding, and RBFs offer different trade-offs between generality, computational cost, and ability to represent interactions.
* **LSTD** directly computes the TD fixed point with $O(d^2)$ computation per step and no step-size parameter, making it more data efficient but computationally more expensive than semi-gradient TD.
* **Deep reinforcement learning** combines deep neural networks with RL, using techniques like experience replay and target networks to stabilize training.

## Chapter 10: On-policy Control with Approximation

In this chapter we return to the control problem, now with parametric approximation of the action-value function $\hat{q}(s, a, \mathbf{w}) \approx q_*(s, a)$, where $\mathbf{w} \in \mathbb{R}^d$ is a finite-dimensional weight vector. We continue to restrict attention to the on-policy case, leaving off-policy methods to Chapter 11. The present chapter features the semi-gradient Sarsa algorithm, the natural extension of semi-gradient TD(0) to action values and to on-policy control.

### 10.1 Episodic Semi-gradient Control

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Semi-gradient Sarsa for Action Values)</span></p>

The extension of the semi-gradient prediction methods of Chapter 9 to action values is straightforward. Now the approximate action-value function $\hat{q} \approx q_\pi$ is represented as a parameterized functional form with weight vector $\mathbf{w}$. We consider random training examples of the form $S_t, A_t \mapsto U_t$. The general gradient-descent update for action-value prediction is:

$$\mathbf{w}_{t+1} \doteq \mathbf{w}_t + \alpha\bigl[U_t - \hat{q}(S_t, A_t, \mathbf{w}_t)\bigr] \nabla \hat{q}(S_t, A_t, \mathbf{w}_t).$$

For example, the update for the one-step Sarsa method is:

$$\mathbf{w}_{t+1} \doteq \mathbf{w}_t + \alpha\bigl[R_{t+1} + \gamma \hat{q}(S_{t+1}, A_{t+1}, \mathbf{w}_t) - \hat{q}(S_t, A_t, \mathbf{w}_t)\bigr] \nabla \hat{q}(S_t, A_t, \mathbf{w}_t).$$

We call this method **episodic semi-gradient one-step Sarsa**. For a constant policy, this method converges in the same way that TD(0) does, with the same kind of error bound.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(From Prediction to Control)</span></p>

To form control methods, we couple action-value prediction methods with techniques for policy improvement and action selection. If the action set is discrete and not too large, then we can use the techniques already developed: for each possible action $a$ available in the next state $S_{t+1}$, we can compute $\hat{q}(S_{t+1}, a, \mathbf{w}_t)$ and then find the greedy action $A^*_{t+1} = \argmax_a \hat{q}(S_{t+1}, a, \mathbf{w}_t)$. Policy improvement is then done by changing the estimation policy to a soft approximation of the greedy policy such as the $\varepsilon$-greedy policy.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Episodic Semi-gradient Sarsa for Estimating $\hat{q} \approx q_*$)</span></p>

**Input:** a differentiable action-value function parameterization $\hat{q} : \mathcal{S} \times \mathcal{A} \times \mathbb{R}^d \to \mathbb{R}$

**Algorithm parameters:** step size $\alpha > 0$, small $\varepsilon > 0$

Initialize value-function weights $\mathbf{w} \in \mathbb{R}^d$ arbitrarily (e.g., $\mathbf{w} = \mathbf{0}$)

Loop for each episode:
* $S, A \leftarrow$ initial state and action of episode (e.g., $\varepsilon$-greedy)
* Loop for each step of episode:
  * Take action $A$, observe $R, S'$
  * If $S'$ is terminal:
    * $\mathbf{w} \leftarrow \mathbf{w} + \alpha\bigl[R - \hat{q}(S, A, \mathbf{w})\bigr] \nabla \hat{q}(S, A, \mathbf{w})$
    * Go to next episode
  * Choose $A'$ as a function of $\hat{q}(S', \cdot, \mathbf{w})$ (e.g., $\varepsilon$-greedy)
  * $\mathbf{w} \leftarrow \mathbf{w} + \alpha\bigl[R + \gamma \hat{q}(S', A', \mathbf{w}) - \hat{q}(S, A, \mathbf{w})\bigr] \nabla \hat{q}(S, A, \mathbf{w})$
  * $S \leftarrow S'$; $A \leftarrow A'$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Mountain Car Task)</span></p>

Consider the task of driving an underpowered car up a steep mountain road. Gravity is stronger than the car's engine, and even at full throttle the car cannot accelerate up the steep slope. The only solution is to first move away from the goal and up the opposite slope on the left, then by applying full throttle, build enough inertia to carry it up the steep slope even though it is slowing down.

The reward is $-1$ on all time steps until the car moves past its goal position at the top of the mountain. There are three possible actions: full throttle forward ($+1$), full throttle reverse ($-1$), and zero throttle ($0$). The car's position $x_t$ and velocity $\dot{x}_t$ are updated by:

$$x_{t+1} \doteq \text{bound}[x_t + \dot{x}_{t+1}], \qquad \dot{x}_{t+1} \doteq \text{bound}[\dot{x}_t + 0.001 A_t - 0.0025 \cos(3x_t)],$$

where the bound operation enforces $-1.2 \leq x_{t+1} \leq 0.5$ and $-0.07 \leq \dot{x}_{t+1} \leq 0.07$.

Using tile coding with 8 tilings and $\varepsilon$-greedy action selection, the action-value function was approximated linearly: $\hat{q}(s, a, \mathbf{w}) \doteq \mathbf{w}^\top \mathbf{x}(s, a)$. The initial action values were all zero, which was optimistic (all true values are negative), causing extensive exploration even with $\varepsilon = 0$. Performance improved from about 428 steps per episode initially to near-optimal behavior over episodes.

</div>

### 10.2 Semi-gradient $n$-step Sarsa

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Semi-gradient $n$-step Sarsa)</span></p>

We can obtain an $n$-step version of episodic semi-gradient Sarsa by using an $n$-step return as the update target. The $n$-step return immediately generalizes from its tabular form to a function approximation form:

$$G_{t:t+n} \doteq R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n \hat{q}(S_{t+n}, A_{t+n}, \mathbf{w}_{t+n-1}), \quad t + n < T,$$

with $G_{t:t+n} \doteq G_t$ if $t + n \geq T$, as usual. The $n$-step update equation is:

$$\mathbf{w}_{t+n} \doteq \mathbf{w}_{t+n-1} + \alpha\bigl[G_{t:t+n} - \hat{q}(S_t, A_t, \mathbf{w}_{t+n-1})\bigr] \nabla \hat{q}(S_t, A_t, \mathbf{w}_{t+n-1}), \quad 0 \leq t < T.$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Episodic Semi-gradient $n$-step Sarsa for Estimating $\hat{q} \approx q_*$ or $q_\pi$)</span></p>

**Input:** a differentiable action-value function parameterization $\hat{q} : \mathcal{S} \times \mathcal{A} \times \mathbb{R}^d \to \mathbb{R}$

**Input:** a policy $\pi$ (if estimating $q_\pi$)

**Algorithm parameters:** step size $\alpha > 0$, small $\varepsilon > 0$, a positive integer $n$

Initialize value-function weights $\mathbf{w} \in \mathbb{R}^d$ arbitrarily (e.g., $\mathbf{w} = \mathbf{0}$)

All store and access operations ($S_t$, $A_t$, and $R_t$) can take their index mod $n + 1$

Loop for each episode:
* Initialize and store $S_0 \neq \text{terminal}$
* Select and store an action $A_0 \sim \pi(\cdot \mid S_0)$ or $\varepsilon$-greedy wrt $\hat{q}(S_0, \cdot, \mathbf{w})$
* $T \leftarrow \infty$
* Loop for $t = 0, 1, 2, \dots$:
  * If $t < T$, then:
    * Take action $A_t$
    * Observe and store the next reward as $R_{t+1}$ and the next state as $S_{t+1}$
    * If $S_{t+1}$ is terminal, then: $T \leftarrow t + 1$
    * else: Select and store an action $A_{t+1} \sim \pi(\cdot \mid S_{t+1})$ or $\varepsilon$-greedy wrt $\hat{q}(S_{t+1}, \cdot, \mathbf{w})$
  * $\tau \leftarrow t - n + 1$ &ensp; ($\tau$ is the time whose estimate is being updated)
  * If $\tau \geq 0$:
    * $G \leftarrow \sum_{i=\tau+1}^{\min(\tau+n, T)} \gamma^{i-\tau-1} R_i$
    * If $\tau + n < T$, then: $G \leftarrow G + \gamma^n \hat{q}(S_{\tau+n}, A_{\tau+n}, \mathbf{w})$ &ensp; ($G_{\tau:\tau+n}$)
    * $\mathbf{w} \leftarrow \mathbf{w} + \alpha\bigl[G - \hat{q}(S_\tau, A_\tau, \mathbf{w})\bigr] \nabla \hat{q}(S_\tau, A_\tau, \mathbf{w})$
* Until $\tau = T - 1$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Effect of $n$ on the Mountain Car Task)</span></p>

As with tabular methods, performance is generally best if an intermediate level of bootstrapping is used, corresponding to an $n$ larger than 1. On the Mountain Car task, the 8-step semi-gradient Sarsa ($n = 8$) learned faster and obtained a better asymptotic performance than one-step semi-gradient Sarsa ($n = 1$). Empirically, an intermediate level of bootstrapping ($n = 4$) performed best across a range of step sizes $\alpha$.

</div>

### 10.3 Average Reward: A New Problem Setting for Continuing Tasks

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Average Reward Setting)</span></p>

The **average reward** setting is a third classical setting — alongside the episodic and discounted settings — for formulating the goal in Markov decision problems. It applies to continuing problems where the interaction between agent and environment goes on and on forever without termination or start states. Unlike the discounted setting, there is no discounting — the agent cares just as much about delayed rewards as it does about immediate reward.

In the average-reward setting, the quality of a policy $\pi$ is defined as the average rate of reward, or simply **average reward**, denoted $r(\pi)$:

$$r(\pi) \doteq \lim_{h \to \infty} \frac{1}{h} \sum_{t=1}^{h} \mathbb{E}[R_t \mid S_0, A_{0:t-1} \sim \pi] = \sum_s \mu_\pi(s) \sum_a \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a) r,$$

where $\mu_\pi(s) \doteq \lim_{t \to \infty} \Pr\lbrace S_t = s \mid A_{0:t-1} \sim \pi \rbrace$ is the steady-state distribution under $\pi$. The second and third equations hold if the steady-state distribution exists and is independent of $S_0$, i.e., if the MDP is **ergodic**.

We consider all policies that attain the maximal value of $r(\pi)$ to be optimal. This quantity $r(\pi)$ is essentially the **reward rate**.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Differential Return and Differential Value Functions)</span></p>

In the average-reward setting, returns are defined in terms of differences between rewards and the average reward:

$$G_t \doteq R_{t+1} - r(\pi) + R_{t+2} - r(\pi) + R_{t+3} - r(\pi) + \cdots$$

This is known as the **differential return**, and the corresponding value functions are known as **differential value functions**:

$$v_\pi(s) \doteq \mathbb{E}_\pi[G_t \mid S_t = s], \qquad q_\pi(s, a) \doteq \mathbb{E}_\pi[G_t \mid S_t = s, A_t = a].$$

Differential value functions also have Bellman equations, just slightly different from those we have seen earlier. We simply remove all $\gamma$s and replace all rewards by the difference between the reward and the true average reward:

$$v_\pi(s) = \sum_a \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a)\bigl[r - r(\pi) + v_\pi(s')\bigr],$$

$$q_\pi(s, a) = \sum_{s', r} p(s', r \mid s, a)\bigl[r - r(\pi) + \sum_{a'} \pi(a' \mid s') q_\pi(s', a')\bigr].$$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Differential TD Errors)</span></p>

There is also a differential form of the two TD errors:

$$\delta_t \doteq R_{t+1} - \bar{R}_t + \hat{v}(S_{t+1}, \mathbf{w}_t) - \hat{v}(S_t, \mathbf{w}_t),$$

$$\delta_t \doteq R_{t+1} - \bar{R}_t + \hat{q}(S_{t+1}, A_{t+1}, \mathbf{w}_t) - \hat{q}(S_t, A_t, \mathbf{w}_t),$$

where $\bar{R}_t$ is an estimate at time $t$ of the average reward $r(\pi)$. With these alternate definitions, most of our algorithms and many theoretical results carry through to the average-reward setting without change.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Differential Semi-gradient Sarsa for Estimating $\hat{q} \approx q_*$)</span></p>

**Input:** a differentiable action-value function parameterization $\hat{q} : \mathcal{S} \times \mathcal{A} \times \mathbb{R}^d \to \mathbb{R}$

**Algorithm parameters:** step sizes $\alpha, \beta > 0$, small $\varepsilon > 0$

Initialize value-function weights $\mathbf{w} \in \mathbb{R}^d$ arbitrarily (e.g., $\mathbf{w} = \mathbf{0}$)

Initialize average reward estimate $\bar{R} \in \mathbb{R}$ arbitrarily (e.g., $\bar{R} = 0$)

Initialize state $S$, and action $A$

Loop for each step:
* Take action $A$, observe $R, S'$
* Choose $A'$ as a function of $\hat{q}(S', \cdot, \mathbf{w})$ (e.g., $\varepsilon$-greedy)
* $\delta \leftarrow R - \bar{R} + \hat{q}(S', A', \mathbf{w}) - \hat{q}(S, A, \mathbf{w})$
* $\bar{R} \leftarrow \bar{R} + \beta \delta$
* $\mathbf{w} \leftarrow \mathbf{w} + \alpha \delta \nabla \hat{q}(S, A, \mathbf{w})$
* $S \leftarrow S'$; $A \leftarrow A'$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Convergence of Differential Semi-gradient Sarsa)</span></p>

One limitation of this algorithm is that it does not converge to the differential values but to the differential values plus an arbitrary offset. This is because the Bellman equations and TD errors are unaffected if all the values are shifted by the same amount. Thus, the offset may not matter in practice.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Access-Control Queuing Task)</span></p>

This is a decision task involving access control to a set of 10 servers. Customers of four different priorities arrive at a single queue. If given access to a server, the customers pay a reward of 1, 2, 4, or 8, depending on their priority. In each time step, the customer at the head of the queue is either accepted (assigned to one of the servers) or rejected (removed with reward zero). The queue never empties. Each busy server becomes free with probability $p = 0.06$ on each time step.

Using a tabular version of differential semi-gradient Sarsa with parameters $\alpha = 0.01$, $\beta = 0.01$, and $\varepsilon = 0.1$, the algorithm learned to reject low-priority customers when few servers are free and accept high-priority customers. The value learned for $\bar{R}$ was about 2.31 after 2 million steps.

</div>

### 10.4 Deprecating the Discounted Setting

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(The Futility of Discounting in Continuing Problems)</span></p>

In the continuing case with function approximation, the discounted problem formulation is questionable. For policy $\pi$, the average of the discounted returns is always $r(\pi)/(1 - \gamma)$, that is, it is essentially the average reward $r(\pi)$. In particular, the *ordering* of all policies in the average discounted return setting would be exactly the same as in the average-reward setting. The discount rate $\gamma$ thus has no effect on the problem formulation and the ranking would be unchanged.

This is proven by considering the discounted objective $J(\pi) = \sum_s \mu_\pi(s) v_\pi^\gamma(s)$, where $v_\pi^\gamma$ is the discounted value function. Through expansion using the Bellman equation and the steady-state distribution property, one obtains:

$$J(\pi) = \frac{1}{1-\gamma} r(\pi).$$

The discount rate $\gamma$ does not influence the ordering! This strongly suggests that discounting has no role to play in the definition of the control problem with function approximation.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Loss of the Policy Improvement Theorem)</span></p>

The root cause of the difficulties with the discounted control setting is that with function approximation we have lost the **policy improvement theorem** (Section 4.2). It is no longer true that if we change the policy to improve the discounted value of one state then we are guaranteed to have improved the overall policy in any useful sense. That guarantee was key to the theory of our reinforcement learning control methods. With function approximation we have lost it!

In Chapter 13, an alternative class of reinforcement learning algorithms based on parameterized policies is introduced, and there we have a theoretical guarantee called the "policy-gradient theorem" which plays a similar role.

</div>

### 10.5 Differential Semi-gradient $n$-step Sarsa

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Differential $n$-step Return)</span></p>

In order to generalize to $n$-step bootstrapping, we need an $n$-step version of the TD error. We begin by generalizing the $n$-step return to its differential form, with function approximation:

$$G_{t:t+n} \doteq R_{t+1} - \bar{R}_{t+n-1} + \cdots + R_{t+n} - \bar{R}_{t+n-1} + \hat{q}(S_{t+n}, A_{t+n}, \mathbf{w}_{t+n-1}), \quad t + n < T,$$

where $\bar{R}$ is an estimate of $r(\pi)$, $n \geq 1$, and $t + n < T$. If $t + n \geq T$, then we define $G_{t:t+n} \doteq G_t$ as usual. The $n$-step TD error is then:

$$\delta_t \doteq G_{t:t+n} - \hat{q}(S_t, A_t, \mathbf{w}),$$

after which we can apply our usual semi-gradient Sarsa update.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Differential Semi-gradient $n$-step Sarsa for Estimating $\hat{q} \approx q_\pi$ or $q_*$)</span></p>

**Input:** a differentiable function $\hat{q} : \mathcal{S} \times \mathcal{A} \times \mathbb{R}^d \to \mathbb{R}$, a policy $\pi$

Initialize value-function weights $\mathbf{w} \in \mathbb{R}^d$ arbitrarily (e.g., $\mathbf{w} = \mathbf{0}$)

Initialize average-reward estimate $\bar{R} \in \mathbb{R}$ arbitrarily (e.g., $\bar{R} = 0$)

**Algorithm parameters:** step sizes $\alpha, \beta > 0$, small $\varepsilon > 0$, a positive integer $n$

All store and access operations ($S_t$, $A_t$, and $R_t$) can take their index mod $n + 1$

Initialize and store $S_0$ and $A_0$

Loop for each step, $t = 0, 1, 2, \dots$:
* Take action $A_t$
* Observe the next reward as $R_{t+1}$ and the next state as $S_{t+1}$
* Select and store an action $A_{t+1} \sim \pi(\cdot \mid S_{t+1})$, or $\varepsilon$-greedy wrt $\hat{q}(S_{t+1}, \cdot, \mathbf{w})$
* $\tau \leftarrow t - n + 1$ &ensp; ($\tau$ is the time whose estimate is being updated)
* If $\tau \geq 0$:
  * $\delta \leftarrow \sum_{i=\tau+1}^{\tau+n} (R_i - \bar{R}) + \hat{q}(S_{\tau+n}, A_{\tau+n}, \mathbf{w}) - \hat{q}(S_\tau, A_\tau, \mathbf{w})$
  * $\bar{R} \leftarrow \bar{R} + \beta \delta$
  * $\mathbf{w} \leftarrow \mathbf{w} + \alpha \delta \nabla \hat{q}(S_\tau, A_\tau, \mathbf{w})$

</div>

### 10.6 Summary

| Method | Setting | Update Target | Key Feature |
| --- | --- | --- | --- |
| Episodic semi-gradient Sarsa | Episodic | $R + \gamma \hat{q}(S', A', \mathbf{w})$ | Direct extension of tabular Sarsa |
| Semi-gradient $n$-step Sarsa | Episodic | $G_{t:t+n}$ (discounted) | Interpolates between 1-step and MC |
| Differential semi-gradient Sarsa | Continuing (avg. reward) | $R - \bar{R} + \hat{q}(S', A', \mathbf{w})$ | Uses differential TD error |
| Differential $n$-step Sarsa | Continuing (avg. reward) | $G_{t:t+n}$ (differential) | $n$-step version of above |

Key takeaways from Chapter 10:

* The extension of parameterized function approximation and semi-gradient descent to control is immediate for the **episodic case**: we simply switch from state values to action values and use $\varepsilon$-greedy action selection for policy improvement.
* **Semi-gradient $n$-step Sarsa** with function approximation generalizes the tabular $n$-step Sarsa algorithm, and intermediate values of $n$ typically perform best.
* For the **continuing case**, we must introduce a new problem formulation based on maximizing the **average reward** $r(\pi)$ per time step. This involves **differential** versions of value functions, Bellman equations, and TD errors, where all rewards are replaced by their difference from the average reward.
* The **discounted setting is problematic** for continuing tasks with function approximation: the ordering of policies by discounted return is equivalent to ordering by average reward, so $\gamma$ has no effect on the problem. Moreover, the policy improvement theorem is lost with function approximation.
* The average reward formulation involves new differential algorithms, but all conceptual changes are small — we simply remove all $\gamma$s and replace all rewards by $R_t - \bar{R}$.

## Chapter 11: *Off-policy Methods with Approximation

This chapter treats off-policy learning with function approximation. The extension to function approximation is significantly different and harder for off-policy learning than for on-policy learning. The tabular off-policy methods from Chapters 6 and 7 extend to semi-gradient algorithms, but these algorithms do not converge as robustly as they do under on-policy training.

The challenge of off-policy learning has two parts:
1. The **target of the update** — dealt with by importance sampling (Chapters 5 and 7).
2. The **distribution of updates** — in the off-policy case this does not match the on-policy distribution, which is important for the stability of semi-gradient methods.

In off-policy learning we seek to learn a value function for a *target policy* $\pi$, given data due to a different *behavior policy* $b$. In the prediction case, both policies are static and given, and we seek to learn either state values $\hat{v} \approx v_\pi$ or action values $\hat{q} \approx q_\pi$. In the control case, action values are learned, and both policies typically change during learning.

### 11.1 Semi-gradient Methods

Semi-gradient methods address the first part of the challenge of off-policy learning (changing the update targets) but not the second part (changing the update distribution). These methods may diverge in some cases, but they are guaranteed stable and asymptotically unbiased for the tabular case.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Per-step Importance Sampling Ratio)</span></p>

Many off-policy algorithms use the per-step importance sampling ratio:

$$\rho_t \doteq \rho_{t:t} = \frac{\pi(A_t \mid S_t)}{b(A_t \mid S_t)}.$$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Semi-gradient Off-policy TD(0) for State Values)</span></p>

The one-step, state-value algorithm is semi-gradient off-policy TD(0), which is like the on-policy algorithm except for the addition of $\rho_t$:

$$\mathbf{w}_{t+1} \doteq \mathbf{w}_t + \alpha \rho_t \delta_t \nabla \hat{v}(S_t, \mathbf{w}_t),$$

where $\delta_t$ is defined appropriately depending on whether the problem is episodic and discounted, or continuing and undiscounted using average reward:

$$\delta_t \doteq R_{t+1} + \gamma \hat{v}(S_{t+1}, \mathbf{w}_t) - \hat{v}(S_t, \mathbf{w}_t), \quad \text{or}$$

$$\delta_t \doteq R_{t+1} - \bar{R}_t + \hat{v}(S_{t+1}, \mathbf{w}_t) - \hat{v}(S_t, \mathbf{w}_t).$$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Semi-gradient Expected Sarsa for Action Values)</span></p>

For action values, the one-step algorithm is semi-gradient Expected Sarsa:

$$\mathbf{w}_{t+1} \doteq \mathbf{w}_t + \alpha \delta_t \nabla \hat{q}(S_t, A_t, \mathbf{w}_t),$$

with

$$\delta_t \doteq R_{t+1} + \gamma \sum_a \pi(a \mid S_{t+1}) \hat{q}(S_{t+1}, a, \mathbf{w}_t) - \hat{q}(S_t, A_t, \mathbf{w}_t), \quad \text{(episodic)}$$

$$\delta_t \doteq R_{t+1} - \bar{R}_t + \sum_a \pi(a \mid S_{t+1}) \hat{q}(S_{t+1}, a, \mathbf{w}_t) - \hat{q}(S_t, A_t, \mathbf{w}_t). \quad \text{(continuing)}$$

This algorithm does not use importance sampling because the only sample action is $A_t$, and we do not need to consider any other actions.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($n$-step Semi-gradient Sarsa)</span></p>

In the multi-step generalizations, both state-value and action-value algorithms involve importance sampling. The $n$-step version of semi-gradient Sarsa is:

$$\mathbf{w}_{t+n} \doteq \mathbf{w}_{t+n-1} + \alpha \rho_{t+1} \cdots \rho_{t+n}\left[G_{t:t+n} - \hat{q}(S_t, A_t, \mathbf{w}_{t+n-1})\right] \nabla \hat{q}(S_t, A_t, \mathbf{w}_{t+n-1}),$$

with

$$G_{t:t+n} \doteq R_{t+1} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n \hat{q}(S_{t+n}, A_{t+n}, \mathbf{w}_{t+n-1}), \quad \text{(episodic)}$$

$$G_{t:t+n} \doteq R_{t+1} - R_t + \cdots + R_{t+n} - \bar{R}_{t+n-1} + \hat{q}(S_{t+n}, A_{t+n}, \mathbf{w}_{t+n-1}). \quad \text{(continuing)}$$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($n$-step Semi-gradient Tree Backup)</span></p>

The $n$-step tree-backup algorithm does not involve importance sampling at all. Its semi-gradient version is:

$$\mathbf{w}_{t+n} \doteq \mathbf{w}_{t+n-1} + \alpha \left[G_{t:t+n} - \hat{q}(S_t, A_t, \mathbf{w}_{t+n-1})\right] \nabla \hat{q}(S_t, A_t, \mathbf{w}_{t+n-1}),$$

$$G_{t:t+n} \doteq \hat{q}(S_t, A_t, \mathbf{w}_{t+n-1}) + \sum_{k=t}^{t+n-1} \delta_k \prod_{i=t+1}^{k} \gamma \pi(A_i \mid S_i).$$

</div>

### 11.2 Examples of Off-policy Divergence

This section discusses the second part of the challenge: the distribution of updates does not match the on-policy distribution. Semi-gradient and other simple algorithms can be unstable and diverge.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(The $w$-to-$2w$ Divergence)</span></p>

Suppose there are two states whose estimated values are of the functional form $w$ and $2w$, where $\mathbf{w}$ consists of a single component $w$. In the first state, only one action is available, and it results deterministically in a transition to the second state with reward 0.

The TD error on this transition is:

$$\delta_t = R_{t+1} + \gamma \hat{v}(S_{t+1}, \mathbf{w}_t) - \hat{v}(S_t, \mathbf{w}_t) = 0 + \gamma 2w_t - w_t = (2\gamma - 1) w_t,$$

and the off-policy semi-gradient TD(0) update is:

$$w_{t+1} = w_t + \alpha \rho_t \delta_t \nabla \hat{v}(S_t, w_t) = \big(1 + \alpha(2\gamma - 1)\big) w_t.$$

If $1 + \alpha(2\gamma - 1) > 1$, i.e., whenever $\gamma > 0.5$, the system is unstable and $w$ diverges to $\pm \infty$. This instability does not depend on the step size $\alpha > 0$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Baird's Counterexample)</span></p>

An episodic seven-state, two-action MDP. The **dashed** action takes the system to one of the six upper states with equal probability, whereas the **solid** action takes the system to the seventh state. The behavior policy $b$ selects dashed and solid with probabilities $\frac{6}{7}$ and $\frac{1}{7}$. The target policy $\pi$ always takes the solid action. The reward is zero on all transitions, so $v_\pi(s) = 0$ for all $s$, which is exactly representable with $\mathbf{w} = \mathbf{0}$.

If we apply semi-gradient TD(0), the weights diverge to infinity — for any positive step size. This instability also occurs under dynamic programming (expected updates done synchronously).

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Tsitsiklis and Van Roy's Counterexample)</span></p>

Extending the $w$-to-$2w$ example with a terminal state. Even if we change the value function at each step to the best, least-squares approximation (minimizing the $\overline{\text{VE}}$), stability is not guaranteed. The sequence $\lbrace w_k \rbrace$ diverges when $\gamma > \frac{5}{6 - 4\varepsilon}$ and $w_0 \neq 0$:

$$w_{k+1} = \frac{6 - 4\varepsilon}{5} \gamma w_k.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Averagers)</span></p>

Stability is guaranteed for function approximation methods that do not extrapolate from the observed targets. These methods, called **averagers**, include nearest neighbor methods and locally weighted regression, but not popular methods such as tile coding and artificial neural networks.

</div>

### 11.3 The Deadly Triad

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The Deadly Triad)</span></p>

The danger of instability and divergence arises whenever we combine all of the following three elements:

1. **Function approximation** — A powerful, scalable way of generalizing from a state space much larger than the memory and computational resources (e.g., linear function approximation or ANNs).
2. **Bootstrapping** — Update targets that include existing estimates (as in DP or TD methods) rather than relying exclusively on actual rewards and complete returns (as in MC methods).
3. **Off-policy training** — Training on a distribution of transitions other than that produced by the target policy. Sweeping through the state space and updating all states uniformly, as in dynamic programming, does not respect the target policy and is an example of off-policy training.

If any two elements of the deadly triad are present, but not all three, then instability can be avoided.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Can We Give Up Any Element?)</span></p>

* **Function approximation** cannot be given up — we need methods that scale to large problems.
* **Bootstrapping** is possible to give up, but at significant cost to computational and data efficiency. MC methods require storing everything until the final return, and bootstrapping greatly increases learning speed.
* **Off-policy learning** is often adequate with on-policy methods (e.g., Sarsa instead of Q-learning), but off-policy learning is *essential* for learning many value functions in parallel from a single stream of experience, which is important for building powerful intelligent agents.

</div>

### 11.4 Linear Value-function Geometry

To better understand the stability challenge, it is helpful to think about value function approximation geometrically. The space of all possible state-value functions ($v : \mathcal{S} \to \mathbb{R}$) is a $\lvert \mathcal{S} \rvert$-dimensional space. The subspace representable by a linear function approximator with parameter $\mathbf{w}$ is a lower-dimensional subspace.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Value-function Norm)</span></p>

We define the distance between value functions using the norm weighted by a distribution $\mu : \mathcal{S} \to [0,1]$:

$$\lVert v \rVert_\mu^2 \doteq \sum_{s \in \mathcal{S}} \mu(s) v(s)^2.$$

Note that $\overline{\text{VE}}$ from Section 9.2 can be written as $\overline{\text{VE}}(\mathbf{w}) = \lVert v_\mathbf{w} - v_\pi \rVert_\mu^2$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Projection Operator $\Pi$)</span></p>

The projection operator $\Pi$ takes an arbitrary value function to the representable function closest in our norm:

$$\Pi v \doteq v_\mathbf{w} \quad \text{where} \quad \mathbf{w} = \operatorname*{argmin}_{\mathbf{w} \in \mathbb{R}^d} \lVert v - v_\mathbf{w} \rVert_\mu^2.$$

For a linear function approximator, $\Pi$ can be represented as a $\lvert \mathcal{S} \rvert \times \lvert \mathcal{S} \rvert$ matrix:

$$\Pi \doteq \mathbf{X} (\mathbf{X}^\top \mathbf{D} \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{D},$$

where $\mathbf{D}$ is the $\lvert \mathcal{S} \rvert \times \lvert \mathcal{S} \rvert$ diagonal matrix with $\mu(s)$ on the diagonal, and $\mathbf{X}$ is the $\lvert \mathcal{S} \rvert \times d$ matrix whose rows are the feature vectors $\mathbf{x}(s)^\top$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Bellman Error and Bellman Operator)</span></p>

The **Bellman error** at state $s$ measures how far off $v_\mathbf{w}$ is from $v_\pi$, using the Bellman equation:

$$\bar{\delta}_\mathbf{w}(s) \doteq \left(\sum_a \pi(a \mid s) \sum_{s',r} p(s',r \mid s,a) \left[r + \gamma v_\mathbf{w}(s')\right]\right) - v_\mathbf{w}(s) = \mathbb{E}_\pi\left[R_{t+1} + \gamma v_\mathbf{w}(S_{t+1}) - v_\mathbf{w}(S_t) \mid S_t = s, A_t \sim \pi\right].$$

The **Bellman operator** $B_\pi : \mathbb{R}^{\lvert \mathcal{S} \rvert} \to \mathbb{R}^{\lvert \mathcal{S} \rvert}$ is:

$$(B_\pi v)(s) \doteq \sum_a \pi(a \mid s) \sum_{s',r} p(s',r \mid s,a)\left[r + \gamma v(s')\right].$$

The Bellman error vector is $\bar{\delta}_\mathbf{w} = B_\pi v_\mathbf{w} - v_\mathbf{w}$. The true value function $v_\pi$ is the unique fixed point: $v_\pi = B_\pi v_\pi$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Error Objectives)</span></p>

Several error objectives can be defined for measuring the quality of an approximate value function:

* **Mean Square Value Error:** $\overline{\text{VE}}(\mathbf{w}) = \lVert v_\mathbf{w} - v_\pi \rVert_\mu^2$
* **Mean Square Bellman Error:** $\overline{\text{BE}}(\mathbf{w}) = \lVert \bar{\delta}_\mathbf{w} \rVert_\mu^2$
* **Mean Square Projected Bellman Error:** $\overline{\text{PBE}}(\mathbf{w}) = \lVert \Pi \bar{\delta}_\mathbf{w} \rVert_\mu^2$

With linear function approximation, there is always a value function (within the subspace) with zero $\overline{\text{PBE}}$; this is the TD fixed point $\mathbf{w}_\text{TD}$. It is generally different from the value function that minimizes $\overline{\text{VE}}$ or $\overline{\text{BE}}$.

</div>

### 11.5 Gradient Descent in the Bellman Error

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Mean Square TD Error)</span></p>

The one-step TD error with discounting is $\delta_t = R_{t+1} + \gamma \hat{v}(S_{t+1}, \mathbf{w}_t) - \hat{v}(S_t, \mathbf{w}_t)$. A possible objective function is the **mean square TD error**:

$$\overline{\text{TDE}}(\mathbf{w}) = \sum_{s \in \mathcal{S}} \mu(s) \mathbb{E}\left[\delta_t^2 \mid S_t = s, A_t \sim \pi\right] = \mathbb{E}_b\left[\rho_t \delta_t^2\right].$$

Following standard SGD on this objective gives the update:

$$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha \rho_t \delta_t \left(\nabla \hat{v}(S_t, \mathbf{w}_t) - \gamma \nabla \hat{v}(S_{t+1}, \mathbf{w}_t)\right),$$

which is the same as the semi-gradient TD algorithm except for the additional final term $-\gamma \nabla \hat{v}(S_{t+1}, \mathbf{w}_t)$. This makes it a true SGD algorithm. It is called the **naive residual-gradient algorithm**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Problems with the Naive Residual-gradient Algorithm)</span></p>

Although the naive residual-gradient algorithm converges robustly, it does not necessarily converge to a desirable place. The **A-split example** demonstrates this: in a three-state episodic MRP (A splits to B or C with equal probability), the algorithm converges to values of $\frac{3}{4}$ (B) and $\frac{1}{4}$ (C), rather than the true values of $1$ and $0$. Minimizing the $\overline{\text{TDE}}$ achieves temporal smoothing rather than accurate prediction.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Residual-gradient Algorithm)</span></p>

To minimize the mean square Bellman error ($\overline{\text{BE}}$), the update is:

$$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha \left[\mathbb{E}_b\left[\rho_t(R_{t+1} + \gamma \hat{v}(S_{t+1}, \mathbf{w})) - \hat{v}(S_t, \mathbf{w})\right]\right] \left[\nabla \hat{v}(S_t, \mathbf{w}) - \gamma \mathbb{E}_b\left[\rho_t \nabla \hat{v}(S_{t+1}, \mathbf{w})\right]\right].$$

This is the **residual-gradient algorithm**. It requires *two* independent samples of the next state $S_{t+1}$ from $S_t$ to get an unbiased estimate, which is only possible in deterministic environments or simulated environments. As a true SGD method, it is guaranteed to converge to a minimum of the $\overline{\text{BE}}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Shortcomings of the Residual-gradient Method)</span></p>

Three problems with the residual-gradient method:
1. It is empirically much slower than semi-gradient methods.
2. It converges to the wrong values in some cases (the **A-presplit example** shows that even with a deterministic variant, the $\overline{\text{BE}}$ objective itself can be flawed).
3. The $\overline{\text{BE}}$ objective is not learnable from data alone (see next section).

</div>

### 11.6 The Bellman Error is Not Learnable

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Learnability)</span></p>

A hypothesis is said to be **learnable** if it can be determined from an infinite amount of experience (observable data). Many quantities of apparent interest in reinforcement learning cannot be learned even from infinite data — they are well defined and can be computed given knowledge of the internal structure of the environment, but cannot be computed or estimated from the observed sequence of feature vectors, actions, and rewards alone.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The $\overline{\text{VE}}$ is Not Learnable, but Its Optimum Is)</span></p>

Two different MDPs can produce the same data distribution yet have different $\overline{\text{VE}}$ values, proving the $\overline{\text{VE}}$ is not learnable. However, all MDPs with the same data distribution have the same optimal parameter vector $\mathbf{w}^*$. Thus the $\overline{\text{VE}}$ remains a usable objective: it is not learnable, but the parameter that optimizes it is.

This is because the **mean square return error** $\overline{\text{RE}}(\mathbf{w}) = \mathbb{E}\left[(G_t - \hat{v}(S_t, \mathbf{w}))^2\right] = \overline{\text{VE}}(\mathbf{w}) + \mathbb{E}\left[(G_t - v_\pi(S_t))^2\right]$ differs from $\overline{\text{VE}}$ only by a variance term independent of $\mathbf{w}$, so they share the same optimum. The $\overline{\text{RE}}$ is learnable.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The $\overline{\text{BE}}$ is Not Learnable)</span></p>

Unlike $\overline{\text{VE}}$, the $\overline{\text{BE}}$ is not learnable *and* its minimizing parameter vector is also not learnable. Two different MDPs can generate the same data distribution but have different $\overline{\text{BE}}$-minimizing $\mathbf{w}$ values. This means it is impossible in principle to pursue $\overline{\text{BE}}$ as an objective for learning from data alone — knowledge of the underlying MDP beyond the feature vectors is required. This is the strongest reason not to seek the $\overline{\text{BE}}$ objective.

The $\overline{\text{PBE}}$ and $\overline{\text{TDE}}$ objectives and their (different) minima *can* be directly determined from data and thus are learnable.

</div>

### 11.7 Gradient-TD Methods

Gradient-TD methods are true SGD methods for minimizing the $\overline{\text{PBE}}$. They have robust convergence properties even under off-policy training and nonlinear function approximation. They achieve $O(d)$ complexity at the cost of a rough doubling of computational complexity (a second parameter vector).

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($\overline{\text{PBE}}$ in Matrix Form)</span></p>

Assuming linear function approximation, the $\overline{\text{PBE}}$ can be written in matrix terms as:

$$\overline{\text{PBE}}(\mathbf{w}) = \lVert \Pi \bar{\delta}_\mathbf{w} \rVert_\mu^2 = \left(\mathbf{X}^\top \mathbf{D} \bar{\delta}_\mathbf{w}\right)^\top \left(\mathbf{X}^\top \mathbf{D} \mathbf{X}\right)^{-1} \left(\mathbf{X}^\top \mathbf{D} \bar{\delta}_\mathbf{w}\right).$$

The gradient is:

$$\nabla \overline{\text{PBE}}(\mathbf{w}) = 2 \mathbb{E}\left[\rho_t(\gamma \mathbf{x}_{t+1} - \mathbf{x}_t) \mathbf{x}_t^\top\right] \mathbb{E}\left[\mathbf{x}_t \mathbf{x}_t^\top\right]^{-1} \mathbb{E}\left[\rho_t \delta_t \mathbf{x}_t\right].$$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(GTD2 and TDC Algorithms)</span></p>

Gradient-TD methods estimate the product of the second two factors in the $\overline{\text{PBE}}$ gradient as a $d$-vector $\mathbf{v}$:

$$\mathbf{v} \approx \mathbb{E}\left[\mathbf{x}_t \mathbf{x}_t^\top\right]^{-1} \mathbb{E}\left[\rho_t \delta_t \mathbf{x}_t\right],$$

which is learned via the Least Mean Square (LMS) rule:

$$\mathbf{v}_{t+1} \doteq \mathbf{v}_t + \beta \rho_t \left(\delta_t - \mathbf{v}_t^\top \mathbf{x}_t\right) \mathbf{x}_t,$$

where $\beta > 0$ is a second step-size parameter.

**GTD2:** Using $\mathbf{v}_t$ to approximate the gradient gives:

$$\mathbf{w}_{t+1} \approx \mathbf{w}_t + \alpha \rho_t \left(\mathbf{x}_t - \gamma \mathbf{x}_{t+1}\right) \mathbf{x}_t^\top \mathbf{v}_t.$$

**TDC** (TD(0) with gradient correction), also known as **GTD(0):** Derived by doing a few more analytic steps before substituting $\mathbf{v}_t$:

$$\mathbf{w}_{t+1} \approx \mathbf{w}_t + \alpha \rho_t \left(\delta_t \mathbf{x}_t - \gamma \mathbf{x}_{t+1} \mathbf{x}_t^\top \mathbf{v}_t\right).$$

Both algorithms are $O(d)$ in storage and per-step computation. The convergence proofs are *two-time-scale* proofs requiring $\beta \to 0$ and $\frac{\alpha}{\beta} \to 0$ (i.e., $\mathbf{v}$ learns on a faster time scale than $\mathbf{w}$).

</div>

### 11.8 Emphatic-TD Methods

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Emphatic-TD Methods)</span></p>

Emphatic-TD methods are an alternative approach: instead of performing true gradient descent (as in Gradient-TD), they reweight updates using importance sampling to restore the on-policy distribution, making semi-gradient methods stable.

The one-step Emphatic-TD algorithm for learning episodic state values is:

$$\delta_t = R_{t+1} + \gamma \hat{v}(S_{t+1}, \mathbf{w}_t) - \hat{v}(S_t, \mathbf{w}_t),$$

$$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha M_t \rho_t \delta_t \nabla \hat{v}(S_t, \mathbf{w}_t),$$

$$M_t = \gamma \rho_{t-1} M_{t-1} + I_t,$$

with $I_t$ the *interest* (arbitrary), $M_t$ the *emphasis*, and initialization $M_{-1} = 0$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Properties of Emphatic-TD)</span></p>

On Baird's counterexample (with $I_t = 1$ for all $t$), the expected behavior of Emphatic-TD converges — the $\overline{\text{VE}}$ goes to zero. However, the variance is so high that it is nearly impossible to get consistent results in computational experiments. The algorithm converges to the optimal solution in theory, but in practice it does not. This motivates variance reduction methods.

</div>

### 11.9 Reducing Variance

Off-policy learning is inherently of greater variance than on-policy learning because importance sampling involves products of policy ratios that can be very high or zero. High variance is problematic for SGD because it means taking steps that vary greatly in size.

Techniques for reducing variance include:
* **Momentum** and **Polyak-Ruppert averaging** of parameter vectors.
* **Adaptive step sizes** — setting separate step sizes for different components of the parameter vector.
* **Weighted importance sampling** — significantly better behaved than ordinary importance sampling, but adapting it to function approximation is challenging and can probably only be done approximately with $O(d)$ complexity.
* **Tree Backup algorithms** — performing off-policy learning without using importance sampling at all, extended to the off-policy case to produce stable and more efficient methods.
* **Constraining the target policy** — defining the target policy in part by the behavior policy so that it never creates large importance sampling ratios.

### 11.10 Summary

| Method | Objective | Key Property |
| --- | --- | --- |
| Semi-gradient TD (off-policy) | $\overline{\text{PBE}} = 0$ (TD fixed point) | Simple, $O(d)$, may diverge |
| Naive residual-gradient | $\overline{\text{TDE}}$ | Converges robustly, wrong values |
| Residual-gradient | $\overline{\text{BE}}$ | Needs two samples, slow, $\overline{\text{BE}}$ not learnable |
| GTD2 / TDC | $\overline{\text{PBE}}$ | True SGD, $O(d)$, two time scales |
| Emphatic-TD | Restores on-policy distribution | Simple semi-gradient, high variance |

Key takeaways from Chapter 11:

* The challenge of off-policy learning is divided into two parts: correcting update *targets* (addressed by importance sampling) and correcting the *distribution* of updates (much harder with function approximation).
* Semi-gradient methods extend easily from on-policy to off-policy, but may diverge due to the **deadly triad**: function approximation + bootstrapping + off-policy training.
* The **geometry of linear value-function approximation** provides insight through three error objectives: $\overline{\text{VE}}$, $\overline{\text{BE}}$, and $\overline{\text{PBE}}$, each with different properties and minimizers.
* Minimizing the $\overline{\text{TDE}}$ (naive residual-gradient) converges but to wrong values. Minimizing the $\overline{\text{BE}}$ (residual-gradient) is problematic because the $\overline{\text{BE}}$ is **not learnable** from data alone.
* **Gradient-TD methods** (GTD2, TDC) perform true SGD on the $\overline{\text{PBE}}$, which is learnable, achieving $O(d)$ complexity with robust convergence guarantees.
* **Emphatic-TD methods** reweight updates to restore the on-policy distribution, making semi-gradient methods stable but at the cost of high variance.

## Chapter 12: Eligibility Traces

Eligibility traces are one of the basic mechanisms of reinforcement learning. When TD methods are augmented with eligibility traces, they produce a family of methods spanning a spectrum that has Monte Carlo methods at one end ($\lambda = 1$) and one-step TD methods at the other ($\lambda = 0$). In between are intermediate methods that are often better than either extreme.

The mechanism is a short-term memory vector, the **eligibility trace** $\mathbf{z}_t \in \mathbb{R}^d$, that parallels the long-term weight vector $\mathbf{w}_t \in \mathbb{R}^d$. When a component of $\mathbf{w}_t$ participates in producing an estimated value, the corresponding component of $\mathbf{z}_t$ is bumped up and then begins to fade away. Learning occurs in that component of $\mathbf{w}_t$ if a nonzero TD error occurs before the trace falls back to zero. The trace-decay parameter $\lambda \in [0, 1]$ determines the rate at which the trace falls.

There are two views for formulating and implementing learning algorithms:
* **Forward views** — look forward in time to all future rewards and decide how best to combine them (e.g., $\lambda$-return). Conceptually clear but complex to implement since future information is needed.
* **Backward views** — use the current TD error and look backward to recently visited states using an eligibility trace. Computationally congenial and incremental.

### 12.1 The $\lambda$-return

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($n$-step Return)</span></p>

The general form of the $n$-step return, for any parameterized function approximator, is:

$$G_{t:t+n} \doteq R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n \hat{v}(S_{t+n}, \mathbf{w}_{t+n-1}), \quad 0 \le t \le T - n.$$

Any valid update can be done toward any *average* of $n$-step returns for different $n$s, as long as the weights are positive and sum to 1. An update that averages simpler component updates is called a **compound update**.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($\lambda$-return)</span></p>

The TD($\lambda$) algorithm can be understood as one particular way of averaging all the $n$-step updates, each weighted proportionally to $\lambda^{n-1}$ (where $\lambda \in [0, 1)$), and normalized by a factor of $1 - \lambda$ to ensure that the weights sum to 1. The resulting update target is the **$\lambda$-return**:

$$G_t^\lambda \doteq (1 - \lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_{t:t+n}.$$

Separating post-termination terms from the main sum for the episodic case:

$$G_t^\lambda = (1 - \lambda) \sum_{n=1}^{T-t-1} \lambda^{n-1} G_{t:t+n} + \lambda^{T-t-1} G_t.$$

* For $\lambda = 1$: reduces to the conventional return $G_t$ (Monte Carlo).
* For $\lambda = 0$: reduces to $G_{t:t+1}$, the one-step return (one-step TD).

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Off-line $\lambda$-return Algorithm)</span></p>

The off-line $\lambda$-return algorithm makes no changes to the weight vector during the episode. At the end, a whole sequence of off-line updates is made using the $\lambda$-return as the target:

$$\mathbf{w}_{t+1} \doteq \mathbf{w}_t + \alpha \left[G_t^\lambda - \hat{v}(S_t, \mathbf{w}_t)\right] \nabla \hat{v}(S_t, \mathbf{w}_t), \quad t = 0, \ldots, T-1.$$

</div>

### 12.2 TD($\lambda$)

TD($\lambda$) improves over the off-line $\lambda$-return algorithm in three ways: (1) it updates on every step rather than only at the end, (2) its computations are equally distributed in time, and (3) it can be applied to continuing problems.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Eligibility Trace for TD($\lambda$))</span></p>

The eligibility trace is initialized to zero at the beginning of each episode and is incremented on each time step by the value gradient, then fades away by $\gamma \lambda$:

$$\mathbf{z}_{-1} \doteq \mathbf{0},$$

$$\mathbf{z}_t \doteq \gamma \lambda \mathbf{z}_{t-1} + \nabla \hat{v}(S_t, \mathbf{w}_t), \quad 0 \le t \le T.$$

This is called an **accumulating trace**. In linear function approximation, $\nabla \hat{v}(S_t, \mathbf{w}_t) = \mathbf{x}_t$, so the trace is just a sum of past, fading input vectors.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(TD($\lambda$) Update)</span></p>

The TD error is:

$$\delta_t \doteq R_{t+1} + \gamma \hat{v}(S_{t+1}, \mathbf{w}_t) - \hat{v}(S_t, \mathbf{w}_t).$$

The weight vector is updated on each step proportional to the scalar TD error and the vector eligibility trace:

$$\mathbf{w}_{t+1} \doteq \mathbf{w}_t + \alpha \delta_t \mathbf{z}_t.$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Semi-gradient TD($\lambda$) for estimating $\hat{v} \approx v_\pi$)</span></p>

**Input:** policy $\pi$, differentiable function $\hat{v} : \mathcal{S}^+ \times \mathbb{R}^d \to \mathbb{R}$ with $\hat{v}(\text{terminal}, \cdot) = 0$
**Parameters:** step size $\alpha > 0$, trace decay rate $\lambda \in [0, 1]$
**Initialize:** $\mathbf{w}$ arbitrarily

Loop for each episode:
* Initialize $S$
* $\mathbf{z} \leftarrow \mathbf{0}$
* Loop for each step of episode:
  * Choose $A \sim \pi(\cdot \mid S)$
  * Take action $A$, observe $R, S'$
  * $\mathbf{z} \leftarrow \gamma \lambda \mathbf{z} + \nabla \hat{v}(S, \mathbf{w})$
  * $\delta \leftarrow R + \gamma \hat{v}(S', \mathbf{w}) - \hat{v}(S, \mathbf{w})$
  * $\mathbf{w} \leftarrow \mathbf{w} + \alpha \delta \mathbf{z}$
  * $S \leftarrow S'$
* until $S'$ is terminal

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Special Cases of TD($\lambda$))</span></p>

* **$\lambda = 0$ (TD(0)):** The trace at time $t$ is exactly the value gradient for $S_t$. Only the current state is updated by the TD error. This is one-step semi-gradient TD.
* **$\lambda = 1$ (TD(1)):** The credit given to earlier states falls only by $\gamma$ per step. With $\gamma = 1$, the traces do not decay at all, and the method behaves like a Monte Carlo method. TD(1) is a more general way to implement MC that works online, incrementally, and on continuing tasks.
* **$0 < \lambda < 1$:** Each more temporally distant state is updated less because its eligibility trace is smaller, giving less *credit* for the TD error.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Convergence of Linear TD($\lambda$))</span></p>

Linear TD($\lambda$) has been proved to converge in the on-policy case if the step-size parameter is reduced over time according to the usual conditions. Convergence is not to the minimum-error weight vector, but to a nearby weight vector that depends on $\lambda$. For the continuing discounted case, the bound on solution quality is:

$$\overline{\text{VE}}(\mathbf{w}_\infty) \le \frac{1 - \gamma \lambda}{1 - \gamma} \min_\mathbf{w} \overline{\text{VE}}(\mathbf{w}).$$

As $\lambda$ approaches 1, the bound approaches the minimum error. However, in practice $\lambda = 1$ is often the poorest choice.

</div>

### 12.3 $n$-step Truncated $\lambda$-return Methods

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Truncated $\lambda$-return)</span></p>

The off-line $\lambda$-return uses $G_t^\lambda$ which depends on $n$-step returns for arbitrarily large $n$. A natural approximation is to truncate the sequence after some number of steps. The **truncated $\lambda$-return** for time $t$, given data only up to some later horizon $h$, is:

$$G_{t:h}^\lambda \doteq (1 - \lambda) \sum_{n=1}^{h-t-1} \lambda^{n-1} G_{t:t+n} + \lambda^{h-t-1} G_{t:h}, \quad 0 \le t < h \le T.$$

The horizon $h$ plays the role that $T$ (the time of termination) played in the $\lambda$-return.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Truncated TD($\lambda$), or TTD($\lambda$))</span></p>

The truncated $\lambda$-return immediately gives rise to a family of $n$-step $\lambda$-return algorithms. Updates are delayed by $n$ steps and take into account the first $n$ rewards, but now all the $k$-step returns for $1 \le k \le n$ are included, weighted geometrically. The update is:

$$\mathbf{w}_{t+n} \doteq \mathbf{w}_{t+n-1} + \alpha \left[G_{t:t+n}^\lambda - \hat{v}(S_t, \mathbf{w}_{t+n-1})\right] \nabla \hat{v}(S_t, \mathbf{w}_{t+n-1}), \quad 0 \le t < T.$$

The $k$-step $\lambda$-return can be efficiently computed as:

$$G_{t:t+k}^\lambda = \hat{v}(S_t, \mathbf{w}_{t-1}) + \sum_{i=t}^{t+k-1} (\gamma \lambda)^{i-t} \delta_i',$$

where $\delta_i' \doteq R_{i+1} + \gamma \hat{v}(S_{i+1}, \mathbf{w}_i) - \hat{v}(S_i, \mathbf{w}_{i-1})$.

</div>

### 12.4 Redoing Updates: Online $\lambda$-return Algorithm

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Online $\lambda$-return Algorithm)</span></p>

The idea is that on each time step, as you gather a new increment of data, you go back and redo all the updates since the beginning of the current episode. Each pass uses a slightly longer horizon, giving slightly better results.

Let $\mathbf{w}_t^h$ denote the weights used to generate the value at time $t$ in the sequence up to horizon $h$. The general form of the update is:

$$\mathbf{w}_{t+1}^h \doteq \mathbf{w}_t^h + \alpha \left[G_{t:h}^\lambda - \hat{v}(S_t, \mathbf{w}_t^h)\right] \nabla \hat{v}(S_t, \mathbf{w}_t^h), \quad 0 \le t < h \le T.$$

Together with $\mathbf{w}_t \doteq \mathbf{w}_t^t$, this defines the **online $\lambda$-return algorithm**. It is fully online, determining a new weight vector $\mathbf{w}_t$ at each step $t$ using only information available at time $t$. Its main drawback is high computational complexity — it passes over the entire episode so far on every step.

</div>

### 12.5 True Online TD($\lambda$)

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(True Online TD($\lambda$))</span></p>

True online TD($\lambda$) is an exact, computationally congenial backward-view implementation of the online $\lambda$-return algorithm for the case of linear function approximation ($\hat{v}(s, \mathbf{w}) = \mathbf{w}^\top \mathbf{x}(s)$). It produces exactly the same sequence of weight vectors as the online $\lambda$-return algorithm, but with $O(d)$ per-step complexity.

The update rules are:

$$\mathbf{w}_{t+1} \doteq \mathbf{w}_t + \alpha \delta_t \mathbf{z}_t + \alpha \left(\mathbf{w}_t^\top \mathbf{x}_t - \mathbf{w}_{t-1}^\top \mathbf{x}_t\right)(\mathbf{z}_t - \mathbf{x}_t),$$

where $\delta_t$ is the usual TD error, and $\mathbf{z}_t$ is the **dutch trace**:

$$\mathbf{z}_t \doteq \gamma \lambda \mathbf{z}_{t-1} + \left(1 - \alpha \gamma \lambda \mathbf{z}_{t-1}^\top \mathbf{x}_t\right) \mathbf{x}_t.$$

The memory requirements are identical to those of conventional TD($\lambda$), while per-step computation is increased by about 50% (one more inner product in the trace update). Overall complexity remains $O(d)$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(True Online TD($\lambda$) for estimating $\mathbf{w}^\top \mathbf{x} \approx v_\pi$)</span></p>

**Input:** policy $\pi$, feature function $\mathbf{x} : \mathcal{S}^+ \to \mathbb{R}^d$ with $\mathbf{x}(\text{terminal}, \cdot) = \mathbf{0}$
**Parameters:** step size $\alpha > 0$, trace decay rate $\lambda \in [0, 1]$
**Initialize:** $\mathbf{w} \in \mathbb{R}^d$

Loop for each episode:
* Initialize state and obtain initial feature vector $\mathbf{x}$
* $\mathbf{z} \leftarrow \mathbf{0}$, $V_{old} \leftarrow 0$
* Loop for each step of episode:
  * Choose $A \sim \pi$
  * Take action $A$, observe $R$, $\mathbf{x}'$ (feature vector of next state)
  * $V \leftarrow \mathbf{w}^\top \mathbf{x}$, $V' \leftarrow \mathbf{w}^\top \mathbf{x}'$
  * $\delta \leftarrow R + \gamma V' - V$
  * $\mathbf{z} \leftarrow \gamma \lambda \mathbf{z} + (1 - \alpha \gamma \lambda \mathbf{z}^\top \mathbf{x})\, \mathbf{x}$
  * $\mathbf{w} \leftarrow \mathbf{w} + \alpha(\delta + V - V_{old})\mathbf{z} - \alpha(V - V_{old})\mathbf{x}$
  * $V_{old} \leftarrow V'$
  * $\mathbf{x} \leftarrow \mathbf{x}'$
* until $\mathbf{x}' = \mathbf{0}$ (terminal)

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Types of Eligibility Traces)</span></p>

Three kinds of eligibility traces have been used:

* **Accumulating trace:** $\mathbf{z}_t \doteq \gamma \lambda \mathbf{z}_{t-1} + \nabla \hat{v}(S_t, \mathbf{w}_t)$. Used in conventional TD($\lambda$).
* **Dutch trace:** $\mathbf{z}_t \doteq \gamma \lambda \mathbf{z}_{t-1} + (1 - \alpha \gamma \lambda \mathbf{z}_{t-1}^\top \mathbf{x}_t)\, \mathbf{x}_t$. Used in true online TD($\lambda$). Usually performs better and has a clearer theoretical basis.
* **Replacing trace:** $z_{i,t} \doteq 1$ if $x_{i,t} = 1$, else $\gamma \lambda z_{i,t-1}$. Defined only for binary features. Nowadays seen as a crude approximation to dutch traces, which largely supersede them.

</div>

### 12.6 *Dutch Traces in Monte Carlo Learning

Eligibility traces arise even in Monte Carlo learning (without TD). The linear gradient MC algorithm (from Chapter 9) makes updates at the end of the episode:

$$\mathbf{w}_{T} = \mathbf{w}_{T-1} + \alpha\left(G - \mathbf{w}_{T-1}^\top \mathbf{x}_{T-1}\right) \mathbf{x}_{T-1}.$$

This can be rewritten using a **forgetting matrix** $\mathbf{F}_t \doteq \mathbf{I} - \alpha \mathbf{x}_t \mathbf{x}_t^\top$ and two auxiliary vectors $\mathbf{a}_t$ and $\mathbf{z}_t$, updated incrementally with $O(d)$ per-step complexity:

$$\mathbf{w}_T = \mathbf{a}_{T-1} + \alpha G \mathbf{z}_{T-1},$$

where $\mathbf{z}_t$ is a dutch-style eligibility trace (for the case $\gamma \lambda = 1$) and $\mathbf{a}_t$ tracks the accumulated forgetting effects. This shows that eligibility traces are not specific to TD learning — they arise whenever one tries to learn long-term predictions efficiently.

### 12.7 Sarsa($\lambda$)

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Sarsa($\lambda$))</span></p>

To extend eligibility traces to action values, we use approximate action values $\hat{q}(s, a, \mathbf{w})$ and the action-value form of the $n$-step return:

$$G_{t:t+n} \doteq R_{t+1} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n \hat{q}(S_{t+n}, A_{t+n}, \mathbf{w}_{t+n-1}), \quad t + n < T.$$

The $\lambda$-return for action values simply uses $\hat{q}$ rather than $\hat{v}$:

$$\mathbf{w}_{t+1} \doteq \mathbf{w}_t + \alpha \left[G_t^\lambda - \hat{q}(S_t, A_t, \mathbf{w}_t)\right] \nabla \hat{q}(S_t, A_t, \mathbf{w}_t), \quad t = 0, \ldots, T-1.$$

The temporal-difference method for action values, known as **Sarsa($\lambda$)**, approximates this forward view with the same update rule as TD($\lambda$):

$$\mathbf{w}_{t+1} \doteq \mathbf{w}_t + \alpha \delta_t \mathbf{z}_t,$$

with the action-value form of the TD error:

$$\delta_t \doteq R_{t+1} + \gamma \hat{q}(S_{t+1}, A_{t+1}, \mathbf{w}_t) - \hat{q}(S_t, A_t, \mathbf{w}_t),$$

and the action-value form of the eligibility trace:

$$\mathbf{z}_{-1} \doteq \mathbf{0}, \quad \mathbf{z}_t \doteq \gamma \lambda \mathbf{z}_{t-1} + \nabla \hat{q}(S_t, A_t, \mathbf{w}_t), \quad 0 \le t \le T.$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Sarsa($\lambda$) with binary features and linear function approximation)</span></p>

**Input:** feature function $\mathcal{F}(s, a)$ returning active feature indices; policy $\pi$
**Parameters:** step size $\alpha > 0$, trace decay rate $\lambda \in [0, 1]$, small $\varepsilon > 0$
**Initialize:** $\mathbf{w} = \mathbf{0}$, $\mathbf{z} = \mathbf{0}$

Loop for each episode:
* Initialize $S$; Choose $A \sim \pi(\cdot \mid S)$ or $\varepsilon$-greedy wrt $\hat{q}(S, \cdot, \mathbf{w})$
* $\mathbf{z} \leftarrow \mathbf{0}$
* Loop for each step of episode:
  * Take action $A$, observe $R, S'$
  * $\delta \leftarrow R$
  * Loop for $i$ in $\mathcal{F}(S, A)$: $\delta \leftarrow \delta - w_i$; $z_i \leftarrow z_i + 1$ (accumulating) or $z_i \leftarrow 1$ (replacing)
  * If $S'$ is terminal: $\mathbf{w} \leftarrow \mathbf{w} + \alpha \delta \mathbf{z}$; go to next episode
  * Choose $A' \sim \pi(\cdot \mid S')$ or $\varepsilon$-greedy wrt $\hat{q}(S', \cdot, \mathbf{w})$
  * Loop for $i$ in $\mathcal{F}(S', A')$: $\delta \leftarrow \delta + \gamma w_i$
  * $\mathbf{w} \leftarrow \mathbf{w} + \alpha \delta \mathbf{z}$
  * $\mathbf{z} \leftarrow \gamma \lambda \mathbf{z}$
  * $S \leftarrow S'$; $A \leftarrow A'$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(True Online Sarsa($\lambda$))</span></p>

**Input:** feature function $\mathbf{x} : \mathcal{S}^+ \times \mathcal{A} \to \mathbb{R}^d$ with $\mathbf{x}(\text{terminal}, \cdot) = \mathbf{0}$; policy $\pi$ (if estimating $q_\pi$)
**Parameters:** step size $\alpha > 0$, trace decay rate $\lambda \in [0, 1]$, small $\varepsilon > 0$
**Initialize:** $\mathbf{w} \in \mathbb{R}^d$

Loop for each episode:
* Initialize $S$; Choose $A \sim \pi(\cdot \mid S)$ or $\varepsilon$-greedy wrt $\hat{q}(S, \cdot, \mathbf{w})$
* $\mathbf{x} \leftarrow \mathbf{x}(S, A)$; $\mathbf{z} \leftarrow \mathbf{0}$; $Q_{old} \leftarrow 0$
* Loop for each step of episode:
  * Take action $A$, observe $R, S'$
  * Choose $A' \sim \pi(\cdot \mid S')$ or $\varepsilon$-greedy wrt $\hat{q}(S', \cdot, \mathbf{w})$
  * $\mathbf{x}' \leftarrow \mathbf{x}(S', A')$; $Q \leftarrow \mathbf{w}^\top \mathbf{x}$; $Q' \leftarrow \mathbf{w}^\top \mathbf{x}'$
  * $\delta \leftarrow R + \gamma Q' - Q$
  * $\mathbf{z} \leftarrow \gamma \lambda \mathbf{z} + (1 - \alpha \gamma \lambda \mathbf{z}^\top \mathbf{x})\, \mathbf{x}$
  * $\mathbf{w} \leftarrow \mathbf{w} + \alpha(\delta + Q - Q_{old})\mathbf{z} - \alpha(Q - Q_{old})\mathbf{x}$
  * $Q_{old} \leftarrow Q'$
  * $\mathbf{x} \leftarrow \mathbf{x}'$; $A \leftarrow A'$
* until $S'$ is terminal

</div>

### 12.8 Variable $\lambda$ and $\gamma$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(State-dependent Bootstrapping and Discounting)</span></p>

In the most general forms of TD learning algorithms, $\lambda$ and $\gamma$ are generalized beyond constant parameters to functions potentially dependent on the state and action. That is, each time step can have a different $\lambda_t$ and $\gamma_t$:

$$\lambda_t \doteq \lambda(S_t, A_t), \quad \gamma_t \doteq \gamma(S_t).$$

Introducing variable $\gamma$, the **termination function**, changes the return:

$$G_t \doteq R_{t+1} + \gamma_{t+1} G_{t+1} = \sum_{k=t}^{\infty} \left(\prod_{i=t+1}^{k} \gamma_i\right) R_{k+1}.$$

Setting $\gamma(s) = 0$ for terminal states and $\gamma(s) = \gamma$ elsewhere recovers the classical episodic setting. State-dependent termination unifies the episodic and discounted-continuing cases through the notion of **pseudo termination**.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(General $\lambda$-returns)</span></p>

The new state-based $\lambda$-return can be written recursively as:

$$G_t^{\lambda s} \doteq R_{t+1} + \gamma_{t+1}\left((1 - \lambda_{t+1})\hat{v}(S_{t+1}, \mathbf{w}_t) + \lambda_{t+1} G_{t+1}^{\lambda s}\right).$$

The action-value forms are:

* **Sarsa form:** $G_t^{\lambda a} \doteq R_{t+1} + \gamma_{t+1}\left((1 - \lambda_{t+1})\hat{q}(S_{t+1}, A_{t+1}, \mathbf{w}_t) + \lambda_{t+1} G_{t+1}^{\lambda a}\right).$
* **Expected Sarsa form:** $G_t^{\lambda a} \doteq R_{t+1} + \gamma_{t+1}\left((1 - \lambda_{t+1})\bar{V}_t(S_{t+1}) + \lambda_{t+1} G_{t+1}^{\lambda a}\right),$

where $\bar{V}_t(s) \doteq \sum_a \pi(a \mid s) \hat{q}(s, a, \mathbf{w}_t)$.

</div>

### 12.9 Off-policy Traces with Control Variates

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Off-policy $\lambda$-return with Control Variates)</span></p>

For off-policy learning, the state-value $\lambda$-return with per-decision importance sampling and control variates is:

$$G_t^{\lambda s} \doteq \rho_t \left(R_{t+1} + \gamma_{t+1}\left((1 - \lambda_{t+1})\hat{v}(S_{t+1}, \mathbf{w}_t) + \lambda_{t+1} G_{t+1}^{\lambda s}\right)\right) + (1 - \rho_t)\hat{v}(S_t, \mathbf{w}_t).$$

This can be approximated as a sum of TD errors:

$$G_t^{\lambda s} \approx \hat{v}(S_t, \mathbf{w}_t) + \rho_t \sum_{k=t}^{\infty} \delta_k^s \prod_{i=t+1}^{k} \gamma_i \lambda_i \rho_i.$$

The general accumulating eligibility trace for state values becomes:

$$\mathbf{z}_t \doteq \rho_t \left(\gamma_t \lambda_t \mathbf{z}_{t-1} + \nabla \hat{v}(S_t, \mathbf{w}_t)\right).$$

This trace, together with the usual semi-gradient parameter-update rule $\mathbf{w}_{t+1} \doteq \mathbf{w}_t + \alpha \delta_t^s \mathbf{z}_t$, forms a general TD($\lambda$) algorithm applicable to both on-policy and off-policy data. On-policy, it reduces exactly to TD($\lambda$) since $\rho_t = 1$ always.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Off-policy Action-value Traces)</span></p>

For action values with the Expected Sarsa form, the off-policy $\lambda$-return is:

$$G_t^{\lambda a} \doteq R_{t+1} + \gamma_{t+1}\left(\bar{V}_t(S_{t+1}) + \lambda_{t+1} \rho_{t+1}\left[G_{t+1}^{\lambda a} - \hat{q}(S_{t+1}, A_{t+1}, \mathbf{w}_t)\right]\right),$$

with the action-based TD error:

$$\delta_t^a = R_{t+1} + \gamma_{t+1} \bar{V}_t(S_{t+1}) - \hat{q}(S_t, A_t, \mathbf{w}_t),$$

and the eligibility trace for action values:

$$\mathbf{z}_t \doteq \gamma_t \lambda_t \rho_t \mathbf{z}_{t-1} + \nabla \hat{q}(S_t, A_t, \mathbf{w}_t).$$

This trace with the update rule $\mathbf{w}_{t+1} \doteq \mathbf{w}_t + \alpha \delta_t^a \mathbf{z}_t$ forms a general Expected Sarsa($\lambda$) algorithm.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Off-policy Stability)</span></p>

If $\lambda < 1$, all these off-policy algorithms involve bootstrapping and the deadly triad applies (Section 11.3). They can be guaranteed stable only for the tabular case, state aggregation, and other limited forms of function approximation. For linear and more-general function approximation, the parameter vector may diverge to infinity. Off-policy eligibility traces deal effectively with the first part of the off-policy challenge (correcting targets) but not the second (correcting the distribution of updates). Algorithmic strategies for addressing the second part are given in Section 12.11.

</div>

### 12.10 Watkins's Q($\lambda$) to Tree-Backup($\lambda$)

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Watkins's Q($\lambda$))</span></p>

**Watkins's Q($\lambda$)** decays its eligibility traces in the usual way as long as a greedy action is taken, then cuts the traces to zero after the first non-greedy action. This was one of the first proposals for extending Q-learning to eligibility traces.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Tree-Backup($\lambda$), TB($\lambda$))</span></p>

Tree-Backup($\lambda$) is the eligibility trace version of the $n$-step Tree Backup algorithm. It does not use importance sampling, making it the true successor to Q-learning. The $\lambda$-return for TB($\lambda$) uses action values with the Expected Sarsa form, expanding the bootstrapping case after the model of Section 7.16:

$$G_t^{\lambda a} \doteq R_{t+1} + \gamma_{t+1}\left(\bar{V}_t(S_{t+1}) + \lambda_{t+1} \pi(A_{t+1} \mid S_{t+1})\left(G_{t+1}^{\lambda a} - \hat{q}(S_{t+1}, A_{t+1}, \mathbf{w}_t)\right)\right).$$

The resulting eligibility trace involves the target-policy probabilities of the selected actions:

$$\mathbf{z}_t \doteq \gamma_t \lambda_t \pi(A_t \mid S_t) \mathbf{z}_{t-1} + \nabla \hat{q}(S_t, A_t, \mathbf{w}_t).$$

This, together with the usual parameter-update rule, defines the TB($\lambda$) algorithm. Like all semi-gradient algorithms, TB($\lambda$) is not guaranteed to be stable with off-policy data and powerful function approximation.

</div>

### 12.11 Stable Off-policy Methods with Traces

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(GTD($\lambda$) and GQ($\lambda$))</span></p>

**GTD($\lambda$)** is the eligibility-trace algorithm analogous to TDC (the better of the two Gradient-TD prediction algorithms from Section 11.7). It learns state values from off-policy data with guaranteed stability:

$$\mathbf{w}_{t+1} \doteq \mathbf{w}_t + \alpha \delta_t^s \mathbf{z}_t - \alpha \gamma_{t+1}(1 - \lambda_{t+1})\left(\mathbf{z}_t^\top \mathbf{v}_t\right) \mathbf{x}_{t+1},$$

$$\mathbf{v}_{t+1} \doteq \mathbf{v}_t + \beta \delta_t^s \mathbf{z}_t - \beta \left(\mathbf{v}_t^\top \mathbf{x}_t\right) \mathbf{x}_t,$$

with $\delta_t^s$, $\mathbf{z}_t$, and $\rho_t$ defined in the usual ways, and $\beta > 0$ is a second step-size parameter.

**GQ($\lambda$)** is the Gradient-TD algorithm for action values with eligibility traces. Its goal is to learn $\hat{q}(s, a, \mathbf{w}_t) \doteq \mathbf{w}_t^\top \mathbf{x}(s, a) \approx q_\pi(s, a)$ from off-policy data. Its update replaces $\mathbf{x}_{t+1}$ with $\bar{\mathbf{x}}_t \doteq \sum_a \pi(a \mid S_t) \mathbf{x}(S_t, a)$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(HTD($\lambda$))</span></p>

**HTD($\lambda$)** is a hybrid state-value algorithm combining aspects of GTD($\lambda$) and TD($\lambda$). Its most appealing feature is that if the behavior policy happens to be the same as the target policy, then HTD($\lambda$) becomes the same as TD($\lambda$), which is not true for GTD($\lambda$). It is defined by:

$$\mathbf{w}_{t+1} \doteq \mathbf{w}_t + \alpha \delta_t^s \mathbf{z}_t + \alpha \left((\mathbf{z}_t - \mathbf{z}_t^b)^\top \mathbf{v}_t\right) (\mathbf{x}_t - \gamma_{t+1} \mathbf{x}_{t+1}),$$

$$\mathbf{v}_{t+1} \doteq \mathbf{v}_t + \beta \delta_t^s \mathbf{z}_t - \beta \left(\mathbf{z}_t^{b\top} \mathbf{v}_t\right) (\mathbf{x}_t - \gamma_{t+1} \mathbf{x}_{t+1}),$$

where $\mathbf{z}_t^b$ is a second set of eligibility traces for the behavior policy: $\mathbf{z}_t^b \doteq \gamma_t \lambda_t \mathbf{z}_{t-1}^b + \mathbf{x}_t$. These become equal to $\mathbf{z}_t$ if all the $\rho_t$ are 1, causing the last term in the $\mathbf{w}$ update to be zero and the overall update to reduce to TD($\lambda$).

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Emphatic TD($\lambda$))</span></p>

**Emphatic TD($\lambda$)** extends the one-step Emphatic-TD algorithm (Sections 9.11 and 11.8) to eligibility traces. It retains strong off-policy convergence guarantees while enabling any degree of bootstrapping, albeit at the cost of high variance and potentially slow convergence:

$$\mathbf{w}_{t+1} \doteq \mathbf{w}_t + \alpha \delta_t \mathbf{z}_t,$$

$$\delta_t \doteq R_{t+1} + \gamma_{t+1} \mathbf{w}_t^\top \mathbf{x}_{t+1} - \mathbf{w}_t^\top \mathbf{x}_t,$$

$$\mathbf{z}_t \doteq \rho_t \left(\gamma_t \lambda_t \mathbf{z}_{t-1} + M_t \mathbf{x}_t\right), \quad \text{with } \mathbf{z}_{-1} \doteq \mathbf{0},$$

$$M_t \doteq \lambda_t I_t + (1 - \lambda_t) F_t,$$

$$F_t \doteq \rho_{t-1} \gamma_t F_{t-1} + I_t, \quad \text{with } F_0 \doteq i(S_0),$$

where $M_t \ge 0$ is the **emphasis**, $F_t \ge 0$ is the **followon trace**, and $I_t \ge 0$ is the **interest**.

In the on-policy case ($\rho_t = 1$ for all $t$), Emphatic-TD($\lambda$) is similar to conventional TD($\lambda$), but still significantly different. Emphatic-TD($\lambda$) is guaranteed to converge for all state-dependent $\lambda$ functions, while TD($\lambda$) is guaranteed convergent only for all constant $\lambda$.

</div>

### 12.12 Implementation Issues

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Practical Implementation of Eligibility Traces)</span></p>

In the tabular case, a naive implementation would require updating every state's trace on every time step. However, for typical values of $\lambda$ and $\gamma$, the traces of almost all states are nearly zero — only recently visited states have significant traces. Implementations can track and update only the few traces that are significantly greater than zero, making the computational expense just a few times that of a one-step method.

With function approximation (e.g., ANNs with backpropagation), eligibility traces generally cause only a doubling of the required memory and computation per step. Truncated $\lambda$-return methods (Section 12.3) can be computationally efficient though they always require some additional memory.

</div>

### 12.13 Conclusions

| Method | View | Trace Type | Key Feature |
| --- | --- | --- | --- |
| Off-line $\lambda$-return | Forward | — | Ideal target, off-line only |
| TD($\lambda$) | Backward | Accumulating | Classic, simple, online |
| Truncated TD($\lambda$) | Forward | — | Bounded delay, $n$-step hybrid |
| Online $\lambda$-return | Forward | — | Ideal online, expensive |
| True online TD($\lambda$) | Backward | Dutch | Exact match to online $\lambda$-return, $O(d)$ |
| Sarsa($\lambda$) | Backward | Accumulating/Replacing | Action-value control |
| True online Sarsa($\lambda$) | Backward | Dutch | Best Sarsa variant |
| TB($\lambda$) | Backward | $\pi(A_t \mid S_t)$-weighted | No importance sampling |
| GTD($\lambda$) / GQ($\lambda$) | Backward | Off-policy accumulating | Gradient-TD, stable off-policy |
| HTD($\lambda$) | Backward | Dual traces ($\mathbf{z}$, $\mathbf{z}^b$) | Reduces to TD($\lambda$) on-policy |
| Emphatic TD($\lambda$) | Backward | Emphasis-weighted | Strong off-policy guarantees |

Key takeaways from Chapter 12:

* Eligibility traces unify TD and Monte Carlo methods through the **$\lambda$-return**, providing a continuous spectrum from one-step TD ($\lambda = 0$) to Monte Carlo ($\lambda = 1$), with intermediate values typically performing best.
* **TD($\lambda$)** is the classic backward-view algorithm using an accumulating trace. **True online TD($\lambda$)** achieves an exact equivalence with the ideal online $\lambda$-return algorithm using the **dutch trace**, with only $O(d)$ per-step complexity.
* The convergence bound $\overline{\text{VE}}(\mathbf{w}_\infty) \le \frac{1-\gamma\lambda}{1-\gamma} \min_\mathbf{w} \overline{\text{VE}}(\mathbf{w})$ tightens as $\lambda \to 1$, though in practice intermediate $\lambda$ values are generally best.
* **Sarsa($\lambda$)** extends traces to action-value control. The fading-trace bootstrapping strategy is often the most efficient, updating all action values along the trajectory to different degrees.
* With **variable $\lambda$ and $\gamma$** (state-dependent), pseudo termination unifies episodic and continuing settings. Off-policy traces use per-decision importance sampling with control variates.
* **Stable off-policy methods with traces** include GTD($\lambda$), GQ($\lambda$), HTD($\lambda$), and Emphatic TD($\lambda$), each with different tradeoffs between simplicity, convergence guarantees, and variance.


## Chapter 13: Policy Gradient Methods

So far in this book almost all the methods have been *action-value methods*; they learned the values of actions and then selected actions based on their estimated action values. In this chapter we consider methods that instead learn a **parameterized policy** that can select actions without consulting a value function. A value function may still be used to *learn* the policy parameter, but is not required for action selection. We use the notation $\boldsymbol{\theta} \in \mathbb{R}^{d'}$ for the policy's parameter vector and write $\pi(a \mid s, \boldsymbol{\theta}) = \Pr\lbrace A_t = a \mid S_t = s, \boldsymbol{\theta}_t = \boldsymbol{\theta} \rbrace$ for the probability that action $a$ is taken at time $t$ given state $s$ and parameter $\boldsymbol{\theta}$.

These methods seek to *maximize* performance, so their updates approximate gradient *ascent* in a scalar performance measure $J(\boldsymbol{\theta})$:

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t + \alpha \widehat{\nabla J(\boldsymbol{\theta}_t)},$$

where $\widehat{\nabla J(\boldsymbol{\theta}_t)} \in \mathbb{R}^{d'}$ is a stochastic estimate whose expectation approximates the gradient of the performance measure. Methods that learn approximations to both policy and value functions are often called **actor–critic methods**, where "actor" refers to the learned policy and "critic" refers to the learned value function (usually a state-value function).

### 13.1 Policy Approximation and its Advantages

In policy gradient methods, the policy can be parameterized in any way, as long as $\pi(a \mid s, \boldsymbol{\theta})$ is differentiable with respect to its parameters — that is, as long as $\nabla \pi(a \mid s, \boldsymbol{\theta})$ exists and is finite for all $s \in \mathcal{S}$, $a \in \mathcal{A}(s)$, and $\boldsymbol{\theta} \in \mathbb{R}^{d'}$. In practice, to ensure exploration we generally require that the policy never becomes deterministic (i.e., that $\pi(a \mid s, \boldsymbol{\theta}) \in (0, 1)$ for all $s, a, \boldsymbol{\theta}$).

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Soft-max in Action Preferences)</span></p>

If the action space is discrete and not too large, a natural parameterization is to form parameterized numerical preferences $h(s, a, \boldsymbol{\theta}) \in \mathbb{R}$ for each state–action pair. The actions with the highest preferences are given the highest probabilities of being selected, according to an exponential soft-max distribution:

$$\pi(a \mid s, \boldsymbol{\theta}) \doteq \frac{e^{h(s,a,\boldsymbol{\theta})}}{\sum_b e^{h(s,b,\boldsymbol{\theta})}}.$$

The action preferences can be parameterized arbitrarily. For example, they might be computed by a deep artificial neural network, or simply be linear in features:

$$h(s, a, \boldsymbol{\theta}) = \boldsymbol{\theta}^\top \mathbf{x}(s, a),$$

using feature vectors $\mathbf{x}(s, a) \in \mathbb{R}^{d'}$.

</div>

Advantages of parameterizing policies according to the soft-max in action preferences:

1. The approximate policy can approach a **deterministic policy**, whereas with $\varepsilon$-greedy action selection there is always an $\varepsilon$ probability of selecting a random action. Action preferences are driven to produce the optimal stochastic policy; if the optimal policy is deterministic, the preferences of the optimal actions will be driven infinitely higher than all suboptimal actions.
2. It enables the selection of actions with **arbitrary probabilities**. In problems with significant function approximation, the best approximate policy may be stochastic (e.g., bluffing in Poker). Action-value methods have no natural way of finding stochastic optimal policies, whereas policy approximating methods can.
3. The policy may be a **simpler function** to approximate than the action-value function, leading to faster learning and a superior asymptotic policy.
4. The choice of policy parameterization is sometimes a good way of injecting **prior knowledge** about the desired form of the policy into the reinforcement learning system.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(13.1: Short Corridor with Switched Actions)</span></p>

Consider a small corridor gridworld with three nonterminal states and two actions: **right** and **left**. Actions have their usual consequences in the first and third states, but in the second state they are reversed (right moves left, left moves right). The reward is $-1$ per step. Under the function approximation $\mathbf{x}(s, \text{right}) = [1, 0]^\top$ and $\mathbf{x}(s, \text{left}) = [0, 1]^\top$ for all $s$, all states appear identical.

An $\varepsilon$-greedy action-value method with $\varepsilon = 0.1$ is forced to choose between just two policies, achieving values of at most about $-44$ and $-82$. A method that can learn a specific probability with which to select **right** achieves a best probability of about 0.59 and a value of about $-11.6$, which is the optimal stochastic policy under this parameterization.

</div>

### 13.2 The Policy Gradient Theorem

With continuous policy parameterization the action probabilities change smoothly as a function of the learned parameter, whereas in $\varepsilon$-greedy selection the action probabilities may change dramatically for an arbitrarily small change in the estimated action values, if that change results in a different action having the maximal value. Largely because of this, stronger convergence guarantees are available for policy-gradient methods than for action-value methods.

In the episodic case, we define the performance measure as the value of the start state of the episode:

$$J(\boldsymbol{\theta}) \doteq v_{\pi_\boldsymbol{\theta}}(s_0).$$

The challenge is that performance depends on both the action selections and the distribution of states in which those selections are made, and both are affected by the policy parameter. The effect of the policy on the state distribution is typically unknown.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Policy Gradient Theorem — Episodic Case)</span></p>

The policy gradient theorem provides an analytic expression for the gradient of performance with respect to the policy parameter that does **not** involve the derivative of the state distribution:

$$\nabla J(\boldsymbol{\theta}) \propto \sum_s \mu(s) \sum_a q_\pi(s,a) \nabla \pi(a \mid s, \boldsymbol{\theta}),$$

where the gradients are column vectors of partial derivatives with respect to the components of $\boldsymbol{\theta}$, and $\pi$ denotes the policy corresponding to parameter vector $\boldsymbol{\theta}$. The symbol $\propto$ means "proportional to." In the episodic case, the constant of proportionality is the average length of an episode, and in the continuing case it is 1 (an equality). The distribution $\mu$ is the on-policy distribution under $\pi$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of the Policy Gradient Theorem (episodic case)</summary>

With elementary calculus and re-arranging of terms, we prove the policy gradient theorem from first principles. All gradients are with respect to $\boldsymbol{\theta}$. The gradient of the state-value function can be written as

$$\nabla v_\pi(s) = \nabla \left[\sum_a \pi(a \mid s) q_\pi(s, a)\right]$$

$$= \sum_a \bigl[\nabla\pi(a \mid s) q_\pi(s,a) + \pi(a \mid s)\nabla q_\pi(s,a)\bigr]$$

$$= \sum_a \bigl[\nabla\pi(a \mid s) q_\pi(s,a) + \pi(a \mid s)\sum_{s'} p(s' \mid s,a) \nabla v_\pi(s')\bigr].$$

After repeated unrolling:

$$= \sum_{x \in \mathcal{S}} \sum_{k=0}^{\infty} \Pr(s \to x, k, \pi) \sum_a \nabla\pi(a \mid x) q_\pi(x, a),$$

where $\Pr(s \to x, k, \pi)$ is the probability of transitioning from $s$ to $x$ in $k$ steps under policy $\pi$. Then:

$$\nabla J(\boldsymbol{\theta}) = \nabla v_\pi(s_0) = \sum_s \eta(s) \sum_a \nabla\pi(a \mid s) q_\pi(s,a) \propto \sum_s \mu(s) \sum_a \nabla\pi(a \mid s) q_\pi(s,a). \quad \blacksquare$$

</details>
</div>

### 13.3 REINFORCE: Monte Carlo Policy Gradient

The policy gradient theorem gives an exact expression proportional to the gradient; all that is needed is some way of sampling whose expectation equals or approximates this expression. The right-hand side of the policy gradient theorem is a sum over states weighted by how often the states occur under the target policy $\pi$; if $\pi$ is followed, then states will be encountered in these proportions. Thus

$$\nabla J(\boldsymbol{\theta}) \propto \sum_s \mu(s) \sum_a q_\pi(s,a) \nabla\pi(a \mid s, \boldsymbol{\theta}) = \mathbb{E}_\pi \left[\sum_a q_\pi(S_t, a) \nabla\pi(a \mid S_t, \boldsymbol{\theta})\right].$$

Introducing $A_t$ by replacing a sum over the random variable's possible values by an expectation under $\pi$, then sampling:

$$\nabla J(\boldsymbol{\theta}) \propto \mathbb{E}_\pi\left[G_t \frac{\nabla\pi(A_t \mid S_t, \boldsymbol{\theta})}{\pi(A_t \mid S_t, \boldsymbol{\theta})}\right].$$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(REINFORCE Update)</span></p>

Using this sample to instantiate the generic stochastic gradient-ascent algorithm yields the REINFORCE update:

$$\boldsymbol{\theta}_{t+1} \doteq \boldsymbol{\theta}_t + \alpha G_t \frac{\nabla\pi(A_t \mid S_t, \boldsymbol{\theta}_t)}{\pi(A_t \mid S_t, \boldsymbol{\theta}_t)}.$$

Each increment is proportional to the product of a return $G_t$ and a vector, the gradient of the probability of taking the action actually taken divided by the probability of taking that action. The vector $\frac{\nabla\pi(A_t \mid S_t, \boldsymbol{\theta}_t)}{\pi(A_t \mid S_t, \boldsymbol{\theta}_t)} = \nabla \ln \pi(A_t \mid S_t, \boldsymbol{\theta}_t)$ is called the **eligibility vector**.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(REINFORCE: Monte-Carlo Policy-Gradient Control for $\pi_*$)</span></p>

**Input:** a differentiable policy parameterization $\pi(a \mid s, \boldsymbol{\theta})$

**Algorithm parameter:** step size $\alpha > 0$

**Initialize** policy parameter $\boldsymbol{\theta} \in \mathbb{R}^{d'}$ (e.g., to $\mathbf{0}$)

**Loop forever** (for each episode):
* Generate an episode $S_0, A_0, R_1, \ldots, S_{T-1}, A_{T-1}, R_T$, following $\pi(\cdot \mid \cdot, \boldsymbol{\theta})$
* Loop for each step of the episode $t = 0, 1, \ldots, T-1$:
  * $G \leftarrow \sum_{k=t+1}^{T} \gamma^{k-t-1} R_k$
  * $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha \gamma^t G \nabla \ln \pi(A_t \mid S_t, \boldsymbol{\theta})$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Properties of REINFORCE)</span></p>

REINFORCE uses the complete return from time $t$, which includes all future rewards up until the end of the episode. In this sense REINFORCE is a Monte Carlo algorithm and is well defined only for the episodic case with all updates made in retrospect after the episode is completed.

As a stochastic gradient method, REINFORCE has good theoretical convergence properties. By construction, the expected update over an episode is in the same direction as the performance gradient. This assures improvement in expected performance for sufficiently small $\alpha$, and convergence to a local optimum under standard stochastic approximation conditions. However, as a Monte Carlo method REINFORCE may be of **high variance** and thus produce slow learning.

</div>

### 13.4 REINFORCE with Baseline

The policy gradient theorem can be generalized to include a comparison of the action value to an arbitrary *baseline* $b(s)$:

$$\nabla J(\boldsymbol{\theta}) \propto \sum_s \mu(s) \sum_a \bigl(q_\pi(s,a) - b(s)\bigr) \nabla\pi(a \mid s, \boldsymbol{\theta}).$$

The baseline can be any function, even a random variable, as long as it does not vary with $a$; the equation remains valid because the subtracted quantity is zero:

$$\sum_a b(s) \nabla\pi(a \mid s, \boldsymbol{\theta}) = b(s) \nabla \sum_a \pi(a \mid s, \boldsymbol{\theta}) = b(s) \nabla 1 = 0.$$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(REINFORCE with Baseline Update)</span></p>

The update rule that includes a general baseline is:

$$\boldsymbol{\theta}_{t+1} \doteq \boldsymbol{\theta}_t + \alpha \bigl(G_t - b(S_t)\bigr) \frac{\nabla\pi(A_t \mid S_t, \boldsymbol{\theta}_t)}{\pi(A_t \mid S_t, \boldsymbol{\theta}_t)}.$$

Because the baseline could be uniformly zero, this update is a strict generalization of REINFORCE. In general, the baseline leaves the expected value of the update unchanged, but it can have a large effect on its **variance** (and thus speed the learning).

One natural choice for the baseline is an estimate of the state value, $\hat{v}(S_t, \mathbf{w})$, where $\mathbf{w} \in \mathbb{R}^d$ is a weight vector learned by one of the methods presented in previous chapters.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(REINFORCE with Baseline, for estimating $\pi_\boldsymbol{\theta} \approx \pi_*$)</span></p>

**Input:** a differentiable policy parameterization $\pi(a \mid s, \boldsymbol{\theta})$

**Input:** a differentiable state-value function parameterization $\hat{v}(s, \mathbf{w})$

**Algorithm parameters:** step sizes $\alpha^\boldsymbol{\theta} > 0$, $\alpha^\mathbf{w} > 0$

**Initialize** policy parameter $\boldsymbol{\theta} \in \mathbb{R}^{d'}$ and state-value weights $\mathbf{w} \in \mathbb{R}^d$ (e.g., to $\mathbf{0}$)

**Loop forever** (for each episode):
* Generate an episode $S_0, A_0, R_1, \ldots, S_{T-1}, A_{T-1}, R_T$, following $\pi(\cdot \mid \cdot, \boldsymbol{\theta})$
* Loop for each step of the episode $t = 0, 1, \ldots, T-1$:
  * $G \leftarrow \sum_{k=t+1}^{T} \gamma^{k-t-1} R_k$
  * $\delta \leftarrow G - \hat{v}(S_t, \mathbf{w})$
  * $\mathbf{w} \leftarrow \mathbf{w} + \alpha^\mathbf{w} \delta \nabla \hat{v}(S_t, \mathbf{w})$
  * $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha^\boldsymbol{\theta} \gamma^t \delta \nabla \ln \pi(A_t \mid S_t, \boldsymbol{\theta})$

</div>

### 13.5 Actor–Critic Methods

In REINFORCE with baseline, the learned state-value function estimates the value of the *first* state of each state transition. This estimate sets a baseline for the subsequent return, but is made prior to the transition's action and thus cannot be used to assess that action. In actor–critic methods, the state-value function is applied also to the *second* state of the transition. The estimated value of the second state, when discounted and added to the reward, constitutes the one-step return $G_{t:t+1}$, which is a useful estimate of the actual return and thus *is* a way of assessing the action.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Actor–Critic)</span></p>

When the state-value function is used to assess actions in this way it is called a **critic**, and the overall policy-gradient method is termed an **actor–critic** method. Note that the bias in the gradient estimate is not due to bootstrapping as such; the actor would be biased even if the critic was learned by a Monte Carlo method.

One-step actor–critic methods replace the full return of REINFORCE with the one-step return (and use a learned state-value function as the baseline):

$$\boldsymbol{\theta}_{t+1} \doteq \boldsymbol{\theta}_t + \alpha \bigl(G_{t:t+1} - \hat{v}(S_t, \mathbf{w})\bigr) \frac{\nabla\pi(A_t \mid S_t, \boldsymbol{\theta}_t)}{\pi(A_t \mid S_t, \boldsymbol{\theta}_t)}$$

$$= \boldsymbol{\theta}_t + \alpha \bigl(R_{t+1} + \gamma \hat{v}(S_{t+1}, \mathbf{w}) - \hat{v}(S_t, \mathbf{w})\bigr) \frac{\nabla\pi(A_t \mid S_t, \boldsymbol{\theta}_t)}{\pi(A_t \mid S_t, \boldsymbol{\theta}_t)}$$

$$= \boldsymbol{\theta}_t + \alpha \delta_t \frac{\nabla\pi(A_t \mid S_t, \boldsymbol{\theta}_t)}{\pi(A_t \mid S_t, \boldsymbol{\theta}_t)}.$$

This is now a fully online, incremental algorithm, with states, actions, and rewards processed as they occur and then never revisited.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(One-step Actor–Critic (episodic), for estimating $\pi_\boldsymbol{\theta} \approx \pi_*$)</span></p>

**Input:** a differentiable policy parameterization $\pi(a \mid s, \boldsymbol{\theta})$

**Input:** a differentiable state-value function parameterization $\hat{v}(s, \mathbf{w})$

**Parameters:** step sizes $\alpha^\boldsymbol{\theta} > 0$, $\alpha^\mathbf{w} > 0$

**Initialize** policy parameter $\boldsymbol{\theta} \in \mathbb{R}^{d'}$ and state-value weights $\mathbf{w} \in \mathbb{R}^d$ (e.g., to $\mathbf{0}$)

**Loop forever** (for each episode):
* Initialize $S$ (first state of episode)
* $I \leftarrow 1$
* Loop while $S$ is not terminal (for each time step):
  * $A \sim \pi(\cdot \mid S, \boldsymbol{\theta})$
  * Take action $A$, observe $S', R$
  * $\delta \leftarrow R + \gamma \hat{v}(S', \mathbf{w}) - \hat{v}(S, \mathbf{w})$ &emsp; (if $S'$ is terminal, then $\hat{v}(S', \mathbf{w}) \doteq 0$)
  * $\mathbf{w} \leftarrow \mathbf{w} + \alpha^\mathbf{w} \delta \nabla \hat{v}(S, \mathbf{w})$
  * $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha^\boldsymbol{\theta} I \delta \nabla \ln \pi(A \mid S, \boldsymbol{\theta})$
  * $I \leftarrow \gamma I$
  * $S \leftarrow S'$

</div>

The generalizations to $n$-step methods and then to a $\lambda$-return algorithm are straightforward. The one-step return is merely replaced by $G_{t:t+n}$ or $G_t^\lambda$ respectively, using separate eligibility traces for the actor and critic.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Actor–Critic with Eligibility Traces (episodic), for estimating $\pi_\boldsymbol{\theta} \approx \pi_*$)</span></p>

**Input:** a differentiable policy parameterization $\pi(a \mid s, \boldsymbol{\theta})$

**Input:** a differentiable state-value function parameterization $\hat{v}(s, \mathbf{w})$

**Parameters:** trace-decay rates $\lambda^\boldsymbol{\theta} \in [0, 1]$, $\lambda^\mathbf{w} \in [0, 1]$; step sizes $\alpha^\boldsymbol{\theta} > 0$, $\alpha^\mathbf{w} > 0$

**Initialize** policy parameter $\boldsymbol{\theta} \in \mathbb{R}^{d'}$ and state-value weights $\mathbf{w} \in \mathbb{R}^d$ (e.g., to $\mathbf{0}$)

**Loop forever** (for each episode):
* Initialize $S$ (first state of episode)
* $\mathbf{z}^\boldsymbol{\theta} \leftarrow \mathbf{0}$ ($d'$-component eligibility trace vector)
* $\mathbf{z}^\mathbf{w} \leftarrow \mathbf{0}$ ($d$-component eligibility trace vector)
* $I \leftarrow 1$
* Loop while $S$ is not terminal (for each time step):
  * $A \sim \pi(\cdot \mid S, \boldsymbol{\theta})$
  * Take action $A$, observe $S', R$
  * $\delta \leftarrow R + \gamma \hat{v}(S', \mathbf{w}) - \hat{v}(S, \mathbf{w})$ &emsp; (if $S'$ is terminal, then $\hat{v}(S', \mathbf{w}) \doteq 0$)
  * $\mathbf{z}^\mathbf{w} \leftarrow \gamma \lambda^\mathbf{w} \mathbf{z}^\mathbf{w} + \nabla \hat{v}(S, \mathbf{w})$
  * $\mathbf{z}^\boldsymbol{\theta} \leftarrow \gamma \lambda^\boldsymbol{\theta} \mathbf{z}^\boldsymbol{\theta} + I \nabla \ln \pi(A \mid S, \boldsymbol{\theta})$
  * $\mathbf{w} \leftarrow \mathbf{w} + \alpha^\mathbf{w} \delta \mathbf{z}^\mathbf{w}$
  * $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha^\boldsymbol{\theta} \delta \mathbf{z}^\boldsymbol{\theta}$
  * $I \leftarrow \gamma I$
  * $S \leftarrow S'$

</div>

### 13.6 Policy Gradient for Continuing Problems

For continuing problems without episode boundaries we need to define performance in terms of the average rate of reward per time step:

$$J(\boldsymbol{\theta}) \doteq r(\pi) \doteq \lim_{h \to \infty} \frac{1}{h} \sum_{t=1}^{h} \mathbb{E}[R_t \mid S_0, A_{0:t-1} \sim \pi] = \sum_s \mu(s) \sum_a \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a) r,$$

where $\mu(s) \doteq \lim_{t \to \infty} \Pr\lbrace S_t = s \mid A_{0:t} \sim \pi \rbrace$ is the steady-state distribution under $\pi$ (assumed to exist and be independent of $S_0$ — an ergodicity assumption). The steady-state distribution satisfies:

$$\sum_s \mu(s) \sum_a \pi(a \mid s, \boldsymbol{\theta}) p(s' \mid s, a) = \mu(s'), \quad \text{for all } s' \in \mathcal{S}.$$

In the continuing case, values are defined with respect to the **differential return**:

$$G_t \doteq R_{t+1} - r(\pi) + R_{t+2} - r(\pi) + R_{t+3} - r(\pi) + \cdots.$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Policy Gradient Theorem — Continuing Case)</span></p>

With the alternate definitions for the continuing case ($v_\pi$ and $q_\pi$ defined with respect to the differential return), the policy gradient theorem remains true:

$$\nabla J(\boldsymbol{\theta}) = \sum_s \mu(s) \sum_a \nabla\pi(a \mid s) q_\pi(s, a).$$

Note that in the continuing case this is an exact equality (the constant of proportionality is 1).

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Actor–Critic with Eligibility Traces (continuing), for estimating $\pi_\boldsymbol{\theta} \approx \pi_*$)</span></p>

**Input:** a differentiable policy parameterization $\pi(a \mid s, \boldsymbol{\theta})$

**Input:** a differentiable state-value function parameterization $\hat{v}(s, \mathbf{w})$

**Algorithm parameters:** $\lambda^\mathbf{w} \in [0, 1]$, $\lambda^\boldsymbol{\theta} \in [0, 1]$, $\alpha^\mathbf{w} > 0$, $\alpha^\boldsymbol{\theta} > 0$, $\alpha^{\bar{R}} > 0$

**Initialize** $\bar{R} \in \mathbb{R}$ (e.g., to 0)

**Initialize** state-value weights $\mathbf{w} \in \mathbb{R}^d$ and policy parameter $\boldsymbol{\theta} \in \mathbb{R}^{d'}$ (e.g., to $\mathbf{0}$)

**Initialize** $S \in \mathcal{S}$ (e.g., to $s_0$)

$\mathbf{z}^\mathbf{w} \leftarrow \mathbf{0}$ ($d$-component eligibility trace vector)

$\mathbf{z}^\boldsymbol{\theta} \leftarrow \mathbf{0}$ ($d'$-component eligibility trace vector)

**Loop forever** (for each time step):
* $A \sim \pi(\cdot \mid S, \boldsymbol{\theta})$
* Take action $A$, observe $S', R$
* $\delta \leftarrow R - \bar{R} + \hat{v}(S', \mathbf{w}) - \hat{v}(S, \mathbf{w})$
* $\bar{R} \leftarrow \bar{R} + \alpha^{\bar{R}} \delta$
* $\mathbf{z}^\mathbf{w} \leftarrow \lambda^\mathbf{w} \mathbf{z}^\mathbf{w} + \nabla \hat{v}(S, \mathbf{w})$
* $\mathbf{z}^\boldsymbol{\theta} \leftarrow \lambda^\boldsymbol{\theta} \mathbf{z}^\boldsymbol{\theta} + \nabla \ln \pi(A \mid S, \boldsymbol{\theta})$
* $\mathbf{w} \leftarrow \mathbf{w} + \alpha^\mathbf{w} \delta \mathbf{z}^\mathbf{w}$
* $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha^\boldsymbol{\theta} \delta \mathbf{z}^\boldsymbol{\theta}$
* $S \leftarrow S'$

</div>

### 13.7 Policy Parameterization for Continuous Actions

Policy-based methods offer practical ways of dealing with large action spaces, even continuous spaces with an infinite number of actions. Instead of computing learned probabilities for each of the many actions, we instead learn statistics of the probability distribution. For example, the action set might be the real numbers, with actions chosen from a normal (Gaussian) distribution.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Gaussian Policy Parameterization)</span></p>

The policy can be defined as the normal probability density over a real-valued scalar action, with mean and standard deviation given by parametric function approximators that depend on the state:

$$\pi(a \mid s, \boldsymbol{\theta}) \doteq \frac{1}{\sigma(s, \boldsymbol{\theta})\sqrt{2\pi}} \exp\left(-\frac{(a - \mu(s, \boldsymbol{\theta}))^2}{2\sigma(s, \boldsymbol{\theta})^2}\right),$$

where $\mu : \mathcal{S} \times \mathbb{R}^{d'} \to \mathbb{R}$ and $\sigma : \mathcal{S} \times \mathbb{R}^{d'} \to \mathbb{R}^+$ are two parameterized function approximators. The policy's parameter vector is divided into two parts, $\boldsymbol{\theta} = [\boldsymbol{\theta}_\mu, \boldsymbol{\theta}_\sigma]^\top$. The mean can be approximated as a linear function and the standard deviation as the exponential of a linear function (to ensure positivity):

$$\mu(s, \boldsymbol{\theta}) \doteq \boldsymbol{\theta}_\mu^\top \mathbf{x}_\mu(s) \qquad \text{and} \qquad \sigma(s, \boldsymbol{\theta}) \doteq \exp\bigl(\boldsymbol{\theta}_\sigma^\top \mathbf{x}_\sigma(s)\bigr).$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Eligibility Vector for Gaussian Policies)</span></p>

For the Gaussian policy parameterization, the eligibility vector has the following two parts:

$$\nabla \ln \pi(a \mid s, \boldsymbol{\theta}_\mu) = \frac{1}{\sigma(s, \boldsymbol{\theta})^2}\bigl(a - \mu(s, \boldsymbol{\theta})\bigr) \mathbf{x}_\mu(s),$$

$$\nabla \ln \pi(a \mid s, \boldsymbol{\theta}_\sigma) = \left(\frac{(a - \mu(s, \boldsymbol{\theta}))^2}{\sigma(s, \boldsymbol{\theta})^2} - 1\right) \mathbf{x}_\sigma(s).$$

</div>

### 13.8 Summary

| Method | Type | Update Target | Key Feature |
| --- | --- | --- | --- |
| REINFORCE | Monte Carlo | Full return $G_t$ | Unbiased but high variance |
| REINFORCE with Baseline | Monte Carlo | $G_t - \hat{v}(S_t, \mathbf{w})$ | Reduced variance, still unbiased |
| One-step Actor–Critic | TD (bootstrapping) | $\delta_t = R_{t+1} + \gamma \hat{v}(S_{t+1}) - \hat{v}(S_t)$ | Fully online, incremental |
| Actor–Critic with Traces | TD + eligibility traces | $\delta_t$ with $\mathbf{z}^\boldsymbol{\theta}$, $\mathbf{z}^\mathbf{w}$ | Unifies TD and MC via $\lambda$ |
| Continuing Actor–Critic | Average reward | $R - \bar{R} + \hat{v}(S') - \hat{v}(S)$ | No episodes needed |

Key takeaways from Chapter 13:

* **Policy gradient methods** learn a parameterized policy $\pi(a \mid s, \boldsymbol{\theta})$ directly, updating $\boldsymbol{\theta}$ in the direction of an estimate of $\nabla J(\boldsymbol{\theta})$. They can learn specific probabilities, approach deterministic policies, handle continuous action spaces, and inject prior knowledge through the parameterization.
* The **policy gradient theorem** provides an exact, analytic expression for the gradient of performance that does *not* involve derivatives of the state distribution: $\nabla J(\boldsymbol{\theta}) \propto \sum_s \mu(s) \sum_a q_\pi(s,a) \nabla\pi(a \mid s, \boldsymbol{\theta})$.
* **REINFORCE** is the classic Monte Carlo policy-gradient method. Adding a state-value **baseline** $\hat{v}(S_t, \mathbf{w})$ does not change the expected update but can greatly reduce its variance, speeding learning.
* **Actor–critic** methods bootstrap by using the TD error $\delta_t$ instead of the full return, introducing bias but substantially reducing variance. The critic (state-value function) assesses the action chosen by the actor (policy). Eligibility traces provide a smooth spectrum between REINFORCE and one-step actor–critic.
* For **continuous actions**, the policy can be parameterized as a Gaussian with learned mean and standard deviation, enabling all the algorithms of this chapter to be applied to real-valued action spaces.

## Chapter 14: Psychology

This chapter looks at reinforcement learning algorithms from the perspective of psychology and its study of how animals learn. Many of the basic reinforcement learning algorithms were inspired by psychological theories, and in some cases, these algorithms have contributed to the development of new animal learning models. The clear formalism provided by reinforcement learning that systemizes tasks, returns, and algorithms is proving to be enormously useful in making sense of experimental data and in suggesting new kinds of experiments.

### 14.1 Prediction and Control

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Prediction and Control in Psychology)</span></p>

The algorithms described in this book fall into two broad categories that correspond to categories of learning extensively studied by psychologists:

* **Prediction** $\leftrightarrow$ **Classical (Pavlovian) conditioning:** Prediction algorithms estimate quantities that depend on how features of an agent's environment are expected to unfold over the future. The correspondence with classical conditioning rests on their common property of predicting upcoming stimuli.
* **Control** $\leftrightarrow$ **Instrumental (operant) conditioning:** In instrumental conditioning the reinforcing stimulus is *contingent* on the animal's behavior — the animal learns to increase rewarded behavior and decrease penalized behavior. This corresponds to policy-improvement algorithms.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Terminology: Reinforcement in Psychology)</span></p>

In psychology, *reinforcement* originally referred to the strengthening of a pattern of behavior (increasing either its intensity or frequency). A stimulus considered to be the cause of the behavioral change is called a *reinforcer*, whether or not it is contingent on the animal's previous behavior. The term is frequently also used for the weakening of a behavior pattern. Many terms in reinforcement learning are borrowed from animal learning theories, but the computational meanings do not always coincide with their meanings in psychology.

</div>

### 14.2 Classical Conditioning

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Classical / Pavlovian Conditioning)</span></p>

While studying the digestive system, Ivan Pavlov found that an animal's innate responses to certain triggering stimuli can come to be triggered by other stimuli that are quite unrelated to the inborn triggers. The key terms are:

* **Unconditioned Response (UR):** An innate response (e.g., salivation in response to food).
* **Unconditioned Stimulus (US):** The natural triggering stimulus (e.g., food). Called a *reinforcer* because it reinforces producing a CR in response to the CS.
* **Conditioned Stimulus (CS):** An initially neutral stimulus (e.g., metronome sound) that the animal learns predicts the US, and so comes to produce a CR.
* **Conditioned Response (CR):** The new response triggered by the CS (e.g., salivation to the metronome).

Two common experimental arrangements:
* **Delay conditioning:** The CS extends throughout the interstimulus interval (ISI), from CS onset to US onset.
* **Trace conditioning:** The CS begins after the CS ends, and the time between CS offset and US onset is the trace interval.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Anticipatory Nature of CRs)</span></p>

Some CRs are similar to the UR but begin earlier and differ in ways that increase their effectiveness. For example, in rabbit nictitating membrane conditioning, a tone CS reliably predicts a puff of air (the US). After conditioning, the tone triggers a CR consisting of membrane closure that begins *before* the air puff and is timed so that peak closure occurs just when the air puff is likely. This anticipatory CR offers better protection than simply reacting to the US. The ability to act in anticipation of important events by learning about predictive relationships among stimuli is widely present across the animal kingdom.

</div>

#### 14.2.1 Blocking and Higher-order Conditioning

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Blocking)</span></p>

**Blocking** occurs when an animal fails to learn a CR when a potential CS is presented along with another CS that had been used previously to condition the animal to produce that CR. For example:
1. A rabbit is first conditioned with a tone CS and an air puff US.
2. A second stimulus (light) is added to the tone to form a compound tone/light CS followed by the same air puff US.
3. The light alone is tested — the rabbit produces very few, or no, CRs.

Learning to the light had been *blocked* by the previous learning to the tone. This challenged the idea that conditioning depends only on simple temporal contiguity (a US frequently following a CS closely in time).

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Higher-order Conditioning)</span></p>

**Higher-order conditioning** occurs when a previously-conditioned CS acts as a US in conditioning another initially neutral stimulus. For example:
1. A dog is conditioned to salivate to a metronome (CS) that predicts food (US).
2. A black square (new CS) is paired with the metronome sound (now acting as US) — *without* food.
3. After just ten trials, the dog salivates upon seeing the black square alone.

This is **second-order conditioning**. A stimulus that consistently predicts primary reinforcement becomes a reinforcer itself — a **secondary reinforcer** or **conditioned reinforcer**. Conditioned reinforcement is a key phenomenon: it explains why we work for conditioned reinforcers like money, whose worth derives solely from what is predicted by having it.

</div>

#### 14.2.2 The Rescorla–Wagner Model

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Rescorla–Wagner Model)</span></p>

The core idea of the Rescorla–Wagner model is that **an animal only learns when events violate its expectations** — in other words, only when the animal is surprised. The model adjusts the "associative strength" of each component stimulus of a compound CS.

For a compound CS AX (components A and X) followed by a US Y:

$$\Delta V_A = \alpha_A \beta_Y (R_Y - V_{AX}), \qquad \Delta V_X = \alpha_X \beta_Y (R_Y - V_{AX}),$$

where $\alpha_A \beta_Y$ and $\alpha_X \beta_Y$ are step-size parameters, $R_Y$ is the asymptotic level of associative strength the US can support, and $V_{AX} = V_A + V_X$ is the aggregate associative strength.

Recast in reinforcement learning notation with linear function approximation:
* **State** $s$ described by feature vector $\mathbf{x}(s) = (x_1(s), \ldots, x_d(s))^\top$ where $x_i(s) = 1$ if CS$_i$ is present.
* **Aggregate associative strength** (value estimate): $\hat{v}(s, \mathbf{w}) = \mathbf{w}^\top \mathbf{x}(s)$.
* **Update:** $\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha \delta_t \mathbf{x}(S_t)$, where the **prediction error** is $\delta_t = R_t - \hat{v}(S_t, \mathbf{w}_t)$.

This is essentially the **Least Mean Square (LMS)**, or Widrow–Hoff, learning rule.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(How the Rescorla–Wagner Model Explains Blocking)</span></p>

As long as $V_{AX}$ is below $R_Y$, the prediction error $R_Y - V_{AX}$ is positive and the associative strengths increase. When a new component CS is added to a compound CS that has already been conditioned, further conditioning produces little or no increase in the added component's associative strength because the error has already been reduced to (near) zero. The US is already predicted nearly perfectly, so there is little surprise — blocking occurs.

</div>

#### 14.2.3 The TD Model

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(TD Model of Classical Conditioning)</span></p>

The TD model extends the Rescorla–Wagner model from a *trial-level* model to a *real-time* model. Instead of updating once per complete trial, the TD model updates at every time step within and between trials, addressing how timing relationships among stimuli influence learning. The TD model also naturally accounts for higher-order conditioning through bootstrapping.

The key equations are:
* **Weight update:** $\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha \delta_t \mathbf{z}_t$
* **TD error:** $\delta_t = R_{t+1} + \gamma \hat{v}(S_{t+1}, \mathbf{w}_t) - \hat{v}(S_t, \mathbf{w}_t)$
* **Eligibility trace:** $\mathbf{z}_t = \gamma \lambda \mathbf{z}_{t-1} + \mathbf{x}(S_t)$

where $\gamma$ is the discount factor and $\lambda$ is the eligibility trace decay parameter.

If $\gamma = 0$, the TD model reduces to the Rescorla–Wagner model (with differences in the meaning of $t$: trial number vs. time step, and a one-time-step lead in the prediction target).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Stimulus Representations in the TD Model)</span></p>

Three stimulus representations have been used with the TD model:

| Representation | Description | Temporal Generalization |
| --- | --- | --- |
| **Complete Serial Compound (CSC)** | Each external stimulus initiates a sequence of precisely-timed short-duration internal signals (like a "tapped delay line") | None between nearby time points |
| **Microstimulus (MS)** | Each stimulus initiates a cascade of overlapping, extended internal stimuli that become progressively wider and lower in amplitude | Middle ground |
| **Presence** | A single feature with value 1 whenever a CS component is present, 0 otherwise | Complete between nearby time points |

The choice of stimulus representation strongly influences the time course of the US prediction (CR profile). The CSC produces an exponentially increasing prediction; the presence representation produces a nearly constant prediction; the MS representation approximates the CSC curve at asymptote through a linear combination of microstimuli.

</div>

#### 14.2.4 TD Model Simulations

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Key Phenomena Explained by the TD Model)</span></p>

The TD model with the simple presence representation accounts for all the basic properties of classical conditioning that the Rescorla–Wagner model explains, plus additional phenomena that are beyond the scope of trial-level models:

* **ISI dependency:** Conditioning generally requires a positive ISI; it is negligible for zero or negative ISI, increases to a maximum at a positive ISI, then decreases.
* **Facilitation of remote associations:** In serial-compound conditioning, if the trace interval between CSA and the US is filled by a second CS (CSB), conditioning to CSA is facilitated by CSB's presence.
* **Blocking with temporal primacy:** The TD model predicts that blocking is reversed if the blocked stimulus is moved earlier in time so that its onset occurs before the blocking stimulus. This prediction was confirmed experimentally (Kehoe, Schreurs, and Graham, 1987).
* **Higher-order conditioning:** Because $\gamma \hat{v}(S_{t+1}, \mathbf{w}_t)$ appears in the TD error, a temporal difference can have the same status as $R_{t+1}$ — there is no difference between a temporal difference and the occurrence of a US as far as learning is concerned. This is the bootstrapping idea that connects to dynamic programming.

</div>

### 14.3 Instrumental Conditioning

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Instrumental / Operant Conditioning)</span></p>

In **instrumental conditioning** (also called **operant conditioning**), learning depends on the consequences of behavior: the delivery of a reinforcing stimulus is contingent on what the animal does. This contrasts with classical conditioning where the reinforcing stimulus (US) is delivered independently of behavior.

The roots of instrumental conditioning go back to Thorndike's experiments with cats in "puzzle boxes." Thorndike observed that escape time decreased with successive experiences, and formulated the **Law of Effect**: learning by trial and error, where successful acts are "stamped in" by resulting pleasure and unsuccessful acts are "stamped out."

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Correspondences with Reinforcement Learning)</span></p>

Essential features of reinforcement learning algorithms correspond to features of animal learning described by the Law of Effect:

1. **Selectional:** RL algorithms try alternatives and select among them by comparing their consequences — like trial and error. (Natural selection is selectional but not associative; supervised learning is associative but not selectional.)
2. **Associative:** The alternatives found by selection are associated with particular situations or states to form the agent's policy — like Thorndike's "selecting and connecting."
3. **Search and memory:** Search in the form of trying and selecting among actions; memory in the form of associations linking situations with actions that work best (policy, value function, or model).

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Shaping)</span></p>

**Shaping** is the process of training an animal (or agent) by reinforcing successive approximations of the desired behavior, progressively altering reinforcement contingencies. B. F. Skinner recognized this technique and compared it to a sculptor shaping clay.

Shaping is a powerful technique for computational reinforcement learning as well: when it is difficult for an agent to receive any non-zero reward signal, starting with an easier problem and incrementally increasing its difficulty as the agent learns can be an effective, and sometimes indispensable, strategy.

</div>

### 14.4 Delayed Reinforcement

The problem of **delayed reinforcement** is related to what Minsky (1961) called the "credit-assignment problem for learning systems": how do you distribute credit for success among the many decisions that may have been involved in producing it? Reinforcement learning algorithms include two basic mechanisms for addressing this problem:

1. **Eligibility traces:** Decaying traces of past state visitations or state–action pairs. These correspond to stimulus traces proposed by Pavlov (1927) and Hull's (1943) "molar stimulus traces" — internal stimuli whose traces decay exponentially as functions of time since an action was taken.
2. **TD methods and value functions:** Learning value functions via TD methods provides nearly immediate evaluations of actions (in instrumental conditioning) or immediate prediction targets (in classical conditioning). The TD error acts as a **conditioned reinforcement signal**, providing immediate evaluation even when the primary reward is considerably delayed.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Actor–Critic and Delayed Reinforcement)</span></p>

The actor–critic architecture illustrates the correspondence between RL and Hull's theory most clearly. The critic uses a TD algorithm to learn a value function (predicting the current policy's return). The actor updates the current policy based on changes in the critic's predictions. The TD error produced by the critic acts as a conditioned reinforcement signal for the actor, providing an immediate evaluation of performance even when the primary reward signal itself is considerably delayed.

</div>

### 14.5 Cognitive Maps

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Cognitive Maps and Latent Learning)</span></p>

Model-based reinforcement learning algorithms use environment models that have elements in common with what psychologists call **cognitive maps**. Environment models consist of two parts:
* **State-transition model:** Knowledge about the effect of actions on state changes.
* **Reward model:** Knowledge about the reward signals expected for each state or state–action pair.

**Latent learning** experiments demonstrated that animals could learn a "cognitive map of the environment" in the absence of rewards or penalties, and use the map later when motivated to reach a goal. In the earliest experiment (Blodgett, 1929), rats explored a maze without reward, then rapidly caught up with rewarded controls once food was introduced — evidence that they had been building an internal model all along.

The cognitive map explanation is analogous to the claim that animals use model-based algorithms, and that environment models can be learned even without explicit rewards or penalties, through forms of supervised learning (e.g., learning S–S' or SA–S' associations — what control engineers call *system identification*).

</div>

### 14.6 Habitual and Goal-directed Behavior

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Habitual vs. Goal-directed Behavior)</span></p>

The distinction between **model-free** and **model-based** reinforcement learning algorithms corresponds to the distinction psychologists make between **habitual** and **goal-directed** control of learned behavioral patterns:

| | Habitual | Goal-directed |
| --- | --- | --- |
| **Controlled by** | Antecedent stimuli | Consequences (value of goals) |
| **RL analog** | Model-free (cached values/policy) | Model-based (planning with a model) |
| **Adapts to change** | Slowly (requires relearning from experience) | Rapidly (planning with updated model) |
| **Speed** | Fast, automatic | Slower, deliberate |

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Outcome-Devaluation Experiments)</span></p>

**Outcome-devaluation experiments** test whether an animal's behavior is habitual or goal-directed. In the Adams and Dickinson (1981) experiment:
1. Rats were trained to press a lever for sucrose pellets (instrumental conditioning).
2. In a separate context (lever retracted), pellets were paired with nausea-inducing lithium chloride — devaluing the pellets.
3. In extinction trials (lever present, but disconnected), the injected rats pressed the lever significantly less than non-injected rats *from the start* — without ever experiencing the devalued reward as a result of lever-pressing.

The rats "knew" (via their cognitive map) that lever pressing $\to$ pellets $\to$ nausea, and so reduced pressing immediately. This is evidence for **goal-directed (model-based)** control. In a follow-up (Adams, 1982), overtrained rats (500 presses) showed no sensitivity to devaluation — their behavior had become **habitual (model-free)**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Transition from Goal-directed to Habitual)</span></p>

An idea developed by computational neuroscientists Daw, Niv, and Dayan (2005) is that animals use both model-free and model-based processes. Each process proposes an action, and the action chosen for execution is the one proposed by the process judged to be more trustworthy, as determined by measures of confidence maintained throughout learning.

* **Early in learning:** The model-based (planning) process is more trustworthy because it chains together short-term predictions, which can become accurate with less experience than the long-term predictions of the model-free process.
* **With continued experience:** The model-free process becomes more trustworthy because planning is prone to mistakes due to model inaccuracies and shortcuts like tree-pruning.

This predicts a shift from goal-directed to habitual behavior as experience accumulates — consistent with experimental observations.

</div>

### 14.7 Summary

| RL Concept | Psychological Correspondence |
| --- | --- |
| Prediction algorithms | Classical (Pavlovian) conditioning |
| Control algorithms | Instrumental (operant) conditioning |
| TD error / prediction error | Surprise (Rescorla–Wagner); dopamine signal (Chapter 15) |
| Rescorla–Wagner model | LMS / Widrow–Hoff learning rule (trial-level) |
| TD model | Real-time extension of Rescorla–Wagner with eligibility traces |
| Eligibility traces | Stimulus traces (Pavlov, Hull) |
| Value functions / conditioned reinforcement | Secondary reinforcement; Hull's goal gradients |
| Environment models | Cognitive maps (Tolman) |
| Model-free algorithms | Habitual behavior |
| Model-based algorithms | Goal-directed behavior |
| Shaping (reward curriculum) | Skinner's successive approximations |

Key takeaways from Chapter 14:

* The prediction/control distinction in RL parallels the **classical/instrumental conditioning** distinction in animal learning psychology. The Rescorla–Wagner model is essentially the LMS rule; the **TD model** extends it to real-time, accounting for within-trial timing effects, blocking, higher-order conditioning, and ISI dependency through bootstrapping and eligibility traces.
* Thorndike's **Law of Effect** describes learning by trial and error — the selectional and associative processes at the heart of RL. Skinner's concept of **shaping** (reinforcing successive approximations) is also a powerful technique for computational RL.
* RL's mechanisms for **delayed reinforcement** — eligibility traces and TD-learned value functions providing conditioned reinforcement — correspond closely to similar mechanisms proposed in theories of animal learning (Pavlov's stimulus traces, Hull's goal gradients).
* The distinction between **model-free and model-based** RL corresponds to the psychological distinction between **habitual and goal-directed** behavior. Outcome-devaluation experiments reveal which type of control governs an animal's behavior, and the computational perspective helps explain transitions between these modes as experience accumulates.

## Chapter 15: Neuroscience

One of the most exciting aspects of reinforcement learning is the mounting evidence from neuroscience that the nervous systems of humans and many other animals implement algorithms that correspond in striking ways to reinforcement learning algorithms. The main objective of this chapter is to explain these parallels and what they suggest about the neural basis of reward-related learning in animals.

The most remarkable point of contact involves **dopamine**, a chemical deeply involved in reward processing in the brains of mammals. Dopamine appears to convey temporal-difference (TD) errors to brain structures where learning and decision making take place.

### Neuroscience Basics

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Neuron and Neural Signaling)</span></p>

**Neurons** are cells specialized for processing and transmitting information using electrical and chemical signals. A neuron typically has:

* A **cell body**
* **Dendrites:** structures that branch from the cell body to receive input from other neurons (or external signals for sensory neurons)
* A single **axon:** a fiber that carries the neuron's output to other neurons (or to muscles or glands)

A neuron's output consists of sequences of electrical pulses called **action potentials** (also called **spikes**). A neuron is said to **fire** when it generates a spike. A neuron's **firing rate** is the average number of spikes per unit of time.

A neuron's axon can branch widely so that its action potentials reach many targets. The branching structure is called the neuron's **axonal arbor**.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Synapse and Neurotransmission)</span></p>

A **synapse** is a structure at the termination of an axon branch that mediates communication from one neuron to another. A synapse transmits information from the **presynaptic** neuron's axon to a dendrite or cell body of the **postsynaptic** neuron.

Upon the arrival of an action potential, synapses release a chemical **neurotransmitter** that diffuses across the **synaptic cleft** and binds to receptors on the postsynaptic neuron to excite or inhibit its activity.

* The strength or effectiveness by which a neurotransmitter influences the postsynaptic neuron is the synapse's **efficacy**.
* A **neuromodulator** is a neurotransmitter that has effects other than direct fast excitation or inhibition. Neuromodulatory systems can distribute a scalar-like signal (e.g. a reinforcement signal) to alter synapses in widely distributed sites.
* The ability of synaptic efficacies to change is called **synaptic plasticity** — one of the primary mechanisms responsible for learning.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Background, Phasic, and Tonic Activity)</span></p>

* **Background activity:** A neuron's level of activity (firing rate) when the neuron does not appear to be driven by synaptic input related to the task of interest.
* **Phasic activity:** Bursts of spiking activity usually caused by synaptic input, in contrast to background activity.
* **Tonic activity:** Activity that varies slowly and often in a graded manner, whether as background activity or not.

</div>

### Reward Signals, Reinforcement Signals, Values, and Prediction Errors

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Distinguishing Reward-Related Signals)</span></p>

In connecting neuroscience to RL theory, it is important to distinguish several types of signals:

* **Reward signal ($R_t$):** Like a reward signal in an animal's brain — an abstraction summarizing the overall effect of a multitude of neural signals that assess rewarding or punishing qualities. It is unlikely that a unitary master reward signal like $R_t$ exists in the brain.
* **Reinforcement signal:** Different from the reward signal. It directs changes in an agent's policy, value estimates, or environment models. For a TD method, the reinforcement signal at time $t$ is the TD error $\delta_{t-1} = R_t + \gamma V(S_t) - V(S_{t-1})$, not just $R_t$.
* **Value signals ($V$ or $Q$):** Predictions of the total reward an agent can expect to accumulate over the future.
* **Prediction errors (RPEs):** Discrepancies between expected and actual signals. TD errors are special kinds of RPEs that signal discrepancies between current and earlier expectations of reward over the long term.

When neuroscientists refer to RPEs, they generally mean TD RPEs.

</div>

### The Reward Prediction Error Hypothesis

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Reward Prediction Error Hypothesis of Dopamine Neuron Activity)</span></p>

The **reward prediction error hypothesis of dopamine neuron activity** proposes that one of the functions of the phasic activity of dopamine-producing neurons in mammals is to deliver an error between an old and a new estimate of expected future reward to target areas throughout the brain.

This hypothesis was first explicitly stated by Montague, Dayan, and Sejnowski (1996), who showed how the TD error concept from reinforcement learning accounts for many features of the phasic activity of dopamine neurons in mammals. The experiments that led to this hypothesis were performed in the 1980s and early 1990s in the laboratory of neuroscientist Wolfram Schultz.

Montague et al. (1996) compared the TD errors of the TD model of classical conditioning (basically the semi-gradient-descent $\text{TD}(\lambda)$ algorithm with linear function approximation) with the phasic activity of dopamine-producing neurons during classical conditioning experiments.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(TD Error–Dopamine Parallels)</span></p>

In simulated trials, TD errors of the TD model are remarkably similar to dopamine neuron phasic activity. The TD errors parallel the following features of dopamine neuron activity:

1. The phasic response of a dopamine neuron only occurs when a rewarding event is **unpredicted**.
2. Early in learning, neutral cues that precede a reward do not cause substantial phasic dopamine responses, but with continued learning these cues gain predictive value and come to elicit phasic dopamine responses.
3. If an even earlier cue reliably precedes a cue that has already acquired predictive value, the phasic dopamine response **shifts to the earlier cue**, ceasing for the later cue.
4. If after learning the predicted rewarding event is **omitted**, a dopamine neuron's response **decreases below its baseline** level shortly after the expected time of the rewarding event.

</div>

### Dopamine

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Dopamine)</span></p>

**Dopamine** is produced as a neurotransmitter by neurons whose cell bodies lie mainly in two clusters of neurons in the midbrain of mammals:

* The **substantia nigra pars compacta (SNpc)**
* The **ventral tegmental area (VTA)**

Dopamine plays essential roles in motivation, learning, action-selection, most forms of addiction, and the disorders schizophrenia and Parkinson's disease. Dopamine is called a neuromodulator because it performs functions other than direct fast excitation or inhibition.

Dopamine neurons have huge axonal arbors, each releasing dopamine at 100 to 1,000 times more synaptic sites than reached by the axons of typical neurons. Each axon of a SNpc or VTA dopamine neuron makes roughly **500,000 synaptic contacts** on the dendrites of neurons in targeted brain areas.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Dopamine as Reinforcement Signal, Not Reward Signal)</span></p>

If the reward prediction error hypothesis is correct, the traditional view that dopamine neuron activity signals reward is not entirely correct: phasic responses of dopamine neurons signal **reward prediction errors, not reward itself**. A dopamine neuron's phasic response at time $t$ corresponds to $\delta_{t-1} = R_t + \gamma V(S_t) - V(S_{t-1})$, not to $R_t$.

In many RL algorithms, $\delta$ functions as a reinforcement signal and is the main driver of learning. The reward signal $R_t$ is a crucial component of $\delta_{t-1}$, but the additional term $\gamma V(S_t) - V(S_{t-1})$ is the higher-order reinforcement part. Even if reward occurs ($R_t \neq 0$), the TD error can be silent if the reward is fully predicted.

</div>

### Experimental Support for the Reward Prediction Error Hypothesis

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Schultz's Experiments)</span></p>

Romo and Schultz (1990) took the first steps toward the reward prediction error hypothesis by recording dopamine neuron activity while monkeys performed reward-related tasks. Key findings:

* Dopamine neurons respond with bursts of activity to intense, novel, or unexpected visual and auditory stimuli, but very little of their activity is related to the movements themselves.
* When monkeys were trained to reach for food in a bin, dopamine neurons produced phasic responses whenever the monkey first touched a food morsel, but not when it touched the wire or explored an empty bin.
* After training with a predictive cue (sight and sound of bin opening), the phasic responses shifted from the reward itself to the earlier predictive stimulus.

Schultz's group further showed that phasic responses correspond to **TD errors** (not simpler Rescorla–Wagner errors):

* Monkeys trained to press a lever after a trigger cue: dopamine neurons initially responded to the reward but shifted to the trigger cue with training.
* When an even earlier instruction cue was added, responses shifted to the instruction cue, ceasing for the trigger cue.
* When monkeys pressed the wrong lever and received no reward, dopamine neurons showed a sharp **decrease in firing rate below baseline** shortly after the reward's usual delivery time — even without any external cue marking that time.

</div>

### TD Error/Dopamine Correspondence

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Reward-Predicting States)</span></p>

To explain the TD error/dopamine correspondence, consider a policy-evaluation task with a sequence of states in each trial, ending with a non-zero reward $R^*$:

* **Earliest reward-predicting state:** The first state in a trial that reliably predicts reward. More generally, it is an **unpredicted predictor** of reward — states that do not predict reward have low values, so transitioning to the earliest reward-predicting state produces a positive TD error.
* **Latest reward-predicting state:** The state immediately preceding the rewarding state. Its value, being correct, cancels the reward, producing a zero TD error at the rewarding state.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(TD Error Behavior During Learning)</span></p>

Using $\text{TD}(0)$ with a CSC representation, $V$ initialized to zero, and $\gamma \approx 1$:

* **Early in learning:** All $V$-values are zero. The TD error is zero until the rewarding state is reached, where $\delta_{t-1} = R_t + 0 - 0 = R^*$. This is analogous to dopamine neurons responding to an unpredicted reward.
* **Learning complete:** Values of all reward-predicting states equal $R^*$. TD errors are zero for transitions between reward-predicting states ($0 + R^* - R^* = 0$) and at the rewarding state ($R^* + 0 - R^* = 0$). A positive TD error occurs only at the transition *to* the earliest reward-predicting state ($0 + R^* - 0 = R^*$). This is analogous to dopamine responses persisting only at the earliest predictive stimulus.
* **Reward omitted:** At the usual reward time, $\delta_{t-1} = 0 + 0 - R^* = -R^*$. This is analogous to dopamine neuron activity decreasing below baseline when expected reward is omitted.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Discrepancies in the TD Error/Dopamine Correspondence)</span></p>

Not every property of dopamine neuron phasic activity matches $\delta$ perfectly. A notable discrepancy involves reward arriving **earlier than expected**: dopamine neurons respond with a positive TD error (consistent), but at the later time when reward is expected but omitted, the TD error is negative whereas dopamine activity does **not** drop below baseline as the TD model predicts. This suggests something more complicated is going on in the brain than simple TD learning with a CSC representation.

Some mismatches can be addressed by selecting suitable parameter values and using stimulus representations other than CSC (e.g. microstimulus representations). Despite imperfections, the reward prediction error hypothesis has received wide acceptance and proven remarkably resilient.

</div>

### Neural Actor–Critic

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Neural Actor–Critic)</span></p>

Evidence supporting the reward prediction error hypothesis, together with the fact that phasic dopamine responses act as reinforcement signals, suggests the brain might implement something like an **actor–critic algorithm**:

* The **actor** learns policies and is hypothesized to reside in the **dorsal striatum**, which is primarily implicated in influencing action selection.
* The **critic** learns the state-value function and is hypothesized to reside in the **ventral striatum**, which is thought to be critical for different aspects of reward processing, including the assignment of affective value to sensations.

Two distinctive features of actor–critic algorithms support this brain hypothesis:

1. The two components (actor and critic) map onto two parts of the striatum (dorsal and ventral subdivisions), both critical for reward-based learning.
2. The TD error has the dual role of being the reinforcement signal for both the actor and the critic, though with a different influence on learning in each. This fits well with the fact that dopamine neuron axons target both subdivisions, and dopamine's effect on a target structure depends on properties of the target structure, not just the neuromodulator.

</div>

### Actor and Critic Learning Rules

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Actor–Critic Update Rules)</span></p>

On each transition from $S_t$ to $S_{t+1}$, taking action $A_t$ and receiving reward $R_{t+1}$, the actor–critic algorithm computes:

$$\delta_t = R_{t+1} + \gamma \hat{v}(S_{t+1}, \mathbf{w}) - \hat{v}(S_t, \mathbf{w}),$$

$$\mathbf{z}_t^{\mathbf{w}} = \gamma \lambda^{\mathbf{w}} \mathbf{z}_{t-1}^{\mathbf{w}} + \nabla \hat{v}(S_t, \mathbf{w}),$$

$$\mathbf{z}_t^{\boldsymbol{\theta}} = \gamma \lambda^{\boldsymbol{\theta}} \mathbf{z}_{t-1}^{\boldsymbol{\theta}} + \nabla \ln \pi(A_t \mid S_t, \boldsymbol{\theta}),$$

$$\mathbf{w} \leftarrow \mathbf{w} + \alpha^{\mathbf{w}} \delta_t \mathbf{z}_t^{\mathbf{w}},$$

$$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha^{\boldsymbol{\theta}} \delta_t \mathbf{z}_t^{\boldsymbol{\theta}},$$

where $\gamma \in [0, 1)$ is the discount rate, $\lambda^{\mathbf{w}} \in [0,1]$ and $\lambda^{\boldsymbol{\theta}} \in [0,1]$ are bootstrapping parameters for the critic and actor respectively, and $\alpha^{\mathbf{w}} > 0$ and $\alpha^{\boldsymbol{\theta}} > 0$ are step-size parameters.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Non-Contingent vs. Contingent Eligibility Traces)</span></p>

With the critic unit as a single linear neuron ($\hat{v}(s, \mathbf{w}) = \mathbf{w}^\top \mathbf{x}(s)$), each synapse has its own eligibility trace:

* **Non-contingent eligibility traces** (critic): The trace $\mathbf{z}_t^{\mathbf{w}}$ accumulates according to the level of presynaptic activity $\mathbf{x}(S_t)$ and decays toward zero. It depends **only on presynaptic activity** and is not contingent on postsynaptic activity. This corresponds to the TD model of classical conditioning.

* **Contingent eligibility traces** (actor): The trace $\mathbf{z}_t^{\boldsymbol{\theta}}$ depends on the activity of the actor unit itself, in addition to presynaptic activity. It is contingent on postsynaptic activity. The factor $A_t - \pi(1 \mid S_t, \boldsymbol{\theta})$ is positive when $A_t = 1$ (the neuron fires) and negative otherwise.

The **postsynaptic contingency in the eligibility traces of actor units is the only difference between the critic and actor learning rules**. By keeping information about what actions were taken in what states, contingent eligibility traces allow credit for reward (positive $\delta$) or blame for punishment (negative $\delta$) to be apportioned among the policy parameters.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Two-Factor and Three-Factor Learning Rules)</span></p>

* **Two-factor learning rule** (critic): The interaction is between the reinforcement signal $\delta$ and eligibility traces that depend only on presynaptic signals. Neuroscientists call this a two-factor learning rule.
* **Three-factor learning rule** (actor): The eligibility traces depend on both presynaptic and postsynaptic activity, in addition to $\delta$. The relative timing of the factors is critical, with eligibility traces intervening to allow the reinforcement signal to affect synapses that were active in the recent past.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Spike-Timing-Dependent Plasticity)</span></p>

**Spike-timing-dependent plasticity (STDP)** is a form of Hebbian plasticity in which changes in a synapse's efficacy depend on the relative timing of presynaptic and postsynaptic action potentials:

* A synapse **increases** in strength if spikes incoming via that synapse arrive shortly **before** the postsynaptic neuron fires.
* A synapse **decreases** in strength if the presynaptic spike arrives shortly **after** the postsynaptic neuron fires.

**Reward-modulated STDP** is a three-factor form of STDP in which neuromodulatory input (such as dopamine) must follow appropriately-timed pre- and postsynaptic spikes. Synaptic changes occur only if there is neuromodulatory input within a time window (that can last up to 10 seconds) after a presynaptic spike is closely followed by a postsynaptic spike. Evidence is accumulating that reward-modulated STDP occurs at the spines of medium spiny neurons of the dorsal striatum — the sites where actor learning takes place in the hypothetical neural actor–critic architecture.

</div>

### Hedonistic Neurons

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Hedonistic Neuron Hypothesis)</span></p>

In his **hedonistic neuron hypothesis**, Klopf (1972, 1982) conjectured that individual neurons seek to maximize the difference between synaptic input treated as rewarding and synaptic input treated as punishing, by adjusting the efficacies of their synapses on the basis of rewarding or punishing consequences of their own action potentials.

Key features:

* Individual neurons can be trained with response-contingent reinforcement, like an animal in an instrumental conditioning task.
* Synaptically-local traces of past pre- and postsynaptic activity make synapses **eligible** for modification by later reward or punishment — the origin of the term "eligibility trace."
* The shape of a synaptic eligibility trace is like a histogram of the durations of the feedback loops in which the neuron is embedded.
* Klopf extended his hypothesis to argue that intelligent behavior can be understood as the collective behavior of a population of self-interested hedonistic neurons.

</div>

### Collective Reinforcement Learning

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Team Problem / Collective Reinforcement Learning)</span></p>

In multi-agent reinforcement learning, when all agents try to maximize a common reward signal that they simultaneously receive, this is known as a **cooperative game** or a **team problem**.

What makes a team problem challenging is the **structural credit assignment problem**: which team members deserve credit for a favorable reward signal, or blame for an unfavorable one? Each individual agent has only limited ability to affect the common reward signal because any single agent contributes just one component of the collective action.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Requirements for Collective Learning)</span></p>

Two essential requirements for collective learning in a team:

1. **Contingent eligibility traces:** Non-contingent traces do not work in the team setting because they provide no way to correlate actions with consequent changes in the reward signal. Non-contingent traces are adequate for learning to *predict* (critic), but they do not support learning to *control* (actor). Contingent eligibility is an essential but preliminary step in the credit assignment process.

2. **Variability in actions:** Team members must independently explore their own action spaces through persistent variability. A team of Bernoulli-logistic REINFORCE units implements a policy gradient algorithm *as a whole* with respect to the average rate of the team's common reward signal. When interconnected to form a multilayer ANN, such a team learns like a network trained by error backpropagation, but with the broadcasted reward signal replacing the backpropagation process — making it more plausible as a neural mechanism.

</div>

### Model-Based Methods in the Brain

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Neural Substrates of Model-Free and Model-Based Behavior)</span></p>

The actor–critic hypothesis applies to an animal's **habitual mode** of behavior (model-free). For **goal-directed behavior** (model-based), other brain structures play important roles:

* **Dorsolateral striatum (DLS):** Inactivating this region impairs habit learning, causing the animal to rely more on goal-directed processes. Associated with **model-free** processes.
* **Dorsomedial striatum (DMS):** Inactivating this region impairs goal-directed processes, requiring the animal to rely more on habit learning. Associated with **model-based** processes.
* **Orbitofrontal cortex (OFC):** The part of the prefrontal cortex immediately above the eyes, implicated in planning and decision making. Reveals strong activity related to subjective reward value and reward expected as a consequence of actions. May be critical for the **reward part** of an animal's environment model.
* **Hippocampus:** Critical for memory and spatial navigation. Experiments show that at choice points in a maze, the hippocampus sweeps forward along possible paths, suggesting it is part of a system that uses the model to simulate possible future state sequences — a form of **planning**.

The picture remains unclear because model-free and model-based processes do not appear to be neatly separated in the brain. Dopamine signals themselves can exhibit the influence of model-based information in addition to model-free processes.

</div>

### Addiction

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Addiction as Destabilized TD Learning)</span></p>

The reward prediction error hypothesis and TD learning are the basis of a model of addiction due to Redish (2004):

* Administration of cocaine and other addictive drugs produces a transient increase in dopamine, assumed to increase the TD error $\delta$ in a way that **cannot be cancelled out by changes in the value function**.
* Whereas $\delta$ is normally reduced as a normal reward becomes predicted by antecedent events, the contribution to $\delta$ from an addictive stimulus does not decrease — drug rewards cannot be "predicted away."
* This prevents $\delta$ from ever becoming negative when the reward is due to an addictive drug, eliminating the error-correcting feature of TD learning. The values of states associated with drug intake increase without bound, making actions leading to these states preferred above all others.

Reinforcement learning theory has been influential in the development of **computational psychiatry**, which aims to improve understanding of mental disorders through mathematical and computational methods.

</div>

### Summary

The neural pathways involved in the brain's reward system are complex and incompletely understood, but neuroscience research is revealing striking correspondences between the brain's reward system and the theory of reinforcement learning.

| RL Concept | Neuroscience Parallel |
| --- | --- |
| TD error $\delta_t$ | Phasic activity of dopamine neurons |
| Reinforcement signal (not reward signal) | Dopamine neuron responses signal RPEs, not reward itself |
| Actor (policy learning) | Dorsal striatum |
| Critic (value learning) | Ventral striatum |
| Dopamine neurons (SNpc, VTA) | Broadcast reinforcement signal via huge axonal arbors |
| Non-contingent eligibility traces | Critic learning rule (two-factor) |
| Contingent eligibility traces | Actor learning rule (three-factor), reward-modulated STDP |
| Model-free RL | Habitual behavior (DLS) |
| Model-based RL | Goal-directed behavior (DMS, OFC, hippocampus) |
| Team problem / collective RL | Populations of neurons learning under globally-broadcast dopamine |
| Destabilized TD learning | Model of drug addiction (Redish, 2004) |

A remarkable aspect is that the reinforcement learning algorithms and theory that connect so well with properties of the dopamine system were developed from a computational perspective in total absence of any knowledge about dopamine neurons — TD learning and its connections to optimal control and dynamic programming were developed many years before the experiments revealing the TD-like nature of dopamine neuron activity. This unplanned correspondence suggests that the TD error/dopamine parallel captures something significant about brain reward processes.

## Chapter 16: Applications and Case Studies

This chapter presents several case studies of reinforcement learning. These are substantial applications of potential economic significance. The presentations illustrate trade-offs and issues that arise in real applications: how domain knowledge is incorporated into the formulation and solution of the problem, and the representation issues that are so often critical to successful applications.

### TD-Gammon

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(TD-Gammon)</span></p>

**TD-Gammon** (Tesauro, 1992–2002) was a program that learned to play backgammon at near-world-champion level, requiring little backgammon knowledge. The learning algorithm was a straightforward combination of:

* Nonlinear $\text{TD}(\lambda)$ with a multilayer ANN trained by backpropagating TD errors
* Self-play to generate training games

The estimated value $\hat{v}(s, \mathbf{w})$ of any state (board position) $s$ was meant to estimate the probability of winning starting from state $s$. Rewards were zero for all time steps except when the game was won.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(TD-Gammon Architecture and Update Rule)</span></p>

The ANN consisted of an input layer (198 units), a hidden layer (40–160 units), and a single output unit. Hidden unit $j$ computed:

$$h(j) = \sigma\!\left(\sum_i w_{ij} x_i\right) = \frac{1}{1 + e^{-\sum_i w_{ij} x_i}},$$

and the output unit applied the same sigmoid. The semi-gradient $\text{TD}(\lambda)$ update rule was:

$$\mathbf{w}_{t+1} \doteq \mathbf{w}_t + \alpha \Big[R_{t+1} + \gamma \hat{v}(S_{t+1}, \mathbf{w}_t) - \hat{v}(S_t, \mathbf{w}_t)\Big] \mathbf{z}_t,$$

with eligibility traces: $\mathbf{z}_t \doteq \gamma \lambda \mathbf{z}_{t-1} + \nabla \hat{v}(S_t, \mathbf{w}_t)$.

For backgammon ($\gamma = 1$, reward zero except on winning), the TD error was usually just $\hat{v}(S_{t+1}, \mathbf{w}) - \hat{v}(S_t, \mathbf{w})$. Moves were selected by evaluating all $\sim$20 possible **afterstates** (resulting positions) and choosing the one with the highest estimated value.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(TD-Gammon Input Representation)</span></p>

For each of the 24 points on the backgammon board, four units for white and four for black encoded the number of pieces:

* **0 pieces:** all four units = 0
* **1 piece:** first unit = 1 (encodes a "blot" — can be hit)
* **2 pieces:** second unit = 1 (a "made point" — opponent cannot land)
* **3 pieces:** third unit = 1 (a "single spare")
* **$n > 3$ pieces:** fourth unit = $(n-3)/2$ (linear encoding of "multiple spares")

Plus units for pieces on the bar, pieces borne off, and whose turn it is, totaling 198 input units. Each conceptually distinct possibility had one unit, scaled between 0 and 1.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(TD-Gammon Results and Evolution)</span></p>

| Program | Hidden Units | Training Games | Opponents | Results |
| --- | --- | --- | --- | --- |
| TD-Gammon 0.0 | 40 | 300,000 | other programs | tied for best |
| TD-Gammon 1.0 | 80 | 300,000 | Robertie, Magriel, ... | $-13$ pts / 51 games |
| TD-Gammon 2.0 | 40 | 800,000 | various Grandmasters | $-7$ pts / 38 games |
| TD-Gammon 2.1 | 80 | 1,500,000 | Robertie | $-1$ pt / 40 games |
| TD-Gammon 3.0 | 80 | 1,500,000 | Kazaros | $+6$ pts / 20 games |

* **TD-Gammon 0.0** used essentially zero backgammon knowledge, yet matched the best previous programs that used extensive expert knowledge.
* **TD-Gammon 1.0** added hand-crafted backgammon features, becoming substantially better than all previous programs.
* **TD-Gammon 2.0/2.1/3.0** added a selective two-ply or three-ply search procedure (looking ahead to the opponent's possible dice rolls and moves), combining learned value functions with decision-time search as in MCTS.
* TD-Gammon 3.0 appeared to play at close to, or possibly better than, the best human players in the world. It also changed how the best human players play certain opening positions.

</div>

### Samuel's Checkers Player

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Samuel's Checkers Player)</span></p>

Arthur Samuel (1959, 1967) was one of the first to make effective use of heuristic search methods and temporal-difference learning. His checkers-playing programs played by performing a lookahead search from each position, evaluating terminal positions with a **linear function approximation** ("scoring polynomial"), and using a **minimax procedure** (backed-up score) to find the best move.

Samuel used two main learning methods:

1. **Rote learning:** Saved board positions with their backed-up values, effectively amplifying search depth when positions reappeared. Samuel added a "discounting" trick — decreasing a position's value as it was backed up a level — which he found essential to successful learning.

2. **Learning by generalization:** Played the program against itself, updating the value function parameters after each move toward the minimax value of a search from the next position. This is conceptually the same as TD learning — the method used much later by Tesauro in TD-Gammon.

A potential problem: Samuel's learning procedure was not constrained to find useful evaluation functions. It could become worse with experience, and Samuel reported observing this during self-play training. His fix was to set the weight with the largest absolute value back to zero.

Despite these issues, Samuel's checkers player using the generalization method approached "better-than-average" play. The program was widely recognized as a significant achievement in AI and machine learning.

</div>

### Watson's Daily-Double Wagering

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Watson's Daily-Double Wagering)</span></p>

IBM Watson's winning *Jeopardy!* strategy relied on a Daily-Double (DD) wagering strategy adapted from TD-Gammon. Watson selected bets by comparing **action values** $\hat{q}(s, bet)$ that estimated the probability of winning for each legal bet from the current game state $s$:

$$\hat{q}(s, bet) = p_{DD} \times \hat{v}(S_W + bet, \ldots) + (1 - p_{DD}) \times \hat{v}(S_W - bet, \ldots),$$

where $S_W$ is Watson's current score, $\hat{v}$ is a state-value function learned by nonlinear $\text{TD}(\lambda)$ via self-play against carefully-crafted models of human players, and $p_{DD}$ is the in-category DD confidence (estimated likelihood of responding correctly).

Key design decisions:

* **Risk abatement:** Subtracted a fraction of the standard deviation over correct/incorrect afterstate evaluations, and prohibited bets that would cause the wrong-answer afterstate value to decrease below a certain limit.
* **Self-play was not used** to learn $\hat{v}$ (unlike TD-Gammon), because Watson was so different from human contestants. Instead, millions of simulated games were played against models of human players extracted from a fan-created archive of nearly 300,000 clues.
* **Endgame improvement:** Near the game's end, Monte-Carlo trials (simulating play to the game's end) produced better wagering decisions than the ANN, significantly improving Watson's performance.

Watson's win rate improved from 61% (baseline heuristic) to 64% (learned values) to 67% (with live in-category DD confidence).

</div>

### Optimizing Memory Control

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(RL-Based DRAM Memory Controller)</span></p>

İpek, Mutlu, Martínez, and Caruana (2008) designed a reinforcement learning memory controller for DRAM that significantly improved the speed of program execution. They modeled DRAM access scheduling as an **MDP**:

* **States:** Contents of the memory transaction queue
* **Actions:** Commands to the DRAM system — *precharge*, *activate*, *read*, *write*, and *NoOp*
* **Reward:** 1 whenever the action is *read* or *write*, 0 otherwise
* **Action constraints:** $A_t \in \mathcal{A}(S_t)$ — only actions that do not violate timing and resource constraints are allowed, ensuring the system's integrity

The scheduling agent used **Sarsa** (Section 6.4) to learn an action-value function with **linear function approximation** implemented by **tile coding with hashing** (32 tilings, 256 action values each as 16-bit fixed point numbers). Exploration was $\varepsilon$-greedy with $\varepsilon = 0.05$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(DRAM Controller Results)</span></p>

Key results and design features:

* The RL controller improved over the best conventional controller (FR-FCFS) by **7% to 33%** across nine memory-intensive benchmark applications, with an average improvement of **19%**.
* The RL controller closed the gap with the unrealizable ideal (Optimistic) controller's upper bound by an impressive **27%**.
* **Online learning** was 8% better than a fixed pre-learned policy, demonstrating the value of adapting to changing workloads.
* The complete controller and learning algorithm were designed to be implemented **directly on a processor chip**, with two five-stage pipelines calculating and comparing two action values at every processor clock cycle.
* The tile coding features for the action-value function were **different** from the features used to specify action-constraint sets $\mathcal{A}(S_t)$, highlighting that different aspects of the state are relevant for different parts of the problem.

</div>

### Human-Level Video Game Play

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Deep Q-Network (DQN))</span></p>

Mnih et al. (2013, 2015) developed a reinforcement learning agent called **deep Q-network (DQN)** that combined Q-learning with a **deep convolutional ANN**. DQN demonstrated that a single learning system, using the same raw input, network architecture, and hyperparameters, could achieve human-level performance on a large fraction of 49 different Atari 2600 video games.

DQN's architecture:

* **Input:** $84 \times 84 \times 4$ preprocessed image stacks (4 most recent frames of $84 \times 84$ luminance values)
* **Three hidden convolutional layers:** $32$ filters of $20 \times 20$, $64$ filters of $9 \times 9$, $64$ filters of $7 \times 7$ (with ReLU activations)
* **One fully connected hidden layer:** 512 units
* **Output layer:** one unit for each possible action in the Atari game (4–18 actions)

Each output unit's activation level was the estimated optimal action value $\hat{q}(s, a, \mathbf{w})$ for the corresponding state–action pair.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(DQN Key Innovations)</span></p>

DQN modified standard Q-learning in three important ways to improve stability:

1. **Experience replay** (Lin, 1992): Stored each transition $(S_t, A_t, R_{t+1}, S_{t+1})$ in a replay memory. At each time step, multiple Q-learning updates were performed on **mini-batches** of 32 experiences sampled uniformly at random from the replay memory. This broke correlations between successive updates, reduced variance, and allowed each experience to be reused many times.

2. **Target network:** Every $C$ updates, the current network weights $\mathbf{w}$ were copied into a separate **target network** with weights held fixed for the next $C$ updates. Q-learning targets used this frozen network:

$$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha \Big[R_{t+1} + \gamma \max_a \bar{q}(S_{t+1}, a, \mathbf{w}_t) - \hat{q}(S_t, A_t, \mathbf{w}_t)\Big] \nabla \hat{q}(S_t, A_t, \mathbf{w}_t),$$

where $\bar{q}$ denotes the output of the target network. This stabilized learning by removing the dependence of the targets on the parameters being updated.

3. **Error clipping:** The TD error $R_{t+1} + \gamma \max_a \bar{q}(S_{t+1}, a, \mathbf{w}_t) - \hat{q}(S_t, A_t, \mathbf{w}_t)$ was clipped to $[-1, 1]$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(DQN Results and Significance)</span></p>

* DQN learned on each game by interacting with the game emulator for 50 million frames ($\approx$38 days of experience). Scores were averaged over 30 sessions of up to 5 minutes each.
* DQN reached or exceeded **human-level play on 29 of 46 games** (considering $\geq 75\%$ of human score as comparable).
* DQN played better than the best previous RL systems on all but 6 of the games, and beat the human tester on 22 of them.
* The reward signal was standardized: $+1$ when the game score increased, $-1$ when it decreased, $0$ otherwise. This made a single step-size parameter work across all games.
* DQN used $\varepsilon$-greedy exploration with $\varepsilon$ decreasing linearly over the first million frames.
* The **same architecture and hyperparameters** were used for all 49 games — no game-specific modifications.
* Games requiring deep planning (e.g. Montezuma's Revenge) remained very difficult for DQN.
* Ablation studies showed that experience replay and the target network **each** significantly improved performance, with the combination yielding the best results. The deep convolutional network was also critical compared to a single linear layer.

</div>

### Mastering the Game of Go

Go's search space is significantly larger than chess ($\approx 250$ legal moves per position vs. $\approx 35$; $\approx 150$ moves per game vs. $\approx 80$). The major stumbling block is the difficulty of defining an adequate position evaluation function. The major step forward was the introduction of **Monte Carlo tree search (MCTS)**.

#### AlphaGo

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(AlphaGo)</span></p>

**AlphaGo** (Silver et al., 2016) combined deep ANNs with a novel version of MCTS called **APV-MCTS** (asynchronous policy and value MCTS). The key innovation was guiding MCTS with both a **policy** and a **value function** learned by reinforcement learning with deep convolutional ANNs.

AlphaGo used four networks in its pipeline:

1. **SL policy network:** A 13-layer deep convolutional ANN trained by supervised learning to predict human expert moves from a database of nearly 30 million moves. Achieved 57% accuracy (vs. 44.4% state-of-the-art at the time).

2. **RL policy network:** Same architecture, initialized with SL policy network weights, then improved by **policy-gradient reinforcement learning** via self-play. Won $>80\%$ of games against the SL policy and 85% against a strong MCTS Go program.

3. **Value network:** Same architecture with a single output unit, trained by **Monte Carlo policy evaluation** on 30 million positions from self-play games to estimate the probability of winning from each position.

4. **Rollout policy:** A fast, simple linear network trained by supervised learning on 8 million human moves, used for fast rollout simulations ($\sim$1,000 complete games per second per thread).

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(AlphaGo's APV-MCTS)</span></p>

APV-MCTS differed from basic MCTS in two key ways:

1. **Expansion guided by SL policy:** Instead of stored action values, the SL policy network's probabilities guided edge selection during tree expansion.

2. **Evaluation mixing rollouts with value network:** Newly-added leaf nodes were evaluated by combining the value network estimate $v_\theta(s)$ with the rollout return $G$:

$$v(s) = (1 - \eta) v_\theta(s) + \eta G,$$

where $\eta$ controlled the mixing. The best play resulted from $\eta = 0.5$, indicating that the value network and rollouts **complemented each other**: the value network evaluated the high-performance RL policy (too slow for rollouts), while the fast rollout policy added precision for specific positions during games.

Interestingly, AlphaGo played better when APV-MCTS used the SL policy (rather than the stronger RL policy) for expansion, because the SL policy was tuned to the broader set of human moves. However, the value function derived from the RL policy performed better.

</div>

#### AlphaGo Zero

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(AlphaGo Zero)</span></p>

**AlphaGo Zero** (Silver et al., 2017a) used **no human data or guidance beyond the basic rules of the game**. It learned exclusively from self-play reinforcement learning, implementing a form of **policy iteration** (interleaving policy evaluation with policy improvement).

Key differences from AlphaGo:

* **Single "two-headed" network** $f_\theta$: A deep convolutional ANN whose input was a raw $19 \times 19 \times 17$ image stack (8 feature planes each for current and opponent stone positions over 7 past board configurations + 1 color plane). It output both:
  * A vector $\mathbf{p}$ of $19^2 + 1 = 362$ move probabilities
  * A scalar $v$ estimating the win probability
* The network had **41 convolutional layers** with batch normalization and residual (skip) connections, computing move probabilities through 43 layers and values through 44 layers.
* **No rollouts** — MCTS simulations ended at leaf nodes, evaluated by the network. No rollout policy was needed.
* **No human features** — input was raw board representations only.
* MCTS was used as a **policy improvement operator**: the policy $\pi_i$ returned by MCTS (based on network output $\mathbf{p}$) was better than the network's raw policy.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(AlphaGo Zero Training and Results)</span></p>

Training procedure:

* Trained over **4.9 million self-play games** in about 3 days.
* Each move selected by running MCTS for **1,600 iterations** ($\approx$0.4 seconds per move).
* Network weights updated over 700,000 batches of 2,048 board configurations.
* At every 1,000 training steps, the new policy was tested against the current best in 400 games. If the new policy won by a margin, it replaced the current best for subsequent self-play.

Results:

* AlphaGo Zero (Elo 4,308) defeated the version of AlphaGo that beat Fan Hui 5–0, and defeated the version that beat Lee Sedol 4–1, winning **100 out of 100 games** against the latter.
* A supervised-learning player initially predicted human moves better, but played less well after AlphaGo Zero was trained for just one day.
* AlphaGo Zero **discovered novel variations** of classical move sequences, developing a strategy different from how humans play.
* A larger version trained for 40 days (Elo 5,185) defeated **AlphaGo Master** (Elo 4,858, which had beaten top human professionals 60–0) by a score of **89–11**.
* A further generalization, **AlphaZero** (Silver et al., 2017b), does not even incorporate knowledge of Go — it is a general RL algorithm that improves over the world's best programs in Go, chess, and shogi.

</div>

### Personalized Web Services

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Personalized Web Services)</span></p>

Reinforcement learning can improve personalized recommendation policies. Two key approaches:

* **Contextual bandits / A/B testing:** Li, Chu, Langford, and Schapire (2010) applied a contextual bandit algorithm to personalizing the Yahoo! Front Page Today, maximizing **click-through rate (CTR)** — improving over a non-associative bandit by 12.5%.

* **Life-time value (LTV) optimization:** Theocharous, Thomas, and Ghavamzadeh (2015) at Adobe formulated personalized recommendation as a full **MDP**, maximizing clicks over repeated visits by the same user rather than treating each visit independently:

$$\text{CTR} = \frac{\text{Total } \# \text{ of Clicks}}{\text{Total } \# \text{ of Visits}}, \quad \text{LTV} = \frac{\text{Total } \# \text{ of Clicks}}{\text{Total } \# \text{ of Visitors}}.$$

LTV is larger than CTR to the extent that individual users revisit the site. LTV optimization used **fitted Q iteration (FQI)**, a batch-mode RL algorithm adapted from Q-learning where the entire data set is available from the start. As expected, greedy optimization was best by CTR, while **LTV optimization was best by LTV** — demonstrating that RL policies designed for long-term engagement outperform greedy policies.

A critical aspect was **high confidence off-policy evaluation** to assess new policies from data collected under different policies before deployment.

</div>

### Thermal Soaring

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Thermal Soaring)</span></p>

Reddy, Celani, Sejnowski, and Vergassola (2016) used reinforcement learning to investigate thermal soaring policies — gaining altitude by exploiting rising columns of turbulent air. The problem was modeled as a **continuing MDP with discounting**:

* **Actions:** $3^2 = 9$ possible actions (increment, decrement, or hold each of bank angle and angle of attack by $5°$ and $2.5°$ respectively).
* **Reward:** A linear combination of vertical wind velocity and vertical wind acceleration observed on the previous time step.
* **Algorithm:** One-step **Sarsa** with **state aggregation** (four-dimensional state space discretized into three bins each) and a **soft-max action selection** based on normalized action-value preferences.

Key findings:

* Only **two sensory cues** — vertical wind acceleration and torques (difference in vertical wind velocities at the two wing tips) — were sufficient for effective soaring. These features give information about the gradient of vertical wind velocity in two directions, allowing the controller to select between turning and flying straight.
* The learned policy had the glider spiral within rising columns of air to gain altitude.
* Effective soaring required a high discount factor ($\gamma = 0.99$), emphasizing long-term effects of control decisions.
* Different turbulence levels led to different policies: strong turbulence favored conservative small bank angles, while weak turbulence favored sharp banking.

</div>

### Summary

| Application | Algorithm | Key Innovation | Result |
| --- | --- | --- | --- |
| TD-Gammon | Nonlinear $\text{TD}(\lambda)$ + self-play | ANN function approximation with afterstate evaluation | Near-world-champion backgammon |
| Samuel's Checkers | TD-like + minimax search | First use of heuristic search + TD learning | "Better-than-average" play |
| Watson (Jeopardy!) | Nonlinear $\text{TD}(\lambda)$ + opponent models | Risk-abated action-value wagering; endgame Monte Carlo | Won against human champions |
| DRAM Controller | Sarsa + tile coding | On-chip RL implementation; action constraints for safety | 7–33% speedup over best conventional |
| DQN (Atari) | Q-learning + deep conv. ANN | Experience replay + target network + error clipping | Human-level on 29/46 games |
| AlphaGo | Policy gradient + MCTS + deep ANN | APV-MCTS combining SL policy, RL policy, and value network | Beat world champion at Go |
| AlphaGo Zero | Self-play RL + MCTS | No human data; single two-headed network; MCTS as policy improvement | Defeated AlphaGo 100–0 |
| Web Services | Contextual bandits / FQI | LTV optimization over repeated visits; off-policy evaluation | Improved long-term engagement |
| Thermal Soaring | Sarsa + state aggregation | Identified minimal sensory cues for soaring | Learned spiral soaring behavior |

## Chapter 17: Frontiers

This final chapter touches on topics beyond the scope of the book that are particularly important for the future of reinforcement learning. Many bring us beyond what is reliably known, and some bring us beyond the MDP framework.

### General Value Functions and Auxiliary Tasks

Over the course of the book, the notion of value function has become quite general. With off-policy learning we allowed a value function to be conditional on an arbitrary target policy. In Section 12.8 we generalized discounting to a *termination function* $\gamma : \mathcal{S} \mapsto [0, 1]$, so that a different discount rate could be applied at each time step in determining the return. The next step is to generalize beyond rewards to permit predictions about arbitrary signals.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(General Value Function)</span></p>

Whatever signal is added up in a value-function-like prediction, we call it the **cumulant** of that prediction. We formalize it in a **cumulant signal** $C_t \in \mathbb{R}$. Using this, a **general value function** (GVF) is written

$$v_{\pi, \gamma, C}(s) \doteq \mathbb{E}\!\left[\sum_{k=t}^{\infty} \left(\prod_{i=t+1}^{k} \gamma(S_i)\right) C_{k+1} \;\middle|\; S_t = s,\, A_{t:\infty} \sim \pi\right].$$

Because a GVF has no necessary connection to reward, it is perhaps a misnomer to call it a *value* function. You might simply call it a prediction or, to make it more distinctive, a **forecast**.

</div>

GVFs are in the form of a value function and thus can be learned in the usual ways using the methods developed in this book. Along with the learned predictions, we might also learn policies to maximize the predictions by Generalized Policy Iteration (Section 4.6) or by actor–critic methods. In this way an agent could learn to predict and control great numbers of signals, not just long-term reward.

**Auxiliary tasks** are extra tasks (in addition to) the main task of maximizing reward. The ability to predict and control a diverse multitude of signals can constitute a powerful kind of environmental model. Two simpler ways in which a multitude of diverse predictions can be helpful to a reinforcement learning agent:

1. **Shared representations.** Auxiliary tasks may require some of the same representations as needed on the main task. Some of the auxiliary tasks may be easier, with less delay and a clearer connection between actions and outcomes. If good features can be found early on easy auxiliary tasks, then those features may significantly speed learning on the main task.

2. **Multi-headed learning.** An artificial neural network (ANN) in which the last layer is split into multiple parts, or *heads*, each working on a different task. One head might produce the approximate value function for the main task (with reward as its cumulant) whereas the others produce solutions to various auxiliary tasks. All heads propagate errors by stochastic gradient descent into the same body — the shared preceding part of the network — which would then try to form representations, in its next-to-last layer, to support all the heads.

3. **Pavlovian control.** Analogous to classical conditioning (Section 14.2): evolution has built in a reflexive (non-learned) association to a particular action from the prediction of a particular signal. Agent designers can do something similar, connecting by design (without learning) predictions of specific events to predetermined actions.

Finally, perhaps the most important role for auxiliary tasks is in moving beyond the assumption that the state representation is fixed and given to the agent (see Section 17.3).

### Temporal Abstraction via Options

An appealing aspect of the MDP formalism is that it can be applied usefully to tasks at many different time scales — from deciding which muscles to twitch to which job to take to lead a satisfying life. All involve interaction with the world, sequential decision making, and a goal usefully conceived of as accumulating rewards over time.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Option)</span></p>

An **option** is a generalized notion of action defined as a pair $\omega = \langle \pi_\omega, \gamma_\omega \rangle$, where $\pi_\omega$ is a policy and $\gamma_\omega$ is a state-dependent termination function.

To execute an option $\omega = \langle \pi_\omega, \gamma_\omega \rangle$ at time $t$: obtain the action $A_t$ from $\pi_\omega(\cdot | S_t)$, then terminate at time $t+1$ with probability $1 - \gamma_\omega(S_{t+1})$. If the option does not terminate, then $A_{t+1}$ is selected from $\pi_\omega(\cdot | S_{t+1})$, and so on until eventual termination.

It is convenient to consider low-level actions to be special cases of options — each action $a$ corresponds to an option $\langle \pi_\omega, \gamma_\omega \rangle$ whose policy picks the action $a$ ($\pi_\omega(s) = a$ for all $s \in \mathcal{S}$) and whose termination function is zero ($\gamma_\omega(s) = 0$ for all $s \in \mathcal{S}^+$). Options effectively extend the action space.

</div>

Options are designed to be interchangeable with low-level actions. The notion of an action-value function $q_\pi$ naturally generalizes to an **option-value function** that takes a state and option as input and returns the expected return starting from that state, executing that option to termination, and thereafter following the policy $\pi$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Hierarchical Policy)</span></p>

A **hierarchical policy** selects from options rather than actions, where options, when selected, execute until termination. With these ideas, many of the algorithms in this book can be generalized to learn approximate option-value functions and hierarchical policies.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Option Models)</span></p>

The conventional model of an action is the state-transition probabilities and the expected immediate reward for taking the action in each state. These generalize to **option models**. For options, the appropriate model has two parts: one corresponding to the state transition resulting from executing the option and one corresponding to the expected cumulative reward along the way.

The **reward part** of an option model, analogous to the expected reward for state–action pairs, is:

$$r(s, \omega) \doteq \mathbb{E}\bigl[R_1 + \gamma R_2 + \gamma^2 R_3 + \cdots + \gamma^{\tau - 1} R_\tau \mid S_0 = s,\, A_{0:\tau-1} \sim \pi_\omega,\, \tau \sim \gamma_\omega\bigr],$$

for all options $\omega$ and all states $s \in \mathcal{S}$, where $\tau$ is the random time step at which the option terminates according to $\gamma_\omega$.

The **state-transition part** of an option model is:

$$p(s' \mid s, \omega) \doteq \sum_{k=1}^{\infty} \gamma^k \Pr\lbrace S_k = s',\, \tau = k \mid S_0 = s,\, A_{0:k-1} \sim \pi_\omega,\, \tau \sim \gamma_\omega\rbrace.$$

Note that, because of the factor of $\gamma^k$, this $p(s' \mid s, \omega)$ is no longer a transition probability and no longer sums to one over all values of $s'$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Bellman Equations for Options)</span></p>

The general Bellman equation for the state values of a hierarchical policy $\pi$ is:

$$v_\pi(s) = \sum_{\omega \in \Omega(s)} \pi(\omega | s) \left[ r(s, \omega) + \sum_{s'} p(s' \mid s, \omega) v_\pi(s') \right],$$

where $\Omega(s)$ denotes the set of options available in state $s$. If $\Omega(s)$ includes only the low-level actions, this equation reduces to the usual Bellman equation (3.14), except that $\gamma$ is included in the new $p$ (17.3) and thus does not appear separately.

The value iteration algorithm with options, analogous to (4.10), is:

$$v_{k+1}(s) \doteq \max_{\omega \in \Omega(s)} \left[ r(s, \omega) + \sum_{s'} p(s' \mid s, \omega) v_k(s') \right], \quad \text{for all } s \in \mathcal{S}.$$

If $\Omega(s)$ includes all the low-level actions, this algorithm converges to the conventional $v_*$, from which the optimal policy can be computed. When only a subset of options are considered in each state, value iteration will converge to the best hierarchical policy limited to the restricted set.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Learning Option Models with GVFs)</span></p>

One natural way to learn an option model is to formulate it as a collection of GVFs and then learn the GVFs using the methods presented in this book.

* **Reward part:** Choose one GVF's cumulant to be the reward ($C_t = R_t$), its policy to be the option's policy ($\pi = \pi_\omega$), and its termination function to be the discount rate times the option's termination function ($\gamma(s) = \gamma \cdot \gamma_\omega(s)$). Then $v_{\pi,\gamma,C}(s) = r(s, \omega)$.

* **State-transition part:** Allocate one GVF for each state that the option might terminate in. Choose the cumulant to be $C_t = (1 - \gamma_\omega(S_t))\mathbb{1}_{S_t = s'}$. The GVF's policy and termination functions are the same as for the reward part. Then $v_{\pi,\gamma,C}(s) = p(s' \mid s, \omega)$.

</div>

### Observations and State

Throughout this book we have written the learned approximate value functions (and the policies in Chapter 13) as functions of the environment's state. This is a significant limitation: in many cases of interest the sensory input gives only partial information about the state of the world. Potentially important aspects of the environment's state are not directly observable, and it is a strong, unrealistic, and limiting assumption to assume that the learned value function is implemented as a table over the environment's state space.

The framework of parametric function approximation (Part II) is far less restrictive: we retained the assumption that the learned value functions are functions of the environment's state, but allowed these functions to be arbitrarily restricted by the parameterization. If there is a state variable that is not observable, then the parameterization can be chosen such that the approximate value does not depend on that state variable. Because of this, all the results obtained for the parameterized case apply to partial observability without change.

**Four steps** to extend reinforcement learning to partial observability:

**Step 1.** Change the problem formulation. The environment would emit not its states, but only *observations* — signals that depend on its state but provide only partial information. The environmental interaction would then be an alternating sequence of actions $A_t \in \mathcal{A}$ and observations $O_t \in \mathcal{O}$:

$$A_0, O_1, A_1, O_2, A_2, O_3, A_3, O_4, \ldots$$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(History and State)</span></p>

**Step 2.** Recover the idea of state from the sequence of observations and actions. The **history** $H_t$ is the initial portion of the trajectory up to an observation:

$$H_t \doteq A_0, O_1, \ldots, A_{t-1}, O_t.$$

A state must be a function of history, $S_t = f(H_t)$. The summary would be informationally perfect if it retained all information about the history.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Markov State and Markov Property)</span></p>

The state $S_t$ and the function $f$ are said to have the **Markov property**, and $S_t$ is a **Markov state** if and only if, for any test $\tau$ (a specific sequence of alternating actions and observations), and for any two histories $h$ and $h'$ that map to the same state under $f$, the test's probabilities given the two histories are equal:

$$f(h) = f(h') \;\Rightarrow\; p(\tau | h) = p(\tau | h'), \qquad \text{for all } h, h', \tau \in \lbrace\mathcal{A} \times \mathcal{O}\rbrace^*.$$

where $p(\tau | h) \doteq \Pr\lbrace O_{t+1} = o_1, O_{t+2} = o_2, O_{t+3} = o_3 \mid H_t = h, A_t = a_1, A_{t+1} = a_2, A_{t+2} = a_3\rbrace$ for a three-step test $\tau = a_1 o_1 a_2 o_2 a_3 o_3$.

A Markov state summarizes all the information in the history necessary for determining any test's probability — and thus all that is necessary for making any prediction (including any GVF) and for optimal behavior.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(State-Update Function)</span></p>

**Step 3.** Deal with computational considerations. We want the state to be *compact* — relatively small compared to the history. Instead of a function $f$ that takes whole histories, we want an $f$ that can be compactly implemented with an incremental, recursive update that computes $S_{t+1}$ from $S_t$, incorporating only the next increment of data, $A_t$ and $O_{t+1}$:

$$S_{t+1} \doteq u(S_t, A_t, O_{t+1}), \quad \text{for all } t \ge 0,$$

with the first state $S_0$ given. The function $u$ is called the **state-update function**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(One-Step Predictions and the Markov Property)</span></p>

An important fact is that if an $f$ is incrementally updatable, then it is Markov if and only if all one-step tests can be accurately predicted — that is, if and only if

$$f(h) = f(h') \;\Rightarrow\; \Pr\lbrace O_{t+1} = o \mid H_t = h, A_t = a\rbrace = \Pr\lbrace O_{t+1} = o \mid H_t = h', A_t = a\rbrace,$$

for all $h, h' \in \lbrace\mathcal{A} \times \mathcal{O}\rbrace^*$, $o \in \mathcal{O}$, and $a \in \mathcal{A}$.

Accurate one-step predictions are informationally sufficient, together with the state-update function, to accurately predict the probability of any test of any length. However, determining long-term predictions from single-step predictions is exponentially complex. Moreover, if there is any error or approximation in the one-step predictions, then it can compound to make the long-term predictions wildly inaccurate.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(POMDP and Belief State)</span></p>

In **Partially Observable MDPs** (POMDPs), the environment is assumed to have a well-defined latent state $X_t$ that underlies and produces the observations, but is never available to the agent. The natural Markov state $S_t$ for a POMDP is the *distribution* over the latent states given the history, called the **belief state**.

For concreteness, assume a finite number of hidden states, $X_t \in \lbrace 1, 2, \ldots, d \rbrace$. Then the belief state is the vector $S_t \doteq \mathbf{s}_t \in [0, 1]^d$ with components

$$\mathbf{s}_t[i] \doteq \Pr\lbrace X_t = i \mid H_t\rbrace, \quad \text{for all possible latent states } i \in \lbrace 1, 2, \ldots, d\rbrace.$$

The belief state remains the same size (same number of components) even as $t$ grows. It can be incrementally updated by Bayes' rule:

$$u(\mathbf{s}, a, o)[i] = \frac{\sum_{x=1}^{d} \mathbf{s}[x]\, p(i, o | x, a)}{\sum_{x=1}^{d} \sum_{x'=1}^{d} \mathbf{s}[x]\, p(x', o | x, a)}, \quad \text{for } a \in \mathcal{A},\ o \in \mathcal{O},$$

where $p(x', o | x, a) \doteq \Pr\lbrace X_t = x', O_t = o \mid X_{t-1} = x, A_{t-1} = a\rbrace$ is the latent-state model.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Predictive State Representations)</span></p>

**Predictive State Representations** (PSRs) address the weakness that POMDP agent state $S_t$ is grounded in the environment state $X_t$, which is never observed and thus is difficult to learn about. In PSRs and related approaches, the semantics of the agent state is instead grounded in predictions about future observations and actions, which are readily observable.

In PSRs, a Markov state is defined as a $d$-vector of the probabilities of $d$ specially chosen "core" tests. The vector is then updated by a state-update function $u$ that is analogous to Bayes rule, but with a semantics grounded in observable data.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Approximate State)</span></p>

**Step 4.** Re-introduce approximation. To approach artificial intelligence ambitiously we must embrace approximation. We must accept and work with an approximate notion of state, even though it may not be Markov.

Perhaps the simplest example of an approximate state is just the latest observation, $S_t \doteq O_t$. A better approach is to use the last $k$ observations and actions, $S_t \doteq O_t, A_{t-1}, O_{t-1}, \ldots, A_{t-k}$, for some $k \ge 1$, which can be achieved by a state-update function that just shifts the new data in and the oldest data out. This **$k$th-order history** approach is simple, but can greatly increase the agent's capabilities.

The general idea is that a state that is good for some predictions is also good for others — in particular, that a Markov state, sufficient for one-step predictions, is also sufficient for all others. Taken together with the discussion in Section 17.1, these ideas suggest an approach to both partial observability and representation learning in which multiple predictions are pursued and used to direct the construction of state features.

</div>

### Designing Reward Signals

A major advantage of reinforcement learning over supervised learning is that reinforcement learning does not rely on detailed instructional information: generating a reward signal does not depend on knowledge of what the agent's correct actions should be. But the success of a reinforcement learning application strongly depends on how well the reward signal frames the goal and how well the signal assesses progress in reaching that goal.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Sparse Reward Problem)</span></p>

Even when there is a simple and easily identifiable goal, the problem of **sparse reward** often arises. Delivering non-zero reward frequently enough to allow the agent to achieve the goal once, let alone to learn to achieve it efficiently from multiple initial conditions, can be a daunting challenge. The agent may wander aimlessly for long periods of time (the **"plateau problem"**, Minsky, 1961).

</div>

Approaches to designing reward signals:

* **Augmenting the value function.** Rather than modifying the reward signal with supplemental rewards for subgoals (which may lead the agent to behave differently from what is intended), a better approach is to leave the reward signal alone and instead augment the value-function approximation with an initial guess $v_0 : \mathcal{S} \to \mathbb{R}$ of what it should ultimately be:

$$\hat{v}(s, \mathbf{w}) \doteq \mathbf{w}^\top \mathbf{x}(s) + v_0(s),$$

and update the weights $\mathbf{w}$ as usual. If the initial weight vector is $\mathbf{0}$, then the initial value function will be $v_0$.

* **Shaping.** A technique introduced by the psychologist B. F. Skinner. Involves changing the reward signal as learning proceeds, starting from a reward signal that is not sparse given the agent's initial behavior, and gradually modifying it toward a reward signal suited to the problem of original interest. At each stage, the agent is frequently rewarded given its current behavior.

* **Imitation / apprenticeship learning.** If there is another agent (e.g. a human expert) already expert at the task, the learner can benefit from the expert agent. Learning from an expert's behavior can be done either by learning directly by supervised learning or by extracting a reward signal using **inverse reinforcement learning** and then using a reinforcement learning algorithm with that reward signal to learn a policy.

* **Automated reward search.** From an application perspective, the reward signal is a parameter of the learning algorithm. The search for a good reward signal can be automated by defining a space of feasible candidates and applying an optimization algorithm. The optimization evaluates each candidate reward signal by running the reinforcement learning system with that signal for some number of steps, and then scoring the overall result by a "high-level" objective function intended to encode the designer's true goal.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Reward Signal Sensitivity)</span></p>

Computational experiments with the bilevel optimization approach have confirmed that intuition alone is not always adequate to devise a good reward signal. The performance of a reinforcement learning agent can be very sensitive to details of the agent's reward signal in subtle ways determined by the agent's limitations and the environment in which it acts and learns.

An agent's goal should not always be the same as the goal of the agent's designer. Under various constraints (limited computational power, limited access to information, limited time to learn), learning to achieve a goal that is *different* from the designer's goal can sometimes end up getting closer to the designer's goal than if that goal were pursued directly. Evolution, for example, gave us a reward signal (certain tastes) rather than the objective function itself (nutritional value) because of our limited sensory abilities and the risks of trial and error.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Intrinsically-Motivated Reinforcement Learning)</span></p>

A reinforcement learning agent is not necessarily like a complete organism or robot; it can be a component of a larger behaving system. Reward signals may be influenced by things inside the larger behaving agent, such as motivational states, memories, ideas, or even hallucinations. Reward signals may also depend on properties of the learning process itself, such as measures of how much progress learning is making. This leads to the idea of **"intrinsically-motivated reinforcement learning"**.

</div>

### Remaining Issues

Six further issues that remain to be addressed by future research:

1. **Scalable incremental function approximation.** Current deep learning methods struggle to learn rapidly in incremental, online settings. The problem of "catastrophic interference" or "correlated data" — when something new is learned it tends to replace what has previously been learned. Techniques such as "replay buffers" are often used but are not well suited to online learning.

2. **Representation learning / constructive induction / meta-learning.** Methods for learning features such that subsequent learning generalizes well. How can we use experience not just to learn a given desired function, but to learn inductive biases such that future learning generalizes better and is thus faster? The problem of representation learning in reinforcement learning can be identified with the problem of learning the state-update function discussed in Section 17.3.

3. **Planning with learned environment models.** Cases of full model-based reinforcement learning, in which the environment model is learned from data and then used for planning, are rare. The learning of the model needs to be selective because the scope of a model strongly affects planning efficiency. Environment models should be constructed judiciously with the goal of optimizing the planning process.

4. **Automating task selection.** Automating the choice of tasks on which an agent works to structure its developing competence. Currently it is usual for human designers to set the tasks that the learning agent is expected to master. Looking ahead, we want the agent to make its own choices about what tasks to master — GVFs, options, and the cumulant/policy/termination function within them.

5. **Curiosity and intrinsic reward.** The interaction between behavior and learning via some computational analog of *curiosity*. When reward is not available or not strongly influenced by behavior, the agent can use some measure of learning progress as an internal or "intrinsic" reward. This implements a computational form of curiosity — a computational analog of *play*.

6. **Safety of embedded agents.** Developing methods to make it acceptably safe to embed reinforcement learning agents into physical environments. Risk management for embedded reinforcement learning is similar to what control engineers have had to confront: a highly-developed body of theory aimed at ensuring convergence and stability of adaptive controllers. One of the most pressing areas for future research is to adapt and extend methods developed in control engineering.

### Reinforcement Learning and the Future of Artificial Intelligence

By the time of the second edition, the promise of AI has transitioned to applications that are changing the lives of millions of people. Some of the most remarkable developments in artificial intelligence have involved reinforcement learning, most notably "deep reinforcement learning" — reinforcement learning with function approximation by deep artificial neural networks.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Safety and the Reward Design Problem)</span></p>

Reinforcement learning is based on optimization, and inherits the plusses and minuses of all optimization methods. The problem of devising objective functions or reward signals such that optimization produces the desired results while avoiding undesirable results is a longstanding challenge. As Wiener, the founder of cybernetics, warned: "... it grants you what you ask for, not what you have asked for or what you intend" (Wiener, 1964).

Key safety challenges include:
* Reinforcement learning agents can discover unexpected ways to make their environments deliver reward, some of which might be undesirable or even dangerous.
* It may be impossible for the agent to achieve the designer's goal no matter what its reward signal is, due to constraints such as limited computational power, limited access to information, or limited time to learn.
* Ensuring safe behavior *while learning* is a distinct challenge from what the agent might learn *eventually*. Risk management and mitigation for embedded reinforcement learning is an active area of research.

Approaches that have been developed include adding hard and soft constraints to optimization, restricting optimization to robust and risk-sensitive policies, and optimizing with multiple objective functions.

</div>
