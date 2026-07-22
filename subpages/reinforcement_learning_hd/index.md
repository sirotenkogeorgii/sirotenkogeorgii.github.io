---
title: Reinforcement Learning (HD)
layout: default
noindex: true
math: true
tags:
  - machine-learning
  - reinforcement-learning
  - markov-decision-process
  - multi-armed-bandits
  - dynamic-programming
  - temporal-difference-learning
  - monte-carlo
  - policy-gradients
---

<style>
  .accordion summary {
    font-weight: 600;
    color: var(--accent-strong, #2c3e94);
    background-color: var(--accent-soft, #f5f6ff);
    padding: 0.35rem 0.6rem;
    border-left: 3px solid var(--accent-strong, #2c3e94);
    border-radius: 0.25rem;
  }

  .rl-diagram {
    max-width: 860px;
    margin: 2.5rem auto;
  }

  .rl-diagram svg {
    display: block;
    width: 100%;
    height: auto;
    border: 1px solid var(--line, #dbe1ee);
    border-radius: 0.75rem;
    background: var(--surface, #ffffff);
    box-shadow: 0 16px 30px rgba(15, 23, 42, 0.08);
  }

  .rl-diagram text {
    font-family: inherit;
    fill: var(--text, #172033);
  }

  .rl-diagram .muted {
    fill: var(--muted, #64748b);
  }

  .rl-diagram .box {
    fill: var(--surface-muted, #f8fafc);
    stroke: var(--line, #dbe1ee);
  }

  .rl-diagram .accent {
    fill: var(--accent-soft, #eef2ff);
    stroke: var(--accent-strong, #2c3e94);
  }

  .rl-diagram .green {
    fill: #ecfdf5;
    stroke: #047857;
  }

  .rl-diagram .amber {
    fill: #fffbeb;
    stroke: #b45309;
  }

  .rl-diagram .red {
    fill: #fef2f2;
    stroke: #b91c1c;
  }

  .rl-diagram .line {
    stroke: var(--muted, #64748b);
    stroke-width: 2;
    fill: none;
  }

  .rl-diagram .strong-line {
    stroke: var(--accent-strong, #2c3e94);
    stroke-width: 3;
    fill: none;
  }
</style>

**Table of Contents**
- TOC
{:toc}

# Reinforcement Learning

## Problems

[Selected Problems](/subpages/reinforcement_learning_hd/problems/)

## Lecture 2: Multi-Armed Bandits

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Setup</span><span class="math-callout__name">(Multi-Armed Bandits)</span></p>

In full reinforcement learning, an action simultaneously affects three things: 
1. the **reward** we receive now, 
2. the **state** we transition into,
3. and through that state — **what data we get to see in the future**. 
These three entanglements make the full problem hard. The multi-armed bandit is the setting obtained by deliberately stripping the last two entanglements away: 
* there are **no states to transition between**
* there are **no delayed credit to assign**
The only thing left is the interaction loop

$$
\text{choose action } A_t \;\longrightarrow\; \text{observe reward } R_t,
$$

and yet even here, one essential RL difficulty refuses to disappear: the tension between **exploration** and **exploitation**. Because of this, bandits are the cleanest laboratory for isolating the central learning problem of RL — how to act in a world that only tells you how good your choices were, never what the right choice would have been.

</div>

### Evaluative versus Instructive Feedback

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Evaluative vs. Instructive Feedback)</span></p>

Given a learning agent that selects an action and receives a response, the feedback is:

* **Instructive** if the response tells the learner the *correct* action, independently of the action it chose. Supervised learning is the canonical example: for an input, the label identifies the right class no matter what the model predicts.
* **Evaluative** if the response only tells the learner *how good* the chosen action was. No information is provided about actions that were not taken — the counterfactuals are missing.

Reinforcement learning operates under evaluative feedback.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Evaluative feedback forces exploration)</span></p>

$$\boxed{\text{If you never try an action, you never observe its reward, so you never learn its value.}}$$

Under evaluative feedback this is not a minor inconvenience — it is a hard wall. Unlike supervised learning, where a teacher can reveal the right label for an input the model has never actively chosen, an RL agent learns *only* about actions it samples. Exploration is therefore not optional: it is the only channel through which information about unused actions reaches the agent.

</div>

### The $k$-armed Bandit: Formal Setup

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($k$-armed Bandit)</span></p>

A $k$-armed bandit is specified by

* a finite **action set** $a \in \lbrace 1, \dots, k \rbrace$ (the "arms"),
* at each time step $t$, the agent selects an action $A_t$ and observes a scalar reward $R_t$ drawn from the reward distribution of that arm,
* a **true action value**

  $$q_*(a) \doteq \mathbb{E}[R_t \mid A_t = a],$$

  which is unknown to the agent.

The agent maintains an **estimate** $Q_t(a) \approx q_\ast(a)$ and its **goal** is to maximize expected cumulative reward

$$
\sum_{t=1}^{T} R_t.
$$

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(Two jobs the agent has to do)</span></p>

Every bandit algorithm implicitly performs two tasks:

* **Estimation:** learn action values $Q_t(a)$ from the rewards seen so far.
* **Control:** decide which action $A_t$ to play next.

Estimation is usually the easy part — the law of large numbers does most of the work. The real difficulty is control: *which* actions to play so that estimation gets the informative data it needs.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(The Restaurant Problem)</span></p>

You have just moved to a new city with $k = 10$ restaurants nearby. Each night you pick one and rate your meal. Your goal is to maximize total enjoyment over the year. The dilemma is painfully concrete:

* **Exploit:** return to the best restaurant you have found so far.
* **Explore:** try somewhere new — it might turn out to be even better.

Every bandit algorithm is an answer to the question: *how do you balance trying new options against sticking with what works?*

</div>

### Sample Averages and $\varepsilon$-greedy

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Idea</span><span class="math-callout__name">(Straightforward Value Estimator)</span></p>

The most natural estimator of $q_\ast(a)$ is the empirical mean of the rewards observed from arm $a$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Sample-Average Action-Value Estimate)</span></p>

Let $N_t(a)$ denote the number of times action $a$ has been selected before time $t$. The **sample-average estimate** of $q_\ast(a)$ is

$$
Q_t(a) \;=\; \frac{\sum_{i=1}^{t-1} R_i \, \mathbf{1}\lbrace A_i = a\rbrace}{\sum_{i=1}^{t-1} \mathbf{1}\lbrace A_i = a\rbrace}
\;=\; \frac{1}{N_t(a)} \sum_{i<t:\, A_i = a} R_i.
$$

In a stationary problem, by the law of large numbers $Q_t(a) \to q_\ast(a)$ as $N_t(a) \to \infty$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(What "stationary" means here)</span></p>

**Formal statement.** A bandit problem is **stationary** if the reward distribution of every arm is fixed in time: for every action $a$ and every pair of time steps $s, t$,

$$
R_s \mid A_s = a \;\stackrel{d}{=}\; R_t \mid A_t = a.
$$

In particular the true action value $q_\ast(a) = \mathbb{E}[R_t \mid A_t = a]$ does **not** depend on $t$ — it is a single unknown number per arm that never moves.

**Why the assumption matters here.** The convergence $Q_t(a) \to q_\ast(a)$ we just stated relies on it in two places:

* the rewards $\lbrace R_i : A_i = a\rbrace$ sampled from arm $a$ are i.i.d. draws from one fixed distribution, so the LLN applies,
* the limit $q_\ast(a)$ that the sample mean converges to is itself a well-defined constant.

If either of these breaks, sample averages are the wrong estimator.

**Non-stationary, for contrast.** A non-stationary bandit is one where $q_\ast(a)$ is allowed to *drift over time*: the true means themselves change as the agent interacts. Classical examples are slot machines whose underlying probabilities shift, or a restaurant whose cook changes mid-year. In that regime, averaging *all* past rewards equally is actively harmful — ancient rewards from a stale regime pull the estimate just as strongly as fresh ones do. We return to this case in the section on constant step sizes, where the goal is no longer convergence to a fixed number but **tracking** a moving target.

**Why "almost" was avoided.** Unlike LLN statements that speak of almost-sure convergence of arbitrary $(X_i)\_{i \ge 1}$, here convergence of $Q_t(a)$ is contingent on the agent *actually sampling arm $a$ infinitely often* — $N_t(a) \to \infty$. A control rule that stops pulling arm $a$ after finitely many steps prevents convergence entirely, even in a stationary problem. The stationarity assumption concerns the environment; the "sample often enough" condition concerns the algorithm.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Estimation is rarely the bottleneck)</span></p>

Sample averages are asymptotically exact, so *given enough pulls* they recover the true value of any arm. The bottleneck is not the quality of the estimator — it is the quality of the control policy. We need a rule that *chooses* actions in a way that gathers enough informative data about the right arms. The rest of this lecture is essentially a taxonomy of such rules.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Greedy Action and Pure-Greedy Selection)</span></p>

Given estimates $Q_t(a)$, the **greedy action** at time $t$ is any action that currently looks best:

$$
A_t^{\text{greedy}} \;=\; \arg\max_{a} Q_t(a),
$$

with ties broken arbitrarily (usually at random). The pure-greedy policy always plays $A_t = \arg\max_a Q_t(a)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why pure greedy can fail catastrophically)</span></p>

Pure greedy has no explicit exploration, and three things conspire against it:

* Early random rewards can make a genuinely bad arm *look* good.
* Once the greedy rule stops sampling a competitor, its estimate stops improving — so it can never "come back".
* Crucially, in bandits the distribution of data the agent sees depends on the algorithm itself. A bad decision early on can freeze the data distribution in a way that prevents correction forever.

This self-reinforcing failure mode is the first distinctive feature that separates sequential decision-making from standard supervised learning.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">($\varepsilon$-greedy Action Selection)</span></p>

Given estimates $Q_t(a)$ and a parameter $\varepsilon \in [0,1]$:

$$
A_t \;=\; \begin{cases}
\arg\max_a Q_t(a), & \text{with probability } 1 - \varepsilon, \\
\text{a uniform random action}, & \text{with probability } \varepsilon.
\end{cases}
$$

If $a_g$ is the (unique) greedy action, the selection probabilities are

$$
\Pr(A_t = a_g) = 1 - \varepsilon + \frac{\varepsilon}{k}, \qquad
\Pr(A_t = a) = \frac{\varepsilon}{k} \quad \text{for } a \neq a_g.
$$

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(What $\varepsilon$-greedy buys us)</span></p>

$\varepsilon$-greedy is the minimal fix for pure greedy. **It guarantees that every action continues to be sampled** with probability at least $\varepsilon / k$ at every time step. Combined with the LLN, this means $Q_t(a) \to q_\ast(a)$ for *every* arm $a$ — so in the long run the greedy argmax identifies the true best arm. The price is that an $\varepsilon$ fraction of steps are "wasted" on uniformly random exploration even after the problem is well-understood.

</div>

### Evaluating Bandit Policies: The 10-armed Testbed

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Bandit Testbed)</span></p>

A **testbed** is a fixed *distribution over bandit instances* used to evaluate a learning algorithm. The experimental protocol is:

1. Sample a fresh $k$-armed bandit instance from the distribution.
2. Run the learning algorithm for many time steps.
3. Repeat for many independent runs and average the learning curves.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(The 10-armed Testbed)</span></p>

* **Across runs:** each arm's true value is drawn independently, $q_\ast(a) \sim \mathcal{N}(0, 1)$, for $a = 1, \dots, 10$.
* **Within a run:** rewards are noisy samples $R_t \sim \mathcal{N}(q_\ast(A_t), 1)$.

This testbed contains *two sources of randomness* that a learning curve has to average over:

* **Across-run randomness:** the problem itself is different on each run.
* **Within-run randomness:** rewards are noisy even if the underlying means were known.

</div>

<figure>
  <img src="{{ '/assets/images/notes/rl_hd/reward_distribution.png' | relative_url }}" alt="Dirichlet function plotted on a dense set of rationals; its Lebesgue integral equals that of the zero function" loading="lazy">
</figure>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(Reading the standard learning curves)</span></p>

Two quantities are typically plotted on a testbed:

* **Average reward** at step $t$: how much reward the method collects over time.
* **% optimal action** at step $t$: how often it selects the true best arm.

On the 10-armed testbed, a few empirical regularities stand out:

* Pure greedy ($\varepsilon = 0$) often looks excellent early — it commits fast.
* $\varepsilon$-greedy with $\varepsilon = 0.1$ is slower initially but *wins eventually*, because it keeps correcting early mistakes.
* $\varepsilon = 0.01$ sits in between: slow to correct but asymptotically better than $\varepsilon = 0.1$, because it wastes less of its budget on random actions.

The general lesson: greedy looks good *early*, exploration pays off *late*.

</div>

<figure>
  <img src="{{ '/assets/images/notes/rl_hd/reward_curves.png' | relative_url }}" alt="Dirichlet function plotted on a dense set of rationals; its Lebesgue integral equals that of the zero function" loading="lazy">
</figure>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Tracking the Updates by Hand)</span></p>

Setup: $k = 2$ arms, $Q_1(1) = Q_1(2) = 0$, $\varepsilon = 0.2$. The first three steps produce:

| Step | Action | Reward | New Estimates |
| :--: | :----: | :----: | :------------ |
| 1    | 1      | 1      | $Q(1) = 1,\; Q(2) = 0$ |
| 2    | 1      | 0      | $Q(1) = 0.5,\; Q(2) = 0$ |
| 3    | 2      | 2      | $Q(1) = 0.5,\; Q(2) = 2$ |

At step $4$, which arm does $\varepsilon$-greedy choose?

* The greedy action is arm $2$ because $Q(2) = 2 > Q(1) = 0.5$.
* $\Pr(\text{arm } 2) = (1 - 0.2) + \tfrac{0.2}{2} = 0.9.$
* $\Pr(\text{arm } 1) = \tfrac{0.2}{2} = 0.1.$

</div>

### Incremental Implementation

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Problem</span><span class="math-callout__name">(A thought experiment)</span></p>

If you have already computed the average of $99$ numbers and now receive a $100$th, do you really need to re-sum all $100$ of them to update the average?

The **answer is no**, and the observation behind this is the single most reused pattern in RL.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Derivation</span><span class="math-callout__name">(Incremental Mean)</span></p>

Fix one action, and let $Q_n = \frac{1}{n-1} \sum_{i=1}^{n-1} R_i$ be the mean of its first $n - 1$ rewards. After observing a new reward $R_n$,

$$
Q_{n+1} \;=\; \frac{1}{n} \sum_{i=1}^n R_i \;=\; \frac{1}{n}\!\left(\sum_{i=1}^{n-1} R_i + R_n\right).
$$

Since $\sum_{i=1}^{n-1} R_i = (n-1)\,Q_n$, this becomes

$$
Q_{n+1} \;=\; \frac{1}{n}\bigl((n-1)Q_n + R_n\bigr) \;=\; Q_n + \frac{1}{n}(R_n - Q_n).
$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Rule</span><span class="math-callout__name">(The Universal RL Update Template)</span></p>

Almost every learning rule in this course can be read as an instance of

$$
\text{new estimate} \;\leftarrow\; \text{old estimate} + \alpha \, (\text{target} - \text{old estimate}),
$$

with three named pieces:

* **Target:** the value we want to move the estimate toward.
* **Prediction error:** $(\text{target} - \text{old})$.
* **Step size $\alpha$:** how far along the error direction we move.

For the sample-average rule we just derived, the target is $R_n$ and the step size is $1/n$. For many later rules it will be something else — but the skeleton is identical.

</div>

### Non-stationarity and Constant Step Sizes

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Problem</span><span class="math-callout__name">(Sample averages fails in nonstationary problems)</span></p>

The sample average weights every observed reward *equally*. This is ideal when the true value $q_\ast(a)$ is a fixed constant — the LLN then tells us $Q_t(a) \to q_\ast(a)$. But if $q_\ast(a)$ *drifts over time*, equal weighting is a disaster: ancient rewards from a stale regime drag the estimate away from the current truth just as hard as fresh rewards pull it toward it. 

**In a non-stationary bandit, convergence to a fixed number is the *wrong* goal** 

$$\implies$$

What we want is **tracking** — an estimate that follows the moving target.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Rule</span><span class="math-callout__name">(Constant Step-Size Update)</span></p>

For a non-stationary bandit we replace the shrinking step size $1/n$ by a **constant** $\alpha \in (0, 1]$:

$$Q_{n+1} \;=\; Q_n + \alpha\,(R_n - Q_n) \;=\; (1 - \alpha)\, Q_n + \alpha\, R_n.$$

A constant step size makes recent rewards matter more than old ones.

</div>

[A bit on non-stationary processes](/subpages/reinforcement_learning_hd/subpages/non-stationary-processes/)

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Exponential Recency Weighting)</span></p>

Unrolling the constant step-size recursion from an initial estimate $Q_1$:

$$Q_{n+1} \;=\; (1 - \alpha)^n Q_1 + \sum_{i=1}^n \alpha \,(1 - \alpha)^{n - i}\, R_i.$$

The coefficient of $R_i$ is $\alpha(1 - \alpha)^{n-i}$, which decays geometrically in $n - i$ — the age of the reward.

* Recent rewards receive the largest weights.
* Older rewards are downweighted exponentially fast.
* A constant step size therefore implements an **exponentially recency-weighted average**.

This is the mechanism that lets the estimate forget obsolete data and track the current value.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

**By induction.**

For $n=1$, we have

$$Q_2 = (1-\alpha)Q_1 + \alpha R_1,$$

which agrees with the claimed formula since

$$(1-\alpha)^1 Q_1 + \sum_{i=1}^{1}\alpha(1-\alpha)^{1-i}R_i = (1-\alpha)Q_1 + \alpha R_1.$$

Assume that the formula holds for some $n-1 \in \mathbb{N}$, i.e.

$$Q_n = (1-\alpha)^{n-1}Q_1 + \sum_{i=1}^{n-1}\alpha(1-\alpha)^{n-1-i}R_i.$$

Then

$$
\begin{aligned}
Q_{n+1}
&= (1-\alpha)Q_n + \alpha R_n \\
&= (1-\alpha)\left((1-\alpha)^{n-1}Q_1
+ \sum_{i=1}^{n-1}\alpha(1-\alpha)^{n-1-i}R_i\right)
+ \alpha R_n \\
&= (1-\alpha)^n Q_1
+ \sum_{i=1}^{n-1}\alpha(1-\alpha)^{n-i}R_i
+ \alpha R_n \\
&= (1-\alpha)^n Q_1
+ \sum_{i=1}^{n}\alpha(1-\alpha)^{n-i}R_i.
\end{aligned}
$$

Thus, the formula holds for all $n \in \mathbb{N}$.

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Normalization and Exponential Recency Weighting)</span></p>

Verify that all weights

$$(1-\alpha)^n, \qquad \alpha(1-\alpha)^{n-i}, \quad i=1,\ldots,n,$$

in the expression above sum to $1$. Why is this called an **exponential recency-weighted average**?

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

The weights in the expression from Task 1 are

$$(1-\alpha)^n, \qquad \alpha(1-\alpha)^{n-i}, \quad i=1,\dots,n.$$

Their sum is

$$
\begin{aligned}
(1-\alpha)^n
+ \sum_{i=1}^{n}\alpha(1-\alpha)^{n-i}
&= (1-\alpha)^n
+ \alpha \sum_{j=0}^{n-1}(1-\alpha)^j \\
&= (1-\alpha)^n
+ \alpha \cdot \frac{1-(1-\alpha)^n}{1-(1-\alpha)} \\
&= (1-\alpha)^n
+ 1 - (1-\alpha)^n \\
&= 1.
\end{aligned}
$$

Therefore, $Q_{n+1}$ is a weighted average of the initial value $Q_1$ and the rewards $R_1,\dots,R_n$.

It is called an exponential recency-weighted average because the weight of reward $R_i$ is

$$\alpha(1-\alpha)^{n-i}.$$

More recent rewards have larger weights. For example, $R_n$ has weight $\alpha$, $R_{n-1}$ has weight $\alpha(1-\alpha)$, and older rewards are discounted by higher powers of $1-\alpha$. Thus, the influence of older rewards decays exponentially with time.

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Sample Means as Uniform Memory)</span></p>

Consider now the sample-mean rule

$$\alpha_n=\frac1n, \qquad n\in\mathbb N.$$

Show that the update

$$Q_{n+1} = Q_n+\frac1n(R_n-Q_n), \qquad n\in\mathbb N,$$

unrolls to

$$Q_{n+1} = \frac1n\sum_{i=1}^n R_i$$

for all $n\in\mathbb N$.

What weight does each past reward receive? How does the initial value $Q_1$ enter the expression, and how does this compare to the initial-value bias $(1-\alpha)^nQ_1$ of the constant-$\alpha$ rule?

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

**By induction.**

For $n=1$,

$$Q_2 = Q_1 + (R_1 - Q_1) = R_1 = \frac{1}{1}\sum_{i=1}^{1}R_i.$$

Hence the claim holds for $n=1$.

Assume that for some $n-1 \in \mathbb{N}$,

$$Q_n = \frac{1}{n-1}\sum_{i=1}^{n-1}R_i.$$

Then

$$
\begin{aligned}
Q_{n+1}
&= \left(1-\frac{1}{n}\right)Q_n + \frac{1}{n}R_n \\
&= \frac{n-1}{n}\cdot \frac{1}{n-1}\sum_{i=1}^{n-1}R_i
+ \frac{1}{n}R_n \\
&= \frac{1}{n}\sum_{i=1}^{n-1}R_i + \frac{1}{n}R_n \\
&= \frac{1}{n}\sum_{i=1}^{n}R_i.
\end{aligned}
$$

Thus, the update unrolls to the sample average.

In this case, each past reward $R_i$, $i=1,\dots,n$, receives the same weight $1/n$. The initial value $Q_1$ does not appear in the expression after the first update, because $\alpha_1=1$ gives $Q_2=R_1$. Therefore, the sample-mean rule has no persistent initial-value bias.

This differs from the constant-$\alpha$ rule, where the initial value enters as

$$(1-\alpha)^n Q_1.$$

For $0<\alpha<1$, this term decays exponentially but does not become exactly zero after finitely many updates.

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Decay of Initial-Value Bias: How Fast Optimistic Initialisation Is Forgotten)</span></p>

Suppose $Q_1(a)=5$ for all arms in a $10$-armed testbed where

$$q^\ast(a)\sim\mathcal N(0,1).$$

Find the minimum number of pulls $n\in\mathbb N$ of the same arm such that the initial bias for the exponential-decay rule is smaller than $0.1$ for $\alpha=0.1$. Give an exact expression and a numerical value.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

For the constant-$\alpha$ update, from Task 1 we have

$$Q_{n+1} = (1-\alpha)^n Q_1(a) + \sum_{i=1}^{n}\alpha(1-\alpha)^{n-i}R_i.$$

Let $\beta = 1-\alpha$. Then

$$Q_{n+1} = \beta^n Q_1(a) + \sum_{i=1}^{n}\alpha\beta^{n-i}R_i.$$

For a fixed arm $a$, the rewards satisfy

$$\mathbb{E}[R_i \mid q_\ast(a)] = q_\ast(a).$$

Therefore,

$$
\begin{aligned}
\mathbb{E}[Q_{n+1} \mid q_*(a)]
&=
\beta^n Q_1(a)
+
\sum_{i=1}^{n}\alpha\beta^{n-i}\mathbb{E}[R_i \mid q_*(a)] \\
&=
\beta^n Q_1(a)
+
q_*(a)\sum_{i=1}^{n}\alpha\beta^{n-i}.
\end{aligned}
$$

Since

$$\sum_{i=1}^{n}\alpha\beta^{n-i} = 1-\beta^n,$$

we get

$$\mathbb{E}[Q_{n+1} \mid q_*(a)] = \beta^n Q_1(a) + (1-\beta^n)q_*(a).$$

Hence the conditional bias is

$$
\begin{aligned}
\mathbb{E}[Q_{n+1} \mid q_*(a)] - q_*(a)
&=
\beta^n Q_1(a)
+
(1-\beta^n)q_*(a)
-
q_\ast(a) \\
&=
\beta^n\bigl(Q_1(a)-q_\ast(a)\bigr).
\end{aligned}
$$

Now we use the fact that

$$q_\ast(a) \sim \mathcal{N}(0,1),$$

so

$$\mathbb{E}[q_\ast(a)] = 0.$$

Taking expectation also over the random choice of $q_\ast(a)$ gives

$$\mathbb{E}\left[\mathbb{E}[Q_{n+1} \mid q_\ast(a)] - q_\ast(a) \right] = \beta^n\left(Q_1(a)-\mathbb{E}[q_\ast(a)]\right).$$

Since $Q_1(a)=5$ and $\mathbb{E}[q_\ast(a)]=0$, the expected initial bias is

$$5\beta^n.$$

For $\alpha=0.1$, we have $\beta=1-\alpha=0.9$. We want the smallest $n\in\mathbb{N}$ such that

$$5(0.9)^n < 0.1.$$

$$(0.9)^n < 0.02.$$

$$n\log(0.9) < \log(0.02).$$

$$n > \frac{\log(0.02)}{\log(0.9)}.$$

$$n_{\min} = \left\lfloor \frac{\log(0.02)}{\log(0.9)} \right\rfloor + 1.$$

$$\frac{\log(0.02)}{\log(0.9)} \approx 37.13,$$

$$n_{\min}=38.$$

</details>
</div>

TODO: add ar, var, attention, stochastic differential equation, alpha as a function (time-adaptive step-size), gd, kalman filter, kalman gain

### Optimism and Confidence Bounds

Both optimistic initialization and UCB address exploration by biasing the estimates themselves, rather than by occasionally acting randomly.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Optimistic Initial Values)</span></p>

Set the initial estimate of every action to the same artificially high value $Q_0$ — higher than any plausible true value:

$$
Q_1(a) \;=\; Q_0 \quad \text{for all } a.
$$

A greedy agent starting from this configuration initially believes every arm is promising. Every arm therefore gets pulled, and as it is sampled its estimate drops toward the true $q_\ast(a)$. Arms that have *not* yet been tried still look attractive by comparison, so they continue to be selected.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Optimism is implicit exploration — and its limits)</span></p>

Optimism creates exploration *without any random action selection*: the policy can be purely greedy and still explore the whole action set early on. On stationary problems, optimistic-greedy typically sweeps the arms broadly in the first few hundred steps and ends up picking the true best arm more reliably than realistic $\varepsilon$-greedy.

**Strength:**
* Useful in stationary tasks.
* Encourages a broad initial sweep of the arms.

The limitation is structural: the optimism budget is **spent once**. Once every arm has been tried and its estimate has dropped to its true value, the artificially high initialization no longer drives exploration. If the environment later changes, the original optimism does not magically return — this method is essentially a one-time mechanism and is **ill-suited to non-stationary problems**.

**Weakness:**
* Mostly a one-time mechanism.
* If the environment changes later, the optimism does not magically return.

</div>

<figure>
  <img src="{{ '/assets/images/notes/rl_hd/optimistic_greedy_vs_realistic_epsilon_greedy.png' | relative_url }}" alt="Dirichlet function plotted on a dense set of rationals; its Lebesgue integral equals that of the zero function" loading="lazy">
</figure>

<figure>
  <img src="{{ '/assets/images/notes/rl_hd/ucb_epsilon_greedy.png' | relative_url }}" alt="Dirichlet function plotted on a dense set of rationals; its Lebesgue integral equals that of the zero function" loading="lazy">
</figure>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why $\varepsilon$-greedy is crude)</span></p>

$\varepsilon$-greedy treats all non-greedy actions identically: it explores **uniformly** at random. It does not care

* which non-greedy arms are *almost* as good as the greedy one,
* nor which arms are *especially uncertain* (rarely sampled).

A better principle is: **explore where the uncertainty is large**. This is exactly what UCB formalizes.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Upper Confidence Bound — UCB)</span></p>

At time $t$, select

$$
A_t \;=\; \arg\max_{a} \left[\, Q_t(a) + c\,\sqrt{\frac{\ln t}{N_t(a)}}\,\right],
$$

where $c > 0$ is a tuning parameter.

* $Q_t(a)$ rewards arms that currently *look* good.
* $N_t(a)$ appears in the denominator, so arms pulled many times receive a smaller bonus.
* $c\sqrt{\ln t / N_t(a)}$ is an **uncertainty bonus** — an upper confidence correction that grows when $a$ has been explored little.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(How the UCB bonus evolves)</span></p>

The bonus has two competing dynamics:

* Whenever action $a$ is selected, $N_t(a)$ increases — so the bonus for $a$ **shrinks**.
* Whenever $a$ is *not* selected, $t$ still increases, $\ln t$ grows, and the bonus for $a$ **slowly rises**.

The net effect is that UCB revisits uncertain actions again and again, but with decreasing frequency, concentrating progressively on the arms that turn out to be genuinely good. In this sense UCB explores **directedly**: 
* it prefers actions that are simultaneously *plausible* (high $Q_t$) and *insufficiently tested* (low $N_t$).

</div>

<figure>
  <img src="{{ '/assets/images/notes/rl_hd/ucb_epsilon_greedy.png' | relative_url }}" alt="Dirichlet function plotted on a dense set of rationals; its Lebesgue integral equals that of the zero function" loading="lazy">
</figure>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(Placing the methods on one axis)</span></p>

It is useful to line the value-based methods up on a single **exploration–exploitation spectrum**:

$$
\underbrace{\text{pure explore}}_{\text{high info gain, low reward now}}
\;\longleftarrow\;
\varepsilon\text{-greedy} \;\prec\; \text{UCB} \;\prec\; \text{optimism}
\;\longrightarrow\;
\underbrace{\text{pure exploit}}_{\text{low info gain, high reward now}}
$$

All three live between the extremes; they only differ in *how* they strike the balance.

</div>

<figure>
  <img src="{{ '/assets/images/notes/rl_hd/ExplorationExploitationSpectrum.png' | relative_url }}" alt="Dirichlet function plotted on a dense set of rationals; its Lebesgue integral equals that of the zero function" loading="lazy">
</figure>

### Gradient Bandits

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">New Paradigm</span><span class="math-callout__name">(From learning expecrted rewards (values) to learning action policy)</span></p>

* Every **method above** shares the same basic recipe: estimate $Q_t(a)$, then act nearly greedily on those estimates. 
* **Gradient bandits** take a genuinely different route — they parameterize the **policy directly** and improve it by gradient ascent on expected reward. This is the first appearance in the course of the idea of **policy gradients**.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Preferences and the Softmax Policy)</span></p>

Instead of action-value estimates, maintain a real-valued **preference** $H_t(a) \in \mathbb{R}$ for every action. The preferences induce a stochastic **softmax policy**:

$$
\pi_t(a) \;\doteq\; \Pr(A_t = a) \;=\; \frac{e^{H_t(a)}}{\sum_{b=1}^{k} e^{H_t(b)}}.
$$

* Larger $H_t(a) \Rightarrow$ larger $\pi_t(a)$, but every action keeps positive probability.
* The policy is **shift-invariant**: 
  * adding the same constant to every $H_t(a)$ leaves $\pi_t$ unchanged. 
  * only *relative* preferences matter.
* We are no longer learning values directly — we are learning which actions to *prefer*.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Objective</span><span class="math-callout__name">(Expected Reward under the Policy)</span></p>

Suppress the time index of $H$ and $\pi$ for clarity. The policy objective is the expected one-step reward:

$$
J(H) \;=\; \mathbb{E}[R]
\;=\; \mathbb{E}_A\bigl[\,\mathbb{E}[R \mid A]\,\bigr]
\;=\; \sum_{a=1}^{k} \Pr(A = a)\,\mathbb{E}[R \mid A = a]
\;=\; \sum_{a=1}^{k} \pi(a)\, q_*(a).
$$

We update $H$ by gradient ascent:

$$
H(a) \;\leftarrow\; H(a) + \alpha\, \frac{\partial J(H)}{\partial H(a)}.
$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(The Softmax Derivative)</span></p>

For the softmax policy $\pi(a) = e^{H(a)} / \sum_{b} e^{H(b)}$,

$$
\frac{\partial \pi(a)}{\partial H(c)} \;=\; \pi(a)\bigl(\mathbf{1}\lbrace a = c\rbrace - \pi(c)\bigr),
$$

which splits into the two cases

$$
\frac{\partial \pi(a)}{\partial H(c)}
\;=\;
\begin{cases}
\pi(c)\bigl(1 - \pi(c)\bigr), & a = c, \\[4pt]
-\pi(a)\,\pi(c), & a \neq c.
\end{cases}
$$

Increasing $H(c)$ raises $\pi(c)$ and — by the constraint $\sum_a \pi(a) = 1$ — lowers every other $\pi(a)$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Softmax Derivative</summary>

TODO: rewrite from my paper notes.

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Derivation</span><span class="math-callout__name">(From the Exact Gradient to an Interpretable Form)</span></p>

Apply the chain rule to $J(H) = \sum_x \pi(x)\, q_\ast(x)$:

$$
\frac{\partial J}{\partial H(a)}
\;=\; \sum_{x} q_*(x)\, \frac{\partial \pi(x)}{\partial H(a)}
\;=\; \sum_{x} q_*(x)\, \pi(x)\bigl(\mathbf{1}\lbrace x = a\rbrace - \pi(a)\bigr).
$$

Expanding and using 
* $\sum_x \pi(x) = 1$,
* $\sum_x \pi(x) q_\ast(x) = J(H)$

gives

$$
\frac{\partial J}{\partial H(a)}
\;=\; \pi(a)\Bigl(\; \underbrace{q_*(a)}_{=\,\mathbb{E}[R \mid A = a]} \;-\; \underbrace{J(H)}_{=\,\mathbb{E}_{A \sim \pi}[R]}\;\Bigr).
$$

So the gradient says: **raise the preference of actions whose value is above the current policy average, and lower the rest.**

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(The Gradient as an Expectation)</span></p>

An equivalent form of the same exact gradient is

$$
\frac{\partial J}{\partial H(a)}
\;=\; \mathbb{E}_{A \sim \pi}\!\left[\, q_*(A)\bigl(\mathbf{1}\lbrace A = a\rbrace - \pi(a)\bigr)\,\right].
$$

This rewriting is useful precisely because it expresses the gradient as an expectation over actions the policy already samples — so we can estimate it from sampled data.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

$$
\frac{\partial J}{\partial H(a)}\;=\; \sum_{x} q_*(x)\, \pi(x)\bigl(\mathbf{1}\lbrace x = a\rbrace - \pi(a)\bigr) = \mathbb{E}_{A \sim \pi}\!\left[\, q_*(A)\bigl(\mathbf{1}\lbrace A = a\rbrace - \pi(a)\bigr)\,\right].
$$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Gradient-Bandit Update with Baseline)</span></p>

Sample $A_t \sim \pi_t$ and observe $R_t$. Since $\mathbb{E}[R_t \mid A_t] = q_\ast(A_t)$, the random quantity

$$
R_t\bigl(\mathbf{1}\lbrace A_t = a\rbrace - \pi_t(a)\bigr)
$$

is an **unbiased** sample-based estimate of $\partial J / \partial H(a)$. Subtracting a baseline $\bar R_t$ (typically the running average of past rewards) does not change the expected gradient but reduces its variance:

$$
(R_t - \bar R_t)\bigl(\mathbf{1}\lbrace A_t = a\rbrace - \pi_t(a)\bigr).
$$

This yields the **gradient-bandit update**

$$
H_{t+1}(a) \;=\; H_t(a) + \alpha\,(R_t - \bar R_t)\bigl(\mathbf{1}\lbrace A_t = a\rbrace - \pi_t(a)\bigr),
$$

which, separated into the selected and non-selected actions, reads:

$$
\begin{aligned}
H_{t+1}(A_t) &= H_t(A_t) + \alpha\,(R_t - \bar R_t)\bigl(1 - \pi_t(A_t)\bigr), \\
H_{t+1}(a) &= H_t(a) - \alpha\,(R_t - \bar R_t)\,\pi_t(a) \qquad \text{for } a \neq A_t.
\end{aligned}
$$

* If the reward is **above** the baseline, the chosen action's preference goes up and every other preference goes down.
* If the reward is **below**, the chosen action is pushed down and the others up.

</div>

<figure>
  <img src="{{ '/assets/images/notes/rl_hd/baseline_subtraction_trick.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>Baseline: same expected direction, less noise. That typically makes learning smoother and faster.</figcaption>
</figure>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why the baseline helps)</span></p>

Subtracting $\bar R_t$ does **not** change the expected gradient direction — 

$$\mathbb{E}\_{A \sim \pi}[(\mathbf{1}\lbrace A = a\rbrace - \pi(a))] = 0$$

and we get

$$\mathbb E\left[ (R_t-\bar R_t) \left(\mathbf 1_{\lbrace A_t=a\rbrace}-\pi_t(a)\right) \right] = \mathbb E\left[ R_t \left(\mathbf 1_{lbrace A_t=a\rbrace}-\pi_t(a)\right) \right],$$

so any action-independent baseline cancels in expectation. What it *does* change is the variance of the stochastic estimate. By centering the rewards around a running average, the magnitude of the update stays comparable across problems with very different reward scales, and the learning curves become visibly smoother and faster — which is borne out by the experimental comparison of "with baseline" versus "without baseline" on the testbed.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Interpretation of the gradient $\frac{\partial J}{\partial H(a)}$)</span></p>

Yes, that is essentially correct.

Because the policy is softmax, changing $H(a)$ changes **all** action probabilities:

$$\frac{\partial \pi(x)}{\partial H(a)} = \pi(x)\bigl(\mathbf 1{x=a}-\pi(a)\bigr).$$

So

$$\frac{\partial J}{\partial H(a)} = \sum_x q_*(x)\frac{\partial \pi(x)}{\partial H(a)}$$

adds together:

$$\underbrace{q_*(a)\pi(a)(1-\pi(a))}_{\text{gain from increasing }\pi(a)} = \underbrace{\sum_{x\neq a}q_*(x)\pi(x)\pi(a)}_{\text{loss from decreasing other probabilities}}.$$

This simplifies to

$$\frac{\partial J}{\partial H(a)} = \pi(a)\bigl(q_*(a)-J(H)\bigr).$$

Thus it is exactly a **net trade-off**:

* increasing $H(a)$ shifts probability mass toward $a$;
* that mass must be taken from the other actions;
* the gradient measures whether this redistribution raises or lowers expected reward.

The criterion is especially clean:

$$q_*(a)>J(H) \quad\Longrightarrow\quad \frac{\partial J}{\partial H(a)}>0.$$

So the preference of $a$ should increase precisely when $a$ is better than the current policy’s average action.

One nuance: the gradient is with respect to the **preference $H(a)$**, not directly with respect to $\pi(a)$. The softmax coupling is what automatically accounts for how all other action probabilities react. 

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(Value-based vs. policy-based methods — the core contrast)</span></p>

All of the bandit methods we have seen fall into one of two camps:

* **Value-based methods** (sample-average / $\varepsilon$-greedy / optimistic / UCB): estimate $Q_t(a)$, then pick an action *from* those estimates.
* **Policy-based methods** (gradient bandit): learn action preferences $H_t(a)$, and sample an action directly from the induced policy $\pi_t(a)$.

The appeal of the policy-based view is that it *directly* learns a stochastic policy, so it naturally produces action probabilities and maintains exploration without any bolt-on mechanism. This is the seed of what later becomes the full class of policy-gradient algorithms in RL.

</div>

### Contextual Bandits

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Contextual Bandit)</span></p>

A **contextual bandit** extends the ordinary $k$-armed bandit by providing a **context** $s_t$ at each round before the agent acts. The policy now maps contexts to action distributions:

$$
\pi(a \mid s).
$$

Different contexts may have entirely different best actions.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(What Is Context Worth?)</span></p>

Two cases occur with equal probability at every round:

* **Case A:** $(q_\ast(1),\, q_\ast(2)) = (10,\, 20)$.
* **Case B:** $(q_\ast(1),\, q_\ast(2)) = (90,\, 80)$.

**Without context**, each action's marginal value is

$$
\mathbb{E}[R \mid a = 1] = \tfrac{1}{2}(10) + \tfrac{1}{2}(90) = 50, \qquad
\mathbb{E}[R \mid a = 2] = \tfrac{1}{2}(20) + \tfrac{1}{2}(80) = 50.
$$

The best achievable expected reward is therefore **$50$**.

**With context**, pick $a = 2$ in case A and $a = 1$ in case B:

$$
\mathbb{E}[R] = 0.5 \cdot 20 + 0.5 \cdot 90 = 55.
$$

Observing context raises the achievable reward because the policy can *specialize* to the situation.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why contextual bandits are still not full RL)</span></p>

Contextual bandits add *input-dependence* to the policy but keep one crucial simplification of plain bandits:

* **Contextual bandit:** the chosen action affects the immediate reward **only**. It does *not* influence the next context.
* **Full RL:** the chosen action affects both the immediate reward *and* the future state distribution. Now planning through time matters, because what you do now changes what situations you will face later.

The contextual bandit is the cleanest stepping stone into full RL: it introduces "state" as an *observation* without yet introducing "state" as something the agent's actions *control*.

</div>

### Summary

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(Method Map — what each idea fixes)</span></p>

| Method            | Core idea                          | Strength              | Main limitation                 |
| :---------------- | :--------------------------------- | :-------------------- | :------------------------------ |
| Greedy            | Act on current best estimate       | Fast early reward     | Can get stuck                   |
| $\varepsilon$-greedy | Random forced exploration       | Simple, robust        | Wastes exploration late         |
| Optimistic init.  | Explore via high initial values    | Good initial sweep    | Mostly one-shot                 |
| UCB               | Add uncertainty bonus              | Directed exploration  | More bookkeeping / tuning       |
| Gradient bandit   | Learn action probabilities directly | Natural policy view  | Higher variance without baseline |

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(Core Takeaways)</span></p>

* Evaluative feedback makes exploration unavoidable.
* The central tension is always **exploration vs. exploitation**.
* The update pattern
  
  $$\text{estimate} \leftarrow \text{estimate} + \alpha(\text{target} - \text{estimate})$$
  
  appears everywhere in RL.
* In non-stationary problems, **constant step sizes** let us *track* changing values instead of merely converging.
* UCB explores by uncertainty; gradient bandits learn action probabilities directly.
* Contextual bandits are the simplest bridge toward full RL.

**Final message.** Bandits are a compact setting where the essential logic of RL becomes visible before we add states, transitions, and long-term planning.

</div>

## Lecture 3: Finite Markov Decision Processes

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Setup</span><span class="math-callout__name">(From One-Step Bandits to Sequential Decisions)</span></p>

In bandits we deliberately stripped away two of the three things an action can do: it could no longer change the state of the world, and it could no longer affect what data we would see in the future. Only the first effect remained — the immediate reward. **Markov decision processes (MDPs) put those two things back in.** An action now affects three quantities at once:

1. the **immediate reward** $R_{t+1}$,
2. the **next state** $S_{t+1}$ — the situation we will face on the next step,
3. and, indirectly through that state, the **distribution of all future rewards**.

The third effect is what people call **delayed consequences** or the **credit-assignment problem**: a decision made now can change the reward we receive ten steps from now, and the agent has to learn to attribute that distant reward to the action that actually caused it. The remaining lectures of the course are essentially a long answer to one **guiding question**:

> *How do we evaluate actions when their main effect is not just the immediate reward, but also the future states they lead to?*

</div>

### From Bandits to MDPs

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(What MDPs add beyond bandits)</span></p>

Two new objects are introduced and one familiar object is upgraded.

* **State $S_t$** tells us *which situation we are in*. In bandits we had no states — every step was a fresh, identical instance of the same one-shot decision.
* **Dynamics** tell us *how actions change the future*. Choosing $a$ in state $s$ now produces a (possibly random) next state $s'$ in addition to a reward.
* **Value** is no longer indexed by an action only, but by a **state** (or by a state-action pair):

  $$
  \text{Bandits: } q_*(a) \quad\Longrightarrow\quad \text{MDPs: } v_*(s),\; q_*(s, a).
  $$

The contextual bandit at the end of the previous lecture was a small step in this direction: it added an *observation* (context) before each action, but the agent's actions still did not control which context would appear next. An MDP is the next step — actions now actually *steer* the state.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Gridworld — value depends on where you are)</span></p>

Place an agent on a $5 \times 5$ grid. Some cells are walls (obstacles), one cell is a terminal goal, and the agent receives a constant per-step penalty until it reaches the goal. Two non-terminal cells $(4,1)$ and $(5,1)$ both *look* like ordinary cells. Yet they are not equally good places to be: one is closer to the goal, has fewer obstacles between it and the terminal cell, and so it requires fewer per-step penalties to escape from. **The "value" of a state is exactly this kind of information** — and it is something a one-step bandit could never represent, because in a bandit there is no notion of "where you are".

</div>

<figure>
  <img src="{{ '/assets/images/notes/rl_hd/gridworld.png' | relative_url }}" alt="Gridworld with obstacles and a terminal cell — a small concrete finite MDP" loading="lazy">
  <figcaption>A small finite MDP. Different non-terminal cells are not equally good — their value depends on how cheaply the agent can reach the terminal cell from there.</figcaption>
</figure>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(Why MDPs are the core RL model)</span></p>

MDPs sit at the intersection of four properties that, taken together, are the defining features of the RL problem:

* **Evaluative feedback.** The environment only tells us *how good* the chosen action was — not what we should have done instead. (Carried over from bandits.)
* **Associative setting.** The best action depends on the current state. The same action can be excellent in one state and terrible in another.
* **Delayed consequences.** An action's effect on reward may not show up until many steps later.
* **Credit assignment.** Given a reward, the agent has to figure out *which* earlier decision was responsible.

Bandits had only the first; contextual bandits added the second; MDPs add the last two — and that is what makes them the standard mathematical model for **sequential decision making under uncertainty**.

</div>

### The Agent–Environment Interface

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Agent–Environment Loop)</span></p>

At each discrete time step $t = 0, 1, 2, \dots$ the agent and environment exchange three quantities:

1. The agent **observes a state** $S_t \in \mathcal{S}$.
2. The agent **chooses an action** $A_t \in \mathcal{A}(S_t)$, where $\mathcal{A}(s)$ is the set of actions allowed in state $s$.
3. The environment **emits a reward** $R_{t+1} \in \mathcal{R} \subset \mathbb{R}$ and **transitions** to a next state $S_{t+1} \in \mathcal{S}$.

This produces an alternating trajectory

$$
S_0,\, A_0,\, R_1,\, S_1,\, A_1,\, R_2,\, S_2,\, A_2,\, \dots
$$

The reward is conventionally indexed at $t+1$ to emphasize that it is a *consequence* of taking $A_t$ in $S_t$, not part of the state at time $t$.

</div>

<figure>
  <img src="{{ '/assets/images/notes/rl_hd/agent_environment_loop.png' | relative_url }}" alt="Agent-environment interaction loop with action, reward, and next state arrows" loading="lazy">
  <figcaption>The closed loop: the agent emits an action, the environment responds with a reward and a next state, and the cycle repeats.</figcaption>
</figure>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Finite MDP — Tabular Setting)</span></p>

The MDP is **finite** when the state and action sets are finite,

$$
\mathcal{S} = \lbrace s^{(1)}, \dots, s^{(n)} \rbrace, \qquad \mathcal{A}(s) \text{ finite for each } s,
$$

and rewards are bounded (or, more weakly, have finite expectation). In this regime the value functions $v(s)$ and $q(s, a)$ are quite literally **tables of numbers** — one entry per state, or per state-action pair.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why "finite" matters here)</span></p>

The whole first half of an RL course is built on top of the finite/tabular case for one reason: it lets us write everything as elementary arithmetic on finite sums, with no approximation involved. Once $\mathcal{S}$ becomes huge or continuous, value functions can no longer be stored as tables and we have to *approximate* them with parametric functions — neural networks, linear features, etc. That extension (function approximation) is genuinely harder, but the entire conceptual machinery — Bellman equations, optimal policies, greedy improvement — is developed cleanly in the tabular case first.

</div>

### MDP Dynamics and the Markov Property

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(One-Step Dynamics)</span></p>

The environment is fully described by the **joint one-step dynamics**

$$
p(s', r \mid s, a) \;\doteq\; \Pr\lbrace S_{t+1} = s',\; R_{t+1} = r \mid S_t = s,\, A_t = a \rbrace.
$$

For each fixed $(s, a)$ this is a probability distribution over next-state/reward pairs:

$$
\sum_{s' \in \mathcal{S}} \sum_{r \in \mathcal{R}} p(s', r \mid s, a) \;=\; 1.
$$

A single function $p$ contains all the information about how the environment responds to actions.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Derived Models from $p$)</span></p>

Several quantities are routinely derived from $p$ by marginalisation:

* **State-transition probability** (rewards marginalised out):

  $$
  p(s' \mid s, a) \;=\; \sum_{r \in \mathcal{R}} p(s', r \mid s, a).
  $$

* **Expected reward given state and action**:

  $$
  r(s, a) \;\doteq\; \mathbb{E}[R_{t+1} \mid S_t = s,\, A_t = a] \;=\; \sum_{s', r} r\, p(s', r \mid s, a).
  $$

* **Expected reward given state, action, *and* next state**:

  $$
  r(s, a, s') \;\doteq\; \mathbb{E}[R_{t+1} \mid S_t = s,\, A_t = a,\, S_{t+1} = s'] \;=\; \sum_{r} r \,\frac{p(s', r \mid s, a)}{p(s' \mid s, a)}.
  $$

The last form is a conditioning step: we incorporate the additional information that the next state turned out to be $s'$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Markov Property)</span></p>

A state representation is **Markov** if the joint distribution of the next state and reward depends on the past *only through the current state and action*:

$$
\Pr(S_{t+1}, R_{t+1} \mid S_0, A_0, \dots, S_t, A_t) \;=\; \Pr(S_{t+1}, R_{t+1} \mid S_t, A_t).
$$

Equivalently: once the present is known, the rest of the past adds no extra predictive power for the future.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Markov is a constraint on state design, not on the world)</span></p>

The Markov property is best read **not** as an assumption about reality, but as a *requirement on what we put inside the state vector*. Almost no real environment is Markov in its raw observation: a single image of a Pong screen does not tell you the ball's velocity. But if you stack the last few frames into the state, the joint object becomes Markov — at least to a good approximation. So:

* "The world is Markov" is usually false.
* "The state I designed is Markov" is something *I* am responsible for making true, by including in $S_t$ everything the future depends on.

Whenever a method seems to fail because past information matters, the right diagnosis is usually **the state is too small**, not that the math is wrong.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Where is the agent–environment boundary?)</span></p>

The split between "agent" and "environment" is a **modelling choice**, not a physical fact. A working rule of thumb:

* Anything the agent **cannot arbitrarily change** is part of the environment — including the robot's own sensors, motors, and battery.
* The agent is the locus of *decision-making*: the policy and the learned value functions.

The agent may *know* the environment dynamics, but it does **not** get to redefine the reward function: the reward is what tells the agent what we want, and letting the agent choose its own reward defeats the purpose.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Finite Markov Decision Process)</span></p>

A **finite Markov decision process** is a tuple

$$
(\mathcal{S},\, \mathcal{A},\, p,\, \gamma),
$$

where

* $\mathcal{S}$ is a finite state set;
* $\mathcal{A}$ is the action set, or $\mathcal{A}(s)$ the set of admissible actions per state;
* $p(s', r \mid s, a)$ specifies the one-step dynamics;
* $\gamma \in [0, 1]$ is the **discount factor** (defined precisely in the next section).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Deterministic vs. stochastic MDPs)</span></p>

The transition kernel can be either:

* **Deterministic**, $p(s' \mid s, a) \in \lbrace 0, 1 \rbrace$ — every $(s, a)$ has exactly one possible next state. Gridworld with rigid movement is the canonical example.
* **Stochastic**, $p(\cdot \mid s, a)$ is a non-trivial distribution — slot machines, weather, opponents, sensor noise; anywhere the world is not perfectly predictable.

Two sources of randomness coexist in any RL trajectory:

* **Environment randomness**: the next state and reward are drawn from $p(\cdot, \cdot \mid s, a)$.
* **Policy randomness**: the agent itself can act stochastically.

Both can be present, both can be absent — and they are independent design choices.

</div>

### Goals, Rewards, and Return

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Hypothesis</span><span class="math-callout__name">(The Reward Hypothesis)</span></p>

> Everything we mean by the agent's *goal* can be expressed as the maximisation of the expected value of the cumulative sum of a received scalar signal, called **reward**.

The agent does not get to negotiate this signal: at each step, the environment emits a scalar $R_{t+1}$, and the entire job of the agent is to *maximise the expected total reward over time*.

The reward specifies **what** we want — not **how** to achieve it.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Reward design and reward hacking)</span></p>

The reward signal carries the entire definition of "what the agent should do", so getting it right is unusually consequential. Some canonical examples:

* **Chess.** $R = +1$ on win, $0$ on draw, $-1$ on loss, $0$ otherwise. The reward says nothing about *how* to play — it only specifies the outcome we care about.
* **Maze / shortest path.** $R = -1$ per step until the goal is reached. Maximising total reward $=$ minimising number of steps.
* **Recycling robot.** Positive reward for picking up cans, large negative reward if the battery runs out. The trade-off between collecting and recharging emerges by itself.

The pitfall is when the designer rewards a **proxy** instead of the true objective:

* Reward "moving toward the goal" $\to$ the agent learns to approach the goal cell *and stay near it*, never entering, because entering ends the episode.
* Reward "eating points" in a video game $\to$ the agent finds a wall-glitch that lets it loop a single point forever.

This is **reward hacking**: the agent maximises exactly what you wrote down, which turns out not to be what you wanted. The right principle is *reward the goal, not the path you imagine taking to it*.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Return — Episodic Tasks)</span></p>

In an **episodic task** the interaction terminates at a (random) time $T$, and the agent's objective at step $t$ is the **return**

$$
G_t \;\doteq\; R_{t+1} + R_{t+2} + \cdots + R_T.
$$

By convention $G_T = 0$: once the terminal state has been reached, no more reward can be accrued.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Return — Continuing Tasks, Discounting)</span></p>

In a **continuing task** the interaction never terminates, and an undiscounted sum can diverge. We instead define the **discounted return**

$$G_t \;\doteq\; R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots \;=\; \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}, \qquad 0 \le \gamma < 1.$$

The discount factor $\gamma$ controls how much the agent cares about distant rewards: $\gamma$ near $0$ makes it short-sighted, $\gamma$ near $1$ makes it far-sighted.

</div>

<figure>
  <img src="{{ '/assets/images/notes/rl_hd/discount_decay.png' | relative_url }}" alt="Geometric decay of gamma^k for several values of gamma" loading="lazy">
  <figcaption>How heavily a reward $k$ steps in the future contributes to today's return. Smaller $\gamma$ produces a short-sighted agent; $\gamma$ near $1$ produces a far-sighted one.</figcaption>
</figure>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Bounded rewards with discouning gamma guarantees finite returns)</span></p>

If rewards are uniformly bounded by $\lvert R_{t+k} \rvert \le R_{\max}$ and $\gamma < 1$, then

$$\lvert G_t \rvert \;\le\; \sum_{k=0}^{\infty} \gamma^k R_{\max} \;=\; \frac{R_{\max}}{1 - \gamma} \;<\; \infty.$$

So the discounted return is **guaranteed** to be a well-defined finite random variable, and expectations of it are finite. This is the basic reason that the entire MDP machinery does not break on continuing tasks.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Derivation</span><span class="math-callout__name">(The Recursive Identity for the Return)</span></p>

Starting from the discounted return,

$$
G_t \;=\; R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots,
$$

factor a $\gamma$ out of the tail:

$$
G_t \;=\; R_{t+1} + \gamma \underbrace{\bigl( R_{t+2} + \gamma R_{t+3} + \cdots \bigr)}_{=\, G_{t+1}}.
$$

This gives the **fundamental recursive identity**

$$
\boxed{\,G_t \;=\; R_{t+1} + \gamma\, G_{t+1}\,}.
$$

The return at time $t$ equals the immediate reward plus a discounted version of the *return one step later*. This is the seed from which every Bellman equation is grown — and the formal reason behind the bootstrap idea ("estimate now from estimate next") that pervades the rest of RL.

</div>

<figure>
  <img src="{{ '/assets/images/notes/rl_hd/recursive_return.png' | relative_url }}" alt="Recursive identity G_t = R_{t+1} + gamma G_{t+1}" loading="lazy">
  <figcaption>Factoring out one step turns an infinite tail into a single discounted copy of the return — the trick on which every Bellman equation rests.</figcaption>
</figure>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Unifying episodic and continuing tasks)</span></p>

Episodic and continuing tasks look like two different objects, but they can be merged into a single framework with a small trick. Add an **absorbing terminal state**: a special state that, once entered, transitions to itself forever and emits reward $0$ on every step. After termination, the trajectory continues — but contributes nothing further to any sum. Then *any* episodic task can be written as

$$
G_t \;=\; \sum_{k=0}^{\infty} \gamma^k R_{t+k+1},
$$

with the same formula as for continuing tasks. From now on we can write a single sum and not worry about whether the episode "ended".

</div>

### Policies and Value Functions

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Policy)</span></p>

A **policy** is a (possibly stochastic) map from states to action distributions:

$$
\pi(a \mid s) \;\doteq\; \Pr(A_t = a \mid S_t = s).
$$

For each state $s$, $\pi(\cdot \mid s)$ is a valid probability distribution on $\mathcal{A}(s)$. Deterministic policies are the special case where $\pi(\cdot \mid s)$ puts mass $1$ on a single action.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(RL in one sentence)</span></p>

> **Reinforcement learning is the problem of changing the policy on the basis of experience so as to increase expected return.**

Every algorithm in the rest of the course is a particular answer to "how do you change $\pi$?", under the constraint that the only thing the agent ever observes is the trajectory $(S_0, A_0, R_1, S_1, A_1, \dots)$ that the current policy produces.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(State-Value and Action-Value Functions)</span></p>

Fix a policy $\pi$. 

* The **state-value function** measures how good it is to *be in state $s$* and follow $\pi$ thereafter:

  $$v_\pi(s) \;\doteq\; \mathbb{E}_\pi[\,G_t \mid S_t = s\,].$$

* The **action-value function** (or Q-function) measures how good it is to *take action $a$ in state $s$* and follow $\pi$ thereafter:

  $$q_\pi(s, a) \;\doteq\; \mathbb{E}_\pi[\,G_t \mid S_t = s,\, A_t = a\,].$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(State-value function is weighted Action-value function)</span></p>

The two are related by averaging $q$ over actions chosen by the policy:

$$v_\pi(s) \;=\; \sum_{a} \pi(a \mid s)\, q_\pi(s, a).$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why both $v$ and $q$ exist)</span></p>

At first glance $v$ and $q$ encode the same information — and indeed $v_\pi$ can be recovered from $q_\pi$ via the formula above. So why carry both?

* **$v_\pi(s)$ is what you want when you only have to evaluate states.** Cheaper to store ($\lvert\mathcal{S}\rvert$ vs. $\lvert\mathcal{S}\rvert\,\lvert\mathcal{A}\rvert$ entries) and easier to interpret as "how good is this situation".
* **$q_\pi(s, a)$ is what you need for control**, because acting greedily means picking $\arg\max_a$. With only $v_\pi$ you cannot pick a good action without also knowing the dynamics $p$ to look one step ahead.

The control algorithms in later lectures (Q-learning, SARSA, $\dots$) all learn $q$ for exactly this reason.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Two Fundamental Problems — Prediction and Control)</span></p>

* **Prediction (policy evaluation).** Given an MDP and a fixed policy $\pi$, compute $v_\pi$ and $q_\pi$.
* **Control.** Find an *optimal* policy $\pi^*$ that maximises $v_\pi(s)$ at every state:

  $$
  \pi^* \;\in\; \arg\max_{\pi}\, v_\pi(s) \quad \forall s \in \mathcal{S}.
  $$

Almost every method in the next lectures fits into one of these two slots — or alternates between them, as in policy iteration.

</div>

### Bellman Expectation Equations

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Problem</span><span class="math-callout__name">(Why Bellman equations are needed)</span></p>

Computing $v_\pi(s) = \mathbb{E}\_\pi[G_t \mid S_t = s]$ directly would require averaging over the **entire future trajectory**

$$(A_t,\, S_{t+1},\, R_{t+1},\, A_{t+1},\, S_{t+2},\, R_{t+2},\, \dots),$$

i.e. summing over an exponentially large branching tree of all paths. This is hopeless for any non-trivial MDP.

**Bellman's idea.** Replace the full path expectation by a **recursive one-step decomposition**: 
* relate $v_\pi(s)$ to the values of states reachable in one step. 
* The exponential blow-up collapses to a fixed-point equation in $\lvert\mathcal{S}\rvert$ unknowns.

</div>

<figure>
  <img src="{{ '/assets/images/notes/rl_hd/trajectory_tree.png' | relative_url }}" alt="Branching tree of all future trajectories from S_t, alternating policy and dynamics layers" loading="lazy">
  <figcaption>Computing $\mathbb{E}_\pi[G_t \mid S_t = s]$ directly means averaging over this exponentially branching tree. Bellman's trick replaces it by a single one-step lookahead plus a value at the next level.</figcaption>
</figure>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(Where the randomness comes from)</span></p>

The expectation in $v_\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s]$ averages over **all** future random draws, which come from two independent sources:

* **Agent randomness:** $A_t \sim \pi(\cdot \mid S_t)$.
* **Environment randomness:** $(S_{t+1}, R_{t+1}) \sim p(\cdot, \cdot \mid S_t, A_t)$.

The same loop then repeats from $S_{t+1}$ onwards, so $G_t$ is a function of the random sequence 

$$(A_t, S_{t+1}, R_{t+1}, A_{t+1}, S_{t+2}, R_{t+2}, \dots).$$

Splitting the expectation into "first action, then environment, then *the rest of the future*" is exactly what the Bellman equation does.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Bellman Expectation Equation for $v_\pi$)</span></p>

For every state $s \in \mathcal{S}$,

$$
v_\pi(s) \;=\; \sum_{a} \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a) \bigl[\, r + \gamma\, v_\pi(s') \,\bigr].
$$

Reading the formula left to right traces the three layers of the Bellman backup:

* first average over the action selected by $\pi$;
* then average over the environment outcome $(s', r)$;
* finally back up the future return through the value of the successor state $v_\pi(s')$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Start from the definition and substitute the recursive identity for $G_t$:

$$v_\pi(s) \;=\; \mathbb{E}_\pi[G_t \mid S_t = s] \;=\; \mathbb{E}_\pi[R_{t+1} + \gamma G_{t+1} \mid S_t = s].$$

**Step 1.** Condition on the first action via the law of total expectation. Since $\Pr(A_t = a \mid S_t = s) = \pi(a \mid s)$,

$$v_\pi(s) \;=\; \sum_{a} \pi(a \mid s)\, \mathbb{E}_\pi[R_{t+1} + \gamma G_{t+1} \mid S_t = s,\, A_t = a].$$

**Step 2.** Expand the inner expectation over the environment outcome $(S_{t+1}, R_{t+1}) \sim p(\cdot, \cdot \mid s, a)$:

$$\mathbb{E}_\pi[R_{t+1} + \gamma G_{t+1} \mid s, a] \;=\; \sum_{s', r} p(s', r \mid s, a)\, \bigl[\, r + \gamma\, \mathbb{E}_\pi[G_{t+1} \mid s, a, s', r]\,\bigr].$$

**Step 3.** Apply the **Markov property**: once $S_{t+1} = s'$ is known, the past $(s, a, r)$ adds no information about the future, so

$$\mathbb{E}_\pi[G_{t+1} \mid S_t = s,\, A_t = a,\, S_{t+1} = s',\, R_{t+1} = r] \;=\; \mathbb{E}_\pi[G_{t+1} \mid S_{t+1} = s'] \;=\; v_\pi(s').$$

Combining,

$$v_\pi(s) \;=\; \sum_{a} \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a)\, \bigl[\, r + \gamma\, v_\pi(s')\,\bigr].$$

Each of the three lines uses exactly one MDP ingredient: the **policy**, the **dynamics**, and the **Markov property**.

</details>
</div>

<figure>
  <img src="{{ '/assets/images/notes/rl_hd/backup_v_pi.png' | relative_url }}" alt="Backup diagram for v_pi: state branches over actions weighted by policy, then over (s', r) pairs weighted by dynamics" loading="lazy">
  <figcaption>Backup diagram for $v_\pi$. Open circles are states, filled circles are state-action pairs. Blue edges are weighted by the policy $\pi(a \mid s)$; green edges by the dynamics $p(s', r \mid s, a)$.</figcaption>
</figure>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Bellman Expectation Equation for $q_\pi$)</span></p>

For every $(s, a)$,

$$q_\pi(s, a) \;=\; \sum_{s', r} p(s', r \mid s, a) \Bigl[\, r + \gamma \sum_{a'} \pi(a' \mid s')\, q_\pi(s', a') \,\Bigr].$$

The order of operations is reversed compared to $v_\pi$: we first commit to taking action $a$ in state $s$, *then* let the environment draw $(s', r)$, *then* let $\pi$ control all subsequent actions.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

$$
\begin{aligned}
q_\pi(s,a)
&= \mathbb{E}_\pi\left[G_t\mid S_t=s,A_t=a\right] \\
&= \mathbb{E}_\pi\left[R_{t+1}+\gamma G_{t+1}\mid S_t=s,A_t=a\right] \\
&= \underbrace{\mathbb{E}_\pi\left[R_{t+1}\mid S_t=s,A_t=a\right]}_{(*)}
   + \gamma\underbrace{\mathbb{E}_\pi\left[G_{t+1}\mid S_t=s,A_t=a\right]}_{(**)}.
\end{aligned}
$$

$$
\begin{aligned}
(*)
&= \sum_r p(r\mid s,a)r = \sum_{s',r} p(s',r\mid s,a)r.
\end{aligned}
$$

$$
\begin{aligned}
(**)
&= \sum_{s',r} p(s',r\mid s,a)
   \mathbb{E}_\pi\left[G_{t+1}\mid S_{t+1}=s',R_{t+1}=r,S_t=s,A_t=a\right] \\
&= \sum_{s',r} p(s',r\mid s,a)
   \sum_{a'} \pi(a'\mid s')
   \mathbb{E}_\pi\left[G_{t+1}\mid S_{t+1}=s',A_{t+1}=a'\right] \\
&= \sum_{s',r} p(s',r\mid s,a)
   \sum_{a'} \pi(a'\mid s') q_\pi(s',a').
\end{aligned}
$$

Therefore

$$
\begin{aligned}
q_\pi(s,a)
&= \sum_{s',r} p(s',r\mid s,a)
\left[ r + \gamma \sum_{a'}\pi(a'\mid s')q_\pi(s',a')\right].
\end{aligned}
$$

</details>
</div>

<figure>
  <img src="{{ '/assets/images/notes/rl_hd/backup_q_pi.png' | relative_url }}" alt="Backup diagram for q_pi: rooted at (s,a), branching over environment outcomes then over policy actions" loading="lazy">
  <figcaption>Backup diagram for $q_\pi$. The root is now a state-action pair: the environment branches first (green), then the policy branches at the next state (blue).</figcaption>
</figure>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Backup diagrams — picturing the equation)</span></p>

The Bellman expectation equation is best read off a **backup diagram**: a small two-level tree rooted at $s$ with open circles for states and filled circles for state-action pairs.

* From $s$ we branch over actions $a$, weighted by $\pi(a \mid s)$.
* From each $(s, a)$ we branch over outcomes $(s', r)$, weighted by $p(s', r \mid s, a)$.
* At each leaf we read off $r + \gamma\, v_\pi(s')$ and average everything.

A Bellman backup is therefore *always* the same operation — **one-step lookahead, then plug in the future value** — even though it shows up in many disguises in subsequent algorithms (TD(0), SARSA, expected SARSA, DP sweeps, $\dots$).

</div>

### Gridworld Example

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Gridworld as a Concrete Finite MDP)</span></p>

* **State space:** $\mathcal{S} = \lbrace (i, j) : i \in \lbrace 1, \dots, H \rbrace,\, j \in \lbrace 1, \dots, W \rbrace \rbrace$.
* **Action space:** $\mathcal{A}(s) = \lbrace \mathsf{N},\, \mathsf{S},\, \mathsf{E},\, \mathsf{W} \rbrace$ for non-terminal $s$.
* **Dynamics (deterministic).** Define a transition function $s' = f(s, a)$:
  - if the move stays inside the grid, go to the neighbouring cell;
  - if the move would leave the grid, stay put ("bump rule").

  The transition kernel collapses to

  $$
  P(\tilde s \mid s, a) \;=\; \begin{cases} 1 & \tilde s = f(s, a), \\ 0 & \text{otherwise.} \end{cases}
  $$

* **Reward.** A constant per-step cost $r(s, a, s') = -1$ until a terminal cell is reached.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(One Concrete Bellman Equation — top-left corner)</span></p>

In the simple gridworld with $r(s, a, s') = -1$ and $\gamma \in (0, 1)$, the Bellman expectation equation simplifies because the reward is constant:

$$
v_\pi(s) \;=\; \sum_{a} \pi(a \mid s)\bigl[\, -1 + \gamma\, v_\pi(f(s, a))\,\bigr].
$$

Pick the corner state $s = (1, 1)$. Under the bump rule,

$$
f((1,1), \mathsf{N}) = (1,1), \quad f((1,1), \mathsf{W}) = (1,1), \quad f((1,1), \mathsf{E}) = (1,2), \quad f((1,1), \mathsf{S}) = (2,1).
$$

For a uniform random policy $\pi(a \mid s) = \tfrac{1}{4}$,

$$
v_\pi(1,1) \;=\; \tfrac{1}{4}\bigl[-1 + \gamma\, v_\pi(1,1)\bigr] + \tfrac{1}{4}\bigl[-1 + \gamma\, v_\pi(1,1)\bigr] + \tfrac{1}{4}\bigl[-1 + \gamma\, v_\pi(1,2)\bigr] + \tfrac{1}{4}\bigl[-1 + \gamma\, v_\pi(2,1)\bigr],
$$

which simplifies to

$$
v_\pi(1,1) \;=\; -1 + \frac{\gamma}{4}\bigl(\, 2\, v_\pi(1,1) + v_\pi(1,2) + v_\pi(2,1)\,\bigr).
$$

Doing the same for *every* state gives a linear system in $\lvert\mathcal{S}\rvert$ unknowns — the Bellman expectation equation as a *single matrix equation* — which is the engine behind the policy-evaluation step of dynamic programming.

</div>

### Two States Deterministic MDP Example

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise Setting</span><span class="math-callout__name">(Value Function)</span></p>

Consider a finite MDP with two non-terminal states $\mathcal S=\lbrace s_1,s_2\rbrace$ and two actions $\mathcal A=\lbrace a_1,a_2\rbrace$.

Under the fixed policy $\pi$, each state has a single action: 

$$\pi(a_1\mid s_1)=1, \qquad \pi(a_2\mid s_2)=1.$$

The deterministic transition and reward structure is given by the following table:

| State | Action | Next state $s'$ | Reward $r$ |
|---|---|---|---|
| $s_1$ | $a_1$ | $s_2$ | $+1$ |
| $s_2$ | $a_2$ | $s_1$ | $0$ |

Let $\gamma\in(0,1)$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(From MDP Dynamics to Bellman Equations)</span></p>

Write down the Bellman expectation equations for $v_\pi(s_1)$ and $v_\pi(s_2)$ by substituting the given dynamics directly into 

$$ v_\pi(s) = \sum_{a\in\mathcal A} \pi(a\mid s) \sum_{s',r\in\mathcal S\times\lbrace 0,1\rbrace} p(s',r\mid s,a) \left(r+\gamma v_\pi(s')\right).$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

**Key idea**

Under the fixed deterministic policy, the process alternates forever:

$$s_1 \xrightarrow{r=1} s_2 \xrightarrow{r=0} s_1 \xrightarrow{r=1} \cdots$$

Thus each Bellman expectation equation contains only one nonzero transition term.

From $s_1$, action $a_1$ is chosen with probability $1$, producing reward $1$ and next state $s_2$. Hence

$$v_\pi(s_1)=1+\gamma v_\pi(s_2).$$

From $s_2$, action $a_2$ is chosen with probability $1$, producing reward $0$ and next state $s_1$. Hence

$$v_\pi(s_2)=\gamma v_\pi(s_1).$$

Therefore,

$$
\boxed{
\begin{aligned}
v_\pi(s_1)&=1+\gamma v_\pi(s_2), \\
v_\pi(s_2)&=\gamma v_\pi(s_1).
\end{aligned}}
$$

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Closing the Loop in a Two-State Cycle: Solving the Two-State Value System)</span></p>

Solve the resulting $2\times2$ linear system in closed form to obtain $v_\pi(s_1)$ and $v_\pi(s_2)$ as functions of $\gamma$. Show all algebraic steps.

</div>

<figure class="rl-diagram">
  <svg viewBox="0 0 680 330" role="img" aria-label="Two-state cycle: from s1 take action a1 for reward +1 into s2, from s2 take action a2 for reward 0 back to s1">
    <defs>
      <marker id="tc-arrow-cycle" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">
        <path d="M0,0 L10,4 L0,8 Z" fill="#64748b"></path>
      </marker>
    </defs>

    <text x="30" y="34" font-size="18" font-weight="700">Two-state cycle</text>
    <text x="30" y="56" font-size="14" class="muted">deterministic policy π,  γ ∈ (0, 1)</text>

    <circle cx="185" cy="185" r="56" class="accent"></circle>
    <text x="185" y="193" text-anchor="middle" font-size="30" font-weight="700">s₁</text>

    <circle cx="495" cy="185" r="56" class="box"></circle>
    <text x="495" y="193" text-anchor="middle" font-size="30" font-weight="700">s₂</text>

    <path d="M228 150 Q 340 78 452 150" class="line" marker-end="url(#tc-arrow-cycle)"></path>
    <text x="340" y="66" text-anchor="middle" font-size="16" font-weight="700">a₁  :  r = +1</text>

    <path d="M452 220 Q 340 292 228 220" class="line" marker-end="url(#tc-arrow-cycle)"></path>
    <text x="340" y="312" text-anchor="middle" font-size="16" font-weight="700">a₂  :  r = 0</text>
  </svg>
  <figcaption>The deterministic cycle behind the value system: from $s_1$ the only available action gives reward $+1$ and moves to $s_2$; from $s_2$ the only available action gives reward $0$ and returns to $s_1$. The process loops forever, so $v_\pi(s_1)=1+\gamma v_\pi(s_2)$ and $v_\pi(s_2)=\gamma v_\pi(s_1)$.</figcaption>
</figure>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

Substitute the second equation into the first:

$$v_\pi(s_1) = 1+\gamma\bigl(\gamma v_\pi(s_1)\bigr) = 1+\gamma^2v_\pi(s_1).$$

Therefore,

$$(1-\gamma^2)v_\pi(s_1)=1.$$

Since $\gamma\in(0,1)$, we have $1-\gamma^2>0$, and so

$$\boxed{v_\pi(s_1)=\frac{1}{1-\gamma^2}}.$$

Using $v_\pi(s_2)=\gamma v_\pi(s_1)$,

$$\boxed{v_\pi(s_2)=\frac{\gamma}{1-\gamma^2}}.$$

Equivalently,

$$
\boxed{
\begin{aligned}
v_\pi(s_1)&=\frac{1}{1-\gamma^2},\\
v_\pi(s_2)&=\frac{\gamma}{1-\gamma^2}.
\end{aligned}}
$$

Indeed, starting from $s_1$, the return is the geometric series

$$1+\gamma^2+\gamma^4+\cdots = \frac{1}{1-\gamma^2}.$$

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Discounting as a Regulator of Infinite-Horizon Value)</span></p>

Interpret the limiting values 

$$\lim_{\gamma\to0}v_\pi(s_1) \qquad\text{and}\qquad \lim_{\gamma\to1}v_\pi(s_1).$$ 

Why does $v_\pi(s_1)\to\infty$ as $\gamma\to1$? What does this say about the role of the discount factor in continuing tasks?

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

As $\gamma\to0$,

$$\boxed{\lim_{\gamma\to0}v_\pi(s_1)=1}.$$

Only the immediate reward matters; all future rewards are suppressed.

As $\gamma\to1$,

$$\boxed{\lim_{\gamma\to1}v_\pi(s_1)=+\infty}.$$

The process receives reward $1$ every two time steps forever. When $\gamma<1$, the geometric discount makes

$$1+\gamma^2+\gamma^4+\cdots$$

finite. As $\gamma\to1$, future rewards are discounted less and less, and this series approaches the undiscounted sum

$$1+1+1+\cdots,$$

which diverges.

More precisely,

$$v_\pi(s_1) = \frac{1}{(1-\gamma)(1+\gamma)} \sim \frac{1}{2(1-\gamma)} \qquad (\gamma\to1).$$

Thus, in continuing tasks, the discount factor does two things: it determines how strongly future rewards matter and ensures that an infinite stream of bounded rewards has a finite total value.

</details>
</div>

### Optimal Policies and Optimal Values

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Ordering of Policies and Optimality)</span></p>

Compare policies **pointwise** by their state values:

$$\pi \succeq \pi' \quad\Longleftrightarrow\quad v_\pi(s) \ge v_{\pi'}(s) \quad \forall s \in \mathcal{S}.$$

A policy $\pi^\ast$ is **optimal** if it dominates every other policy, $\pi^\ast \succeq \pi$ for all $\pi$. Equivalently, $\pi^\ast$ is at least as good as every other policy in every state simultaneously.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Many optimal policies, one optimal value)</span></p>

A few facts about optimality that are often glossed over:

* **Optimal policies need not be unique.** Whenever two actions tie for the argmax in some state, *any* of them gives an optimal policy. Tie-breaking does not change the value.
* **Optimal value functions are unique.** For any discounted finite MDP, $v_\ast$ and $q_\ast$ are well-defined functions. Any optimal policy attains *the same* values.
* **A deterministic optimal policy always exists.** This is non-obvious — one might worry that some MDPs require randomisation — but for finite discounted MDPs there is always at least one purely deterministic policy that is optimal.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Optimal Value Functions)</span></p>

The **optimal state-value function** is the best $v_\pi(s)$ achievable over policies:

$$v_*(s) \;\doteq\; \max_{\pi}\, v_\pi(s).$$

The **optimal action-value function** is defined analogously:

$$q_*(s, a) \;\doteq\; \max_{\pi}\, q_\pi(s, a).$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Optimal State-value function is maximal action-value function)</span></p>

The two are related by maximising over actions:

$$v_*(s) \;=\; \max_{a}\, q_*(s, a).$$

</div>

### Bellman Optimality Equations

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Bellman Optimality Equation for $v_*$)</span></p>

For every state $s$,

$$
v_*(s) \;=\; \max_{a \in \mathcal{A}(s)} \sum_{s', r} p(s', r \mid s, a)\bigl[\, r + \gamma\, v_*(s')\,\bigr].
$$

Compare with the Bellman *expectation* equation: the **expectation over actions weighted by $\pi$** has been replaced by a **max over actions**. Interpretation: choose the best first action, then act optimally thereafter.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Bellman Optimality Equation for $q_*$)</span></p>

For every $(s, a)$,

$$q_*(s, a) \;=\; \sum_{s', r} p(s', r \mid s, a)\Bigl[\, r + \gamma\, \max_{a'}\, q_*(s', a')\,\Bigr].$$

Take action $a$ once, then — after arriving in $s'$ — pick the best possible *next* action. This is exactly the equation behind value iteration and (later) Q-learning.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Expectation backup vs. max backup)</span></p>

The Bellman expectation and Bellman optimality equations differ in **exactly one** place:

* Expectation backup: average over actions according to $\pi$.
* Optimality backup: replace that average by a maximisation.

That single change is what turns *evaluation of a fixed policy* into *control*. Diagrammatically, the upper branching (over actions) becomes a max instead of a weighted sum; the lower branching (over environment outcomes) is unchanged because randomness in the world is not something we can max away.

</div>

<figure>
  <img src="{{ '/assets/images/notes/rl_hd/backup_optimality.png' | relative_url }}" alt="Side-by-side backup diagrams comparing expectation backup and max backup" loading="lazy">
  <figcaption>The two Bellman backups differ at exactly one layer: expectation averages over actions weighted by $\pi$ (left); optimality keeps only the best action and discards the others (right).</figcaption>
</figure>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(An Average Cannot Exceed Its Best Component)</span></p>

In this task, you will show that randomizing over actions at optimality is never beneficial; that is, there always exists an optimal policy that is **deterministic**.

---

**(a)** Let $\pi$ be any possibly stochastic policy. Using the identity

$$v_\pi(s) = \sum_a \pi(a\mid s)q_\pi(s,a),$$

show that 

$$v_\pi(s) \le \max_{a\in\mathcal A(s)}q_\pi(s,a)$$

for all $s\in\mathcal S$. When does equality hold, and what does this say about the form of the maximizing policy?

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

For each fixed state $s$,

$$v_\pi(s) =\sum_a\pi(a\mid s)q_\pi(s,a) \le \sum_a\pi(a\mid s)\max_{a'}q_\pi(s,a') =\max_{a'}q_\pi(s,a')\sum_a\pi(a\mid s) =\max_{a'}q_\pi(s,a').$$

Equality holds if the policy assigns positive probability only to actions that maximize $q_\pi(s,a)$. Thus a maximizing policy may be deterministic: choose any action $a_s\in\arg\max_{a}q_\pi(s,a)$.

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Greedy Selection Recovers the Optimal Fixed Point)</span></p>

**(b)** Now let $v^\ast$ and $q^\ast$ denote the optimal value functions, and define the greedy deterministic policy 

$$\pi_d(s) \in \arg\max_a q^*(s,a).$$

Show that 

$$v_{\pi_d}(s)=v^*(s)$$

for all $s\in\mathcal S$ by verifying that $v_{\pi_d}$ satisfies the Bellman optimality equation. 

*Hint.* The Bellman optimality equation has a unique fixed point. Use this to conclude equality rather than merely an inequality.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

We show that $v_{\pi_d}=v^\ast$. Since $v^\ast(s)=\max_a q^\ast(s,a)$, and the way $\pi_d(s)$ is chosen, $v^\ast(s)=q^\ast(s,\pi_d(s))$. 

Using the Bellman optimality equation for $q^*$,

$$v^*(s) =q^*(s,\pi_d(s)) =\sum_{s',r}p(s',r\mid s,\pi_d(s)) \left[r+\gamma\max_{a'}q^*(s',a')\right] =\sum_{s',r}p(s',r\mid s,\pi_d(s)) \left[r+\gamma v^*(s')\right].$$

This is exactly the Bellman expectation equation for the deterministic policy $\pi_d$.

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Why History-Dependent Mixing Adds No Value)</span></p>

**(c)** Explain in one or two sentences why a stochastic mixture over actions at state $s$ can never exceed the optimal value $v^\ast(s)$, even if the mixture weights are chosen adaptively based on the full history.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

A stochastic mixture over actions at a state $s$ gives a weighted average of the action values $\sum_a\pi(a\mid s)q^*(s,a)$. A weighted average cannot exceed the largest value being averaged, so

$$\sum_a\pi(a\mid s)q^*(s,a)\le \max_a q^*(s,a)=v^*(s).$$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Greedy Policies from Optimal Values)</span></p>

Given the optimal state-value function $v_*$, define the **greedy policy** by

$$\pi_{\text{greedy}}(s) \;\in\; \arg\max_{a} \sum_{s', r} p(s', r \mid s, a)\bigl[\, r + \gamma\, v_*(s')\,\bigr].$$

Then **any policy greedy with respect to $v_\ast$ is optimal.** With $q_\ast$ the rule is even simpler — and no model is required:

$$\pi^*(s) \;\in\; \arg\max_{a}\, q_*(s, a).$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Punchline of MDP theory)</span></p>

$$\boxed{\text{Once you can solve the Bellman optimality equations, optimal control reduces to a one-step argmax in every state.}}$$

</div>

### Summary

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(Method Map — the five core objects)</span></p>

| Object         | Meaning                                  | Bellman form                                                  |
| :------------- | :--------------------------------------- | :------------------------------------------------------------ |
| $\pi(a \mid s)$  | behaviour rule                           | choose actions in each state                                  |
| $v_\pi(s)$     | value of state $s$ under $\pi$           | expectation over action *and* environment                     |
| $q_\pi(s, a)$  | value of taking $a$ then following $\pi$ | expectation over environment, then policy                     |
| $v_*(s)$       | best achievable state value              | max over actions                                              |
| $q_*(s, a)$    | best achievable value after forcing $(s, a)$ | one-step transition + max over future actions             |

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(Core Takeaways)</span></p>

* MDPs extend bandits by adding **state** and **dynamics** — actions now affect *what happens next*, not just *what we get now*.
* The **Markov property** is a constraint on state design: the state must summarise enough of the past for the future to be predictable from it alone.
* **Return** formalises the agent's objective; **discounting** keeps it finite for continuing tasks and underwrites the contraction-mapping arguments that make later algorithms converge.
* **Policies** define behaviour; **value functions** evaluate long-term quality. Two fundamental problems: **prediction** ($v_\pi$, $q_\pi$ for a fixed $\pi$) and **control** (find $\pi^\ast$).
* **Bellman expectation equations** convert long-horizon reasoning into recursive one-step lookahead; **Bellman optimality equations** define the unique optimal value functions.
* Once $q_\ast$ is known, optimal control collapses to $\arg\max_a q_\ast(s, a)$ — a *one-step* decision per state, no planning required.

**Final message.** RL on MDPs is built around a single core idea: **solve the Bellman equations exactly or approximately**. Everything that follows — dynamic programming, Monte Carlo, TD, Q-learning, policy gradients, actor-critic — is a different way of attacking those equations.

</div>

## Lecture 4: Dynamic Programming

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Setup</span><span class="math-callout__name">(Dynamic Programming — planning with a known model)</span></p>

The previous lecture ended with two punchlines:

* the **Bellman expectation equations** characterise $v_\pi$ and $q_\pi$ for a fixed policy;
* the **Bellman optimality equations** characterise $v_\ast$ and $q_\ast$, and once we have $q_\ast$ optimal control is a one-step argmax.

Both are *equations*, not algorithms. Dynamic programming is the first systematic answer to the question **"how do we actually solve them?"** — under the strongest possible assumption that we have a **perfect model** of the environment in the form of the one-step dynamics $p(s', r \mid s, a)$.

In this lecture there is no learning from experience yet: no sampled rewards, no exploration, no noise. The agent has access to the full transition kernel and can compute every required expectation *exactly*. We will treat DP as **planning with a model**, and the rest of the course will progressively relax this assumption — first by sampling returns (Monte Carlo), then by bootstrapping from sampled transitions (TD), and finally by combining both with function approximation.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(Why DP still matters even though it is rarely run as-is)</span></p>

Classical DP has two practical handicaps:

* it requires a **known model** $p(s', r \mid s, a)$, which is precisely what real-world RL does *not* have;
* it sweeps the entire state space, so its cost scales with $\lvert \mathcal{S} \rvert$ (and with $\lvert \mathcal{S} \rvert \lvert \mathcal{A} \rvert$ for $q$-style updates).

But every later algorithm in this course inherits its conceptual skeleton from DP:

* **prediction** = repeatedly apply a Bellman *expectation* backup;
* **improvement** = act greedily with respect to the current value estimate;
* **control** = interleave the two.

Monte Carlo replaces the expectation by sample averages; TD replaces it by a single sampled transition; Q-learning replaces it by a sampled max backup; actor-critic replaces the greedy step by a gradient step on a parameterised policy. The structure is always the same — DP is the *clean* version, with all randomness averaged away.

</div>

### From Bellman Equations to Update Rules

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Setting</span><span class="math-callout__name">(Finite MDP with Known Dynamics)</span></p>

From this lecture onward we assume a **finite MDP**:

* finite state set $\mathcal{S}$,
* finite action set $\mathcal{A}(s)$ at each state,
* bounded rewards (so all expected returns are well-defined),
* known one-step dynamics $p(s', r \mid s, a)$.

The crucial consequence of "known dynamics" is that every expectation in a Bellman backup — over actions, over next states, over rewards — can be computed *exactly* as a finite sum. No sampling is needed. This is precisely what separates DP from Monte Carlo and temporal-difference methods in the next lectures.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Idea</span><span class="math-callout__name">(DP in one sentence)</span></p>

Dynamic programming solves a sequential decision problem by repeatedly:

1. **estimating how good states are** (policy evaluation),
2. **choosing better actions using those estimates** (policy improvement),
3. **repeating until nothing changes** (iteration to a fixed point).

The three primitives are **evaluation**, **improvement**, and **iteration**. The two main DP algorithms — *policy iteration* and *value iteration* — differ only in *how much* evaluation is done between improvement steps.

</div>

### Policy Evaluation (Prediction)

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Prediction Problem)</span></p>

Given a policy $\pi$, compute its state-value function

$$v_\pi(s) \;=\; \mathbb{E}_\pi[\, G_t \mid S_t = s \,] \qquad \text{for all } s \in \mathcal{S}.$$

This is exactly the *prediction* problem from the previous lecture. The Bellman expectation equation supplies the recursive characterisation we need:

$$v_\pi(s) \;=\; \sum_{a} \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a)\bigl[\, r + \gamma\, v_\pi(s')\,\bigr].$$

There is **one equation per state** in the unknowns $\lbrace v_\pi(s) \rbrace_{s \in \mathcal{S}}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Two ways to read the same equation)</span></p>

Looking at $v_\pi = r_\pi + \gamma P_\pi v_\pi$ (the vector form developed below), one can think of policy evaluation in two different ways:

* **Algebraic.** It is a *linear system* with $\lvert \mathcal{S} \rvert$ equations and $\lvert \mathcal{S} \rvert$ unknowns. Solve it directly with linear algebra.
* **Fixed-point.** It is the fixed-point equation $v = T_\pi v$ of the **Bellman expectation operator** $T_\pi$. Iterate $v_{k+1} = T_\pi v_k$ until it stops moving.

Both viewpoints give the *same* answer; they differ only in computational strategy. Iterative methods are the ones that survive into the model-free regime, so we develop them in detail.

</div>

#### Iterative Policy Evaluation

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Iterative Policy Evaluation)</span></p>

Initialise $v_0(s)$ arbitrarily for all $s \in \mathcal{S}$ (and $v(\text{terminal}) = 0$). Then repeatedly apply the **Bellman expectation update**:

$$v_{k+1}(s) \;=\; \sum_{a} \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a)\bigl[\, r + \gamma\, v_k(s')\,\bigr].$$

For discounted finite MDPs with $\gamma < 1$, the sequence $\lbrace v_k \rbrace$ converges to the true value $v_\pi$ as $k \to \infty$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why this is called an "expected update")</span></p>

Each update uses an **expectation over all possible successor states and rewards**, weighted by the known dynamics. This is what DP terminology calls an *expected update* — in contrast to:

* a **sample update**, used by Monte Carlo and TD, which replaces the sum $\sum_{s', r} p(s', r \mid s, a)$ by a single sampled transition,
* a **max update**, used by value iteration, which replaces $\sum_a \pi(a \mid s)$ by $\max_a$.

The two key features of an expected update are **bootstrapping** (using the current estimate $v_k(s')$ in place of the true $v_\pi(s')$) and **full backups** (averaging over the entire next-state distribution).

</div>

<figure>
  <img src="{{ '/assets/images/notes/rl_hd/backup_v_pi.png' | relative_url }}" alt="One-step backup diagram: state s branches over actions then over successor states" loading="lazy">
  <figcaption>Backup diagram for a Bellman expectation update: <em>new value of $s$ = expected immediate reward + $\gamma\,\cdot\,$ expected successor value</em>. A Bellman backup is always the same operation — one-step lookahead, then plug in the current value estimate at each leaf.</figcaption>
</figure>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Two-array vs. in-place updates)</span></p>

There are two natural ways to implement the iteration in code:

* **Two-array (synchronous) update.** Keep arrays `old` and `new`. The update for $v_{k+1}(s)$ reads only from `old`; after the sweep, swap arrays. Every state in iteration $k+1$ uses values from iteration $k$.
* **In-place (asynchronous) update.** Maintain a single array $V$ and overwrite $V(s)$ as soon as it is computed. Later states in the same sweep already see the freshly updated values of earlier states.

Both versions target the **same fixed point** $v_\pi$, and both converge under the same conditions. In-place is what one would actually code — it uses half the memory and typically propagates information faster (especially if the sweep order is chosen well), because newer estimates are available immediately for downstream states.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Iterative Policy Evaluation — pseudocode)</span></p>

Initialise $V(s)$ arbitrarily for all $s \in \mathcal{S}$, and $V(\text{terminal}) = 0$.

Repeat:

1. $\Delta \leftarrow 0$.
2. For each state $s \in \mathcal{S}$:
   * $v \leftarrow V(s)$,
   * $V(s) \leftarrow \sum_{a} \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a)\bigl[\, r + \gamma\, V(s')\,\bigr]$,
   * $\Delta \leftarrow \max\bigl(\Delta,\, \lvert v - V(s)\rvert\bigr)$.

until $\Delta < \theta$ for a small tolerance $\theta > 0$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Why a tolerance stop works)</span></p>

The Bellman operator is a $\gamma$-contraction on $\mathbb{R}^{\lvert \mathcal{S}\rvert}$ in the max-norm (proved later), so a sweep with maximum change $\Delta$ guarantees 

$$\lVert V - v_\pi \rVert_\infty \le \Delta / (1 - \gamma).$$

Stopping at $\Delta < \theta$ thus controls the worst-case error.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Turning Successive-Iterate Error into an Optimality Guarantee)</span></p>

Derive a stopping criterion. Given a tolerance $\varepsilon>0$, show that once 

$$\lVert v_{k+1}-v_k\rVert_\infty < \frac{\varepsilon(1-\gamma)}{\gamma}$$

(or equivalently, $\lVert v_{k+1}-v_k\rVert_\infty \frac{\gamma}{1-\gamma} < \varepsilon$) the current iterate satisfies 

$$\lVert v_k-v^*\rVert_\infty<\varepsilon.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

I believe there is an error in the problem statement. I am considering the statement about $\|v_{k+1} - v^\ast\|\_\infty$:

$$\lVert v_k-v^*\rVert_\infty \le \lVert v_{k+1}- v_k\rVert_\infty + \lVert v_{k+1}-v^*\rVert_\infty$$

$$\lVert v_{k+1}-v^*\rVert_\infty \le \gamma\lVert v_{k+1}- v^*\rVert_\infty + \gamma\lVert v_{k+1}-v_k\rVert_\infty$$

$$(1-\gamma)\lVert v_{k+1}-v^*\rVert_\infty < \gamma\frac{\varepsilon(1-\gamma)}{\gamma}$$

$$\lVert v_{k+1}-v^*\rVert_\infty < \varepsilon$$

<figure>
  <img src="{{ '/assets/images/notes/rl_hd/stopping_criterion.png' | relative_url }}" alt="Log-scale convergence plot of the measurable successive-iterate gap, its amplified guaranteed bound, and the true error, all versus iteration, with the tolerance and stopping threshold marked" loading="lazy">
  <figcaption>Iterative policy evaluation $v_{k+1}=T_\pi v_k$ on an 8-state chain ($\gamma=0.9$). The orange measurable gap $\lVert v_{k+1}-v_k\rVert_\infty$ is all the algorithm ever sees; scaling it by $\gamma/(1-\gamma)$ gives the red bound, which sits above the blue true error $\lVert v_{k+1}-v^\ast\rVert_\infty$ at every iteration. The instant the gap falls below the threshold $\varepsilon(1-\gamma)/\gamma$ (orange dotted), the certified bound — and hence the true error — is guaranteed below the tolerance $\varepsilon$ (green dotted).</figcaption>
</figure>

</details>
</div>

<figure class="rl-diagram">
  <svg viewBox="0 0 680 300" role="img" aria-label="A value-space line showing how the measurable step between successive iterates certifies the unknown distance to the fixed point">
    <text x="30" y="34" font-size="18" font-weight="700">From a measurable step to a certified error</text>
    <text x="30" y="56" font-size="14" class="muted">one backup T is a γ-contraction, so the visible gap between iterates bounds the invisible error</text>

    <line x1="120" y1="118" x2="640" y2="118" class="line"></line>

    <line x1="170" y1="113" x2="170" y2="150" class="line" stroke-dasharray="4 4"></line>
    <line x1="430" y1="113" x2="430" y2="150" class="line" stroke-dasharray="4 4"></line>
    <line x1="600" y1="113" x2="600" y2="150" class="line" stroke-dasharray="4 4"></line>

    <circle cx="170" cy="118" r="6" class="box"></circle>
    <circle cx="430" cy="118" r="6" class="box"></circle>
    <circle cx="600" cy="118" r="6" class="accent"></circle>

    <text x="170" y="102" text-anchor="middle" font-size="15" font-weight="700">v<tspan baseline-shift="sub" font-size="10">k</tspan></text>
    <text x="430" y="102" text-anchor="middle" font-size="15" font-weight="700">v<tspan baseline-shift="sub" font-size="10">k+1</tspan></text>
    <text x="600" y="102" text-anchor="middle" font-size="15" font-weight="700">v<tspan baseline-shift="super" font-size="10">*</tspan></text>
    <text x="600" y="88" text-anchor="middle" font-size="12" class="muted">fixed point</text>

    <line x1="170" y1="150" x2="430" y2="150" class="strong-line"></line>
    <line x1="170" y1="145" x2="170" y2="155" class="strong-line"></line>
    <line x1="430" y1="145" x2="430" y2="155" class="strong-line"></line>
    <text x="300" y="174" text-anchor="middle" font-size="15" font-weight="700">‖v<tspan baseline-shift="sub" font-size="10">k+1</tspan> − v<tspan baseline-shift="sub" font-size="10">k</tspan>‖<tspan baseline-shift="sub" font-size="10">∞</tspan></text>
    <text x="300" y="192" text-anchor="middle" font-size="13" class="muted">measurable step</text>

    <line x1="430" y1="150" x2="600" y2="150" class="line" stroke-dasharray="6 4"></line>
    <line x1="600" y1="145" x2="600" y2="155" class="line"></line>
    <text x="515" y="174" text-anchor="middle" font-size="15" font-weight="700">‖v<tspan baseline-shift="sub" font-size="10">k+1</tspan> − v<tspan baseline-shift="super" font-size="10">*</tspan>‖<tspan baseline-shift="sub" font-size="10">∞</tspan></text>
    <text x="515" y="192" text-anchor="middle" font-size="13" class="muted">true error (unknown)</text>

    <rect x="70" y="212" width="540" height="72" rx="8" class="accent"></rect>
    <text x="340" y="242" text-anchor="middle" font-size="15" font-weight="700">‖v<tspan baseline-shift="sub" font-size="10">k+1</tspan> − v<tspan baseline-shift="super" font-size="10">*</tspan>‖<tspan baseline-shift="sub" font-size="10">∞</tspan>  ≤  γ ⁄ (1−γ) · ‖v<tspan baseline-shift="sub" font-size="10">k+1</tspan> − v<tspan baseline-shift="sub" font-size="10">k</tspan>‖<tspan baseline-shift="sub" font-size="10">∞</tspan></text>
    <text x="340" y="268" text-anchor="middle" font-size="13" class="muted">so once the step drops below  ε (1−γ) ⁄ γ,  the true error is below ε</text>
  </svg>
  <figcaption>The step you can measure, $\lVert v_{k+1}-v_k\rVert_\infty$, controls the error you cannot, $\lVert v_{k+1}-v^\ast\rVert_\infty$. The triangle inequality gives $\lVert v_{k+1}-v^\ast\rVert_\infty \le \gamma\lVert v_k-v^\ast\rVert_\infty \le \gamma\bigl(\lVert v_{k+1}-v_k\rVert_\infty+\lVert v_{k+1}-v^\ast\rVert_\infty\bigr)$, and solving for the true error yields the amplification factor $\gamma/(1-\gamma)$. Choosing the stopping threshold $\varepsilon(1-\gamma)/\gamma$ exactly cancels it.</figcaption>
</figure>

#### Policy Evaluation as a Linear System

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Vector Form of Policy Evaluation)</span></p>

Order the $n = \lvert \mathcal{S} \rvert$ states and stack the value function into a vector $v \in \mathbb{R}^n$. Define:

* **Expected one-step reward under $\pi$:**

  $$r_\pi(s_i) \;=\; \sum_{a} \pi(a \mid s_i) \sum_{s', r} p(s', r \mid s_i, a)\, r,$$

  stacked into a vector $r_\pi \in \mathbb{R}^n$.
* **Policy-induced transition matrix:**

  $$(P_\pi)_{ij} \;=\; \sum_{a} \pi(a \mid s_i)\, P(s_j \mid s_i, a),$$

  the probability of moving from $s_i$ to $s_j$ in one step under $\pi$.

The Bellman expectation equation for $\pi$ then collapses to the linear identity

$$v_\pi \;=\; r_\pi + \gamma\, P_\pi v_\pi.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Using the identity $v_\pi(s)=\sum_a \pi(a\mid s)q_\pi(s,a)$ and substituting the result of Task 1 gives

$$
\begin{aligned}
v_\pi(s)
&= \sum_a \pi(a\mid s)q_\pi(s,a) \\
&= \sum_a \pi(a\mid s)
\sum_{s',r}p(s',r\mid s,a)
\left[ r+\gamma\sum_{a'}\pi(a'\mid s')q_\pi(s',a')\right] \\
&= \underbrace{\sum_a \pi(a\mid s)\sum_{s',r}r\,p(s',r\mid s,a)}_{(*)}
+ \gamma
\underbrace{\sum_a \pi(a\mid s)\sum_{s',r}p(s',r\mid s,a)
\sum_{a'}\pi(a'\mid s')q_\pi(s',a')}_{(**)}.
\end{aligned}
$$

The first term is the policy-induced reward by definition:

$$(*)=(r_\pi)_s.$$

For the second term:

$$
\begin{aligned}
(**)
&= \sum_a \pi(a\mid s)\sum_{s'}p(s'\mid s,a)v_\pi(s') = \sum_{s'}\left(\sum_a\pi(a\mid s)p(s'\mid s,a)\right)v_\pi(s') \\
&= \sum_{s'}(P_\pi)_{s,s'}v_\pi(s') = (P_\pi v_\pi)_s.
\end{aligned}
$$


</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Uniqueness of Bellman expectation equation from a vector form)</span></p>

$$v_\pi(s)=(r_\pi)_s+\gamma(P_\pi v_\pi)_s$$ 

or wirtten in vectorized form 

$$v_\pi = r_\pi + \gamma P_\pi v_\pi$$

* We get $v_\pi=(I-\gamma P_\pi)^{-1}r_\pi$, if $I-\gamma P_\pi$ is invertible. 
* Since $P_\pi$ is row-stochastic, every eigenvalue $\lambda$ of $P_\pi$ satisfies $\lvert\lambda\rvert\leq 1$. 
* Therefore every eigenvalue of $I-\gamma P_\pi$ has the form $1-\gamma\lambda$. 
* Since $\gamma\in[0,1)$ and $\lvert\lambda\rvert\leq 1$, we cannot have $1-\gamma\lambda=0$. 
* Hence $0$ is not an eigenvalue of $I-\gamma P_\pi$, so $I-\gamma P_\pi$ is invertible.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(A fixed policy turns the MDP into a Markov chain)</span></p>

* Once $\pi$ is fixed, the action is no longer a decision
  * it is a random variable with distribution $\pi(\cdot \mid s)$.
  * marginalising over it leaves only a **state-to-state transition kernel** $P_\pi$: 
    * $(P_\pi)\_{ij} = \sum_{a} \pi(a \mid s_i) P(s_j \mid s_i, a),$
  * plus a state-indexed expected reward $r_\pi$:
    * $r_\pi(s_i) = \sum_{a} \pi(a \mid s_i) \sum_{s', r} p(s', r \mid s_i, a) r.$
*  This is exactly a (reward-augmented) **Markov chain**: 
   *  the MDP structure has been collapsed to its dynamics-under-$\pi$. 
   *  every later **prediction algorithm** (TD(0), every-visit MC, $\lambda$-returns) is, in effect, trying to compute $v_\pi$ for this chain.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Bellman Expectation Operator)</span></p>

Define $T_\pi : \mathbb{R}^n \to \mathbb{R}^n$ by

$$T_\pi v \;\doteq\; r_\pi + \gamma\, P_\pi v.$$

$T_\pi$ is an **affine map** on value vectors. The iterative-policy-evaluation update is simply

$$v_{k+1} \;=\; T_\pi v_k,$$

and the true value function $v_\pi$ is the unique vector satisfying 

$$v_\pi = T_\pi v_\pi$$

— the **fixed point** of $T_\pi$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Two equivalent solution strategies)</span></p>

Rearranging the fixed-point equation:

$$(I - \gamma P_\pi)\, v_\pi \;=\; r_\pi.$$

For finite discounted MDPs with $\gamma < 1$, the matrix $I - \gamma P_\pi$ is invertible (its spectrum lies in $\lbrace 1 - \gamma \mu : \lvert \mu \rvert \le 1\rbrace$, none of which crosses the origin), so

$$v_\pi \;=\; (I - \gamma P_\pi)^{-1}\, r_\pi.$$

**Two viewpoints, one answer:**

* **Linear-system view:** solve once 
  
  $$(I - \gamma P_\pi)\, v_\pi = r_\pi$$

* **Fixed-point view:** iterate until convergence
  
  $$v_{k+1} = T_\pi v_k$$

**Why iteration is preferred in practice.**

* A direct linear solve costs $O(\lvert \mathcal{S} \rvert^3)$, infeasible for large state spaces.
* Transition matrices are typically **huge and very sparse**; iterative sweeps exploit sparsity cheaply.
* Policies change repeatedly during control — re-solving a fresh linear system after each tiny policy change is wasteful.
* Iterative Bellman updates **generalise to model-free RL**, where $P_\pi$ is unknown and $T_\pi v$ must be sample-approximated. The linear-algebra view does not.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Why is spectral radius of a row stochastic matrix $leq 1$?</summary>

Because two separate facts combine:

**1. Row sums equal to $1$ force $1$ to be an eigenvalue.**

Let

$$\mathbf 1=(1,\ldots,1)^\top.$$

The $i$-th component of $P\mathbf 1$ is

$$(P\mathbf 1)_i=\sum_{j=1}^n P_{ij}=1.$$

Hence

$$P\mathbf 1=\mathbf 1,$$

so $\mathbf 1$ is an eigenvector with eigenvalue $1$. Therefore

$$\rho(P)\ge 1.$$

**2. Nonnegativity prevents any eigenvalue from having modulus greater than $1$.**

If $Pv=\lambda v$, choose $i$ such that

$$|v_i|=\max_j |v_j|.$$

Then

$$|\lambda|,|v_i| =|(Pv)_i| =\left|\sum_j P_{ij}v_j\right| \le \sum_j P_{ij}|v_j| \le |v_i|\sum_j P_{ij} =|v_i|.$$

Since $v_i\neq 0$,

$$|\lambda|\le 1.$$

Thus every eigenvalue lies in the closed unit disk, while $1$ itself is an eigenvalue. Consequently,

$$\boxed{\rho(P)=1}.$$

The nonnegativity assumption is essential. Row sums equal to $1$ alone only guarantee that $1$ is an eigenvalue; for example,

$$
A=
\begin{pmatrix}
2&-1\\
0&1
\end{pmatrix}
$$

has row sums $1$, but eigenvalues $2$ and $1$, so $\rho(A)=2$.

</details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Row stochastic matrix could have negative eigenvalues</summary>

For a row-stochastic matrix $P$, the correct statement is

$$\boxed{|\lambda|\le 1}$$

for every eigenvalue $\lambda$, not necessarily $0<\lambda\le 1$.

Eigenvalues may be negative. For example,

$$
P=
\begin{pmatrix}
0&1\\
1&0
\end{pmatrix}
$$

is row-stochastic, but its eigenvalues are

$$1,-1.$$

They may also be complex. The cyclic permutation matrix

$$
P=
\begin{pmatrix}
0&1&0\\
0&0&1\\
1&0&0
\end{pmatrix}
$$

has eigenvalues

$$1,\quad e^{2\pi i/3},\quad e^{4\pi i/3}.$$

All have modulus $1$, but only one lies in $(0,1]$.

Zero is also possible. For instance,

$$
P=
\begin{pmatrix}
1&0\\
1&0
\end{pmatrix}
$$

has eigenvalues $1$ and $0$.

Thus the spectrum satisfies

$$\sigma(P)\subseteq{z\in\mathbb C:|z|\le1},$$

and because $1$ is always an eigenvalue,

$$\rho(P)=1.$$

Eigenvalues lie in $[0,1]$ only under extra assumptions, for example when $P$ is symmetric positive semidefinite.

</details>
</div>


<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Geometric picture)</span></p>

Iterative policy evaluation is a sequence of points $v_0,\, v_1,\, v_2, \dots$ in $\mathbb{R}^{\lvert \mathcal{S} \rvert}$ obtained by repeatedly applying the same affine map $T_\pi$. Each step pulls $v_k$ closer to $v_\pi$:

$$\lVert v_{k+1} - v_\pi \rVert_\infty \;=\; \lVert T_\pi v_k - T_\pi v_\pi \rVert_\infty \;\le\; \gamma\,\lVert v_k - v_\pi \rVert_\infty.$$

So convergence is **geometric with rate $\gamma$** — the smaller $\gamma$, the faster the chase. This same contraction estimate underwrites the convergence of every algorithm built on top of expected backups.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

For the policy evaluation operator,

$$
\begin{aligned}
(\Tpi u)(s)-(\Tpi v)(s)
&= \sum_{s',r}p(s',r\mid s,\pi(s))\gamma\bigl(u(s')-v(s')\bigr).
\end{aligned}
$$

$$
\begin{aligned}
\left|(\Tpi u)(s)-(\Tpi v)(s)\right|
&\le \gamma\sum_{s',r}p(s',r\mid s,\pi(s))|u(s')-v(s')| \\
&\le \gamma\lVert u-v\rVert_\infty \sum_{s',r}p(s',r\mid s,\pi(s)) \\
&= \gamma\lVert u-v\rVert_\infty.
\end{aligned}
$$

the maximum over $s$ gives 

$$\lVert T_\pi u- T_\pi v\rVert_\infty \le \gamma\lVert u-v\rVert_\infty$$

</details>
</div>

<figure>
  <img src="{{ '/assets/images/notes/rl_hd/contraction_fixed_point.png' | relative_url }}" alt="A two-dimensional trace of v_0, v_1, ... shrinking towards the fixed point v_pi" loading="lazy">
  <figcaption>Each Bellman backup is the same affine map applied to the value vector. The iterates form a contracting sequence in $\mathbb{R}^{\lvert \mathcal{S} \rvert}$ — every step shrinks the distance to the fixed point $v_\pi$ by a factor of at most $\gamma$ in the max-norm. Dashed circles indicate the successive worst-case error bounds.</figcaption>
</figure>

#### Gridworld: Evaluation in Action

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Small Gridworld under the Random Policy)</span></p>

* **Non-terminal states:** $\mathcal{S} = \lbrace 1, 2, \dots, 14 \rbrace$, arranged on a $4 \times 4$ grid with two terminal corners.
* **Actions:** $\lbrace \mathsf{up}, \mathsf{down}, \mathsf{left}, \mathsf{right}\rbrace$, with deterministic transitions; bumping into a wall leaves the state unchanged.
* **Reward:** $R_{t+1} = -1$ on every transition, until termination.
* **Policy:** equiprobable random, $\pi(a \mid s) = \tfrac{1}{4}$ for all $a$.

Under the random policy the value $v_\pi(s)$ has a very transparent meaning:

* states adjacent to a terminal have small (close to zero) magnitudes — the agent escapes quickly on average;
* states far from a terminal accumulate more negative values — the random walk takes longer to terminate.

**One Bellman equation in the corner.** For the state immediately to the right of the upper-left terminal, $\mathsf{up}$ keeps the agent in place, $\mathsf{right}$ and $\mathsf{down}$ move to neighbouring non-terminals $s_R$ and $s_D$, and $\mathsf{left}$ enters the terminal. Averaging over the four equiprobable actions:

$$v_\pi(s) \;=\; \tfrac{1}{4}[-1 + \gamma\, v_\pi(s)] \;+\; \tfrac{1}{4}[-1 + \gamma\, v_\pi(s_R)] \;+\; \tfrac{1}{4}[-1 + \gamma\, v_\pi(s_D)] \;+\; \tfrac{1}{4}[-1 + \gamma \cdot 0].$$

A state's value depends only on the values of states **reachable in one step** — that is the whole structural content of a Bellman equation.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(How value information propagates)</span></p>

Watching the iterates $v_k$ in gridworld is a useful intuition pump:

* At $k = 0$, all values are zero.
* After one sweep, every non-terminal state picks up only the immediate step cost $-1$ (because every successor still has value zero).
* After more sweeps, terminal-proximity information **diffuses outward** from the terminal cells through the grid.

Iterative policy evaluation is, geometrically, a kind of *information diffusion* over the state graph — one bond of distance per sweep, weighted by $\gamma$.

</div>

<figure>
  <img src="{{ '/assets/images/notes/rl_hd/policy_eval_gridworld.png' | relative_url }}" alt="Iterative policy evaluation on a 4x4 gridworld at iterations k=0,1,2,3,10 and infinity, with greedy-action arrows superimposed" loading="lazy">
  <figcaption>Iterative policy evaluation on the small $4 \times 4$ gridworld under the equiprobable random policy. Each panel shows the value estimate $v_k$ after $k$ sweeps; arrows point in the direction(s) of the greedy action(s) at every non-terminal cell. <strong>Two things to notice:</strong> (i) information about the terminals diffuses outward one bond per sweep, taking many iterations to reach $v_\pi$; (ii) the <em>greedy</em> policy with respect to $v_k$ is already optimal by $k = 3$ — long before evaluation converges. This is the intuition that justifies value iteration's "truncate after one sweep" idea.</figcaption>
</figure>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Section Takeaway</span><span class="math-callout__name">(Policy Evaluation)</span></p>

* Policy evaluation computes $v_\pi$ for a fixed policy $\pi$.
* With a known model, the Bellman expectation equation gives an **exact** update rule.
* Iterative policy evaluation applies that update repeatedly until convergence.
* Equivalent linear-algebra view: solve 
  
  $$(I - \gamma P_\pi)\, v_\pi = r_\pi.$$

We can now answer *"how good is $\pi$?"* — the natural next question is **"how do we improve it?"**.

</div>

### Policy Improvement

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Idea</span><span class="math-callout__name">(From prediction to control)</span></p>

Policy evaluation answers a passive question — **how good is $\pi$?** — and produces a value function $v_\pi$. Control asks a stronger, more useful question — **how can we make $\pi$ better?** — and produces a *new* policy $\pi'$. The key insight is that the value function $v_\pi$ is *already* enough to do this, because comparing two candidate first actions only requires a one-step lookahead and a copy of $v_\pi$ at the successor states.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(One-Step Lookahead Value)</span></p>

Suppose we start in state $s$, take an action $a$ *once*, and then follow $\pi$ thereafter. The expected return of this one-step deviation is

$$q_\pi(s, a) \;=\; \sum_{s', r} p(s', r \mid s, a)\bigl[\, r + \gamma\, v_\pi(s')\,\bigr].$$

$q_\pi(s, a)$ measures how good it is to take $a$ now and behave according to $\pi$ afterwards. Compared with following $\pi$ from the start, the only difference is the first action.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Comparing actions — the local-improvement test)</span></p>

We already know the value of following $\pi$ from state $s$: it is $v_\pi(s)$. We can compare it against the value of *deviating once* and then returning to $\pi$:

* if $q_\pi(s, a) > v_\pi(s)$, taking $a$ in state $s$ is a strict one-step improvement,
* if $q_\pi(s, a) = v_\pi(s)$ for every $a$ used by $\pi$, no local improvement exists in state $s$.

The seductive question is: if a *one-step* deviation helps, does **permanently** switching to that action also help? The policy improvement theorem says yes — at every state, simultaneously.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Policy Improvement Theorem)</span></p>

Let $\pi$ and $\pi'$ be any pair of deterministic policies. If for **every** state $s$,

$$q_\pi(s, \pi'(s)) \;\ge\; v_\pi(s),$$

then $\pi'$ is at least as good as $\pi$ everywhere:

$$v_{\pi'}(s) \;\ge\; v_\pi(s) \qquad \text{for all } s \in \mathcal{S}.$$

If the first inequality is strict in some state, $\pi'$ is strictly better there.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof I</summary>

Assume that for every state $s\in S$ set we have $q_\pi(s,\pi'(s))\ge v_\pi(s)$. Then

$$v_\pi(s) \le \mathbb E\bigl[R_{t+1}+\gamma v_\pi(S_{t+1}) \mid S_t=s,A_t=\pi'(s)\bigr].$$

$$v_\pi(S_{t+1}) \le \mathbb E\bigl[R_{t+2}+\gamma v_\pi(S_{t+2}) \mid S_{t+1},A_{t+1}=\pi'(S_{t+1})\bigr].$$

Substituting this inequality into the previous expectation gives

$$v_\pi(s) \le \mathbb E_{\pi'}\bigl[R_{t+1}+\gamma R_{t+2}+\gamma^2v_\pi(S_{t+2})\mid S_t=s\bigr].$$

Repeating the same argument $n$ times gives

$$v_\pi(s) \le \mathbb E_{\pi'}\left[ \sum_{i=0}^{n-1}\gamma^i R_{t+i+1} + \gamma^n v_\pi(S_{t+n}) \mid S_t=s \right].$$

Because the MDP is finite and $\gamma<1$, the value function $v_\pi$ is bounded, we have $\gamma^n v_\pi(S_{t+n})\to 0$ with $n$ goes to infinity. Taking the limit gives

$$v_\pi(s) \le \mathbb E_{\pi'}\left[ \sum_{i=0}^{\infty}\gamma^i R_{t+i+1} \mid S_t=s \right] = v_{\pi'}(s).$$

</details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof II</summary>

$$q_\pi(s,\pi'(s))=(T_{\pi'}v_\pi)(s).$$

Hence the assumption gives

$$v_\pi\le T_{\pi'}v_\pi.$$

Since $T_{\pi'}$ is monotone,

$$v_\pi\le T_{\pi'}v_\pi\le T_{\pi'}^2v_\pi\le\cdots.$$

But $T_{\pi'}^n v_\pi\to v_{\pi'}$, so

$$\boxed{v_\pi\le v_{\pi'}}.$$

If the first inequality is strict at $s$, then

$$v_\pi(s)<T_{\pi'}v_\pi(s)\le v_{\pi'}(s).$$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Derivation</span><span class="math-callout__name">(Why local one-step improvement implies global improvement)</span></p>

Unroll the assumption 

$$q_\pi(s, \pi'(s)) \ge v_\pi(s)$$

by repeated substitution. Starting from $S_t = s$ and using $\pi'$ to choose the *first* action, then $\pi$ thereafter, then again $\pi'$ for the second action, and so on:

$$\pi, \pi, \pi, \dots \;\xrightarrow{\text{1 swap}}\; \pi', \pi, \pi, \dots \;\xrightarrow{\text{2 swaps}}\; \pi', \pi', \pi, \dots \;\xrightarrow{\text{}\cdots\text{}}\; \pi', \pi', \pi', \dots$$

Each swap can only *increase* the value (by assumption). In the limit of swapping every step, the trajectory follows $\pi'$ throughout, and the limit value is $v_{\pi'}(s)$. So 

$$v_{\pi'}(s) \ge v_\pi(s).$$

The "for all $s$" assumption is essential: after the first action, the agent may land in *any* successor state, and we must be sure that using $\pi'$ from there is still safe.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Greedy Policy Improvement)</span></p>

Given the value function $v_\pi$, define the **greedy policy with respect to $v_\pi$** by

$$\pi'(s) \;\in\; \arg\max_{a}\, q_\pi(s, a) \;=\; \arg\max_{a} \sum_{s', r} p(s', r \mid s, a)\bigl[\, r + \gamma\, v_\pi(s')\,\bigr].$$

By construction 

$$q_\pi(s, \pi'(s)) = \max_a q_\pi(s, a) \ge q_\pi(s, \pi(s)) = v_\pi(s)$$

at every state, so the improvement theorem applies: $\pi'$ is at least as good as $\pi$ everywhere, and strictly better whenever $\pi$ was not already greedy.

</div>

<figure>
  <img src="{{ '/assets/images/notes/rl_hd/greedy_improvement.png' | relative_url }}" alt="Bar chart of q-values for four actions in a state with the maximum-q action highlighted" loading="lazy">
  <figcaption>Greedy improvement at a single state $s$. The bars are the action values $q_\pi(s, a)$ under the current policy $\pi$; the dashed line is $v_\pi(s) = \sum_a \pi(a \mid s)\, q_\pi(s, a)$ — the value of <em>following</em> $\pi$ from $s$. Switching $\pi$'s action at $s$ to the argmax (right) is a strict one-step improvement: $q_\pi(s, \pi'(s)) > v_\pi(s)$. The policy improvement theorem promises this local gain extends to a global one.</figcaption>
</figure>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Evaluation vs. improvement — never confuse the two arrows)</span></p>

The two DP primitives operate in opposite directions:

* **Policy evaluation:** $\pi \longrightarrow v_\pi$ — *change values*, keep the policy fixed. Answers "how good is $\pi$?".
* **Policy improvement:** $v_\pi \longrightarrow \pi'$ — *change the policy*, treat values as a fixed input. Answers "can we act better?".

A short memory aid: **evaluation changes values; improvement changes the policy.** Every DP algorithm (and every later RL algorithm) is some interleaving of these two arrows.

</div>

### Policy Iteration

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Policy Iteration)</span></p>

1. Initialise a policy $\pi$ arbitrarily.
2. **Policy evaluation.** Compute $v_\pi$ (e.g. by iterative policy evaluation to tolerance $\theta$).
3. **Policy improvement.** For every state, set

   $$\pi(s) \;\leftarrow\; \arg\max_{a} \sum_{s', r} p(s', r \mid s, a)\bigl[\, r + \gamma\, v_\pi(s')\,\bigr].$$

4. If the policy has not changed in any state, stop. Otherwise return to step 2.

$$
\pi_0 \;\xrightarrow{E}\; v_{\pi_0} \;\xrightarrow{I}\; \pi_1 \;\xrightarrow{E}\; v_{\pi_1} \;\xrightarrow{I}\; \pi_2 \;\xrightarrow{E}\; \cdots \;\xrightarrow{I}\; \pi_\ast.
$$

</div>

<figure>
  <img src="{{ '/assets/images/notes/rl_hd/policy_iteration_cycle.png' | relative_url }}" alt="Zig-zag of policies and value functions alternating evaluation and improvement steps, ending at the optimal policy" loading="lazy">
  <figcaption>Policy iteration alternates two arrows: <strong>E</strong> (evaluation, blue) maps a policy to its value function; <strong>I</strong> (improvement, green) maps a value function to its greedy policy. Each <strong>I</strong>-step is monotone — $v_{\pi_{k+1}} \ge v_{\pi_k}$ pointwise — and the number of deterministic policies is finite, so the chain must terminate at $\pi_\ast$.</figcaption>
</figure>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(A Stable Greedy Policy Is Already Optimal)</span></p>

Suppose policy improvement does not change the policy; that is, the greedy policy $\pi'$ obtained from $v_\pi$ satisfies $\pi'=\pi$. Show that 

$$v_\pi(s) = \max_a \sum_{s',r} p(s',r\mid s,a) \left( r+\gamma v_\pi(s') \right)$$

for all $s\in\mathcal S$, and conclude that $v_\pi=v^\ast$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

Suppose the greedy policy $\pi'$ obtained from $v_\pi$ satisfies $\pi'=\pi$:

$$\pi(s) \in \arg\max_a \sum_{s',r}p(s',r\mid s,a)\bigl[r+\gamma v_\pi(s')\bigr].$$

$$
\begin{aligned}
v_\pi(s) = q_\pi(s,\pi(s)) 
&= \sum_{s',r}p(s',r\mid s,\pi(s))\bigl[r+\gamma v_\pi(s')\bigr] \\
&= \max_a \sum_{s',r}p(s',r\mid s,a)\bigl[r+\gamma v_\pi(s')\bigr].
\end{aligned}
$$

This is the Bellman optimality equation. In a finite discounted MDP, the Bellman optimality equation has a unique solution, namely the optimal value function $v^\ast$.

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Why policy iteration converges — and in *finitely* many steps)</span></p>

For a finite discounted MDP:

* Each improvement step yields a policy that is at least as good as the previous one (policy improvement theorem).
* If improvement makes any change, the new policy is **strictly** better in at least one state — so the same policy is never revisited.
* The number of deterministic policies is finite ($\lvert \mathcal{A} \rvert^{\lvert \mathcal{S} \rvert}$ at most).

Therefore the algorithm cannot keep improving forever:
* it must terminate in finitely many iterations with a policy $\pi$ satisfying 

  $$\pi(s) \in \arg\max_a q_\pi(s, a)$$

  for every $s$. 
* That fixed-point condition is precisely the Bellman optimality equation
  * so the limit policy is **optimal**.

In practice, policy iteration tends to need surprisingly *few* outer iterations (often single digits even on substantial problems), because each greedy step makes large jumps in policy space.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Finite Policy Spaces Force Termination)</span></p>

Argue that policy iteration must converge to an optimal policy in finitely many steps. Your argument should use: 
1. the Policy Improvement Theorem from Task 1; 
2. the stopping criterion from Task 2; 
3. the finiteness of the number of deterministic policies.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

* At each step of policy iteration, we perform policy evaluation plus policy improvement.  
* Let $\pi_k$ be the current policy and let $\pi_{k+1}$ be the greedy policy with respect to $v_{\pi_k}$. 
* By the Policy Improvement Theorem from Task 1, we have $v_{\pi_{k+1}}(s) \ge v_{\pi_k}(s)$ for all states $s$.
* Thus the sequence of value functions is monotonically non-decreasing. 
  * If the improvement step does not change the policy
    * then by Task 2 the current policy satisfies the Bellman optimality equation, 
    * hence $v_{\pi_k}=v^\ast$.
    * therefore $\pi_k$ is optimal and policy iteration terminates. 
  * If the policy changes,
    * then under a fixed tie-breaking rule policy iteration moves to a new deterministic policy. 
* Since the state and action spaces are finite, the number of deterministic policies is finite. 
* Therefore policy iteration cannot generate infinitely many distinct improved deterministic policies. 
* Hence after finitely many steps it must reach a policy that is unchanged by the improvement step and by Task 2, this policy is optimal.

</details>
</div>

#### Jack's Car Rental: Policy Iteration in Action

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Jack's Car Rental — a textbook continuing MDP)</span></p>

Jack manages two rental locations:

* Each day customers request cars (Poisson-distributed) and return cars (Poisson-distributed).
* Each completed rental produces \$10 revenue.
* Cars can be moved overnight between the two locations; moving a car costs \$2.
* Capacity: at most 20 cars per location.
* At most 5 cars moved overnight in either direction.

**State.** $s = (n_1, n_2)$, the number of cars at each location at the end of the day. So $\mathcal{S} = \lbrace 0, \dots, 20\rbrace^2$, with $\lvert \mathcal{S} \rvert = 441$.

**Action.** $a \in \lbrace -5, -4, \dots, 5\rbrace$: positive $a$ moves $\lvert a \rvert$ cars from location 1 to location 2; negative $a$ moves them the other way.

**Why this is a good DP illustration.** The dynamics are *known* (Poissons are tabulated and the action is deterministic in its effect on the morning count), so $p(s' \mid s, a)$ can be computed in closed form. The state space is finite but non-trivial (441 states $\times$ 11 actions $\times$ ${\sim}20{,}000$ successor terms per backup), and the problem is rich enough to have a *non-obvious* optimal policy that does not coincide with any human heuristic.

Running policy iteration produces a sequence $\pi_0, \pi_1, \pi_2, \dots$ where each map shows the optimal overnight transfer as a function of $(n_1, n_2)$. The algorithm typically converges in a handful of outer iterations, and the final policy reveals a sharp staircase-shaped frontier in $(n_1, n_2)$ space dictating when to ferry cars between locations.

</div>

<figure>
  <img src="{{ '/assets/images/notes/rl_hd/jacks_car_rental.png' | relative_url }}" alt="Five policy maps over the (n1, n2) state space showing successive improvement steps, plus the converged value surface" loading="lazy">
  <figcaption>Policy iteration on Jack's Car Rental. The five small panels show the policy $\pi_k(n_1, n_2)$ after each improvement step — red cells move cars $1 \to 2$, blue cells move them $2 \to 1$, white means "do nothing". The 3D panel on the right is the converged value surface $v_{\pi_\ast}$: highest in balanced, moderately-stocked configurations where future rentals are most likely to materialise without overflow penalties. The decision boundary in $\pi_\ast$ has a staircase shape that no simple human heuristic would discover.</figcaption>
</figure>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The DP method map so far)</span></p>

| Method | Main question | Bellman backup |
| :----- | :------------ | :------------- |
| Policy evaluation | How good is $\pi$? | expectation over actions under $\pi$ |
| Policy improvement | Can $\pi$ be improved? | greedy one-step lookahead |
| Policy iteration | How do we alternate both? | evaluate, then improve |
| Value iteration | Can we combine both? | max backup |
| Asynchronous DP | Can we avoid full sweeps? | update selected states |

The next algorithm — value iteration — fills in the "combine both" row by noticing that we do not actually need to *fully* evaluate a policy before improving it.

</div>

### Value Iteration

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Idea</span><span class="math-callout__name">(Why fully evaluating each policy is wasteful)</span></p>

Policy iteration alternates evaluation and improvement, but evaluation itself can require many sweeps (it is an *inner* iterative procedure). And after a few sweeps the greedy step typically already produces the same policy as it would at full convergence — so most of the inner work is wasted. The key idea of **value iteration** is to interleave the two arrows at the finest possible granularity: do *one* sweep of evaluation, then immediately improve. Better still, *merge* the two steps into a single update that takes a max over actions instead of an average.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(From Bellman Expectation to Bellman Optimality Backup)</span></p>

Compare the two backups on the same value function:

* **Policy evaluation backup (expectation over actions):**

  $$v_{k+1}(s) \;=\; \sum_{a} \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a)\bigl[\, r + \gamma\, v_k(s')\,\bigr].$$

* **Value iteration backup (max over actions):**

  $$v_{k+1}(s) \;=\; \max_{a} \sum_{s', r} p(s', r \mid s, a)\bigl[\, r + \gamma\, v_k(s')\,\bigr].$$

The two differ in **exactly one place**: averaging over $a$ by $\pi$ is replaced by taking the max over $a$. The first iterates toward $v_\pi$; the second iterates toward $v_\ast$ directly — it uses the current estimate $v_k$ as a *guess for the optimal* future value and applies a greedy one-step lookahead.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Value Iteration)</span></p>

Initialise $V(s)$ arbitrarily for all $s \in \mathcal{S}$, with $V(\text{terminal}) = 0$.

Repeat:

1. $\Delta \leftarrow 0$.
2. For each state $s \in \mathcal{S}$:
   * $v \leftarrow V(s)$,
   * $V(s) \leftarrow \displaystyle \max_{a} \sum_{s', r} p(s', r \mid s, a)\bigl[\, r + \gamma\, V(s')\,\bigr]$,
   * $\Delta \leftarrow \max\bigl(\Delta,\, \lvert v - V(s) \rvert\bigr)$.

until $\Delta < \theta$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Optimal Greedy Policy from Value Iteration)</span></p>

After convergence, $V \approx v_\ast$, and the **optimal greedy policy** is extracted in one final sweep:

$$\pi^\ast(s) \;\in\; \arg\max_{a} \sum_{s', r} p(s', r \mid s, a)\bigl[\, r + \gamma\, V(s')\,\bigr].$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Bellman Operators Contract Value Errors: Discounting Makes Bellman Updates Contractive)</span></p>

Let $\pi$ be a deterministic policy. Define

* **Bellman expectation operator:** $T_\pi:\mathbb R^{\lvert\mathcal S\rvert}\to\mathbb R^{\lvert\mathcal S\rvert}$
* **Bellman optimality operator:** $T:\mathbb R^{\lvert\mathcal S\rvert}\to\mathbb R^{\lvert\mathcal S\rvert}$

$$(T_\pi v)(s) = \sum_{s',r} p(s',r\mid s,\pi(s)) \left( r+\gamma v(s') \right),$$ 

$$(Tv)(s) = \max_a \sum_{s',r} p(s',r\mid s,a) \left( r+\gamma v(s') \right)$$

for $s\in\mathcal S$. 

Let $\lVert v\rVert_\infty = \max_{s\in\mathcal S}\lvert v(s)\rvert$. Denote the supremum norm on $\mathbb R^{\lvert\mathcal S\rvert}$. Let $\gamma\in[0,1)$, and assume that rewards are bounded $\lvert r\rvert\le R_{\max}$ for all $(s,a,s',r)$. 

---

Prove that both $T_\pi$ and $T$ are $\gamma$-contractions in the supremum norm; that is, for any two value functions $u,v\in\mathbb R^{\lvert\mathcal S\rvert}$, show that 

$$\lVert T_\pi u-T_\pi v\rVert_\infty \le \gamma\lVert u-v\rVert_\infty$$

$$\lVert Tu-Tv\rVert_\infty \le \gamma\lVert u-v\rVert_\infty.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

For $T_\pi$ we have already proven.

For the Bellman optimality operator.

$$Q_u(s,a) := \sum_{s',r}p(s',r\mid s,a)\bigl[r+\gamma u(s')\bigr].$$

$$
\begin{aligned}
|T u(s)-T v(s)|
&= \left|\max_a Q_u(s,a)-\max_a Q_v(s,a)\right| \\
&\le \max_a |Q_u(s,a)-Q_v(s,a)| \\
&= \max_a \left| \gamma\sum_{s',r}p(s',r\mid s,a)\bigl(u(s')-v(s')\bigr) \right| \\
&\le \gamma\lVert u-v\rVert_\infty.
\end{aligned}
$$

the maximum over $s$ gives $\lVert \T u-\T v\rVert\_\infty \le \gamma\lVert u-v\rVert\_\infty$.

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Repeated Bellman Updates Erase Initial Error: Geometric Convergence to the Optimal Fixed Point)</span></p>

Use the contraction property to show that value iteration 

$$v_{k+1}=Tv_k,$$

starting from any $v_0\in\mathbb R^{\lvert\mathcal S\rvert}$, satisfies 

$$\lVert v_k-v^*\rVert_\infty \le \gamma^k\lVert v_0-v^*\rVert_\infty$$

for all $k\in\mathbb N_0$, and conclude that 

$$v_k\to v^*$$ 

geometrically in $k$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

$$\lVert v_{k+1}-v^*\rVert_\infty =\lVert T v_k-T v^*\rVert_\infty \le \gamma\lVert v_k-v^*\rVert_\infty.$$

Repeating this gives $\lVert v_k-v^\ast\rVert\_\infty \le \gamma^k\lVert v_0-v^\ast\rVert\_\infty$. Since $0\le\gamma<1$, then $\gamma^k\to0$ and $v_k\to v^\ast$, concluding the proof.

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(The Cost of Discount Factors Near One)</span></p>

Suppose that $\gamma=0.9$, $\varepsilon=0.01$ and $\lVert v_0-v^\ast\rVert_\infty\le10$.

Find the minimum number of iterations $k^\ast$ such that $\lVert v_{k^\ast}-v^\ast\rVert_\infty<\varepsilon$.

Give an exact expression and a numerical value. How does $k^\ast$ scale as $\gamma\to1$?

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

Assuming $\gamma=0.9$, $\varepsilon=0.01$ and $\lVert v_0-v^*\rVert\_\infty\le 10$, 

$$\lVert v_k-v^*\rVert_\infty \le 0.9^k\cdot 10$$

$$10\cdot 0.9^k<0.01.$$

$$k> \frac{\log(10^{-3})}{\log(0.9)}.$$

$$k^* = \left\lfloor \frac{\log(10^{-3})}{\log(0.9)} \right\rfloor+1 = 66$$

As $\gamma\to1$, then $-\log\gamma\sim 1-\gamma$ implying that value iteration becomes slow when $\gamma$ is close to $1$.

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Value iteration as "truncated" policy iteration)</span></p>

A clean way to think about value iteration is as **policy iteration with the evaluation step truncated to a single sweep**:

* one Bellman expectation sweep $\to$ greedy improvement collapses into a single Bellman *optimality* sweep,
* the explicit policy never has to be maintained between sweeps; it is implicit in the max.

Conversely, policy iteration is value iteration with the *evaluation* step run to convergence between maxes. They are two ends of a spectrum: how much evaluation do you do per improvement? **All of it** (policy iteration), **none of it past one sweep** (value iteration), or **somewhere in between** (modified / generalised policy iteration).

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(Policy iteration vs. value iteration — side by side)</span></p>

|                  | Policy iteration                       | Value iteration                       |
| :--------------- | :------------------------------------- | :------------------------------------ |
| Main object      | policy $+$ value function              | value function                        |
| Evaluation       | usually many sweeps                    | one sweep at a time                   |
| Improvement      | explicit greedy step                   | built into the max update             |
| Update rule      | evaluate, then improve                 | improve *while* evaluating            |
| Typical view     | cleaner conceptually                   | often more efficient                  |

**Memory aid.** Policy iteration says **evaluate, then improve**. Value iteration says **improve during every backup.**

</div>

<figure>
  <img src="{{ '/assets/images/notes/rl_hd/pi_vs_vi_flow.png' | relative_url }}" alt="Two vertical chains: left side alternates explicit policies and value functions; right side updates only value functions until convergence" loading="lazy">
  <figcaption>Two ways of arranging the same Bellman backups. <strong>Left (policy iteration):</strong> evaluation produces true value functions $v_{\pi_k}$ of explicit policies; improvement re-greedifies. <strong>Right (value iteration):</strong> no explicit policy is maintained — each sweep applies a max backup directly to the value vector, driving $v_k$ toward $v_\ast$. The optimal policy is extracted only at the very end by a single greedy argmax.</figcaption>
</figure>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Common confusion — what is being computed at each level)</span></p>

The biggest source of confusion is that *both* algorithms produce a sequence of value functions and a final policy, but:

* In **policy iteration** the *intermediate* value functions $v_{\pi_0}, v_{\pi_1}, \dots$ are *true* values of *real* (deterministic) policies. The trace is 
  
  $$\pi_0 \to v_{\pi_0} \to \pi_1 \to v_{\pi_1} \to \cdots.$$

* In **value iteration** the intermediate value functions $v_0, v_1, v_2, \dots$ are *not* the values of any particular policy. They are estimates of $v_\ast$ that happen to be related to the value of the greedy policy at sweep $k$, but no explicit policy is materialised until the very end.

This distinction matters for analysis (error bounds, monotonicity) and for any later algorithm that wants to use intermediate policies (e.g. on-policy methods later in the course).

</div>

#### Gambler's Problem: Value Iteration on a Tiny MDP

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Gambler's Problem)</span></p>

A gambler repeatedly bets on coin flips.

* **State:** current capital $s \in \lbrace 1, 2, \dots, 99 \rbrace$ (with $0$ and $100$ as terminals).
* **Actions:** stake amount $a \in \lbrace 0, 1, \dots, \min(s,\, 100 - s) \rbrace$ — you cannot stake more than you have, and you never need to stake more than is required to reach $100$.
* **Dynamics:** heads (probability $p_h$) increases capital by $a$; tails (probability $1 - p_h$) decreases it by $a$.
* **Reward:** $0$ on every non-terminal transition, $+1$ if capital reaches $100$, $0$ if capital reaches $0$.
* **Goal:** maximise the probability of reaching $100$ before reaching $0$.

Because rewards are $0$ everywhere except at the winning terminal, the value function has a beautiful interpretation:

$$
v(s) \;=\; \text{probability of eventually winning from capital } s.
$$

**The MDP is exactly suited to value iteration:** the state space is one-dimensional, the dynamics are explicit, and there is no need to maintain an explicit policy during sweeps.

</div>

<figure>
  <img src="{{ '/assets/images/notes/rl_hd/gamblers_problem.png' | relative_url }}" alt="Optimal value function curve (smooth, monotonically increasing) and optimal-stake stem plot (irregular spikes at 25, 50, 75)" loading="lazy">
  <figcaption>Value iteration on the Gambler's problem with $p_h = 0.4$. <strong>Top:</strong> the optimal value function $v_\ast(s)$, equal to the probability of eventually reaching capital $100$ starting from $s$. It is monotonically increasing and visually smooth. <strong>Bottom:</strong> the corresponding optimal policy $\pi_\ast(s)$ — wildly irregular, with characteristic spikes at $s = 25, 50, 75$ where a single all-in bet either reaches $50$ exactly or hits $100$ directly. The lesson: <em>optimal policies can be discontinuous even when the optimal value function is smooth</em>.</figcaption>
</figure>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Interpreting the optimal solution)</span></p>

The value function $v_\ast$ produced by value iteration is monotone in capital — more money means a higher chance of winning. The optimal policy, however, is **not** smooth at all: it has dramatic *spikes* at certain capital levels.

* At some capital levels (e.g. $s = 50$ when $p_h < 1/2$), the gambler bets *everything* — a single bet can reach $100$ exactly.
* At other levels the optimal bet is small.

**Why the spikes happen.** At certain capitals, a particular stake lands the agent in a state from which winning is much more likely (often because it can reach $100$ or $50$ exactly). These states create sudden jumps in success probability, and the greedy argmax inherits those jumps — so the policy looks irregular even though the underlying value function is smooth.

This is a good lesson: **the optimal *policy* can be much less regular than the optimal *value function***. Any later method that approximates the policy directly (policy gradient, actor-critic) has to be careful about parametric families flexible enough to represent such discontinuities.

</div>

### Generalised Picture and Section Summary

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(Generalised Policy Iteration — the unifying frame)</span></p>

Policy iteration and value iteration are two extreme points of the same idea:

* **Generalised policy iteration (GPI)** is the umbrella term for any algorithm that maintains *some* value function $V$ and *some* policy $\pi$ and repeatedly improves them against each other — without insisting that either be exact at any intermediate step.
* Policy iteration: bring $V$ all the way to $v_\pi$ before re-greedifying.
* Value iteration: do exactly one Bellman optimality sweep before re-greedifying.
* Asynchronous DP: update *some* states at *some* sweeps, in any order — as long as every state is visited infinitely often.

Every algorithm in the rest of the course (Monte Carlo control, SARSA, Q-learning, actor-critic, $\dots$) is an instance of GPI; the differences are *how* the two arrows are implemented (with samples? bootstrapping? function approximation?) but the skeleton is the same.

</div>

<figure>
  <img src="{{ '/assets/images/notes/rl_hd/gpi_diagram.png' | relative_url }}" alt="Two manifolds (v = v_pi and pi = greedy(v)) intersecting at the fixed point, with a zig-zag trajectory of evaluation/improvement steps approaching that intersection" loading="lazy">
  <figcaption>The classical GPI picture: the two arrows of evaluation (blue, change $v$) and improvement (green, change $\pi$) each push the current $(v, \pi)$ pair onto a respective manifold — $v = v_\pi$ and $\pi = \mathrm{greedy}(v)$. Alternating partial moves toward each manifold produces a zig-zag that converges to their unique intersection $(\pi_\ast, v_\ast)$. Policy iteration takes <em>full</em> projections onto the evaluation line; value iteration takes <em>tiny</em> ones (one sweep); model-free RL replaces the projections by sample-based estimates. The unifying skeleton is the same.</figcaption>
</figure>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(Core Takeaways — Dynamic Programming)</span></p>

* **DP = planning with a known model.** With $p(s', r \mid s, a)$ in hand, every Bellman backup is a *finite, exact* expectation.
* **Policy evaluation** = repeated Bellman *expectation* backups, converging geometrically (rate $\gamma$) to $v_\pi$. Equivalent to solving the linear system $(I - \gamma P_\pi)\, v_\pi = r_\pi$.
* **Policy improvement** = acting greedily with respect to $v_\pi$. The improvement theorem guarantees that a *local one-step* improvement is automatically a *global* improvement.
* **Policy iteration** alternates the two until the policy stops changing; it converges in *finitely many* outer iterations on finite MDPs.
* **Value iteration** is policy iteration with evaluation truncated to a single max-backup sweep. It tracks $v_\ast$ directly and extracts $\pi^\ast$ at the end by a one-step argmax.
* The unifying view is **generalised policy iteration**: every later RL algorithm replaces one of the two arrows (evaluation or improvement) with a sample-based, model-free surrogate, but the alternation structure is preserved.

**Final message.** DP is the *exact* version of the picture: an oracle environment, full sweeps, expected backups. The rest of the course is a long answer to a single question — **how do we approximate the DP arrows when the model is unknown and the state space is too large to sweep?**

</div>

## Lecture 5: Monte Carlo Methods

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Setup</span><span class="math-callout__name">(Model-free prediction and control from sampled episodes)</span></p>

The previous lecture closed with the question "how do we approximate the DP arrows when the model is unknown?". **Monte Carlo (MC) methods are the first honest answer.** They drop the strongest assumption of DP — that we have $p(s', r \mid s, a)$ in closed form — and replace expected backups with *sample averages of complete returns*. The agent now interacts with the environment (or a simulator) and learns from what actually happens, not from what the dynamics say should happen on average.

The methods we develop in this lecture share four defining features:

1. **Model-free.** No transition kernel is ever consulted; estimates come from sampled experience only.
2. **Episodic.** Updates are made *after the episode ends*, using the full sampled return $G_t$ from a state onward.
3. **No bootstrapping.** Unlike DP (and unlike TD in the next lecture), an MC update does **not** use the value estimate of another state — it uses the realised return.
4. **GPI-shaped.** The skeleton is still policy evaluation $\to$ policy improvement, only with samples replacing expectations.

The guiding question of this lecture is:

> *How do we evaluate and improve a policy when we have **no model** of the environment, only sampled trajectories?*

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(Where MC sits in the RL taxonomy)</span></p>

It helps to keep the broader map in view. Methods in this course split along two axes:

* **Model-based vs. model-free.** DP and optimal control assume a known $p(s', r \mid s, a)$. MC, TD, and everything after them do not.
* **Bootstrapping vs. not.** DP and TD update one value estimate using another; MC does not — it waits for the actual return.

MC therefore occupies a *unique* corner of the taxonomy: **model-free but non-bootstrapping**. TD is the other model-free family; it adds bootstrapping back in to enable online updates. The pedagogical reason to develop MC first is that, by ruling out bootstrapping, every estimate is an honest sample mean of a well-defined target — convergence proofs reduce to the strong law of large numbers, with no fixed-point machinery needed.

</div>

<figure class="rl-diagram">
  <svg viewBox="0 0 760 430" role="img" aria-label="Taxonomy of reinforcement learning methods by model use and bootstrapping">
    <defs>
      <marker id="mc-arrow-taxonomy" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">
        <path d="M0,0 L10,4 L0,8 Z" fill="#64748b"></path>
      </marker>
    </defs>
    <line x1="140" y1="355" x2="685" y2="355" class="line" marker-end="url(#mc-arrow-taxonomy)"></line>
    <line x1="140" y1="355" x2="140" y2="70" class="line" marker-end="url(#mc-arrow-taxonomy)"></line>
    <text x="672" y="361" text-anchor="end" font-size="15" class="muted">uses model</text>
    <text x="45" y="68" font-size="15" class="muted">bootstraps</text>
    <text x="145" y="385" font-size="14" class="muted">model-free</text>
    <text x="575" y="385" font-size="14" class="muted">model-based</text>
    <text x="54" y="350" font-size="14" class="muted">no</text>
    <text x="58" y="115" font-size="14" class="muted">yes</text>

    <rect x="180" y="95" width="200" height="105" rx="8" class="box"></rect>
    <text x="280" y="135" text-anchor="middle" font-size="22" font-weight="700">TD</text>
    <text x="280" y="164" text-anchor="middle" font-size="14" class="muted">model-free</text>
    <text x="280" y="184" text-anchor="middle" font-size="14" class="muted">bootstrapping</text>

    <rect x="455" y="95" width="200" height="105" rx="8" class="box"></rect>
    <text x="555" y="135" text-anchor="middle" font-size="22" font-weight="700">DP</text>
    <text x="555" y="164" text-anchor="middle" font-size="14" class="muted">model-based</text>
    <text x="555" y="184" text-anchor="middle" font-size="14" class="muted">bootstrapping</text>

    <rect x="180" y="235" width="200" height="105" rx="8" class="accent"></rect>
    <text x="280" y="275" text-anchor="middle" font-size="22" font-weight="700">MC</text>
    <text x="280" y="304" text-anchor="middle" font-size="14" class="muted">model-free</text>
    <text x="280" y="324" text-anchor="middle" font-size="14" class="muted">complete returns</text>

    <rect x="455" y="235" width="200" height="105" rx="8" class="box"></rect>
    <text x="555" y="275" text-anchor="middle" font-size="20" font-weight="700">Rollouts</text>
    <text x="555" y="304" text-anchor="middle" font-size="14" class="muted">model-based</text>
    <text x="555" y="324" text-anchor="middle" font-size="14" class="muted">sampled trajectories</text>
  </svg>
  <figcaption>MC is the model-free, non-bootstrapping corner: it does not know the transition model and it does not update from another value estimate. It waits for full sampled returns.</figcaption>
</figure>

### Monte Carlo: Big Picture

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Idea</span><span class="math-callout__name">(From expectation backups to sample averages)</span></p>

The Bellman expectation equation says

$$
v_\pi(s) \;=\; \mathbb{E}_\pi[\, G_t \mid S_t = s \,] \;=\; \sum_{a, s', r} \pi(a \mid s)\, p(s', r \mid s, a)\bigl[\, r + \gamma\, v_\pi(s')\,\bigr].
$$

DP evaluates the right-hand sum *exactly* because $p$ is known. **MC instead estimates the left-hand expectation directly by averaging samples:**

$$
v_\pi(s) \;\approx\; \frac{1}{n}\sum_{i=1}^{n} G^{(i)},
$$

where each $G^{(i)}$ is the return of an episode that passed through $s$ under $\pi$. The model never appears — and that is exactly the point.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Core Ingredients of Monte Carlo)</span></p>

MC methods are built from three primitives.

* **Sampled experience.** A finite trajectory generated by interacting with the environment under some policy:

  $$(S_0,\, A_0,\, R_1,\, S_1,\, A_1,\, R_2,\, \dots,\, S_T).$$

* **Episodic structure.** Every episode terminates in finite time with probability $1$ (so $T < \infty$ a.s.). This is what makes the return well-defined as a *realised number*, not just an expectation.

* **Return from time $t$:**

  $$G_t \;\doteq\; \sum_{k=0}^{T-t-1} \gamma^{k}\, R_{t+k+1}.$$

The MC estimate of $v_\pi(s)$ is then literally a sample average of returns observed from visits to $s$:

$$v_\pi(s) \;=\; \mathbb{E}_\pi[\, G_t \mid S_t = s \,] \;\approx\; \frac{1}{n}\sum_{i=1}^{n} G_i(s).$$

</div>

<figure class="rl-diagram">
  <svg viewBox="0 0 860 300" role="img" aria-label="Monte Carlo return as a discounted sum along one sampled episode">
    <defs>
      <marker id="mc-arrow-episode" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">
        <path d="M0,0 L10,4 L0,8 Z" fill="#2c3e94"></path>
      </marker>
    </defs>
    <text x="32" y="42" font-size="18" font-weight="700">One sampled episode</text>
    <text x="32" y="68" font-size="14" class="muted">The update target for a visit is the realised future from that point onward.</text>

    <g transform="translate(54 122)">
      <circle cx="0" cy="0" r="24" class="box"></circle>
      <text x="0" y="5" text-anchor="middle" font-size="16">S0</text>
      <line x1="24" y1="0" x2="112" y2="0" class="strong-line" marker-end="url(#mc-arrow-episode)"></line>
      <text x="68" y="-18" text-anchor="middle" font-size="14" class="muted">R1</text>

      <circle cx="140" cy="0" r="24" class="accent"></circle>
      <text x="140" y="5" text-anchor="middle" font-size="16">St</text>
      <line x1="164" y1="0" x2="252" y2="0" class="strong-line" marker-end="url(#mc-arrow-episode)"></line>
      <text x="208" y="-18" text-anchor="middle" font-size="14" font-weight="700">R<tspan baseline-shift="sub" font-size="10">t+1</tspan></text>

      <circle cx="280" cy="0" r="24" class="box"></circle>
      <text x="280" y="5" text-anchor="middle" font-size="16">S<tspan baseline-shift="sub" font-size="10">t+1</tspan></text>
      <line x1="304" y1="0" x2="392" y2="0" class="strong-line" marker-end="url(#mc-arrow-episode)"></line>
      <text x="348" y="-18" text-anchor="middle" font-size="14" font-weight="700">R<tspan baseline-shift="sub" font-size="10">t+2</tspan></text>

      <circle cx="420" cy="0" r="24" class="box"></circle>
      <text x="420" y="5" text-anchor="middle" font-size="16">S<tspan baseline-shift="sub" font-size="10">t+2</tspan></text>
      <line x1="444" y1="0" x2="532" y2="0" class="strong-line" marker-end="url(#mc-arrow-episode)"></line>
      <text x="488" y="-18" text-anchor="middle" font-size="14" class="muted">...</text>

      <circle cx="560" cy="0" r="24" class="box"></circle>
      <text x="560" y="5" text-anchor="middle" font-size="16">S<tspan baseline-shift="sub" font-size="10">T-1</tspan></text>
      <line x1="584" y1="0" x2="672" y2="0" class="strong-line" marker-end="url(#mc-arrow-episode)"></line>
      <text x="628" y="-18" text-anchor="middle" font-size="14" font-weight="700">R<tspan baseline-shift="sub" font-size="10">T</tspan></text>

      <circle cx="700" cy="0" r="26" class="green"></circle>
      <text x="700" y="5" text-anchor="middle" font-size="16">terminal</text>
    </g>

    <path d="M195 160 C260 230, 555 230, 756 160" class="strong-line"></path>
    <text x="475" y="245" text-anchor="middle" font-size="18" font-weight="700">G<tspan baseline-shift="sub" font-size="12">t</tspan> = R<tspan baseline-shift="sub" font-size="12">t+1</tspan> + gamma R<tspan baseline-shift="sub" font-size="12">t+2</tspan> + gamma<tspan baseline-shift="super" font-size="11">2</tspan> R<tspan baseline-shift="sub" font-size="12">t+3</tspan> + ...</text>
  </svg>
  <figcaption>A Monte Carlo target is not a one-step lookahead. Once a visit to $S_t$ occurs, the episode is followed to termination and the whole realised discounted tail becomes $G_t$.</figcaption>
</figure>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Backup diagrams — one trajectory, not one expectation)</span></p>

The visual difference between DP and MC is striking. A DP backup is a *shallow, wide* diagram: from $s$ it fans out to every successor $(s', r)$, weighted by $p(s', r \mid s, a)$ and $\pi(a \mid s)$, and combines the results in one expectation. **An MC backup is the opposite — narrow and deep:** from $s$ a *single sampled trajectory* runs all the way to the terminal state, and the entire realised return $G_t$ is used as the update target.

Two consequences follow immediately.

* **MC does not propagate value information laterally.** Each state learns *purely from its own returns*; the estimate at $s'$ never enters the update for $s$. This is the precise statement of "no bootstrapping".
* **MC updates are local in the trajectory.** A single episode contributes information to *each* state it visited, but only via the realised future of that episode — nothing else.

</div>

<figure>
  <img src="{{ '/assets/images/notes/rl_hd/mc_backup_diagram.svg' | relative_url }}" alt="Monte Carlo backup diagram: a single trajectory from a state runs all the way to a terminal leaf, contrasted with a shallow expectation backup over all one-step successors" loading="lazy">
  <figcaption>Backup diagrams contrasted. <strong>Left (DP):</strong> shallow and wide — one-step expectation over <em>all</em> successors, weighted by the model. <strong>Right (MC):</strong> narrow and deep — one sampled trajectory followed all the way to termination, the realised $G_t$ used as the update target. The DP diagram averages over what <em>could</em> happen; the MC diagram uses what <em>did</em> happen.</figcaption>
</figure>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(Key property — no bootstrapping)</span></p>

MC methods estimate values **directly from complete returns**. They do *not* use other value estimates in their updates. This single design choice is the source of both their strengths and their weaknesses.

**Pros.**

* **Unbiased.** The update target $G_t$ is an honest sample of the random variable whose expectation is $v_\pi(s)$.
* **Conceptually simple.** A sample mean is all there is to the estimator.
* **No model required.** Only the ability to *generate* episodes is assumed.

**Cons.**

* **Delayed updates.** Nothing can be learned until the episode terminates.
* **High variance.** A single realised return can deviate wildly from its expectation, especially in long episodes.

The contrast with TD methods next lecture is exact: TD trades a little bias (because it bootstraps) for much lower variance and the ability to update online.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(DP vs. MC vs. TD — taxonomy preview)</span></p>

| Method | Model needed? | Bootstraps? | Updates when? |
| :----- | :-----------: | :---------: | :------------ |
| DP     | yes           | yes         | planning sweeps; expected one-step backups |
| MC     | no            | no          | after the episode ends |
| TD     | no            | yes         | after each sampled step |

MC sits in the unique "model-free, non-bootstrapping" corner; TD will occupy the "model-free, bootstrapping" corner; DP is the "model-based, bootstrapping" corner. The fourth corner (model-based, non-bootstrapping) corresponds essentially to plain trajectory-rollout planning.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Running example — Blackjack)</span></p>

Throughout this lecture we use **Blackjack** as a recurring concrete example because it has exactly the structure MC needs.

* **State.** $s = (\text{player sum},\, \text{dealer's showing card},\, \text{usable ace})$.
* **Actions.** *hit* or *stick*.
* **Reward.** $+1$ for a win, $0$ for a draw, $-1$ for a loss.
* **Episodes.** Each game is a single episode that terminates with probability $1$.

The fit with MC is unusually clean. The transition dynamics $p(s', r \mid s, a)$ are *technically* available (one can sum over all card sequences) but combinatorially nasty, while *sampling* games is trivial. And the return from any state is just the final game outcome:

$$
G_0 = \text{game result} \in \lbrace +1,\, 0,\, -1 \rbrace.
$$

So Blackjack lets us see MC prediction and MC control work end-to-end without the model ever being touched.

</div>

### MC Prediction

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Goal</span><span class="math-callout__name">(MC Prediction Problem)</span></p>

Given a **fixed** policy $\pi$, estimate

$$
v_\pi(s) \;=\; \mathbb{E}_\pi[\, G_t \mid S_t = s \,] \qquad \text{for all } s \in \mathcal{S},
$$

using only sampled episodes generated under $\pi$.

**Monte Carlo answer.**

* Generate many episodes following $\pi$.
* For each state $s$, average the returns that followed visits to $s$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Visit, First-visit vs. Every-visit)</span></p>

A **visit** to state $s$ in an episode is an occurrence of $s$ as $S_t$ for some $t$. States can repeat within an episode, so a single episode may provide *several* returns for the same $s$.

For a fixed state $s$ and one sampled episode $S_0, A_0, R_1, S_1, \dots, S_T$, define the set of visit times

$$
\mathcal{T}(s) \;=\; \lbrace\, t \in \lbrace 0, \dots, T-1 \rbrace : S_t = s \,\rbrace \;=\; \lbrace t_1 < t_2 < \cdots < t_m \rbrace.
$$

The return from visit time $t_i$ is

$$
G_{t_i} \;=\; \sum_{k=0}^{T - t_i - 1} \gamma^{k}\, R_{t_i + k + 1}.
$$

Two natural estimators arise:

* **First-visit MC.** If $\mathcal{T}(s) \neq \emptyset$, update $V(s)$ using only $G_{t_1}$.
* **Every-visit MC.** Update $V(s)$ using the average of $G_{t_1}, G_{t_2}, \dots, G_{t_m}$.

</div>

<figure class="rl-diagram">
  <svg viewBox="0 0 860 300" role="img" aria-label="First-visit and every-visit Monte Carlo updates on an episode with repeated states">
    <defs>
      <marker id="mc-arrow-visits" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">
        <path d="M0,0 L10,4 L0,8 Z" fill="#64748b"></path>
      </marker>
    </defs>
    <text x="32" y="42" font-size="18" font-weight="700">Repeated state in one episode</text>
    <text x="32" y="68" font-size="14" class="muted">The same state can occur more than once, giving several possible returns from the same episode.</text>

    <g transform="translate(72 122)">
      <circle cx="0" cy="0" r="25" class="accent"></circle>
      <text x="0" y="5" text-anchor="middle" font-size="16">s</text>
      <line x1="25" y1="0" x2="115" y2="0" class="line" marker-end="url(#mc-arrow-visits)"></line>
      <text x="70" y="-18" text-anchor="middle" font-size="14" class="muted">R1</text>

      <circle cx="145" cy="0" r="25" class="box"></circle>
      <text x="145" y="5" text-anchor="middle" font-size="16">x</text>
      <line x1="170" y1="0" x2="260" y2="0" class="line" marker-end="url(#mc-arrow-visits)"></line>
      <text x="215" y="-18" text-anchor="middle" font-size="14" class="muted">R2</text>

      <circle cx="290" cy="0" r="25" class="accent"></circle>
      <text x="290" y="5" text-anchor="middle" font-size="16">s</text>
      <line x1="315" y1="0" x2="405" y2="0" class="line" marker-end="url(#mc-arrow-visits)"></line>
      <text x="360" y="-18" text-anchor="middle" font-size="14" class="muted">R3</text>

      <circle cx="435" cy="0" r="25" class="box"></circle>
      <text x="435" y="5" text-anchor="middle" font-size="16">y</text>
      <line x1="460" y1="0" x2="550" y2="0" class="line" marker-end="url(#mc-arrow-visits)"></line>
      <text x="505" y="-18" text-anchor="middle" font-size="14" class="muted">R4</text>

      <circle cx="580" cy="0" r="26" class="green"></circle>
      <text x="580" y="5" text-anchor="middle" font-size="15">terminal</text>
    </g>

    <path d="M72 162 C110 220, 435 220, 652 162" class="strong-line"></path>
    <text x="350" y="238" text-anchor="middle" font-size="15" font-weight="700">first-visit uses only this first tail</text>

    <path d="M362 151 C410 185, 530 185, 652 151" stroke="#047857" stroke-width="3" fill="none"></path>
    <text x="520" y="203" text-anchor="middle" font-size="15" font-weight="700">every-visit also uses this tail</text>

    <rect x="680" y="100" width="145" height="90" rx="8" class="box"></rect>
    <text x="752" y="130" text-anchor="middle" font-size="14" font-weight="700">Estimator choice</text>
    <text x="752" y="154" text-anchor="middle" font-size="13" class="muted">first-visit: 1 return</text>
    <text x="752" y="174" text-anchor="middle" font-size="13" class="muted">every-visit: 2 returns</text>
  </svg>
  <figcaption>First-visit and every-visit MC differ only in which realised tails are added to the sample average. First-visit keeps one return per state per episode; every-visit uses all repeated occurrences.</figcaption>
</figure>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why two variants exist)</span></p>

The split looks pedantic but it has real statistical content.

* **First-visit returns are i.i.d. across episodes.** Each episode contributes at most one return for $s$, and different episodes are independent — so we are averaging an i.i.d. sample of the random variable whose mean *is* $v_\pi(s)$. The classical LLN applies directly.
* **Every-visit returns within the same episode are *not* independent.** Two visits to $s$ within one episode share a common future tail, so their returns are correlated. The estimator is still consistent, but the convergence proof is more delicate (it has finite-sample bias that vanishes as $n \to \infty$).

Both estimators converge to $v_\pi(s)$. First-visit is the *theoretically* cleaner one; every-visit is sometimes preferred in practice because it uses more data per episode.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(First-visit vs. Every-visit on a tiny episode)</span></p>

Take $\gamma = 0.5$ and one sampled episode

$$S_0 = s,\quad S_1 = x,\quad S_2 = s,\quad S_3 = \text{terminal}, \qquad R_1 = 2,\; R_2 = 1,\; R_3 = 4.$$

The state $s$ is visited at times $\mathcal{T}(s) = \lbrace 0, 2 \rbrace$.

* **First-visit MC.** Use only the return from the first visit:

  $$G_0 \;=\; R_1 + \gamma R_2 + \gamma^2 R_3 \;=\; 2 + 0.5 \cdot 1 + 0.25 \cdot 4 \;=\; 3.5, \qquad V(s) \leftarrow 3.5.$$

* **Every-visit MC.** Average the returns from both visits:

  $$G_0 = 3.5, \qquad G_2 = R_3 = 4, \qquad V(s) \leftarrow \frac{3.5 + 4}{2} \;=\; 3.75.$$

The numerical disagreement on a single episode is expected; both estimators agree in the limit as the number of episodes goes to infinity.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(First-Visit MC Prediction)</span></p>

Initialise $V(s)$ arbitrarily for all $s \in \mathcal{S}$, and maintain a sample list `Returns(s)` for each state.

Repeat for each episode:

1. Generate an episode following 
   * $\pi$: $S_0, A_0, R_1, \dots, S_T$.
2. For each state $s$ that appears in the episode:
   * Let $G$ be the return after the **first** visit to $s$.
   * Append $G$ to `Returns(s)`.
   * $V(s) \leftarrow \text{average}(\texttt{Returns}(s))$.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(In practice — incremental mean)</span></p>

Storing the full list `Returns(s)` is wasteful. Replacing the explicit average by the **incremental mean update**

$$V(s) \;\leftarrow\; V(s) + \frac{1}{N(s)}\bigl(\, G - V(s) \,\bigr),$$

where $N(s)$ counts how many returns have been seen for $s$, gives an *exactly equivalent* estimator with constant memory. This formula is also the template all later sample-based RL updates inherit — the only thing that changes is *what is plugged in for $G$* (a sampled return for MC, a bootstrapped target for TD, an importance-weighted return for off-policy MC, $\dots$).

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">()</span></p>

Show that the first-visit MC estimator $V_n(s)$ can be updated *incrementally*: if a new episode yields return $G$ at the first visit to $s$, then

$$V_{n+1}(s) = V_n(s) + \frac{1}{n+1}\bigl(G-V_n(s)\bigr).$$

Generalise this update by replacing $\frac{1}{n+1}$ with a fixed step size $\alpha\in(0,1]$ and explain what role $\alpha$ plays in a non-stationary environment.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

By definition, $V_n(s) = \frac1n\sum_{i=1}^n G^{(i)}$. After observing one additional return $G$, we have

$$V_{n+1}(s) = \frac1{n+1}\left(\sum_{i=1}^n G^{(i)} + G\right).$$

Since $\sum_{i=1}^n G^{(i)} = nV_n(s)$, we get

$$V_{n+1}(s) = \frac{nV_n(s)+G}{n+1} = V_n(s) + \frac1{n+1}\bigl(G - V_n(s)\bigr).$$

The term $G - V_n(s)$ is the prediction error: $V_{n+1}(s) = (1-\alpha)V_n(s)+\alpha G$. In a non-stationary environment, the value $v_\pi(s)$ may change over time. A decreasing step size such as $\frac1{n+1}$ eventually becomes too small to track such changes. A fixed step size $\alpha$ gives more weight to recent returns, allowing the estimator to adapt to changing environment.

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(One sampled Blackjack episode)</span></p>

A single Blackjack episode might be

$$(\text{player 13, dealer 10}) \to \textbf{hit}, \quad (\text{player 18, dealer 10}) \to \textbf{stick}, \quad \text{dealer plays} \to \text{player wins.}$$

The final return is $G = +1$, so this *one* episode contributes to two state estimates:

* the return after the first visit to $(13, 10, \cdot)$ is $G = +1$,
* the return after the first visit to $(18, 10, \cdot)$ is $G = +1$.

The MC update rule is just

$$V(s) \;\leftarrow\; \text{average of all observed first-visit returns to } s.$$

After many games this average concentrates around the true $v_\pi(s)$ for each player-sum/dealer-showing state.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">()</span></p>

Let $G^{(1)},G^{(2)},\ldots$ be the sequence of returns observed at the first visit to $s$ across successive episodes. Explain why these random variables are identically distributed with

$$\mathbb{E}\left[G^{(i)}\right]=v_\pi(s),$$

and why they are mutually independent.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

Let $\tau_i$ be the first hitting time of $s$ in episode $i$ (only episodes with $\tau_i < \infty$ are kept). By the Markov property applied to $\tau_i$, conditional on $\tau_i < \infty$, the post-$\tau_i$ trajectory has the law of a fresh episode started from $s$ under $\pi$. The return $G^{(i)} := G_{\tau_i}$ depends only on that post-$\tau_i$ trajectory, so its conditional law equals that of $G_0$ under $\mathbb{P}\pi(\cdot \mid S_0 = s)$. In particular, 

$$\mathbb{E}[G^{(i)}] = \mathbb{E}\pi[G_0 \mid S_0 = s] =: v_\pi(s).$$

Moreover, every episode is generated independently from the same policy $\pi$, starting from the same initial state $s_0$. Each $G^{(i)}$ is a function only of the $i$-th episode. Therefore the random variables $G^{(1)},G^{(2)},\dots$ are mutually independent. They are also identically distributed, because every episode is generated according to the same transition probabilities and the same fixed policy $\pi$. Hence $G^{(1)},G^{(2)},\dots$ are iid random variables with $\mathbb{E}[G^{(i)}]=v_\pi(s)$.

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">()</span></p>

Use Task 1 to show that the first-visit MC estimator $V_n(s)$ is an unbiased estimator of $v_\pi(s)$, i.e.

$$\mathbb{E}[V_n(s)]=v_\pi(s).$$

Then apply the strong law of large numbers to conclude that

$$V_n(s)\xrightarrow{\mathrm{a.s.}}v_\pi(s) \qquad\text{as }n\to\infty.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

$$\mathbb{E}[V_n(s)] = \mathbb{E}\left[ \frac1n \sum_{i=1}^n G^{(i)} \right] = \frac1n \sum_{i=1}^n \mathbb{E}[G^{(i)}] = \frac1n \sum_{i=1}^n v_\pi(s) = v_\pi(s).$$

Therefore $V_n(s)$ is an unbiased estimator of $v_\pi(s)$. Since the rewards are bounded and $\gamma\in[0,1)$, $G^{(i)}$ is also bounded. Since the boundedness and iid of $G^{(i)}$, strong law of large numbers gives $V_\pi (s)\overset{\text{a.s.}}{\longrightarrow} v_\pi(s)$ as $n\to\infty$.

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Convergence of First-Visit MC)</span></p>

Fix a policy $\pi$. The first-visit MC estimate $V(s)$ converges almost surely to $v_\pi(s)$ as the number of first visits to $s$ goes to infinity.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof Idea</summary>

* Across independently generated episodes, the first-visit returns to $s$ are i.i.d. samples of a random variable with mean $v_\pi(s)$.
* By the strong law of large numbers,

  $$\frac{1}{n}\sum_{i=1}^{n} G^{(i)} \xrightarrow{\text{a.s.}} \mathbb{E}_\pi[\, G_t \mid S_t = s \,] \;=\; v_\pi(s).$$

* The key point is that "the return after the first visit to $s$" is *exactly* the random quantity whose expectation defines $v_\pi(s)$ — no model, no Bellman fixed point, no operator contraction is needed.

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">()</span></p>

Explain why every-visit MC is *not* an unbiased estimator of $v_\pi(s)$ for a single episode, but argue that it is still consistent (converges to $v_\pi(s)$ as the number of episodes grows). Under what condition do first-visit and every-visit MC estimates coincide? 

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

The reason every-visit MC is not an unbiased estimator of $v_\pi(s)$ is that the returns from the same episode are not independent. They are correlated because they come from the same trajectory. However, every-visit Monte Carlo is still consistent. Across many independent episodes, the total collection of visits to $s$ produces more and more returns generated under the same policy $\pi$. Although returns within a single episode may be dependent, different episodes are independent, and the Markov property implies that each visit to $s$ has the correct conditional expected return $v_\pi(s)$. Hence, as the number of episodes grows, the average over all visits converges to \$v_\pi(s)$.

First-visit and every-visit Monte Carlo estimates coincide whenever $s$ is visited at most once in each episode. In that case, the set of all visits to $s$ is exactly the same as the set of first visits to $s$.


</details>
</div>

<figure>
  <img src="{{ '/assets/images/notes/rl_hd/blackjack_mc_value.svg' | relative_url }}" alt="Two pairs of 3D surfaces of estimated Blackjack value functions (with and without a usable ace) after 10,000 and 500,000 sampled episodes, becoming visibly smoother as the sample size grows" loading="lazy">
  <figcaption>MC prediction on Blackjack. The 3D surfaces show estimated state-values for a policy that sticks only on 20 or 21, split by whether the player holds a usable ace. <strong>Left:</strong> after 10,000 episodes the surfaces are jagged; many states have few samples. <strong>Right:</strong> after 500,000 episodes they have settled into smooth functions of (player sum, dealer showing). No model of the dealer's strategy was ever used — only sampled game outcomes.</figcaption>
</figure>

### From Prediction to Control

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Idea</span><span class="math-callout__name">(Why state-values are no longer enough)</span></p>

In DP, policy improvement was a one-step lookahead using the *model*:

$$\pi'(s) \;=\; \arg\max_{a} \sum_{s', r} p(s', r \mid s, a)\bigl[\, r + \gamma\, v(s')\,\bigr].$$

The model $p(s', r \mid s, a)$ was the bridge that let us turn a *state*-value function into a *better policy*. **Without $p$, that bridge collapses.** Knowing $v_\pi(s)$ alone tells us nothing about which action to take in $s$, because we cannot evaluate the one-step lookahead any more.

The fix is to estimate **action-values** $q_\pi(s, a)$ instead — and to estimate them in the same MC way we estimated $v_\pi$, by averaging returns.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(MC Action-Value Estimation)</span></p>

For each state-action pair $(s, a)$, define the MC estimate

$$q_\pi(s, a) \;\doteq\; \mathbb{E}_\pi[\, G_t \mid S_t = s,\, A_t = a \,] \;\approx\; \frac{1}{n}\sum_{i=1}^{n} G_i(s, a),$$

where each $G_i(s, a)$ is the return following a visit to the *pair* $(s, a)$ in some sampled episode. Once $q_\pi(s, a)$ is available, policy improvement becomes a one-line, *model-free* operation:

$$\pi'(s) \;=\; \arg\max_{a} q_\pi(s, a).$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The exploration problem MC inherits)</span></p>

There is one new obstacle that DP did not have. If $\pi$ is **deterministic**, then in each state $s$ only one action is ever taken, so only that one $q_\pi(s, a)$ is ever sampled. All other actions remain unestimated $\Longrightarrow$ no informed improvement is possible.

This is the **same dilemma** that drove the bandit lecture:

> *If you never try an action, you can never learn whether it is good.*

So MC control inherits an inescapable need for **exploration**: we must keep playing some non-greedy actions just to keep all $q$-values estimable. The question is not *whether* to explore but **how to enforce it while still improving the policy**.

</div>

### On-Policy MC Control

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($\varepsilon$-Soft Policies, $\varepsilon$-Greedy)</span></p>

A policy $\pi$ is **$\varepsilon$-soft** for some $\varepsilon > 0$ if it always assigns *at least* a uniform-exploration probability to every action:

$$\pi(a \mid s) \;\geq\; \frac{\varepsilon}{\lvert \mathcal{A}(s) \rvert} \qquad \forall s,\, \forall a \in \mathcal{A}(s).$$

The simplest $\varepsilon$-soft policy is **$\varepsilon$-greedy with respect to $Q$**:

$$
\pi(a \mid s) \;=\;
\begin{cases}
1 - \varepsilon + \dfrac{\varepsilon}{\lvert \mathcal{A}(s) \rvert}, & a = \arg\max_{a'} Q(s, a'), \\[4pt]
\dfrac{\varepsilon}{\lvert \mathcal{A}(s) \rvert}, & \text{otherwise.}
\end{cases}
$$

Restricting attention to $\varepsilon$-soft policies guarantees that every $q(s, a)$ keeps being sampled — so MC estimation never starves.

</div>

<figure class="rl-diagram">
  <svg viewBox="0 0 720 430" role="img" aria-label="The total probability 1 is split into an exploration budget epsilon, spent as a uniform floor of epsilon over the number of actions to every action, and a greedy leftover of one minus epsilon handed entirely to the greedy action; the resulting per-action probabilities are shown below">
    <text x="32" y="34" font-size="18" font-weight="700">ε-greedy as a probability budget</text>
    <text x="32" y="58" font-size="13" class="muted">the whole budget must sum to 1 · illustration: |A(s)| = 4 actions, ε = 0.4</text>

    <!-- Panel 1: the budget bar [0,1] -->
    <text x="32" y="98" font-size="14" font-weight="700">Spend the budget [0, 1]</text>

    <!-- exploration budget: 4 equal floor slices, one per action -->
    <rect x="110" y="112" width="58" height="46" class="amber"></rect>
    <rect x="168" y="112" width="58" height="46" class="amber"></rect>
    <rect x="226" y="112" width="58" height="46" class="amber"></rect>
    <rect x="284" y="112" width="58" height="46" class="amber"></rect>
    <text x="139" y="140" text-anchor="middle" font-size="13" font-weight="700">a★</text>
    <text x="197" y="140" text-anchor="middle" font-size="13" font-weight="700">a₂</text>
    <text x="255" y="140" text-anchor="middle" font-size="13" font-weight="700">a₃</text>
    <text x="313" y="140" text-anchor="middle" font-size="13" font-weight="700">a₄</text>

    <!-- greedy leftover: one minus epsilon, all to the greedy action -->
    <rect x="342" y="112" width="348" height="46" class="accent"></rect>
    <text x="516" y="136" text-anchor="middle" font-size="14" font-weight="700">1 − ε = 0.6</text>
    <text x="516" y="152" text-anchor="middle" font-size="12" class="muted">all to greedy a★</text>

    <!-- ticks -->
    <line x1="110" y1="112" x2="110" y2="176" class="line"></line>
    <line x1="342" y1="112" x2="342" y2="176" class="line" stroke-dasharray="4 4"></line>
    <line x1="690" y1="112" x2="690" y2="176" class="line"></line>
    <text x="110" y="192" text-anchor="middle" font-size="12" class="muted">0</text>
    <text x="342" y="192" text-anchor="middle" font-size="12" class="muted">ε = 0.4</text>
    <text x="690" y="192" text-anchor="middle" font-size="12" class="muted">1</text>

    <!-- brace labels -->
    <text x="226" y="216" text-anchor="middle" font-size="12">exploration budget ε → ε/|A(s)| = 0.1 to every action</text>
    <text x="516" y="216" text-anchor="middle" font-size="12">greedy leftover 1 − ε → the argmax action</text>

    <!-- Panel 2: resulting per-action probabilities -->
    <text x="32" y="262" font-size="14" font-weight="700">Resulting π(a | s)</text>

    <!-- greedy action: floor slice + leftover -->
    <rect x="110" y="278" width="58" height="24" class="amber"></rect>
    <rect x="168" y="278" width="348" height="24" class="accent"></rect>
    <text x="96" y="295" text-anchor="end" font-size="13" font-weight="700">a★</text>
    <text x="524" y="295" font-size="12">0.7 = (1 − ε) + ε/|A(s)|</text>

    <rect x="110" y="312" width="58" height="24" class="amber"></rect>
    <text x="96" y="329" text-anchor="end" font-size="13" font-weight="700">a₂</text>
    <text x="176" y="329" font-size="12">0.1 = ε/|A(s)|</text>

    <rect x="110" y="346" width="58" height="24" class="amber"></rect>
    <text x="96" y="363" text-anchor="end" font-size="13" font-weight="700">a₃</text>
    <text x="176" y="363" font-size="12" class="muted">0.1</text>

    <rect x="110" y="380" width="58" height="24" class="amber"></rect>
    <text x="96" y="397" text-anchor="end" font-size="13" font-weight="700">a₄</text>
    <text x="176" y="397" font-size="12" class="muted">0.1</text>
  </svg>
  <figcaption>The total probability $1$ is a budget. First spend $\varepsilon$ on a uniform <em>floor</em> — $\tfrac{\varepsilon}{\lvert\mathcal{A}(s)\rvert}$ handed to <em>every</em> action, which is exactly what keeps each $\pi(a\mid s) > 0$ so no action-value ever starves. The leftover $1-\varepsilon$ is handed entirely to the greedy action. The greedy action therefore collects <em>both</em> pieces, $\pi(a^\star\mid s) = (1-\varepsilon) + \tfrac{\varepsilon}{\lvert\mathcal{A}(s)\rvert}$, while every other action keeps just its floor $\tfrac{\varepsilon}{\lvert\mathcal{A}(s)\rvert}$. The slices sum to $1$: $\bigl[(1-\varepsilon)+\tfrac{\varepsilon}{\lvert\mathcal{A}(s)\rvert}\bigr] + (\lvert\mathcal{A}(s)\rvert - 1)\tfrac{\varepsilon}{\lvert\mathcal{A}(s)\rvert} = 1$.</figcaption>
</figure>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(The tension in MC control)</span></p>

Model-free control under MC needs two things at the same time:

* estimate **action-values** $q(s, a)$ (no model is available for the lookahead),
* keep **exploring** to discover better actions.

So we restrict attention to $\varepsilon$-soft policies. But this raises a worry: forcing exploration means we are *deliberately* sometimes taking suboptimal actions. Can policy improvement still work under this constraint, or do we get stuck oscillating around a sub-optimal policy?

**Key question.** *If we are forced to keep taking suboptimal exploratory actions, can policy improvement still be guaranteed?*

The next three callouts prove that the answer is **yes** — within the class of $\varepsilon$-soft policies, $\varepsilon$-greedy improvement is monotone.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Policy Improvement for $\varepsilon$-Soft Policies — Step 1)</span></p>

Let $\pi$ be any $\varepsilon$-soft policy with action-value function $q_\pi(s, a)$. Define $\pi'$ as the **$\varepsilon$-greedy policy with respect to $q_\pi$**: it picks the greedy action with probability $1 - \varepsilon$ and explores uniformly with probability $\varepsilon$.

Write

$$q_\pi(s, \pi') \;\doteq\; \sum_{a} \pi'(a \mid s)\, q_\pi(s, a),$$

the expected $q_\pi$-value obtained by choosing the first action according to $\pi'$ and then following $\pi$. Because $\pi'$ is $\varepsilon$-greedy,

$$q_\pi(s, \pi') \;=\; (1 - \varepsilon)\, \max_{a} q_\pi(s, a) \;+\; \frac{\varepsilon}{\lvert \mathcal{A}(s) \rvert} \sum_{a} q_\pi(s, a).$$

So $\pi'$ is a *mixture*: mostly the best action, plus a small uniform-exploration average.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Policy Improvement for $\varepsilon$-Soft Policies — Step 2)</span></p>

Now compare with the value of the *old* policy $\pi$,

$$v_\pi(s) \;=\; \sum_{a} \pi(a \mid s)\, q_\pi(s, a).$$

Since $\pi$ is $\varepsilon$-soft, 

$$\pi(a \mid s) \geq \varepsilon / \lvert \mathcal{A}(s) \rvert \quad \forall a,$$

so we can split

$$\pi(a \mid s) \;=\; \underbrace{\tfrac{\varepsilon}{\lvert \mathcal{A}(s) \rvert}}_{\text{forced exploration}} \;+\; \underbrace{(1 - \varepsilon)\,\tilde{\pi}(a \mid s)}_{\text{remaining mass}},$$

where $\tilde{\pi}(\cdot \mid s)$ is another probability distribution. Plugging into $v_\pi(s)$,

$$v_\pi(s) \;=\; \frac{\varepsilon}{\lvert \mathcal{A}(s) \rvert} \sum_{a} q_\pi(s, a) \;+\; (1 - \varepsilon)\sum_{a} \tilde{\pi}(a \mid s)\, q_\pi(s, a).$$

The second sum is an average of $q_\pi(s, a)$ under $\tilde{\pi}$, so it cannot exceed $\max_{a} q_\pi(s, a)$. Therefore

$$v_\pi(s) \;\leq\; \frac{\varepsilon}{\lvert \mathcal{A}(s) \rvert} \sum_{a} q_\pi(s, a) \;+\; (1 - \varepsilon)\, \max_{a} q_\pi(s, a).$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Policy Improvement for $\varepsilon$-Soft Policies — Step 3)</span></p>

Comparing the two expressions side by side:

$$q_\pi(s, \pi') \;=\; (1 - \varepsilon)\, \max_{a} q_\pi(s, a) \;+\; \frac{\varepsilon}{\lvert \mathcal{A}(s) \rvert} \sum_{a} q_\pi(s, a),$$

$$v_\pi(s) \;\leq\; (1 - \varepsilon)\, \max_{a} q_\pi(s, a) \;+\; \frac{\varepsilon}{\lvert \mathcal{A}(s) \rvert} \sum_{a} q_\pi(s, a).$$

The right-hand sides are *identical*. Hence

$$q_\pi(s, \pi') \;\geq\; v_\pi(s) \qquad \forall s.$$

Choosing the first action according to $\pi'$ and then following $\pi$ has expected return at least as large as following $\pi$ from the start. By the **policy improvement theorem** (DP lecture),

$$\boxed{\, v_{\pi'}(s) \;\geq\; v_\pi(s) \qquad \forall s. \,}$$

**Conclusion.** Even with forced exploration, the $\varepsilon$-greedy policy improves — or at least does not worsen — the old $\varepsilon$-soft policy. The argument is robust: the $\varepsilon/\lvert \mathcal{A}(s) \rvert$ "forced exploration" term cancels exactly on both sides, leaving the improvement entirely to the 

$$(1 - \varepsilon)\max_a q_\pi(s, a)$$ 

contribution.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">($\varepsilon$-Greedy improvement on two actions)</span></p>

Take a single state $s$ with two actions and action-values

$$q_\pi(s, a_1) = 10, \qquad q_\pi(s, a_2) = 0.$$

Suppose the old $\varepsilon$-soft policy is

$$\pi(a_1 \mid s) = 0.6, \qquad \pi(a_2 \mid s) = 0.4, \qquad v_\pi(s) = 0.6 \cdot 10 + 0.4 \cdot 0 = 6.$$

The new **$\varepsilon$-greedy policy** with $\varepsilon = 0.2$ assigns

$$\pi'(a_1 \mid s) = 0.9, \qquad \pi'(a_2 \mid s) = 0.1, \qquad q_\pi(s, \pi') = 0.9 \cdot 10 + 0.1 \cdot 0 = 9.$$

So 

$$q_\pi(s, \pi'(s)) = 9 \geq 6 = v_\pi(s),$$

confirming the theorem on this single example.

The mechanism is intuitive: **$\varepsilon$-greedy shifts probability mass toward the best action, increasing expected value**, while the small forced exploration mass is the *same* on both sides of the comparison and cancels.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(On-Policy First-Visit MC Control)</span></p>

Initialise action-value estimates $Q(s, a)$ arbitrarily, and pick any initial $\varepsilon$-soft policy $\pi$ so that every action can be sampled.

Repeat for each episode:

1. **Generate** an episode using the current policy $\pi$.
2. **Policy evaluation.** For each first visit to a pair $(s, a)$ in the episode, update

   $$Q(s, a) \;\leftarrow\; \text{average of observed returns after } (s, a).$$

3. **Policy improvement.** For every visited state $s$, replace the action probabilities by

   $$\pi(\cdot \mid s) \;\leftarrow\; \varepsilon\text{-greedy w.r.t. } Q(s, \cdot).$$

The same policy that *generates* the data is the one being *improved* — hence the name **on-policy**. By the previous theorem, each improvement step is guaranteed to be monotone within the class of $\varepsilon$-soft policies.

</div>

<figure class="rl-diagram">
  <svg viewBox="0 0 860 360" role="img" aria-label="On-policy Monte Carlo control loop with episode generation, return averaging, and epsilon-greedy improvement">
    <defs>
      <marker id="mc-arrow-onpolicy" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">
        <path d="M0,0 L10,4 L0,8 Z" fill="#2c3e94"></path>
      </marker>
    </defs>
    <text x="32" y="42" font-size="18" font-weight="700">On-policy MC control loop</text>
    <text x="32" y="68" font-size="14" class="muted">The same epsilon-soft policy collects data and is then improved from that data.</text>

    <rect x="78" y="130" width="185" height="88" rx="8" class="accent"></rect>
    <text x="170" y="163" text-anchor="middle" font-size="17" font-weight="700">policy pi</text>
    <text x="170" y="188" text-anchor="middle" font-size="14" class="muted">epsilon-soft</text>

    <rect x="338" y="130" width="185" height="88" rx="8" class="box"></rect>
    <text x="430" y="163" text-anchor="middle" font-size="17" font-weight="700">episode</text>
    <text x="430" y="188" text-anchor="middle" font-size="14" class="muted">S, A, R, ..., terminal</text>

    <rect x="598" y="130" width="185" height="88" rx="8" class="green"></rect>
    <text x="690" y="163" text-anchor="middle" font-size="17" font-weight="700">Q(s,a)</text>
    <text x="690" y="188" text-anchor="middle" font-size="14" class="muted">average returns</text>

    <line x1="263" y1="174" x2="338" y2="174" class="strong-line" marker-end="url(#mc-arrow-onpolicy)"></line>
    <text x="300" y="154" text-anchor="middle" font-size="13" class="muted">generate</text>

    <line x1="523" y1="174" x2="598" y2="174" class="strong-line" marker-end="url(#mc-arrow-onpolicy)"></line>
    <text x="560" y="154" text-anchor="middle" font-size="13" class="muted">evaluate</text>

    <path d="M690 218 C690 288, 170 288, 170 220" class="strong-line" marker-end="url(#mc-arrow-onpolicy)"></path>
    <text x="430" y="306" text-anchor="middle" font-size="15" font-weight="700">improve: make pi epsilon-greedy with respect to Q</text>

    <rect x="295" y="88" width="270" height="26" rx="13" class="amber"></rect>
    <text x="430" y="106" text-anchor="middle" font-size="13" font-weight="700">Generalised Policy Iteration, implemented with sampled returns</text>
  </svg>
  <figcaption>On-policy MC control is GPI with sampling: generate an episode under the current exploratory policy, average returns into $Q$, then move the same policy toward $\varepsilon$-greedy behaviour.</figcaption>
</figure>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">()</span></p>

Let $\pi$ be any $\varepsilon$-soft policy and let $\pi'$ be the $\varepsilon$-greedy policy with respect to $q_\pi$. Prove the $\varepsilon$-soft policy improvement theorem:

$$v_{\pi'}(s)\geq v_\pi(s) \qquad \text{for all }s\in\mathcal S.$$

> **Hint.** Start from
>
> $$v_\pi(s)=\sum_a\pi(a\mid s)q_\pi(s,a)$$
>
> and write
>
> $$\pi(a\mid s) = \left(\pi(a\mid s)-\frac{\varepsilon}{|\mathcal A|}\right) + \frac{\varepsilon}{|\mathcal A|}.$$
>
> Use the $\varepsilon$-soft condition to bound the first term from above.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

Let

$$a^*(s)\in\arg\max_a q_\pi(s,a), \qquad M(s):=\max_a q_\pi(s,a).$$

**Separate the mandatory exploration mass.**

Since $\pi$ is $\varepsilon$-soft, we may write

$$\pi(a\mid s)=\frac{\varepsilon}{m}+\delta(a\mid s),$$

where

$$\delta(a\mid s)\ge 0, \qquad \sum_a\delta(a\mid s)=1-\varepsilon.$$

Therefore,

$$
\begin{aligned}
v_\pi(s)
&=\sum_a\pi(a\mid s)q_\pi(s,a)\\
&=\frac{\varepsilon}{m}\sum_aq_\pi(s,a)
+\sum_a\delta(a\mid s)q_\pi(s,a)\\
&\le
\frac{\varepsilon}{m}\sum_aq_\pi(s,a)
+M(s)\sum_a\delta(a\mid s)\\
&=
\frac{\varepsilon}{m}\sum_aq_\pi(s,a)
+(1-\varepsilon)M(s).
\end{aligned}
$$

But $\pi'$ is $\varepsilon$-greedy with respect to $q_\pi$, so

$$
\begin{aligned}
\sum_a\pi'(a\mid s)q_\pi(s,a)
&=
\left(1-\varepsilon+\frac{\varepsilon}{m}\right)M(s)
+\sum_{a\ne a^*(s)}\frac{\varepsilon}{m}q_\pi(s,a)\\
&=
(1-\varepsilon)M(s)
+\frac{\varepsilon}{m}\sum_aq_\pi(s,a).
\end{aligned}
$$

Consequently,

$$v_\pi(s) \le \sum_a\pi'(a\mid s)q_\pi(s,a) = (T_{\pi'}v_\pi)(s).$$

This is exactly the one-step policy-improvement inequality established in the lecture notes. 

**Iterate the Bellman operator.**

Since $T_{\pi'}$ is monotone,

$$v_\pi \le T_{\pi'}v_\pi \le T_{\pi'}^2v_\pi \le\cdots.$$

Moreover, $T_{\pi'}$ is a $\gamma$-contraction, so

$$T_{\pi'}^nv_\pi\longrightarrow v_{\pi'}.$$

Taking limits gives

$$\boxed{v_{\pi'}(s)\ge v_\pi(s)\qquad\text{for every }s.}$$

Equality in the one-step inequality holds precisely when the extra mass

$$\pi(a\mid s)-\frac{\varepsilon}{m}$$

is supported only on maximizing actions. With a unique maximizing action, this means that $\pi$ is already $\varepsilon$-greedy at $s$.

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">()</span></p>

Suppose $\varepsilon$-greedy policy improvement does not change the policy, i.e. $\pi'=\pi$. Show that in this case

$$v_\pi(s) = \frac{\varepsilon}{|\mathcal A|} \sum_a q_\pi(s,a) + (1-\varepsilon)\max_a q_\pi(s,a) \qquad \text{for all }s\in\mathcal S,$$

and interpret this equation as a fixed-point condition.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

Suppose policy improvement changes nothing:

$$\pi'=\pi.$$

Then $\pi$ itself is $\varepsilon$-greedy with respect to $q_\pi$. Hence

$$
\begin{aligned}
v_\pi(s)
&=\sum_a\pi(a\mid s)q_\pi(s,a)\\
&= \left(1-\varepsilon+\frac{\varepsilon}{m}\right) \max_a q_\pi(s,a) + \sum_{a\ne a^*(s)} \frac{\varepsilon}{m}q_\pi(s,a)\\
&= (1-\varepsilon)\max_aq_\pi(s,a) + \frac{\varepsilon}{m}\sum_aq_\pi(s,a).
\end{aligned}
$$

Thus

$$\boxed{v_\pi(s) = \frac{\varepsilon}{m}\sum_a q_\pi(s,a)+(1-\varepsilon)\max_a q_\pi(s,a).}$$

This is the equation requested on the sheet. 

**Interpret it as a Bellman fixed point.**

For an arbitrary value function $v$, define

$$q_v(s,a) := r(s,a)+\gamma\sum_{s'}p(s'\mid s,a)v(s')$$

and the $\varepsilon$-soft optimality operator

$$(T_\varepsilon v)(s) := \frac{\varepsilon}{m}\sum_a q_v(s,a) + (1-\varepsilon)\max_a q_v(s,a).$$

Since $q_\pi=q_{v_\pi}$, the preceding equation becomes

$$\boxed{v_\pi=T_\varepsilon v_\pi.}$$

The operator $T_\varepsilon$ is a $\gamma$-contraction. Indeed,

$$|T_\varepsilon v-T_\varepsilon w|_\infty \le \gamma|v-w|_\infty.$$

It therefore has a unique fixed point. This fixed point is the optimal value function among all $\varepsilon$-soft policies. Thus, when $\varepsilon$-greedy improvement leaves $\pi$ unchanged, $\pi$ is optimal within the constrained class of $\varepsilon$-soft policies.

It need not be optimal among all policies when $\varepsilon>0$, because it is forced to take non-greedy actions with positive probability.

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">()</span></p>

A sequence of policies $\lbrace \pi_k\rbrace$ is called **GLIE** (*Greedy in the Limit with Infinite Exploration*) if:

1. Every state-action pair is visited infinitely often:

   $$\sum_{k=1}^{\infty} \mathbf 1[(s,a)\text{ visited in episode }k] = \infty \qquad \text{for all }(s,a).$$

2. The policies converge to the greedy policy:

   $$\pi_k(a\mid s) \longrightarrow \mathbf 1\left[ a=\arg\max_{a'}q(s,a') \right] \qquad \text{as }k\to\infty$$

   for all $s$.

Show that $\varepsilon$-greedy with

$$\varepsilon_k=\frac{1}{k}$$

satisfies both GLIE conditions, assuming every state-action pair is visited at least once per episode. 

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

Let

$$\varepsilon_k=\frac1k.$$

The two GLIE requirements are infinite exploration and convergence toward greedy action selection. 

**Greedy in the limit**

Let $a^\ast(s)$ be the consistently chosen greedy action. Then

$$
\pi_k(a\mid s)
=
\begin{cases}
1-\dfrac1k+\dfrac1{km},&a=a^*(s), \\
\dfrac1{km},&a\ne a^*(s).
\end{cases}
$$

Therefore,

$$\pi_k(a^*(s)\mid s) = 1-\frac{1-1/m}{k} \longrightarrow 1,$$

while for every non-greedy action,

$$\pi_k(a\mid s)=\frac1{km}\longrightarrow0.$$

Hence

$$\boxed{\pi_k(a\mid s)\longrightarrow \mathbf 1_{{a=a^*(s)}}.}$$

Thus the policies become greedy in the limit.

**Infinite exploration**

Under the assumption literally stated on the sheet—that every state–action pair is visited at least once per episode—we have

$$\mathbf 1_{{(s,a)\text{ visited in episode }k}}=1$$

for every $k$. Hence

$$\sum_{k=1}^{\infty} = \mathbf 1_{{(s,a)\text{ visited in episode }k}} =  \sum_{k=1}^{\infty}1 = \infty.$$

Thus every pair is visited infinitely often.

Therefore the schedule $\varepsilon_k=1/k$ satisfies both GLIE conditions.

**More standard interpretation**

The assumption on the sheet is stronger than necessary and makes infinite exploration immediate. Usually one assumes only that state $s$ is encountered infinitely often. At the $k$-th opportunity,

$$\mathbb P(A=a\mid S=s)\ge \frac1{km}.$$

Since

$$\sum_{k=1}^{\infty}\frac1{km} = \frac1m\sum_{k=1}^{\infty}\frac1k = \infty,$$

the exploration probabilities are not summable, so each action continues to be selected infinitely often almost surely. The harmonic decay $1/k$ is exactly slow enough to preserve infinite exploration while still tending to zero.

</details>
</div>

### Off-Policy MC Learning

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Idea</span><span class="math-callout__name">(Why off-policy)</span></p>

In **on-policy** learning a single policy plays two roles at once: it generates the data *and* is the policy being improved. The price is that the learned policy must remain exploratory forever — we cannot turn $\varepsilon$ down to $0$ without losing the exploration channel.

**Off-policy** learning separates these two roles.

* The **behavior policy** $\mu$ is the one that actually interacts with the environment. It is *exploratory* by design — for example $\varepsilon$-soft.
* The **target policy** $\pi$ is the one we want to evaluate or improve. It can be *greedy*, even deterministic, and need not match $\mu$.

**Key advantage.** Off-policy learning lets us **explore with one policy and learn about another** — we can collect richly diverse data with $\mu$ while training $\pi$ toward the nearly-deterministic policy we actually intend to deploy.

</div>

<figure class="rl-diagram">
  <svg viewBox="0 0 860 330" role="img" aria-label="Off-policy Monte Carlo separates the behavior policy from the target policy">
    <defs>
      <marker id="mc-arrow-offpolicy" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">
        <path d="M0,0 L10,4 L0,8 Z" fill="#2c3e94"></path>
      </marker>
      <marker id="mc-arrow-offpolicy-muted" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">
        <path d="M0,0 L10,4 L0,8 Z" fill="#64748b"></path>
      </marker>
    </defs>
    <text x="32" y="42" font-size="18" font-weight="700">Two policies, two roles</text>
    <text x="32" y="68" font-size="14" class="muted">The behaviour policy supplies data; the target policy is the object being evaluated or improved.</text>

    <rect x="70" y="115" width="185" height="88" rx="8" class="amber"></rect>
    <text x="162" y="148" text-anchor="middle" font-size="17" font-weight="700">behaviour mu</text>
    <text x="162" y="173" text-anchor="middle" font-size="14" class="muted">exploratory data</text>

    <rect x="338" y="115" width="185" height="88" rx="8" class="box"></rect>
    <text x="430" y="148" text-anchor="middle" font-size="17" font-weight="700">episodes</text>
    <text x="430" y="173" text-anchor="middle" font-size="14" class="muted">sampled under mu</text>

    <rect x="606" y="115" width="185" height="88" rx="8" class="accent"></rect>
    <text x="698" y="148" text-anchor="middle" font-size="17" font-weight="700">target pi</text>
    <text x="698" y="173" text-anchor="middle" font-size="14" class="muted">policy we learn about</text>

    <line x1="255" y1="159" x2="338" y2="159" class="strong-line" marker-end="url(#mc-arrow-offpolicy)"></line>
    <text x="296" y="139" text-anchor="middle" font-size="13" class="muted">acts</text>

    <line x1="523" y1="159" x2="606" y2="159" class="strong-line" marker-end="url(#mc-arrow-offpolicy)"></line>
    <text x="565" y="139" text-anchor="middle" font-size="13" class="muted">reweight</text>

    <path d="M698 204 C698 260, 430 276, 162 204" class="line" marker-end="url(#mc-arrow-offpolicy-muted)"></path>
    <text x="430" y="285" text-anchor="middle" font-size="14" class="muted">coverage: if pi can choose an action, mu must sometimes choose it too</text>
  </svg>
  <figcaption>Off-policy MC is not "learning from the wrong policy" naively. It uses behaviour-policy samples and corrects their distribution so they estimate target-policy quantities.</figcaption>
</figure>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The Off-Policy Prediction Problem)</span></p>

We want the target-policy expectation

$$v_\pi(s) \;=\; \mathbb{E}_\pi[\, G_t \mid S_t = s \,],$$

but the returns we observe come from episodes generated by $\mu$:

$$G_t^{(1)}, G_t^{(2)}, \dots \;\sim\; \mu, \qquad \text{not } \pi.$$

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Problem</span><span class="math-callout__name">(Core difficulty of Off-Policy Prediction Problem)</span></p>

We need to compute an expectation under one distribution ($\mathbb{E}\_\pi[\cdot]$) while observing samples from another ($\mathbb{E}\_\mu[\cdot]$). Naively averaging the observed $G_t^{(i)}$ would estimate $v_\mu$, not $v_\pi$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Coverage Assumption)</span></p>

For off-policy learning to be possible at all, the behavior policy must "see" every action the target policy might take:

$$\pi(a \mid s) > 0 \quad \Longrightarrow \quad \mu(a \mid s) > 0.$$

That is, if the target policy would ever choose action $a$ in state $s$, the behavior policy must sometimes choose it too. Otherwise there are entire pieces of target-policy experience that *never* appear in the data — and no statistical trick can recover them.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Idea</span><span class="math-callout__name">(Importance sampling — the statistical idea)</span></p>

* Suppose we want 

  $$\mathbb{E}_\pi[f(X)] = \sum_x \pi(x) f(x),$$

  but our samples come from a *different* distribution $\mu$. 

* Whenever coverage holds, multiply and divide by $\mu(x)$:

  $$\mathbb{E}_\pi[f(X)] \;=\; \sum_{x} \pi(x) f(x) \;=\; \sum_{x} \mu(x) \frac{\pi(x)}{\mu(x)} f(x) \;=\; \mathbb{E}_\mu\!\left[\, \frac{\pi(X)}{\mu(X)}\, f(X) \,\right].$$

So samples drawn from $\mu$, after being **reweighted by the ratio $\pi(X)/\mu(X)$**, have the same expectation as if they had come from $\pi$. The ratio

$$\frac{\pi(X)}{\mu(X)} \;=\; \frac{\text{target probability}}{\text{behavior probability}}$$

is called the **importance sampling ratio**.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(Importance sampling in RL — trajectories, not single variables)</span></p>

In RL the random variable is not a single $X$ but an entire **trajectory** generated by a policy. So the correction factor is

$$\frac{\Pr\nolimits_\pi(\text{trajectory})}{\Pr\nolimits_\mu(\text{trajectory})}.$$

**Intuition.** Trajectories that are more likely under the target policy $\pi$ than under the behavior policy $\mu$ receive **larger weight**; trajectories less typical of $\pi$ receive **smaller weight**. Importance sampling thus *re-emphasises* the parts of $\mu$'s experience that are typical of $\pi$ and *down-weights* the rest.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(The IS Ratio Depends Only on the Policies)</span></p>

For a trajectory 

$$S_t, A_t, S_{t+1}, \dots, S_T$$

generated under policy $\pi$ versus under $\mu$,

$$
\frac{\Pr\nolimits_\pi(\text{trajectory})}{\Pr\nolimits_\mu(\text{trajectory})} \;=\; \frac{\prod_{k=t}^{T-1} \pi(A_k \mid S_k)\, P(S_{k+1} \mid S_k, A_k)}{\prod_{k=t}^{T-1} \mu(A_k \mid S_k)\, P(S_{k+1} \mid S_k, A_k)} \;=\; \prod_{k=t}^{T-1} \frac{\pi(A_k \mid S_k)}{\mu(A_k \mid S_k)} \;=:\; \rho_{t}^{T}.
$$

The environment transition probabilities $P(S_{k+1} \mid S_k, A_k)$ **cancel** — they are the same under either policy.

**Takeaway.** The importance sampling ratio $\rho_{t}^{T}$ can be computed *from the policies alone*. **No model of the environment is needed.** This is the structural reason IS is usable in model-free RL.

</div>

<figure class="rl-diagram">
  <svg viewBox="0 0 860 360" role="img" aria-label="Importance sampling ratio cancels environment probabilities and keeps policy ratios">
    <defs>
      <marker id="mc-arrow-is" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">
        <path d="M0,0 L10,4 L0,8 Z" fill="#64748b"></path>
      </marker>
    </defs>
    <text x="32" y="42" font-size="18" font-weight="700">Why the model cancels</text>
    <text x="32" y="68" font-size="14" class="muted">The same environment probabilities appear under both policies; only the action probabilities differ.</text>

    <g transform="translate(70 120)">
      <rect x="0" y="0" width="150" height="58" rx="8" class="accent"></rect>
      <text x="75" y="24" text-anchor="middle" font-size="14" font-weight="700">pi(A<tspan baseline-shift="sub" font-size="10">k</tspan>|S<tspan baseline-shift="sub" font-size="10">k</tspan>)</text>
      <text x="75" y="44" text-anchor="middle" font-size="13" class="muted">target action</text>
      <line x1="150" y1="29" x2="238" y2="29" class="line" marker-end="url(#mc-arrow-is)"></line>

      <rect x="248" y="0" width="150" height="58" rx="8" class="green"></rect>
      <text x="323" y="24" text-anchor="middle" font-size="14" font-weight="700">P environment</text>
      <text x="323" y="44" text-anchor="middle" font-size="13" class="muted">environment</text>
      <line x1="398" y1="29" x2="486" y2="29" class="line" marker-end="url(#mc-arrow-is)"></line>

      <rect x="496" y="0" width="150" height="58" rx="8" class="accent"></rect>
      <text x="571" y="24" text-anchor="middle" font-size="14" font-weight="700">pi(A<tspan baseline-shift="sub" font-size="10">k+1</tspan>|S<tspan baseline-shift="sub" font-size="10">k+1</tspan>)</text>
      <text x="571" y="44" text-anchor="middle" font-size="13" class="muted">target action</text>
    </g>

    <line x1="92" y1="210" x2="748" y2="210" stroke="#dbe1ee" stroke-width="2"></line>

    <g transform="translate(70 235)">
      <rect x="0" y="0" width="150" height="58" rx="8" class="amber"></rect>
      <text x="75" y="24" text-anchor="middle" font-size="14" font-weight="700">mu(A<tspan baseline-shift="sub" font-size="10">k</tspan>|S<tspan baseline-shift="sub" font-size="10">k</tspan>)</text>
      <text x="75" y="44" text-anchor="middle" font-size="13" class="muted">behaviour action</text>
      <line x1="150" y1="29" x2="238" y2="29" class="line" marker-end="url(#mc-arrow-is)"></line>

      <rect x="248" y="0" width="150" height="58" rx="8" class="green"></rect>
      <text x="323" y="24" text-anchor="middle" font-size="14" font-weight="700">P environment</text>
      <text x="323" y="44" text-anchor="middle" font-size="13" class="muted">same environment</text>
      <line x1="398" y1="29" x2="486" y2="29" class="line" marker-end="url(#mc-arrow-is)"></line>

      <rect x="496" y="0" width="150" height="58" rx="8" class="amber"></rect>
      <text x="571" y="24" text-anchor="middle" font-size="14" font-weight="700">mu(A<tspan baseline-shift="sub" font-size="10">k+1</tspan>|S<tspan baseline-shift="sub" font-size="10">k+1</tspan>)</text>
      <text x="571" y="44" text-anchor="middle" font-size="13" class="muted">behaviour action</text>
    </g>

    <text x="742" y="218" text-anchor="middle" font-size="22" font-weight="700">=</text>
    <text x="742" y="242" text-anchor="middle" font-size="13" class="muted">cancel P terms</text>
    <text x="430" y="330" text-anchor="middle" font-size="16" font-weight="700">rho = product of target-probability / behaviour-probability factors</text>
  </svg>
  <figcaption>The likelihood ratio looks like a trajectory-model ratio, but the transition model appears in numerator and denominator. The only surviving factors are $\pi(A_k \mid S_k) / \mu(A_k \mid S_k)$.</figcaption>
</figure>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(IS-Corrected Off-Policy MC Estimate)</span></p>

The goal is to estimate the target-policy value

$$v_\pi(s) \;=\; \mathbb{E}_\pi[\, G_t \mid S_t = s \,]$$

from episodes generated by $\mu$. Importance sampling rewrites this as an expectation under $\mu$:

$$v_\pi(s) \;=\; \mathbb{E}_\mu\!\bigl[\, \rho_{t}^{T}\, G_t \mid S_t = s \,\bigr], \qquad \rho_{t}^{T} = \prod_{k=t}^{T-1} \frac{\pi(A_k \mid S_k)}{\mu(A_k \mid S_k)}.$$

Replacing the expectation by an empirical average over the set $\mathcal{T}(s)$ of sampled visit times to $s$ gives the **ordinary off-policy MC estimator**

$$V(s) \;\approx\; \frac{1}{\lvert \mathcal{T}(s) \rvert} \sum_{t \in \mathcal{T}(s)} \rho_{t}^{T}\, G_t.$$

**Takeaway.** Off-policy MC $=$ ordinary MC averaging $+$ importance weights.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Importance sampling on Blackjack)</span></p>

Take

* target policy $\pi$: **stick whenever player sum $\geq 20$** (otherwise hit);
* behavior policy $\mu$: exploratory, picks both actions with non-zero probability everywhere.

Observe the *tail* of an episode starting at player sum 18:

$$(18, \cdot) : \text{hit} \;\to\; (20, \cdot) : \text{stick}.$$

The importance weight for this tail is

$$\rho \;=\; \frac{\pi(\text{hit} \mid 18)}{\mu(\text{hit} \mid 18)} \cdot \frac{\pi(\text{stick} \mid 20)}{\mu(\text{stick} \mid 20)}.$$

If $\pi$ would also hit on 18, the first factor is positive. If $\pi$ would *never* hit on 18, then $\pi(\text{hit} \mid 18) = 0$ and so $\rho = 0$.

**Interpretation.** Behaviour trajectories *incompatible* with the target policy receive zero weight; they contribute nothing to the estimate. Only the parts of $\mu$'s experience that "look like" $\pi$ count.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Why importance weights become unstable — toy model)</span></p>

Suppose an episode has length $H$, and at each step the per-step ratio

$$
X_k \;=\; \frac{\pi(A_k \mid S_k)}{\mu(A_k \mid S_k)} \;=\;
\begin{cases}
3,         & \text{with probability } 0.25, \\[2pt]
\tfrac{1}{3}, & \text{with probability } 0.75.
\end{cases}
$$

The total importance weight is

$$\rho \;=\; \prod_{k=1}^{H} X_k.$$

**Question.** What typically happens to $\rho$ as $H$ grows? Do most trajectories carry large weight, small weight, or moderate weight? Which trajectories dominate MC averages?

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Working out the instability for $H = 4$)</span></p>

Each step multiplies the weight by either $3$ or $1/3$. With $H = 4$ the possible total weights are:

| Sequence of ratios | $\rho$ |
| :----------------- | :----- |
| $3, 3, 3, 3$ | $81$ |
| $3, 3, 3, 1/3$ | $9$ |
| $3, 1/3, 1/3, 1/3$ | $1/9$ |
| $1/3, 1/3, 1/3, 1/3$ | $1/81$ |

Two facts follow.

* The possible weights span the range $\tfrac{1}{81}$ to $81$, so the **variance** of $\rho$ is huge.
* The shrinking factor $1/3$ occurs three times more often than $3$, so most trajectories receive **very small weight**, while a few rare trajectories (many factors of $3$) receive **enormous weight**.

Even when $\mathbb{E}\_\mu[\rho] = 1$ (the likelihood ratio is correctly normalised on average), products of likelihood ratios become **very spread out** as $H$ grows. A handful of rare trajectories can therefore *dominate* the MC average. This is the structural reason ordinary IS becomes unreliable on long episodes.

</div>

<figure class="rl-diagram">
  <svg viewBox="0 0 860 340" role="img" aria-label="Importance sampling weights spread out multiplicatively over episode length">
    <text x="32" y="42" font-size="18" font-weight="700">Multiplicative weights spread out</text>
    <text x="32" y="68" font-size="14" class="muted">After only four steps, products of per-step ratios already span several orders of magnitude.</text>

    <line x1="105" y1="270" x2="760" y2="270" class="line"></line>
    <line x1="105" y1="90" x2="105" y2="270" class="line"></line>
    <text x="80" y="94" text-anchor="end" font-size="13" class="muted">large</text>
    <text x="80" y="270" text-anchor="end" font-size="13" class="muted">small</text>

    <rect x="145" y="92" width="95" height="178" class="red"></rect>
    <text x="192" y="84" text-anchor="middle" font-size="16" font-weight="700">81</text>
    <text x="192" y="295" text-anchor="middle" font-size="13" class="muted">3,3,3,3</text>

    <rect x="295" y="196" width="95" height="74" class="amber"></rect>
    <text x="342" y="188" text-anchor="middle" font-size="16" font-weight="700">9</text>
    <text x="342" y="295" text-anchor="middle" font-size="13" class="muted">three 3s</text>

    <rect x="445" y="252" width="95" height="18" class="box"></rect>
    <text x="492" y="244" text-anchor="middle" font-size="16" font-weight="700">1/9</text>
    <text x="492" y="295" text-anchor="middle" font-size="13" class="muted">one 3</text>

    <rect x="595" y="266" width="95" height="4" class="box"></rect>
    <text x="642" y="258" text-anchor="middle" font-size="16" font-weight="700">1/81</text>
    <text x="642" y="295" text-anchor="middle" font-size="13" class="muted">no 3s</text>

    <path d="M210 112 C360 42, 520 44, 666 248" stroke="#b91c1c" stroke-width="3" fill="none" stroke-dasharray="8 8"></path>
    <text x="510" y="106" font-size="14" class="muted">rare high-weight tails can dominate the average</text>
  </svg>
  <figcaption>Ordinary importance sampling is unbiased, but the product form can make a few trajectories carry enormous weight while most carry almost none. Weighted IS trades finite-sample bias for much lower variance.</figcaption>
</figure>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Ordinary vs. Weighted Importance Sampling)</span></p>

Two estimators are standard.

* **Ordinary importance sampling.**

  $$V(s) \;=\; \frac{1}{\lvert \mathcal{T}(s) \rvert} \sum_{t \in \mathcal{T}(s)} \rho_{t}^{T(t)}\, G_t.$$

* **Weighted importance sampling.**

  $$V(s) \;=\; \frac{\sum_{t \in \mathcal{T}(s)} \rho_{t}^{T(t)}\, G_t}{\sum_{t \in \mathcal{T}(s)} \rho_{t}^{T(t)}}.$$

| Estimator       | Bias                  | Variance                         |
| :-------------- | :-------------------- | :------------------------------- |
| Ordinary IS     | none (unbiased)       | often enormous                   |
| Weighted IS     | finite-sample bias    | dramatically smaller             |

**Reading.** Ordinary IS is unbiased but its variance explodes with episode length (as the toy example just showed). Weighted IS introduces a small finite-sample bias — the normaliser is itself a random variable — but the bias vanishes as $\lvert \mathcal{T}(s) \rvert \to \infty$, and its variance is much better behaved. In practice **weighted IS is almost always preferred**.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Incremental Weighted IS for First-Visit Off-Policy MC Prediction)</span></p>

For each state $s$ maintain two scalars:

$$V(s) \;\approx\; v_\pi(s), \qquad C(s) \;=\; \text{cumulative sum of importance weights for } s.$$

For each episode generated by the behaviour policy $\mu$:

1. Compute the returns $G_t$ for all time steps.
2. For each state $s$ that appears in the episode, let

   $$t_s = \min\lbrace\, t : S_t = s \,\rbrace$$

   be its first visit time.

3. Compute the importance weight from that first visit onward:

   $$\rho_{t_s}^{T} \;=\; \prod_{k=t_s}^{T-1} \frac{\pi(A_k \mid S_k)}{\mu(A_k \mid S_k)}.$$

4. **Update:**

   $$C(s) \;\leftarrow\; C(s) + \rho_{t_s}^{T},$$

   $$V(s) \;\leftarrow\; V(s) + \frac{\rho_{t_s}^{T}}{C(s)}\bigl(\, G_{t_s} - V(s)\,\bigr).$$

**Meaning.** Each state is updated once per episode, using its first observed return and its first-visit importance weight. The update is a *weighted* incremental mean: weights $\rho_{t_s}^{T}$ accumulate in $C(s)$ and form the denominator of an effective sample average.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Recall</summary>

$$\bar{Q_N} = \frac{Q_1 + \dots + Q_N}{N} = \frac{Q_1 + \dots + Q_{N-1}}{N} + \frac{Q_N}{N} = \frac{\bar{Q_{N-1}(N-1)}}{N} + \frac{Q_N}{N} = \bar{Q_{N-1}} + \frac{1}{N}(Q_N - \bar{Q_{N-1}})$$

</details>
</div>

### Off-Policy MC Control

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Idea</span><span class="math-callout__name">(From $V$ to $Q$ in the off-policy setting)</span></p>

Off-policy *prediction* estimated $V(s)$. Off-policy *control* must estimate $Q(s, a)$ — because, as in the on-policy case, policy improvement without a model requires comparing actions:

$$v_\pi(s) = \mathbb{E}_\pi[\, G_t \mid S_t = s \,] \quad\Longrightarrow\quad q_\pi(s, a) = \mathbb{E}_\pi[\, G_t \mid S_t = s,\, A_t = a \,].$$

Episodes are generated by a soft exploratory behaviour policy $\mu$, while the target policy $\pi$ is improved **greedily**:

$$Q(s, a) \;\approx\; \frac{\sum_{t \in \mathcal{T}(s, a)} \rho_{t+1}^{T(t)}\, G_t}{\sum_{t \in \mathcal{T}(s, a)} \rho_{t+1}^{T(t)}}, \qquad \rho_{t+1}^{T} = \prod_{k = t+1}^{T-1} \frac{\pi(A_k \mid S_k)}{\mu(A_k \mid S_k)},$$

$$\pi(s) \;=\; \arg\max_{a} Q(s, a).$$

Note that the ratio starts at $k = t+1$, not $k = t$: the action $A_t$ at the visited pair $(s, a)$ is *given* (we are conditioning on it), so only the *subsequent* actions need to be re-weighted. The rest is the same off-policy machinery as before: coverage, importance weights, weighted averaging, greedy improvement.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(A fundamental weakness of off-policy MC control)</span></p>

Off-policy MC control can learn a *greedy* target policy $\pi$ from exploratory data generated by $\mu$, but MC updates only use trajectory tails **consistent with $\pi$**.

If the target policy is **deterministic greedy**, then any non-greedy behaviour action has zero probability under $\pi$,

$$A_t \neq \pi(S_t) \quad\Longrightarrow\quad \pi(A_t \mid S_t) = 0,$$

and the importance weight becomes

$$\rho \;=\; \prod_{k} \frac{\pi(A_k \mid S_k)}{\mu(A_k \mid S_k)} \;=\; 0.$$

**Consequence.** In long episodes, *many* MC updates receive zero or unstable weights. As soon as the behaviour policy takes even one non-greedy action, the entire remaining trajectory is discarded. Off-policy MC control is therefore notoriously **sample-inefficient**: episodes that *almost* match the greedy target policy contribute nothing past their first divergence.

This is one of the main motivations to move on to **off-policy TD methods** (Q-learning), where bootstrapping bypasses the multiplicative explosion of importance weights by replacing the full return with a one-step backup.

</div>

### Summary

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(Method map — Monte Carlo at a glance)</span></p>

| Object | Meaning | Key idea |
| :----- | :------ | :------- |
| $G_t$ | discounted return | actual sum of future rewards |
| $V(s)$ | MC estimate of $v_\pi(s)$ | average of first-visit returns |
| $Q(s, a)$ | MC estimate of $q_\pi(s, a)$ | needed for model-free control |
| $\pi_\varepsilon$ | $\varepsilon$-soft policy | exploration guarantee |
| $\rho_{t}^{T}$ | IS ratio | corrects for off-policy distribution |

Reading the table top to bottom recapitulates the whole lecture: average returns to get $V$, average returns conditioned on actions to get $Q$, force exploration via $\pi_\varepsilon$ to keep $Q$ estimable, and use $\rho_{t}^{T}$ when the data was generated by a *different* policy from the one you want to learn.

</div>

<figure class="rl-diagram">
  <svg viewBox="0 0 860 370" role="img" aria-label="Monte Carlo method map from sampled returns to prediction, control, and off-policy correction">
    <defs>
      <marker id="mc-arrow-map" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">
        <path d="M0,0 L10,4 L0,8 Z" fill="#2c3e94"></path>
      </marker>
    </defs>
    <text x="32" y="42" font-size="18" font-weight="700">Monte Carlo method map</text>
    <text x="32" y="68" font-size="14" class="muted">Almost every MC variant is obtained by changing what is averaged and which policy generated the episode.</text>

    <rect x="50" y="135" width="145" height="70" rx="8" class="box"></rect>
    <text x="122" y="164" text-anchor="middle" font-size="16" font-weight="700">episodes</text>
    <text x="122" y="186" text-anchor="middle" font-size="13" class="muted">sampled data</text>

    <rect x="260" y="70" width="160" height="70" rx="8" class="accent"></rect>
    <text x="340" y="99" text-anchor="middle" font-size="16" font-weight="700">average G<tspan baseline-shift="sub" font-size="10">t</tspan></text>
    <text x="340" y="121" text-anchor="middle" font-size="13" class="muted">prediction: V(s)</text>

    <rect x="260" y="200" width="160" height="70" rx="8" class="green"></rect>
    <text x="340" y="224" text-anchor="middle" font-size="16" font-weight="700">average G<tspan baseline-shift="sub" font-size="10">t</tspan></text>
    <text x="340" y="246" text-anchor="middle" font-size="13" class="muted">after (s,a): Q(s,a)</text>

    <rect x="500" y="200" width="160" height="70" rx="8" class="amber"></rect>
    <text x="580" y="229" text-anchor="middle" font-size="16" font-weight="700">epsilon-greedy</text>
    <text x="580" y="251" text-anchor="middle" font-size="13" class="muted">keeps exploring</text>

    <rect x="500" y="70" width="160" height="70" rx="8" class="box"></rect>
    <text x="580" y="99" text-anchor="middle" font-size="16" font-weight="700">multiply by rho</text>
    <text x="580" y="121" text-anchor="middle" font-size="13" class="muted">off-policy correction</text>

    <rect x="705" y="135" width="110" height="70" rx="8" class="accent"></rect>
    <text x="760" y="164" text-anchor="middle" font-size="16" font-weight="700">target pi</text>
    <text x="760" y="186" text-anchor="middle" font-size="13" class="muted">evaluate or improve</text>

    <line x1="195" y1="155" x2="260" y2="113" class="strong-line" marker-end="url(#mc-arrow-map)"></line>
    <line x1="195" y1="185" x2="260" y2="223" class="strong-line" marker-end="url(#mc-arrow-map)"></line>
    <line x1="420" y1="235" x2="500" y2="235" class="strong-line" marker-end="url(#mc-arrow-map)"></line>
    <line x1="420" y1="105" x2="500" y2="105" class="strong-line" marker-end="url(#mc-arrow-map)"></line>
    <line x1="660" y1="105" x2="730" y2="135" class="strong-line" marker-end="url(#mc-arrow-map)"></line>
    <line x1="660" y1="235" x2="730" y2="205" class="strong-line" marker-end="url(#mc-arrow-map)"></line>

    <text x="340" y="318" text-anchor="middle" font-size="14" class="muted">on-policy: behaviour policy = target policy</text>
    <text x="580" y="318" text-anchor="middle" font-size="14" class="muted">off-policy: behaviour policy differs, so use rho</text>
  </svg>
  <figcaption>The lecture can be read as a sequence of small modifications: average complete returns for prediction, condition on actions for control, enforce exploration for on-policy control, and add importance weights when data comes from a different policy.</figcaption>
</figure>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(Core Takeaways — Monte Carlo)</span></p>

* **Model-free.** MC methods learn directly from sampled episodes; the transition kernel $p(s', r \mid s, a)$ is never consulted.
* **Complete returns, no bootstrapping.** Updates use the realised $G_t$ as the target, not another value estimate.
* **Action-values are required for control.** State-values alone are not enough without a model, because improvement is otherwise impossible.
* **$\varepsilon$-greedy improvement within the $\varepsilon$-soft class is monotone.** Forced exploration does not break policy improvement — the proof shows the $\varepsilon$ term cancels exactly.
* **Off-policy = separate exploration from optimisation.** The behaviour policy $\mu$ collects diverse data; the target policy $\pi$ is the one we evaluate or improve.
* **Importance sampling makes off-policy MC possible.** The ratio $\rho_{t}^{T}$ depends only on $\pi$ and $\mu$ (the environment cancels). **Weighted IS is preferred in practice** because its variance is dramatically smaller than ordinary IS.
* **Off-policy MC control has a fundamental weakness.** In long episodes, only the *tails* consistent with $\pi$ contribute — most data is wasted.

**Bridge to TD.** Monte Carlo waits for the final outcome of each episode before updating. **Temporal-difference methods** in the next lecture learn while the episode is *still unfolding* — they combine the model-free advantage of MC with the online efficiency (and bootstrapping bias) of DP. That combination is the cornerstone of modern RL.

</div>

## Lecture 6: Temporal-Difference Learning

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Problem and Motivation</span><span class="math-callout__name">(Monte Carlo: High Variance and Long)</span></p>

The MC method has very high variance (returns are sums of many random rewards) and is limited to episodic tasks (must unroll to the end of the episode). **Temporal difference** (TD) learning is a more efficient technique that incrementally reduces the Bellman error based on transitions instead of full trajectories.

Given a state transition $(s_t, a_t, r_t, s_{t+1})$ where $a_t \sim \pi(s_t)$, we change the estimate $V(s_t)$ so that it moves towards the **target value** $y_t = r_t + \gamma V(s_{t+1}) \approx G_{t:t+1}$:

$$
V(s_t) \leftarrow V(s_t) + \eta\left[\underbrace{r_t + \gamma V(s_{t+1}) - V(s_t)}_{\delta_t}\right]
$$

The term $\delta_t = y_t - V(s_t)$ is the **TD error**. A more general form for parametric value function representations is:

$$
\boldsymbol{w} \leftarrow \boldsymbol{w} + \eta\left[r_t + \gamma V_{\boldsymbol{w}}(s_{t+1}) - V_{\boldsymbol{w}}(s_t)\right] \nabla_{\boldsymbol{w}} V_{\boldsymbol{w}}(s_t)
$$

The potential divergence arises because the update does not correspond to a gradient update on any objective function, despite having a form similar to SGD. Instead, TD learning is an example of **bootstrapping**: the estimate $V_{\boldsymbol{w}}(s_t)$ is updated to approach a target $r_t + \gamma V_{\boldsymbol{w}}(s_{t+1})$ that is itself defined by the value function estimate. MC estimation avoids this issue by using complete trajectory returns, but is often much less efficient.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Bootstrapping)</span></p>

The potential divergence arises because the update does not correspond to a gradient update on any objective function, despite having a form similar to SGD. Instead, TD learning is an example of **bootstrapping**: the estimate $V_{\boldsymbol{w}}(s_t)$ is updated to approach a target $r_t + \gamma V_{\boldsymbol{w}}(s_{t+1})$ that is itself defined by the value function estimate. This idea is shared by DP methods like value iteration, although they rely on the complete MDP model to compute an exact Bellman backup. In contrast, TD learning can be viewed as using sampled transitions to approximate such backups.

An example of a non-bootstrapping approach is the Monte Carlo estimation in the previous lecture. It samples a complete trajectory, rather than individual transitions, to perform an update; this avoids the divergence issue, but is often much less efficient.

</div>

<figure>
  <img src="{{ '/assets/images/notes/rl_hd/diagram_mc_td0_dp.png' | relative_url }}" alt="Two pairs of 3D surfaces of estimated Blackjack value functions (with and without a usable ace) after 10,000 and 500,000 sampled episodes, becoming visibly smoother as the sample size grows" loading="lazy">
  <figcaption>Backup diagrams of $V(s_t)$ for Monte Carlo, temporal difference, and dynamic programming updates of the state-value function.</figcaption>
</figure>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Setup</span><span class="math-callout__name">(Sampling experience, bootstrapping predictions)</span></p>

The previous lecture left us with a clean but expensive idea: estimate values by averaging **complete returns** $G\_t$, which forces us to wait until the episode ends. **Temporal-difference (TD) learning** is the hybrid that removes that wait. It is *the* central idea of model-free RL, and almost every algorithm in the rest of the course is best read as "define a TD error, then take a step on it."

The lecture builds up in the following order:

1. **The central idea.** TD = *sampling* (like Monte Carlo) + *bootstrapping* (like Dynamic Programming).
2. **Prediction.** The TD(0) update, the TD error, and where the TD target comes from (the Bellman expectation equation).
3. **Why bootstrap?** Computational advantages over DP and MC, soundness (convergence), and what TD converges to (certainty equivalence).
4. **Control.** Moving from $v\_\pi$ to $q\_\pi$: 
   * **Sarsa** (on-policy), 
   * **Q-learning** (off-policy),
   * **Double Q-learning** (bias correction).

The guiding question of the lecture is:

> *Can we update our value estimates using our **own current estimates** as targets? If yes, why does it work, and when does it work better than waiting for actual returns?*

</div>

### The Big Picture

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Idea</span><span class="math-callout__name">(TD is sampling **and** bootstrapping)</span></p>

Temporal-difference learning combines two ideas we have already met separately:

* **Monte Carlo idea:** learn from *raw experience*, with no model of the environment needed.
* **DP idea:** *bootstrap* — update an estimate toward a target that is itself built from other estimates.

The behavioural contrast is the cleanest way to remember it:

* **MC** uses the **complete return** $G\_t$ as its target — and must wait until the episode ends to know it.
* **TD** uses the **one-step prediction** $R\_{t+1} + \gamma V(S\_{t+1})$ as its target — and can update *immediately*, after a single transition.

In one phrase: **TD learning is learning a prediction from another prediction.**

</div>

<figure class="rl-diagram">
  <svg viewBox="0 0 860 300" role="img" aria-label="Temporal-difference learning combines one sampled transition with bootstrapping from the next state's current value estimate">
    <defs>
      <marker id="td-hybrid-arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
        <path d="M0,0 L0,6 L9,3 z" fill="#64748b"></path>
      </marker>
      <marker id="td-hybrid-arrow-strong" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
        <path d="M0,0 L0,6 L9,3 z" fill="#2c3e94"></path>
      </marker>
    </defs>

    <rect x="35" y="70" width="190" height="105" rx="8" class="green"></rect>
    <text x="130" y="107" text-anchor="middle" font-size="17" font-weight="700">Sampling</text>
    <text x="130" y="133" text-anchor="middle" font-size="13" class="muted">observe one transition</text>
    <text x="130" y="155" text-anchor="middle" font-size="13">S_t, R_{t+1}, S_{t+1}</text>

    <rect x="35" y="200" width="190" height="62" rx="8" class="accent"></rect>
    <text x="130" y="228" text-anchor="middle" font-size="17" font-weight="700">Bootstrapping</text>
    <text x="130" y="250" text-anchor="middle" font-size="13" class="muted">reuse current next-state guess</text>

    <circle cx="355" cy="122" r="30" class="box"></circle>
    <text x="355" y="128" text-anchor="middle" font-size="16" font-weight="700">S_t</text>
    <circle cx="525" cy="122" r="30" class="box"></circle>
    <text x="525" y="128" text-anchor="middle" font-size="16" font-weight="700">S_{t+1}</text>
    <path d="M386 122 L494 122" class="strong-line" marker-end="url(#td-hybrid-arrow-strong)"></path>
    <rect x="424" y="80" width="60" height="27" rx="5" class="amber"></rect>
    <text x="454" y="99" text-anchor="middle" font-size="12">R_{t+1}</text>

    <rect x="445" y="198" width="160" height="52" rx="8" class="accent"></rect>
    <text x="525" y="229" text-anchor="middle" font-size="15" font-weight="700">V(S_{t+1})</text>
    <path d="M525 153 L525 197" class="line" marker-end="url(#td-hybrid-arrow)"></path>

    <rect x="665" y="110" width="160" height="94" rx="8" class="green"></rect>
    <text x="745" y="142" text-anchor="middle" font-size="15" font-weight="700">TD target</text>
    <text x="745" y="169" text-anchor="middle" font-size="14">R_{t+1} + gamma</text>
    <text x="745" y="190" text-anchor="middle" font-size="14">V(S_{t+1})</text>

    <path d="M555 122 C600 122 615 132 660 145" class="strong-line" marker-end="url(#td-hybrid-arrow-strong)"></path>
    <path d="M605 224 C630 218 645 194 664 177" class="line" marker-end="url(#td-hybrid-arrow)"></path>
    <path d="M225 122 L324 122" class="line" marker-end="url(#td-hybrid-arrow)"></path>
    <path d="M225 231 C300 231 350 225 444 225" class="line" marker-end="url(#td-hybrid-arrow)"></path>
  </svg>
  <figcaption>TD(0) builds its target from two ingredients at once: a <em>sampled</em> reward and successor state, plus a <em>bootstrapped</em> current estimate of the successor value. This is the concrete meaning of "sampling + bootstrapping".</figcaption>
</figure>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(The DP / MC / TD taxonomy)</span></p>

All three methods estimate the *same object* — the value 

$$v_\pi(s) = \mathbb{E}_\pi[\, G_t \mid S_t = s\,].$$

They differ only in **how the expectation is approximated**.

| Method | Source of estimate | Backup type |
| :----- | :----------------- | :---------- |
| **DP** | model $+$ bootstrap | full expectation backup |
| **MC** | samples, no bootstrap | complete return |
| **TD** | samples $+$ bootstrap | one-step sample backup |

* **DP** needs a model and averages over *all* successors; 
* **MC** needs samples but no model and uses the *whole* trajectory; 
* **TD** needs only samples and replaces both the full expectation (by one sample) and the unknown true value (by the current estimate).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(DP does not approximate, it computes the value function **exactly**)</span></p>

Dynamic Programming (DP) computes the value function **exactly**, rather than approximating it through sampled experience like Monte Carlo (MC) methods do.

Here is exactly how DP handles these computations:

* **Perfect Knowledge of the Model:** DP operates under the assumption that you have a perfect model of the environment, specifically the complete one-step transition dynamics $p(s', r \mid s, a)$.
* **Exact Expectations:** Because this transition kernel is fully known, every expectation in a Bellman backup—whether it is over actions, next states, or rewards—can be calculated exactly as a finite sum. There is absolutely no sampling involved.
* **Contrast with MC:** Monte Carlo methods are entirely "model-free". Since they do not have access to the environment's transition model, they must estimate the value function by averaging the actual complete returns gathered from sampled trajectories.

In short, DP acts as a planner using full, exact mathematical expectations, whereas MC acts as a learner estimating values through the empirical averages of real, unfolding experiences.

</div>

### Prediction: TD(0) vs MC

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Setup</span><span class="math-callout__name">(The prediction problem)</span></p>

Fix a policy $\pi$ and generate experience under it,

$$S_0, A_0, R_1, S_1, A_1, R_2, S_2, \dots,$$

with the goal of estimating $v\_\pi(s)$ for every nonterminal $s$. Both methods are instances of the same incremental-mean template 

$$V(S_t) \leftarrow V(S_t) + \alpha\,[\,\text{target} - V(S_t)\,]$$

they differ only in the **target**:

$$\underbrace{V(S_t) \;\leftarrow\; V(S_t) + \alpha\bigl[\,G_t - V(S_t)\,\bigr]}_{\textbf{constant-}\alpha\text{ Monte Carlo, target } = \text{ full return}},$$

$$\underbrace{V(S_t) \;\leftarrow\; V(S_t) + \alpha\bigl[\,R_{t+1} + \gamma V(S_{t+1}) - V(S_t)\,\bigr]}_{\textbf{TD(0)},\; \text{target } = R_{t+1} + \gamma V(S_{t+1})}.$$

* MC plugs in the realised return $G\_t$ (known only at episode end); 
* TD(0) plugs in the **TD target** $R\_{t+1} + \gamma V(S\_{t+1})$ (known one step later).

This is the single substitution that turns a batch, episodic method into an online, incremental one.

</div>

<figure class="rl-diagram">
  <svg viewBox="0 0 860 330" role="img" aria-label="Monte Carlo waits until terminal return while TD updates after one transition">
    <defs>
      <marker id="td-timing-arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
        <path d="M0,0 L0,6 L9,3 z" fill="#64748b"></path>
      </marker>
      <marker id="td-timing-strong" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
        <path d="M0,0 L0,6 L9,3 z" fill="#2c3e94"></path>
      </marker>
    </defs>

    <text x="80" y="78" text-anchor="middle" font-size="17" font-weight="700">MC</text>
    <text x="80" y="218" text-anchor="middle" font-size="17" font-weight="700">TD(0)</text>

    <line x1="145" y1="75" x2="725" y2="75" class="line"></line>
    <circle cx="170" cy="75" r="18" class="box"></circle>
    <text x="170" y="81" text-anchor="middle" font-size="13">S_t</text>
    <circle cx="310" cy="75" r="18" class="box"></circle>
    <text x="310" y="81" text-anchor="middle" font-size="13">S_{t+1}</text>
    <circle cx="450" cy="75" r="18" class="box"></circle>
    <text x="450" y="81" text-anchor="middle" font-size="13">S_{t+2}</text>
    <circle cx="590" cy="75" r="18" class="box"></circle>
    <text x="590" y="81" text-anchor="middle" font-size="13">...</text>
    <rect x="702" y="57" width="46" height="36" rx="5" fill="#64748b"></rect>
    <text x="725" y="81" text-anchor="middle" font-size="13" font-weight="700" fill="#ffffff">T</text>
    <path d="M170 112 C300 152 575 152 725 112" stroke="#b45309" stroke-width="3" fill="none" marker-end="url(#td-timing-arrow)"></path>
    <text x="445" y="166" text-anchor="middle" font-size="13" class="muted">wait until terminal return G_t is known</text>
    <rect x="632" y="112" width="150" height="36" rx="6" class="amber"></rect>
    <text x="707" y="135" text-anchor="middle" font-size="13" font-weight="700">update after episode</text>

    <line x1="145" y1="215" x2="725" y2="215" class="line"></line>
    <circle cx="170" cy="215" r="18" class="box"></circle>
    <text x="170" y="221" text-anchor="middle" font-size="13">S_t</text>
    <circle cx="310" cy="215" r="18" class="accent"></circle>
    <text x="310" y="221" text-anchor="middle" font-size="13">S_{t+1}</text>
    <circle cx="450" cy="215" r="18" class="box"></circle>
    <text x="450" y="221" text-anchor="middle" font-size="13">S_{t+2}</text>
    <circle cx="590" cy="215" r="18" class="box"></circle>
    <text x="590" y="221" text-anchor="middle" font-size="13">...</text>
    <rect x="702" y="197" width="46" height="36" rx="5" fill="#64748b"></rect>
    <text x="725" y="221" text-anchor="middle" font-size="13" font-weight="700" fill="#ffffff">T</text>
    <path d="M190 215 L288 215" class="strong-line" marker-end="url(#td-timing-strong)"></path>
    <rect x="216" y="158" width="188" height="40" rx="6" class="green"></rect>
    <text x="310" y="183" text-anchor="middle" font-size="13" font-weight="700">update after one step</text>
    <path d="M310 198 L310 235" class="line"></path>
    <text x="310" y="267" text-anchor="middle" font-size="13" class="muted">target: R_{t+1} + gamma V(S_{t+1})</text>
  </svg>
  <figcaption>Monte Carlo must wait for the realised return $G_t$. TD(0) only waits for the next transition, then updates $V(S_t)$ toward $R_{t+1} + \gamma V(S_{t+1})$ while the episode is still unfolding.</figcaption>
</figure>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(One-step TD error)</span></p>

The **TD error** is the difference between the old prediction and the (improved) one-step target:

$$\delta_t \;\doteq\; R_{t+1} + \gamma V(S_{t+1}) - V(S_t).$$

With it, the TD(0) update is simply

$$V(S_t) \;\leftarrow\; V(S_t) + \alpha\,\delta_t.$$

So $\delta\_t$ measures the gap between the **old prediction** $V(S\_t)$ and the **updated prediction** $R\_{t+1} + \gamma V(S\_{t+1})$ — the difference between two successive predictions of the same quantity. Hence the name *temporal difference*.

</div>

<figure class="rl-diagram">
  <svg viewBox="0 0 760 250" role="img" aria-label="TD error as the gap between old value estimate and TD target">
    <defs>
      <marker id="td-error-arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
        <path d="M0,0 L0,6 L9,3 z" fill="#2c3e94"></path>
      </marker>
    </defs>

    <line x1="95" y1="195" x2="665" y2="195" class="line"></line>
    <text x="95" y="222" text-anchor="middle" font-size="13" class="muted">low value</text>
    <text x="665" y="222" text-anchor="middle" font-size="13" class="muted">high value</text>

    <rect x="160" y="134" width="150" height="52" rx="8" class="box"></rect>
    <text x="235" y="157" text-anchor="middle" font-size="14" font-weight="700">old estimate</text>
    <text x="235" y="176" text-anchor="middle" font-size="13">V(S_t)</text>

    <rect x="455" y="58" width="190" height="62" rx="8" class="green"></rect>
    <text x="550" y="84" text-anchor="middle" font-size="14" font-weight="700">one-step target</text>
    <text x="550" y="105" text-anchor="middle" font-size="13">R_{t+1} + gamma V(S_{t+1})</text>

    <path d="M315 160 C365 132 406 105 451 94" class="strong-line" marker-end="url(#td-error-arrow)"></path>
    <text x="378" y="113" text-anchor="middle" font-size="16" font-weight="700">delta_t</text>

    <circle cx="235" cy="195" r="7" fill="#64748b"></circle>
    <circle cx="550" cy="195" r="7" fill="#047857"></circle>
    <circle cx="329" cy="195" r="7" fill="#2c3e94"></circle>
    <path d="M235 195 L329 195" class="strong-line" marker-end="url(#td-error-arrow)"></path>
    <text x="282" y="184" text-anchor="middle" font-size="13" class="muted">alpha delta_t</text>
    <text x="329" y="222" text-anchor="middle" font-size="13" font-weight="700">new V(S_t)</text>
  </svg>
  <figcaption>The TD error is the signed gap from the old estimate to the one-step target. The update does not jump all the way to the target unless $\alpha = 1$; it moves partway by $\alpha\,\delta_t$.</figcaption>
</figure>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why the TD error is the central object)</span></p>

A large fraction of the algorithms still ahead — Sarsa, Q-learning, actor-critic, and the deep-RL methods built on them — are best read in one line:

> *define a TD error, then take a (stochastic-gradient) step on it.*

Everything that changes from method to method is **what goes into the target** inside $\delta\_t$: a sampled action value for Sarsa, a maximised action value for Q-learning, a decoupled pair of tables for Double Q-learning. Getting comfortable with $\delta\_t = (\text{target}) - (\text{old estimate})$ now pays off repeatedly.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Monte Carlo Error as Accumulated TD Error)</span></p>

Consider an episode

$$S_0,R_1,S_1,R_2,\ldots,R_T,S_T$$

with $V(S_T)=0$. Show that the sum of discounted TD errors along the episode telescopes into the MC error:

$$G_0-V(S_0),$$

where

$$\sum_{k=0}^{T-1}\gamma^kR_{k+1}$$

is the full return, assuming $V$ does not change during the episode.

Interpret this result: what does it say about the relationship between the TD and MC updates?

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

Assume that $V$ is held fixed during the episode. For $t=0,\ldots,T-1$,

$$\delta_t = R_{t+1}+\gamma V(S_{t+1})-V(S_t).$$

Multiplying by $\gamma^t$ and summing gives

$$
\begin{aligned}
\sum_{t=0}^{T-1}\gamma^t\delta_t
&= \sum_{t=0}^{T-1}\gamma^t R_{t+1} + \sum_{t=0}^{T-1}\gamma^{t+1}V(S_{t+1}) - \sum_{t=0}^{T-1}\gamma^t V(S_t).
\end{aligned}
$$

The first term is the Monte Carlo return

$$G_0=\sum_{t=0}^{T-1}\gamma^tR_{t+1}.$$

For the value terms, reindex the middle sum:

$$\sum_{t=0}^{T-1}\gamma^{t+1}V(S_{t+1}) = \sum_{j=1}^{T}\gamma^jV(S_j).$$

Therefore

$$
\begin{aligned}
\sum_{t=0}^{T-1}\gamma^t\delta_t
&= G_0 + \sum_{j=1}^{T}\gamma^jV(S_j) - \sum_{j=0}^{T-1}\gamma^jV(S_j) \\
&= G_0 - V(S_0)+\gamma^T V(S_T).
\end{aligned}
$$

Since $V(S_T)=0$, this becomes

$$\sum_{t=0}^{T-1}\gamma^t\delta_t = G_0-V(S_0).$$

Thus the Monte Carlo error $G_0-V(S_0)$ is exactly the discounted sum of all one-step TD errors along the episode. MC waits until the complete return is available and then updates using the full error. TD breaks the same total error into local one-step prediction errors and can therefore update immediately after each transition.

</details>
</div>

### Where the TD Target Comes From

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Derivation</span><span class="math-callout__name">(The TD target is the Bellman expectation equation)</span></p>

The TD target is not an ad-hoc guess; it falls straight out of the definition of value. Start from

$$v_\pi(s) = \mathbb{E}_\pi[\, G_t \mid S_t = s\,],$$

and use the one-step recursion of the return,

$$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} = R_{t+1} + \gamma \sum_{k=0}^{\infty} \gamma^k R_{t+k+2} = R_{t+1} + \gamma\, G_{t+1}.$$

Substituting and applying the tower property of conditional expectation,

$$
\begin{aligned}
v_\pi(s) &= \mathbb{E}_\pi\bigl[\,R_{t+1} + \gamma G_{t+1} \mid S_t = s\,\bigr] \\
&= \mathbb{E}_\pi\bigl[\,R_{t+1} + \gamma\, \mathbb{E}_\pi[\,G_{t+1} \mid S_{t+1}\,] \,\big|\, S_t = s\,\bigr] \\
&= \mathbb{E}_\pi\bigl[\,R_{t+1} + \gamma\, v_\pi(S_{t+1}) \,\big|\, S_t = s\,\bigr].
\end{aligned}
$$

This is exactly the **Bellman expectation equation**. The TD target is what you get by turning this *exact identity* into something estimable with two approximations.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(The TD recipe — two approximations)</span></p>

Starting from the exact identity 

$$v_\pi(s) = \mathbb{E}_\pi[\,R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s\,],$$

1. **Replace the expectation by a single sample** — use the one transition $(R\_{t+1}, S\_{t+1})$ that actually occurred instead of averaging over all of them (this is the *sampling* part, shared with MC).
2. **Replace the unknown $v\_\pi$ by the current estimate $V$** — we do not know the true value of the successor, so we bootstrap from our running guess (this is the *bootstrapping* part, shared with DP).

$$\underbrace{R_{t+1} + \gamma\, v_\pi(S_{t+1})}_{\text{Bellman target (exact)}} \;\rightsquigarrow\; \underbrace{R_{t+1} + \gamma\, V(S_{t+1})}_{\text{TD(0) target (sampled + bootstrapped)}}.$$

This makes precise where each method sits relative to the same Bellman equation:

* **MC** samples the *whole* return $G\_t$ — no bootstrap.
* **DP** computes the expectation *exactly* via the model — but bootstraps with $V$.
* **TD** samples $(R\_{t+1}, S\_{t+1})$ *and* bootstraps with $V(S\_{t+1})$ — it does both approximations at once.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(From Bellman Expectation to the TD Target)</span></p>

Start from the Bellman expectation equation

$$\mathbb{E}\pi \left[ R{t+1}+\gamma v_\pi(S_{t+1}) \mid S_t=s \right]$$

and explain the two approximations that turn this identity into the TD(0) target

$$R_{t+1}+\gamma V(S_{t+1}).$$

Why is the TD(0) target a biased estimator of

$$\sum_{k=0}^{\infty}\gamma^kR_{t+k+1},$$

the true return, yet an unbiased estimator of

$$R_{t+1}+\gamma v_\pi(S_{t+1})$$

when conditioned on $S_t$?

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

$$v_\pi(s) = \mathbb E_\pi\! \left[R_{t+1}+\gamma v_\pi(S_{t+1}) \,\middle|\, S_t=s \right].$$

TD(0) turns this into the one-step target

$$R_{t+1}+\gamma V(S_{t+1})$$

by two approximations:

* **Sampling:** replace the conditional expectation by the observed transition $(S_t,R_{t+1},S_{t+1})$.
* **Bootstrapping:** replace the unknown value function $v_\pi$ by the current estimate $V$.

Thus the TD(0) target is

$$Y_t(V):=R_{t+1}+\gamma V(S_{t+1}).$$

The target is usually biased as an estimator of the full return

$$G_t=\sum_{k=0}^\infty \gamma^k R_{t+k+1},$$

because it replaces the random future return $G_{t+1}$ by the current prediction $V(S_{t+1})$:

$$\mathbb E_\pi[Y_t(V)\mid S_t=s] = \mathbb E_\pi[ R_{t+1}+\gamma V(S_{t+1})\mid S_t=s] = (T_\pi V)(s),$$

which need not equal

$$v_\pi(s)=\mathbb E_\pi[G_t\mid S_t=s].$$

The bias disappears when $V=v_\pi$.

On the other hand, with the true value function inserted, the one-step Bellman sample

$$R_{t+1}+\gamma v_\pi(S_{t+1})$$

is unbiased for its conditional expectation:

$$\mathbb E_\pi\!\left[ R_{t+1}+\gamma v_\pi(S_{t+1}) \,\middle|\, S_t=s \right] =v_\pi(s).$$

So the sampling step is unbiased; the possible bias comes from replacing $v_\pi$ by $V$.

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Tabular TD(0) for estimating $v_\pi$)</span></p>

* **Input:** policy $\pi$ to be evaluated.
* **Parameter:** step size $\alpha \in (0, 1]$.

Initialise $V(s)$ arbitrarily for all $s \in \mathcal{S}^+$, with $V(\text{terminal}) = 0$.

Loop for each episode:

1. Initialise $S$.
2. Loop for each step of the episode, until $S$ is terminal:
   * $A \leftarrow$ action given by $\pi$ for $S$.
   * Take action $A$.
   * Observe reward $R$ and next state $S'$.
   * $V(S) \leftarrow V(S) + \alpha\bigl[\,R + \gamma V(S') - V(S)\,\bigr]$.
   * $S \leftarrow S'$.

Updates happen **online**, after every transition — which makes TD(0) well suited to very long episodes and to continuing (non-terminating) tasks where MC simply cannot wait for a return.

</div>

### Backup Diagrams

<figure class="rl-diagram">
  <svg viewBox="0 0 760 330" role="img" aria-label="Backup diagrams for DP, TD(0), and Monte Carlo">
    <text x="150" y="36" text-anchor="middle" font-size="17" font-weight="700">DP</text>
    <text x="380" y="36" text-anchor="middle" font-size="17" font-weight="700">TD(0)</text>
    <text x="610" y="36" text-anchor="middle" font-size="17" font-weight="700">MC</text>

    <!-- DP: full expectation backup -->
    <line x1="150" y1="70" x2="95" y2="130" class="line"></line>
    <line x1="150" y1="70" x2="150" y2="130" class="line"></line>
    <line x1="150" y1="70" x2="205" y2="130" class="line"></line>
    <line x1="95" y1="140" x2="70" y2="200" class="line"></line>
    <line x1="95" y1="140" x2="120" y2="200" class="line"></line>
    <line x1="150" y1="140" x2="150" y2="200" class="line"></line>
    <line x1="205" y1="140" x2="180" y2="200" class="line"></line>
    <line x1="205" y1="140" x2="230" y2="200" class="line"></line>
    <circle cx="150" cy="70" r="11" class="box"></circle>
    <circle cx="95" cy="135" r="7" fill="#2c3e94"></circle>
    <circle cx="150" cy="135" r="7" fill="#2c3e94"></circle>
    <circle cx="205" cy="135" r="7" fill="#2c3e94"></circle>
    <circle cx="70" cy="205" r="11" class="box"></circle>
    <circle cx="120" cy="205" r="11" class="box"></circle>
    <circle cx="150" cy="205" r="11" class="box"></circle>
    <circle cx="180" cy="205" r="11" class="box"></circle>
    <circle cx="230" cy="205" r="11" class="box"></circle>
    <text x="150" y="245" text-anchor="middle" font-size="13" class="muted">full expectation</text>

    <!-- TD(0): one sample step -->
    <line x1="380" y1="70" x2="380" y2="130" class="strong-line"></line>
    <line x1="380" y1="140" x2="380" y2="200" class="strong-line"></line>
    <circle cx="380" cy="70" r="11" class="box"></circle>
    <circle cx="380" cy="135" r="7" fill="#2c3e94"></circle>
    <circle cx="380" cy="205" r="11" class="box"></circle>
    <text x="380" y="245" text-anchor="middle" font-size="13" class="muted">one sample step</text>

    <!-- MC: full sample trajectory -->
    <line x1="610" y1="70" x2="610" y2="270" class="strong-line"></line>
    <circle cx="610" cy="70" r="11" class="box"></circle>
    <circle cx="610" cy="110" r="7" fill="#2c3e94"></circle>
    <circle cx="610" cy="150" r="11" class="box"></circle>
    <circle cx="610" cy="190" r="7" fill="#2c3e94"></circle>
    <circle cx="610" cy="230" r="11" class="box"></circle>
    <rect x="600" y="262" width="20" height="20" fill="#64748b"></rect>
    <text x="610" y="305" text-anchor="middle" font-size="13" class="muted">full sample trajectory</text>
  </svg>
  <figcaption>Open circles are states, filled circles are state–action pairs, the filled square is a terminal state. DP fans out to <em>all</em> successors (a full backup); TD samples <em>one</em> successor (a shallow sample backup); MC samples a <em>whole</em> trajectory to termination (a deep sample backup).</figcaption>
</figure>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Two axes: depth and width)</span></p>

The three diagrams differ along two independent axes:

* **Depth of the backup.** TD is *shallow* (one step), MC is *deep* (to the end of the episode). This is the axis later unified by $n$-step methods and TD($\lambda$).
* **Width of the backup.** DP is *wide* (it sweeps the full successor distribution), while MC and TD are *narrow* (a single sampled branch). This is the axis that separates model-based from model-free.

TD(0) is the corner that is both shallow and narrow — and therefore the cheapest possible nontrivial backup.

</div>

### The Driving-Home Example

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Predicting time to drive home)</span></p>

A canonical intuition pump. The **state** is the situation along the route (leaving office, in the car, exiting the highway, …); the **reward** is the minutes elapsed on each leg ($\gamma = 1$); the **value** $V(s)$ is the *expected remaining* travel time; the **return** is the *actual* remaining time; and the **predicted total** is elapsed-so-far $+\, V(s)$.

| state | elapsed | predicted total | how we get $V(s)$ |
| :---- | :------ | :-------------- | :----------------- |
| leaving office | 0 | 30 | $30 - 0 = 30$ |
| reach car (raining) | 5 | 40 | $40 - 5 = 35$ |
| exiting highway | 20 | 35 | $35 - 20 = 15$ |
| 2ndary road, truck | 30 | 40 | $40 - 30 = 10$ |
| home street | 40 | 43 | $43 - 40 = 3$ |
| arrive home | 43 | 43 | $43 - 43 = 0$ |

The actual trip took 43 minutes. The question — *when should we revise each prediction?* — is exactly the MC-vs-TD question.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(MC vs TD updates on the road)</span></p>

Both methods apply 

$$V(s) \leftarrow V(s) + \alpha\,(\text{target} - V(s)),$$

but with different targets:

$$\text{MC target} = \underbrace{G_t}_{\text{total time } - \text{ elapsed}}, \qquad \text{TD target} = \underbrace{R_{t+1}}_{\text{next elapsed } - \text{ current elapsed}} + V(S_{t+1}).$$

| state | old $V$ | MC target | MC error | TD target | TD error |
| :---- | :------ | :-------- | :------- | :-------- | :------- |
| leaving office | 30 | $43$ | $+13$ | $5 + 35 = 40$ | $+10$ |
| reach car | 35 | $38$ | $+3$ | $15 + 15 = 30$ | $-5$ |
| exit highway | 15 | $23$ | $+8$ | $10 + 10 = 20$ | $+5$ |
| 2ndary road | 10 | $13$ | $+3$ | $10 + 3 = 13$ | $+3$ |
| home street | 3 | $3$ | $0$ | $3 + 0 = 3$ | $0$ |
| arrive home | 0 | $0$ | $0$ | — | — |

**The key visual.** MC updates *every* prediction toward the *final* outcome (43 min) — so a single surprise late in the trip is only felt at the end. TD updates *each* prediction toward the *next* prediction — so the "stuck behind a truck" surprise propagates backward to the immediately preceding state right away, without waiting to arrive home.

</div>

<figure class="rl-diagram">
  <svg viewBox="0 0 860 360" role="img" aria-label="Driving home example comparing Monte Carlo updates to final outcome with TD updates to next prediction">
    <defs>
      <marker id="td-drive-arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
        <path d="M0,0 L0,6 L9,3 z" fill="#64748b"></path>
      </marker>
      <marker id="td-drive-strong" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
        <path d="M0,0 L0,6 L9,3 z" fill="#2c3e94"></path>
      </marker>
    </defs>

    <text x="78" y="92" text-anchor="middle" font-size="16" font-weight="700">MC</text>
    <text x="78" y="245" text-anchor="middle" font-size="16" font-weight="700">TD</text>

    <g>
      <circle cx="160" cy="90" r="20" class="box"></circle>
      <text x="160" y="95" text-anchor="middle" font-size="12">office</text>
      <circle cx="285" cy="90" r="20" class="box"></circle>
      <text x="285" y="95" text-anchor="middle" font-size="12">car</text>
      <circle cx="410" cy="90" r="20" class="box"></circle>
      <text x="410" y="95" text-anchor="middle" font-size="12">exit</text>
      <circle cx="535" cy="90" r="20" class="box"></circle>
      <text x="535" y="95" text-anchor="middle" font-size="12">truck</text>
      <circle cx="660" cy="90" r="20" class="box"></circle>
      <text x="660" y="95" text-anchor="middle" font-size="12">street</text>
      <rect x="750" y="70" width="58" height="40" rx="6" fill="#047857"></rect>
      <text x="779" y="95" text-anchor="middle" font-size="13" font-weight="700" fill="#ffffff">43</text>

      <path d="M160 118 C285 175 650 175 779 114" class="line" marker-end="url(#td-drive-arrow)"></path>
      <path d="M285 116 C390 155 660 158 779 114" class="line" marker-end="url(#td-drive-arrow)"></path>
      <path d="M410 115 C488 140 665 142 779 114" class="line" marker-end="url(#td-drive-arrow)"></path>
      <path d="M535 114 C600 128 690 128 779 114" class="line" marker-end="url(#td-drive-arrow)"></path>
      <path d="M660 110 C694 112 727 112 750 98" class="line" marker-end="url(#td-drive-arrow)"></path>
      <text x="470" y="188" text-anchor="middle" font-size="13" class="muted">all states update toward the final realised trip time</text>
    </g>

    <g>
      <circle cx="160" cy="245" r="20" class="box"></circle>
      <text x="160" y="250" text-anchor="middle" font-size="12">office</text>
      <circle cx="285" cy="245" r="20" class="box"></circle>
      <text x="285" y="250" text-anchor="middle" font-size="12">car</text>
      <circle cx="410" cy="245" r="20" class="box"></circle>
      <text x="410" y="250" text-anchor="middle" font-size="12">exit</text>
      <circle cx="535" cy="245" r="20" class="amber"></circle>
      <text x="535" y="250" text-anchor="middle" font-size="12">truck</text>
      <circle cx="660" cy="245" r="20" class="box"></circle>
      <text x="660" y="250" text-anchor="middle" font-size="12">street</text>
      <rect x="750" y="225" width="58" height="40" rx="6" fill="#047857"></rect>
      <text x="779" y="250" text-anchor="middle" font-size="13" font-weight="700" fill="#ffffff">home</text>

      <path d="M181 245 L263 245" class="strong-line" marker-end="url(#td-drive-strong)"></path>
      <path d="M306 245 L388 245" class="strong-line" marker-end="url(#td-drive-strong)"></path>
      <path d="M431 245 L513 245" class="strong-line" marker-end="url(#td-drive-strong)"></path>
      <path d="M556 245 L638 245" class="strong-line" marker-end="url(#td-drive-strong)"></path>
      <path d="M681 245 L748 245" class="strong-line" marker-end="url(#td-drive-strong)"></path>
      <text x="470" y="307" text-anchor="middle" font-size="13" class="muted">each state updates toward the next prediction immediately</text>
      <text x="535" y="214" text-anchor="middle" font-size="13" font-weight="700" fill="#b45309">surprise enters here</text>
    </g>
  </svg>
  <figcaption>In MC, the correction for every earlier prediction is delayed until the trip outcome is known. In TD, each leg passes its surprise one step backward through the chain of predictions.</figcaption>
</figure>

### Why Bootstrap?

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(Three questions about "a guess from a guess")</span></p>

The TD target $R\_{t+1} + \gamma V(S\_{t+1})$ uses an *estimate* $V(S\_{t+1})$, not the truth $v\_\pi(S\_{t+1})$ — we are quite literally **learning a guess from a guess**. Three natural questions organise the rest of the prediction story:

1. Why is bootstrapping **useful** computationally? *(advantages over DP and MC)*
2. Is it **correct** — does TD converge? *(soundness)*
3. With limited data, who is **faster**, TD or MC? *(efficiency)*

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Advantage over DP — no model required)</span></p>

DP requires the **full model** $p(s', r \mid s, a)$ to compute its expected backup. TD requires only **sampled transitions** $(S\_t, R\_{t+1}, S\_{t+1})$. So TD works in *any* environment we can *interact* with, even when we cannot *model* it.

The mechanism is the **sample backup**: TD replaces DP's full-expectation backup by a single observed transition that stands in for the whole expectation. One sampled successor does the job of the entire $\sum\_{s', r} p(s', r \mid s, a)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Advantage over MC — online and incremental)</span></p>

**MC** must wait for the **end of the episode** to know $G\_t$; **TD** updates after a *single* transition using $(S\_t, R\_{t+1}, S\_{t+1})$. This matters because:

* some applications have **very long episodes**, where end-of-episode updates are far too slow;
* some are **continuing tasks** with *no* terminal state at all, so $G\_t$ is never observed;
* some MC variants must **discount or discard** episodes containing exploratory actions, whereas TD just uses every transition.

In short, **every transition is a learning opportunity** — TD is the natural fit for streaming, online data.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(TD(0) prediction convergence)</span></p>

For a fixed policy $\pi$, in the **tabular** case:

* with a **small constant** $\alpha$, TD(0) converges *in the mean* to $v\_\pi$;
* with a **diminishing** $\alpha\_t$ satisfying the standard stochastic-approximation conditions $\sum\_t \alpha\_t = \infty$ and $\sum\_t \alpha\_t^2 < \infty$, TD(0) converges to $v\_\pi$ *with probability 1*.

So bootstrapping is a **convergent** prediction method.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(The True Value Function as an Expected TD Fixed Point)</span></p>

Show that if $V=v_\pi$, the true value function, then the TD error has zero mean conditioned on $S_t$:

$$\mathbb{E}_\pi[\delta_t\mid S_t=s]=0 \qquad \text{for all }s.$$

Use this to argue informally why $V=v_\pi$ is a fixed point of the expected TD(0) updates.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

Assume $V=v_\pi$. Then the TD error is

$$\delta_t = R_{t+1}+\gamma v_\pi(S_{t+1})-v_\pi(S_t).$$

Conditioning on $S_t=s$, we obtain

$$
\begin{aligned}
\mathbb E_\pi[\delta_t\mid S_t=s]
&= \mathbb E_\pi\! \left[ R_{t+1}+\gamma v_\pi(S_{t+1})-v_\pi(S_t) \,\middle|\, S_t=s \right] \\
&= \mathbb E_\pi\! \left[ R_{t+1}+\gamma v_\pi(S_{t+1}) \,\middle|\, S_t=s \right] -v_\pi(s).
\end{aligned}
$$

By the Bellman expectation equation,

$$\mathbb E_\pi\! \left[ R_{t+1}+\gamma v_\pi(S_{t+1}) \,\middle|\, S_t=s \right] = v_\pi(s).$$

Hence

$$\mathbb E_\pi[\delta_t\mid S_t=s]=0 \qquad \forall s\in \mathcal{S}.$$

The TD(0) update is

$$V(S_t)\leftarrow V(S_t)+\alpha\delta_t.$$

If $V=v_\pi$, then the expected update at every state is zero:

$$\mathbb E_\pi\! \left[ V_{\mathrm{new}}(s)-V(s) \,\middle|\, S_t=s \right] = \alpha\mathbb E_\pi[\delta_t\mid S_t=s] =0.$$

Thus $v_\pi$ is a fixed point of the expected TD(0) update.

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Who learns faster?)</span></p>

Both MC and TD converge asymptotically to $v\_\pi$ for a fixed policy, and **no general theorem says one is always faster**. Even *defining* "faster" rigorously is subtle — it tangles together bias, variance, step size, and non-stationarity. But **empirically**, on stochastic problems, TD often reduces error faster than constant-$\alpha$ MC. 

**The intuition:**
* the TD target has **lower variance** (it depends on one random transition, not a whole random trajectory), 
* at the cost of some **bias** (it depends on the current estimate $V$).

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>The price for lower variance is primarily bias from bootstrapping.</summary>

The price is primarily **bias from bootstrapping**.

For Monte Carlo,

$$G_t$$

is a noisy but unbiased target:

$$\mathbb E[G_t\mid S_t=s]=v_\pi(s).$$

For TD(0), the target is

$$Y_t=R_{t+1}+\gamma \hat v(S_{t+1}).$$

Its expectation is

$$\mathbb E[Y_t\mid S_t=s] = (T^\pi \hat v)(s),$$

which equals $v_\pi(s)$ only when $\hat v=v_\pi$. Thus, during learning, TD updates toward a target containing its **own current error**.

**So the trade-off is**

$$\boxed{\text{MC: high variance, unbiased targets}} \qquad\text{vs.}\qquad \boxed{\text{TD: lower variance, biased targets}}.$$

This creates several practical costs.

**Errors propagate through bootstrapping.**
If $\hat v(S_{t+1})$ is wrong, that error is inserted into the estimate of $S_t$. TD can therefore propagate incorrect values backward before observing the true eventual outcome.

**The target moves during learning.**
Changing $\hat v$ changes both the prediction and the target. This makes optimization less like ordinary supervised regression.

**Credit travels only locally.**
TD(0) initially propagates information one transition at a time. A reward far in the future may require many updates to influence an earlier state. MC immediately assigns the entire return to every visited state.

**TD can become unstable with approximation and off-policy learning.**
In the tabular on-policy case, TD converges to $v_\pi$ under standard step-size and visitation assumptions. But combining

1. bootstrapping,
2. function approximation,
3. off-policy sampling

can cause divergence—the “deadly triad.” MC does not suffer from bootstrapping instability in the same way because its targets do not depend on the current value estimate.

Thus TD’s finite-time estimates are biased, but this does **not** mean TD necessarily converges to the wrong value. In the standard tabular setting, the Bellman operator repeatedly corrects the bootstrapping bias, and TD converges to the same true value as MC.

The deeper picture is that TD replaces uncertain future randomness with its current prediction of that future. This reduces noise, but it means trusting an estimate that may currently be wrong.

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(The Bias–Variance Trade-off between TD and MC: Trading Bias for Variance through Bootstrapping)</span></p>

Explain qualitatively why TD(0) typically has lower variance but higher bias than constant-$\alpha$ MC for value prediction.

In what kinds of environments—long episodes, continuing tasks, or non-stationary dynamics—is each advantage most beneficial?

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary> </summary>

TD(0) usually has lower variance than Monte Carlo prediction because its target

$$R_{t+1}+\gamma V(S_{t+1})$$

only samples one reward and one next state. The random tail of the future trajectory is replaced by the current estimate $V(S_{t+1})$. By contrast, the Monte Carlo target is the full return

$$G_t = R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\cdots.$$

It contains randomness from all future rewards, transitions, and actions. For long episodes or for $\gamma\approx 1$, this can have large variance.

The cost of TD(0) is higher bias. The bootstrap term $V(S_{t+1})$ may be wrong, especially early in learning. Hence

$$R_{t+1}+\gamma V(S_{t+1})$$

may point systematically away from the true value. This is the bootstrap bias.

Monte Carlo prediction does not bootstrap. It uses the actually realized return $G_t$. For a fixed policy in a stationary episodic environment,

$$\mathbb E_\pi[G_t\mid S_t=s]=v_\pi(s),$$

so MC has an unbiased target for $v_\pi(s)$, although often a noisy one.

TD(0) is useful in long episodes, continuing tasks, tasks with $\gamma\approx 1$, and online learning settings. In these cases MC either has to wait too long before updating or suffers from high-variance returns. TD is also useful in non-stationary environments when combined with a constant step size, because it can keep tracking changing values from local updates.

Monte Carlo prediction is useful in short episodic tasks, when full returns are cheap to observe, and when avoiding bootstrap bias is more important than reducing variance. 

</details>
</div>

### The Random-Walk Example

<figure class="rl-diagram">
  <svg viewBox="0 0 760 150" role="img" aria-label="Five-state random walk Markov reward process">
    <rect x="40" y="55" width="50" height="50" rx="6" fill="#64748b"></rect>
    <text x="65" y="86" text-anchor="middle" font-size="16" font-weight="700" fill="#ffffff">0</text>
    <circle cx="170" cy="80" r="26" class="box"></circle>
    <text x="170" y="86" text-anchor="middle" font-size="16" font-weight="700">A</text>
    <circle cx="280" cy="80" r="26" class="box"></circle>
    <text x="280" y="86" text-anchor="middle" font-size="16" font-weight="700">B</text>
    <circle cx="390" cy="80" r="26" class="accent"></circle>
    <text x="390" y="86" text-anchor="middle" font-size="16" font-weight="700">C</text>
    <circle cx="500" cy="80" r="26" class="box"></circle>
    <text x="500" y="86" text-anchor="middle" font-size="16" font-weight="700">D</text>
    <circle cx="610" cy="80" r="26" class="box"></circle>
    <text x="610" y="86" text-anchor="middle" font-size="16" font-weight="700">E</text>
    <rect x="670" y="55" width="50" height="50" rx="6" fill="#047857"></rect>
    <text x="695" y="86" text-anchor="middle" font-size="16" font-weight="700" fill="#ffffff">1</text>

    <line x1="144" y1="80" x2="92" y2="80" class="line"></line>
    <line x1="196" y1="80" x2="254" y2="80" class="line"></line>
    <line x1="306" y1="80" x2="364" y2="80" class="line"></line>
    <line x1="416" y1="80" x2="474" y2="80" class="line"></line>
    <line x1="526" y1="80" x2="584" y2="80" class="line"></line>
    <line x1="636" y1="80" x2="668" y2="80" class="line"></line>
    <text x="390" y="135" text-anchor="middle" font-size="13" class="muted">start</text>
  </svg>
  <figcaption>Five nonterminal states $A, B, C, D, E$. Every episode starts at $C$; each step goes left or right with probability $1/2$. Reward $+1$ only on terminating to the <em>right</em>, $0$ otherwise; undiscounted episodic ($\gamma = 1$).</figcaption>
</figure>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Random walk — true values)</span></p>

Because $\gamma = 1$ and the only nonzero reward is $+1$ on terminating to the right, the value of a state equals the **probability of eventually exiting on the right**, which for a symmetric walk is linear in position:

$$v(A) = \tfrac{1}{6}, \quad v(B) = \tfrac{2}{6}, \quad v(C) = \tfrac{3}{6}, \quad v(D) = \tfrac{4}{6}, \quad v(E) = \tfrac{5}{6}.$$

The estimates are initialised to $V(s) = 0.5$ for all states (correct only for the centre $C$).

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(What the experiments show)</span></p>

* **TD propagates information one step at a time.** After a *single* episode, only one entry of $V$ has changed — the surprise reaches the directly adjacent state first and ripples outward over subsequent episodes. After 100 episodes the estimates closely track the true line.
* **Constant $\alpha$ never fully settles.** With a fixed step size, $V$ keeps fluctuating around the truth indefinitely (it tracks rather than converges pointwise) — this is the price of online responsiveness.
* **TD beats MC across step sizes.** Averaging RMS error over states and many seeds, TD reduces error faster than constant-$\alpha$ MC over a wide range of $\alpha$ — a concrete instance of the variance advantage.

</div>

### Online vs Batch Updating

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(From online updates to batch updates)</span></p>

* So far updates are **online**: each increment is applied immediately, so $V$ changes between transitions. 
* To compare MC and TD *cleanly* — independent of step-size and update-order artefacts — consider **batch** updating instead:
  * freeze $V$;
  * walk through every transition $(S\_t, R\_{t+1}, S\_{t+1})$ in a fixed dataset $\mathcal{D}$, accumulating 
    
    $$\Delta_t = \alpha\,[\,\text{target}_t - V(S_t)\,]$$

  * apply the *summed* increment once, then repeat the sweep to convergence.

The only difference between the two methods is again the target: 
* **batch MC** uses $G\_t$; 
* **batch TD(0)** uses $R\_{t+1} + \gamma V(S\_{t+1})$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Batch MC and batch TD have different fixed points)</span></p>

* For small enough $\alpha$, batch TD(0) and batch MC each converge to a **unique fixed point**, independent of $\alpha$. 
* But on the *same* dataset these are **different** fixed points.

The reason is the per-step target: 
* batch MC drives $V$ toward the full sample return 
  
  $$G_t,$$

* while batch TD(0) drives $V$ toward the one-step Bellman backup 
  
  $$R_{t+1} + \gamma V(S_{t+1}).$$

Identical data, different notion of "best fit."

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(What should $V(A)$ be?)</span></p>

Eight observed episodes ($\gamma = 1$):

$$A, 0, B, 0 \quad\mid\quad B, 1 \quad\mid\quad B, 1 \quad\mid\quad B, 1 \quad\mid\quad B, 1 \quad\mid\quad B, 1 \quad\mid\quad B, 1 \quad\mid\quad B, 0.$$

Everyone agrees on $B$: it was visited 8 times and the return was $1$ in 6 of them, so

$$V(B) = \tfrac{6}{8} = \tfrac{3}{4}.$$

But what is $V(A)$? State $A$ was visited exactly once, in the episode $A, 0, B, 0$. There are **two reasonable answers**.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(Two reasonable answers for $V(A)$)</span></p>

* **Batch MC.** Asks *what returns were observed after visiting $A$?* Only one return: $G = 0$. So batch MC drives $V(A) \to 0$. This **minimises mean-square error on the observed data** — it fits the past perfectly (zero training error).
* **Batch TD.** Asks *what one-step transition was observed from $A$?* Always $A \xrightarrow{0} B$. So batch TD enforces $V(A) = 0 + V(B) = \tfrac{3}{4}$. This is the answer an **exact solution on the inferred Markov model** would give.

Which is "right"? 

* MC fits the past data perfectly; 
* TD generalises via the **Markov structure** of the data. 
* On a Markov process, TD's answer is usually the better predictor of *future* returns
  * which is what we actually care about.

</div>

<figure class="rl-diagram">
  <svg viewBox="0 0 860 340" role="img" aria-label="Batch Monte Carlo fits observed returns while batch TD solves the empirical Markov model">
    <defs>
      <marker id="td-batch-arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
        <path d="M0,0 L0,6 L9,3 z" fill="#64748b"></path>
      </marker>
      <marker id="td-batch-strong" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
        <path d="M0,0 L0,6 L9,3 z" fill="#2c3e94"></path>
      </marker>
    </defs>

    <rect x="40" y="48" width="235" height="235" rx="8" class="box"></rect>
    <text x="158" y="82" text-anchor="middle" font-size="16" font-weight="700">Observed data</text>
    <circle cx="105" cy="150" r="24" class="accent"></circle>
    <text x="105" y="156" text-anchor="middle" font-size="16" font-weight="700">A</text>
    <circle cx="210" cy="150" r="24" class="box"></circle>
    <text x="210" y="156" text-anchor="middle" font-size="16" font-weight="700">B</text>
    <path d="M129 150 L183 150" class="strong-line" marker-end="url(#td-batch-strong)"></path>
    <text x="157" y="137" text-anchor="middle" font-size="13">r=0</text>
    <text x="158" y="210" text-anchor="middle" font-size="13" class="muted">B returns: six 1s, two 0s</text>
    <text x="158" y="235" text-anchor="middle" font-size="13" class="muted">so V(B) = 3/4</text>

    <rect x="335" y="48" width="210" height="235" rx="8" class="amber"></rect>
    <text x="440" y="82" text-anchor="middle" font-size="16" font-weight="700">Batch MC</text>
    <text x="440" y="122" text-anchor="middle" font-size="13">fit returns seen after A</text>
    <rect x="380" y="152" width="120" height="56" rx="7" fill="#ffffff" stroke="#b45309"></rect>
    <text x="440" y="176" text-anchor="middle" font-size="14" font-weight="700">only G(A)=0</text>
    <text x="440" y="196" text-anchor="middle" font-size="13">V(A) -> 0</text>
    <text x="440" y="243" text-anchor="middle" font-size="13" class="muted">best fit to this sample</text>

    <rect x="610" y="48" width="210" height="235" rx="8" class="green"></rect>
    <text x="715" y="82" text-anchor="middle" font-size="16" font-weight="700">Batch TD</text>
    <text x="715" y="122" text-anchor="middle" font-size="13">fit one-step model</text>
    <circle cx="675" cy="170" r="21" fill="#ffffff" stroke="#047857" stroke-width="2"></circle>
    <text x="675" y="176" text-anchor="middle" font-size="14" font-weight="700">A</text>
    <circle cx="755" cy="170" r="21" fill="#ffffff" stroke="#047857" stroke-width="2"></circle>
    <text x="755" y="176" text-anchor="middle" font-size="14" font-weight="700">B</text>
    <path d="M697 170 L732 170" stroke="#047857" stroke-width="3" fill="none" marker-end="url(#td-batch-strong)"></path>
    <text x="715" y="219" text-anchor="middle" font-size="13">V(A) = 0 + V(B)</text>
    <text x="715" y="241" text-anchor="middle" font-size="13" font-weight="700">V(A) -> 3/4</text>

    <path d="M275 165 L332 165" class="line" marker-end="url(#td-batch-arrow)"></path>
    <path d="M545 165 L607 165" class="line" marker-end="url(#td-batch-arrow)"></path>
  </svg>
  <figcaption>Batch MC treats the one return following $A$ as the target, so it memorises $V(A)=0$. Batch TD treats the observed transition $A \to B$ as evidence about the Markov structure, so it backs up from the learned value of $B$.</figcaption>
</figure>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Batch TD computes the certainty-equivalence estimate)</span></p>

Build the **maximum-likelihood Markov reward process** from the data:

$$\hat{p}(i \to j) = \text{fraction of observed transitions from } i \text{ that go to } j,$$

$$\qquad \hat{r}(i \to j) = \text{average reward on those transitions.}$$

The **certainty-equivalence estimate** treats $(\hat{p}, \hat{r})$ as if it were the true model and solves the Bellman equations *exactly* on it. The (perhaps surprising) fact:

> Under batch updating, **TD(0) converges to the certainty-equivalence estimate**.

So batch TD is implicitly doing **model estimation $+$ planning** — without ever building the model explicitly.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why not just compute certainty equivalence directly?)</span></p>

Because it does not scale. For $N = \lvert\mathcal{S}\rvert$ states, forming the certainty-equivalence estimate directly costs

* $O(N^2)$ memory to store $\hat{p}$, and
* $O(N^3)$ to solve the Bellman equations with generic methods.

TD approximates the *same* answer with only $O(N)$ memory (just $V$) and repeated sweeps over experience (or streaming online updates). On large state spaces, **TD is the only feasible way to approximate the certainty-equivalence value function** — this is a large part of why bootstrapping is so valuable in practice.

</div>

### From Prediction to Control

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(Control with TD — GPI once more)</span></p>

Control still follows **Generalised Policy Iteration**: 
* evaluate the current policy, 
* improve it toward greediness, 
* repeat.

With TD, the twist is that **evaluation and improvement are interleaved at the finest grain**
* updates happen step-by-step *within* each episode, and 
* the policy changes as soon as the value table changes.

As in model-free MC control, we cannot improve a policy from $V$ alone without a model, so we learn **action values** $Q(s, a)$ instead. Everything that follows is "TD(0), but on $Q$."

</div>

### Sarsa: On-Policy TD Control

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Sarsa update)</span></p>

Treat the trajectory as alternating state–action pairs $S\_t, A\_t, R\_{t+1}, S\_{t+1}, A\_{t+1}, \dots$ and apply the TD(0) idea to each pair as though it were one enlarged "state":

$$Q(S_t, A_t) \;\leftarrow\; Q(S_t, A_t) + \alpha\Bigl[\, \underbrace{R_{t+1} + \gamma\, Q(S_{t+1}, A_{t+1})}_{\text{TD target}} - Q(S_t, A_t) \,\Bigr].$$

The update uses the quintuple $(S\_t, A\_t, R\_{t+1}, S\_{t+1}, A\_{t+1})$ — which spells the name **Sarsa**. Crucially, $A\_{t+1}$ is the action *actually taken* under the current policy (this is what makes Sarsa **on-policy**). If $S\_{t+1}$ is terminal, the TD target is just $R\_{t+1}$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Sarsa — on-policy TD control)</span></p>

**Parameters:** step size $\alpha \in (0, 1]$, exploration $\varepsilon > 0$.

Initialise $Q(s, a)$ arbitrarily, with $Q(\text{terminal}, \cdot) = 0$.

Loop for each episode:

1. Initialise $S$; 
2. Choose $A$ from $S$ using an $\varepsilon$-greedy policy derived from $Q$.
3. Loop for each step of the episode:
   * Take action $A$; observe $R, S'$.
   * **If $S'$ is terminal:** 
     
     $$Q(S, A) \leftarrow Q(S, A) + \alpha\,[\,R - Q(S, A)\,]$$

     then break.

   * **Else:** 
     * choose $A'$ from $S'$ using the $\varepsilon$-greedy policy; 
     * update
     
       $$Q(S, A) \leftarrow Q(S, A) + \alpha\bigl[\,R + \gamma Q(S', A') - Q(S, A)\,\bigr];$$
     
     * then $S \leftarrow S'$, $A \leftarrow A'$.

Each TD update *evaluates* the current policy a little, and each $\varepsilon$-greedy action choice *improves* the behaviour a little — GPI at the granularity of a single transition.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Sarsa convergence — GLIE)</span></p>

Convergence to the optimal action-value function needs **two simultaneous conditions**:

1. **exploration:** every $(s, a)$ pair is visited infinitely often;
2.  **greedy in the limit:** the policy becomes greedy with respect to $Q$ as $t \to \infty$.

* Any schedule satisfying both is called **GLIE** (*Greedy in the Limit with Infinite Exploration*); 
  * a concrete recipe is $\varepsilon$-greedy with $\varepsilon\_t = 1/t$.
* Under GLIE *and* the standard step-size conditions, tabular Sarsa converges to the optimal action-value function $q\_\ast$ and an optimal policy.

</div>

### Q-Learning: Off-Policy TD Control

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Idea</span><span class="math-callout__name">(From Sarsa to Q-learning)</span></p>

Compare the two targets:

$$\text{Sarsa: } R_{t+1} + \gamma\, Q(S_{t+1}, \underbrace{A_{t+1}}_{\text{action taken}}), \qquad \text{Q-learning: } R_{t+1} + \gamma \max_a Q(S_{t+1}, a).$$

* Sarsa learns the value of the **policy it actually follows**. 
* Q-learning learns the value of the **greedy policy** — regardless of which action it actually takes next. 

This is what makes Q-learning **off-policy**:
* the **behaviour** policy is any $\varepsilon$-greedy policy (it must visit all $(s, a)$);
* the **target** policy is greedy with respect to the current $Q$;
* the two need not match.

</div>

<figure class="rl-diagram">
  <svg viewBox="0 0 860 360" role="img" aria-label="Sarsa backs up the actually selected next action while Q-learning backs up the maximum next action">
    <defs>
      <marker id="td-control-arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
        <path d="M0,0 L0,6 L9,3 z" fill="#64748b"></path>
      </marker>
      <marker id="td-control-strong" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
        <path d="M0,0 L0,6 L9,3 z" fill="#2c3e94"></path>
      </marker>
    </defs>

    <text x="215" y="48" text-anchor="middle" font-size="17" font-weight="700">Sarsa: use the action actually taken</text>
    <text x="645" y="48" text-anchor="middle" font-size="17" font-weight="700">Q-learning: use the greedy action</text>

    <g>
      <circle cx="92" cy="170" r="27" class="box"></circle>
      <text x="92" y="176" text-anchor="middle" font-size="14" font-weight="700">S_t</text>
      <circle cx="225" cy="170" r="27" class="box"></circle>
      <text x="225" y="176" text-anchor="middle" font-size="14" font-weight="700">S'</text>
      <path d="M120 170 L197 170" class="strong-line" marker-end="url(#td-control-strong)"></path>
      <rect x="138" y="128" width="44" height="25" rx="5" class="amber"></rect>
      <text x="160" y="146" text-anchor="middle" font-size="12">R</text>

      <circle cx="315" cy="105" r="22" class="box"></circle>
      <text x="315" y="110" text-anchor="middle" font-size="12">left</text>
      <circle cx="315" cy="170" r="22" class="amber"></circle>
      <text x="315" y="175" text-anchor="middle" font-size="12">actual</text>
      <circle cx="315" cy="235" r="22" class="box"></circle>
      <text x="315" y="240" text-anchor="middle" font-size="12">right</text>
      <path d="M252 170 L291 170" stroke="#b45309" stroke-width="3" fill="none" marker-end="url(#td-control-arrow)"></path>
      <text x="215" y="282" text-anchor="middle" font-size="13" class="muted">target uses Q(S', A') from the behavior policy</text>
    </g>

    <line x1="430" y1="72" x2="430" y2="300" stroke="#dbe1ee" stroke-width="2"></line>

    <g>
      <circle cx="522" cy="170" r="27" class="box"></circle>
      <text x="522" y="176" text-anchor="middle" font-size="14" font-weight="700">S_t</text>
      <circle cx="655" cy="170" r="27" class="box"></circle>
      <text x="655" y="176" text-anchor="middle" font-size="14" font-weight="700">S'</text>
      <path d="M550 170 L627 170" class="strong-line" marker-end="url(#td-control-strong)"></path>
      <rect x="568" y="128" width="44" height="25" rx="5" class="amber"></rect>
      <text x="590" y="146" text-anchor="middle" font-size="12">R</text>

      <circle cx="745" cy="105" r="22" class="box"></circle>
      <text x="745" y="110" text-anchor="middle" font-size="12">left</text>
      <circle cx="745" cy="170" r="22" class="box"></circle>
      <text x="745" y="175" text-anchor="middle" font-size="12">actual</text>
      <circle cx="745" cy="235" r="22" class="green"></circle>
      <text x="745" y="240" text-anchor="middle" font-size="12">max</text>
      <path d="M682 170 C710 173 719 208 729 219" stroke="#047857" stroke-width="3" fill="none" marker-end="url(#td-control-arrow)"></path>
      <text x="645" y="282" text-anchor="middle" font-size="13" class="muted">target uses max_a Q(S', a), even if behavior picked another action</text>
    </g>
  </svg>
  <figcaption>Sarsa and Q-learning observe the same transition. They differ only in the next-state action used inside the TD target: Sarsa follows the sampled behaviour action $A'$, while Q-learning backs up the greedy action value.</figcaption>
</figure>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Q-learning update)</span></p>

$$Q(S_t, A_t) \;\leftarrow\; Q(S_t, A_t) + \alpha\Bigl[\, \underbrace{R_{t+1} + \gamma \max_a Q(S_{t+1}, a)}_{\text{TD target}} - Q(S_t, A_t) \,\Bigr].$$

It uses the quadruple $(S\_t, A\_t, R\_{t+1}, S\_{t+1})$ — there is **no $A\_{t+1}$**: 
* The term $\max\_a Q(S\_{t+1}, a)$ is the best possible value at the next state; 
* If $S\_{t+1}$ is terminal
  * $\max\_a Q(S\_{t+1}, a) \doteq 0$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Q-learning — off-policy TD control)</span></p>

**Parameters:** step size $\alpha \in (0, 1]$, exploration $\varepsilon > 0$. 

Initialise $Q(s, a)$ arbitrarily, with $Q(\text{terminal}, \cdot) = 0$.

Loop for each episode:

1. Initialise $S$.
2. Loop for each step of the episode, until $S$ is terminal:
   * Choose $A$ from $S$ using a policy derived from $Q$ (e.g. $\varepsilon$-greedy).
   * Take action $A$; 
   * Observe $R, S'$.
   * $Q(S, A) \leftarrow Q(S, A) + \alpha\bigl[\,R + \gamma \max\_a Q(S', a) - Q(S, A)\,\bigr]$.
   * $S \leftarrow S'$.

Only *one* action is sampled from the next state, but the target uses the **max** over actions. This mixed pattern — **sample the successor, maximise over actions** — is the heart of model-free greedy control. Under the standard step-size conditions and visiting all $(s, a)$ infinitely often, $Q \to q\_\ast$ with probability 1, *regardless of how exploratory the behaviour policy is.*

</div>

<figure class="rl-diagram">
  <svg viewBox="0 0 760 230" role="img" aria-label="Cliff walking gridworld with safe and risky paths">
    <rect x="120" y="40" width="520" height="120" rx="6" fill="none" stroke="#64748b" stroke-width="2"></rect>
    <rect x="170" y="120" width="420" height="40" fill="#fef2f2" stroke="#b91c1c"></rect>
    <text x="380" y="145" text-anchor="middle" font-size="14" font-weight="700" fill="#b91c1c">The Cliff (R = -100)</text>
    <rect x="120" y="120" width="50" height="40" fill="#ecfdf5" stroke="#047857"></rect>
    <text x="145" y="145" text-anchor="middle" font-size="15" font-weight="700">S</text>
    <rect x="590" y="120" width="50" height="40" fill="#ecfdf5" stroke="#047857"></rect>
    <text x="615" y="145" text-anchor="middle" font-size="15" font-weight="700">G</text>

    <path d="M145 120 L145 65 L615 65 L615 120" class="strong-line"></path>
    <path d="M150 150 L585 150" stroke="#b45309" stroke-width="3" stroke-dasharray="7 5" fill="none"></path>

    <line x1="200" y1="200" x2="240" y2="200" class="strong-line"></line>
    <text x="250" y="205" font-size="13" class="muted">Sarsa: safer detour</text>
    <line x1="430" y1="200" x2="470" y2="200" stroke="#b45309" stroke-width="3" stroke-dasharray="7 5"></line>
    <text x="480" y="205" font-size="13" class="muted">Q-learning: optimal/risky edge</text>
  </svg>
  <figcaption>Cliff-walking gridworld: start $S$ bottom-left, goal $G$ bottom-right, reward $-1$ per step, and $-100$ (with reset to $S$) for stepping into the cliff. Undiscounted episodic, $\gamma = 1$.</figcaption>
</figure>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Explanation of classical Cliff-walking gridworld example</summary>

The key point is that **Q-learning learns the best greedy policy, but behaves exploratorily while learning**. SARSA learns the value of the exploratory policy it is actually executing.

Assume the classical cliff grid:

* Every move gives reward $-1$.
* Entering the cliff gives $-100$ and returns the agent to $S$.
* Actions are chosen $\varepsilon$-greedily.

**What Q-learning evaluates.**

Its update is

$$Q(S_t,A_t)\leftarrow Q(S_t,A_t) +\alpha\left(R_{t+1}+\gamma\max_a Q(S_{t+1},a)-Q(S_t,A_t)\right).$$

The $\max_a$ means:

> “After reaching $S_{t+1}$, suppose I behave perfectly greedily.”

Under perfectly greedy behaviour, the shortest route is the path immediately beside the cliff. No mistake occurs, so taking a longer detour only accumulates more $-1$ rewards. Therefore Q-learning correctly identifies the cliff-edge route as the optimal deterministic policy.

But during training the agent is not actually greedy—it is $\varepsilon$-greedy. At every state near the cliff, exploration may select the downward action and produce reward $-100$.

Thus Q-learning has a mismatch:

$$\text{policy being evaluated}=\text{greedy policy}, \qquad \text{policy generating experience}=\varepsilon\text{-greedy policy}.$$

It knows the edge route is excellent **provided that no exploratory mistake is made**, while continuing to make exploratory mistakes.

---

**What SARSA evaluates.**

SARSA uses

$$Q(S_t,A_t)\leftarrow Q(S_t,A_t) +\alpha\left(R_{t+1}+\gamma Q(S_{t+1},A_{t+1})-Q(S_t,A_t)\right),$$

where $A_{t+1}$ is the action actually selected by the $\varepsilon$-greedy policy.

Consequently, SARSA learns:

> “What return will I obtain if I continue behaving $\varepsilon$-greedily?”

Near the cliff, this includes the possibility that $A_{t+1}$ is an exploratory action into the cliff. Hence states close to the cliff receive lower values. A route one row farther away may require a few extra steps, but exploratory actions there are much less catastrophic.

So SARSA learns the safer route because it internalizes its own imperfect behaviour.

---

**Why a small exploration probability matters.**

Suppose there are four actions and $\varepsilon=0.1$. At a state beside the cliff, the probability of selecting “down” purely through exploration is approximately

$$\frac{\varepsilon}{4}=0.025.$$

If the route contains ten such dangerous decisions, the probability of at least one downward exploratory move is approximately

$$1-(1-0.025)^{10}\approx 0.224.$$

So there is roughly a $22%$ chance of falling during a single traversal, even though exploration at each individual state is small. A few extra $-1$ steps are easily preferable to repeated $-100$ penalties.

---

**In what sense is Q-learning inferior?**

Only in terms of **online reward while exploration remains active**:

$$\text{SARSA: safer behaviour, better reward during learning},$$

$$\text{Q-learning: riskier behaviour, worse reward during learning}.$$

Q-learning is not learning the wrong optimal policy. In this deterministic environment, its final greedy policy is the true shortest-path policy. If exploration is turned off after learning, Q-learning follows the cliff edge safely and outperforms the longer SARSA route.

With a decaying $\varepsilon$ satisfying the usual convergence conditions, SARSA can also eventually converge to the optimal greedy policy. The famous difference appears when exploration remains substantial: **Q-learning optimizes for the agent it wishes it were; SARSA optimizes for the exploratory agent it actually is.**


</details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>But is not Q’s route more rewarding since it is shorter the therefore accumulates less negative rewards?</summary>

Yes — **under purely greedy execution**, Q-learning’s route is better because it is shorter:

$$G_{\text{edge}}=-L_{\text{edge}} \quad\text{and}\quad G_{\text{safe}}=-L_{\text{safe}},$$

with $L_{\text{edge}}<L_{\text{safe}}$. Hence

$$-L_{\text{edge}}>-L_{\text{safe}}.$$

The difference arises because, during learning, the agent is still $\varepsilon$-greedy.

Suppose the safe path costs only two additional steps. Then it loses an extra $2$ reward units. But if the cliff-edge path has probability $p$ of incurring the $-100$ cliff penalty, its approximate additional expected cost is $100p$.

Thus the edge route is worse whenever

$$100p>2,$$

or

$$p>0.02.$$

So even a $2%$ chance of falling can erase the advantage of saving two steps.

Q-learning estimates the route as though future actions were greedy:

$$R_{t+1}+\gamma\max_a Q(S_{t+1},a),$$

so it effectively evaluates the edge route with no exploratory mistakes. SARSA uses the actual next $\varepsilon$-greedy action:

$$R_{t+1}+\gamma Q(S_{t+1},A_{t+1}),$$

and therefore includes the risk of falling.

So:

* **Exploration off:** Q-learning’s shorter route is more rewarding.
* **Exploration still active:** SARSA’s longer route can have a higher expected return because it avoids rare but enormous $-100$ losses.

Q-learning is inferior here only in **online performance during persistent exploration**, not in the quality of its final greedy policy.

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Cliff walking — on-policy vs off-policy)</span></p>

* **Q-learning** learns the values of the *greedy* (optimal) path that runs right along the cliff edge. 
  * But its $\varepsilon$-greedy *behaviour* occasionally slips off the cliff, so its **online return is poor**. 
* **Sarsa** learns the value of the $\varepsilon$-greedy policy *itself* — which includes the chance of falling
  * so it prefers the **safer detour** one row up, and earns a **higher online return**.

This crystallises the distinction:

* **On-policy** methods (Sarsa) evaluate *what they actually do*, exploration included.
* **Off-policy** methods (Q-learning) evaluate *what they would do greedily*, ignoring exploration.

Online performance reflects the former — which is why Sarsa looks better *during learning* even though Q-learning has learned the truly optimal path.

</div>

### Maximization Bias and Double Q-Learning

* [Reddit: Is my understanding of double Q-learning correct?](https://www.reddit.com/r/MachineLearning/comments/57ec9z/discussion_is_my_understanding_of_double/)

<figure>
  <img src="{{ '/assets/images/notes/rl_hd/Q-learning_Double_Q-learning.png' | relative_url }}" alt="Two pairs of 3D surfaces of estimated Blackjack value functions (with and without a usable ace) after 10,000 and 500,000 sampled episodes, becoming visibly smoother as the sample size grows" loading="lazy">
  <figcaption>Comparison of Q-learning and double Q-learning on a simple episodic MDP using $\varepsilon$-greedy action selection with $\varepsilon= 0.1$. The initial state is A, and squares denote absorbing states.</figcaption>
</figure>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Problem</span><span class="math-callout__name">(Maximization bias)</span></p>

Standard Q-learning suffers from a problem known as the optimizer’s curse, or the maximization bias. The problem refers to the simple statistical inequality:

$$\mathbb{E}[\max_a X_a] \geq \max_a \mathbb{E}[X_a]$$

for a set of random variables $\lbrace X_a\rbrace$. Thus, if we pick actions greedily according to their random scores $\lbrace X_a\rbrace$, we might pick a wrong action just because random noise makes it appealing.

Figure above gives a simple example of how this can happen in an MDP. The start state is A. The right action gives a reward $0$ and terminates the episode. The left action also gives a reward of $0$, but then enters state B, from which there are many possible actions, with rewards drawn from $\mathcal N(−0.1,1.0)$. Thus the expected return for any trajectory starting with the left action is $−0.1$, making it suboptimal. Nevertheless, the RL algorithm may pick the left action due to the maximization bias making B appear to have a positive value.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Mapping this problem to our context: Q-learning</summary>

You are right: the lecture skips the crucial intermediate step. The expectation is **not part of the implemented Q-learning update**; it appears when we analyze the update as a random estimator.

Q-learning uses

$$Y_t =R_{t+1} + \gamma\max_a \widehat Q_t(S_{t+1},a).$$

The true Bellman target would involve the true action values:

$$Y_t^\ast = R_{t+1} + \gamma\max_a q_\ast(S_{t+1},a).$$

The issue is that $\widehat Q_t(s,a)$ is noisy because it was learned from random trajectories, rewards, transitions, and initialization. Across possible training histories, it is therefore a random variable.

Fix the next state $S_{t+1}=s'$. Taking expectation over the randomness of the learned estimates gives

$$\mathbb E[Y_t\mid S_{t+1}=s'] = \mathbb E[R_{t+1}\mid S_{t+1}=s'] + \gamma\mathbb E\left[\max_a\widehat Q_t(s',a)\right].$$

$$\mathbb E[Y_t^\ast\mid S_{t+1}=s'] = \mathbb E[R_{t+1}\mid S_{t+1}=s'] + \gamma\max_a q^\ast_t(s',a).$$

Now the inequality appears:

$$\mathbb E\left[\max_a\widehat Q_t(s',a)\right] \geq \max_a\mathbb E[\widehat Q_t(s',a)].$$

Suppose every individual estimate is unbiased:

$$\mathbb E[\widehat Q_t(s',a)] = q_\ast(s',a).$$

Then

$$\mathbb E\left[\max_a\widehat Q_t(s',a)\right] \geq \max_a q_\ast(s',a).$$

Therefore,

$$\boxed{ \mathbb E[Y_t] \geq \mathbb E[Y_t^\ast].}$$

So the Q-learning target is systematically too optimistic on average. Since the update is

$$\widehat Q_t(S_t,A_t) \leftarrow \widehat Q_t(S_t,A_t) + \alpha\left[Y_t-\widehat Q_t(S_t,A_t)\right],$$

the upward-biased target creates an upward drift in the updated value. The notes state the target and the inequality, but do not explicitly insert this expectation-taking step between them. 

**Double Q-learning**

This is also why Double Q-learning separates

$$A^\ast=\arg\max_a Q_1(s',a)$$

from its evaluation

$$Q_2(s',A^\ast).$$

The positive error in $Q_1$ that caused the action to be selected is not reused as its evaluated value. 

This maximization bias is mainly a finite-sample learning problem; ordinary tabular Q-learning can still converge to $q_\ast$ under the standard diminishing-step-size and exploration assumptions.

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(A subtle problem in Q-learning)</span></p>

The Q-learning target 

$$R_{t+1} + \gamma \max_a Q(S_{t+1}, a)$$

uses the *same* estimates $Q$ to do two jobs at once: 
* **select** the action with the largest estimate, and 
* **evaluate** that selected action. 

This double-duty introduces **maximization bias** — even if every individual estimate is unbiased, the maximum is biased *upward*:

$$\mathbb{E}\Bigl[\max_a Q(s, a)\Bigr] \;\ge\; \max_a \mathbb{E}\bigl[Q(s, a)\bigr].$$

In particular, if 

$$\mathbb{E}[Q(s, a)] = q(s, a),$$

then 

$$\mathbb{E}[\max_a Q(s, a)] \ge \max_a q(s, a)$$ 

— Q-learning is systematically optimistic about the value of the best action.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>What is meant by two jobs</summary>

You are right to distinguish the two actions. The remark uses **“evaluate” in a different sense** from “update the action that was executed.”

At time $t$:

1. The behavior policy chooses $A_t$ in $S_t$.
2. Executing $A_t$ produces $R_{t+1}$ and $S_{t+1}$.
3. Q-learning updates the executed pair $Q(S_t,A_t)$:

$$Q(S_t,A_t)\leftarrow Q(S_t,A_t) +\alpha\left[ R_{t+1} +\gamma\max_a Q(S_{t+1},a) -Q(S_t,A_t) \right].$$

So yes: the **entry being updated** corresponds to the behavior action $A_t$ that led to $S_{t+1}$. 

But inside the target, Q-learning defines

$$A^*=\arg\max_a Q(S_{t+1},a)$$

and then uses

$$Q(S_{t+1},A^*)$$

as its estimate of the future value. This is what the remark calls:

* **selection:** use $Q$ to choose $A^\ast$;
* **evaluation:** use the same $Q$ to assign the numerical value $Q(S_{t+1},A^\ast)$ to that selected action.

The word “evaluate” here does **not** mean that $A^\ast$ was executed in the environment. It means that its value estimate is inserted into the TD target.

The action roles are therefore:

$$\underbrace{A_t}_{\substack{\text{behavior action. }\text{actually executed}}} \quad\longrightarrow\quad R_{t+1},S_{t+1},$$

then

$$\underbrace{A^*=\arg\max_a Q(S_{t+1},a)}_{\substack{\text{hypothetical greedy next action. }\text{selected and valued inside target}}}.$$

The actual next behavior action $A_{t+1}$ may be different because the behavior policy is $\varepsilon$-greedy. Q-learning ignores $A_{t+1}$ in its update; Sarsa uses it:

$$\text{Sarsa target} = R_{t+1}+\gamma Q(S_{t+1},A_{t+1}),$$

$$\text{Q-learning target} = R_{t+1}+\gamma Q(S_{t+1},A^*).$$

So your objection is valid about the wording: **Q-learning updates the action that was behaviorally executed, but it “selects and evaluates” a possibly unexecuted greedy action at the next state.** Maximization bias arises in this second part. 

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Derivation</span><span class="math-callout__name">(Why the max overestimates)</span></p>

Let 

$$b^\ast \in \arg\max\_b \mathbb{E}[Q(s, b)].$$

Since the maximum is at least as large as any particular action value,

$$\max_a Q(s, a) \;\ge\; Q(s, b^\ast).$$

Taking expectations on both sides,

$$\mathbb{E}\Bigl[\max_a Q(s, a)\Bigr] \;\ge\; \mathbb{E}[Q(s, b^\ast)].$$

By the definition of $b^\ast$, 

$$\mathbb{E}[Q(s, b^\ast)] = \max_b \mathbb{E}[Q(s, b)],$$

hence

$$\mathbb{E}\Bigl[\max_a Q(s, a)\Bigr] \;\ge\; \max_b \mathbb{E}[Q(s, b)].$$

So taking a maximum over *noisy* estimates injects optimism — the inequality is strict whenever the estimates have genuine variance and are not all maximised at the same action.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Intuition — the max picks positive errors)</span></p>

Suppose two actions are *truly equally good*, $q(s, a\_1) = q(s, a\_2) = 0$, but our estimates are noisy,

$$Q(s, a_1) = q(s, a_1) + \text{error}_1, \qquad Q(s, a_2) = q(s, a_2) + \text{error}_2,$$

with 

$$\mathbb{E}[\text{error}_1] = \mathbb{E}[\text{error}_2] = 0.$$

Even so,

$$\max\lbrace Q(s, a_1), Q(s, a_2)\rbrace$$
 
tends to select whichever estimate happens to have the **more positive error**.

The key realisation: each estimate may be unbiased, but the *selected* one is **not random** — it is, by construction, the one that currently looks best. In Q-learning the same table $Q$ both 
* **selects** $A^\ast = \arg\max\_a Q(S\_{t+1}, a)$ and
* **evaluates** $Q(S\_{t+1}, A^\ast)$,

so the positive selection error is preserved straight into the TD target.

</div>

<figure class="rl-diagram">
  <svg viewBox="0 0 860 350" role="img" aria-label="Maximization bias picks the most positive noisy estimate; Double Q-learning separates selection from evaluation">
    <defs>
      <marker id="td-bias-arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
        <path d="M0,0 L0,6 L9,3 z" fill="#64748b"></path>
      </marker>
      <marker id="td-bias-strong" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
        <path d="M0,0 L0,6 L9,3 z" fill="#2c3e94"></path>
      </marker>
    </defs>

    <text x="215" y="48" text-anchor="middle" font-size="17" font-weight="700">Ordinary max: select and evaluate with Q</text>
    <text x="645" y="48" text-anchor="middle" font-size="17" font-weight="700">Double Q: split the two jobs</text>

    <g>
      <line x1="75" y1="205" x2="355" y2="205" stroke="#64748b" stroke-width="2"></line>
      <text x="64" y="210" text-anchor="end" font-size="12" class="muted">true 0</text>
      <rect x="118" y="170" width="48" height="35" class="box"></rect>
      <text x="142" y="228" text-anchor="middle" font-size="12">a1</text>
      <rect x="206" y="112" width="48" height="93" class="green"></rect>
      <text x="230" y="228" text-anchor="middle" font-size="12">a2</text>
      <rect x="294" y="190" width="48" height="15" class="box"></rect>
      <text x="318" y="228" text-anchor="middle" font-size="12">a3</text>
      <text x="230" y="96" text-anchor="middle" font-size="13" font-weight="700" fill="#047857">largest noisy estimate</text>
      <path d="M230 108 L230 112" class="line" marker-end="url(#td-bias-arrow)"></path>
      <rect x="95" y="262" width="240" height="43" rx="7" class="amber"></rect>
      <text x="215" y="288" text-anchor="middle" font-size="13" font-weight="700">positive noise enters the target</text>
    </g>

    <line x1="430" y1="72" x2="430" y2="312" stroke="#dbe1ee" stroke-width="2"></line>

    <g>
      <rect x="490" y="94" width="135" height="64" rx="8" class="accent"></rect>
      <text x="557" y="121" text-anchor="middle" font-size="15" font-weight="700">Q_1</text>
      <text x="557" y="143" text-anchor="middle" font-size="12">select argmax</text>
      <rect x="665" y="94" width="135" height="64" rx="8" class="green"></rect>
      <text x="732" y="121" text-anchor="middle" font-size="15" font-weight="700">Q_2</text>
      <text x="732" y="143" text-anchor="middle" font-size="12">evaluate selected</text>
      <path d="M626 126 L662 126" class="strong-line" marker-end="url(#td-bias-strong)"></path>
      <text x="645" y="185" text-anchor="middle" font-size="13" class="muted">selection noise and evaluation noise are independent</text>

      <rect x="500" y="222" width="290" height="62" rx="8" class="box"></rect>
      <text x="645" y="247" text-anchor="middle" font-size="13">target uses Q_2(S', argmax_a Q_1(S',a))</text>
      <text x="645" y="269" text-anchor="middle" font-size="13">or the symmetric update with Q_1 and Q_2 swapped</text>
      <path d="M732 159 L732 220" class="line" marker-end="url(#td-bias-arrow)"></path>
    </g>
  </svg>
  <figcaption>The max operator tends to choose the estimate with the most favourable noise. Double Q-learning reduces that optimism by using one table to choose the action and the other table to evaluate it.</figcaption>
</figure>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Double learning — separate selection from evaluation)</span></p>

Maintain **two** independent action-value estimates $Q\_1(s, a)$ and $Q\_2(s, a)$. Use one to *select* the maximising action and the *other* to *evaluate* its value:

$$A^\ast = \arg\max_a Q_1(s, a), \qquad \text{value} = Q_2(s, A^\ast).$$

Because the noise in $Q\_1$ (which made the selection) is **independent** of the noise in $Q\_2$ (which does the evaluation), the upward bias is removed in expectation. Swapping the roles symmetrically yields a second unbiased estimate.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Double Q-learning)</span></p>

**Parameters:** step size $\alpha \in (0, 1]$, exploration $\varepsilon > 0$. 

Initialise $Q\_1(s, a)$ and $Q\_2(s, a)$ arbitrarily, with 

$$Q_1(\text{terminal}, \cdot) = Q_2(\text{terminal}, \cdot) = 0$$

Loop for each episode, and for each step:

1. Choose $A$ from $S$ using an $\varepsilon$-greedy policy derived from $Q\_1 + Q\_2$; 
2. Take $A$;
3. Observe $R, S'$;
4. With probability $\tfrac{1}{2}$, **update $Q\_1$** (select with $Q\_1$, evaluate with $Q\_2$):

   $$Q_1(S, A) \leftarrow Q_1(S, A) + \alpha\Bigl[\,R + \gamma\, Q_2\bigl(S', \arg\max_a Q_1(S', a)\bigr) - Q_1(S, A)\,\Bigr].$$

5. **Otherwise update $Q\_2$** symmetrically (select with $Q\_2$, evaluate with $Q\_1$):

   $$Q_2(S, A) \leftarrow Q_2(S, A) + \alpha\Bigl[\,R + \gamma\, Q_1\bigl(S', \arg\max_a Q_2(S', a)\bigr) - Q_2(S, A)\,\Bigr].$$

6. $S \leftarrow S'$ (and at a terminal $S'$ the target is just $R$).

The behaviour policy is $\varepsilon$-greedy with respect to $Q\_1 + Q\_2$. The cost is a **$\times 2$ memory** footprint; the per-step computation is the same as ordinary Q-learning.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Intuition validation</summary>

> Is my intuition of of double Q-learning correct: we basically have two interleaving Q-learning process with different experiences, which makes their noise independent and each of the processes getting less / no biased on its own biased / noisy experience and they have sort of cooperate and try to agree with each other, because they one after another take actions according  to any of them and another one has to update its own knowledge, which looks like they introduce their own biases and noisy into each other, which on average gives no bias, because the biases are random, so it cancels out

This intuition is **close**, especially the idea that two estimates prevent an estimator from validating an action selected because of its own noise. But the precise mechanism is slightly different.

Write

$$Q_1(s,a)=q(s,a)+\varepsilon_1(a), \qquad Q_2(s,a)=q(s,a)+\varepsilon_2(a).$$

Ordinary Q-learning uses

$$\max_a Q_1(s,a) = Q_1\left(s,\arg\max_a Q_1(s,a)\right).$$

The selected action tends to have an unusually positive $\varepsilon_1(a)$, precisely because that noise helped it win the maximum. The same positive error is then used to evaluate it.

Double Q-learning instead uses, for example,

$$A^*=\arg\max_a Q_1(s,a), \qquad \text{target value}=Q_2(s,A^*).$$

Now $A^\ast$ may have been selected because $\varepsilon_1(A^\ast)$ was unusually positive, but its evaluation uses $\varepsilon_2(A^\ast)$, which has no reason to be positive merely because $\varepsilon_1(A^\ast)$ was positive. This breaks the selection–evaluation correlation responsible for maximization bias. 

### What is correct in your intuition

There are indeed two interleaved estimators. At each transition, one is chosen for updating:

$$Q_1(S,A)\leftarrow Q_1(S,A) +\alpha\left[ R+\gamma Q_2\left(S',\arg\max_a Q_1(S',a)\right)-Q_1(S,A) \right],$$

or symmetrically with $Q_1,Q_2$ exchanged. 

Thus each table sometimes selects the next action and the other table evaluates it.

### Three corrections

**They do not generally receive separate independent trajectories.**
Usually both tables observe the same stream of transitions, with each transition randomly assigned to update one table. Their errors are therefore not perfectly independent. The algorithm merely tries to keep them sufficiently decorrelated.

**The other table does not get updated.**
When updating $Q_1$, $Q_1$ selects the maximizing action, $Q_2$ evaluates it, but the entry actually changed is $Q_1(S,A)$. $Q_2$ remains fixed on that step.

**The biases do not cancel by averaging opposite signs.**
The mechanism is not

$$\text{positive bias}+\text{negative bias}=0.$$

Rather, the noise responsible for winning the maximization is not reused for evaluation. Conditional on the action selected by $Q_1$, the error from $Q_2$ is approximately mean-zero:

$$\mathbb E\left[\varepsilon_2(A^*)\mid A^* \text{ selected using }Q_1\right]\approx 0.$$

So the upward selection bias disappears. Double Q-learning can sometimes mildly underestimate; it is not guaranteed to be exactly unbiased at every finite stage.

The clean interpretation is:

$$\boxed{\text{Two estimators do not cancel noise; they decouple selection noise from evaluation noise.}}$$

They still cooperate through shared targets and through the behaviour policy, commonly chosen $\varepsilon$-greedily with respect to $Q_1+Q_2$, so over time both tables approach the same optimal action values. 

</details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Show how double Q-learning removes this problem</summary>

**Key idea:** Double Q-learning does not take a maximum of the estimates used for evaluation. It uses one estimator to choose the action and the other to evaluate it.

Write two estimates at the next state $s'$:

$$Q_1(s',a)=q_*(s',a)+\varepsilon_{1,a}, \qquad Q_2(s',a)=q_*(s',a)+\varepsilon_{2,a},$$

where the errors have mean zero and are approximately independent or at least decorrelated.

### Ordinary Q-learning

Ordinary Q-learning uses

$$\max_a Q_1(s',a).$$

Equivalently, define

$$A^*=\arg\max_a Q_1(s',a).$$

Then the value inserted into the target is

$$Q_1(s',A^*).$$

The problem is that $A^\ast$ was selected partly because $\varepsilon_{1,A^\ast}$ was unusually large. Therefore,

$$\mathbb E[\varepsilon_{1,A^*}]>0$$

in typical noisy settings. Selection and evaluation reuse the same favorable error.

---

### Double Q-learning

When updating $Q_1$, Double Q-learning instead uses

$$A^*=\arg\max_a Q_1(s',a)$$

for selection, but evaluates this action with $Q_2$:

$$Q_2(s',A^*).$$

Condition on the whole table $Q_1$. Then $A^\ast$ is fixed. Assuming $Q_2$'s error is independent of the selection noise,

$$
\begin{aligned}
\mathbb E\left[Q_2(s',A^*)\mid Q_1\right] &= \mathbb E\left[q_*(s',A^*)+\varepsilon_{2,A^*}\mid Q_1\right]\ &= q_*(s',A^*).
\end{aligned}
$$

The error that made the action win in $Q_1$ is not present in its evaluation by $Q_2$. Hence

$$\mathbb E\left[Q_2(s',A^*)\right] = \mathbb E\left[q_*(s',A^*)\right] \leq \max_a q_*(s',a).$$

So the systematic upward bias

$$\mathbb E[\max_a Q(s',a)] \geq \max_a q_*(s',a)$$

is gone.

### The two-action example

Suppose both actions have true value $0$, and both tables independently estimate each action as either $-1$ or $+1$, each with probability $1/2$.

Ordinary Q-learning gives

$$\mathbb E[\max{Q_1(a_1),Q_1(a_2)}] = \frac12.$$

Now let

$$A^*=\arg\max_a Q_1(a).$$

Double Q-learning evaluates using $Q_2(A^\ast)$. Since $Q_2$ is independent of which action happened to look best in $Q_1$,

$$\mathbb E[Q_2(A^*)\mid A^*]=0,$$

and therefore

$$\mathbb E[Q_2(A^*)]=0.$$

Thus:

$$\boxed{\text{ordinary Q-learning: expected target value }=\frac12}$$

while

$$\boxed{\text{Double Q-learning: expected target value }=0.}$$

The actual update alternates symmetrically:

$$Q_1(S,A)\leftarrow Q_1(S,A)+ \alpha\left[ R+\gamma Q_2\left(S',\arg\max_aQ_1(S',a)\right)-Q_1(S,A) \right],$$

or the same formula with $Q_1,Q_2$ exchanged. 

One nuance: Double Q-learning removes the **positive maximization bias**, but it can mildly underestimate because $Q_1$ may select a genuinely suboptimal action, which $Q_2$ then evaluates honestly. The noises do not cancel; the noise responsible for selection is simply not reused for evaluation.

</details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Why is the exception of Q is q^*? Is it guaranteed?</summary>

You mean **expectation**, and no: it is **not automatically guaranteed** that

$$\mathbb E[Q_i(s,a)] = q_*(s,a)$$

at every stage of Q-learning.

That equality was an **idealized assumption used to isolate maximization bias**.

**What the derivation assumes.**
Imagine that each action-value estimate has the form

$$Q_i(s,a)=q_*(s,a)+\varepsilon_{i,a}, \qquad \mathbb E[\varepsilon_{i,a}]=0.$$

Then immediately,

$$\mathbb E[Q_i(s,a)]=q_*(s,a).$$

This says that each action is estimated correctly **on average**, even though any particular estimate is noisy. The derivation then shows that maximization alone introduces bias:

$$\mathbb E\left[\max_a Q_i(s,a)\right] \geq \max_a q_*(s,a).$$

So the point is stronger than “Q-learning estimates are bad”: **even perfectly unbiased individual estimates become biased after taking their maximum**. This is the assumption made in the notes’ derivation. 

**Actual Q-learning estimates need not be unbiased.**

During learning, $Q_t(s,a)$ can be biased for many reasons:

* initialization;
* bootstrapping from other inaccurate estimates;
* changing targets as the policy improves;
* finite data;
* correlated samples.

Thus generally,

$$\mathbb E[Q_t(s,a)]\neq q_*(s,a)$$

for finite $t$.

What is guaranteed under the standard tabular convergence assumptions is an **asymptotic statement**:

$$Q_t(s,a)\longrightarrow q_*(s,a)$$

with probability $1$, provided, roughly, that every state-action pair is visited infinitely often, the step sizes satisfy the Robbins–Monro conditions, rewards are bounded, and the discounted or episodic MDP is well behaved.

This convergence is not the same as finite-time unbiasedness.

**What Double Q-learning actually needs for the clean argument.**

Suppose $Q_1$ selects

$$A^*=\arg\max_a Q_1(s,a),$$

and $Q_2$ evaluates it. The clean calculation assumes that $Q_2$'s evaluation error is mean-zero and independent of the information used to select $A^\ast$:

$$Q_2(s,a)=q_*(s,a)+\varepsilon_{2,a}.$$

Then

$$\mathbb E[\varepsilon_{2,A^*}\mid Q_1]=0,$$

so

$$\mathbb E[Q_2(s,A^*)\mid Q_1] = q_*(s,A^*).$$

Notice that this is generally **not**

$$q_*(s,a_*), \qquad a_*\in\arg\max_a q_*(s,a),$$

because $Q_1$ may select the wrong action. Therefore,

$$\mathbb E[Q_2(s,A^*)] = \mathbb E[q_*(s,A^*)] \leq \max_a q_*(s,a).$$

So Double Q-learning removes the **upward bias caused by reusing the winning noise**, but it may introduce some underestimation because the selecting estimator can choose a genuinely suboptimal action.

The precise conclusion is therefore:

$$\boxed{ \mathbb E[Q_i]=q_* \text{ is an analytical assumption, not a finite-time guarantee.} }$$

Double Q-learning’s real convergence result is asymptotic; the unbiased-noise model is used only to make the maximization-bias mechanism transparent.

</details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>In q-learning do we really have this bias in estimation? I mean I would expect it going to her over time. The error in coupled system could introduce bias in estimation, but asymptotically it should be similar to double q-learning, no?</summary>

Yes—your intuition is basically correct.

**Under the standard tabular Q-learning assumptions, the maximization bias is a finite-time phenomenon, not a different asymptotic limit.**

If

$$Q_t(s,a)\xrightarrow[t\to\infty]{a.s.}q_*(s,a)$$

for every state-action pair, then continuity of the finite maximum gives

$$\max_a Q_t(s,a)\xrightarrow[t\to\infty]{a.s.}\max_a q_*(s,a).$$

Hence the noise vanishes, and there is eventually no positive estimation error for the maximum to select. Ordinary Q-learning and Double Q-learning both converge to $q_\ast$, subject to the usual conditions: sufficient exploration, bounded rewards, and decreasing step sizes satisfying the stochastic-approximation conditions.

**Why then is maximization bias considered a problem?**

At finite time, write

$$Q_t(s,a)=q_*(s,a)+\varepsilon_{t,a}.$$

Even when each error is individually centered, the maximum tends to select a positive error:

$$\mathbb E!\left[\max_a Q_t(s,a)\right] \geq \max_a \mathbb E[Q_t(s,a)].$$

This inflated estimate is placed inside another Q-value’s bootstrap target:

$$R_{t+1}+\gamma\max_a Q_t(S_{t+1},a).$$

Thus the bias can propagate backward through the coupled Bellman updates. It may cause poor action choices, which then change what data are collected, slowing down correction. The example in your notes is exactly of this kind: Q-learning substantially prefers the suboptimal left action early in training, even though it would eventually recover under ideal convergence conditions. 

**Double Q-learning improves the path, not usually the final destination.**

Both algorithms aim at

$$q_*.$$

The difference is roughly

$$\begin{array}{c} \text{Q-learning: noisy and systematically optimistic route to }q_*, \\ \text{Double Q-learning: less optimistic route to }q_*. \end{array}$$

Double Q-learning can therefore learn a better policy much earlier and avoid wasting many samples on actions whose values were inflated by noise.

**The bias need not vanish in practical variants.**

The asymptotic equivalence relies on the estimation noise actually going to zero. That can fail or become irrelevant when:

* a constant step size is used, so estimates keep fluctuating;
* the environment is nonstationary;
* function approximation continually introduces approximation error;
* data are limited;
* deep Q-learning uses noisy targets and imperfect optimization.

In these regimes,

$$\operatorname{Var}(Q_t(s,a))\not\to 0,$$

so maximization can keep generating overestimation indefinitely. This is why Double DQN is useful even when training for a long time.

So the clean conclusion is

$$\boxed{\text{Tabular, diminishing-step-size case: same asymptotic }q_*,\text{ but different finite-time behavior.}}$$

The derivation in the notes explains a bias in the **current noisy bootstrap target**, not a claim that ordinary Q-learning necessarily converges to a permanently biased limit. 

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Double Q-learning is not generally unbiased)</span></p>

Double Q-Learning removes maximisation bias in expectation. Does this make its targets unbiased? Identify at least two remaining sources of error.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Answer</summary>

No. Double Q-learning removes the **systematic upward bias caused by using the same noise for selection and evaluation**, but its target is not generally unbiased for the optimal Bellman target.

When updating $Q_1$, define

$$A^\ast=\arg\max_a Q_1(S',a),$$

and use

$$Y^{\mathrm{Double}} = R+\gamma Q_2(S',A^\ast).$$

This is exactly the selection–evaluation split described in the notes. 

Assume, ideally, that $Q_2$ is independent of the selection noise in $Q_1$ and individually unbiased:

$$\mathbb E[Q_2(s',a)]=q_\ast(s',a).$$

Then, conditional on the selected action,

$$\mathbb E[Q_2(s',A^\ast)\mid A^\ast] = q_\ast(s',A^\ast).$$

Therefore,

$$\mathbb E[Q_2(s',A^\ast)] = \mathbb E[q_\ast(s',A^\ast)] \leq \max_a q_\ast(s',a).$$

Equality holds only when $Q_1$ selects an optimal action almost surely. Thus Double Q-learning removes the upward maximization bias, but it can instead **underestimate** because the selection table sometimes chooses a suboptimal action.

## Remaining sources of error

**1. Action-selection error**

Because $Q_1$ is imperfect,

$$\arg\max_a Q_1(s',a)$$

may not equal

$$\arg\max_a q_\ast(s',a).$$

The independent table $Q_2$ then accurately evaluates the wrong action. Double Q fixes biased **evaluation of the selected action**, not incorrect action selection.

**2. Error in the evaluation table**

$Q_2(s',a)$ is itself learned from finite data and bootstrapped targets. It may have:

* finite-sample error;
* initialization error;
* bootstrapping error;
* function-approximation error.

Independence prevents the selection error in $Q_1$ from preferentially selecting positive errors in $Q_2$, but it does not make $Q_2$ exact.

**3. Reward and transition sampling noise**

The target uses one sampled transition:

$$R+\gamma Q_2(S',A^\ast).$$

Even with perfect value estimates, random $R$ and $S'$ make this a noisy sample of the expected Bellman target. TD methods remain sample-based and bootstrapped. 

**4. Imperfect independence**

In practice, $Q_1$ and $Q_2$ are trained from the same environment stream and often start similarly, so their errors may be correlated. Then the theoretical cancellation of selection-induced optimism may be incomplete.

So the precise statement is:

$$\boxed{\text{Double Q removes one source of upward bias; it does not make the whole target unbiased or error-free.}}$$

The most important two remaining errors are **selecting a suboptimal action** and **inaccuracy in the independent evaluation estimate**.

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Maximization bias — Q-learning vs Double Q-learning)</span></p>

From state $A$: going **right** terminates immediately with reward $0$ (this is optimal). Going **left** leads to state $B$, from which $10$ different actions all give $\mathcal{N}(-0.1,\, 1)$ rewards — *negative in expectation*. The true optimum is therefore "always go right"; the only reason to ever go left is forced $\varepsilon$-exploration, whose floor is $\varepsilon/2 = 5\%$.

* **Q-learning is fooled:** the noisy $\mathcal{N}(-0.1, 1)$ rewards occasionally look positive, maximization bias inflates the value of going left, and the agent picks `left` about $65\%$ of the time early in training.
* **Double Q-learning** decouples selection from evaluation and stays close to the optimal $5\%$ throughout.

This is the cleanest demonstration that overestimation is not a cosmetic issue — it actively corrupts the learned policy.

</div>

### Summary

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(Method map — TD at a glance)</span></p>

| Method | Update target | Notes |
| :----- | :------------ | :---- |
| TD(0) prediction | $R\_{t+1} + \gamma V(S\_{t+1})$ | sample $+$ bootstrap, online |
| Sarsa | $R\_{t+1} + \gamma Q(S\_{t+1}, A\_{t+1})$ | on-policy |
| Q-learning | $R\_{t+1} + \gamma \max\_a Q(S\_{t+1}, a)$ | off-policy, learns greedy target |
| Double Q-learning | decouple selection & evaluation across $Q\_1, Q\_2$ | removes maximization bias |

Every row is the same skeleton $Q \leftarrow Q + \alpha\,[\,\text{target} - Q\,]$; only the target changes.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(Core takeaways — Temporal-Difference Learning)</span></p>

* **TD = sampling $+$ bootstrapping.** The defining hybrid of Monte Carlo (learn from raw experience) and Dynamic Programming (update toward an estimate built from estimates).
* **The TD error is the central object.** 
  * $\delta\_t = R\_{t+1} + \gamma V(S\_{t+1}) - V(S\_t)$; 
  * nearly every later algorithm is "define a TD error, then step on it."
* **Why TD is useful:** no model needed (vs DP), online and incremental (vs MC), often more data-efficient, and batch TD equals **certainty equivalence** on the empirical Markov model.
* **Control via action values.** **Sarsa** (on-policy) learns the value of the behaviour it follows, exploration included; **Q-learning** (off-policy) learns the value of the greedy target regardless of behaviour; **Double Q-learning** decouples selection from evaluation to remove maximization bias.
* **Final message.** TD methods are not merely RL algorithms — they are general-purpose tools for *learning long-term predictions from other predictions* in dynamical systems.

**Bridge ahead.** TD(0) is the shallowest possible bootstrap (one step) and MC the deepest (full return). The next step unifies them along the depth axis with **$n$-step TD** and **TD($\lambda$)**, and along the width axis with **function approximation**, carrying the TD-error machinery developed here into the function-approximation and deep-RL settings.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Does not TD operate in dynamical programming manner, i.e. to update the current $V(S_t)$, we need to know $V(S_{t+1})$? Why does it allow to make the updates faster without waiting the whole trajectory to end, if we still need the final state to send the updates  backwards in DP chain?</summary>

You are completely correct that TD bootstraps—it uses the value of the next state, $V(S_{t+1})$, to update the current state, $V(S_t)$. Your intuition about having to wait for the final state to send updates "backwards" perfectly describes how Monte Carlo (MC) works.

Here is why TD escapes that requirement and allows for immediate, online updates.

### The Secret: Learning a Guess from a Guess

The reason we do not have to wait for the trajectory to end is that TD does not look for the *true* final value of $S_{t+1}$. Instead, it simply looks up its **current, imperfect estimate** of $V(S_{t+1})$ that is already stored in memory.

Here is exactly how the timeline of a single step works in TD(0):

1. **Take a step:** The agent is in state $S_t$, takes an action, receives an immediate reward $R_{t+1}$, and lands in $S_{t+1}$.
2. **Look up the guess:** The episode is still ongoing, but the agent immediately looks at its value table and pulls its *current* estimated value for $V(S_{t+1})$.
3. **Form the target:** It creates the TD target: $R_{t+1} + \gamma V(S_{t+1})$.
4. **Update immediately:** It updates $V(S_t)$ toward that target right then and there.

By doing this, TD combines the **sampling** of Monte Carlo (taking one actual step in the world to get $R_{t+1}$) with the **bootstrapping** of Dynamic Programming (using an existing estimate $V$ instead of waiting for the real outcome).

### The "Driving Home" Analogy

To make this concrete, imagine you are predicting how long it takes to drive home from work.

* **The Monte Carlo Way:** You predict the trip will take 30 minutes. You hit a terrible traffic jam halfway through and arrive in 43 minutes. You have to wait until you are parked in your driveway to look back and update the values of every street you took.
* **The TD Way:** You are driving and suddenly get stuck behind a slow-moving truck. The moment you hit that delay, you generate a "TD error" (a surprise). You immediately update your estimate for the *previous* street you were just on, passing the surprise one step backwards. You do not need to arrive home to know that you are running late; you just update your current prediction based on your *next* prediction.

### Why does this actually work?

It feels like a house of cards to update a guess using another guess, but it is mathematically sound. Even though $V(S_{t+1})$ might be highly inaccurate early in training, the immediate reward $R_{t+1}$ is a real, grounded piece of data from the environment.

With a small step size, these local, step-by-step updates pull the estimates toward the true values over time. This makes TD incredibly powerful because **every single transition is a learning opportunity**, meaning you can learn from long episodes or even continuing tasks that never terminate, without ever having to wait for the end.

</details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>So, we use $V(S_{t+1})$ value from the previous iteration?</summary>

In online Temporal-Difference (TD) learning, we don't look back at a fixed table from a "previous iteration". Instead, tabular TD updates happen **in-place** and **online** after every single transition.

We maintain a single, live value table $V$ in memory. When the agent steps from $S_t$ into $S_{t+1}$, it reads **whatever value happens to be sitting in the $S_{t+1}$ slot of that table at that exact millisecond**.

### How the In-Place Memory Lookup Works

Because TD updates are online and asynchronous rather than sweeping the entire state space in structured blocks, there are no global "outer iterations". The values evolve dynamically based on the path the agent travels:

* **If $S_{t+1}$ was updated recently:** If the agent visited $S_{t+1}$ a few steps ago (or in a previous episode) and changed its value, the target for $S_t$ immediately benefits from that newly updated information.
* **If $S_{t+1}$ hasn't been visited yet:** The lookup simply pulls the original initialized value (or whatever baseline guess was set at the start).

### Contrast with Dynamic Programming

This is distinct from the synchronous **two-array update** style often used in textbook Dynamic Programming sweeps:

* **Synchronous DP sweeps:** You maintain an `old` array and a `new` array. Every state's new value reads *exclusively* from the frozen `old` values of the previous iteration.
* **Online TD:** You maintain a **single array**. New updates are written directly into the table and are instantly available to be read by any subsequent step.

By modifying the table in-place, surprise and reward information ripple through the value function much faster, leveraging the most up-to-date estimates available the moment a state transition occurs.

</details>
</div>

## Lecture 7: $n$-step Bootstrapping

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Setup</span><span class="math-callout__name">(Bridging Monte Carlo and one-step TD)</span></p>

The previous lecture set up the two extremes of model-free value estimation. 
* **Monte Carlo** waits for the *complete return* $G\_t$ before updating; 
* **one-step TD** updates immediately, after a single transition, by bootstrapping from $V(S\_{t+1})$. 

This lecture fills in everything *between* them: the family of **$n$-step methods**, which use the first $n$ real rewards and then bootstrap.

The build-up is:

1. **The spectrum.** Why neither pure MC nor pure one-step TD is usually best, and how $n$-step methods decouple *how often we act* from *how far we bootstrap*.
2. **$n$-step prediction.** The $n$-step return $G\_{t:t+n}$, the $n$-step TD update, and the **error-reduction property** that guarantees soundness.
3. **Control.** $n$-step **Sarsa** and **Expected Sarsa**.
4. **Off-policy.** Importance sampling over $n$ steps, and **control variates** that cut its variance.

The guiding question of the lecture is:

> *One-step TD bootstraps after a single step; Monte Carlo waits until the end. Why is the **best** answer almost always somewhere in between — and how do we get there?*

</div>

### The Space Between MC and TD

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Idea</span><span class="math-callout__name">(One target, three depths)</span></p>

All three methods update a state's value toward a *target*; they differ only in how many real rewards the target contains before it bootstraps:

* **Monte Carlo:** update toward the *full return* — all rewards until termination.
* **One-step TD:** update toward *one* reward plus the bootstrapped value of the next state.
* **$n$-step:** update toward the first $n$ rewards plus the bootstrapped value $n$ steps later.

Laid side by side, the targets form a continuum indexed by depth:

$$
\underbrace{R_{t+1} + \gamma V(S_{t+1})}_{\text{1-step TD}}
\;\;\cdots\;\;
\underbrace{R_{t+1} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^{n} V(S_{t+n})}_{n\text{-step}}
\;\;\cdots\;\;
\underbrace{R_{t+1} + \cdots + \gamma^{T-t-1} R_{T}}_{\text{Monte Carlo}}.
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">($n$-step methods are still TD methods)</span></p>

It is tempting to read $n$-step methods as "almost Monte Carlo". They are not — they remain **TD methods**: they change an earlier estimate based on a *later estimate*, just $n$ steps later instead of one. Monte Carlo is the single special case where the bootstrap term vanishes because the episode has ended. The bootstrapping machinery, the TD-error viewpoint, and the convergence arguments all carry over unchanged.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why one step is a compromise — decoupling acting from bootstrapping)</span></p>

With one-step TD a *single* time step is forced to control **two different things at once**:

* how often the agent can **change its action**, and
* the interval over which **bootstrapping** occurs.

These pull in opposite directions. We usually want to *act fast* — react to whatever just changed — but *bootstrap over a longer interval*, long enough for a meaningful state change to register. One step forces a compromise between the two.

**$n$-step methods break the tie:** the agent still acts every step, but bootstraps over $n$ steps. Acting frequency and bootstrap horizon become independent design choices.

</div>

### $n$-step TD Prediction

<figure class="rl-diagram">
  <svg viewBox="0 0 860 270" role="img" aria-label="Backup-diagram spectrum from one-step TD through n-step to Monte Carlo: the only thing that changes is the depth of the backup">
    <!-- 1-step TD : x=120 -->
    <line x1="120" y1="44" x2="120" y2="70" class="line"></line>
    <line x1="120" y1="70" x2="120" y2="96" class="line"></line>
    <circle cx="120" cy="44" r="11" class="box"></circle>
    <circle cx="120" cy="70" r="8" fill="#2c3e94"></circle>
    <circle cx="120" cy="96" r="11" class="box"></circle>
    <text x="120" y="124" text-anchor="middle" font-size="13" class="muted">1-step TD</text>

    <!-- 2-step : x=270 -->
    <line x1="270" y1="44" x2="270" y2="148" class="line"></line>
    <circle cx="270" cy="44" r="11" class="box"></circle>
    <circle cx="270" cy="70" r="8" fill="#2c3e94"></circle>
    <circle cx="270" cy="96" r="11" class="box"></circle>
    <circle cx="270" cy="122" r="8" fill="#2c3e94"></circle>
    <circle cx="270" cy="148" r="11" class="box"></circle>
    <text x="270" y="176" text-anchor="middle" font-size="13" class="muted">2-step</text>

    <!-- 3-step : x=420 -->
    <line x1="420" y1="44" x2="420" y2="200" class="line"></line>
    <circle cx="420" cy="44" r="11" class="box"></circle>
    <circle cx="420" cy="70" r="8" fill="#2c3e94"></circle>
    <circle cx="420" cy="96" r="11" class="box"></circle>
    <circle cx="420" cy="122" r="8" fill="#2c3e94"></circle>
    <circle cx="420" cy="148" r="11" class="box"></circle>
    <circle cx="420" cy="174" r="8" fill="#2c3e94"></circle>
    <circle cx="420" cy="200" r="11" class="box"></circle>
    <text x="420" y="228" text-anchor="middle" font-size="13" class="muted">3-step</text>

    <!-- n-step : x=580 -->
    <line x1="580" y1="44" x2="580" y2="96" class="line"></line>
    <line x1="580" y1="96" x2="580" y2="148" stroke="#64748b" stroke-width="2" stroke-dasharray="2 5"></line>
    <line x1="580" y1="148" x2="580" y2="200" class="line"></line>
    <circle cx="580" cy="44" r="11" class="box"></circle>
    <circle cx="580" cy="70" r="8" fill="#2c3e94"></circle>
    <circle cx="580" cy="96" r="11" class="box"></circle>
    <circle cx="580" cy="148" r="8" fill="#2c3e94"></circle>
    <circle cx="580" cy="174" r="11" class="box"></circle>
    <text x="580" y="202" text-anchor="middle" font-size="13" class="muted" font-style="italic">n-step</text>

    <!-- inf / MC : x=740 -->
    <line x1="740" y1="44" x2="740" y2="96" class="line"></line>
    <line x1="740" y1="96" x2="740" y2="174" stroke="#64748b" stroke-width="2" stroke-dasharray="2 5"></line>
    <line x1="740" y1="174" x2="740" y2="200" class="line"></line>
    <circle cx="740" cy="44" r="11" class="box"></circle>
    <circle cx="740" cy="70" r="8" fill="#2c3e94"></circle>
    <circle cx="740" cy="96" r="11" class="box"></circle>
    <circle cx="740" cy="148" r="8" fill="#2c3e94"></circle>
    <rect x="730" y="190" width="20" height="20" class="box"></rect>
    <text x="740" y="232" text-anchor="middle" font-size="13" class="muted">∞-step / MC</text>
  </svg>
  <figcaption>The backup-diagram spectrum. Open circles are states, filled dots are actions, the square is termination. From one-step TD to Monte Carlo the <em>only</em> thing that changes is the depth of the backup.</figcaption>
</figure>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($n$-step return)</span></p>

The target of an $n$-step update is the **$n$-step return**:

$$G_{t:t+n} \;\doteq\; R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^{n} V_{t+n-1}(S_{t+n}),$$

defined for $n \geq 1$ and $0 \leq t < T - n$. Here $V\_{t+n-1}$ denotes the value-function estimate *as it stands at time $t+n-1$* (estimates change over time, so the bootstrap uses the most recent one).

* The first $n$ rewards are **real**; everything after is replaced by the **bootstrap** $\gamma^{n} V\_{t+n-1}(S\_{t+n})$.
* If the return runs to or past termination ($t + n \geq T$), the missing terms are zero and $G\_{t:t+n} \doteq G\_t$ — the ordinary full return. So Monte Carlo is the limit $n \to \infty$.
* **Timing subtlety.** $G\_{t:t+n}$ needs $R\_{t+n}$ and $V\_{t+n-1}$, which only exist at time $t+n$. So **no update for $S\_t$ can be made until then.**

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">($n$-step TD prediction)</span></p>

The natural learning rule, applied when the data becomes available (at time $t + n$):

$$V_{t+n}(S_t) \;\doteq\; V_{t+n-1}(S_t) + \alpha\,\bigl[\, G_{t:t+n} - V_{t+n-1}(S_t) \,\bigr], \qquad 0 \leq t < T,$$

with all other states left unchanged ($V\_{t+n}(s) = V\_{t+n-1}(s)$ for $s \neq S\_t$). Two book-keeping consequences of the timing subtlety:

* No updates happen during the *first $n - 1$ steps* of an episode (the first return is not yet complete).
* To compensate, $n - 1$ extra updates are made *after* termination, before the next episode starts, so the last $n-1$ visited states still get updated.

**Special cases:** 
* $n = 1$ recovers one-step TD; 
* $n = \infty$ (or any $n \geq T$) recovers the constant-$\alpha$ Monte Carlo update.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(A Continuum from Bootstrapping to Full Returns)</span></p>

Show that the $n$-step return interpolates between one-step TD, corresponding to $n=1$, and Monte Carlo, corresponding to $n\geq T-t$.

Specifically, write out

$$G_{t+1}$$

and

$$G_{t}$$

explicitly and identify them as the usual TD target and the full episodic return $G_t$, respectively.

Then argue why $n$-step methods are still TD methods: they update an earlier estimate using a later estimate, only $n$ steps later.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

For $n=1$, the $n$-step return is $G_{t:t+1} = R_{t+1}+\gamma V(S_{t+1})$. This is the usual one-step TD target. The corresponding TD(0) update is

$$V(S_t) \leftarrow V(S_t) + \alpha \bigl( R_{t+1}+\gamma V(S_{t+1})-V(S_t) \bigr).$$

If $n\geq T-t$, then the backup reaches the terminal time and no bootstrap term remains. Hence

$$G_{t:T} = R_{t+1} +\gamma R_{t+2} +\cdots +\gamma^{T-t-1}R_T.$$

This is the full episodic Monte Carlo return $G_t$. Thus the $n$-step return interpolates between one-step TD and Monte Carlo:

$$
\begin{aligned}
\underbrace{R_{t+1}+\gamma V(S_{t+1})}_{\text{one-step TD}}
&\quad\longrightarrow\quad \underbrace{ R_{t+1} +\cdots +\gamma^{n-1}R_{t+n} +\gamma^n V(S_{t+n}) }_{n\text{-step TD}} \\
&\quad\longrightarrow\quad \underbrace{ R_{t+1} +\cdots +\gamma^{T-t-1}R_T}_{\text{Monte Carlo}}.
\end{aligned}
$$

The method is still a temporal-difference method whenever the bootstrap term is present. The estimate at the earlier state $S_t$ is updated using the later estimate $V(S_{t+n})$. Compared with TD(0), the only difference is that the later estimate is used after $n$ observed rewards rather than after one observed reward.

</details>
</div>

### Why It Works: The Error-Reduction Property

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Error-reduction property of the $n$-step return)</span></p>

The expected $n$-step return is a **contraction toward $v\_\pi$** in the worst-state (sup-norm) sense:

$$\max_s \bigl|\, \mathbb{E}_\pi[\, G_{t:t+n} \mid S_t = s \,] - v_\pi(s) \,\bigr| \;\leq\; \gamma^{n} \max_s \bigl|\, V_{t+n-1}(s) - v_\pi(s) \,\bigr|.$$

In words: the worst error of the *expected* $n$-step return is at most $\gamma^{n}$ times the worst error of the current estimate. Since $\gamma^{n} < 1$, repeated application shrinks the error.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

By iterating the Bellman expectation equation $n$ times we obtain

$$v_\pi(s) = \mathbb E_\pi\! \left[ R_{t+1} +\gamma R_{t+2} +\cdots +\gamma^{n-1}R_{t+n} +\gamma^n v_\pi(S_{t+n}) \,\middle|\, S_t=s \right].$$

Subtracting this identity from the expected $n$-step return gives

$$
\begin{aligned}
&\mathbb E_\pi[G_{t:t+n}\mid S_t=s]-v_\pi(s)\\
&\qquad = \mathbb E_\pi\! \left[ \gamma^n \bigl( V(S_{t+n})-v_\pi(S_{t+n}) \bigr) \,\middle|\, S_t=s \right].
\end{aligned}
$$

Therefore

$$
\begin{aligned}
\left| \mathbb E_\pi[G_{t:t+n}\mid S_t=s]-v_\pi(s) \right| &\leq \gamma^n \mathbb E_\pi\! \left[ \left| V(S_{t+n})-v_\pi(S_{t+n}) \right| \,\middle|\, S_t=s \right] \\
&\leq \gamma^n \max_{s'} |V(s')-v_\pi(s')|.
\end{aligned}
$$

Taking the maximum over $s$ gives

$$\max_s \left| \mathbb E_\pi[G_{t:t+n}\mid S_t=s]-v_\pi(s) \right| \leq \gamma^n \max_s |V(s)-v_\pi(s)|.$$

Since $\gamma^n<1$, the expected $n$-step target is closer to $v_\pi$ than the current estimate $V$ in the supremum norm. Hence the expected $n$-step backup is a contraction toward $v_\pi$. With the usual stochastic approximation assumptions on the step sizes and sufficient state visitation, this contraction property explains the convergence of $n$-step TD prediction to $v_\pi$.

</details>
</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Takeaway</span><span class="math-callout__name">(All $n$-step methods are sound)</span></p>

Because of error reduction, **all $n$-step TD methods converge to $v\_\pi$** under the usual conditions (in expectation, with appropriate step sizes). One-step TD and Monte Carlo are not special — they are just the two extreme members of a single sound family.

</div>

### The Empirical Story: Bias, Variance, and the Best $n$

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(5-state random walk — credit travels $n$ states)</span></p>

Consider a short chain 

$$0 \leftrightarrow A \leftrightarrow B \leftrightarrow C \leftrightarrow D \leftrightarrow E \leftrightarrow 1,$$

starting in $C$, with reward $+1$ only on terminating at the right end. Suppose the first episode goes $C \to D \to E \to$ terminate-right, return $+1$, with all values initialised at $0.5$.

* **1-step:** only $V(E)$ moves toward $1$ — credit travels just one state.
* **2-step:** both $V(D)$ and $V(E)$ move toward $1$.
* **$n$-step ($n > 2$):** all three visited states ($C, D, E$) move toward $1$, equally.

**Intuition:** larger $n$ propagates the outcome back to *more* states from a single episode — but, as the next example shows, it can also over-commit to a noisy return.

</div>

<figure class="rl-diagram">
  <svg viewBox="0 0 860 296" role="img" aria-label="On a five-state random walk, after the episode C to D to E to the right terminal, one-step TD updates only state E, two-step updates D and E, and n-step with n at least three updates C, D and E equally">
    <defs>
      <marker id="rw-traj" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
        <path d="M0,0 L0,6 L8,3 z" fill="#b45309"></path>
      </marker>
    </defs>

    <!-- backbone -->
    <line x1="80" y1="70" x2="800" y2="70" class="line"></line>

    <!-- terminals + states -->
    <rect x="56" y="46" width="48" height="48" rx="6" class="box"></rect>
    <text x="80" y="77" text-anchor="middle" font-size="17" font-weight="700">0</text>
    <circle cx="200" cy="70" r="24" class="box"></circle>
    <text x="200" y="76" text-anchor="middle" font-size="17" font-weight="700">A</text>
    <circle cx="320" cy="70" r="24" class="box"></circle>
    <text x="320" y="76" text-anchor="middle" font-size="17" font-weight="700">B</text>
    <circle cx="440" cy="70" r="24" class="box"></circle>
    <text x="440" y="76" text-anchor="middle" font-size="17" font-weight="700">C</text>
    <circle cx="560" cy="70" r="24" class="box"></circle>
    <text x="560" y="76" text-anchor="middle" font-size="17" font-weight="700">D</text>
    <circle cx="680" cy="70" r="24" class="box"></circle>
    <text x="680" y="76" text-anchor="middle" font-size="17" font-weight="700">E</text>
    <rect x="776" y="46" width="48" height="48" rx="6" class="green"></rect>
    <text x="800" y="77" text-anchor="middle" font-size="17" font-weight="700">1</text>
    <text x="800" y="30" text-anchor="middle" font-size="13" font-weight="700" fill="#047857">+1</text>
    <text x="440" y="112" text-anchor="middle" font-size="12" class="muted">start</text>

    <!-- trajectory C -> D -> E -> terminal -->
    <path d="M462 56 C500 30 520 30 558 50" class="strong-line" stroke="#b45309" marker-end="url(#rw-traj)"></path>
    <path d="M582 56 C620 30 640 30 678 50" class="strong-line" stroke="#b45309" marker-end="url(#rw-traj)"></path>
    <path d="M702 56 C740 30 760 32 776 48" class="strong-line" stroke="#b45309" marker-end="url(#rw-traj)"></path>

    <!-- credit rows -->
    <text x="40" y="160" font-size="13" font-weight="700">1-step</text>
    <text x="40" y="210" font-size="13" font-weight="700">2-step</text>
    <text x="40" y="260" font-size="13" font-weight="700">n ≥ 3</text>

    <!-- row 1: only E -->
    <circle cx="200" cy="155" r="11" class="box"></circle>
    <circle cx="320" cy="155" r="11" class="box"></circle>
    <circle cx="440" cy="155" r="11" class="box"></circle>
    <circle cx="560" cy="155" r="11" class="box"></circle>
    <circle cx="680" cy="155" r="11" fill="#047857"></circle>
    <!-- row 2: D, E -->
    <circle cx="200" cy="205" r="11" class="box"></circle>
    <circle cx="320" cy="205" r="11" class="box"></circle>
    <circle cx="440" cy="205" r="11" class="box"></circle>
    <circle cx="560" cy="205" r="11" fill="#047857"></circle>
    <circle cx="680" cy="205" r="11" fill="#047857"></circle>
    <!-- row 3: C, D, E -->
    <circle cx="200" cy="255" r="11" class="box"></circle>
    <circle cx="320" cy="255" r="11" class="box"></circle>
    <circle cx="440" cy="255" r="11" fill="#047857"></circle>
    <circle cx="560" cy="255" r="11" fill="#047857"></circle>
    <circle cx="680" cy="255" r="11" fill="#047857"></circle>

    <text x="745" y="160" font-size="12" class="muted">updated</text>
    <text x="745" y="260" font-size="12" class="muted">toward 1</text>
  </svg>
  <figcaption>One episode $C \to D \to E \to$ right-terminal (reward $+1$). Filled green = a state whose value moves toward $1$ after this single episode. Deeper backups (<em>larger $n$</em>) push the outcome's credit further back along the visited path — one extra state per extra step.</figcaption>
</figure>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(19-state random walk — intermediate $n$ wins)</span></p>

Running $n$-step TD directly on a 19-state random walk and plotting RMS error against step size $\alpha$ for each fixed $n$ gives a family of U-shaped curves:

* **Intermediate $n$ wins.** The lowest error is achieved around $n \approx 4$ — not at $n = 1$ (one-step TD) nor at $n = \infty$ (Monte Carlo).
* **Larger $n$ needs a smaller $\alpha$.** Deeper backups carry higher-variance targets, so they must take smaller steps to stay stable.

The generalization genuinely **outperforms both** extremes — this is the empirical payoff for decoupling the bootstrap horizon from the acting frequency.

</div>

<figure class="rl-diagram">
  <svg viewBox="0 0 860 412" role="img" aria-label="Average RMS error versus step size alpha for n-step TD on the 19-state random walk. Each curve is a fixed n; all are U-shaped. Intermediate n around 4 reaches the lowest error; larger n bottoms out at smaller alpha and then diverges.">
    <!-- gridlines -->
    <line x1="85" y1="45" x2="85" y2="355" class="line"></line>
    <line x1="85" y1="355" x2="775" y2="355" class="line"></line>
    <line x1="85" y1="45" x2="775" y2="45" stroke="#dbe1ee" stroke-width="1" stroke-dasharray="2 4"></line>
    <line x1="85" y1="148.3" x2="775" y2="148.3" stroke="#dbe1ee" stroke-width="1" stroke-dasharray="2 4"></line>
    <line x1="85" y1="251.7" x2="775" y2="251.7" stroke="#dbe1ee" stroke-width="1" stroke-dasharray="2 4"></line>

    <!-- y ticks (RMS error) -->
    <text x="78" y="50" text-anchor="end" font-size="12" class="muted">0.55</text>
    <text x="78" y="153" text-anchor="end" font-size="12" class="muted">0.45</text>
    <text x="78" y="256" text-anchor="end" font-size="12" class="muted">0.35</text>
    <text x="78" y="359" text-anchor="end" font-size="12" class="muted">0.25</text>
    <!-- x ticks (alpha) -->
    <text x="85" y="374" text-anchor="middle" font-size="12" class="muted">0</text>
    <text x="223" y="374" text-anchor="middle" font-size="12" class="muted">0.2</text>
    <text x="361" y="374" text-anchor="middle" font-size="12" class="muted">0.4</text>
    <text x="499" y="374" text-anchor="middle" font-size="12" class="muted">0.6</text>
    <text x="637" y="374" text-anchor="middle" font-size="12" class="muted">0.8</text>
    <text x="775" y="374" text-anchor="middle" font-size="12" class="muted">1.0</text>
    <text x="430" y="398" text-anchor="middle" font-size="13">α (step size)</text>
    <text x="26" y="200" text-anchor="middle" font-size="13" transform="rotate(-90 26 200)">Avg. RMS error (first 10 episodes)</text>

    <!-- curves -->
    <polyline fill="none" stroke="#94a3b8" stroke-width="2" points="96.5,55.3 108.0,62.8 119.5,70.1 131.0,77.0 142.5,83.7 154.0,90.3 165.5,96.7 177.0,102.9 188.5,108.9 200.0,114.8 211.5,120.9 223.0,126.5 234.5,132.2 246.0,137.7 257.5,143.3 269.0,148.2 280.5,153.5 292.0,158.7 303.5,163.9 315.0,168.5 326.5,173.0 338.0,178.1 349.5,182.8 361.0,187.2 372.5,192.3 384.0,196.3 395.5,200.3 407.0,204.4 418.5,208.7 430.0,212.1 441.5,216.7 453.0,220.4 464.5,223.3 476.0,227.1 487.5,230.0 499.0,233.7 510.5,236.5 522.0,239.8 533.5,242.7 545.0,245.1 556.5,247.7 568.0,248.7 579.5,251.1 591.0,252.5 602.5,254.6 614.0,256.5 625.5,255.9 637.0,258.5 648.5,258.0 660.0,256.6 671.5,255.7 683.0,255.5 694.5,253.4 706.0,249.5 717.5,248.0 729.0,243.6 740.5,239.2 752.0,233.0 763.5,225.1 775.0,211.1"></polyline>
    <polyline fill="none" stroke="#93c5fd" stroke-width="2" points="96.5,62.4 108.0,76.6 119.5,90.2 131.0,103.2 142.5,115.8 154.0,128.0 165.5,139.8 177.0,151.2 188.5,162.3 200.0,173.2 211.5,183.9 223.0,193.5 234.5,203.4 246.0,213.4 257.5,222.1 269.0,231.0 280.5,239.4 292.0,248.9 303.5,257.0 315.0,265.0 326.5,271.6 338.0,278.6 349.5,286.1 361.0,292.8 372.5,296.8 384.0,302.1 395.5,308.7 407.0,312.1 418.5,317.2 430.0,320.1 441.5,322.0 453.0,323.4 464.5,324.1 476.0,325.6 487.5,326.2 499.0,328.2 510.5,326.5 522.0,325.8 533.5,323.7 545.0,320.3 556.5,318.7 568.0,314.7 579.5,310.2 591.0,306.7 602.5,300.7 614.0,297.9 625.5,291.5 637.0,287.0 648.5,279.3 660.0,276.0 671.5,263.1 683.0,257.6 694.5,250.3 706.0,238.3 717.5,228.0 729.0,219.9 740.5,207.4 752.0,194.6 763.5,178.6 775.0,160.9"></polyline>
    <polyline fill="none" stroke="#2563eb" stroke-width="2" points="96.5,94.8 108.0,136.2 119.5,172.4 131.0,203.8 142.5,230.1 154.0,256.2 165.5,275.0 177.0,289.4 188.5,303.5 200.0,316.0 211.5,321.2 223.0,321.9 234.5,328.8 246.0,329.3 257.5,326.2 269.0,328.4 280.5,322.3 292.0,316.7 303.5,319.0 315.0,310.9 326.5,298.2 338.0,298.9 349.5,291.5 361.0,285.2 372.5,278.2 384.0,270.9 395.5,267.2 407.0,256.5 418.5,255.7 430.0,244.5 441.5,239.5 453.0,235.6 464.5,228.8 476.0,219.6 487.5,208.3 499.0,205.1 510.5,195.0 522.0,195.8 533.5,184.3 545.0,178.8 556.5,171.9 568.0,164.1 579.5,153.0 591.0,152.1 602.5,145.3 614.0,136.9 625.5,125.1 637.0,122.9 648.5,113.6 660.0,108.7 671.5,94.6 683.0,84.8 694.5,84.2 706.0,71.7 717.5,62.5 729.0,52.6"></polyline>
    <polyline fill="none" stroke="#1e3a8a" stroke-width="2" points="96.5,119.8 108.0,175.6 119.5,216.2 131.0,246.8 142.5,266.7 154.0,283.8 165.5,287.3 177.0,294.4 188.5,291.6 200.0,284.5 211.5,278.7 223.0,279.0 234.5,273.1 246.0,259.6 257.5,251.4 269.0,245.0 280.5,236.0 292.0,233.6 303.5,223.6 315.0,211.2 326.5,204.4 338.0,200.9 349.5,185.3 361.0,184.1 372.5,176.3 384.0,168.4 395.5,159.1 407.0,151.7 418.5,143.8 430.0,140.8 441.5,132.3 453.0,122.1 464.5,114.3 476.0,114.3 487.5,107.4 499.0,97.3 510.5,92.7 522.0,85.9 533.5,76.3 545.0,70.3 556.5,68.0 568.0,56.5 579.5,54.8 591.0,46.1"></polyline>
    <!-- n = 4 : the winner, drawn last and thicker -->
    <polyline fill="none" stroke="#b91c1c" stroke-width="3" points="96.5,74.9 108.0,100.5 119.5,124.2 131.0,146.5 142.5,167.6 154.0,187.3 165.5,205.2 177.0,222.9 188.5,239.3 200.0,253.9 211.5,267.5 223.0,281.6 234.5,291.6 246.0,301.3 257.5,311.6 269.0,318.6 280.5,323.8 292.0,329.7 303.5,332.5 315.0,337.2 326.5,339.9 338.0,341.4 349.5,340.7 361.0,338.9 372.5,341.1 384.0,334.1 395.5,335.2 407.0,330.7 418.5,324.3 430.0,326.0 441.5,318.3 453.0,316.1 464.5,312.9 476.0,304.3 487.5,299.8 499.0,297.6 510.5,290.7 522.0,280.3 533.5,275.0 545.0,268.4 556.5,263.4 568.0,255.0 579.5,251.1 591.0,241.8 602.5,235.7 614.0,229.6 625.5,220.6 637.0,209.4 648.5,201.9 660.0,192.7 671.5,187.0 683.0,176.7 694.5,167.9 706.0,156.7 717.5,147.3 729.0,135.1 740.5,120.2 752.0,112.4 763.5,97.4 775.0,82.4"></polyline>

    <!-- lowest-error marker -->
    <circle cx="338" cy="341.4" r="4.5" fill="#b91c1c"></circle>
    <line x1="338" y1="341.4" x2="392" y2="318" stroke="#b91c1c" stroke-width="1"></line>
    <text x="396" y="315" font-size="12" font-weight="700" fill="#b91c1c">lowest error (n ≈ 4)</text>

    <!-- legend -->
    <rect x="624" y="52" width="146" height="118" rx="8" class="box"></rect>
    <line x1="636" y1="70" x2="664" y2="70" stroke="#94a3b8" stroke-width="2"></line>
    <text x="672" y="74" font-size="12">n = 1</text>
    <line x1="636" y1="90" x2="664" y2="90" stroke="#93c5fd" stroke-width="2"></line>
    <text x="672" y="94" font-size="12">n = 2</text>
    <line x1="636" y1="110" x2="664" y2="110" stroke="#b91c1c" stroke-width="3"></line>
    <text x="672" y="114" font-size="12" font-weight="700">n = 4</text>
    <line x1="636" y1="130" x2="664" y2="130" stroke="#2563eb" stroke-width="2"></line>
    <text x="672" y="134" font-size="12">n = 8</text>
    <line x1="636" y1="150" x2="664" y2="150" stroke="#1e3a8a" stroke-width="2"></line>
    <text x="672" y="154" font-size="12">n = 16</text>
  </svg>
  <figcaption>Reproduced by running $n$-step TD directly on the 19-state random walk (600 runs, first 10 episodes, values initialised to $0$). Each curve fixes $n$ and sweeps the step size $\alpha$. Every curve is U-shaped; <strong>$n \approx 4$ attains the lowest error</strong>. Larger $n$ bottoms out at a <em>smaller</em> $\alpha$ and then diverges as the high-variance targets destabilise — the curves leave the top of the band. Neither extreme ($n=1$ TD, $n=\infty$ MC) is best.</figcaption>
</figure>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">($n$ is a bias–variance dial)</span></p>

The whole spectrum is one bias–variance trade-off. **Small $n$** leans on the bootstrap (more bias from a possibly-wrong $V$, but low variance). **Large $n$** uses more real, sampled reward (less bias, but high variance because long reward sequences are noisy). The U-shaped curves are exactly what a bias–variance trade-off looks like; the best $n$ is the bottom of the U, problem-dependent and usually small but greater than one.

</div>

### $n$-step Sarsa (Control)

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Idea</span><span class="math-callout__name">(From prediction to control — switch to action values)</span></p>

Moving from prediction ($v\_\pi$) to control ($q\_\pi$) repeats exactly the move made for one-step TD when we went from TD(0) to **Sarsa**: switch from states to **state–action pairs**, act $\varepsilon$-greedily, and bootstrap from an action value. The $n$-step return for action values is

$$G_{t:t+n} \;\doteq\; R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^{n} Q_{t+n-1}(S_{t+n}, A_{t+n}),$$

and the update is **$n$-step Sarsa**:

$$Q_{t+n}(S_t, A_t) \;\doteq\; Q_{t+n-1}(S_t, A_t) + \alpha\,\bigl[\, G_{t:t+n} - Q_{t+n-1}(S_t, A_t) \,\bigr].$$

As with prediction, $n = 1$ recovers ordinary Sarsa(0) and $n = \infty$ recovers Monte Carlo control.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($n$-step Sarsa — the core idea)</span></p>

Read against ordinary Sarsa, which updates after **one** transition,

$$G = R_{t+1} + \gamma\, Q(S_{t+1}, A_{t+1}),$$

$n$-step Sarsa simply **waits longer and uses more real rewards** before it bootstraps,

$$G = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^{n} Q(S_{t+n}, A_{t+n}),$$

and then applies the *same* Sarsa update to the original pair:

$$Q(S_t, A_t) \;\leftarrow\; Q(S_t, A_t) + \alpha\,\bigl[\, G - Q(S_t, A_t) \,\bigr].$$

It is the same algorithm — only the return is longer.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Why $n$-step control learns faster)</span></p>

Picture a gridworld episode that wanders to the goal $G$, where reaching $G$ is the *only* reward. After that single episode:

* **One-step Sarsa** strengthens just the **last** action — the one that stepped into the goal. Every earlier action along the path is untouched; their values only improve on later episodes, one step further back each time.
* **10-step Sarsa** strengthens the **last ten** actions at once — so far more is learned from the same single trajectory.

Deeper backups propagate the goal's credit many steps back per episode, which is exactly why intermediate $n$ accelerates control learning.

</div>

<figure class="rl-diagram">
  <svg viewBox="0 0 860 300" role="img" aria-label="Three gridworld panels showing one path from start to goal G. One-step Sarsa strengthens only the final action; ten-step Sarsa strengthens the last ten actions of the same trajectory.">
    <defs>
      <marker id="gw-arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto" markerUnits="strokeWidth">
        <path d="M0,0 L0,6 L7,3 z" fill="#b45309"></path>
      </marker>
    </defs>

    <!-- panel titles -->
    <text x="132" y="64" text-anchor="middle" font-size="14" font-weight="700">Path taken (one episode)</text>
    <text x="417" y="64" text-anchor="middle" font-size="14" font-weight="700">1-step Sarsa</text>
    <text x="702" y="64" text-anchor="middle" font-size="14" font-weight="700">10-step Sarsa</text>

    <!-- ===== Panel 1 ===== -->
    <g stroke="#e2e8f0" stroke-width="1">
      <line x1="48" y1="87" x2="48" y2="255"></line><line x1="76" y1="87" x2="76" y2="255"></line><line x1="104" y1="87" x2="104" y2="255"></line><line x1="132" y1="87" x2="132" y2="255"></line><line x1="160" y1="87" x2="160" y2="255"></line><line x1="188" y1="87" x2="188" y2="255"></line><line x1="216" y1="87" x2="216" y2="255"></line>
      <line x1="48" y1="255" x2="216" y2="255"></line><line x1="48" y1="227" x2="216" y2="227"></line><line x1="48" y1="199" x2="216" y2="199"></line><line x1="48" y1="171" x2="216" y2="171"></line><line x1="48" y1="143" x2="216" y2="143"></line><line x1="48" y1="115" x2="216" y2="115"></line><line x1="48" y1="87" x2="216" y2="87"></line>
    </g>
    <rect x="204" y="75" width="24" height="24" rx="3" class="green"></rect>
    <text x="216" y="92" text-anchor="middle" font-size="12" font-weight="700" fill="#065f46">G</text>
    <polyline fill="none" stroke="#94a3b8" stroke-width="2.5" points="48,255 76,255 76,227 104,227 104,199 132,199 132,171 160,171 160,143 188,143 188,115 216,115 216,87"></polyline>
    <circle cx="48" cy="255" r="5" fill="#1f2933"></circle>

    <!-- ===== Panel 2 ===== -->
    <g stroke="#e2e8f0" stroke-width="1">
      <line x1="333" y1="87" x2="333" y2="255"></line><line x1="361" y1="87" x2="361" y2="255"></line><line x1="389" y1="87" x2="389" y2="255"></line><line x1="417" y1="87" x2="417" y2="255"></line><line x1="445" y1="87" x2="445" y2="255"></line><line x1="473" y1="87" x2="473" y2="255"></line><line x1="501" y1="87" x2="501" y2="255"></line>
      <line x1="333" y1="255" x2="501" y2="255"></line><line x1="333" y1="227" x2="501" y2="227"></line><line x1="333" y1="199" x2="501" y2="199"></line><line x1="333" y1="171" x2="501" y2="171"></line><line x1="333" y1="143" x2="501" y2="143"></line><line x1="333" y1="115" x2="501" y2="115"></line><line x1="333" y1="87" x2="501" y2="87"></line>
    </g>
    <rect x="489" y="75" width="24" height="24" rx="3" class="green"></rect>
    <text x="501" y="92" text-anchor="middle" font-size="12" font-weight="700" fill="#065f46">G</text>
    <polyline fill="none" stroke="#cbd5e1" stroke-width="2.5" points="333,255 361,255 361,227 389,227 389,199 417,199 417,171 445,171 445,143 473,143 473,115 501,115 501,87"></polyline>
    <polyline fill="none" stroke="#b45309" stroke-width="4" points="501,115 501,87" marker-end="url(#gw-arrow)"></polyline>
    <circle cx="333" cy="255" r="5" fill="#1f2933"></circle>

    <!-- ===== Panel 3 ===== -->
    <g stroke="#e2e8f0" stroke-width="1">
      <line x1="618" y1="87" x2="618" y2="255"></line><line x1="646" y1="87" x2="646" y2="255"></line><line x1="674" y1="87" x2="674" y2="255"></line><line x1="702" y1="87" x2="702" y2="255"></line><line x1="730" y1="87" x2="730" y2="255"></line><line x1="758" y1="87" x2="758" y2="255"></line><line x1="786" y1="87" x2="786" y2="255"></line>
      <line x1="618" y1="255" x2="786" y2="255"></line><line x1="618" y1="227" x2="786" y2="227"></line><line x1="618" y1="199" x2="786" y2="199"></line><line x1="618" y1="171" x2="786" y2="171"></line><line x1="618" y1="143" x2="786" y2="143"></line><line x1="618" y1="115" x2="786" y2="115"></line><line x1="618" y1="87" x2="786" y2="87"></line>
    </g>
    <rect x="774" y="75" width="24" height="24" rx="3" class="green"></rect>
    <text x="786" y="92" text-anchor="middle" font-size="12" font-weight="700" fill="#065f46">G</text>
    <polyline fill="none" stroke="#cbd5e1" stroke-width="2.5" points="618,255 646,255 646,227 674,227 674,199 702,199 702,171 730,171 730,143 758,143 758,115 786,115 786,87"></polyline>
    <polyline fill="none" stroke="#b45309" stroke-width="4" points="646,227 674,227 674,199 702,199 702,171 730,171 730,143 758,143 758,115 786,115 786,87" marker-end="url(#gw-arrow)"></polyline>
    <circle cx="618" cy="255" r="5" fill="#1f2933"></circle>

    <text x="430" y="285" text-anchor="middle" font-size="12.5" class="muted">● start &nbsp;&nbsp; ■ goal (only reward) &nbsp;&nbsp; <tspan fill="#b45309" font-weight="700">orange</tspan> = action values strengthened by this episode</text>
  </svg>
  <figcaption>A single trajectory reaches the goal $G$ (the only reward). <strong>One-step Sarsa</strong> strengthens just the <em>last</em> action that entered the goal; <strong>10-step Sarsa</strong> strengthens the last ten actions at once — so a single episode teaches far more of the path, which is why deeper backups speed up control.</figcaption>
</figure>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">($n$-step Sarsa — on-policy TD control)</span></p>

**Parameters:** step size $\alpha \in (0, 1]$, exploration $\varepsilon > 0$, integer $n \geq 1$.

Initialise $Q(s, a)$ arbitrarily, with $Q(\text{terminal}, \cdot) = 0$.

Loop for each episode:

1. Initialise and store $S\_0 \neq$ terminal; 
2. Choose and store $A\_0 \sim \varepsilon\text{-greedy}(\cdot \mid S\_0)$.
3. Set $T \leftarrow \infty$.
4. Loop $t = 0, 1, 2, \dots$:
   * **If $t < T$:** take action $A\_t$; observe and store $R\_{t+1}, S\_{t+1}$.
     * If $S\_{t+1}$ is terminal: 
       * $T \leftarrow t + 1$.
     * Else: choose and store 
       * $A\_{t+1} \sim \varepsilon\text{-greedy}(\cdot \mid S\_{t+1})$.
   * Let $\tau \leftarrow t - n + 1$ ($\tau$ is the time of the estimate now being updated).
   * **If $\tau \geq 0$:**
     
     $$G \leftarrow \sum_{i=\tau+1}^{\min(\tau+n,\, T)} \gamma^{\,i-\tau-1} R_i;$$
     
     if $\tau + n < T$, add the bootstrap $G \leftarrow G + \gamma^{n} Q(S_{\tau+n}, A_{\tau+n})$; then
     
     $$Q(S_\tau, A_\tau) \leftarrow Q(S_\tau, A_\tau) + \alpha\,\bigl[\, G - Q(S_\tau, A_\tau) \,\bigr].$$
     
   * Until $\tau = T - 1$.

The guard `if $t < T$` stops *sampling* once the episode ends, while the loop keeps *issuing updates* until $\tau = T - 1$, so the last $n - 1$ visited pairs are still updated.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($n$-step Expected Sarsa)</span></p>

Same update, but the **last** node is a full expectation over the target policy rather than a single sampled action:

$$G_{t:t+n} \;\doteq\; R_{t+1} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^{n} \bar V_{t+n-1}(S_{t+n}),$$

where the **expected approximate value** is

$$\bar V_{t+n-1}(S_{t+n}) \;\doteq\; \sum_a \pi(a \mid S_{t+n})\, Q_{t+n-1}(S_{t+n}, a).$$

The terminal branch averages over *all* actions, weighted by $\pi$. This has **lower variance** than Sarsa's sampled last step; the quantity $\bar V$ reappears in several later contexts (notably the off-policy control variates below).

</div>

### $n$-step Off-policy Learning

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Off-policy $n$-step learning via importance sampling)</span></p>

To learn about a target policy $\pi$ while following a behaviour policy $b$, we reweight each $n$-step return by the **importance-sampling ratio** — the product of per-step action-probability ratios over the segment:

$$\rho_{t:h} \;\doteq\; \prod_{k=t}^{\min(h,\, T-1)} \frac{\pi(A_k \mid S_k)}{b(A_k \mid S_k)},$$

where $T$ is the terminal time step and $h$ is the last action index included in the product. The off-policy updates simply premultiply the TD error by the appropriate ratio.

**Off-policy $n$-step TD (state values):**

$$V_{t+n}(S_t) \;\doteq\; V_{t+n-1}(S_t) + \alpha\, \rho_{t:t+n-1}\,\bigl[\, G_{t:t+n} - V_{t+n-1}(S_t) \,\bigr].$$

**Off-policy $n$-step Sarsa (action values):**

$$Q_{t+n}(S_t, A_t) \;\doteq\; Q_{t+n-1}(S_t, A_t) + \alpha\, \rho_{t+1:t+n}\,\bigl[\, G_{t:t+n} - Q_{t+n-1}(S_t, A_t) \,\bigr].$$

The ratio for the action-value version starts at $t+1$, not $t$: the pair $(S\_t, A\_t)$ being updated is *given*, so its own action is not reweighted — only the actions chosen *afterwards* need correcting.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Reading the ratios)</span></p>

The product $\rho\_{t:h}$ has a clean interpretation, recalling the importance-sampling idea from the off-policy Monte Carlo lecture:

* If $\pi$ would **never** take one of the actions ($\pi(A\_k \mid S\_k) = 0$), the whole return gets weight zero — it is **ignored**.
* If $\pi$ would take an action *far more often* than $b$, the return is **up-weighted** — it is characteristic of $\pi$ but rare in the data.
* If $\pi = b$ (on-policy), every ratio is $1$ and we recover the on-policy updates exactly.

</div>

### Control Variates: Taming Off-policy Variance

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Control variate for state values)</span></p>

Write the $n$-step return recursively, 

$$G^{\text{plain}}_{t:h} = R_{t+1} + \gamma\, G_{t+1:h}.$$

A naive weighting multiplies the *whole* thing by $\rho\_t$ — but if $\rho\_t = 0$ the target collapses to **zero**, injecting variance (the estimate is yanked toward $0$ for no good reason). The fix folds in a **control variate**:

$$G_{t:h} \;\doteq\; \rho_t\,\bigl(R_{t+1} + \gamma\, G_{t+1:h}\bigr) + (1 - \rho_t)\, V_{h-1}(S_t).$$

* If $\rho\_t = 0$, the target becomes $V\_{h-1}(S\_t)$ — the *current estimate* — so **no spurious change** is made.
* Given $S\_t$, the term $V\_{h-1}(S\_t)$ is fixed and $\mathbb{E}\_b[\,1 - \rho\_t \mid S\_t\,] = 0$
  * because

    $$\mathbb{E}_b[\rho_t \mid S_t = s] = \sum_a b(a \mid s)\tfrac{\pi(a\mid s)}{b(a\mid s)} = \sum_a \pi(a\mid s) = 1.$$

    So the control variate has **zero conditional expectation**: it **does not bias** the update — it only reduces variance.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Control variate for action values)</span></p>

For action values we update the already-chosen pair $(S\_t, A\_t)$, so importance sampling starts at the *next* action. The naive form is 

$$G^{\text{naive}}_{t:h} \doteq R_{t+1} + \gamma\, \rho_{t+1}\, G_{t+1:h}.$$

Rewriting the next return around its current estimate,

$$G_{t+1:h} = Q_{h-1}(S_{t+1}, A_{t+1}) + \bigl(G_{t+1:h} - Q_{h-1}(S_{t+1}, A_{t+1})\bigr),$$

we importance-sample **only the correction** and add back the target-policy expectation:

$$G_{t:h} \;\doteq\; R_{t+1} + \gamma\Bigl(\, \bar V_{h-1}(S_{t+1}) + \rho_{t+1}\bigl[\, G_{t+1:h} - Q_{h-1}(S_{t+1}, A_{t+1}) \,\bigr] \Bigr),$$

where 

$$\bar V_{h-1}(S_{t+1}) \doteq \sum_a \pi(a \mid S_{t+1})\, Q_{h-1}(S_{t+1}, a).$$

This is the **Expected-Sarsa-flavoured, variance-reduced off-policy return**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why subtract $Q_{h-1}$?)</span></p>

The bracketed term

$$G^{\text{cv}}_{t+1:h} - Q_{h-1}(S_{t+1}, A_{t+1})$$

is the **surprise** beyond the current estimate. The construction splits the return into a default and a correction:

* $\bar V\_{h-1}(S\_{t+1})$ is the default expected value under $\pi$ — what we believe *before* seeing the sampled continuation.
* $\rho\_{t+1}[\cdots]$ corrects only the *sampled deviation* from the current action-value estimate.
* If $\rho\_{t+1} = 0$, the target falls back to $\bar V\_{h-1}(S\_{t+1})$ — **not zero** — because 
  
  $$\mathbb{E}_b[\rho_{t+1} Q_{h-1}(S_{t+1}, A_{t+1}) \mid S_{t+1}] = \bar V_{h-1}(S_{t+1}).$$

Subtracting $Q\_{h-1}$ is precisely what makes the importance-sampled piece a *zero-mean correction* on top of a low-variance baseline — the defining trick of a control variate.

</div>

### Summary

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Method</span><span class="math-callout__name">(Map of the $n$-step family)</span></p>

| Method | On/off-policy | Last-step handling |
| :----- | :------------ | :----------------- |
| **$n$-step TD** | prediction | bootstrap state value $V\_{t+n-1}(S\_{t+n})$ |
| **$n$-step Sarsa** | on-policy | sample final action: $Q\_{t+n-1}(S\_{t+n}, A\_{t+n})$ |
| **$n$-step Expected Sarsa** | on/off-policy | expectation over actions: $\bar V\_{t+n-1}(S\_{t+n})$ |
| **Off-policy $n$-step Sarsa** | off-policy | importance-sampled, optionally with control variates |

Every row is the *same* skeleton — $n$ real rewards then a bootstrap — differing only in how the final node is handled.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(Core takeaways — $n$-step Bootstrapping)</span></p>

* **A spectrum, not two camps.** $n$-step methods span a continuum from one-step TD ($n = 1$) to Monte Carlo ($n = \infty$); intermediate $n$ is usually best, and they remain TD methods throughout.
* **The $n$-step return is the central object.** $n$ real rewards plus a bootstrap, 
  
  $$G_{t:t+n} = R_{t+1} + \cdots + \gamma^{n-1}R_{t+n} + \gamma^{n} V_{t+n-1}(S_{t+n});$$
  
  the **error-reduction property** ($\gamma^{n}$ contraction) guarantees soundness for the whole family.

* **$n$ is a bias–variance dial.** Small $n$ = more bootstrapping (bias, low variance); large $n$ = more real reward (less bias, high variance). Larger $n$ needs a smaller $\alpha$.
* **Faster control.** **$n$-step Sarsa** propagates credit back $n$ steps per episode, so a single trajectory teaches far more than one-step Sarsa does; **Expected Sarsa** trades the sampled last action for a lower-variance expectation.
* **Off-policy needs reweighting.** **Importance sampling** corrects for following $b$ while learning $\pi$; **control variates** keep the update unbiased while cutting the variance that the raw ratios inject.

**Bridge ahead.** Choosing a single $n$ is itself a compromise — short returns are biased, long ones noisy. The next step removes the choice by *averaging over all $n$ at once* with an exponentially weighted geometric mixture, giving **TD($\lambda$)** and **eligibility traces**: a mechanism that achieves the $n$-step backup incrementally and online, without storing the trajectory, carrying this lecture's TD-error and bias–variance lessons into the function-approximation setting.

</div>

## Lecture 8: Planning and Learning with Tabular Methods

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Setup</span><span class="math-callout__name">(Unifying model-free and model-based RL)</span></p>

Every method so far has improved value estimates from **real experience**:
* Monte Carlo from complete returns, 
* TD and $n$-step methods from sampled transitions.

Dynamic Programming (Lecture 4) was the exception — it improved values *without acting*, by sweeping a known model of the environment. This lecture closes the gap between those two worlds.

The organising claim is simple:

> *Planning and learning are the same operation — a **backup** that improves a value estimate — applied to two different sources of experience. Learning backs up **real** transitions; planning backs up **simulated** ones produced by a model.*

The build-up is:

1. **Models and planning.** What a model is (distribution vs. sample), and what it means to *plan* with one.
2. **Dyna.** Interleaving acting, model learning, direct RL, and planning in one loop.
3. **When the model is wrong.** How imperfect models fail, and how exploration repairs them (Dyna-Q+).
4. **Search control.** *Where* to spend planning updates: prioritized sweeping (backward, error-driven) and trajectory sampling / RTDP (forward, on-policy).
5. **Decision-time planning.** Spending computation on the *current* state only: heuristic search, rollout, and Monte Carlo Tree Search.

</div>

### Models and Planning

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Idea</span><span class="math-callout__name">(Both learning and planning are backups)</span></p>

Earlier lectures emphasised **learning from real experience**. We now add a **model** and ask how to learn from *simulated* experience as well. This unifies the two families seen so far:

* **model-free methods** — TD and Monte Carlo, which learn from real transitions;
* **model-based methods** — dynamic programming and search, which compute from a model.

The central message: planning and learning are closer than they first appear. **Both improve value estimates by backing up information from successor states** — they differ mainly in *where the experience comes from*.

</div>

It helps to locate the new method against everything covered so far. The only genuinely new row is the last one — a method that is **model-based but does not assume the model is given**, learning it from experience instead.

| Method | Model-based | True model | Bootstrapping | Prediction | Control |
| :----- | :---------: | :--------: | :-----------: | :--------: | :-----: |
| Dynamic Programming (DP) | Yes | Yes | Yes | Yes | Yes |
| Monte Carlo (MC) | No | No | No | Yes | Yes |
| TD prediction: TD(0), $n$-step TD | No | No | Yes | Yes | No |
| TD control: Sarsa, $n$-step Sarsa | No | No | Yes | No | Yes |
| TD control: Expected Sarsa | No | No | Yes | No | Yes |
| TD control: Q-learning, Double Q | No | No | Yes | No | Yes |
| **Dyna-Q / learned-model planning** | **Yes** | **No** | **Yes** | No | **Yes** |

#### What Is a Model?

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Model of the environment)</span></p>

A **model** is anything the agent can use to predict what the environment does next — given a state and action, it returns information about the resulting next state and reward. Models come in two flavours:

* **Distribution model.** Represents *all* possible next state–reward pairs together with their probabilities: 
  
  $$\hat p(s', r \mid s, a).$$
  
  It carries complete information but is harder to obtain and store.
* **Sample model.** Produces a *single* sampled outcome 
  
  $$(s,a) \mapsto (S', R)$$
  
  drawn according to the true dynamics. It is easier to simulate from and often far easier to obtain in practice (e.g. a game engine or physics simulator).

A distribution model can always *generate* samples; a sample model cannot recover the full distribution. Either kind can be used to **simulate experience**: starting from any state–action pair, a model can produce a transition, and chaining transitions produces an entire simulated trajectory.

</div>

#### What Is Planning?

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Planning)</span></p>

**Planning** is any computational process that takes a *model* as input and produces or improves a *policy*:

$$\text{model} \;\longrightarrow\; \text{simulated experience} \;\longrightarrow\; \text{backups} \;\longrightarrow\; \text{better policy}.$$

The model simulates experience; the agent backs up values through that simulated experience; the improved values yield better decisions. This is exactly the learning loop with the data source swapped — which is why a single backup rule can serve both.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Takeaway</span><span class="math-callout__name">(Planning is learning from imagined experience)</span></p>

Once planning is framed as *backups over simulated transitions*, every value-update rule from the earlier lectures — TD(0), Sarsa, Q-learning, expected updates — becomes a candidate **planning** rule. The model just replaces the environment as the generator of $(S, A, R, S')$ tuples.

</div>

<figure class="rl-diagram">
  <svg viewBox="0 0 820 330" role="img" aria-label="Learning and planning use the same backup; learning receives real transitions from the environment, while planning receives simulated transitions from a model">
    <rect x="55" y="45" width="185" height="62" rx="8" class="box"></rect>
    <text x="147" y="72" text-anchor="middle" font-size="15">Environment</text>
    <text x="147" y="91" text-anchor="middle" font-size="12" class="muted">real transition</text>

    <rect x="55" y="210" width="185" height="62" rx="8" class="green"></rect>
    <text x="147" y="237" text-anchor="middle" font-size="15">Model</text>
    <text x="147" y="256" text-anchor="middle" font-size="12" class="muted">simulated transition</text>

    <rect x="330" y="105" width="170" height="70" rx="8" class="accent"></rect>
    <text x="415" y="133" text-anchor="middle" font-size="15">Same backup</text>
    <text x="415" y="154" text-anchor="middle" font-size="12" class="muted">update value estimate</text>

    <rect x="590" y="105" width="175" height="70" rx="8" class="box"></rect>
    <text x="678" y="133" text-anchor="middle" font-size="15">Improved policy</text>
    <text x="678" y="154" text-anchor="middle" font-size="12" class="muted">better action choice</text>

    <path d="M240,76 C282,76 292,120 326,130" class="strong-line" marker-end="url(#backup-source-arrow)"></path>
    <path d="M240,241 C282,241 292,161 326,150" class="strong-line" marker-end="url(#backup-source-arrow)"></path>
    <line x1="502" y1="140" x2="586" y2="140" class="strong-line" marker-end="url(#backup-source-arrow)"></line>

    <text x="282" y="67" text-anchor="middle" font-size="12" class="muted">learning</text>
    <text x="284" y="232" text-anchor="middle" font-size="12" class="muted">planning</text>
    <text x="415" y="235" text-anchor="middle" font-size="13" class="muted">The data source changes; the backup machinery does not.</text>

    <defs>
      <marker id="backup-source-arrow" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
        <path d="M0,0 L7,3 L0,6 Z" fill="#64748b"></path>
      </marker>
    </defs>
  </svg>
  <figcaption>Planning is not a new kind of value update. It is the familiar backup operation applied to model-generated experience instead of environment-generated experience.</figcaption>
</figure>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>What are backups?</summary>

In RL, a **backup** means one update that propagates value information **backward** from possible future states to an earlier state or state–action pair.

For example, the Q-learning backup is

$$Q(S,A) \leftarrow Q(S,A) + \alpha \left[ R+\gamma\max_a Q(S',a)-Q(S,A) \right].$$

Here, information from the successor $S'$ is “backed up” into the predecessor pair $(S,A)$:

$$S,A \longrightarrow R,S' \qquad\text{but value information flows backward}\qquad S' \longrightarrow S,A.$$

So a backup is essentially **one Bellman-style value update**, not a stored copy.

### Expected versus sample backups

An **expected backup** averages over every possible next outcome:

$$Q(s,a) \leftarrow \sum_{s',r} p(s',r\mid s,a) \left[ r+\gamma\max_{a'}Q(s',a') \right].$$

This is the usual DP-style backup. It requires the distribution model.

A **sample backup** uses one observed or simulated transition:

$$(s,a,r,s')$$

and updates toward

$$r+\gamma\max_{a'}Q(s',a').$$

TD, Q-learning and Dyna-Q commonly use sample backups.

### Why this matters in Lecture 8

Lecture 8 discusses planning with learned models. Both real experience and model-generated experience can produce the same backup:

$$Q(S,A) \leftarrow Q(S,A)+\alpha \left[ R+\gamma\max_aQ(S',a)-Q(S,A) \right].$$

The only difference is where $(S,A,R,S')$ came from:

* **direct RL:** a real environment transition;
* **planning:** a transition generated by the model.

Thus when Dyna-Q performs $n$ planning steps, it performs $n$ additional backups. Prioritized sweeping asks which state–action pairs should be backed up first. 

A useful mental translation is:

$$\boxed{\text{backup}=\text{use later-value information to revise an earlier value estimate}.}$$

</details>
</div>

### Dyna: Integrating Learning and Planning

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Idea</span><span class="math-callout__name">(The Dyna recipe)</span></p>

Dyna interleaves four ideas in one online loop:

1. **Act** and learn from **real** experience (direct RL).
2. **Learn a model** from that same real experience.
3. Use the model to **simulate** additional experience.
4. **Learn again** from the simulated experience (planning).

The point is *data efficiency*: Dyna turns **one real transition into many learning opportunities** by replaying its consequences through a learned model, rather than discarding each transition after a single update.

</div>

#### The Dyna Architecture

<figure class="rl-diagram">
  <svg viewBox="0 0 820 360" role="img" aria-label="Dyna architecture: real experience updates both the value function (direct RL) and the model (model learning); the model generates simulated experience that updates the value function (planning); the value function drives acting in the environment">
    <!-- Value/policy box -->
    <rect x="300" y="20" width="220" height="48" rx="8" class="accent"></rect>
    <text x="410" y="49" text-anchor="middle" font-size="15">Policy / value functions</text>

    <!-- Experience node -->
    <ellipse cx="410" cy="180" rx="78" ry="30" class="box"></ellipse>
    <text x="410" y="178" text-anchor="middle" font-size="14">real</text>
    <text x="410" y="195" text-anchor="middle" font-size="14">experience</text>

    <!-- Model box -->
    <rect x="610" y="156" width="150" height="48" rx="8" class="green"></rect>
    <text x="685" y="185" text-anchor="middle" font-size="15">Model</text>

    <!-- Environment box -->
    <rect x="300" y="300" width="220" height="48" rx="8" class="box"></rect>
    <text x="410" y="329" text-anchor="middle" font-size="15">Environment</text>

    <!-- direct RL update: experience -> value (left, up) -->
    <line x1="360" y1="156" x2="360" y2="70" class="strong-line" marker-end="url(#ah)"></line>
    <text x="250" y="116" text-anchor="middle" font-size="12" class="muted">direct RL</text>
    <text x="250" y="132" text-anchor="middle" font-size="12" class="muted">update</text>

    <!-- planning update: model/simulated -> value (right, up) -->
    <line x1="560" y1="156" x2="470" y2="70" class="strong-line" marker-end="url(#ah)"></line>
    <text x="585" y="116" text-anchor="middle" font-size="12" class="muted">planning</text>
    <text x="585" y="132" text-anchor="middle" font-size="12" class="muted">update</text>

    <!-- model learning: experience -> model -->
    <line x1="488" y1="180" x2="606" y2="180" class="line" marker-end="url(#ah)"></line>
    <text x="548" y="170" text-anchor="middle" font-size="12" class="muted">model learning</text>

    <!-- search control / simulated experience: model -> experience -->
    <line x1="606" y1="196" x2="488" y2="196" class="line" marker-end="url(#ah)"></line>
    <text x="548" y="220" text-anchor="middle" font-size="12" class="muted">search control</text>

    <!-- acting loop: value -> environment -> experience -->
    <line x1="410" y1="300" x2="410" y2="212" class="line" marker-end="url(#ah)"></line>
    <text x="455" y="258" text-anchor="middle" font-size="12" class="muted">acting</text>

    <defs>
      <marker id="ah" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
        <path d="M0,0 L7,3 L0,6 Z" fill="#64748b"></path>
      </marker>
    </defs>
  </svg>
  <figcaption>The Dyna loop. Real experience feeds <em>two</em> learners — direct RL (value function) and model learning. The model then produces simulated experience that feeds the same value function through planning. <strong>Search control</strong> decides which state–action pairs to simulate next.</figcaption>
</figure>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(The four Dyna processes)</span></p>

* **Acting** — interact with the environment to gather real experience.
* **Direct RL** — update the value function from real experience.
* **Model learning** — improve the model from real experience.
* **Planning** — update the value function from simulated experience.

A fifth, easily overlooked role is **search control**: the planner must also decide *which* state–action pairs to simulate next. This question — *where* to spend planning effort — drives the second half of the lecture.

</div>

#### Tabular Dyna-Q

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Tabular Dyna-Q)</span></p>

In **tabular Dyna-Q** the three value-touching processes share one update rule:

* real experience updates $Q$ directly;
* real experience also updates a deterministic model $\hat M(S, A) = (R, S')$;
* planning then performs $n$ extra Q-learning-style updates on *model-generated* transitions.

The shared backup is the Q-learning update,

$$
Q(S, A) \;\leftarrow\; Q(S, A) + \alpha\Bigl[\, R + \gamma \max_{a} Q(S', a) - Q(S, A) \,\Bigr],
$$

applied identically whether $(S, A, R, S')$ came from the environment or from $\hat M$. **Only the source of experience changes** — the learning machinery is the same.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Tabular Dyna-Q)</span></p>

**Maintain:** 
* action values $Q(s,a)$; 
* a model $\hat M(s,a) = (r, s')$; 
* the set of previously seen state–action pairs.

For each real step:

1. Choose $A$ from current state $S$ by $\varepsilon$-greedy on $Q$.
2. Execute $A$; observe reward $R$ and next state $S'$.
3. **Direct RL update:**
   
   $$Q(S, A) \leftarrow Q(S, A) + \alpha\bigl[\, R + \gamma \max_{a} Q(S', a) - Q(S, A)\,\bigr].$$

4. **Model learning:** store 
   
   $$\hat M(S, A) \leftarrow (R, S').$$

5. **Planning — repeat $n$ times:**
   * sample a previously seen pair $(\tilde S, \tilde A)$;
   * retrieve $(\tilde R, \tilde S') = \hat M(\tilde S, \tilde A)$;
   * update $Q(\tilde S, \tilde A)$ with the *same* rule as step 3.
6. $S \leftarrow S'$.

The parameter $n$ is the number of planning updates per real step; $n = 0$ recovers ordinary one-step Q-learning (direct RL only).

</div>

#### Example: The Dyna Maze

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Dyna maze — planning buys speed)</span></p>

A gridworld agent must reach a goal $G$ from a start $S$ around a few walls. Holding the **real** experience stream fixed and varying only the number of planning steps:

* $n = 0$ (direct RL only) needs many episodes before the steps-per-episode curve falls toward the optimal path length.
* $n = 5$ converges substantially faster.
* $n = 50$ reaches near-optimal behaviour in a handful of episodes.

**More planning per real step propagates each observed transition much further before the agent takes its next action — so the policy improves dramatically faster for the *same* amount of real interaction.**

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Intuition</span><span class="math-callout__name">(Planning races ahead of the agent)</span></p>

Compare the policy after the first episode. **Without planning** ($n = 0$), only a tiny region near the goal has been backed up — value information has barely spread. **With planning** ($n = 50$), value has already propagated far back toward the start, so much of the route is laid out *while the agent is still near where it began*.

Planning lets value information race ahead of the agent's physical location: each real step is replayed enough times for its consequences to reach states the agent has not recently visited.

</div>

<figure class="rl-diagram">
  <svg viewBox="0 0 820 340" role="img" aria-label="Dyna maze propagation comparing no planning to many planning updates; value information spreads farther from the goal when planning is used">
    <text x="205" y="42" text-anchor="middle" font-size="15">n = 0: direct RL only</text>
    <text x="615" y="42" text-anchor="middle" font-size="15">n = 50: many simulated backups</text>

    <!-- left maze -->
    <g transform="translate(65,70)">
      <rect x="0" y="0" width="280" height="180" rx="8" fill="#ffffff" stroke="#dbe1ee"></rect>
      <rect x="0" y="0" width="35" height="30" class="box"></rect><rect x="35" y="0" width="35" height="30" class="box"></rect><rect x="70" y="0" width="35" height="30" class="box"></rect><rect x="105" y="0" width="35" height="30" class="box"></rect><rect x="140" y="0" width="35" height="30" class="box"></rect><rect x="175" y="0" width="35" height="30" class="box"></rect><rect x="210" y="0" width="35" height="30" class="box"></rect><rect x="245" y="0" width="35" height="30" class="box"></rect>
      <rect x="0" y="30" width="35" height="30" class="box"></rect><rect x="35" y="30" width="35" height="30" class="box"></rect><rect x="70" y="30" width="35" height="30" class="box"></rect><rect x="105" y="30" width="35" height="30" class="box"></rect><rect x="140" y="30" width="35" height="30" class="box"></rect><rect x="175" y="30" width="35" height="30" class="box"></rect><rect x="210" y="30" width="35" height="30" class="box"></rect><rect x="245" y="30" width="35" height="30" class="box"></rect>
      <rect x="0" y="60" width="35" height="30" class="box"></rect><rect x="35" y="60" width="35" height="30" class="box"></rect><rect x="70" y="60" width="35" height="30" class="box"></rect><rect x="105" y="60" width="35" height="30" class="box"></rect><rect x="140" y="60" width="35" height="30" class="box"></rect><rect x="175" y="60" width="35" height="30" class="box"></rect><rect x="210" y="60" width="35" height="30" class="box"></rect><rect x="245" y="60" width="35" height="30" class="green"></rect>
      <rect x="0" y="90" width="35" height="30" class="accent"></rect><rect x="35" y="90" width="35" height="30" class="box"></rect><rect x="70" y="90" width="35" height="30" class="box"></rect><rect x="105" y="90" width="35" height="30" class="box"></rect><rect x="140" y="90" width="35" height="30" class="box"></rect><rect x="175" y="90" width="35" height="30" fill="#172033"></rect><rect x="210" y="90" width="35" height="30" class="amber"></rect><rect x="245" y="90" width="35" height="30" class="green"></rect>
      <rect x="0" y="120" width="35" height="30" class="box"></rect><rect x="35" y="120" width="35" height="30" class="box"></rect><rect x="70" y="120" width="35" height="30" fill="#172033"></rect><rect x="105" y="120" width="35" height="30" fill="#172033"></rect><rect x="140" y="120" width="35" height="30" fill="#172033"></rect><rect x="175" y="120" width="35" height="30" fill="#172033"></rect><rect x="210" y="120" width="35" height="30" class="box"></rect><rect x="245" y="120" width="35" height="30" class="box"></rect>
      <rect x="0" y="150" width="35" height="30" class="box"></rect><rect x="35" y="150" width="35" height="30" class="box"></rect><rect x="70" y="150" width="35" height="30" class="box"></rect><rect x="105" y="150" width="35" height="30" class="box"></rect><rect x="140" y="150" width="35" height="30" class="box"></rect><rect x="175" y="150" width="35" height="30" class="box"></rect><rect x="210" y="150" width="35" height="30" class="box"></rect><rect x="245" y="150" width="35" height="30" class="box"></rect>
      <text x="17" y="110" text-anchor="middle" font-size="13">S</text>
      <text x="262" y="80" text-anchor="middle" font-size="13">G</text>
      <path d="M229,104 L247,104" class="strong-line" marker-end="url(#dyna-maze-arrow)"></path>
      <text x="140" y="210" text-anchor="middle" font-size="12" class="muted">only states near the goal get useful backups</text>
    </g>

    <!-- right maze -->
    <g transform="translate(475,70)">
      <rect x="0" y="0" width="280" height="180" rx="8" fill="#ffffff" stroke="#dbe1ee"></rect>
      <rect x="0" y="0" width="35" height="30" class="box"></rect><rect x="35" y="0" width="35" height="30" class="box"></rect><rect x="70" y="0" width="35" height="30" class="box"></rect><rect x="105" y="0" width="35" height="30" class="box"></rect><rect x="140" y="0" width="35" height="30" class="box"></rect><rect x="175" y="0" width="35" height="30" class="box"></rect><rect x="210" y="0" width="35" height="30" class="box"></rect><rect x="245" y="0" width="35" height="30" class="green"></rect>
      <rect x="0" y="30" width="35" height="30" class="box"></rect><rect x="35" y="30" width="35" height="30" class="box"></rect><rect x="70" y="30" width="35" height="30" class="box"></rect><rect x="105" y="30" width="35" height="30" class="box"></rect><rect x="140" y="30" width="35" height="30" class="amber"></rect><rect x="175" y="30" width="35" height="30" class="amber"></rect><rect x="210" y="30" width="35" height="30" class="green"></rect><rect x="245" y="30" width="35" height="30" class="green"></rect>
      <rect x="0" y="60" width="35" height="30" class="amber"></rect><rect x="35" y="60" width="35" height="30" class="amber"></rect><rect x="70" y="60" width="35" height="30" class="amber"></rect><rect x="105" y="60" width="35" height="30" class="amber"></rect><rect x="140" y="60" width="35" height="30" class="amber"></rect><rect x="175" y="60" width="35" height="30" class="green"></rect><rect x="210" y="60" width="35" height="30" class="green"></rect><rect x="245" y="60" width="35" height="30" class="green"></rect>
      <rect x="0" y="90" width="35" height="30" class="accent"></rect><rect x="35" y="90" width="35" height="30" class="amber"></rect><rect x="70" y="90" width="35" height="30" class="amber"></rect><rect x="105" y="90" width="35" height="30" class="amber"></rect><rect x="140" y="90" width="35" height="30" class="amber"></rect><rect x="175" y="90" width="35" height="30" fill="#172033"></rect><rect x="210" y="90" width="35" height="30" class="green"></rect><rect x="245" y="90" width="35" height="30" class="green"></rect>
      <rect x="0" y="120" width="35" height="30" class="box"></rect><rect x="35" y="120" width="35" height="30" class="box"></rect><rect x="70" y="120" width="35" height="30" fill="#172033"></rect><rect x="105" y="120" width="35" height="30" fill="#172033"></rect><rect x="140" y="120" width="35" height="30" fill="#172033"></rect><rect x="175" y="120" width="35" height="30" fill="#172033"></rect><rect x="210" y="120" width="35" height="30" class="box"></rect><rect x="245" y="120" width="35" height="30" class="box"></rect>
      <rect x="0" y="150" width="35" height="30" class="box"></rect><rect x="35" y="150" width="35" height="30" class="box"></rect><rect x="70" y="150" width="35" height="30" class="box"></rect><rect x="105" y="150" width="35" height="30" class="box"></rect><rect x="140" y="150" width="35" height="30" class="box"></rect><rect x="175" y="150" width="35" height="30" class="box"></rect><rect x="210" y="150" width="35" height="30" class="box"></rect><rect x="245" y="150" width="35" height="30" class="box"></rect>
      <text x="17" y="110" text-anchor="middle" font-size="13">S</text>
      <text x="262" y="80" text-anchor="middle" font-size="13">G</text>
      <path d="M20,104 C82,70 160,70 244,75" class="strong-line" marker-end="url(#dyna-maze-arrow)"></path>
      <text x="140" y="210" text-anchor="middle" font-size="12" class="muted">simulated backups spread value far along the route</text>
    </g>

    <defs>
      <marker id="dyna-maze-arrow" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
        <path d="M0,0 L7,3 L0,6 Z" fill="#64748b"></path>
      </marker>
    </defs>
  </svg>
  <figcaption>In the Dyna maze, planning changes the speed of propagation, not the real experience stream. Extra simulated backups push the goal information back toward the start before the agent physically revisits those states.</figcaption>
</figure>

### When the Model Is Wrong

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Models are usually imperfect)</span></p>

So far the model behaved kindly: it started empty and then filled with *correct* transitions. In practice a learned model can be **wrong** because

* the environment is stochastic and only finitely sampled,
* data is limited,
* function approximation generalises badly, or
* the environment itself **changes** over time.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Caveat</span><span class="math-callout__name">(A wrong model is dangerous)</span></p>

Planning amplifies whatever the model believes. **A bad model can make planning compute a bad policy faster** — the same mechanism that propagates correct information efficiently will propagate incorrect information just as efficiently. The remedy is not to trust the model blindly but to keep *testing* it against reality.

</div>

#### Failure Mode 1: The Blocking Maze

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Blocking maze — the model becomes too optimistic)</span></p>

A short path exists for a while, then the environment changes and that path is **blocked**. The agent's model still records the old, now-impossible transition.

* Standard **Dyna-Q** eventually adapts: real experience overwrites the stale entry once the agent stumbles into the wall.
* **Dyna-Q+** adapts *faster*, because it keeps re-testing long-untried transitions and so rediscovers the change sooner.

The lesson: *exploration in planning is about checking whether the model is still right*, not only about chasing reward.

</div>

#### Failure Mode 2: The Shortcut Maze

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Shortcut maze — the model is too pessimistic)</span></p>

Now the environment change *opens* a new, shorter path. The agent's model says no shortcut exists, so planning keeps reinforcing the old longer route and the agent may **never discover** the improvement.

Standard Dyna-Q has no incentive to revisit a region it believes it already understands. Dyna-Q+, by contrast, periodically rechecks neglected actions and can find the new path.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Failure mode</span><span class="math-callout__name">(Outdated models suppress their own repair)</span></p>

The shortcut maze exposes the core danger: **if planning trusts an outdated model too much, it can suppress the very exploration needed to repair that model.** The agent is confident precisely where it is wrong, and confidence kills the exploration that would correct it.

</div>

<figure class="rl-diagram">
  <svg viewBox="0 0 820 330" role="img" aria-label="Two stale-model failures: blocking maze closes an old short path; shortcut maze opens a new path that the model does not know about">
    <text x="205" y="42" text-anchor="middle" font-size="15">Blocking: old path closes</text>
    <text x="615" y="42" text-anchor="middle" font-size="15">Shortcut: new path opens</text>

    <g transform="translate(65,70)">
      <rect x="0" y="0" width="280" height="160" rx="8" fill="#ffffff" stroke="#dbe1ee"></rect>
      <rect x="0" y="0" width="40" height="40" class="accent"></rect><rect x="40" y="0" width="40" height="40" class="amber"></rect><rect x="80" y="0" width="40" height="40" class="amber"></rect><rect x="120" y="0" width="40" height="40" class="amber"></rect><rect x="160" y="0" width="40" height="40" class="red"></rect><rect x="200" y="0" width="40" height="40" class="green"></rect><rect x="240" y="0" width="40" height="40" class="green"></rect>
      <rect x="0" y="40" width="40" height="40" class="box"></rect><rect x="40" y="40" width="40" height="40" class="box"></rect><rect x="80" y="40" width="40" height="40" class="box"></rect><rect x="120" y="40" width="40" height="40" fill="#172033"></rect><rect x="160" y="40" width="40" height="40" fill="#172033"></rect><rect x="200" y="40" width="40" height="40" fill="#172033"></rect><rect x="240" y="40" width="40" height="40" class="green"></rect>
      <rect x="0" y="80" width="40" height="40" class="box"></rect><rect x="40" y="80" width="40" height="40" class="box"></rect><rect x="80" y="80" width="40" height="40" class="box"></rect><rect x="120" y="80" width="40" height="40" class="box"></rect><rect x="160" y="80" width="40" height="40" class="box"></rect><rect x="200" y="80" width="40" height="40" class="box"></rect><rect x="240" y="80" width="40" height="40" class="green"></rect>
      <rect x="0" y="120" width="40" height="40" class="box"></rect><rect x="40" y="120" width="40" height="40" class="box"></rect><rect x="80" y="120" width="40" height="40" class="box"></rect><rect x="120" y="120" width="40" height="40" class="box"></rect><rect x="160" y="120" width="40" height="40" class="box"></rect><rect x="200" y="120" width="40" height="40" class="box"></rect><rect x="240" y="120" width="40" height="40" class="box"></rect>
      <text x="20" y="25" text-anchor="middle" font-size="13">S</text>
      <text x="260" y="25" text-anchor="middle" font-size="13">G</text>
      <text x="180" y="25" text-anchor="middle" font-size="17">×</text>
      <path d="M22,20 C70,18 114,18 155,20" class="strong-line" stroke-dasharray="5 4" marker-end="url(#stale-model-arrow)"></path>
      <text x="140" y="195" text-anchor="middle" font-size="12" class="muted">model still believes the dashed path works</text>
    </g>

    <g transform="translate(475,70)">
      <rect x="0" y="0" width="280" height="160" rx="8" fill="#ffffff" stroke="#dbe1ee"></rect>
      <rect x="0" y="0" width="40" height="40" class="accent"></rect><rect x="40" y="0" width="40" height="40" class="amber"></rect><rect x="80" y="0" width="40" height="40" class="amber"></rect><rect x="120" y="0" width="40" height="40" class="amber"></rect><rect x="160" y="0" width="40" height="40" class="amber"></rect><rect x="200" y="0" width="40" height="40" class="green"></rect><rect x="240" y="0" width="40" height="40" class="green"></rect>
      <rect x="0" y="40" width="40" height="40" class="box"></rect><rect x="40" y="40" width="40" height="40" class="box"></rect><rect x="80" y="40" width="40" height="40" class="box"></rect><rect x="120" y="40" width="40" height="40" fill="#172033"></rect><rect x="160" y="40" width="40" height="40" fill="#172033"></rect><rect x="200" y="40" width="40" height="40" fill="#172033"></rect><rect x="240" y="40" width="40" height="40" class="green"></rect>
      <rect x="0" y="80" width="40" height="40" class="box"></rect><rect x="40" y="80" width="40" height="40" class="box"></rect><rect x="80" y="80" width="40" height="40" class="amber"></rect><rect x="120" y="80" width="40" height="40" class="amber"></rect><rect x="160" y="80" width="40" height="40" class="green"></rect><rect x="200" y="80" width="40" height="40" class="green"></rect><rect x="240" y="80" width="40" height="40" class="green"></rect>
      <rect x="0" y="120" width="40" height="40" class="box"></rect><rect x="40" y="120" width="40" height="40" class="box"></rect><rect x="80" y="120" width="40" height="40" class="box"></rect><rect x="120" y="120" width="40" height="40" class="box"></rect><rect x="160" y="120" width="40" height="40" class="box"></rect><rect x="200" y="120" width="40" height="40" class="box"></rect><rect x="240" y="120" width="40" height="40" class="box"></rect>
      <text x="20" y="25" text-anchor="middle" font-size="13">S</text>
      <text x="260" y="25" text-anchor="middle" font-size="13">G</text>
      <path d="M22,20 C62,74 116,98 235,98" class="strong-line" marker-end="url(#stale-model-arrow)"></path>
      <path d="M22,20 C72,18 138,18 195,20" class="line" stroke-dasharray="5 4" marker-end="url(#stale-model-arrow)"></path>
      <text x="140" y="195" text-anchor="middle" font-size="12" class="muted">new route exists, but the stale model plans the old one</text>
    </g>

    <defs>
      <marker id="stale-model-arrow" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
        <path d="M0,0 L7,3 L0,6 Z" fill="#64748b"></path>
      </marker>
    </defs>
  </svg>
  <figcaption>Planning amplifies the model. If the model is stale, it can keep rehearsing a blocked route or ignore a newly opened shortcut until exploration forces a correction.</figcaption>
</figure>

#### Dyna-Q+: Exploration as Model Maintenance

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Dyna-Q+ exploration bonus)</span></p>

**Dyna-Q+** adds an exploration bonus *inside planning*. If a transition has not been tried for $\tau$ time steps, planning treats its reward as

$$
r^{+} = r + \kappa \sqrt{\tau},
$$

for a small $\kappa > 0$. Long-untried state–action pairs accrue extra value, so the agent is encouraged to recheck parts of the world it has neglected — which is exactly how it discovers that the environment has changed.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Dyna-Q+ — change from Dyna-Q)</span></p>

Dyna-Q+ keeps the entire Dyna-Q loop and adds one piece of bookkeeping and one change to planning.

**Extra state.** For every pair, track $\tau(s, a)$ — the number of time steps since $(s, a)$ was last tried.

**Planning change.** During planning, use the bonus-augmented reward

$$
r^{+} = r + \kappa \sqrt{\tau(s, a)}
$$

in place of the stored reward $r$. Recently tried actions receive a negligible bonus; long-neglected actions become worth rechecking.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Takeaway</span><span class="math-callout__name">(Curiosity as model upkeep)</span></p>

Dyna-Q+ explores **not just for reward, but to test whether its model has gone stale**. The $\sqrt{\tau}$ bonus is a form of *computational curiosity*: it makes "I haven't checked this in a while" a reason to act, which keeps the model honest in non-stationary environments.

</div>

### Prioritized Sweeping

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Uniform random planning wastes effort)</span></p>

Dyna-Q samples the state–action pairs to plan from *uniformly at random*. This is easy but often **inefficient**:

* early in learning, almost every update changes nothing, because most values are still zero and the backed-up target is zero too;
* when one state's value changes a lot, only its **predecessors** are immediately affected;
* so planning should concentrate where value changes are likely to propagate next, not spread itself evenly.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Intuition</span><span class="math-callout__name">(Work backward from what just changed)</span></p>

If one state suddenly becomes important — its value jumps — then the states that *lead into it* are the ones whose values are now stale. So back up **into the predecessors** of any state whose value just changed, and recurse. This focuses computation on the frontier where information is actually moving.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Prioritized sweeping)</span></p>

**Prioritized sweeping** focuses planning **backward** from states whose values have changed. It maintains a **priority queue** of state–action pairs keyed by the magnitude of their Bellman error:

* a large Bellman error $\Rightarrow$ high priority;
* after updating a pair, examine its **predecessors** and compute how much *their* values would change;
* push important predecessors onto the queue.

In short: **do the most urgent backups first.**

</div>

<figure class="rl-diagram">
  <svg viewBox="0 0 820 360" role="img" aria-label="Prioritized sweeping backs up from a changed state to its predecessors using a priority queue of Bellman errors">
    <rect x="330" y="125" width="150" height="70" rx="8" class="green"></rect>
    <text x="405" y="153" text-anchor="middle" font-size="15">changed state</text>
    <text x="405" y="174" text-anchor="middle" font-size="12" class="muted">large new value</text>

    <rect x="80" y="55" width="140" height="52" rx="8" class="box"></rect>
    <text x="150" y="86" text-anchor="middle" font-size="13">predecessor A</text>
    <rect x="80" y="150" width="140" height="52" rx="8" class="box"></rect>
    <text x="150" y="181" text-anchor="middle" font-size="13">predecessor B</text>
    <rect x="80" y="245" width="140" height="52" rx="8" class="box"></rect>
    <text x="150" y="276" text-anchor="middle" font-size="13">predecessor C</text>

    <path d="M220,81 C265,90 288,122 326,143" class="line" marker-end="url(#ps-arrow)"></path>
    <path d="M220,176 L326,163" class="line" marker-end="url(#ps-arrow)"></path>
    <path d="M220,271 C265,250 288,204 326,179" class="line" marker-end="url(#ps-arrow)"></path>

    <path d="M330,160 C285,160 260,176 224,176" class="strong-line" stroke-dasharray="5 4" marker-end="url(#ps-arrow)"></path>
    <text x="275" y="205" text-anchor="middle" font-size="12" class="muted">sweep backward</text>

    <rect x="560" y="55" width="185" height="245" rx="8" class="accent"></rect>
    <text x="652" y="86" text-anchor="middle" font-size="15">Priority queue</text>
    <rect x="585" y="112" width="135" height="38" rx="6" class="red"></rect>
    <text x="652" y="136" text-anchor="middle" font-size="12">B: |δ| = 0.91</text>
    <rect x="585" y="162" width="135" height="38" rx="6" class="amber"></rect>
    <text x="652" y="186" text-anchor="middle" font-size="12">A: |δ| = 0.42</text>
    <rect x="585" y="212" width="135" height="38" rx="6" class="box"></rect>
    <text x="652" y="236" text-anchor="middle" font-size="12">C: |δ| = 0.08</text>
    <text x="652" y="274" text-anchor="middle" font-size="12" class="muted">pop highest first</text>

    <line x1="481" y1="160" x2="556" y2="130" class="strong-line" marker-end="url(#ps-arrow)"></line>
    <text x="520" y="116" text-anchor="middle" font-size="12" class="muted">enqueue</text>

    <defs>
      <marker id="ps-arrow" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
        <path d="M0,0 L7,3 L0,6 Z" fill="#64748b"></path>
      </marker>
    </defs>
  </svg>
  <figcaption>Prioritized sweeping is search control, not a new target. A value change at one state makes its predecessors suspicious; the queue orders those predecessor backups by urgency.</figcaption>
</figure>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Prioritized sweeping — deterministic environment)</span></p>

Initialise $Q(s,a)$, a model $\hat M(s,a)$, and an empty priority queue $PQueue$. For a small threshold $\theta > 0$, loop forever:

1. $S \leftarrow$ current (nonterminal) state; 
   
   $$A \leftarrow \text{ policy }(S, Q).$$

2. Execute $A$; observe $R, S'$, and store 
   
   $$\hat M(S, A) \leftarrow (R, S').$$

3. Compute the priority 
   
   $$P \leftarrow \lvert R + \gamma \max_{a} Q(S', a) - Q(S, A) \rvert.$$

4. If $P > \theta$, insert $(S, A)$ into $PQueue$ with priority $P$.
5. Repeat $n$ times while $PQueue$ is nonempty:
   * pop the highest-priority pair $(S, A)$; retrieve 
     
     $$(R, S') = \hat M(S, A);$$

   * update 
     
     $$Q(S, A) \leftarrow Q(S, A) + \alpha\bigl[\, R + \gamma \max_{a} Q(S', a) - Q(S, A)\,\bigr];$$

   * for each predicted predecessor $(\bar S, \bar A)$ of $S$ with reward $\bar R$, compute
     
     $$P \leftarrow \bigl\lvert \bar R + \gamma \max_{a} Q(S, a) - Q(\bar S, \bar A) \bigr\rvert;$$

     * if $P > \theta$, insert $(\bar S, \bar A)$ into $PQueue$ with priority $P$.

On large gridworlds, prioritized sweeping can reach the optimal solution with **orders of magnitude fewer updates** than uniform Dyna-Q — **search control matters as much as the update rule itself.**

</div>

#### Expected vs Sample Updates

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Expected vs sample update)</span></p>

For control, the two ways to back up a single $(s, a)$ are:

* **Expected update** — average over *all* possible successors weighted by the model:
  
  $$Q(s, a) \leftarrow \sum_{s', r} \hat p(s', r \mid s, a)\bigl[\, r + \gamma \max_{a'} Q(s', a')\,\bigr].$$

* **Sample update** — use *one* sampled transition:
  
  $$Q(s, a) \leftarrow Q(s, a) + \alpha\bigl[\, R + \gamma \max_{a'} Q(S', a') - Q(s, a)\,\bigr].$$

Expected updates are **exact given the model** but cost work proportional to the branching factor. Sample updates are **noisy but cheap**, and pay off when computation — not data — is the bottleneck, since many cheap sample updates can outrun a few expensive expected ones.

</div>

<figure class="rl-diagram">
  <svg viewBox="0 0 820 330" role="img" aria-label="Expected update averages all possible successors while sample update backs up one sampled successor">
    <text x="205" y="42" text-anchor="middle" font-size="15">Expected update</text>
    <text x="615" y="42" text-anchor="middle" font-size="15">Sample update</text>

    <g transform="translate(60,75)">
      <circle cx="145" cy="42" r="24" class="accent"></circle>
      <text x="145" y="47" text-anchor="middle" font-size="13">(s,a)</text>
      <circle cx="50" cy="155" r="22" class="green"></circle>
      <circle cx="145" cy="155" r="22" class="green"></circle>
      <circle cx="240" cy="155" r="22" class="green"></circle>
      <line x1="133" y1="63" x2="64" y2="135" class="strong-line" marker-end="url(#expected-sample-arrow)"></line>
      <line x1="145" y1="66" x2="145" y2="129" class="strong-line" marker-end="url(#expected-sample-arrow)"></line>
      <line x1="157" y1="63" x2="226" y2="135" class="strong-line" marker-end="url(#expected-sample-arrow)"></line>
      <text x="50" y="160" text-anchor="middle" font-size="12">s1</text>
      <text x="145" y="160" text-anchor="middle" font-size="12">s2</text>
      <text x="240" y="160" text-anchor="middle" font-size="12">s3</text>
      <text x="92" y="103" text-anchor="middle" font-size="11" class="muted">p=.2</text>
      <text x="166" y="103" text-anchor="middle" font-size="11" class="muted">p=.5</text>
      <text x="208" y="103" text-anchor="middle" font-size="11" class="muted">p=.3</text>
      <rect x="17" y="220" width="256" height="42" rx="8" class="box"></rect>
      <text x="145" y="246" text-anchor="middle" font-size="12">average every branch</text>
    </g>

    <g transform="translate(470,75)">
      <circle cx="145" cy="42" r="24" class="accent"></circle>
      <text x="145" y="47" text-anchor="middle" font-size="13">(s,a)</text>
      <circle cx="50" cy="155" r="22" class="box"></circle>
      <circle cx="145" cy="155" r="22" class="green"></circle>
      <circle cx="240" cy="155" r="22" class="box"></circle>
      <line x1="133" y1="63" x2="64" y2="135" class="line" stroke-dasharray="5 4"></line>
      <line x1="145" y1="66" x2="145" y2="129" class="strong-line" marker-end="url(#expected-sample-arrow)"></line>
      <line x1="157" y1="63" x2="226" y2="135" class="line" stroke-dasharray="5 4"></line>
      <text x="50" y="160" text-anchor="middle" font-size="12">s1</text>
      <text x="145" y="160" text-anchor="middle" font-size="12">S'</text>
      <text x="240" y="160" text-anchor="middle" font-size="12">s3</text>
      <rect x="17" y="220" width="256" height="42" rx="8" class="box"></rect>
      <text x="145" y="246" text-anchor="middle" font-size="12">back up one draw</text>
    </g>

    <defs>
      <marker id="expected-sample-arrow" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
        <path d="M0,0 L7,3 L0,6 Z" fill="#64748b"></path>
      </marker>
    </defs>
  </svg>
  <figcaption>Expected updates spend more computation per backup to remove sampling noise. Sample updates are cheaper and noisy, so a fixed compute budget can buy many more of them.</figcaption>
</figure>

### Trajectory Sampling and RTDP

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Idea</span><span class="math-callout__name">(Two criteria for where to update)</span></p>

We now have *two* complementary answers to the question "**where** should planning spend its updates?"

* **Prioritized sweeping** (already seen) — works **backward** from states whose value just changed. 
  * Criterion: **Bellman-error magnitude**. 
  * Goal: propagate new information to the predecessors that depend on it. *"Fix what is most wrong first."*
* **Trajectory sampling** (next) — works **forward** from the current state along the policy. 
  * Criterion: **on-policy visitation**. 
  * Goal: concentrate compute on states the agent actually reaches. *"Work on what is most relevant."*

One is error-driven and backward; the other is distribution-driven and forward. They are **complementary**, and can be combined.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(On-policy trajectory sampling can win big)</span></p>

Generating updates by **following the current policy** focuses effort on the states that matter under that policy and ignores unreachable ones. Empirically, on-policy trajectory sampling often wins **early**: on large problems it reaches good values far faster than uniform sweeps, which waste computation on states the policy never visits. Uniform sweeps may catch up eventually on *small* problems, but on large state spaces, focusing updates where the policy actually goes is a major advantage.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Real-Time Dynamic Programming, RTDP)</span></p>

**RTDP** is an on-policy, trajectory-sampling version of value iteration. It

* follows **real or simulated** trajectories from the start state;
* at each visited state performs a value-iteration-style **expected update**;
* always acts greedily with respect to the current value function.

The contrast with classical DP is purely one of *coverage*:

* **DP** — expected backups over *many / all* states each sweep;
* **RTDP** — expected backups *only along sampled trajectories*.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(RTDP — sketch)</span></p>

**Maintain:** 
* state values $V(s)$
* transition model $\hat p$.

For each trial:

1. Start at an initial state $S$.
2. Repeat until terminal:
   * choose a greedy action under the current model,
     
     $$A \in \arg\max_{a} \sum_{s', r} \hat p(s', r \mid S, a)\bigl[\, r + \gamma V(s')\,\bigr];$$

   * update the value with an expected backup,
     
     $$V(S) \leftarrow \max_{a} \sum_{s', r} \hat p(s', r \mid S, a)\bigl[\, r + \gamma V(s')\,\bigr];$$

   * move to the next state $S'$.

Because it only ever backs up states reachable under greedy play, RTDP can solve goal-directed problems while leaving most of the state space untouched.

</div>

#### The Space of Updates

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Idea</span><span class="math-callout__name">(Two axes organise every method)</span></p>

Every backup discussed in this course can be placed on two axes:

* **sample vs expected** — does the backup use one sampled successor, or average over all of them?
* **one-step (shallow) vs deep** — does it look one step ahead, or simulate far into the future?

|  | **Sample updates** | **Expected updates** |
| :--- | :--- | :--- |
| **One step** | Dyna-Q | RTDP / DP |
| **Deep** | Rollout, MCTS | Heuristic search |

</div>

<figure class="rl-diagram">
  <svg viewBox="0 0 820 420" role="img" aria-label="Two-axis map of backup methods: sample versus expected updates and shallow versus deep updates">
    <line x1="150" y1="330" x2="705" y2="330" class="line" marker-end="url(#update-space-arrow)"></line>
    <line x1="150" y1="330" x2="150" y2="65" class="line" marker-end="url(#update-space-arrow)"></line>
    <text x="428" y="380" text-anchor="middle" font-size="14">width of update: sample → expected</text>
    <text x="65" y="205" text-anchor="middle" font-size="14" transform="rotate(-90 65 205)">depth of update: shallow → deep</text>

    <rect x="185" y="210" width="210" height="85" rx="8" class="accent"></rect>
    <text x="290" y="240" text-anchor="middle" font-size="15">Dyna-Q</text>
    <text x="290" y="263" text-anchor="middle" font-size="12" class="muted">sample, one-step</text>

    <rect x="455" y="210" width="210" height="85" rx="8" class="green"></rect>
    <text x="560" y="240" text-anchor="middle" font-size="15">RTDP / DP</text>
    <text x="560" y="263" text-anchor="middle" font-size="12" class="muted">expected, one-step</text>

    <rect x="185" y="95" width="210" height="85" rx="8" class="amber"></rect>
    <text x="290" y="125" text-anchor="middle" font-size="15">Rollout / MCTS</text>
    <text x="290" y="148" text-anchor="middle" font-size="12" class="muted">sample trajectories</text>

    <rect x="455" y="95" width="210" height="85" rx="8" class="box"></rect>
    <text x="560" y="125" text-anchor="middle" font-size="15">Heuristic search</text>
    <text x="560" y="148" text-anchor="middle" font-size="12" class="muted">deep tree + leaf heuristic</text>

    <line x1="425" y1="80" x2="425" y2="315" stroke="#dbe1ee" stroke-width="2" stroke-dasharray="6 5"></line>
    <line x1="170" y1="195" x2="685" y2="195" stroke="#dbe1ee" stroke-width="2" stroke-dasharray="6 5"></line>
    <text x="260" y="350" text-anchor="middle" font-size="12" class="muted">sample</text>
    <text x="560" y="350" text-anchor="middle" font-size="12" class="muted">expected</text>
    <text x="125" y="265" text-anchor="end" font-size="12" class="muted">shallow</text>
    <text x="125" y="140" text-anchor="end" font-size="12" class="muted">deep</text>

    <defs>
      <marker id="update-space-arrow" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
        <path d="M0,0 L7,3 L0,6 Z" fill="#64748b"></path>
      </marker>
    </defs>
  </svg>
  <figcaption>The lecture's methods differ less by algebra than by backup geometry: how wide the update is and how far into the future it looks.</figcaption>
</figure>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The backup shape hides search control)</span></p>

Two methods can share the *same* backup diagram yet behave completely differently, because they update **different states in a different order**. The backup shape says *how* one value is computed; **search control** says *which* values get computed and *when*:

* **uniform sweep** — fair but unfocused, spreads updates evenly;
* **prioritized sweeping** — concentrates on the largest Bellman error;
* **trajectory sampling / RTDP** — concentrates where the policy goes;
* **decision-time planning** — pours all effort onto the *current* state.

The backup diagram is only half the story; the distribution of effort across states is the other half.

</div>

<figure class="rl-diagram">
  <svg viewBox="0 0 820 440" role="img" aria-label="Four search-control patterns: uniform sweep, prioritized sweeping, trajectory sampling, and decision-time planning">
    <text x="205" y="38" text-anchor="middle" font-size="15">Uniform sweep</text>
    <text x="615" y="38" text-anchor="middle" font-size="15">Prioritized sweeping</text>
    <text x="205" y="238" text-anchor="middle" font-size="15">Trajectory sampling / RTDP</text>
    <text x="615" y="238" text-anchor="middle" font-size="15">Decision-time planning</text>

    <g transform="translate(95,60)">
      <rect x="0" y="0" width="220" height="135" rx="8" fill="#ffffff" stroke="#dbe1ee"></rect>
      <circle cx="35" cy="35" r="12" class="amber"></circle><circle cx="85" cy="35" r="12" class="amber"></circle><circle cx="135" cy="35" r="12" class="amber"></circle><circle cx="185" cy="35" r="12" class="amber"></circle>
      <circle cx="35" cy="80" r="12" class="amber"></circle><circle cx="85" cy="80" r="12" class="amber"></circle><circle cx="135" cy="80" r="12" class="amber"></circle><circle cx="185" cy="80" r="12" class="amber"></circle>
      <text x="110" y="120" text-anchor="middle" font-size="12" class="muted">updates spread evenly</text>
    </g>

    <g transform="translate(505,60)">
      <rect x="0" y="0" width="220" height="135" rx="8" fill="#ffffff" stroke="#dbe1ee"></rect>
      <circle cx="35" cy="35" r="12" class="box"></circle><circle cx="85" cy="35" r="12" class="red"></circle><circle cx="135" cy="35" r="12" class="amber"></circle><circle cx="185" cy="35" r="12" class="box"></circle>
      <circle cx="35" cy="80" r="12" class="box"></circle><circle cx="85" cy="80" r="12" class="red"></circle><circle cx="135" cy="80" r="12" class="amber"></circle><circle cx="185" cy="80" r="12" class="box"></circle>
      <path d="M150,35 C125,35 110,35 98,35" class="strong-line" marker-end="url(#search-control-arrow)"></path>
      <text x="110" y="120" text-anchor="middle" font-size="12" class="muted">largest errors first</text>
    </g>

    <g transform="translate(95,260)">
      <rect x="0" y="0" width="220" height="135" rx="8" fill="#ffffff" stroke="#dbe1ee"></rect>
      <circle cx="35" cy="35" r="12" class="accent"></circle><circle cx="85" cy="35" r="12" class="accent"></circle><circle cx="135" cy="35" r="12" class="accent"></circle><circle cx="185" cy="35" r="12" class="green"></circle>
      <circle cx="35" cy="80" r="12" class="box"></circle><circle cx="85" cy="80" r="12" class="box"></circle><circle cx="135" cy="80" r="12" class="box"></circle><circle cx="185" cy="80" r="12" class="green"></circle>
      <path d="M35,35 L85,35 L135,35 L185,35 L185,80" class="strong-line" fill="none" marker-end="url(#search-control-arrow)"></path>
      <text x="110" y="120" text-anchor="middle" font-size="12" class="muted">follow reachable trajectories</text>
    </g>

    <g transform="translate(505,260)">
      <rect x="0" y="0" width="220" height="135" rx="8" fill="#ffffff" stroke="#dbe1ee"></rect>
      <circle cx="110" cy="30" r="15" class="accent"></circle>
      <line x1="100" y1="43" x2="55" y2="85" class="strong-line"></line>
      <line x1="110" y1="45" x2="110" y2="88" class="strong-line"></line>
      <line x1="120" y1="43" x2="165" y2="85" class="line"></line>
      <circle cx="55" cy="96" r="12" class="green"></circle>
      <circle cx="110" cy="100" r="12" class="green"></circle>
      <circle cx="165" cy="96" r="12" class="box"></circle>
      <text x="110" y="120" text-anchor="middle" font-size="12" class="muted">focus on this root state</text>
    </g>

    <defs>
      <marker id="search-control-arrow" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
        <path d="M0,0 L7,3 L0,6 Z" fill="#64748b"></path>
      </marker>
    </defs>
  </svg>
  <figcaption>Search control decides the distribution of computation. The same backup can be weak or powerful depending on whether updates are scattered uniformly, pulled by error, pushed along trajectories, or concentrated on the present decision.</figcaption>
</figure>

### Decision-Time Planning

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Background vs decision-time planning)</span></p>

Planning can be used in two fundamentally different ways:

* **Background planning** — gradually improve a value function or policy that will be *reused across many states*. Example: Dyna. Planning runs "in the background", building reusable knowledge.
* **Decision-time planning** — plan *specifically for the current state*, use the result to choose **one action now**, then discard most of the work and start again next step.

Background planning learns reusable knowledge; decision-time planning **spends computation directly on the current choice**. The methods below — heuristic search, rollout, and MCTS — are all decision-time.

</div>

#### Heuristic Search

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Heuristic search)</span></p>

**Heuristic search** plans at decision time by building a look-ahead tree from the current state:

1. search forward from the current state;
2. expand a look-ahead tree **selectively** (only promising lines);
3. back up values from the leaves toward the root;
4. choose the best root action.

Leaf nodes are evaluated with a **heuristic** value estimate $\hat v$ rather than expanded indefinitely. It is, in effect, a deeper greedy policy: instead of asking only "what happens next?", it asks "what happens after that, and after that?"

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Intuition</span><span class="math-callout__name">(Heuristic search)</span></p>

Heuristic search is like a deeper greedy policy: it does not just ask “what happens next?” but “what happens after that, and after that?”

</div>

<figure>
  <img src="{{ '/assets/images/notes/rl_hd/heuristic_search_example.png' | relative_url }}" alt="Expected return on the short corridor as a function of the probability of choosing right, a smooth curve peaking near 0.59, with the two probability levels reachable by epsilon-greedy marked as dashed vertical lines far from the peak" loading="lazy">
  <!-- <figcaption>The short corridor objective $J(p)$ as a function of $p = \pi(\text{right} \mid s)$. The best stochastic policy sits at $p^* = 2 - \sqrt{2} \approx 0.59$; $\varepsilon$-greedy ($\varepsilon = 0.1$) is confined to the two dashed lines, earning about $-82$ or $-44$.</figcaption> -->
</figure>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Backing up a small look-ahead tree)</span></p>

Take a tree rooted at $s$ with $\gamma = 1$. From $s$, action $a\_1$ leads to node $x$ and action $a\_2$ to a leaf valued $4$. From $x$, action $b\_1$ leads through a **stochastic** action $o\_1$ to leaves $\lbrace 12, 2\rbrace$ with equal probability, and $b\_2$ to a leaf valued $6$. Backing up bottom-up:

* **average** at the stochastic node $o\_1$: 
  
  $$\mathbb{E} = 0.5(12) + 0.5(2) = 7$$

* **max** at each decision node: 
  
  $$v(m) = \max(7, 5) = 7, \text{ then } v(x) = \max(7, 6) = 7;$$

* at the root: 
  
  $$Q(s, a_1) = 7, Q(s, a_2) = 4 \Rightarrow \text{ play } a_1.$$

Bottom-up, the operators **alternate**: *average* at stochastic (chance) nodes, *max* at decision nodes. Expanding only promising lines — selective depth — is what makes the search *heuristic* rather than exhaustive.

</div>

<figure class="rl-diagram">
  <svg viewBox="0 0 820 420" role="img" aria-label="Heuristic search tree with max backups at decision nodes and average backups at chance nodes">
    <circle cx="410" cy="45" r="22" class="accent"></circle>
    <text x="410" y="50" text-anchor="middle" font-size="14">s</text>
    <text x="410" y="18" text-anchor="middle" font-size="12" class="muted">choose max at root</text>

    <line x1="394" y1="61" x2="278" y2="120" class="strong-line"></line>
    <line x1="426" y1="61" x2="542" y2="120" class="line"></line>
    <text x="322" y="85" text-anchor="middle" font-size="12" class="muted">a1</text>
    <text x="500" y="85" text-anchor="middle" font-size="12" class="muted">a2</text>

    <circle cx="260" cy="135" r="22" class="accent"></circle>
    <text x="260" y="140" text-anchor="middle" font-size="14">x</text>
    <rect x="515" y="113" width="70" height="45" rx="8" class="box"></rect>
    <text x="550" y="140" text-anchor="middle" font-size="14">v=4</text>

    <line x1="248" y1="154" x2="188" y2="215" class="strong-line"></line>
    <line x1="272" y1="154" x2="335" y2="215" class="line"></line>
    <text x="209" y="182" text-anchor="middle" font-size="12" class="muted">b1</text>
    <text x="317" y="182" text-anchor="middle" font-size="12" class="muted">b2</text>

    <circle cx="175" cy="232" r="20" class="amber"></circle>
    <text x="175" y="237" text-anchor="middle" font-size="13">o1</text>
    <rect x="315" y="211" width="70" height="42" rx="8" class="box"></rect>
    <text x="350" y="237" text-anchor="middle" font-size="14">v=6</text>

    <line x1="163" y1="249" x2="112" y2="310" class="line"></line>
    <line x1="187" y1="249" x2="238" y2="310" class="line"></line>
    <text x="125" y="282" text-anchor="middle" font-size="11" class="muted">0.5</text>
    <text x="224" y="282" text-anchor="middle" font-size="11" class="muted">0.5</text>

    <rect x="75" y="315" width="70" height="42" rx="8" class="green"></rect>
    <text x="110" y="341" text-anchor="middle" font-size="14">12</text>
    <rect x="205" y="315" width="70" height="42" rx="8" class="green"></rect>
    <text x="240" y="341" text-anchor="middle" font-size="14">2</text>

    <rect x="475" y="235" width="260" height="118" rx="8" class="box"></rect>
    <text x="605" y="266" text-anchor="middle" font-size="13">chance node: average</text>
    <text x="605" y="291" text-anchor="middle" font-size="13">0.5 · 12 + 0.5 · 2 = 7</text>
    <text x="605" y="320" text-anchor="middle" font-size="13">decision nodes: max</text>
    <text x="605" y="343" text-anchor="middle" font-size="13">max(7, 6) = 7; max(7, 4) = 7</text>

    <path d="M110,315 C155,285 183,272 175,255" class="strong-line" stroke-dasharray="5 4" marker-end="url(#heuristic-arrow)"></path>
    <path d="M175,210 C185,175 220,152 238,141" class="strong-line" stroke-dasharray="5 4" marker-end="url(#heuristic-arrow)"></path>
    <path d="M260,112 C300,70 360,56 388,49" class="strong-line" stroke-dasharray="5 4" marker-end="url(#heuristic-arrow)"></path>

    <text x="320" y="330" text-anchor="middle" font-size="12" class="muted">dashed arrows show bottom-up backup flow</text>

    <defs>
      <marker id="heuristic-arrow" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
        <path d="M0,0 L7,3 L0,6 Z" fill="#64748b"></path>
      </marker>
    </defs>
  </svg>
  <figcaption>Heuristic search builds a finite look-ahead tree, evaluates leaves, then backs values up toward the root. Decision nodes take maxima; chance nodes take expectations.</figcaption>
</figure>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Heuristic search)</span></p>

At the current state $s$:

1. build a look-ahead tree rooted at $s$;
2. expand promising actions up to a depth or budget limit;
3. evaluate leaves with a heuristic or value estimate $\hat v$;
4. back up values from leaves to root (max at decision nodes, expectation at chance nodes);
5. choose the root action with the largest backed-up value.

**Key idea:** spend computation on the *current decision only* — do not sweep the whole state space.

</div>

#### Rollout Algorithms

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Rollout algorithm)</span></p>

A **rollout algorithm** is decision-time Monte Carlo planning. For each candidate action available now:

* simulate many trajectories that begin with that action;
* after the first action, follow a fixed **rollout policy**;
* estimate the action's value by the **average return** of those trajectories;
* pick the best-looking current action.

Rollout does **not** learn a full global value function — it estimates *just enough* to decide what to do now, then throws the estimates away and repeats at the next state.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Rollout)</span></p>

**Inputs:** current state $s$; rollout policy $\pi$; number of simulations $K$ per candidate action.

1. For each action $a \in \mathcal{A}(s)$:
   * simulate $K$ trajectories starting with $a$;
   * follow $\pi$ after the first step;
   * average the returns,
     
     $$\hat q(s, a) = \frac{1}{K} \sum_{i=1}^{K} G^{(i)}.$$

2. Choose the action with the largest $\hat q(s, a)$.

A better rollout policy $\pi$ gives a better estimate; more simulations $K$ reduce its variance.

</div>

<figure class="rl-diagram">
  <svg viewBox="0 0 820 360" role="img" aria-label="Rollout planning simulates multiple trajectories for each candidate action and averages returns to choose the current action">
    <circle cx="100" cy="175" r="24" class="accent"></circle>
    <text x="100" y="180" text-anchor="middle" font-size="14">s</text>
    <text x="100" y="130" text-anchor="middle" font-size="12" class="muted">current state</text>

    <rect x="210" y="55" width="80" height="42" rx="8" class="box"></rect>
    <rect x="210" y="154" width="80" height="42" rx="8" class="green"></rect>
    <rect x="210" y="253" width="80" height="42" rx="8" class="box"></rect>
    <text x="250" y="81" text-anchor="middle" font-size="13">a1</text>
    <text x="250" y="180" text-anchor="middle" font-size="13">a2</text>
    <text x="250" y="279" text-anchor="middle" font-size="13">a3</text>

    <line x1="123" y1="162" x2="206" y2="82" class="line"></line>
    <line x1="124" y1="175" x2="206" y2="175" class="strong-line"></line>
    <line x1="123" y1="188" x2="206" y2="269" class="line"></line>

    <path d="M295,76 C370,35 460,35 535,68" class="line" marker-end="url(#rollout-arrow)"></path>
    <path d="M295,76 C370,78 455,88 535,90" class="line" marker-end="url(#rollout-arrow)"></path>
    <path d="M295,76 C370,112 455,112 535,103" class="line" marker-end="url(#rollout-arrow)"></path>

    <path d="M295,175 C370,132 460,134 535,154" class="strong-line" marker-end="url(#rollout-arrow)"></path>
    <path d="M295,175 C370,175 455,174 535,175" class="strong-line" marker-end="url(#rollout-arrow)"></path>
    <path d="M295,175 C370,218 455,218 535,196" class="strong-line" marker-end="url(#rollout-arrow)"></path>

    <path d="M295,274 C370,232 460,236 535,253" class="line" marker-end="url(#rollout-arrow)"></path>
    <path d="M295,274 C370,274 455,276 535,274" class="line" marker-end="url(#rollout-arrow)"></path>
    <path d="M295,274 C370,315 455,315 535,296" class="line" marker-end="url(#rollout-arrow)"></path>

    <rect x="575" y="50" width="175" height="65" rx="8" class="box"></rect>
    <text x="662" y="78" text-anchor="middle" font-size="13">mean return</text>
    <text x="662" y="99" text-anchor="middle" font-size="13">mean(a1)=3.8</text>
    <rect x="575" y="143" width="175" height="65" rx="8" class="green"></rect>
    <text x="662" y="171" text-anchor="middle" font-size="13">mean return</text>
    <text x="662" y="192" text-anchor="middle" font-size="13">mean(a2)=5.4</text>
    <rect x="575" y="236" width="175" height="65" rx="8" class="box"></rect>
    <text x="662" y="264" text-anchor="middle" font-size="13">mean return</text>
    <text x="662" y="285" text-anchor="middle" font-size="13">mean(a3)=4.1</text>

    <text x="405" y="330" text-anchor="middle" font-size="12" class="muted">after the first action, every trajectory follows the rollout policy π</text>

    <defs>
      <marker id="rollout-arrow" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
        <path d="M0,0 L7,3 L0,6 Z" fill="#64748b"></path>
      </marker>
    </defs>
  </svg>
  <figcaption>Rollout planning is flat decision-time Monte Carlo: estimate each candidate root action by simulated returns, then take the action with the largest average.</figcaption>
</figure>

#### Monte Carlo Tree Search

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Monte Carlo Tree Search, MCTS)</span></p>

**MCTS** improves on plain rollout by **saving and refining statistics in a partial search tree** instead of restarting from scratch each time. It repeats four steps while time remains:

1. **Selection** — from the root, descend the *existing* tree by the **tree policy** to a node with an untried action.
2. **Expansion** — add one new node.
3. **Simulation** — roll out from that node to the end using the **rollout policy**, returning a return $G$.
4. **Backup** — propagate $G$ up the visited path, updating each node's statistics.

MCTS focuses simulation effort on **promising prefixes** rather than restarting every trajectory from the root.

</div>

<figure>
  <img src="{{ '/assets/images/notes/rl_hd/MCTS01.png' | relative_url }}" alt="Expected return on the short corridor as a function of the probability of choosing right, a smooth curve peaking near 0.59, with the two probability levels reachable by epsilon-greedy marked as dashed vertical lines far from the peak" loading="lazy">
  <!-- <figcaption>The short corridor objective $J(p)$ as a function of $p = \pi(\text{right} \mid s)$. The best stochastic policy sits at $p^* = 2 - \sqrt{2} \approx 0.59$; $\varepsilon$-greedy ($\varepsilon = 0.1$) is confined to the two dashed lines, earning about $-82$ or $-44$.</figcaption> -->
</figure>

<figure class="rl-diagram">
  <svg viewBox="0 0 760 320" role="img" aria-label="MCTS search tree: a root node with visit counts N and mean values Q, an in-memory tree, and a dashed frontier where rollouts begin; the tree policy selects inside the tree and the rollout policy takes over beyond the frontier">
    <!-- root -->
    <circle cx="380" cy="40" r="20" class="accent"></circle>
    <text x="380" y="20" text-anchor="middle" font-size="12" class="muted">N=40</text>

    <!-- level 1 -->
    <line x1="380" y1="60" x2="250" y2="120" class="strong-line"></line>
    <line x1="380" y1="60" x2="510" y2="120" class="line"></line>
    <circle cx="250" cy="135" r="18" class="accent"></circle>
    <text x="250" y="113" text-anchor="middle" font-size="12" class="muted">N=28, Q=.70</text>
    <circle cx="510" cy="135" r="18" class="box" stroke-dasharray="4 3"></circle>
    <text x="510" y="113" text-anchor="middle" font-size="12" class="muted">N=12, Q=.30</text>

    <!-- level 2 -->
    <line x1="250" y1="153" x2="180" y2="210" class="strong-line"></line>
    <line x1="250" y1="153" x2="320" y2="210" class="line"></line>
    <circle cx="180" cy="225" r="16" class="accent"></circle>
    <text x="180" y="205" text-anchor="middle" font-size="12" class="muted">N=18, Q=.80</text>
    <circle cx="320" cy="225" r="16" class="box" stroke-dasharray="4 3"></circle>
    <text x="320" y="205" text-anchor="middle" font-size="11" class="muted">N=10</text>

    <!-- level 3 (frontier) -->
    <line x1="180" y1="241" x2="180" y2="285" class="strong-line"></line>
    <circle cx="180" cy="298" r="14" class="box" stroke-dasharray="4 3"></circle>
    <text x="150" y="302" text-anchor="end" font-size="11" class="muted">N=3</text>

    <!-- rollout arrow from a frontier node -->
    <line x1="320" y1="241" x2="320" y2="288" class="line" stroke-dasharray="3 3" marker-end="url(#mah)"></line>
    <text x="430" y="270" text-anchor="middle" font-size="12" class="muted">rollouts start here</text>
    <text x="430" y="288" text-anchor="middle" font-size="12" class="muted">(dashed = frontier)</text>

    <defs>
      <marker id="mah" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
        <path d="M0,0 L7,3 L0,6 Z" fill="#64748b"></path>
      </marker>
    </defs>
  </svg>
  <figcaption>The MCTS tree lives in memory: each node stores a visit count $N$ and a mean value $Q$. It is <strong>partial and local</strong> (only states reachable from now), <strong>grown from scratch</strong> (one node per iteration), and split by a frontier — the <strong>tree policy</strong> acts inside, the <strong>rollout policy</strong> beyond.</figcaption>
</figure>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(One MCTS iteration in detail)</span></p>

A single iteration touches the tree as follows:

1. **Select** — from the root, descend the existing tree by the tree policy to a node with an untried action.
2. **Expand** — add *one* new node.
3. **Simulate** — roll out from it to the end, returning a return $G$.
4. **Backup** — along the path just taken, increment each visit count and fold $G$ into each running mean:
   
   $$N \leftarrow N + 1, \qquad Q \leftarrow Q + \tfrac{1}{N}\,(G - Q).$$

The rollout trajectory itself is **not stored** — only the statistics on the in-tree nodes are kept. The tree thus stores only the slice of the future worth thinking about for the decision in front of the agent.

</div>

<figure>
  <img src="{{ '/assets/images/notes/rl_hd/MCTS02.png' | relative_url }}" alt="Expected return on the short corridor as a function of the probability of choosing right, a smooth curve peaking near 0.59, with the two probability levels reachable by epsilon-greedy marked as dashed vertical lines far from the peak" loading="lazy">
  <!-- <figcaption>The short corridor objective $J(p)$ as a function of $p = \pi(\text{right} \mid s)$. The best stochastic policy sits at $p^* = 2 - \sqrt{2} \approx 0.59$; $\varepsilon$-greedy ($\varepsilon = 0.1$) is confined to the two dashed lines, earning about $-82$ or $-44$.</figcaption> -->
</figure>

<figure class="rl-diagram">
  <svg viewBox="0 0 820 390" role="img" aria-label="One MCTS iteration selects a path, expands a new node, simulates a rollout, and backs up the return to update N and Q">
    <text x="410" y="35" text-anchor="middle" font-size="15">One MCTS iteration: select → expand → simulate → backup</text>

    <circle cx="410" cy="80" r="22" class="accent"></circle>
    <text x="410" y="85" text-anchor="middle" font-size="13">root</text>
    <text x="410" y="50" text-anchor="middle" font-size="12" class="muted">N: 40→41</text>

    <line x1="397" y1="98" x2="300" y2="158" class="strong-line"></line>
    <line x1="423" y1="98" x2="520" y2="158" class="line"></line>
    <circle cx="285" cy="170" r="20" class="accent"></circle>
    <text x="285" y="175" text-anchor="middle" font-size="12">A</text>
    <text x="285" y="143" text-anchor="middle" font-size="12" class="muted">N: 28→29</text>
    <circle cx="535" cy="170" r="20" class="box"></circle>
    <text x="535" y="175" text-anchor="middle" font-size="12">B</text>

    <line x1="274" y1="187" x2="220" y2="242" class="strong-line"></line>
    <line x1="296" y1="187" x2="350" y2="242" class="line"></line>
    <circle cx="208" cy="255" r="18" class="accent"></circle>
    <text x="208" y="260" text-anchor="middle" font-size="12">C</text>
    <text x="208" y="229" text-anchor="middle" font-size="12" class="muted">N: 18→19</text>
    <circle cx="362" cy="255" r="18" class="box"></circle>
    <text x="362" y="260" text-anchor="middle" font-size="12">D</text>

    <line x1="208" y1="273" x2="208" y2="315" class="strong-line"></line>
    <circle cx="208" cy="335" r="17" class="green"></circle>
    <text x="208" y="340" text-anchor="middle" font-size="12">new</text>
    <text x="260" y="339" text-anchor="start" font-size="12" class="muted">expand: N 0→1, Q 0→G</text>

    <path d="M225,335 C340,365 470,340 590,300" class="line" stroke-dasharray="5 4" marker-end="url(#mcts-iter-arrow)"></path>
    <text x="440" y="363" text-anchor="middle" font-size="12" class="muted">simulate rollout beyond the frontier</text>

    <rect x="590" y="250" width="145" height="58" rx="8" class="amber"></rect>
    <text x="662" y="276" text-anchor="middle" font-size="13">return</text>
    <text x="662" y="297" text-anchor="middle" font-size="13">G = 1</text>

    <path d="M610,248 C490,188 345,170 305,170" class="strong-line" stroke-dasharray="5 4" marker-end="url(#mcts-iter-arrow)"></path>
    <path d="M305,158 C360,120 388,105 403,100" class="strong-line" stroke-dasharray="5 4" marker-end="url(#mcts-iter-arrow)"></path>
    <text x="590" y="190" text-anchor="middle" font-size="12" class="muted">backup G along selected path</text>
    <text x="590" y="211" text-anchor="middle" font-size="12" class="muted">Q ← Q + (G − Q)/N</text>

    <defs>
      <marker id="mcts-iter-arrow" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
        <path d="M0,0 L7,3 L0,6 Z" fill="#64748b"></path>
      </marker>
    </defs>
  </svg>
  <figcaption>One MCTS iteration stores only the new tree node and the updated statistics on the selected path. The rollout is used for its return, then discarded.</figcaption>
</figure>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(Choosing the real move, then moving on)</span></p>

After the simulation budget is spent, pick the real move by

$$\arg\max_{a} N(s_0, a),$$

the **most-visited** root action — *not* the one with the highest mean $Q$. Why most visits rather than best mean?

* the tree policy already uses $Q$ during search, so promising moves keep attracting simulations and accumulate high $N$;
* a high $Q$ on *few* visits may just be luck — $N$ is the more robust signal.

**Then move on (decision-time):** play the chosen action, observe the true $s', r$, **reuse the subtree** under that move as the new root, and replan from $s'$.

</div>

<figure>
  <img src="{{ '/assets/images/notes/rl_hd/MCTS03.png' | relative_url }}" alt="Expected return on the short corridor as a function of the probability of choosing right, a smooth curve peaking near 0.59, with the two probability levels reachable by epsilon-greedy marked as dashed vertical lines far from the peak" loading="lazy">
  <!-- <figcaption>The short corridor objective $J(p)$ as a function of $p = \pi(\text{right} \mid s)$. The best stochastic policy sits at $p^* = 2 - \sqrt{2} \approx 0.59$; $\varepsilon$-greedy ($\varepsilon = 0.1$) is confined to the two dashed lines, earning about $-82$ or $-44$.</figcaption> -->
</figure>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Takeaway</span><span class="math-callout__name">(Why MCTS mattered)</span></p>

MCTS combines several ideas this lecture has built up:

* it marries **decision-time planning** with **incremental sample-based value estimation**;
* it stores only a **partial tree** near the current state;
* it avoids globally approximating action values everywhere;
* it **reuses previous simulations** to guide future exploration.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Method</span><span class="math-callout__name">(Decision-time planning: heuristic search → rollout → MCTS)</span></p>

| | **Heuristic search** | **Rollout** | **MCTS** |
| :--- | :--- | :--- | :--- |
| Structure built | fixed-depth tree | none — flat trajectories | tree, grown 1 node/iter |
| Leaf value from | heuristic $\hat v$ | full rollout return | rollout return, then stored |
| Backup | one max/expectation pass | average of $K$ returns | incremental running mean |
| Search control | expand to depth $d$ | $K$ sims per action (uniform) | selection: explore/exploit |
| Memory kept? | no (rebuilt each move) | no | **yes** — reused next move |
| More compute means | deeper $d$ | larger $K$ | more iterations |

Heuristic search needs a good $\hat v$; rollout drops it for sampled returns; **MCTS keeps the sampled returns and adds a reusable tree plus selective search on top.**

</div>

### Summary

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Method</span><span class="math-callout__name">(Method map — what each idea adds)</span></p>

| Method | Core idea | Strength | Main limitation |
| :--- | :--- | :--- | :--- |
| Dyna-Q | learn and plan with the same update rule | makes each real step more valuable | depends on model quality |
| Dyna-Q+ | exploration bonus in planning | adapts to change better | more exploratory cost |
| Prioritized sweeping | back up urgent predecessors first | great search efficiency | needs predecessor bookkeeping |
| Expected updates | exact given the model | no sampling error | can be expensive |
| Sample updates | cheap stochastic backups | many backups quickly | noisy |
| RTDP | on-policy value iteration by trajectories | focuses on relevant states | best for goal-directed structure |
| Rollout | simulate candidate actions now | strong decision-time planning | little long-term memory |
| MCTS | build and reuse a partial search tree | focused effort at decision time | more machinery / tuning |

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Idea</span><span class="math-callout__name">(The dimensions view)</span></p>

All of Part I of the course fits in a single design space spanned by two axes:

* **horizontal axis — width of the update:** *sample* $\leftrightarrow$ *expected* (one successor vs. all of them);
* **vertical axis — depth of the update:** *shallow* $\leftrightarrow$ *deep* (one step vs. full return).

The four corners are familiar: **TD learning** (sample, shallow), **dynamic programming** (expected, shallow), **Monte Carlo** (sample, deep), and **exhaustive search** (expected, deep). Every planning method in this lecture is just a different position — and a different *search control* — in the same space.

</div>

<figure>
  <img src="{{ '/assets/images/notes/rl_hd/planning_dimensions_view.png' | relative_url }}" alt="Expected return on the short corridor as a function of the probability of choosing right, a smooth curve peaking near 0.59, with the two probability levels reachable by epsilon-greedy marked as dashed vertical lines far from the peak" loading="lazy">
  <!-- <figcaption>The short corridor objective $J(p)$ as a function of $p = \pi(\text{right} \mid s)$. The best stochastic policy sits at $p^* = 2 - \sqrt{2} \approx 0.59$; $\varepsilon$-greedy ($\varepsilon = 0.1$) is confined to the two dashed lines, earning about $-82$ or $-44$.</figcaption> -->
</figure>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(Core takeaways — Planning and Learning)</span></p>

* **A model lets us create simulated experience.** Distribution models give probabilities; sample models give one outcome and are usually easier to obtain.
* **Planning and learning share machinery.** Both back up values; they differ only in whether the experience is real or simulated — so the same update rule serves both.
* **Dyna integrates direct RL, model learning, and planning** in one loop, turning each real transition into many updates.
* **Models can be wrong**, and planning amplifies their errors; **exploration (Dyna-Q+) is model maintenance**, re-testing stale transitions to keep the model honest.
* **Search control is crucial** — *where* you spend updates matters enormously. Prioritized sweeping works backward by Bellman error; trajectory sampling / RTDP works forward by on-policy visitation.
* **Sample updates can outperform expected updates** when computation, not data, is the bottleneck.
* **Decision-time planning** — heuristic search, rollout, and **MCTS** — shows increasingly focused ways to spend computation on the *current* decision.

**Final message.** *Planning is reinforcement learning applied to imagined experience.* Once that equivalence is clear, the whole tabular toolkit — backups, bootstrapping, exploration, and search control — transfers directly to the model-based setting, setting up the move from tables to **function approximation** in Part II.

</div>

## Lecture 9: On-Policy Prediction with Function Approximation

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Setup</span><span class="math-callout__name">(From tables to parameterized functions)</span></p>

Part I closed by reframing planning as *reinforcement learning on imagined experience* — but every method so far, real or simulated, stored **one number per state** (or per state–action pair). This lecture opens Part II by removing that assumption. We keep the *prediction* problem — estimate $v\_\pi$ for a fixed policy $\pi$ — but replace the lookup table $V(s)$ by a **parameterized function** $\hat v(s,\mathbf{w}) \approx v\_\pi(s)$ whose weight vector $\mathbf{w}$ has far fewer components than there are states.

The build-up is:

1. **Why tabular methods break.** When the state space is large or continuous, a table is both too big to store *and* too sparse to fill — almost every state is seen at most once.
2. **The objective.** Once we cannot be exactly right everywhere, we must say *which errors matter*. This gives the **mean-squared value error** $\overline{\text{VE}}$, weighted by how often the policy visits each state.
3. **Stochastic gradient descent.** Minimize $\overline{\text{VE}}$ by SGD; understand why **bootstrapping** breaks the true-gradient guarantee and forces **semi-gradient** methods.
4. **Linear methods.** The cleanest, most analyzable case — gradient equals the feature vector, the objective is convex, and on-policy TD has a convergence guarantee with a known error bound.
5. **Feature construction.** State aggregation, Fourier bases, coarse / tile coding, and RBFs — the design choices that usually matter more than the algorithm.
6. **Nonlinear approximation.** Neural networks lift the feature-design burden but lose the theory, leaning on a toolbox of stability tricks instead.

The guiding question of the lecture is:

> *How do we evaluate a policy when we cannot store one number per state, and almost every state we encounter is effectively new?*

</div>

### Why Tabular Methods Break

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Problem</span><span class="math-callout__name">(The double trouble with tables)</span></p>

Tabular RL stores one number per state and works only when the state set is small **and** every relevant state is visited many times. Real problems violate both conditions: their states are **continuous** (sensors, joint angles), **very high-dimensional** (images, structured descriptions), or **combinatorial** (positions of many objects). Two failures strike at once:

* **(a) Memory.** A table indexed by states is simply too large to store.
* **(b) Coverage.** Most states are seen *at most once* — so even if memory were free, repeated visits would never accumulate to refine each entry.

Trouble (b) is the deeper one: it survives even infinite memory, and it is what forces us to **generalize** rather than merely compress.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(How astronomically large an image state space is)</span></p>

Suppose the state is a small grayscale image, $84 \times 84$ pixels, each pixel one of $256$ levels. The number of possible configurations is

$$256^{\,84 \times 84} \;=\; 256^{\,7056},$$

which is astronomically larger than the number of atoms in the observable universe. No table can index such a set, and no agent could ever revisit a particular image. The only way forward is to **act sensibly in states never seen before**, by exploiting regularities learned from the states that *were* seen.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Idea</span><span class="math-callout__name">(Generalization is the central issue)</span></p>

In large state spaces *almost every* encountered state is effectively new. Learning must therefore rely on **using experience in one set of states to make predictions in many others**. This is exactly what function approximation buys us:

* a **compact** representation — far fewer parameters than states;
* **controlled sharing** of information across states — one experience can improve the estimate at many related states at once.

Generalization is not a bonus here; it is the only thing that makes the problem learnable at all.

</div>

### From Tables to Parameterized Functions

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Parameterized value function)</span></p>

Replace the table $V(s)$ by a parameterized approximation

$$\hat v(s,\mathbf{w}) \;\approx\; v_\pi(s), \qquad \mathbf{w} \in \mathbb{R}^{d}, \qquad d \ll |\mathcal{S}|,$$

where $\mathbf{w}$ is the **weight vector**, updated incrementally. Three consequences follow immediately from $d \ll \lvert\mathcal{S}\rvert$:

* with far fewer parameters than states, **exact** equality $\hat v = v\_\pi$ is impossible in general;
* one weight change affects the prediction at **many** states at once — so generalization is automatic;
* this is also the *cost* of generalization: updates **couple** states. Reducing the error at one state cannot leave all the others untouched.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(Examples of function approximators)</span></p>

The form of $\hat v$ is a free design choice. Common families:

* **Linear in features:** $\hat v(s,\mathbf{w}) = \mathbf{w}^\top \mathbf{x}(s)$ — a weighted sum of fixed features. Most of the theory lives here.
* **Neural networks:** $\hat v(s,\mathbf{w})$ a multi-layer nonlinear function of $\mathbf{w}$.
* **Decision / regression trees.**
* **Memory-based / kernel methods.**

Anything differentiable in $\mathbf{w}$ can be plugged into the gradient machinery below; the linear and neural cases dominate practice.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Idea</span><span class="math-callout__name">(The "$s \mapsto u$" update view)</span></p>

Every prediction update we have ever seen has the same shape

$$
S_t \;\mapsto\; U_t,
$$

read as: *"the value at $S\_t$ should look more like the target $U\_t$."* The methods differ only in the **target** $U\_t$:

* **Monte Carlo:** $U\_t = G\_t$.
* **TD(0):** $U\_t = R\_{t+1} + \gamma\,\hat v(S\_{t+1},\mathbf{w})$.
* **$n$-step TD:** $U\_t = G\_{t:t+n}$.
* **DP:** $U\_t = \mathbb{E}\_\pi[\,R\_{t+1} + \gamma\,\hat v(S\_{t+1},\mathbf{w}) \mid S\_t\,]$.

**The reframing:** RL produces a stream of training examples $S\_t \mapsto U\_t$, and a function approximator just has to fit them. The twist is that in ordinary regression the targets are external labels handed to us by an *oracle*; here we **compute the targets ourselves**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why RL approximation is harder than supervised learning)</span></p>

Plugging a regression method into RL is not as innocent as the $S\_t\mapsto U\_t$ view suggests. Three difficulties have no analogue in ordinary supervised learning:

* **We learn online**, from states visited under the **state-visitation distribution** $d\_\pi$. Nearby examples are *correlated*, coverage is uneven, and the shared weights couple the effects of different updates.
* **Bootstrapping makes the target depend on the current weights:** 
  
  $$U_t = R_{t+1} + \gamma\,\hat v(S_{t+1},\mathbf{w}_t)$$
  
  The "label" therefore **moves as we learn** — we are chasing a target we are simultaneously editing.
* **In control, the value function feeds back into the policy**, which changes the future data distribution — the problem is *nonstationary* by construction.

The three recurring headaches are thus **coupled updates**, **moving targets / bootstrapping bias**, and **nonstationarity**. Keep them in mind; the rest of the lecture is largely about taming them.

</div>

### The Prediction Objective $\overline{\text{VE}}$

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Problem</span><span class="math-callout__name">(From a tabular update to a loss function)</span></p>

The tabular prediction update

$$V(S_t) \;\leftarrow\; V(S_t) + \alpha\,[\,U_t - V(S_t)\,]$$

gives each state its **own** estimate; updating $S\_t$ leaves $V(s)$ for every other state untouched. With function approximation

$$\hat v(s,\mathbf{w}) \approx v_\pi(s),$$

a *single* weight vector $\mathbf{w}$ is shared across many states, so changing $\mathbf{w}$ to improve one state can change predictions elsewhere. This raises a question that simply did not exist in the tabular world: **when you cannot be right everywhere, what should count as the best compromise across states?**

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Mean-squared value error $\overline{\text{VE}}$)</span></p>

Choose a non-negative **state weighting** $\mu(s)$ saying how much we care about being accurate at each state, with

$$\mu(s) \geq 0, \qquad \sum_s \mu(s) = 1.$$

The objective is the **mean-squared value error**

$$\overline{\text{VE}}(\mathbf{w}) \;=\; \sum_s \mu(s)\,\bigl[\,v_\pi(s) - \hat v(s,\mathbf{w})\,\bigr]^2 .$$

For **on-policy prediction** the natural choice is $\mu(s) = $ *how often policy $\pi$ visits state $s$.* States we visit often weigh more — both because we have more data there **and** because those are the states relevant to evaluating $\pi$ in the first place.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(What distribution is $\mu$?)</span></p>

$\mu$ is not an arbitrary modelling knob — it is the state distribution *generated by the prediction problem we actually run*:

* In a **continuing task**, assuming the Markov chain induced by $\pi$ is ergodic, $\mu(s)$ is the **long-run fraction of time** spent in $s$ while following $\pi$ — the stationary distribution $d\_\pi$.
* In an **episodic task**, $\mu(s)$ depends on **both** the start-state distribution and $\pi$.

A practical consequence: changing *where episodes start* changes which states are visited often, which changes $\mu$ — and therefore changes what counts as a good approximation. Same policy and same environment can yield a different $\overline{\text{VE}}$ landscape under a different start distribution.

</div>

### Stochastic Gradient Descent

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Derivation</span><span class="math-callout__name">(From $\overline{\text{VE}}$ to an SGD update)</span></p>

The objective is a $\mu$-weighted sum over **all** states:

$$\overline{\text{VE}}(\mathbf{w}) \;=\; \sum_s \mu(s)\,\bigl[\,v_\pi(s) - \hat v(s,\mathbf{w})\,\bigr]^2 .$$

Differentiating keeps the same $\mu$-weighting, and the $\mu$-weighted sum is exactly an expectation over $S \sim \mu$:

$$
\nabla_{\mathbf{w}}\,\overline{\text{VE}}(\mathbf{w})
\;=\; -2\sum_s \mu(s)\,\bigl[\,v_\pi(s) - \hat v(s,\mathbf{w})\,\bigr]\,\nabla_{\mathbf{w}}\hat v(s,\mathbf{w})
\;=\; -2\,\mathbb{E}_{S \sim \mu}\!\bigl[\,(v_\pi(S) - \hat v(S,\mathbf{w}))\,\nabla_{\mathbf{w}}\hat v(S,\mathbf{w})\,\bigr].
$$

Under on-policy sampling, visited states occur with long-run frequency $\mu(s)$, so the **visited state is itself a stochastic sample** of this expectation:

$$-\tfrac{1}{2}\,\nabla_{\mathbf{w}}\,\overline{\text{VE}}(\mathbf{w}) \;\approx\; \bigl[\,v_\pi(S_t) - \hat v(S_t,\mathbf{w})\,\bigr]\,\nabla_{\mathbf{w}}\hat v(S_t,\mathbf{w}).$$

Following this sample gradient downhill on the per-sample squared error gives, assuming for now that we know $v\_\pi$,

$$
\mathbf{w}_{t+1}
\;=\; \mathbf{w}_t - \alpha\,\nabla_{\mathbf{w}}\tfrac{1}{2}\bigl[\,v_\pi(S_t) - \hat v(S_t,\mathbf{w}_t)\,\bigr]^2
\;=\; \mathbf{w}_t + \alpha\,\bigl[\,v_\pi(S_t) - \hat v(S_t,\mathbf{w}_t)\,\bigr]\,\nabla_{\mathbf{w}}\hat v(S_t,\mathbf{w}_t).
$$

The step is deliberately **small**: driving the error at $S\_t$ all the way to zero in one move would worsen the errors at every other state sharing those parameters.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Convergence with a sample target $U_t$)</span></p>

We never actually know $v\_\pi(S\_t)$, so substitute a **target** $U\_t$:

$$\mathbf{w}_{t+1} \;=\; \mathbf{w}_t + \alpha\,\bigl[\,U_t - \hat v(S_t,\mathbf{w}_t)\,\bigr]\,\nabla_{\mathbf{w}}\hat v(S_t,\mathbf{w}_t).$$

If the target is **correct in expectation** (unbiased),

$$\mathbb{E}[\,U_t \mid S_t = s\,] \;=\; v_\pi(s),$$

then under standard stochastic-approximation conditions — a Robbins–Monro step-size schedule $\alpha\_t$ that decreases to zero but not too quickly — the weights converge to a **local minimum** of $\overline{\text{VE}}$:

$$\mathbf{w}_t \;\to\; \mathbf{w}^\ast, \qquad \mathbf{w}^\ast \text{ a local minimum of } \overline{\text{VE}}.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Caveat of convergence: Bootstrapping targets are generally biased)</span></p>

Bootstrapping targets are generally **biased**, so they fall outside this guarantee. And in nonstationary problems one often keeps a **constant** step size $\alpha$ instead, so the estimate keeps adapting rather than fully converging.

</div>

### Monte Carlo with Function Approximation

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Gradient Monte Carlo)</span></p>

With $U\_t = G\_t$ (the actual return) the target is **unbiased**, so the SGD theorem applies directly. The update is

$$
\mathbf{w} \;\leftarrow\; \mathbf{w} + \alpha\,\bigl[\,G_t - \hat v(S_t,\mathbf{w})\,\bigr]\,\nabla_{\mathbf{w}}\hat v(S_t,\mathbf{w}).
$$

**One episode:**

1. Generate $S\_0, A\_0, R\_1, \dots, R\_T, S\_T$ following $\pi$.
2. For $t = 0, 1, \dots, T-1$: apply the update above with the realized return $G\_t$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Monte Carlo with Function Approximation)</span></p>

* **Pro:** true SGD on $\overline{\text{VE}}$ — converges to a local optimum (and, in the linear case, the global one). 
* **Con:** must wait until the episode ends, and the returns are **high variance**.

</div>

### Bootstrapping and Semi-Gradients

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Bootstrapping breaks the SGD guarantee)</span></p>

TD(0)'s target depends on $\mathbf{w}\_t$ **itself**:

$$U_t \;=\; R_{t+1} + \gamma\,\hat v(S_{t+1},\mathbf{w}_t).$$

* A *true* gradient method would differentiate through **both** appearances of the approximation — $\hat v(S\_t,\cdot)$ and $\hat v(S\_{t+1},\cdot)$. 
* But the bootstrapped target is not an external label; it moves with $\mathbf{w}$.
* Worse, because the target is a biased estimate of $v\_\pi$, the unbiasedness condition $\mathbb{E}[U\_t \mid S\_t] = v\_\pi(S\_t)$ fails, so the convergence theorem no longer covers it.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Semi-gradient TD(0))</span></p>

The practical compromise: take the gradient with respect to the prediction **but treat the target as a constant** (do *not* differentiate through $\hat v(S\_{t+1},\cdot)$). This gives the **semi-gradient TD(0)** update

$$
\mathbf{w}_{t+1} \;=\; \mathbf{w}_t + \alpha\,\bigl[\,R_{t+1} + \gamma\,\hat v(S_{t+1},\mathbf{w}_t) - \hat v(S_t,\mathbf{w}_t)\,\bigr]\,\nabla_{\mathbf{w}}\hat v(S_t,\mathbf{w}_t).
$$

It is "semi"-gradient because only **half** the dependence on $\mathbf{w}$ — the prediction half — is followed; the target half is frozen. In exchange for giving up the clean SGD guarantee, it learns **online**, with **lower variance**, and works in **continuing** tasks.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Method</span><span class="math-callout__name">(Gradient MC vs. semi-gradient TD — the big picture)</span></p>

| | **Gradient MC** | **Semi-gradient TD** |
| :--- | :--- | :--- |
| Target $U\_t$ | $G\_t$ (unbiased) | $R\_{t+1} + \gamma\,\hat v(S\_{t+1},\mathbf{w})$ (biased) |
| True gradient? | yes | only "half" |
| Convergence (general) | yes (local optimum) | not guaranteed |
| Convergence (linear, on-policy) | yes ($\overline{\text{VE}}$ optimum) | yes (TD fixed point) |
| Variance | high | lower |
| Online / continuing | no | yes |

The trade is the familiar one: MC is **sound but slow and noisy**; TD is **fast and low-variance but biased**, and keeps its guarantees only in the linear on-policy corner.

</div>

### Linear Function Approximation

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Feature map and linear value function)</span></p>

A **feature** is a function from states to numbers, $x\_i : \mathcal{S} \to \mathbb{R}$. Stack $d$ of them into a **feature vector** $\mathbf{x}(s) \in \mathbb{R}^{d}$, **chosen by us and then fixed**. The linear value function reads the value off as a weighted sum of features:

$$
\hat v(s,\mathbf{w}) \;=\; \mathbf{w}^\top \mathbf{x}(s) \;=\; \sum_{i=1}^{d} w_i\, x_i(s).
$$

Any raw state — a position, an image, a board — is first turned into a numeric vector by the hand-designed map $\mathbf{x}(\cdot)$; only the weights $\mathbf{w}$ are learned.

</div>

<figure class="rl-diagram">
  <svg viewBox="0 0 820 300" role="img" aria-label="Pipeline: a raw state (position, image or board) passes through a fixed hand-designed feature map x(s) to produce a feature vector in R^d, which a linear read-out w transpose x turns into a single value. Only w is learned.">
    <!-- raw state inputs -->
    <text x="95" y="40" text-anchor="middle" font-size="13" class="muted">raw state s</text>
    <rect x="20" y="60" width="150" height="44" rx="8" class="box"></rect>
    <text x="95" y="87" text-anchor="middle" font-size="13">position (x, y)</text>
    <rect x="20" y="128" width="150" height="44" rx="8" class="box"></rect>
    <text x="95" y="155" text-anchor="middle" font-size="13">image / pixels</text>
    <rect x="20" y="196" width="150" height="44" rx="8" class="box"></rect>
    <text x="95" y="223" text-anchor="middle" font-size="13">board layout</text>

    <!-- arrows into feature map -->
    <line x1="170" y1="82"  x2="285" y2="140" class="line" marker-end="url(#fa-arrow)"></line>
    <line x1="170" y1="150" x2="285" y2="150" class="line" marker-end="url(#fa-arrow)"></line>
    <line x1="170" y1="218" x2="285" y2="162" class="line" marker-end="url(#fa-arrow)"></line>

    <!-- feature map block -->
    <rect x="290" y="110" width="130" height="84" rx="10" class="accent"></rect>
    <text x="355" y="145" text-anchor="middle" font-size="14" font-weight="700">feature map</text>
    <text x="355" y="166" text-anchor="middle" font-size="13">x(·)</text>
    <text x="355" y="184" text-anchor="middle" font-size="11" class="muted">fixed, hand-designed</text>

    <!-- feature vector -->
    <line x1="420" y1="152" x2="475" y2="152" class="line" marker-end="url(#fa-arrow)"></line>
    <text x="512" y="70" text-anchor="middle" font-size="12" class="muted">feature vector x(s) ∈ ℝᵈ</text>
    <rect x="486" y="84"  width="52" height="20" class="box"></rect>
    <rect x="486" y="106" width="52" height="20" class="box"></rect>
    <rect x="486" y="128" width="52" height="20" class="accent"></rect>
    <rect x="486" y="150" width="52" height="20" class="accent"></rect>
    <rect x="486" y="172" width="52" height="20" class="box"></rect>
    <rect x="486" y="194" width="52" height="20" class="box"></rect>
    <rect x="486" y="216" width="52" height="20" class="accent"></rect>

    <!-- linear read-out -->
    <line x1="538" y1="152" x2="600" y2="152" class="line" marker-end="url(#fa-arrow)"></line>
    <rect x="604" y="124" width="150" height="56" rx="10" class="green"></rect>
    <text x="679" y="148" text-anchor="middle" font-size="13">value (a number)</text>
    <text x="679" y="169" text-anchor="middle" font-size="13">v̂ = wᵀx(s)</text>
    <text x="679" y="206" text-anchor="middle" font-size="11" class="muted">only w is learned</text>

    <defs>
      <marker id="fa-arrow" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
        <path d="M0,0 L7,3 L0,6 Z" fill="#64748b"></path>
      </marker>
    </defs>
  </svg>
  <figcaption>The state is mapped through a <strong>fixed</strong> feature map $\mathbf{x}(\cdot)$ into a numeric vector; the value is a linear read-out $\mathbf{w}^\top\mathbf{x}(s)$. Learning only adjusts $\mathbf{w}$ — the features are designed in advance and never change.</figcaption>
</figure>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(A tiny worked feature map)</span></p>

Let the state be a 1-D position $s \in [0,1]$. Choose **three** features — soft "bumps" centred at $0.25, 0.5, 0.75$:

$$
x_i(s) \;=\; \exp\!\Bigl(-\tfrac{(s - c_i)^2}{2\sigma^2}\Bigr), \qquad \sigma = 0.15.
$$

At $s = 0.4$ the three features fire by different amounts, so the value is a weighted blend:

$$\mathbf{x}(0.4) \approx (0.61,\, 0.80,\, 0.07)^\top, \qquad \hat v(0.4,\mathbf{w}) = 0.61\,w_1 + 0.80\,w_2 + 0.07\,w_3.$$

**Reading the vector:** each weight $w\_i$ "owns" a region of state space. A state's value blends the weights whose features it activates — so nearby states (firing the *same* features) get *similar* values automatically. That automatic local similarity is precisely the generalization we wanted.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(The gradient is the feature vector)</span></p>

With a fixed feature map, the value is linear in $\mathbf{w}$, so its gradient is just the feature vector:

$$\nabla_{\mathbf{w}}\,\hat v(s,\mathbf{w}) \;=\; \nabla_{\mathbf{w}}\,\mathbf{w}^\top\mathbf{x}(s) \;=\; \mathbf{x}(s).$$

Two clean consequences:

* $\overline{\text{VE}}$ is **convex in $\mathbf{w}$** — a *single* optimum, no local minima.
* The semi-gradient update moves $\mathbf{w}$ **directly along $\mathbf{x}(s)$**.

**Take-home:** almost all convergence guarantees in RL with function approximation are for the *linear* case.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Gradient of linear function approximation)</span></p>

* $\overline{\text{VE}}$ is **convex in $\mathbf{w}$** — a *single* optimum, no local minima.
* The **semi-gradient** update moves $\mathbf{w}$ **directly along $\mathbf{x}(s)$**.

**Take-home:** almost all convergence guarantees in RL with function approximation are for the *linear* case.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Rule</span><span class="math-callout__name">(The linear update rules)</span></p>

Substituting $\nabla\_{\mathbf{w}}\hat v(S\_t,\mathbf{w}) = \mathbf{x}\_t$ with $\mathbf{x}\_t \doteq \mathbf{x}(S\_t)$, the general update collapses to

$$\mathbf{w}_{t+1} \;=\; \mathbf{w}_t + \alpha\,\bigl[\,U_t - \mathbf{w}_t^\top \mathbf{x}_t\,\bigr]\,\mathbf{x}_t .$$

The method is chosen entirely by the target:

* **Gradient Monte Carlo:** $U\_t = G\_t$.
* **Semi-gradient TD(0):** $U\_t = R\_{t+1} + \gamma\,\mathbf{w}\_t^\top \mathbf{x}\_{t+1}$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Linear semi-gradient TD converges to the TD fixed point)</span></p>

Semi-gradient TD uses bootstrapping, so it is **not** ordinary gradient descent on $\overline{\text{VE}}$. Yet in one important case there *is* a guarantee:

$$\textbf{linear function approximation } + \textbf{ on-policy data.}$$

Then semi-gradient TD(0) converges to a **TD fixed point** $\mathbf{w}\_{TD}$, where predictions stop changing **on average**. This is not, in general, the weight vector that minimizes $\overline{\text{VE}}$ — but its error is **bounded relative to the best achievable** with these features:

$$\overline{\text{VE}}(\mathbf{w}_{TD}) \;\leq\; \frac{1}{1-\gamma}\,\min_{\mathbf{w}}\,\overline{\text{VE}}(\mathbf{w}).$$

* The guarantee is special to the **linear, on-policy** setting (Sutton 1988; Tsitsiklis & Van Roy 1997). 
  * With nonlinear approximation or off-policy data, TD can become **unstable**. 
* The factor $\tfrac{1}{1-\gamma}$ can be large for $\gamma$ near $1$, so "bounded" does not mean "tight".

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">($n$-step semi-gradient TD)</span></p>

The $n$-step return target carries over unchanged, now bootstrapping from the *approximate* value:

$$G_{t:t+n} \;=\; R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^{n}\,\hat v(S_{t+n},\mathbf{w}_{t+n-1}),$$

with update

$$\mathbf{w}_{t+n} \;=\; \mathbf{w}_{t+n-1} + \alpha\,\bigl[\,G_{t:t+n} - \hat v(S_t,\mathbf{w}_{t+n-1})\,\bigr]\,\nabla_{\mathbf{w}}\hat v(S_t,\mathbf{w}_{t+n-1}).$$

**Bias / variance dial.** Small $n$ means more bootstrapping ⇒ low variance, high bias; large $n$ approaches Monte Carlo ⇒ low bias, high variance. The same dial governs **both** linear and nonlinear $\hat v$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Bias / Variance: $n$-step semi-gradient TD)</span></p>

* Small $n$ means more bootstrapping ⇒ low variance, high bias; 
* Large $n$ approaches Monte Carlo ⇒ low bias, high variance. The same dial governs **both** linear and nonlinear $\hat v$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Choosing $n$ — the bias–variance trade-off)</span></p>

Sweeping $n$ traces the textbook bias–variance curve: **bias decreases** with $n$, **variance grows** with $n$, and their sum — the total error — is **U-shaped**, minimized at some *intermediate* $n$. The best value is **problem-dependent**: it depends on how long you train and how stable the step size $\alpha$ is. The lesson is the same as in plain $n$-step TD — neither extreme ($n=1$ or $n=\infty$) is usually best.

</div>

<figure>
  <img src="{{ '/assets/images/notes/rl_hd/Bias–VarianceTrade-off_n-step-td.png' | relative_url }}" alt="Soft-max policy probability rising smoothly with the preference gap, versus the epsilon-greedy step function that can only occupy two probability levels and jumps at the decision boundary" loading="lazy">
  <!-- <figcaption>Why "smooth" matters. $\varepsilon$-greedy can only reach two probability levels and <em>jumps</em> at the decision boundary; the soft-max moves $\pi(\text{right} \mid s)$ continuously with $\theta$ — exactly what $\nabla_\theta J$ needs — and it can approach a deterministic optimum.</figcaption> -->
</figure>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Method</span><span class="math-callout__name">(Which on-policy prediction methods converge?)</span></p>

For on-policy prediction with function approximation, the strongest clean guarantees are for **linear** methods:

| Method | Converges? | Where it converges |
| :--- | :--- | :--- |
| Gradient Monte Carlo | Yes | local minimum of $\overline{\text{VE}}$ |
| Linear gradient MC | Yes | **global** minimum of $\overline{\text{VE}}$ |
| Linear semi-gradient TD(0), on-policy | Yes | TD fixed point |
| $n$-step semi-gradient TD, linear, on-policy | Yes | TD-like fixed point |
| Nonlinear semi-gradient TD | No general guarantee | can be unstable |
| Off-policy + bootstrapping + FA | Not in general | can diverge |

The pattern: **linearity buys global optima for gradient methods and bounded-error fixed points for semi-gradient methods**; leave the linear on-policy corner and the guarantees evaporate.

</div>

### Feature Construction for Linear Methods

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Idea</span><span class="math-callout__name">(Features are everything)</span></p>

The set of functions a linear model can represent — its **hypothesis space** — is exactly the span of the features:

$$
\hat v(\cdot,\mathbf{w}) \in \operatorname{span}\lbrace\,x_1(\cdot), \dots, x_d(\cdot)\,\rbrace.
$$

If $v\_\pi$ lies **outside** this span, no $\mathbf{w}$ recovers it exactly. Feature design therefore answers two questions at once:

* **Capacity:** can the basis express the regularities of $v\_\pi$?
* **Generalization:** which states should *share* parameters (and so be forced to have related values)?

Good features encode **domain knowledge** as useful priors: a robot's location suggests spatial features; pole-balancing suggests angle, angular velocity, **and their product**. Because the algorithm is fixed once the features are chosen, *choosing features is often more important than choosing the algorithm.*

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(State aggregation — the simplest linear method)</span></p>

**State aggregation** partitions states into groups and learns one weight per group — the simplest possible feature map. With one-hot indicator features

$$x_i(s) = \mathbf{1}[\,s \in \text{group } i\,], \qquad \hat v(s,\mathbf{w}) = w_{g(s)},$$

where $g(s)$ is the group of $s$. Properties:

* $\nabla\_{\mathbf{w}}\hat v(s,\mathbf{w})$ is a **one-hot** vector;
* updating one group does **not** affect any other — exactly the tabular non-interaction, at coarser granularity;
* within each group $\hat v$ is **constant** — the approximation is a **staircase**.

It is the degenerate, zero-generalization end of the feature-design spectrum: groups share nothing across boundaries.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(1000-state random walk with state aggregation)</span></p>

States $1, \dots, 1000$, episodes start at $500$. From each non-terminal state the agent jumps **uniformly** to one of the $\pm 100$ nearest neighbours; reward is $-1$ on left-termination, $+1$ on right-termination, $0$ otherwise. The true $v\_\pi$ is nearly **linear** in the state index.

**Approximation:** $10$ groups of $100$ states each, one weight per group. Gradient MC with this aggregation learns a **staircase**: within each block of $100$ states the estimate is flat, so it tracks the underlying line only block-by-block. It also reveals a subtle bias — within a block the learned value is pulled toward the values of the states *most visited* under $\mu$ (here, the block's interior near the center), not a plain average. The staircase is the visible price of zero within-group generalization.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Fourier cosine basis)</span></p>

For a 1-D state $s \in [0,1]$ and integer order $i \geq 0$, the **Fourier cosine features** are

$$x_i(s) \;=\; \cos(i\pi s).$$

Here $i = 0$ is the constant feature; higher $i$ gives **higher frequency**. Stacking orders $0, \dots, n$ gives a basis; e.g. for $n = 3$,

$$\mathbf{x}(s) = \bigl(1,\, \cos\pi s,\, \cos 2\pi s,\, \cos 3\pi s\bigr)^\top .$$

Low-frequency terms capture broad structure; higher-frequency terms add finer detail. For a $k$-dimensional state $s = (s\_1, \dots, s\_k) \in [0,1]^k$, choose a frequency vector $\mathbf{c} \in \lbrace 0, \dots, n\rbrace^k$ per feature,

$$x_{\mathbf{c}}(s) \;=\; \cos\bigl(\pi\, \mathbf{c}^\top s\bigr) \;=\; \cos\bigl(\pi(c_1 s_1 + \cdots + c_k s_k)\bigr).$$

A **zero** entry in $\mathbf{c}$ ignores that dimension; **larger** entries oscillate faster; the **ratio** of entries sets the orientation of the wave. A useful per-feature step-size rescaling is $\alpha\_i = \alpha / \lVert \mathbf{c}^{(i)} \rVert\_2$ (and $\alpha\_i = \alpha$ when $\mathbf{c}^{(i)} = \mathbf{0}$), so high-frequency features are not over-stepped.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Coarse coding)</span></p>

**Coarse coding** uses **binary indicators of overlapping regions** (e.g. circles covering the state space):

$$x_i(s) \in \lbrace 0,1\rbrace, \qquad x_i(s) = 1 \iff s \in \text{region } i .$$

The **generalization mechanism** is geometric:

* an update at $s$ moves the weight of every feature **active** at $s$;
* this affects every *other* state where that feature is also active;
* the **strength** of generalization between two states is proportional to the **number of features they share**.

Two states inside many common regions generalize strongly; two states sharing one region generalize weakly; states sharing none do not generalize at all.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Coarse coding on a 1-D interval)</span></p>

Let $s \in [0,1]$ (say, a cartpole position). Cover the interval with overlapping regions

$$[0.0,0.3],\quad [0.2,0.5],\quad [0.4,0.7],\quad [0.6,0.9],\quad [0.8,1.0].$$

Each region is one binary feature. If $s = 0.45$ lies in regions $2$ and $3$, then

$$\mathbf{x}(0.45) = (0, 1, 1, 0, 0)^\top, \qquad \hat v(0.45,\mathbf{w}) = \mathbf{w}^\top\mathbf{x}(0.45) = w_2 + w_3 .$$

**Controlled generalization:** updating at $s = 0.45$ changes only $w\_2$ and $w\_3$. It therefore *also* nudges nearby states that share those active features, such as $s' = 0.35$, but leaves far-away states such as $s'' = 0.95$ untouched.

</div>

<figure class="rl-diagram">
  <svg viewBox="0 0 820 250" role="img" aria-label="Coarse coding on a line: five overlapping intervals act as binary features. The query point at 0.45 falls inside intervals 2 and 3, so its feature vector activates those two features; a nearby point at 0.35 shares one of them, a far point at 0.95 shares none.">
    <!-- axis -->
    <line x1="60" y1="170" x2="760" y2="170" class="line"></line>
    <text x="60"  y="200" text-anchor="middle" font-size="12" class="muted">0.0</text>
    <text x="760" y="200" text-anchor="middle" font-size="12" class="muted">1.0</text>

    <!-- helper: x = 60 + 700*s -->
    <!-- intervals as rounded bars at staggered heights -->
    <rect x="60"  y="60"  width="210" height="16" rx="8" class="box"></rect>
    <text x="165" y="54" text-anchor="middle" font-size="11" class="muted">region 1 [0.0,0.3]</text>
    <rect x="200" y="84"  width="210" height="16" rx="8" class="accent"></rect>
    <text x="305" y="118" text-anchor="middle" font-size="11" class="muted">region 2 [0.2,0.5]</text>
    <rect x="340" y="108" width="210" height="16" rx="8" class="accent"></rect>
    <text x="445" y="78" text-anchor="middle" font-size="11" class="muted">region 3 [0.4,0.7]</text>
    <rect x="480" y="84"  width="210" height="16" rx="8" class="box"></rect>
    <text x="585" y="54" text-anchor="middle" font-size="11" class="muted">region 4 [0.6,0.9]</text>
    <rect x="620" y="60"  width="140" height="16" rx="8" class="box"></rect>
    <text x="700" y="118" text-anchor="middle" font-size="11" class="muted">region 5 [0.8,1.0]</text>

    <!-- query points: s=0.45 -> x=375 ; s=0.35 -> x=305 ; s=0.95 -> x=725 -->
    <line x1="375" y1="50" x2="375" y2="178" stroke="#b91c1c" stroke-width="2" stroke-dasharray="4 3"></line>
    <circle cx="375" cy="170" r="6" fill="#b91c1c"></circle>
    <text x="375" y="222" text-anchor="middle" font-size="12" font-weight="700" fill="#b91c1c">s = 0.45</text>

    <circle cx="305" cy="170" r="5" fill="#047857"></circle>
    <text x="305" y="222" text-anchor="middle" font-size="12" fill="#047857">s' = 0.35</text>

    <circle cx="725" cy="170" r="5" class="muted" fill="#64748b"></circle>
    <text x="725" y="222" text-anchor="middle" font-size="12" class="muted">s'' = 0.95</text>
  </svg>
  <figcaption>Coarse coding: the query $s=0.45$ activates the two highlighted regions, so $\mathbf{x}(0.45)=(0,1,1,0,0)^\top$. The nearby point $s'=0.35$ shares region 2 (weak generalization); the far point $s''=0.95$ shares none (no generalization).</figcaption>
</figure>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Receptive-field width: speed now vs. accuracy later)</span></p>

For a *fixed* set of feature centres, the **width** of each receptive field controls **how broadly** a single update spreads:

* **Narrow** fields ⇒ updates stay **local** (sharp, slow-spreading);
* **Broad** fields ⇒ updates have **smooth, global** influence.

The key subtlety: width sets the **initial generalization speed**, but the **final accuracy** is governed mostly by the **number** of features, not their width. Wide fields learn fast and coarse early; the asymptote depends on how finely the features tile the space.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Tile coding — structured coarse coding)</span></p>

**Tile coding** uses **multiple tilings** (partitions) of the state space. Each tiling cuts the space into $m$ **non-overlapping** tiles; with $n$ tilings there are $m \times n$ feature weights. Key facts:

* a *single* tiling alone is exactly **state aggregation** (non-overlapping groups);
* **multiple offset tilings** create overlapping receptive fields **without overhead**: a query state activates **exactly one tile per tiling**, so exactly $n$ features fire — sparse and binary;
* the features are $O(\sharp\text{tilings})$ active per state, giving a cheap, fast update.

Because the active set is always $n$ tiles, the step size has a simple, interpretable form

$$\alpha \;=\; \frac{1}{\tau\, n},$$

where $n$ is the number of tilings: each update changes the prediction by about a $1/\tau$ fraction of the TD error, so $\tau$ is roughly the **number of samples we want to average over**. Hashing multi-dimensional tile coordinates into a fixed-size weight vector avoids storing every possible tile.

</div>

<figure class="rl-diagram">
  <svg viewBox="0 0 820 330" role="img" aria-label="Tile coding: a continuous 2D state space covered by four offset square-grid tilings drawn in different colors. A single query point falls inside exactly one tile of each tiling, so four tiles overlap at the point and together represent it.">
    <text x="200" y="28" text-anchor="middle" font-size="14">continuous 2-D state space</text>
    <!-- base region -->
    <rect x="60" y="50" width="280" height="240" rx="6" fill="#f8fafc" stroke="#dbe1ee"></rect>

    <!-- tiling 1 (blue) offset 0 -->
    <g stroke="#2c3e94" stroke-width="1.4" fill="none" opacity="0.85">
      <line x1="60"  y1="130" x2="340" y2="130"></line>
      <line x1="60"  y1="210" x2="340" y2="210"></line>
      <line x1="140" y1="50"  x2="140" y2="290"></line>
      <line x1="220" y1="50"  x2="220" y2="290"></line>
    </g>
    <!-- tiling 2 (amber) offset -->
    <g stroke="#b45309" stroke-width="1.4" fill="none" opacity="0.8">
      <line x1="60"  y1="155" x2="340" y2="155"></line>
      <line x1="60"  y1="235" x2="340" y2="235"></line>
      <line x1="165" y1="50"  x2="165" y2="290"></line>
      <line x1="245" y1="50"  x2="245" y2="290"></line>
    </g>
    <!-- tiling 3 (green) offset -->
    <g stroke="#047857" stroke-width="1.4" fill="none" opacity="0.8">
      <line x1="60"  y1="105" x2="340" y2="105"></line>
      <line x1="60"  y1="185" x2="340" y2="185"></line>
      <line x1="115" y1="50"  x2="115" y2="290"></line>
      <line x1="195" y1="50"  x2="195" y2="290"></line>
    </g>

    <!-- query point -->
    <circle cx="185" cy="170" r="6" fill="#b91c1c"></circle>
    <text x="185" y="312" text-anchor="middle" font-size="12" font-weight="700" fill="#b91c1c">query state s</text>

    <!-- right-hand explanation -->
    <text x="585" y="80" text-anchor="middle" font-size="13" font-weight="700">3 offset tilings</text>
    <text x="585" y="110" text-anchor="middle" font-size="13">⇒ exactly 3 active tiles</text>
    <rect x="430" y="140" width="310" height="120" rx="10" class="box"></rect>
    <text x="585" y="168" text-anchor="middle" font-size="13">one tile per tiling fires at s</text>
    <text x="585" y="192" text-anchor="middle" font-size="13">sparse binary features</text>
    <text x="585" y="216" text-anchor="middle" font-size="13">nearby states share many tiles</text>
    <text x="585" y="240" text-anchor="middle" font-size="13">far states share few or none</text>
  </svg>
  <figcaption>Tile coding with offset tilings. The query state activates <strong>exactly one tile per tiling</strong>; offsetting the tilings turns the union of activations into graded generalization — nearby states share many active tiles, distant states share few or none.</figcaption>
</figure>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why tile coding beats a single coarse partition)</span></p>

Fix the memory budget so the comparison is fair: 

$$\sharp\text{weights} = \sharp\text{tilings} \times \sharp\text{tiles per tiling}$$

On the 1000-state random walk:

* **State aggregation:** $1$ tiling $\times\ 1000$ tiles $= 1000$ weights — a coarse staircase.
* **Tile coding:** $50$ tilings $\times\ 20$ tiles $= 1000$ weights — far lower error.

Same memory, very different behaviour. The reason is **graded generalization**: offset tilings make nearby states share many active tiles and far states share few or none, so the value surface is smooth where it should be and still localized. A single tiling can only do one or the other.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Radial basis functions)</span></p>

**RBFs** are the continuous version of coarse coding — Gaussian bumps instead of hard indicators:

$$
x_i(s) \;=\; \exp\!\Bigl(-\frac{\lVert s - c_i \rVert^2}{2\sigma_i^2}\Bigr),
$$

centred at $c\_i$ with width $\sigma\_i$. Advantages and limits:

* feature responses are **smooth and differentiable** (graded, not 0/1);
* but in **high dimensions** they are expensive and suffer edge effects — the very regime where tile coding's hard edges are also hard, so RBFs do not help much there.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Method</span><span class="math-callout__name">(Feature-construction cheat sheet)</span></p>

| Family | Strength | Weakness |
| :--- | :--- | :--- |
| State aggregation | trivial, exact gradient | coarse staircase |
| Polynomials | familiar, simple | global, scales as $(n+1)^k$ |
| Fourier cosine | smooth, strong empirically | ringing at discontinuities |
| Coarse coding | local generalization | must design receptive fields |
| Tile coding | sparse, fast, flexible | axis-aligned bias |
| RBFs | smooth, differentiable | expensive, edge effects |

**Bottom line:** choosing features is often more important than choosing the algorithm — and **tile coding is the classical workhorse**.

</div>

### Nonlinear Function Approximation: Neural Networks

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(Why move beyond linear methods)</span></p>

Linear methods are the cleanest, most analyzable case — but limited:

* the **hypothesis space is fixed** by the features we chose;
* **hand-engineering features** for high-dimensional inputs (images, audio) is impractical;
* real problems demand **representation learning** — the agent should *discover* useful features.

The trade-off of going nonlinear: **power ↑, theory ↓, stability ↓**. Much of modern deep RL is effort spent patching the stability problems this trade creates.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Feedforward neural network and a single unit)</span></p>

A **feedforward network** stacks layers: the **input layer** is the features (or the raw observation), the **hidden layers** are *learned* feature representations, and the **output** is the scalar $\hat v(s,\mathbf{w})$ (or a vector of action values). Each "semi-linear" unit computes a weighted sum followed by a nonlinear activation:

$$z = \mathbf{w}^\top \mathbf{x} + b, \qquad a = \phi(z).$$

Common activations:

* **Sigmoid:** $\phi(z) = 1/(1 + e^{-z})$.
* **tanh:** zero-centred.
* **ReLU:** $\max(0, z)$ — the modern default.

The crucial difference from the linear case is that the hidden layers' features are **learned together with the weights**, not fixed in advance.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Nonlinearity is mandatory; universal approximation)</span></p>

A network of purely *linear* units is itself a linear function.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof Sketch</summary>

if layer $\ell$ computes 

$$\mathbf{a}^{(\ell)} = W^{(\ell)}\mathbf{a}^{(\ell-1)} + \mathbf{b}^{(\ell)}$$

with no nonlinearity, unrolling all $L$ layers gives

$$\mathbf{a}^{(L)} \;=\; \Bigl(\textstyle\prod_{\ell=L}^{1} W^{(\ell)}\Bigr)\mathbf{a}^{(0)} + \tilde{\mathbf{b}},$$

which is again **affine** in $\mathbf{a}^{(0)}$ — stacking adds no expressiveness. The nonlinear activations are what make depth meaningful.

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Universal approximation (Cybenko, 1989))</span></p>

A *single* hidden layer with enough sigmoid units can approximate any continuous function on a compact domain to arbitrary accuracy.

**Catch:** expressiveness is *not* the bottleneck — **optimization** is. Being able to represent $v\_\pi$ in principle says nothing about being able to *find* the weights that do.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Training by SGD via backpropagation)</span></p>

The same generic update rule applies, with the gradient now flowing through many layers:

$$\mathbf{w} \;\leftarrow\; \mathbf{w} + \alpha\,\delta_t\,\nabla_{\mathbf{w}}\hat v(S_t,\mathbf{w}), \qquad \delta_t = U_t - \hat v(S_t,\mathbf{w}).$$

The gradient $\nabla\_{\mathbf{w}}\hat v$ is computed by **backpropagation**, in two passes:

* **Forward pass.** Compute every layer's activations and store them.
* **Backward pass.** Apply the chain rule, layer by layer, to compute $\partial L / \partial w$ for every weight.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Difficulties for deep networks: vanishing / exploding gradients)</span></p>

**Vanishing / Exploding gradients** are mitigated by residual connections, batch normalization, and careful initialization.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The deadly triad)</span></p>

Three properties that, **together**, can make TD-style updates **diverge**:

* **Function approximation** — so updates couple states;
* **Bootstrapping** — so targets depend on $\mathbf{w}$;
* **Off-policy** learning — so the update distribution is "wrong" (we update along a distribution that does not match the policy generating the targets).

---

* Any **two** are usually safe; it is the **combination of all three** that risks divergence. 
* **Caveat**: with deep nets and bootstrapping, even *on-policy* learning can wobble, which is why deep RL leans so heavily on engineering tricks to stay stable.

</div>

<figure class="rl-diagram">
  <svg viewBox="0 0 720 330" role="img" aria-label="The deadly triad shown as three overlapping circles labelled function approximation, bootstrapping, and off-policy learning. Their common intersection in the middle is labelled risk of divergence.">
    <text x="360" y="36" text-anchor="middle" font-size="16" font-weight="700">The Deadly Triad</text>
    <circle cx="290" cy="150" r="105" fill="rgba(44,62,148,0.12)" stroke="#2c3e94" stroke-width="2"></circle>
    <circle cx="430" cy="150" r="105" fill="rgba(180,83,9,0.12)" stroke="#b45309" stroke-width="2"></circle>
    <circle cx="360" cy="245" r="105" fill="rgba(185,28,28,0.10)" stroke="#b91c1c" stroke-width="2"></circle>

    <text x="225" y="120" text-anchor="middle" font-size="13" font-weight="700" fill="#2c3e94">Function</text>
    <text x="225" y="138" text-anchor="middle" font-size="13" font-weight="700" fill="#2c3e94">approximation</text>
    <text x="500" y="120" text-anchor="middle" font-size="13" font-weight="700" fill="#b45309">Bootstrapping</text>
    <text x="360" y="300" text-anchor="middle" font-size="13" font-weight="700" fill="#b91c1c">Off-policy</text>
    <text x="360" y="318" text-anchor="middle" font-size="13" font-weight="700" fill="#b91c1c">training</text>

    <text x="360" y="180" text-anchor="middle" font-size="12" font-weight="700">risk of</text>
    <text x="360" y="197" text-anchor="middle" font-size="12" font-weight="700">divergence</text>
  </svg>
  <figcaption>The deadly triad: function approximation, bootstrapping, and off-policy training. Any two together are usually safe; the central region — all three at once — is where TD updates can diverge.</figcaption>
</figure>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Method</span><span class="math-callout__name">(Six stability tricks that make deep RL work)</span></p>

Modern deep RL stays stable by a toolbox of engineering fixes, each targeting a specific failure of naive SGD on bootstrapped, correlated data:

1. **Target networks.** The bootstrap target 
   
   $$U_t = R_{t+1} + \gamma\max_a \hat v(S_{t+1},\mathbf{w}_t)$$
   
   uses the *same* weights we are updating — a feedback loop that can spiral.
   * **Fix:** keep a frozen copy $\mathbf{w}^- = \theta^-$ used **only** to compute targets, refreshed $\mathbf{w}^- \leftarrow \mathbf{w}$ every $C$ steps (DQN). A momentarily *stationary* target turns a moving-goalpost problem into ordinary supervised regression between refreshes.
2. **Experience replay.** Consecutive transitions are **highly correlated** (the agent moves smoothly), violating the i.i.d. assumption and wasting each rare experience after one use. 
   * **Fix:** store transitions $(S, A, R, S')$ in a **replay buffer** and sample a uniform random minibatch per update — this **decorrelates** samples *and* makes each transition **reusable** (data efficiency).
3. **Batch normalization.** As lower layers learn, the distribution of inputs to higher layers keeps shifting (**internal covariate shift**). 
   * **Fix:** per minibatch, normalize a layer's pre-activations to mean $0$, variance $1$, then rescale by learned $\gamma, \beta$: 
   
   $$\hat z = (z - \mu_B)/\sqrt{\sigma_{B}^2 + \epsilon}$$

   $$\tilde z = \gamma\hat z + \beta$$
   
   Well-scaled signals keep gradients healthy and allow larger steps.
4. **Residual / skip connections.** In very deep stacks gradients shrink as they propagate back (**vanishing gradients**). 
   * **Fix:** add an identity skip around a block, $\mathbf{y} = F(\mathbf{x}) + \mathbf{x}$, so the block only learns the **residual** $F(\mathbf{x})$ and gradients get a short-circuit straight to earlier layers.
5. **Gradient clipping.** A single sharp target error (common with bootstrapping) can produce a huge gradient that throws the weights far off. 
   * **Fix:** cap the gradient norm at a threshold $c$, 
   
   $$\mathbf{g} \leftarrow \mathbf{g}\cdot\min\!\bigl(1, c/\lVert\mathbf{g}\rVert\bigr)$$ 
   
   — direction is kept, only the length is limited.
6. **Dropout.** Units **co-adapt**, so the network overfits and generalizes poorly. 
   * **Fix:** on each training pass randomly "drop" each hidden unit with probability $p$; at test time use the full network with weights scaled to match. Training a different random sub-network each step is like averaging an **ensemble**, forcing each unit to be individually useful.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(Deep convolutional networks)</span></p>

For high-dimensional **spatial** inputs (images, sensor arrays), convolutional networks (LeCun et al. 1998) stack convolutional and subsampling layers, then fully-connected layers. Two structural priors do the heavy lifting:

* **Weight sharing** across spatial locations — the same filter is applied everywhere, drastically cutting parameters;
* **Subsampling** for spatial invariance — small shifts of the input do not change the output much.

The whole stack is trainable **end-to-end** by ordinary backpropagation, so the convolutional features are learned, not designed.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Method</span><span class="math-callout__name">(Linear vs. nonlinear approximation)</span></p>

| | **Linear** | **Neural networks** |
| :--- | :--- | :--- |
| Expressiveness | limited by features | very high |
| Theory | strong | weak |
| Stability | high (on-policy) | fragile |
| Computation | cheap | expensive |
| Feature engineering | needed | largely automatic |

The two columns are the two ends of the **power vs. stability** trade: linear methods give guarantees and demand hand-built features; neural networks learn the features but forfeit the theory.

</div>

### Summary

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(What we learned)</span></p>

1. **Tabular methods do not scale** — we have to approximate.
2. Approximation forces us to choose an **objective**: the $\mu$-weighted mean-squared value error $\overline{\text{VE}}$.
3. **Gradient methods** on $\overline{\text{VE}}$ are well behaved when targets are **unbiased** (Monte Carlo).
4. **Bootstrapping introduces bias** ⇒ **semi-gradient** methods, which give up the clean SGD guarantee for speed and online learning.
5. **Linear + on-policy** ⇒ semi-gradient TD converges to the **TD fixed point** — stable, though not the $\overline{\text{VE}}$ minimizer (within the $\tfrac{1}{1-\gamma}$ bound).
6. **Feature design encodes domain knowledge** — often mattering more than the choice of algorithm; **tile coding** is a workhorse.
7. **Neural networks** lift the feature-design burden but lose the theory, leaning on **stability tricks** instead.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Idea</span><span class="math-callout__name">(Three axes of approximate prediction)</span></p>

Almost every algorithm in modern RL is a particular point in one 3-D design space:

| Axis | Choices | Trade-off |
| :--- | :--- | :--- |
| Function class | linear / NN / memory-based | power vs. stability |
| Update target $U\_t$ | MC / TD($n$) / TD(0) | bias vs. variance |
| Features (linear) | aggregation / tile / Fourier / RBF | locality vs. smoothness |

Choosing a prediction method *is* choosing a coordinate on these three axes. The control methods of the next lectures inherit the very same axes, plus the policy-improvement loop on top.

</div>

## Lecture 10: Policy Gradient Methods

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Setup</span><span class="math-callout__name">(From learned values to learned policies)</span></p>

Every control method so far — from $\varepsilon$-greedy bandits through Sarsa, Q-learning, and their function-approximation versions — follows the same recipe: *learn action values, then act (nearly) greedily on them*. The policy is implicit, a by-product of the value estimates. This lecture flips the arrangement: we parameterize the **policy itself**, $\pi(a \mid s, \boldsymbol{\theta})$, and improve it by **gradient ascent** on a scalar performance measure $J(\boldsymbol{\theta}) = v\_{\pi\_{\boldsymbol{\theta}}}(s\_0)$. We already met the stateless seed of this idea in the **gradient bandits** of Lecture 2; here it grows into a full MDP method. The build-up:

1. **Why parameterize the policy?** Smooth approach to determinism, genuinely stochastic optima, and policies that are simpler than their value functions.
2. **Soft-max in action preferences.** The workhorse parameterization $\pi(a \mid s, \boldsymbol{\theta}) \propto e^{h(s,a,\boldsymbol{\theta})}$.
3. **The short corridor.** A tiny counterexample where *every* $\varepsilon$-greedy action-value method loses to a modest stochastic policy.
4. **The policy gradient theorem.** The central result: a formula for $\nabla J$ that does **not** require differentiating the state distribution.
5. **REINFORCE.** Turning the theorem into a Monte Carlo algorithm, one sampled state–action pair at a time.
6. **Baselines.** Same expected gradient, drastically lower variance.
7. **Actor–critic.** Bootstrapping the return estimate turns the baseline into a true *critic*: fully online, incremental updates driven by the TD error.
8. **Case study: AlphaGo.** The algorithms of this lecture at scale — REINFORCE self-play plus a value-network baseline.

The guiding question of the lecture:

> *How can we follow the gradient of expected return with respect to the policy parameters, even though the state distribution itself depends on the policy?*

The arc continues in the next lecture: a **Gaussian policy** extends the same score-function machinery to **continuous** action spaces.

</div>

### Motivation: Why Parameterize the Policy?

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Parameterized policy)</span></p>

So far we learned $q(s, a)$ and then acted greedily or $\varepsilon$-greedily on it,

$$
\pi(s) = \arg\max_a q(s, a) \qquad \text{or} \qquad \varepsilon\text{-greedy on } q.
$$

Now we parameterize the policy **directly**:

$$
\pi(a \mid s, \boldsymbol{\theta}) \;\doteq\; \Pr\lbrace A_t = a \mid S_t = s,\ \boldsymbol{\theta}_t = \boldsymbol{\theta}\rbrace,
\qquad \boldsymbol{\theta} \in \mathbb{R}^{d'},
$$

where $\pi(\cdot \mid s, \boldsymbol{\theta})$ is a probability distribution over actions for every state, differentiable in $\boldsymbol{\theta}$. Note the division of labour between the two parameter vectors we now carry:

* $\boldsymbol{\theta} \in \mathbb{R}^{d'}$ — the **policy** parameters;
* $\mathbf{w} \in \mathbb{R}^{d}$ — the weights of a value function $\hat v(s, \mathbf{w})$, which *may* still be used to **help learn** $\boldsymbol{\theta}$ but is **not required for action selection**.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(The new objective: gradient ascent on performance)</span></p>

Choose a scalar performance measure $J(\boldsymbol{\theta})$ and follow its gradient:

$$
\boldsymbol{\theta}_{t+1} \;=\; \boldsymbol{\theta}_t + \alpha\, \widehat{\nabla J(\boldsymbol{\theta}_t)},
$$

where $\widehat{\nabla J(\boldsymbol{\theta}\_t)}$ is a *stochastic estimate* whose expectation approximates the true gradient of $J$ at $\boldsymbol{\theta}\_t$. Every method of this form is a **policy-gradient method**, whether or not it also learns a value function. Methods that learn *both* a parameterized policy and a value function are called **actor–critic** methods — the policy is the *actor*, the value function the *critic*.

</div>

<figure class="rl-diagram">
  <svg viewBox="0 0 780 360" role="img" aria-label="Two ways to get a policy: the value-based route learns q and derives an implicit policy by maximization; the policy-based route learns the policy directly and acts by sampling from it, with an optional critic that helps learn the parameters but is not needed to act.">
    <text x="190" y="38" text-anchor="middle" font-size="16" font-weight="700">Value-based</text>
    <text x="590" y="38" text-anchor="middle" font-size="16" font-weight="700" fill="#2c3e94">Policy-based</text>

    <rect x="80" y="60" width="220" height="52" rx="8" class="box"></rect>
    <text x="190" y="91" text-anchor="middle" font-size="14" font-weight="700">learn  q(s, a)</text>

    <rect x="80" y="152" width="220" height="52" rx="8" class="box"></rect>
    <text x="190" y="174" text-anchor="middle" font-size="13">derive policy:</text>
    <text x="190" y="193" text-anchor="middle" font-size="13">arg maxₐ q(s, a)</text>

    <rect x="80" y="244" width="220" height="52" rx="8" class="box"></rect>
    <text x="190" y="266" text-anchor="middle" font-size="13">act</text>
    <text x="190" y="285" text-anchor="middle" font-size="12" class="muted">(deterministic / ε-greedy)</text>

    <line x1="190" y1="112" x2="190" y2="148" class="line" marker-end="url(#pg-flow-arrow)"></line>
    <line x1="190" y1="204" x2="190" y2="240" class="line" marker-end="url(#pg-flow-arrow)"></line>

    <line x1="390" y1="60" x2="390" y2="150" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="6 6"></line>
    <line x1="390" y1="215" x2="390" y2="330" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="6 6"></line>
    <text x="390" y="176" text-anchor="middle" font-size="12" font-style="italic" class="muted">policy is</text>
    <text x="390" y="194" text-anchor="middle" font-size="12" font-style="italic" class="muted">implicit</text>

    <rect x="480" y="60" width="220" height="52" rx="8" class="accent"></rect>
    <text x="590" y="82" text-anchor="middle" font-size="14" font-weight="700" fill="#2c3e94">learn  π(a | s, θ)</text>
    <text x="590" y="101" text-anchor="middle" font-size="14" font-weight="700" fill="#2c3e94">directly</text>

    <rect x="480" y="152" width="220" height="52" rx="8" class="box"></rect>
    <text x="590" y="174" text-anchor="middle" font-size="13">act</text>
    <text x="590" y="193" text-anchor="middle" font-size="12" class="muted">(sample from π)</text>

    <rect x="480" y="256" width="220" height="72" rx="8" class="green"></rect>
    <text x="590" y="278" text-anchor="middle" font-size="12" fill="#047857">optional critic  v̂(s, w)</text>
    <text x="590" y="296" text-anchor="middle" font-size="12" fill="#047857">helps learn θ;</text>
    <text x="590" y="314" text-anchor="middle" font-size="12" fill="#047857">not needed to act</text>

    <line x1="590" y1="112" x2="590" y2="148" class="line" marker-end="url(#pg-flow-arrow)"></line>
    <line x1="590" y1="252" x2="590" y2="212" stroke="#047857" stroke-width="2" stroke-dasharray="5 5" marker-end="url(#pg-critic-arrow)"></line>

    <defs>
      <marker id="pg-flow-arrow" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
        <path d="M0,0 L7,3 L0,6 Z" fill="#64748b"></path>
      </marker>
      <marker id="pg-critic-arrow" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
        <path d="M0,0 L7,3 L0,6 Z" fill="#047857"></path>
      </marker>
    </defs>
  </svg>
  <figcaption>Two ways to get a policy. On the value-based route the policy is implicit — it is read off the value estimates by (near-)maximization. On the policy-based route the policy is the learned object itself; a value function is at most an optional helper for the update, never needed to select actions.</figcaption>
</figure>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Three reasons to parameterize the policy)</span></p>

1. **Approach determinism smoothly.** $\varepsilon$-greedy *always* selects a random action with probability $\varepsilon$; a parameterized policy can drive the optimal-action probabilities to $1$ continuously, without a hand-tuned exploration schedule.
2. **Represent stochastic optima.** Under function approximation or partial observability the best policy may be *genuinely* stochastic — think of bluffing in poker: any deterministic bluffing rule is exploitable.
3. **Sometimes the policy is simpler than the value function.** Action preferences can be far easier to represent than a full $q\_\ast$; the policy only has to rank actions, not predict returns accurately.

**Bonus:** policy parameterization handles *continuous* action spaces naturally — a theme we return to in the next lecture.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Caveat</span><span class="math-callout__name">(The challenge)</span></p>

Performance depends on **both** the action probabilities **and** the resulting state distribution. The latter is a function of the unknown environment: changing $\boldsymbol{\theta}$ changes where the policy goes, and we cannot differentiate through dynamics we do not know. Computing $\nabla J$ therefore *looks like* it may require a model of the environment. Resolving this tension is exactly the job of the policy gradient theorem below.

</div>

### Policy Parameterization: Soft-max in Action Preferences

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Action preferences and the soft-max policy)</span></p>

For a discrete action space, define numerical **action preferences** $h(s, a, \boldsymbol{\theta}) \in \mathbb{R}$ and set

$$\pi(a \mid s, \boldsymbol{\theta}) \;\doteq\; \frac{e^{h(s,a,\boldsymbol{\theta})}}{\sum_b e^{h(s,b,\boldsymbol{\theta})}}.$$

The denominator is a normalizer: for any fixed $s$, $\pi(\cdot \mid s, \boldsymbol{\theta})$ is a probability distribution over actions.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Action preferences and the soft-max policy)</span></p>

Two standard choices for the preferences:

* **Linear in features:**

  $$h(s, a, \boldsymbol{\theta}) = \boldsymbol{\theta}^\top \mathbf{x}(s, a), \qquad \mathbf{x}(s, a) \in \mathbb{R}^{d'},$$

  with state–action feature vectors $\mathbf{x}(s,a)$ built by any of the constructions from Lecture 9.
* **Nonlinear:** $h(s, a, \boldsymbol{\theta})$ represented by a neural network with weights $\boldsymbol{\theta}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why preferences, not action values?)</span></p>

* Preferences are **relative**: adding the same constant to every preference in a state changes nothing — only differences matter. 
* This is precisely the freedom action values lack: a soft-max over *values* would tie the action probabilities to the *scale* of the returns, so approaching determinism would require a hand-tuned temperature schedule. 
* With preferences, the optimal action's preference can simply **grow relative to the others**, so $\pi(a^\ast \mid s) \to 1$ on its own. 
* This is the same shift-invariance we proved for the gradient-bandit soft-max in Lecture 2 — $h(s, a, \boldsymbol{\theta})$ generalizes the bandit preferences $H\_t(a)$ from a table over actions to a parameterized function of state *and* action.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(Soft-max vs. $\varepsilon$-greedy)</span></p>

Side by side:

* **$\varepsilon$-greedy on $q$-values:** always at least $\varepsilon$ random; tiny changes in $q$ can flip the greedy action *discontinuously*; cannot represent arbitrary non-trivial stochastic policies.
* **Soft-max in preferences:** probabilities change *smoothly* with $\boldsymbol{\theta}$; can approach a deterministic optimum; can also remain stochastic if that is what is optimal.

The smoothness is not cosmetic: to follow $\nabla\_{\boldsymbol{\theta}} J(\boldsymbol{\theta})$ we need a **continuous, differentiable** dependence of $\pi$ on $\boldsymbol{\theta}$. A discontinuous policy class has no usable gradient.

</div>

<figure>
  <img src="{{ '/assets/images/notes/rl_hd/pg_softmax_vs_epsgreedy.png' | relative_url }}" alt="Soft-max policy probability rising smoothly with the preference gap, versus the epsilon-greedy step function that can only occupy two probability levels and jumps at the decision boundary" loading="lazy">
  <figcaption>Why "smooth" matters. $\varepsilon$-greedy can only reach two probability levels and <em>jumps</em> at the decision boundary; the soft-max moves $\pi(\text{right} \mid s)$ continuously with $\theta$ — exactly what $\nabla_\theta J$ needs — and it can approach a deterministic optimum.</figcaption>
</figure>

### A Motivating Example: the Short Corridor

<figure class="rl-diagram">
  <svg viewBox="0 0 760 250" role="img" aria-label="Short corridor with switched actions: three nonterminal states in a row followed by a goal. Reward is minus one per step. In the middle state the actions are reversed, and all three states share the same features so they look identical to the agent.">
    <text x="140" y="48" text-anchor="middle" font-size="12" class="muted">reward −1 per step</text>

    <circle cx="140" cy="120" r="34" class="accent"></circle>
    <text x="140" y="126" text-anchor="middle" font-size="15" font-weight="700">S₀</text>
    <text x="140" y="180" text-anchor="middle" font-size="12" class="muted">start</text>

    <circle cx="310" cy="120" r="34" class="amber"></circle>
    <text x="310" y="126" text-anchor="middle" font-size="15" font-weight="700">S₁</text>

    <circle cx="480" cy="120" r="34" class="box"></circle>
    <text x="480" y="126" text-anchor="middle" font-size="15" font-weight="700">S₂</text>

    <rect x="580" y="86" width="110" height="68" rx="10" class="green"></rect>
    <text x="635" y="126" text-anchor="middle" font-size="15" font-weight="700" fill="#047857">goal</text>

    <line x1="178" y1="108" x2="272" y2="108" class="line" marker-end="url(#pg-corridor-arrow)"></line>
    <text x="225" y="98" text-anchor="middle" font-size="11" class="muted">right</text>
    <line x1="348" y1="108" x2="442" y2="108" class="line" marker-end="url(#pg-corridor-arrow)"></line>
    <line x1="518" y1="108" x2="576" y2="108" class="line" marker-end="url(#pg-corridor-arrow)"></line>
    <text x="547" y="98" text-anchor="middle" font-size="11" class="muted">right</text>

    <path d="M285,158 C260,190 215,190 190,158" stroke="#b45309" stroke-width="2.5" fill="none" marker-end="url(#pg-corridor-arrow-amber)"></path>
    <text x="310" y="196" text-anchor="middle" font-size="12" font-weight="700" fill="#b45309">right → moves left!</text>
    <text x="310" y="214" text-anchor="middle" font-size="12" fill="#b45309">actions reversed here</text>

    <text x="380" y="240" text-anchor="middle" font-size="12" class="muted">all three nonterminal states share the same features — to the agent they look identical</text>

    <defs>
      <marker id="pg-corridor-arrow" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
        <path d="M0,0 L7,3 L0,6 Z" fill="#64748b"></path>
      </marker>
      <marker id="pg-corridor-arrow-amber" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
        <path d="M0,0 L7,3 L0,6 Z" fill="#b45309"></path>
      </marker>
    </defs>
  </svg>
  <figcaption>The short corridor with switched actions: three nonterminal states, actions <code>left</code>/<code>right</code>, reward $-1$ per step, undiscounted episodic ($\gamma = 1$). In the middle state the actions are reversed.</figcaption>
</figure>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Short corridor with switched actions)</span></p>

The corridor has three nonterminal states and a goal; the reward is $-1$ per step, so maximizing return means reaching the goal fast. Two twists make it interesting:

* In the **middle** state the actions are **reversed**: `right` moves left and `left` moves right.
* All nonterminal states **share the same action features**,

  $$\mathbf{x}(s, \text{right}) = [1, 0]^\top, \qquad \mathbf{x}(s, \text{left}) = [0, 1]^\top,$$

  so under this representation the states are *indistinguishable* and every policy must behave identically in all of them.

**Why action-value methods lose here.** Since the states look identical, a deterministic policy must treat them the same way — and always-right loops forever in the middle state, while always-left never leaves the start. An $\varepsilon$-greedy action-value method can only realize

$$\pi(\text{right}) \in \lbrace \varepsilon/2,\ 1 - \varepsilon/2 \rbrace ,$$

and for $\varepsilon = 0.1$ the better of the two choices earns expected return about $-44$. A **stochastic** policy, by contrast, can learn *any* $p = \pi(\text{right}) \in (0,1)$; the closed-form performance is

$$J(p) \;=\; -\frac{2(2 - p)}{p(1 - p)},$$

maximized at 

$$p^\ast = 2 - \sqrt{2} \approx 0.59$$

with 

$$J(p^\ast) \approx -11.6$$

— roughly *four times better* than anything $\varepsilon$-greedy can reach.

</div>

<figure>
  <img src="{{ '/assets/images/notes/rl_hd/pg_short_corridor_objective.png' | relative_url }}" alt="Expected return on the short corridor as a function of the probability of choosing right, a smooth curve peaking near 0.59, with the two probability levels reachable by epsilon-greedy marked as dashed vertical lines far from the peak" loading="lazy">
  <figcaption>The short corridor objective $J(p)$ as a function of $p = \pi(\text{right} \mid s)$. The best stochastic policy sits at $p^* = 2 - \sqrt{2} \approx 0.59$; $\varepsilon$-greedy ($\varepsilon = 0.1$) is confined to the two dashed lines, earning about $-82$ or $-44$.</figcaption>
</figure>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Stochasticity is not just exploration noise)</span></p>

Until now, randomness in the policy was a *means* — exploration bolted onto an essentially deterministic target policy. The short corridor shows it can be the *end*: under function approximation (aliased states) the **best representable policy is itself stochastic**, and the optimal action probabilities are specific numbers, not "greedy plus noise". Only a method that can *search the space of stochastic policies* — moving action probabilities smoothly — can find $p^\ast$. That is precisely what policy-gradient methods do.

</div>

### The Performance Objective

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Start-state performance objective, episodic)</span></p>

Assume every episode starts in a fixed state $s\_0$. Define the performance of $\boldsymbol{\theta}$ as the value of the start state under the policy it induces:

$$J(\boldsymbol{\theta}) \;\doteq\; v_{\pi_{\boldsymbol{\theta}}}(s_0)\;=\; \mathbb{E}_{\pi_{\boldsymbol{\theta}}}\!\left[\, \sum_{t=0}^{T-1} \gamma^{t} R_{t+1} \,\middle|\, S_0 = s_0 \right].$$

For undiscounted episodic examples (like the short corridor) set $\gamma = 1$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Does not MC from the previous lectures estimate the same thing?</summary>

Yes. **For a fixed policy and fixed start state, Monte Carlo prediction can estimate exactly the same scalar quantity.**

Lecture 10 defines

$$J(\boldsymbol\theta) = v_{\pi_{\boldsymbol\theta}}(s_0) =  \mathbb E_{\pi_{\boldsymbol\theta}} \left[G_0\mid S_0=s_0\right].$$

Thus, if we generate $N$ episodes from $s_0$ under the fixed policy $\pi_{\boldsymbol\theta}$, then

$$\widehat J_N(\boldsymbol\theta) = \frac1N\sum_{i=1}^N G_0^{(i)}$$

is the ordinary Monte Carlo estimate of $v_{\pi_{\boldsymbol\theta}}(s_0)$. 

**The difference is the role assigned to this quantity.**

Earlier MC prediction treats $\pi$ as fixed and asks:

$$\text{What is }v_\pi(s)?$$

It may estimate $v_\pi(s)$ for every visited state $s$, using complete returns. 

Policy-gradient methods treat the policy as parameterized and ask:

$$\text{How should }\boldsymbol\theta\text{ change to increase } J(\boldsymbol\theta)?$$

So $J$ is not a new kind of return. It is the familiar value function, restricted to the start state and viewed as an **objective function over policies**:

$$\boxed{J(\boldsymbol\theta) = v_{\pi_{\boldsymbol\theta}}(s_0).}$$

The conceptual progression is:

$$
\begin{array}{c|c}
\text{MC prediction} &
\text{estimate }J(\boldsymbol\theta)\text{ for fixed }\boldsymbol\theta \\
\text{Policy gradient} &
\text{estimate }\nabla_{\boldsymbol\theta}J(\boldsymbol\theta)
\text{ and change }\boldsymbol\theta
\end{array}
$$

REINFORCE still uses Monte Carlo returns, but now $G_t$ supplies a sample of

$$q_\pi(S_t,A_t) = \mathbb E_\pi[G_t\mid S_t,A_t],$$

which is inserted into the policy-gradient update. 

So yes: **MC already knows how to evaluate the objective. Lecture 10 introduces how to differentiate and optimize that objective with respect to the policy itself.**

</details>
</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Problem</span><span class="math-callout__name">(The technical obstacle)</span></p>

$J(\boldsymbol{\theta})$ depends on the *state distribution* induced by $\pi\_{\boldsymbol{\theta}}$, which in turn depends on the unknown dynamics $p(s', r \mid s, a)$. Differentiating $J$ naively would require knowing how a change in $\boldsymbol{\theta}$ shifts the whole distribution of visited states — a quantity we can neither compute nor sample without a model. What we need is a gradient formula that **does not require differentiating the state distribution**.

</div>

### The Policy Gradient Theorem

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Policy gradient theorem — episodic case)</span></p>

For any differentiable policy $\pi(a \mid s, \boldsymbol{\theta})$,

$$\nabla J(\boldsymbol{\theta}) \;\propto\; \sum_s \mu(s) \sum_a q_\pi(s, a)\, \nabla \pi(a \mid s, \boldsymbol{\theta}),$$

where $\mu$ is the **on-policy state distribution** under $\pi\_{\boldsymbol{\theta}}$ — the same visitation weighting that appeared in the $\overline{\text{VE}}$ objective of Lecture 9.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Policy gradient theorem — episodic case: Expactation form)</span></p>

Using

$$\nabla \pi(a \mid s, \boldsymbol{\theta}) = \pi(a \mid s, \boldsymbol{\theta})\, \nabla \log \pi(a \mid s, \boldsymbol{\theta}),$$

the same statement reads as an expectation:

$$\nabla J(\boldsymbol{\theta}) \;\propto\; \mathbb{E}_{S \sim \mu,\; A \sim \pi(\cdot \mid S, \boldsymbol{\theta})}\bigl[\, q_\pi(S, A)\, \nabla \log \pi(A \mid S, \boldsymbol{\theta}) \,\bigr].$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Reading the theorem as a procedure)</span></p>

The expectation form is a recipe with three steps:

1. **Sample a state** $S \sim \mu$ — i.e. from *where the policy actually goes* (states the policy visits often carry more weight);
2. **Sample an action** $A \sim \pi(\cdot \mid S, \boldsymbol{\theta})$ — *what the policy actually does there*;
3. **Adjust the log-probability**: push $\log \pi(A \mid S, \boldsymbol{\theta})$ *up* in proportion to $q\_\pi(S, A)$ — strongly up for high-value actions, down for negative-value ones.

**Why this matters.** The right-hand side involves only $\nabla \pi$ — how changing $\boldsymbol{\theta}$ changes the **action probabilities**. The effect of $\boldsymbol{\theta}$ on the future **state distribution** $\mu(s)$ appears nowhere: the theorem has already accounted for it. This is what makes model-free policy-gradient learning possible.

</div>

<figure>
  <img src="{{ '/assets/images/notes/rl_hd/PolicyGradientTheoremReading.png' | relative_url }}" alt="Learning curves of REINFORCE on the short corridor for three step sizes: the middle step size converges near the optimum, the smallest converges more slowly, and the largest plateaus well below the optimum" loading="lazy">
  <!-- <figcaption>REINFORCE on the short corridor ($\gamma = 1$, $\theta_0$ chosen so that $\pi(\text{right}) \approx 0.05$; episodes capped at 500 steps; smoothed average of 200 runs). $\alpha = 2^{-13}$ converges toward $v_*(s_0)$; $\alpha = 2^{-14}$ converges more slowly; $\alpha = 2^{-12}$ is too aggressive — in a fraction of runs the noisy updates temporarily collapse $\pi(\text{right})$ toward $0$, and the average pays for it.</figcaption> -->
</figure>

#### Proof of the Policy Gradient Theorem

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(Proof setup)</span></p>

We suppress the dependence of $\pi$ on $\boldsymbol{\theta}$, and all gradients are with respect to $\boldsymbol{\theta}$. Recall from Lecture 3:

$$v_\pi(s) = \sum_a \pi(a \mid s)\, q_\pi(s, a), \qquad q_\pi(s, a) = \sum_{s', r} p(s', r \mid s, a)\bigl[ r + \gamma\, v_\pi(s') \bigr].$$

We want $\nabla J(\boldsymbol{\theta}) = \nabla v\_\pi(s\_0)$, so we expand $\nabla v\_\pi(s)$ for an arbitrary state $s$. The plan:

1. apply the **product rule** to $v\_\pi(s)$;
2. use the **Bellman equation** for $q\_\pi$ to expand $\nabla q\_\pi$;
3. **unroll** the resulting recursion — $\nabla v\_\pi$ at one state is expressed through $\nabla v\_\pi$ at successor states.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Derivation</span><span class="math-callout__name">(Step 1/5 — apply the product rule)</span></p>

Differentiate 

$$v_\pi(s) = \sum_a \pi(a \mid s)\, q_\pi(s,a)$$

using the product rule:

$$
\nabla v_\pi(s)
= \nabla\!\left[ \sum_a \pi(a \mid s)\, q_\pi(s, a) \right]
= \sum_a \Bigl[ \underbrace{\nabla \pi(a \mid s)\, q_\pi(s, a)}_{\text{action-selection term}}
\;+\; \underbrace{\pi(a \mid s)\, \nabla q_\pi(s, a)}_{\text{future-value term}} \Bigr].
$$

The first term involves the policy gradient *at* $s$ — this is the part we want. The second term still hides $\nabla v\_\pi$ at successor states inside $\nabla q\_\pi$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Derivation</span><span class="math-callout__name">(Step 2/5 — differentiate $q_\pi$ via Bellman)</span></p>

In the Bellman equation for $q\_\pi$, the dynamics $p(s', r \mid s, a)$ and the reward value $r$ do **not** depend on $\boldsymbol{\theta}$ — only $v\_\pi(s')$ does:

$$
\begin{aligned}
\nabla q_\pi(s, a)
&= \sum_{s', r} p(s', r \mid s, a)\, \nabla \bigl[ r + \gamma\, v_\pi(s') \bigr] \\
&= \sum_{s', r} p(s', r \mid s, a)\, \gamma\, \nabla v_\pi(s') \\
&= \gamma \sum_{s'} p(s' \mid s, a)\, \nabla v_\pi(s').
\end{aligned}
$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Derivation</span><span class="math-callout__name">(Step 3/5 — a one-step recursion in $\nabla v_\pi$)</span></p>

Substitute Step 2 into Step 1:

$$\nabla v_\pi(s)= \underbrace{\sum_a \nabla \pi(a \mid s)\, q_\pi(s, a)}_{=:\ \varphi(s)}\;+\; \gamma \sum_a \pi(a \mid s) \sum_{s'} p(s' \mid s, a)\, \nabla v_\pi(s').$$

Define the **local term** and the **transition kernel under $\pi$**:

$$\varphi(s) \doteq \sum_a \nabla \pi(a \mid s)\, q_\pi(s, a), \qquad P_\pi(s' \mid s) \doteq \sum_a \pi(a \mid s)\, p(s' \mid s, a).$$

The recursion becomes

$$\nabla v_\pi(s) \;=\; \varphi(s) + \gamma \sum_{s'} P_\pi(s' \mid s)\, \nabla v_\pi(s').$$

**Same shape as Bellman:** this is a Bellman-style recursion for the *gradient* of the value function, with $\varphi(s)$ playing the role of the immediate reward.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Derivation</span><span class="math-callout__name">(Step 4/5 — unroll the recursion)</span></p>

Apply the same identity to the future state $s'$, substitute back, and repeat. After one substitution,

$$\nabla v_\pi(s) = \varphi(s) + \gamma \sum_{s'} P_\pi(s' \mid s)\, \varphi(s') + \gamma^2 \sum_{s''} \underbrace{\sum_{s'} P_\pi(s' \mid s) P_\pi(s'' \mid s')}_{=\ \Pr(s \to s'',\, 2,\, \pi)} \nabla v_\pi(s''),$$

and after $K$ unrollings,

$$\nabla v_\pi(s) = \sum_{k=0}^{K-1} \gamma^k \sum_x \Pr(s \to x,\, k,\, \pi)\, \varphi(x) \;+\; \gamma^K \sum_x \Pr(s \to x,\, K,\, \pi)\, \nabla v_\pi(x),$$

where $\Pr(s \to x, k, \pi)$ is the probability of being in state $x$ after $k$ steps when starting from $s$ and following $\pi$. The accumulated sum collects the local policy-gradient contributions $\varphi(x)$ along the way; the leftover recursion term vanishes as $K \to \infty$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Derivation</span><span class="math-callout__name">(Step 5/5 — from $\nabla v_\pi(s_0)$ to $\mu$)</span></p>

Letting $K \to \infty$,

$$\nabla v_\pi(s) = \sum_{k=0}^{\infty} \gamma^k \sum_x \Pr(s \to x,\, k,\, \pi)\, \varphi(x) = \sum_x \left( \sum_{k=0}^{\infty} \gamma^k \Pr(s \to x,\, k,\, \pi) \right) \varphi(x).$$

Set $s = s\_0$ and define the **discounted expected visitation count**

$$\eta(x) \;\doteq\; \sum_{k=0}^{\infty} \gamma^k \Pr(s_0 \to x,\, k,\, \pi)$$

("add up the probability of being in state $x$ at every future time step, with later time steps discounted"). Substituting $\varphi(x) = \sum\_a \nabla \pi(a \mid x)\, q\_\pi(x, a)$:

$$\nabla J(\boldsymbol{\theta}) = \nabla v_\pi(s_0)= \sum_x \eta(x) \sum_a \nabla \pi(a \mid x)\, q_\pi(x, a).$$

**Normalize.** Define $C \doteq \sum\_{x'} \eta(x')$ and $\mu(x) = \eta(x)/C$. Then

$$\nabla J(\boldsymbol{\theta})= C \sum_x \mu(x) \sum_a q_\pi(x, a)\, \nabla \pi(a \mid x)\;\propto\; \sum_x \mu(x) \sum_a q_\pi(x, a)\, \nabla \pi(a \mid x). \qquad \blacksquare$$

$C$ is the total discounted visitation mass; if $\gamma = 1$ in an episodic task, $C$ is the expected episode length. The constant is absorbed into the step size $\alpha$ during gradient ascent, which is why the theorem is stated with $\propto$ rather than $=$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Local Policy Sensitivity and Its Bellman Propagation)</span></p>

Consider a finite discounted episodic MDP with fixed start-state distribution $d_0^\top$ and discount $\gamma \in [0,1)$. Let $P_\pi$ denote the transition matrix induced by the differentiable policy $\pi_\theta$, and let $v_\pi$ be its value vector. For a state $s$, define

$$\phi(s) := \sum_a \nabla_\theta \pi(a \mid s,\theta),q_\pi(s,a).$$

All gradients below are with respect to $\theta$.

---

Starting from

$$v_\pi(s)=\sum_a \pi(a\mid s)q_\pi(s,a),$$

show that

$$\nabla v_\pi=\phi+\gamma P_\pi\nabla v_\pi.$$

You may use that the environment transition kernel is independent of $\theta$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

Starting from

$$v_\pi(s)=\sum_a\pi(a\mid s)q_\pi(s,a),$$

the product rule gives

$$\nabla v_\pi(s) = \sum_a \nabla\pi(a\mid s)q_\pi(s,a) + \sum_a \pi(a\mid s)\nabla q_\pi(s,a).$$

The first term is precisely (\phi(s)).

For the second term, use the Bellman equation

$$q_\pi(s,a) = \sum_{s',r} p(s',r\mid s,a) \bigl(r+\gamma v_\pi(s')\bigr).$$

Because the transition and reward kernel is independent of $\theta$,

$$\nabla q_\pi(s,a) = \gamma\sum_{s'}p(s'\mid s,a)\nabla v_\pi(s').$$

Consequently,

$$
\begin{aligned}
\nabla v_\pi(s)
&= \phi(s) + \gamma \sum_a\pi(a\mid s) \sum_{s'}p(s'\mid s,a)\nabla v_\pi(s') \\
&= \phi(s) + \gamma\sum_{s'}P_\pi(s,s')\nabla v_\pi(s').
\end{aligned}
$$

Stacking these identities over all states gives

$$\boxed{\nabla v_\pi=\phi+\gamma P_\pi\nabla v_\pi. }$$

This is exactly the Bellman-style recursion developed in the lecture notes. 

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Resolvent Form of the Policy Gradient: Summing Future Sensitivities via the Bellman Resolvent)</span></p>

Use $\gamma<1$ to solve the recursion and prove

$$\nabla J(\theta) = d_0^\top (I-\gamma P_\pi)^{-1}\phi, \qquad J(\theta)=d_0^\top v_\pi.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

**The recursion is solved by the discounted resolvent of $P_\pi$.**

Rearranging,

$$(I-\gamma P_\pi)\nabla v_\pi=\phi.$$

Since $P_\pi$ is stochastic or substochastic,

$$\rho(P_\pi)\leq 1.$$

Because $0\leq\gamma<1$,

$$\rho(\gamma P_\pi)<1,$$

so $I-\gamma P_\pi$ is invertible. Therefore

$$\nabla v_\pi=(I-\gamma P_\pi)^{-1}\phi.$$

Since $d_0$ is fixed and

$$J(\theta)=d_0^\top v_\pi,$$

we have

$$\nabla J(\theta) = d_0^\top\nabla v_\pi.$$

Hence

$$\boxed{\nabla J(\theta) = d_0^\top(I-\gamma P_\pi)^{-1}\phi.}$$

</details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Is the suggested form of $J(\theta)$ suggested in the exercise actually correct? Does not it suggest the expectation of the state-value function over all state instead of the being conditioned like in the definition on a fixed state?</summary>

You are right to distinguish them. **As written, Exercise 2 uses a slightly more general objective than the lecture definition.**

The lecture defines

$$J(\theta)=v_{\pi_\theta}(s_0) = \mathbb E_{\pi_\theta}[G_0\mid S_0=s_0],$$

where $s_0$ is deterministic.

Exercise 2 instead assumes a fixed initial-state distribution $d_0$ and writes

$$J(\theta)=d_0^\top v_\pi = \sum_s d_0(s)v_\pi(s) = \mathbb E_{S_0\sim d_0,\pi_\theta}[G_0].$$

So these are not literally the same objective for arbitrary $d_0$.

**The lecture objective is recovered by choosing a point mass.**

Set

$$d_0=\delta_{s_0},$$

or, in vector notation, $d_0=e_{s_0}$. Then

$$d_0^\top v_\pi = e_{s_0}^\top v_\pi = v_\pi(s_0),$$

and hence

$$J(\theta) = \mathbb E_{\pi_\theta}[G_0\mid S_0=s_0].$$

Therefore:

$$\boxed{\text{Exercise 2 is correct, but it generalizes the lecture’s fixed-start-state objective.}}$$

It is not “the expectation over all states” in the sense of averaging uniformly over the state space. It averages only over the possible **initial states**, weighted by $d_0$. Nevertheless, unless $d_0=\delta_{s_0}$, it is a genuinely different performance objective from the one stated in the lecture.

The sheet does explicitly say “fixed start-state distribution $d_0^\top$,” rather than “fixed start state $s_0$.”  So this appears intentional, although it would have been clearer to state that the lecture formulation is the special case $d_0=\delta_{s_0}$.


</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Discounted State-Visitation Measure)</span></p>

Define the row vector

$$\eta^\top:=d_0^\top(I-\gamma P_\pi)^{-1}.$$

Prove that

$$\eta^\top = \sum_{t=0}^{\infty}\gamma^t d_0^\top P_\pi^t,$$

and interpret $\eta(s)$ as a discounted expected visitation count.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

**The inverse is the Neumann series.**

Because $\rho(\gamma P_\pi)<1$,

$$(I-\gamma P_\pi)^{-1} = \sum_{t=0}^\infty(\gamma P_\pi)^t = \sum_{t=0}^\infty\gamma^tP_\pi^t.$$

Thus

$$\boxed{ \eta^\top = d_0^\top(I-\gamma P_\pi)^{-1} = \sum_{t=0}^\infty \gamma^t d_0^\top P_\pi^t.}$$

Now

$$\bigl(d_0^\top P_\pi^t\bigr)(s) = \Pr_\pi(S_t=s),$$

where $S_0\sim d_0$. Therefore

$$\eta(s) = \sum_{t=0}^\infty\gamma^t\Pr_\pi(S_t=s).$$

Equivalently,

$$\boxed{ \eta(s) = \mathbb E_\pi\left[ \sum_{t=0}^\infty \gamma^t\mathbf 1_{{S_t=s}} \right]. }$$

Thus $\eta(s)$ is the expected number of visits to $s$, with a visit at time $t$ weighted by $\gamma^t$. It is an unnormalized discounted visitation measure.

</details>
</div>


### REINFORCE: Monte-Carlo Policy Gradient

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Derivation</span><span class="math-callout__name">(State sampling — the outer sum becomes an on-policy average)</span></p>

For clarity, consider first the undiscounted episodic case ($\gamma = 1$). The policy-gradient theorem says

$$\nabla J(\boldsymbol{\theta}) \;\propto\; \sum_s \mu(s) \sum_a q_\pi(s, a)\, \nabla \pi(a \mid s, \boldsymbol{\theta}) \;=\; \mathbb{E}_{S \sim \mu}\!\left[ \sum_a q_\pi(S, a)\, \nabla \pi(a \mid S, \boldsymbol{\theta}) \right].$$

This is where **on-policy sampling** earns its keep: when we generate episodes using the *current* policy $\pi(\cdot \mid \cdot, \boldsymbol{\theta})$, the visited states $S\_t$ *are* samples from $\mu$. An empirical average over visited on-policy states therefore approximates the state expectation for free — no reweighting, no importance sampling.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Derivation</span><span class="math-callout__name">(Action sampling — from $q_\pi$ to the sampled return $G_t$)</span></p>

* Now fix one sampled on-policy state $S\_t$. 
* The inner sum over actions is not yet an expectation over the *sampled* action
  * fix that by multiplying and dividing by $\pi(a \mid S\_t, \boldsymbol{\theta})$:

$$
\begin{aligned}
\sum_a q_\pi(S_t, a)\, \nabla \pi(a \mid S_t, \boldsymbol{\theta})
&= \sum_a \pi(a \mid S_t, \boldsymbol{\theta})\, q_\pi(S_t, a)\, \frac{\nabla \pi(a \mid S_t, \boldsymbol{\theta})}{\pi(a \mid S_t, \boldsymbol{\theta})} \\
&= \sum_a \pi(a \mid S_t, \boldsymbol{\theta})\, q_\pi(S_t, a)\, \nabla \ln \pi(a \mid S_t, \boldsymbol{\theta}) \\
&= \mathbb{E}_{A \sim \pi(\cdot \mid S_t)}\bigl[\, q_\pi(S_t, A)\, \nabla \ln \pi(A \mid S_t, \boldsymbol{\theta}) \,\bigr].
\end{aligned}
$$

TODO: cannot we do this trick only if $\pi(a \mid S_t, \boldsymbol{\theta})$ \neq 0 for all $a$ in the sum? Is the non-zero probability guaranteed by a softmax that probably gives non-zero probability to any action?

* The action $A\_t \sim \pi(\cdot \mid S\_t, \boldsymbol{\theta})$ generated during the episode is a sample from exactly this expectation. 
* One unknown remains: $q\_\pi(S\_t, A\_t)$. 

But by definition 

$$q_\pi(S_t, A_t) = \mathbb{E}_\pi[\, G_t \mid S_t, A_t \,],$$

so the observed **Monte Carlo return** $G\_t$ is an unbiased sample of it. One on-policy time step thus yields a Monte Carlo sample of the *full* policy gradient:

$$\widehat{\nabla J} \;\propto\; G_t\, \nabla \ln \pi(A_t \mid S_t, \boldsymbol{\theta}).$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why later actions get the extra $\gamma^t$)</span></p>

For the *discounted* start-state objective, the sample direction at time $t$ picks up an additional weight:

$$\text{sample direction at time } t \;=\; \gamma^{t}\, G_t\, \nabla \ln \pi(A_t \mid S_t, \boldsymbol{\theta}_t).$$

The key point is the **perspective of the objective** 

$$J(\boldsymbol{\theta}) = v_\pi(s_0) = \mathbb{E}_\pi[G_0].$$

We are asking: *"how much does the action at time $t$ matter for $G\_0$?"* Take the action $A\_3$ as an example. Its consequences unfold in

$$G_3 = R_4 + \gamma R_5 + \gamma^2 R_6 + \cdots,$$

the future return *from time 3 onward*. But inside the original objective $G\_0$, that same future is **delayed**:

$$\gamma^3 G_3 = \gamma^3 R_4 + \gamma^4 R_5 + \gamma^5 R_6 + \cdots$$

So $G\_t$ estimates $q\_\pi(S\_t, A\_t)$ — the *quality* of the action — while $\gamma^t$ weights *how much time $t$ contributes* to $J(\boldsymbol{\theta}) = \mathbb{E}[G\_0]$. An action can only influence $G\_0$ through its delayed, $\gamma^t$-weighted slice, so its update carries the same $\gamma^t$.

</div>

<figure class="rl-diagram">
  <svg viewBox="0 0 760 300" role="img" aria-label="A chain of states and actions over one episode, with bars of geometrically decreasing height below each action showing that the weight on the action's update decays like gamma to the power t.">
    <circle cx="95" cy="55" r="21" class="accent"></circle>
    <text x="95" y="60" text-anchor="middle" font-size="13" font-weight="700">S₀</text>
    <circle cx="205" cy="55" r="21" class="box"></circle>
    <text x="205" y="60" text-anchor="middle" font-size="13" font-weight="700">S₁</text>
    <circle cx="315" cy="55" r="21" class="box"></circle>
    <text x="315" y="60" text-anchor="middle" font-size="13" font-weight="700">S₂</text>
    <circle cx="425" cy="55" r="21" class="box"></circle>
    <text x="425" y="60" text-anchor="middle" font-size="13" font-weight="700">S₃</text>
    <circle cx="535" cy="55" r="21" class="box"></circle>
    <text x="535" y="60" text-anchor="middle" font-size="13" font-weight="700">S₄</text>
    <circle cx="645" cy="55" r="21" class="box"></circle>
    <text x="645" y="60" text-anchor="middle" font-size="13" font-weight="700">S₅</text>

    <line x1="118" y1="55" x2="180" y2="55" class="line" marker-end="url(#pg-credit-arrow)"></line>
    <text x="149" y="40" text-anchor="middle" font-size="11" class="muted">A₀</text>
    <line x1="228" y1="55" x2="290" y2="55" class="line" marker-end="url(#pg-credit-arrow)"></line>
    <text x="259" y="40" text-anchor="middle" font-size="11" class="muted">A₁</text>
    <line x1="338" y1="55" x2="400" y2="55" class="line" marker-end="url(#pg-credit-arrow)"></line>
    <text x="369" y="40" text-anchor="middle" font-size="11" class="muted">A₂</text>
    <line x1="448" y1="55" x2="510" y2="55" class="line" marker-end="url(#pg-credit-arrow)"></line>
    <text x="479" y="40" text-anchor="middle" font-size="11" class="muted">A₃</text>
    <line x1="558" y1="55" x2="620" y2="55" class="line" marker-end="url(#pg-credit-arrow)"></line>
    <text x="589" y="40" text-anchor="middle" font-size="11" class="muted">A₄</text>
    <line x1="668" y1="55" x2="715" y2="55" class="line" marker-end="url(#pg-credit-arrow)"></line>
    <text x="730" y="60" font-size="11" class="muted">end</text>

    <text x="60" y="160" text-anchor="middle" font-size="11" class="muted">weight on</text>
    <text x="60" y="176" text-anchor="middle" font-size="11" class="muted">the action's</text>
    <text x="60" y="192" text-anchor="middle" font-size="11" class="muted">update</text>

    <rect x="127" y="115" width="44" height="130" fill="#0f9b6c" opacity="0.85"></rect>
    <text x="149" y="107" text-anchor="middle" font-size="12" fill="#0f9b6c" font-weight="700">γ⁰</text>
    <rect x="237" y="141" width="44" height="104" fill="#0f9b6c" opacity="0.85"></rect>
    <text x="259" y="133" text-anchor="middle" font-size="12" fill="#0f9b6c" font-weight="700">γ¹</text>
    <rect x="347" y="162" width="44" height="83" fill="#0f9b6c" opacity="0.85"></rect>
    <text x="369" y="154" text-anchor="middle" font-size="12" fill="#0f9b6c" font-weight="700">γ²</text>
    <rect x="457" y="178" width="44" height="67" fill="#0f9b6c" opacity="0.85"></rect>
    <text x="479" y="170" text-anchor="middle" font-size="12" fill="#0f9b6c" font-weight="700">γ³</text>
    <rect x="567" y="192" width="44" height="53" fill="#0f9b6c" opacity="0.85"></rect>
    <text x="589" y="184" text-anchor="middle" font-size="12" fill="#0f9b6c" font-weight="700">γ⁴</text>

    <line x1="100" y1="245" x2="680" y2="245" stroke="#64748b" stroke-width="1.5"></line>
    <text x="380" y="280" text-anchor="middle" font-size="12" class="muted">the contribution of Aₜ to the start-state objective G₀ is discounted by γᵗ</text>

    <defs>
      <marker id="pg-credit-arrow" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
        <path d="M0,0 L7,3 L0,6 Z" fill="#64748b"></path>
      </marker>
    </defs>
  </svg>
  <figcaption>Credit assignment over time: the update for the action taken at time $t$ is weighted by $\gamma^t$, because that is exactly how much of the action's future the start-state objective $G_0$ still "sees".</figcaption>
</figure>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(The REINFORCE update)</span></p>

The REINFORCE update for the discounted episodic case is

$$\boldsymbol{\theta}_{t+1} \;\doteq\; \boldsymbol{\theta}_t + \alpha\, \gamma^{t}\, G_t\, \nabla \ln \pi(A_t \mid S_t, \boldsymbol{\theta}_t).$$

Using $\nabla \ln \pi = \nabla \pi / \pi$, this is equivalently

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t + \alpha\, \gamma^{t}\, G_t\, \underbrace{\frac{\nabla \pi(A_t \mid S_t, \boldsymbol{\theta}_t)}{\pi(A_t \mid S_t, \boldsymbol{\theta}_t)}}_{\text{eligibility vector}}.$$

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(The REINFORCE update in words)</span></p>

*Increase the log-probability of the sampled action $A\_t$ in state $S\_t$, scaled by its return estimate $G\_t$ and its time weight $\gamma^t$.* The division by $\pi(A\_t \mid S\_t, \boldsymbol{\theta}\_t)$ in the eligibility vector also corrects for how often the action would be chosen anyway: frequently-taken actions get proportionally smaller pushes, so favourites do not win merely by being sampled more.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(REINFORCE is credit assignment over time)</span></p>

Think of one episode as a chain of decisions

$$S_0 \xrightarrow{\ A_0\ } S_1 \xrightarrow{\ A_1\ } S_2 \xrightarrow{\ A_2\ } \cdots \xrightarrow{\ A_{T-1}\ } \text{end}.$$

At every time step the policy asks: *"should I make this sampled action more or less likely next time?"* REINFORCE answers with 

$$\Delta \boldsymbol{\theta}_t \propto G_t\, \nabla \ln \pi(A_t \mid S_t, \boldsymbol{\theta}),$$

whose two factors have clean roles:

* $G\_t$ — *increase or decrease?* The observed quality of everything that followed the action; TODO: does not the magnitude of the G_t matters, not the sign. The policy could be any non-linear function of the current state, so what matters more is the direction of the update + the magnitude of the update. We do not know, which direction correspond to the lowering the probability of action to be chosen in the given state without knowing a specific policy function, do not we?
* $\nabla \ln \pi(A\_t \mid S\_t, \boldsymbol{\theta})$ — *which direction in parameter space* increases $\pi(A\_t \mid S\_t)$.

Locally: 
* $G\_t > 0 \Rightarrow$ make $A\_t$ more likely in $S\_t$;
* $G\_t < 0 \Rightarrow$ make it less likely.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(REINFORCE — Monte-Carlo policy-gradient control, episodic)</span></p>

* **Input:** 
  * a differentiable policy $\pi(a \mid s, \boldsymbol{\theta})$.
* **Parameter:** 
  * step size $\alpha > 0$. 

Initialise $\boldsymbol{\theta} \in \mathbb{R}^{d'}$.

Loop forever (for each episode):

1. Generate an episode 
   
   $$S_0, A_0, R_1, \dots, S_{T-1}, A_{T-1}, R_T$$
   
   following $\pi(\cdot \mid \cdot, \boldsymbol{\theta})$.
2. Loop for each step of the episode, $t = 0, 1, \dots, T-1$:
   * $G \leftarrow \sum\_{k=t+1}^{T} \gamma^{k-t-1} R\_k$  (this is $G\_t$);
   * $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha\, \gamma^{t}\, G\, \nabla \ln \pi(A\_t \mid S\_t, \boldsymbol{\theta})$.


</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Intuition</span><span class="math-callout__name">(REINFORCE — Monte-Carlo policy-gradient control, episodic)</span></p>

Try an action according to your current stochastic policy. See how good the outcome was. If the outcome was good, make that action more likely next time; if it was bad, make it less likely.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Eligibility vector for the soft-max linear policy)</span></p>

For $\pi(a \mid s, \boldsymbol{\theta}) \propto e^{\boldsymbol{\theta}^\top \mathbf{x}(s,a)}$,

$$\ln \pi(a \mid s, \boldsymbol{\theta}) = \boldsymbol{\theta}^\top \mathbf{x}(s, a) - \ln \sum_b e^{\boldsymbol{\theta}^\top \mathbf{x}(s, b)}.$$

Differentiating,

$$
\nabla \ln \pi(a \mid s, \boldsymbol{\theta})
= \mathbf{x}(s, a) - \frac{\sum_b e^{\boldsymbol{\theta}^\top \mathbf{x}(s,b)}\, \mathbf{x}(s, b)}{\sum_b e^{\boldsymbol{\theta}^\top \mathbf{x}(s,b)}}
= \mathbf{x}(s, a) - \sum_b \pi(b \mid s, \boldsymbol{\theta})\, \mathbf{x}(s, b).
$$

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Interpretation</span><span class="math-callout__name">(Eligibility vector for the soft-max linear policy)</span></p>

The eligibility vector points from the **expected feature vector under $\pi$** toward the **feature vector of the action actually taken**. (This is the state-dependent generalization of the soft-max derivative we computed for gradient bandits in Lecture 2.)

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(One REINFORCE step by hand)</span></p>

**Setup.** 
* Short corridor, 
* features 
  * $\mathbf{x}(s, \text{right}) = [1, 0]^\top$, 
  * $\mathbf{x}(s, \text{left}) = [0, 1]^\top$, 
* current parameters $\boldsymbol{\theta} = [0.4,\, -0.1]^\top$, 
* step size $\alpha = 0.1$, 
* $\gamma = 1$.

**Policy.**

$$
\pi(\text{right} \mid s) = \frac{e^{0.4}}{e^{0.4} + e^{-0.1}} = \frac{1.492}{1.492 + 0.905} \approx 0.622.
$$

**Suppose** we observe $A\_t = \text{right}$ and the rest of the episode produced $G\_t = -7$.

**Eligibility.**

$$
\nabla \ln \pi(\text{right} \mid s, \boldsymbol{\theta})
= \underbrace{\begin{bmatrix} 1 \\ 0 \end{bmatrix}}_{\mathbf{x}(s,\text{right})}
- \underbrace{0.622}_{\pi(\text{right} \mid s)} \begin{bmatrix} 1 \\ 0 \end{bmatrix}
- \underbrace{0.378}_{\pi(\text{left} \mid s)} \begin{bmatrix} 0 \\ 1 \end{bmatrix}
= \begin{bmatrix} 0.378 \\ -0.378 \end{bmatrix}.
$$

**Update.**

$$
\boldsymbol{\theta} \;\leftarrow\; \boldsymbol{\theta} + \alpha\, G_t\, \nabla \ln \pi
= \begin{bmatrix} 0.4 \\ -0.1 \end{bmatrix}
+ 0.1 \cdot (-7) \cdot \begin{bmatrix} 0.378 \\ -0.378 \end{bmatrix}
= \begin{bmatrix} 0.135 \\ 0.165 \end{bmatrix}.
$$

$\Rightarrow$ $\pi(\text{right} \mid s)$ drops from $0.622$ to $\approx 0.493$. A negative return $\Rightarrow$ decrease the chosen action's probability — exactly the local interpretation above.

</div>

<figure>
  <img src="{{ '/assets/images/notes/rl_hd/pg_reinforce_learning_curves.png' | relative_url }}" alt="Learning curves of REINFORCE on the short corridor for three step sizes: the middle step size converges near the optimum, the smallest converges more slowly, and the largest plateaus well below the optimum" loading="lazy">
  <figcaption>REINFORCE on the short corridor ($\gamma = 1$, $\theta_0$ chosen so that $\pi(\text{right}) \approx 0.05$; episodes capped at 500 steps; smoothed average of 200 runs). $\alpha = 2^{-13}$ converges toward $v_*(s_0)$; $\alpha = 2^{-14}$ converges more slowly; $\alpha = 2^{-12}$ is too aggressive — in a fraction of runs the noisy updates temporarily collapse $\pi(\text{right})$ toward $0$, and the average pays for it.</figcaption>
</figure>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(REINFORCE — what we can guarantee, and what it costs)</span></p>

* **Unbiased:** the sample direction is an unbiased estimator of the policy-gradient direction, up to the theorem's constant of proportionality.
* **Convergent:** $\boldsymbol{\theta}\_t$ converges to a *local* optimum of $J$ under the standard stochastic-approximation step-size conditions and suitable regularity assumptions.
* **Slow:** the return $G\_t$ has **high variance**, so learning is slow — and, as the experiment above shows, sensitive to the step size.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Caveat</span><span class="math-callout__name">(Variance is the practical bottleneck)</span></p>

* The same parameter step depends on the **entire random future trajectory**: 
  * every reward from $t+1$ to the end of the episode enters $G\_t$. 
* Two episodes that visit similar state–action sequences can still produce very different returns — and hence very different updates. 
* This single weakness motivates the two classical fixes: 
  * **bootstrapping** (next lecture, actor–critic) and
  * **baselines** (below). 

</div>

### REINFORCE with Baseline

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Derivation</span><span class="math-callout__name">(Subtracting a baseline does not change the gradient)</span></p>

Start from the policy-gradient theorem and subtract an arbitrary **state-dependent baseline** $b(s)$ from $q\_\pi$:

$$
\sum_s \mu(s) \sum_a \bigl( q_\pi(s, a) - b(s) \bigr)\, \nabla \pi(a \mid s, \boldsymbol{\theta})
= \sum_s \mu(s) \sum_a q_\pi(s, a)\, \nabla \pi(a \mid s, \boldsymbol{\theta})
- \sum_s \mu(s) \sum_a b(s)\, \nabla \pi(a \mid s, \boldsymbol{\theta}).
$$

For each fixed state $s$, the subtracted term vanishes because the action probabilities always sum to one:

$$\sum_a b(s)\, \nabla \pi(a \mid s, \boldsymbol{\theta}) = b(s)\, \nabla \sum_a \pi(a \mid s, \boldsymbol{\theta}) = b(s)\, \nabla 1 = 0.$$

Therefore the policy gradient theorem **also holds with a baseline**:

$$\nabla J(\boldsymbol{\theta}) \;\propto\; \sum_s \mu(s) \sum_a \bigl( q_\pi(s, a) - b(s) \bigr)\, \nabla \pi(a \mid s, \boldsymbol{\theta}).$$

(The bandit ancestor of this trick: subtracting the running average reward $\bar R\_t$ in the gradient-bandit update of Lecture 2.)

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why baselines reduce variance)</span></p>

The discounted episodic update with a baseline is

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t + \alpha\, \gamma^{t} \bigl( G_t - b(S_t) \bigr)\, \nabla \ln \pi(A_t \mid S_t, \boldsymbol{\theta}_t).$$

* Adding $-b(s)\, \nabla \ln \pi$ to the estimator does **not** change its mean
  * see the derivation above.
* But it can **drastically change its variance**. 
  * A good baseline removes *common-mode noise*: 
  * if all actions in $s$ lead to a high return, the update should reflect the **relative** preference among actions, not the absolute return size.

**Natural choice:** 

$$b(s) = \hat v(s, \mathbf{w}),$$

a learned state-value estimate. Then 

$$G_t - \hat v(S_t, \mathbf{w})$$

is approximately the **advantage** of the chosen action — *was this action better or worse than what we typically achieve from this state?*

</div>

<figure>
  <img src="{{ '/assets/images/notes/rl_hd/pg_baseline_variance_histograms.png' | relative_url }}" alt="Two histograms of update signals collected early in training: raw returns are large and widely spread, while advantages after subtracting the learned baseline are small and concentrated around zero" loading="lazy">
  <figcaption>What a baseline actually does (update signals collected early in training on the short corridor). Subtracting $b(S_t)$ leaves the expected gradient untouched but removes the common-mode part of the return: the update reflects how much <em>better than typical</em> the action was, not the raw return level.</figcaption>
</figure>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(REINFORCE with baseline — episodic)</span></p>

* **Input:** 
  * a differentiable policy $\pi(a \mid s, \boldsymbol{\theta})$
  * a differentiable value function $\hat v(s, \mathbf{w})$.
* **Parameters:** 
  * step size $\alpha^{\boldsymbol{\theta}} > 0$
  * step size $\alpha^{\mathbf{w}} > 0$. 

Initialise $\boldsymbol{\theta}$, $\mathbf{w}$.

Loop forever (for each episode):

1. Generate an episode 
   
   $$S_0, A_0, R_1, \dots, S_{T-1}, A_{T-1}, R_T$$
   
   following $\pi(\cdot \mid \cdot, \boldsymbol{\theta})$.
2. Loop for each step of the episode, $t = 0, 1, \dots, T-1$:
   * $G \leftarrow \sum\_{k=t+1}^{T} \gamma^{k-t-1} R\_k$;
   * $\delta \leftarrow G - \hat v(S\_t, \mathbf{w})$;
   * $\mathbf{w} \leftarrow \mathbf{w} + \alpha^{\mathbf{w}}\, \delta\, \nabla \hat v(S\_t, \mathbf{w})$;
   * $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha^{\boldsymbol{\theta}}\, \gamma^{t}\, \delta\, \nabla \ln \pi(A\_t \mid S\_t, \boldsymbol{\theta})$.

Two step sizes: 
* $\alpha^{\mathbf{w}}$ (a plain Monte Carlo regression of $\hat v$ onto observed returns, as in Lecture 9) is often easy to tune; 
* $\alpha^{\boldsymbol{\theta}}$ depends on the reward scale and the policy parameterization and is the delicate one.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remarks on the baseline — and what it is <em>not</em>)</span></p>

* The baseline is usually a **learned value function**, $b(s) = \hat v(s, \mathbf{w})$, updated by Monte Carlo regression 
  
  $$\mathbf{w} \leftarrow \mathbf{w} + \alpha^{\mathbf{w}} \bigl( G_t - \hat v(S_t, \mathbf{w}) \bigr) \nabla \hat v(S_t, \mathbf{w}).$$

* The policy update uses an **advantage-like signal** $G\_t - \hat v(S\_t, \mathbf{w})$: we ask whether the action was *better or worse than expected*, not whether the return was large in absolute terms.
* This does **not** turn the method into a value-based method. The policy $\pi(a \mid s, \boldsymbol{\theta})$ is still the **actor** — the value function never selects actions, it only sharpens the update. (Since $\hat v$ here evaluates but does not yet *bootstrap*, it is not a critic in the strict actor–critic sense; that step comes in the next section.)
* With **neural networks**, replace the linear preferences by logits: 
  
  $$h_{\boldsymbol{\theta}}(s, a) = \text{NN}_{\boldsymbol{\theta}}(s)_a$$
  
  and 
  
  $$\pi(a \mid s, \boldsymbol{\theta}) = \operatorname{softmax}(h_{\boldsymbol{\theta}}(s))_a;$$ 
  
  the gradients $\nabla \ln \pi$ are then computed by backpropagation.

</div>

<figure>
  <img src="{{ '/assets/images/notes/rl_hd/pg_reinforce_baseline_comparison.png' | relative_url }}" alt="Learning curves comparing plain REINFORCE with REINFORCE plus a learned baseline on the short corridor; the baseline version converges within about a hundred episodes while plain REINFORCE takes many hundreds" loading="lazy">
  <figcaption>Baseline in practice (short corridor; smoothed average of 200 runs). Plain REINFORCE at its best step size $\alpha = 2^{-13}$ takes hundreds of episodes; REINFORCE with a learned scalar baseline ($\alpha^\theta = 2^{-9}$, $\alpha^w = 2^{-6}$) converges much faster. Same expected gradient direction, lower variance — the baseline itself introduces no bias.</figcaption>
</figure>

### Actor–Critic Methods

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Idea</span><span class="math-callout__name">(From baseline to critic: bootstrapping)</span></p>

In REINFORCE with baseline, $\hat v(S\_t, \mathbf{w})$ is *only* a baseline. It does **not** replace the Monte Carlo return $G\_t$ — every update still waits for the end of the episode.

What if we bootstrap, replacing $G\_t$ with the **one-step return**

$$G_{t:t+1} \;=\; R_{t+1} + \gamma\, \hat v(S_{t+1}, \mathbf{w})\,?$$

Then the value function is used **both** as a baseline at $S\_t$ **and** as a forecast of the future from $S\_{t+1}$. It is now a **critic**.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Actor–critic method)</span></p>

An **actor–critic** method is a policy-gradient method that uses a learned value function to **bootstrap** the return estimate.

* **Actor:** the policy $\pi(a \mid s, \boldsymbol{\theta})$ — it selects the actions.
* **Critic:** the value function $\hat v(s, \mathbf{w})$ — it evaluates (criticizes) the actor's action choices.

</div>

<figure class="rl-diagram">
  <svg viewBox="0 0 800 380" role="img" aria-label="The actor-critic loop: the actor sends an action to the environment, the environment returns reward and next state to the critic, the critic feeds its value estimates into the TD error, and the TD error both steers the actor and improves the critic.">
    <rect x="90" y="60" width="210" height="64" rx="10" class="accent"></rect>
    <text x="195" y="87" text-anchor="middle" font-size="15" font-weight="700" fill="#2c3e94">ACTOR</text>
    <text x="195" y="108" text-anchor="middle" font-size="13" fill="#2c3e94">policy π(a | s, θ)</text>

    <rect x="90" y="260" width="210" height="64" rx="10" class="green"></rect>
    <text x="195" y="287" text-anchor="middle" font-size="15" font-weight="700" fill="#047857">CRITIC</text>
    <text x="195" y="308" text-anchor="middle" font-size="13" fill="#047857">value v̂(s, w)</text>

    <rect x="350" y="160" width="130" height="62" rx="10" class="box"></rect>
    <text x="415" y="186" text-anchor="middle" font-size="13" font-weight="700">TD error</text>
    <text x="415" y="208" text-anchor="middle" font-size="14">δₜ</text>

    <rect x="580" y="155" width="170" height="72" rx="10" class="box"></rect>
    <text x="665" y="197" text-anchor="middle" font-size="14" font-weight="700">ENVIRONMENT</text>

    <path d="M300,84 C450,52 570,74 638,150" class="line" stroke="#2c3e94" marker-end="url(#pg-ac-blue)"></path>
    <text x="470" y="56" text-anchor="middle" font-size="12" fill="#2c3e94">action A</text>

    <path d="M638,230 C560,326 420,312 306,300" class="line" marker-end="url(#pg-ac-gray)"></path>
    <text x="490" y="330" text-anchor="middle" font-size="12" class="muted">R, S′</text>

    <path d="M290,258 C320,240 340,232 362,226" fill="none" stroke="#047857" stroke-width="2" marker-end="url(#pg-ac-green)"></path>
    <text x="288" y="242" text-anchor="middle" font-size="11" fill="#047857">v̂(Sₜ), v̂(Sₜ₊₁)</text>

    <path d="M366,158 C336,142 316,134 288,128" fill="none" stroke="#b45309" stroke-width="2" marker-end="url(#pg-ac-amber)"></path>
    <text x="352" y="132" text-anchor="middle" font-size="11" fill="#b45309">steer policy</text>

    <path d="M398,226 C360,248 320,256 292,260" fill="none" stroke="#b45309" stroke-width="2" marker-end="url(#pg-ac-amber)"></path>
    <text x="392" y="252" text-anchor="middle" font-size="11" fill="#b45309">improve value</text>

    <text x="400" y="362" text-anchor="middle" font-size="12" class="muted">δₜ = Rₜ₊₁ + γ v̂(Sₜ₊₁) − v̂(Sₜ)   drives both updates</text>

    <defs>
      <marker id="pg-ac-gray" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
        <path d="M0,0 L7,3 L0,6 Z" fill="#64748b"></path>
      </marker>
      <marker id="pg-ac-blue" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
        <path d="M0,0 L7,3 L0,6 Z" fill="#2c3e94"></path>
      </marker>
      <marker id="pg-ac-green" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
        <path d="M0,0 L7,3 L0,6 Z" fill="#047857"></path>
      </marker>
      <marker id="pg-ac-amber" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
        <path d="M0,0 L7,3 L0,6 Z" fill="#b45309"></path>
      </marker>
    </defs>
  </svg>
  <figcaption>The actor–critic loop. One signal, $\delta_t$, does double duty: it <em>steers the actor</em> and <em>improves the critic</em>. And because $\delta_t$ uses $\hat v(S_{t+1})$, the critic now <em>bootstraps</em> — enabling fully online, incremental updates.</figcaption>
</figure>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Derivation</span><span class="math-callout__name">(What is the critic estimating?)</span></p>

The policy-gradient theorem suggests an actor update of the form

$$\Delta \boldsymbol{\theta} \;\propto\; q_\pi(S_t, A_t)\, \nabla \ln \pi(A_t \mid S_t, \boldsymbol{\theta}).$$

A critic can estimate the action value directly:

$$q_\pi(S_t, A_t) \;\approx\; \hat q(S_t, A_t, \mathbf{w}).$$

This is actor–critic **without an explicit advantage baseline**. More commonly, we use a *value* critic and subtract a baseline:

$$\widehat{A}_t \;=\; \underbrace{R_{t+1} + \gamma \hat v(S_{t+1}, \mathbf{w})}_{=\, \hat q(S_t, A_t, \mathbf{w})} \;-\; \hat v(S_t, \mathbf{w}).$$

**Terminology:** *actor–critic* is the general family. *Advantage actor–critic* uses an advantage estimate to reduce variance.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Derivation</span><span class="math-callout__name">(The one-step actor–critic update)</span></p>

Start from the REINFORCE-with-baseline update:

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t + \alpha\, \gamma^{t} \underbrace{\bigl( G_t - \hat v(S_t, \mathbf{w}) \bigr)}_{\text{Monte Carlo advantage estimate}} \nabla \ln \pi(A_t \mid S_t, \boldsymbol{\theta}_t).$$

Actor–critic replaces the full return $G\_t$ by the one-step bootstrapped estimate of $q\_\pi(S\_t, A\_t)$:

$$G_t \;\rightsquigarrow\; R_{t+1} + \gamma\, \hat v(S_{t+1}, \mathbf{w}).$$

Therefore the **advantage estimate** becomes

$$\widehat{A}_t = R_{t+1} + \gamma \hat v(S_{t+1}, \mathbf{w}) - \hat v(S_t, \mathbf{w}) = \delta_t$$

— exactly the **TD error** of Lecture 6, computed with the learned $\hat v$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Method</span><span class="math-callout__name">(One-step actor–critic update, discounted episodic)</span></p>

$$\boldsymbol{\theta}_{t+1} \;=\; \boldsymbol{\theta}_t + \alpha\, \gamma^{t}\, \delta_t\, \nabla \ln \pi(A_t \mid S_t, \boldsymbol{\theta}_t).$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Caveat</span><span class="math-callout__name">(Bias–variance trade-off)</span></p>

Replacing $G\_t$ by a bootstrapped estimate usually **lowers variance**, but can **introduce bias** when $\hat v$ is inaccurate. This is the same dial we turned in Lecture 7: Monte Carlo at one end, one-step TD at the other, $n$-step targets in between — except now it sits *inside* the policy-gradient update rather than inside a value-learning rule.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(One-step actor–critic — episodic)</span></p>

* **Input:**
  * a differentiable policy $\pi(a \mid s, \boldsymbol{\theta})$
  * a differentiable value function $\hat v(s, \mathbf{w})$.
* **Parameters:**
  * step size $\alpha^{\boldsymbol{\theta}} > 0$
  * step size $\alpha^{\mathbf{w}} > 0$.

Initialise $\boldsymbol{\theta}$, $\mathbf{w}$.

Loop forever (for each episode):

1. Initialise $S$ (first state of the episode); $I \leftarrow 1$.
2. Loop while $S$ is not terminal:
   * $A \sim \pi(\cdot \mid S, \boldsymbol{\theta})$;
   * take action $A$;
   * observe $S'$, $R$;
   * $\delta \leftarrow R + \gamma \hat v(S', \mathbf{w}) - \hat v(S, \mathbf{w})$  
     * with $\hat v(\text{terminal}) \doteq 0$;
   * $\mathbf{w} \leftarrow \mathbf{w} + \alpha^{\mathbf{w}}\, \delta\, \nabla \hat v(S, \mathbf{w})$;
   * $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha^{\boldsymbol{\theta}}\, I\, \delta\, \nabla \ln \pi(A \mid S, \boldsymbol{\theta})$;
   * $I \leftarrow \gamma I$;  
   * $S \leftarrow S'$.

Two features worth noting:

* **Fully online and incremental** — every step of experience produces an update; there is no need to wait for the episode to end. (The running factor $I = \gamma^t$ plays the role of the explicit $\gamma^t$ in REINFORCE.)
* **Generalizes to $n$-step returns** — replace the one-step target by an $n$-step target (or an eligibility-trace variant) to tune the bias–variance trade-off.

</div>

<figure class="rl-diagram">
  <svg viewBox="0 0 780 360" role="img" aria-label="Family tree of policy-gradient methods: the policy gradient theorem at the top, REINFORCE below it, and two children of REINFORCE at the bottom - REINFORCE with baseline on the left and actor-critic on the right.">
    <rect x="250" y="36" width="280" height="66" rx="10" class="accent"></rect>
    <text x="390" y="62" text-anchor="middle" font-size="14" font-weight="700" fill="#2c3e94">Policy Gradient Theorem</text>
    <text x="390" y="86" text-anchor="middle" font-size="13" fill="#2c3e94">∇J ∝ Σₛ μ(s) Σₐ q<tspan font-size="10" dy="3">π</tspan><tspan dy="-3"> ∇π</tspan></text>

    <rect x="290" y="152" width="200" height="62" rx="10" class="box"></rect>
    <text x="390" y="178" text-anchor="middle" font-size="14" font-weight="700">REINFORCE</text>
    <text x="390" y="200" text-anchor="middle" font-size="13" class="muted">γᵗ Gₜ ∇ln π</text>

    <rect x="70" y="266" width="270" height="62" rx="10" class="box"></rect>
    <text x="205" y="292" text-anchor="middle" font-size="14" font-weight="700">REINFORCE w/ baseline</text>
    <text x="205" y="314" text-anchor="middle" font-size="13" class="muted">γᵗ (Gₜ − v̂) ∇ln π</text>

    <rect x="460" y="266" width="220" height="62" rx="10" class="box"></rect>
    <text x="570" y="292" text-anchor="middle" font-size="14" font-weight="700">Actor–Critic</text>
    <text x="570" y="314" text-anchor="middle" font-size="13" class="muted">γᵗ δₜ ∇ln π</text>

    <line x1="390" y1="102" x2="390" y2="146" class="line" marker-end="url(#pg-tree-arrow)"></line>
    <line x1="350" y1="214" x2="235" y2="260" class="line" marker-end="url(#pg-tree-arrow)"></line>
    <line x1="430" y1="214" x2="545" y2="260" class="line" marker-end="url(#pg-tree-arrow)"></line>

    <defs>
      <marker id="pg-tree-arrow" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
        <path d="M0,0 L7,3 L0,6 Z" fill="#64748b"></path>
      </marker>
    </defs>
  </svg>
  <figcaption>The family tree of this lecture. <strong>Baseline:</strong> variance reduction without bias. <strong>Critic:</strong> bootstrapping enables online learning and trades a little bias for lower variance.</figcaption>
</figure>

<figure class="rl-diagram">
  <svg viewBox="0 0 860 280" role="img" aria-label="A horizontal axis from more bootstrapping on the left to full Monte Carlo on the right, with four methods placed along it: one-step actor-critic (low variance, some bias), n-step or lambda actor-critic (tunable trade-off), REINFORCE with baseline (unbiased, lower variance), and REINFORCE (unbiased, high variance).">
    <line x1="50" y1="200" x2="806" y2="200" class="line" marker-end="url(#pg-axis-arrow)"></line>
    <text x="60" y="230" font-size="12" class="muted">more bootstrapping</text>
    <text x="800" y="230" text-anchor="end" font-size="12" class="muted">full Monte-Carlo</text>

    <text x="160" y="58" text-anchor="middle" font-size="11" font-style="italic" fill="#047857">low variance</text>
    <text x="160" y="74" text-anchor="middle" font-size="11" font-style="italic" fill="#047857">some bias</text>
    <rect x="85" y="96" width="150" height="56" rx="8" class="green"></rect>
    <text x="160" y="119" text-anchor="middle" font-size="13" font-weight="700" fill="#047857">one-step</text>
    <text x="160" y="138" text-anchor="middle" font-size="13" font-weight="700" fill="#047857">actor–critic</text>
    <line x1="160" y1="152" x2="160" y2="192" class="line"></line>
    <g fill="#047857"><circle cx="160" cy="200" r="7"></circle></g>

    <text x="350" y="58" text-anchor="middle" font-size="11" font-style="italic" fill="#2c3e94">tunable</text>
    <text x="350" y="74" text-anchor="middle" font-size="11" font-style="italic" fill="#2c3e94">trade-off</text>
    <rect x="275" y="96" width="150" height="56" rx="8" class="accent"></rect>
    <text x="350" y="119" text-anchor="middle" font-size="13" font-weight="700" fill="#2c3e94">n-step / λ</text>
    <text x="350" y="138" text-anchor="middle" font-size="13" font-weight="700" fill="#2c3e94">actor–critic</text>
    <line x1="350" y1="152" x2="350" y2="192" class="line"></line>
    <circle cx="350" cy="200" r="7" fill="#2c3e94"></circle>

    <text x="565" y="58" text-anchor="middle" font-size="11" font-style="italic" fill="#b45309">unbiased</text>
    <text x="565" y="74" text-anchor="middle" font-size="11" font-style="italic" fill="#b45309">lower variance</text>
    <rect x="480" y="96" width="170" height="56" rx="8" class="amber"></rect>
    <text x="565" y="119" text-anchor="middle" font-size="13" font-weight="700" fill="#b45309">REINFORCE</text>
    <text x="565" y="138" text-anchor="middle" font-size="13" font-weight="700" fill="#b45309">w/ baseline</text>
    <line x1="565" y1="152" x2="565" y2="192" class="line"></line>
    <g fill="#b45309"><circle cx="565" cy="200" r="7"></circle></g>

    <text x="740" y="58" text-anchor="middle" font-size="11" font-style="italic" fill="#b45309">unbiased</text>
    <text x="740" y="74" text-anchor="middle" font-size="11" font-style="italic" fill="#b45309">high variance</text>
    <rect x="665" y="96" width="150" height="56" rx="8" class="amber"></rect>
    <text x="740" y="128" text-anchor="middle" font-size="13" font-weight="700" fill="#b45309">REINFORCE</text>
    <line x1="740" y1="152" x2="740" y2="192" class="line"></line>
    <g fill="#b45309"><circle cx="740" cy="200" r="7"></circle></g>

    <defs>
      <marker id="pg-axis-arrow" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
        <path d="M0,0 L7,3 L0,6 Z" fill="#64748b"></path>
      </marker>
    </defs>
  </svg>
  <figcaption>One axis: how much do we bootstrap? Moving left trades a little bias for much lower variance and online learning; moving right keeps the estimator unbiased but pays in variance.</figcaption>
</figure>

### Case Study: AlphaGo

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Problem</span><span class="math-callout__name">(The game of Go, as an MDP)</span></p>

* Two players, **Black** and **White**, alternately place stones on a $19 \times 19$ grid.
* **Goal:** surround and control more territory (and capture enemy stones) than the opponent.
* Hard for computers: about **250 legal moves** per turn over $\sim 150$ moves $\Rightarrow \sim 10^{170}$ positions.
* And positions are **hard to evaluate** by hand (unlike chess piece values).

As an MDP:

* **State** = board position, **actions** = legal moves, **reward** = $+1$ win / $-1$ loss at the end (zero along the way).
* Brute-force search is hopeless; AlphaGo *learns* a policy and a value to guide a focused search.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(AlphaGo — this lecture, at scale)</span></p>

* AlphaGo (DeepMind, 2016) was the first program to beat a professional Go player on a full board — **Fan Hui 5–0**, then **Lee Sedol 4–1**.
* Its policy was improved by **exactly the algorithm in this lecture**: REINFORCE on self-play games.
* On the second training pass it used a learned value function as a **baseline** — the variance-reduction trick from the previous section.

**The one-line summary.** Training consists of 3 steps: **supervised learning** to give it a strong starting policy; **REINFORCE self-play** to improve it; and a **value network** (baseline + leaf evaluator) and **tree search** to evaluate.

</div>

<figure class="rl-diagram">
  <svg viewBox="0 0 860 400" role="img" aria-label="AlphaGo's training pipeline in three stages: supervised learning trains an SL policy and a fast rollout policy from human expert games; reinforcement learning improves an RL policy by REINFORCE self-play, initialized from the SL policy; and the value network is trained by regression on self-play outcomes.">
    <text x="150" y="42" text-anchor="middle" font-size="15" font-weight="700" fill="#2c3e94">1. Supervised</text>
    <text x="430" y="42" text-anchor="middle" font-size="15" font-weight="700" fill="#047857">2. Reinforcement</text>
    <text x="710" y="42" text-anchor="middle" font-size="15" font-weight="700" fill="#b45309">3. Evaluation</text>

    <line x1="295" y1="62" x2="295" y2="370" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="6 6"></line>
    <line x1="575" y1="62" x2="575" y2="370" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="6 6"></line>

    <rect x="55" y="80" width="190" height="48" rx="8" class="box"></rect>
    <text x="150" y="109" text-anchor="middle" font-size="13">human expert games</text>

    <line x1="150" y1="128" x2="150" y2="176" class="line" marker-end="url(#pg-ago-gray)"></line>
    <text x="164" y="158" font-size="11" class="muted">train</text>

    <rect x="55" y="180" width="190" height="60" rx="8" class="accent"></rect>
    <text x="150" y="204" text-anchor="middle" font-size="13" font-weight="700" fill="#2c3e94">SL policy  p<tspan font-size="10" dy="3">σ</tspan></text>
    <text x="150" y="226" text-anchor="middle" font-size="11" class="muted">supervised, ~57%</text>

    <path d="M66,128 C24,180 24,250 62,296" class="line" marker-end="url(#pg-ago-gray)"></path>

    <rect x="55" y="290" width="190" height="60" rx="8" class="box"></rect>
    <text x="150" y="314" text-anchor="middle" font-size="13" font-weight="700">rollout policy  p<tspan font-size="10" dy="3">π</tspan></text>
    <text x="150" y="336" text-anchor="middle" font-size="11" class="muted">fast, ~24%</text>

    <line x1="245" y1="210" x2="331" y2="210" stroke="#2c3e94" stroke-width="2" stroke-dasharray="5 5" marker-end="url(#pg-ago-blue)"></line>
    <text x="288" y="200" text-anchor="middle" font-size="11" class="muted">init</text>

    <rect x="335" y="180" width="190" height="60" rx="8" class="green"></rect>
    <text x="430" y="204" text-anchor="middle" font-size="13" font-weight="700" fill="#047857">RL policy  p<tspan font-size="10" dy="3">ρ</tspan></text>
    <text x="430" y="226" text-anchor="middle" font-size="11" class="muted">self-play; reward z = ±1</text>

    <path d="M385,176 C365,110 495,110 475,176" fill="none" stroke="#047857" stroke-width="2" marker-end="url(#pg-ago-green)"></path>
    <text x="430" y="112" text-anchor="middle" font-size="12" font-weight="700" fill="#047857">REINFORCE</text>

    <line x1="525" y1="210" x2="611" y2="210" class="line" marker-end="url(#pg-ago-gray)"></line>
    <text x="568" y="196" text-anchor="middle" font-size="10" class="muted">self-play</text>
    <text x="568" y="232" text-anchor="middle" font-size="10" class="muted">data</text>

    <rect x="615" y="180" width="190" height="60" rx="8" class="amber"></rect>
    <text x="710" y="204" text-anchor="middle" font-size="13" font-weight="700" fill="#b45309">value net  v<tspan font-size="10" dy="3">θ</tspan></text>
    <text x="710" y="226" text-anchor="middle" font-size="11" class="muted">regress to outcome</text>

    <defs>
      <marker id="pg-ago-gray" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
        <path d="M0,0 L7,3 L0,6 Z" fill="#64748b"></path>
      </marker>
      <marker id="pg-ago-blue" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
        <path d="M0,0 L7,3 L0,6 Z" fill="#2c3e94"></path>
      </marker>
      <marker id="pg-ago-green" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
        <path d="M0,0 L7,3 L0,6 Z" fill="#047857"></path>
      </marker>
    </defs>
  </svg>
  <figcaption>AlphaGo's training pipeline. The REINFORCE stage updates the RL policy by $\Delta\rho \propto \nabla_{\rho} \log p_\rho(a_t \mid s_t)\,(z_t - b)$ with baseline $b = v_\theta$ — the REINFORCE-with-baseline update of this lecture, with the game outcome $z = \pm 1$ as the (undiscounted) return. Concept re-creation. Source: Silver <em>et al.</em>, "Mastering the game of Go with deep neural networks and tree search," <em>Nature</em> 529:484–489 (2016).</figcaption>
</figure>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(The four networks)</span></p>

| Network | Trained how | Role |
|---|---|---|
| $p\_\pi$ — rollout | supervised, linear soft-max | fast move sampling inside MCTS rollouts ($\sim 24\%$) |
| $p\_\sigma$ — SL policy | supervised on $\sim 30$M expert moves | strong prior; seeds MCTS priors ($\sim 57\%$) |
| $p\_\rho$ — RL policy | **REINFORCE** self-play, init from $p\_\sigma$ | generates self-play data to train $v\_\theta$ |
| $v\_\theta$ — value | regression to self-play outcomes | leaf evaluation + REINFORCE baseline |

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(A subtlety worth noting)</span></p>

The *SL* policy $p\_\sigma$ (not the stronger RL policy $p\_\rho$) gave better MCTS priors — it is more **diverse**. The RL policy's real job was to produce the data that trains the value network $v\_\theta$.

</div>

<figure class="rl-diagram">
  <svg viewBox="0 0 860 300" role="img" aria-label="Shared network architecture: stacked input planes of size 19 by 19 by 48 feed a shared convolutional body of 13 layers with 192 filters, which branches into a policy head producing move probabilities and a value head producing a win probability.">
    <text x="430" y="36" text-anchor="middle" font-size="12" font-style="italic" class="muted">Three networks share one conv body; the value net adds an FC layer + scalar output.</text>

    <rect x="78" y="106" width="76" height="76" rx="4" class="box"></rect>
    <rect x="70" y="98" width="76" height="76" rx="4" class="box"></rect>
    <rect x="62" y="90" width="76" height="76" rx="4" class="box"></rect>
    <rect x="54" y="82" width="76" height="76" rx="4" class="box"></rect>
    <text x="104" y="216" text-anchor="middle" font-size="11" class="muted">input 19 × 19 × 48</text>

    <line x1="160" y1="130" x2="216" y2="130" class="line" marker-end="url(#pg-net-gray)"></line>

    <rect x="224" y="78" width="11" height="104" rx="2" fill="#94a3b8"></rect>
    <rect x="243" y="78" width="11" height="104" rx="2" fill="#94a3b8"></rect>
    <rect x="262" y="78" width="11" height="104" rx="2" fill="#94a3b8"></rect>
    <rect x="281" y="78" width="11" height="104" rx="2" fill="#94a3b8"></rect>
    <rect x="300" y="78" width="11" height="104" rx="2" fill="#94a3b8"></rect>
    <rect x="319" y="78" width="11" height="104" rx="2" fill="#94a3b8"></rect>
    <rect x="338" y="78" width="11" height="104" rx="2" fill="#94a3b8"></rect>
    <text x="286" y="216" text-anchor="middle" font-size="11" class="muted">13 conv layers, 192 filters</text>

    <path d="M355,112 C385,96 400,92 424,88" class="line" marker-end="url(#pg-net-gray)"></path>
    <path d="M355,148 C385,164 400,168 424,172" class="line" marker-end="url(#pg-net-gray)"></path>

    <rect x="430" y="58" width="170" height="56" rx="8" class="accent"></rect>
    <text x="515" y="81" text-anchor="middle" font-size="13" font-weight="700" fill="#2c3e94">policy head</text>
    <text x="515" y="102" text-anchor="middle" font-size="12" fill="#2c3e94">p<tspan font-size="10" dy="3">σ</tspan><tspan dy="-3">, p</tspan><tspan font-size="10" dy="3">ρ</tspan></text>

    <line x1="600" y1="86" x2="656" y2="86" class="line" marker-end="url(#pg-net-gray)"></line>

    <rect x="662" y="58" width="150" height="56" rx="8" class="box"></rect>
    <text x="737" y="81" text-anchor="middle" font-size="13">move probs</text>
    <text x="737" y="102" text-anchor="middle" font-size="12" class="muted">p(a | s)</text>

    <rect x="430" y="146" width="170" height="56" rx="8" class="amber"></rect>
    <text x="515" y="169" text-anchor="middle" font-size="13" font-weight="700" fill="#b45309">value head</text>
    <text x="515" y="190" text-anchor="middle" font-size="12" fill="#b45309">v<tspan font-size="10" dy="3">θ</tspan></text>

    <line x1="600" y1="174" x2="656" y2="174" class="line" marker-end="url(#pg-net-gray)"></line>

    <rect x="662" y="146" width="150" height="56" rx="8" class="box"></rect>
    <text x="737" y="169" text-anchor="middle" font-size="13">win prob</text>
    <text x="737" y="190" text-anchor="middle" font-size="12" class="muted">v(s)</text>

    <defs>
      <marker id="pg-net-gray" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
        <path d="M0,0 L7,3 L0,6 Z" fill="#64748b"></path>
      </marker>
    </defs>
  </svg>
  <figcaption>Shared architecture, different heads. Concept re-creation. Source: Silver <em>et al.</em>, <em>Nature</em> 529:484–489 (2016).</figcaption>
</figure>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Playing a move: look-ahead search — propose, judge, decide)</span></p>

To choose its *real* move, AlphaGo does not just sample the policy network. It runs thousands of look-ahead simulations, using **MCTS** — the selection / expansion / evaluation / backup loop from Lecture 8 — and then plays the move that held up best under all that scrutiny.

The division of labour:

* the **policy network** *proposes* which moves are worth looking at (it seeds the tree's priors),
* the **value network** *judges* how good the resulting positions are (leaf evaluation, alongside fast $p\_\pi$ rollouts),
* and the **search** *decides* — by accumulating those judgements over many simulations.

</div>

### Summary

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(What we learned)</span></p>

1. **Policies can be learned directly.** Parameterize $\pi(a \mid s, \boldsymbol{\theta})$ and do gradient ascent on $J(\boldsymbol{\theta}) = v\_{\pi\_{\boldsymbol{\theta}}}(s\_0)$; a value function becomes optional machinery for the update, not the decision-maker.
2. **Soft-max in preferences** gives a smooth, differentiable policy class that can approach determinism *and* represent stochastic optima — $\varepsilon$-greedy can do neither.
3. **The short corridor** shows the stakes: under state aliasing the best representable policy is genuinely stochastic, and every $\varepsilon$-greedy action-value method is beaten by a modest stochastic policy ($-11.6$ vs. $-44$).
4. **The policy gradient theorem** is the enabling result: 
   
   $$\nabla J \propto \sum_s \mu(s) \sum_a q_\pi(s,a) \nabla \pi(a \mid s, \boldsymbol{\theta})$$
   
   no derivative of the state distribution required.

5. **REINFORCE** samples the theorem: states from on-policy visitation, actions from the policy, $q\_\pi$ from the Monte Carlo return; the $\gamma^t$ weight is the objective's own credit assignment over time.
6. **REINFORCE is unbiased but high-variance** — it works, but slowly and with delicate step sizes.
7. **A baseline** $b(s) = \hat v(s, \mathbf{w})$ subtracts common-mode noise: identical expected gradient, much lower variance, dramatically faster learning.
8. **Actor–critic** bootstraps: replacing $G\_t$ by the one-step return $R\_{t+1} + \gamma \hat v(S\_{t+1}, \mathbf{w})$ collapses the advantage estimate to the TD error $\delta\_t$, and the update becomes fully online and incremental — lower variance, possible bias when $\hat v$ is off.
9. **One axis — how much to bootstrap:** one-step actor–critic, $n$-step / $\lambda$ variants, REINFORCE with baseline, and plain REINFORCE all sit on the same bias–variance dial as in Lecture 7.
10. **AlphaGo** is this lecture at scale: REINFORCE self-play improved its policy, and a learned value network served both as the baseline and as the leaf evaluator inside MCTS.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(Method map)</span></p>

| Method | Update direction | Trade-off |
|---|---|---|
| REINFORCE | $\gamma^t G\_t \nabla \ln \pi(A\_t \mid S\_t, \boldsymbol{\theta})$ | unbiased, high variance |
| REINFORCE with baseline | $\gamma^t (G\_t - \hat v(S\_t, \mathbf{w})) \nabla \ln \pi$ | unbiased, lower variance |
| One-step actor–critic | $\gamma^t \delta\_t \nabla \ln \pi$ with $\delta\_t = R\_{t+1} + \gamma \hat v(S\_{t+1}) - \hat v(S\_t)$ | bootstrapped, online |
| $n$-step / $\lambda$ actor–critic | $\gamma^t (G\_t^{(n)} - \hat v(S\_t)) \nabla \ln \pi$ or trace variants | tunable bias–variance |
| Gaussian PG | Gaussian score function $\nabla \ln \pi$ | continuous control *(next lecture)* |

**Common skeleton.** Every method here is

$$\boldsymbol{\theta} \;\leftarrow\; \boldsymbol{\theta} + \alpha\, \widehat{A}_t\, \nabla \ln \pi(A_t \mid S_t, \boldsymbol{\theta}),$$

where $\widehat{A}\_t$ is some estimate of how much better than average $A\_t$ was in $S\_t$. For discounted episodic start-state objectives, $\widehat{A}\_t$ includes the outer factor $\gamma^t$.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Takeaway</span><span class="math-callout__name">(Final message)</span></p>

Policy gradients exchange *"find the best action for each state"* for *"shape a probability distribution over actions."* The gradient of $\ln \pi$ is where the parameterization enters; the return or advantage term supplies the credit signal.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Idea</span><span class="math-callout__name">(Where this goes next: continuous actions)</span></p>

One thread is left deliberately hanging: nothing in the policy-gradient machinery requires a finite action set. With a **Gaussian policy** — mean and standard deviation as parameterized functions of the state — the same score-function idea handles **continuous action spaces** directly; that is the "Gaussian PG" row of the method map above.

This is the subject of the next lecture.

</div>

## What to read to continue

### Proximal Policy Optimization

* [A Natural Policy Gradient (2001)](https://papers.nips.cc/paper_files/paper/2001/file/4b86abe48d358ecf194c56c69108433e-Paper.pdf)
* [Trust Region Policy Optimization (2015)](https://arxiv.org/abs/1502.05477)
* [Proximal Policy Optimization Algorithms (2017)](https://arxiv.org/abs/1707.06347)


<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Why Action-Independent Baselines Do Not Change the Policy Gradient: The Zero-Mean Geometry of Baselines)</span></p>

Let

$$g(s,a):=\nabla_\theta \ln \pi(a\mid s,\theta)$$

and let

$$\hat v=v_\pi+e$$

be an approximate critic, where $e$ is its approximation error. Define the one-step TD error

$$\hat\delta_t = R_{t+1} + \gamma \hat v(S_{t+1}) - \hat v(S_t).$$

---

Prove the baseline identity

$$\sum_a \pi(a\mid s,\theta)b(s)g(s,a)=0$$

for every function $b(s)$ independent of the sampled action.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

**Any state-only term is orthogonal in expectation to the score.**

For a function $b(s)$ independent of the sampled action,

$$
\begin{aligned}
\sum_a\pi(a\mid s)b(s)g(s,a)
&= b(s)\sum_a \pi(a\mid s)\nabla\log\pi(a\mid s) \\
&=b(s)\sum_a\nabla\pi(a\mid s) \\
&= b(s)\nabla\sum_a\pi(a\mid s) \\
&= b(s)\nabla 1 \\ 
&=0.
\end{aligned}
$$

Therefore

$$\boxed{\sum_a\pi(a\mid s,\theta)b(s)g(s,a)=0.}$$

This is why a state-value baseline can reduce variance without changing the expected policy-gradient direction. 

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(The Exact TD Error as an Unbiased Advantage Estimator: Exact TD Errors Recover the Advantage Signal)</span></p>

Let

$$\delta_t^\star = R_{t+1} + \gamma v_\pi(S_{t+1}) - v_\pi(S_t)$$

use the exact value function. Show that

$$\mathbb E!\left[\delta_t^\star\mid S_t=s,A_t=a\right] = q_\pi(s,a)-v_\pi(s) =:A_\pi(s,a).$$

Deduce that the expected actor direction based on

$$\delta_t^\star g(S_t,A_t)$$

equals the policy-gradient direction with the state-value baseline.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

Let

$$\delta_t^\star= R_{t+1} + \gamma v_\pi(S_{t+1}) - v_\pi(S_t).$$

Conditioning on $S_t=s,A_t=a$,

$$
\begin{aligned}
\mathbb E[\delta_t^\star\mid s,a]
&= \mathbb E[ R_{t+1}+\gamma v_\pi(S_{t+1}) \mid s,a] - v_\pi(s) \\
&= q_\pi(s,a)-v_\pi(s).
\end{aligned}
$$

Thus

$$\boxed{\mathbb E[\delta_t^\star\mid S_t=s,A_t=a] = A_\pi(s,a).}$$

Now consider the conditional expected actor direction:

$$
\begin{aligned}
\mathbb E[ \delta_t^\star g(S_t,A_t) \mid S_t=s]
&= \sum_a \pi(a\mid s) A_\pi(s,a)g(s,a) \\
&= \sum_a A_\pi(s,a)\nabla\pi(a\mid s).
\end{aligned}
$$

Since $A_\pi(s,a)=q_\pi(s,a)-v_\pi(s)$,

$$
\begin{aligned}
\sum_a A_\pi(s,a)\nabla\pi(a\mid s)
&= \sum_aq_\pi(s,a)\nabla\pi(a\mid s) - v_\pi(s)\sum_a\nabla\pi(a\mid s) \\
&= \sum_aq_\pi(s,a)\nabla\pi(a\mid s).
\end{aligned}
$$

Therefore

$$\boxed{ \mathbb E[ \delta_t^\star g(S_t,A_t) \mid S_t=s] = \sum_aq_\pi(s,a)\nabla\pi(a\mid s).}$$

After weighting states by $\eta(s)$, this is exactly the policy-gradient direction with the state-value baseline $v_\pi(s)$.

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">()</span></p>

Show that an approximate critic changes the conditional mean of the TD error to

$$\mathbb E[\hat\delta_t\mid s,a] = A_\pi(s,a) + \gamma \sum_{s'}p(s'\mid s,a)e(s') - e(s).$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

Since $\widehat v=v_\pi+e$,

$$
\begin{aligned}
\widehat\delta_t
&= R_{t+1} + \gamma\widehat v(S_{t+1}) - \widehat v(S_t) \\
&= R_{t+1} + \gamma v_\pi(S_{t+1}) - v_\pi(S_t) + \gamma e(S_{t+1}) - e(S_t) \\
&= \delta_t^\star + \gamma e(S_{t+1}) - e(S_t).
\end{aligned}
$$

Conditioning on $S_t=s,A_t=a$,

$$
\begin{aligned}
\mathbb E[\widehat\delta_t\mid s,a]
&= A_\pi(s,a) + \gamma\mathbb E[e(S_{t+1})\mid s,a] - e(s) \\
&= A_\pi(s,a) + \gamma\sum_{s'}p(s'\mid s,a)e(s') - e(s).
\end{aligned}
$$

Hence

$$\boxed{ \mathbb E[\widehat\delta_t\mid s,a] = A_\pi(s,a) + \gamma\sum_{s'}p(s'\mid s,a)e(s') - e(s).}$$

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">()</span></p>

Derive the action-dependent bias term in the expected actor direction. Show that the term involving $-e(s)$ cancels, but in general the remaining term

$$B(s) = \gamma \sum_a \nabla \pi(a\mid s,\theta) \sum_{s'} p(s'\mid s,a)e(s')$$

need not vanish. Give a sufficient condition under which

$$B(s)=0.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

Let

$$\widehat D(s) = \mathbb E[ \widehat\delta_tg(S_t,A_t) \mid S_t=s].$$

Using Task 3,

$$
\begin{aligned}
\widehat D(s)
&= \sum_a\pi(a\mid s)g(s,a) \left[ A_\pi(s,a) + \gamma\sum_{s'}p(s'\mid s,a)e(s') - e(s) \right].
\end{aligned}
$$

The first term is the exact policy-gradient direction. The term involving $-e(s)$ is

$$-e(s)\sum_a\pi(a\mid s)g(s,a)=0$$

by the baseline identity. Therefore

$$\widehat D(s) = D^\star(s)+B(s),$$

where

$$\boxed{B(s) = \gamma \sum_a \nabla_\theta\pi(a\mid s,\theta) \sum_{s'} p(s'\mid s,a)e(s').}$$

The error $-e(s)$ is harmless because it is the same for all actions at $s$. By contrast, the expected successor error

$$\overline e(s,a) := \sum_{s'}p(s'\mid s,a)e(s')$$

may vary with $a$, and therefore acts like an erroneous action-dependent advantage.

A sufficient condition for the bias to vanish is that $\overline e(s,a)$ be independent of $a$: suppose

$$\sum_{s'}p(s'\mid s,a)e(s')=c(s) \qquad\text{for every }a.$$

Then

$$B(s) =  \gamma c(s)\sum_a\nabla\pi(a\mid s) = 0.$$

Thus

$$\boxed{ B(s)=0 \quad\text{whenever the expected next-state critic error is action-independent.} }$$

In particular, this holds when $e\equiv0$, when $e$ is constant over all possible successor states, or when the next-state distribution is independent of the action. This is the source of the usual actor–critic bias–variance trade-off: bootstrapping lowers variance but an inaccurate critic can introduce bias. 

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">()</span></p>

A critic error at the current state behaves like an ordinary baseline and cancels. Only critic error that predicts different future errors for different actions can rotate the expected actor update away from the true policy gradient.

</div>

---

## Exercise 2 — $\varepsilon$-Greedy MC Control

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">()</span></p>

Let $\pi$ be any $\varepsilon$-soft policy and let $\pi'$ be the $\varepsilon$-greedy policy with respect to $q_\pi$. Prove the $\varepsilon$-soft policy improvement theorem:

$$v_{\pi'}(s)\geq v_\pi(s) \qquad \text{for all }s\in\mathcal S.$$

> **Hint.** Start from
>
> $$v_\pi(s)=\sum_a\pi(a\mid s)q_\pi(s,a)$$
>
> and write
>
> $$\pi(a\mid s) = \left(\pi(a\mid s)-\frac{\varepsilon}{|\mathcal A|}\right) + \frac{\varepsilon}{|\mathcal A|}.$$
>
> Use the $\varepsilon$-soft condition to bound the first term from above.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

Let

$$a^*(s)\in\arg\max_a q_\pi(s,a), \qquad M(s):=\max_a q_\pi(s,a).$$

**Separate the mandatory exploration mass.**

Since $\pi$ is $\varepsilon$-soft, we may write

$$\pi(a\mid s)=\frac{\varepsilon}{m}+\delta(a\mid s),$$

where

$$\delta(a\mid s)\ge 0, \qquad \sum_a\delta(a\mid s)=1-\varepsilon.$$

Therefore,

$$
\begin{aligned}
v_\pi(s)
&=\sum_a\pi(a\mid s)q_\pi(s,a)\\
&=\frac{\varepsilon}{m}\sum_aq_\pi(s,a)
+\sum_a\delta(a\mid s)q_\pi(s,a)\\
&\le
\frac{\varepsilon}{m}\sum_aq_\pi(s,a)
+M(s)\sum_a\delta(a\mid s)\\
&=
\frac{\varepsilon}{m}\sum_aq_\pi(s,a)
+(1-\varepsilon)M(s).
\end{aligned}
$$

But $\pi'$ is $\varepsilon$-greedy with respect to $q_\pi$, so

$$
\begin{aligned}
\sum_a\pi'(a\mid s)q_\pi(s,a)
&=
\left(1-\varepsilon+\frac{\varepsilon}{m}\right)M(s)
+\sum_{a\ne a^*(s)}\frac{\varepsilon}{m}q_\pi(s,a)\\
&=
(1-\varepsilon)M(s)
+\frac{\varepsilon}{m}\sum_aq_\pi(s,a).
\end{aligned}
$$

Consequently,

$$v_\pi(s) \le \sum_a\pi'(a\mid s)q_\pi(s,a) = (T_{\pi'}v_\pi)(s).$$

This is exactly the one-step policy-improvement inequality established in the lecture notes. 

**Iterate the Bellman operator.**

Since $T_{\pi'}$ is monotone,

$$v_\pi \le T_{\pi'}v_\pi \le T_{\pi'}^2v_\pi \le\cdots.$$

Moreover, $T_{\pi'}$ is a $\gamma$-contraction, so

$$T_{\pi'}^nv_\pi\longrightarrow v_{\pi'}.$$

Taking limits gives

$$\boxed{v_{\pi'}(s)\ge v_\pi(s)\qquad\text{for every }s.}$$

Equality in the one-step inequality holds precisely when the extra mass

$$\pi(a\mid s)-\frac{\varepsilon}{m}$$

is supported only on maximizing actions. With a unique maximizing action, this means that $\pi$ is already $\varepsilon$-greedy at $s$.

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">()</span></p>

Suppose $\varepsilon$-greedy policy improvement does not change the policy, i.e. $\pi'=\pi$. Show that in this case

$$v_\pi(s) = \frac{\varepsilon}{|\mathcal A|} \sum_a q_\pi(s,a) + (1-\varepsilon)\max_a q_\pi(s,a) \qquad \text{for all }s\in\mathcal S,$$

and interpret this equation as a fixed-point condition.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

Suppose policy improvement changes nothing:

$$\pi'=\pi.$$

Then $\pi$ itself is $\varepsilon$-greedy with respect to $q_\pi$. Hence

$$
\begin{aligned}
v_\pi(s)
&=\sum_a\pi(a\mid s)q_\pi(s,a)\\
&= \left(1-\varepsilon+\frac{\varepsilon}{m}\right) \max_a q_\pi(s,a) + \sum_{a\ne a^*(s)} \frac{\varepsilon}{m}q_\pi(s,a)\\
&= (1-\varepsilon)\max_aq_\pi(s,a) + \frac{\varepsilon}{m}\sum_aq_\pi(s,a).
\end{aligned}
$$

Thus

$$\boxed{v_\pi(s) = \frac{\varepsilon}{m}\sum_a q_\pi(s,a)+(1-\varepsilon)\max_a q_\pi(s,a).}$$

This is the equation requested on the sheet. 

**Interpret it as a Bellman fixed point.**

For an arbitrary value function $v$, define

$$q_v(s,a) := r(s,a)+\gamma\sum_{s'}p(s'\mid s,a)v(s')$$

and the $\varepsilon$-soft optimality operator

$$(T_\varepsilon v)(s) := \frac{\varepsilon}{m}\sum_a q_v(s,a) + (1-\varepsilon)\max_a q_v(s,a).$$

Since $q_\pi=q_{v_\pi}$, the preceding equation becomes

$$\boxed{v_\pi=T_\varepsilon v_\pi.}$$

The operator $T_\varepsilon$ is a $\gamma$-contraction. Indeed,

$$|T_\varepsilon v-T_\varepsilon w|_\infty \le \gamma|v-w|_\infty.$$

It therefore has a unique fixed point. This fixed point is the optimal value function among all $\varepsilon$-soft policies. Thus, when $\varepsilon$-greedy improvement leaves $\pi$ unchanged, $\pi$ is optimal within the constrained class of $\varepsilon$-soft policies.

It need not be optimal among all policies when $\varepsilon>0$, because it is forced to take non-greedy actions with positive probability.

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">()</span></p>

A sequence of policies $\lbrace \pi_k\rbrace$ is called **GLIE** (*Greedy in the Limit with Infinite Exploration*) if:

1. Every state-action pair is visited infinitely often:

   $$\sum_{k=1}^{\infty} \mathbf 1[(s,a)\text{ visited in episode }k] = \infty \qquad \text{for all }(s,a).$$

2. The policies converge to the greedy policy:

   $$\pi_k(a\mid s) \longrightarrow \mathbf 1\left[ a=\arg\max_{a'}q(s,a') \right] \qquad \text{as }k\to\infty$$

   for all $s$.

Show that $\varepsilon$-greedy with

$$\varepsilon_k=\frac{1}{k}$$

satisfies both GLIE conditions, assuming every state-action pair is visited at least once per episode. 

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

Let

$$\varepsilon_k=\frac1k.$$

The two GLIE requirements are infinite exploration and convergence toward greedy action selection. 

**Greedy in the limit**

Let $a^\ast(s)$ be the consistently chosen greedy action. Then

$$
\pi_k(a\mid s)
=
\begin{cases}
1-\dfrac1k+\dfrac1{km},&a=a^*(s), \\
\dfrac1{km},&a\ne a^*(s).
\end{cases}
$$

Therefore,

$$\pi_k(a^*(s)\mid s) = 1-\frac{1-1/m}{k} \longrightarrow 1,$$

while for every non-greedy action,

$$\pi_k(a\mid s)=\frac1{km}\longrightarrow0.$$

Hence

$$\boxed{\pi_k(a\mid s)\longrightarrow \mathbf 1_{{a=a^*(s)}}.}$$

Thus the policies become greedy in the limit.

**Infinite exploration**

Under the assumption literally stated on the sheet—that every state–action pair is visited at least once per episode—we have

$$\mathbf 1_{{(s,a)\text{ visited in episode }k}}=1$$

for every $k$. Hence

$$\sum_{k=1}^{\infty} = \mathbf 1_{{(s,a)\text{ visited in episode }k}} =  \sum_{k=1}^{\infty}1 = \infty.$$

Thus every pair is visited infinitely often.

Therefore the schedule $\varepsilon_k=1/k$ satisfies both GLIE conditions.

**More standard interpretation**

The assumption on the sheet is stronger than necessary and makes infinite exploration immediate. Usually one assumes only that state $s$ is encountered infinitely often. At the $k$-th opportunity,

$$\mathbb P(A=a\mid S=s)\ge \frac1{km}.$$

Since

$$\sum_{k=1}^{\infty}\frac1{km} = \frac1m\sum_{k=1}^{\infty}\frac1k = \infty,$$

the exploration probabilities are not summable, so each action continues to be selected infinitely often almost surely. The harmonic decay $1/k$ is exactly slow enough to preserve infinite exploration while still tending to zero.

</details>
</div>

---
