---
title: Reinforcement Learning (HD)
layout: default
noindex: true
math: true
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

In full reinforcement learning, an action simultaneously affects three things: the reward we receive now, the state we transition into, and — through that state — *what data we get to see in the future*. These three entanglements make the full problem hard. The multi-armed bandit is the setting obtained by deliberately stripping the last two entanglements away: there are no states to transition between and no delayed credit to assign. The only thing left is the interaction loop

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
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why evaluative feedback forces exploration)</span></p>

**If you never try an action, you never observe its reward, so you never learn its value.**

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

The most natural estimator of $q_\ast(a)$ is the empirical mean of the rewards observed from arm $a$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Sample-Average Action-Value Estimate)</span></p>

Let $N_t(a)$ denote the number of times action $a$ has been selected before time $t$. The **sample-average estimate** of $q_\ast(a)$ is

$$
Q_t(a) \;=\; \frac{\sum_{i=1}^{t-1} R_i \, \mathbf{1}\{A_i = a\}}{\sum_{i=1}^{t-1} \mathbf{1}\{A_i = a\}}
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

* the rewards $\{R_i : A_i = a\}$ sampled from arm $a$ are i.i.d. draws from one fixed distribution, so the LLN applies,
* the limit $q_\ast(a)$ that the sample mean converges to is itself a well-defined constant.

If either of these breaks, sample averages are the wrong estimator.

**Non-stationary, for contrast.** A non-stationary bandit is one where $q_\ast(a)$ is allowed to *drift over time*: the true means themselves change as the agent interacts. Classical examples are slot machines whose underlying probabilities shift, or a restaurant whose cook changes mid-year. In that regime, averaging *all* past rewards equally is actively harmful — ancient rewards from a stale regime pull the estimate just as strongly as fresh ones do. We return to this case in the section on constant step sizes, where the goal is no longer convergence to a fixed number but **tracking** a moving target.

**Why "almost" was avoided.** Unlike LLN statements that speak of almost-sure convergence of arbitrary $(X_i)_{i \ge 1}$, here convergence of $Q_t(a)$ is contingent on the agent *actually sampling arm $a$ infinitely often* — $N_t(a) \to \infty$. A control rule that stops pulling arm $a$ after finitely many steps prevents convergence entirely, even in a stationary problem. The stationarity assumption concerns the environment; the "sample often enough" condition concerns the algorithm.

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

$$
Q_{n+1} \;=\; Q_n + \alpha\,(R_n - Q_n)
\;=\; (1 - \alpha)\, Q_n + \alpha\, R_n.
$$

A constant step size makes recent rewards matter more than old ones.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Derivation</span><span class="math-callout__name">(Exponential Recency Weighting)</span></p>

Unrolling the constant step-size recursion from an initial estimate $Q_1$:

$$
Q_{n+1} \;=\; (1 - \alpha)^n Q_1 + \sum_{i=1}^n \alpha \,(1 - \alpha)^{n - i}\, R_i.
$$

The coefficient of $R_i$ is $\alpha(1 - \alpha)^{n-i}$, which decays geometrically in $n - i$ — the age of the reward.

* Recent rewards receive the largest weights.
* Older rewards are downweighted exponentially fast.
* A constant step size therefore implements an **exponentially recency-weighted average**.

This is the mechanism that lets the estimate forget obsolete data and track the current value.

</div>

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

Every method above shares the same basic recipe: estimate $Q_t(a)$, then act nearly greedily on those estimates. Gradient bandits take a genuinely different route — they parameterize the **policy directly** and improve it by gradient ascent on expected reward. This is the first appearance in the course of the idea of **policy gradients**.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Preferences and the Softmax Policy)</span></p>

Instead of action-value estimates, maintain a real-valued **preference** $H_t(a) \in \mathbb{R}$ for every action. The preferences induce a stochastic **softmax policy**:

$$
\pi_t(a) \;\doteq\; \Pr(A_t = a) \;=\; \frac{e^{H_t(a)}}{\sum_{b=1}^{k} e^{H_t(b)}}.
$$

* Larger $H_t(a) \Rightarrow$ larger $\pi_t(a)$, but every action keeps positive probability.
* The policy is **shift-invariant**: adding the same constant to every $H_t(a)$ leaves $\pi_t$ unchanged. Only *relative* preferences matter.
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

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Derivation</span><span class="math-callout__name">(From the Exact Gradient to an Interpretable Form)</span></p>

Apply the chain rule to $J(H) = \sum_x \pi(x)\, q_\ast(x)$:

$$
\frac{\partial J}{\partial H(a)}
\;=\; \sum_{x} q_*(x)\, \frac{\partial \pi(x)}{\partial H(a)}
\;=\; \sum_{x} q_*(x)\, \pi(x)\bigl(\mathbf{1}\lbrace x = a\rbrace - \pi(a)\bigr).
$$

Expanding and using $\sum_x \pi(x) = 1$ and $\sum_x \pi(x) q_\ast(x) = J(H)$ gives

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

If the reward is **above** the baseline, the chosen action's preference goes up and every other preference goes down; if it is **below**, the chosen action is pushed down and the others up.

</div>

<figure>
  <img src="{{ '/assets/images/notes/rl_hd/baseline_subtraction_trick.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>Baseline: same expected direction, less noise. That typically makes learning smoother and faster.</figcaption>
</figure>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why the baseline helps)</span></p>

Subtracting $\bar R_t$ does **not** change the expected gradient direction — $\mathbb{E}\_{A \sim \pi}[(\mathbf{1}\lbrace A = a\rbrace - \pi(a))] = 0$, so any action-independent baseline cancels in expectation. What it *does* change is the variance of the stochastic estimate. By centering the rewards around a running average, the magnitude of the update stays comparable across problems with very different reward scales, and the learning curves become visibly smoother and faster — which is borne out by the experimental comparison of "with baseline" versus "without baseline" on the testbed.

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

$$
G_t \;\doteq\; R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots \;=\; \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}, \qquad 0 \le \gamma < 1.
$$

The discount factor $\gamma$ controls how much the agent cares about distant rewards: $\gamma$ near $0$ makes it short-sighted, $\gamma$ near $1$ makes it far-sighted.

</div>

<figure>
  <img src="{{ '/assets/images/notes/rl_hd/discount_decay.png' | relative_url }}" alt="Geometric decay of gamma^k for several values of gamma" loading="lazy">
  <figcaption>How heavily a reward $k$ steps in the future contributes to today's return. Smaller $\gamma$ produces a short-sighted agent; $\gamma$ near $1$ produces a far-sighted one.</figcaption>
</figure>

If rewards are uniformly bounded by $\lvert R_{t+k} \rvert \le R_{\max}$ and $\gamma < 1$, then

$$
\lvert G_t \rvert \;\le\; \sum_{k=0}^{\infty} \gamma^k R_{\max} \;=\; \frac{R_{\max}}{1 - \gamma} \;<\; \infty.
$$

So the discounted return is *guaranteed* to be a well-defined finite random variable, and expectations of it are finite. This is the basic reason that the entire MDP machinery does not break on continuing tasks.

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

Fix a policy $\pi$. The **state-value function** measures how good it is to *be in state $s$* and follow $\pi$ thereafter:

$$
v_\pi(s) \;\doteq\; \mathbb{E}_\pi[\,G_t \mid S_t = s\,].
$$

The **action-value function** (or Q-function) measures how good it is to *take action $a$ in state $s$* and follow $\pi$ thereafter:

$$
q_\pi(s, a) \;\doteq\; \mathbb{E}_\pi[\,G_t \mid S_t = s,\, A_t = a\,].
$$

The two are related by averaging $q$ over actions chosen by the policy:

$$
v_\pi(s) \;=\; \sum_{a} \pi(a \mid s)\, q_\pi(s, a).
$$

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

Computing $v_\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s]$ directly would require averaging over the **entire future trajectory**

$$
(A_t,\, S_{t+1},\, R_{t+1},\, A_{t+1},\, S_{t+2},\, R_{t+2},\, \dots),
$$

i.e. summing over an exponentially large branching tree of all paths. This is hopeless for any non-trivial MDP.

**Bellman's idea.** Replace the full path expectation by a **recursive one-step decomposition**: relate $v_\pi(s)$ to the values of states reachable in one step. The exponential blow-up collapses to a fixed-point equation in $\lvert\mathcal{S}\rvert$ unknowns.

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

The same loop then repeats from $S_{t+1}$ onwards, so $G_t$ is a function of the random sequence $(A_t, S_{t+1}, R_{t+1}, A_{t+1}, S_{t+2}, R_{t+2}, \dots)$. Splitting the expectation into "first action, then environment, then *the rest of the future*" is exactly what the Bellman equation does.

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

<figure>
  <img src="{{ '/assets/images/notes/rl_hd/backup_v_pi.png' | relative_url }}" alt="Backup diagram for v_pi: state branches over actions weighted by policy, then over (s', r) pairs weighted by dynamics" loading="lazy">
  <figcaption>Backup diagram for $v_\pi$. Open circles are states, filled circles are state-action pairs. Blue edges are weighted by the policy $\pi(a \mid s)$; green edges by the dynamics $p(s', r \mid s, a)$.</figcaption>
</figure>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Derivation</span><span class="math-callout__name">(Bellman Expectation for $v_\pi$ — three lines)</span></p>

Start from the definition and substitute the recursive identity for $G_t$:

$$
v_\pi(s) \;=\; \mathbb{E}_\pi[G_t \mid S_t = s] \;=\; \mathbb{E}_\pi[R_{t+1} + \gamma G_{t+1} \mid S_t = s].
$$

**Step 1.** Condition on the first action via the law of total expectation. Since $\Pr(A_t = a \mid S_t = s) = \pi(a \mid s)$,

$$
v_\pi(s) \;=\; \sum_{a} \pi(a \mid s)\, \mathbb{E}_\pi[R_{t+1} + \gamma G_{t+1} \mid S_t = s,\, A_t = a].
$$

**Step 2.** Expand the inner expectation over the environment outcome $(S_{t+1}, R_{t+1}) \sim p(\cdot, \cdot \mid s, a)$:

$$
\mathbb{E}_\pi[R_{t+1} + \gamma G_{t+1} \mid s, a] \;=\; \sum_{s', r} p(s', r \mid s, a)\, \bigl[\, r + \gamma\, \mathbb{E}_\pi[G_{t+1} \mid s, a, s', r]\,\bigr].
$$

**Step 3.** Apply the **Markov property**: once $S_{t+1} = s'$ is known, the past $(s, a, r)$ adds no information about the future, so

$$
\mathbb{E}_\pi[G_{t+1} \mid S_t = s,\, A_t = a,\, S_{t+1} = s',\, R_{t+1} = r] \;=\; \mathbb{E}_\pi[G_{t+1} \mid S_{t+1} = s'] \;=\; v_\pi(s').
$$

Combining,

$$
v_\pi(s) \;=\; \sum_{a} \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a)\, \bigl[\, r + \gamma\, v_\pi(s')\,\bigr].
$$

Each of the three lines uses exactly one MDP ingredient: the **policy**, the **dynamics**, and the **Markov property**.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Bellman Expectation Equation for $q_\pi$)</span></p>

For every $(s, a)$,

$$
q_\pi(s, a) \;=\; \sum_{s', r} p(s', r \mid s, a) \Bigl[\, r + \gamma \sum_{a'} \pi(a' \mid s')\, q_\pi(s', a') \,\Bigr].
$$

The order of operations is reversed compared to $v_\pi$: we first commit to taking action $a$ in state $s$, *then* let the environment draw $(s', r)$, *then* let $\pi$ control all subsequent actions.

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

### Optimal Policies and Optimal Values

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Ordering of Policies and Optimality)</span></p>

Compare policies **pointwise** by their state values:

$$
\pi \succeq \pi' \quad\Longleftrightarrow\quad v_\pi(s) \ge v_{\pi'}(s) \quad \forall s \in \mathcal{S}.
$$

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

$$
v_*(s) \;\doteq\; \max_{\pi}\, v_\pi(s).
$$

The **optimal action-value function** is defined analogously:

$$
q_*(s, a) \;\doteq\; \max_{\pi}\, q_\pi(s, a).
$$

The two are related by maximising over actions:

$$
v_*(s) \;=\; \max_{a}\, q_*(s, a).
$$

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

$$
q_*(s, a) \;=\; \sum_{s', r} p(s', r \mid s, a)\Bigl[\, r + \gamma\, \max_{a'}\, q_*(s', a')\,\Bigr].
$$

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

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Greedy Policies from Optimal Values)</span></p>

Given the optimal state-value function $v_*$, define the **greedy policy** by

$$
\pi_{\text{greedy}}(s) \;\in\; \arg\max_{a} \sum_{s', r} p(s', r \mid s, a)\bigl[\, r + \gamma\, v_*(s')\,\bigr].
$$

Then **any policy greedy with respect to $v_\ast$ is optimal.** With $q_\ast$ the rule is even simpler — and no model is required:

$$
\pi^*(s) \;\in\; \arg\max_{a}\, q_*(s, a).
$$

This is the punchline of MDP theory: once you can solve the Bellman optimality equations, optimal control reduces to a *one-step argmax* in every state.

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

$$
v_\pi(s) \;=\; \mathbb{E}_\pi[\, G_t \mid S_t = s \,] \qquad \text{for all } s \in \mathcal{S}.
$$

This is exactly the *prediction* problem from the previous lecture. The Bellman expectation equation supplies the recursive characterisation we need:

$$
v_\pi(s) \;=\; \sum_{a} \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a)\bigl[\, r + \gamma\, v_\pi(s')\,\bigr].
$$

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

$$
v_{k+1}(s) \;=\; \sum_{a} \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a)\bigl[\, r + \gamma\, v_k(s')\,\bigr].
$$

For discounted finite MDPs with $\gamma < 1$, the sequence $\lbrace v_k \rbrace$ converges to the true value $v_\pi$ as $k \to \infty$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why this is called an "expected update")</span></p>

Each update uses an **expectation over all possible successor states and rewards**, weighted by the known dynamics. This is what DP terminology calls an *expected update* — in contrast to:

* a **sample update**, used by Monte Carlo and TD, which replaces the sum $\sum_{s', r} p(s', r \mid s, a) \cdot$ by a single sampled transition,
* a **max update**, used by value iteration, which replaces $\sum_a \pi(a \mid s) \cdot$ by $\max_a$.

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

**Why a tolerance stop works.** The Bellman operator is a $\gamma$-contraction on $\mathbb{R}^{\lvert \mathcal{S}\rvert}$ in the max-norm (proved later), so a sweep with maximum change $\Delta$ guarantees $\lVert V - v_\pi \rVert_\infty \le \Delta / (1 - \gamma)$. Stopping at $\Delta < \theta$ thus controls the worst-case error.

</div>

#### Policy Evaluation as a Linear System

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Vector Form of Policy Evaluation)</span></p>

Order the $n = \lvert \mathcal{S} \rvert$ states and stack the value function into a vector $v \in \mathbb{R}^n$. Define:

* **Expected one-step reward under $\pi$:**

  $$
  r_\pi(s_i) \;=\; \sum_{a} \pi(a \mid s_i) \sum_{s', r} p(s', r \mid s_i, a)\, r,
  $$

  stacked into a vector $r_\pi \in \mathbb{R}^n$.
* **Policy-induced transition matrix:**

  $$
  (P_\pi)_{ij} \;=\; \sum_{a} \pi(a \mid s_i)\, P(s_j \mid s_i, a),
  $$

  the probability of moving from $s_i$ to $s_j$ in one step under $\pi$.

The Bellman expectation equation for $\pi$ then collapses to the linear identity

$$
v_\pi \;=\; r_\pi + \gamma\, P_\pi v_\pi.
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(A fixed policy turns the MDP into a Markov chain)</span></p>

Once $\pi$ is fixed, the action is no longer a decision — it is a random variable with distribution $\pi(\cdot \mid s)$. Marginalising over it leaves only a **state-to-state transition kernel** $P_\pi$, plus a state-indexed expected reward $r_\pi$. This is exactly a (reward-augmented) **Markov chain**: the MDP structure has been collapsed to its dynamics-under-$\pi$. Every later prediction algorithm — TD(0), every-visit MC, $\lambda$-returns — is, in effect, trying to compute $v_\pi$ for this chain.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Bellman Expectation Operator)</span></p>

Define $T_\pi : \mathbb{R}^n \to \mathbb{R}^n$ by

$$
T_\pi v \;\doteq\; r_\pi + \gamma\, P_\pi v.
$$

$T_\pi$ is an **affine map** on value vectors. The iterative-policy-evaluation update is simply

$$
v_{k+1} \;=\; T_\pi v_k,
$$

and the true value function $v_\pi$ is the unique vector satisfying $v_\pi = T_\pi v_\pi$ — the **fixed point** of $T_\pi$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Two equivalent solution strategies)</span></p>

Rearranging the fixed-point equation:

$$
(I - \gamma P_\pi)\, v_\pi \;=\; r_\pi.
$$

For finite discounted MDPs with $\gamma < 1$, the matrix $I - \gamma P_\pi$ is invertible (its spectrum lies in $\lbrace 1 - \gamma \mu : \lvert \mu \rvert \le 1\rbrace$, none of which crosses the origin), so

$$
v_\pi \;=\; (I - \gamma P_\pi)^{-1}\, r_\pi.
$$

**Two viewpoints, one answer:**

* **Linear-system view:** solve $(I - \gamma P_\pi)\, v_\pi = r_\pi$ once.
* **Fixed-point view:** iterate $v_{k+1} = T_\pi v_k$ until convergence.

**Why iteration is preferred in practice.**

* A direct linear solve costs $O(\lvert \mathcal{S} \rvert^3)$, infeasible for large state spaces.
* Transition matrices are typically **huge and very sparse**; iterative sweeps exploit sparsity cheaply.
* Policies change repeatedly during control — re-solving a fresh linear system after each tiny policy change is wasteful.
* Iterative Bellman updates **generalise to model-free RL**, where $P_\pi$ is unknown and $T_\pi v$ must be sample-approximated. The linear-algebra view does not.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Geometric picture)</span></p>

Iterative policy evaluation is a sequence of points $v_0,\, v_1,\, v_2, \dots$ in $\mathbb{R}^{\lvert \mathcal{S} \rvert}$ obtained by repeatedly applying the same affine map $T_\pi$. Each step pulls $v_k$ closer to $v_\pi$:

$$
\lVert v_{k+1} - v_\pi \rVert_\infty \;=\; \lVert T_\pi v_k - T_\pi v_\pi \rVert_\infty \;\le\; \gamma\,\lVert v_k - v_\pi \rVert_\infty.
$$

So convergence is **geometric with rate $\gamma$** — the smaller $\gamma$, the faster the chase. This same contraction estimate underwrites the convergence of every algorithm built on top of expected backups.

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

$$
v_\pi(s) \;=\; \tfrac{1}{4}[-1 + \gamma\, v_\pi(s)] \;+\; \tfrac{1}{4}[-1 + \gamma\, v_\pi(s_R)] \;+\; \tfrac{1}{4}[-1 + \gamma\, v_\pi(s_D)] \;+\; \tfrac{1}{4}[-1 + \gamma \cdot 0].
$$

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
* Equivalent linear-algebra view: solve $(I - \gamma P_\pi)\, v_\pi = r_\pi$.

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

$$
q_\pi(s, a) \;=\; \sum_{s', r} p(s', r \mid s, a)\bigl[\, r + \gamma\, v_\pi(s')\,\bigr].
$$

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

$$
q_\pi(s, \pi'(s)) \;\ge\; v_\pi(s),
$$

then $\pi'$ is at least as good as $\pi$ everywhere:

$$
v_{\pi'}(s) \;\ge\; v_\pi(s) \qquad \text{for all } s \in \mathcal{S}.
$$

If the first inequality is strict in some state, $\pi'$ is strictly better there.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Derivation</span><span class="math-callout__name">(Why local one-step improvement implies global improvement)</span></p>

Unroll the assumption $q_\pi(s, \pi'(s)) \ge v_\pi(s)$ by repeated substitution. Starting from $S_t = s$ and using $\pi'$ to choose the *first* action, then $\pi$ thereafter, then again $\pi'$ for the second action, and so on:

$$
\pi, \pi, \pi, \dots \;\xrightarrow{\text{1 swap}}\; \pi', \pi, \pi, \dots \;\xrightarrow{\text{2 swaps}}\; \pi', \pi', \pi, \dots \;\xrightarrow{\text{}\cdots\text{}}\; \pi', \pi', \pi', \dots
$$

Each swap can only *increase* the value (by assumption). In the limit of swapping every step, the trajectory follows $\pi'$ throughout, and the limit value is $v_{\pi'}(s)$. So $v_{\pi'}(s) \ge v_\pi(s)$.

The "for all $s$" assumption is essential: after the first action, the agent may land in *any* successor state, and we must be sure that using $\pi'$ from there is still safe.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Greedy Policy Improvement)</span></p>

Given the value function $v_\pi$, define the **greedy policy with respect to $v_\pi$** by

$$
\pi'(s) \;\in\; \arg\max_{a}\, q_\pi(s, a) \;=\; \arg\max_{a} \sum_{s', r} p(s', r \mid s, a)\bigl[\, r + \gamma\, v_\pi(s')\,\bigr].
$$

By construction $q_\pi(s, \pi'(s)) = \max_a q_\pi(s, a) \ge q_\pi(s, \pi(s)) = v_\pi(s)$ at every state, so the improvement theorem applies: $\pi'$ is at least as good as $\pi$ everywhere, and strictly better whenever $\pi$ was not already greedy.

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

   $$
   \pi(s) \;\leftarrow\; \arg\max_{a} \sum_{s', r} p(s', r \mid s, a)\bigl[\, r + \gamma\, v_\pi(s')\,\bigr].
   $$

4. If the policy has not changed in any state, stop. Otherwise return to step 2.

In a diagram,

$$
\pi_0 \;\xrightarrow{E}\; v_{\pi_0} \;\xrightarrow{I}\; \pi_1 \;\xrightarrow{E}\; v_{\pi_1} \;\xrightarrow{I}\; \pi_2 \;\xrightarrow{E}\; \cdots \;\xrightarrow{I}\; \pi_\ast.
$$

</div>

<figure>
  <img src="{{ '/assets/images/notes/rl_hd/policy_iteration_cycle.png' | relative_url }}" alt="Zig-zag of policies and value functions alternating evaluation and improvement steps, ending at the optimal policy" loading="lazy">
  <figcaption>Policy iteration alternates two arrows: <strong>E</strong> (evaluation, blue) maps a policy to its value function; <strong>I</strong> (improvement, green) maps a value function to its greedy policy. Each <strong>I</strong>-step is monotone — $v_{\pi_{k+1}} \ge v_{\pi_k}$ pointwise — and the number of deterministic policies is finite, so the chain must terminate at $\pi_\ast$.</figcaption>
</figure>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Why policy iteration converges — and in *finitely* many steps)</span></p>

For a finite discounted MDP:

* Each improvement step yields a policy that is at least as good as the previous one (policy improvement theorem).
* If improvement makes any change, the new policy is **strictly** better in at least one state — so the same policy is never revisited.
* The number of deterministic policies is finite ($\lvert \mathcal{A} \rvert^{\lvert \mathcal{S} \rvert}$ at most).

Therefore the algorithm cannot keep improving forever; it must terminate in finitely many iterations with a policy $\pi$ satisfying $\pi(s) \in \arg\max_a q_\pi(s, a)$ for every $s$. That fixed-point condition is precisely the Bellman optimality equation — so the limit policy is **optimal**.

In practice, policy iteration tends to need surprisingly *few* outer iterations (often single digits even on substantial problems), because each greedy step makes large jumps in policy space.

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

  $$
  v_{k+1}(s) \;=\; \sum_{a} \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a)\bigl[\, r + \gamma\, v_k(s')\,\bigr].
  $$

* **Value iteration backup (max over actions):**

  $$
  v_{k+1}(s) \;=\; \max_{a} \sum_{s', r} p(s', r \mid s, a)\bigl[\, r + \gamma\, v_k(s')\,\bigr].
  $$

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

After convergence, $V \approx v_\ast$, and the **optimal greedy policy** is extracted in one final sweep:

$$
\pi^\ast(s) \;\in\; \arg\max_{a} \sum_{s', r} p(s', r \mid s, a)\bigl[\, r + \gamma\, V(s')\,\bigr].
$$

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

* In **policy iteration** the *intermediate* value functions $v_{\pi_0}, v_{\pi_1}, \dots$ are *true* values of *real* (deterministic) policies. The trace is $\pi_0 \to v_{\pi_0} \to \pi_1 \to v_{\pi_1} \to \cdots$.
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

  $$
  (S_0,\, A_0,\, R_1,\, S_1,\, A_1,\, R_2,\, \dots,\, S_T).
  $$

* **Episodic structure.** Every episode terminates in finite time with probability $1$ (so $T < \infty$ a.s.). This is what makes the return well-defined as a *realised number*, not just an expectation.

* **Return from time $t$:**

  $$
  G_t \;\doteq\; \sum_{k=0}^{T-t-1} \gamma^{k}\, R_{t+k+1}.
  $$

The MC estimate of $v_\pi(s)$ is then literally a sample average of returns observed from visits to $s$:

$$
v_\pi(s) \;=\; \mathbb{E}_\pi[\, G_t \mid S_t = s \,] \;\approx\; \frac{1}{n}\sum_{i=1}^{n} G_i(s).
$$

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

$$
S_0 = s,\quad S_1 = x,\quad S_2 = s,\quad S_3 = \text{terminal}, \qquad R_1 = 2,\; R_2 = 1,\; R_3 = 4.
$$

The state $s$ is visited at times $\mathcal{T}(s) = \lbrace 0, 2 \rbrace$.

* **First-visit MC.** Use only the return from the first visit:

  $$
  G_0 \;=\; R_1 + \gamma R_2 + \gamma^2 R_3 \;=\; 2 + 0.5 \cdot 1 + 0.25 \cdot 4 \;=\; 3.5, \qquad V(s) \leftarrow 3.5.
  $$

* **Every-visit MC.** Average the returns from both visits:

  $$
  G_0 = 3.5, \qquad G_2 = R_3 = 4, \qquad V(s) \leftarrow \frac{3.5 + 4}{2} \;=\; 3.75.
  $$

The numerical disagreement on a single episode is expected; both estimators agree in the limit as the number of episodes goes to infinity.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(First-Visit MC Prediction)</span></p>

Initialise $V(s)$ arbitrarily for all $s \in \mathcal{S}$, and maintain a sample list `Returns(s)` for each state.

Repeat for each episode:

1. Generate an episode following $\pi$: $S_0, A_0, R_1, \dots, S_T$.
2. For each state $s$ that appears in the episode:
   * Let $G$ be the return after the **first** visit to $s$.
   * Append $G$ to `Returns(s)`.
   * $V(s) \leftarrow \text{average}(\texttt{Returns}(s))$.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(In practice — incremental mean)</span></p>

Storing the full list `Returns(s)` is wasteful. Replacing the explicit average by the **incremental mean update**

$$
V(s) \;\leftarrow\; V(s) + \frac{1}{N(s)}\bigl(\, G - V(s) \,\bigr),
$$

where $N(s)$ counts how many returns have been seen for $s$, gives an *exactly equivalent* estimator with constant memory. This formula is also the template all later sample-based RL updates inherit — the only thing that changes is *what is plugged in for $G$* (a sampled return for MC, a bootstrapped target for TD, an importance-weighted return for off-policy MC, $\dots$).

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(One sampled Blackjack episode)</span></p>

A single Blackjack episode might be

$$
(\text{player 13, dealer 10}) \to \textbf{hit}, \quad (\text{player 18, dealer 10}) \to \textbf{stick}, \quad \text{dealer plays} \to \text{player wins.}
$$

The final return is $G = +1$, so this *one* episode contributes to two state estimates:

* the return after the first visit to $(13, 10, \cdot)$ is $G = +1$,
* the return after the first visit to $(18, 10, \cdot)$ is $G = +1$.

The MC update rule is just

$$
V(s) \;\leftarrow\; \text{average of all observed first-visit returns to } s.
$$

After many games this average concentrates around the true $v_\pi(s)$ for each player-sum/dealer-showing state.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Convergence of First-Visit MC)</span></p>

Fix a policy $\pi$. The first-visit MC estimate $V(s)$ converges almost surely to $v_\pi(s)$ as the number of first visits to $s$ goes to infinity.

**Proof idea.**

* Across independently generated episodes, the first-visit returns to $s$ are i.i.d. samples of a random variable with mean $v_\pi(s)$.
* By the strong law of large numbers,

  $$
  \frac{1}{n}\sum_{i=1}^{n} G^{(i)} \xrightarrow{\text{a.s.}} \mathbb{E}_\pi[\, G_t \mid S_t = s \,] \;=\; v_\pi(s).
  $$

* The key point is that "the return after the first visit to $s$" is *exactly* the random quantity whose expectation defines $v_\pi(s)$ — no model, no Bellman fixed point, no operator contraction is needed.

</div>

<figure>
  <img src="{{ '/assets/images/notes/rl_hd/blackjack_mc_value.svg' | relative_url }}" alt="Two pairs of 3D surfaces of estimated Blackjack value functions (with and without a usable ace) after 10,000 and 500,000 sampled episodes, becoming visibly smoother as the sample size grows" loading="lazy">
  <figcaption>MC prediction on Blackjack. The 3D surfaces show estimated state-values for a policy that sticks only on 20 or 21, split by whether the player holds a usable ace. <strong>Left:</strong> after 10,000 episodes the surfaces are jagged; many states have few samples. <strong>Right:</strong> after 500,000 episodes they have settled into smooth functions of (player sum, dealer showing). No model of the dealer's strategy was ever used — only sampled game outcomes.</figcaption>
</figure>

### From Prediction to Control

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Idea</span><span class="math-callout__name">(Why state-values are no longer enough)</span></p>

In DP, policy improvement was a one-step lookahead using the *model*:

$$
\pi'(s) \;=\; \arg\max_{a} \sum_{s', r} p(s', r \mid s, a)\bigl[\, r + \gamma\, v(s')\,\bigr].
$$

The model $p(s', r \mid s, a)$ was the bridge that let us turn a *state*-value function into a *better policy*. **Without $p$, that bridge collapses.** Knowing $v_\pi(s)$ alone tells us nothing about which action to take in $s$, because we cannot evaluate the one-step lookahead any more.

The fix is to estimate **action-values** $q_\pi(s, a)$ instead — and to estimate them in the same MC way we estimated $v_\pi$, by averaging returns.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(MC Action-Value Estimation)</span></p>

For each state-action pair $(s, a)$, define the MC estimate

$$
q_\pi(s, a) \;\doteq\; \mathbb{E}_\pi[\, G_t \mid S_t = s,\, A_t = a \,] \;\approx\; \frac{1}{n}\sum_{i=1}^{n} G_i(s, a),
$$

where each $G_i(s, a)$ is the return following a visit to the *pair* $(s, a)$ in some sampled episode. Once $q_\pi(s, a)$ is available, policy improvement becomes a one-line, *model-free* operation:

$$
\pi'(s) \;=\; \arg\max_{a} q_\pi(s, a).
$$

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

$$
\pi(a \mid s) \;\geq\; \frac{\varepsilon}{\lvert \mathcal{A}(s) \rvert} \qquad \forall s,\, \forall a \in \mathcal{A}(s).
$$

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

$$
q_\pi(s, \pi') \;\doteq\; \sum_{a} \pi'(a \mid s)\, q_\pi(s, a),
$$

the expected $q_\pi$-value obtained by choosing the first action according to $\pi'$ and then following $\pi$. Because $\pi'$ is $\varepsilon$-greedy,

$$
q_\pi(s, \pi') \;=\; (1 - \varepsilon)\, \max_{a} q_\pi(s, a) \;+\; \frac{\varepsilon}{\lvert \mathcal{A}(s) \rvert} \sum_{a} q_\pi(s, a).
$$

So $\pi'$ is a *mixture*: mostly the best action, plus a small uniform-exploration average.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Policy Improvement for $\varepsilon$-Soft Policies — Step 2)</span></p>

Now compare with the value of the *old* policy $\pi$,

$$
v_\pi(s) \;=\; \sum_{a} \pi(a \mid s)\, q_\pi(s, a).
$$

Since $\pi$ is $\varepsilon$-soft, $\pi(a \mid s) \geq \varepsilon / \lvert \mathcal{A}(s) \rvert$ for every action, so we can split

$$
\pi(a \mid s) \;=\; \underbrace{\tfrac{\varepsilon}{\lvert \mathcal{A}(s) \rvert}}_{\text{forced exploration}} \;+\; \underbrace{(1 - \varepsilon)\,\tilde{\pi}(a \mid s)}_{\text{remaining mass}},
$$

where $\tilde{\pi}(\cdot \mid s)$ is another probability distribution. Plugging into $v_\pi(s)$,

$$
v_\pi(s) \;=\; \frac{\varepsilon}{\lvert \mathcal{A}(s) \rvert} \sum_{a} q_\pi(s, a) \;+\; (1 - \varepsilon)\sum_{a} \tilde{\pi}(a \mid s)\, q_\pi(s, a).
$$

The second sum is an average of $q_\pi(s, a)$ under $\tilde{\pi}$, so it cannot exceed $\max_{a} q_\pi(s, a)$. Therefore

$$
v_\pi(s) \;\leq\; \frac{\varepsilon}{\lvert \mathcal{A}(s) \rvert} \sum_{a} q_\pi(s, a) \;+\; (1 - \varepsilon)\, \max_{a} q_\pi(s, a).
$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Policy Improvement for $\varepsilon$-Soft Policies — Step 3)</span></p>

Comparing the two expressions side by side:

$$
q_\pi(s, \pi') \;=\; (1 - \varepsilon)\, \max_{a} q_\pi(s, a) \;+\; \frac{\varepsilon}{\lvert \mathcal{A}(s) \rvert} \sum_{a} q_\pi(s, a),
$$

$$
v_\pi(s) \;\leq\; (1 - \varepsilon)\, \max_{a} q_\pi(s, a) \;+\; \frac{\varepsilon}{\lvert \mathcal{A}(s) \rvert} \sum_{a} q_\pi(s, a).
$$

The right-hand sides are *identical*. Hence

$$
q_\pi(s, \pi') \;\geq\; v_\pi(s) \qquad \forall s.
$$

Choosing the first action according to $\pi'$ and then following $\pi$ has expected return at least as large as following $\pi$ from the start. By the **policy improvement theorem** (DP lecture),

$$
\boxed{\, v_{\pi'}(s) \;\geq\; v_\pi(s) \qquad \forall s. \,}
$$

**Conclusion.** Even with forced exploration, the $\varepsilon$-greedy policy improves — or at least does not worsen — the old $\varepsilon$-soft policy. The argument is robust: the $\varepsilon/\lvert \mathcal{A}(s) \rvert$ "forced exploration" term cancels exactly on both sides, leaving the improvement entirely to the $(1 - \varepsilon)\max_a q_\pi(s, a)$ contribution.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">($\varepsilon$-Greedy improvement on two actions)</span></p>

Take a single state $s$ with two actions and action-values

$$
q_\pi(s, a_1) = 10, \qquad q_\pi(s, a_2) = 0.
$$

Suppose the old $\varepsilon$-soft policy is

$$
\pi(a_1 \mid s) = 0.6, \qquad \pi(a_2 \mid s) = 0.4, \qquad v_\pi(s) = 0.6 \cdot 10 + 0.4 \cdot 0 = 6.
$$

The new **$\varepsilon$-greedy policy** with $\varepsilon = 0.2$ assigns

$$
\pi'(a_1 \mid s) = 0.9, \qquad \pi'(a_2 \mid s) = 0.1, \qquad q_\pi(s, \pi') = 0.9 \cdot 10 + 0.1 \cdot 0 = 9.
$$

So $q_\pi(s, \pi'(s)) = 9 \geq 6 = v_\pi(s)$, confirming the theorem on this single example.

The mechanism is intuitive: **$\varepsilon$-greedy shifts probability mass toward the best action, increasing expected value**, while the small forced exploration mass is the *same* on both sides of the comparison and cancels.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(On-Policy First-Visit MC Control)</span></p>

Initialise action-value estimates $Q(s, a)$ arbitrarily, and pick any initial $\varepsilon$-soft policy $\pi$ so that every action can be sampled.

Repeat for each episode:

1. **Generate** an episode using the current policy $\pi$.
2. **Policy evaluation.** For each first visit to a pair $(s, a)$ in the episode, update

   $$
   Q(s, a) \;\leftarrow\; \text{average of observed returns after } (s, a).
   $$

3. **Policy improvement.** For every visited state $s$, replace the action probabilities by

   $$
   \pi(\cdot \mid s) \;\leftarrow\; \varepsilon\text{-greedy w.r.t. } Q(s, \cdot).
   $$

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

$$
v_\pi(s) \;=\; \mathbb{E}_\pi[\, G_t \mid S_t = s \,],
$$

but the returns we observe come from episodes generated by $\mu$:

$$
G_t^{(1)}, G_t^{(2)}, \dots \;\sim\; \mu, \qquad \text{not } \pi.
$$

**Core difficulty.** We need to compute an expectation under one distribution ($\mathbb{E}\_\pi[\cdot]$) while observing samples from another ($\mathbb{E}\_\mu[\cdot]$). Naively averaging the observed $G_t^{(i)}$ would estimate $v_\mu$, not $v_\pi$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Coverage Assumption)</span></p>

For off-policy learning to be possible at all, the behavior policy must "see" every action the target policy might take:

$$
\pi(a \mid s) > 0 \quad \Longrightarrow \quad \mu(a \mid s) > 0.
$$

That is, if the target policy would ever choose action $a$ in state $s$, the behavior policy must sometimes choose it too. Otherwise there are entire pieces of target-policy experience that *never* appear in the data — and no statistical trick can recover them.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Idea</span><span class="math-callout__name">(Importance sampling — the statistical idea)</span></p>

Suppose we want $\mathbb{E}_\pi[f(X)] = \sum_x \pi(x) f(x)$, but our samples come from a *different* distribution $\mu$. Whenever coverage holds ($\mu(x) > 0$ whenever $\pi(x) > 0$), multiply and divide by $\mu(x)$:

$$
\mathbb{E}_\pi[f(X)] \;=\; \sum_{x} \pi(x) f(x) \;=\; \sum_{x} \mu(x) \frac{\pi(x)}{\mu(x)} f(x) \;=\; \mathbb{E}_\mu\!\left[\, \frac{\pi(X)}{\mu(X)}\, f(X) \,\right].
$$

So samples drawn from $\mu$, after being **reweighted by the ratio $\pi(X)/\mu(X)$**, have the same expectation as if they had come from $\pi$. The ratio

$$
\frac{\pi(X)}{\mu(X)} \;=\; \frac{\text{target probability}}{\text{behavior probability}}
$$

is called the **importance sampling ratio**.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(Importance sampling in RL — trajectories, not single variables)</span></p>

In RL the random variable is not a single $X$ but an entire **trajectory** generated by a policy. So the correction factor is

$$
\frac{\Pr\nolimits_\pi(\text{trajectory})}{\Pr\nolimits_\mu(\text{trajectory})}.
$$

**Intuition.** Trajectories that are more likely under the target policy $\pi$ than under the behavior policy $\mu$ receive **larger weight**; trajectories less typical of $\pi$ receive **smaller weight**. Importance sampling thus *re-emphasises* the parts of $\mu$'s experience that are typical of $\pi$ and *down-weights* the rest.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(The IS Ratio Depends Only on the Policies)</span></p>

For a trajectory $S_t, A_t, S_{t+1}, \dots, S_T$ generated under policy $\pi$ versus under $\mu$,

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

$$
v_\pi(s) \;=\; \mathbb{E}_\pi[\, G_t \mid S_t = s \,]
$$

from episodes generated by $\mu$. Importance sampling rewrites this as an expectation under $\mu$:

$$
v_\pi(s) \;=\; \mathbb{E}_\mu\!\bigl[\, \rho_{t}^{T}\, G_t \mid S_t = s \,\bigr], \qquad \rho_{t}^{T} = \prod_{k=t}^{T-1} \frac{\pi(A_k \mid S_k)}{\mu(A_k \mid S_k)}.
$$

Replacing the expectation by an empirical average over the set $\mathcal{T}(s)$ of sampled visit times to $s$ gives the **ordinary off-policy MC estimator**

$$
V(s) \;\approx\; \frac{1}{\lvert \mathcal{T}(s) \rvert} \sum_{t \in \mathcal{T}(s)} \rho_{t}^{T}\, G_t.
$$

**Takeaway.** Off-policy MC $=$ ordinary MC averaging $+$ importance weights.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Importance sampling on Blackjack)</span></p>

Take

* target policy $\pi$: **stick whenever player sum $\geq 20$** (otherwise hit);
* behavior policy $\mu$: exploratory, picks both actions with non-zero probability everywhere.

Observe the *tail* of an episode starting at player sum 18:

$$
(18, \cdot) : \text{hit} \;\to\; (20, \cdot) : \text{stick}.
$$

The importance weight for this tail is

$$
\rho \;=\; \frac{\pi(\text{hit} \mid 18)}{\mu(\text{hit} \mid 18)} \cdot \frac{\pi(\text{stick} \mid 20)}{\mu(\text{stick} \mid 20)}.
$$

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

$$
\rho \;=\; \prod_{k=1}^{H} X_k.
$$

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

Even when $\mathbb{E}_\mu[\rho] = 1$ (the likelihood ratio is correctly normalised on average), products of likelihood ratios become **very spread out** as $H$ grows. A handful of rare trajectories can therefore *dominate* the MC average. This is the structural reason ordinary IS becomes unreliable on long episodes.

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

  $$
  V(s) \;=\; \frac{1}{\lvert \mathcal{T}(s) \rvert} \sum_{t \in \mathcal{T}(s)} \rho_{t}^{T(t)}\, G_t.
  $$

* **Weighted importance sampling.**

  $$
  V(s) \;=\; \frac{\sum_{t \in \mathcal{T}(s)} \rho_{t}^{T(t)}\, G_t}{\sum_{t \in \mathcal{T}(s)} \rho_{t}^{T(t)}}.
  $$

| Estimator       | Bias                  | Variance                         |
| :-------------- | :-------------------- | :------------------------------- |
| Ordinary IS     | none (unbiased)       | often enormous                   |
| Weighted IS     | finite-sample bias    | dramatically smaller             |

**Reading.** Ordinary IS is unbiased but its variance explodes with episode length (as the toy example just showed). Weighted IS introduces a small finite-sample bias — the normaliser is itself a random variable — but the bias vanishes as $\lvert \mathcal{T}(s) \rvert \to \infty$, and its variance is much better behaved. In practice **weighted IS is almost always preferred**.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Incremental Weighted IS for First-Visit Off-Policy MC Prediction)</span></p>

For each state $s$ maintain two scalars:

$$
V(s) \;\approx\; v_\pi(s), \qquad C(s) \;=\; \text{cumulative sum of importance weights for } s.
$$

For each episode generated by the behaviour policy $\mu$:

1. Compute the returns $G_t$ for all time steps.
2. For each state $s$ that appears in the episode, let

   $$
   t_s = \min\lbrace\, t : S_t = s \,\rbrace
   $$

   be its first visit time.

3. Compute the importance weight from that first visit onward:

   $$
   \rho_{t_s}^{T} \;=\; \prod_{k=t_s}^{T-1} \frac{\pi(A_k \mid S_k)}{\mu(A_k \mid S_k)}.
   $$

4. **Update:**

   $$
   C(s) \;\leftarrow\; C(s) + \rho_{t_s}^{T},
   $$

   $$
   V(s) \;\leftarrow\; V(s) + \frac{\rho_{t_s}^{T}}{C(s)}\bigl(\, G_{t_s} - V(s)\,\bigr).
   $$

**Meaning.** Each state is updated once per episode, using its first observed return and its first-visit importance weight. The update is a *weighted* incremental mean: weights $\rho_{t_s}^{T}$ accumulate in $C(s)$ and form the denominator of an effective sample average.

</div>

### Off-Policy MC Control

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Idea</span><span class="math-callout__name">(From $V$ to $Q$ in the off-policy setting)</span></p>

Off-policy *prediction* estimated $V(s)$. Off-policy *control* must estimate $Q(s, a)$ — because, as in the on-policy case, policy improvement without a model requires comparing actions:

$$
v_\pi(s) = \mathbb{E}_\pi[\, G_t \mid S_t = s \,] \quad\Longrightarrow\quad q_\pi(s, a) = \mathbb{E}_\pi[\, G_t \mid S_t = s,\, A_t = a \,].
$$

Episodes are generated by a soft exploratory behaviour policy $\mu$, while the target policy $\pi$ is improved **greedily**:

$$
Q(s, a) \;\approx\; \frac{\sum_{t \in \mathcal{T}(s, a)} \rho_{t+1}^{T(t)}\, G_t}{\sum_{t \in \mathcal{T}(s, a)} \rho_{t+1}^{T(t)}}, \qquad \rho_{t+1}^{T} = \prod_{k = t+1}^{T-1} \frac{\pi(A_k \mid S_k)}{\mu(A_k \mid S_k)},
$$

$$
\pi(s) \;=\; \arg\max_{a} Q(s, a).
$$

Note that the ratio starts at $k = t+1$, not $k = t$: the action $A_t$ at the visited pair $(s, a)$ is *given* (we are conditioning on it), so only the *subsequent* actions need to be re-weighted. The rest is the same off-policy machinery as before: coverage, importance weights, weighted averaging, greedy improvement.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(A fundamental weakness of off-policy MC control)</span></p>

Off-policy MC control can learn a *greedy* target policy $\pi$ from exploratory data generated by $\mu$, but MC updates only use trajectory tails **consistent with $\pi$**.

If the target policy is **deterministic greedy**, then any non-greedy behaviour action has zero probability under $\pi$,

$$
A_t \neq \pi(S_t) \quad\Longrightarrow\quad \pi(A_t \mid S_t) = 0,
$$

and the importance weight becomes

$$
\rho \;=\; \prod_{k} \frac{\pi(A_k \mid S_k)}{\mu(A_k \mid S_k)} \;=\; 0.
$$

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
  <p class="math-callout__title"><span class="math-callout__label">Setup</span><span class="math-callout__name">(Sampling experience, bootstrapping predictions)</span></p>

The previous lecture left us with a clean but expensive idea: estimate values by averaging **complete returns** $G\_t$, which forces us to wait until the episode ends. **Temporal-difference (TD) learning** is the hybrid that removes that wait. It is *the* central idea of model-free RL, and almost every algorithm in the rest of the course is best read as "define a TD error, then take a step on it."

The lecture builds up in the following order:

1. **The central idea.** TD = *sampling* (like Monte Carlo) + *bootstrapping* (like Dynamic Programming).
2. **Prediction.** The TD(0) update, the TD error, and where the TD target comes from (the Bellman expectation equation).
3. **Why bootstrap?** Computational advantages over DP and MC, soundness (convergence), and what TD converges to (certainty equivalence).
4. **Control.** Moving from $v\_\pi$ to $q\_\pi$: **Sarsa** (on-policy), **Q-learning** (off-policy), and **Double Q-learning** (bias correction).

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

All three methods estimate the *same object* — the value $v\_\pi(s) = \mathbb{E}\_\pi[\, G\_t \mid S\_t = s\,]$. They differ only in **how the expectation is approximated**.

| Method | Source of estimate | Backup type |
| :----- | :----------------- | :---------- |
| **DP** | model $+$ bootstrap | full expectation backup |
| **MC** | samples, no bootstrap | complete return |
| **TD** | samples $+$ bootstrap | one-step sample backup |

DP needs a model and averages over *all* successors; MC needs samples but no model and uses the *whole* trajectory; TD needs only samples and replaces both the full expectation (by one sample) and the unknown true value (by the current estimate).

</div>

### Prediction: TD(0) vs MC

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Setup</span><span class="math-callout__name">(The prediction problem)</span></p>

Fix a policy $\pi$ and generate experience under it,

$$
S_0, A_0, R_1, S_1, A_1, R_2, S_2, \dots,
$$

with the goal of estimating $v\_\pi(s)$ for every nonterminal $s$. Both methods are instances of the same incremental-mean template $V(S\_t) \leftarrow V(S\_t) + \alpha\,[\,\text{target} - V(S\_t)\,]$; they differ only in the **target**:

$$
\underbrace{V(S_t) \;\leftarrow\; V(S_t) + \alpha\bigl[\,G_t - V(S_t)\,\bigr]}_{\textbf{constant-}\alpha\text{ Monte Carlo, target } = \text{ full return}},
$$

$$
\underbrace{V(S_t) \;\leftarrow\; V(S_t) + \alpha\bigl[\,R_{t+1} + \gamma V(S_{t+1}) - V(S_t)\,\bigr]}_{\textbf{TD(0)},\; \text{target } = R_{t+1} + \gamma V(S_{t+1})}.
$$

MC plugs in the realised return $G\_t$ (known only at episode end); TD(0) plugs in the **TD target** $R\_{t+1} + \gamma V(S\_{t+1})$ (known one step later). This is the single substitution that turns a batch, episodic method into an online, incremental one.

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

$$
\delta_t \;\doteq\; R_{t+1} + \gamma V(S_{t+1}) - V(S_t).
$$

With it, the TD(0) update is simply

$$
V(S_t) \;\leftarrow\; V(S_t) + \alpha\,\delta_t.
$$

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

### Where the TD Target Comes From

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Derivation</span><span class="math-callout__name">(The TD target is the Bellman expectation equation)</span></p>

The TD target is not an ad-hoc guess; it falls straight out of the definition of value. Start from

$$
v_\pi(s) = \mathbb{E}_\pi[\, G_t \mid S_t = s\,],
$$

and use the one-step recursion of the return,

$$
G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} = R_{t+1} + \gamma \sum_{k=0}^{\infty} \gamma^k R_{t+k+2} = R_{t+1} + \gamma\, G_{t+1}.
$$

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

Starting from the exact identity $v\_\pi(s) = \mathbb{E}\_\pi[\,R\_{t+1} + \gamma v\_\pi(S\_{t+1}) \mid S\_t = s\,]$:

1. **Replace the expectation by a single sample** — use the one transition $(R\_{t+1}, S\_{t+1})$ that actually occurred instead of averaging over all of them (this is the *sampling* part, shared with MC).
2. **Replace the unknown $v\_\pi$ by the current estimate $V$** — we do not know the true value of the successor, so we bootstrap from our running guess (this is the *bootstrapping* part, shared with DP).

$$
\underbrace{R_{t+1} + \gamma\, v_\pi(S_{t+1})}_{\text{Bellman target (exact)}} \;\rightsquigarrow\; \underbrace{R_{t+1} + \gamma\, V(S_{t+1})}_{\text{TD(0) target (sampled + bootstrapped)}}.
$$

This makes precise where each method sits relative to the same Bellman equation:

* **MC** samples the *whole* return $G\_t$ — no bootstrap.
* **DP** computes the expectation *exactly* via the model — but bootstraps with $V$.
* **TD** samples $(R\_{t+1}, S\_{t+1})$ *and* bootstraps with $V(S\_{t+1})$ — it does both approximations at once.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Tabular TD(0) for estimating $v\_\pi$)</span></p>

**Input:** policy $\pi$ to be evaluated. **Parameter:** step size $\alpha \in (0, 1]$.

Initialise $V(s)$ arbitrarily for all $s \in \mathcal{S}^+$, with $V(\text{terminal}) = 0$.

Loop for each episode:

1. Initialise $S$.
2. Loop for each step of the episode, until $S$ is terminal:
   * $A \leftarrow$ action given by $\pi$ for $S$.
   * Take action $A$; observe reward $R$ and next state $S'$.
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

Both methods apply $V(s) \leftarrow V(s) + \alpha\,(\text{target} - V(s))$, but with different targets:

$$
\text{MC target} = \underbrace{G_t}_{\text{total time } - \text{ elapsed}}, \qquad \text{TD target} = \underbrace{R_{t+1}}_{\text{next elapsed } - \text{ current elapsed}} + V(S_{t+1}).
$$

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

MC must wait for the **end of the episode** to know $G\_t$; TD updates after a *single* transition using $(S\_t, R\_{t+1}, S\_{t+1})$. This matters because:

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

So bootstrapping is a **convergent** prediction method. The proof is technical (it belongs to stochastic-approximation theory), but the takeaway is simple: learning a guess from a guess is *sound*.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Who learns faster?)</span></p>

Both MC and TD converge asymptotically to $v\_\pi$ for a fixed policy, and **no general theorem says one is always faster**. Even *defining* "faster" rigorously is subtle — it tangles together bias, variance, step size, and non-stationarity. But **empirically**, on stochastic problems, TD often reduces error faster than constant-$\alpha$ MC. The intuition: the TD target has *lower variance* (it depends on one random transition, not a whole random trajectory), at the cost of some *bias* (it depends on the current estimate $V$).

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

$$
v(A) = \tfrac{1}{6}, \quad v(B) = \tfrac{2}{6}, \quad v(C) = \tfrac{3}{6}, \quad v(D) = \tfrac{4}{6}, \quad v(E) = \tfrac{5}{6}.
$$

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

So far updates are **online**: each increment is applied immediately, so $V$ changes between transitions. To compare MC and TD *cleanly* — independent of step-size and update-order artefacts — consider **batch** updating instead:

* freeze $V$;
* walk through every transition $(S\_t, R\_{t+1}, S\_{t+1})$ in a fixed dataset $\mathcal{D}$, accumulating $\Delta\_t = \alpha\,[\,\text{target}\_t - V(S\_t)\,]$;
* apply the *summed* increment once, then repeat the sweep to convergence.

The only difference between the two methods is again the target: **batch MC** uses $G\_t$; **batch TD(0)** uses $R\_{t+1} + \gamma V(S\_{t+1})$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Batch MC and batch TD have different fixed points)</span></p>

For small enough $\alpha$, batch TD(0) and batch MC each converge to a **unique fixed point**, independent of $\alpha$. But on the *same* dataset these are **different** fixed points.

The reason is the per-step target: batch MC drives $V$ toward $G\_t$ (the full sample return), while batch TD(0) drives $V$ toward $R\_{t+1} + \gamma V(S\_{t+1})$ (the one-step Bellman backup). Identical data, different notion of "best fit."

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(What should $V(A)$ be?)</span></p>

Eight observed episodes ($\gamma = 1$):

$$
A, 0, B, 0 \quad\mid\quad B, 1 \quad\mid\quad B, 1 \quad\mid\quad B, 1 \quad\mid\quad B, 1 \quad\mid\quad B, 1 \quad\mid\quad B, 1 \quad\mid\quad B, 0.
$$

Everyone agrees on $B$: it was visited 8 times and the return was $1$ in 6 of them, so

$$
V(B) = \tfrac{6}{8} = \tfrac{3}{4}.
$$

But what is $V(A)$? State $A$ was visited exactly once, in the episode $A, 0, B, 0$. There are **two reasonable answers**.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(Two reasonable answers for $V(A)$)</span></p>

* **Batch MC.** Asks *what returns were observed after visiting $A$?* Only one return: $G = 0$. So batch MC drives $V(A) \to 0$. This **minimises mean-square error on the observed data** — it fits the past perfectly (zero training error).
* **Batch TD.** Asks *what one-step transition was observed from $A$?* Always $A \xrightarrow{0} B$. So batch TD enforces $V(A) = 0 + V(B) = \tfrac{3}{4}$. This is the answer an **exact solution on the inferred Markov model** would give.

Which is "right"? MC fits the past data perfectly; TD generalises via the **Markov structure** of the data. On a Markov process, TD's answer is usually the better predictor of *future* returns — which is what we actually care about.

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

$$
\hat{p}(i \to j) = \text{fraction of observed transitions from } i \text{ that go to } j, \qquad \hat{r}(i \to j) = \text{average reward on those transitions.}
$$

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

Control still follows **Generalised Policy Iteration**: evaluate the current policy, improve it toward greediness, repeat. With TD, the twist is that **evaluation and improvement are interleaved at the finest grain** — updates happen step-by-step *within* each episode, and the policy changes as soon as the value table changes.

As in model-free MC control, we cannot improve a policy from $V$ alone without a model, so we learn **action values** $Q(s, a)$ instead. Everything that follows is "TD(0), but on $Q$."

</div>

### Sarsa: On-Policy TD Control

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Sarsa update)</span></p>

Treat the trajectory as alternating state–action pairs $S\_t, A\_t, R\_{t+1}, S\_{t+1}, A\_{t+1}, \dots$ and apply the TD(0) idea to each pair as though it were one enlarged "state":

$$
Q(S_t, A_t) \;\leftarrow\; Q(S_t, A_t) + \alpha\Bigl[\, \underbrace{R_{t+1} + \gamma\, Q(S_{t+1}, A_{t+1})}_{\text{TD target}} - Q(S_t, A_t) \,\Bigr].
$$

The update uses the quintuple $(S\_t, A\_t, R\_{t+1}, S\_{t+1}, A\_{t+1})$ — which spells the name **Sarsa**. Crucially, $A\_{t+1}$ is the action *actually taken* under the current policy (this is what makes Sarsa **on-policy**). If $S\_{t+1}$ is terminal, the TD target is just $R\_{t+1}$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Sarsa — on-policy TD control)</span></p>

**Parameters:** step size $\alpha \in (0, 1]$, exploration $\varepsilon > 0$.

Initialise $Q(s, a)$ arbitrarily, with $Q(\text{terminal}, \cdot) = 0$.

Loop for each episode:

1. Initialise $S$; choose $A$ from $S$ using an $\varepsilon$-greedy policy derived from $Q$.
2. Loop for each step of the episode:
   * Take action $A$; observe $R, S'$.
   * **If $S'$ is terminal:** $Q(S, A) \leftarrow Q(S, A) + \alpha\,[\,R - Q(S, A)\,]$, then break.
   * **Else:** choose $A'$ from $S'$ using the $\varepsilon$-greedy policy; update
     $$
     Q(S, A) \leftarrow Q(S, A) + \alpha\bigl[\,R + \gamma Q(S', A') - Q(S, A)\,\bigr];
     $$
     then $S \leftarrow S'$, $A \leftarrow A'$.

Each TD update *evaluates* the current policy a little, and each $\varepsilon$-greedy action choice *improves* the behaviour a little — GPI at the granularity of a single transition.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Sarsa convergence — GLIE)</span></p>

Convergence to the optimal action-value function needs **two simultaneous conditions**:

* **exploration:** every $(s, a)$ pair is visited infinitely often;
* **greedy in the limit:** the policy becomes greedy with respect to $Q$ as $t \to \infty$.

Any schedule satisfying both is called **GLIE** (*Greedy in the Limit with Infinite Exploration*); a concrete recipe is $\varepsilon$-greedy with $\varepsilon\_t = 1/t$. Under GLIE *and* the standard step-size conditions, tabular Sarsa converges to the optimal action-value function $q\_\ast$ and an optimal policy.

</div>

### Q-Learning: Off-Policy TD Control

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Idea</span><span class="math-callout__name">(From Sarsa to Q-learning)</span></p>

Compare the two targets:

$$
\text{Sarsa: } R_{t+1} + \gamma\, Q(S_{t+1}, \underbrace{A_{t+1}}_{\text{action taken}}), \qquad \text{Q-learning: } R_{t+1} + \gamma \max_a Q(S_{t+1}, a).
$$

Sarsa learns the value of the **policy it actually follows**. Q-learning learns the value of the **greedy policy** — regardless of which action it actually takes next. This is what makes Q-learning **off-policy**:

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

$$
Q(S_t, A_t) \;\leftarrow\; Q(S_t, A_t) + \alpha\Bigl[\, \underbrace{R_{t+1} + \gamma \max_a Q(S_{t+1}, a)}_{\text{TD target}} - Q(S_t, A_t) \,\Bigr].
$$

It uses the quadruple $(S\_t, A\_t, R\_{t+1}, S\_{t+1})$ — there is **no $A\_{t+1}$**. The term $\max\_a Q(S\_{t+1}, a)$ is the best possible value at the next state; if $S\_{t+1}$ is terminal, $\max\_a Q(S\_{t+1}, a) \doteq 0$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Q-learning — off-policy TD control)</span></p>

**Parameters:** step size $\alpha \in (0, 1]$, exploration $\varepsilon > 0$. Initialise $Q(s, a)$ arbitrarily, with $Q(\text{terminal}, \cdot) = 0$.

Loop for each episode:

1. Initialise $S$.
2. Loop for each step of the episode, until $S$ is terminal:
   * Choose $A$ from $S$ using a policy derived from $Q$ (e.g. $\varepsilon$-greedy).
   * Take action $A$; observe $R, S'$.
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

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Cliff walking — on-policy vs off-policy)</span></p>

Q-learning learns the values of the *greedy* (optimal) path that runs right along the cliff edge. But its $\varepsilon$-greedy *behaviour* occasionally slips off the cliff, so its **online return is poor**. Sarsa learns the value of the $\varepsilon$-greedy policy *itself* — which includes the chance of falling — so it prefers the **safer detour** one row up, and earns a **higher online return**.

This crystallises the distinction:

* **On-policy** methods (Sarsa) evaluate *what they actually do*, exploration included.
* **Off-policy** methods (Q-learning) evaluate *what they would do greedily*, ignoring exploration.

Online performance reflects the former — which is why Sarsa looks better *during learning* even though Q-learning has learned the truly optimal path.

</div>

### Maximization Bias and Double Q-Learning

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(A subtle problem in Q-learning)</span></p>

The Q-learning target $R\_{t+1} + \gamma \max\_a Q(S\_{t+1}, a)$ uses the *same* estimates $Q$ to do two jobs at once: **select** the action with the largest estimate, and **evaluate** that selected action. This double-duty introduces **maximization bias** — even if every individual estimate is unbiased, the maximum is biased *upward*:

$$
\mathbb{E}\Bigl[\max_a Q(s, a)\Bigr] \;\ge\; \max_a \mathbb{E}\bigl[Q(s, a)\bigr].
$$

In particular, if $\mathbb{E}[Q(s, a)] = q(s, a)$, then $\mathbb{E}[\max\_a Q(s, a)] \ge \max\_a q(s, a)$ — Q-learning is systematically optimistic about the value of the best action.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Derivation</span><span class="math-callout__name">(Why the max overestimates)</span></p>

Let $b^\ast \in \arg\max\_b \mathbb{E}[Q(s, b)]$. Since the maximum is at least as large as any particular action value,

$$
\max_a Q(s, a) \;\ge\; Q(s, b^\ast).
$$

Taking expectations on both sides,

$$
\mathbb{E}\Bigl[\max_a Q(s, a)\Bigr] \;\ge\; \mathbb{E}[Q(s, b^\ast)].
$$

By the definition of $b^\ast$, $\mathbb{E}[Q(s, b^\ast)] = \max\_b \mathbb{E}[Q(s, b)]$, hence

$$
\mathbb{E}\Bigl[\max_a Q(s, a)\Bigr] \;\ge\; \max_b \mathbb{E}[Q(s, b)].
$$

So taking a maximum over *noisy* estimates injects optimism — the inequality is strict whenever the estimates have genuine variance and are not all maximised at the same action.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Intuition — the max picks positive errors)</span></p>

Suppose two actions are *truly equally good*, $q(s, a\_1) = q(s, a\_2) = 0$, but our estimates are noisy,

$$
Q(s, a_1) = q(s, a_1) + \text{error}_1, \qquad Q(s, a_2) = q(s, a_2) + \text{error}_2,
$$

with $\mathbb{E}[\text{error}\_1] = \mathbb{E}[\text{error}\_2] = 0$. Even so, $\max\lbrace Q(s, a\_1), Q(s, a\_2)\rbrace$ tends to select whichever estimate happens to have the **more positive error**.

The key realisation: each estimate may be unbiased, but the *selected* one is **not random** — it is, by construction, the one that currently looks best. In Q-learning the same table $Q$ both selects $A^\ast = \arg\max\_a Q(S\_{t+1}, a)$ and evaluates $Q(S\_{t+1}, A^\ast)$, so the positive selection error is preserved straight into the TD target.

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

$$
A^\ast = \arg\max_a Q_1(s, a), \qquad \text{value} = Q_2(s, A^\ast).
$$

Because the noise in $Q\_1$ (which made the selection) is **independent** of the noise in $Q\_2$ (which does the evaluation), the upward bias is removed in expectation. Swapping the roles symmetrically yields a second unbiased estimate.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Double Q-learning)</span></p>

**Parameters:** step size $\alpha \in (0, 1]$, exploration $\varepsilon > 0$. Initialise $Q\_1(s, a)$ and $Q\_2(s, a)$ arbitrarily, with $Q\_1(\text{terminal}, \cdot) = Q\_2(\text{terminal}, \cdot) = 0$.

Loop for each episode, and for each step:

1. Choose $A$ from $S$ using an $\varepsilon$-greedy policy derived from $Q\_1 + Q\_2$; take $A$, observe $R, S'$.
2. With probability $\tfrac{1}{2}$, **update $Q\_1$** (select with $Q\_1$, evaluate with $Q\_2$):
   $$
   Q_1(S, A) \leftarrow Q_1(S, A) + \alpha\Bigl[\,R + \gamma\, Q_2\bigl(S', \arg\max_a Q_1(S', a)\bigr) - Q_1(S, A)\,\Bigr].
   $$
3. **Otherwise update $Q\_2$** symmetrically (select with $Q\_2$, evaluate with $Q\_1$):
   $$
   Q_2(S, A) \leftarrow Q_2(S, A) + \alpha\Bigl[\,R + \gamma\, Q_1\bigl(S', \arg\max_a Q_2(S', a)\bigr) - Q_2(S, A)\,\Bigr].
   $$
4. $S \leftarrow S'$ (and at a terminal $S'$ the target is just $R$).

The behaviour policy is $\varepsilon$-greedy with respect to $Q\_1 + Q\_2$. The cost is a **$\times 2$ memory** footprint; the per-step computation is the same as ordinary Q-learning.

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
* **The TD error is the central object.** $\delta\_t = R\_{t+1} + \gamma V(S\_{t+1}) - V(S\_t)$; nearly every later algorithm is "define a TD error, then step on it."
* **Why TD is useful:** no model needed (vs DP), online and incremental (vs MC), often more data-efficient, and batch TD equals **certainty equivalence** on the empirical Markov model.
* **Control via action values.** **Sarsa** (on-policy) learns the value of the behaviour it follows, exploration included; **Q-learning** (off-policy) learns the value of the greedy target regardless of behaviour; **Double Q-learning** decouples selection from evaluation to remove maximization bias.
* **Final message.** TD methods are not merely RL algorithms — they are general-purpose tools for *learning long-term predictions from other predictions* in dynamical systems.

**Bridge ahead.** TD(0) is the shallowest possible bootstrap (one step) and MC the deepest (full return). The next step unifies them along the depth axis with **$n$-step TD** and **TD($\lambda$)**, and along the width axis with **function approximation**, carrying the TD-error machinery developed here into the function-approximation and deep-RL settings.

</div>
