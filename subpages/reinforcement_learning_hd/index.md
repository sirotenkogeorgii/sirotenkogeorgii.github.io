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
