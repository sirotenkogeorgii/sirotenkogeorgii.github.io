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

In full reinforcement learning, an action simultaneously affects three things: the reward we receive now, the state we transition into, and — through that state — *what data we get to see in the future*. These three entanglements make the full problem hard. The multi-armed bandit is the setting obtained by deliberately stripping the last two entanglements away: there are no states to transition between and no delayed credit to assign. The only thing left is the interaction loop

$$
\text{choose action } A_t \;\longrightarrow\; \text{observe reward } R_t,
$$

and yet even here, one essential RL difficulty refuses to disappear: the tension between **exploration** and **exploitation**. Because of this, bandits are the cleanest laboratory for isolating the central learning problem of RL — how to act in a world that only tells you how good your choices were, never what the right choice would have been.

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

If you never try an action, you never observe its reward, so you never learn its value. Under evaluative feedback this is not a minor inconvenience — it is a hard wall. Unlike supervised learning, where a teacher can reveal the right label for an input the model has never actively chosen, an RL agent learns *only* about actions it samples. Exploration is therefore not optional: it is the only channel through which information about unused actions reaches the agent.

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
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Estimation is rarely the bottleneck)</span></p>

Sample averages are asymptotically exact, so *given enough pulls* they recover the true value of any arm. The bottleneck is not the quality of the estimator — it is the quality of the control policy. We need a rule that *chooses* actions in a way that gathers enough informative data about the right arms. The rest of this lecture is essentially a taxonomy of such rules.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Greedy Action and Pure-Greedy Selection)</span></p>

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

$\varepsilon$-greedy is the minimal fix for pure greedy. It guarantees that every action continues to be sampled with probability at least $\varepsilon / k$ at every time step. Combined with the LLN, this means $Q_t(a) \to q_\ast(a)$ for *every* arm $a$ — so in the long run the greedy argmax identifies the true best arm. The price is that an $\varepsilon$ fraction of steps are "wasted" on uniformly random exploration even after the problem is well-understood.

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
* **Within a run:** rewards are noisy samples $R_t \sim \mathcal{N}(q_*(A_t), 1)$.

This testbed contains *two sources of randomness* that a learning curve has to average over:

* **Across-run randomness:** the problem itself is different on each run.
* **Within-run randomness:** rewards are noisy even if the underlying means were known.

</div>

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

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Question</span><span class="math-callout__name">(A thought experiment)</span></p>

If you have already computed the average of $99$ numbers and now receive a $100$th, do you really need to re-sum all $100$ of them to update the average?

</div>

The answer is no, and the observation behind this is the single most reused pattern in RL.

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

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why sample averages fail when the world drifts)</span></p>

The sample average weights every observed reward *equally*. This is ideal when the true value $q_\ast(a)$ is a fixed constant — the LLN then tells us $Q_t(a) \to q_\ast(a)$. But if $q_\ast(a)$ *drifts over time*, equal weighting is a disaster: ancient rewards from a stale regime drag the estimate away from the current truth just as hard as fresh rewards pull it toward it. In a non-stationary bandit, convergence to a fixed number is the *wrong* goal. What we want is **tracking** — an estimate that follows the moving target.

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

The limitation is structural: the optimism budget is **spent once**. Once every arm has been tried and its estimate has dropped to its true value, the artificially high initialization no longer drives exploration. If the environment later changes, the original optimism does not magically return — this method is essentially a one-time mechanism and is ill-suited to non-stationary problems.

</div>

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

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span><span class="math-callout__name">(How the UCB bonus evolves)</span></p>

The bonus has two competing dynamics:

* Whenever action $a$ is selected, $N_t(a)$ increases — so the bonus for $a$ **shrinks**.
* Whenever $a$ is *not* selected, $t$ still increases, $\ln t$ grows, and the bonus for $a$ **slowly rises**.

The net effect is that UCB revisits uncertain actions again and again, but with decreasing frequency, concentrating progressively on the arms that turn out to be genuinely good. In this sense UCB explores **directedly**: it prefers actions that are simultaneously *plausible* (high $Q_t$) and *insufficiently tested* (low $N_t$).

</div>

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

### Gradient Bandits

Every method above shares the same basic recipe: estimate $Q_t(a)$, then act nearly greedily on those estimates. Gradient bandits take a genuinely different route — they parameterize the **policy directly** and improve it by gradient ascent on expected reward. This is the first appearance in the course of the idea of **policy gradients**.

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
