---
title: "Cheatsheet — Lecture 8: Planning & Learning (Tabular)"
layout: default
noindex: true
math: true
tags:
  - reinforcement-learning
  - planning
  - dyna-q
  - prioritized-sweeping
  - monte-carlo-tree-search
  - cheatsheet
---

# Cheatsheet — Lecture 8: Planning & Learning (Tabular)

*Exam recap: formulas and key results only. Full explanations live in the [main notes]({{ '/subpages/reinforcement_learning_hd/' | relative_url }}#lecture-8-planning-and-learning-with-tabular-methods).*

## The unifying idea

* **Planning and learning use the *same* backups** — they differ only in the *source* of experience: **real** (learning) vs **simulated from a model** (planning).
* **Planning** = any computation that takes a **model** as input and improves a policy/value function.
* **Model** types:
  * **Distribution model** — full $p(s',r\mid s,a)$ (all outcomes + probabilities) → enables **expected** backups.
  * **Sample model / simulator** — returns *one* sampled $(s',r)$ → enables **sample** backups only.

## Expected vs sample backups

$$\text{expected: } \sum_{s',r}p(s',r\mid s,a)[\cdots] \qquad\text{vs}\qquad \text{sample: one drawn }(s',r).$$

* **Expected** = exact, no sampling error, cost $\propto$ branching factor $b$ (needs distribution model).
* **Sample** = cheap ($O(1)$ per backup, needs only a simulator/real experience); with large $b$, many sample updates make more progress per unit compute.
* Both target the **same Bellman fixed point**.

## Dyna: learning + planning + acting

Each real step does three things with **one shared update rule**:

1. **Direct RL** — Q-learning backup on the *real* transition.
2. **Model learning** — store $\hat M(S,A)=(R,S')$ (tabular = remembered transitions).
3. **Planning** — $n$ backups on *simulated* transitions sampled from $\hat M$.

**Tabular Dyna-Q update** (identical for real and simulated transitions):

$$\boxed{\,Q(S,A) \leftarrow Q(S,A) + \alpha\bigl[\,R + \gamma\max_a Q(S',a) - Q(S,A)\,\bigr]\,}$$

* $n$ planning steps ⇒ each real transition supports **many** backups ⇒ value propagates in far fewer real interactions.
* Planning here is **replay**: the model is *replay memory with structure*, not a generator of unseen dynamics — its value is **propagating later-learned values through earlier stored transitions**.

## When the model is wrong

| Failure | Cause | Consequence | Cure |
| :-- | :-- | :-- | :-- |
| **Blocking maze** | old path closes → model too **optimistic** | keeps planning a blocked route | real experience corrects it; **Dyna-Q+** adapts *faster* |
| **Shortcut maze** | new path opens → model too **pessimistic** | may **never** discover the shortcut | **Dyna-Q+** rechecks neglected actions |

* **Lesson:** planning *amplifies* the model — a stale model suppresses the exploration needed to repair it ("confident precisely where it's wrong").

## Dyna-Q+ (exploration as model maintenance)

Inside *planning*, a pair untried for $\tau(s,a)$ steps gets a bonus reward:

$$\boxed{\,r^{+} = r + \kappa\sqrt{\tau(s,a)}\,}\qquad (\kappa>0 \text{ small})$$

* Changes the **planning target**, *not* which pair is sampled (search-control stays uniform).
* Bonus grows without bound ⇒ every long-neglected action **eventually** gets rechecked ⇒ real transition observed, model corrected, $\tau$ reset.
* Neglected $a\_2$ beats greedy $a\_1$ once

  $$\tau(s,a_2) > \left(\frac{r_1-r_2}{\kappa}\right)^2.$$

## Search control — *where* to spend backups

* **Prioritized sweeping** (backward, error-driven): keep a priority queue keyed by **Bellman-error magnitude**; back up predecessors of recently-changed states first. *"Fix what's most wrong first."*
* **Backup order matters** (deterministic chain, $\alpha=1$): a **backward** sweep propagates the terminal reward to *all* predecessors in **one** sweep ($Q(i,a)=\gamma^{L-1-i}$); a **forward** sweep advances value only **one edge per sweep** ($L$ sweeps needed).
* **Trajectory sampling** (forward, on-policy): back up states the current policy actually visits. *"Work on what's relevant."*
* **RTDP** = asynchronous value iteration along greedy trajectories (planning, given model):

  $$V(s) \leftarrow \max_a \sum_{s',r}\hat p(s',r\mid s,a)\bigl[r+\gamma V(s')\bigr].$$

  Converges to optimal on the **relevant** states without sweeping the whole space.

## Decision-time planning (plan for the *current* state, then discard)

| | **Heuristic search** | **Rollout** | **MCTS** |
| :-- | :-- | :-- | :-- |
| Structure | fixed-depth tree | none (flat trajectories) | tree grown 1 node/iter |
| Leaf value | heuristic $\hat v$ | full rollout return | rollout return, **stored** |
| Backup | one max/expectation pass | average of $K$ returns | incremental running mean |
| Search control | expand to depth $d$ | $K$ sims/action (uniform) | **selection: explore/exploit** |
| Memory reused? | no | no | **yes** |

* **Heuristic search:** back up **max at decision nodes, expectation at chance nodes** (unrolled Bellman optimality); leaves cut at depth and valued by $\hat v$.
* **Rollout:** for each candidate first action, average returns of simulations that then follow a fixed **rollout policy**; pick the best. No memory.
* **MCTS** = rollout **+ reusable tree + UCB selection**. Four phases per iteration:

  $$\text{select (tree policy)} \to \text{expand} \to \text{simulate (rollout policy)} \to \text{backup}.$$

  * **Tree policy** = UCT: $\;a=\arg\max_a\bigl[Q(s,a)+c\sqrt{\ln N(s)/N(s,a)}\bigr]$ (untried actions first, $N=0\Rightarrow$ bonus $=\infty$).
  * Backup = running mean $Q \leftarrow Q + \tfrac{1}{N}(G-Q)$.
  * **Real move played = most-visited root action** $\arg\max_a N(s\_0,a)$, *not* highest $Q$; then reuse the subtree.

## One-glance map

| Axis | Options |
| :-- | :-- |
| Experience source | **real** (learning) ↔ **simulated** (planning) — same backups |
| Backup width | **expected** (all successors, needs $p$) ↔ **sample** (one successor) |
| When to plan | **background** (Dyna, sweeping, RTDP) ↔ **decision-time** (heuristic search, rollout, MCTS) |
| Where to back up | uniform ↔ **prioritized** (backward, error) ↔ **on-policy trajectory** |
