---
title: Ising Model
layout: default
noindex: true
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

# Ising Model

## Introduction

The **Ising model** (or Lenz–Ising model), named after the physicists Ernst Ising and Wilhelm Lenz, is a mathematical model of **ferromagnetism** in statistical mechanics. The model consists of discrete variables that represent magnetic dipole moments of atomic "spins" that can be in one of two states (+1 or −1). The spins are arranged in a graph, usually a lattice (where the local structure repeats periodically in all directions), allowing each spin to interact with its neighbors. Neighboring spins that agree have a lower energy than those that disagree; the system tends to the lowest energy but heat disturbs this tendency, thus creating the possibility of different structural phases. The two-dimensional square-lattice Ising model is one of the simplest statistical models to show a phase transition. Though it is a highly simplified model of a magnetic material, the Ising model can still provide qualitative and sometimes quantitative results applicable to real physical systems.

The Ising problem without an external field can be equivalently formulated as a **graph maximum cut** (Max-Cut) problem that can be solved via **combinatorial optimization**.

<figure>
  <img src="{{ 'assets/images/notes/monte-carlo-methods/ising-model/2d-ising-model.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>Two-dimensional Ising model shown as a lattice of interacting spins.</figcaption>
</figure>

**Ferromagnetism**:
<div class="accordion">
  <details markdown="1">
    <summary>Ferromagnetism</summary>

Ферромагнетизм — это явление сильного магнитного упорядочивания в материалах (ферромагнетиках), при котором их атомы самопроизвольно намагничиваются даже без внешнего поля, образуя домены (магнитные области) с параллельными магнитными моментами, что позволяет им оставаться постоянными магнитами и притягиваться к другим магнитам, в отличие, например, от парамагнетиков или диамагнетиков. 
Ключевые особенности ферромагнетизма:
* Спонтанная намагниченность: Вещество намагничено само по себе, без внешнего поля.
* Сильное взаимодействие: Энергия взаимодействия между соседними атомными магнитными моментами преобладает над их взаимодействием с внешним полем.
* Точка Кюри: Ферромагнитные свойства исчезают при нагревании выше определенной температуры, называемой температурой Кюри, после чего материал переходит в парамагнитное состояние.
* Доменная структура: Внутри ферромагнетика существуют микроскопические области (домены), где магнитные моменты ориентированы одинаково. В ненамагниченном состоянии эти домены ориентированы хаотично, но при намагничивании они выстраиваются в одном направлении.
Примеры: Железо, никель, кобальт и их сплавы являются классическими ферромагнетиками. 

Ферромагнетизм — появление спонтанной намагниченности при температуре ниже температуры Кюри вследствие упорядочения магнитных моментов, при котором большая их часть параллельна друг другу. Это основной механизм, с помощью которого определённые материалы (например, железо) образуют постоянные магниты или притягиваются к магнитам. Вещества, в которых возникает ферромагнитное упорядочение магнитных моментов, называются ферромагнетиками.

В физике принято различать несколько типов магнетизма. Ферромагнетизм (наряду с аналогичным эффектом ферримагнетизма) является самым сильным типом магнетизма и ответственен за физическое явление магнетизма в магнитах, встречающееся в повседневной жизни. Вещества с тремя другими типами магнетизма — парамагнетизмом, диамагнетизмом и антиферромагнетизмом, слабее реагируют на магнитные поля, — но силы обычно настолько слабы, что их можно обнаружить только с помощью чувствительных приборов в лаборатории.

Повседневный пример ферромагнетизма — магнит на холодильник, который используется для хранения записок на дверце холодильника.

Ferromagnetism is **a type of magnetic behavior a material can have**. It’s not just “how magnetized it is,” but it *does* strongly affect what magnetization can do.

### Is it a “property of magnetization” or a “binary label”?

* **Mostly a label/classification:** Calling a material *ferromagnetic* means it has an internal tendency for electron spins to align (below a certain temperature), so it *can* form magnetic domains and *can* become strongly magnetized.
* **Not strictly binary in real life:** It’s more like “this material has ferromagnetic order under these conditions.”

  * **Temperature matters**: above the **Curie temperature**, the same material stops being ferromagnetic (it becomes paramagnetic).
  * Composition and crystal structure matter too (some alloys are ferromagnetic, some aren’t).

So: it’s not “a certain magnetization value,” it’s “this material supports a ferromagnetic ordered state (domains) under the right conditions.”

### Does ferromagnetism happen on its own, or only with an external field?

**Both, depending on what you mean:**

1. **The *ferromagnetic order* (domains) happens on its own**
   Below the Curie temperature, the material naturally splits into **domains** where spins are aligned. That’s *spontaneous*—it doesn’t require an external magnetic field.

2. **But a piece of ferromagnetic material often has *no net magnetization* on its own**
   Even though each domain is magnetized, the domains can point in different directions, so they **cancel out overall**. That’s why an unmagnetized chunk of iron might not behave like a strong magnet.

   (*Net magnetization (M)* is the overall magnetic alignment of all individual magnetic moments (like protons) in a substance, acting like one big magnet, resulting from the vector sum of these tiny moments, which are usually random but align in an external magnetic field (like in MRI) to produce a detectable signal, forming longitudinal (Mz) and transverse (Mxy) components.)

3. **An external magnetic field helps create a net magnetization**
   A field tends to make domains aligned with the field **grow**, and others **shrink/flip**, giving the whole object a **net** magnetization. After removing the field:

* **Soft ferromagnets** (like many forms of iron) may lose most of it.
* **Hard ferromagnets** (permanent magnet materials) keep a lot of it (**remanence**) because domain walls get “pinned.”

### A simple mental model

* Ferromagnetism = the material *can* have many tiny magnets naturally aligned in domains.
* External field = persuades those domains to line up more in the same direction, making the object act like a magnet overall.

If you tell me a material you’re thinking of (iron? steel? fridge magnets?), I can map this to that concrete example.

Yes — but the key is: **“ferromagnetism” isn’t a single slider.** Different materials can be “more magnetic” in different ways.

Here are the main knobs people mean when they ask that, and examples of materials that are “more” in each sense:

### 1) Stronger *maximum* magnetization (how big it can get)

This is about how large the net magnetic moment can be when fully aligned (called **saturation magnetization**).

* Very high: **iron (Fe)**, **cobalt (Co)**, some **iron–cobalt alloys**
* Lower: **nickel (Ni)**
  So, in that sense, some materials can become “stronger magnets” than others.

### 2) Higher temperature where it stays ferromagnetic

That’s the **Curie temperature**: above it, ferromagnetism disappears.

* High: **cobalt** (stays ferromagnetic to a higher temperature than iron)
* Medium: **iron**
* Lower: **nickel**
  So some materials “keep ferromagnetism” better when hot.

### 3) Easier vs harder to magnetize (and whether it stays magnetized)

Two big categories:

* **Soft ferromagnets**: magnetize easily, demagnetize easily
  Used in transformer cores, electromagnets.

  * Examples: **pure iron**, **silicon steel**, **permalloy (Ni-Fe)**

* **Hard ferromagnets**: harder to magnetize, but **stay magnetized strongly** (permanent magnets)

  * Examples: **NdFeB (neodymium magnets)**, **SmCo**, **ferrites** (ceramic magnets)

So a “neodymium magnet” isn’t “more ferromagnetic” because of a bigger *ability to align* (that’s complicated), but because it has **very high coercivity + high remanence**, so it keeps a strong magnetization.

### 4) Domain behavior and microstructure matter a lot

Even the *same* chemical material can behave differently depending on:

* heat treatment,
* grain structure,
* impurities/defects,
* whether it’s an alloy.

That’s why “steel” can range from barely magnetic to strongly magnetic depending on the alloy and processing.

  </details>
</div>

<figure>
  <img src="{{ 'assets/images/notes/monte-carlo-methods/ising-model/ferromagnetism.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>Domains alignment under the influence of a external magnetic field $B_0$.</figcaption>
</figure>

## Definition

Let $\Lambda$ denote a set of lattice sites equipped with an adjacency relation (equivalently, the vertex set of a graph) inducing a $d$-dimensional lattice structure. To each site $k\in\Lambda$ we associate a spin variable $\sigma_k \in \lbrace -1,+1\rbrace$. A **spin configuration** is the collection $\sigma=\lbrace\sigma_k\rbrace_{k\in\Lambda}$, i.e., an assignment of a spin value to every site.

For each nearest-neighbor pair $i,j\in\Lambda$ (denoted $\langle ij\rangle$), the model specifies an interaction strength $J_{ij}$, and each site $j\in\Lambda$ is subject to an external magnetic field $h_j$. The energy of a configuration $\sigma$ is defined by the **Ising Hamiltonian**

$$H(\sigma) = -\sum_{\langle ij\rangle} J_{ij}\sigma_i\sigma_j - \mu\sum_{j\in\Lambda} h_j\sigma_j,$$

where the first sum ranges over all unordered nearest-neighbor pairs (each pair counted exactly once) and $\mu$ denotes the magnetic moment. The Ising Hamiltonian is an example of a **pseudo-Boolean function** ($f:\lbrace -1,+1\rbrace^{\Lambda}\to\mathbb{R}$).

The *configuration probability* is given by the **Boltzmann distribution** with inverse temperature $\beta \geq 0$:

$$P_\beta(\sigma) = \frac{e^{-\beta H(\sigma)}}{Z_\beta},
\qquad \text{where } \beta=\frac{1}{k_B T},$$

and where the partition function $Z_\beta$ (the normalizing constant) is given by

$$Z_\beta = \sum_{\sigma} e^{-\beta H(\sigma)},$$

with the sum taken over all configurations $\sigma\in\lbrace -1,+1\rbrace^{\Lambda}$. For any observable $f:\lbrace -1,+1\rbrace^{\Lambda}\to\mathbb{R}$, its expectation under $P_\beta$ is

$$\langle f\rangle_\beta = \sum_{\sigma} f(\sigma)P_\beta(\sigma),$$

which represents the thermal mean of $f$ at inverse temperature $\beta$.

The configuration probabilities $P_{\beta }(\sigma)$ represent the probability that (in equilibrium) the system is in a state with configuration $\sigma$.

<figure>
  <img src="{{ 'assets/images/notes/monte-carlo-methods/ising-model/2d_ising_model_different_temp_three_config.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>Three configurations of the Ising model at different temperatures.</figcaption>
</figure>

Cool interactive Ising model simulation: https://mattbierbaum.github.io/ising.js/

### Discussion

The negative sign in each term of the Hamiltonian $H(\sigma)$ is simply a convention. With this convention, Ising models are categorized by the sign of the interaction between spins $i$ and $j$:

* If $J_{ij} > 0$, the interaction is **ferromagnetic**.
* If $J_{ij} < 0$, the interaction is **antiferromagnetic**.
* If $J_{ij} = 0$, the spins **do not interact**.

A model is called ferromagnetic (or antiferromagnetic) when all interactions share the same sign. Historically, the original Ising models were ferromagnetic, and “Ising model” is still often used to mean the ferromagnetic case by default.

In a ferromagnet, neighboring spins prefer to match, so configurations with adjacent spins of the same sign are more likely. In an antiferromagnet, neighbors instead tend to have opposite signs.

This sign convention also clarifies the effect of an external field on a spin at site $j$: the spin tends to align with the field. In particular:

* If $h_j > 0$, site $j$ is biased toward the positive direction.
* If $h_j < 0$, it is biased toward the negative direction.
* If $h_j = 0$, there is no external influence at that site.

### Simplifications

Often, one studies the Ising model without any external field, i.e. $h=0$ for all $j\in\Lambda$. Then the Hamiltonian reduces to

$$H(\sigma)= -\sum_{\langle i, j\rangle} J_{ij}\sigma_i\sigma_j.$$


When the field is zero everywhere $(h=0)$, the model is symmetric under flipping all spins; introducing a nonzero field breaks that symmetry.

Another frequent simplification is to assume every nearest-neighbor pair $\langle i j\rangle$ has the same coupling strength, so $J_{ij}=J$ for all such pairs. In that case,

$$H(\sigma)= -J\sum_{\langle i, j\rangle} \sigma_i\sigma_j $$


## Ising Model Simulation (Metropolis-Hastings algorithm)

More about the algorithm is [here](/subpages/monte-carlo-methods/mcmc/metropolis–hastings-algorithm/).


## Connection to Graph Maximum Cut

Consider a weighted undirected graph $G$. The size of the **cut** is the sum of the weights of the edges between $S$ and $G\setminus S$. A **maximum cut** size is at least the size of any other cut.

For an Ising model on a graph $G$ with no external field, the Hamiltonian can be written as a sum over edges:

$$H(\sigma) = - \sum_{ij\in E(G)} J_{ij}\sigma_i\sigma_j.$$

Each vertex $i$ carries a spin $\sigma_i\in\lbrace\pm1\rbrace$. A spin assignment $\sigma$ partitions the vertices into
$V^+$ (spin up) and $V^-$ (spin down). Let $\delta(V^+)$ be the set of edges crossing between $V^+$ and $V^-$. The weighted size of this cut is

$$\lvert \delta(V^+)\rvert = \frac{1}{2}\sum_{ij\in \delta(V^+)} W_{ij},$$

where $W_{ij}$ is the edge weight (the factor $1/2$ corrects for double counting since $W_{ij}=W_{ji}$).

Using the partition into $V^+$ and $V^-$, one can decompose the Hamiltonian as

$$H(\sigma)
= -\sum_{ij\in E(V^+)} J_{ij}-\sum_{ij\in E(V^-)} J_{ij}+\sum_{ij\in \delta(V^+)} J_{ij}
= -\sum_{ij\in E(G)} J_{ij}+2\sum_{ij\in \delta(V^+)} J_{ij}.$$

The first term is independent of $\sigma$, so minimizing $H(\sigma)$ over $\sigma$ is equivalent to minimizing $\sum_{ij\in \delta(V^+)} J_{ij}$.

If we set the edge weights to $W_{ij}=-J_{ij}$, the zero-field Ising optimization becomes a **Max-Cut** problem: maximizing the cut size $\lvert\delta(V^+)\rvert$. With this substitution, the Hamiltonian relates to the cut value via

$$H(\sigma) = \sum_{ij\in E(G)} W_{ij}-4\lvert\delta(V^+)\rvert.$$

