---
title: Physical Nature of Information
layout: default
noindex: true
---

# Physical Nature of Information

## 1. Thermodynamics and Statistical Physics

Our knowledge is always partial. When studying macroscopic systems, some degrees of freedom remain hidden. In mechanics, electricity and magnetism, we found a way around partial knowledge by working with *closed sets of equations describing explicitly known degrees of freedom*. Even then, our description applies only to things that can be considered independent of the unknown within a given accuracy. Even with closed equations, initial or boundary conditions are known only with finite precision, which has dramatic consequences for unstable systems where small variations in initial data lead to large deviations in evolution.

This chapter deals with *observable manifestations of the hidden degrees of freedom*. While we do not know their state, we do know their nature — whether those degrees of freedom are related to moving particles, spins, bacteria or market traders. In particular, we know the symmetries and conservation laws of the system.

### 1.1 Basics of Thermodynamics

*Thermodynamics studies restrictions on the possible macroscopic properties that follow from the fundamental conservation laws.* Therefore, thermodynamics does not predict numerical values but rather sets inequalities and establishes relations among different properties.

One starts building thermodynamics by identifying a conserved quantity, which can be exchanged but not created — it could be matter, money, energy, etc. For most physical systems, the basic symmetry is invariance of the fundamental laws with respect to time shifts. Evolution of an isolated physical system is usually governed by the Hamiltonian (the energy written in canonical variables), whose time-independence means energy conservation.

#### First Law of Thermodynamics

The conserved quantity of thermodynamics is called energy and denoted $E$. We ascribe to the states of the system the values of $E$. *Equilibrium* states are completely characterized by the *static* values of observable variables.

Passing from state to state under external action involves the energy change, which generally consists of two parts: the energy change of visible degrees of freedom (work) and the energy change of hidden degrees of freedom (heat).

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(First Law of Thermodynamics)</span></p>

If the visible energy balance does not hold then the energy of the hidden must change. The energy is a function of state so we use differential, but we use $\delta$ for heat and work, which aren't differentials of any function:

$$dE = \delta Q - \delta W.$$

Heat exchange and work depend on the path taken from A to B, that is they refer to particular forms of energy transfer (not energy content).

</div>

#### Entropy and the Basic Problem

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Entropy and the Extremum Principle)</span></p>

**The basic problem** of thermodynamics is the determination of the equilibrium state that eventually results after all internal constraints are removed in a closed composite system. The problem is solved with the help of the extremum principle: there exists a quantity $S$ called **entropy** which is a function of the parameters of the system. The values assumed by the parameters in the absence of an internal constraint **maximize the entropy** over the manifold of available states (Clausius 1865).

</div>

#### Carnot Efficiency

A heat engine works by delivering heat from a reservoir at higher $T_1$ to another reservoir at $T_2$, doing some work in the process. The work $W$ is the difference between the heat given by the hot reservoir $Q_1$ and the heat absorbed by the cold one $Q_2$. Carnot in 1824 stated that $Q_2/T_2 \ge Q_1/T_1$, so the efficiency is bounded from above:

$$\frac{W}{Q_1} = \frac{Q_1 - Q_2}{Q_1} \le 1 - \frac{T_2}{T_1}.$$

The entropy decrease of the hot reservoir, $\Delta S_1 = Q_1/T_1$, must be less than the entropy increase of the cold one, $\Delta S_2 = Q_2/T_2$. Maximal work is achieved for minimal (zero) total entropy change, $\Delta S_2 = \Delta S_1$, which happens for **reversible processes**.

#### Thermodynamic Limit

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Thermodynamic Limit)</span></p>

Traditionally, thermodynamics deals with extensive parameters whose value grows linearly with the number of degrees of freedom. To treat energy as an additive variable we make two assumptions: (i) the forces of interaction are short-range and act only along the boundary, (ii) take the thermodynamic limit $V \to \infty$ where one can neglect surface terms that scale as $V^{2/3} \propto N^{2/3}$ in comparison with the bulk terms that scale as $V \propto N$.

In that limit, thermodynamic entropy is also an extensive variable and is a homogeneous first-order function of all extensive parameters:

$$S(\lambda E, \lambda V, \ldots) = \lambda S(E, V, \ldots).$$

</div>

This function $S(E, V, \ldots)$, called the **fundamental relation**, is *everything* one needs to know to solve the basic problem (and others) in thermodynamics.

#### Entropy-Energy Duality and Convexity

For every interval of a definite derivative sign, say $(\partial E / \partial S)_X > 0$, we can solve $S = S(E, V, \ldots)$ uniquely for $E(S, V, \ldots)$, which is an equivalent fundamental relation. Any entropy extremum is also an energy extremum. At equilibrium:

$$(\partial^2 E / \partial X^2)_S = -(\partial^2 S / \partial X^2)_E (\partial E / \partial S)_X.$$

The equilibrium is an entropy maximum, so $-(\partial^2 S / \partial X^2)_E$ is negative. When the temperature is positive, the equilibrium is the energy minimum. The equilibrium curve $S(E)$ is **convex**, which guarantees stability of a homogeneous state: if the system were to break into two halves with slightly different energies, the entropy must decrease: $2S(E) > S(E+\Delta) + S(E-\Delta) = 2S(E) + S'' \Delta^2$, which requires $S'' < 0$.

#### Equations of State

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Equations of State)</span></p>

The partial derivatives of an extensive variable with respect to its arguments (also extensive parameters) are **intensive parameters**. In thermodynamics we have only extensive and intensive variables (because we take the thermodynamic limit $N \to \infty$, $V \to \infty$ keeping $N/V$ finite). For the energy one writes

$$\frac{\partial E}{\partial S} \equiv T(S,V,N), \quad \frac{\partial E}{\partial V} \equiv -P(S,V,N), \quad \frac{\partial E}{\partial N} \equiv \mu(S,V,N), \ldots$$

These relations are called the **equations of state** and they serve as *definitions* for temperature $T$, pressure $P$ and chemical potential $\mu$, corresponding to the respective extensive variables $S, V, N$.

</div>

Entropy is the missing information, so temperature is the energetic price of information. Our entropy is dimensionless, so that $T$ is assumed to be multiplied by the Boltzmann constant $k = 1.3 \cdot 10^{-23}\, J/K$ and have the same dimensionality as the energy. From the equations of state we write

$$dE = \delta Q - \delta W = TdS - PdV + \mu dN.$$

Entropy is thus responsible for hidden degrees of freedom (i.e. heat) while other extensive parameters describe macroscopic degrees of freedom. In equilibrium the missing information is maximal for hidden degrees of freedom.

#### Euler Equation and Gibbs-Duhem Relation

Both energy and entropy are homogeneous first-order functions of their variables. Differentiating $E(\lambda S, \lambda V, \lambda N) = \lambda E(S, V, N)$ with respect to $\lambda$ and setting $\lambda = 1$ one gets the **Euler equation**:

$$E = TS - PV + \mu N.$$

The equations of state are homogeneous of zero order, e.g. $T(\lambda E, \lambda V, \lambda N) = T(E, V, N)$. From the Euler equation one can derive the **Gibbs-Duhem relation**, which states that because $dS = (dE + PdV - \mu dN)/T$, the sum of products of the extensive parameters and the differentials of the corresponding intensive parameters vanish:

$$Ed(1/T) + Vd(P/T) - Nd(\mu/T) = 0.$$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Summary of Formal Structure)</span></p>

The fundamental relation is equivalent to the three equations of state. If only two equations of state are given, then the Gibbs-Duhem relation may be integrated to obtain the third relation up to an integration constant. Alternatively, one may integrate the molar relation $de = Tds - Pdv$ to get $e(s,v)$, again with an undetermined constant of integration.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Two-Level System)</span></p>

Consider a system characterized solely by its energy, which can change between zero and $E_{max}$. The equation of state is the energy-temperature relation $E/E_{max} = (1 + e^{\epsilon/T})^{-1}$, which tends to $1/2$ at $T \gg \epsilon$ and is exponentially small at $T \ll \epsilon$. In Section 1.3, we shall identify this with a set of elements with two energy levels, $0$ and $\epsilon$. To find the fundamental relation in the entropy representation, we integrate the equation of state:

$$\frac{1}{T} = \frac{dS}{dE} = \frac{1}{\epsilon}\ln\frac{E_{max} - E}{E} \;\Rightarrow\; S(E) = \frac{E_{max}}{\epsilon}\ln\frac{E_{max}}{E_{max} - E} + \frac{E}{\epsilon}\ln\frac{E_{max} - E}{E}.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Ideal Monatomic Gas)</span></p>

Consider an ideal monatomic gas characterized by two equations of state (found experimentally, with $R \simeq 8.3\,\text{J/mole K} \simeq 2\,\text{cal/mole K}$):

$$PV = NRT, \quad E = 3NRT/2.$$

The extensive parameters are $E, V, N$ so we want the fundamental equation in the entropy representation $S(E,V,N)$. We write the Euler equation $S = E/T + VP/T - N\mu/T$ and express the intensive variables $1/T, P/T, \mu/T$ via extensive variables. The equations of state give us two of them:

$$\frac{P}{T} = \frac{NR}{V} = \frac{R}{v}, \quad \frac{1}{T} = \frac{3NR}{2E} = \frac{3R}{e}.$$

To find $\mu/T$ as a function of $e, v$, we use the Gibbs-Duhem relation in the entropy representation. Computing $d(1/T) = -3Rde/2e^2$ and $d(P/T) = -Rdv/v^2$, and substituting into the Gibbs-Duhem relation gives:

$$d\!\left(\frac{\mu}{T}\right) = -\frac{3R}{2e}de - \frac{R}{v}dv, \quad \frac{\mu}{T} = C - \frac{3R}{2}\ln e - R\ln v,$$

$$s = \frac{1}{T}e + \frac{P}{T}v - \frac{\mu}{T} = s_0 + \frac{3R}{2}\ln\frac{e}{e_0} + R\ln\frac{v}{v_0}.$$

</div>

### 1.2 Thermodynamic Potentials

The fundamental relation always relates extensive quantities. Even though it is always possible to eliminate, say, $S$ from $E = E(S,V,N)$ and $T = T(S,V,N)$ getting $E = E(T,V,N)$, this *is not* a fundamental relation and does not contain all the information. Indeed, $E = E(T,V,N)$ is actually a partial differential equation (because $T = \partial E / \partial S$) and even if it can be integrated the result would contain an undetermined function of $V, N$. Still, it is easier to measure temperature than entropy, so it is convenient to have a complete formalism with an intensive parameter as operationally independent variable and an extensive parameter as a derived quantity. This is achieved by the **Legendre transform**.

#### The Legendre Transform

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Legendre Transform)</span></p>

We want to pass from the relation $Y = Y(X)$ to that in terms of $P = \partial Y / \partial X$. It is not enough to eliminate $X$ and consider $Y = Y[X(P)] = Y(P)$, because such function determines $Y = Y(X)$ only up to a shift along $X$. Instead, for every $P$ we specify the position $\psi(P)$ where the straight line tangent to the curve intercepts the $Y$-axis: $\psi = Y - PX$.

The function $\psi(P) = Y[X(P)] - PX(P)$ completely defines the curve; here one substitutes $X(P)$ found from $P = dY/dX$. The function $\psi(P)$ is the **Legendre transform** of $Y(X)$.

From $d\psi = -PdX - XdP + dY = -XdP$ one gets $-X = d\psi/dP$, i.e. the inverse transform is the same up to a sign: $Y = \psi + XP$.

The transform is possible when for every $X$ there is one $P$, that is $P(X)$ is monotonic and $Y(X)$ is **convex**: $dP/dX = d^2Y/dX^2 \neq 0$.

</div>

#### Free Energy (Helmholtz Potential)

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Free Energy)</span></p>

**Free energy** $F = E - TS$ (also called the Helmholtz potential) is the partial Legendre transform of $E$ which replaces the entropy by the temperature as an independent variable:

$$dF(T,V,N,\ldots) = -SdT - PdV + \mu dN + \ldots$$

Counterpart to $(\partial E / \partial S)\_{VN} = T$ is $(\partial F / \partial T)\_{VN} = -S$.

</div>

The free energy is particularly convenient for the description of a system in thermal contact with a heat reservoir because then the temperature is fixed and we have one variable less to care about. The maximal work that can be done under a constant temperature (equal to that of the reservoir) is minus the differential of the free energy. Indeed, the work done *by the system and the thermal reservoir* is:

$$d(E + E_r) = dE + T_r dS_r = dE - T_r dS = d(E - T_r S) = d(E - TS) = dF.$$

In other words, the free energy $F = E - TS$ is that part of the internal energy which is *free* to turn into work, the rest of the energy $TS$ we must keep to sustain a constant temperature. The equilibrium state minimizes $F$, not absolutely, but over the manifold of states with the temperature equal to that of the reservoir.

System can reach the minimum of the free energy minimizing energy and maximizing entropy. The former often requires creating some order in the system, while increasing entropy requires disorder. Which of these tendencies wins depends on temperature, setting their relative importance.

#### Enthalpy

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Enthalpy)</span></p>

**Enthalpy** $H = E + PV$ is that partial Legendre transform of $E$ which replaces the volume by the pressure:

$$dH(S, P, N, \ldots) = TdS + VdP + \mu dN + \ldots$$

It is particularly convenient for situations in which the pressure is maintained constant by a pressure reservoir. Just as the energy acts as a potential at constant entropy and the free energy as potential at constant temperature, so the enthalpy is a potential for the work done *by the system and the pressure reservoir* at constant pressure.

</div>

The heat received by the system at constant pressure (and $N$) is the enthalpy change: $\delta Q = dQ = TdS = dH$. Compare it with the fact that the heat received by the system at constant volume (and N) is the energy change since the work is zero.

#### Gibbs Free Energy and Grand Canonical Potential

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Gibbs Free Energy and Grand Canonical Potential)</span></p>

One can replace both entropy and volume obtaining the **Gibbs thermodynamic potential** $G = E - TS + PV$ which has $dG(T,P,N,\ldots) = -SdT + VdP + \mu dN + \ldots$ and is minimal in equilibrium at constant temperature and pressure. From the Euler equation:

$$F = -P(T,V)V + \mu(T,V)N, \quad H = TS + \mu N, \quad G = \mu(T,P)N.$$

When there is a possibility of change in the number of particles (because the system is in contact with a particle source having a fixed chemical potential), it is convenient to use the **grand canonical potential** $\Omega(T,V,\mu) = E - TS - \mu N$ which has $d\Omega = -SdT - PdV - Nd\mu$. The grand canonical potential reaches its minimum under constant temperature and chemical potential.

</div>

Since the Legendre transform is invertible, all potentials are equivalent and contain the same information. The choice of the potential for a given physical situation is that of convenience: we usually take what is fixed as a variable to diminish the number of effective variables.

### 1.3 Microcanonical Distribution

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Microcanonical Distribution)</span></p>

Consider a *closed* system with the fixed number of particles $N$ and the energy $E_0$. Boltzmann *assumed* that all microstates with the same energy have equal probability (ergodic hypothesis), which gives the **microcanonical distribution**:

$$\rho(p,q) = \Gamma^{-1}\delta[E(p_1 \ldots p_N, q_1 \ldots q_N) - E_0],$$

where $\Gamma$ is the volume of the phase space occupied by the system:

$$\Gamma(E,V,N,\Delta) = \int \delta[E(p_1 \ldots p_N, q_1 \ldots q_N) - E_0]\, d^{3N}p\, d^{3N}q.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Ideal Gas Phase Volume)</span></p>

For $N$ noninteracting particles (ideal gas) the states with energy $E = \sum p^2/2m$ are in the $\mathbf{p}$-space near the hyper-sphere with the radius $\sqrt{2mE}$. Using the surface area of the hyper-sphere with radius $R$ in $3N$-dimensional space, $2\pi^{3N/2} R^{3N-1}/(3N/2-1)!$, we have

$$\Gamma(E,V,N,\Delta) \propto E^{3N/2-1}V^N/(3N/2-1)! \approx (E/N)^{3N/2}V^N.$$

</div>

#### Boltzmann Entropy

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Boltzmann Entropy)</span></p>

To link statistical physics with thermodynamics one must define a thermodynamic potential as a function of respective variables. For microcanonical distribution, Boltzmann introduced the entropy as

$$S(E,V,N) = \ln \Gamma(E,V,N).$$

This is one of the most important formulas in physics (on a par with $F = ma$, $E = mc^2$ and $E = \hbar\omega$).

</div>

Noninteracting subsystems are statistically independent. The statistical weight of the composite system is a product — for every state of one subsystem we have all the states of another. If the weight is a product then the entropy is a sum. For interacting subsystems, this is true only for short-range forces in the thermodynamic limit $N \to \infty$.

#### Equilibrium from Statistics

Consider two subsystems, 1 and 2, that can exchange energy. Then

$$\Gamma(E) = \sum_{i=1}^{E/\Delta} \Gamma_1(E_i)\Gamma_2(E - E_i).$$

The maximum of $\Gamma$ corresponds to the thermal equilibrium. Computing the derivative and setting it to zero gives $(\partial S_1/\partial E_1)\_{\bar{E}_1} = (\partial S_2/\partial E_2)\_{\bar{E}_2}$, that is the temperatures of the subsystems are equal. In the thermodynamic limit, $S(E) = S_1(\bar{E}_1) + S_2(\bar{E}_2) + O(\log N)$.

#### The Gibbs Paradox and Indistinguishability

Taking the logarithm of $\Gamma$ for the ideal gas gives a result containing a non-extensive term $N \ln V$. The resolution is that we need to treat the particles as indistinguishable, otherwise we need to account for the entropy of mixing different species. This requires dividing $\Gamma$ by the number of permutations $N!$, which makes the resulting entropy of the ideal gas extensive and in agreement with thermodynamics:

$$S(E,V,N) = (3N/2)\ln E/N + N\ln eV/N + \text{const}.$$

Defining temperature in the usual way, $T^{-1} = \partial S / \partial E = 3N/2E$, we get the correct expression $E = 3NT/2$.

#### Two-Level System (Discrete)

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Two-Level System — Negative Temperature)</span></p>

Consider $N$ particles (spins, neurons), each with two energy levels $0$ and $\epsilon$. If the energy of the set is $E$ then there are $L = E/\epsilon$ upper levels occupied. The statistical weight is $\Gamma(N,L) = C_N^L = N!/L!(N-L)!$ and the entropy is $S(E,N) = \ln\Gamma$.

At the thermodynamic limit $N \gg 1$ and $L \gg 1$, the entropy is $S(E,N) \approx N\ln[N/(N-L)] + L\ln[(N-L)/L]$, which coincides with the earlier result from integrating the equation of state.

The entropy is symmetric about $E = N\epsilon/2$ and is zero at $E = 0, N\epsilon$ when all the particles are in the same state. The temperature-energy relation is $T^{-1} = \partial S / \partial E \approx \epsilon^{-1}\ln[(N-L)/L]$. When $E > N\epsilon/2$, the population of the higher level is larger than that of the lower one (inverse population as in a laser) and the **temperature is negative**.

Negative temperature may happen only in systems with the upper limit of energy levels and simply means that by adding energy beyond some level we actually decrease the entropy (the number of accessible states). Negative temperatures are actually "hotter" than positive: if you put your hand on a negative temperature system, you feel heat flowing into you.

</div>

### 1.4 Canonical Distribution and Fluctuations

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Canonical Distribution and Partition Function)</span></p>

Consider a system exchanging energy with a thermostat, which can be thought of as consisting of infinitely many copies of our system — this is the so-called **canonical ensemble**, characterized by $N, V, T$. The probability to be in a given microstate $a$ with the energy $E$ is derived from the microcanonical distribution of the whole system. Since $E \ll E_0$, expanding the thermostat's entropy $S_0(E_0 - E) \approx S_0(E_0) - E/T$ in the exponent gives:

$$w_a(E) = Z^{-1}\exp(-E/T),$$

$$Z = \sum_a \exp(-E_a/T).$$

The normalization factor $Z(T,V,N)$ is a sum over all states accessible to the system and is called the **partition function**.

</div>

The Gibbs canonical distribution tells that the probability of a given microstate exponentially decays with the energy of the state, while the probability of a *given energy* $W(E) = \Gamma(E)Z^{-1}\exp(-E/T)$ has a peak (since $\Gamma(E)$ grows with $E$ very fast but $\exp(-E/T)$ decays faster than any power).

#### Alternative Derivation via Gibbs Ensemble

An alternative derivation uses the Gibbs idea of the canonical ensemble as a virtual set, of which the single member is the system under consideration and the energy of the total set is fixed. The probability to have our chosen system in the state $a$ is given by the average number of systems $\bar{n}_a$ in this state divided by the total number of systems $\mathcal{N}$. Using the method of Lagrangian multipliers to maximize $\ln W$ under the constraints $\sum_a n_a = \mathcal{N}$ and $\sum_a E_a n_a = \epsilon \mathcal{N}$, one obtains

$$\frac{n_a^*}{\mathcal{N}} = \frac{\exp(-\beta E_a)}{\sum_a \exp(-\beta E_a)},$$

where $\beta$ is determined implicitly by $E/\mathcal{N} = \epsilon = \sum_a E_a \exp(-\beta E_a)/\sum_a \exp(-\beta E_a)$.

#### Free Energy from the Partition Function

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Free Energy and Partition Function)</span></p>

To get thermodynamics from the Gibbs distribution one defines the free energy via the partition function:

$$F(T,V,N) = -T\ln Z(T,V,N).$$

To prove that, differentiate the identity $Z = \exp(-F/T) = \sum_a \exp(-E_a/T)$ with respect to temperature, which gives $F = \bar{E} + T\frac{\partial F}{\partial T} = E - TS$, which is the thermodynamic free energy.

</div>

#### Gibbs Entropy

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Gibbs Entropy)</span></p>

For a closed system Boltzmann defined $S = \ln\Gamma$ while the probability of state was $w_a = 1/\Gamma$. In other words, the entropy was minus the log of the probability, $S = -\ln w_a$. For a subsystem at fixed temperature, different states have different probabilities, and both energy and entropy fluctuate. For a system that has a Gibbs distribution, $\ln w_a$ is linear in $E_a$, so that the entropy at a mean energy is the **mean entropy**:

$$\langle S \rangle = -\langle\ln w_a\rangle = -\sum w_a \ln w_a = \sum w_a(E_a/T + \ln Z) = E/T + \ln Z = (E - F)/T.$$

Even though the Gibbs entropy $S = -\sum w_a \ln w_a$ is derived here for equilibrium, this definition can be used for any set of probabilities $w_a$, since it provides a useful measure of our ignorance about the system.

</div>

#### Equivalence of Ensembles

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Equivalence of Ensembles)</span></p>

Are canonical and microcanonical statistical descriptions equivalent? The descriptions are equivalent only when fluctuations are neglected and consideration is restricted to mean values. This takes place in thermodynamics, where the distributions just produce different fundamental relations: $S(E,N)$ for microcanonical, $F(T,N)$ for canonical, $\Omega(T,\mu)$ for grand canonical. These relations are related by the Legendre transforms.

As far as fluctuations are concerned, there is a natural hierarchy: microcanonical distribution neglects fluctuations in energy and number of particles, canonical distribution neglects fluctuations in $N$ but accounts for fluctuations in $E$. Grand canonical distribution accounts for fluctuations both in $E$ and $N$.

</div>

#### Energy Fluctuations

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Energy Fluctuations)</span></p>

The magnitude of fluctuations is determined by the **second derivative** of the respective thermodynamic potential. Expanding the probability distribution around the mean value using the second derivatives $\partial^2 S/\partial E^2$ and $\partial^2 S/\partial N^2$ (which must be negative for stability) gives Gaussian distributions of $E - \bar{E}$ and $N - \bar{N}$.

The energy variance is:

$$\overline{(E - \bar{E})^2} = -\frac{\partial\bar{E}}{\partial\beta} = T^2 C_V,$$

where $C_V$ is the heat capacity at constant volume. Since

$$\frac{\partial^2 S}{\partial E^2} = \frac{\partial}{\partial E}\frac{1}{T} = -\frac{1}{T^2}\frac{\partial T}{\partial E} = -\frac{1}{T^2 C_V},$$

the sharper the extremum (the higher the second derivative), the better the system parameters are confined to the mean values. Since both $\bar{E}$ and $C_V$ are proportional to $N$, the relative fluctuations are small: $\overline{(E - \bar{E})^2}/\bar{E}^2 \propto N^{-1}$.

</div>

### 1.5 Evolution in the Phase Space

We now focus on a broad class of energy-conserving systems that can be described by Hamiltonian evolution. Every such system is characterized by its momenta $p$ and coordinates $q$, together comprising the phase space. We define probability for a system to be in some $\Delta p \Delta q$ region of the phase space as the fraction $\Delta t$ of the total observation time $T$ it spends there: $w = \Delta t/T$. Introducing the statistical distribution in the phase space as density: $dw = \rho(p,q)dpdq$, the average with the statistical distribution is equivalent to the time average:

$$\bar{f} = \int f(p,q)\rho(p,q)\,dpdq = \frac{1}{T}\int_0^T f(t)\,dt.$$

#### Liouville Theorem

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Liouville Theorem)</span></p>

For not very long time, we can neglect interaction between subsystems, so that the motion can be described by the Hamiltonian dynamics of the subsystem itself: $\dot{q}_i = \partial\mathcal{H}/\partial p_i$ and $\dot{p}_i = -\partial\mathcal{H}/\partial q_i$. The evolution of the density is given by

$$\frac{\partial\rho}{\partial t} = \sum_i \frac{\partial\mathcal{H}}{\partial p_i}\frac{\partial\rho}{\partial q_i} - \frac{\partial\mathcal{H}}{\partial q_i}\frac{\partial\rho}{\partial p_i} \equiv \lbrace\rho,\mathcal{H}\rbrace.$$

Any Hamiltonian flow in the phase space is **incompressible**: it conserves area in each plane $p_i, q_i$ and the total volume: $\operatorname{div} \mathbf{v} = \partial\dot{q}_i/\partial q_i + \partial\dot{p}_i/\partial p_i = 0$. That gives the **Liouville theorem**: $d\rho/dt = \partial\rho/\partial t + (\mathbf{v}\nabla)\rho = -\rho\operatorname{div}\mathbf{v} = 0$. The statistical distribution is thus conserved along the phase trajectories of any subsystem.

</div>

As a result, at equilibrium $\rho$ must be expressed solely via the integrals of motion. Assuming short-range forces, different macroscopic subsystems interact weakly and are statistically independent so that the distribution for a composite system $\rho_{12}$ is factorized: $\rho_{12} = \rho_1\rho_2$. Since in equilibrium $\ln\rho$ is an additive quantity, then it must be expressed linearly via the additive integrals of motions which for a general mechanical system are momentum $\mathbf{P}(p,q)$, the momentum of momentum $\mathbf{M}(p,q)$ and energy $E(p,q)$:

$$\ln\rho_a = \alpha_a + \beta E_a(p,q) + \mathbf{c}\cdot\mathbf{P}_a(p,q) + \mathbf{d}\cdot\mathbf{M}(p,q).$$

Considering a subsystem which neither moves nor rotates, we are down to the single integral, energy, which corresponds to the Gibbs' canonical distribution:

$$\rho(p,q) = A\exp[-\beta E(p,q)].$$

The canonical equilibrium distribution corresponds to the maximum of the Gibbs entropy $S = -\int\rho\ln\rho\,dpdq$, under the condition of the given mean energy $\bar{E} = \int\rho(p,q)E(p,q)\,dpdq$. For an isolated system with a fixed energy, the entropy maximum corresponds to a uniform microcanonical distribution.

## 2. Appearance of Irreversibility

The most obvious contradiction we face is between distribution preservation by Hamiltonian evolution and the growth of its entropy. More generally, the puzzle here is how irreversible entropy growth appears out of reversible laws of mechanics. If we screen the movie of any evolution backwards, it will be a legitimate solution of the equations of motion. Will it have its entropy decreasing? Can we also decrease entropy by employing the Maxwell demon who can distinguish fast molecules from slow ones and selectively open a window between two boxes to increase the temperature difference?

These conceptual questions were posed already in the 19th century. It took the better part of the 20th century to answer them. This required two things: (i) better understanding dynamics and revealing the mechanism of randomization called dynamical chaos, (ii) consistent use of information theory which turned out to be just another form of statistical physics.

Irreversibility and relaxation to equilibrium essentially follows from necessity to consider ensembles (regions in phase space) due to incomplete knowledge. Initially small regions spread over the whole phase space under reversible Hamiltonian dynamics, very much like flows of an incompressible liquid are mixing. Such spreading and mixing in phase space correspond to the approach to equilibrium. On the contrary, to deviate a system from equilibrium, one adds external forcing and dissipation, which makes its phase flow compressible and distribution non-uniform. Difference between equilibrium and non-equilibrium distributions in phase space can then be expressed by the difference between incompressible and compressible flows.

### 2.1 Kinetic Equation and H-Theorem

How does the system come to equilibrium and reach the entropy maximum? The Hamiltonian evolution is an incompressible flow in the phase space ($\operatorname{div}\mathbf{v} = 0$), so it conserves the total Gibbs entropy: $dS/dt = -\int d\mathbf{x}\,\ln\rho\frac{\partial\rho}{\partial t} = \int d\mathbf{x}\,\ln\rho\,\operatorname{div}(\rho\mathbf{v}) = \int d\mathbf{x}\,\rho\operatorname{div}\mathbf{v} = 0$. How then can the entropy grow?

Boltzmann answered this question by deriving the equation on the one-particle momentum probability distribution. Such equation must follow from integrating the $N$-particle Liouville equation over all $N$ coordinates and $N-1$ momenta. Consider the phase-space probability density $\rho(\mathbf{x},t)$ in the space $\mathbf{x} = (\mathbf{P},\mathbf{Q})$, where 

$$\mathbf{P} = \lbrace\mathbf{p}_1\ldots\mathbf{p}_N\rbrace$$

and 

$$\mathbf{Q} = \lbrace\mathbf{q}_1\ldots\mathbf{q}_N\rbrace$$

For the system with the Hamiltonian 

$$\mathcal{H} = \sum_i p_i^2/2m + \sum_{i<j}U(\mathbf{q}_i - \mathbf{q}_j)$$

the evolution of the density is described by the Liouville equation:

$$\frac{\partial\rho(\mathbf{P},\mathbf{Q},t)}{\partial t} = \lbrace\rho(\mathbf{P},\mathbf{Q},t),\mathcal{H}\rbrace = \left[-\sum_i^N\frac{\mathbf{p}_i}{m}\frac{\partial}{\partial\mathbf{q}_i} + \sum_{i<j}\theta_{ij}\right]\rho(\mathbf{P},\mathbf{Q},t),$$

where

$$\theta_{ij} = \theta(\mathbf{q}_i,\mathbf{p}_i,\mathbf{q}_j,\mathbf{p}_j) = \frac{\partial U(\mathbf{q}_i - \mathbf{q}_j)}{\partial\mathbf{q}_i}\left(\frac{\partial}{\partial\mathbf{p}_i} - \frac{\partial}{\partial\mathbf{p}_j}\right)$$

has a meaning of an inverse typical time of the momentum change due to interaction. For a reduced description of the single-particle distribution over momenta $\rho(\mathbf{p},t) = \int \rho(\mathbf{P},\mathbf{Q},t)\delta(\mathbf{p}_1 - \mathbf{p})\,d\mathbf{p}_1\ldots d\mathbf{p}_N d\mathbf{q}_1\ldots d\mathbf{q}_N$, we integrate the Liouville equation. The terms with $\partial/\partial\mathbf{q}_i$ do not contribute, and we get:

$$\frac{\partial\rho(\mathbf{p},t)}{\partial t} = \int \delta(\mathbf{p}_1 - \mathbf{p})\theta(\mathbf{q}_1,\mathbf{p}_1;\mathbf{q}_2,\mathbf{p}_2)\rho(\mathbf{q}_1,\mathbf{p}_1;\mathbf{q}_2,\mathbf{p}_2)\,d\mathbf{q}_1 d\mathbf{p}_1 d\mathbf{q}_2 d\mathbf{p}_2.$$

This equation is not closed since the right-hand side contains the two-particle probability distribution. The consistent procedure is to assume short-range interaction and low density, so that the mean distance between particles much exceeds the radius of interaction. In this case we may assume that for every binary collision the particles come from large distances and their momenta are not correlated. Statistical independence allows one to replace the two-particle momenta distribution by the product of one-particle distributions.

#### Boltzmann Kinetic Equation

For a dilute gas, only two-particle collisions matter. Consider the collision of two particles having momenta $\mathbf{p}, \mathbf{p}_1$ that scatter into $\mathbf{p}', \mathbf{p}_1'$. We assume that the particle velocity is independent of the position and that the momenta of two particles are statistically independent: $\rho(\mathbf{p},\mathbf{p}_1) = \rho(\mathbf{p})\rho(\mathbf{p}_1)$. These very strong assumptions constitute the *hypothesis of molecular chaos*.

The rate of probability change is the difference between the number of particles coming and leaving the given region of phase space around $\mathbf{p}$:

$$\frac{\partial\rho}{\partial t} = \int (w'\rho'\rho_1' - w\rho\rho_1)\,d\mathbf{p}_1 d\mathbf{p}' d\mathbf{p}_1'.$$

The scattering probabilities $w \equiv w(\mathbf{p},\mathbf{p}_1;\mathbf{p}',\mathbf{p}_1')$ and $w' \equiv w(\mathbf{p}',\mathbf{p}_1';\mathbf{p},\mathbf{p}_1)$ are nonzero only for quartets satisfying conservation of energy and momentum. Assuming time-reversal invariance and spatial inversion invariance, all three symmetries combined give the **detailed balance**:

$$w \equiv w(\mathbf{p},\mathbf{p}_1;\mathbf{p}',\mathbf{p}_1') = w(\mathbf{p}',\mathbf{p}_1';\mathbf{p},\mathbf{p}_1) \equiv w'.$$

Using detailed balance, we obtain the **Boltzmann kinetic equation** (1872):

$$\frac{\partial\rho}{\partial t} = \int w'(\rho'\rho_1' - \rho\rho_1)\,d\mathbf{p}_1 d\mathbf{p}' d\mathbf{p}_1' \equiv I.$$

#### H-Theorem

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Boltzmann H-Theorem)</span></p>

The one-particle entropy $S = -\int \rho \ln\rho\,d\mathbf{p}$ is non-decreasing in time under the Boltzmann kinetic equation:

$$\frac{dS}{dt} = -\int \frac{\partial\rho}{\partial t}\ln\rho\,d\mathbf{p} = -\int I\ln\rho\,d\mathbf{p}.$$

By exploiting the symmetry of the collision integral under the interchanges $\mathbf{p}_1 \leftrightarrow \mathbf{p}$ and $\mathbf{p},\mathbf{p}_1 \leftrightarrow \mathbf{p}',\mathbf{p}_1'$:

$$\frac{dS}{dt} = \frac{1}{2}\int w'\rho\rho_1 \ln\frac{\rho\rho_1}{\rho'\rho_1'}\,d\mathbf{p} d\mathbf{p}_1 d\mathbf{p}' d\mathbf{p}_1' \ge 0,$$

where the inequality follows from $x\ln x - x + 1 \ge 0$ with $x = \rho\rho_1/\rho'\rho_1'$.

</div>

Even though we use scattering probabilities obtained from mechanics reversible in time, $w(-\mathbf{p},-\mathbf{p}_1;-\mathbf{p}',-\mathbf{p}_1') = w(\mathbf{p}',\mathbf{p}_1';\mathbf{p},\mathbf{p}_1)$, the use of the molecular chaos hypothesis made the kinetic equation irreversible. Equilibrium realizes the entropy maximum and so the distribution must be a steady solution of the Boltzmann equation. Indeed, the collision integral turns into zero by virtue of $\rho_0(\mathbf{p})\rho_0(\mathbf{p}_1) = \rho_0(\mathbf{p}')\rho_0(\mathbf{p}_1')$, since $\ln\rho_0$ is a linear function of the integrals of motion as was explained in Section 1.5.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Resolution of the Irreversibility Paradox)</span></p>

The full entropy of the $N$-particle distribution is conserved. Yet the one-particle entropy grows. Is there a contradiction? The answer is "no" to both questions. The resolution requires the central notion of *mutual information*. We broke time reversibility and set the arrow of time when we assumed particles uncorrelated *before* the collision and not after. Starting from uncorrelated particles, the interaction builds correlations. The total distribution changes, but the total entropy does not. Entropy lowering by correlations is compensated by the growth of the single-particle entropy described by the Boltzmann equation.

Motivation for choosing an uncorrelated initial state is that it is most likely in any generic ensemble. Running the Boltzmann equation backwards from a correlated state is statistically unlikely, since it requires momenta to be correlated in such a way that a definite state is produced after time $t$.

</div>

Going ahead of ourselves, we can say that neglecting inter-particle correlations by factorizing the two-particle distribution $\rho_{12} = \rho(\mathbf{q}_1,\mathbf{p}_1;\mathbf{q}_2,\mathbf{p}_2) = \rho_1\rho_2$ means using incomplete information. This naturally leads to a further increase of uncertainty, that is of entropy. For dilute gases, such a factorization is just the first term of the cluster expansion:

$$\rho_{12} = \rho_1\rho_2 + \int d\mathbf{q}_3 d\mathbf{p}_3\, J_{123}\rho_1\rho_2\rho_3 + \ldots$$

This so-called cluster expansion is well-defined only for equilibrium distributions. For non-equilibrium distributions, starting from some term (depending on the space dimensionality), all higher terms diverge. The same divergencies take place if one tries to apply the expansion to kinetic coefficients like diffusivity, conductivity or viscosity, which are non-equilibrium properties by their nature. These divergencies can be related to the fact that non-equilibrium distributions do not fill the phase space, as described in Section 2.4. Boltzmann equation looks nice, but corrections to it are ugly. The corrections also violate the H-theorem — indeed, dropping all the terms is part of passing from the Liouville equation to the Boltzmann equation and is what leads to the loss of information and entropy growth.

### 2.2 Phase-Space Mixing and Entropy Growth

We have seen that one-particle entropy can grow even when the full $N$-particle entropy is conserved. But thermodynamics requires the full entropy to grow. To accomplish that, let us return to the full $N$-particle distribution and recall that we have an incomplete knowledge of the system. That means that we always measure coordinates and momenta within some intervals, i.e. characterize the system not by a point in phase space but by a finite region there. We shall see that quite general dynamics stretches this finite domain into a very thin convoluted strip whose parts can be found everywhere in the available phase space. The dynamics thus provides a stochastic-like element of mixing in phase space that is responsible for the approach to equilibrium.

Yet by itself this stretching and mixing does not change the phase volume and entropy. Another ingredient needed is the necessity to continually treat our system with finite precision, which follows from the insufficiency of information. Such consideration is called *coarse graining* and, together with mixing, it is responsible for the irreversibility of statistical laws and for the entropy growth.

#### Dynamical Mechanism of Entropy Growth

The dynamical mechanism of the entropy growth is the separation of trajectories in phase space: trajectories started from a small neighborhood are found farther and farther away as time proceeds. Denote again by $\mathbf{x} = (\mathbf{P},\mathbf{Q})$ the $6N$-dimensional vector of the position and by $\mathbf{v} = (\dot{\mathbf{P}},\dot{\mathbf{Q}})$ the velocity in the phase space. The relative motion of two points, separated by $\mathbf{r}$, is determined by their velocity difference: $\delta v_i = r_j \partial v_i / \partial x_j = r_j \sigma_{ij}$.

We can decompose the tensor of velocity derivatives into an antisymmetric part (which describes rotation) and a symmetric part $S_{ij} = (\partial v_i/\partial x_j + \partial v_j/\partial x_i)/2$ (which describes deformation). According to the Liouville theorem, a Hamiltonian dynamics is an incompressible flow in the phase space, so that the trace of the tensor, which is the rate of the volume change, must be zero: $\operatorname{Tr}\sigma_{ij} = \sum_i S_i = \operatorname{div}\mathbf{v} = 0$. That means that some components are positive (rates of stretching) and some are negative (rates of contraction) in respective directions.

The equation for the distance between two points along a principal direction has the form $\dot{r}_i = \delta v_i = r_i S_i$, with the solution

$$r_i(t) = r_i(0)\exp\left[\int_0^t S_i(t')\,dt'\right].$$

For a time-independent strain, the growth/decay is exponential in time. A purely straining motion converts a spherical element into an ellipsoid with principal diameters that grow (or decay) in time.

#### Lyapunov Exponents

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Lyapunov Exponents)</span></p>

The net stretching and separation of trajectories is formally described by a random strain matrix $\hat{\sigma}(t)$ and the transfer matrix $\hat{W}$ defined by $\mathbf{r}(t) = \hat{W}(t,t_1)\mathbf{r}(t_1)$, which satisfies $d\hat{W}/dt = \hat{\sigma}\hat{W}$. The Liouville theorem $\operatorname{tr}\hat{\sigma} = 0$ means $\det\hat{W} = 1$.

The main result (Furstenberg and Kesten 1960; Oseledec, 1968) states that in almost every realization $\hat{\sigma}(t)$, the matrix $\frac{1}{t}\ln\hat{W}^T(t,0)\hat{W}(t,0)$ tends to a finite limit as $t \to \infty$. Its eigenvectors tend to $d$ fixed orthonormal eigenvectors $\mathbf{f}_i$. The limiting eigenvalues

$$\lambda_i = \lim_{t\to\infty} t^{-1}\ln|\hat{W}\mathbf{f}_i|$$

define the so-called **Lyapunov exponents**, which can be thought of as the mean stretching rates. The sum of the exponents is the mean volume growth rate, which is zero due to the Liouville theorem. As long as there is no special degeneracy, there exists at least one positive exponent which gives stretching.

</div>

Even when the average rate of separation along a given direction $\Lambda_i(t) = \int_0^t S_i(t')\,dt'/t$ is zero, the average exponent of it is larger than unity (and generally growing with time):

$$\lim_{t\to\infty}\int_0^t S_i(t')\,dt' = 0, \quad \left\langle\frac{r_i(t)}{r_i(0)}\right\rangle = \lim_{T\to\infty}\frac{1}{T}\int_0^T dt\exp\left[\int_0^t S_i(t')\,dt'\right] \ge 1.$$

This is because the intervals of time with positive $\Lambda(t)$ give more contribution into the exponent than the intervals with negative $\Lambda(t)$. That follows from the *concavity* of the exponential function.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Saddle-Point Flow)</span></p>

Consider the simplest two-dimensional pure strain, which corresponds to an incompressible saddle-point flow: $v_x = \lambda x$, $v_y = -\lambda y$. Here we have one expanding direction and one contracting direction, with equal rates. Since $x(t) = x_0\exp(\lambda t)$ and 

$$y(t) = y_0\exp(-\lambda t) = x_0 y_0/x(t)$$

every trajectory is a hyperbola.

A unit vector initially forming an angle $\varphi$ with the $x$ axis will have its length 

$$[\cos^2\varphi\exp(2\lambda T) + \sin^2\varphi\exp(-2\lambda T)]^{1/2}$$ 

after time $T$. The vector is stretched if 

$$\cos\varphi \ge [1 + \exp(2\lambda T)]^{-1/2} < 1/\sqrt{2}$$

i.e. the fraction of stretched directions is larger than half. When along the motion all orientations are equally probable, the net effect is stretching, increasing with the persistence time $T$.

</div>

As time increases, the ellipsoid is more and more elongated and it is less and less likely that the hierarchy of the ellipsoid axes will change. The probability to find a ball turning into an exponentially stretching ellipse thus goes to unity as time increases. To reverse it, one needs to contract the long axis of the ellipse — the direction of contraction must be inside the narrow angle defined by the ellipse eccentricity, which is less likely than being outside the angle.

#### Coarse Graining and the Second Law

Armed with the understanding of the exponential stretching, we now return to the dynamical foundation of the second law of thermodynamics. We assume that our finite resolution does not allow us to distinguish between the states within some square in the phase space. That square is our "grain" in coarse-graining.

A black square of initial conditions is stretched in one (unstable) direction and contracted in another (stable) direction so that it turns into a long narrow strip. Later in time, our resolution is still restricted — rectangles in the right box show finite resolution (this is coarse-graining). Viewed with such resolution, our set of points occupies larger phase volume at $t = \pm T$ than at $t = 0$. Larger phase volume corresponds to larger entropy.

*Time reversibility of any trajectory* in the phase space does not contradict the *time-irreversible filling of the phase space by the set of trajectories* considered with a finite resolution. By reversing time we exchange stable and unstable directions (i.e. those of contraction and expansion), but the fact of space filling persists.

#### Entropy Growth Rate and Kolmogorov-Sinai Entropy

When the density spreads, entropy grows (as the logarithm of the volume occupied). If initially our system was within the phase-space volume $\epsilon^{6N}$, then its density was $\rho_0 = \epsilon^{-6N}$ inside and zero outside. After stretching to some larger volume $e^{\lambda t}\epsilon^{6N}$ the entropy $S = -\int\rho\ln\rho\,d\mathbf{x}$ has increased by $\lambda t$. The positive Lyapunov exponent $\lambda$ determines the rate of the entropy growth.

If in a $d$-dimensional space there are $k$ stretching and $d-k$ contracting directions, then contractions eventually stabilize at the resolution scale, while expansions continue. Therefore, the volume growth rate is determined by the sum of the positive Lyapunov exponents $\sum_{i=1}^k \lambda_i$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Kolmogorov-Sinai Entropy)</span></p>

The acquisition rate of information about the past of trajectories is the sum of the positive Lyapunov exponents and is called the **Kolmogorov-Sinai entropy**. As time lag from the present moment increases, we can say less and less where we shall be and more and more where we came from. It illustrates Kierkegaard's remark that the irony of life is that it is lived forward but understood backwards.

</div>

#### Mixing and Ergodicity

After the strip length reaches the scale of the velocity change (when one already cannot approximate the phase-space flow by a linear profile $\hat{\sigma}r$), the strip starts to fold. Still, the strip continues locally the exponential stretching. Eventually, one can find the points from the initial ball everywhere, which means that the flow is mixing, also called **ergodic**.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Ergodicity)</span></p>

The formal definition is that the flow is called ergodic in the domain if the trajectory of almost every point (except possibly a set of zero volume) passes arbitrarily close to every other point. An equivalent definition is that there are no finite-volume subsets of the domain invariant with respect to the flow except the domain itself. Ergodic flow on an energy surface in the phase space provides for a micro-canonical distribution (i.e. constant), since time averages are equivalent to the average over the surface.

</div>

At even larger time scales than the time of the velocity change for a trajectory, one can consider the motion as a series of uncorrelated random steps. That produces random walk, where the spread of the probability density $\rho(\mathbf{r},t)$ is described by a simple diffusion: $\partial\rho/\partial t = \kappa\Delta\rho$. The total probability $\int\rho(\mathbf{r},t)\,d\mathbf{r}$ is conserved but the entropy increases monotonically under diffusion:

$$\frac{dS}{dt} = -\frac{d}{dt}\int\rho(\mathbf{r},t)\ln\rho(\mathbf{r},t)\,d\mathbf{r} = -\kappa\int\Delta\rho\ln\rho\,d\mathbf{r} = \kappa\int\frac{(\nabla\rho)^2}{\rho}\,d\mathbf{r} \ge 0.$$

Asymptotically in time the solution of the diffusion equation takes the universal form $\rho(\mathbf{r},t) = (4\pi\kappa t)^{-d/2}\exp(-r^2/4\kappa t)$; substituting gives a universal entropy production rate $dS/dt = 1/2t$, independent of $\kappa$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Determinism vs Dynamical Chaos)</span></p>

Two concluding remarks are in order. First, the notion of an exponential separation of trajectories puts an end to the old dream of Laplace to be able to predict the future if only all coordinates and momenta are given. Even if we were able to measure all relevant phase-space initial data, we can do it only with a finite precision $\epsilon$. However small is the indeterminacy in the data, it is amplified exponentially with time so that eventually $\epsilon\exp(\lambda T)$ is large and we cannot predict the outcome. Mathematically speaking, limits $\epsilon \to 0$ and $T \to \infty$ do not commute. Second, the above arguments did not use the usual mantra of the thermodynamic limit, which means that even systems with a small number of degrees of freedom need statistics for their description at long times if their dynamics has a positive Lyapunov exponent (which is generic) — this is sometimes called *dynamical chaos*.

</div>

### 2.3 Baker Map

One can think of any Hamiltonian dynamics as a map of phase space into itself. We consider a toy model of such a map, which is of great illustrative value for the applications of chaos theory to statistical mechanics.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Baker Map)</span></p>

Take the phase-space to be a unit square in the $(x,y)$-plane, with $0 < x, y < 1$. The measure-preserving transformation is an expansion in the $x$-direction and a contraction in the $y$-direction, arranged in such a way that the unit square is mapped onto itself at each step. The transformation consists of two steps:
1. The unit square is contracted in the $y$-direction and stretched in the $x$-direction by a factor of 2. The unit square becomes a rectangle occupying the region $0 < x < 2$; $0 < y < 1/2$.
2. The rectangle is cut in the middle and the right half is put on top of the left half to recover a square.

This transformation is reversible except on the lines where the area was cut in two and glued back.

</div>

If we consider two initially closed points, then after $n$ such steps the distance along $x$ and $y$ will be multiplied respectively by $2^n = e^{n\ln 2}$ and $2^{-n} = e^{-n\ln 2}$. There are two Lyapunov exponents corresponding to the discrete time $n$: $\lambda_+ = \ln 2$ (expanding direction, along $x$-axis) and $\lambda_- = -\ln 2$ (contracting direction, along $y$-axis). The sum of the Lyapunov exponents is zero, reflecting the fact that the baker's transformation is area-preserving.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Baker Map is Mixing)</span></p>

The baker transformation is mixing, that is spreading the measure uniformly over the whole phase space. Indeed, if a measure is initially concentrated in any domain, after sufficiently long time the domain is transformed into a large number of very thin horizontal strips of length unity, distributed more and more uniformly in the vertical direction. Eventually any set in the unit square will have the same fraction of its area occupied by these little strips of pasta as any other set. This is the indicator of a mixing system. If we add coarse-graining at a small scale $\epsilon$, at sufficiently long time it blurs our measure to a constant one. We conclude that a sufficiently smooth initial distribution function defined on the unit square will approach a uniform (microcanonical) distribution on the square.

</div>

#### Toral Map

To avoid the impression that cutting and gluing of the baker map are necessary for mixing, consider a smooth model which has similar behavior. Consider a unit two-dimensional torus, that is a unit square with periodic boundary conditions, so that all distances are measured modulo 1 in the $x$- and $y$-direction. The transformation matrix $T$ (an analog of the transfer matrix $\hat{W}$) maps unit torus into itself if $a, b, c, d$ are all integers:

$$\begin{pmatrix} x' \\ y' \end{pmatrix} = T \cdot \begin{pmatrix} x \\ y \end{pmatrix} \pmod{1}, \quad T = \begin{pmatrix} a & b \\ c & d \end{pmatrix}.$$

The eigenvalues $\lambda_{1,2} = (a+d)/2 \pm \sqrt{(a-d)^2/4 + bc}$ are real when $(a-d)^2/4 + bc \ge 0$, in particular when the matrix is symmetric. For the transform to be area-preserving, the determinant of the matrix $T$ must be unity: $\lambda_1\lambda_2 = ad - bc = 1$. In a general case, one eigenvalue is larger than unity and one is smaller, which corresponds respectively to positive and negative Lyapunov exponents $\ln\lambda_1$ and $\ln\lambda_2$.

Baker map is area-preserving and does not change entropy, yet when we allow for repeating coarse-graining along with the evolution, then the entropy grows and eventually reaches the maximum, which is the logarithm of the phase volume, which corresponds to the equilibrium microcanonical distribution.

### 2.4 Entropy Decrease and Non-Equilibrium Fractal Measures

As we have seen in the previous two sections, if we have indeterminacy in the data or consider an ensemble of systems, then Hamiltonian dynamics (an incompressible flow) effectively mixes and makes distribution uniform in the phase space. Since we have considered isolated systems, they conserve their integrals of motion, so that the distribution is uniform over the respective surface. In particular, dynamical chaos justifies micro-canonical distribution, uniform over the energy surface.

But what if the dynamics is non-Hamiltonian, that is Liouville theorem is not valid? The flow in the phase space is then generally compressible. The simplest non-conservative effect is dissipation of kinetic energy, which shrinks all momenta and thus decreases the phase volume. We are interested, however, in a non-equilibrium steady state where we keep the energy non-decreasing. To compensate for the loss of the momentum of the particles with the dissipation rates $\gamma_i$, we act on them by external forces $f_i$, so that the equations of motion take the form: $\dot{p}_i = f_i - \gamma_i p_i - \partial H/\partial q_i$, $\dot{q}_i = \partial H/\partial p_i$, which gives generally $\operatorname{div}\mathbf{v} = \sum_i(\partial f_i/\partial p_i - \gamma_i) \neq 0$.

Since $\operatorname{div}\mathbf{v} \neq 0$, the probability density generally changes along a flow: $d\rho/dt = -\rho\operatorname{div}\mathbf{v}$. That produces entropy:

$$\frac{dS}{dt} = \int\rho(\mathbf{r},t)\operatorname{div}\mathbf{v}(\mathbf{r},t)\,d\mathbf{r} = \langle\operatorname{div}\mathbf{v}\rangle,$$

with the rate equal to the Lagrangian mean of the phase-volume local expansion rate. If the system does not on average heat or cool (expand or contract), then the whole phase volume does not change, meaning that the volume integral of the local expansion rate is zero: $\int\operatorname{div}\mathbf{v}\,d\mathbf{r} = 0$. Yet for a non-uniform density, the entropy is not the log of the phase volume but the minus *mean* log of the phase density, $S(t) = -\langle\ln\rho\rangle = -\int\rho(\mathbf{r},t)\ln\rho(\mathbf{r},t)\,d\mathbf{r}$, whose derivative is non-zero because of correlations between $\rho$ and $\operatorname{div}\mathbf{v}$. Since $\rho$ is always smaller in the expanding regions where $\operatorname{div}\mathbf{v} > 0$, then *the entropy production rate is non-positive*. The entropy decreases, meaning that the distribution is getting more non-uniform.

#### Density Evolution in Compressible Flows

The density of an arbitrary fluid element evolves as:

$$\rho(t)/\rho(0) = \exp\left[-\int_0^t \operatorname{div}\mathbf{v}(t')\,dt'\right] = e^{C(t)}.$$

If a mean is zero, the mean exponent generally exceeds unity because of the concavity of the exponential function. The contraction factor averaged over the whole flow is zero at any time, $\langle C \rangle = 0$, and its average exponent is larger than unity:

$$\langle\rho(t)/\rho(0)\rangle = \lim_{T\to\infty}\frac{1}{T}\int_0^T dt\exp\left[-\int_0^t \operatorname{div}\mathbf{v}(t')\,dt'\right] = \langle e^C \rangle > 1.$$

For a generic random flow the density of most fluid elements must grow non-stop as they move. Since the total measure is conserved, growth of density at some places must be compensated by its decrease in other places, so that the distribution is getting more and more non-uniform, which decreases the entropy. Looking at the phase space one sees it more and more emptied with the density concentrated asymptotically in time on a fractal set. That is opposite to the mixing by Hamiltonian incompressible flow.

The long-time Lagrangian average (along the flow) of the volume change rate is the sum of the Lyapunov exponents, which is then non-positive:

$$\left\langle\frac{dS}{dt}\right\rangle = \langle\operatorname{div}\mathbf{v}\rangle = \lim_{t\to\infty}\frac{1}{t}\int_0^t \operatorname{div}\mathbf{v}(t')\,dt' = \sum_i \lambda_i.$$

It is important that we allowed for a compressibility of a phase-space flow $\mathbf{v}(\mathbf{r},t)$ but did not require its irreversibility. Indeed, even if the system is invariant with respect to $t \to -t$, $\mathbf{v} \to -\mathbf{v}$, the entropy production rate is generally non-negative and the sum of the Lyapunov exponents is non-positive for the same simple reason that contracting regions have more measure and give higher contributions. Backwards in time the measure also concentrates, only on a different set.

#### Generalized Baker Map and Fractal Dimension

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Generalized Baker Map)</span></p>

This can be illustrated by a slight generalization of the baker map, expanding one region and contracting another, keeping the whole volume of the phase space unity. The transformation has the form

$$x' = \begin{cases} x/l & \text{for } 0 < x < l \\ (x-l)/r & \text{for } l < x < 1 \end{cases}, \quad y' = \begin{cases} ry & \text{for } 0 < x < l \\ r + ly & \text{for } l < x < 1 \end{cases},$$

where $r + l = 1$. The Jacobian of the transformation is not identically equal to unity when $r \neq l$:

$$J = \left|\frac{\partial(x',y')}{\partial(x,y)}\right| = \begin{cases} r/l & \text{for } 0 < x < l \\ l/r & \text{for } l < x < 1 \end{cases}.$$

If $l > 1/2$, then $r = 1 - l < l$, so that $J < 1$ in the shadowed region where $x < l$, and $J > 1$ in the white region where $x > l$. The mean Jacobian $\bar{J} = r + l$ is of course unity.

If during $n$ steps the points find themselves $n_1$ times in the region $0 < x < l$ and $n_2 = n - n_1$ times inside $l < x < 1$ then the distances along $x$ and $y$ will be multiplied respectively by $l^{-n_1}r^{-n_2}$ and $r^{n_1}l^{n_2}$. Taking the log and the limit we obtain the Lyapunov exponents:

$$\lambda_+ = -l\ln l - r\ln r, \quad \lambda_- = r\ln l + l\ln r.$$

The sum of the Lyapunov exponents, $\lambda_+ + \lambda_- = (l-r)\ln(r/l)$, is non-positive and is zero only for $l = r = 1/2$. The volume contraction means that the expansion in the $\lambda_+$-direction proceeds slower than the contraction in the $\lambda_-$-direction. Asymptotically the strips of pasta concentrate on a fractal set.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Box-Counting Dimension)</span></p>

Define the (box-counting) dimension of a set as follows:

$$d_f = \lim_{\epsilon\to 0}\frac{\ln N(\epsilon)}{\ln(L/\epsilon)},$$

where $N(\epsilon)$ is the number of boxes of length $\epsilon$ on a side needed to cover the set of size $L$.

</div>

After $n$ iterations of the map, a square having initial side $\delta \ll L$ will be stretched into a long thin rectangle of length $\delta\exp(n\lambda_+)$ and width $\delta\exp(n\lambda_-)$. To cover the contracting direction, we choose $\epsilon = \delta\exp(n\lambda_-)$, then $N(\epsilon) = \exp[n(\lambda_+ - \lambda_-)]$, so that the dimension is

$$d_f = 1 + \frac{\lambda_+}{|\lambda_-|}.$$

Since $\|\lambda_-\| \ge \lambda_+$, the dimension is between 1 and 2. The set is smooth in the $x$-direction and fractal in the $y$-direction, which respectively gives two terms in the formula.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Kaplan-Yorke Conjecture)</span></p>

The general (Kaplan-Yorke) conjecture is that $d_f = j + \sum_{i=1}^j \lambda_i/\lambda_{j+1}$, where $j$ is the largest number for which $\sum_{i=1}^j \lambda_i \ge 0$ and $\sum_{i=1}^{j+1} \lambda_i < 0$. For incompressible flows, $j = d$.

</div>

Fractalization of the measure proceeds until the coarse-graining stops it. In distinction from the incompressible flow, coarse-graining at a small scale $\epsilon$ does not make the distribution uniform, but it makes the entropy finite: $S = \ln N(\epsilon)$. An equilibrium uniform (microcanonical) distribution in $d$-dimensional phase space has the entropy $S_0 = d\ln(L/\epsilon)$; the non-equilibrium steady state (NESS) generally has a lower dimensionality $d_f < d$ with a lower entropy, $S_0 \simeq d_f\ln(L/\epsilon)$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Entropy Growth vs Entropy Decay)</span></p>

Thus for dynamical systems, both temporal and spatial properties of the entropy are determined by the Lyapunov exponents. Entropy dependence on time (both forward and backward) is governed by the Kolmogorov-Sinai entropy, which is the sum of the positive Lyapunov exponents. Entropy dependence on spatial resolution is determined by the dimensionality.

The dramatic difference between equilibrium equipartition and non-equilibrium fractal distribution: relation between compressibility and non-equilibrium is natural. To make a system non-Hamiltonian one needs to act by some external forces, which pump energy into some degrees of freedom and, to keep a steady state, absorb it from other degrees of freedom — expansion and contraction of the momentum part of the phase-space volume. Long-time average volume contraction of a fluid element and respective entropy decay is the analog of the second law of thermodynamics: to deviate a system from equilibrium, one needs to constantly lower its entropy until the resolution limit is reached.

</div>

## 3. Physics of Information

This Chapter presents an elementary introduction into information theory from the viewpoint of a natural scientist. It re-tells the story of statistical physics using a different language, which lets us see the Boltzmann and Gibbs entropies in a new light. Here we switch from continuous thinking in terms of phase-space flows to discrete combinatoric manipulations. What is attractive about the information viewpoint is that it erases paradoxes and makes the second law of thermodynamics trivial. It also allows us to see generality and commonality in the approaches (to partially known systems) of physicists, engineers, computer scientists, biologists, brain researchers, social scientists, market speculators, spies and flies. The same tools used in setting limits on thermal engines are used in setting limits on communications, measurements and learning (which are essentially the same phenomena). The main mathematical tool exploits universality appearing upon summing many independent random numbers.

The central idea is that information lowers uncertainty. A convenient way to quantify it is by the number of questions whose answers together eliminate the uncertainty. If we are uncertain about the events with a priori equal probabilities, the number of such questions is a logarithm of the number $n$ of possible outcomes, which is the Boltzmann entropy. One needs $\log_2 n$ of yes-no questions to locate one out of $n$ equally probable objects. If we know the probabilities $p_i$ of the events, we find the information rate per answer on average to be equal to the Gibbs entropy, $S = -\sum_i p_i \log_2 p_i$ bits. That follows from the fact that the number of typical $N$-sequences of outcomes grows with $N$ as $2^{NS}$, so that any such sequence brings $\log_2 2^{NS} = NS$ bits, that is $S$ bits per outcome on average.

But what if the answers are not completely reliable? In other words, we have an imperfect channel whose output $A$ specifies the event (input) $B_j$ not completely, but with some remaining uncertainty, which is characterized by the conditional entropy $S(B\mid A)$. The information received is then equal to $I(A,B) = S(B) - S(B\mid A)$ called the mutual information. Next Chapter will describe how fast widens the region of applications of the universal notions of entropy and mutual information: from physics, communications and computations to brain research, artificial intelligence and quantum computing.

### 3.1 Information as a Choice

> *"Information is the resolution of uncertainty."* — C. Shannon, 1948

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Information Content)</span></p>

We want to know in which of $n$ boxes a candy is hidden, that is we are faced with a choice among $n$ equal possibilities. How much information we need to get the candy? Let us denote the missing information by $I(n)$. Clearly, $I(1) = 0$, and we want the information to be a monotonically increasing function of $n$. If we have several independent problems then information must be additive. For example, consider each box to have $m$ compartments. To know in which from $mn$ compartments is the candy, we need to know first in which box and then in which compartment inside the box: $I(nm) = I(n) + I(m)$. Now, we can write (Fisher 1925, Hartley 1927, Shannon 1948):

$$I(n) = I(e)\ln n = k\ln n.$$

If we measure information in binary choices or **bits** (abbreviation of "binary digits"), then $I(n) = \log_2 n$, that is $k^{-1} = \ln(2)$. To arrive at a destination via the road with $N$ forks one needs $N$ bits, while via a street with $M$ intersections $M\log_2 3$ bits, since there are three possible ways at an intersection.

</div>

We can think of information received through words and symbols. If we have an alphabet with $n$ symbols then every symbol we receive is a choice out of $n$ and brings the information $k\ln n$. If symbols come independently then the message of length $N$ can potentially be one of $n^N$ possibilities so that it brings the information $kN\ln n$. To convey the same information by a smaller alphabet, one needs a longer message.

If all the 26 letters of the English alphabet were used with the same frequency then the word "love" would bring the information equal to $4\log_2 26 \approx 4 \cdot 4.7 = 18.8$ bits. Here and below we assume that the receiver has no other prior knowledge on subjects like correlations between letters.

#### Average Information and Gibbs-Shannon Entropy

In reality, every letter brings on average even less information than $\log_2 26$ since we *know* that letters are used with different frequencies. Consider the situation when there is a probability $p_i$ assigned to each letter (or box) $i = 1,\ldots,n$. Different letters bring different information. Let us evaluate the *average* information per symbol in a long message. To average, we consider the limit $N \to \infty$, then we know that the $i$-th letter appears $Np_i$ times *in a typical sequence*. What any message of length $N$ brings is the order in which different symbols appear. The total number of different typical sequences is equal to $N!/\Pi_i(Np_i)!$, and the information that we obtained from a string of $N$ symbols is the logarithm of that number:

$$I_N = k\ln\frac{N!}{\Pi_i(Np_i)} \approx k\left(N\ln N - \sum_i Np_i\ln Np_i\right) = -Nk\sum_i p_i\ln p_i.$$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Gibbs-Shannon Entropy)</span></p>

The mean information per symbol coincides with the Gibbs entropy:

$$S(p_1\ldots p_n) = \lim_{N\to\infty} I_N/N = -k\sum_{i=1}^n p_i\ln p_i.$$

</div>

Alternatively, one can derive this without any mention of randomness. Consider again $n$ boxes and define $p_i = m_i/\sum_{i=1}^n m_i = m_i/M$, where $m_i$ is the number of compartments in the box number $i$. When each compartment can be chosen independently of the box it is in, the $i$-th box is chosen with the frequency $p_i$, that is a given box is chosen more frequently if it has more compartments. The information on a specific compartment is a choice out of $M$ and brings information $k\ln M$. That information must be a sum of the information about the box $I_n$ plus the information about the compartment, $k\sum_{i=1}^n p_i\ln m_i$, summed over the boxes. That gives the information $I_n$ about the box (letter) as the difference:

$$I_n = k\ln M - k\sum_{i=1}^n p_i\ln m_i = k\sum_{i=1}^n p_i\ln M - k\sum_{i=1}^n p_i\ln m_i = -k\sum_{i=1}^n p_i\ln p_i = S.$$

A little more formally, one can prove that the entropy is the only measure of uncertainty that is a continuous function of $p_i$, symmetric with respect to their transmutations, and satisfies the inductive relation:

$$S(p_1, p_2, p_3\ldots p_n) = S(p_1 + p_2, p_3\ldots p_n) + (p_1 + p_2)S\left(\frac{p_1}{p_1 + p_2}, \frac{p_2}{p_1 + p_2}\right).$$

That relation comes from considering a subdivision: first, receive the information whether one of the first two possibilities appeared, second, distinguish between 1 and 2.

#### Asymptotic Equipartition

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Asymptotic Equipartition Property)</span></p>

What if we look at the given sequence of symbols $y_1,\ldots,y_N$ and ask: how probable it is? If the sequence is long enough and the symbols are independently chosen, then for a *typical* sequence the law of large numbers gives:

$$\frac{1}{N}\ln P(y_1,\ldots,y_N) \to -\sum_{j=1}^n p(y^j)\ln p(y^j) = -\langle\ln p(y)\rangle = S(Y).$$

This means that the log of the probability converges to $N$ times the entropy of $y$. The probability of the typical sequence thus decreases with $N$ exponentially: $P(y_1,\ldots,y_N) = \exp[-NS(y)]$. That probability is independent of the values $y_1,\ldots,y_N$, that is the same for all typical sequences — we found the best uniform (microcanonical!) distribution approximating $P(y_1,\ldots,y_N)$.

Equivalently, the number of typical sequences grows with $N$ exponentially, and the entropy sets the rate of growth. That focus on typical sequences, which all have the same (maximal) probability, is known as **asymptotic equipartition** and formulated as "almost all events are almost equally probable".

</div>

In physics, asymptotic equipartition is used, for instance, when we claim that the Boltzmann entropy is equivalent to the Gibbs entropy for systems whose energy is separable into independent parts in the thermodynamic limit. Like we argued for equivalence of energies in Section 1.4, we consider the microcanonical distribution taken at the energy equal to the mean energy of the canonical distribution (the typical set of the canonical ensemble). Then the Boltzmann entropy of such microcanonical distribution is equal to the entropy of the canonical distribution in the thermodynamic limit.

#### Convexity and the Entropy Inequality

The entropy is zero for a delta-distribution $p_i = \delta_{ij}$; it is generally less than the information $I(n) = \ln n$ and coincides with it only for equal probabilities $p_i = 1/n$, when the entropy is maximum. Indeed, equal probabilities we ascribe when there is no extra information, i.e. in a state of maximum ignorance.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Entropy Convexity)</span></p>

Mathematically, the property

$$S(1/n,\ldots,1/n) \ge S(p_1\ldots p_n)$$

is called **convexity**. It follows from the fact that the function of a single variable $s(p) = -p\ln p$ is strictly **concave** since its second derivative, $-1/p$, is everywhere negative for positive $p$. For any concave function, the average over the set of points $p_i$ is less or equal to the function at the average value (so-called Jensen inequality):

$$\frac{1}{n}\sum_{i=1}^n s(p_i) \le s\left(\frac{1}{n}\sum_{i=1}^n p_i\right).$$

From here one gets the entropy inequality:

$$S(p_1\ldots p_n) = \sum_{i=1}^n s(p_i) \le ns\left(\frac{1}{n}\sum_{i=1}^n p_i\right) = ns\left(\frac{1}{n}\right) = S\left(\frac{1}{n},\ldots,\frac{1}{n}\right).$$

</div>

Note that the Boltzmann information $I(n) = \ln n$ corresponds to the microcanonical Boltzmann entropy as a logarithm of the number of states, while the Gibbs-Shannon entropy $S = -\sum p_i \ln p_i$ corresponds to the canonical Gibbs entropy giving it as an average. An advantage of Gibbs-Shannon entropy is that it is defined for arbitrary distributions, not necessarily equilibrium.

### 3.2 Communication Theory

Here we start learning how to treat everything as a message. After we learnt what information messages bring on average, we are ready to discuss the best ways to transmit them. Communication Theory is interested in two key issues, speed and reliability:

i) How much can a message be compressed; i.e., how redundant is the information? In other words, what is the maximal rate of transmission in bits per symbol?

ii) At what rate can we communicate reliably over a noisy channel; i.e., how much redundancy must be incorporated into a message to protect against errors?

Both questions concern redundancy — how unexpected is every letter of the message, on the average. Entropy quantifies redundancy. A communication channel transmitting independent symbols on average transmits one unit of the information $S$ per symbol. Receiving letter (box) number $i$ through a binary channel brings information $\log_2(1/p_i) = \log_2 M - \log_2 m_i$ bits. The information measures the degree of surprise: less frequent events are more surprising. The entropy $-\sum_{i=a}^z p_i\log_2 p_i$ is the mean information content per letter. Less probable symbols bring larger information content, but they happen more rarely.

#### Source Coding (Data Compression)

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Shannon Source Coding Theorem)</span></p>

Consider for simplicity a message of $N$ bits, where 0 comes with probability $1-p$ and 1 with probability $p$. To compress the message to a shorter string that conveys essentially the same information it suffices to choose a code that treats effectively the *typical* strings — those that contain $N(1-p)$ zeroes and $Np$ ones. The number of such strings is given by the binomial $C_{Np}^N$ which for large $N$ is $2^{NS(p)}$, where $S(p) = -p\log_2 p - (1-p)\log_2(1-p)$.

To distinguish between these $2^{NS(p)}$ messages, we encode any one using a binary string with lengths starting from one and ending at $NS(p)$ bits. The maximal word length $NS(p)$ is less than $N$, since $0 \le S(p) \le 1$ for $0 \le p \le 1$. In other words, to encode all $2^N$ sequences we need words of $N$ bits, but to encode all typical sequences, we need only words up to $NS(p)$ bits. We indeed achieve compression with the sole exception of the case of equal probability where $S(1/2) = 1$.

Entropy sets both the mean and the maximal rate in the limit of long sequences. It gives the transfer rate of information when all the redundancy has been squeezed out.

</div>

The notion of typical messages in the limit $N \to \infty$ is an information-theory analog of ensemble equivalence in the thermodynamic limit.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Mean Codeword Length Bounds)</span></p>

Maximal rate of transmission corresponds to the shortest mean length of the codeword. If we encode $n$ objects by an alphabet with $q$ symbols, the mean codeword cannot be shorter than $\log n / \log q = \log_q n$. If we know that the source has the probability distribution $p(i)$, $i = 1,\ldots,n$, then we can use this information to shorten the mean codeword thus increasing the rate. Shannon proved that the shortest mean length of the codeword $\ell$ is bounded by:

$$-\sum_i p(i)\log_q p(i) \le \ell < -\sum_i p(i)\log_q p(i) + 1.$$

</div>

#### Huffman Coding

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Huffman Code for DNA Bases)</span></p>

Not any encoding guarantees the shortest mean codeword and the maximal rate of transmission. Designating sequences of the same length to objects with different probabilities is apparently sub-optimal. Consider a fictional creature whose DNA contains four bases A, T, C, G occurring with probabilities $p_i$ listed in the table:

| Symbol | $p_i$ | Code 1 | Code 2 |
| --- | --- | --- | --- |
| A | 1/2 | 00 | 0 |
| T | 1/4 | 01 | 10 |
| C | 1/8 | 10 | 110 |
| G | 1/8 | 11 | 111 |

Code 1 has exactly 4 two-bit words and uses 2 bits for every base. However, the entropy of the distribution $S = -\sum_{i=1}^4 p_i\log_2 p_i = 7/4$ is lower than 2. One then may suggest a variable-length Code 2. It is built by starting from the least probable C and G, which have the longest codewords of the same length differing by one (last) binary digit. We then combine C and G into a single source symbol with probability 1/4, coinciding with the probability of T. To distinguish from C,G, we code T by two-bit word placing 0 in the second position. The combined $C,G$ is now encoded 11, while T is encoded 10. We then can code A by one-bit word 0 to distinguish it from the combined T,C,G.

The mean length of Code 2 is exactly equal to the entropy: $(1/2)\cdot 1 + (1/4)\cdot 2 + (1/4)\cdot 3 = 7/4$. This is an example of the **Huffman code**, which draws a binary tree starting from its leaves: first, ascribe to the two least probable symbols two longest codewords differing in the last digit; second, combine these two symbols into a single one and repeat. The procedure ends after $n-1$ steps. Such codes are prefix-free: no codeword can be mistaken for the beginning of another one.

</div>

The most efficient code has the mean codeword length (the number of bits per base) equal to the entropy of the distribution, which determines the fastest mean transmission rate, that is the shortest mean codeword length.

#### Redundancy in Natural Languages

The inequality $S(1/n,\ldots,1/n) \ge S(p_1\ldots p_n)$ tells us, in particular, that using an alphabet is not optimal for the speech transmission rate as long as the probabilities of the letters are different. For example, if we use 26 letters, space and 5 punctuation marks (.!?-), we need 5-bit words to encode these 32 symbols. We can use fewer symbols but variable codeword length to make the average codeword shorter than 5. Morse code uses just three symbols (dot, dash and space) to encode any language. In English, the probability of "E" is 13% and of "Q" is 0.1%, so Morse encodes "E" by a single dot and "Q" by "$- - \cdot -$". One-letter probabilities give for the written English language the information per symbol as follows:

$$-\sum_{i=a}^z p_i\log_2 p_i \approx 4.11 \text{ bits},$$

which is lower than $\log_2 26 = 4.7$ bits.

### 3.3 Correlations in the Signals

The first British telegraph managed to do without C, J, Q, U, X, which tells us that some letters can be guessed from their neighbors, and more generally that there is a correlation between letters. Apart from one-letter probabilities, one can utilize more knowledge about the language by accounting for two-letter correlation (say, that "Q" is almost always followed by "U", "H" often follows "T", etc). That will further lower the entropy.

#### Markov Sources

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Markov Chain and Stochastic Matrix)</span></p>

A simple universal model with neighboring correlations is a Markov chain. It is specified by the conditional probability $p(j\mid i)$ that the letter $i$ is followed by $j$. For example $p(U\mid Q) = 1$. The probability is normalized for every $i$: $\sum_j p(j\mid i) = 1$. The matrix $p_{ij} = p(j\mid i)$, whose elements are positive and in every column sum to unity, is called **stochastic**.

Do the vector of probabilities $p(i)$ and the transition matrix $p_{ij}$ bring independent information? The answer is no, because the matrix $p_{ij}$ and the vector $p_i$ are not independent, but are related by the condition of stationarity: $p(i) = \sum_j p(j) p_{ji}$, that is $\mathbf{p} = \lbrace p(a),\ldots p(z)\rbrace$ is an eigenvector with the unit eigenvalue of the matrix $p_{ij}$.

</div>

The probability of any $N$-string is then the product of $N-1$ transition probabilities times the probability of the initial letter. As in the asymptotic equipartition case, minus the logarithm of the probability of a long $N$-string is a sum of uncorrelated numbers:

$$\log_2 p(i_1,\ldots,i_N) = \log_2 p(i_1) + \sum_{k=2}^N \log_2 p(i_{k+1}\mid i_k).$$

At large $N$ the sum grows linearly with $N$ with the rate, which is the mean value of the logarithm of conditional probability, $-\sum_j p(j\mid i)\log_2 p(j\mid i) = S_i$, called the conditional entropy $S_i$. The number of typical sequences starting from $i$ grows with $N$ exponentially, as $2^{NS_i}$. To get the mean rate of growth for all sequences, it must be averaged over different $i$ with their probabilities $p(i)$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Information Rate of a Markov Source)</span></p>

The entropy of the language for a Markov source is expressed via $p(i)$ and $p(j\mid i)$ by averaging over $i$ the entropy of the transition probability distribution:

$$S = -\sum_i p_i \sum_j p(j\mid i)\log_2 p(j\mid i).$$

</div>

One can go beyond two-letter correlations and statistically calculate the entropy of the next letter when the previous $N-1$ letters are known (Shannon 1950). As $N$ increases, the entropy approaches the limit which can be called the entropy of the language. Long-range correlations and the fact that we cannot make up words further lower the entropy of English down to approximately 1.4 bits per letter, *if no other information given*. Comparing 1.4 and 4.7, we conclude that the letters in an English text are about 70% redundant. This redundancy makes possible data compression, error correction and crosswords.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Redundancy of the Genetic Code)</span></p>

How redundant is the genetic code? There are four bases, which must encode twenty amino acids. There are $4^2$ two-letter words, which is not enough. The designer then must use a triplet code with $4^3 = 64$ words, so that the redundancy factor is again about 3. Number of ways to encode a given amino acid is approximately proportional to its frequency of appearance.

What are the error rates in the transmission of the genetic code? Typical energy cost of a mismatched DNA base pair is that of a hydrogen bond, which is about ten times the room temperature. If the DNA molecule was in thermal equilibrium with the environment, thermal noise would cause error probability $e^{-10} \simeq 10^{-4}$ per base. This is deadly. A typical protein has about 300 amino acids, that is encoded by about 1000 bases; we cannot have mutations in every tenth protein. Nature operates a highly non-equilibrium state, so that bonding involves extra irreversible steps and burning more energy. This way of sorting molecules is called **kinetic proofreading** (Hopfield 1974, Ninio 1975) and is very much similar to the Maxwell demon.

</div>

### 3.4 Mutual Information as a Universal Tool

Answering the question i) in Section 3.2, we have found that the entropy of the set of symbols to be transferred determines the minimum mean number of bits per symbol, that is the maximal rate of information transfer. In this section, we turn to the question ii) and find out how this rate is lowered if the transmission channel can make errors, so that one cannot unambiguously restore the input $B$ from the output $A$.

In this context one can treat measurements $A$ as messages about the value of the quantity $B$ we measure. One can also view storing and retrieving information as sending a message through time rather than space. We can include into the same scheme forecast and observation, asking how much information about the experimental data $B$ is contained in the theoretical predictions $A$.

#### Conditional Entropy

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Conditional Entropy)</span></p>

The relation between the message (measurement) $A_i$ and the event (quantity) $B_j$ is characterized by the conditional probability $P(B_j\mid A_i)$. For every $A_i$, this is a normalized probability distribution, and one can define its entropy $S(B\mid A_i) = -\sum_j P(B_j\mid A_i)\log_2 P(B_j\mid A_i)$. Since we are interested in the mean quality of transmission, we average this entropy over all values of $A_j$, which defines the **conditional entropy**:

$$S(B|A) = \sum_i P(A_i)S(B\mid A_i) = -\sum_{ij} P(A_i)P(B_j\mid A_i)\log_2 P(B_j\mid A_i).$$

</div>

The conditional entropy measures what on average remains unknown about $B$ after the value of $A$ is known. The missing information was $S(B)$ before the measurement and is equal to the conditional entropy $S(B|A)$ after it.

#### Mutual Information

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Mutual Information)</span></p>

What the measurements bring on average is their difference called the **mutual information**:

$$I(A,B) = S(B) - S(B\mid A) = \sum_{ij} P(A_i,B_j)\log_2\left[\frac{P(B_j\mid A_i)}{P(B_j)}\right].$$

Indeed, information is a decrease in uncertainty, so that the mutual information is non-negative. That means that measurements on average lower uncertainty by increasing the conditional probability relative to unconditional:

$$\langle\log_2[P(B_j\mid A_i)/P(B_j)]\rangle \ge 0.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Choice out of $n$ with $m$ Remaining)</span></p>

Let $B$ be a choice out of $n$ equal possibilities: $P(B) = 1/n$ and $S(B) = \log_2 n$. If for every $A_i$ we can have $m$ different values of $B$, that is $P(B\mid A) = 1/m$, then $S(B\mid A) = \log_2 m$ and $I(A,B) = S(B) - S(B\mid A) = \log_2(n/m)$ bits. It is non-negative, since evidently $m \le n$. Note that in this case, knowledge of $B$ fixes $A$, so that $S(A\mid B) = 0$ and $I(A,B) = S(A)$.

When there is one-to-one correspondence, $m = 1$, and $A$ tells us all we need to know about $B$.

</div>

#### Chain Rule and Symmetry of Mutual Information

Probabilities are multiplied and entropies added for independent events. For correlated events, one uses conditional probabilities and entropies in what is called the **chain rule**:

$$P(A_i,B_j) = P(B_j\mid A_i)P(A_i),$$

$$S(A,B) = S(A) + S(B\mid A) = S(B) + S(A\mid B).$$

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Properties of Mutual Information)</span></p>

The mutual information can be written in the symmetric form:

$$I(A,B) = \sum_{ij} P(A_i,B_j)\log_2\left[\frac{P(A_i,B_j)}{P(A_i)P(B_j)}\right] = S(B) - S(B\mid A) = S(A) - S(A\mid B) = S(A) + S(B) - S(A,B).$$

Key properties:
- **Symmetry**: $I(A,B) = I(B,A)$, even though $A$ and $B$ can be of very different nature.
- **Non-negativity**: $I(A,B) \ge 0$, with equality iff $A$ and $B$ are independent.
- **Sub-additivity of entropy**: $S(A) + S(B) > S(A,B)$ when $A, B$ are correlated.
- **Self-information**: $I(A,A) = S(A)$. So one can call entropy self-information.
- **Bounds**: $I(A,B)$ exceeds neither $S(A)$ nor $S(B)$. $A$ cannot contain more information about $B$ than $B$ contains about itself.

It is important to stress that measuring $A$ decreases the entropy of $B$ only *on average* over all values $A_i$: $S(B\mid A) \le S(B)$. Yet for any particular $A_i$ the entropy $S(B\mid A_i)$ can be either smaller or larger than $S(B)$, depending on how this measurement changes the probability distribution.

</div>

### 3.5 Channel Capacity

If the mutual information is what on average brings an imperfect channel, how reliable it is? It is tempting to assume that the mutual information plays for noisy channels the same role the entropy plays for ideal channels, in particular, sets the maximal rate of reliable communication in the limit of long messages, thus answering the question ii) from Section 3.2.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Channel Capacity)</span></p>

Let us characterize the channel itself, maximizing $I(A,B)$ over all choices of the source statistics $P(B)$. That quantity is called **Shannon's channel capacity**, which quantifies the quality of communication systems in bits per symbol:

$$\mathcal{C} = \max_{P(B)} I(A,B).$$

The channel capacity is the log of the maximal number of distinguishable inputs.

</div>

If there are different outputs for the same input (as in the $k-l$ example), the rate of information transfer is lower than for a one-to-one correspondence, since we need to divide our $k$ outputs into groups of $l$, distinguishing only between the groups. More formally, for each typical $N$-sequence of independently chosen $B$-s, we have $[P(A\mid B)]^{-N} = 2^{NS(A\mid B)}$ possible output sequences, all of them equally likely. To get the rate of the useful information about distinguishing the inputs, we need to divide the total number of typical outputs $2^{NS(A)}$ into sets of size $2^{NS(A\mid B)}$ corresponding to different inputs. Therefore, we can distinguish at most $2^{NS(A)}/2^{NS(A|B)} = 2^{NI(A,B)}$ sequences of the length $N$, which sets $I(A,B)$ as the maximal rate of information transfer.

#### Noisy Channel Coding Theorem

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Shannon's Noisy Channel Coding Theorem)</span></p>

What about the most generic case with random errors, when one cannot separate inputs/outputs into completely disjoint groups? Shannon showed (in the noisy channel theorem) that one can keep a finite transmission rate $R$ and yet make the probability of error arbitrarily small at the limit $N \to \infty$. The idea is that to correct errors one needs to send extra bits, so to get the rate we need to compute how many bits are devoted to error correction and how many to transferring the information itself.

It is possible to make the probability of error arbitrarily small when sending information with a finite rate $R$, if there is any correlation between output A and input B, that is $\mathcal{C} > 0$. Then the probability of an error can be made $2^{-N(\mathcal{C}-R)}$, that is asymptotically small in the limit of $N \to \infty$, if the rate is lower than the channel capacity.

</div>

This (arguably the most important) result of the communication theory is rather counter-intuitive: if the channel makes errors all the time, how one can decrease the error probability treating long messages? Shannon's argument is based on typical sequences and average equipartition, that is on the law of large numbers.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Binary Symmetric Channel)</span></p>

If in a binary channel the probability of every single bit going wrong is $q$, then $A$ is binary random variable with equal probabilities of 0 and 1, so that $S(A) = \log_2 2 = 1$. Conditional probabilities are $P(1\mid 0) = P(0\mid 1) = q$ and $P(1\mid 1) = P(0\mid 0) = 1 - q$, so that $S(A\mid B) = S(B\mid A) = S(q) = -q\log_2 q - (1-q)\log_2(1-q)$. The mutual information $I(A,B) = S(A) - S(A\mid B) = 1 - S(q)$. This is actually the maximum, that is the channel capacity: $\mathcal{C} = \max_{P(B)}[S(B) - S(B\mid A)] = 1 - S(q)$, because the maximal entropy is unity for a binary variable and corresponds to $P(0) = P(1) = 1/2$.

In a message of length $N$, there are on average $qN$ errors and there are $N!/(qN)!(N-qN)! \approx 2^{NS(q)}$ ways to distribute them. We then need to devote some $m$ bits in the message not to data transmission but to error correction. The number of possibilities provided by these extra bits, $2^m$, must exceed $2^{NS(q)}$, which means that $m > NS(q)$, and the transmission rate $R = (N-m)/N < 1 - S(q)$.

</div>

#### Additive Noise and the Gaussian Channel

The conditional entropy $S(B\mid A)$ is often independent of the input statistics $P(B)$. Maximal mutual information, that is capacity, is then achieved for maximal $S(B)$. If no other restrictions are imposed, that corresponds to the uniform distribution $P(B)$.

If the measurement/transmission noise $\xi$ is additive, that is the output is $A = g(B) + \xi$ with an invertible function $g$, then $S(A\mid B) = S(\xi)$ and

$$I(A,B) = S(A) - S(\xi).$$

The more choices of the output are recognizable despite the noise, the more is the capacity of the channel. When the conditional entropy $S(A\mid B)$ is given, then to maximize the mutual information we need to choose the measurement/coding procedure, for instance $g(B)$, that maximizes the entropy of the output $S(A)$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Gaussian Channel Capacity)</span></p>

In a continuous case, an indeterminacy is infinite. If we agree to know the position of a point on an interval $L$ with an accuracy $\epsilon$, the entropy is $S(B) = \log_2(L/\epsilon)$. A measurement $A$ with precision $\Delta$ brings the information $I(A,B) = S(B) - S(B\mid A) = \log(L/\Delta)$, independent of $\epsilon$.

More generally, for a continuous distribution $\rho(x)$, the entropy in the limit $\epsilon \to 0$ consists of two parts:

$$-\sum_i p_i\log p_i \to -\int dx\,\rho(x)\log\rho(x) - \log\epsilon.$$

The first term is called **differential entropy** $S(\rho)$. For example, the differential entropy of the Gaussian distribution $P(\xi) = (2\pi\mathcal{N})^{-1/2}\exp[-\xi^2/2\mathcal{N}]$ is $S(\xi) = \frac{1}{2}\log_2 2\pi e\mathcal{N}$.

Consider a linear noisy channel $A = B + \xi$, such that the noise is independent of $B$ and Gaussian with $\langle\xi\rangle = 0$ and $\langle\xi^2\rangle = \mathcal{N}$. If in addition we have a Gaussian input signal with $P(B) = (2\pi\mathcal{S})^{-1/2}\exp(-B^2/2\mathcal{S})$, then $P(A) = [2\pi(\mathcal{N}+\mathcal{S})]^{-1/2}\exp[-A^2/2(\mathcal{N}+\mathcal{S})]$. The mutual information is:

$$I(A,B) = S(A) - S(\xi) = \frac{1}{2}[\log_2 2\pi e(\mathcal{S}+\mathcal{N}) - \log_2 2\pi e\mathcal{N}] = \frac{1}{2}\log_2(1 + SNR),$$

where $SNR = \mathcal{S}/\mathcal{N}$ is the signal to noise ratio. This determines the capacity of the Gaussian channel in bits per transmission: $\mathcal{C} = \log_2\sqrt{(\mathcal{N}+\mathcal{S})/\mathcal{N}}$. That means that receiving a value $A$ allows to distinguish between $2^{\mathcal{C}}$ values, that is noise effectively makes a continuous channel discrete.

The estimate of $B$ is linearly related to the measurement $A$:

$$\bar{B}(A) = \int BP(B\mid A)\,dB = \frac{A\mathcal{S}}{\mathcal{S}+\mathcal{N}} = A\frac{SNR}{1+SNR}.$$

At high SNR we use the unity factor, while at low SNR we scale down the output since most of what we are seeing must be noise.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Compression and Transmission Limits)</span></p>

Mutual information also sets the limit on the data compression $A \to C$: if coding has a random element so that its entropy $S(C)$ is nonzero, the maximal data compression, that is the minimal coding length in bits, is $\min I(A,C)$.

Take-home lesson: entropy of the symbol set is the ultimate data compression rate; channel capacity is the ultimate transmission rate. Since we cannot compress below the entropy of the alphabet and cannot transfer faster than the capacity, then transmission is possible only if the former exceeds the latter.

</div>

### 3.6 Hypothesis Testing and Bayes' Formula

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Bayes' Formula)</span></p>

All empirical sciences need a quantitative tool for confronting hypothesis with data. One (rational) way to do that is statistical: update prior beliefs in light of the evidence. It is done using conditional probability. Indeed, for any $e$ and $h$, we have $P(e,h) = P(e\mid h)P(h) = P(h\mid e)P(e)$. If we now call $h$ hypothesis and $e$ evidence, we obtain the rule for updating the probability of hypothesis to be true:

$$P(h\mid e) = P(h)\frac{P(e\mid h)}{P(e)}.$$

This form of the chain rule has been named after Bayes, who first introduced it (in 1763). The new (posterior) probability $P(h\mid e)$ is the prior probability $P(h)$ times the quotient $P(e\mid h)/P(e)$ which presents the support $e$ provides for $h$.

</div>

Without exaggeration, one can say that most errors made by data analysis in science and most conspiracy theories are connected to neglect or abuse of Bayes' formula. A common mistake is the *inversion of the conditional*: evaluating $P(e\mid h)$ instead of $P(h\mid e)$. Even when $P(e\mid h)$ is high, $P(h\mid e)$ could be small if the prior probability $P(h)$ is vanishingly small and the total evidence probability $P(e)$ is not negligible.

#### Two Mutually Exclusive Hypotheses

If we choose between two mutually exclusive hypotheses $h_1$ and $h_2$, the total probability of evidence consists of two terms: $P(e) = P(e,h_1) + P(e,h_2) = P(h_1)P(e\mid h_1) + P(h_2)P(e\mid h_2)$. The posterior probability is:

$$P(h_1\mid e) = P(h_1)\frac{P(e\mid h_1)}{P(h_1)P(e\mid h_1) + P(h_2)P(e\mid h_2)}.$$

For checking an a priori improbable hypothesis, $P(h_1) \ll P(h_2)$, it is better to design an experiment that minimizes $P(e\mid h_2)$ rather than maximizes $P(e\mid h_1)$ — that is, *rule out the alternative* rather than support the hypothesis.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Drug Test in a Mostly Drug-Free Population)</span></p>

Suppose a drug test is 99% sensitive and 99% specific: $P(e\mid h_1) = 0.99$ and $P(e\mid h_2) = 0.01$, where $e$ is a positive test result, $h_1$ is "drug user", $h_2$ is "clean". If 0.5% of the population are drug users, i.e. $P(h_1) = 0.005$, the probability that a randomly selected individual with a positive test is actually a drug user is:

$$P(h_1\mid e) = \frac{0.005 \cdot 0.99}{0.99 \cdot 0.005 + 0.01 \cdot 0.995} \approx 0.332,$$

which is less than half. The result is more sensitive to specificity approaching unity (when $P(e\mid h_2) \to 0$) than to sensitivity.

</div>

#### Posterior Odds and Occam's Razor

The choice between two (not necessarily exclusive) hypotheses is determined by the ratio of their posterior probabilities:

$$\frac{P(h_1\mid e)}{P(h_2\mid e)} = \frac{P(h_1)}{P(h_2)}\frac{P(e\mid h_1)}{P(e\mid h_2)}.$$

Both factors quantify **Occam's razor** — the preference for simpler hypotheses. The second factor (the *likelihood ratio*) is applied to data: a more complex hypothesis spreads its probability over the data space more thinly, so if the evidence is compatible with both hypotheses, the simpler one generally assigns more probability to the evidence.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Minimum Description Length)</span></p>

One can interpret higher probability as lower information brought by the choice. This leads to the **minimum description length** principle: prefer the hypothesis that communicates the data in the smaller number of bits. The total message length is

$$-\log_2 P(h) - \log_2 P(e\mid h) = -\log_2 P(e,h).$$

A simpler model is communicated in fewer bits and also communicates data predictions in fewer bits since a more narrow distribution has lower entropy. The evaluation of $P(e|h)$ is itself a two-step process: first specify the model parameters, then communicate the data in those terms. Increasing the number of parameters allows better data fitting which shortens the error list, but lengthens the model specification — optimizing this trade-off is the essence of model selection.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Frequentist vs Bayesian Interpretation)</span></p>

The traditional sampling approach treats probability as the *frequency of outcomes in repeating trials*. The Bayesian approach defines probability as a *degree of belief*, which allows wider applications — particularly when we cannot have repeating identical trials nor an ensemble of identical objects. The approach may seem unscientific since it depends on prior beliefs, which can be subjective. However, repeatedly subjecting our hypothesis to variable enough testing, the resulting flow in the space of probabilities will eventually come close to a fixed point independent of the starting position.

Making prior assumptions explicit is important, both computationally and conceptually. There are neither inference nor prediction without assumptions.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Arithmetic vs Fibonacci Sequence)</span></p>

Given $5, 8, \ldots$ as two numbers of a sequence, one may put forward two hypotheses: $h_1$ predicts an arithmetic sequence $5, 8, 11, \ldots$; $h_2$ predicts a Fibonacci sequence $5, 8, 13, \ldots$ If the next number comes through a noisy channel as $12 \pm 1$, then $P(e\mid h_1) = P(e\mid h_2)$ and the choice is due to priors. Engineers and accountants would argue for arithmetic sequences (more frequently encountered), while natural scientists would point to pine cones, floral petals and seed heads to argue for Fibonacci.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Bayes' Formula for Design)</span></p>

Experimentalists measure the sensory response $A$ of an animal to the stimulus $B$, which gives $P(A\mid B)/P(A)$, or build a robot with the prescribed response. Then they measure the distribution of stimulus $P(B)$ in the natural habitat. After that, one obtains:

$$P(B\mid A) = P(B)\frac{P(A\mid B)}{P(A)},$$

which allows the animal/robot to perceive the environment and function effectively in that habitat.

</div>

### 3.7 Relative Entropy

The mutual information $I(A,B)$ measures the degree of correlation, which is essentially the difference between the true joint distribution $P(A,B)$ and the product distribution $P(A)P(B)$ of two independent quantities. As such, it is a particular case of a more general measure of difference between distributions.

#### Invalidating an Incorrect Hypothesis

Let us ask: how fast can data invalidate an incorrect hypothesis? If the true distribution is $p$ but our hypothetical distribution is $q$, what number $N$ of trials is sufficient to decrease $P(h\mid e)$ by some a priori set factor? We need to estimate how fast the factor $\mathcal{P} = P(e\mid h)/P(e)$ decreases with $N$. The result $i$ is observed $p_i N$ times. We judge the probability of that happening as $q_i^{p_i N}$ times the number of sequences with those frequencies:

$$\mathcal{P} = \prod_i q_i^{p_i N} \frac{N!}{\prod_j (p_j N)!}.$$

In the limit of large $N$, we obtain a large-deviation-type relation:

$$\mathcal{P} \propto \exp\!\left[-N\sum_i p_i \ln(p_i/q_i)\right].$$

The probability of a not-exactly-correct hypothesis to approximate the data exponentially decreases with the number of trials.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Relative Entropy / Kullback-Leibler Divergence)</span></p>

The rate of the exponential decrease in the probability of an incorrect hypothesis is the **relative entropy** (also called **Kullback-Leibler divergence**):

$$D(p\|q) = \sum_i p_i \ln(p_i/q_i) = \langle\ln(p/q)\rangle.$$

The relative entropy determines how many trials we need: we prove our hypothesis wrong when $ND(p\|q)$ becomes large. The closer our hypothesis is to the true distribution, the larger the number of trials needed. When $ND(p\|q)$ is not large, our hypothetical distribution is just fine.

</div>

#### Properties of Relative Entropy

The relative entropy measures how different the hypothetical distribution $q$ is from the true distribution $p$. Note that $D(p\|q)$ is *not* the difference between entropies (which just measures difference in uncertainties). The relative entropy is not a true geometrical distance since it does not satisfy the triangle inequality and is asymmetric: $D(p\|q) \neq D(q\|p)$. Indeed, there is no symmetry between reality and our version of it.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Nonnegativity of Relative Entropy)</span></p>

$D(p\|q) \geq 0$, with equality if and only if $p_i = q_i$ for all $i$.

**Proof.** Using the inequality $\ln x \leq x - 1$ (with equality only for $x = 1$):

$$-D(p\|q) = \sum_i p_i \ln(q_i/p_i) \leq \sum_i (q_i - p_i) = 0.$$

</div>

#### Asymptotic Equipartition and Types

Relative entropy quantifies how close to reality is the asymptotic equipartition estimate of the probability of a given sequence. Assume we have an $N$-sequence where the values appear with frequencies $q_k$, $k = 1, \ldots, K$. The asymptotic equipartition (the law of large numbers) suggests that the probability of that sequence is $\prod_k q_k^{Nq_k} = \exp[-NS(q)]$. But the frequencies in a finite sequence are generally somewhat different from the true probabilities $\lbrace p_k\rbrace$. The positivity of the relative entropy guarantees that the asymptotic equipartition *underestimates* the probability — the true probability is actually higher:

$$\prod_k p_k^{Nq_k} = \exp\!\left[N\sum_k q_k \ln p_k\right] = \exp\lbrace -N[S(q) - D(q\|p)]\rbrace.$$

How many different probability distributions $\lbrace q_k\rbrace$ (called **types** in information theory) exist for an $N$-sequence from an alphabet with $K$ symbols? Since $q_k$ can take any of $N+1$ values $0, 1/N, \ldots, 1$, the number of possible $K$-vectors is at most $(N+1)^K$, which grows with $N$ only polynomially (where $K$ sets the power). The number of sequences grows exponentially with $N$, so there is an exponential number of possible sequences for each type. The probability to observe a given type (empirical distribution) is determined by the relative entropy: $\mathcal{P}\lbrace q_k\rbrace \propto \exp[-ND(q\|p)]$.

#### Mutual Information as Relative Entropy

Mutual information is a particular case of relative entropy where we compare the true joint probability $p(x_i, y_j)$ with the product of marginals $q(x_i, y_j) = p(x_i)p(y_j)$:

$$D(p\|q) = S(X) + S(Y) - S(X,Y) = I(X,Y) \geq 0.$$

If $i$ in $p_i$ runs from 1 to $M$, we can introduce $D(p\|u) = \log_2 M - S(p)$, where $u$ is the uniform distribution. This allows one to show that both relative entropy and mutual information inherit convexity properties from entropy: $D(p\|q)$ is convex with respect to both $p$ and $q$, while $I(X,Y)$ is a concave function of $p(x)$ for fixed $p(y\mid x)$ and a convex function of $p(y\mid x)$ for fixed $p(x)$. Convexity is important for ensuring that the extremum we seek is unique and lies at the boundary of allowed states.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Price of Non-Optimal Coding)</span></p>

Relative entropy measures the price of non-optimal coding. The natural way to achieve optimal coding is to assign codeword length according to probability: $l_i = -\log_d p_i$. But $l_i$ must be integers, while $-\log_d p_i$ are generally not. A set of integer $l_i$ effectively corresponds to another distribution $q_i = d^{-l_i}/\sum_i d^{-l_i}$. Assuming we find encoding with $\sum_i d^{-l_i} = 1$, the mean codeword length is:

$$\bar{l} = \sum_i p_i l_i = -\sum_i p_i \log_d q_i = S(p) + D(p\|q),$$

which is larger than the optimal value $S(p)$, so the transmission rate is lower. If one takes $l_i = \lceil\log_d(1/p_i)\rceil$ (the ceiling), then $S(p) \leq \bar{l} \leq S(p) + 1$, i.e., non-optimality is at most one bit.

</div>

#### Monotonicity and Irreducible Correlations

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Monotonicity of Relative Entropy)</span></p>

If we observe fewer variables, the relative entropy is less:

$$D[p(x_i, y_j)\|q(x_i, y_j)] \geq D[p(x_i)\|q(x_i)],$$

where $p(x_i) = \sum_j p(x_i, y_j)$ and $q(x_i) = \sum_j q(x_i, y_j)$. When we observe fewer variables we need larger $N$ to have the same confidence. In other words, *information does not hurt* (but only on average!).

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Strong Sub-Additivity)</span></p>

For three variables, defining $q(x_i, y_j, z_k) = p(x_i)p(y_j, z_k)$ (neglecting correlations between $X$ and the rest) and using monotonicity when integrating out $Z$:

$$D[p(x_i, y_j, z_k)\|q(x_i, y_j, z_k)] \geq D[p(x_i, y_j)\|q(x_i, y_j)].$$

When $q$ is a product, $D$ turns into $I$ and we obtain:

$$S(X,Y) + S(Y,Z) - S(Y) - S(X,Y,Z) \geq 0,$$

which is called **strong sub-additivity**. It can be presented as the positivity of the **conditional mutual information**:

$$I(X,Z\mid Y) = S(X\mid Y) + S(Z\mid Y) - S(X,Z\mid Y) = S(X,Y) + S(Y,Z) - S(Y) - S(X,Y,Z) \geq 0.$$

</div>

#### Interaction Information

The straightforward generalization of mutual information for many objects, $I(X_1, \ldots, X_k) = \sum S(X_i) - S(X_1, \ldots, X_k)$, simply measures the total correlation. A more sophisticated measure is the **interaction (or multivariate) information**, which measures the irreducible information in a set of variables beyond that present in any subset.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Interaction Information)</span></p>

For three variables, the interaction information (McGill 1954) measures the difference between the total correlation and that encoded in all pairs:

$$II = I(X,Z) - I(X,Z\mid Y) = S(X) + S(Y) + S(Z) - S(X,Y) - S(X,Z) - S(Y,Z) + S(X,Y,Z).$$

Equivalently: $II = I(X,Y) + I(X,Z) + I(Y,Z) - I(X,Y,Z)$.

Interaction information measures the influence of a third variable on the amount of information shared between the other two and can be of **either sign**:
- **Positive $II$**: the third variable accounts for some correlation between the other two (its knowledge diminishes the correlation) — this measures **redundancy**.
- **Negative $II$**: the knowledge of the third variable facilitates correlation between the other two — this measures **synergy**.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Channels and Love Triangles)</span></p>

A channel with input $X$, noise $Z$ and output $Y$ corresponds to $I(X,Z) = 0$ and $I(X,Z\mid Y) > 0$, that is $II(X,Y,Z) < 0$ (synergy). Once you know the output, the unknown noise and input become related.

Love triangles can be either redundant or synergetic. If $Y$ dates either $X$, both $X,Z$, or none, then the dating states of $X$ and $Z$ are correlated, and knowing one tells us more about another when $Y$'s state is unknown: $I(X,Z) > I(X,Z\mid Y)$ (redundancy). Conversely, if $Y$ can date one, another, both, or none with equal probability, the states of $X$ and $Z$ are uncorrelated, but knowledge of $Y$ induces correlation between them (synergy).

</div>

The generalization for arbitrary number of variables is:

$$I_n = \sum_{i=1}^{n} S(X_i) - \sum_{ij} S(X_i, X_j) + \sum_{ijk} S(X_i, X_j, X_k) - \cdots + (-1)^{n+1}S(X_1, \ldots, X_n).$$

Entropy, mutual information, and interaction information are the first three members of this hierarchy. An important property of both relative entropy and all $I_n$ for $n > 1$ is that they are independent of the additive constants in the entropies, that is of the choice of units or bin sizes.

#### Connections to Statistical Physics

The second law of thermodynamics becomes trivial from the perspective of mutual information. Even when we follow evolution with infinite precision, the full $N$-particle entropy is conserved, but one-particle entropy grows. There is no contradiction: subsequent collisions impose more and more correlation between particles, so that mutual information growth compensates that of one-particle entropy.

The thermodynamic entropy of a gas is the sum of entropies of different particles $\sum S(p_i, q_i)$. In the thermodynamic limit, we neglect inter-particle correlations measured by the generalized mutual information $\sum_i S(p_i, q_i) - S(p_1 \ldots p_n, q_1, \ldots, q_n) = I(p_1, q_1; \ldots; p_n, q_n)$. Deriving the Boltzmann kinetic equation, we replaced two-particle probability by the product of one-particle probabilities, which gave the H-theorem (growth of uncorrelated entropy). Since the Liouville theorem guarantees that phase volume and true entropy $S(p_1 \ldots p_n, q_1, \ldots, q_n)$ do not change, the increase of the uncorrelated part must be compensated by the increase of mutual information.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Information-Theoretic Second Law)</span></p>

One can replace the usual second law of thermodynamics by the law of conservation of total entropy (or information): the increase in thermodynamic (uncorrelated) entropy is exactly compensated by the increase in correlations between particles expressed by mutual information. The usual second law results simply from our renunciation of all correlation knowledge.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Generalized Second Law with Correlations)</span></p>

If two systems at temperatures $T_1$ and $T_2$ exchange heat $dE_1$ (from system 2 to system 1) and their degree of correlation changes by $\Delta I$, the second law generalizes to:

$$\left(\frac{1}{T_1} - \frac{1}{T_2}\right)dE_1 - \Delta I \geq 0.$$

If correlations appear upon contact ($\Delta I > 0$), heat still flows from hot to cold: $dE_1(T_2 - T_1) \geq T_1 T_2 \Delta I > 0$. However, if initial correlations are *destroyed* during heat exchange ($\Delta I < 0$), heat could flow from cold to hot. An information-theoretic resource can be spent to perform refrigeration — this connects to Maxwell's demon.

</div>

Relative entropy also generalizes the second law for non-equilibrium processes. Entropy itself can either increase (evolution towards thermal equilibrium) or decrease (evolution towards a non-equilibrium state). However, the relative entropy between the distribution and the steady-state distribution monotonously decreases with time. Also, the conditional entropy between values of any quantity taken at different times, $S(X_{t+\tau}|X_t)$, grows with $\tau$ when it exceeds the correlation time.

## 4. Applications of Information Theory

This chapter puts content into the general notions introduced above. The simplest and probably the most important lesson is that looking for a conditional entropy maximum is not restricted to thermal equilibrium, but is a universal approach.

### 4.1 The Whole Truth and Nothing but the Truth

So far, we defined entropy and information via the distribution. In practical applications, however, the distribution is usually unknown and we need to guess it from data. Information theory / statistical physics is a systematic way of guessing, making use of partial information.

We assume that the information is given as $\langle R_j(x,t)\rangle = r_j$, i.e. as expectation (mean) values of some dynamical quantities including normalization $R_0 = r_0 = 1$. How to get the best guess for the probability distribution $\rho(x,t)$, based on that information?

Our distribution must contain *nothing but the truth*: it must maximize the missing information, which is the entropy $S = -\langle\ln\rho\rangle$. This provides the widest set of possibilities for future use, compatible with the existing information.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Maximum Entropy Distribution)</span></p>

Looking for the extremum of

$$S + \sum_j \lambda_j \langle R_j(x,t)\rangle = \int \rho(x,t)\left\{-\ln[\rho(x,t)] + \sum_j \lambda_j R_j(x,t)\right\}dx,$$

we differentiate with respect to $\rho(x,t)$ and obtain $\ln[\rho(x,t)] = \sum_j \lambda_j R_j(x,t)$, giving the distribution:

$$\rho(x,t) = \exp\!\left[\sum_j \lambda_j R_j(x,t)\right] = \frac{1}{Z}\exp\!\left[\sum_{j=1} \lambda_j R_j(x,t)\right].$$

The normalization factor (partition function) is:

$$Z(\lambda_i) = e^{-\lambda_0} = \int \exp\!\left[\sum_{j=1} \lambda_j R_j(x,t)\right]dx,$$

and the measured quantities are recovered via $\frac{\partial\ln Z}{\partial\lambda_i} = r_i$.

</div>

That this distribution is indeed a *maximum* follows from the positivity of relative entropy. Consider any other normalized distribution $g(x)$ satisfying the same constraints $\int dx\,g(x)R_j(x) = r_j$. Then:

$$S(\rho) - S(g) = -\int dx\,(g\ln\rho - g\ln g) = \int dx\,g\ln(g/\rho) = D(g\|\rho) \geq 0.$$

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Gibbs, Maxwell, and Boltzmann Distributions)</span></p>

The Gibbs distribution is the maximum entropy distribution with $R_1$ being energy. When $R_1$ is the kinetic energy of molecules, we get the Maxwell distribution; when it is potential energy in an external field, we get the Boltzmann distribution.

For the "candy-in-the-box" problem (an impurity atom on a lattice), with box number $j$ and measured mean value $\langle j\rangle = r_1$, the distribution giving maximal entropy for a fixed mean is the **geometric distribution**: $\rho(j) = (1-p)p^j$, where $p = r_1/(1+r_1)$.

If we scatter X-rays on the lattice with wavenumber $k$ and find $\langle\cos(kj)\rangle = 0.3$, then $\rho(j) = Z^{-1}(\lambda)\exp[-\lambda\cos(kj)]$ where $Z(\lambda) = \sum_{j=1}^{n}\exp[\lambda\cos(kj)]$ and $\langle\cos(kj)\rangle = d\log Z/d\lambda = 0.3$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Self-Contradictory or Insufficient Constraints)</span></p>

The set of equations for the Lagrange multipliers may be self-contradictory or insufficient, so that the data do not allow to define the distribution or allow it non-uniquely. For example, with $R_i = \int x^i\rho(x)\,dx$ for $i = 0,1,2,3$, the distribution cannot be normalized if $\lambda_3 \neq 0$, and having only three constants $\lambda_0, \lambda_1, \lambda_2$ one generally cannot satisfy four conditions. That means we cannot reach the entropy maximum, yet we can come arbitrarily close to the entropy of the Gaussian distribution $\ln[2\pi e(r_2 - r_1^2)]^{1/2}$.

</div>

#### Information and Time

If we know information at some time $t_1$ and want to make guesses about some other time $t_2$, our information generally gets less relevant as the distance $|t_1 - t_2|$ increases. In the particular case of guessing the distribution in phase space, the mechanism of losing information is due to separation of trajectories. The set of trajectories started at $t_1$ from some region fills larger and larger regions as $|t_1 - t_2|$ increases. Therefore, missing information (entropy) increases with $|t_1 - t_2|$. Note that this works both into the future and into the past — there is really no contradiction between the reversibility of equations of motion and the growth of entropy.

There is one class of quantities where information does not age: **integrals of motion**. A situation in which only integrals of motion are known is called *equilibrium*. When we leave the system alone, all currents dissipate and gradients diffuse. The distribution then takes the equilibrium form — either canonical (if environment temperature is known) or microcanonical (if only total energy is known).

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Information Perspective on Equilibrium)</span></p>

From the information point of view, thermal equilibrium means all information is forgotten except the integrals of motion. If we possess information about averages of non-conserved quantities whose averages do not coincide with their equilibrium values, then the maximum entropy distribution deviates from equilibrium.

The traditional operational view says: if we leave the system alone, it is in equilibrium; we must act on it to deviate it. The informational view offers a new light: if we leave the system alone, our ignorance is maximal and so is the entropy — the system is in thermal equilibrium. When we act on it in a way that gives us more knowledge, entropy is lowered and the system deviates from equilibrium.

</div>

#### Neuron Ensembles and the Ising Model

A beautiful example of the maximum entropy approach is obtaining the statistical distribution of neuron ensembles (Schneiderman, Berry, Segev and Bialek, 2006). In a small time window, a single neuron either generates an action potential or remains silent, described by binary vectors $\sigma_i = \pm 1$. The most fundamental measurement results are the mean spike probability $\langle\sigma_i\rangle$ and pairwise correlations $\langle\sigma_i\sigma_j\rangle$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Maximum Entropy Distribution for Neurons)</span></p>

The maximum entropy distribution consistent with measured means $\langle\sigma_i\rangle$ and pairwise correlations $\langle\sigma_i\sigma_j\rangle$ is:

$$\rho(\{\sigma\}) = Z^{-1}\exp\!\left[\sum_i h_i\sigma_i + \frac{1}{2}\sum_{i<j}J_{ij}\sigma_i\sigma_j\right],$$

where the Lagrange multipliers $h_i, J_{ij}$ are chosen so that averages in this distribution agree with the experiment. This is the **Ising model** — originally used in physics for describing systems of spins. Here it describes brain activity, which is apparently far from thermal equilibrium.

</div>

Despite apparent patterns of collective behavior involving many neurons, it turns out to be enough to account for pairwise correlations to describe the statistical distribution remarkably well. Accounting for pairwise correlations changes entropy significantly, while accounting for further correlations changes entropy relatively little. The sufficiency of pairwise interactions is an enormous simplification: the brain actually develops and constantly modifies its own predictive model of probability, and the dominance of pairwise interactions means that learning rules based on pairwise correlations could be sufficient to generate nearly optimal internal models.

It is interesting how the entropy scales with the number of interacting neurons $N$. The entropy of non-interacting neurons is extensive (proportional to $N$). Data show that $J_{ij}$ are non-zero for distant neurons as well, so the entropy of an interacting set is lower at least by the sum of mutual information terms between all pairs — a negative contribution proportional to $N(N-1)/2$. One can estimate from low-$N$ data a "critical" $N$ when the quadratic term is expected to turn entropy to zero. That critical $N$ corresponds well to the empirically observed sizes of clusters of strongly correlated cells.

### 4.2 Exorcizing Maxwell Demon

Making a measurement $R$ changes the distribution from $\rho(x)$ to the (generally non-equilibrium) conditional distribution $\rho(x|R)$, which has its own **conditional entropy**:

$$S(x|R) = -\int dx\,dR\,\rho(x,R)\ln\rho(x|R).$$

The conditional entropy quantifies the remaining ignorance about $x$ once $R$ is known. Measurement decreases the entropy of the system by the mutual information:

$$S(x) - S(x|R) = \int \rho(x,R)\ln[\rho(x,R)/\rho(x)\rho(R)]\,dx\,dR = S(x,R) - S(R) - S(x).$$

#### Energy Cost of Measurement

All measurements happen at a finite temperature. Assume our system is in contact with a thermostat at temperature $T$ (which does not mean the system is in thermal equilibrium). We define a free energy $F(\rho) = E - TS(\rho)$. The Gibbs-Shannon entropy and mutual information can be defined for arbitrary distributions. If the measurement does not change energy (like knowing which half of the box the particles are in), then the entropy decrease increases the free energy — that is the total work we are able to do. The first law of thermodynamics then requires that the minimal work to perform such a measurement is:

$$F(\rho(x|R)) - F(\rho(x)) = T[S(x) - S(x|R)].$$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Information Interprets Free Energy)</span></p>

Thermodynamics interprets $F$ as the energy we are *free* to use (keeping the temperature). Information theory reinterprets this: if we knew everything, we could use the whole energy to do work; the less we know, the more is the missing information $S$ and the less work we can extract. The decrease of $F = E - TS$ with the growth of $S$ measures how available energy decreases with the loss of information about the system. As Maxwell understood in 1878: "Suppose our senses sharpened to such a degree that we could trace molecules as we now trace large bodies, the distinction between work and heat would vanish."

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Entropy Is Not a Property of the System)</span></p>

The concept of entropy as missing information (Brillouin 1949) allows one to understand that Maxwell demon or any other information-processing device does not really decrease entropy. If at the beginning one has information on position or velocity of any molecule, the entropy was less by this amount from the start; after using and processing the information, the entropy can only increase.

Consider a particle in a box at temperature $T$. If we know which half it is in, the entropy (logarithm of *available* states) is $\ln(V/2)$. That information has thermodynamic (energetic) value at finite temperature: by placing a piston at the half and allowing the particle to hit and move it, we can get the work $T\Delta S = T\ln 2$ out of the thermal energy of the particle.

</div>

#### Landauer's Principle

The law of energy conservation tells that to get such information one must make a measurement whose minimum energetic cost at fixed temperature is $W_{meas} = T\Delta S = T\ln 2$ (realized by Szilard in 1929, who also introduced "bit" as a unit of information). In a general case, the entropy change is the difference between $S(A)$ and $S(A,M)$. When there is a change in the free energy $\Delta F_M$ of the measuring device:

$$W_{meas} \geq T\Delta S + \Delta F_M = T[S(A) - S(A,M)] + \Delta F_M.$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Landauer's Principle)</span></p>

The work at the expense of thermal energy does not break the first law of thermodynamics. But can we break the second law by constructing a perpetuum mobile of the second kind — regularly measuring particle position and using thermal energy to do work? To make a full thermodynamic cycle, we need to return the demon's memory to the initial state. The energy price of **erasing** information involves compression of the phase space and is irreversible:

$$W_{eras} \geq TS(M) - \Delta F_M.$$

Together, the energy price of the full cycle is:

$$W_{eras} + W_{meas} \geq T[S(A) + S(M) - S(A,M)] = TI,$$

where $I$ is the **mutual information** between the measured system and the memory. The bound depends only on the mutual correlation between the measured system and the memory, not on the information content itself. This expresses the trade-off between erasure and measurement work: when one is smaller, the other must be larger.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Information Is Physical)</span></p>

Take-home lesson: information is physical. We can get extra work out of it, for instance improving the efficiency of thermal engines beyond the Carnot limit. Processing information without storing an ever-increasing amount of it must be accompanied by a finite heat release at a finite temperature. No matter how slowly we process information, we cannot make the dissipation rate lower than $T\ln 2$ per bit. This is in distinction from usual thermodynamic processes (where there is no information processing involved) where we can make heat release arbitrarily small by making the process slower.

Landauer's principle also imposes the fundamental physical limit on computations. Any Boolean function that maps several input states onto the same output state (such as AND, NAND, OR, XOR) is logically irreversible. When a computer performs logically irreversible operations, information is erased and heat must be generated. In principle, any computation can be done using only reversible steps (Bennett 1973), but that requires the computer to reverse all steps after printing the answer.

</div>

### 4.3 Renormalization Group and the Art of Forgetting

Statistical physics in general is about lack of information. One of the most fruitful ideas of the 20th century is to look at how one loses information step by step and what universal features appear in the process. A general formalism describing how to reduce description keeping only most salient features is called the **renormalization group** (RG). It consists in subsequently eliminating degrees of freedom, renormalizing remaining ones and looking for fixed points.

There is a shift of paradigm brought by RG: instead of being interested in any particular probability distribution, we are interested in *flows in the space of distributions*. Whole families (**universality classes**) of different systems described by different distributions flow under RG transformation to the same fixed point, i.e. have the same asymptotic distribution.

#### RG for Independent Random Variables: Central Limit Theorem

Consider a set of random iid variables $\{x_1 \ldots x_N\}$, each with probability density $\rho(x)$ with zero mean and unit variance. The two-step RG replaces any two of them by their sum re-scaled to keep the variance: $z_i = (x_{2i-1} + x_{2i})/\sqrt{2}$. The new variables each have the distribution:

$$\rho'(z) = \sqrt{2}\int dx\,dy\,\rho(x)\rho(y)\delta(x + y - z\sqrt{2}).$$

The distribution that does not change under this procedure (the **fixed point**) satisfies:

$$\rho(x) = \sqrt{2}\int dy\,\rho(y)\rho(\sqrt{2}x - y).$$

Solving via the Fourier transform $\rho(k) = \int\rho(x)e^{ikx}\,dx$, we get $\rho(k\sqrt{2}) = \rho^2(k)$, whose solution is $\rho_0(k) \sim e^{-k^2}$ and $\rho_0(x) = (2\pi)^{-1/2}e^{-x^2/2}$. The **Gaussian distribution** is the fixed point of repetitive summation and re-scaling, keeping variance fixed. This is not surprising since it has maximal entropy among distributions with the same variance.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Central Limit Theorem via RG)</span></p>

To turn the fixed-point result into the central limit theorem, we need to show the distribution is **linearly stable** — that RG indeed flows towards it. Near the fixed point $\rho = \rho_0(1 + h)$, the transform linearizes to $h'(k) = 2h(k/\sqrt{2})$. The eigenfunctions are $h_m(k) = k^m$ with eigenvalues $2^{1-m/2} = h'_m(k)/h_m(k)$.

Three conservation laws constrain the transformation: $\int x^n\rho(x)\,dx$ must be preserved for $n=0$ (normalization), $n=1$ (zero mean) and $n=2$ (unit variance). Since moments of $\rho(x)$ are derivatives of $\rho(k)$ at $k=0$, the conservation laws mean $h(0) = h'(0) = h''(0) = 0$, so only $m > 2$ perturbations are admissible. All these eigenvalues are $< 1$, meaning deviations from the fixed point **decrease** — in the space of distributions with the same variance, the RG-flow eventually brings us to the Gaussian, forgetting all information except normalization, mean, and variance.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Cauchy Distribution as Another Fixed Point)</span></p>

Another natural transformation replaces a pair by their mean $z_i = (x_{2i-1} + x_{2i})/2$. The fixed point satisfies $\rho(k) = \rho^2(k/2)$, giving $\rho(k) = \exp(-|k|)$ and $\rho(x) = (1+x^2)^{-1}$, the **Cauchy distribution**. This distribution has infinite variance, and RG preserves only the mean (zero) and normalization. More generally, with re-scaling $z_i = (x_{2i-1} + x_{2i})/2^\mu$, one gets the family of universal distributions $\rho(k) = \exp(-|k|^\mu)$ characterized by the parameter $\mu$.

</div>

#### RG for Interacting Systems: The Ising Model

When dealing with strongly correlated random variables, we consider the Ising model of interacting spins with the block spin transformation. We divide all spins into groups (blocks) of side $k$ ($k^d$ spins per block in $d$ dimensions) and assign to each block a new variable $\sigma' = \pm 1$ corresponding to the majority orientation.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(1D Ising Chain Decimation)</span></p>

Consider a 1D chain with nearest-neighbor interaction, Gibbs distribution $Z^{-1}\exp(-K\sum_{ij}\sigma'_i\sigma'_j)$, where $K = J/2T$. The partition function is:

$$Z(K) = \sum_{\{\sigma=\pm 1\}}\exp\!\left[K\sum_i \sigma_i\sigma_{i+1}\right] = 2(2\cosh K)^{N-1}.$$

Using decimation (eliminating every other spin, $k=3$ blocks, summing over intermediate spins with $x = \tanh K$), we obtain the recursion relations:

$$K' = \tanh^{-1}(\tanh^3 K) \quad\text{or equivalently}\quad x' = x^3,$$

$$g(K) = \ln\!\left(\frac{\cosh K'}{4\cosh^3 K}\right).$$

The partition function of the renormalized system becomes $\sum_{\{\sigma'\}}\exp[-g(K)N/3 + K'\sum_i \sigma'_i\sigma'_{i+1}]$.

</div>

#### Fixed Points and Phase Transitions

Since $K \propto 1/T$, then $T \to \infty$ and $T \to 0$ correspond to $x \to 0+$ and $x \to 1-$ respectively. The transformation $x \to x^3$ has two fixed points: $x = 0$ (stable, disordered/paramagnetic) and $x = 1$ (unstable, fully ordered). For $0 < x < 1$, iterating the process drives $x$ to zero and effective temperature to infinity — large-scale degrees of freedom are described by a paramagnetic state. This is consistent with the impossibility of long-range order in one-dimensional systems with short-range interaction.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Entropy and Mutual Information under RG)</span></p>

What entropic measure monotonically grows along RG, quantifying the irreversibility of forgetting? Eliminating degrees of freedom decreases the entropy of the whole system even as RG moves towards a more disordered state. It is more natural to consider the entropy per spin or the mutual information between eliminated and remaining degrees of freedom.

By monotonicity of relative entropy (decreasing upon elimination of variables, Section 3.7), we can define the mutual information between two sub-lattices (eliminated and remaining). Its positivity implies the monotonic growth of the entropy per site $h(K) = \lim_{N\to\infty}S(K,N)/N$.

For a finite system with short-range correlations, $S(N) = hN + C$, where $C$ is the **excess entropy** (also called mutual information between past and future of a message). For the 1d Ising chain: $h = \ln(2\cosh K) - K\tanh K$ and $C = K\tanh K - \ln(\cosh K)$. Upon RG flow, $h$ monotonously changes from $h(K) \approx 3e^{-2K}$ at large $K$ to $h(K) \approx \ln 2$ at $K \to 0$, and $C \approx \ln 2$ at large $K$ to $C \to 0$ at $K \to 0$. Here $C = \ln q$ where $q$ is the degeneracy of the ground state ($q = 2$ for two ground states with opposite magnetization at zero temperature, $q = 1$ for the fully disordered state).

</div>

#### Higher Dimensions and Universality

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Phase Transitions in Higher Dimensions)</span></p>

In higher spatial dimensions ($d > 1$), there can exist fixed points (limiting distributions) describing neither fully ordered nor fully disordered state, but a **critical state** of the phase transition. In 1d, the zero-temperature fixed point is unstable ($K$ decreases under RG). But in $d$ dimensions, the renormalized coupling scales as $K' \propto k^{d-1}K$, so for $d = 2$ with $k = 3$ we have $K' \approx 3K$ at high temperatures — meaning the low-temperature fixed point $K = \infty$ is stable at $d > 1$. The paramagnetic fixed point $K = 0$ is also stable. There must be an unstable fixed point at some $K_c$ corresponding to a **critical temperature** $T_c$.

This unstable fixed point is what physicists care most about. The RG flow in multi-dimensional parameter space has an attractor whose dimensionality is set by Lyapunov exponents. The critical surface is a separatrix dividing paramagnetic from ferromagnetic behavior. Different physical systems with only nearest-neighbor coupling ($K_2 = 0$) or also next-nearest-neighbor coupling ($K_2 \neq 0$) arrive at *the same fixed point* — this is the universality of long-distance critical behavior.

</div>

#### Information Flow and Entanglement Entropy

What quantifies the rate of information loss by RG flow in a multi-dimensional space? In 2d, one can consider a (finite) line $L$ and break direct interactions between degrees of freedom on different sides. The statistics of such a set is characterized not by a scalar function but by a matrix (similar to the density matrix in quantum statistics). For that density matrix $\rho_L$ one defines the von Neumann entropy $S_L = -\text{Tr}\,\rho_L\log\rho_L$.

For long lines in short-correlated systems, $S_L$ depends only on the distance $r$ between the end points (information flows like an incompressible fluid). At criticality, this dependence is logarithmic (fluctuations of all scales, infinite correlation radius). The function $c(r) = r\,dS_L(r)/dr$ is shown to be a monotonic zero-degree function, changing monotonically under RG flow and taking a finite value at the fixed point equal to the **central charge** of the respective conformal field theory. The central charge is a measure of relevant degrees of freedom that respond to boundary perturbations.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Topological Entanglement Entropy)</span></p>

For a quantum system in 2+1 dimensions, the relative entanglement of three finite planar regions $A, B, C$ with common boundaries gives the **topological entanglement entropy**:

$$S_A + S_B + S_C + S_{ABC} - S_{AB} - S_{BC} - S_{AC}.$$

This is a quantum analog of the interaction information. For some classes of systems, the terms depending on boundary lengths cancel and what remains (if any) is independent of the deformations of the boundaries, thus characterizing the **topological order** of the system (Kitaev, Preskill 2006).

</div>

### 4.4 Information Is Life

One may be excused thinking that living beings consume energy and matter to survive, unless one is a physicist and knows that energy and matter are conserved and cannot be consumed. All the energy, absorbed by plants from sunlight and by us from food, is emitted as heat. Life-sustaining substance is **entropy**: we consume information and generate entropy by intercepting flows from low-entropy energy sources to high-entropy body heat.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why We Eat and Why Plants Need the Sun)</span></p>

For plants, the Sun is a low-entropy energy source due to its high temperature (the same is true for the whole Earth, which exports much more entropy than it receives). We consume matter only to make it more disordered: what we consume has much lower entropy than what comes out of our excretory system. In other words, we decrease entropy inside and increase it outside of our bodies. Consuming information is our way to resist (temporarily) the second law of thermodynamics and survive. On a higher level, the nervous system maintains body integrity by consuming information through active inference.

</div>

Our way to stay out of thermal equilibrium is to use **replication** to generate highly ordered (improbable) structures. The instructions for replication are encoded in genes. Evolution as natural selection is an increasingly efficient encoding of information about the environment in the gene pool of its inhabitants. The ultimate survivor is the information in the genes, which continues to exist long after many its former carriers went extinct.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Energy Price of Information Processing in Life)</span></p>

If an elementary act of life as information processing generates $\Delta S$, we can ask about its energy price. Similar to the thermal engine efficiency, assume one takes $Q$ from a reservoir at $T_1$ and delivers $Q - W$ to the environment at $T_2$. Then $\Delta S = S_2 - S_1 = (Q-W)/T_2 - Q/T_1$ and the energy price is:

$$Q = \frac{T_2\Delta S + W}{1 - T_2/T_1}.$$

When $T_1 \to T_2$, information processing becomes prohibitively ineffective, just like the thermal engine. In the other limit, $T_1 \gg T_2$, one can neglect the entropy change on the source, and $Q = T_2\Delta S + W$. Hot Sun is indeed a low-entropy source.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Human Information Processing Rate)</span></p>

An average lazy human being dissipates about $W = 200$ watts of power at $T = 300\,K$. Since the Boltzmann constant is $k = 1.38 \times 10^{-23}$, that gives about $W/kT \simeq 10^{23}$ bits per second. The amount of information processed per unit of subjective time (per thought) is about the same, assuming each moment of consciousness lasts about a second (Dyson, 1979).

</div>

#### Hick's Law: Reaction Time and Entropy

Do the Gibbs entropy and mutual information have any quantitative relation to the way we react to signals? When one must react differently to different stimuli, the average choice-reaction time was found experimentally to be **linearly proportional to the entropy** of the statistical distribution of stimuli (Hick 1952, Hyman 1953). The more uncertainty, the longer it takes to recognize the event.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Reaction Time and Entropy)</span></p>

When one needs to name a letter or number that appears randomly on a screen, the average response time grows **logarithmically** with the size of the set (logarithmic dependence means the decision is made by a subdividing strategy — like binary search). When the number of elements stays constant but their frequencies are made unequal (lowering entropy), the average response time decreases proportionally.

Even more remarkably, when experimentalists introduce correlations between subsequent stimuli, the response time goes down in proportion to the **conditional entropy**, which is less than unconditional entropy. Conversely, as reaction time gets shorter, we make more errors, diminishing the mutual information $I(i,j) = S(j) - S(j|i)$ between input $i$ and output name $j$. Living beings use Boltzmann and Gibbs entropies as well as mutual information.

</div>

Since the time of processing is proportional to the amount of information, one can conclude that the system works to keep uniform an average amount of information processed per unit time, that is the rate.

#### Maximizing Capacity: The Infomax Principle

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Optimal Sensory Response Function)</span></p>

Consider designing the response function $y = g(x)$ for a sensory system of a living being. The problem is to choose thresholds for switching to the next level of response, or equivalently, to choose the function of the input for which we take equidistant thresholds.

The representation with maximal capacity corresponds to the maximum of the mutual information between input and output: $I(x,y) = S(y) - S(y|x)$. For a one-to-one (error-free) relation $y = g(x)$, the conditional entropy $S(y|x)$ is zero. We need to maximize the entropy of the output. Since $\rho(y)\,dy = \rho(x)\,dx$, we have $\rho(y) = \rho(x)/g'(x)$, and:

$$S(y) = -\int \rho(x)\ln[\rho(x)/g'(x)]\,dx = S(x) + \langle\ln[g'(x)]\rangle.$$

The entropy for a distribution on a finite interval is maximal when $\rho(y)$ is constant. Maximizing $S(y)$ gives $g'(x) = C\rho(x)$, that is the optimal response function is the **cumulative probability function** of the input:

$$g(x) = C\int^x \rho(x')\,dx'.$$

In other words, we choose equal bins for the output variable — more frequent inputs are coded by shorter codewords (analogy with efficient encoding).

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Fly Eye and Laughlin's Experiment)</span></p>

This was discovered in one of the first applications of information theory to biology (Laughlin 1981). First-order interneurons of the insect eye were found to code contrast rather than absolute light intensity. The response function $y = g(x)$ measured from fly neurons matched the cumulative probability distribution of contrasts measured in the fly's natural habitat (woodlands and lakeside) — with no fitting parameters. The same approach works for biochemical and genetic input-output relations.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Amplitude vs Frequency Modulation in Neurons)</span></p>

A neuron either fires a standard pulse (action potential) or stays silent, making it natural to assume information is encoded as binary digits in discrete equal time intervals (**amplitude modulation**). Alternatively, information could be encoded by the time delays between subsequent pulses (**frequency modulation**). In the former case, the maximal rate depends only on the minimal time delay between pulses (recovery time). In the latter, the rate depends on both the minimal error of timing measurement and the admissible maximal time between pulses. In reality, brain activity depends on all information-bearing parameters — both presence/absence as a binary digit and precise timing.

</div>

### 4.5 Theory of Mind

An ambitious application of information theory is the attempt to understand and quantify sentient behavior. The idea going back to Helmholtz is to view "perception as unconscious inference". There is evidence that perception is inferential, based on prediction and hypothesis testing (as manifested by binocular rivalry, where different pictures presented to two eyes cause alternating perception rather than a stable amalgam). Signals travel between brain and sensory organs in both directions simultaneously, suggesting even unconscious activity uses rational Bayes' rule.

#### Perception as Bayesian Inference

Perception is treated not as bottom-up encoding of sensory states $Y$ into internal neuronal representation of environmental states $X$, but as a combination of **top-down prior expectation** with **bottom-up sensory signals**. The mechanics of the sensory system determines $P(Y|X)$ — the conditional probability of sensory input for a given state of environment. Upon receiving input $y$, the simplest inference is maximum likelihood: taking the value $x$ that maximizes $P(y|x)$.

However, to make a decision or action based on inference, we need a measure of confidence — the whole posterior distribution $P(X|Y)$. Sharp distribution gives high confidence, flat gives low. Using Bayes' formula:

$$P(X|Y) = P(Y|X)P(X)/P(Y).$$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Variational Free Energy for Perception)</span></p>

The mind has a so-called generative model, represented by the joint distribution $P(X,Y) = P(Y|X)P(X)$. Exact computation of $P(X|Y)$ can be impossible or unpractical due to the need to average over many hidden states. It is natural to assume the brain uses a **variational approach**: for a given observation $y$, look for the posterior distribution $Q(X)$ which minimizes the **variational free energy**:

$$F[Q,y] = D[Q(x)\|P(x)] - \sum_x Q(x)\log P(y|x) = \sum_x Q(x)\left[\log\frac{Q(x)}{P(x)} - \log P(y|x)\right].$$

This can be rewritten in three equivalent forms:

$$F[Q,y] = D[Q(x)\|P(x|y)] - \log P(y).$$

The three lines suggest three different interpretations:
- **Line 1:** Trade-off between inertia (divergence from prior) and force of data (log-likelihood weighted by $Q$).
- **Line 2:** Can be written as $E/T - S(Q)$, where $\log P(x,y) = -E(x,y)$. Minimization requires the trade-off between energy-imposed "truth" and entropy maximization ("nothing but the truth").
- **Line 3:** The free energy is bounded from below by the sensory surprise $-\log P(y)$. Only when $Q(x)$ equals the exact posterior $P(x|y)$ does the free energy reach its global minimum.

</div>

#### Active Inference

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Active Inference)</span></p>

The third line of the free energy suggests that perceptual inference — computing $Q(x)$ — is not the only way to minimize $F(Q,y)$. Another way is to change the sensory data $y$ itself. That requires **action**: switching the channel, looking the other way, or moving. This is the **active inference** approach: living beings survive by adapting an action-perception loop with their environment.

Every sensory input is not obtained passively but is predicted by the brain and solicited by an action intended for that predicted input. Mismatch between predicted and actually received sensory input leads to updating the predictive (generative) model, which triggers new action leading to new sensory observations better corresponding to expectations. Perception and action are complementary ways to diminish the mismatch:
- **Perception** changes your mind — replacing prior beliefs by posterior ones.
- **Action** changes the world — to make it more compatible with the beliefs.

In a nutshell: **surprise minimization by active inference is our way to survive**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Art, Emotions, and Hallucinations)</span></p>

Poetry and music appeal to our ever-predicting mind by creating expectations (using rhythm or melody) and then partially fulfilling and partially breaking them. An optimal mixture of expected and surprising is what makes great art.

When top-down signals totally dominate, one has hallucinations; what is considered normal perception could then be called "controlled hallucination". The active inference approach also provides a treatment of emotions not solely as fixed universal patterns inherited from animal ancestors, but as constructed and learnt patterns of prediction and reaction amenable to significant variability and plasticity.

</div>

### 4.6 Rate Distortion and Information Bottleneck

When we transfer information, we look for maximal transfer rate and define channel capacity as the maximal mutual information between input and output. But when we *encode* information, we may look for the opposite: what is the *minimal* number of bits sufficient to encode the data with a given accuracy.

#### Rate Distortion Theory

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Rate Distortion Function)</span></p>

Representation of a continuous input $B$ by a finite discrete output encoding $A$ generally leads to some distortion, characterized by the real function $d(A,B)$. The mean distortion is $\mathcal{D} = \sum_{ij}P(A_i, B_j)d(A_i, B_j)$. For Gaussian statistics (completely determined by the variance), one chooses the squared error function $d(A,B) = (A-B)^2$ — this is why we use least squares approximations: minimizing variance minimizes the entropy of a Gaussian distribution and thus the amount of information needed to characterize it.

The **rate distortion function** $R(\mathcal{D})$ is the minimal rate $R$ (mutual information $I(A,B)$) sufficient to provide for distortion not exceeding $\mathcal{D}$. We look not for the maximum of $I(A,B)$ (as in channel capacity) but for the **minimum** over all encodings $P(B|A)$ such that the distortion does not exceed $\mathcal{D}$. Since $I(A,B) = S(B) - S(B|A)$, minimizing $I$ means maximizing $S(B|A)$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Gaussian Rate Distortion)</span></p>

Consider a Gaussian $B$ with $\langle B\rangle = 0$ and $\langle B^2\rangle = \sigma^2$. With one bit, we can only convey the sign of $B$. The simplest encoding is $A = \pm\sigma\sqrt{2/\pi}$ (to minimize squared error). For a fixed variance, maximal $S(B|A)$ corresponds to Gaussian distortion with variance $\mathcal{D}$, corresponding to an imaginary Gaussian channel with $\langle(B-A)^2\rangle = \mathcal{D}$. The minimal rate is:

$$R(\mathcal{D}) = \frac{1}{2}\log_2\frac{\sigma^2}{\mathcal{D}}.$$

It goes to infinity for $\mathcal{D} \to 0$ and to zero for $\mathcal{D} = \sigma^2$ (at which point we can take $A = 0$ with probability one — zero mutual information).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Allocating Bits Among Multiple Signals)</span></p>

Often we need to represent by $R$ bits $m$ independent Gaussian signals with different variances $\sigma_i$. To minimize total distortion $\sum_i \mathcal{D}_i = \mathcal{D}$ subject to $\sum_i R(\mathcal{D}_i) = R$, we find that all $\mathcal{D}_i$ should equal the same constant $\mathcal{D}/m$, as long as this constant is less than all $\sigma_i$. As we reduce $R$, we increase $\mathcal{D}$ until $\mathcal{D}/m$ exceeds some $\sigma_j$ — then we allocate zero bits to that component (its variance is already small enough). This is the logic of rate distortion theory.

</div>

#### General Variational Solution

For non-quadratic distortion functions, the conditional probability $P(A|B)$ is not Gaussian. One minimizes the functional:

$$F = I + \beta\mathcal{D} = \sum_{ij}P(A_i|B_j)P(B_j)\left\{\ln\frac{P(A_i|B_j)}{P(A_i)} + \beta\,d(A_i, B_j)\right\}.$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Gibbs Form of Optimal Encoding)</span></p>

Variation with respect to $P(A_i|B_j)$ gives:

$$P(A_i|B_j) = \frac{P(A_i)}{Z(B_j,\beta)}\,e^{-\beta\,d(A_i,B_j)},$$

where $Z(B_j,\beta) = \sum_i P(A_j)e^{-\beta\,d(A_i,B_j)}$ is the normalization factor and $P(A_i) = \sum_i P(A_i|B_j)P(B_j)$. This is a Gibbs distribution with "energy" equal to the distortion function. The inverse temperature $\beta$ reflects our priorities: small $\beta$ means we minimize information without much regard to distortion; large $\beta$ requires the conditional probability to be sharply peaked at the minima of the distortion function.

</div>

#### Information Bottleneck

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Information Bottleneck)</span></p>

In image processing and pattern recognition, the measured quantity $A$ contains too much data of low information value. We wish to compress $A$ to $C$ while keeping as much as possible information about the relevant event $B$. We formalize this as finding a short code for $\{A\}$ that preserves the maximum information about $\{B\}$. We squeeze the information through a "bottleneck" formed by a limited set of codewords $\{C\}$ (Tishby et al 2000).

One looks for the minimum of the functional:

$$I(C,A) - \beta\,I(C,B).$$

The coding $A \to C$ is stochastic, characterized by $P(C|A)$. The rate $I(C,A)$ is the average number of bits per message needed to specify an element in the codebook. The single parameter $\beta$ represents the tradeoff between complexity of representation $I(C,A)$ and accuracy $I(C,B)$:
- At $\beta = 0$: the most sketchy quantization — everything is assigned to a single point.
- As $\beta$ grows: we are pushed toward detailed quantization.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Information Bottleneck Solution)</span></p>

Variation of the functional with respect to $P(C_j|A_i)$ gives:

$$P(C_j|A_i) = \frac{P(C_j)}{Z(A_i,\beta)}\exp\left\{-\beta\,D[P(B_k|A_i)\|P(B_k|C_j)]\right\},$$

where the conditional probabilities of $A, B$ given $C$ are related by Bayes' rule:

$$P(B_k|C_j) = \frac{1}{P(C_j)}\sum_i P(A_i)P(B_k|A_i)P(C_j|A_i).$$

The relative entropy $D$ between the two conditional probability distributions $P(B|A)$ and $P(B|C)$ emerged as the effective distortion measure. The system of equations is solved by iterations. Doing compression many times, $A \to C_1 \to C_2 \ldots$, is used in multi-layered **Deep Learning** algorithms. Knowledge of statistical physics helps in identifying phase transitions (with respect to $\beta$) and the relation between processing from layer to layer and the renormalization group.

</div>

### 4.7 Information Is Money

#### Kelly Criterion: Optimal Betting

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Kelly Criterion)</span></p>

Consider a simple game: bet on a coin with unequal probabilities $p > 1/2$ and $1-p$, doubling your bet if you win and losing it if you lose. You bet a fraction $f$ of your money on the more probable side and put $1-f$ on the less probable side. After $N$ bets with $n$ wins, your money is multiplied by $(2f)^n[2(1-f)]^{N-n} = 2^{N\Lambda}$, where the rate is:

$$\Lambda(f) = 1 + \frac{n}{N}\log_2 f + \left(1 - \frac{n}{N}\right)\log_2(1-f).$$

As $N \to \infty$, the mean geometric rate approaches $\lambda = 1 + p\ln f + (1-p)\ln(1-f)$. Differentiating with respect to $f$, the maximal growth rate corresponds to $f = p$ (**proportional gambling**) and equals:

$$\lambda(p) = 1 + p\log_2 p + (1-p)\log_2(1-p) = S(u) - S(p) = 1 - S(p),$$

where $S(u) = 1$ bit is the entropy of the uniform distribution. The maximal rate of money growth equals the **entropy decrease** — that is, it equals the information you have (Kelly 1950).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Geometric vs Arithmetic Mean)</span></p>

An important lesson is that we maximize not the mean return but its mean **logarithm** — the geometric mean. Since it is a self-averaging quantity, the probability of growth approaches unity as $N \to \infty$. Note, however, that the geometric mean is less than the arithmetic mean. We may have a situation where the arithmetic growth rate is larger than unity while the geometric mean is smaller — meaning the probability to lose it all tends to unity as $N \to \infty$, even though the mean returns grow unbounded.

</div>

#### Horse Races and Generalized Kelly Criterion

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Generalized Kelly Criterion for Horse Races)</span></p>

For gambling on horse races or investing, where many outcomes have different probabilities $p_i$ and payoffs $g_i$, we maximize $\sum p_i\log(f_i g_i)$. Since $\sum f_i = 1$, we can treat $f_i$ as a distribution. The relative entropy $\sum p_i\log(p_i/f_i)$ is non-negative, so $\sum p_i\log f_i$ reaches its maximum when $f_i = p_i$ (proportional gambling). The rate is then:

$$\lambda(p,g) = \sum_i p_i\ln(p_i g_i).$$

If the operator assumes probabilities $q_i$ and sets payoffs $g_i = 1/Zq_i$ with $Z > 1$, then:

$$\lambda(p,q) = -\ln Z + \sum_i p_i\ln(p_i/q_i) = -\ln Z + D(p\|q).$$

The relative entropy $D(p\|q)$ determines the rate with which your winnings can grow. Using an incorrect distribution $q'$ instead of the true $p$ gives $\lambda(p,q,q') = -\ln Z + D(p\|q) - D(p\|q')$. If you can encode the data (sequence of track winners) shorter than the operator, you get paid in proportion to that shortening.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Bacteria and Proportional Gambling)</span></p>

Bacteria follow the same strategy without ever taking a course on statistical physics. Analogously to coin flipping, bacteria face the choice between growing fast (but being vulnerable to antibiotic) or growing slow (but being resistant). They use proportional gambling to allocate fractions of populations to different choices — this is called **phenotype switching**. The same strategy is used by many plants: a fraction of seeds does not germinate in the same year they were dispersed; the fraction increases with the environment variability.

</div>

#### Growth Rate and Mutual Information

More generally, the environment can be characterized by parameters $A$ and the internal state of the organism/gambler by parameters $B$. The growth rate is:

$$\lambda = \int dA\,dB\,P(A,B)\,r(A,B) = \int dA\,P(A)\int dB\,P(B|A)\,r(A,B).$$

The coordination is determined by $P(B|A)$, which determines the mutual information:

$$I(A,B) = \int dA\,P(A)\int dB\,P(B|A)\log_2\frac{P(B|A)}{P(B)}.$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Growth-Information Tradeoff)</span></p>

Acquiring information has its own cost $aI$. One looks for a tradeoff between maximizing growth and minimizing information cost: maximize $F = \lambda - aI$. This gives, similarly to the rate distortion theory:

$$P(B|A) = \frac{P(B)}{Z(A,\beta)}\,e^{\beta r(A,B)},$$

where $\beta = a^{-1}\ln 2$. This is the same as rate distortion theory, with the energy now being minus the growth rate. The choice of $\beta$ reflects relative costs of information and metabolism. All possible states in the plane $(r, I)$ lie below a monotonic convex curve, much like the energy-entropy plane in thermodynamics.

</div>

#### Multi-Armed Bandits and the Gittins Index

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Multi-Armed Bandit and Gittins Index)</span></p>

In the multi-armed bandit problem, we can make one bet at a time, choosing among several options (arms of slot-machines). Each arm gives reward $r = 1$ (win) or $r = 0$ (loss). We do not know the winning probabilities $s_i$, and assume uniform prior: $P(s_i) = 1$, $0 \leq s_i \leq 1$. After $w_i$ wins and $l_i$ losses on arm $i$, the posterior probability is the binomial distribution:

$$P(s_i) = s_i^{w_i}(1-s_i)^{l_i}\frac{(w_i + l_i + 1)!}{w_i!\,l_i!}.$$

Later rewards are discounted by factor $0 < \gamma < 1$. The **Gittins index** $\nu_i$ (Gittins, 1979) is the ratio of the expected sum of rewards to the discounted time, under the probability that the arm will be terminated in the future:

$$\nu(l_i, n_i, t) = \sup_{\tau > 0}\frac{\sum_{k=0}^{\tau-1}\gamma^k\langle r_{t+k-1}\rangle}{\sum_{k=0}^{\tau-1}\gamma^k}.$$

The optimal strategy is to play at each step the arm with the maximal Gittins index. At the beginning, each index equals $1/2$ (prior probability of winning). We start from an arbitrary arm and play until losses make its index fall below $1/2$, then switch. In the limit $l_i + w_i \to \infty$, the probability shrinks to $P(s_i) = \delta(s_i - p_i)$, and the optimal strategy reduces to choosing the arm with the highest $p_i$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Exploration vs Exploitation)</span></p>

The Gittins index elegantly solves the exploration-exploitation tradeoff. The finite-time index is larger than its infinite-time asymptotics, accounting for the possibility that the actual probability is larger than the observed one. As we play an arm, its posterior distribution narrows and the index decreases, making it possible to switch to another arm. Switching arms provides exploration and new information.

</div>

#### Model Selection and Fitting Parameters

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Choosing the Number of Fitting Parameters)</span></p>

How many fitting parameters to use? If we work with a given class of functions (say, Fourier harmonics), increasing the number $K$ of functions lets us approximate our $N$ data points better, so the deviation $D(K)$ (relative entropy between our hypothetical and the true distribution) decreases with $K$. But our data contain noise, so it does not make sense to approximate every fluctuation. Every fitting parameter introduces its own entropy $s_i$ (for white noise, all Fourier harmonics have the same $s_i$).

We minimize the mutual information of the representation: $ND(K) + \sum_{i=1}^{K}s_i$. The first term comes from imperfect fitting (relative entropy), the second from the entropy related to $K$ degrees of freedom. The extremum comes from competition between these two terms. When we obtain more data and $N$ grows large, the value of $K$ that gives a minimum usually saturates.

</div>

## 5. Stochastic Processes

So far, we quantified uncertainty mostly by counting possibilities. In this chapter, we explore and exploit the fundamental process of random walk in different environments: discrete and continuous versions, Brownian motion, modern generalizations of the second law, and fluctuation-dissipation relations.

### 5.1 Random Walk and Diffusion

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Random Walk on a Lattice)</span></p>

Consider a particle that can hop randomly to a neighboring site of a $d$-dimensional cubic lattice, starting from the origin at $t = 0$. Let $a$ be the lattice spacing, $\tau$ the time between hops, and $\mathbf{e}_i$ the orthogonal lattice vectors satisfying $\mathbf{e}_i \cdot \mathbf{e}_j = a^2\delta_{ij}$. The probability to be at a given site $\mathbf{x}$ evolves according to:

$$P(\mathbf{x}, t+\tau) = \frac{1}{2d}\sum_{i=1}^{d}\left[P(\mathbf{x}+\mathbf{e}_i, t) + P(\mathbf{x}-\mathbf{e}_i, t)\right].$$

</div>

Rewriting in a form convenient for taking the continuous limit:

$$\frac{P(\mathbf{x}, t+\tau) - P(\mathbf{x},t)}{\tau} = \frac{a^2}{2d\tau}\sum_{i=1}^{d}\frac{P(\mathbf{x}+\mathbf{e}_i, t) + P(\mathbf{x}-\mathbf{e}_i, t) - 2P(\mathbf{x},t)}{a^2}.$$

This is a finite difference approximation to the diffusion equation if we take the continuous limit $a \to 0$, $\tau \to 0$ keeping the ratio $\kappa = a^2/2d\tau$ finite:

$$(\partial_t - \kappa\Delta)P(\mathbf{x},t) = 0.$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Gaussian Solution of the Diffusion Equation)</span></p>

The space density $\rho(x,t) = P(\mathbf{x},t)a^{-d}$ satisfies the same diffusion equation. With the initial condition $\rho(\mathbf{x},0) = \delta(\mathbf{x})$, the solution is the Gaussian distribution:

$$\rho(\mathbf{x},t) = (4\pi\kappa t)^{-d/2}\exp\!\left(-\frac{x^2}{4\kappa t}\right).$$

The diffusion equation conserves total probability $\int\rho(\mathbf{x},t)\,d\mathbf{x}$, since it has the form of a continuity equation $\partial_t\rho(\mathbf{x},t) = -\mathrm{div}\,\mathbf{j}$ with the current $\mathbf{j} = -\kappa\nabla\rho$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Random Walk as a Sum of Random Variables)</span></p>

Another way to describe a random walk is to treat $\mathbf{e}_i$ as a random variable with $\langle\mathbf{e}_i\rangle = 0$ and $\langle\mathbf{e}_i\mathbf{e}_j\rangle = a^2\delta_{ij}$, so that $\mathbf{x} = \sum_{i=1}^{t/\tau}\mathbf{e}_i$. The probability of the sum is the product of Gaussian distributions of the components, with variance growing linearly with $t$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Dimensionality of Random Walks)</span></p>

A path of a random walker behaves rather like a surface than a line. The two-dimensionality of the random walk reflects the square-root diffusion law: $\langle x\rangle \propto \sqrt{t}$. Using the box-counting dimension with box size $a$, the number of elements is the number of steps $N(a) = t/\tau \propto a^{-2}$, so the dimension is $d = 2$.

The mean time spent on a given site, $\sum_{t=0}^{\infty}P(\mathbf{x},t)$, is finite or infinite depending on the space dimensionality: $\int\rho(\mathbf{x},t)\,dt \propto \int^{\infty}t^{-d/2}\,dt$, which diverges for $d \le 2$. The walker in 1d and 2d returns to any point infinite number of times. Surfaces generally intersect along curves in 3d and meet at isolated points in 4d and do not meet at $d > 4$. This is reflected in special properties of critical phenomena in 2d (where the random walker fills the surface) and 4d (where random walkers do not meet and hence do not interact).

</div>

#### Path Integral Representation

The transition probability $\rho(\mathbf{x},t;0,0)$ — the conditional probability of $x$ given the walker was at $0$ at time $-t$ — satisfies the convolution identity:

$$\rho(\mathbf{x},t;0,0) = \int\rho(\mathbf{x},t;\mathbf{x}_1,t_1)\rho(\mathbf{x}_1,t_1;0,0)\,d\mathbf{x}_1.$$

Dividing the time interval $t$ into an arbitrary large number of intervals leads to the **path integral representation**:

$$\rho(\mathbf{x},t;0,0) \to \int\mathcal{D}\mathbf{x}(t')\exp\!\left[-\frac{1}{4\kappa}\int_0^t dt'\,\dot{x}^2(t')\right].$$

The notation $\mathcal{D}\mathbf{x}(t')$ implies integration over the positions at intermediate times normalized by square roots of the time differences. The exponential gives the weight of every trajectory.

### 5.2 Brownian Motion

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Physical Setup)</span></p>

Consider a small particle of mass $M$ in a fluid. Its momentum $\mathbf{p} = M\mathbf{v}$ changes because of collisions with molecules. Thermal equipartition guarantees that the mean kinetic energy equals $T/2$ per degree of freedom. When $M$ is much larger than the molecular mass $m$, the rms particle velocity $v = \sqrt{T/M}$ is small compared to the typical molecular velocity $v_T = \sqrt{T/m}$.

The force $\mathbf{f}(\mathbf{p})$ on the particle can be expanded to first order in $\mathbf{p}$: $f_i(\mathbf{p},t) = f_i(0,t) + p_j\partial f_i(0,t)/\partial p_j$. The first term is random force with zero mean; the second term is resistance (friction), approximated as $\partial f_i/\partial p_j = -\lambda\delta_{ij}$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Stokes Formula and Langevin Equation)</span></p>

If the particle radius $R$ is larger than the mean free path $\ell$, we treat the fluid as a continuous medium with viscosity $\eta$. For a slow particle ($v \ll v_T\ell/R$), the resistance is given by the **Stokes formula**:

$$\lambda = 6\pi\eta R / M.$$

The equation of motion is the **Langevin equation**:

$$\dot{\mathbf{p}} = \mathbf{f} - \lambda\mathbf{p}.$$

Its solution is $\mathbf{p}(t) = \int_{-\infty}^{t}\mathbf{f}(t')e^{\lambda(t'-t)}\,dt'$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Fluctuation-Dissipation Theorem for Brownian Motion)</span></p>

The force $\mathbf{f}(t)$ is random with $\langle\mathbf{f}\rangle = 0$ and correlation $\langle\mathbf{f}(t')\cdot\mathbf{f}(t'+t)\rangle = 3C(t)$ decaying during the correlation time $\tau \ll \lambda^{-1}$. The momentum has Gaussian statistics $\rho(\mathbf{p}) = (2\pi\sigma^2)^{-3/2}\exp(-p^2/2\sigma^2)$ with

$$\sigma^2 = \langle p_x^2\rangle = \langle p_y^2\rangle = \langle p_z^2\rangle \approx \frac{1}{2\lambda}\int_{-\infty}^{\infty}C(t')\,dt'.$$

By equipartition $\langle p_x^2\rangle = MT$, so the friction coefficient is expressed via the force correlation function:

$$\lambda = \frac{1}{2TM}\int_{-\infty}^{\infty}C(t')\,dt'.$$

This is a particular case of the **fluctuation-dissipation theorem**: friction (dissipation) is determined by the force fluctuations.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Velocity Correlation and Mean Squared Displacement)</span></p>

The velocity autocorrelation function is:

$$\langle\mathbf{v}(t)\cdot\mathbf{v}(0)\rangle = (3T/M)\exp(-\lambda\lvert t\rvert).$$

The mean squared displacement is:

$$\langle\lvert\Delta\mathbf{r}\rvert^2(t')\rangle = \frac{6T}{M\lambda^2}(\lambda t' + e^{-\lambda t'} - 1).$$

At short times ($\lambda t' \ll 1$), the growth is **ballistic** (quadratic). At long times ($\lambda t' \gg 1$), the growth is **diffusive**: $\langle(\Delta\mathbf{r})^2\rangle \approx 6Tt'/M\lambda = 2d\kappa t$ with the **Einstein relation**:

$$\kappa = \frac{T}{M\lambda} = \frac{T}{6\pi\eta R}.$$

The diffusivity depends on the particle radius but not the mass. Measuring diffusion of particles with a known size allows one to determine the temperature.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Historical Significance of the Einstein Relation)</span></p>

With temperature in degrees, the Einstein relation contains the Boltzmann constant $k = \kappa M\lambda/T$, which was actually determined by this relation and found constant, i.e. independent of the medium and the type of particle. That proved the reality of atoms — after all, $kT$ is the kinetic energy of a single atom.

</div>

#### Brownian Motion in an External Field

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Langevin Equation in an External Potential)</span></p>

In the external field $V(\mathbf{q})$, the particle satisfies $\dot{\mathbf{p}} = -\lambda\mathbf{p} + \mathbf{f} - \partial_q V$ and $\dot{\mathbf{q}} = \mathbf{p}/M$. The Hamiltonian is $\mathcal{H} = p^2/2M + V(\mathbf{q})$, and the balance between friction $-\lambda\mathbf{p}$ and agitation $\mathbf{f}$ expressed by the fluctuation-dissipation theorem means the thermostat is in equilibrium.

</div>

In the over-damped limit ($\lambda\tau \gg 1$), we average over a moving time window, neglect acceleration, and reduce the second-order equation to the first-order:

$$\lambda M\dot{\mathbf{q}} = \mathbf{f} - \partial_q V.$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Fokker-Planck Equation)</span></p>

The probability distribution $\rho(q,t)$ satisfies the **Fokker-Planck equation**, combining diffusion and advection:

$$\frac{\partial\rho}{\partial t} = \kappa\nabla^2\rho + \frac{1}{\lambda M}\frac{\partial}{\partial q_i}\rho\frac{\partial V}{\partial q_i} = -\mathrm{div}\,\mathbf{J}.$$

The stationary solution, corresponding to thermal equilibrium with the surrounding molecules, is the **Boltzmann-Gibbs distribution**:

$$\rho(q) \propto \exp[-V(q)/\lambda M\kappa] = \exp[-V(q)/T].$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Path Integral with Drift)</span></p>

In the presence of an external field, the path integral representation becomes:

$$\rho(\mathbf{q},t;0,0) = \int\mathcal{D}\mathbf{q}(t')\exp\!\left[-\frac{1}{4\kappa}\int_0^t dt'\,\lvert\dot{\mathbf{q}} - \mathbf{w}\rvert^2\right],$$

where $\mathbf{w} = -\partial_{\mathbf{q}}V/\lambda M$ is the drift velocity. The Fokker-Planck equation can be derived from this by considering the convolution identity for an infinitesimal time change.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Analogy with Quantum Mechanics)</span></p>

The Fokker-Planck equation $\partial_t\rho = -\hat{H}_{FP}\rho$ can be viewed in a Hilbert space of functions, analogously to the Schrödinger equation $d\lvert\psi\rangle/dt = -\hat{H}_{FP}\lvert\psi\rangle$. The Fokker-Planck operator is:

$$H_{FP} = -\frac{\partial}{\partial x}\!\left(\frac{\partial V}{\partial x} + T\frac{\partial}{\partial x}\right).$$

The only difference with quantum mechanics is that the Schrödinger equation has imaginary time — corresponding to imaginary diffusivity: $(i\hbar\Delta - 2mV)\lvert\psi\rangle = 0$.

</div>

### 5.3 General Fluctuation-Dissipation Relation

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Context)</span></p>

Recently, a significant generalization of equilibrium statistical physics appeared for systems with one or few degrees of freedom deviated arbitrary far from equilibrium. This is under the assumption that the rest of the degrees of freedom is in equilibrium and can be represented by a thermostat generating thermal noise. This approach also allows one to treat non-thermodynamic fluctuations, like negative entropy change.

</div>

#### Detailed Balance

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Detailed Balance)</span></p>

Without a coordinate-dependent field $V(x)$, the transition probability is symmetric: $\rho(x',t;x,0) = \rho(x,t;x',0)$, formally manifested by the fact that the Fokker-Planck operator $\partial_x^2$ is Hermitian. This property is called **detailed balance**.

In an external time-independent potential $V$, the Gibbs steady state $\rho(x) = Z_0^{-1}\exp[-\beta V(x)]$ satisfies a **modified detailed balance**: the probability current density at the starting point times the transition probability must be equal forward and backward in equilibrium:

$$\rho(x',t;x,0)e^{-V(x)/T} = \rho(x,t;x',0)e^{-V(x')/T}.$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Adjoint Fokker-Planck Operator)</span></p>

From the detailed balance relation, $e^{-tH_{FP}^{\dagger}} = e^{V/T}e^{-tH_{FP}}e^{-V/T}$, so that:

$$H_{FP}^{\dagger} \equiv \left(\frac{\partial V}{\partial x} - T\frac{\partial}{\partial x}\right)\frac{\partial}{\partial x} = e^{V/T}H_{FP}e^{-V/T},$$

i.e. $e^{V/2T}H_{FP}e^{-V/2T}$ is Hermitian, translating the detailed balance property to that of the evolution operator.

</div>

#### Jarzynski Relation

Consider the over-damped Langevin equation $\dot{x} = -\partial_x V + \eta$ with a time-dependent potential $V(x,t)$ and random noise $\langle\eta(0)\eta(t)\rangle = 2T\delta(t)$. When the potential changes in time, the system goes away from equilibrium. Define the work done along a trajectory:

$$W_t = \int_0^t dt'\,\frac{\partial V(x(t'),t')}{\partial t'}.$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Jarzynski Relation)</span></p>

Starting from a Gibbs distribution at $t = 0$ and evolving the potential arbitrarily (possibly far from equilibrium), the mean of the exponentiated work is remarkably expressed via equilibrium partition functions:

$$\langle e^{-\beta W}\rangle = \frac{Z_t}{Z_0} = \frac{\int e^{-\beta V(x,t)}\,dx}{\int e^{-\beta V(x,0)}\,dx}.$$

Introducing the free energy $F_t = -T\ln Z_t$, this becomes $\langle e^{-\beta W}\rangle = e^{\beta(F_0 - F_t)}$ (Jarzynski, 1997).

The bracket means double averaging: over the initial distribution $\rho(x,0)$ and over all realizations of the noise $\eta(t)$ during $(0,t)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Consequences of the Jarzynski Relation)</span></p>

Using Jensen's inequality $\langle e^A\rangle \ge e^{\langle A\rangle}$, one obtains the usual second law:

$$\langle\beta W_d\rangle = \langle\Delta S\rangle \ge 0,$$

where $W_d = W - F_t + F_0$ is the dissipation (work minus free energy change). The Jarzynski relation is also a generalization of the fluctuation-dissipation theorem (considering $V(x,t) = V_0(x) - f(t)x$ and expanding for small $f$, one recovers the response via the second moment). Remarkably, the free energy difference can be determined from non-equilibrium measurements that could be arbitrarily fast, rather than adiabatically slow as in traditional thermodynamics.

</div>

#### Crooks Fluctuation Theorem

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Crooks Fluctuation Theorem)</span></p>

The modern form of the second law of thermodynamics is an equality rather than an inequality. The probability of entropy change $-\Delta S$ in a time-reversed process relates to the forward probability as (Crooks, 1999):

$$\rho^{\dagger}(-\Delta S) = \rho(\Delta S)e^{-\Delta S}.$$

Integrating gives the Jarzynski relation $\langle e^{-\Delta S}\rangle = 1$. The mean entropy production is the relative entropy between forward and backward evolution:

$$\langle\Delta S\rangle = \left\langle\ln[\rho(\Delta S)/\rho^{\dagger}(-\Delta S)]\right\rangle.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Crooks Relation for the Generalized Baker Map)</span></p>

The Crooks relation can be verified for the generalized baker map (from Section 2.4). At every step, the volume contraction factor is the Jacobian $J = r/l$ for $x \in (0,l)$ and $J = l/r$ for $x \in (l,1)$. During $n$ steps, if a rectangular element finds itself $n_1$ times in $(0,l)$ and $n_2 = n - n_1$ times in $(l,1)$, the volume contraction is $(l/r)^{n_2 - n_1}$ and the entropy change is $\Delta S = n_1\ln(r/l) + n_2\ln(l/r)$. Time reversal corresponds to $x \to 1-y$, $y \to 1-x$. Denoting the entropy production rate $\sigma = -\ln J$:

$$\frac{P(\Delta S)}{P(-\Delta S)} = \left(\frac{l}{r}\right)^{n_2-n_1} = e^{n\sigma} = e^{\Delta S}.$$

</div>

#### Non-potential Forces and the Kramers Equation

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Kramers Equation with Non-potential Forces)</span></p>

In a multi-dimensional case, apart from a time-dependent potential, one can add a non-potential (non-gradient) coordinate-dependent force $\mathbf{F}(\mathbf{q})$. The equation on the full phase-space distribution $\rho(\mathbf{p},\mathbf{q},t)$ then takes the form of the **Kramers equation**:

$$\partial_t\rho = \lbrace\mathcal{H}, \rho\rbrace + T\Delta_p\rho + \partial_{\mathbf{p}}\rho\mathbf{F} = H_K\rho,$$

where $\mathcal{H} = p^2/2M + V$. Only without $\mathbf{F}$, the Gibbs distribution $\exp(-\mathcal{H}/T)$ is a steady solution and detailed balance holds. The non-potential force breaks the detailed balance. A close analog of the Jarzynski relation holds for the entropy production rate averaged during time $t$:

$$\sigma_t = \frac{1}{tT}\int_0^t(\mathbf{F}\cdot\dot{\mathbf{q}})\,dt,$$

satisfying $P(\sigma_t)/P(-\sigma_t) \propto e^{t\sigma_t}$. These are called **detailed fluctuation-dissipation relations** and are stronger than the integral relations of the Jarzynski type.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Information Processing and the Second Law)</span></p>

When information processing is involved, it must be treated on equal footing, allowing one to decrease the work and dissipation below the free energy difference:

$$\langle e^{-\beta W_d - I}\rangle = \langle e^{-\Delta S}\rangle = 1,$$

(Sagawa and Uedo, 2012). This is a generalization of the inequality $\langle W_d\rangle \ge -IT = -T\Delta S$ encountered in Section 4.2.

</div>

### 5.4 Stochastic Web Surfing and Google's PageRank

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Motivation)</span></p>

For efficient information retrieval from the Web Library, webpages need to be ranked by their **importance** to order search results. A reasonable measure is to count the number of links referring to a page. Not all links are equal — links from more important pages bring more importance, while a link from a page with many outgoing links brings less importance. Two rules emerge: (i) every page relays its importance score to the pages it links to, dividing equally, (ii) the importance score of a page is the sum of all scores obtained by links.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(PageRank as an Eigenvector Problem)</span></p>

For the Internet with $n$ pages, organize scores into a vector $\mathbf{p} = \lbrace p_1,\ldots,p_n\rbrace$ with $\sum_{i=1}^{n}p_i = 1$. The rules give $p_i = \sum_j p_j/n_j$ where $n_j$ is the number of outgoing links on page $j$. We are looking for the eigenvector of the **hyperlink matrix** $\hat{A}$, where $a_{ij} = 1/n_j$ if $j$ links to $i$ and $a_{ij} = 0$ otherwise:

$$\mathbf{p}\hat{A} = \mathbf{p}.$$

The iterative algorithm to find this eigenvector is called **PageRank** (Brin and Page, 1998). Starting from $p_i(0) = 1/n$, the new distribution is generated by the Markov chain:

$$\mathbf{p}(t+1) = \mathbf{p}(t)\hat{A}.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Convergence Problems: Sinks and Loops)</span></p>

Two problems arise: some pages do not link to any other page (sinks — rows of zeroes in $\hat{A}$, accumulating score without sharing it), and loops can trap the surfer. If the largest eigenvalue $\lambda_1$ of $\hat{A}$ exceeds unity, iterations diverge; if $\lambda_1 < 1$, iterations converge to zero. We need $\lambda_1 = 1$ corresponding to a unique eigenvector. Convergence speed is determined by the second largest eigenvalue $\lambda_2 < 1$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Google Matrix and Random Teleportation)</span></p>

To release the random surfer from sinks and loops, the original PageRank algorithm allows random teleportation: the surfer clicks on a link with probability $d$ or opens a random page with probability $1-d$. This replaces $\hat{A}$ by the **Google matrix**:

$$\hat{G} = d\hat{A} + (1-d)\hat{E},$$

where $\hat{E} = \mathbf{e}\mathbf{e}^T/n$ is the teleportation matrix with all entries $1/n$. Now all matrix entries $g_{ij}$ are strictly positive and the graph is fully connected.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Perron-Frobenius and PageRank Convergence)</span></p>

The matrix $\hat{G}$ has positive elements in every column whose sum is unity — such matrices are called **stochastic**. Every stochastic matrix has unity as the largest eigenvalue: since $\sum_j g_{ij} = 1$, the vector $\mathbf{e}$ is an eigenvector of the transposed matrix $\hat{G}^T\mathbf{e} = \mathbf{e}$.

By convexity: for any vector $\mathbf{p}$, every element of $\mathbf{p}\hat{G}$ is a convex combination $\sum_j p_j g_{ij}$, which cannot exceed the largest element of $\mathbf{p}$. Therefore no eigenvector with an eigenvalue exceeding unity can exist. This is a particular case of the **Perron-Frobenius theorem**: the eigenvalue with the largest absolute value of a positive square matrix is positive, belongs to a positive eigenvector, and all other eigenvectors are smaller in absolute value (Markov 1906, Perron 1907).

The iterative process $\mathbf{p}(t+1) = \hat{G}\mathbf{p}(t)$ cannot be caught in a loop (since $G_{ii} \neq 0$) and converges. The eigenvalues of $\hat{G}$ are $1, d\lambda_2,\ldots,d\lambda_n$ where $\lbrace\lambda_i\rbrace$ are eigenvalues of $\hat{A}$. The standard Google choice $d = 0.85$ comes from estimating how often an average surfer uses bookmarks. The process usually converges after about 50 iterations.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Personalized PageRank and Markov Chains)</span></p>

One can design a personalized ranking by replacing the teleportation matrix by $\hat{E} = \mathbf{e}\mathbf{v}^T$, where $\mathbf{v}$ has all nonzero entries chosen according to the individual user's history of searches and visits. The sequence of probability vectors defined by the PageRank iterations is a **Markov chain**: three random quantities $X \to Y \to Z$ form a Markov triplet if $Y$ is completely determined by $X,Z$ while $X,Z$ are independent conditional on $Y$, that is $I(X,Z\lvert Y) = 0$.

</div>

## 6. Quantum Information

Since our world is fundamentally quantum mechanical, it is interesting what that reveals about the nature of information and uncertainty. Quantum mechanics is inherently statistical — predictions and measurement results are truly random (not because we did not bother to learn more about the system). Measurement dramatically differs in a quantum world since it irreversibly changes the system and this change cannot be made arbitrarily small.

Apart from that fundamental difference, interest in quantum information is also pragmatic. Classical systems are limited by locality (operations have only local effects) and by the fact that systems can be in only one state at a time. A quantum system can be in a superposition of many different states simultaneously, meaning quantum evolution exhibits interference effects and proceeds in a space of factorially more dimensions than the respective classical system — this is the source of the parallelism of quantum computations. Moreover, spatially separated quantum systems may be **entangled** with each other and operations may have non-local effects.

### 6.1 Quantum Mechanics and Quantum Information

Quantum mechanics is mathematically based on linear algebra — vector states and linear operations on them. A state of a physical system is a vector, denoted $\psi_i$ or in Dirac notation $\lvert i\rangle$. The dual (row) vector is $\langle i\rvert$ and the inner product is $\langle i\lvert j\rangle$. The fundamental statement is that any system can be in a *pure* state $\psi_i$ or in a superposition of states, $\psi = \sum_i \sqrt{p_i}\,\psi_i$. The possibility of a superposition is the total breakdown from classical physics, where those states (say, with different energies) are mutually exclusive.

There are two things we can do with a quantum state: let it evolve (unitarily) without touching it, or measure it. Measurement is classical — it produces one and only pure state from the initial superposition; immediately repeated measurements produce the same outcome. However, repeated measurement of identically prepared initial superpositions find different states, state $i$ appearing with probability $p_i$. A property that can be measured is called an **observable** and is described by a self-adjoint operator (matrix).

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Heisenberg Uncertainty Principle)</span></p>

There is uncertainty already in a pure state of an isolated quantum system. If the operators of two observables are non-commuting, $[\hat{A},\hat{B}] = \hat{A}\hat{B} - \hat{B}\hat{A} \neq 0$, then the product of their variances is restricted from below:

$$\lvert\langle\psi\lvert[\hat{A},\hat{B}]\rvert\psi\rangle\rvert^2 \le 4\langle\psi\lvert\hat{A}^2\rvert\psi\rangle\langle\psi\lvert\hat{B}^2\rvert\psi\rangle.$$

For momentum $\hat{A} = \hat{\mathbf{p}} - \langle\mathbf{p}\rangle$ and coordinate $\hat{B} = \hat{\mathbf{q}} - \langle\mathbf{q}\rangle$, using $[\hat{p}_x, x] = -i\hbar$:

$$\sqrt{\sigma_p\sigma_q} \ge \hbar/2.$$

This means we cannot describe quantum states as points in the phase space $(p,q)$. Quantum entanglement is ultimately related to the fact that one cannot localize quantum states in a finite region — if coordinates are fixed somewhere, then the momenta are not.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Entropic Uncertainty Relation)</span></p>

The Heisenberg relation is the "undergraduate version." Taking logarithms: $\log(2\sigma_p/\hbar) + \log(2\sigma_q/\hbar) = S(p) + S(q) \ge 0$, recasting it as requirements on the entropies of the Gaussian probability distributions. For non-Gaussian distributions, the variance formulation does not make sense, yet the entropic uncertainty relation remains universally valid (Deutsch 1982).

If we measure a quantum state $\psi$ by projecting onto the orthonormal basis $\lbrace\lvert x\rangle\rbrace$, the outcomes define a classical distribution $p(x) = \langle x\lvert\psi\rvert x\rangle$. If $\lbrace\lvert z\rangle\rbrace$ is another orthonormal basis and the projecting operators on $x,z$ do not commute, there is a tradeoff between our uncertainty about $X$ and $Z$:

$$S(X) + S(Z) \ge \log(1/c), \quad c = \max_{x,z}\lvert\langle x\lvert z\rangle\rvert^2.$$

Two bases $\lbrace\lvert x\rangle\rbrace$, $\lbrace\lvert z\rangle\rbrace$ for a $d$-dimensional space are called **mutually unbiased** if $\lvert\langle x_i\lvert z_k\rangle\rvert^2 = 1/d$ for all $i,k$. For measurements in two mutually unbiased bases on a pure state:

$$S(X) + S(Z) \ge \log d.$$

</div>

#### Qubits

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Qubit)</span></p>

A **qubit** is a quantum system having only two states: $\lvert 0\rangle$ and $\lvert 1\rangle$. The most general state of a qubit $A$ is a superposition $\psi_A = a\lvert 0\rangle + b\lvert 1\rangle$ with $\lvert a\rvert^2 + \lvert b\rvert^2 = 1$. Any observable has the expectation value:

$$\langle\psi_A\lvert\hat{O}_A\rvert\psi_A\rangle = \lvert a\rvert^2\langle 0\lvert\hat{O}_A\rvert 0\rangle + \lvert b\rvert^2\langle 1\lvert\hat{O}_A\rvert 1\rangle + (a^*b + ab^*)\langle 0\lvert\hat{O}_A\rvert 1\rangle.$$

If the overall phase does not matter, a qubit is characterized by two real numbers (say the amplitude $\lvert a\rvert$ and the relative phase between $a$ and $b$). The qubit represents the unit of quantum information the same way the bit represents the unit of classical information — but quantum systems operate with much more information, since one needs many bits to record a complex number with reasonable precision.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Qubit vs Classical Bit)</span></p>

A qubit is not a classical bit: it can be in a superposition, nor can it be considered a random ensemble of classical bits with probability $\lvert a\rvert^2$ in the state $\lvert 0\rangle$, because the phase difference of the complex numbers $a,b$ matters. Measurements of a qubit bring either $\lvert 0\rangle$ with probability $\lvert a\rvert^2$ or $\lvert 1\rangle$ with probability $\lvert b\rvert^2 = 1 - \lvert a\rvert^2$. A quantum coin can defy gravity and stand on its edge at an arbitrary angle, but any measurement collapses it to one side — either heads or tails. We cannot measure the quantum state of the qubit (the complex numbers $a,b$), but we can communicate it and manipulate it indirectly via entanglement.

</div>

### 6.2 Quantum Statistics and Entanglement Entropy

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Density Matrix)</span></p>

To consider subsystems, we pass from quantum mechanics to quantum statistics and introduce the **density matrix**. Consider a composite system $AB$ in a pure state $\psi_{AB}$. Denote by $x$ the coordinates on $A$ and by $y$ on $B$. The expectation value of any $O(x)$ is $\bar{O} = \sum_{x,y}\psi_{AB}^*(x,y)\hat{O}(x)\psi_{AB}(x,y)$.

For independent subsystems, $\psi_{AB}(x,y) = \psi_A(x)\psi_B(y)$ and $\bar{O} = \sum_x \psi_A^*(x)\hat{O}(x)\psi_A(x)$. But generally, dependencies on $x$ and $y$ are not factorized. We then characterize $A$ by the **density matrix**:

$$\rho(x,x') = \sum_y \psi_{AB}^*(x',y)\psi_{AB}(x,y),$$

so that $\bar{O} = \sum_x[\hat{O}(x)\rho(x,x')]_{x'=x}$. The density matrix is Hermitian, has all non-negative eigenvalues and unit trace. Every such matrix can be "purified" — presented (non-uniquely) as a density matrix of some pure state $\psi_{AB}$ in the extended system $AB$. Purification is quantum mechanical with no classical analog.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Schmidt Decomposition and Entanglement)</span></p>

A general pure state $\psi_{AB}$ can be decomposed as a sum of tensor products (**Schmidt decomposition**):

$$\psi_{AB} = \sum_i \sqrt{p_i}\,\psi_A^i \otimes \psi_B^i,$$

by applying unitary transformations in $A$ and $B$ independently ($\psi_{AB} \to U\psi_{AB}V$). The $\sqrt{p_i}$ are positive eigenvalues and $\sum_i p_i = 1$. The density matrix for subsystem $A$ is then:

$$\rho_A = \sum_i p_i\lvert\psi_A^i\rangle\langle\psi_A^i\rvert.$$

If there is more than one term in this sum, we call subsystems $A$ and $B$ **entangled**. There is no factorization of the dependencies in such a state.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Mixed States vs Superpositions)</span></p>

A statistical density matrix describes a **mixed state** — an ensemble of states. A mixed state described by a matrix must be distinguished from a quantum-mechanical superposition described by a vector. The superposition is in both states simultaneously; the ensemble is in perhaps one or perhaps the other, characterized by probabilities. The uncertainty appears because we do not have any information about the state of the $B$-subsystem.

For a two-qubit system with $\psi_{AB} = a\lvert 00\rangle + b\lvert 11\rangle$, the density matrix of $A$ is:

$$\rho_A = \lvert a\rvert^2\lvert 0\rangle\langle 0\rvert + \lvert b\rvert^2\lvert 1\rangle\langle 1\rvert = \begin{bmatrix}\lvert a\rvert^2 & 0 \\ 0 & \lvert b\rvert^2\end{bmatrix}.$$

Being in a mixed state (where relative phases of $\lvert 0\rangle,\lvert 1\rangle$ are experimentally inaccessible) is not the same as being in a superposition.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Von Neumann Entropy and Entanglement Entropy)</span></p>

The **von Neumann entropy** of a density matrix $\rho_A$ is defined analogously to the Gibbs-Shannon entropy:

$$S(\rho_A) = -\mathrm{Tr}\,\rho_A\log\rho_A.$$

Since we deal with diagonalizable matrices, $\log\rho = \sum_k \log(p_k)\lvert k\rangle\langle k\rvert$ for diagonal $\rho = \sum_k p_k\lvert k\rangle\langle k\rvert$. The von Neumann entropy:

- Is invariant under unitary transformations $\rho_A \to U\rho_A U^{-1}$.
- Is non-negative, equals zero only for a pure state.
- Reaches its maximum $\log d$ for equipartition (all $d$ non-zero eigenvalues equal).
- The purifying system $B$ has the same entropy as $A$ (since the same $p_i$ appear in the density matrix of both).

The purely quantum correlation between different parts is called **entanglement**, and the von Neumann entropy of a subsystem of a pure state is called **entanglement entropy**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Quantum vs Classical Mutual Information)</span></p>

Classically, the mutual information of perfectly correlated quantities equals each of their entropies, and the sum is at most twice that. But in quantum mechanics, when $AB$ is in an entangled pure state, $S(\rho_{AB}) = 0$ while $S(\rho_A) = S(\rho_B) > 0$, so the quantum mutual information is:

$$I(\rho_{AB}) = S(\rho_A) + S(\rho_B) - S(\rho_{AB}) = 2S(\rho_A).$$

Quantum correlations are stronger than classical: nonlocality of information encoding is raised to a whole new level in the quantum world.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Entanglement Entropy of Two Qubits)</span></p>

For $\psi_{AB} = a\lvert 00\rangle + b\lvert 11\rangle$, the von Neumann entropy is $S(\rho_A) = -\lvert a\rvert^2\log_2\lvert a\rvert^2 - \lvert b\rvert^2\log_2\lvert b\rvert^2$. The maximum $S(\rho_A) = 1$ is reached when $\lvert a\rvert^2 = \lvert b\rvert^2 = 1/2$ — the state of **maximal entanglement**. When we trace out $B$ (or $A$), we wipe out this information: any measurement on $A$ or $B$ alone cannot tell us anything about the pair, since both outcomes are equally probable. Conversely, when either $b \to 0$ or $a \to 0$, the entropy goes to zero and measurements give definite information on the state.

</div>

#### General Uncertainty Relation for Mixed States

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(General Uncertainty Relation)</span></p>

If we measure a mixed state $\rho$ by projecting onto orthonormal basis $\lbrace\lvert x\rangle\rbrace$, the outcomes define a density matrix $\hat{M}_x\rho = \rho_x = \sum_x\lvert x\rangle\langle x\lvert\rho\rvert x\rangle\langle x\rvert$. Using monotonicity of the relative entropy $D(\rho\lvert\rho_x)$ under the action of the measurement $\hat{M}_z$:

$$D(\rho\lvert\rho_x) \ge D(\hat{M}_z\rho\lvert\hat{M}_z\rho_x) = D(\rho_z\lvert\hat{M}_z\rho_x) = -S(Z) - \mathrm{Tr}\,\rho_z\log\hat{M}_z\rho_x.$$

Since $\log\!\left(\sum_x\langle z\lvert x\rangle\langle x\lvert\rho\rvert x\rangle\langle x\lvert z\rangle\right) \le \log\!\left(\max_{x,z}\lvert\langle x\lvert z\rangle\rvert^2\sum_x\langle x\lvert\rho\rvert x\rangle\right) = \log\!\left(\max_{x,z}\lvert\langle x\lvert z\rangle\rvert^2\right)$, the generalization for a mixed state of the uncertainty relations is:

$$S(X) + S(Z) \ge \log(1/c) + S(\rho), \quad c = \max_{x,z}\lvert\langle x\lvert z\rangle\rvert^2.$$

The von Neumann entropy quantifies the increase in uncertainty compared to a pure state.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Measurements Increase Shannon Entropy Beyond Von Neumann Entropy)</span></p>

Let $\rho_A = \sum_k p_k\lvert\phi_A^k\rangle\langle\phi_A^k\rvert$ be diagonal in the basis of eigenvectors $\lbrace\lvert\phi_A^k\rangle\rbrace$, but we measure by projecting onto a different orthogonal set $\lbrace\lvert\psi_A^i\rangle\rbrace$. Outcome $i$ occurs with probability $q'(i) = \sum_k p_k D_{ik}$ where $D_{ik} = \lvert\langle\psi_A^i\lvert\phi_A^k\rangle\rvert^2$ is a doubly stochastic matrix. Then:

$$S(q') = S(p) + \sum_{ik}p_k D_{ik}\log\!\left(\sum_n D_{in}/p_k\right) = S(p) + D(q'\lvert p) \ge S(p) = S(\rho_A).$$

Such measurements are less predictable — the Shannon entropy of the measurement outcomes is larger than the von Neumann entropy.

</div>

#### Coming to Equilibrium

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Quantum Thermalization)</span></p>

When a classical system is attached to a thermostat, it comes to thermal equilibrium, attaining maximum entropy determined by the temperature. But what if a quantum system is attached to a large system with which they together form a pure quantum state with zero entropy? Thermalization takes place for any subsystem of a large system if the dynamics is ergodic, characterized by the growth of the entanglement entropy. The system as a whole acts as a thermal reservoir for its subsystems, provided those are small enough.

Information initially encoded locally in an out-of-equilibrium state becomes encoded more and more nonlocally as the system evolves, eventually becoming invisible to an observer confined to the subsystem. The relative entropy between the evolving density matrix $\rho$ and the Gibbs state $\rho_0 = Z^{-1}\exp(-\beta H)$ gives:

$$D(\rho\lvert\rho_0) = \mathrm{Tr}\,\rho\ln\rho - \mathrm{Tr}\,\rho\ln\rho_0 = \beta[F(\rho) - F_0] \ge 0,$$

where $F(\rho) = E - \beta^{-1}S(\rho)$. Therefore, the Gibbs state has the lowest free energy. Unitary evolution of the subsystem and its environment induces a decrease (by monotonicity) of $D(\rho,\rho_0)$, eventually bringing the subsystem to the Gibbs state.

</div>

### 6.3 Quantum Communications

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Classical Information from Quantum Systems)</span></p>

How many bits of classical information can be recovered from a quantum system? Even though any qubit potentially contains a complex number, any measurement will only give one or another state, so a pure state of a qubit can store one classical bit. The four orthogonal maximally entangled states of a qubit pair, $(\lvert 00\rangle \pm \lvert 11\rangle)/\sqrt{2}$ and $(\lvert 01\rangle \pm \lvert 10\rangle)/\sqrt{2}$, can store two bits. Generally, sending a quantum system whose state is determined by a $d$-dimensional complex vector, one can send at most $\log d$ bits of classical information.

The maximal number of bits Alice can get from her measurements about Bob (and vice versa) is the classical mutual information $I(C_A,C_B)$. By monotonicity: $I(C_A,C_B) \le I(\rho_{AB}) \le \log d$, where $I(\rho_{AB}) = S(\rho_A) + S(\rho_B) - S(\rho_{AB})$ is the quantum mutual information.

</div>

#### Quantum Information Compression

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Schumacher's Noiseless Quantum Coding)</span></p>

How much can a message of quantum states be compressed? The letters are quantum states picked with probabilities $p_k$, each described by a density matrix, and the message is a tensor product of $N$ vectors. If the states are mutually orthogonal and the density matrix is diagonal, the answer is the Shannon entropy $S(p) = -\sum_k p_k\log p_k$ (same as von Neumann entropy in this case), and compression proceeds as for a classical source.

The new issue is that **non-orthogonal quantum states cannot be perfectly distinguished**. The difference between the Shannon entropy of the mixture and the von Neumann entropy of the matrix, $S\lbrace q_i\rbrace - S(\rho_A)$, is non-negative and quantifies how much distinguishability is lost when we mix non-orthogonal pure states. The best possible rate of quantum information transfer is given by the **von Neumann entropy** $S(\rho)$ of the density matrix of the source.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Compression of Non-orthogonal States)</span></p>

Consider a source producing non-orthogonal states $\lvert 0\rangle$ and $\lvert s\rangle = (\lvert 0\rangle + \lvert 1\rangle)/\sqrt{2}$ with probabilities $p$ and $1-p$. The Shannon entropy is $S(p) = 1$ bit (for $p = 1/2$). The density matrix in the $\lvert 0\rangle,\lvert 1\rangle$ basis is:

$$\rho = \frac{1}{4}\begin{bmatrix}3 & 1 \\ 1 & 1\end{bmatrix}.$$

The eigenvalues are $q = (2+\sqrt{2})/4 = \sin^2(\pi/8)$ and $1-q = \cos^2(\pi/8)$. The von Neumann entropy is $S(\rho) \approx 0.6$ bits, indeed less than $S(p) = 1$ bit. The redundancy from the non-orthogonality of our alphabet allows tighter compression.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Holevo Information and Transfer Rates)</span></p>

The information transfer rate depends on the type of states used:

1. **Orthogonal pure states**: rate equals $S(p) = S(\rho)$.
2. **Non-orthogonal pure states**: rate equals $S(\rho)$, which is less than $S(p)$.
3. **Orthogonal mixed states**: rate equals $S(p)$, which is less than $S(\rho)$.

For non-orthogonal mixed states, it is believed that the **Holevo information**:

$$\chi(\rho_k, p_k) = S\!\left(\sum_k p_k\rho_k\right) - \sum_k p_k S(\rho_k)$$

defines the limiting compression rate in all cases. The Holevo information is a kind of mutual information — it tells us how much, on average, the von Neumann entropy of an ensemble is reduced when we know which preparation was chosen. Classical Shannon information is a mutual von Neumann information.

</div>

### 6.4 Conditional Entropy and Teleportation

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Quantum Conditional Entropy)</span></p>

Similar to the classical conditional entropy, one defines for von Neumann entropy:

$$S(\rho_{AB}\lvert\rho_B) = S(\rho_{AB}) - S(\rho_B).$$

However, this is **not** an entropy conditional on something known — it is not zero for correlated quantities but **negative**! For pure $AB$: $S(\rho_{AB}\lvert\rho_B) = -S(\rho_B) < 0$.

Classically, $S(A\lvert B)$ measures how many additional bits Alice must send to Bob (who already has $B$) for him to fully determine $A$. The quantum analog involves qubits rather than classical bits.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Quantum Teleportation)</span></p>

**Teleportation** moves quantum states without a quantum channel. Suppose Alice has a qubit $A_0$ in unknown state $\alpha\lvert 0\rangle + \beta\lvert 1\rangle$ that she wants Bob to recreate. She can only send classical bits. However, Alice and Bob have previously shared an entangled pair $A_1,B_1$ in the state:

$$\psi_{A_1B_1} = \frac{1}{\sqrt{2}}(\lvert 00\rangle + \lvert 11\rangle)_{A_1B_1}.$$

Alice performs a joint measurement on $A_0A_1$ in the **Bell basis** (four maximally entangled states):

$$\frac{1}{\sqrt{2}}(\lvert 00\rangle \pm \lvert 11\rangle)_{A_0A_1}, \quad \frac{1}{\sqrt{2}}(\lvert 01\rangle \pm \lvert 10\rangle)_{A_0A_1}.$$

This measurement is chosen so that Alice learns nothing about $A_0$ (each outcome is equally likely regardless of $\alpha,\beta$), yet it creates instantaneous correlation: after the measurement, $B_1$ is in a state that differs from the original $\alpha\lvert 0\rangle + \beta\lvert 1\rangle$ by at most a known unitary transformation. Alice sends Bob two classical bits (her measurement outcome), and Bob applies the corresponding operation on $B_1$ to recreate the state of $A_0$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Teleportation Protocol in Detail)</span></p>

The initial state of $A_0A_1B_1$ is:

$$\frac{1}{\sqrt{2}}(\alpha\lvert 000\rangle + \alpha\lvert 011\rangle + \beta\lvert 100\rangle + \beta\lvert 111\rangle)_{A_0A_1B_1}.$$

If Alice's measurement reveals $A_0A_1$ in the state $\frac{1}{\sqrt{2}}(\lvert 00\rangle - \lvert 11\rangle)$, then $B_1$ is in the state $(\alpha\lvert 0\rangle - \beta\lvert 1\rangle)_{B_1}$. Bob multiplies his qubit by $\begin{bmatrix}1 & 0 \\ 0 & -1\end{bmatrix}$ (switching the sign of the second component) to recover $\alpha\lvert 0\rangle + \beta\lvert 1\rangle$. The beauty is that Alice learnt and communicated not *what* was the state of $A_0$, but *how to recreate it*.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Role of Conditional Entropy in Teleportation)</span></p>

Teleportation is possible only when $S(\rho_{AB}\lvert\rho_B)$ is non-positive. Notice that $A_1$ and $B_1$ are maximally entangled, so $S(\rho_B) = \log_2 2 = 1$, while $A_1B_1$ is a pure state with $S(\rho_{A_1B_1}) = 0$. Adding the reference system $R$ maximally entangled with $A_0$:

$$S(\rho_{AB}\lvert\rho_B) = S(\rho_{A_0A_1}) - S(\rho_{B_1}) = S(\rho_{A_0}) - S(\rho_{B_1}) = 0.$$

If $S(\rho_{AB}\lvert\rho_B) > 0$, Alice can simply send her states, or do teleportation by first sending Bob entangled pairs. Each pair sent increases $S(\rho_B)$ by 1 and decreases $S(\rho_{AB}\lvert\rho_B)$ by 1. So $S(\rho_{AB}\lvert\rho_B)$, if positive, is the number of qubits Alice must send to enable teleportation. Negative quantum conditional entropy measures the number of possible future qubit teleportations — entanglement is an important resource in quantum communications.

</div>

### 6.5 Black Hole is a Way Out of Our World

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Black Holes and Information)</span></p>

A black hole eliminates all uncertainty about a system by swallowing it, forever removing it from our world. Our belief that uncertainty can only increase leads us to the entropy of a black hole and the ultimate restriction on the amount of information that can be encoded in a physical system.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Black Hole Entropy — Area Law)</span></p>

A black hole is an object whose size is smaller than its horizon $r_h = 2GM/c^2$, where $M$ is the mass and $G$ the gravitational constant. The quantum entanglement entropy (between interior and exterior) is thought to be responsible for the entropy of black holes.

The energy is $E_{BH} = Mc^2 = c^4 r_h/2G$. The temperature is determined by the Hawking radiation (particle-antiparticle pairs straddling the horizon): $T = \hbar c/4\pi r_h$. Integrating the equation of state $T = dE/dS$:

$$S_{BH} = \frac{\pi r_h^2 c^3}{G\hbar} = \frac{\pi r_h^2}{l_p^2} = \frac{4\pi GM^2}{\hbar c},$$

where $l_p = \sqrt{\hbar G/c^3} \simeq 10^{-17}\,\text{cm}$ is the **Planck length**. The entropy is proportional to the **area** of the horizon (in Planck units), not the volume — this is the **area law**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Holographic Principle)</span></p>

The area law behavior of entanglement entropy could be related to the **holographic principle** — the conjecture that the information contained in a volume of space can be completely encoded by the degrees of freedom which live on the boundary of that region (like a hologram where a 3d image is encoded on a 2d surface).

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Bekenstein Bound)</span></p>

The **Bekenstein bound** gives a universal limit on the entropy of a physical system. On dimensional grounds, the entropy must be the total energy $E$ divided by a temperature. The temperature must be determined by the system size $R$ — confining a system to a smaller region increases kinetic energy. The only combination with the dimensionality of energy from $R$ and world constants $\hbar,c$ is $\hbar c/R$. This suggests:

$$S \le \frac{2\pi RE}{\hbar c}$$

(Bekenstein 1981, Casini 2008). Assuming the body is not itself a black hole ($R > r_h(E) = 2GE/c^4$), the bound can be stated purely in terms of the radius:

$$S \le \frac{\pi R^2 c^3}{G\hbar}.$$

Comparing with the black hole entropy, a system must be a black hole to realize the capacity limit. In a thermodynamic limit, entropy is extensive (proportional to volume). When there is so much matter or so little space that the system turns into a black hole, the entropy becomes proportional to the area — the holographic bound.

</div>

## 7. Conclusion

### 7.1 Take-Home Lessons

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Thermodynamics and Entropy)</span></p>

1. **Thermodynamics** studies restrictions imposed by hidden on observable. Energy $E$ is conserved; its changes divide into work (observable) and heat (hidden). Entropy $S$ can only increase for a closed system and reaches its maximum in thermal equilibrium, where $S(E)$ is a convex function. All available states lie below this convex curve in the $S - E$ plane.

2. **Convexity** of $E(S)$ introduces temperature as $\partial E/\partial S$. Extremum of the entropy means temperatures of connected subsystems are equal in equilibrium. The entropy increase (second law) imposes restrictions on thermal engine efficiency: $W/Q_1 \le 1 - T_2/T_1$. If information processing generates $\Delta S$, the energy price of information is $Q = (T_2\Delta S + W)/(1 - T_2/T_1)$.

3. **Statistical physics** arises from incomplete knowledge. The Boltzmann entropy of a closed system is $S = \log\Gamma$; the Gibbs entropy for a subsystem is $S = -\sum_i w_i\log w_i$; the probability distribution maximizing entropy for a given mean energy gives $\log w_i \propto -E_i$. Information theory generalizes this approach.

4. **Irreversibility** of entropy growth seems to contradict time-reversible Hamiltonian dynamics. The lesson: if we follow all degrees of freedom precisely, entropy is conserved. But if we follow only part, the entropy of that part grows as it interacts with the rest. Similarly, thermalization of a quantum subsystem increases entanglement entropy as information is encoded nonlocally.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Phase Space, Sums, and Convexity)</span></p>

5. **Total entropy growth** can appear even if we follow all degrees of freedom, when we track finite phase-space regions. Instability leads to separation of trajectories, spreading and mixing in phase space — like flows of an incompressible liquid. For unstable systems, any extra digit in precision adds a new degree of freedom.

6. **Sum of independent random numbers** $X = \sum_{i=1}^{N}y_i$: (i) $X$ approaches its mean value exponentially fast in $N$ (law of large numbers); (ii) the distribution $\mathcal{P}(X)$ is Gaussian in the vicinity $N^{-1/2}$ of the maximum (central limit theorem); (iii) for larger deviations, $\mathcal{P}(X) \propto e^{-NH(X/N)}$ with $H \ge 0$ and $H(\langle y\rangle) = 0$ (large deviations). The probability is independent of a sequence for most of them, the number of typical sequences grows exponentially, and **the entropy is the rate**.

7. **Convexity** is used throughout: in thermodynamics (extremum on boundary, Legendre transforms), for exponential separation of trajectories, for Jensen's inequality $\langle e^{-\Delta S}\rangle = 1 \Rightarrow \langle\Delta S\rangle \ge 0$, and to establish hierarchies and find distributions that provide an extremum.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Information Theory Core)</span></p>

8. **Entropy as measure of uncertainty**: information is $\log_2$ of the number of equally probable possibilities (Boltzmann entropy) or the mean logarithm when probabilities differ (Shannon-Gibbs entropy). Convexity of $-w\log w$ proves that information/entropy has its maximum for equal probabilities.

9. **Coding and typical sequences**: the number of typical binary sequences of length $N$ is $2^{NS}$, which cannot exceed $2^N$. Efficient encoding uses words of lengths from unity to $NS$, which is less than $N$ when probabilities of 0 and 1 are unequal. The entropy is both the mean and the fastest rate of information reception.

10. **Channel capacity**: if the channel $B \to A$ makes errors, the conditional entropy $S(B\lvert A) = S(A,B) - S(A)$ is the mean rate of growth of the number of possible errors. The transmission rate is the mutual information $I(A,B) = S(B) - S(B\lvert A)$. The channel capacity is the maximum of $I$ over all source statistics — the maximal rate of asymptotically error-free transmission.

11. **Rate distortion theory** looks for the minimal rate $I$ of information transfer under the restriction that signal distortion does not exceed $\mathcal{D}$, by minimizing $I + \beta\mathcal{D}$. Another minimization task: separate signal into independent components with as little mutual information between them as possible.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Inference, Free Energy, and Physical Nature)</span></p>

12. **Bayes' rule** $P(h\lvert e) = P(h)P(e\lvert h)/P(e)$ gives the posterior probability that a hypothesis is correct. Bayes' approach demonstrates there is no inference without prior assumption. The **relative entropy** $D(p\lvert q) = \langle\log_2(p/q)\rangle$ measures the difference between the true distribution $p$ and the hypothetical $q$ — it is the rate with which the error probability grows. Both the relative entropy and mutual information are invariant under invertible transformations, which facilitates their ever-widening applications.

13. **Maximum entropy principle**: how best to guess a distribution from data? Maximize the entropy under constraints — "the truth and nothing but the truth." This explains constrained equilibria and gives rise to free energy $F = E - TS$ as the functional whose (conditional) minima describe physical systems and serve as optimization tools, from Bayesian brains to machine-learning algorithms.

14. **Information is physical**: to learn $\Delta S = S(A) - S(A,M)$ one does the work $T\Delta S$. To erase information, one converts $TS(M)$ into heat. The energetic price of a cycle is $T\cdot I(A,M)$. The Bekenstein bound limits how much entropy one can squeeze inside a given radius — surprisingly proportional to the area, not the volume, and realized by black holes.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Renormalization, Quantum Information, and the Second Law)</span></p>

15. **Renormalization Group** is the best known way to forget information. The entropy of the partially averaged and renormalized distribution is the proper measure of forgetting. In physical systems with many degrees of freedom, the mutual information (between remaining and eliminated degrees of freedom, or between different parts) shows the area law when $I$ is sub-extensive.

16. **Quantum information** theory was initially motivated by the area law for black hole entropy. The density matrix and von Neumann entropy are fundamental — the quantum entropy of the whole can be less than the entropy of a part. All entropy of a subsystem of a pure state comes from entanglement, which is a quantum sibling of mutual information.

17. **Two stronger forms of the second law**: the first, $\langle e^{-\Delta S}\rangle = 1$, is the analog of a Liouville theorem. The second relates forward and backward process probabilities: $\rho^{\dagger}(-\Delta S) = \rho(\Delta S)e^{-\Delta S}$.

</div>

### 7.2 Epilogue

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Central Ideas)</span></p>

The central idea of this course is that learning about the world means building a model — finding an efficient representation of the data. Entropy plays two roles: it defines maximum transmission and minimum compression.

Entropy is not a property of the physical world, but is information we lack about it. Yet information is physical — it has an energetic value and a monetary price. The difference between work and heat is that we have information about the former but not the latter: one can turn information into work and must release heat to erase information. The physical nature of information is manifested in the universal limit on how much of it we can squeeze into a given area of space.

The unifying mathematical notions across all these analogies — measurements, predictions, communication, optimal strategy in economics and biology, data processing, perceptual inference — are the **relative entropy** and **free energy**. Convexity is another recurring theme unifying different approaches to the classes of phenomena.

</div>

## 8. Appendix

### 8.1 Central Limit Theorem and Large Deviations

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Large Deviation Theorem)</span></p>

Consider $X = \sum_1^N y_i$, a sum of i.i.d. random numbers with mean $\langle X\rangle = N\langle y\rangle$. The PDF of $X$ has asymptotically the form:

$$\mathcal{P}(X) \propto e^{-NH(X/N)},$$

where $H$ is the **Cramér (rate) function**, non-negative and convex. It measures the rate of probability decay with the growth of $N$ for every $X/N$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Derivation via Generating Function)</span></p>

Introducing the generating function $\langle e^{zy}\rangle \equiv e^{G(z)}$, whose derivatives at zero give the moments of $y$ (and derivatives of $\log G(z)$ give cumulants):

$$\mathcal{P}(X) = \int_{-\infty}^{\infty}dp\,e^{-ipX + NG(ip)}.$$

For large $N$, this integral is dominated by the saddle point $z_0$ such that $G'(z_0) = X/N$. Substituting $X = NG'(z_0)$ into $-zX + NG(z)$:

$$H = -G(z_0) + z_0 G'(z_0).$$

Thus $-H$ and $G$ are related by the **Legendre transform** (which always appear in saddle-point approximations of integral Fourier or Laplace transforms).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Properties of the Rate Function)</span></p>

$H$ is convex as long as $G(z)$ is convex (which holds since $G''(z) = \langle y^2\rangle_z - \langle y\rangle_z^2 \ge 0$ by Cauchy-Bunyakovsky-Schwarz inequality). $H$ takes its minimum at $z_0 = 0$, i.e. at $X = N\langle y\rangle = NG'(0)$. Since $G(0) = 0$, the minimum value is zero. The maximum of probability does not necessarily coincide with the mean value, but they approach each other as $N$ grows — the law of large numbers.

Any smooth function is quadratic around its minimum: $H''(0) = \Delta^{-1}$, where $\Delta = G''(0) = \langle y^2\rangle - \langle y\rangle^2$ is the variance. Quadratic entropy means Gaussian probability near the maximum — this is the essence of the **central limit theorem**. Non-Gaussianity appears when deviations of $X/N$ from the mean are large, of order $\Delta/G'''(0)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Generalizations and Invariant Distributions)</span></p>

The central limit theorem and the large-deviation approach generalize in two directions: (i) for non-identical variables $y_i$, as long as all variances are finite and none dominates, it still works with the mean and variance of $X$ given by the averages; (ii) if $y_i$ is correlated with a finite number of neighboring variables, one can group "correlated sums" into new variables that can be considered independent.

Is there a distribution invariant under averaging, i.e. that does not change upon passing from $y_i$ to $\sum y_i/N$? That requires $H \equiv 0$, i.e. $G(z) = kz$, corresponding to the **Cauchy distribution** $\mathcal{P}(y) \propto (y^2 + k^2)^{-1}$. Since averaging decreases the variance, the invariant distribution must have infinite variance.

</div>

### 8.2 Continuous Distributions

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Differential Entropy)</span></p>

For a continuous distribution, we generalize by dividing into cells with $p_i = \rho(x_i)\Delta x_i$. The entropy at the limit consists of two parts:

$$-\sum_i p_i\log p_i \to -\int dx\,\rho(x)\log\rho(x) - \log(\Delta x).$$

The second part is an additive constant depending on resolution. The first term is called **differential entropy** $S(x)$. It can be positive, negative, or $-\infty$. Unlike discrete entropy, it is invariant with respect to shifts but not re-scaling: $S(ax + b) = S(x) + \log a$. Different choices of variables to define equal cells give different answers.

For a Hamiltonian system, physics requires that equal volumes in phase space contain equal numbers of states, so the measure is uniform in canonical coordinates. The entropy in terms of the phase space density $\rho(P,Q,t)$ is:

$$S(P,Q,t) = -\int\rho\log\rho\,dP\,dQ,$$

maximal for the uniform distribution $\rho = 1/\Gamma$, giving $I = \ln\Gamma$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Relative Entropy for Continuous Distributions)</span></p>

If the density of discrete points is inhomogeneous, say $m(\mathbf{x})$, the proper generalization is:

$$S = -\int\rho(\mathbf{x})\ln[\rho(\mathbf{x})/m(\mathbf{x})]\,d\mathbf{x}.$$

This is invariant under arbitrary changes of variables $\mathbf{x} \to \mathbf{y}(\mathbf{x})$ since both $\rho$ and $m$ transform the same way. Introducing the normalized distribution $\rho'(\mathbf{x}) = m(\mathbf{x})/\Gamma$:

$$S = \ln\Gamma - \int\rho(\mathbf{x})\ln[\rho(\mathbf{x})/\rho'(\mathbf{x})]\,d\mathbf{x}.$$

The last term is the relative entropy. Both the relative entropy and the mutual information are invariant with respect to invertible transformations of variables — this is a fundamental advantage over the differential entropy alone.

</div>

### 8.3 On Zipf Law

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Zipf Law)</span></p>

The frequency of the $r$-th most frequent word in English is well approximated by:

$$p_r = \begin{cases}(10r)^{-1} & \text{for } r = 1,\ldots,12367 \\ 0 & \text{for } r > 12367\end{cases}.$$

If words were independent, the entropy per word would be $\approx 9.7$ bits. With average word length 4.7 letters, that gives approximately 2 bits per letter.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Random Typing Model and Variational Derivation)</span></p>

The simplest model producing Zipf's law is **random typing**: all letters plus the space are taken with equal probability (Wentian Li, 1992). A word with length $L$ is flanked by two spaces and has probability $P_i(L) = (M+1)^{-L-2}/Z$, where $M$ is the alphabet size. In the limit of large alphabet $M \gg 1$: $P(r) = (r+1)^{-1}$.

A variational interpretation: maximizing information transfer $S = -\sum_r P(r)\log P(r)$ subject to a constraint on mean effort $W = \sum_r P(r)\log r$ (effort grows with rank, e.g. proportional to word length) gives $P(r) \propto r^{-\lambda}$. Zipf's law corresponds to $\lambda = 1$, when goals and means are balanced.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Words Have Meaning)</span></p>

Does the Zipf law trivially appear because both the number of words (inverse probability) and rank increase exponentially with word size? The answer is negative — the number of distinct words of the same length in real language is not exponential and is not even monotonic. Our texts are statistically distinguishable from those of a random monkey with a typewriter. The number of meanings of a word grows approximately as the square root of its frequency: $m_i \propto \sqrt{P_i}$. Meanings correspond to reference objects with their own probabilities $p_i$, and language groups them so that $P_i = m_i p_i \propto m_i^2$, suggesting a balance between minimizing efforts of writers/speakers and readers/listeners.

</div>

### 8.4 Landauer Bound Experiment

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Experimental Verification of the Landauer Bound)</span></p>

The erasure bounds were verified experimentally only recently (Berut et al., Nature 2012; Jun, Gavrilov, Bechhoefer, PRL 2014). The experiment treats a colloidal particle in a double-well potential as a one-bit memory. The initial entropy of the system is $\ln 2$. The procedure to erase a bit (reset to the right well irrespective of initial position): (a) lower the barrier, (b) apply a tilting force to bring the particle into the right well, (c) raise the barrier back. The heat/work was determined by observing the particle trajectory $x(t)$ and computing:

$$W = Q(\tau) = -\int_0^{\tau}\dot{x}(t)\frac{\partial U(x,t)}{\partial x}\,dt.$$

Averaged over 600 realizations, the second law gives $\langle Q\rangle \ge -T\Delta S = T\ln 2$. The experiment confirms that this limit is approached as the duration of the process increases (becoming more quasi-static). Present-day silicon computers still exceed the Landauer limit by a factor $10^2$–$10^3$, but this factor is decreasing fast.

</div>

### 8.5 Unsupervised Learning and Infomax Principle

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Infomax Principle)</span></p>

The maximal-capacity approach from Section 4.4 is useful for image and speech recognition via unsupervised learning. Given a training set of inputs $x$ with density $\rho(x)$, choose a response function $y = g(x,w)$ and find optimal $w$ using online stochastic gradient ascent:

$$\Delta w \propto \frac{\partial}{\partial w}\ln\!\left(\frac{\partial g(x,w)}{\partial x}\right).$$

For example, the generalized logistic function defined implicitly by $dy/du = y^p(1-y)^r$ provides flexibility for both symmetric and skewed distributions via parameters $p,r$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Multi-Channel Processing and Blind Separation)</span></p>

For $N$ inputs and outputs, the input vector $\mathbf{x} = (x_1,\ldots,x_N)$ is transformed monotonically into output $\mathbf{y}(\mathbf{x})$ with $\det[\partial y_i/\partial x_k] \neq 0$. The multivariate output density is $\rho(\mathbf{y}) = \rho(\mathbf{x})/\det[\partial y_i/\partial x_k]$. To **maximize** the mutual information between input and output, we often (but not always) first **minimize** the mutual information between the output components: $S(y_1,y_2) = S(y_1) + S(y_2) - I(y_1,y_2)$.

This is particularly effective for natural signals where most redundancy comes from strong correlations (e.g. neighboring pixels in images). Finding an encoding in terms of least dependent components facilitates pattern recognition and associative learning. This is the **infomax principle**, and the specific technique is called **independent component analysis (ICA)**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Blind Source Separation and Deconvolution)</span></p>

The "cocktail-party problem": $N$ microphones record $N$ people speaking simultaneously. Uncorrelated sources $s_1,\ldots,s_N$ are mixed linearly by an unknown matrix $\hat{A}$; we receive superpositions $x_1,\ldots,x_N$ and must find a square matrix $\hat{W}$ (the inverse of $\hat{A}$, up to permutations and rescaling) to recover the original sources. Closely related is **blind deconvolution**: a signal $s(t)$ convolved with an unknown filter $a(t)$ gives $x(t) = \int a(t-t')s(t')\,dt'$, and we must find the inverse filter $w(t)$ by learning.

The simplest energy functional for statistical independence is:

$$E = \sum_i S(y_i) - \beta\ln\det[\partial y_i/\partial x_k].$$

The parameter $\beta$ reflects priorities — whether statistical independence or increase in indeterminacy is more important.

</div>
