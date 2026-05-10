---
layout: default
title: "The Variational Principles of Mechanics — Cornelius Lanczos"
date: 2025-01-01
excerpt: Notes on Lanczos's classic treatise covering analytical mechanics, the calculus of variations, Lagrangian and Hamiltonian formulations, and canonical transformations.
tags:
  - mechanics
  - calculus-of-variations
  - physics
  - mathematics
---

# The Variational Principles of Mechanics — Cornelius Lanczos

## Introduction

### 1. The Variational Approach to Mechanics

Ever since Newton laid the solid foundation of dynamics by formulating the laws of motion, the science of mechanics developed along two main lines:

1. **Vectorial mechanics** starts directly from Newton's laws of motion. It aims at recognizing all the forces acting on any given particle, its motion being uniquely determined by the known forces acting on it at every instant. The analysis and synthesis of forces and moments is thus the basic concern of vectorial mechanics.

2. **Analytical mechanics** originates from Leibniz, who advocated the *vis viva* (living force) as the proper gauge for the dynamical action of a force. The *vis viva* of Leibniz coincides --- apart from the unessential factor 2 --- with what we call today "kinetic energy." Leibniz replaced the "momentum" of Newton by the "kinetic energy" and the "force" of Newton by the "work of the force," which was later replaced by a still more basic quantity, the "work function" (frequently replaceable by the "potential energy").

Analytical mechanics bases the entire study of equilibrium and motion on two fundamental scalar quantities: the **kinetic energy** and the **work function** (potential energy).

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Scalars vs. Vectors)</span></p>

Since motion is by its very nature a *directed* phenomenon, it seems puzzling that two scalar quantities should be sufficient to determine it. The energy theorem, which states that the sum of the kinetic and potential energies remains unchanged during the motion, yields only *one* equation, while a single particle in space requires *three* equations. For systems of two or more particles the discrepancy becomes even greater. And yet these two fundamental scalars contain the complete dynamics of even the most complicated material system, provided they are used as the basis of a **principle** rather than of an equation.

</div>

### 2. The Procedure of Euler and Lagrange

Consider a particle at a point $P_1$ at time $t_1$, with known velocity. Assume that the particle will be at a point $P_2$ after a given time has elapsed. Although we do not know the path taken by the particle, it is possible to establish that path completely by mathematical experimentation, provided that the kinetic and potential energies are given for any possible velocity and any possible position.

Euler and Lagrange, the first discoverers of the exact principle of least action, proceed as follows. Connect the two points $P_1$ and $P_2$ by *any* tentative path --- an arbitrary continuous curve that will in all probability *not* coincide with the actual path that nature has chosen for the motion. However, we can gradually *correct* our tentative solution and eventually arrive at a curve which can be designated as the *actual* path of motion.

For this purpose we let the particle move along the tentative path in accordance with the energy principle. The sum of the kinetic and potential energies is kept constant and always equal to that value $E$ which the actual motion has revealed at time $t_1$. This restriction assigns a definite velocity to any point of our path and thus determines the motion. We can choose our path freely, but once this is done the conservation of energy determines the motion uniquely.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Action in the Euler--Lagrange Sense)</span></p>

The **action** is the time-integral of the *vis viva*, i.e., of double the kinetic energy, extended over the entire motion from $P_1$ to $P_2$:

$$A = \int_{P_1}^{P_2} 2T \, dt$$

where all paths share the same end-points $P_1$, $P_2$ and the same given energy constant $E$.

</div>

The value of this "action" will vary from path to path. For some paths it will come out larger, for others smaller. Mathematically we can imagine that *all* possible paths have been tried. There must exist one definite path (at least if $P_1$ and $P_2$ are not too far apart) for which the action assumes a minimum value.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Principle of Least Action, Euler--Lagrange Form)</span></p>

The principle of least action asserts that *this particular path is the one chosen by nature as the actual path of motion*.

That is, among all paths connecting $P_1$ and $P_2$ with the same energy constant $E$, the actual path of motion is the one for which the action $A = \int 2T\,dt$ is minimized.

</div>

This principle, explained here for one single particle, can be generalized to any number of particles and any arbitrarily complicated mechanical system.

### 3. Hamilton's Procedure

We encounter problems of mechanics for which the work function is a function not only of the position of the particle but also of the time. For such systems the law of the conservation of energy does not hold, and the principle of Euler and Lagrange is not applicable, but that of Hamilton is.

In Hamilton's procedure we again start with the given initial point $P_1$ and the given end-point $P_2$. But now we do not restrict the trial motion in any way. Not only can the path be chosen arbitrarily --- save for natural continuity conditions --- but also the motion in time is at our disposal. All that we require now is that our tentative motion shall start at the observed time $t_1$ of the actual motion and end at the observed time $t_2$. (This condition is not satisfied in the procedure of Euler--Lagrange, because there the energy theorem restricts the motion, and the time taken to go from $P_1$ to $P_2$ in the tentative motion will generally differ from the time taken in the actual motion.)

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Action in the Hamiltonian Sense)</span></p>

The characteristic quantity that we now use as the measure of action is the time-integral of the *difference between the kinetic and potential energies*:

$$A = \int_{t_1}^{t_2} (T - V) \, dt$$

where $T$ is the kinetic energy and $V$ is the potential energy.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Hamilton's Principle of Least Action)</span></p>

The Hamiltonian formulation of the principle of least action asserts that *the actual motion realized in nature is that particular motion for which this action assumes its smallest value*.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Euler--Lagrange as a Consequence of Hamilton)</span></p>

One can show that in the case of "conservative" systems, i.e. systems which satisfy the law of the conservation of energy, the principle of Euler--Lagrange is a consequence of Hamilton's principle. But the latter principle remains valid even for non-conservative systems.

</div>

### 4. The Calculus of Variations

The mathematical problem of minimizing an integral is dealt with in a special branch of the calculus, called "calculus of variations." The mathematical theory shows that our final results can be established without taking into account the infinity of tentatively possible paths. We can restrict our mathematical experiment to such paths as are *infinitely near* to the actual path.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Variation)</span></p>

A tentative path which differs from the actual path in an arbitrary but still *infinitesimal* degree is called a **variation** of the actual path. The calculus of variations investigates the changes in the value of an integral caused by such infinitesimal variations of the path.

</div>

### 5. Comparison Between the Vectorial and the Variational Treatments of Mechanics

The vectorial and the variational theories of mechanics are two different mathematical descriptions of the same realm of natural phenomena:

| | Vectorial Mechanics | Analytical (Variational) Mechanics |
| --- | --- | --- |
| **Founded by** | Newton | Euler and Lagrange (inspired by Leibniz) |
| **Fundamental quantities** | Two vectors: *momentum* and *force* | Two scalars: *kinetic energy* and *work function* |
| **Constraints** | Must account for constraint forces explicitly; Newton's third law ("action equals reaction") does not embrace all cases | Constraints are handled naturally by letting the system move along tentative paths in harmony with them |
| **Coordinate freedom** | Well suited to rectangular frames; curvilinear coordinates are cumbersome without tensor calculus | Complete freedom in choosing coordinates; equations remain valid for an arbitrary choice of coordinates |
| **Scope of forces** | Does not restrict the nature of a force; frictional forces are included without difficulty | Assumes that acting forces are derivable from a scalar "work function"; frictional forces (which have no work function) are outside the realm of variational principles |

### 6. Mathematical Evaluation of the Variational Principles

Many elementary problems of physics and engineering are solvable by vectorial mechanics and do not require variational methods. But in all more complicated problems the superiority of the variational treatment becomes conspicuous. This superiority is due to the complete freedom we have in choosing the appropriate coordinates for our problem.

A key concept is that of **cyclic** (or **ignorable**) coordinates. If we hit on a certain type of coordinates called "cyclic" or "ignorable," a partial integration of the basic differential equations is at once accomplished. If *all* our coordinates are ignorable, our problem is completely solved.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Coordinate Transformations and Ignorable Coordinates)</span></p>

In the Euler--Lagrangian form of mechanics it is more or less accidental if we hit on the right coordinates, because we have no systematic way of producing ignorable coordinates. But the later developments of the theory by Hamilton and Jacobi broadened the original procedures immensely by introducing the "canonical equations," with their much wider transformation properties. Here we are able to produce a *complete* set of ignorable coordinates by solving one single partial differential equation.

Although the actual solution of this differential equation is possible only for a restricted class of problems, it so happens that many important problems of theoretical physics belong precisely to this class. And thus the most advanced form of analytical mechanics turns out to be not only esthetically and logically most satisfactory, but at the same time very practical by providing a tool for the solution of many dynamical problems which are not accessible to elementary methods.

</div>

### 7. Philosophical Evaluation of the Variational Approach to Mechanics

The idea of enlarging reality by including "tentative" possibilities and then selecting one of these by the condition that it minimizes a certain quantity, seems to bring a *purpose* to the flow of natural events. This is in contradiction to the usual *causal* description of things.

Yet in the 17th and 18th centuries, the two forms of thinking did not necessarily appear contradictory. The keynote of that entire period was the seemingly pre-established harmony between "reason" and "world." Leibniz's "deus intellectualis" and Fermat's derivation of geometrical optics on the basis of his "principle of quickest arrival" could not fail to impress the philosophically-oriented scientists of those days.

The sober, practical, matter-of-fact nineteenth century suspected all speculative and interpretative tendencies as "metaphysical" and limited its programme to the pure description of natural events. E. Mach, in "The Science of Mechanics" (*Open Court*, 1893, p. 480), wrote:

> "No fundamental light can be expected from this branch of mechanics. On the contrary, the discovery of matters of principle must be substantially completed before we can think of framing analytical mechanics the sole aim of which is a perfect *practical* mastery of problems."

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Relativity and the Variational Method)</span></p>

In the light of the discoveries of relativity, the variational foundation of mechanics deserves more than purely formalistic appraisal. Far from being nothing but an alternative formulation of the Newtonian laws of motion, the following points suggest the supremacy of the variational method:

1. **The Principle of Relativity** requires that the laws of nature shall be formulated in an "invariant" fashion, independently of any special frame of reference. The methods of the calculus of variations automatically satisfy this principle, because the minimum of a scalar quantity does not depend on the coordinates in which it is measured. While the Newtonian equations of motion did not satisfy the principle of relativity, the principle of least action remained valid, with only the modification that the basic action quantity had to be brought into harmony with the requirement of invariance.

2. **The Theory of General Relativity** has shown that matter cannot be separated from field and is in fact an outgrowth of the field. Hence, the basic equations of physics must be formulated as partial rather than ordinary differential equations. While Newton's particle picture can hardly be brought into harmony with the field concept, the variational methods are not restricted to the mechanics of particles but can be extended to the mechanics of continua.

3. **The Principle of General Relativity** is automatically satisfied if the fundamental "action" of the variational principle is chosen as an invariant under any coordinate transformation. Since the differential geometry of Riemann furnishes us such invariants, we have no difficulty in setting up the required field equations.

</div>

## Chapter I: The Basic Concepts of Analytical Mechanics

### 1. The Principal Viewpoints of Analytical Mechanics

The analytical form of mechanics, as introduced by Euler and Lagrange, differs considerably in its method and viewpoint from vectorial mechanics. The fundamental law of mechanics as stated by Newton --- "mass times acceleration equals moving force" --- holds in the first instance for a single particle only. If the particle is not free but associated with other particles (as in a solid body or a fluid), the Newtonian equation is still applicable if the proper precautions are observed: one has to isolate the particle from all other particles and determine the force exerted on it by the surrounding particles. Each particle is an independent unit which follows the law of motion of a free particle.

The analytical approach to the problem of motion is quite different. The particle is no longer an isolated unit but part of a "system." A **mechanical system** signifies an assembly of particles which interact with each other. The single particle has no significance; it is the system as a whole which counts. It is enough to know one single function, depending on the positions of the moving particles; this **work function** contains implicitly all the forces acting on the particles of the system. They can be obtained from that function by mere differentiation.

Another fundamental difference concerns **auxiliary conditions**. It frequently happens that certain kinematical conditions exist between the particles of a moving system which can be stated *a priori*. The analytical treatment does not require the knowledge of the forces which maintain these conditions --- we can take the given kinematical conditions for granted.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Four Principal Differences Between Vectorial and Analytical Mechanics)</span></p>

1. Vectorial mechanics isolates the particle and considers it as an individual; analytical mechanics considers the system as a whole.
2. Vectorial mechanics constructs a separate acting force for each moving particle; analytical mechanics considers one single function: the work function (or potential energy). This one function contains all the necessary information concerning forces.
3. If strong forces maintain a definite relation between the coordinates of a system, and that relation is empirically given, the vectorial treatment has to consider the forces necessary to maintain it. The analytical treatment takes the given relation for granted, without requiring knowledge of the forces which maintain it.
4. In the analytical method, the entire set of equations of motion can be developed from one unified principle which implicitly includes all these equations. This principle takes the form of minimizing a certain quantity, the "action." Since a minimum principle is independent of any special reference system, the equations of analytical mechanics hold for any set of coordinates.

</div>

### 2. Generalized Coordinates

In the elementary vectorial treatment of mechanics the abstract concept of a "coordinate" does not enter the picture. The method is essentially geometrical in character. Vector methods are eminently useful in problems of statics, but for problems of motion the number of such problems which can be solved by pure vector methods is relatively small.

Analytical mechanics is a completely mathematical science. Everything is done by calculations in the abstract realm of quantities. The physical world is translated into mathematical relations with the help of coordinates: they establish a one-to-one correspondence between the points of physical space and numbers.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Generalized Coordinates)</span></p>

Consider a mechanical system of $N$ free particles with rectangular coordinates $x_i, y_i, z_i$ ($i = 1, 2, \ldots, N$). The same problem is likewise solved if the $x_i, y_i, z_i$ are expressed in terms of some other quantities

$$q_1, q_2, \ldots, q_{3N}$$

provided these quantities $q_k$ are determined as functions of time $t$. The general form of such a **coordinate transformation** is:

$$x_1 = f_1(q_1, \ldots, q_{3N}), \quad \ldots, \quad z_N = f_{3N}(q_1, \ldots, q_{3N}).$$

</div>

The flexibility of the reference system makes it possible to choose coordinates which are particularly suitable for the given problem. For example, in the planetary problem (a particle revolving around a fixed attracting centre), polar coordinates are better suited than rectangular ones.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Degrees of Freedom)</span></p>

If a mechanical system consists of $N$ particles and there are $m$ independent kinematical conditions imposed, it will be possible to characterize the configuration of the system uniquely by

$$n = 3N - m$$

independent parameters $q_1, q_2, \ldots, q_n$. The number $n$ is a characteristic constant of the given mechanical system called the number of **degrees of freedom**. The parameters $q_1, \ldots, q_n$ are the **generalized coordinates** of the system.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Degrees of Freedom)</span></p>

* *One degree of freedom:* A piston moving up and down. A rigid body rotating about a fixed axis.
* *Two degrees of freedom:* A particle moving on a given surface.
* *Three degrees of freedom:* A particle moving in space. A rigid body rotating about a fixed point (top).
* *Four degrees of freedom:* Two components of a double star revolving in the same plane.
* *Five degrees of freedom:* Two particles kept at a constant distance from each other.
* *Six degrees of freedom:* Two planets revolving about a fixed sun. A rigid body moving freely in space.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Regularity Conditions on Generalized Coordinates)</span></p>

The generalized coordinates $q_1, q_2, \ldots, q_n$ may or may not have a geometrical significance. It is necessary, however, that the functions (12.8) shall be finite, single valued, continuous and differentiable, and that the Jacobian of at least *one* combination of $n$ functions shall be different from zero. These conditions may be violated at certain singular points, which have to be excluded from consideration.

It is not always advisable to eliminate all the kinematical conditions by introducing suitable generalized coordinates. We sometimes prefer to eliminate only *some* of the kinematical conditions, and to leave the others as additional restricting conditions of the form:

$$\varphi_i(q_1, q_2, \ldots, q_n) = 0, \quad (i = 1, \ldots, m).$$

</div>

### 3. The Configuration Space

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Configuration Space, C-point, C-curve)</span></p>

If we associate with the $n$ numbers $q_1, q_2, \ldots, q_n$ the rectangular coordinates of a "point" $P$ in an $n$-dimensional space, the entire mechanical system is pictured as a single point of a many-dimensional space, called the **configuration space**. The point which symbolizes the position of the mechanical system in the configuration space is called the **C-point**, while the curve traced out by that point during the motion is the **C-curve**.

The solution of a dynamical problem takes the form

$$q_1 = q_1(t), \quad \ldots, \quad q_n = q_n(t),$$

and in the associated geometrical picture we have a point $P$ of an $n$-dimensional space which moves along a given curve of that space.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Topological vs. Metrical Properties)</span></p>

For the purposes of the calculus of variations such "topological" properties of space are the really important things, while the "metrical" properties of space, such as distances, angles, areas, etc., are irrelevant. For this reason even the simplified picture of a configuration space without a corresponding geometry is still an excellent aid in visualizing some abstract analytical processes.

</div>

### 4. Mapping of the Space on Itself

Since the significance of the $n$ generalized coordinates $q_1, \ldots, q_n$ is not specified beyond the requirement that it shall allow a complete characterization of the system, we may choose another set of quantities $\bar{q}_1, \bar{q}_2, \ldots, \bar{q}_n$ as generalized coordinates. There must exist a functional relationship between the two sets expressible in the form:

$$\bar{q}_1 = f_1(q_1, \ldots, q_n), \quad \ldots, \quad \bar{q}_n = f_n(q_1, \ldots, q_n).$$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Point Transformation and the Jacobian)</span></p>

A **point transformation** is a mapping of the $n$-dimensional space on itself: to a definite point $P$ of the $q$-space corresponds a definite point $\bar{P}$ of the $\bar{q}$-space. The functions $f_1, \ldots, f_n$ must satisfy the ordinary regularity conditions: they must be finite, single valued, continuous and differentiable functions of the $q_k$, with a **Jacobian** $\Delta$ which is different from zero.

Differentiation gives

$$d\bar{q}_i = \sum_{k=1}^{n} \frac{\partial f_i}{\partial q_k} \, dq_k.$$

No matter what functional relations exist between the two sets of coordinates, their *differentials* are always *linearly* dependent. The functional determinant $\Delta$ is just the determinant of these linear equations. Geometrically this determinant represents the ratio of the volume $\bar{\tau}$ of the new parallelepiped to the volume $\tau$ of the original one. The non-vanishing of $\Delta$ ensures that the entire neighborhood of $P$ shall be mapped on the entire neighborhood of $\bar{P}$, which is necessary for the one-to-one correspondence.

</div>

### 5. Kinetic Energy and Riemannian Geometry

The use of arbitrary generalized coordinates in describing the motion of a mechanical system is one of the principal features of analytical mechanics. The equations of analytical mechanics have a structure that can be stated in a form which is independent of the coordinates used. This property links analytical mechanics with the theory of invariants and covariants, and ultimately with Riemannian geometry.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Riemannian Line Element)</span></p>

The geometry of an $n$-dimensional space is determined by postulating the **line element** given by

$$d\bar{s}^2 = \sum_{i,k=1}^{n} g_{ik} \, dx_i \, dx_k,$$

with the additional symmetry condition $g_{ik} = g_{ki}$, which makes the **metrical tensor** $g_{ik}$ a "symmetric tensor." The quantities $g_{ik}$ are generally not constants but functions of the variables $x_1, \ldots, x_n$. They are constants only if rectangular (or more generally "rectilinear") coordinates are employed.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Riemannian vs. Euclidean Geometry)</span></p>

Riemann's discovery was that the definition of the line element gives not only a new, but a much more *general*, basis for building geometry than the older basis of Euclidean postulates. The $g_{ik}$ have to belong to a *certain class of functions* in order to yield the Euclidean type of geometry. If the $g_{ik}$ are not thus restricted, a new type of geometry emerges, characterized by two fundamental properties:

1. The properties of space change from point to point, but only in a *continuous* fashion.
2. Although Euclidean geometry does not hold in large regions, it holds in *infinitesimal* regions.

Riemann showed how one may obtain by differentiation a characteristic quantity, the "curvature tensor," which tests the nature of the geometry. If all the components of the curvature tensor vanish, the geometry is Euclidean, otherwise not.

</div>

The kinetic energy is a scalar quantity and defined as $T = \frac{1}{2}\sum_{i=1}^{N} m_i v_i^2$ for a system of particles. Now let us define the line element of a $3N$-dimensional space by

$$d\bar{s}^2 = 2T \, dt^2 = \sum_{i=1}^{N} m_i(dx_i^2 + dy_i^2 + dz_i^2).$$

Then the total kinetic energy of the mechanical system may be written as

$$T = \tfrac{1}{2} m \left(\frac{ds}{dt}\right)^2, \quad \text{with } m = 1.$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Kinetic Energy as Riemannian Line Element)</span></p>

*The kinetic energy of the whole system may be replaced by the kinetic energy of one single particle of mass 1.* This imaginary particle is a point of the $3N$-dimensional configuration space which symbolizes the position of the mechanical system. In this space one point is sufficient to represent the mechanical system, and hence we can carry over the mechanics of a free particle to any mechanical system if we place that particle in a space of the proper number of dimensions and proper geometry.

If kinematical conditions are present, the line element in terms of the $n$ generalized coordinates $q_i$ takes the form

$$d\bar{s}^2 = \sum_{i,k=1}^{n} a_{ik} \, dq_i \, dq_k,$$

which is now truly Riemannian. *The mechanical problem is translated into a problem of differential geometry.*

</div>

### 6. Holonomic and Non-holonomic Mechanical Systems

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Holonomic and Non-holonomic Conditions)</span></p>

A kinematical condition of the form

$$f(q_1, \ldots, q_n) = 0$$

is called a **holonomic** condition. By differentiation it yields

$$\frac{\partial f}{\partial q_1} dq_1 + \cdots + \frac{\partial f}{\partial q_n} dq_n = 0.$$

However, if we start with a differential relation of the form

$$A_1 \, dq_1 + \cdots + A_n \, dq_n = 0,$$

where the coefficients $A_k$ are given functions of the $q_k$, this relation is convertible into a finite (holonomic) form only if certain integrability conditions are satisfied. If the integrability conditions are *not* satisfied, the kinematical condition is called **non-holonomic**.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Ball Rolling on a Table)</span></p>

A characteristic example is the rolling of a ball on a table. The ball, moving freely in space, has six degrees of freedom. Since it rests on the surface, the height of the centre is a given constant, reducing the degrees of freedom to five (two rectangular coordinates $x$ and $y$ of the point of contact, plus three angles $\alpha$, $\beta$, $\gamma$ fixing the orientation of the ball relative to its centre).

If the ball can slide along the surface, it can make use of all five degrees of freedom. However, if it is confined to rolling, the point of contact has to be momentarily at rest, which requires the instantaneous axis of rotation to go through the point of contact. This condition of "pure rolling" cuts down the degrees of freedom to two.

The *differentials* of $\alpha$, $\beta$, $\gamma$ are expressible in terms of the differentials of $x$ and $y$, but these differential relations are *not integrable*. They cannot be changed into finite relations between the coordinates. This is a non-holonomic system.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Testing Integrability)</span></p>

For the simplest case of one condition between three variables $dq_3 = B_1\,dq_1 + B_2\,dq_2$, the integrability condition is:

$$\frac{\partial B_1}{\partial q_2} + \frac{\partial B_1}{\partial q_3} B_2 = \frac{\partial B_2}{\partial q_1} + \frac{\partial B_2}{\partial q_3} B_1.$$

If this relation is satisfied identically for all values of $q_1$ and $q_2$, the condition is holonomic; otherwise it is non-holonomic. In the case of more than two independent variables, all the integrability conditions $\frac{\partial B_i}{\partial q_k} = \frac{\partial B_k}{\partial q_i}$ have to be tested in a similar manner.

Holonomic conditions can be attacked in *two* ways: we can eliminate $m$ of the variables and reduce the problem to $n - m$ independent variables, or we can operate with a surplus number of variables and retain the given relations as auxiliary conditions. Non-holonomic conditions necessitate the *second* form of treatment --- a reduction in variables is not possible.

</div>

### 7. Work Function and Generalized Force

The two sides of the Newtonian equation of motion correspond to two fundamentally different aspects of mechanical problems. On the left side we have the inertial quality of mass, absorbed by the kinetic energy in the analytical treatment. The right side, the "moving force," describes the dynamical behavior of an external field in its action on the particle. The analytical treatment shows that it is not the force but the *work* done by the force which is of primary importance, while the force is a secondary quantity derived from the work.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Generalized Force)</span></p>

Let each particle of mass $m_i$ with rectangular coordinates $x_i, y_i, z_i$ be acted on by a force $F_i$ with components $X_i, Y_i, Z_i$. The total work of all the impressed forces for arbitrary infinitesimal displacements is

$$d\bar{w} = \sum_{i=1}^{N} (X_i \, dx_i + Y_i \, dy_i + Z_i \, dz_i).$$

When the rectangular coordinates are expressed in terms of the generalized coordinates $q_1, \ldots, q_n$, the infinitesimal work comes out as a linear differential form:

$$d\bar{w} = F_1 \, dq_1 + F_2 \, dq_2 + \cdots + F_n \, dq_n.$$

The coefficients $F_1, F_2, \ldots, F_n$ are called the components of the **generalized force**. These quantities form the components of a vector of the $n$-dimensional configuration space. They are analytically defined as the coefficients of an invariant differential form of the first order which gives the total work of all the impressed forces for an arbitrary infinitesimal change of the position of the system.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Work Function and Potential Energy)</span></p>

If the infinitesimal work $d\bar{w}$ is the true differential of a certain function $U$, then we put

$$d\bar{w} = dU, \quad \text{where } U = U(q_1, q_2, \ldots, q_n),$$

and the function $U$ is called the **work function**. It follows that

$$F_i = \frac{\partial U}{\partial q_i}.$$

The present practice is to use the *negative* work function, called the **potential energy** $V$:

$$V = -U, \quad \text{so that} \quad F_i = -\frac{\partial V}{\partial q_i}.$$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Monogenic and Polygenic Forces)</span></p>

Forces which are derivable from a scalar quantity (a work function), irrespective of whether they are conservative or not, are called **monogenic** (single-generated). Forces which are not derivable from a scalar function --- such as friction --- are called **polygenic**.

If the work function depends only on the position coordinates, $U = U(q_1, \ldots, q_n)$, then the forces are called **conservative forces**, because they satisfy the law of the conservation of energy.

The work function in the most general case may also be a function of the coordinates *and the velocities*:

$$U = U(q_1, \ldots, q_n; \dot{q}_1, \ldots, \dot{q}_n; t).$$

In this case the connection between force and the work function is given by the more general equation:

$$F_i = \frac{\partial U}{\partial q_i} - \frac{d}{dt}\frac{\partial U}{\partial \dot{q}_i}.$$

</div>

### 8. Scleronomic and Rheonomic Systems. The Law of the Conservation of Energy

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Scleronomic and Rheonomic Systems)</span></p>

Boltzmann used the terms **rheonomic** and **scleronomic** as distinguishing names for kinematical conditions which do or do not involve the time $t$:

* A kinematical condition is **scleronomic** if it does not contain the time $t$ explicitly: $f(q_1, \ldots, q_n) = 0$.
* A kinematical condition is **rheonomic** if it contains the time $t$ explicitly: $f(q_1, \ldots, q_n, t) = 0$.

This is the situation, for example, if a mass point moves on a surface which itself is moving according to a given law, or a pendulum whose length is being constantly changed by pulling the thread.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Consequences of Rheonomic Conditions)</span></p>

If rheonomic conditions are present, the elimination of these conditions by a proper choice of curvilinear coordinates will have the consequence that the transformation equations $x_i = f_i(q_1, \ldots, q_n, t)$ contain $t$ explicitly. Differentiating with respect to $t$:

$$\dot{x}_1 = \frac{\partial f_1}{\partial q_1}\dot{q}_1 + \cdots + \frac{\partial f_1}{\partial q_n}\dot{q}_n + \frac{\partial f_1}{\partial t}.$$

If these expressions are substituted in the definition of kinetic energy, it will not come out as a purely quadratic form of the generalized velocities $\dot{q}_i$; we obtain additional new terms which are linear in the velocities, and others which are independent of the velocities. The Riemannian geometry of the configuration space ceases to play the role that it has before.

A similar situation arises even without time-dependent kinematical conditions, if the coordinates chosen belong to a reference system *which is in motion*.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Conservation of Energy)</span></p>

If both the kinetic energy and the work function are scleronomic, i.e. time-independent, the equations of motion yield a fundamental theorem called the **law of the conservation of energy**. The total energy, defined as

$$E = T + V,$$

remains constant during the motion. Here the potential energy $V$ of the mechanical system is defined as:

$$V = \sum_i \frac{\partial U}{\partial \dot{q}_i} \dot{q}_i - U.$$

In the frequent case where the work function is independent of the velocities, the potential energy becomes simply $V = -U$, and we can consider the kinetic energy $T$ and the potential energy $V$ as the two basic scalars of mechanics.

If either kinetic energy or work function or both are rheonomic (time-dependent), such a conservation law cannot be found. For this reason scleronomic systems are frequently referred to as **conservative systems**.

</div>

## Chapter II: The Calculus of Variations

### 1. The General Nature of Extremum Problems

Extremum problems have aroused the interest and curiosity of all ages. Walking in a straight line is the instinctive solution of an extremum problem: we want to reach the end point of our destination with as little detour as possible. The proverbial "path of least resistance" is another acknowledgment of our instinctive desire for minimum solutions.

Mathematically we speak of an "extremum problem" whenever the largest or smallest possible value of a quantity is involved. For the solution of such problems a special branch of mathematics, called the "calculus of variations," has been developed.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Historical Origins)</span></p>

From the formal viewpoint the problem of minimizing a definite integral is considered as the proper domain of the calculus of variations, while the problem of minimizing a function belongs to ordinary calculus. Historically, the two problems arose simultaneously and a clear-cut distinction was not made till the time of Lagrange, who developed the technique of the calculus of variations. The famous problem of Dido, well known to the ancient geometers, was a variational problem which involved the minimization of an integral. Hero of Alexandria derived the law of reflection from the principle that a light ray emitted at $A$ and proceeding to $B$ after reflection from a mirror reaches its destination in the shortest possible time. Fermat extended this principle to derive the law of refraction. The problem of the "brachistochrone" --- the curve of quickest descent --- was proposed by John Bernoulli and solved independently by him, Newton and Leibniz. The basic differential equation of variational problems was discovered by Euler and Lagrange. A general method for the solution of variational problems was introduced by Lagrange in his *Mécanique Analytique* (1788).

</div>

### 2. The Stationary Value of a Function

We consider a function of an arbitrary number of variables:

$$F = F(u_1, u_2, \ldots, u_n).$$

These variables can be pictured as the rectangular coordinates of a point $P$ in a space of $n$ dimensions. The question of a maximum or minimum involves by its very nature a *comparison*. We distinguish between a **local** maximum (or minimum) and an **absolute** maximum (or minimum).

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Stationary Value)</span></p>

We say that *a function has a "stationary value" at a certain point if the rate of change of the function in every possible direction from that point vanishes*.

A **variation** is a virtual and infinitesimal change of the position. Lagrange introduced a special symbol $\delta$ for the process of variation, in order to emphasize its virtual character. The analogy to $d$ brings to mind that both symbols refer to *infinitesimal changes*. However, $d$ refers to an *actual* change, $\delta$ to a *virtual* change.

</div>

The infinitesimal virtual changes of coordinates are written as $\delta u_1, \delta u_2, \ldots, \delta u_n$. The corresponding change of the function $F$ becomes by the rules of elementary calculus:

$$\delta F = \frac{\partial F}{\partial u_1} \delta u_1 + \frac{\partial F}{\partial u_2} \delta u_2 + \cdots + \frac{\partial F}{\partial u_n} \delta u_n.$$

This expression is called the **first variation** of the function $F$. To operate with finite rather than infinitesimal quantities, we put $\delta u_i = \epsilon a_i$, where $a_1, \ldots, a_n$ are the direction cosines of the virtual direction and $\epsilon$ tends toward zero. The rate of change becomes:

$$\frac{\delta F}{\epsilon} = \frac{\partial F}{\partial u_1} a_1 + \frac{\partial F}{\partial u_2} a_2 + \cdots + \frac{\partial F}{\partial u_n} a_n.$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Necessary and Sufficient Condition for a Stationary Value)</span></p>

The necessary and sufficient condition that a function $F$ of $n$ variables shall have a stationary value at a certain point $P$ is that the $n$ partial derivatives of $F$ with respect to all the $n$ variables shall vanish at that point $P$:

$$\frac{\partial F}{\partial u_k} = 0, \quad (k = 1, 2, \ldots, n).$$

These equations determine the *position* of a stationary value rather than the stationary value itself. If we have found the $n$ values $u_1, \ldots, u_n$ which satisfy these $n$ equations, we can substitute them in $F$ to obtain the stationary value of the function.

</div>

### 3. The Second Variation

Let us once more determine the infinitesimal change of a function due to a virtual variation of the coordinates. Writing the variation in the form $\delta u_i = \epsilon a_i$, we have to evaluate:

$$\Delta F = F(u_1 + \epsilon a_1, u_2 + \epsilon a_2, \ldots, u_n + \epsilon a_n) - F(u_1, u_2, \ldots, u_n).$$

Expanding in powers of $\epsilon$:

$$\Delta F = \epsilon \sum_{k=1}^{n} \frac{\partial F}{\partial u_k} a_k + \frac{1}{2} \epsilon^2 \sum_{i,k=1}^{n} \frac{\partial^2 F}{\partial u_i \partial u_k} a_i a_k + \cdots$$

At a point where $F$ has a stationary value, the first sum vanishes and $\Delta F$ starts with the second order terms.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Second Variation)</span></p>

If $\epsilon$ is chosen sufficiently small, we can neglect the terms of order higher than the second and write:

$$\Delta F = \frac{1}{2} \delta^2 F,$$

where

$$\delta^2 F = \epsilon^2 \sum_{i,k=1}^{n} \frac{\partial^2 F}{\partial u_i \partial u_k} a_i a_k.$$

This expression is called the **second variation** of $F$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Sign of the Second Variation and Extrema)</span></p>

The sign of the second variation determines the existence of an extremum:

- If $\delta^2 F$ is always **positive** (no matter what values the direction cosines $a_i$ take, subject to the constraint that their squares sum to 1), then $F$ is *increasing* in every possible direction from $P$ and we have a real **minimum**.
- If $\delta^2 F$ is always **negative**, then $F$ is *decreasing* in every direction and we have a real **maximum**.
- If $\delta^2 F$ is positive in some directions and negative in others, then we have **neither a maximum nor a minimum** (a saddle point).

To test whether $\delta^2 F$ can change sign, one checks whether the equation $\delta^2 F = 0$ has real solutions for the $a_i$. If it does, no extremum exists. If no real solution exists, then $\delta^2 F$ cannot change sign and an extremum is present.

</div>

### 4. Stationary Value Versus Extremum Value

We should keep well in mind the difference between stationary value and extremum and the mutual relation of these two problems.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Stationary Value vs. Extremum)</span></p>

A **stationary value** requires solely the vanishing of the first variation, without any restriction on the second variation. An **extremum** requires the vanishing of the first variation *plus* further conditions on the second variation.

We have investigated the question of an extremum under the condition that we are *inside* the boundaries of the configuration space. A function which does not assume any extremum inside a certain region may well assume it on the **boundary** of that region. On the boundary the displacements are no longer reversible, and hence the argument that the first variation must vanish --- because otherwise it can be made positive as well as negative --- no longer holds. For non-reversible displacements a function may well assume an extremum *without* having a stationary value at that point.

For the general aims of dynamics, problems of motion require merely the stationary value, and not necessarily the minimum, of a certain definite integral.

</div>

### 5. Auxiliary Conditions. The Lagrangian $\lambda$-method

The problem of minimizing a function does not always present itself as a free variation problem. The configuration space in which the point $P$ can move may be restricted to less than $n$ dimensions by certain kinematical relations called "auxiliary conditions."

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Lagrangian Multiplier Method --- Single Constraint)</span></p>

Consider the variation of the function $F = F(u_1, u_2, \ldots, u_n)$ with a single auxiliary condition $f(u_1, \ldots, u_n) = 0$.

Taking the variation of the auxiliary condition gives

$$\delta f = \frac{\partial f}{\partial u_1} \delta u_1 + \cdots + \frac{\partial f}{\partial u_n} \delta u_n = 0,$$

while the stationarity of $F$ requires

$$\delta F = \frac{\partial F}{\partial u_1} \delta u_1 + \cdots + \frac{\partial F}{\partial u_n} \delta u_n = 0.$$

The Lagrangian method multiplies $\delta f$ by an undetermined factor $\lambda$ and adds it to $\delta F$:

$$\sum_{k=1}^{n} \left( \frac{\partial F}{\partial u_k} + \lambda \frac{\partial f}{\partial u_k} \right) \delta u_k = 0.$$

We choose $\lambda$ so that the factor multiplying $\delta u_n$ vanishes: $\frac{\partial F}{\partial u_n} + \lambda \frac{\partial f}{\partial u_n} = 0$. After this elimination, all remaining $\delta u_k$ are free variations, leading to:

$$\frac{\partial F}{\partial u_k} + \lambda \frac{\partial f}{\partial u_k} = 0, \quad (k = 1, 2, \ldots, n-1).$$

Combined with the condition on $\lambda$, we obtain $n$ equations that hold for *all* $k = 1, \ldots, n$, *just as if all the variations $\delta u_k$ were free*.

Equivalently, instead of putting the first variation of $F$ equal to zero, we modify the function $F$ to

$$\bar{F} = F + \lambda f,$$

and put its first variation equal to zero for *arbitrary* variations of the $u_k$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Lagrangian Multiplier Method --- Multiple Constraints)</span></p>

If the stationary value of $F$ is sought under $m$ independent restricting conditions

$$f_1(u_1, \ldots, u_n) = 0, \quad \ldots, \quad f_m(u_1, \ldots, u_n) = 0, \quad (m < n),$$

then the result generalizes: instead of asking for the stationary value of $F$, we ask for the stationary value of the modified function

$$\bar{F} = F + \lambda_1 f_1 + \cdots + \lambda_m f_m,$$

dropping the auxiliary conditions and handling this as a free variation problem. This yields $n$ equations

$$\frac{\partial F}{\partial u_k} + \lambda_1 \frac{\partial f_1}{\partial u_k} + \cdots + \lambda_m \frac{\partial f_m}{\partial u_k} = 0, \quad (k = 1, \ldots, n),$$

together with the $m$ auxiliary conditions, giving $n + m$ equations for the $n + m$ unknowns $u_1, \ldots, u_n; \lambda_1, \ldots, \lambda_m$.

The method of Lagrange permits the use of surplus coordinates --- a great convenience in many considerations of mechanics. It preserves the full symmetry of all coordinates by making it unnecessary to distinguish between dependent and independent variables.

</div>

### 6. Non-holonomic Auxiliary Conditions

As was pointed out in Chapter I, Section 6, the restrictions on the mechanical variables of a problem may be given in a differential instead of a finite form. We then have a variation problem with non-holonomic auxiliary conditions.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Lagrangian $\lambda$-method for Non-holonomic Conditions)</span></p>

Non-holonomic conditions cannot be handled by the elimination method, because the equations for eliminating some variables as dependent variables do not exist. The Lagrangian $\lambda$-method, however, is again available. The non-holonomic conditions can be written in the form:

$$\bar{\delta} f_i = A_{i1} \delta u_1 + A_{i2} \delta u_2 + \cdots + A_{in} \delta u_n = 0, \quad (i = 1, \ldots, m).$$

Here the $A_{ik}$ are given functions of the $u_i$ which cannot be considered as the partial derivatives of a function $f_i$. By exactly the same procedure as before, we obtain an equation analogous to the holonomic case:

$$\delta F + \lambda_1 \bar{\delta} f_1 + \cdots + \lambda_m \bar{\delta} f_m = 0,$$

and again all the $\delta u_k$ are handled as free variations. The only difference lies in the fact that we cannot proceed to the integrated form $\delta(F + \lambda_1 f_1 + \cdots + \lambda_m f_m) = 0$ and have to be content with the differential formulation. The reduction of a conditioned variation problem to a free variation problem is once more accomplished.

</div>

### 7. The Stationary Value of a Definite Integral

The analytical problems of motion involve a special type of extremum problem: the stationary value of a *definite integral*. The branch of mathematics dealing with problems of this nature is called the Calculus of Variations.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(The Brachistochrone)</span></p>

A typical problem is that of the brachistochrone (the curve of quickest descent), first formulated and solved by John Bernoulli (1696). We wish to find a suitable plane curve along which a particle descends in the shortest possible time, starting from $A$ and arriving at $B$. If the unknown curve is $y = f(x)$, the time to be minimized is given by:

$$t = \frac{1}{\sqrt{2g}} \int_a^b \frac{\sqrt{1 + y'^2}}{\sqrt{\alpha - y}} \, dx.$$

Here $y$ is an unknown function of $x$ which is to be determined, subject to the boundary conditions $f(a) = \alpha$, $f(b) = \beta$.

</div>

More generally, we are given a function $F$ of three variables $F = F(y, y', x)$ and the definite integral

$$I = \int_a^b F(y, y', x) \, dx,$$

with boundary conditions $f(a) = \alpha$, $f(b) = \beta$. The problem is to find a function $y = f(x)$ --- restricted by the customary regularity conditions --- which will make the integral $I$ an extremum, or at least give it a stationary value.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Function Space and Hilbert's Approach)</span></p>

At first sight this problem appears utterly different from the previous problem where we dealt with the extremum of a function $F(u_1, \ldots, u_n)$ of a set of variables. Instead of a *function*, a *definite integral* must be minimized. Moreover, instead of a *set of variables* $u_1, \ldots, u_n$ we have a certain unknown *function* $y = f(x)$ at our disposal. Yet closer inspection reveals that the mathematical nature of this new problem is not substantially different from the previous one.

Using the concept of the "function space" originated by Hilbert, the arbitrary function $y = f(x)$ can be expanded in an infinite Fourier series throughout the given range between $a$ and $b$. The coefficients of this expansion are uniquely determined and can be plotted as rectangular coordinates of a point $P$ in a $(2n+1)$-dimensional space. The problem of finding the function $f(x)$ which minimizes the definite integral $I$ is translated into the problem of finding the deepest point of a surface in a space of $2n + 2$ dimensions.

</div>

Euler has shown how the problem can be solved by elementary means, without resorting to the tools of a specific calculus. A definite integral can be replaced by a sum of an increasing number of terms, and the derivative can be replaced by a difference coefficient. We divide the interval between $x = a$ and $x = b$ into many equal small intervals, obtaining abscissa values $x_0 = a, x_1, x_2, \ldots, x_n, x_{n+1} = b$ and corresponding ordinates $y_0 = \alpha, y_1, \ldots, y_n, y_{n+1} = \beta$. The derivative $f'(x_k)$ is replaced by the difference coefficient

$$z_k = \frac{y_{k+1} - y_k}{x_{k+1} - x_k},$$

and the integral by the sum

$$S' = \sum_{j=0}^{n} F(y_{j+1}, z_j, x_j)(x_{j+1} - x_j).$$

Setting the partial derivatives of $S'$ with respect to $y_{k+1}$ equal to zero and passing to the limit as $\Delta x \to 0$, we arrive at a differential equation.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Euler--Lagrange Differential Equation)</span></p>

The fundamental equation discovered independently by Euler and Lagrange, called the **Euler--Lagrange differential equation**, is:

$$\frac{\partial F}{\partial y} - \frac{d}{dx}\left(\frac{\partial F}{\partial y'}\right) = 0, \quad (a \leq x \leq b).$$

This equation is the necessary and sufficient condition for the integral $I = \int_a^b F(y, y', x)\,dx$ to be stationary, given boundary conditions $y(a) = \alpha$, $y(b) = \beta$. The two limiting ordinates $y_0$ and $y_{n+1}$ are *given quantities* and thus remain unvaried.

</div>

### 8. The Fundamental Processes of the Calculus of Variations

Lagrange realized that the problem of minimizing a definite integral requires specific tools, different from those of the ordinary calculus. With the help of these tools we can attack the problem directly, instead of reverting to the limiting process by which Euler obtained the solution.

Consider the function $y = f(x)$ which by hypothesis gives a stationary value to the definite integral $I$. In order to prove that we *do* have a stationary value, we have to evaluate the same integral for a slightly *modified* function $\bar{f}(x)$ and show that the rate of change of the integral due to the change in the function is zero.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Variation of a Function)</span></p>

The modified function $\bar{f}(x)$ can be written in the form

$$\bar{f}(x) = f(x) + \epsilon \phi(x),$$

where $\phi(x)$ is some arbitrary new function which satisfies the same general continuity conditions as $\bar{f}(x)$; hence $\phi(x)$ has to be continuous and differentiable. The selection of the hypothetical function $f(x)$ which solves our variation problem has to be made from the class of continuous and differentiable functions.

</div>

We now compare the values of the modified function $\bar{f}(x)$ with the values of the original function $f(x)$ *at a certain definite point* $x$ of the independent variable, by forming the difference between $\bar{f}(x)$ and $f(x)$. This difference is called the "variation" of the function $f(x)$ and is denoted by $\delta y$:

$$\delta y = \bar{f}(x) - f(x) = \epsilon \phi(x).$$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Difference Between $\delta y$ and $dy$)</span></p>

Both $\delta y$ and $dy$ are infinitesimal changes of the function $y$. However, $dy$ refers to the infinitesimal change of the given function $f(x)$ caused by the infinitesimal change $dx$ of the independent variable, while $\delta y$ is an infinitesimal change of $y$ which produces a *new function* $y + \delta y$.

It is in the nature of the process of variation that only the dependent function $y$ should be varied while the variation of $x$ serves no useful purposes. Hence we always put $\delta x = 0$. Moreover, if the two limiting ordinates $f(a)$ and $f(b)$ of the function $f(x)$ are prescribed, these two ordinates cannot be varied, which means

$$[\delta f(x)]_{x=a} = 0, \quad [\delta f(x)]_{x=b} = 0.$$

We then speak of a "variation between definite limits."

</div>

### 9. The Commutative Properties of the $\delta$-process

The variation of the function $f(x)$ defines an entirely new function $\epsilon\phi(x)$. We can take the derivative of this function. On the other hand, we can take the derivative of the new function $\bar{f}(x)$ and the derivative of the original function $f(x)$. The difference of these two derivatives can naturally be called the variation of the derivative $f'(x)$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Commutativity of Variation and Differentiation)</span></p>

The "derivative of the variation" is:

$$\frac{d}{dx} \delta y = \frac{d}{dx}[\bar{f}(x) - f(x)] = \frac{d}{dx} \epsilon\phi(x) = \epsilon\phi'(x).$$

The "variation of the derivative" is:

$$\delta \frac{d}{dx} f(x) = \bar{f}'(x) - f'(x) = (y' + \epsilon\phi') - y' = \epsilon\phi'(x).$$

This gives

$$\frac{d}{dx} \delta y = \delta \frac{d}{dx} y,$$

showing that *the derivative of the variation is equal to the variation of the derivative*. Variation and differentiation are **permutable** (commutative) processes.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Commutativity of Variation and Integration)</span></p>

The variation of a definite integral is equal to the definite integral of the variation:

$$\delta \int_a^b F(x)\,dx = \int_a^b \bar{F}(x)\,dx - \int_a^b F(x)\,dx = \int_a^b [\bar{F}(x) - F(x)]\,dx = \int_a^b \delta F(x)\,dx.$$

Hence

$$\delta \int_a^b F(x)\,dx = \int_a^b \delta F(x)\,dx.$$

Variation and integration are **permutable** processes.

</div>

### 10. The Stationary Value of a Definite Integral Treated by the Calculus of Variations

We consider once more the problem of Section 7, but this time treated by the direct methods of the calculus of variations. Given the definite integral $I = \int_a^b F(y, y', x)\,dx$ with boundary conditions $y(a) = \alpha$, $y(b) = \beta$, the stationary value of this integral is to be found.

We start out with the variation of the integrand $F(y, y', x)$ itself, caused by the variation of $y$ (remembering that $F$ is a *given function* of the three variables $y$, $y'$, $x$ and this functional dependence is *not altered* by the process of variation):

$$\delta F(y, y', x) = F(y + \epsilon\phi, y' + \epsilon\phi', x) - F(y, y', x) = \epsilon\left(\frac{\partial F}{\partial y}\phi + \frac{\partial F}{\partial y'}\phi'\right).$$

Using the commutativity of variation and integration:

$$\delta \int_a^b F\,dx = \int_a^b \delta F\,dx = \epsilon \int_a^b \left(\frac{\partial F}{\partial y}\phi + \frac{\partial F}{\partial y'}\phi'\right)dx.$$

The rate of change of the integral (dividing by $\epsilon$) which has to vanish for a stationary value is:

$$\frac{\delta I}{\epsilon} = \int_a^b \left(\frac{\partial F}{\partial y}\phi + \frac{\partial F}{\partial y'}\phi'\right)dx.$$

This expression is not accessible to further analysis in its present form, because $\phi(x)$ and $\phi'(x)$ are not independent of each other. The difficulty can be removed by an ingenious application of integration by parts:

$$\int_a^b \frac{\partial F}{\partial y'}\phi'\,dx = \left[\frac{\partial F}{\partial y'}\phi\right]_a^b - \int_a^b \frac{d}{dx}\left(\frac{\partial F}{\partial y'}\right)\phi\,dx.$$

The first term drops out since we vary between definite limits so that $\phi(x)$ vanishes at the two end-points $a$ and $b$. Making use of this transformation:

$$\frac{\delta I}{\epsilon} = \int_a^b \left(\frac{\partial F}{\partial y} - \frac{d}{dx}\frac{\partial F}{\partial y'}\right)\phi\,dx.$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Euler--Lagrange Equation via the Calculus of Variations)</span></p>

Introducing the notation

$$E(x) = \frac{\partial F}{\partial y} - \frac{d}{dx}\frac{\partial F}{\partial y'},$$

the condition for a stationary value of $I$ takes the form

$$\int_a^b E(x)\phi(x)\,dx = 0.$$

Now it is not difficult to see that this integral can vanish for *arbitrary* functions $\phi(x)$ only if $E(x)$ vanishes everywhere between $a$ and $b$. Indeed, we may arrange that $\phi(x)$ shall vanish everywhere with the exception of an arbitrarily small interval around a point $x = \xi$. Within this interval $E(x)$ is practically constant and can be put in front of the integral sign. The vanishing of $\delta I / \epsilon$ requires the vanishing of the first factor, and since $\xi$ may be *any* point of the interval between $a$ and $b$, we obtain the differential equation

$$\frac{\partial F}{\partial y} - \frac{d}{dx}\frac{\partial F}{\partial y'} = 0, \quad (a \leq x \leq b).$$

This condition is *necessary* for the vanishing of $\delta I$. On the other hand, it is also *sufficient*, because if the integrand vanishes, the integral also vanishes. *The differential equation is thus the necessary and sufficient condition that the definite integral $I$ shall be stationary under the given boundary conditions.*

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Higher-order Derivatives in the Integrand)</span></p>

If the definite integral contains the first *and second* derivatives of $y$,

$$I = \int_a^b F(y, y', y'', x)\,dx,$$

the condition for a stationary value is found by forming the variation of the integral and applying integration by parts *twice*. This leads to the differential equation

$$\frac{\partial F}{\partial y} - \frac{d}{dx}\frac{\partial F}{\partial y'} + \frac{d^2}{dx^2}\frac{\partial F}{\partial y''} = 0,$$

and the boundary term

$$\delta I = \left[\left(\frac{\partial F}{\partial y'} - \frac{d}{dx}\frac{\partial F}{\partial y''}\right)\delta y + \frac{\partial F}{\partial y''}\delta y'\right]_a^b,$$

which vanishes if $y$ and $y'$ are prescribed at the two end-points.

</div>

### 11. The Euler--Lagrange Differential Equations for $n$ Degrees of Freedom

In mechanics the problem of variation presents itself in the following form. Find the stationary value of a definite integral

$$I = \int_{t_1}^{t_2} L(q_1, \ldots, q_n; \dot{q}_1, \ldots, \dot{q}_n; t)\,dt,$$

with the boundary conditions that the $q_k$ are given (and thus their variation is zero) at the two end-points $t_1$ and $t_2$:

$$[\delta q_k(t)]_{t=t_1} = 0, \quad [\delta q_k(t)]_{t=t_2} = 0.$$

The $q_1, \ldots, q_n$ are unknown functions of $t$, to be determined by the condition that the actual motion shall make the integral $I$ stationary: $\delta I = 0$, for arbitrary independent variations of the $q_k$, subject only to the boundary conditions.

We can select one definite $q_k$ and vary it all by itself, leaving the other $q_i$ unchanged. Hence we can apply the Euler--Lagrange equation to our present problem, after adapting the notation: $y$ corresponds to $q_k$, $y'$ to $\dot{q}_k$, the independent variable $x$ is now the time $t$, and the function $F$ is denoted by $L$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Euler--Lagrange Equations for $n$ Degrees of Freedom)</span></p>

The conditions for the stationary value of $I$ come out in the form of the following *system of simultaneous differential equations*:

$$\frac{d}{dt}\frac{\partial L}{\partial \dot{q}_k} - \frac{\partial L}{\partial q_k} = 0, \quad (k = 1, 2, \ldots, n).$$

These are called the **differential equations of Euler and Lagrange**, or, if applied to problems of mechanics, the **Lagrangian equations of motion**.

The variations employed so far are but *special* variations --- we vary each $q_k$ one at a time. But on account of the superposition principle of infinitesimal processes, a simultaneous variation of all the $q_k$ would not bring in additional conditions. Denoting by $\delta_k I$ the variation of $I$ produced by varying $q_k$ alone, the simultaneous variation of all the $q_k$ produces:

$$\delta I = \delta_1 I + \delta_2 I + \cdots + \delta_n I.$$

If each $\delta_k I$ vanishes separately, then $\delta I$ is zero for *arbitrary* variations of the $q_k$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Nature of the Lagrangian Equations)</span></p>

With the exception of the singular case in which the function $L$ depends on some or all the $\dot{q}_k$ in a *linear* way, the partial derivatives $\frac{\partial L}{\partial \dot{q}_k}$ will contain all the $\dot{q}_k$, so that differentiation with respect to $t$ brings in all the second derivatives $\ddot{q}_k$. We can solve the equations for the $\ddot{q}_k$ algebraically and thus rewrite the differential equations in the following explicit form:

$$\ddot{q}_k = \phi_k(q_1, \ldots, q_n, \dot{q}_1, \ldots, \dot{q}_n, t).$$

The integration of such a system of differential equations of the second order involves $2n$ constants of integration, so that the complete solution may be written as:

$$q_k = q_k(A_1, \ldots, A_n; B_1, \ldots, B_n; t).$$

The constants $A_k$ and $B_k$ can be adjusted to the given boundary conditions. In mechanical problems more frequently *initial conditions* take the place of boundary conditions. The freedom of $2n$ constants of integration allows all the initial position coordinates and velocities to be prescribed arbitrarily.

</div>

### 12. Variation with Auxiliary Conditions

We consider once more the problem of the previous paragraph, with the modification that the variables $q_1, \ldots, q_n$ shall not be independent, but restricted by given auxiliary conditions of the form:

$$f_1(q_1, \ldots, q_n, t) = 0, \quad \ldots, \quad f_m(q_1, \ldots, q_n, t) = 0.$$

It is possible to eliminate $m$ of the $q_k$ in terms of the other variables and thus reduce the problem to $n - m$ degrees of freedom. However, this elimination may be rather cumbersome, and the conditions between the variables may make the distinction between dependent and independent variables artificial. Here again the method of the Lagrangian multiplier, studied before in Section 5, gives an adequate solution.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Lagrangian $\lambda$-method for Variational Problems with Auxiliary Conditions)</span></p>

Given the variational problem of making the integral $I = \int_{t_1}^{t_2} L\,dt$ stationary, with $m$ auxiliary conditions $f_i(q_1, \ldots, q_n, t) = 0$ $(i = 1, \ldots, m)$, we multiply the variation of each auxiliary condition by an undetermined $\lambda$-factor and add it to $\delta I$:

$$\delta I' = \delta \int_{t_1}^{t_2} L\,dt + \int_{t_1}^{t_2} (\lambda_1 \delta f_1 + \cdots + \lambda_m \delta f_m)\,dt = 0.$$

Since the auxiliary conditions are prescribed for every value of $t$, the $\lambda$-factors have to be applied for every value of $t$, making them *functions of $t$*. Performing the standard integration by parts and collecting terms, the resulting equations are:

$$\frac{\partial L}{\partial q_k} - \frac{d}{dt}\frac{\partial L}{\partial \dot{q}_k} + \lambda_1 \frac{\partial f_1}{\partial q_k} + \cdots + \lambda_m \frac{\partial f_m}{\partial q_k} = 0, \quad (k = 1, \ldots, n).$$

This is equivalent to replacing the original integrand $L$ by the modified integrand

$$L' = L + \lambda_1 f_1 + \cdots + \lambda_m f_m,$$

and considering the variation of $I' = \int_{t_1}^{t_2} L'\,dt$, dropping the auxiliary conditions and handling all the $q_k$ as independent variables.

The solution of the $n$ differential equations together with the $m$ auxiliary conditions determines both the $q_k$ and the $\lambda_i$ as functions of $t$. In the matter of initial conditions we can choose arbitrarily only $n - m$ position coordinates and $n - m$ velocities, because the remaining $q_i$ and $\dot{q}_i$ are determined by the auxiliary conditions.

</div>

### 13. Non-holonomic Conditions

If the auxiliary conditions of the variational problem are not given as algebraic relations between the variables, but as differential relations --- cf. Chapter I, Section 6 and Chapter II, Section 6 --- the Lagrangian $\lambda$-method is still applicable. The only modification is that the $\frac{\partial f_i}{\partial q_k}$ are replaced by the coefficients $A_{ik}$ of the non-holonomic conditions.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Non-holonomic Conditions in Variational Problems)</span></p>

A difference exists in the matter of initial conditions. The coordinates $q_i$ are at present not restricted by any conditions, only their differentials. Hence the initial values of all the $q_i$ can now be prescribed arbitrarily. The velocities, however, are restricted on account of the given conditions

$$A_{i1}\dot{q}_1 + \cdots + A_{in}\dot{q}_n = 0, \quad (i = 1, \ldots, m).$$

We can thus assign arbitrarily $n$ initial position coordinates and $n - m$ initial velocities.

The $m$ equations above serve not only the purpose of eliminating $m$ of the initial velocities. They have the further function of determining the $\lambda$-factors which enter the equations of motion as undetermined multipliers.

Non-holonomic auxiliary conditions which are rheonomic, i.e. time-dependent, require particular care. Here it is necessary to know what conditions exist between the $\delta q_k$ if the variation is not performed instantaneously but during the infinitesimal time $\delta t$. The auxiliary conditions now take the form

$$A_{i1}\delta q_1 + \cdots + A_{in}\delta q_n + B_i\delta t = 0,$$

with coefficients $A_{ik}$ and $B_i$ which are in general functions of the $q_i$ and the time $t$. The quantities $B_i$ do not enter into the equations of motion since the virtual displacements $\delta q_k$ are performed *without* varying the time; but they do enter into the relations which exist between the velocities:

$$A_{i1}\dot{q}_1 + \cdots + A_{in}\dot{q}_n + B_i = 0.$$

</div>

### 14. Isoperimetric Conditions

It can happen that an auxiliary condition does not appear as an algebraic relation between the $q_k$, but in the form of a definite integral which must have a prescribed value $C$:

$$\int_{t_1}^{t_2} f(q_1, \ldots, q_n, t)\,dt = C.$$

Auxiliary conditions of this form are called **isoperimetric conditions**, since the first historically recorded extremum problem --- that of finding the maximum area bounded by a perimeter of given length (Dido's problem) --- prescribes a condition of this nature.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Lagrangian $\lambda$-method for Isoperimetric Conditions)</span></p>

Consider the variational problem of making $I = \int_{t_1}^{t_2} L\,dt$ stationary, subject to the isoperimetric condition $\int_{t_1}^{t_2} f(q_1, \ldots, q_n, t)\,dt = C$.

The condition must hold not only for the actual functions $q_1, \ldots, q_n$ which make $I$ stationary, but also for the varied functions $\bar{q}_1, \ldots, \bar{q}_n$. Taking the variation of the isoperimetric condition gives the integral relation between the $\delta q_k$:

$$\int_{t_1}^{t_2} \left(\frac{\partial f}{\partial q_1}\delta q_1 + \cdots + \frac{\partial f}{\partial q_n}\delta q_n\right)dt = 0.$$

It is allowable to multiply the left-hand side by some undetermined constant $\lambda$ and add it to $\delta I$. By similar reasoning as in Section 12, we can always determine $\lambda$ such that the integral shall vanish for *arbitrary* variations of the $\delta q_k$. This gives the principle that with a proper choice of $\lambda$ we have

$$\delta \int_{t_1}^{t_2} (L + \lambda f)\,dt = 0,$$

for arbitrary variations. We have once more transformed our conditioned variation problem into a free variation problem by changing the original $L$ into

$$L' = L + \lambda f,$$

where $\lambda$ is an undetermined *constant* (not a function of $t$, in contrast to the case of algebraic auxiliary conditions). The constant can be determined by satisfying the given isoperimetric condition.

The same $\lambda$-method applies if the isoperimetric condition depends not only on the $q_k$ but also on their derivatives with respect to $t$:

$$\int_{t_1}^{t_2} f(q_1, \ldots, q_n; \dot{q}_1, \ldots, \dot{q}_n; t)\,dt = C.$$

</div>

### 15. The Calculus of Variations and Boundary Conditions. The Problem of the Elastic Bar

In all our previous considerations we were essentially interested in the *differential equations* which can be deduced as the solution of the problem that a given definite integral shall assume a stationary value. The derivation of these equations by the method of integration by parts revealed that the variation of a definite integral is composed of two parts, namely, an integral extended over the given range, *plus a boundary term*. We did not pay much attention to this boundary term, since we assumed boundary conditions which made it vanish. But actually there are situations where this boundary term plays a more active role.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Imposed and Natural Boundary Conditions)</span></p>

Any problem which involves the solution of differential equations requires a proper number of **boundary conditions** to make the solution unique. These conditions might be determined by the given physical circumstances. But it is also possible that the physical conditions under consideration give no boundary conditions or not enough boundary conditions.

It is a particularly beautiful feature of variational problems that *they always furnish automatically the right number of boundary conditions*. Those boundary conditions which are not imposed on the system by the given external circumstances follow from the nature of the variational problem itself. It is the boundary term of $\delta I$ which is responsible for these surplus boundary conditions. **Imposed** and **natural** boundary conditions, *taken together, provide a unique solution*.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(The Loaded Elastic Bar)</span></p>

Consider a horizontal elastic bar of length $l$, loaded by weights and supported at the two ends. Let $y(x)$ denote the small vertical displacement and $\rho(x)$ the load per unit length, assuming the bar has everywhere the same cross-section. The potential energy due to elastic forces is

$$V_1 = \frac{k}{2}\int_0^l (y'')^2\,dx,$$

while the potential energy due to gravity is

$$V_2 = -\int_0^l \rho y\,dx.$$

The integral to be minimized is thus

$$I = \int_0^l \left(\frac{k}{2}y''^2 - \rho y\right)dx.$$

The Euler--Lagrange differential equation (using the higher-order formula) is:

$$ky''''(x) = \rho(x).$$

If this equation is satisfied, the variation $\delta I$ reduces to a boundary term:

$$\delta I = k\left[-y'''(l)\,\delta y(l) + y'''(0)\,\delta y(0) + y''(l)\,\delta y'(l) - y''(0)\,\delta y'(0)\right].$$

The form of the boundary conditions depends on the physical circumstances:

1. **Clamped ends:** All four boundary conditions are imposed:
   $y(0) = 0$, $y'(0) = 0$, $y(l) = 0$, $y'(l) = 0$.
   Since the variation of fixed quantities is zero, all terms of $\delta I$ vanish. No natural boundary conditions are needed.

2. **Supported ends:** The only imposed conditions are $y(0) = 0$, $y(l) = 0$. The variations $\delta y'(0)$ and $\delta y'(l)$ are arbitrary, so the vanishing of $\delta I$ requires two additional (natural) boundary conditions: $y''(0) = 0$, $y''(l) = 0$.

3. **One end clamped, the other free:** The imposed conditions are $y(0) = 0$, $y'(0) = 0$, while $y(l)$ and $y'(l)$ are not prescribed. The variational problem requires: $y''(l) = 0$, $y'''(l) = 0$.

4. **Both ends free:** All four natural boundary conditions come into play: $y''(0) = 0$, $y'''(0) = 0$, $y''(l) = 0$, $y'''(l) = 0$. There are no imposed conditions, but the natural boundary conditions are present in sufficient number.

In this last case, the homogeneous differential equation $y''''(x) = 0$ has *two* independent solutions: $y = 1$ and $y = x$. The boundary value problem is not solvable unless $\rho(x)$ is "orthogonal" to these homogeneous solutions, i.e.

$$\int_0^l \rho(x)\,dx = 0, \quad \int_0^l \rho(x)\,x\,dx = 0.$$

The physical significance of these conditions is that *the sum of the forces and the sum of the moments of the forces acting on the bar are zero*. These are the integrability conditions of the boundary value problem, furnished here by the calculus of variations.

</div>

## Chapter III: The Principle of Virtual Work

### 1. The Principle of Virtual Work for Reversible Displacements

The first variational principle we encounter in the science of mechanics is the **principle of virtual work**. It controls the equilibrium of a mechanical system and is fundamental for the later development of analytical mechanics.

In the Newtonian form of mechanics a particle is in equilibrium if the resulting force acting on that particle is zero. This form of mechanics isolates the particle and replaces all constraints by forces. The analytical treatment can dispense with all these forces and take only the external force --- i.e. in this case the force of gravity --- into account. This is accomplished by performing only such virtual displacements as are in harmony with the given constraints.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Constraints and Inner Forces)</span></p>

The mechanical behavior of a rigid body is certainly very different from that of a sandpile. There are strong inner forces acting between the particles of a rigid body which keep these particles together and which do not act between the particles of a sandpile. But the presence of these forces can only be detected by trying to *break* the rigid body, i.e. by trying to move the particles relative to each other in a manner which is *not* in harmony with the given constraints. If we merely move a rigid body or a sandpile by rotation and translation, the mechanical difference between the two systems disappears, because the strong inner forces which characterize the rigid body do not come into action.

In the variational treatment of mechanics the "forces of constraint" which maintain certain given kinematical conditions are neglected, and only the work of the "impressed forces" needs to be taken into account. The number of equations obtained by this procedure is smaller than the number of particles, but it is exactly equal to the number of degrees of freedom which characterize the system.

</div>

Let external forces $\mathbf{F}_1, \mathbf{F}_2, \ldots, \mathbf{F}_n$ act at points $P_1, P_2, \ldots, P_n$ of the system. The virtual displacements of these points will be denoted by

$$\delta\mathbf{R}_1, \, \delta\mathbf{R}_2, \, \ldots, \, \delta\mathbf{R}_n.$$

These virtual displacements must be in harmony with the given kinematical constraints, and we shall assume that they are *reversible*, i.e. the given constraints do not prevent us from changing an arbitrary $\delta\mathbf{R}_i$ into $-\delta\mathbf{R}_i$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Principle of Virtual Work)</span></p>

*The given mechanical system will be in equilibrium if, and only if, the total virtual work of all the impressed forces vanishes:*

$$\bar{\delta w} = \mathbf{F}_1 \cdot \delta\mathbf{R}_1 + \mathbf{F}_2 \cdot \delta\mathbf{R}_2 + \ldots + \mathbf{F}_n \cdot \delta\mathbf{R}_n = 0.$$

</div>

Translating this equation into analytical language by expressing the rectangular coordinates $x_i, y_i, z_i$ as functions of the generalized coordinates $q_1, q_2, \ldots, q_n$, the virtual work takes the differential form

$$\bar{\delta w} = F_1 \delta q_1 + F_2 \delta q_2 + \ldots + F_n \delta q_n,$$

where $F_1, F_2, \ldots, F_n$ are the components of the **generalized force**. They form a vector of the $n$-dimensional configuration space. The principle of virtual work requires that

$$F_1 \delta q_1 + F_2 \delta q_2 + \ldots + F_n \delta q_n = 0.$$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Geometric Interpretation)</span></p>

The left-hand side of the equilibrium equation is the "scalar product" of force and virtual displacement. The vanishing of this scalar product means that *the force $F_i$ is perpendicular to any possible virtual displacement*.

If the given mechanical system is free of any constraints, then the $C$-point of the configuration space can be displaced in an arbitrary direction. Then the principle requires that the force $F_i$ shall vanish, because there is no vector which can be perpendicular to all directions in space.

If the $C$-point has to stay within a certain $(n - m)$-dimensional subspace of the configuration space, on account of $m$ given kinematical constraints, then the condition no longer requires the vanishing of the force $F_i$, but only its *perpendicularity* to that subspace. This amounts to $n - m$ equations, in conformity with the $n - m$ degrees of freedom of the mechanical system.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Physical Interpretation — Postulate A)</span></p>

According to Newtonian mechanics, equilibrium requires that the resultant force acting on *any* particle of the system shall vanish. This resultant force is the sum of the impressed force and the forces which maintain the given constraints ("forces of reaction"). Since "impressed force plus resultant force of reaction equals zero," the virtual work of the impressed forces can be replaced by the negative virtual work of the forces of reaction. Hence the principle of virtual work can be formulated as **Postulate $A$**:

> "The virtual work of the forces of reaction is always zero for any virtual displacement which is in harmony with the given kinematic constraints."

This postulate is not restricted to the realm of statics. It applies equally to dynamics, when the principle of virtual work is suitably generalized by means of d'Alembert's principle. Since all the fundamental variational principles of mechanics --- the principles of Euler, Lagrange, Jacobi, Hamilton --- are but alternative mathematical formulations of d'Alembert's principle, Postulate $A$ is actually the *only* postulate of analytical mechanics, and is thus of fundamental importance.

</div>

When the impressed force $F_i$ is **monogenic**, i.e. derivable from a single scalar function, the work function $U(q_1, \ldots, q_n)$, the virtual work is equal to the variation of the work function. Since the work function can be replaced by the negative of the potential energy, the state of equilibrium of a mechanical system is distinguished by the stationary value of the potential energy, i.e. by the condition

$$\delta V = 0.$$

If the equilibrium is *stable*, the potential energy must assume its minimum value --- the minimum understood in the local sense --- while in general, equilibrium does not require the minimum, but only the stationary value, of $V$.

### 2. The Equilibrium of a Rigid Body

A rigid body which can move freely in space has six degrees of freedom: three on account of translation and three on account of rotation. Making use of the superposition principle of infinitesimal quantities, we can apply these two types of displacements independently of one another.

**(A) Translation.** An infinitesimal translation produces at each point of the rigid body the same displacement. Let $\epsilon$ be the extent of the infinitesimal displacement, and $\mathbf{B}$ a vector of unit length. We then have for the virtual displacement $\delta\mathbf{R}_k$ of the particle $P_k$:

$$\delta\mathbf{R}_k = \epsilon\mathbf{B},$$

and the resultant work becomes

$$\bar{\delta w} = \Sigma(\mathbf{F}_k \cdot \epsilon\mathbf{B}) = \epsilon\mathbf{B} \cdot \Sigma\mathbf{F}_k.$$

Since the vector $\mathbf{B}$ can be chosen in any direction, the vanishing of the virtual work requires:

$$\bar{\mathbf{F}} = \Sigma\mathbf{F}_k = 0,$$

which means that *the resultant force $\bar{\mathbf{F}}$ of all the impressed forces vanishes*.

**(B) Rotation.** Let $\epsilon$ be the angle of an infinitesimal rotation, and $\boldsymbol{\Omega}$ a vector of unit length along the axis of rotation. The displacement of the point $P_k$ due to the rotation can be written as

$$\delta\mathbf{R}_k = \epsilon\boldsymbol{\Omega} \times \mathbf{R}_k,$$

where $\mathbf{R}_k$ denotes the position vector of $P_k$ with respect to an origin on the axis of rotation. The work of the force $\mathbf{F}_k$ becomes

$$\bar{\delta w}_k = \mathbf{F}_k \cdot \epsilon\boldsymbol{\Omega} \times \mathbf{R}_k = \epsilon\boldsymbol{\Omega} \cdot (\mathbf{R}_k \times \mathbf{F}_k) = \epsilon\boldsymbol{\Omega} \cdot \mathbf{M}_k,$$

where

$$\mathbf{M}_k = \mathbf{R}_k \times \mathbf{F}_k$$

denotes the "moment of the force" about the origin. Hence the total work of all acting forces becomes

$$\bar{\delta w} = \Sigma\epsilon\boldsymbol{\Omega} \cdot \mathbf{M}_k = \epsilon\boldsymbol{\Omega} \cdot \Sigma\mathbf{M}_k.$$

Since a free body may rotate about any axis, and thus $\boldsymbol{\Omega}$ has an arbitrary direction, the vanishing of the virtual work requires that

$$\bar{\mathbf{M}} = \Sigma\mathbf{M}_k = 0.$$

*The resultant moment of all the impressed forces vanishes.*

Since an arbitrary virtual displacement of a rigid body can always be produced by the superposition of an infinitesimal translation and rotation, these two conditions together determine its equilibrium.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Special Cases of Rigid Body Equilibrium)</span></p>

1. The body may be fixed at the origin, destroying freedom of translation. Then the first condition ("the sum of all forces vanishes") no longer holds; but the second condition, the vanishing of the resultant moment, still holds. The radius vectors $\mathbf{R}_k$ are now measured from the fixed point; hence it is the resulting moment *about that point* which must vanish.

2. Not only one point of the body, but a whole axis may be fixed. In this case $\boldsymbol{\Omega}$ is no longer an arbitrary vector but has the direction of the given axis. The vanishing of the virtual work requires only

   $$\boldsymbol{\Omega} \cdot \bar{\mathbf{M}} = 0,$$

   which means that *only the component of the resultant moment in the direction of the axis of rotation has to vanish*.

</div>

### 3. Equivalence of Two Systems of Forces

The virtual work of the impressed forces determines not only the equilibrium but also the dynamical behaviour of a mechanical system. *Two systems of forces which produce the same virtual displacements are dynamically equivalent.*

Since the virtual work of an arbitrary system of forces acting on a rigid body depends solely on two quantities, viz. the resultant force $\bar{\mathbf{F}}$ and the resultant moment $\bar{\mathbf{M}}$, we get at once the important theorem that *two systems of forces which have the same resultant force and resultant moment are mechanically equivalent*.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Problems on Force Equivalence)</span></p>

**Problem 1.** Any given system of forces acting on a rigid body can be replaced by a single force if, and only if, the resultant moment $\bar{\mathbf{M}}$ and the resultant force $\bar{\mathbf{F}}$ are perpendicular to each other:

$$\bar{\mathbf{F}} \cdot \bar{\mathbf{M}} = 0.$$

**Problem 2.** Two forces can be replaced by a single force if, and only if, the two forces are coplanar.

**Problem 3.** An arbitrary *parallel* system of forces $\mathbf{F}_i = m\mathbf{G}_i$ can always be replaced by a single force $\bar{\mathbf{F}} = \bar{m}\mathbf{G}$ applied at the point

$$\mathbf{R}_0 = \frac{\Sigma m \mathbf{R}}{\Sigma m},$$

provided that $\bar{m} = \Sigma m$ is not zero.

**Problem 4.** An arbitrary system of forces can be replaced by a single force $\bar{\mathbf{F}} = \Sigma\mathbf{F}$ plus a couple whose axis is parallel to $\bar{\mathbf{F}}$.

</div>

### 4. Equilibrium Problems with Auxiliary Conditions

Occasionally, equilibrium problems occur which involve one or more auxiliary conditions. In such problems the infinitesimal virtual work $\bar{\delta w}$, before being equated to zero, must be augmented by the variation of the auxiliary conditions, each one multiplied by an undetermined Lagrangian factor $\lambda$, according to the general method discussed in Chapter II, sections 5 and 12.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(The Problem of Jointed Bars)</span></p>

Consider a system of uniform rigid bars of constant cross-section, freely jointed at their end-points. The two free ends of the chain are suspended. Find the position of equilibrium of the system.

The impressed force is gravity, which is monogenic. We determine the potential energy $V$ of the system which has to be minimized. Denoting by $x_k, y_k$ the rectangular coordinates of the end points of the bars --- the $x$-axis being chosen horizontally, the $y$-axis vertically downwards --- and the given lengths of the bars by $l_k$:

$$l_k^2 = (x_{k+1} - x_k)^2 + (y_{k+1} - y_k)^2 = (\Delta x_k)^2 + (\Delta y_k)^2, \quad (k = 0, 1, \ldots, n-1).$$

If $\sigma$ is the mass per unit length, the potential energy of the system is:

$$V = \frac{\sigma g}{2}\sum_{k=0}^{n-1}(y_k + y_{k+1})l_k.$$

This is the potential energy to be minimized, considering the length equations as $n$ auxiliary conditions. According to the general Lagrangian procedure we form the *modified* potential energy:

$$\bar{V} = \frac{1}{2}\sum_{k=0}^{n-1}(y_k + y_{k+1})l_k + \sum_{k=0}^{n-1}\lambda_k\left[(\Delta x_k)^2 + (\Delta y_k)^2\right].$$

Variation with respect to $x_k$ yields the difference equation

$$\lambda_k \Delta x_k - \lambda_{k-1}\Delta x_{k-1} = 0,$$

which is soluble in the form $\lambda_k = \frac{C}{\Delta x_k}$. Variation with respect to $y_k$ yields

$$\tfrac{1}{2}(l_k + l_{k-1}) - \lambda_k \Delta y_k + \lambda_{k-1}\Delta y_{k-1} = 0,$$

and if we substitute for $\lambda_k$ its value, we obtain the following difference equation as the solution of our equilibrium problem:

$$c\left(\frac{\Delta y_k}{\Delta x_k} - \frac{\Delta y_{k-1}}{\Delta x_{k-1}}\right) = \tfrac{1}{2}(l_k + l_{k-1}).$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(The Catenary — Continuous Limit)</span></p>

If the number of jointed bars increases more and more, while their lengths decrease, the bars approach a smooth, continuous and differentiable curve. In the limit we get the problem of the uniform *chain*. The difference equation approaches the differential equation

$$c\,\frac{d}{ds}\!\left(\frac{dy}{dx}\right) = 1,$$

where $ds$ denotes the line element of the curve. If we characterize the curve by $y = f(x)$, then this may be written as

$$\frac{y''}{\sqrt{1 + y'^2}} = \mathfrak{a},$$

where $\mathfrak{a} = \frac{1}{c}$. This equation can be integrated to give the well-known equation of the catenary:

$$y = \frac{1}{2\mathfrak{a}}\left(e^{\mathfrak{a}x} + e^{-\mathfrak{a}x}\right).$$

Alternatively, applying the direct methods of the calculus of variations, the curve is given in parametric form by $x = x(\tau)$, $y = y(\tau)$. The definite integral to be minimized is

$$V = \int_{\tau_1}^{\tau_2} y\sqrt{x'^2 + y'^2}\,d\tau,$$

with the auxiliary condition that the total length of the curve is constant:

$$\int_{\tau_1}^{\tau_2}\sqrt{x'^2 + y'^2}\,d\tau = \text{Constant}.$$

This is an isoperimetrical problem. The modified integral becomes

$$\bar{V} = \int_{\tau_1}^{\tau_2}(y + \lambda)\sqrt{x'^2 + y'^2}\,d\tau.$$

The essential difference between the two treatments is that in the first case --- where the auxiliary condition is of the local type --- the Lagrangian factor $\lambda$ will be a function of $\tau$; while in the second case --- where the auxiliary condition is an integral extended over the entire range --- $\lambda$ will be a constant.

</div>

### 5. Physical Interpretation of the Lagrangian Multiplier

Let us assume that we have a mechanical system of $n$ degrees of freedom, characterized by the generalized coordinates $q_1, \ldots, q_n$, and that there is a kinematical condition given in the form

$$f(q_1, \ldots, q_n) = 0.$$

The Lagrangian multiplier method requires that

$$\delta V + \lambda\,\delta f = 0.$$

This equation can be expressed in the form $\delta\bar{V} = 0$, where

$$\bar{V} = V + \lambda f.$$

Once more we have the problem of finding the stationary value of a function, but that function is no longer the original potential energy $V$ but a **modified potential energy** $\bar{V}$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Physical Significance of the Lagrangian Multiplier)</span></p>

The additional term $\lambda f$ is not merely a matter of mathematical method but has a very real physical significance. *The modification of the potential energy on account of the Lagrangian $\lambda$-method represents the potential energy of the forces which are responsible for the maintenance of the given auxiliary conditions.*

If we do not restrict the variation of the configuration by the condition $f = 0$ but permit *arbitrary* variations of the $q_i$, then not only the impressed forces will act but also the forces which maintain the given kinematical condition. They too have a potential energy which has to be added to the potential energy of the impressed forces.

The additional potential energy is $V_1 = \lambda f$. Forming the gradient of the additional potential energy, the "force of reaction" $F_{1i}$ is

$$F_{1i} = -\frac{\partial V_1}{\partial x_i} = -\lambda\frac{\partial f}{\partial x_i} - \frac{\partial\lambda}{\partial x_i}f.$$

At the point $C$ of the configuration space which lies on the surface $f = 0$, the second term drops out because it is multiplied by $f$ which vanishes at $C$; and thus the "force of reaction" $F_{1i}$ is reduced to

$$F_{1i} = -\lambda\,\frac{\partial f}{\partial x_i}.$$

We see that the Lagrangian $\lambda$-term has the remarkable property of giving us the force of reaction connected with a given kinematical constraint. The same holds not only in the state of equilibrium but also in the state of motion (cf. Chapter V, section 8).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Holonomic vs. Non-holonomic Conditions)</span></p>

If a given kinematical condition is non-holonomic, we can no longer modify the function which is to be minimized. But the $\lambda$-terms still appear in the conditions of equilibrium. This again has a direct physical significance: the $\lambda$-terms augment the impressed forces by those forces which maintain the given kinematical constraints. No work function exists in this case, *but once more the forces of reaction are provided*.

Hence the ingenious method of the Lagrangian multiplier elucidates the nature of holonomic and non-holonomic kinematical conditions: **a holonomic condition is maintained by monogenic forces; a non-holonomic condition is maintained by polygenic forces.**

</div>

### 6. Fourier's Inequality

All our previous conclusions were made under the tacit assumption that the virtual displacements are reversible. This is the case if we are somewhere *inside* the configuration space, so that motion can occur in every direction. The situation is quite different if we reach the *boundaries* of the configuration space. Here a virtual displacement has to be directed *inward* and the opposite displacement is not possible because it would lead out of the region.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Irreversible Displacements)</span></p>

Consider a ball hanging on the end of a flexible string. That ball can move upwards, which merely relaxes the string. But it cannot move downwards because the string does not permit it.

Again, a ball on the table can move on the surface of the table but it can also move upwards in any direction; however, it cannot move downwards. The virtual displacement is reversible if the motion occurs in the horizontal direction, but it is irreversible in all other directions.

</div>

Fourier pointed out that the ordinary formulation of the principle of virtual work

$$\delta V = 0$$

is restricted to *reversible* displacements; for irreversible displacements it has to be replaced by the inequality

$$\delta V \geq 0.$$

Indeed, our final goal is to minimize the potential energy. This requires for small but finite displacements that $V$ shall always increase: $\Delta V > 0$. For infinitesimal displacements this must be replaced by $\Delta V \geq 0$. There is now no objection to the "greater than" sign: the only reason we were able to remove it before was that "greater than" if applied to the reversed displacement would become "smaller than," and "smaller than zero" contradicts the hypothesis of a minimum. For irreversible displacements, however, the argument is no longer valid, and we have to leave $\Delta V \geq 0$ in its original form because there is no objection to positive changes of the potential energy.

Since the change of the potential energy is the same as the negative of the virtual work of the forces, we can express the inequality in the form

$$\bar{\delta w} \leq 0.$$

In this form Fourier's inequality includes even polygenic forces which have no potential energy. *If the work of the forces for any virtual displacement is zero or negative, the system is in equilibrium.*

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Equilibrium at the Boundary)</span></p>

Any mechanical system which cannot come to equilibrium *within* the configuration space will move to the *boundary* of that space and find its equilibrium there. It is easier to satisfy the inequality $\bar{\delta w} \leq 0$ on the boundary of a region than the equality $\delta V = 0$ inside the region. Notice that on the boundary of the configuration space equilibrium does not require a stationary value of the potential energy.

</div>

## Chapter IV: D'Alembert's Principle

With d'Alembert's principle we leave the realm of statics and enter the realm of dynamics. While the problems of statics of systems of a finite number of degrees of freedom lead to algebraic equations solvable by eliminations and substitutions, the problems of dynamics lead to differential equations. D'Alembert's principle, discussed in this chapter, did not contribute directly to the problem of integration. Yet it is an important landmark in the history of theoretical mechanics, since it gives an interpretation of the force of inertia which is fundamental for the later development of variational methods.

### 1. The Force of Inertia

D'Alembert (1717–1783) had the ingenious idea of extending the applicability of the principle of virtual work from statics to dynamics. Starting with Newton's fundamental law of motion

$$m\mathbf{A} = \mathbf{F},$$

we can rearrange it as

$$\mathbf{F} - m\mathbf{A} = 0.$$

We define a new vector $\mathbf{I}$ by

$$\mathbf{I} = -m\mathbf{A}.$$

This vector can be thought of as a force created by the motion itself. It is called the **force of inertia**. With this concept, Newton's equation takes the form

$$\mathbf{F} + \mathbf{I} = 0.$$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why This Reformulation Matters)</span></p>

At first glance, introducing $\mathbf{I} = -m\mathbf{A}$ seems like a trivial relabelling. But the importance of $\mathbf{F} + \mathbf{I} = 0$ lies in its interpretation as a *principle*: the vanishing of a sum of forces signals equilibrium. Since the addition of the force of inertia to the other acting forces produces equilibrium, any criterion we have for the equilibrium of a mechanical system can immediately be extended to a system which is in motion --- all we have to do is add the new "force of inertia" to the impressed forces. In this way, **dynamics is reduced to statics**.

This does not mean that we can actually *solve* a dynamical problem by statical methods. The resulting equations are *differential equations* which still need to be solved. We have merely *deduced* those differential equations by statical considerations. The addition of the force of inertia $\mathbf{I}$ to the acting force $\mathbf{F}$ converts the problem of motion into a problem of equilibrium.

</div>

We define the **effective force** $\mathbf{F}^e_k$ acting on a particle as the sum of the impressed force and the force of inertia:

$$\mathbf{F}^e_k = \mathbf{F}_k + \mathbf{I}_k.$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(D'Alembert's Principle)</span></p>

The total virtual work of the effective forces is zero for all reversible variations which satisfy the given kinematical conditions:

$$\sum_{k=1}^{N}\mathbf{F}^e_k \cdot \delta\mathbf{R}_k \equiv \sum_{k=1}^{N}(\mathbf{F}_k - m_k\mathbf{A}_k)\cdot\delta\mathbf{R}_k = 0.$$

</div>

Note that the impressed forces $\mathbf{F}_k$ may act at just a few points, while the effective forces $\mathbf{F}^e_k$ are present wherever a mass is in accelerated motion. A given system of impressed forces will generally not be in equilibrium on its own. The body moves in such a way that the additional inertial forces, produced by the motion, bring the balance up to zero. In this way d'Alembert's principle yields *the equations of motion of an arbitrary mechanical system*.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Monogenic vs. Polygenic Character)</span></p>

The impressed forces are usually monogenic --- derivable from a single work function by differentiation --- so that the virtual work of the impressed forces can be written as $\bar{\delta w} = \delta U$. The virtual work of the forces of inertia,

$$\bar{\delta w}^i = -\Sigma m_k \mathbf{A}_k \cdot \delta\mathbf{R}_k,$$

is merely a differential form, not reducible to the variation of a scalar function. (We shall see later how this shortcoming can be remedied by an integration with respect to time.)

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Variable Mass and Momentum)</span></p>

Newton's equation $m\mathbf{A} = \mathbf{F}$ holds only if the mass is a constant. If $m$ changes during the motion, the fundamental equation must be written as

$$\frac{d}{dt}(m\mathbf{v}) = \mathbf{F},$$

i.e. "rate of change of momentum equals moving force." Accordingly, the force of inertia is then defined as the negative rate of change of momentum:

$$\mathbf{I} = -\frac{d}{dt}(m\mathbf{v}).$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Physical Significance and Postulate A)</span></p>

From the definition of the effective force, $\mathbf{F}^e = \mathbf{F} + \mathbf{I}$, this force is zero for a free particle and equals the negative force of reaction for a constrained particle. Hence the application of the principle of virtual work to $\mathbf{F}^e$ is equivalent to the assumption that "the virtual work of the forces of reaction is zero for any virtual displacement which is in harmony with the given constraints." We thus come back to the same **Postulate $A$** that we encountered in Chapter III, section 1. D'Alembert's principle generalizes this postulate from the field of statics to the field of dynamics, without any alteration.

</div>

### 2. The Place of D'Alembert's Principle in Mechanics

D'Alembert's principle provides a complete solution of problems of mechanics. All the different principles of mechanics --- those of Euler, Lagrange, Jacobi, Hamilton --- are merely mathematically different formulations of d'Alembert's principle. The most advanced variational principle, Hamilton's principle, can be obtained from d'Alembert's principle by a mathematical transformation. It is equivalent to d'Alembert's principle where both are applicable. In fact, Hamilton's principle is restricted to holonomic systems, while d'Alembert's principle can equally well be applied to holonomic and non-holonomic systems.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Advantages and Limitations)</span></p>

D'Alembert's principle is more elementary than the later variational principles because it requires no integration with respect to time. But the disadvantage is that the virtual work of the inertial forces is a polygenic quantity and thus not reducible to a single scalar function. This makes the principle unsuited to the analytical use of curvilinear coordinates. However, in many elementary problems of dynamics --- especially those which can be treated with rectangular coordinates, or those that involve non-holonomic constraints ("kinematical variables") --- d'Alembert's principle is of great use.

D'Alembert's principle is fundamental in still another respect: it makes possible the use of moving reference systems, and is thus a forerunner of Einstein's revolutionary ideas concerning the relativity of motion, explaining --- within the scope of Newtonian physics --- the nature of those "apparent forces" which are present in a moving frame of reference.

</div>

### 3. The Conservation of Energy as a Consequence of D'Alembert's Principle

Although d'Alembert's principle is generally of a polygenic nature, in one special case it becomes monogenic and integrable. This special case leads to one of the most fundamental laws of mechanics: the **law of the conservation of energy**.

We consider the general formulation of d'Alembert's principle. Assume that the impressed forces are monogenic, derivable from a potential energy function. Then the work of the impressed forces equals the negative variation of the potential energy, and d'Alembert's principle may be written as

$$\delta V + \Sigma m_k \mathbf{A}_k \cdot \delta\mathbf{R}_k = 0.$$

The second term --- the negative virtual work of the forces of inertia --- cannot in general be written as the variation of anything. But now let us make a special choice: let the tentative virtual displacements $\delta\mathbf{R}_k$ coincide with the *actual* displacements $d\mathbf{R}_k$ that occur during the time $dt$. This is merely a special (but always permissible) application of the variational principle.

For this special choice, $\delta V$ coincides with the actual change $dV$ that occurs during $dt$. Moreover, the second term becomes a perfect differential. Since $\mathbf{A}_k = \ddot{\mathbf{R}}_k$, we have

$$\Sigma m_k \ddot{\mathbf{R}}_k \cdot d\mathbf{R}_k = \Sigma m_k \ddot{\mathbf{R}}_k \cdot \dot{\mathbf{R}}_k \, dt = \frac{d}{dt}\!\left(\tfrac{1}{2}\Sigma m_k \dot{\mathbf{R}}_k^2\right)dt = dT,$$

where

$$T = \tfrac{1}{2}\Sigma m_k \mathbf{v}_k^2$$

is the **kinetic energy** of the mechanical system. D'Alembert's principle now takes the form

$$dV + dT = d(V + T) = 0,$$

which can be integrated to give

$$T + V = \text{constant} = E.$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Conservation of Energy)</span></p>

The sum of the kinetic and potential energies remains unchanged during the motion. This fundamental result is called the "law of the conservation of energy." The constant $E$ is called the **energy constant**. It is a scalar equation and only *one* of the integrals of the equations of motion --- although in itself insufficient for the complete solution of the problem of motion (except for a single degree of freedom), it is one of the most fundamental and universal laws of nature.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Conditions for Validity)</span></p>

The derivation of energy conservation requires careful attention to when $\delta\mathbf{R}_i$ can be identified with $d\mathbf{R}_i$. This identification is always permissible for free particles, but not always for systems with constraints. Letting $\delta q_i$ coincide with $dq_i$ in generalized coordinates will not always mean that the variations $\delta\mathbf{R}_i$ of the particle positions coincide with the actual displacements $d\mathbf{R}_i$. The identification holds in the **scleronomic** case --- where the kinematical conditions do not contain the time explicitly --- but fails in the **rheonomic** case --- where the kinematical conditions involve the time explicitly.

A similar remark holds with regard to the potential energy $V$. If $V$ is a function of the $q_i$ only, then the choice $\delta q_i = dq_i$ gives $\delta V = dV$. But this would no longer be true if $V$ were rheonomic, depending on $t$ explicitly. Since $V$ was defined as $-U$, the conservation of energy requires that the work function $U$ must not depend on $t$ explicitly.

Thus, the law of the conservation of energy holds only for systems which are **scleronomic in both work function and kinematical conditions**. Moreover, the deduction assumed that the masses $m_k$ are *constants*.

</div>

### 4. Apparent Forces in an Accelerated Reference System. Einstein's Equivalence Hypothesis

The fundamental soundness of d'Alembert's principle can be tested by the most exact physical experiments. D'Alembert's principle assumes that the force of inertia is just one more additional force which acts exactly like all the other forces. For the definition of the force of inertia we have to make use of an "absolute reference system." If the reference system in which the accelerations are measured is itself in motion relative to the absolute system, the force of inertia measured in the moving system is no longer the true force of inertia but has to be corrected by additional terms.

According to d'Alembert's principle, the correction terms which enter the force of inertia on account of the motion of the reference system can equally well be interpreted as belonging to the impressed forces. This gives rise to the phenomenon of **"apparent forces"** (or "d'Alembert forces") which occur in reference systems that are in motion.

**(A) Translatory motion.** Consider a reference system $S'$ in purely translatory motion (such as an elevator). The origin $O'$ of $S'$ moves along some given curve $C$ traced out by the vector $\mathbf{C}$ which is a function of time $t$. The radius vector $\mathbf{R}'$ measured in the moving system and the radius vector $\mathbf{R}$ measured in the absolute system are related by

$$\mathbf{R} = \mathbf{C} + \mathbf{R}'.$$

Differentiating twice with respect to time gives

$$\mathbf{A} = \ddot{\mathbf{C}} + \mathbf{A}'.$$

Multiplying by $-m$, the force of inertia decomposes as

$$\mathbf{I} = -m\ddot{\mathbf{C}} + \mathbf{I}'.$$

The quantity $-m\ddot{\mathbf{C}}$ is part of the true force of inertia and is due to the fact that the reference system is in motion relative to the absolute system. Whenever the motion of the reference system generates a force which has to be added to the relative force of inertia $\mathbf{I}'$, measured in that system, we call that force an **"apparent force."** In the moving reference system the apparent force is a perfectly real force which is not distinguishable in its nature from any other impressed force.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Einstein's Equivalence Hypothesis)</span></p>

Suppose we are in a closed laboratory which is pulled upwards with constant acceleration $\mathbf{G}$, while the field of gravity suddenly disappears (Einstein's "box-experiment"). Then d'Alembert's principle implies that no mechanical experiment can detect this change, because it is impossible to distinguish between the following two hypotheses:

1. Our reference system is moving upwards with constant acceleration $\mathbf{G}$; no field of gravity exists.
2. Our reference system is at rest but there is a field of gravity present which pulls every mass $m$ downwards with the force $m\mathbf{G}$.

This "equivalence hypothesis" was introduced by Einstein in his early attempts to solve the riddle of gravity, before establishing the general principle of relativity. For purely mechanical phenomena the equivalence hypothesis is a direct consequence of d'Alembert's principle. Einstein raised it to a general principle of nature. It seems justifiable to call the apparent force

$$\mathbf{E} = -m\ddot{\mathbf{C}},$$

which acts in an accelerated reference system on account of d'Alembert's principle, the **"Einstein force."** Note that all apparent forces, being generated by the force of inertia $-m\mathbf{A}$, are *proportional to the inertial mass* $m$. The empirical fact that the gravitational mass is always proportional to the inertial mass was the decisive clue which enabled Einstein to establish the force of gravity as an apparent force.

</div>

### 5. Apparent Forces in a Rotating Reference System

Any motion of a rigid reference system can be decomposed into a translation plus a rotation. Having studied a pure translation, we now deal with the case of pure rotation (the general problem is a superposition of these two special problems).

If the origin $O$ of the reference system is kept fixed, the radius vectors $\mathbf{R}$ and $\mathbf{R}'$ in the absolute system $S$ and the moving system $S'$ are the same: $\mathbf{R} = \mathbf{R}'$. Nevertheless, the velocities and accelerations measured in the two systems differ because the rates of change observed in the two systems are different. If a certain vector $\mathbf{B}$ is constant in $S'$, it rotates with the system, and thus, if observed in $S$, it undergoes in the time $dt$ an infinitesimal change

$$d\mathbf{B} = (\boldsymbol{\Omega} \times \mathbf{B})\,dt,$$

where $\boldsymbol{\Omega}$ is the angular velocity vector giving the rotation of $S'$ with respect to $S$. The general relation between the rate of change of any vector $\mathbf{B}$ as observed in $S$ and as observed in $S'$ is therefore

$$\frac{d\mathbf{B}}{dt} = \frac{d'\mathbf{B}}{dt} + \boldsymbol{\Omega} \times \mathbf{B},$$

where $d'/dt$ denotes the rate of change as observed in the moving system $S'$.

Applying this to the radius vector $\mathbf{R} = \mathbf{R}'$ gives the relation between velocities:

$$\mathbf{v} = \mathbf{v}' + \boldsymbol{\Omega} \times \mathbf{R}'.$$

Differentiating this relation to obtain the accelerations, and using the same operator identity to express everything in terms of quantities belonging to $S'$, one finds after multiplication by $-m$ the relation between the two forces of inertia:

$$\mathbf{I} = \mathbf{I}' - 2m\,\boldsymbol{\Omega}\times\mathbf{v}' - m\,\boldsymbol{\Omega}\times(\boldsymbol{\Omega}\times\mathbf{R}') - m\,\dot{\boldsymbol{\Omega}}\times\mathbf{R}'.$$

Agreeing to stay consistently in the rotating system $S'$ (and dropping the primes), the motion in the rotating system gives rise to three apparent forces of a different character:

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Apparent Forces in a Rotating Frame)</span></p>

1. **Centrifugal force:**
   $$\mathbf{C} = m\omega^2\mathbf{R}_\perp,$$
   where $\mathbf{R}_\perp$ is the component of the radius vector perpendicular to the rotation axis. This force is monogenic, being derivable from the potential energy $\phi = -\frac{m}{2}\omega^2 r_\perp^2$. On the earth, the plumb-line indicates the direction of the resultant of gravity and the centrifugal force; d'Alembert's principle shows that these two forces cannot be separated by any experiment.

2. **Coriolis force:**
   $$\mathbf{B} = -2m\,\boldsymbol{\Omega}\times\mathbf{v} = 2m\,\mathbf{v}\times\boldsymbol{\Omega},$$
   which depends on the velocity of the moving particle. It is always perpendicular to the velocity and thus does no work. Its horizontal component can be demonstrated by Foucault's celebrated experiment of the precessing pendulum --- the first mechanical demonstration of the rotation of the earth.

3. **Euler force:**
   $$\mathbf{K} = -m\,\dot{\boldsymbol{\Omega}}\times\mathbf{R},$$
   which is present only if the angular velocity vector changes in either direction or magnitude. It vanishes for uniform rotation about a fixed axis.

</div>

The total effective force acting in a rotating system is therefore

$$\mathbf{F}^e = \mathbf{F} + \mathbf{C} + \mathbf{B} + \mathbf{K} + \mathbf{I}$$

(impressed force + centrifugal force + Coriolis force + Euler force + force of inertia).

### 6. Dynamics of a Rigid Body. The Motion of the Centre of Mass

As an illustration of apparent forces and d'Alembert's principle, consider the dynamics of a rigid body which can move freely in space. We assume an observer who makes his measurements in a reference system attached to the moving body. In this reference system the body is at rest and is thus in equilibrium.

From Chapter III, section 2, the equilibrium of a rigid body requires two vector conditions: the sum of all forces and the sum of all moments must vanish. In the present section we consider the first condition only.

In forming the resultant force $\mathbf{F}^e$ we must include, in addition to the impressed forces $\mathbf{F}$, the apparent forces acting in our system. Let the origin $O'$ of our reference system coincide with the centre of mass of the rigid body. This means $\Sigma m\mathbf{R} = 0$.

One can show that the vector sum of the centrifugal forces and the vector sum of the Euler forces both vanish in this reference system. Since these resultant apparent forces vanish, we must have equilibrium between the impressed forces and the Einstein forces. This gives the condition

$$\bar{\mathbf{F}} - \Sigma m\ddot{\mathbf{C}} = 0,$$

or equivalently

$$\bar{m}\,\ddot{\mathbf{C}} = \bar{\mathbf{F}},$$

where $\bar{m} = \Sigma m$ is the total mass.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Motion of the Centre of Mass)</span></p>

The centre of mass of a rigid body moves like a particle in which the total mass of the body is concentrated and on which the resultant of all the forces impressed on the rigid body acts. This is the well-known theorem of the motion of the centre of mass, here derived as a consequence of d'Alembert's principle and the equilibrium of impressed and Einstein forces.

</div>

### 7. Dynamics of a Rigid Body. Euler's Equations

We now turn to the *second* condition for the equilibrium of a rigid body, freely movable in space: the sum of all moments must vanish. Since the Einstein forces have no resultant moment (as can be shown), the sum of the moments of the following three forces must vanish: the impressed forces, the centrifugal forces, and the Euler forces.

We set up a rectangular coordinate system $x, y, z$ with origin at the centre of mass. These axes are rigidly connected with the moving body and coincide with the **"principal axes"** --- meaning that the "products of inertia" are zero:

$$\Sigma myz = \Sigma mzx = \Sigma mxy = 0.$$

We introduce the customary **moments of inertia**:

$$A = \Sigma m(y^2+z^2), \quad B = \Sigma m(z^2+x^2), \quad C = \Sigma m(x^2+y^2).$$

Denoting the components of the resultant moment of the impressed forces by $\bar{M}_x, \bar{M}_y, \bar{M}_z$, and computing the moments of the centrifugal forces and the Euler forces, the vanishing of the total resultant of the moments gives **Euler's equations**:

$$A\dot{\omega}_x - (B-C)\omega_y\omega_z = \bar{M}_x,$$

$$B\dot{\omega}_y - (C-A)\omega_z\omega_x = \bar{M}_y,$$

$$C\dot{\omega}_z - (A-B)\omega_x\omega_y = \bar{M}_z.$$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Interpretation of Euler's Equations)</span></p>

These famous equations describe how the instantaneous axis of rotation $\boldsymbol{\Omega}$ changes with the time relative to a body-centred reference system of the principal axes. They can be interpreted as the conditions for the vanishing of the resultant of three moments: those of the Euler forces, the centrifugal forces, and the external forces. They give a partial solution of the dynamical problem of a freely rotating rigid body, but they have to be completed by fixing the position of the body-centred reference system relative to a space-centred system.

</div>

### 8. Gauss' Principle of Least Constraint

D'Alembert's principle is not a minimum principle. It makes use of an infinitesimal quantity --- the virtual work of the impressed forces, augmented by the virtual work of the forces of inertia --- but the latter quantity is not the variation of some function. Gauss (1777–1855) gave an ingenious reinterpretation of d'Alembert's principle which changes it into a minimum principle.

The idea proceeds as follows. The differential equations of motion determine the *accelerations* of the position coordinates $q_i$. At a certain time $t$ the position and velocity of all the particles of a mechanical system are given. The accelerations may be disposed of at will, though subject to the given constraints. If the actual acceleration of a particle at time $t$ is $\mathbf{A}(t)$, and we change it to $\mathbf{A} + \delta\mathbf{A}$, the position of the particle at time $t + \tau$ is given by Taylor's theorem:

$$\mathbf{R}(t+\tau) = \mathbf{R}(t) + \mathbf{v}(t)\tau + \tfrac{1}{2}\mathbf{A}(t)\tau^2 + \ldots$$

The variation of the acceleration causes a deviation in the path of the particle. Since we do not change the initial position and velocity, if we consider $\tau$ as an arbitrarily small time interval and neglect terms of higher order, we obtain

$$\delta\mathbf{R}(t+\tau) = \tfrac{1}{2}\delta\mathbf{A}(t)\cdot\tau^2.$$

Now, d'Alembert's principle requires that $\sum(\mathbf{F}_i - m_i\mathbf{A}_i)\cdot\delta\mathbf{R}_i = 0$, and since the $\delta\mathbf{R}_i$ are proportional to $\delta\mathbf{A}_i$, this gives $\sum(\mathbf{F}_i - m_i\mathbf{A}_i)\cdot\delta\mathbf{A}_i = 0$. Since the impressed forces $\mathbf{F}_i$ are given and cannot be varied, this condition can be rewritten as

$$\sum_{i=1}^{N}(\mathbf{F}_i - m_i\mathbf{A}_i)\cdot\delta\!\left(\frac{\mathbf{F}_i - m_i\mathbf{A}_i}{m_i}\right) = 0,$$

which means that

$$\delta\sum_{i=1}^{N}\frac{1}{2m_i}(\mathbf{F}_i - m_i\mathbf{A}_i)^2 = 0.$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Gauss' Principle of Least Constraint)</span></p>

Define the **"constraint"** of the motion as

$$Z = \sum_{i=1}^{N}\frac{1}{2m_i}(\mathbf{F}_i - m_i\mathbf{A}_i)^2.$$

Then *the actual motion occurring in nature is such that under the given kinematical conditions the constraint becomes as small as possible*. If the particles are free from constraints, $Z$ can assume its absolute minimum, which is zero, giving $m_i\mathbf{A}_i = \mathbf{F}_i$ --- Newton's law. If constraints prevent the free choice of the $\mathbf{A}_i$, we can still minimize $Z$ under the given auxiliary conditions, and the solution obtained yields the actual motion realized in nature.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Particle on a Surface)</span></p>

A particle is forced to stay on the surface $z = f(x, y)$ and is acted upon by the force $\mathbf{F}$. The constraint of the surface gives the auxiliary condition $\ddot{z} = f_x\ddot{x} + f_y\ddot{y} + f_{xx}\dot{x}^2 + \ldots$ Minimizing the Gaussian constraint $Z = \frac{1}{2m}(\mathbf{F} - m\mathbf{A})^2$ subject to this auxiliary condition yields the equations of motion directly.

</div>

## Chapter V: The Lagrangian Equations of Motion

We now transition from the differential principles considered so far (d'Alembert, Gauss) to the true *variational principles*, which operate with the minimum --- or more generally the stationary value --- of a definite integral. The polygenic character of the inertial forces can be overcome by integrating with respect to time, reducing the dynamical problem to the study of a scalar integral whose stationary value encodes all the equations of motion.

The various names attached to these principles --- Euler, Lagrange, Jacobi, Hamilton --- all refer to closely related formulations of the "principle of least action." We begin with Hamilton's principle, which provides the most direct and natural transformation of d'Alembert's principle into a minimum principle.

### 1. Hamilton's Principle

D'Alembert's principle sets an infinitesimal quantity --- the total virtual work done by impressed and inertial forces --- equal to zero at each instant. The virtual work of the impressed forces is monogenic (derivable from a single work function), but the virtual work of the inertial forces is not: it must be computed particle by particle. Hamilton's key insight is that integrating over time converts the inertial contribution into a monogenic form as well.

We multiply the d'Alembert equation by $dt$ and integrate between times $t_1$ and $t_2$. Splitting the result into an impressed-force part and an inertial-force part, the impressed-force contribution gives $-\delta\int_{t_1}^{t_2}V\,dt$ (assuming the work function is velocity-independent and setting $V = -U$). For the inertial part we integrate by parts, which produces a boundary term and a bulk term involving the variation of the velocities. Using the interchangeability of variation and differentiation, the bulk term becomes the variation of the integral of the kinetic energy.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Lagrangian Function)</span></p>

The **Lagrangian function** is defined as the excess of kinetic energy over potential energy:

$$L = T - V.$$

This single scalar function is the most fundamental quantity in the mathematical analysis of mechanical problems.

</div>

Combining all the terms, the time-integrated virtual work becomes

$$\int_{t_1}^{t_2}\overline{\delta w^e}\,dt = \delta\int_{t_1}^{t_2}L\,dt - \bigl[\Sigma\, m_i \mathbf{V}_i \cdot \delta\mathbf{R}_i\bigr]_{t_1}^{t_2}.$$

So far the virtual displacements $\delta\mathbf{R}_i$ at the endpoints are arbitrary. The crucial step is to require that **the variations vanish at both time limits** $t_1$ and $t_2$:

$$\delta\mathbf{R}_i(t_1) = 0, \qquad \delta\mathbf{R}_i(t_2) = 0.$$

This means we fix the configuration of the system at the initial and final instants and allow arbitrary variations only in between --- we "vary between definite limits." Under this condition the boundary term drops out, and the time-integrated virtual work becomes a true variation of a definite integral.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Action Integral)</span></p>

The **action** is the time-integral of the Lagrangian between fixed initial and final times:

$$A = \int_{t_1}^{t_2} L\,dt.$$

</div>

Since d'Alembert's principle demands that the virtual work vanish at every instant, the left-hand side of the integrated equation vanishes, and we arrive at Hamilton's principle.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Hamilton's Principle)</span></p>

The motion of a mechanical system occurs in such a way that the action integral

$$A = \int_{t_1}^{t_2}L\,dt$$

is **stationary** for arbitrary possible variations of the configuration, provided the initial and final configurations are held fixed:

$$\delta A = 0.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Equivalence with d'Alembert's Principle)</span></p>

The reasoning can be reversed: starting from $\delta A = 0$ one can recover $\overline{\delta w^e} = 0$ at each instant, which is d'Alembert's principle. Hence Hamilton's principle and d'Alembert's principle are mathematically equivalent whenever the impressed forces are monogenic and the kinematical conditions are holonomic. The conservative nature of these forces --- i.e. their independence of time --- is *not* required.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Scope of Hamilton's Principle)</span></p>

Hamilton's principle holds for arbitrary mechanical systems characterized by monogenic forces and holonomic auxiliary conditions. While d'Alembert's principle makes an independent statement at each instant of time, Hamilton's principle treats the motion *as a whole* --- the entire trajectory between $t_1$ and $t_2$ enters at once. This integral viewpoint is the hallmark of variational mechanics and extends far beyond classical mechanics: both the equations of general relativity and quantum wave mechanics are derivable from a "principle of least action," differing only in the choice of Lagrangian $L$.

</div>

### 2. The Lagrangian Equations of Motion and Their Invariance Relative to Point Transformations

In the derivation of Hamilton's principle, rectangular coordinates were used. However, a mechanical system may be subject to holonomic kinematical conditions. If these are present, the $3N$ rectangular position coordinates can be expressed in terms of $n$ generalized coordinates $q_1, \ldots, q_n$. Both the kinetic and potential energies then become functions of the $q_i$ and the generalized velocities $\dot{q}_1, \ldots, \dot{q}_n$.

The motion of the entire system can be visualized as the trajectory of a single representative point (the *C*-point) through the $n$-dimensional configuration space of the $q_i$. Going one step further, we can add the time $t$ as an additional dimension and work in an $(n+1)$-dimensional "extended configuration space." In this picture the successive states of the system trace out a **world-line** --- a curve $C$ whose shape encodes the complete physical history.

Varying the world-line between fixed endpoints $A$ (at $t_1$) and $B$ (at $t_2$) and demanding that the action integral be stationary yields the **Euler--Lagrange equations** --- the necessary and sufficient conditions for stationarity.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Lagrangian Equations of Motion)</span></p>

For a system described by generalized coordinates $q_1, \ldots, q_n$ and Lagrangian $L(q_i, \dot{q}_i, t)$, the equations of motion are

$$\frac{d}{dt}\frac{\partial L}{\partial \dot{q}_i} - \frac{\partial L}{\partial q_i} = 0, \qquad i = 1, \ldots, n.$$

These form a system of $n$ simultaneous second-order ordinary differential equations that completely determine the motion once initial positions and velocities are specified.

</div>

A remarkable property of these equations is their **invariance under arbitrary point transformations**. If we switch from coordinates $q_i$ to a new set $\bar{q}_i$ via any smooth invertible transformation, the Lagrangian $L$ changes its functional form but not its numerical value. Since the action integral is the same number regardless of the coordinate system, the condition $\delta A = 0$ holds in both the old and the new coordinates. Consequently the *complete set* of Lagrangian equations maps into the corresponding set in the new reference system.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Invariance and Moving Frames)</span></p>

Because time $t$ enters as just another variable in the extended configuration space, the invariance persists even when the coordinate transformation depends on $t$. This means the Lagrangian equations remain valid in arbitrarily moving reference systems --- a property that distinguishes analytical mechanics from the Newtonian vectorial approach, where non-inertial frames require fictitious forces. The invariance of the Lagrangian equations was historically the first example of the "principle of invariance" that became a leading idea in 19th-century mathematics and modern physics.

</div>

A practical advantage of the variational approach, compared with d'Alembert's principle, is the use of the single scalar function $L = T - V$. There is no need to determine the acceleration and virtual work for each particle individually; the scalar Lagrangian determines the entire dynamics.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Compound Pendulum)</span></p>

Consider a compound pendulum whose position is described by the angle $\theta$ between the plane through the axis of suspension and the centre of mass and the vertical. The kinetic and potential energies are

$$T = \tfrac{1}{2}I\dot{\theta}^2, \qquad V = Mgl(1 - \cos\theta),$$

where $I$ is the moment of inertia about the axis of suspension, $M$ the total mass, and $l$ the distance from the axis to the centre of mass. Substituting into the Lagrangian equation gives the equation of motion, which for small $\theta$ reduces to simple harmonic oscillation.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Planetary Motion in Polar Coordinates)</span></p>

For a planet of mass $m$ moving under the gravitational attraction of a sun of mass $M$ fixed at the origin, polar coordinates $(r, \theta)$ give

$$T = \frac{m}{2}(\dot{r}^2 + r^2\dot{\theta}^2), \qquad V = -\frac{fmM}{r},$$

where $f$ is the gravitational constant. The two Lagrangian equations yield the radial force balance and the conservation of angular momentum.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Spherical Pendulum)</span></p>

A pendulum of length $l$ and mass $m$ free to swing in any direction is described by spherical polar coordinates $(\theta, \varphi)$. The energies are

$$T = \frac{m}{2}l^2(\dot{\theta}^2 + \sin^2\theta\;\dot{\varphi}^2), \qquad V = mgl(1-\cos\theta).$$

The Lagrangian equations produce two coupled second-order equations governing the motion of the bob on the sphere.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Force of Inertia vs. Moving Force)</span></p>

The Lagrangian equations can be split into two sides by writing them as

$$\frac{d}{dt}\frac{\partial T}{\partial \dot{q}_i} - \frac{\partial T}{\partial q_i} = -\frac{d}{dt}\frac{\partial U}{\partial \dot{q}_i} + \frac{\partial U}{\partial q_i}.$$

The left-hand side represents the $n$ components of the **generalized force of inertia** in configuration space; the right-hand side represents the $n$ components of the **generalized moving (applied) force**. Ordinarily the work function $U$ depends only on the $q_i$ and the first term on the right vanishes, but there is no fundamental reason to exclude velocity-dependent potentials --- in relativistic electrodynamics the force on a charged particle is indeed derived from such a potential.

</div>

### 3. The Energy Theorem as a Consequence of Hamilton's Principle

The law of conservation of energy, previously derived from d'Alembert's principle, can also be obtained directly from Hamilton's principle. The derivation reveals the deep connection between the total energy of a mechanical system and the Lagrangian.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Generalized Momentum)</span></p>

The partial derivatives of the Lagrangian with respect to the generalized velocities play a central role in analytical mechanics. They are called the **generalized momenta** (or conjugate momenta):

$$p_i = \frac{\partial L}{\partial \dot{q}_i}.$$

For a single free particle in rectangular coordinates, these reduce to the ordinary momentum components $m\dot{x}, m\dot{y}, m\dot{z}$. In general, however, the $p_i$ need not have the dimensions of momentum and do not form a vector in configuration space.

</div>

To derive the energy theorem, we employ a special variation: at each instant, let the virtual displacement coincide with the actual displacement occurring during an infinitesimal time $dt = \epsilon$. That is, we set $\delta q_i = dq_i = \epsilon\dot{q}_i$. This variation shifts the world-line horizontally in the $(q, t)$ diagram and does *not* keep the endpoints fixed; it therefore produces a boundary term.

The boundary term takes the form $[\sum p_i\,\delta q_i]_{t_1}^{t_2}$. For a **scleronomic** system (one whose Lagrangian does not depend on $t$ explicitly), the variation $\delta L$ equals $dL = \epsilon\dot{L}$, and equating the boundary terms from both sides yields

$$\sum_{i=1}^{n} p_i\dot{q}_i - L = \text{const.}$$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Total Energy / Hamiltonian)</span></p>

The quantity

$$\Lambda = \sum_{i=1}^{n} p_i\dot{q}_i - L$$

is called the **total energy** of the mechanical system. Together with $L$, it is the most important scalar associated with a mechanical system. In the Hamiltonian formulation of dynamics, $\Lambda$ (expressed in the proper variables) becomes the **Hamiltonian function** $H$ and completely replaces the Lagrangian.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Energy Theorem --- General Form)</span></p>

For any Lagrangian system, the quantity $\sum p_i\dot{q}_i - L$ satisfies

$$\sum_{i=1}^{n} p_i\dot{q}_i - L = \text{const.}$$

whenever $L$ does not depend explicitly on $t$ (scleronomic system). This is the most general form of the energy conservation law in Lagrangian mechanics.

</div>

In the usual mechanical applications, the kinetic energy is a quadratic form in the velocities,

$$T = \tfrac{1}{2}\sum_{i,k} a_{ik}\,\dot{q}_i\,\dot{q}_k,$$

and the potential energy $V$ does not contain the velocities. Under these conditions, Euler's theorem on homogeneous functions gives $\sum p_i\dot{q}_i = 2T$, and the energy theorem assumes its familiar form:

$$T + V = E = \text{const.}$$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Conditions for $T + V = E$)</span></p>

The simple form $T + V = \text{const.}$ requires *two* conditions beyond what the general energy theorem demands: (1) the system must be scleronomic, and (2) the kinetic energy must be a homogeneous quadratic form in the velocities while the potential energy is velocity-independent. Systems with "gyroscopic terms" (kinetic energy terms linear in the velocities) or relativistic kinetic energies satisfy the general conservation law $\sum p_i\dot{q}_i - L = \text{const.}$ but not necessarily $T + V = \text{const.}$

</div>

For systems with one degree of freedom, the energy theorem provides a complete first integral and reduces the problem to a single quadrature:

$$f(q, \dot{q}) = E \quad\Longrightarrow\quad \dot{q} = \varphi(E, q) \quad\Longrightarrow\quad t = \int\frac{dq}{\varphi(E, q)} + \tau,$$

with two constants of integration $E$ and $\tau$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Rheonomic Systems)</span></p>

When the Lagrangian depends explicitly on time --- i.e. the system is **rheonomic** --- the total energy $\Lambda = \sum p_i\dot{q}_i - L$ is no longer conserved. Instead it changes according to

$$\Delta\Lambda = -\int_{t_1}^{t_2}\frac{\partial L}{\partial t}\,dt.$$

The rate of change of the total energy equals minus the explicit time derivative of $L$: the energy leaks in or out through the time-dependent part of the Lagrangian.

</div>

### 4. Kinosthenic (Ignorable) Variables and Their Elimination

A general method for integrating the Lagrangian equations does not exist, but under special circumstances a *partial* integration is possible. A particularly important case arises when one or more generalized coordinates are **ignorable** (also called **kinosthenic** or **cyclic**).

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Ignorable Variable)</span></p>

A coordinate $q_k$ is called **ignorable** (or kinosthenic, cyclic) if it does not appear explicitly in the Lagrangian function, even though its velocity $\dot{q}_k$ does:

$$\frac{\partial L}{\partial q_k} = 0.$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Conservation of Kinosthenic Momentum)</span></p>

If $q_k$ is an ignorable coordinate, the corresponding generalized momentum is a constant of the motion:

$$p_k = \frac{\partial L}{\partial \dot{q}_k} = c_k = \text{const.}$$

This follows immediately from the Lagrangian equation $\dot{p}_k = \partial L / \partial q_k = 0$.

</div>

This result has many physical applications. For instance, in the Kepler problem formulated in polar coordinates, the angle $\theta$ is ignorable. The constancy of the corresponding momentum $p_\theta$ is precisely Kepler's law of areas (conservation of angular momentum). Similarly, for the spherical pendulum, the azimuthal angle $\varphi$ is ignorable and its conjugate momentum is conserved.

The power of ignorable variables lies in the possibility of *eliminating* them, thereby reducing the number of degrees of freedom. The elimination proceeds in three steps:

1. Write down the equation for the kinosthenic momentum: $\partial L / \partial \dot{q}_n = c_n$.
2. Form the **modified Lagrangian function**: $\bar{L} = L - c_n\dot{q}_n$.
3. Solve step 1 for the ignorable velocity $\dot{q}_n$ and substitute into $\bar{L}$, eliminating $\dot{q}_n$ entirely.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Modified Lagrangian / Routhian)</span></p>

The **modified Lagrangian function** (also called the **Routhian**) is obtained by subtracting the products of the ignorable momenta and their velocities from the original Lagrangian:

$$\bar{L} = L - \sum_k c_k\dot{q}_k,$$

where the sum extends over all ignorable coordinates. After eliminating the ignorable velocities, $\bar{L}$ depends only on the non-ignorable coordinates and velocities, and the original variational problem is reduced by as many degrees of freedom as there are ignorable variables.

</div>

The elimination has interesting physical consequences for the structure of the reduced problem. When the kinetic energy is a quadratic form $T = \frac{1}{2}\sum a_{ik}\dot{q}_i\dot{q}_k$, separating out the ignorable velocity produces three types of terms:

- A velocity-independent term $-\frac{1}{2}c_n^2/a_{nn}$ that behaves like an **apparent potential energy** and can be combined with $V$ to give a modified potential $\bar{V} = V + \frac{1}{2}c_n^2/a_{nn}$.
- Terms *linear* in the non-ignorable velocities, called **gyroscopic terms**, which arise whenever there is "kinetic coupling" between the ignorable and non-ignorable velocities (i.e. the cross-coefficients $a_{in}$ with $i \neq n$ do not all vanish).
- The usual quadratic kinetic energy in the remaining velocities.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Gyroscopic Terms and Kinetic Coupling)</span></p>

Whether the reduced problem contains gyroscopic terms depends on the $a_{in}$ coefficients coupling the ignorable and non-ignorable velocities. If all such cross-coefficients vanish, the reduced kinetic energy remains a purely quadratic form and there are no gyroscopic effects. When the coupling is present, the gyroscopic terms --- being linear rather than quadratic in the velocities --- produce forces perpendicular to the velocity that do no work but can profoundly affect the motion. The Coriolis force in a rotating frame and the magnetic force on a charged particle are physical manifestations of gyroscopic terms in the Lagrangian.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Kepler Problem --- Elimination of $\theta$)</span></p>

In the Kepler problem with polar coordinates $(r, \theta)$, the angle $\theta$ is ignorable. Eliminating it produces an apparent repulsive potential proportional to $1/r^2$ (the "centrifugal barrier"), which competes with the attractive $1/r$ gravitational potential. The balance between these two forces creates a stable equilibrium radius, and the oscillation of $r$ about this point explains the pulsation of the orbit between perihelion and aphelion. If the attractive force fell off faster than $1/r^2$, no such stable equilibrium would exist and closed orbits would be impossible.

</div>

### 5. The Forceless Mechanics of Hertz

The properties of ignorable variables inspired Heinrich Hertz to develop an ingenious theory aimed at explaining the deeper significance of potential energy. A mechanical system is described by a definite number of position coordinates, but not all of these need be directly observable. A system may contain "microscopic parameters" --- hidden degrees of freedom that are not experimentally accessible. Because of these hidden coordinates, the effective number of degrees of freedom may appear much smaller than it truly is.

The microscopic motions inside such a system fall into two categories:

- **Non-kinosthenic hidden motions** produce apparently *polygenic* forces in the macroscopic description. Friction is a prime example: macroscopically it appears as a non-conservative, dissipative force, but microscopically it is merely a substitute for the unobservable motion of molecules.
- **Kinosthenic hidden motions** can be eliminated by the procedure described above. After elimination they contribute an apparent potential energy and possibly gyroscopic terms to the reduced Lagrangian, but no polygenic forces.

Hertz's bold hypothesis was that *all* forces in nature might ultimately arise from hidden kinosthenic motions. If this were true, the fundamental Lagrangian of the universe would contain kinetic energy alone --- no potential energy at all. All apparent forces and potentials would be artefacts of eliminating the unseen cyclic coordinates. This vision of a "forceless mechanics" --- in which geometry and inertia alone govern all phenomena --- was ahead of its time and anticipates aspects of general relativity, where gravitation is reinterpreted as the curvature of spacetime rather than a force in the Newtonian sense.

### 6. The Time as a Kinosthenic Variable --- Jacobi's Principle

In Hamilton's principle the time $t$ plays the role of an independent variable: we integrate the Lagrangian between fixed limits $t_1$ and $t_2$, and the time is *not* varied. However, when we pass to the extended configuration space (where $t$ is treated as just another coordinate alongside the $q_i$), a deeper viewpoint emerges. We can parametrise the world-line by an arbitrary parameter $\tau$ and let $t$ itself become a dependent variable on the same footing as the $q_i$.

If the system is **scleronomic** (the Lagrangian does not depend on $t$ explicitly) and **conservative** (the energy theorem $T + V = E$ holds), a remarkable simplification occurs. The time can be treated as a kinosthenic variable --- it does not appear explicitly in the parametrised action, though its "velocity" $t' = dt/d\tau$ does. We can therefore eliminate it using exactly the same procedure developed in Section 4 for ignorable coordinates.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Jacobi's Action)</span></p>

After eliminating the time from Hamilton's action integral for a conservative scleronomic system with total energy $E$, the resulting **Jacobi action** (also called the "abbreviated action") is

$$A^* = \int_{P_1}^{P_2} \sqrt{2(E - V)}\;ds,$$

where $ds^2 = \sum a_{ik}\,dq_i\,dq_k$ is the Riemannian line element of the configuration space derived from the kinetic energy, and the integral is taken between two fixed points in configuration space without any reference to the time.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Jacobi's Principle)</span></p>

For a conservative scleronomic system with prescribed total energy $E$, the actual path of the representative point in configuration space is the one that makes Jacobi's action stationary:

$$\delta\int_{P_1}^{P_2}\sqrt{2(E-V)}\;ds = 0.$$

This determines the *geometry* of the trajectory (the shape of the path) but not the time-law of the motion along it. The time can be recovered afterwards from the energy relation.

</div>

Jacobi's principle is equivalent to Hamilton's principle for conservative scleronomic systems, but it shifts the focus from the trajectory-in-time to the trajectory-in-space. The condition $\delta A^* = 0$ can be interpreted as a geodesic problem in a Riemannian space whose metric is $d\sigma^2 = 2(E - V)\,ds^2$. In other words, the mechanical system moves along the shortest (or more generally, the extremal) path with respect to this modified metric.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Geodesic Interpretation and Free Motion)</span></p>

When no impressed forces act ($V = 0$ or $V = \text{const.}$), Jacobi's principle reduces to requiring that the path be a geodesic of the original Riemannian metric $ds^2 = \sum a_{ik}\,dq_i\,dq_k$. Since the energy theorem further implies that the speed $ds/dt$ is constant, the representative point traverses this geodesic with uniform velocity. This is a far-reaching generalisation of Newton's first law: a free particle moves in a straight line with constant speed, but here the "straight line" is a geodesic in an $n$-dimensional curved Riemannian manifold, and the "particle" represents an entire mechanical system subject to holonomic constraints.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Particle on a Curved Surface)</span></p>

Consider a particle constrained to move on a curved surface with no impressed forces. The Riemannian space is two-dimensional with the intrinsic geometry of the surface. The particle traces out a geodesic of that surface --- the shortest path between two points. This is exactly the behaviour one observes: a marble rolling on a frictionless bowl follows a geodesic arc.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Connection to General Relativity)</span></p>

Jacobi's geodesic principle bears a striking resemblance to Einstein's description of gravity. In general relativity the motion of a planet is governed by the geodesics of a four-dimensional Riemannian (pseudo-Riemannian) spacetime. No gravitational force needs to be postulated; the geometry of spacetime alone dictates the trajectory. The main conceptual difference is that in Jacobi's mechanics the Riemannian structure arises from the kinematical constraints, whereas in relativity the curved geometry is an intrinsic property of the universe itself.

</div>

### 7. Jacobi's Principle and Riemannian Geometry

Jacobi's principle opens the door to a fully geometric description of classical mechanics. The key idea is that the configuration space of a mechanical system, equipped with the metric $ds^2 = \sum a_{ik}\,dq_i\,dq_k$ derived from the kinetic energy, is naturally a Riemannian manifold. The $a_{ik}$ coefficients play the role of the metric tensor, and all the tools of Riemannian geometry --- geodesics, curvature, parallel transport --- become applicable to mechanical problems.

For a conservative system with potential energy $V$, Jacobi's principle tells us to work with the conformally modified metric

$$d\sigma^2 = 2(E - V)\sum a_{ik}\,dq_i\,dq_k.$$

The trajectories of the mechanical system are then geodesics of this modified Riemannian space. The curvature of the modified space encodes the dynamical effects of the potential: regions where $V$ is large (leaving little kinetic energy) correspond to regions where the effective geometry is "compressed," causing the geodesics to curve away.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Geometry Replaces Force)</span></p>

The geometric viewpoint unifies two seemingly separate aspects of a mechanical problem --- the kinematical constraints (which determine $a_{ik}$) and the dynamical forces (which determine $V$) --- into a single Riemannian metric. Rather than solving differential equations driven by forces, we can equivalently find the shortest paths in a curved space. This is not merely an analogy; it is a mathematically exact reformulation that is sometimes more powerful than the direct approach, especially for qualitative analysis of trajectories.

</div>

### 8. Auxiliary Conditions and the Physical Significance of the Lagrangian $\lambda$-Factor

In the general treatment of constrained systems (cf. Chapter II), we saw that holonomic kinematical constraints of the form

$$f_i(q_1, \ldots, q_n, t) = 0, \qquad i = 1, \ldots, m,$$

can be handled in two ways. Either we eliminate $m$ of the coordinates using the constraint equations (reducing the system to $n - m$ free degrees of freedom), or we keep all $n$ coordinates and adjoin the constraints via Lagrange multipliers. In the latter approach we form the modified Lagrangian

$$\bar{L} = L - (\lambda_1 f_1 + \cdots + \lambda_m f_m)$$

and treat the problem as though unconstrained. The Lagrangian equations for $\bar{L}$ then automatically produce the correct equations of motion *including* the forces of reaction that maintain the constraints.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Lagrange Multipliers and Forces of Reaction)</span></p>

When the $\lambda$-method is applied to the variational problem, the forces maintaining the holonomic constraints are given by

$$K_i = -\left(\lambda_1\frac{\partial f_1}{\partial x_i} + \cdots + \lambda_m\frac{\partial f_m}{\partial x_i}\right).$$

This recovers exactly the same expressions encountered in the statics of constrained systems (Chapter III). The $\lambda$-method thus provides the forces of reaction for dynamical problems as well.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Physical Interpretation of $\lambda$)</span></p>

An important subtlety arises when the constraint equations and the potential energy are both time-independent (scleronomic conservative system). By the elimination method, the system is conservative and the constraint forces must likewise be conservative. Yet the Lagrange multipliers $\lambda_i$ typically depend on time, making the apparent potential energy of the constraint forces time-dependent. This paradox is resolved by recognising that we need the potential energy of the constraint forces only *along the actual trajectory* (the $C$-curve), not throughout all of configuration space.

A careful analysis shows that $\lambda$ is proportional to the "microscopic violation" of the constraint: the constraint $f = 0$ is maintained by very strong restoring forces, and $\lambda$ measures the tiny departure of the system from the constraint surface at each instant. In the idealised limit of infinitely strong restoring forces (perfect constraints), $\lambda$ becomes the ratio $f/\epsilon$ where $\epsilon \to 0$, giving the constraint force its correct finite value.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Holonomic Constraints as Conservative Forces)</span></p>

The analysis confirms an important general principle: **scleronomic holonomic auxiliary conditions are mechanically equivalent to conservative monogenic forces.** The $\lambda$-method reveals these forces explicitly without needing to eliminate any coordinates. This equivalence between constraints and forces is one of the most powerful results in analytical mechanics.

</div>

### 9. Non-Holonomic Auxiliary Conditions and Polygenic Forces

When the kinematical constraints are **non-holonomic** --- that is, they take the form of non-integrable differential relations rather than algebraic equations between the coordinates --- the situation changes fundamentally. We can no longer eliminate surplus variables by substitution, because the constraint cannot be integrated to give a relation among the $q_i$ alone.

The $\lambda$-method, however, continues to apply. If the non-holonomic constraints are

$$\sum_k A_{ik}\,dq_k = 0, \qquad i = 1, \ldots, m,$$

the Lagrangian equations become

$$\frac{d}{dt}\frac{\partial L}{\partial \dot{q}_k} - \frac{\partial L}{\partial q_k} = -(\lambda_1 A_{1k} + \lambda_2 A_{2k} + \cdots + \lambda_m A_{mk}).$$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Non-Holonomic Constraints as Polygenic Forces)</span></p>

A crucial difference from the holonomic case: the forces maintaining non-holonomic constraints are **polygenic** --- they cannot be derived from a single work function. The virtual work of these forces takes the form $\delta w = \sum \rho_i\,\delta q_i$ where the $\rho_i$ are not expressible as partial derivatives of a potential. Thus non-holonomic constraints are mechanically equivalent to polygenic forces. Both produce the same effect in the Lagrangian equations: an extra "right-hand side" that cannot be absorbed into $L$.

</div>

More generally, if arbitrary polygenic forces with virtual work $\overline{\delta w} = \rho_1\,\delta q_1 + \cdots + \rho_n\,\delta q_n$ act alongside the monogenic forces already contained in $L$, the Lagrangian equations generalise to

$$\frac{d}{dt}\frac{\partial L}{\partial \dot{q}_i} - \frac{\partial L}{\partial q_i} = \rho_i.$$

The monogenic part is handled by $L$ as usual; the polygenic part appears as a forcing term on the right.

### 10. Small Vibrations About a State of Equilibrium

One of the most impressive applications of the Lagrangian framework is the theory of small oscillations around a stable equilibrium. This theory underlies the analysis of elastic vibrations, acoustics, molecular vibrations, and many other physical phenomena. Its power lies in its **generality**: regardless of how complicated a system may be, the motion near equilibrium is always governed by the same mathematical structure.

#### Linearisation Near Equilibrium

The configuration space of a mechanical system is a Riemannian manifold with metric $ds^2 = \sum a_{ik}\,dq_i\,dq_k$ determined by the kinetic energy. In general this space is curved, but near a given point $P$ (the equilibrium configuration) it can be approximated by a flat (Euclidean) tangent space. We place the origin of our coordinate system at $P$ so that $q_i = 0$ at equilibrium.

Since $P$ is an equilibrium point, the potential energy $V$ has a stationary value there; the linear terms in the Taylor expansion vanish. Dropping the irrelevant constant, the potential energy to leading order is a quadratic form:

$$V \approx \frac{1}{2}\sum_{i,k=1}^{n} b_{ik}\,q_i\,q_k, \qquad b_{ik} = \left(\frac{\partial^2 V}{\partial q_i\,\partial q_k}\right)_0.$$

Similarly, the kinetic energy with the $a_{ik}$ frozen at their equilibrium values is

$$T \approx \frac{1}{2}\sum_{i,k=1}^{n}a_{ik}\,\dot{q}_i\,\dot{q}_k.$$

Both $T$ and $V$ are now homogeneous quadratic forms: $T$ in the velocities, $V$ in the coordinates.

#### The Principal Axis Problem

The problem of small vibrations reduces to a classical algebraic question: **simultaneously diagonalise two quadratic forms.** Geometrically, the level surface $V = \frac{1}{2}$ defines a second-order surface (an ellipsoid, hyperboloid, etc.) in the $n$-dimensional configuration space, and we seek the *principal axes* of this surface relative to the metric defined by $T$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Characteristic Equation)</span></p>

The **characteristic equation** (or secular equation) of the small-vibration problem is obtained by requiring the homogeneous linear system

$$\sum_{k=1}^{n} b_{ik}\,q_k = \lambda\sum_{k=1}^{n} a_{ik}\,q_k, \qquad i = 1, \ldots, n,$$

to have a non-trivial solution. This requires the determinant

$$\det(b_{ik} - \lambda\,a_{ik}) = 0.$$

This is a polynomial equation of degree $n$ in $\lambda$, whose roots $\lambda_1, \ldots, \lambda_n$ are called the **characteristic values** (eigenvalues).

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Properties of the Characteristic Values)</span></p>

For the small-vibration problem with symmetric $a_{ik}$ and $b_{ik}$ (and $a_{ik}$ positive definite):

1. **All eigenvalues $\lambda_i$ are real.** This follows from the symmetry of both matrices and the positive definiteness of $a_{ik}$.
2. **The principal axes $\mathbf{p}_i$ are mutually orthogonal** with respect to the metric $a_{ik}$, i.e. $\sum a_{ik}\,p^{(j)}_i\,p^{(l)}_k = 0$ for $j \neq l$.
3. **The eigenvalues are invariant** under linear coordinate transformations.
4. A single linear transformation of coordinates brings **both** $T$ and $V$ into diagonal form simultaneously.

</div>

#### Normal Coordinates and Normal Modes

By transforming to the principal axes, we obtain new coordinates $u_1, \ldots, u_n$ (called **normal coordinates**) in which the kinetic and potential energies take the simultaneously diagonal forms:

$$T = \frac{1}{2}(\dot{u}_1^2 + \cdots + \dot{u}_n^2), \qquad V = \frac{1}{2}(\lambda_1 u_1^2 + \cdots + \lambda_n u_n^2).$$

The Lagrangian equations in normal coordinates are completely **separated** --- each equation involves only a single coordinate:

$$\ddot{u}_i + \lambda_i\,u_i = 0, \qquad i = 1, \ldots, n.$$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Normal Modes and Natural Frequencies)</span></p>

Each separated equation $\ddot{u}_i + \lambda_i u_i = 0$ describes a **normal mode** of vibration. If $\lambda_i > 0$, the solution is simple harmonic motion

$$u_i = A_i\cos(\nu_i t) + B_i\sin(\nu_i t)$$

with **natural frequency** $\nu_i = \sqrt{\lambda_i}$. The general motion near the equilibrium is a superposition of all $n$ normal modes. Each mode vibrates independently along its own principal axis with its own characteristic frequency.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Double Pendulum)</span></p>

Consider two identical simple pendulums of the same length and mass, where the second is suspended from the bob of the first. Using the small-angle displacements $\theta_1, \theta_2$ as coordinates, the kinetic and potential energies are quadratic forms that can be diagonalised. The two oblique axes of the original coordinate system are at $45°$ to the principal axes. The two normal frequencies turn out to be $\nu_1 = (\sqrt{2} - \sqrt[4]{2})\,\nu$ and $\nu_2 = (\sqrt{2} + \sqrt[4]{2})\,\nu$, where $\nu$ is the natural frequency of a single pendulum. The two normal modes correspond to symmetric and antisymmetric oscillation patterns.

</div>

#### Stability and Instability

The sign of the characteristic values $\lambda_i$ determines the qualitative behaviour of the equilibrium:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Stability Criterion)</span></p>

- **Stable equilibrium:** All $\lambda_i > 0$. Every normal mode oscillates harmonically. The $C$-point remains near $P$ for all time. This occurs precisely when $V$ has a **local minimum** at $P$ --- the potential energy surface is bowl-shaped in every direction.
- **Unstable equilibrium:** At least one $\lambda_i < 0$. The corresponding mode grows exponentially as $u_i = A_i e^{\nu_i t} + B_i e^{-\nu_i t}$ with $\nu_i = \sqrt{-\lambda_i}$. The slightest perturbation drives the system away from $P$.
- **Neutral equilibrium:** No $\lambda_i$ is negative, but one or more are zero. The system neither oscillates nor diverges in those directions.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Stability, the Second Variation, and Buckling)</span></p>

The stability analysis connects directly with the second variation of the potential energy $\delta^2 V$. The characteristic values $\lambda_i$ are the eigenvalues of $\delta^2 V$ relative to the kinetic-energy metric, and their signs determine whether the second variation is positive definite (stable), indefinite (unstable), or semi-definite (neutral). This perspective is of central importance in the theory of elastic stability and **buckling**: a thin structure under load is stable as long as the smallest $\lambda_i$ remains positive. The critical load at which the structure buckles is precisely the load at which the smallest eigenvalue passes through zero. Beyond this point the equilibrium becomes unstable and the structure collapses --- not from a breakdown of the elastic forces, but because the equilibrium itself ceases to be a minimum of the potential energy.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Thermal Vibrations and the Spectrum of Normal Modes)</span></p>

The theory of normal modes has profound applications in statistical mechanics. A solid body consisting of $N$ molecules can be modelled as a $C$-point in a $3N$-dimensional configuration space. Near thermal equilibrium the motion is a superposition of $3N$ normal vibrations along mutually perpendicular principal axes, each with its own characteristic frequency. The resulting "spectrum" of frequencies ranges from very low (elastic waves) to very high (infrared), and the distribution of energy among these modes is governed by the laws of statistical mechanics as a function of the temperature $T$.

</div>

## Chapter VI: The Canonical Equations of Motion

The principle of least action and Hamilton's generalization transfer the problems of mechanics into the realm of the calculus of variations. The Lagrangian equations, which follow from the stationarity of a definite integral, are the basic differential equations of theoretical mechanics. However, the Lagrangian function is quadratic in the velocities, and Hamilton discovered a remarkable transformation that makes it linear in the velocities --- at the cost of doubling the number of mechanical variables. This transformation reduces all Lagrangian problems to a specially simple form called the "canonical" form by Jacobi. The resulting $2n$ first-order differential equations --- replacing the original $n$ second-order ones --- possess a beautiful symmetry that opened a new era in theoretical mechanics.

### 1. Legendre's Dual Transformation

The French mathematician Legendre (1752--1833) discovered an important transformation in his studies on differential equations. Before applying it to the Lagrangian equations, we discuss its general mathematical properties.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Legendre Transformation)</span></p>

Given a function $F = F(u_1, \ldots, u_n)$, introduce new variables $v_i$ via

$$v_i = \frac{\partial F}{\partial u_i}.$$

Assuming the Hessian (the determinant of the matrix of second partial derivatives of $F$) is non-zero, the $u_i$ can be solved as functions of the $v_i$. Define a new function

$$G = \sum_{i=1}^{n} u_i v_i - F.$$

Expressing the $u_i$ in terms of the $v_i$ yields $G = G(v_1, \ldots, v_n)$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Duality of the Legendre Transformation)</span></p>

The Legendre transformation is entirely symmetrical: just as the new variables are the partial derivatives of the old function with respect to the old variables, so the old variables are the partial derivatives of the new function with respect to the new variables:

$$u_i = \frac{\partial G}{\partial v_i}.$$

The "old" and "new" systems are completely equivalent --- applying the same procedure to $G$ recovers $F$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Active and Passive Variables)</span></p>

The transformation can be generalised to the case where $F$ depends on two sets of variables: the $u_i$ (called **active** variables, which participate in the transformation) and additional $w_i$ (called **passive** variables, which appear as mere parameters). For the passive variables, an additional relation holds:

$$\frac{\partial F}{\partial w_i} = -\frac{\partial G}{\partial w_i}.$$

The partial derivative of $F$ with respect to any passive variable equals the negative of the corresponding partial derivative of $G$.

</div>

### 2. Legendre's Transformation Applied to the Lagrangian Function

The Lagrangian function $L$ of a variational problem depends in general on $n$ position coordinates $q_i$, the $n$ velocities $\dot{q}_i$, and the time $t$. We now apply Legendre's transformation, treating the velocities $\dot{q}_i$ as the active variables and the position coordinates $q_i$ together with $t$ as passive variables.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Canonical Momenta and the Hamiltonian)</span></p>

The transformation proceeds in three steps:

1. **Introduce the momenta** $p_i$ (the new variables):

$$p_i = \frac{\partial L}{\partial \dot{q}_i}.$$

2. **Define the Hamiltonian function** $H$ (the "total energy"):

$$H = \sum_{i=1}^{n} p_i \dot{q}_i - L.$$

3. **Express $H$ in terms of the new variables** by solving the defining equations for the $\dot{q}_i$ and substituting:

$$H = H(q_1, \ldots, q_n;\; p_1, \ldots, p_n;\; t).$$

</div>

The dual nature of the transformation is expressed in the following scheme:

| | Old system | New system |
|---|---|---|
| **Function** | Lagrangian $L$ | Hamiltonian $H$ |
| **Active variables** | velocities $\dot{q}_i$ | momenta $p_i$ |
| **Passive variables** | positions $q_i$, time $t$ | positions $q_i$, time $t$ |

The duality relations give:

$$p_i = \frac{\partial L}{\partial \dot{q}_i}, \qquad \dot{q}_i = \frac{\partial H}{\partial p_i},$$

and for the passive variables:

$$\frac{\partial L}{\partial q_i} = -\frac{\partial H}{\partial q_i}, \qquad \frac{\partial L}{\partial t} = -\frac{\partial H}{\partial t}.$$

### 3. Transformation of the Lagrangian Equations of Motion

The Lagrangian equations of motion are differential equations of the second order in the position coordinates $q_i$. By introducing the momenta $p_i$, they can be rewritten as a system of first-order equations.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Conjugate Momenta)</span></p>

The momenta defined by

$$p_i = \frac{\partial L}{\partial \dot{q}_i}$$

are called **conjugate momenta**. Each position coordinate $q_k$ is paired with its own conjugate momentum $p_k$, forming the **conjugate pairs** $(q_k, p_k)$.

</div>

Defining the momenta has the effect of replacing the original system of $n$ second-order differential equations by a system of $2n$ first-order equations. The Lagrangian equations $\dot{p}_i = \frac{\partial L}{\partial q_i}$ can be rewritten, via Legendre's transformation, as:

$$\dot{p}_i = -\frac{\partial H}{\partial q_i}.$$

Combining this with the duality relation $\dot{q}_i = \frac{\partial H}{\partial p_i}$, we arrive at Hamilton's canonical equations.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Hamilton's Canonical Equations)</span></p>

The Lagrangian equations of motion are entirely equivalent to the system

$$\dot{q}_i = \frac{\partial H}{\partial p_i}, \qquad \dot{p}_i = -\frac{\partial H}{\partial q_i}, \qquad i = 1, \ldots, n.$$

These $2n$ first-order differential equations are called **Hamilton's canonical equations**. Time derivatives appear only on the left-hand sides, and the right-hand sides are all derivable from the single function $H$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Advantages of the Canonical Form)</span></p>

Although the canonical equations are mathematically equivalent to the Lagrangian equations, the new form is vastly superior. Differentiation with respect to time is cleanly separated from the algebraic operations: the Hamiltonian function does not contain any derivatives of $q_i$ or $p_i$ with respect to $t$. Furthermore, the entire set of $2n$ equations is determined by a single scalar function $H$.

</div>

### 4. The Canonical Integral

The canonical equations have a double origin. The first set $\dot{q}_i = \frac{\partial H}{\partial p_i}$ follows from the definition of the momenta via Legendre's transformation. The second set $\dot{p}_i = -\frac{\partial H}{\partial q_i}$ is a consequence of the variational principle. Yet the symmetry of the complete system suggests that both sets should be derivable from a single principle --- and this is indeed the case.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The Canonical Integral)</span></p>

Starting from the Hamiltonian function $H$, reconstruct the Lagrangian via $L = \sum p_i \dot{q}_i - H$. The action integral becomes

$$A = \int_{t_1}^{t_2} \left[\sum_{i=1}^{n} p_i \dot{q}_i - H(q_1, \ldots, q_n;\; p_1, \ldots, p_n;\; t)\right] dt.$$

This is called the **canonical integral**. Its distinguishing feature is that the $q_i$ and $p_i$ are treated as $2n$ independent variables, all varied freely. The Euler--Lagrange equations applied to this integral reproduce exactly Hamilton's canonical equations.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Structure of the Canonical Integral)</span></p>

An arbitrary variation of the $p_i$ has no influence on the variation of $L$, because the coefficient of $\delta p_i$ vanishes identically by Legendre's transformation. This means the variational principle remains valid even when the $p_i$ are considered as a second independent set of variables. The kinetic part of the canonical integral is now a simple linear function of the velocities $\dot{q}_i$, namely $\sum p_i \dot{q}_i$, rather than a quadratic form. This remarkable simplification is the hallmark of the Hamiltonian formulation.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Conjugate Pairs and Complex Canonical Equations)</span></p>

Each $q_k$ is associated with its own $p_k$, and the canonical variables can be arranged in pairs $(q_k, p_k)$. A peculiar feature of the canonical equations is their appearance in pairs. If the conjugate variables $q_k$, $p_k$ are replaced by the complex variables

$$u_k = \frac{q_k + ip_k}{\sqrt{2}}, \qquad u_k^* = \frac{q_k - ip_k}{\sqrt{2}},$$

then the double set of canonical equations can be replaced by a single set of complex equations:

$$\frac{du_k}{i\,dt} = -\frac{\partial H}{\partial u_k^*}.$$

Because of the duality of Legendre's transformation, every Hamiltonian problem can be associated with a corresponding Lagrangian problem (and vice versa), by expressing the $p_i$ in terms of $q_i$ and $\dot{q}_i$.

</div>

### 5. The Phase Space and the Phase Fluid

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Phase Space and State Space)</span></p>

The variables of the canonical integral are the $q_i$ and $p_i$, giving $2n$ degrees of freedom. The $2n$-dimensional space whose coordinates are the position coordinates $q_1, \ldots, q_n$ and the momenta $p_1, \ldots, p_n$ is called the **phase space** (a term introduced by Gibbs). A single point $C$ in the phase space represents the complete instantaneous state of the mechanical system.

If the time $t$ is included as an additional variable, we obtain a $(2n+1)$-dimensional space called the **state space** (Cartan's *espace des états*). In the state space the complete solution of the canonical equations is pictured as an infinite manifold of non-crossing curves.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Phase Space vs. Configuration Space)</span></p>

In Lagrangian mechanics the "configuration space" includes only the $n$ position coordinates $q_i$. In configuration space the totality of paths forms a hopeless criss-cross because motion can start from every point in every direction with arbitrary initial velocity. In the phase space, by contrast, the canonical equations are of the first order: specifying the position $C$ of the system (i.e. all $q_i$ and $p_i$) uniquely determines its subsequent motion. From any point of the phase space, only one trajectory passes, and the totality of motions can be pictured in a well-ordered way without any overlapping.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Phase Fluid)</span></p>

The motion in phase space can be pictured as the flow of a $2n$-dimensional **phase fluid**. Each individual streamline of the moving fluid represents the motion of the mechanical system under specified initial conditions; the fluid as a whole represents the complete solution for arbitrary initial conditions. The fluid can be described either by a **particle description** (following individual fluid particles from their initial positions) or by a **field description** (specifying the velocity field at every point of phase space at every time).

</div>

### 6. The Energy Theorem as a Consequence of the Canonical Equations

The canonical equations acquire a new significance when interpreted hydrodynamically, in connection with the motion of the phase fluid. Certain types of fluid motion that are of special interest in ordinary hydrodynamics are also of interest in the motion of the phase fluid.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Steady Phase Flow for Conservative Systems)</span></p>

If the Lagrangian function $L$ does not depend on $t$ (a **conservative** or **scleronomic** system), then the Hamiltonian function $H$ is likewise independent of $t$. In this case the right-hand sides of the canonical equations are independent of the time, meaning that the velocity field at a definite point of phase space is constant. The phase fluid associated with a conservative system is therefore in a state of **steady motion**.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Energy Conservation from the Canonical Equations)</span></p>

For a conservative system with $H = H(q_1, \ldots, q_n;\; p_1, \ldots, p_n)$, differentiation gives

$$\frac{dH}{dt} = \sum_{i=1}^{n}\left(\frac{\partial H}{\partial q_i}\dot{q}_i + \frac{\partial H}{\partial p_i}\dot{p}_i\right).$$

Substituting the canonical equations, every term cancels, yielding $\frac{dH}{dt} = 0$, i.e.

$$H = \text{const.} = E.$$

This is the **energy theorem**: the Hamiltonian is a conserved quantity (the total energy). When $L = T - V$ with $T$ quadratic in the velocities and $V$ independent of the velocities, the Hamiltonian equals $H = T + V$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Energy Surfaces in Phase Space)</span></p>

The equation $H(q_1, \ldots, q_n;\; p_1, \ldots, p_n) = E$ defines a surface in phase space. As $E$ takes arbitrary constant values, we obtain an infinite family of such surfaces. The energy theorem has the geometric interpretation that a fluid particle which starts its motion on a definite energy surface remains constantly on that surface, no matter how long we follow its motion.

</div>

### 7. Liouville's Theorem

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Liouville's Theorem)</span></p>

The phase fluid associated with the canonical equations behaves like an **incompressible fluid**. Analytically, the divergence of the phase-space velocity field vanishes identically:

$$\operatorname{div}\mathbf{v} \equiv \sum_{i=1}^{n}\left(\frac{\partial \dot{q}_i}{\partial q_i} + \frac{\partial \dot{p}_i}{\partial p_i}\right) = 0.$$

This relation holds not only for conservative systems but for any system whose motion is governed by canonical equations. By Green's theorem, the total flux of the phase fluid through any closed surface in phase space is always zero --- the phase fluid moves like an incompressible fluid.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Conservation of Phase-Space Volume)</span></p>

Liouville's theorem adds a powerful conservation law. We may cut out an arbitrary $2n$-dimensional region of phase space with volume $\sigma = \int dq_1 \cdots dq_n\, dp_1 \cdots dp_n$ and follow it as the phase fluid carries it along. Although the region may become distorted, its volume remains unchanged during the motion:

$$\sigma = \text{const.}$$

This theorem was first discovered by Liouville in 1838 and is fundamental to statistical mechanics.

</div>

### 8. Integral Invariants and Helmholtz's Circulation Theorem

Poincaré called any integral associated with the phase fluid that remains unchanged during the motion an "integral invariant." The phase-space volume $\sigma$ (Liouville's theorem) is one example. Another important integral invariant is the **circulation**, introduced by Helmholtz.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Circulation in Phase Space)</span></p>

Let $\mathcal{L}$ be a closed curve in phase space, carried along by the phase fluid as a "material line." The **circulation** is the line integral

$$\Gamma = \oint \sum_{i=1}^{n} p_i\, dq_i.$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Helmholtz's Circulation Theorem for Phase Space)</span></p>

The circulation $\Gamma = \oint \sum p_i\, dq_i$ is an invariant of the motion:

$$\Gamma = \text{const.}$$

This follows from the fact that the variation of the action integral for arbitrary variations of the $q_i$ (and the $p_i$) gives $\delta A = \left[\sum p_i\,\delta q_i\right]_{t_1}^{t_2}$, and for a closed curve the end-point and initial point coincide, causing the integrated contribution to vanish at both times. Since $t_1$ and $t_2$ are arbitrary, the circulation is conserved.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Helmholtz's Theorem and Vortices)</span></p>

For a single particle in a potential field, the configuration space coincides with the physical space, and the momentum vector $\mathbf{P} = m\mathbf{v}$. The circulation becomes $\Gamma = m \oint \mathbf{v} \cdot d\mathbf{s}$, which is exactly the quantity studied in the circulation theorem of Helmholtz in classical hydrodynamics. The invariance of the circulation means that a fluid which is initially free from vortices remains so permanently --- vortices cannot be created or destroyed.

</div>

### 9. The Elimination of Ignorable Variables

Although the canonical equations have a simpler structure than the original Lagrangian equations, we do not possess any general method for their integration. Ignorable variables again play a particularly important role.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Ignorable Variables in the Hamiltonian Formulation)</span></p>

If the Lagrangian function $L$ does not contain a certain coordinate $q_n$, then the Hamiltonian function $H$ will not contain it either. The last of the canonical equations then gives

$$\dot{p}_n = 0, \qquad \Longrightarrow \qquad p_n = \text{const.} = C_n.$$

The constancy of the kinosthenic momentum is thus re-established. The process of reduction in the Hamiltonian form is much simpler than in the Lagrangian form: we drop the contribution of the ignorable variables from the kinetic part of the canonical integral ($\sum p_i \dot{q}_i$), while in the Hamiltonian function the ignorable momenta are replaced by their constant values $C_n$. The reduced action integral becomes

$$\bar{A} = \int_{t_1}^{t_2}\left(\sum_{i=1}^{n-1} p_i \dot{q}_i - H\right) dt.$$

After solving the reduced problem, the ignorable variables are recovered by direct integration of $\dot{q}_m = \frac{\partial H}{\partial p_m}$.

</div>

### 10. The Parametric Form of the Canonical Equations

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Time as a Mechanical Variable)</span></p>

The state space has an odd number of dimensions: all position coordinates $q_i$ are paired with their momenta $p_i$, but the time $t$ stands alone as the independent variable with no associated momentum. Great advantage comes from promoting $t$ to a mechanical variable by setting $t = q_{n+1}$ and introducing an unspecified parameter $\tau$ as the new independent variable. The configuration space then has $n+1$ dimensions, and the phase space has $2n+2$ dimensions with $n+1$ conjugate pairs.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Extended Phase Space and Extended Hamiltonian)</span></p>

Setting $q_{n+1} = t$, the momentum conjugate to $t$ has the physical interpretation of the negative of the total energy: $p_{n+1} = -H$. The canonical integral in parametric form becomes

$$A = \int_{\tau_1}^{\tau_2} \sum_{i=1}^{n+1} p_i\, q_i'\, d\tau,$$

where primes denote differentiation with respect to $\tau$. This integral is to be made stationary under the auxiliary condition

$$K(q_1, \ldots, q_{n+1};\; p_1, \ldots, p_{n+1}) = 0,$$

where $K = p_{n+1} + H$ is the **extended Hamiltonian function**. The extended canonical equations are

$$q_k' = \frac{\partial K}{\partial p_k}, \qquad p_k' = -\frac{\partial K}{\partial q_k}, \qquad k = 1, \ldots, n+1.$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Universal Conservatism of the Extended System)</span></p>

In the extended phase space, the "extended Hamiltonian" $K$ does not depend on the parameter $\tau$, so every system --- whether originally conservative or not --- becomes conservative. The motion of the extended phase fluid is always steady, and every fluid particle remains permanently on the surface $K = \text{const.}$ The auxiliary condition $K = 0$ is permanently satisfied if the initial values of $q_i$ and $p_i$ are chosen consistently at $\tau_1$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Unification of the Minimum Principles)</span></p>

The parametric formulation gives a deeper insight into the mutual relations of the various minimum principles. If the canonical integral is normalised to the form $A = \int \sum p_i q_i'\, d\tau$ with auxiliary condition $K = 0$, then the three great variational principles --- Hamilton's principle, the principle of Euler--Lagrange, and Jacobi's principle --- correspond to different interpretations of this auxiliary condition. In the extended phase space the time $t$ plays simply the role of a parameter, and the particular properties of conservative systems are generalised to arbitrary systems. This parametric formulation can be considered the most advanced form of the canonical equations.

</div>

## Chapter VII: Canonical Transformations

Having brought the differential equations of motion into the canonical form, the ultimate goal is to *solve* these equations. No direct integration method is known in general. The most effective indirect approach is the method of **coordinate transformations**: we seek new coordinates in which the Hamiltonian function is so highly simplified that the equations of motion become directly integrable. The study of transformations which preserve the canonical form --- called **canonical transformations** --- is essentially due to Jacobi and constitutes one of the great achievements of analytical mechanics.

### 1. Coordinate Transformations as a Method of Solving Mechanical Problems

In the Lagrangian formulation, a proper choice of coordinates can greatly facilitate the solution of the equations of motion --- for instance, whenever a variable turns out to be ignorable (kinosthenic), we immediately obtain a first integral. Hence we try to produce ignorable variables by transforming the original set of coordinates.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Advantages of the Hamiltonian Framework for Transformations)</span></p>

In Lagrangian mechanics the significant function is $L = T - V$: simplifying $V$ may complicate $T$ and vice versa. In Hamiltonian mechanics the significant function is $H$ alone, which contains no derivatives --- only the variables themselves. The kinetic part of the canonical integral, $\sum p_i \dot{q}_i$, does not participate in the transformation. Hence we can focus entirely on simplifying $H$.

Furthermore, the doubling of variables from $n$ to $2n$ (positions and momenta) is actually an advantage: we have a much wider class of possible transformations at our disposal. Finally, in Hamiltonian mechanics a definite *systematic* method can be devised for the production of ignorable variables (via the generating function), whereas in Lagrangian mechanics one must rely on lucky guesses.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Viewpoint of Coordinate Transformations)</span></p>

The method of coordinate transformations employs a viewpoint entirely different from that of the direct integration problem. We forget about the equations of motion and consider the $q_i$, $p_i$ merely as *variables* --- coordinates of a point of the phase space. The specific problem of motion is entirely eliminated. What matters is that the transformation shall preserve the canonical form of the equations. If this differential form is preserved, the whole set of canonical equations is preserved. We call such transformations **canonical transformations**.

</div>

### 2. The Lagrangian Point Transformations

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Lagrangian Point Transformation)</span></p>

A **Lagrangian point transformation** is a transformation of the position coordinates alone:

$$q_i = f_i(Q_1, \ldots, Q_n), \qquad i = 1, \ldots, n,$$

where the new coordinates $Q_i$ replace the old $q_i$. The momenta $p_i$ are not independently transformed but follow from the invariance principle.

</div>

The canonical equations arise from a Lagrangian problem whose integrand is normalised to the canonical form $\sum p_i \dot{q}_i - H$. We require this form to be preserved under the transformation. A sufficient condition is the invariance of the differential form

$$\sum_{i=1}^{n} p_i\,\delta q_i = \sum_{i=1}^{n} P_i\,\delta Q_i.$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Canonical Invariance of Lagrangian Point Transformations)</span></p>

A Lagrangian point transformation leaves the canonical equations invariant. The transformation of the momenta is determined entirely by the invariance of the differential form $\sum p_i\,\delta q_i$: since the $\dot{q}_i$ are linearly related to the $\dot{Q}_i$ by the same Jacobian matrix that connects the $\delta q_i$ and $\delta Q_i$, the transformation of the momenta obtained from the invariance principle agrees with the Lagrangian definition $P_i = \frac{\partial L}{\partial \dot{Q}_i}$. The Hamiltonian function $H$ is an **invariant** of the transformation --- it takes the same functional form (after re-expression) in the new variables.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Rheonomic Point Transformations and Gyroscopic Terms)</span></p>

For scleronomic (time-independent) point transformations, the Hamiltonian function is an invariant of the transformation. For rheonomic (time-dependent) transformations of the form $q_i = f_i(Q_1, \ldots, Q_n, t)$, the situation changes: the invariance principle in the extended phase space involves the extended Hamiltonian function $K = p_t + H$. Since the time $t$ is also being transformed (to $\bar{t}$), the invariance of $K$ gives the relation

$$H' = H + \sum_{i=1}^{n} p_i \frac{\partial f_i}{\partial t}.$$

The Hamiltonian function is thus modified by additional terms that are *linear in the momenta*. These are called **gyroscopic terms** and give rise to apparent forces such as the Coriolis and centrifugal forces in a rotating reference frame.

</div>

### 3. Mathieu and Lie Transformations

The Lagrangian point transformations form only a very restricted subgroup of the general group of canonical transformations. A much broader class was studied by the French mathematician Mathieu (1874) and by Sophus Lie, who called them "contact transformations."

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Mathieu Transformation)</span></p>

A **Mathieu transformation** (or contact transformation) is characterised by the requirement that the differential form $\sum p_i\,\delta q_i$ shall be an invariant of the transformation:

$$\sum_{i=1}^{n} p_i\,\delta q_i = \sum_{i=1}^{n} P_i\,\delta Q_i.$$

Unlike the Lagrangian point transformation (where $q_i$ depends on the $Q_i$ alone), the momenta now participate in the position transformation. There must exist at least one functional relation

$$f(q_1, \ldots, q_n;\; Q_1, \ldots, Q_n) = 0$$

between the $q_i$ and $Q_i$ alone, not involving the $p_i$ and $P_i$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Classification of Mathieu Transformations)</span></p>

Mathieu transformations can be classified according to the number $m$ of independent relations that exist between the $q_i$ and $Q_i$ alone ($1 \leq m \leq n$). These $m$ auxiliary conditions are of the form

$$\Omega_k(q_1, \ldots, q_n;\; Q_1, \ldots, Q_n) = 0, \qquad k = 1, \ldots, m.$$

When $m = n$ (the maximum), the $q_i$ can be completely eliminated in terms of the $Q_i$, and we recover the Lagrangian point transformations as a special subgroup. The generating formulae are obtained using the Lagrangian $\lambda$-method:

$$p_i = \lambda_1 \frac{\partial \Omega_1}{\partial q_i} + \cdots + \lambda_m \frac{\partial \Omega_m}{\partial q_i}, \qquad P_i = -\left(\lambda_1 \frac{\partial \Omega_1}{\partial Q_i} + \cdots + \lambda_m \frac{\partial \Omega_m}{\partial Q_i}\right).$$

These equations, together with the $m$ auxiliary conditions, completely determine the transformation.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Rheonomic Mathieu Transformations)</span></p>

For rheonomic (time-dependent) Mathieu transformations, the time $t$ is included via the extended phase space. The invariance of the differential form becomes

$$\sum_{i=1}^{n} p_i\,\delta q_i + p_t\,\delta t = \sum_{i=1}^{n} P_i\,\delta Q_i + p_{\bar{t}}\,\delta \bar{t},$$

and the auxiliary conditions may also depend on $t$. The invariance of the extended Hamiltonian $K$ then gives

$$H' = H + \lambda_1 \frac{\partial \Omega_1}{\partial t} + \cdots + \lambda_m \frac{\partial \Omega_m}{\partial t}.$$

The Hamiltonian function is invariant only for scleronomic transformations; for rheonomic ones it acquires additional correction terms.

</div>

### 4. The General Canonical Transformation

The Mathieu transformations require the invariance of the differential form $\sum p_i\,\delta q_i$ itself. The general canonical transformation relaxes this condition: the differential form need not be preserved exactly, but only up to an exact differential.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Generating Function of a Canonical Transformation)</span></p>

The most general condition for a canonical transformation is

$$\sum_{i=1}^{n}\left(p_i\,\delta q_i - P_i\,\delta Q_i\right) = \delta S,$$

where $S = S(q_1, \ldots, q_n;\; Q_1, \ldots, Q_n)$ is a function of the old and new position coordinates called the **generating function**. The variation of $S$ gives

$$\delta S = \sum_{i=1}^{n}\left(\frac{\partial S}{\partial q_i}\,\delta q_i + \frac{\partial S}{\partial Q_i}\,\delta Q_i\right).$$

Comparing coefficients yields the fundamental equations of the transformation:

$$p_i = \frac{\partial S}{\partial q_i}, \qquad P_i = -\frac{\partial S}{\partial Q_i}.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Mixed Representation)</span></p>

The generating function equations do not give the transformation in explicit form. Instead, the old momenta $p_i$ and the new momenta $P_i$ are expressed as functions of the old and new position coordinates --- a *mixed* representation. To obtain the explicit transformation from old to new (or vice versa), one must solve one set of equations for the position coordinates. The generating function $S$ may also be combined with auxiliary conditions between the $q_i$ and $Q_i$ (as in the Mathieu transformations), producing a "conditioned" canonical transformation with $m$ prescribed conditions ($1 \leq m \leq n$). In this case the modified generating function is $\bar{S} = S + \lambda_1 \Omega_1 + \cdots + \lambda_m \Omega_m$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Invariance of the Canonical Equations under a General Canonical Transformation)</span></p>

Any canonical transformation defined by a generating function $S$ preserves the canonical equations. The canonical integral

$$A = \int_{t_1}^{t_2}\left(\sum p_i\,dq_i - H\,dt\right)$$

is modified to

$$A = \int_{t_1}^{t_2}\left(\sum P_i\,dQ_i - H\,dt\right) + \int_{t_1}^{t_2} dS.$$

The last term is a pure boundary term with no influence on the variation. Hence the vanishing of the variation in the original variables guarantees its vanishing in the new variables --- the canonical equations of motion remain invariant.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Transformation of the Hamiltonian under a General Canonical Transformation)</span></p>

For a scleronomic (time-independent) generating function $S = S(q_i, Q_i)$, the Hamiltonian function is an invariant: $H' = H$. For a rheonomic (time-dependent) generating function $S = S(q_i, Q_i, t)$, the transformed Hamiltonian is

$$H' = H + \frac{\partial S}{\partial t}.$$

</div>

### 5. The Bilinear Differential Form

In any transformation theory there are basic invariants that determine the nature of the transformation. The Riemannian line element $d\bar{s}^2$ was the basic invariant of Lagrangian point transformations and it determined the geometry of configuration space. In the phase space of Hamiltonian mechanics, an analogous role is played by a differential quantity of a different character.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Invariance of the Circulation under Canonical Transformations)</span></p>

For any closed curve $L$ of the phase space, the **circulation**

$$\Gamma = \oint \sum_{i=1}^{n} p_i\,dq_i$$

is invariant under an arbitrary canonical transformation:

$$\oint \sum_{i=1}^{n} p_i\,dq_i = \oint \sum_{i=1}^{n} P_i\,dQ_i.$$

This follows from integrating the defining relation $\sum p_i\,dq_i - \sum P_i\,dQ_i = dS$ around any closed curve, since the integral of an exact differential around a closed path vanishes.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Bilinear Differential Form)</span></p>

The circulation can be converted from a line integral into a surface integral via a generalised Stokes' theorem. Parametrising a two-dimensional region $K$ of phase space by Gaussian curvilinear coordinates $u$ and $v$, we obtain

$$\oint_L \sum_{i=1}^{n} p_i\,dq_i = \int_K \sum_{i=1}^{n}\left(\frac{\partial q_i}{\partial u}\frac{\partial p_i}{\partial v} - \frac{\partial p_i}{\partial u}\frac{\partial q_i}{\partial v}\right) du\,dv.$$

The integrand, evaluated for any two independent infinitesimal displacements $d'q_i, d'p_i$ and $d''q_i, d''p_i$ of the phase space, gives the **bilinear differential form**:

$$\sum_{i=1}^{n}\left(d'p_i\,d''q_i - d''p_i\,d'q_i\right).$$

The invariance of this bilinear differential form is the **necessary and sufficient condition** for a canonical transformation.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Geometry of Phase Space)</span></p>

The bilinear differential form plays a role in phase space analogous to that of $d\bar{s}^2$ in Riemannian configuration space, but with a crucial difference. While the distance element is a *quadratic* form in a *single* infinitesimal displacement, the bilinear differential form is associated with *two* independent infinitesimal displacements. It therefore resembles an *area element* rather than a distance element. The geometry of the phase space is not of the ordinary metrical kind --- it is a geometry in which areas can be measured but distances cannot. The entire theory of canonical transformations can be based on this invariant differential form, and the generating function $S$ is eliminated from the picture.

</div>

### 6. The Bracket Expressions of Lagrange and Poisson

Lagrange anticipated many of the results which are more systematically deducible from the theory of canonical transformations. He noticed that a certain expression is of prime importance for the theory of perturbations.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Lagrange Bracket)</span></p>

Let the two sets of variables $q_i$ and $p_i$ be given as functions of two parameters $u$ and $v$. The **Lagrange bracket** is

$$[u, v] = \sum_{i=1}^{n}\left(\frac{\partial q_i}{\partial u}\frac{\partial p_i}{\partial v} - \frac{\partial q_i}{\partial v}\frac{\partial p_i}{\partial u}\right).$$

It is anti-symmetric with respect to a change of the variables $u$, $v$:

$$[u, v] = -[v, u].$$

</div>

The Lagrange bracket is closely connected with the theory of canonical transformations. It is exactly the quantity encountered when the circulation is expressed as a line integral converted to a surface integral:

$$\oint_L \sum_{i=1}^{n} p_i\,dq_i = \int_K [u, v]\,du\,dv.$$

Since the invariance of the circulation, taken round any closed curve $L$, is a characteristic property of a canonical transformation, the same property can also be expressed as the invariance of the Lagrange bracket: *the canonical transformations are those transformations of the variables $q_i$, $p_i$ to $Q_i$, $P_i$ which leave the Lagrange bracket invariant, no matter how $q_i$, $p_i$ depend on $u$ and $v$.*

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Canonical Conditions via Lagrange Brackets)</span></p>

Let the transformation from the old coordinates to the new be given in explicit form:

$$q_i = f_i(Q_1, \ldots, Q_n;\, P_1, \ldots, P_n), \qquad p_i = g_i(Q_1, \ldots, Q_n;\, P_1, \ldots, P_n).$$

Pick any pair of the new variables $Q_i$ and $Q_k$, or $Q_i$ and $P_k$, or $P_i$ and $P_k$, and consider them as the two parameters $u$, $v$ for which the Lagrange bracket is to be formed, holding the other variables as constants. Then the **necessary and sufficient conditions** for the transformation to be canonical are the following Lagrange bracket relations:

$$[Q_i, Q_k] = 0, \qquad [P_i, P_k] = 0, \qquad [Q_i, P_k] = \delta_{ik},$$

where $\delta_{ik}$ is the Kronecker delta.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Poisson Bracket)</span></p>

Soon after Lagrange's investigation, Poisson (in 1809) introduced another type of bracket expression which is the natural counterpart of the Lagrange bracket. Instead of considering the $q_i$ and $p_i$ as functions of $u$ and $v$, we consider a pair of variables $u$ and $v$ as given functions of the coordinates $q_i$, $p_i$:

$$u = u(q_1, \ldots, q_n;\, p_1, \ldots, p_n), \qquad v = v(q_1, \ldots, q_n;\, p_1, \ldots, p_n).$$

The **Poisson bracket** is then defined as

$$(u, v) = \sum_{i=1}^{n}\left(\frac{\partial u}{\partial q_i}\frac{\partial v}{\partial p_i} - \frac{\partial v}{\partial q_i}\frac{\partial u}{\partial p_i}\right).$$

These brackets are likewise anti-symmetric:

$$(u, v) = -(v, u).$$

</div>

The Lagrange and Poisson brackets are closely related. Given $2n$ variables $u_1, \ldots, u_{2n}$ which are given functions of the $q_i$, $p_i$, the Poisson brackets can be formed; for the second type the Lagrange brackets can be formed. One kind of bracket determines the other. Hence, if the Lagrange brackets are invariants of a canonical transformation, so are the Poisson brackets. This gives an alternative formulation of the conditions for a canonical transformation.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Canonical Conditions via Poisson Brackets)</span></p>

*The canonical transformations are those transformations which leave the Poisson bracket $(u, v)$ invariant, no matter how the functions $u$ and $v$ depend on the coordinates $q_i$, $p_i$.*

Applying this invariance principle to the inverse form of the canonical transformation, expressing the new coordinates in terms of the old:

$$Q_i = Q_i(q_1, \ldots, q_n;\, p_1, \ldots, p_n), \qquad P_i = P_i(q_1, \ldots, q_n;\, p_1, \ldots, p_n),$$

the Poisson bracket conditions become:

$$(Q_i, Q_k) = 0, \qquad (P_i, P_k) = 0, \qquad (Q_i, P_k) = \delta_{ik}.$$

These conditions are equivalent to the Lagrange bracket conditions.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Reciprocity Between Lagrange and Poisson Brackets)</span></p>

Consider $u_1, \ldots, u_{2n}$ as given functions of the $q_i$, $p_i$; then the Lagrange and the Poisson brackets satisfy the reciprocity relation:

$$\sum_{a=1}^{2n} [u_i, u_a]\,(u_k, u_a) = \delta_{ik}.$$

Because of this relation, the Poisson bracket conditions are immediate consequences of the Lagrange bracket conditions and vice versa.

</div>

### 7. Infinitesimal Canonical Transformations

Canonical transformations have the **group property**: the product of two canonical transformations is itself a canonical transformation. This follows directly from the defining relation: if

$$\sum_{i=1}^{n}(p_i\,\delta q_i - P_i\,\delta Q_i) = \delta S, \qquad \sum_{i=1}^{n}(P_i\,\delta Q_i - \bar{p}_i\,\delta\bar{q}_i) = \delta S',$$

then by addition

$$\sum_{i=1}^{n}(p_i\,\delta q_i - \bar{p}_i\,\delta\bar{q}_i) = \delta(S + S'),$$

which shows that the direct transition from $q_i$, $p_i$ to the final $\bar{q}_i$, $\bar{p}_i$ is also canonical. The canonical transformations thus form a *group* in Lie's theory of continuous transformations.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Infinitesimal Canonical Transformation)</span></p>

Consider a generating function $S$ which contains a parameter $t$:

$$S = S(q_1, \ldots, q_n;\, Q_1, \ldots, Q_n;\, t).$$

Each value of $t$ corresponds a definite canonical transformation. An **infinitesimal canonical transformation** is obtained when a transformation belonging to the value $t$ is compared with its neighbouring transformation belonging to $t + \Delta t$. Each point of the phase space is transformed into a neighbouring point:

$$Q_k = q_k + \Delta q_k, \qquad P_k = p_k + \Delta p_k,$$

where $\Delta q_k$ and $\Delta p_k$ are small quantities whose products and squares are negligible.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Explicit Form of Infinitesimal Canonical Transformations)</span></p>

While an arbitrary canonical transformation does not permit an explicit representation, an infinitesimal canonical transformation *can* be obtained in explicit form. The determining equations of the canonical transformation give $p_i = \frac{\partial S}{\partial q_i}$. Expanding $S$ and forming the difference of the two equations for parameters $t$ and $t + \Delta t$, one obtains:

$$\sum_{i=1}^{n}(\Delta p_i\,\delta q_i - \Delta q_i\,\delta p_i) = -\sum_{i=1}^{n}\left(\frac{\partial B}{\partial q_i}\,\delta q_i + \frac{\partial B}{\partial p_i}\,\delta p_i\right)\Delta t,$$

where the function $B$ is constructed from the generating function $S$ by introducing $p_i$ via the canonical relation $p_i = \frac{\partial S}{\partial q_i}$ and setting

$$\frac{\partial S}{\partial t} = -B(q_1, \ldots, q_n;\, p_1, \ldots, p_n;\, t).$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Equations of an Infinitesimal Canonical Transformation)</span></p>

The explicit equations of an infinitesimal canonical transformation are:

$$\Delta q_i = \frac{\partial B}{\partial p_i}\,\Delta t, \qquad \Delta p_i = -\frac{\partial B}{\partial q_i}\,\Delta t.$$

Instead of the "absolute coordinates" $q_i + \Delta q_i$, $p_i + \Delta p_i$ of the new reference system, the "relative coordinates" $\Delta q_i$, $\Delta p_i$ can be used. These coordinates are explicitly expressed in terms of a single function $B$ which characterises the transformation, and can be chosen as an arbitrarily chosen function of the variables $q_i$ and $p_i$.

</div>

### 8. The Motion of the Phase Fluid as a Continuous Succession of Canonical Transformations

If we let $\Delta t$ tend towards zero after dividing both sides of the infinitesimal canonical transformation equations by $\Delta t$, we obtain in the limit the differential equations:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Canonical Equations from Infinitesimal Transformations)</span></p>

$$\frac{dq_i}{dt} = \frac{\partial B}{\partial p_i}, \qquad \frac{dp_i}{dt} = -\frac{\partial B}{\partial q_i}.$$

*These are nothing but the canonical equations of motion*, if the variable parameter $t$ is identified with the time and the function $B$ is identified with the Hamiltonian function $H$.

</div>

This result yields a profound reinterpretation of the dynamics. Imagine the motion of the phase fluid during a time interval $\Delta t$. All fluid particles are displaced by infinitesimal amounts. These infinitesimal displacements represent a canonical transformation of the phase fluid. The process can be repeated any number of times. *The whole motion of the phase fluid is nothing but a continuous succession of canonical transformations.*

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Phase Fluid Picture)</span></p>

This result --- first conceived by Hamilton --- reveals the role of canonical transformations in the study of the phenomena of motion from a new angle. *The entire motion of the mechanical system can be considered as a transformation problem.* The successive states of the phase fluid represent a continuously changing mapping of the space on itself. *This mapping is constantly canonical.*

The invariants of the motion are actually *invariants of any canonical transformation*. The invariance of the circulation, discussed in chapter VI section 8, is a characteristic feature of canonical transformations: in fact it defines these transformations. Liouville's theorem brought the invariance of the volume into evidence, based on the incompressibility of the phase fluid. This theorem may be expressed as: *the value of the functional determinant of any canonical transformation is equal to 1.*

</div>

We can now proceed in the opposite direction. A given problem of motion presents us with the Hamiltonian $H$ as a function of $q_i$, $p_i$ and (possibly) $t$. We replace $p_i$ by $\frac{\partial S}{\partial q_i}$ and try to find the original function $S$ from the equation

$$\frac{\partial S}{\partial t} = -H.$$

This means an *integration problem*. One can show that for any given $H$ a corresponding $S$-function can be found. After constructing this $S$-function we arrive at

$$S = S(q_1, \ldots, q_n;\, Q_1, \ldots, Q_n;\, t)$$

which generates an infinite family of canonical transformations. A point $Q_i$, $P_i$ is transformed into a point $q_i$, $p_i$, the position of which changes continuously with the time $t$, and this motion represents exactly the motion of the mechanical system if we let $t$ change while the $Q_i$, $P_i$ are kept constant. The motion of the entire phase fluid is nothing but *the successive evolution of a time-dependent canonical transformation*.

### 9. Hamilton's Principal Function and the Motion of the Phase Fluid

The results of the previous discussion have a direct bearing on the integration problem of the dynamical equations. We know that the relation between the generating function $S$ and the given Hamiltonian $H$ is

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Hamilton--Jacobi Equation)</span></p>

$$\frac{\partial S}{\partial t} + H = 0,$$

where the $p_i$ of the Hamiltonian function $H$ are replaced by $\partial S / \partial q_i$. If we are able to find a generating function $S$ which satisfies this partial differential equation, then we have succeeded in obtaining the motion of the phase fluid as the successive phases of a time-dependent canonical transformation.

</div>

The transformation equations take the form

$$q_i = f_i(Q_1, \ldots, Q_n;\, P_1, \ldots, P_n;\, t), \qquad p_i = g_i(Q_1, \ldots, Q_n;\, P_1, \ldots, P_n;\, t),$$

which amounts to a *complete integration* of the dynamical problem because all the mechanical variables are expressed as explicit functions of the time $t$ and $2n$ constants $Q_1, \ldots, Q_n$; $P_1, \ldots, P_n$ which can be adjusted to arbitrary initial conditions. The problem of integration is thus shifted to the problem of finding the generating function.

#### Conservative Systems and Hamilton's Principal Function

For a conservative system whose Lagrangian $L$ and Hamiltonian $H$ do not contain the time explicitly, we consider the principle of least action in Lagrange's formulation, changing from configuration space to phase space. The stationary value of the "action"

$$A = 2\int_{\tau_1}^{\tau_2} T\,dt = \int_{\tau_1}^{\tau_2}\sum_{i=1}^{n} p_i\,dq_i$$

is sought under the auxiliary condition that the $C$-point of the phase space stays on the energy surface $H(q_1, \ldots, q_n, p_1, \ldots, p_n) - E = 0$. This principle can be expressed in Jacobi's form: minimise the integral

$$A = \sqrt{2}\int_{\tau_1}^{\tau_2}\sqrt{E - V}\,\overline{ds}$$

without any auxiliary condition, where the Riemannian line-element is

$$d\sigma = \sqrt{2(E-V)}\,\overline{ds}.$$

Jacobi's principle assumes a purely geometrical significance: determine the shortest lines or geodesics of the given Riemannian manifold.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Hamilton's Principal Function --- Conservative Case)</span></p>

Evaluating the integral $A$ along the geodesics, the arc length of a geodesic between two points $M$ and $N$ gives the "distance" between them. This distance is a function of the coordinates $q_i$ of the two end-points $M$ and $N$. *This distance, expressed in the coordinates of the two end-points, is Hamilton's principal function*:

$$W = W(q_1, \ldots, q_n;\, \bar{q}_1, \ldots, \bar{q}_n).$$

The variation of the integral $A$ vanishes between definite limits; varying the limits yields the boundary term

$$\delta A = \sum_{i=1}^{n} p_i\,\delta q_i - \sum_{i=1}^{n}\bar{p}_i\,\delta\bar{q}_i,$$

and since $W$ is by definition the definite integral $A$ expressed in terms of the $q_i$ and $\bar{q}_i$, we get

$$p_i = \frac{\partial W}{\partial q_i}, \qquad \bar{p}_i = -\frac{\partial W}{\partial \bar{q}_i}.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Significance of Hamilton's Principal Function)</span></p>

These equations show that *two positions of the moving phase fluid are connected with each other by means of a canonical transformation*. The role of $W$ in these equations shows that Hamilton's principal function is the generating function of that particular canonical transformation which converts one state of the moving phase fluid into a later state.

Notice that the generating function $S$ of a general time-dependent canonical transformation is more general than Hamilton's $W$-function: a general time-dependent canonical transformation transforms an arbitrary point $Q_i$, $P_i$ of the phase space into the moving point $q_i$, $p_i$, while Hamilton's transformation transforms the *initial position* $\bar{q}_i$, $\bar{p}_i$ of the moving fluid particle into some later position $q_i$, $p_i$.

</div>

Since the coordinates $q_i$, $p_i$ have never left the energy surface $H = E$, the function $W$ satisfies automatically the partial differential equation

$$H\!\left(q_1, \ldots, q_n;\, \frac{\partial W}{\partial q_1}, \ldots, \frac{\partial W}{\partial q_n}\right) - E = 0.$$

The same can be said of the coordinates $\bar{q}_i$, $\bar{p}_i$ of the initial point which belongs to the same manifold. Hence $W$ satisfies a *second* partial differential equation, obtained by replacing $p_i$ by $-\frac{\partial W}{\partial \bar{q}_i}$:

$$H\!\left(\bar{q}_1, \ldots, \bar{q}_n;\, -\frac{\partial W}{\partial \bar{q}_1}, \ldots, -\frac{\partial W}{\partial \bar{q}_n}\right) - E = 0.$$

Hamilton's principal function is thus restricted by a *pair* of partial differential equations.

#### The Time-Dependent Case

For the general (time-dependent) case, we add $t$ to the position coordinates and consider the "extended action"

$$A = \int_{\tau_1}^{\tau_2}\!\left(\sum_{i=1}^{n} p_i\,\dot{q}_i + p_t\right)d\tau = \int_{\tau_1}^{\tau_2}\!\left(\sum_{i=1}^{n} p_i\,dq_i + p_t\,dt\right),$$

under the auxiliary condition $K(q_1, \ldots, q_n, t;\, p_1, \ldots, p_n, p_t) = 0$, i.e. $p_t + H = 0$. A line element for the extended $(n+1)$-dimensional configuration space $q_1, \ldots, q_n, t$ is defined by

$$\overline{d\sigma} = L\,dt = L'\,d\tau,$$

and one again connects two points $\bar{q}_1, \ldots, \bar{q}_n, \bar{t}$ and $q_1, \ldots, q_n, t$ of the $(n+1)$-dimensional space by a shortest line and measures the arc length

$$A = \int_{\tau_1}^{\tau_2}\overline{d\sigma} = \int_{\tau_1}^{\tau_2} L'\,d\tau.$$

Hamilton's principal function in the general case is

$$W = W(q_1, \ldots, q_n, t;\, \bar{q}_1, \ldots, \bar{q}_n, \bar{t}).$$

Varying the limits establishes the relations

$$p_i = \frac{\partial W}{\partial q_i}, \qquad \bar{p}_i = -\frac{\partial W}{\partial \bar{q}_i}, \qquad p_t = \frac{\partial W}{\partial t}, \qquad \bar{p}_t = -\frac{\partial W}{\partial \bar{t}},$$

which confirm the canonical nature of the transformation.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Hamilton--Jacobi Partial Differential Equation)</span></p>

Expressing $K$ in terms of the Hamiltonian $K = p_t + H$, the two partial differential equations which hold for $W$ become

$$\frac{\partial W}{\partial t} + H\!\left(q_1, \ldots, q_n;\, \frac{\partial W}{\partial q_1}, \ldots, \frac{\partial W}{\partial q_n};\, t\right) = 0,$$

$$-\frac{\partial W}{\partial \bar{t}} + H\!\left(\bar{q}_1, \ldots, \bar{q}_n;\, -\frac{\partial W}{\partial \bar{q}_1}, \ldots, -\frac{\partial W}{\partial \bar{q}_n};\, \bar{t}\right) = 0.$$

The last two equations of the transformation follow from the first two and can be dropped. In the remaining equations we obtain the transformation

$$q_i = f_i(\bar{q}_1, \ldots, \bar{q}_n;\, \bar{t},\, t), \qquad p_i = g_i(\bar{q}_1, \ldots, \bar{q}_n;\, \bar{t},\, t),$$

which solves the problem of motion in explicit form, giving the coordinates $q_i$, $p_i$ of the moving point at any time $t$ if the initial position at time $\bar{t}$ is given.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Hamilton vs. Jacobi)</span></p>

Hamilton's integration scheme was simplified and improved by Jacobi. Hamilton's principal function is very restricted and has to satisfy two simultaneous partial differential equations. The solution of this problem would be practically impossible without the broader integration scheme of Jacobi. The generating function $S$ of a time-dependent canonical transformation generates the entire motion of the phase fluid, subject only to the partial differential equation

$$\frac{\partial S}{\partial t} + H\!\left(q_1, \ldots, q_n;\, \frac{\partial S}{\partial q_1}, \ldots, \frac{\partial S}{\partial q_n};\, t\right) = 0.$$

A second differential equation is no longer necessary, since the point $Q_1, \ldots, Q_n$ need *not* lie on the extended energy surface $K = 0$. Moreover, $S$ is a function of $q_1, \ldots, q_n$; $Q_1, \ldots, Q_n$; $t$ only, while Hamilton's principal function contains in addition the surplus variable $\bar{t}$. The relation between the two theories will be discussed in greater detail in the next chapter.

</div>

## Chapter VIII: The Partial Differential Equation of Hamilton--Jacobi

### 1. The Importance of the Generating Function for the Problem of Motion

In the theory of canonical transformations no other theorem is of such importance as that a canonical transformation is completely characterised by knowing one single function $S$, the generating function of the transformation. This parallels the fact that the canonical equations are likewise characterised by one single function, the Hamiltonian function $H$. These two fundamental functions can be linked together by a definite relation. In order to solve the problem of motion it suffices to consider the Hamiltonian function and try to simplify it to a form in which the canonical equations become directly integrable. For this purpose a suitable canonical transformation can be employed. But this transformation depends on one single function. And thus the problem of solving the entire canonical set can be replaced by the problem of solving a single equation --- a partial differential equation.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Practical vs. Philosophical Significance)</span></p>

From the practical viewpoint not much is gained. The solution of a partial differential equation --- even *one* equation --- is no easy task, and in most cases is not simpler than the original integration problem. Those problems which can be solved explicitly by means of the partial differential equation are, in the majority of cases, the same problems which can be solved also by other means. For this reason the Hamiltonian methods were long considered as of purely mathematical interest and of little practical importance. Yet in contemporary physics, the Hamiltonian methods gained recognition because of the optico-mechanical analogy which was made clear by Hamilton's partial differential equation. Since the advent of Schroedinger's wave mechanics, which is based essentially on Hamilton's researches, the leading ideas of Hamiltonian mechanics have found their way into the textbooks of theoretical physics.

</div>

### 2. Jacobi's Transformation Theory

#### The Conservative Case

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Jacobi's Integration Method --- Conservative Systems)</span></p>

Let us consider a conservative mechanical system with a given Hamiltonian function $H$ which does not depend on the time $t$. We wish to transform the mechanical variables $q_1, \ldots, q_n$; $p_1, \ldots, p_n$ into a new set of variables $Q_1, \ldots, Q_n$; $P_1, \ldots, P_n$ by a canonical transformation, except for the single condition that the Hamiltonian function $H$ shall be one of the new variables, say $Q_n$:

$$Q_n = H(q_1, \ldots, q_n;\, p_1, \ldots, p_n).$$

If such a transformation is found, then in the new system *all variables are ignorable* and the canonical equations are completely integrated. The first set of canonical equations gives at once

$$Q_i = \text{const.} = \mathfrak{a}_i, \quad (i = 1, 2, \ldots, n-1),$$

while $Q_n = \text{const.} = E$ is the energy constant. The second set yields

$$P_i = \text{const.} = -\beta_i, \quad (i = 1, \ldots, n-1), \qquad P_n = \tau - t.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Geometric Interpretation)</span></p>

This transformation can be interpreted geometrically: the original world lines of the moving phase fluid fill the state space with an infinite family of curves. By a canonical transformation a mapping of the space on itself is produced which *straightens out these world lines to an infinite bundle of parallel straight lines*, inclined at 45° to the time axis. By merely flattening out the cylindrical surfaces $H = E$ into the parallel planes $Q_n = E$ we ensure that all the previously curved world lines of the phase fluid are now straightened out to parallel straight lines.

</div>

#### The Hamilton--Jacobi Equation for Conservative Systems

The original integration problem is reduced to that of finding a canonical transformation which satisfies the condition that $H$ becomes $Q_n$. We interject an intermediate step: first introduce the $Q_i$ only, keeping the old $q_i$ but eliminating the $p_i$ via $p_i = \frac{\partial S}{\partial q_i}$. Then the Hamiltonian function becomes a function of $q_i$ and $Q_i$, and the condition that it equals $Q_n$ gives:

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Hamilton--Jacobi Equation --- Conservative Form)</span></p>

$$H\!\left(q_1, \ldots, q_n;\, \frac{\partial S}{\partial q_1}, \ldots, \frac{\partial S}{\partial q_n}\right) = E.$$

This is a partial differential equation for the function $S$. It is not enough to find *some* solution; to serve as a generating function, $S$ must have the form

$$S = S(q_1, \ldots, q_n;\, \mathfrak{a}_1, \ldots, \mathfrak{a}_{n-1};\, Q_n),$$

i.e. a **complete solution** containing $n - 1$ essential (non-trivial) constants of integration $\mathfrak{a}_1, \ldots, \mathfrak{a}_{n-1}$. Since $S$ itself does not appear in the differential equation, but only its partial derivatives, an additive constant is irrelevant and can be omitted.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Jacobi's Recipe for Conservative Systems)</span></p>

1. Write down the energy equation: $H(q_1, \ldots, q_n, p_1, \ldots, p_n) = E$.

2. Replace the $p_i$ by the partial derivatives of some function $S$ with respect to $q_i$:

$$H\!\left(q_1, \ldots, q_n;\, \frac{\partial S}{\partial q_1}, \ldots, \frac{\partial S}{\partial q_n}\right) = E.$$

3. Find a complete solution of this equation with $n - 1$ non-trivial constants of integration: $S = S(q_1, \ldots, q_n;\, \mathfrak{a}_1, \ldots, \mathfrak{a}_{n-1};\, E)$.

4. Form the equations

$$\frac{\partial S}{\partial \mathfrak{a}_i} = \beta_i, \qquad \frac{\partial S}{\partial E} = t - \tau.$$

5. Solve these equations for the $q_i$, obtaining

$$q_i = f_i(\mathfrak{a}_1, \ldots, \mathfrak{a}_{n-1},\, E,\, \beta_1, \ldots, \beta_{n-1},\, t - \tau).$$

This completes the solution. The $q_i$ and $p_i$ are determined as explicit functions of the time $t$ and $2n$ constants of integration.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Linear Oscillator)</span></p>

For a single degree of freedom with $V = \frac{1}{2}mk^2 x^2$, the partial differential equation reduces to an ordinary differential equation solvable by quadratures. The Hamilton--Jacobi equation yields the canonical transformation:

$$x = \frac{1}{k}\sqrt{\frac{2Q}{m}}\cos kP, \qquad p = \sqrt{2mQ}\sin kP.$$

The ellipses $H = E$ of the original coordinate system are transformed into the straight lines $Q = E$ of the new coordinate system. The ellipse which encloses a definite area is mapped to a shaded rectangle of the same area, since canonical transformations preserve area (the transformation is non-single-valued: $P$ is restricted to the range $0$ to $2\pi/k$).

</div>

#### The Rheonomic (Time-Dependent) Case

For a general rheonomic system which does not satisfy the law of conservation of energy, we can generalise the conclusions drawn for conservative systems by merely adding the time $t$ to the position coordinates $q_i$ and formulating our problem as a conservative problem in the extended phase space.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Hamilton--Jacobi Equation --- General Rheonomic Form)</span></p>

In the extended phase space with the auxiliary condition $K(q_i, t;\, p_i, p_t) = 0$ where $K = p_t + H$, we apply a canonical transformation such that $\bar{t} = K$. Then the auxiliary condition becomes $\bar{t} = 0$, and all $Q_i$ and $P_i$ are constants:

$$Q_i = \mathfrak{a}_i, \qquad P_i = \beta_i.$$

The condition $\bar{t} = K$ is equivalent to the partial differential equation

$$\frac{\partial S}{\partial t} + H\!\left(q_1, \ldots, q_n, t;\, \frac{\partial S}{\partial q_1}, \ldots, \frac{\partial S}{\partial q_n}\right) = 0,$$

which is the Hamilton--Jacobi equation for arbitrary rheonomic systems. A "complete solution" is now required containing $n$ essential constants of integration $\mathfrak{a}_1, \ldots, \mathfrak{a}_n$.

</div>

An alternative derivation remains within the ordinary phase space, considering a canonical transformation of $q_i$, $p_i$ into $Q_i$, $P_i$ *without* adding the time as a variable. The time enters merely as a parameter. Under such a time-dependent canonical transformation the Hamiltonian function $H$ is not an invariant; the new Hamiltonian is $H' = \frac{\partial S}{\partial t} + H$. We can now require $H' = 0$. This gives exactly the Hamilton--Jacobi equation, now interpreted as the condition that the Hamiltonian function be transformed into zero. If the Hamiltonian of the new coordinate system is zero, then all $Q_i$ and $P_i$ are constants during the motion.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Jacobi's Recipe for Rheonomic Systems)</span></p>

1. Obtain an arbitrary complete solution of the partial differential equation

$$\frac{\partial S}{\partial t} + H\!\left(q_1, \ldots, q_n, t;\, \frac{\partial S}{\partial q_1}, \ldots, \frac{\partial S}{\partial q_n}\right) = 0,$$

i.e. a solution which contains $n$ essential constants of integration $\mathfrak{a}_1, \ldots, \mathfrak{a}_n$:

$$S = S(q_1, \ldots, q_n, t, \mathfrak{a}_1, \ldots, \mathfrak{a}_n).$$

2. Set the partial derivatives of $S$ with respect to the $n$ constants of integration equal to $n$ new constants:

$$\frac{\partial S}{\partial \mathfrak{a}_i} = \beta_i.$$

3. Solve these equations for the $n$ position coordinates $q_i$, obtaining them in the form

$$q_i = f_i(\mathfrak{a}_1, \ldots, \mathfrak{a}_n, \beta_1, \ldots, \beta_n, t).$$

This procedure amounts to a *complete solution of the integration problem*, since the mechanical variables are expressed as explicit functions of the time $t$ and $2n$ constants of integration which can be adjusted to arbitrary initial conditions.

</div>

### 3. Solution of the Partial Differential Equation by Separation

We do not possess any general method for the solution of a partial differential equation. Yet under certain special conditions a complete solution of the Hamilton--Jacobi equation is possible. This special class of problems turned out to be of great importance in the development of theoretical physics, because some of the fundamental problems of Bohr's atomic theory belonged to this class.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Separable Systems)</span></p>

In **separable problems** the one partial differential equation in $n$ variables can be replaced by $n$ ordinary differential equations in a single variable. These equations are completely integrable.

The method of separation consists in setting $S$ equal to a sum of functions, each of which depends on a single variable only:

$$S = S_1(q_1) + S_2(q_2) + \ldots + S_n(q_n).$$

The characteristic feature is that now the momentum $p_k = \frac{\partial S}{\partial q_k} = \frac{dS_k(q_k)}{dq_k}$ *becomes a function of $q_k$ alone*. The energy equation $H - E = 0$ can then be separated, yielding $n$ relations of the form

$$p_k = f_k(q_k, \mathfrak{a}_1, \ldots, \mathfrak{a}_{n-1}, E),$$

where $\mathfrak{a}_1, \ldots, \mathfrak{a}_{n-1}$ are arbitrary constants obtained by the process of separation. Each $S_k$ is obtained by a straightforward quadrature:

$$S_k = \int f_k(q_k, \mathfrak{a}_1, \ldots, \mathfrak{a}_{n-1}, E)\,dq_k + C_k.$$

The "constants of integration" are obtained by the technique of separation; the actual integration does not add any new constants.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Kepler Problem)</span></p>

For the Kepler problem of planetary motion, we use polar coordinates $r$, $\theta$, $\phi$, where the line element is $ds^2 = dr^2 + r^2\,d\theta^2 + r^2\sin^2\theta\,d\phi^2$ and the Hamiltonian is

$$H = \frac{1}{2m}\!\left(p_r^2 + \frac{p_\theta^2}{r^2} + \frac{p_\phi^2}{r^2\sin^2\theta}\right) - \frac{k^2}{r}.$$

The equation $H = E$ can be separated as follows:

$$p_\phi = \text{const.} = \mathfrak{a},$$

$$p_\theta^2 + \frac{\mathfrak{a}^2}{\sin^2\theta} = \text{const.} = \beta^2,$$

$$\frac{1}{2m}\!\left(p_r^2 + \frac{\beta^2}{r^2}\right) - \frac{k^2}{r} = E.$$

*The technique of separation brings in automatically the right number of constants.* The actual integration does not produce any new constants.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Separability and Coordinates)</span></p>

The separable nature of a problem constitutes no inherent feature of the physical properties of a mechanical system, but is entirely a matter of the *right system of coordinates*. A problem which is not separable in a given system of coordinates might become separable after a proper point transformation. Unfortunately, the finding of the right system of coordinates is to some extent a matter of chance, since we do not possess any systematic method of procedure.

The complete solution of a separable system gives the unusual situation that the conjugate variables $q_k$, $p_k$ of each pair interact strictly with one another, without any interference from the other variables. The mechanical system of $n$ degrees of freedom can be considered as a superposition of $n$ systems of one degree of freedom. However, the actual equations of motion $\frac{\partial S}{\partial \mathfrak{a}_i} = \beta_i$, $\frac{\partial S}{\partial E} = t - \tau$ are *not* separated since generally a certain $\mathfrak{a}_i$ --- and also $E$ --- will be present in more than one of the $S_i$.

</div>

### 4. Delaunay's Treatment of Separable Periodic Systems

The method of separation --- if it is applicable --- provides us with the complete solution of the Hamilton--Jacobi equation which is required in Jacobi's integration theory. Now a "complete solution" of a partial differential equation of the first order can appear in many different forms.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Action Variables)</span></p>

The constants of separation $\mathfrak{a}_1, \ldots, \mathfrak{a}_{n-1}$ and the energy constant $E$ which appear as the new position coordinates $Q_i$ of a canonical transformation can be replaced by a new set of variables $J_i$, called **action variables**, since they have the dimensions of "action" --- i.e. of $p$ times $q$.

Assuming that the stream lines in all the $q_k$, $p_k$ planes are closed, the action variables are defined as the line integrals extended over a complete stream line:

$$J_k = \oint p_k\,dq_k = \oint f_k(q_k, \mathfrak{a}_1, \ldots, \mathfrak{a}_{n-1}, E)\,dq_k.$$

This line integral is equal to the area enclosed by the stream line. The $J_k$ are functions of the $\mathfrak{a}_i$ and $E$:

$$J_k = J_k(\mathfrak{a}_1, \ldots, \mathfrak{a}_{n-1}, E).$$

From these equations we can eliminate the $\mathfrak{a}_i$ and $E$, expressing them in terms of the $J_i$:

$$\mathfrak{a}_i = \mathfrak{a}_i(J_1, \ldots, J_n), \qquad E = E(J_1, \ldots, J_n).$$

The generating function then becomes $S = S(q_1, \ldots, q_n, J_1, \ldots, J_n)$, and the function $E$ is actually the Hamiltonian function $H$ of the new system.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Angle Variables)</span></p>

The conjugate momenta $P_i$ of the action variables are called **angle variables**. They are pure numbers without physical dimensions. We prefer to use the negative $P_i$ instead of $P_i$ themselves and denote them by $\omega_1, \ldots, \omega_n$. According to the general transformation scheme:

$$-P_i = \omega_i = \frac{\partial S}{\partial J_i}.$$

The passage of $q_k$ through a complete cycle causes no change in the $\omega_i$, *except for the variable $\omega_k$ which changes by one*:

$$\Delta\omega_i = \frac{\partial J_k}{\partial J_i} = \delta_{ik}.$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Multiply-Periodic Motion and Fundamental Frequencies)</span></p>

Since changing an arbitrary $\omega_i$ by one does not change the values of the $q_i$, the $q_i$ are **periodic functions** of all the angle variables $\omega_1, \ldots, \omega_n$ *with the period one*. This means that the $q_i$ can be expressed as multiple Fourier series of sines and cosines with the arguments

$$2\pi(k_1\omega_1 + k_2\omega_2 + \ldots + k_n\omega_n),$$

where $k_1, k_2, \ldots, k_n$ are arbitrary integers. The amplitudes of these terms are constants.

Now solving the canonical equations in the new coordinate system of $J_i$, $\omega_i$ variables: since $H = E(J_1, \ldots, J_n)$, the first set of canonical equations gives $\dot{J}_i = -\frac{\partial E}{\partial \omega_i} = 0$, confirming $J_i = \text{const.}$ The second set gives

$$\dot{\omega}_i = \frac{\partial E}{\partial J_i} = \text{const.} = \nu_i,$$

and hence $\omega_i = \nu_i t + \delta_i$. The entire motion is analytically represented as a **superposition of simple harmonic motions, with the fundamental frequencies $\nu_i$**. The frequencies are obtained from the total energy expressed in terms of the action variables: the partial derivatives of $E$ with respect to the $J_i$ give $n$ new constants which are the frequencies of the system.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Periodicity and Degeneracy)</span></p>

The path of the moving point in the $\omega$-space becomes a straight line, traversed with uniform velocity. Since the $\omega$-space is reduced to a cube of unit length (the $q_i$ are periodic in the $\omega_i$ with period one), the straight line jumps back and forth between the walls.

The criterion for whether or not the motion is strictly periodic is whether we can find linear relations of the form

$$k_1\nu_1 + k_2\nu_2 + \ldots + k_n\nu_n = 0$$

with *integral values* of the $k_i$. If $n - 1$ relations of this nature exist, the line does return after a finite time to its starting point and the motion is strictly periodic. If no such relation exists, the line *fills the entire unit-cube*, coming eventually arbitrarily near to every point of the cube. If $m$ such conditions exist, the moving point will stay on a definite $(n - m)$-dimensional subspace.

For example, in the case of the Kepler ellipses, $E = -\frac{2\pi^2 mk^2}{(J_1 + J_2 + J_3)^2}$, which gives $\nu_1 = \nu_2 = \nu_3$, i.e. maximal degeneracy --- the motion continues along the same ellipse. The introduction of a weak magnetic field (Zeeman effect) removes one degeneracy by letting the plane of the ellipse precess slowly about the polar axis. The second degeneracy is removed by relativistic corrections which cause the perihelion of the ellipse to precess slowly in its own plane.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Connection to Quantum Theory)</span></p>

Delaunay's method, originally developed for astronomical perturbation problems, was eminently adapted to the problems of the earlier quantum theory. Bohr's quantum theory assumed that only certain orbits were allowed for the revolving electron, with the $2n$ constants of integration "quantized" according to certain rules. For these "quantum conditions" Delaunay's treatment of multiply-periodic systems was eminently suitable. The quantum conditions require that the action variables shall be equal to integral multiples of Planck's fundamental constant $h$:

$$J_k = n_k h.$$

The integers $n_k$ are called "quantum numbers." Einstein (in 1917) gave an invariant formulation of the quantum conditions by requiring that the multi-valuedness of $S$ shall be such that for *any* closed curve of the configuration space the change in $S$ for a complete revolution shall be a multiple of $h$:

$$\sum J_k = \sum \Delta S_k = \Delta S = nh.$$

Einstein's invariant formulation led de Broglie (in 1924) to his fundamental discovery of matter waves.

</div>

### 5. The Role of the Partial Differential Equation in the Theories of Hamilton and Jacobi

It was Hamilton who discovered the fundamental partial differential equation of analytical mechanics. It was likewise he who first conceived the idea of a fundamental function which can provide all the mechanical paths by mere differentiations and eliminations. Yet Hamilton's original scheme was practically unworkable. Moreover, Hamilton's principal function satisfies *two* partial differential equations --- an unnecessary complication from the standpoint of integration theory.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Hamilton's Principal Function vs. Jacobi's Generating Function)</span></p>

Hamilton's principal function $W$ connects the initial position $\bar{q}_i$, $\bar{p}_i$ with the final position $q_i$, $p_i$ of the moving phase fluid by a canonical transformation. The function $W$ satisfies two simultaneous partial differential equations:

$$\frac{\partial W}{\partial t} + H\!\left(q_1, \ldots, q_n;\, \frac{\partial W}{\partial q_1}, \ldots, \frac{\partial W}{\partial q_n};\, t\right) = 0,$$

$$-\frac{\partial W}{\partial \bar{t}} + H\!\left(\bar{q}_1, \ldots, \bar{q}_n;\, -\frac{\partial W}{\partial \bar{q}_1}, \ldots, -\frac{\partial W}{\partial \bar{q}_n};\, \bar{t}\right) = 0.$$

Jacobi's great insight was to replace $W$ by the generating function $S$ of an *arbitrary* time-dependent canonical transformation. The function $S$ has to satisfy only *one* partial differential equation (the Hamilton--Jacobi equation), and it does not contain the surplus variable $\bar{t}$. With the help of Jacobi's much less restricted $S$-function, even Hamilton's $W$-function can be obtained. But it would be practically impossible to find the $W$-function directly by solving two simultaneous partial differential equations.

Jacobi's viewpoint is also more natural. Hamilton considered a very specific canonical transformation --- that which connects two states of the moving phase fluid at two different times. Jacobi considers an *arbitrary* canonical transformation, subject only to the condition that it makes the Hamiltonian vanish (or, in the conservative case, makes $H$ equal one of the new variables). The partial differential equation arises as the analytical expression of this condition.

</div>

#### Hamilton's $W$-Function vs. Jacobi's $S$-Function

Let us concentrate on a *conservative* mechanical system. We connect the point $\bar{q}_1, \ldots, \bar{q}_n$ with the point $q_1, \ldots, q_n$ by a path which makes the integral

$$A = \int_{\tau_1}^{\tau_2} \sum_{k=1}^{n} p_k\,dq_k$$

stationary, under the auxiliary condition that the path lies on the energy surface $H(q_1, \ldots, q_n, p_1, \ldots, p_n) - E = 0$. The value of this integral, evaluated along the stationary path, gives Hamilton's principal function $W$:

$$W = W(q_1, \ldots, q_n, \bar{q}_1, \ldots, \bar{q}_n).$$

Hamilton shows that $W$ satisfies the partial differential equation

$$H\!\left(q_1, \ldots, q_n,\, \frac{\partial W}{\partial q_1}, \ldots, \frac{\partial W}{\partial q_n}\right) - E = 0.$$

So far the analogy with Jacobi's $S$-function, which satisfies the same differential equation, seems perfect. Both functions depend on $2n$ variables, and each can be considered as the generating function of a canonical transformation. From the standpoint of integrating the partial differential equation, both functions contain $n$ constants of integration.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Over-Completeness of Hamilton's $W$-Function)</span></p>

A fundamental difference emerges when we compare the constants of integration. In Jacobi's theory, the energy constant $E$ was *one of the new variables* $Q_n$. Aside from $E$, the solution contained but $n - 1$ *constants of integration*. In Hamilton's theory, *all* variables are on an equal footing and the energy constant $E$ plays the role of a given constant and *not* a variable. Hamilton's solution of the partial differential equation is thus not a complete but an **over-complete** solution which contains *one more* constant of integration than a complete solution. This over-completeness is reflected in the fact that the functional determinant of the transformation vanishes:

$$\left\|\frac{\partial^2 W}{\partial q_i\,\partial \bar{q}_j}\right\| = 0.$$

This is a characteristic property of the $W$-function which has no analogue in Jacobi's theory.

</div>

The vanishing determinant means that the canonical transformation generated by $W$ cannot be a regular point-to-point transformation. Instead, $W$ generates a **point-to-line transformation**: the equations $p_i = \frac{\partial W}{\partial q_i}$ are not solvable for the $\bar{q}_i$ because the transformation exists only if the point $q_i$, $p_i$ of the phase space is chosen somewhere on the energy surface $H = E$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Singular Transformations: Point-to-Line)</span></p>

We can illustrate the situation by a linear transformation of ordinary three-dimensional space onto itself. If the determinant of the transformation is different from zero, we have a regular point-to-point correspondence. But if the determinant vanishes, the coordinates $x$, $y$, $z$ (expressed as linear functions of $X$, $Y$, $Z$) satisfy an identity $Ax + By + Cz = 0$, confining the point $x$, $y$, $z$ to a plane. By normalising so that $z = 0$ and setting up the transformation as $x = X + \alpha Z$, $y = Y + \beta Z$, $z = 0$, the plane $z = 0$ in Space I is mapped to the *entire* Space II. Going from Space II to Space I is point-to-point, but from Space I to Space II the foot of every straight line $L$ is mapped to the entire line --- a point-to-line transformation.

*This is the picture of the transformation generated by Hamilton's principal function.* The surface $H = E$ is transformed into itself, because the points $q_k$, $p_k$ are transformed into lines lying on that surface. The second partial differential equation of Hamilton is exactly the statement that not only the point $q_k$, $p_k$ but also the point $\bar{q}_k$, $\bar{p}_k$ lies on the surface $H = E$.

</div>

In the Jacobian theory the point-to-line transformation arises in an entirely different manner. The variable $Q_n$ is distinguished from the other variables: the point $Q_i$, $P_i$ is fixed, *except* that the coordinate $P_n$ is *not* constant. Hence we need to solve only the $n - 1$ equations $P_k = -\frac{\partial S}{\partial Q_k}$ for $k = 1, \ldots, n - 1$; this gives $n - 1$ equations between the $n$ variables $q_i$, which defines a *line*.

#### The Rheonomic Case

For the general rheonomic case, we add the time $t$ to the mechanical variables and require that the "extended action"

$$A = \int_{\tau_1}^{\tau_2} \left(\sum_{i=1}^{n} p_i\,dq_i + p_t\,dt\right)$$

be stationary, under the auxiliary condition that the $C$-point of the extended phase space remains on the "extended energy surface"

$$p_t + H(q_1, \ldots, q_n, t, p_1, \ldots, p_n) = 0.$$

Again we evaluate the integral between the two points $\bar{q}_1, \ldots, \bar{q}_n, \bar{t}$ and $q_1, \ldots, q_n, t$ of the extended configuration space:

$$W = W(q_1, \ldots, q_n, t;\, \bar{q}_1, \ldots, \bar{q}_n, \bar{t}).$$

The transformation equations are $p_i = \frac{\partial W}{\partial q_i}$, $\bar{p}_i = -\frac{\partial W}{\partial \bar{q}_i}$, $p_t = \frac{\partial W}{\partial t}$, $\bar{p}_t = -\frac{\partial W}{\partial \bar{t}}$. Writing the auxiliary condition for both endpoints yields the two partial differential equations satisfied by the $W$-function. Here again the functional determinant of the transformation vanishes, and the singularity in one direction involves the singularity in the other. The one partial differential equation cannot exist without the other, the common tie between the two being provided by the determinantal condition.

The number of integration constants in Hamilton's solution is not $2n$ but $2n + 1$, in agreement with the over-completeness of the $W$-function. This gives the complete solution of the problem of motion, adjustable to arbitrary initial conditions.

### 6. Construction of Hamilton's Principal Function with the Help of Jacobi's Complete Solution

In spite of the different viewpoints which characterise Hamilton's and Jacobi's integration theories, there is a definite relationship between the $W$-function and the $S$-function. Jacobi's integration theory includes the over-determined $W$-function in an indirect way.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Constructing $W$ from $S$)</span></p>

If we know Jacobi's $S$-function, we can construct Hamilton's principal function by differentiation and elimination. We use the *group property* of canonical transformations: the transformation from $\bar{q}_i$, $\bar{p}_i$ to $Q_i$, $P_i$ and then from $Q_i$, $P_i$ to $q_i$, $p_i$ are both canonical. The generating function of the resulting transformation from $\bar{q}_i$, $\bar{p}_i$ *directly* to $q_i$, $p_i$ is equal to the *difference* between the two generating functions. Thus the following remarkable relation holds:

$$W = \Delta S = S(q_1, \ldots, q_n, Q_1, \ldots, Q_n, t) - S(\bar{q}_1, \ldots, \bar{q}_n, Q_1, \ldots, Q_n, \bar{t}).$$

</div>

We then consider the auxiliary conditions $P_i = \frac{\partial S(q, Q, t)}{\partial Q_i}$ and $P_i = \frac{\partial S(\bar{q}, Q, \bar{t})}{\partial Q_i}$. Subtracting these two equations, we obtain the $n$ conditions

$$\frac{\partial\,\Delta S}{\partial Q_i} = 0.$$

These conditions can be used to eliminate the $Q_i$, so that finally $W$ appears as a function of the $q_i$, $t$ and $\bar{q}_i$, $\bar{t}$ alone.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Procedure for Constructing $W$ from a Complete Solution of the Hamilton--Jacobi Equation)</span></p>

Given a complete solution $S(q_1, \ldots, q_n, t, a_1, \ldots, a_n)$ of the Hamilton--Jacobi partial differential equation:

1. Take the difference $\Delta S = S(q_1, \ldots, q_n, t, a_1, \ldots, a_n) - S(\bar{q}_1, \ldots, \bar{q}_n, \bar{t}, a_1, \ldots, a_n)$.

2. Solve the equations $\frac{\partial\,\Delta S}{\partial a_i} = 0$ for the $a_i$.

3. Substitute these $a_i$ in $\Delta S$, thus obtaining $W = \Delta S$ as a function of $q_1, \ldots, q_n, t$ and $\bar{q}_1, \ldots, \bar{q}_n, \bar{t}$. *This gives Hamilton's principal function $W$.*

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Time-Independent $W$ from the Conservative $S$)</span></p>

For a conservative system, if the time-independent $W$-function between two points of the phase space is known:

$$W = W(q_1, \ldots, q_n, \bar{q}_1, \ldots, \bar{q}_n, E),$$

then the time-dependent $W$-function can be obtained as

$$W_1 = -E(t - \bar{t}) + W,$$

provided we eliminate the energy constant $E$ with the help of the equation $\frac{\partial W}{\partial E} = t - \bar{t}$.

</div>

### 7. Geometrical Solution of the Partial Differential Equation --- Hamilton's Optico-Mechanical Analogy

In the previous considerations we assumed that we possess a *complete* solution of the partial differential equation of Hamilton--Jacobi. We now assume much less: we are satisfied if we have a *particular* solution

$$H\!\left(q_1, \ldots, q_n,\, \frac{\partial S}{\partial q_1}, \ldots, \frac{\partial S}{\partial q_n}\right) = E,$$

without any constants of integration. Such a particular solution can be exploited for integrating *one half* of the complete set of canonical equations: we can express the $p_i$ by $p_i = \frac{\partial S}{\partial q_i}$ and substitute into $\dot{q}_i = \frac{\partial H}{\partial p_i}$, reducing our task to $n$ first-order differential equations instead of the original $2n$.

#### The Geometrical Construction

We can construct a particular solution of the partial differential equation by *geometrical* means. For a particle of mass $m$ in a potential field $V(x,y,z)$, the Hamilton--Jacobi equation takes the form

$$\left(\frac{\partial S}{\partial x}\right)^2 + \left(\frac{\partial S}{\partial y}\right)^2 + \left(\frac{\partial S}{\partial z}\right)^2 = 2m(E - V).$$

We consider $S$ as a function of $x$, $y$, $z$ without any constants of integration: $S = S(x,y,z)$. This particular solution permits a simple geometrical interpretation.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Surfaces of Equal Action)</span></p>

The level surfaces $S(x, y, z) = \text{const.}$ are called **surfaces of equal action**. The gradient of $S$ has the direction of the normal to these surfaces. We consider two neighbouring surfaces $S = C$ and $S = C + \epsilon$. The infinitesimal distance $\sigma$ between them, measured along the normal, satisfies

$$|\operatorname{grad} S| = \frac{\epsilon}{\sigma}.$$

Using the Hamilton--Jacobi equation, we obtain

$$\sigma = \frac{\epsilon}{\sqrt{2m(E - V)}}.$$

</div>

This equation gives a method of constructing the surface $S = C + \epsilon$ from the surface $S = C$: at every point of the given surface, lay off the infinitesimal distance $\sigma$ perpendicular to it. Starting from any given basic surface $S = 0$, we construct the neighbouring surface $S = \epsilon$, from this $S = 2\epsilon$, and so on. Eventually a finite portion of space is filled with surfaces $S = \text{const.}$, giving a particular solution of the Hamilton--Jacobi equation.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Ray Property of Mechanical Paths)</span></p>

Since $\mathbf{p} = \operatorname{grad} S$ and the momentum $\mathbf{p} = m\mathbf{v}$ has the direction of the tangent to the path, *the mechanical path of the moving point is perpendicular to the surfaces $S = \text{const.}$* We obtain a family of possible mechanical paths by constructing the orthogonal trajectories of the surfaces $S = \text{const.}$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Conditions for the Ray Property)</span></p>

The ray property holds for a *selected* family of mechanical paths: those which all start with the same total energy $E$ and are perpendicular to a given surface $S = 0$. These paths behave exactly like light rays of optics, which are the orthogonal trajectories of the wave surfaces. The ray property holds only because the paths are derivable from a variational principle. Without the principle of least action, the ray property of mechanical paths could not be established. The orthogonality should be understood in the intrinsic Riemannian sense: for the Jacobian line element $d\sigma = \sqrt{E - V}\,ds$, the intrinsic orthogonality always leads to orthogonality in the ordinary sense.

</div>

#### The Optico-Mechanical Analogy

The construction of wave surfaces on the basis of Huygens' principle has a mechanical counterpart. In fact, the infinitesimal formulation of Huygens' principle coincides with Hamilton's partial differential equation for geometrical optics.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Wave Surfaces in Optics)</span></p>

In an optically heterogeneous medium with refractive index $n(x,y,z)$ and velocity of light $v = c / n$, Huygens' construction produces a family of surfaces $\phi(x,y,z) = C$ such that $\phi$ represents the time light requires to travel from the basic surface $\phi = 0$ to the given point. The function $\phi$ satisfies the partial differential equation of geometrical optics:

$$\left(\frac{\partial\phi}{\partial x}\right)^2 + \left(\frac{\partial\phi}{\partial y}\right)^2 + \left(\frac{\partial\phi}{\partial z}\right)^2 = \frac{n^2}{c^2}.$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Hamilton's Optico-Mechanical Analogy)</span></p>

Comparing the Hamilton--Jacobi equation for mechanics with the equation of geometrical optics, we can correlate a definite optical problem to a given mechanical problem by defining the refractive index according to the law

$$\frac{n}{c} = a\sqrt{2m(E - V)},$$

where $a$ is an arbitrary constant. This establishes a complete analogy: the surfaces of equal action $S = \text{const.}$ in mechanics correspond to the wave surfaces $\phi = \text{const.}$ in optics, and the mechanical trajectories correspond to the optical rays.

</div>

This analogy, discovered by Hamilton, was one of the most startling developments in the history of physics. John Bernoulli had already treated the motion of a particle in the field of gravity as an optical problem with a fictitious refractive index proportional to $\sqrt{E - V}$.

#### Fermat's Principle and Jacobi's Principle

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Equivalence of Fermat's and Jacobi's Principles)</span></p>

In optics, the orthogonal trajectory $T$ from a point $M$ on the wave surface $\phi = 0$ to a point $N$ on $\phi = n\epsilon$ provides the path of quickest arrival: *light travels from $M$ to $N$ in the smallest possible time.* This is Fermat's principle. Minimising the travel time means minimising the integral

$$I = \int_{\tau_1}^{\tau_2}\frac{n}{c}\,\overline{ds}.$$

In mechanics, the corresponding variational principle is Jacobi's principle of least action:

$$A = \int_{\tau_1}^{\tau_2}\sqrt{2m(E - V)}\,ds.$$

These two principles are entirely equivalent via the optico-mechanical correlation.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Kinetic Focus and the "Least" Feature)</span></p>

Our conclusion that Fermat's principle is a true minimum holds in the *local* sense, provided that all along the trajectory $T$ the wave surfaces are well-defined, single-valued surfaces with definite normals. However, rays diverging from a point $M$ may later converge, and two neighbouring trajectories $T$ and $T'$ may intersect at a point $M'$. At that point the wave surface degenerates to a line or a point. Our conclusion of a true minimum holds only up to $M'$ but cannot be extended *beyond* it. The corresponding mechanical situation is called the **kinetic focus** associated with $M$ along $T$. The principle of least action loses its "least" feature if we pass the kinetic focus.

</div>

#### Emission Theory vs. Undulatory Theory

The analogy between mechanics and optics illuminates the historical debate between the *emission theory* and the *undulatory (wave) theory* of light. Both theories can explain the laws of reflection and refraction on purely mechanical grounds, but they give different predictions for the law of refraction:

- The **emission theory** gives $\frac{\sin i}{\sin r} = \text{const.} = \frac{v_2}{v_1}$: light travels *faster* in optically denser media.
- The **undulatory theory** gives $\frac{\sin i}{\sin r} = \text{const.} = \frac{v_1}{v_2}$: light travels *slower* in denser media.

Foucault's famous experiment (1850) showed that light travels in water more slowly than in air, decisively confirming the wave theory.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Dispersion and Chromatic Aberration)</span></p>

An optical field can include at the same point vibrations of various frequencies. If the refractive index $n$ is a function of the frequency, the original wave surface $\phi = 0$ is propagated in various ways for the different frequencies --- the phenomenon of "dispersion." In mechanics, the corresponding situation arises when mechanical paths starting perpendicularly from a basic surface $S = 0$ might be slightly inhomogeneous with respect to the total energy $E$: particles with slightly different total energies trace slightly different families of surfaces of equal action. This produces a mechanical analogue of "chromatic aberration," as occurs in the electron microscope where thermal fluctuations cause slight variations in the electrons' total energy $E$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Surfaces of Equal Action vs. Surfaces of Simultaneous Position)</span></p>

The optico-mechanical analogy holds only between the mechanical *paths* and the light *rays*. The motion *in time* occurs according to entirely different laws. In optics, the infinitesimal distance between neighbouring wave surfaces is $\sigma = v\epsilon$, directly proportional to the velocity. In mechanics, $\sigma = \frac{\epsilon}{mv}$, *inversely* proportional to the velocity. Hence the surfaces of simultaneous position of the moving particles are entirely different from the surfaces of equal action.

</div>

### 8. The Significance of Hamilton's Partial Differential Equation in the Theory of Wave Motion

Hamilton's partial differential equation in optics expresses Huygens' principle in infinitesimal form. Although Huygens' principle is based on undulatory assumptions, the construction of successive wave-fronts with its help is a method of *geometrical*, not physical, optics. To examine more deeply the relation of Hamilton's partial differential equation to the principles of physical optics, we modify somewhat the definition of wave surfaces.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Wave Surface in Physical Optics)</span></p>

In physical optics, the characteristic quantity of a monochromatic wave is a phase factor which varies periodically along the wave. A "wave surface" in the physical-optical sense is defined as a surface of equal phase. Since the wavelength of a monochromatic wave may change from point to point in a heterogeneous medium, what we now demand of the function $\phi$ is *not* that the infinitesimal distance between two neighbouring surfaces shall represent the time of travel (as in geometrical optics), but that it shall measure the number of wavelengths contained between them:

$$\sigma = \frac{\epsilon}{\lambda},$$

where $\lambda = \lambda(x, y, z)$ is the local wavelength.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(The Wave Equation from Hamilton's Partial Differential Equation)</span></p>

The function $\phi$ satisfies the partial differential equation

$$\left(\frac{\partial\phi}{\partial x}\right)^2 + \left(\frac{\partial\phi}{\partial y}\right)^2 + \left(\frac{\partial\phi}{\partial z}\right)^2 = \frac{1}{\lambda^2}.$$

If the wave is monochromatic with frequency $\nu$ so that $\lambda = v/\nu = c/(n\nu)$, this becomes

$$\left(\frac{\partial\phi}{\partial x}\right)^2 + \left(\frac{\partial\phi}{\partial y}\right)^2 + \left(\frac{\partial\phi}{\partial z}\right)^2 = \frac{n^2\nu^2}{c^2}.$$

This can be interpreted as Hamilton's partial differential equation for a medium whose refractive index has been multiplied by $\nu$. Thus the wave surfaces are orthogonal trajectories of the same set of rays --- the ray pattern is completely independent of the frequency, even though the wave surfaces themselves change with frequency.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(From Geometrical to Physical Optics)</span></p>

The physical-optical interpretation changes the picture profoundly compared with geometrical optics. In geometrical optics the surfaces $\phi = \text{const.}$ were surfaces of *equal time* --- they described how a light pulse propagates. In physical optics the same equation now describes surfaces of *equal phase* --- the spacing of the wave crests. If we substitute

$$\psi = e^{2\pi i \phi}$$

we obtain a wave-like entity where the surfaces $\phi = k$ (integer) are the crests of the wave. Writing $\phi = \nu\chi$ where $\chi$ is the "time-type" $\phi$-function of geometrical optics, we get

$$\psi = e^{2\pi i\nu\chi}.$$

This is essentially the Ansatz from which Schrödinger's wave equation derives its origin.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(De Broglie's Hypothesis and Wave Mechanics)</span></p>

The wave-mechanical interpretation of mechanics rests on the insight that Hamilton's partial differential equation is the geometrical-optics limit of a true wave equation. Geometrical optics works perfectly when the wavelength $\lambda$ is infinitesimally small compared to the other dimensions of the problem. But when the wave surfaces are strongly curved --- when features of the system are comparable to the wavelength --- the description by rays breaks down and must be replaced by the full wave equation. In the mechanical analogy, the $S$-surfaces play the role of wave surfaces and the mechanical trajectories play the role of rays. De Broglie's hypothesis (1924) assigns a wavelength $\lambda = h/p$ to a moving particle. For macroscopic bodies $\lambda$ is negligibly small, and the "geometrical-optics" (classical-mechanics) description is fully adequate. But for atoms and electrons, $\lambda$ becomes comparable to the relevant dimensions and classical mechanics must give way to *wave mechanics* --- just as geometrical optics gives way to physical optics.

</div>

#### The Phase Function and De Broglie's Wavelength

The propagation of a wave of definite frequency $\nu$ may be described by

$$u = A\,e^{i\,2\pi[\nu t - \bar{\phi}(x,y,z)]},$$

where $A$ is the amplitude, $\nu$ the frequency, and $\bar{\phi}$ the phase angle. The wave surfaces are characterised by $\bar{\phi}(x,y,z) = \text{const.}$ The propagation of a definite phase requires $\nu\,dt - d\bar{\phi} = 0$, so that the time taken to pass between neighbouring surfaces $\bar{\phi} = \text{const.}$ is $d\bar{\phi}/\nu$, whereas the corresponding time for the surfaces $\phi = \text{const.}$ of the previous section is $d\phi$. Hence $\bar{\phi} = \nu\phi$, and the partial differential equation satisfied by $\bar{\phi}$ becomes

$$|\operatorname{grad}\bar{\phi}|^2 = \frac{\nu^2 n^2}{c^2} = \frac{1}{\lambda^2},$$

if $\lambda$ denotes the wavelength.

The interpretation of $\bar{\phi}$ as a phase function gave rise to de Broglie's fundamental discovery of matter waves. If we assume that there is physical truth in the optico-mechanical analogy --- notwithstanding the obvious discrepancy between the propagation of matter and the propagation of light waves --- we consider the path of the moving electron as a kind of light ray, forming a closed orbit around the nucleus. If there exists a vibration along a closed path, the returning vibration will add up to a resultant amplitude. This amplitude is zero if the phase angle of the returning vibration is not in resonance with the original phase angle. Hence the selection principle: $\Delta\bar{\phi} = n$ (an integer). On the other hand, Einstein's invariant formulation of the quantum conditions gives $\Delta S = nh$. The resonance condition shows that we can obtain a natural and adequate interpretation if we conceive the action function $S$ as a phase function $\bar{\phi}$, in accordance with the correlation

$$\bar{\phi} = S/h.$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(De Broglie Wavelength)</span></p>

Comparing the Hamilton--Jacobi equation $|\operatorname{grad}S|^2 = 2m(E - V)$ with the phase-function equation $|\operatorname{grad}\bar{\phi}|^2 = 1/\lambda^2$, and using the correlation $\bar{\phi} = S/h$, we obtain

$$\frac{1}{\lambda^2} = \frac{2m(E-V)}{h^2} = \frac{m^2 v^2}{h^2}.$$

This gives the celebrated **de Broglie wavelength**:

$$\lambda = \frac{h}{mv},$$

which has influenced so decisively the course of contemporary atomic physics.

</div>

#### From Hamilton's Equation to Schrödinger's Wave Equation

Hamilton's partial differential equation in optics is equivalent to an infinitesimal formulation of Huygens' principle. But Huygens' principle is only an *approximate* consequence of the true principles of physical optics. Light is a vectorial phenomenon, and the adequate description of optical phenomena proceeds on the basis of Maxwell's electromagnetic field equations. Yet many optical phenomena are explainable on the basis of the simpler scalar theory of Fresnel.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Fresnel's Wave Equation)</span></p>

Fresnel's wave equation for a field function $\phi(x, y, z, t)$ in a medium with refractive index $n(x, y, z)$ is

$$\frac{\partial^2\phi}{\partial x^2} + \frac{\partial^2\phi}{\partial y^2} + \frac{\partial^2\phi}{\partial z^2} - \frac{n^2}{c^2}\frac{\partial^2\phi}{\partial t^2} = 0.$$

This is a *linear* differential equation of the *second* order, in contrast to Hamilton's partial differential equation which is of the *first* order but of the *second* degree.

</div>

If the optical vibration occurs with a definite frequency $\nu$, a separation with respect to the time becomes possible by writing $\phi = e^{2\pi i\nu t}\,\psi(x, y, z)$. The function $\psi$ is called the "amplitude function" and satisfies the **amplitude equation**:

$$\frac{\partial^2\psi}{\partial x^2} + \frac{\partial^2\psi}{\partial y^2} + \frac{\partial^2\psi}{\partial z^2} + \frac{4\pi^2}{\lambda^2}\,\psi = 0.$$

The connection between the wave function $\psi$ and the phase function $\bar{\phi}$ is $\psi = e^{-2\pi i\bar{\phi}}$. Substituting this into the amplitude equation, we obtain

$$|\operatorname{grad}\bar{\phi}|^2 = \frac{1}{\lambda^2} - \frac{i}{2\pi}\left(\frac{\partial^2\bar{\phi}}{\partial x^2} + \frac{\partial^2\bar{\phi}}{\partial y^2} + \frac{\partial^2\bar{\phi}}{\partial z^2}\right).$$

The last term becomes negligible as $\lambda \to 0$. Hence Hamilton's partial differential equation in optics is equivalent to Fresnel's wave equation for the case of infinitely small wavelengths (infinitely large frequencies). For small but finite wavelengths, Hamilton's partial differential equation is only an approximation and should be replaced by the correct amplitude equation for $\psi$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Schrödinger's Wave Equation)</span></p>

Schrödinger had in 1927 the original idea of going beyond the analogy between geometrical optics and mechanics, established by Hamilton's partial differential equation, and changing over from the phase function $\bar{\phi}$ to the wave function $\psi$. Introducing de Broglie's wavelength $1/\lambda^2 = 2m(E - V)/h^2$ into the amplitude equation, we obtain **Schrödinger's famous differential equation**:

$$\frac{\partial^2\psi}{\partial x^2} + \frac{\partial^2\psi}{\partial y^2} + \frac{\partial^2\psi}{\partial z^2} + \frac{8\pi^2 m}{h^2}(E - V)\,\psi = 0,$$

which is the basis of modern wave mechanics.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Historical Landmarks from Classical to Wave Mechanics)</span></p>

The great development from classical mechanics to wave mechanics is characterised by the following landmarks: Delaunay's treatment of separable multiply-periodic mechanical systems; the Sommerfeld--Wilson quantum conditions; Einstein's invariant formulation of the quantum conditions; de Broglie's resonance interpretation of Einstein's quantum condition; Schrödinger's logarithmic transformation from the phase function $S$ to the wave function $\psi$.

</div>

### 9. The Geometrization of Dynamics --- Non-Riemannian Geometries --- The Metrical Significance of Hamilton's Partial Differential Equation

Again and again we have found how much the pictorial language of geometry helps in the deeper understanding of the problems of mechanics. The configuration space with its Riemannian metric made it possible to picture a mechanical system, however complicated, as a single point of a properly defined many-dimensional space. The principles which govern the motion of a single particle could be extended to the motion of arbitrary mechanical systems.

The advantage of a geometrical language is particularly conspicuous if the mechanical system is not subject to external forces. In that case the mechanical path can be conceived as a geodesic of the configuration space (Hertz's principle of the straightest path). Furthermore, if the potential energy does not depend on $t$, we can introduce the auxiliary line element $d\sigma = \sqrt{E - V}\,\overline{ds}$, and once more obtain all the mechanical paths which belong to the same total energy $E$ as the geodesics of this manifold.

However, not all mechanical systems are conservative. Nor is it always true that the work function of the external forces is a function of the coordinates alone: the work function of an electron in an external electromagnetic field depends on the velocities $\dot{q}_i$ and may also depend on $t$. Moreover, if we change from classical to relativistic mechanics, the ordinary form of the kinetic energy responsible for the Riemannian structure of the line element can no longer be retained.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Beyond Riemannian Geometry)</span></p>

The close relation of dynamics to geometry continues to hold even under these more general conditions. The Riemannian geometry is not the only possible form of metrical geometry. Riemann's geometry is distinguished by the unique feature that space flattens out in the neighbourhood of an arbitrary point of the manifold, so that the customary Euclidean geometry holds at least in infinitesimal regions. But the constructions of geometry, based on straight lines and angles, do not require this restriction. In view of the general problems of dynamics a more general form of geometry deserves attention, in which the line element $\overline{ds}$ is defined in a more general manner than the Riemannian line element.

</div>

#### The Lagrangian as a Line Element

For the following discussion we abandon the preferred position which the time $t$ ordinarily occupies in mechanics. We add $t$ to the position coordinates $q_i$ by putting $t = q_{n+1}$ and consider all the variables $q_i$ on an equal footing. The subscript $i$ runs from $1$ to $n + 1$. The configuration space has thus $n + 1$, instead of $n$, dimensions.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Non-Riemannian Line Element)</span></p>

We introduce a line element $\overline{ds}$ in the $(n + 1)$-dimensional space by the definition

$$\overline{ds} = F(q_1, \ldots, q_{n+1},\, dq_1, \ldots, dq_{n+1}).$$

The function $F$ is an arbitrary function of the $2n + 2$ variables $q_i$ and $dq_i$, except for the natural restriction that it be a **homogeneous differential form of the first order** in the variables $dq_i$:

$$F(q_1, \ldots, q_{n+1},\, a\,dq_1, \ldots, a\,dq_{n+1}) = a\,F(q_1, \ldots, q_{n+1},\, dq_1, \ldots, dq_{n+1}).$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Equivalence of Dynamics and Geodesics)</span></p>

Let $q_i = f_i(\tau)$ be an arbitrary curve on the $(n+1)$-dimensional manifold, parametrised by $\tau$. By the homogeneity condition, the line element along this curve can be written as $\overline{ds} = F(q_1, \ldots, q_{n+1}, q'_1, \ldots, q'_{n+1})\,d\tau$. The problem of finding the geodesic lines --- minimising the length of a curve between two points --- leads to the variational problem of minimising

$$A = \int_{\tau_1}^{\tau_2} F(q_1, \ldots, q_{n+1},\, q'_1, \ldots, q'_{n+1})\,d\tau.$$

If we choose $q_{n+1} = t$ as the parameter $\tau$, the geodesic equations take the form $q_i = f_i(t)$ and are found by minimising

$$A = \int_{t_1}^{t_2} F(q_1, \ldots, q_n, t,\, \dot{q}_1, \ldots, \dot{q}_n, 1)\,dt.$$

Here we recognise the standard problem of Lagrangian mechanics. **The function $F$ can thus be interpreted mechanically as the Lagrangian function $L$ of analytical mechanics.** Conversely, given a mechanical problem characterised by a Lagrangian $L$, we can define the line element of an $(n+1)$-dimensional space, and the problem of solving the equations of dynamics becomes the problem of finding the geodesics of a certain --- in general non-Riemannian --- manifold.

</div>

#### The Hamiltonian Function and the Identity $H = 0$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Vanishing of the Hamiltonian)</span></p>

Since $F$ is a homogeneous function of the first order in the variables $q'_i$, the Legendre transformation gives

$$H = \sum_{i=1}^{n+1} \frac{\partial F}{\partial q'_i}\,q'_i - F = 0.$$

The vanishing of the Hamiltonian function is always compensated for by the existence of an identity between the $q_i$ and $p_i$:

$$K(q_1, \ldots, q_{n+1},\, p_1, \ldots, p_{n+1}) = 0.$$

This identity has to be considered as an auxiliary condition, and the variational problem becomes: make the integral $A = \int \sum p_i\,dq_i$ stationary under the auxiliary condition $K = 0$.

</div>

#### Hamilton's Principal Function as the Distance Function

In this geometrical interpretation, Hamilton's "principal function" $W$ is particularly significant. Since the "action" is now geometrically interpreted as "arc length," and "least action" means "least length," Hamilton's $W$-function simply means *the distance between the two points $\bar{q}_i$ and $q_i$ of our manifold*.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(The Gradient Condition and Hamilton's Partial Differential Equation)</span></p>

The surfaces $W = \text{const.}$ are by definition concentric spheres with the common centre $\bar{q}_i$. The radius vector $\mathbf{R}$ is everywhere perpendicular to the surface of the sphere. To proceed in the direction of the normal means to proceed in the direction of the radius vector, and by the definition of $W$ as the length of the radius vector:

$$|\operatorname{grad} W| = \left(\frac{dW}{\overline{ds}}\right)_{\!max} = \frac{dr}{dr} = 1.$$

The perpendicularity of the radius vector to a sphere is not a characteristic property of Euclidean geometry alone --- *it is an invariant property of any kind of metrical geometry.*

The condition $|\operatorname{grad} W| = 1$ is equivalent to Hamilton's partial differential equation:

$$K\!\left(q_1, \ldots, q_{n+1},\, \frac{\partial W}{\partial q_1}, \ldots, \frac{\partial W}{\partial q_{n+1}}\right) = 0.$$

This differential equation, together with the boundary condition that $W$ reduces to the line element $F$ when the two points come arbitrarily near to one another, determines Hamilton's principal function uniquely.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Complete Integration in the Homogeneous Case)</span></p>

In the special case where $F$ depends on the $dq_i$ alone (without the $q_i$), we have $\overline{ds} = F(dq_1, \ldots, dq_{n+1})$, and the problem of geodesics is then completely integrable because all the $q_i$ are ignorable variables. The result is

$$W = F(q_1 - \bar{q}_1, \ldots, q_{n+1} - \bar{q}_{n+1}),$$

which gives the distance between *any* two points of the manifold by mere substitution, without any integration. In this geometry, space is **homogeneous** (the properties of space around every point are the same) but in general *not* isotropic (the properties of space depend on the direction). Figures can be freely translated but generally not rotated.

</div>

#### Wave Surfaces and Parallel Surfaces

The wave surfaces $S = \text{const.}$ and their relation to the mechanical paths (or optical rays) can now be established in full generality. The fundamental differential equation which the function $S$ has to satisfy, in the light of the geometrical interpretation, is:

$$|\operatorname{grad} S| = 1.$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Parallel Surfaces and Orthogonal Trajectories)</span></p>

The equation $|\operatorname{grad} S| = 1$ has a striking geometrical significance: *the surface $S = \text{const.}$ has everywhere the same constant distance from a basic surface $S = 0$*. Hence the surfaces $S = \text{const.}$ represent a family of **parallel surfaces**. The wave surfaces, which before were "surfaces of equal action," now become "surfaces of equal distance," i.e. parallel surfaces.

The orthogonality of the wave surfaces $S = \text{const.}$ to the mechanical (or optical) paths can once more be established. Writing $p_i = \frac{\partial S}{\partial q_i}$, or equivalently $p_i = \frac{1}{|\operatorname{grad} S|}\frac{\partial S}{\partial q_i}$, we see that these are exactly the equations of the normals to the surfaces $S = \text{const.}$ Thus *the mechanical paths are orthogonal trajectories of the wave surfaces* --- a result that now applies in a much more general sense than before, belonging to the non-Euclidean and in general even non-Riemannian geometry which is intrinsically associated with the given mechanical problem.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Gauss's Theorem on Orthogonal Trajectories --- Generalised)</span></p>

It was Gauss who, in his immortal "general investigations of curved surfaces" (*Disquisitiones generales circa superficies curvas*, 1827), first discovered that *the orthogonal trajectories of an arbitrary family of parallel surfaces are always geodesics*. His investigations were naturally restricted to the Riemannian form of metric. But actually we have here a theorem which holds equally in *any* kind of metrical geometry.

In the language of optics and mechanics: light rays in an optically homogeneous medium are straight lines (geodesics); the elementary waves in Huygens' construction are *spheres*, not only in the infinitesimal but even in finite regions; the envelopes of these spheres are *parallel surfaces*; and the optical rays --- or mechanical paths --- are *orthogonal trajectories* of this family of parallel surfaces. All this remains true for arbitrary optical and mechanical systems, provided that we operate in a properly defined metrical space.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Non-Riemannian Geometry in Crystal Optics)</span></p>

An amazing form of non-Riemannian geometry is realised in nature in the optical phenomena associated with crystals. In this geometry the line element is defined as

$$\overline{ds} = \frac{\sqrt{dx^2 + dy^2 + dz^2}}{v},$$

where $v$ --- the velocity of light --- is obtained by solving Fresnel's equation

$$\frac{dx^2}{v^2 - v_1^2} + \frac{dy^2}{v^2 - v_2^2} + \frac{dz^2}{v^2 - v_3^2} = 0.$$

The constants $v_1$, $v_2$, $v_3$ are called the "principal velocities." Since this equation is quadratic in $v$, we have a **double-valued metric** --- the coexistence of two different kinds of geometries in the same problem. As a consequence, an arbitrary light ray splits into two rays, giving rise to the phenomenon of "double refraction." The two rays have different velocities and hence obey different geometries. The Huygens construction operates here with double spheres, which in turn produce double envelopes. The intersections of the two kinds of spheres give rise to conical points and singular directions --- the "optical axes" of the crystal.

</div>

## Chapter IX: Historical Survey

This chapter traces the intellectual lineage of analytical mechanics from antiquity through the modern era, focusing on the key figures whose contributions shaped the variational approach to mechanics.

### The Two Traditions in Mechanics

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Vectorial vs. Analytical Mechanics)</span></p>

The history of mechanics can be understood as the interplay between two distinct traditions. **Vectorial mechanics**, rooted in Newton, works directly with forces and momenta as vector quantities. **Analytical mechanics**, originating with Leibniz, replaces these with two scalar quantities --- kinetic energy and potential energy --- and derives the equations of motion from a *principle* rather than from force balance. Despite the positivistic philosophy of the 19th and early 20th centuries, which tended to dismiss the variational principles as mere mathematical curiosities, these principles have proven to be among the deepest and most enduring ideas in all of physics.

</div>

### Aristotle (384--322 B.C.)

The earliest seeds of the principle of virtual work appear in Aristotle's *Physics*. Aristotle observed that in a lever, the forces balance when they are inversely proportional to their velocities. Although framed in terms of velocities rather than displacements, the essential idea --- that a small virtual perturbation can be used to characterise equilibrium --- is already present. The same principle, reformulated as "what is gained in force is lost in velocity," was later employed by Stevinus (1598--1620) to analyse the equilibrium of pulleys.

### Galileo (1564--1642)

Galileo refined Aristotle's principle by recognising that what matters is not the total velocity of a displaced point, but only its component *in the direction of the applied force*. This insight amounts to the concept of **work** as the scalar product of force and displacement. Galileo applied this principle of virtual work to derive the equilibrium of a body on an inclined plane, recovering the same result that Stevinus had obtained via the energy principle. He further extended the principle to hydrostatics, deducing the laws of hydrostatic pressure from the concept of the centre of mass.

### John Bernoulli (1667--1748)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Bernoulli's Generalisation of Virtual Work)</span></p>

Bernoulli was the first to elevate the principle of virtual work to a fully general principle of statics. Earlier applications always involved only two forces related by a proportion. Bernoulli eliminated the need for proportions and introduced the key quantity: the *product of the force and the virtual velocity in the direction of the force*, taken with the appropriate sign. In a letter to Varignon in 1717, Bernoulli announced that for equilibrium, the sum of all such products over all forces and all possible infinitesimal displacements must vanish. This formulation holds for *any* forces under *any* mechanical conditions.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Bernoulli's Optico-Mechanical Analogy)</span></p>

Bernoulli also drew a remarkable analogy between the motion of a particle in a force field and the propagation of light in an optically heterogeneous medium. He attempted to derive a mechanical theory of the refractive index on this basis. This observation anticipates Hamilton's later discovery that the principle of least action in mechanics and Fermat's principle of shortest time in optics are strikingly analogous, allowing one to interpret optical phenomena in mechanical terms and vice versa.

</div>

### Newton (1642--1727)

Newton formulated the fundamental equations of motion, identifying **momentum** and **the acting force** as the key dynamical quantities. His laws completely determine the motion of an isolated particle. The difficulty arises with systems of interacting particles, where not all forces are known a priori. The Third Law (action and reaction) partly resolves this for rigid bodies by eliminating unknown internal forces, but for systems with more complicated kinematical constraints the Newtonian approach becomes increasingly unwieldy. Newton's followers elevated his laws to the status of absolute universal truths --- a dogmatism that Newton himself would likely not have endorsed --- and this reverence for vectorial particle mechanics delayed the appreciation of the analytical methods developed in the 18th century.

### Leibniz (1646--1716)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Leibniz's Vis Viva and the Analytical Viewpoint)</span></p>

Where Newton measured the dynamical effect of a force by the rate of change of momentum, Leibniz advocated a different measure: the **vis viva** (living force), which --- up to a factor of $1/2$ --- is what we now call **kinetic energy**. Leibniz replaced the Newtonian equation with the statement that "the change of kinetic energy equals the work done by the force." This shift had profound consequences:

- Both kinetic and potential energy are **scalar** quantities, easily generalised from a single particle to an arbitrary system without needing to isolate individual particles.
- The work of all forces (including internal ones) could be captured by a single scalar function --- the **potential energy**, a term coined by J. Rankine in the mid-19th century.
- The variational approach, applying the principle of variation to these two scalar quantities, could account for *all* forces in a system, however elaborate, in a unified way.

The Newton--Leibniz controversy over force is thus fundamentally a question of *method*: Newtonian vectors suit the direct treatment of individual particles, while Leibniz's scalars provide the cornerstones of analytical mechanics.

</div>

### D'Alembert (1717--1785)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(D'Alembert's Principle)</span></p>

D'Alembert achieved a crucial conceptual advance by reducing every problem of *motion* to a problem of *equilibrium*. His device is to introduce an additional "force" --- the **inertial force** (equal to the mass times the acceleration, directed opposite to the motion) --- and to add it to the applied forces. The resulting system of forces is then in equilibrium at every instant, making the principle of virtual work applicable to *dynamics*, not just statics. In this way, all the equations of motion for an arbitrary mechanical system are encompassed by a single variational principle.

</div>

### Maupertuis (1698--1759)

Maupertuis conceived the bold hypothesis that nature always acts so as to minimise a certain quantity called the **action**. While the universality of this assumption was admirable and ahead of its time, Maupertuis lacked the mathematical power to properly define the quantity to be minimised. His treatment of elastic collisions, though yielding the correct result, relied on an incorrect method. His more convincing contribution was showing that Fermat's principle of least time for light refraction can be replaced by the principle of least action --- a result that Bernoulli had anticipated earlier.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Maupertuis--Euler Priority Dispute)</span></p>

The priority of Maupertuis' discovery was challenged by Koenig, who claimed Leibniz had expressed the same idea in a private letter (never produced). Euler vigorously defended Maupertuis, despite having himself discovered the principle at least a year earlier and in a more correct form. Euler knew that both the actual and the varied motions must satisfy conservation of energy --- without this auxiliary condition, Maupertuis's action integral loses its meaning. Euler's generous self-effacement in crediting Maupertuis as the inventor is remarkable in the history of science.

</div>

### Euler (1707--1783)

Euler made foundational contributions to both the mechanics of rigid bodies (introducing angular velocity components as kinematical variables) and to the calculus of variations. He initiated the systematic study of **isoperimetric problems** --- maximum-minimum problems for integrals --- which had attracted the best minds since the early days of the infinitesimal calculus. Euler discovered a differential equation that implicitly solves a broad class of such problems. Although he did not state the principle of least action in the exact form later given by Lagrange, his application of the variational principle to mechanical phenomena is equivalent to Lagrange's explicit formulation.

### Lagrange (1736--1813)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Lagrange's Contributions)</span></p>

Lagrange's achievements, paralleling and extending Euler's work, include:

- **Calculus of variations:** Lagrange independently solved isoperimetric problems using entirely new methods, developing the calculus of variations as a distinct branch of mathematics.
- **Generalised coordinates:** He recognised that the variational approach allows the position of a mechanical system to be described by *any* convenient set of parameters, without being tied to Cartesian coordinates. This freedom, combined with d'Alembert's principle, is the chief advantage of variational mechanics.
- **Invariance under coordinate transformations:** The Lagrangian equations retain their form under arbitrary coordinate changes --- a property of supreme importance.
- **The method of Lagrange multipliers:** This technique for incorporating auxiliary constraints is one of Lagrange's most enduring inventions and plays a central role throughout theoretical mechanics.
- **Mécanique Analytique (1788):** In this landmark treatise, Lagrange showed that any mechanical problem can be solved by purely algebraic and analytical operations, without recourse to geometric constructions or physical intuition, provided the kinetic and potential energies are given in abstract analytical form.

Hamilton himself called Lagrange the "Shakespeare of mathematics" on account of the extraordinary elegance and depth of the Lagrangian methods.

</div>

### Hamilton (1805--1865)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Hamilton's Contributions)</span></p>

Hamilton opened an entirely new world beyond Lagrange's discoveries:

- **Canonical equations:** By treating positions $q_i$ and momenta $p_i$ as independent variables, Hamilton transformed the second-order Lagrangian equations into a set of *first-order* differential equations that are linear and separated in the derivatives. Jacobi named these the **canonical equations**, recognising them as the simplest and most natural form for the equations of a variational problem.
- **Hamilton's principle:** By transforming d'Alembert's principle, Hamilton gave the first *exact* formulation of the principle of least action, valid beyond the conservative (scleronomic) case to which Euler and Lagrange's version was restricted.
- **The optico-mechanical analogy:** One of Hamilton's deepest insights is that problems of mechanics and geometrical optics can be handled from a unified viewpoint. He introduced a **characteristic (principal) function** that, by mere differentiation, yields both the path of a moving particle and the path of an optical ray. Moreover, in both optics and mechanics, this characteristic function satisfies the same partial differential equation --- so solving the Hamilton--Jacobi equation under appropriate boundary conditions is equivalent to solving the equations of motion.

</div>

### Jacobi (1804--1851)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Jacobi's Contributions)</span></p>

Jacobi immediately grasped the power and brilliance of Hamilton's methods and extended them significantly:

- **Canonical transformations:** Jacobi developed the transformation theory of the canonical equations. He interpreted Hamilton's characteristic function within this framework and showed that Hamilton's function is just one special case of a generating function for canonical transformations.
- **Generalised Hamilton--Jacobi theory:** Jacobi proved that *any* complete solution of Hamilton's partial differential equation --- not just one satisfying specific boundary conditions --- suffices for the full integration of the equations of motion. This greatly extended the usefulness of the theory.
- **Reformulation of least action:** Jacobi gave a new formulation of the principle of least action for the time-independent (conservative) case. He criticised the Euler--Lagrange formulation because the limits of the integral did not vary between fixed bounds, and showed how eliminating time from the variational integral leads to a new principle determining the *path* of the motion (a geodesic in configuration space) without reference to *how* the motion unfolds in time. This approach directly connects mechanics to Fermat's principle of least time in optics.

</div>

### Gauss (1777--1855)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Gauss's Principle of Least Constraint)</span></p>

Gauss introduced the **principle of least constraint**, which stands apart from the other variational principles in that it does not use a time-integral. Instead, at every instant, one defines a positive quantity called the **constraint** (depending on the positions, velocities, and accelerations at that instant) and determines the accelerations by minimising this quantity. The Gauss principle has the distinction of yielding a genuine *minimum* rather than merely a stationary value. However, because the constraint involves accelerations in addition to positions and velocities, it does not share the full analytical elegance of the Lagrangian or Hamiltonian formulations.

Hertz later gave a geometrical interpretation of Gauss's constraint as the **geodesic curvature** of the path in a $3N$-dimensional **configuration space**. Under this interpretation, the Gauss principle becomes the "principle of the straightest path," bringing it into close relationship with Jacobi's geometric formulation of least action.

</div>

### Later Developments

Several further contributions enriched the edifice of analytical mechanics:

- **Routh** and independently **Helmholtz** identified the importance of **cyclic (ignorable) variables** --- coordinates that do not appear explicitly in the Lagrangian --- and developed the general procedure for their elimination, leading to the Routhian reduction.
- **J. J. Thomson** explored the idea of a mechanics entirely without forces, in which all potential energy arises from the kinetic energy of hidden motions associated with ignorable coordinates.
- **Hertz** pursued this idea systematically, conceiving the potential energy as hidden kinetic energy and picturing the motion of an arbitrary mechanical system as the trajectory of a single point in a high-dimensional configuration space.
- **S. Lie** introduced the group-theoretic viewpoint into canonical transformations, paying special attention to the group of infinitesimal transformations.
- **H. Poincaré** studied the integral invariants of the canonical equations and made fundamental contributions to perturbation theory in celestial mechanics, especially the three-body problem. His investigation of the geometry of configuration space "in the large" led him to pioneering work in topology.
- **G. Appell** investigated the analytical treatment of **non-holonomic systems** --- systems whose constraints involve velocities and cannot be reduced to constraints on positions alone.

### Analytical Mechanics and Contemporary Physics

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Relativity and Quantum Theory)</span></p>

The two great revolutions of 20th-century physics --- **relativity** and **quantum mechanics** --- are both deeply connected to analytical mechanics.

**General relativity** showed that Newtonian mechanics is only an approximation valid at low velocities. However, the variational formulation survived intact: only the form of the Lagrangian had to be modified. The complete independence of a variational principle from any special coordinate system made it particularly suited for formulating the covariant field equations of general relativity, which demand invariance under arbitrary coordinate transformations.

**Quantum mechanics** is likewise closely allied to the Hamiltonian formulation. Bohr's theory of electronic orbits made excellent use of the Hamiltonian methods, and the importance of separable systems for the old quantum conditions was already clear from the work of Sommerfeld (1916) and Epstein on the Stark effect. The wave-mechanical theories of Schrödinger, Heisenberg, and Dirac all grew out of Hamiltonian ideas:

- Heisenberg, Born, and Jordan gave the $q$- and $p$-variables a **matrix character**.
- Dirac treated conjugate variables as **non-commutative** quantities.
- Schrödinger reinterpreted the Hamilton--Jacobi equation as a **wave equation**, taking Hamilton's optico-mechanical analogy as his starting point.

Despite the radical departures of modern physics from classical mechanics, the basic feature of the differential equations of wave mechanics is their **self-adjoint** character, which means they are derivable from a variational principle. The variational principles of mechanics thus continue to underpin the description of all natural phenomena.

</div>
