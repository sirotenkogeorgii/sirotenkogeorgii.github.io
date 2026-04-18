### Computing eigenvalues

As we already mentioned, eigenvalues are computed only by numerical iterative methods and finding them as roots of the characteristic polynomial is not an efficient approach. In this section we will show a simple estimate for eigenvalues and a simple method for computing the largest eigenvalue.

Since numerical methods are iterative and compute eigenvalues only with a certain precision, it is difficult to express a priori the exact number of operations they will perform. Nevertheless, current methods for both symmetric and non-symmetric matrices have practically cubic complexity. This means they require asymptotically $\alpha n^3$ operations, where $n$ is the dimension of the matrix and $\alpha > 0$ is the corresponding coefficient.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark 10.54 — Alternative form of the spectral decomposition)</span></p>

Let a symmetric $A \in \mathbb{R}^{n \times n}$ have eigenvalues $\lambda_1, \ldots, \lambda_n$ and corresponding orthonormal eigenvectors $x_1, \ldots, x_n$. Thus in the spectral decomposition $A = Q \Lambda Q^\top$ we have $\Lambda_{ii} = \lambda_i$ and $Q_{*i} = x_i$. If we decompose $\Lambda$ as a sum of simpler diagonal matrices

$$\Lambda = \sum_{i=1}^n \lambda_i e_i e_i^\top,$$

then the matrix $A$ can be expressed as

$$A = Q \Lambda Q^\top = Q \left( \sum_{i=1}^n \lambda_i e_i e_i^\top \right) Q^\top = \sum_{i=1}^n \lambda_i Q e_i e_i^\top Q^\top = \sum_{i=1}^n \lambda_i Q_{*i} Q_{*i}^\top = \sum_{i=1}^n \lambda_i x_i x_i^\top.$$

The form 

$$A = \sum_{i=1}^n \lambda_i x_i x_i^\top$$

is thus an alternative expression of the spectral decomposition, in which we decompose the matrix $A$ as a sum of $n$ matrices of rank 0 or 1. Moreover, $x_i x_i^\top$ is the projection matrix onto the line $\operatorname{span}\lbrace x_i \rbrace$, so from a geometric perspective we can view the map $x \mapsto Ax$ as a sum of $n$ maps, where in each one we project onto a line (orthogonal to the others) and scale by the value $\lambda_i$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Theorem 10.58 — Gershgorin discs)</span></p>

Every eigenvalue $\lambda$ of a matrix $A \in \mathbb{C}^{n \times n}$ lies in a disc centered at $a_{ii}$ with radius $\sum_{j \neq i} \lvert a_{ij}\rvert$ for some $i \in \lbrace 1, \ldots, n \rbrace$.

</div>

*Proof.* Let $\lambda$ be an eigenvalue and $x$ the corresponding eigenvector, so $Ax = \lambda x$. Let the $i$-th component of $x$ have the largest absolute value, i.e., $\lvert x_i\rvert = \max_{k=1,\ldots,n} \lvert x_k\rvert$. Since the $i$-th equation has the form 

$$\sum_{j=1}^n a_{ij} x_j = \lambda x_i$$

dividing by $x_i \neq 0$ we get 

$$\lambda = a_{ii} + \sum_{j \neq i} a_{ij} \frac{x_j}{x_i}$$

and therefore 

$$\lvert \lambda - a_{ii}\rvert = \left\| \sum_{j \neq i} a_{ij} \frac{x_j}{x_i} \right\| \le \sum_{j \neq i} \lvert a_{ij}\rvert \frac{\lvert x_j\rvert}{\lvert x_i\rvert} \le \sum_{j \neq i} \lvert a_{ij}\rvert$$

The theorem gives a simple but coarse estimate on the magnitude of eigenvalues (there also exist improvements, e.g., Cassini ovals, etc.). Nevertheless, in some applications such an estimate may suffice.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(10.59)</span></p>

Consider 

$$A = \begin{pmatrix} 2 & 1 & 0 \\ -2 & 5 & 1 \\ -1 & -2 & -3 \end{pmatrix}$$

The eigenvalues of $A$ are $\lambda_1 = -2.78$, $\lambda_2 = 3.39 + 0.6i$, $\lambda_3 = 3.39 - 0.6i$. The Gershgorin discs are: the disc centered at 2 with radius 1, the disc centered at 5 with radius 3, and the disc centered at $-3$ with radius 3. All eigenvalues lie inside one of the discs.

<figure>
  <img src="{{ '/assets/images/notes/linear-algebra/gershgorin_discs.png' | relative_url }}" alt="GPU global memory" loading="lazy">
  <figcaption>Gershgorin discs</figcaption>
</figure>

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(10.60 — Three applications of Gershgorin discs)</span></p>

1. *Stopping criterion for iterative methods.* For example, the Jacobi method for computing eigenvalues consists of gradually reducing the off-diagonal entries of a symmetric matrix so that the matrix converges to a diagonal matrix. The Gershgorin discs then give an upper bound on the accuracy of the computed eigenvalues. If, for instance, a matrix $A \in \mathbb{R}^{n \times n}$ is nearly diagonal in the sense that all off-diagonal entries are less than $10^{-k}$ for some $k \in \mathbb{N}$, then the diagonal entries approximate the eigenvalues with accuracy $10^{-k}(n-1)$.
2. *Diagonally dominant matrices.* Gershgorin discs also give the following sufficient condition for the nonsingularity of a matrix 
   
   $$A \in \mathbb{C}^{n \times n}$: $\lvert a_{ii}\rvert > \sum_{j \neq i} \lvert a_{ij}\rvert \quad \forall i = 1, \ldots, n$$
   
   In this case the discs do not contain the origin, and therefore zero is not an eigenvalue of $A$. Matrices with this property are called diagonally dominant.
3. *Markov matrices.* Let $A$ be the Markov matrix from Example 10.57. All Gershgorin discs of the matrix $A^\top$ have their center at a point in the interval $[0, 1]$ and their right edge touches the value 1 on the real axis. This proves that $\rho(A) \le 1$, and therefore 1 is indeed the largest eigenvalue of the matrix $A$ in absolute value.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(10.61 — Power method)</span></p>

Input: matrix $A \in \mathbb{C}^{n \times n}$.

1. Choose $o \neq x_0 \in \mathbb{C}^n$, $i \coloneqq 1$,
2. **while not** stopping criterion satisfied **do**
3. &emsp; $y_i \coloneqq A x_{i-1}$,
4. &emsp; $x_i \coloneqq \frac{1}{\lVert y_i\rVert_2} y_i$,
5. &emsp; $i \coloneqq i + 1$,
6. **end while**

Output: $\lambda_1 \coloneqq x_{i-1}^\top y_i$ is an estimate of the eigenvalue, $v_1 \coloneqq x_i$ is an estimate of the corresponding eigenvector.

</div>

The method terminates when the value $x_{i-1}^\top y_i$ or the vector $x_i$ stabilizes; then $x_i \approx x_{i-1}$ is an estimate of the eigenvector and 

$$x_{i-1}^\top y_i = x_{i-1}^\top A x_{i-1} \approx x_{i-1}^\top \lambda x_{i-1} \approx \lambda$$

is an estimate of the corresponding eigenvalue. The method can be slow, the error and convergence rate are difficult to estimate, and furthermore the initial choice of $x_0$ matters significantly. On the other hand, it is robust (rounding errors have little effect) and easily applicable to large sparse matrices. It does not always converge, but under certain assumptions convergence can be guaranteed.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(10.63 — Convergence of the power method)</span></p>

Let $A \in \mathbb{R}^{n \times n}$ with eigenvalues $\lvert \lambda_1\rvert > \lvert\lambda_2\rvert \ge \ldots \ge \lvert\lambda_n\rvert$ and corresponding linearly independent eigenvectors $v_1, \ldots, v_n$ of unit size. Let $x_0$ have a nonzero component in the direction of $v_1$. Then $x_i$ converges (up to a scalar multiple) to the eigenvector $v_1$ and $x_{i-1}^\top y_i$ converges to the eigenvalue $\lambda_1$.

</div>

*Proof.* Since the vectors $v_1, \ldots, v_n$ form a basis of $\mathbb{R}^n$, we can express the vector $x_0$ as 

$$x_0 = \sum_{j=1}^n \alpha_j v_j$$

where $\alpha_1 \neq 0$ by assumption. Then 

$$A^i x_0 = \sum_{j=1}^n \alpha_j \lambda_j^i v_j = \lambda_1^i \left( \alpha_1 v_1 + \sum_{j \neq 1} \alpha_j \left(\frac{\lambda_j}{\lambda_1}\right)^i v_j \right)$$

Since the vectors $x_i$ are successively normalized, the factor $\lambda_1^i$ does not matter. The remaining vector gradually converges to $\alpha_1 v_1$, because $\left\|\frac{\lambda_j}{\lambda_1}\right\| < 1$ and therefore $\left\|\frac{\lambda_j}{\lambda_1}\right\|^i \to 0$ as $i \to \infty$.

From the proof we see that the convergence rate depends strongly on the ratio $\left\|\frac{\lambda_2}{\lambda_1}\right\|$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(10.64 — On eigenvalue deflation)</span></p>

Let $A \in \mathbb{R}^{n \times n}$ be symmetric, $\lambda_1, \ldots, \lambda_n$ its eigenvalues, and $v_1, \ldots, v_n$ the corresponding orthonormal eigenvectors. Then the matrix $A - \lambda_1 v_1 v_1^\top$ has eigenvalues $0, \lambda_2, \ldots, \lambda_n$ and eigenvectors $v_1, \ldots, v_n$.

</div>

*Proof.* By Remark 10.54 we can write 

$$A = \sum_{i=1}^n \lambda_i v_i v_i^\top$$

Then 

$$A - \lambda_1 v_1 v_1^\top = 0 v_1 v_1^\top + \sum_{i=2}^n \lambda_i v_i v_i^\top$$

which is the spectral decomposition of the matrix $A - \lambda_1 v_1 v_1^\top$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Mental Model of Power Iteration)</span></p>

A good mental image is an ellipse. If a transformation turns the unit circle into an ellipse, then the longest axis of that ellipse is the dominant eigenvector direction. If you keep applying the transformation, the shape gets more and more elongated along that axis, and almost any point you track ends up pointing along it.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>More detailed intuition on power iteration</summary>

Think of a linear map $A$ as something that stretches space by different factors in a few special directions: the eigenvector directions.

If you start with any vector $x_0$, that vector can be decomposed into eigenvector components:

$$x_0 = c_1 v_1 + c_2 v_2 + \cdots + c_n v_n$$

where $v_1$ is the eigenvector with the largest absolute eigenvalue:

$$|\lambda_1| > |\lambda_2| \ge \cdots$$


After applying $A$ once,

$$Ax_0 = c_1 \lambda_1 v_1 + c_2 \lambda_2 v_2 + \cdots$$

After applying it $k$ times,

$$A^k x_0 = c_1 \lambda_1^k v_1 + c_2 \lambda_2^k v_2 + \cdots$$

The key intuition is:

**each eigen-direction gets amplified at its own rate.**
The direction with the biggest stretch factor grows fastest, so after repeated applications it overwhelms all the others.

Geometrically, imagine every step as:

* take your current arrow,
* apply $A$,
* the part of the arrow pointing in the dominant eigenvector direction gets multiplied by the biggest factor,
* all other directional parts grow more slowly,
* so the arrow gradually “tilts” toward the most amplified direction.

Even if analytically you already know this, the geometric picture is:

**the power method is a competition between directional components, and the dominant eigenvector wins because it is the direction that $A$ magnifies the most.**

A good mental image is an ellipse. If a transformation turns the unit circle into an ellipse, then the longest axis of that ellipse is the dominant eigenvector direction. If you keep applying the transformation, the shape gets more and more elongated along that axis, and almost any point you track ends up pointing along it.

Why “almost any” vector?

Because your starting vector must contain at least a tiny component in that dominant direction. If it is exactly orthogonal to that eigenvector component, then there is nothing for the dominant direction to amplify.

A few extra intuitive remarks:

* **Normalization does not change the direction**, it only prevents the vector from exploding in length.
* If the largest eigenvalue is negative, the vector flips sign each iteration, but the **line** still converges to the dominant eigenvector direction.
* If two eigenvalues have the same largest magnitude, there is no single winner, so the method may not settle to one direction.

So the shortest intuition is:

**Repeated application of $A$ acts like repeated selective amplification. The dominant eigenvector is the direction most amplified by $A$, so every generic vector gets visually pulled toward that direction because all competing components become negligible in comparison.**

If you want, I can also give you a 2D picture-based explanation with a concrete matrix.

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(10.65 — On eigenvalue deflation for a general matrix)</span></p>

Let $\lambda$ be an eigenvalue and $x$ the corresponding eigenvector of a matrix $A \in \mathbb{R}^{n \times n}$. Extend $x$ to a nonsingular matrix $S$ so that $S_{*1} = x$. Then

$$S^{-1}AS = S^{-1}A(x \mid \cdots) = S^{-1}(\lambda x \mid \cdots) = (\lambda e_1 \mid \cdots) = \begin{pmatrix} \lambda & \cdots \\ o & A' \end{pmatrix}.$$

By similarity, the matrix $A'$ has the same eigenvalues as $A$, only $\lambda$ has multiplicity reduced by one. Therefore the remaining eigenvalues of matrix $A$ can be found using $A'$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(10.66 — Google search engine and PageRank)</span></p>

Consider a web network with $N$ web pages. The goal is to determine the importances $x_1, \ldots, x_N$ of individual pages. The basic idea of the authors of Google's PageRank is to set the importance of the $i$-th page to be proportional to the sum of the importances of the pages linking to it. We therefore solve the equation $x_i = \sum_{j=1}^N \frac{a_{ij}}{b_j} x_j$, $i = 1, \ldots, N$, where $a_{ij} = 1$ if the $j$-th page links to the $i$-th page (otherwise 0) and $b_j$ is the number of links from the $j$-th page. In matrix form $A'x = x$, where $a'\_{ij} \coloneqq \frac{a_{ij}}{b_j}$.

Thus $x$ is an eigenvector of the matrix $A'$ corresponding to eigenvalue 1. The eigenvalue 1 is dominant, which is easily seen from the Gershgorin discs for the matrix $A'^\top$ (the column sums of $A'$ equal 1, so all Gershgorin discs have their rightmost point at 1). By Perron's theorem 10.56, the eigenvector $x$ is nonnegative.

In practice, the matrix $A'$ is huge, on the order of $\approx 10^{10}$, and at the same time sparse (most values are zeros). Therefore the power method is well-suited for computing $x$, requiring approximately $\approx 100$ iterations. In practice, the matrix $A'$ is also slightly modified to make it stochastic, aperiodic, and irreducible.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Remark 10.67 — Further applications in graph theory)</span></p>

In conclusion, let us mention the broad use of eigenvalues in graph theory. The eigenvalues of the adjacency matrix and the Laplacian matrix of a graph reveal much about the structure of the graph. They are used to estimate the size of the so-called "bottleneck" in a graph, which is a set of vertices with relatively few outgoing edges. They also provide various estimates on the size of independent sets in graphs and other characteristics.

</div>
