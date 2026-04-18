### Matrix Differentiation and Other Tricks

1. x^\top A y
2. $\int \frac{1}{\sin(x)} \, dx = \ln |\tan\left(\frac{x}{2}\right)| + C$
3. Random vector $X \sim \mathcal{N}(\mu, \Sigma)$, then for a constant vector $k$, the expectation of $\exp(k^\top X)$ is given by: $\mathbb{E}[\exp(k^\top X)] = \exp(k^\top \mu + \frac{1}{2} k^\top \Sigma k)$
4. $\mathbb{E}(\text{tr}(A)) = \text{tr}(\mathbb{E}(A))$
5. $\frac{\partial \text{tr}(CXB)}{\partial X} = C^\top B^\top$
6. $\frac{\partial \text{tr}(X^\top C)}{\partial X} = C$
7. $\frac{\partial \text{tr}(X^\toBXC)}{\partial X} = BXC + B^\top XC^\top$
8. $\frac{\partial \det \Sigma}{\partial \Sigma} = \Sigma^{-1}$
9. $\frac{\partial a^\top \Sigma^{-1} a}{\partial \Sigma} = -\Sigma^{-1}aa^\top\Sigma^{-1}$
10. $d(\Sigma−1)=−\Sigma−1(d\Sigma)\Sigma−1$
11. \text{Cov}(\bar{X}) = \frac{\text{Cov}(X)}{n}
12. \frac{\partial arctan(x)}{\partial x} = 1 / (1 + x^2)
13. $\|Y - XB\|_F^2 = \text{tr}\!\big((Y - XB)^\top(Y - XB)\big)$
14. $\|A\|_F^2 = \text{tr}(A^\top A)$
15. $\operatorname{Cov}\left(\sum_i Y_i\right) = \sum_i \operatorname{Cov}(Y_i) + \sum_{i\neq j}\operatorname{Cov}(Y_i,Y_j)$
16. \text{Cov}(aY) = a^2\text{Cov}(Y)
17. \text{Cov}(AX) = \text{Cov}(AX, AX) = text{Cov}(AX, X)A^\top = A text{Cov}(X, X)A^\top = A\text{Cov}(X)A^\top
18. $f(x) = x^\top A x + b^\top x + c$
   1. $\nabla f(x) = (A + A^\top) x + b$
   2. $\nabla^2 f(x) = A + A^\top$
19. $\frac{\partial \text{arctan}(x)}{\partial X} = \frac{1}{1+x^2}$
20. $\frac{\partial x^\top A x}{\partial A} = xx^\top$
21. $\frac{\partial A^\top A}{\partial A} = 2A$
22. $\frac{\partial Ax}{\partial A} = x^\top$
23. $\frac{\partial A^{-1}}{\partial A} = -A^{-1}(dA)A^{-1}$
24. $\frac{\partial \log \det A}{\partial A} = A^{-T}$
25. $(A + UCV)^{-1} = A^{-1} - A^{-1}U(C^{-1} + VA^{-1}U)^{-1}VA^{-1}$


### Lingebra

1. The decomposition $A=Q\Lambda Q^\top$, where $\Lambda$ is diagonal and $Q$ is orthogonal, exists only for symmetric matrices.

#### Maticové funkce a mocninné řady + Analytical Function

#TODO:

#TODO: Write about the difference between the diagonalization and spectral decomposition
#TODO: Make anki on Maticové funkce a mocninné řady + Analytical Function
#TODO: Read about change of basis

* $\sqrt{\mathbb{E}[(X-\mu)^2]} \geq \mathbb{E}[\sqrt{(X-\mu)^2}]$

#### Probability and Statistics

1. If $S = \sum_i^n X_i$ and $X_i \sim \text{Exp}(\lambda)$, then
   1. $S \sim \text{Gamma}(n, \lambda)$
   2. $\mathbb{E}[\dfrac{1}{S}] = \dfrac{\lambda}{n-1}$









DSTML Remember:

1. Hamiltonian for VL System
2. Hamiltonian function and Potential function are twice continuously differentiable (C^2)