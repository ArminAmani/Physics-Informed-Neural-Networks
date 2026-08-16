# Physics-Informed Neural Network for the Viscous Burgers Equation

This project implements a Physics-Informed Neural Network (PINN) in PyTorch for the one-dimensional viscous Burgers equation.

The Burgers equation is a canonical nonlinear partial differential equation that combines nonlinear convection and viscous diffusion. It provides a compact benchmark for studying numerical PDE solution methods, automatic differentiation, and physics-informed machine learning.

---

## 1. Problem Definition

The one-dimensional viscous Burgers equation considered in this project is

```math
\frac{\partial u}{\partial t}
+
u\frac{\partial u}{\partial x}
-
\nu\frac{\partial^2 u}{\partial x^2}
=
0
```

where `u(t,x)` is the unknown scalar field and `nu` is the kinematic viscosity.

For the present problem,

```math
\nu
=
\frac{0.01}{\pi}
```

The spatial and temporal domains are

```math
x \in [-1,1]
```

and

```math
t \in [0,1]
```

respectively.

---

## 2. Initial and Boundary Conditions

The initial condition is

```math
u(0,x)
=
-\sin(\pi x)
```

Homogeneous Dirichlet boundary conditions are imposed at both ends of the spatial domain.

At the left boundary,

```math
u(t,-1)
=
0
```

and at the right boundary,

```math
u(t,1)
=
0
```

The PINN is therefore trained to satisfy three sources of physical information:

1. the Burgers equation inside the space-time domain;
2. the prescribed initial condition at `t = 0`;
3. the prescribed boundary conditions at `x = -1` and `x = 1`.

---

## 3. PINN Formulation

The neural network approximates the mapping

```math
(t,x)
\rightarrow
u_{\theta}(t,x)
```

where `theta` denotes the trainable neural-network parameters.

The network receives time and space as inputs and returns a continuous approximation of the scalar solution field.

---

## 4. Neural Network Architecture

The implementation uses a fully connected feed-forward neural network.

| Component | Configuration |
|---|---|
| Input variables | `t`, `x` |
| Number of inputs | 2 |
| Hidden layers | 5 |
| Neurons per hidden layer | 40 |
| Activation function | `Tanh` |
| Output variable | `u(t,x)` |
| Number of outputs | 1 |
| Weight initialization | Xavier normal |
| Bias initialization | Zero |
| Input preprocessing | Domain-based normalization |

The implemented constructor uses `n_layer=4`, but the network first creates one hidden layer and then adds four additional hidden layers inside the loop.

Therefore, the actual network contains **5 hidden layers in total**.

The input coordinates are normalized using the lower and upper bounds of the computational domain before being passed through the network.

---

## 5. Automatic Differentiation

PyTorch automatic differentiation is used to evaluate the derivatives required by the governing equation.

For the network prediction

```math
u_{\theta}(t,x)
```

the implementation calculates

```math
\frac{\partial u_{\theta}}{\partial t}
```

```math
\frac{\partial u_{\theta}}{\partial x}
```

and

```math
\frac{\partial^2 u_{\theta}}{\partial x^2}
```

directly from the computational graph.

This allows the governing differential equation to be included in the optimization objective without explicitly discretizing the derivatives with finite differences.

---

## 6. Physics-Informed PDE Residual

The Burgers-equation residual is defined as

```math
f_{\theta}(t,x)
=
\frac{\partial u_{\theta}}{\partial t}
+
u_{\theta}
\frac{\partial u_{\theta}}{\partial x}
-
\nu
\frac{\partial^2 u_{\theta}}{\partial x^2}
```

For an ideal solution,

```math
f_{\theta}(t,x)
=
0
```

throughout the computational domain.

The PDE loss is evaluated over the interior collocation points:

```math
\mathcal{L}_{PDE}
=
\frac{1}{N_c}
\sum_{i=1}^{N_c}
\left[
f_{\theta}(t_i,x_i)
\right]^2
```

where `N_c` is the number of interior collocation points.

---

## 7. Initial-Condition Loss

The prescribed initial condition is

```math
u(0,x)
=
-\sin(\pi x)
```

The corresponding loss term is

```math
\mathcal{L}_{IC}
=
\frac{1}{N_i}
\sum_{i=1}^{N_i}
\left[
u_{\theta}(0,x_i)
+
\sin(\pi x_i)
\right]^2
```

where `N_i` is the number of initial-condition training points.

---

## 8. Boundary-Condition Loss

The boundary conditions are

```math
u(t,-1)=0
```

and

```math
u(t,1)=0
```

The implementation combines the samples from both spatial boundaries into a single boundary dataset.

The boundary-condition loss can therefore be represented as

```math
\mathcal{L}_{BC}
=
\frac{1}{2N_b}
\sum_{i=1}^{2N_b}
\left[
u_{\theta}(t_i,x_i)
-
u_{BC,i}
\right]^2
```

where each prescribed boundary value is zero.

The implementation combines the initial- and boundary-condition losses as

```math
\mathcal{L}_{IC+BC}
=
\mathcal{L}_{IC}
+
\mathcal{L}_{BC}
```

---

## 9. Total Training Objective

The complete physics-informed objective is

```math
\mathcal{L}
=
\mathcal{L}_{IC+BC}
+
\mathcal{L}_{PDE}
```

or equivalently,

```math
\mathcal{L}
=
\mathcal{L}_{IC}
+
\mathcal{L}_{BC}
+
\mathcal{L}_{PDE}
```

This objective forces the network to reduce both the data-free physics residual and the errors associated with the prescribed physical constraints.

---

## 10. Training-Point Sampling

Latin Hypercube Sampling (LHS) is used to generate the initial-condition, boundary-condition, and interior collocation points.

| Point Set | Number of Points |
|---|---:|
| Initial-condition points | 1,000 |
| Left-boundary points | 1,000 |
| Right-boundary points | 1,000 |
| Total boundary points | 2,000 |
| Interior collocation points | 10,000 |

The interior collocation points are distributed throughout the full space-time domain.

No labeled interior solution data are used in the PINN loss.

---

## 11. Optimization Strategy

Training is performed in two sequential optimization stages.

### Stage 1 - Adam

The first stage uses the Adam optimizer.

| Parameter | Value |
|---|---:|
| Learning rate | `0.001` |
| Iterations | `1000` |

Adam provides the initial optimization of the neural-network parameters.

### Stage 2 - L-BFGS

The second stage uses PyTorch's L-BFGS optimizer.

| Parameter | Value |
|---|---:|
| Learning rate | `1.0` |
| Maximum iterations | `10000` |
| Maximum function evaluations | `10000` |
| History size | `100` |
| Gradient tolerance | `1e-10` |
| Line search | `strong_wolfe` |

The L-BFGS stage is applied after Adam to further reduce the physics-informed objective using quasi-Newton optimization.

---

## 12. Predicted Space-Time Solution

After training, the PINN is evaluated over a uniform `100 x 100` space-time grid.

### Solution Field

![PINN solution field](figures/solution-field.png)

The figure represents the predicted field

```math
u_{\theta}(t,x)
```

over the complete computational domain.

The solution evolves from the prescribed sinusoidal initial condition under the combined effects of nonlinear convection and viscous diffusion.

---

## 13. Solution Cross-Sections

The predicted solution is also evaluated at five selected time locations:

```math
t
=
0,\;
0.2,\;
0.45,\;
0.75,\;
1.0
```

![Solution cross-sections](figures/solution-cross-sections.png)

At `t = 0`, the network prediction is compared directly with the prescribed analytical initial condition

```math
u(0,x)
=
-\sin(\pi x)
```

The analytical curve shown at `t = 0` verifies how closely the trained network reproduces the imposed initial condition.

For `t > 0`, the notebook does not include an analytical or independent numerical reference solution.

Therefore, the later curves should be interpreted as **PINN predictions**, not as validated comparisons against an exact solution.

---

## 14. Training-Loss History

The implementation separately records:

- the combined initial- and boundary-condition loss;
- the PDE residual loss.

![Training loss history](figures/training-loss-history.png)

Both quantities are plotted on a logarithmic scale.

The reduction of these loss components indicates progressive enforcement of the prescribed conditions and the Burgers-equation residual during optimization.

### Interpretation of the Iteration Axis

The loss history is updated every time the optimization closure is evaluated.

During the Adam stage, this approximately corresponds to individual optimization iterations.

During the L-BFGS stage, however, the optimizer can evaluate the closure multiple times within a single optimization step.

Therefore, the horizontal axis is more precisely interpreted as a sequence of **loss-function evaluations** rather than a strict count of independent L-BFGS parameter updates.

---

## 15. Implementation Workflow

The notebook follows the computational workflow below:

1. define the physical parameters and computational domain;
2. generate initial, boundary, and collocation points using Latin Hypercube Sampling;
3. construct the fully connected neural network;
4. normalize the network inputs;
5. compute the required derivatives using automatic differentiation;
6. construct the PDE residual;
7. construct the initial- and boundary-condition losses;
8. optimize the network using Adam;
9. refine the solution using L-BFGS;
10. evaluate the trained PINN on a space-time grid;
11. generate solution and training-history visualizations;
12. save the trained network state.

When the notebook is executed, the model state is saved as

```text
burgers_pinn.pt
```

---

## 16. Technologies

The implementation uses:

- Python
- PyTorch
- NumPy
- Matplotlib
- pyDOE
- Jupyter Notebook
- Automatic Differentiation
- Latin Hypercube Sampling
- Adam Optimization
- L-BFGS Optimization

CUDA acceleration is selected automatically when an available GPU is detected.

---

## 17. Repository Contents

```text
01-viscous-burgers-equation/
|-- README.md
|-- viscous-burgers-pinn.ipynb
`-- figures/
    |-- solution-cross-sections.png
    |-- solution-field.png
    `-- training-loss-history.png
```

The notebook contains the complete implementation of the PINN workflow, while the `figures` directory contains the principal generated visualizations used in this README.

---

## 18. Methodological Notes

This project is a computational study of a Physics-Informed Neural Network applied to a nonlinear time-dependent PDE.

The PINN is trained using:

- the governing differential equation;
- the analytical initial condition;
- the spatial boundary conditions.

The current implementation does **not** include:

- a high-resolution numerical reference solution;
- a complete analytical reference solution for `t > 0`;
- a relative error field;
- a quantitative global error metric against an independent benchmark.

Accordingly, the numerical results are presented as **physics-informed predictions** rather than as a fully benchmark-validated PDE solution.

The comparison shown at `t = 0` should specifically be interpreted as validation against the prescribed initial condition, not as validation of the full time-dependent solution.

---

## 19. Potential Extensions

Possible future developments include:

- comparison with a high-resolution finite-difference solution;
- comparison with a spectral reference solution;
- relative L2 error evaluation;
- pointwise error-field visualization;
- adaptive residual-based sampling;
- sensitivity analysis with respect to network depth and width;
- collocation-density studies;
- optimizer comparisons;
- adaptive loss weighting;
- alternative activation functions;
- studies at lower viscosity values;
- investigation of sharper nonlinear gradients;
- parameterized PINN formulations.

---

## 20. Summary

This project demonstrates the use of a Physics-Informed Neural Network to approximate the solution of the one-dimensional viscous Burgers equation.

The implementation combines:

- a nonlinear governing PDE;
- analytical initial and boundary conditions;
- automatic differentiation;
- Latin Hypercube Sampling;
- a fully connected neural network;
- Adam optimization;
- L-BFGS refinement;
- continuous space-time prediction.

The project provides a compact example of how physical governing equations can be incorporated directly into neural-network training for scientific and engineering computation.

---

**Author:** Armin Amani
