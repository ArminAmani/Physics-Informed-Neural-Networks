# Physics-Informed Neural Network for the Viscous Burgers Equation

This project implements a Physics-Informed Neural Network (PINN) in PyTorch to approximate the solution of the one-dimensional viscous Burgers equation.

The Burgers equation provides a compact nonlinear model combining convection and diffusion and is commonly used as a benchmark problem for studying numerical methods and physics-informed machine learning approaches for partial differential equations.

## Governing Equation

The viscous Burgers equation considered here is

\[
\frac{\partial u}{\partial t}
+
u\frac{\partial u}{\partial x}
-
\nu\frac{\partial^2 u}{\partial x^2}
=0,
\]

with kinematic viscosity

\[
\nu=\frac{0.01}{\pi}.
\]

The computational domain is

\[
x\in[-1,1],
\qquad
t\in[0,1].
\]

## Initial and Boundary Conditions

The initial condition is

\[
u(0,x)=-\sin(\pi x),
\]

while homogeneous Dirichlet boundary conditions are imposed at both ends of the spatial domain:

\[
u(t,-1)=0,
\qquad
u(t,1)=0.
\]

## Physics-Informed Neural Network

The neural network approximates the solution

\[
(t,x)\longrightarrow u(t,x).
\]

The implementation uses a fully connected feed-forward network with:

- 2 input variables: \(t\) and \(x\)
- 5 hidden layers
- 40 neurons per hidden layer
- `Tanh` activation functions
- 1 output variable: \(u\)
- Xavier normal weight initialization
- normalized network inputs

Automatic differentiation provided by PyTorch is used to evaluate the derivatives required by the governing equation.

## Physics-Informed Loss

The PDE residual is defined as

\[
f(t,x)=
u_t
+
u\,u_x
-
\nu u_{xx}.
\]

The total training objective combines the initial/boundary-condition loss and the PDE residual loss:

\[
\mathcal{L}
=
\mathcal{L}_{IC+BC}
+
\mathcal{L}_{PDE}.
\]

The PDE contribution is

\[
\mathcal{L}_{PDE}
=
\frac{1}{N_c}
\sum_{i=1}^{N_c}
f(t_i,x_i)^2.
\]

The initial and boundary conditions are enforced through corresponding mean-squared-error terms.

## Training Points

Latin Hypercube Sampling (LHS) is used to distribute training points across the problem domain.

| Point Type | Number of Points |
|---|---:|
| Initial-condition points | 1,000 |
| Boundary points at \(x=-1\) | 1,000 |
| Boundary points at \(x=1\) | 1,000 |
| Interior collocation points | 10,000 |

The collocation points are used to enforce the Burgers equation throughout the interior of the space-time domain.

## Optimization

Training is performed in two stages.

First, the network is optimized using Adam:

- Learning rate: `0.001`
- Iterations: `1000`

The Adam stage is followed by L-BFGS optimization using a strong-Wolfe line search. This second optimization stage is used to further reduce the physics-informed objective.

## Results

The trained PINN produces a continuous approximation of \(u(t,x)\) over the full space-time domain.

### Predicted Solution Field

![PINN solution field](figures/solution-field.png)

The solution field illustrates the evolution of the initial sinusoidal profile under the combined effects of nonlinear convection and viscous diffusion.

### Solution Cross-Sections

![Solution cross-sections](figures/solution-cross-sections.png)

Cross-sections of the PINN prediction are shown at

\[
t=0,\ 0.2,\ 0.45,\ 0.75,\ 1.0.
\]

At \(t=0\), the PINN prediction is compared with the prescribed analytical initial condition

\[
u(0,x)=-\sin(\pi x).
\]

The notebook does not include an analytical or external reference solution for later times, so the remaining cross-sections represent PINN predictions rather than quantitative comparisons against an exact solution.

### Training Loss

![Training loss history](figures/training-loss-history.png)

The loss history tracks the two principal components of the optimization objective:

- initial and boundary condition loss
- PDE residual loss

## Implementation

The implementation is based on:

- Python
- PyTorch
- NumPy
- Matplotlib
- pyDOE
- automatic differentiation
- Latin Hypercube Sampling
- Adam optimization
- L-BFGS optimization

GPU acceleration is used automatically when CUDA is available.

## Repository Contents

```text
01-viscous-burgers-equation/
├── README.md
├── viscous-burgers-pinn.ipynb
└── figures/
    ├── solution-cross-sections.png
    ├── solution-field.png
    └── training-loss-history.png
```

The Jupyter notebook contains the complete PINN implementation, including sampling, neural-network construction, physics-informed loss evaluation, optimization, prediction, and visualization.

## Methodological Notes

This implementation is intended as a computational study of the PINN formulation for a nonlinear time-dependent PDE.

The generated solution satisfies the governing equation through residual minimization together with the prescribed initial and boundary conditions. However, no independent numerical benchmark or full analytical reference solution is used in the current notebook to quantify the prediction error throughout the complete space-time domain.

Consequently, the presented results should be interpreted as physics-informed predictions rather than as a fully benchmark-validated numerical solution.

## Possible Extensions

Potential extensions include quantitative comparison against a high-resolution numerical reference solution, relative error analysis, sensitivity studies with respect to collocation-point density and network architecture, adaptive residual-based sampling, optimizer comparisons, and investigation of PINN performance at lower viscosity values.

---

**Author:** Armin Amani