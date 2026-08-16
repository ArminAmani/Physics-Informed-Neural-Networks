# Physics-Informed Neural Networks for Engineering

A technical repository exploring Physics-Informed Neural Networks (PINNs) for solving differential equations and engineering problems.

The repository emphasizes the integration of physical governing equations, boundary and initial conditions, automatic differentiation, and neural-network optimization for computational engineering applications.

## Overview

Physics-Informed Neural Networks incorporate governing physical equations directly into the training objective of neural networks.

Instead of relying exclusively on labeled datasets, PINNs can use residuals of differential equations together with physical constraints to construct continuous approximations of engineering fields.

This repository is organized around engineering application areas and is intended to grow progressively as additional problems are implemented.

## Current Engineering Domain

### Fluid Dynamics & Aerodynamics

Current implementations include:

| Project | Description |
|---|---|
| [Viscous Burgers Equation](fluid-dynamics-and-aerodynamics/01-viscous-burgers-equation/) | PINN solution of a nonlinear time-dependent convection–diffusion equation |
| [Lid-Driven Cavity Flow](fluid-dynamics-and-aerodynamics/02-lid-driven-cavity-navier-stokes/) | PINN solution of the 2D steady incompressible Navier–Stokes equations at \(Re=100\) |

See the complete section:

[**Fluid Dynamics & Aerodynamics →**](fluid-dynamics-and-aerodynamics/)

## Repository Structure

```text
Physics-Informed-Neural-Networks/
├── README.md
├── fluid-dynamics-and-aerodynamics/
│   ├── README.md
│   ├── 01-viscous-burgers-equation/
│   │   ├── README.md
│   │   ├── viscous-burgers-pinn.ipynb
│   │   └── figures/
│   └── 02-lid-driven-cavity-navier-stokes/
│       ├── README.md
│       ├── lid-driven-cavity-pinn.ipynb
│       └── figures/
└── .gitignore
```

## Computational Approach

The implementations currently use combinations of:

- Physics-Informed Neural Networks
- automatic differentiation
- governing PDE residuals
- physical boundary and initial conditions
- Latin Hypercube Sampling
- fully connected neural networks
- Adam optimization
- L-BFGS optimization
- scientific visualization

## Technologies

- Python
- PyTorch
- NumPy
- Matplotlib
- pyDOE
- Jupyter Notebook
- Git
- GitHub

## Engineering Direction

The repository currently focuses on fluid dynamics and computational flow problems.

Future development may extend toward additional engineering domains such as:

- aerodynamics
- incompressible and compressible flows
- boundary-layer problems
- heat transfer
- solid mechanics
- structural mechanics
- inverse engineering problems
- scientific machine learning

These areas represent future directions rather than currently implemented projects.

## Methodological Philosophy

The goal of this repository is not only to obtain neural-network predictions, but also to document:

- the governing physical problem
- mathematical formulation
- boundary and initial conditions
- network architecture
- physics-informed loss construction
- sampling strategy
- optimization procedure
- numerical outputs
- methodological limitations

Results are presented conservatively, and quantitative validation is only claimed when an independent reference solution or benchmark is available.

---

**Author:** Armin Amani  
**Focus:** Aerospace Engineering · Computational Engineering · Scientific Machine Learning · Physics-Informed Neural Networks