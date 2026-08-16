# Physics-Informed Neural Network for Lid-Driven Cavity Flow

This project applies a Physics-Informed Neural Network (PINN) to the steady two-dimensional incompressible lid-driven cavity problem.

The lid-driven cavity is a classical computational fluid dynamics benchmark in which the motion of the upper wall drives a recirculating viscous flow inside a closed square domain.

---

## 1. Physical Problem

The computational domain is the unit square:

```math
(x,y) \in [0,1] \times [0,1]
```

The upper wall moves horizontally with velocity

```math
U_{\mathrm{lid}} = 1
```

while the left, right, and bottom walls remain stationary.

The kinematic viscosity is

```math
\nu = 0.01
```

Using a characteristic cavity length of `L = 1`, the Reynolds number is

```math
Re
=
\frac{U_{\mathrm{lid}}L}{\nu}
=
100
```

Therefore, the present problem is solved at `Re = 100`.

---

## 2. Governing Equations

The PINN enforces the steady two-dimensional incompressible Navier-Stokes equations.

The neural network predicts the flow variables

```math
(x,y)
\longmapsto
\left(
u(x,y),
v(x,y),
p(x,y)
\right)
```

where:

- `u` is the horizontal velocity component.
- `v` is the vertical velocity component.
- `p` is the pressure field.

### Continuity Equation

Conservation of mass for incompressible flow requires

```math
\frac{\partial u}{\partial x}
+
\frac{\partial v}{\partial y}
=
0
```

### x-Momentum Equation

The horizontal momentum equation is

```math
u\frac{\partial u}{\partial x}
+
v\frac{\partial u}{\partial y}
+
\frac{\partial p}{\partial x}
-
\frac{1}{Re}
\left(
\frac{\partial^2u}{\partial x^2}
+
\frac{\partial^2u}{\partial y^2}
\right)
=
0
```

### y-Momentum Equation

The vertical momentum equation is

```math
u\frac{\partial v}{\partial x}
+
v\frac{\partial v}{\partial y}
+
\frac{\partial p}{\partial y}
-
\frac{1}{Re}
\left(
\frac{\partial^2v}{\partial x^2}
+
\frac{\partial^2v}{\partial y^2}
\right)
=
0
```

PyTorch automatic differentiation is used to evaluate the first- and second-order spatial derivatives required by the governing equations.

---

## 3. Boundary Conditions

The left, right, and bottom walls satisfy the no-slip condition:

```math
u = 0,
\qquad
v = 0
```

The moving upper lid satisfies

```math
u = 1,
\qquad
v = 0
```

The complete velocity boundary conditions are

### Left wall

```math
u(0,y)=0,
\qquad
v(0,y)=0
```

### Right wall

```math
u(1,y)=0,
\qquad
v(1,y)=0
```

### Bottom wall

```math
u(x,0)=0,
\qquad
v(x,0)=0
```

### Moving top lid

```math
u(x,1)=1,
\qquad
v(x,1)=0
```

No explicit pressure-reference condition is imposed in the current implementation.

---

## 4. Physics-Informed Neural Network

The neural network represents the mapping

```math
(x,y)
\longmapsto
(u,v,p)
```

The implemented fully connected architecture has the following configuration:

| Component | Configuration |
|---|---|
| Inputs | `x`, `y` |
| Hidden layers | 9 |
| Neurons per hidden layer | 30 |
| Activation function | `Tanh` |
| Outputs | `u`, `v`, `p` |
| Weight initialization | Xavier normal |
| Input preprocessing | Domain-based normalization |

The implementation parameter is written as `n_layer=8`. However, the network construction first creates one hidden layer and then adds eight additional hidden layers.

Therefore, the implemented network contains **9 hidden layers in total**.

---

## 5. Physics-Informed Residuals

The governing equations are incorporated directly into the training objective through residual functions evaluated at the interior collocation points.

### Continuity Residual

```math
r_c
=
\frac{\partial u}{\partial x}
+
\frac{\partial v}{\partial y}
```

### x-Momentum Residual

```math
r_x
=
u\frac{\partial u}{\partial x}
+
v\frac{\partial u}{\partial y}
+
\frac{\partial p}{\partial x}
-
\frac{1}{Re}
\left(
\frac{\partial^2u}{\partial x^2}
+
\frac{\partial^2u}{\partial y^2}
\right)
```

### y-Momentum Residual

```math
r_y
=
u\frac{\partial v}{\partial x}
+
v\frac{\partial v}{\partial y}
+
\frac{\partial p}{\partial y}
-
\frac{1}{Re}
\left(
\frac{\partial^2v}{\partial x^2}
+
\frac{\partial^2v}{\partial y^2}
\right)
```

The individual physics-informed loss components are defined from the mean squared residuals.

For the continuity equation:

```math
\mathcal{L}_{c}
=
\frac{1}{N_f}
\sum_{i=1}^{N_f}
\left[
r_c(x_i,y_i)
\right]^2
```

For the x-momentum equation:

```math
\mathcal{L}_{x}
=
\frac{1}{N_f}
\sum_{i=1}^{N_f}
\left[
r_x(x_i,y_i)
\right]^2
```

For the y-momentum equation:

```math
\mathcal{L}_{y}
=
\frac{1}{N_f}
\sum_{i=1}^{N_f}
\left[
r_y(x_i,y_i)
\right]^2
```

The total PDE loss is therefore

```math
\mathcal{L}_{\mathrm{PDE}}
=
\mathcal{L}_{c}
+
\mathcal{L}_{x}
+
\mathcal{L}_{y}
```

---

## 6. Boundary-Condition Loss

The velocity boundary-condition loss is based on the differences between the predicted and prescribed velocity components:

```math
\mathcal{L}_{\mathrm{BC}}
=
\frac{1}{N_b}
\sum_{i=1}^{N_b}
\left[
u_{\theta}(x_i,y_i)-u_{\mathrm{BC},i}
\right]^2
+
\frac{1}{N_b}
\sum_{i=1}^{N_b}
\left[
v_{\theta}(x_i,y_i)-v_{\mathrm{BC},i}
\right]^2
```

The complete training objective is

```math
\mathcal{L}
=
\mathcal{L}_{\mathrm{BC}}
+
\mathcal{L}_{\mathrm{PDE}}
```

or equivalently,

```math
\mathcal{L}
=
\mathcal{L}_{\mathrm{BC}}
+
\mathcal{L}_{c}
+
\mathcal{L}_{x}
+
\mathcal{L}_{y}
```

---

## 7. Training-Point Sampling

Latin Hypercube Sampling (LHS) is used to generate the training points.

| Point Set | Number of Points |
|---|---:|
| Interior collocation points | 10,000 |
| Left wall | 500 |
| Right wall | 500 |
| Bottom wall | 500 |
| Moving top lid | 500 |
| **Total boundary points** | **2,000** |

The interior collocation points are used to enforce the governing equations throughout the two-dimensional domain.

---

## 8. Optimization Strategy

Training is performed sequentially using two optimizers.

### Stage 1 - Adam

The first optimization stage uses:

- Optimizer: `Adam`
- Learning rate: `0.001`
- Iterations: `1000`

### Stage 2 - L-BFGS

The second optimization stage uses PyTorch's L-BFGS optimizer with:

- Maximum iterations: `10000`
- Maximum function evaluations: `10000`
- History size: `100`
- Line search: `strong_wolfe`

The L-BFGS stage further minimizes the combined physics-informed and boundary-condition losses after the initial Adam optimization stage.

---

## 9. Predicted Flow Field

After training, the PINN is evaluated on a Cartesian grid with

```math
100 \times 100
```

evaluation points.

The resulting visualization contains:

- velocity magnitude and streamlines;
- horizontal velocity `u`;
- vertical velocity `v`;
- pressure contours;
- boundary-condition loss history;
- PDE loss history.

![Lid-driven cavity PINN results](figures/lid-driven-cavity-results.png)

The predicted velocity field develops a dominant recirculating structure inside the cavity as a consequence of the moving upper wall.

---

## 10. Pressure Interpretation

The present formulation does not impose an explicit pressure reference such as

```math
p(x_0,y_0)=0
```

For incompressible Navier-Stokes flow, the momentum equations constrain pressure through its spatial gradients:

```math
\nabla p
```

Therefore, adding a spatially constant value to the pressure field does not change the pressure gradients:

```math
p^{*}(x,y)
=
p(x,y)+C
```

Consequently, the absolute pressure level shown in the visualization should not be interpreted as a uniquely defined reference pressure.

The upper-corner regions should also be interpreted carefully because the moving lid meets the stationary side walls at these locations.

---

## 11. Training-Loss History

The implementation records:

- boundary-condition loss;
- continuity residual loss;
- x-momentum residual loss;
- y-momentum residual loss;
- total PDE loss.

The generated figure displays the principal boundary-condition and PDE loss histories on a logarithmic scale.

Because L-BFGS can evaluate its closure multiple times during optimization, the horizontal index of the recorded loss history is more accurately interpreted as a sequence of loss-function evaluations rather than a strict count of independent optimizer updates.

---

## 12. Technologies

The implementation uses:

- Python
- PyTorch
- NumPy
- Matplotlib
- pyDOE
- Jupyter Notebook
- Automatic Differentiation
- Latin Hypercube Sampling
- Adam
- L-BFGS

CUDA acceleration is selected automatically when an available GPU is detected.

---

## 13. Repository Contents

```text
02-lid-driven-cavity-navier-stokes/
|-- README.md
|-- lid-driven-cavity-pinn.ipynb
`-- figures/
    `-- lid-driven-cavity-results.png
```

The notebook contains the complete workflow, including:

- training-point generation;
- neural-network construction;
- automatic differentiation;
- physics-informed residual evaluation;
- boundary-condition enforcement;
- Adam optimization;
- L-BFGS optimization;
- flow-field prediction;
- visualization;
- model-state saving.

---

## 14. Methodological Notes

This project demonstrates a Physics-Informed Neural Network formulation for the steady incompressible Navier-Stokes equations at `Re = 100`.

The current implementation does not include:

- an independent CFD reference solution;
- centerline benchmark data;
- quantitative velocity-error metrics;
- an independently validated pressure field.

Therefore, the presented results should be interpreted as **physics-informed flow predictions demonstrating qualitative physical behavior**, rather than as a fully benchmark-validated CFD solution.

---

## 15. Potential Extensions

Possible future developments include:

- comparison with established lid-driven cavity benchmark data;
- centerline velocity validation;
- quantitative error analysis against a CFD reference solution;
- explicit pressure-gauge enforcement;
- adaptive residual-based sampling;
- improved treatment of the upper-corner regions;
- adaptive loss weighting;
- Reynolds-number parameter studies;
- higher-Reynolds-number cavity flows;
- alternative PINN architectures;
- domain decomposition;
- adaptive PINN methods.

---

**Author:** Armin Amani
