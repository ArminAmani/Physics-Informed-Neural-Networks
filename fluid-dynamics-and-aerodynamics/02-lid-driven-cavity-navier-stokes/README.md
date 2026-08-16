# Physics-Informed Neural Network for Lid-Driven Cavity Flow

This project applies a Physics-Informed Neural Network (PINN) to the steady two-dimensional incompressible lid-driven cavity problem.

The lid-driven cavity is a classical computational fluid dynamics benchmark characterized by a moving upper wall that drives a recirculating viscous flow inside a closed square domain.

---

## 1. Physical Problem

The computational domain is a unit square:

$$
(x,y)\in[0,1]\times[0,1].
$$

The upper wall moves horizontally with velocity

$$
U_{\mathrm{lid}} = 1,
$$

while the left, right, and bottom walls remain stationary.

The kinematic viscosity is

$$
\nu = 0.01.
$$

Using the cavity length $L=1$, the Reynolds number is

$$
Re
=
\frac{U_{\mathrm{lid}}L}{\nu}
=
100.
$$

The problem is therefore solved at

$$
\boxed{Re=100}
$$

using a physics-informed neural-network formulation.

---

## 2. Governing Equations

The PINN enforces the dimensionless steady incompressible Navierâ€“Stokes equations.

The network predicts the three flow variables

$$
(x,y)
\longmapsto
\left(
u(x,y),
v(x,y),
p(x,y)
\right),
$$

where:

- $u$ is the horizontal velocity component;
- $v$ is the vertical velocity component;
- $p$ is the pressure field.

### Continuity Equation

Mass conservation for incompressible flow requires

$$
\boxed{
\frac{\partial u}{\partial x}
+
\frac{\partial v}{\partial y}
=
0
}
$$

throughout the fluid domain.

### x-Momentum Equation

The horizontal momentum equation is

$$
\boxed{
u\frac{\partial u}{\partial x}
+
v\frac{\partial u}{\partial y}
+
\frac{\partial p}{\partial x}
-
\frac{1}{Re}
\left(
\frac{\partial^2 u}{\partial x^2}
+
\frac{\partial^2 u}{\partial y^2}
\right)
=
0
}
$$

### y-Momentum Equation

The vertical momentum equation is

$$
\boxed{
u\frac{\partial v}{\partial x}
+
v\frac{\partial v}{\partial y}
+
\frac{\partial p}{\partial y}
-
\frac{1}{Re}
\left(
\frac{\partial^2 v}{\partial x^2}
+
\frac{\partial^2 v}{\partial y^2}
\right)
=
0
}
$$

All required first- and second-order derivatives are calculated using PyTorch automatic differentiation.

---

## 3. Boundary Conditions

The left, right, and bottom walls satisfy the no-slip condition:

$$
u=0,
\qquad
v=0.
$$

The moving upper lid satisfies

$$
u=U_{\mathrm{lid}}=1,
\qquad
v=0.
$$

Thus, the velocity boundary conditions are

$$
\begin{aligned}
u(0,y) &= 0,
&
v(0,y) &= 0,
\\
u(1,y) &= 0,
&
v(1,y) &= 0,
\\
u(x,0) &= 0,
&
v(x,0) &= 0,
\\
u(x,1) &= 1,
&
v(x,1) &= 0.
\end{aligned}
$$

No explicit pressure-reference condition is imposed in the current implementation.

---

## 4. Physics-Informed Neural Network

The neural network represents the mapping

$$
(x,y)
\longmapsto
(u,v,p).
$$

The implemented fully connected architecture contains:

- **Inputs:** 2 â€” $x$ and $y$
- **Hidden layers:** 9
- **Neurons per hidden layer:** 30
- **Activation function:** `Tanh`
- **Outputs:** 3 â€” $u$, $v$, and $p$
- **Weight initialization:** Xavier normal initialization
- **Input preprocessing:** normalization using the domain bounds

Although the implementation parameter is named `n_layer=8`, the network construction first creates one hidden layer and then adds eight additional hidden layers. The resulting network therefore contains **9 hidden layers in total**.

---

## 5. Physics-Informed Residuals

Three PDE residuals are evaluated at interior collocation points.

### Continuity Residual

$$
r_c
=
\frac{\partial u}{\partial x}
+
\frac{\partial v}{\partial y}.
$$

### x-Momentum Residual

$$
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
\right).
$$

### y-Momentum Residual

$$
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
\right).
$$

The corresponding PDE loss is

$$
\mathcal{L}_{\mathrm{PDE}}
=
\mathcal{L}_{c}
+
\mathcal{L}_{x}
+
\mathcal{L}_{y},
$$

where

$$
\mathcal{L}_{c}
=
\operatorname{MSE}(r_c),
$$

$$
\mathcal{L}_{x}
=
\operatorname{MSE}(r_x),
$$

and

$$
\mathcal{L}_{y}
=
\operatorname{MSE}(r_y).
$$

---

## 6. Boundary-Condition Loss

The velocity boundary-condition loss is

$$
\mathcal{L}_{\mathrm{BC}}
=
\operatorname{MSE}
\left(
u_{\theta}-u_{\mathrm{BC}}
\right)
+
\operatorname{MSE}
\left(
v_{\theta}-v_{\mathrm{BC}}
\right).
$$

The total training objective is

$$
\boxed{
\mathcal{L}
=
\mathcal{L}_{\mathrm{BC}}
+
\mathcal{L}_{\mathrm{PDE}}
}
$$

or, equivalently,

$$
\boxed{
\mathcal{L}
=
\mathcal{L}_{\mathrm{BC}}
+
\mathcal{L}_{c}
+
\mathcal{L}_{x}
+
\mathcal{L}_{y}
}
$$

---

## 7. Training-Point Sampling

Latin Hypercube Sampling (LHS) is used to generate both interior and boundary training points.

| Point Set | Number of Points |
|---|---:|
| Interior collocation points | 10,000 |
| Left wall | 500 |
| Right wall | 500 |
| Bottom wall | 500 |
| Moving top lid | 500 |
| **Total boundary points** | **2,000** |

The collocation points enforce the governing equations throughout the two-dimensional flow domain.

---

## 8. Optimization Strategy

Training is performed in two stages.

### Stage 1 â€” Adam

The first stage uses

- Optimizer: Adam
- Learning rate: `0.001`
- Iterations: `1000`

### Stage 2 â€” L-BFGS

The second stage uses PyTorch's L-BFGS optimizer with

- Maximum iterations: `10000`
- Maximum function evaluations: `10000`
- History size: `100`
- Line search: `strong_wolfe`

The L-BFGS stage further minimizes the combined PDE and boundary-condition residuals after the initial Adam training stage.

---

## 9. Predicted Flow Field

The trained PINN is evaluated on a

$$
100\times100
$$

Cartesian grid covering the cavity.

The generated visualization contains:

1. velocity magnitude and streamlines;
2. horizontal velocity $u$;
3. vertical velocity $v$;
4. pressure contours;
5. boundary-condition and PDE loss histories.

![Lid-driven cavity PINN results](figures/lid-driven-cavity-results.png)

The predicted velocity field develops a dominant recirculating motion inside the cavity, driven by the moving upper wall.

---

## 10. Pressure Interpretation

The present formulation does not impose an explicit pressure reference such as

$$
p(x_0,y_0)=0.
$$

For incompressible Navierâ€“Stokes flow, the momentum equations determine pressure through its spatial gradients:

$$
\nabla p.
$$

Therefore, the pressure field is defined only up to an arbitrary additive constant:

$$
p^{*}(x,y)
=
p(x,y)+C.
$$

Consequently, the absolute pressure level shown in the visualization should not be interpreted as a uniquely defined reference pressure.

The regions near the upper corners should also be interpreted carefully because the classical lid-driven cavity problem contains an abrupt transition between the moving top wall and the stationary side walls.

---

## 11. Training-Loss History

The implementation separately records:

- boundary-condition loss;
- continuity residual loss;
- x-momentum residual loss;
- y-momentum residual loss;
- total PDE loss.

The figure displays the principal boundary-condition and PDE loss histories on a logarithmic scale.

Because L-BFGS can evaluate its closure multiple times during an optimization step, the plotted loss-history index is more precisely interpreted as a sequence of loss/closure evaluations rather than a strict count of independent optimizer updates.

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

CUDA acceleration is used automatically when an available GPU is detected.

---

## 13. Repository Contents

```text
02-lid-driven-cavity-navier-stokes/
â”œâ”€â”€ README.md
â”œâ”€â”€ lid-driven-cavity-pinn.ipynb
â””â”€â”€ figures/
    â””â”€â”€ lid-driven-cavity-results.png
```

The notebook contains the complete workflow from point generation and neural-network construction to physics-informed training, flow-field prediction, visualization, and model-state saving.

---

## 14. Methodological Notes

This project demonstrates a PINN formulation for the steady incompressible Navierâ€“Stokes equations at $Re=100$.

The current implementation does not include:

- an independent CFD reference solution;
- Ghia-type centerline benchmark data;
- quantitative velocity-error metrics;
- a separately validated pressure field.

Accordingly, the results should be interpreted as **physics-informed flow predictions demonstrating qualitative behavior**, rather than as a fully benchmark-validated CFD solution.

---

## 15. Potential Extensions

Possible future developments include:

- comparison with established lid-driven cavity benchmark data;
- centerline $u$- and $v$-velocity validation;
- relative error analysis against a CFD reference solution;
- explicit pressure-gauge enforcement;
- adaptive residual-based sampling;
- improved treatment of the upper-corner regions;
- loss-component weighting strategies;
- Reynolds-number parameter studies;
- higher-Reynolds-number cavity flows;
- alternative PINN architectures;
- domain decomposition and adaptive PINN methods.

---

**Author:** Armin Amani
