# CST Element Formulation

## Overview

The Constant Strain Triangle (CST) is the simplest 2D finite element: a
3-node triangle with linear displacement interpolation, giving a strain
field that is constant over the element (hence the name).

## Displacement field

Within an element, displacement varies linearly:

```
u(x, y) = a1 + a2*x + a3*y
v(x, y) = a4 + a5*x + a6*y
```

The six constants are determined by the six nodal displacement values
(u, v at each of the 3 nodes).

## Strain-displacement relation

Since strain is the derivative of displacement, and displacement is linear
in x and y, strain is constant within the element:

```
epsilon = [exx, eyy, gxy]^T = B @ u_e
```

where `B` (3x6) depends only on the nodal coordinates, and `u_e` is the
element's 6x1 nodal displacement vector.

## Constitutive relation (plane stress)

```
sigma = D @ epsilon
```

where `D` (3x3) is built from Young's modulus `E` and Poisson's ratio `nu`,
assuming stress is zero through the thickness (plane stress).

## Element stiffness matrix

From the principle of virtual work:

```
Ke = t * A * B^T @ D @ B
```

where `t` is thickness and `A` is the element's area. `Ke` is 6x6 and maps
nodal displacements to nodal forces: `f_e = Ke @ u_e`.

## Global assembly

Each element's 6x6 `Ke` is scattered into the global stiffness matrix `K`
at the rows/columns corresponding to its 3 nodes' global DOF indices
(2 DOFs per node: x and y). Overlapping contributions from elements sharing
a node are summed.

## Boundary conditions and solving

Given the global system `K @ u = F`:
- Fixed DOFs are removed from the system (their displacement is set to zero)
- The reduced system `Kff @ uf = Ff` is solved for the free DOFs
- Reaction forces at fixed DOFs are recovered as `R = K @ u - F`

## Known limitation: shear locking

CST elements use a linear displacement field, which cannot represent the
curved deformation that occurs in pure bending. This makes CST elements
artificially stiff in bending-dominated problems ("shear locking"). The
mesh convergence study in this project quantifies this effect directly:
error shrinks with mesh refinement but converges slowly compared to
higher-order elements (e.g. 6-node quadratic triangles), which are not
implemented here but would be the natural next step.