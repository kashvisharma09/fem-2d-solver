"""Cantilever beam under a tip load: solve with FEM, validate against theory,
and save plots to results/cantilever/."""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from fem.assembly import assemble_stiffness
from fem.boundary import add_edge_traction, fixed_dofs, nodes_on_line
from fem.material import Material
from fem.mesh import rectangle_mesh
from fem.plotting import plot_contour, plot_deformed, plot_mesh
from fem.postprocess import element_strain_stress, nodal_average, von_mises
from fem.solver import solve
from validation.analytical import cantilever_max_bending_stress, cantilever_tip_deflection

OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "cantilever"
OUT.mkdir(parents=True, exist_ok=True)

# ---- Problem definition -----------------------------------------------
L, H, T = 10.0, 1.0, 0.1          # length, height, thickness [m]
P_total = -1.0e4                  # total tip force [N], downward
NX, NY = 60, 6                    # mesh resolution
mat = Material(E=200e9, nu=0.3, t=T)

# ---- Model prep and meshing -------------------------------------------
nodes, elements = rectangle_mesh(L, H, NX, NY)
print(f"Mesh: {len(nodes)} nodes, {len(elements)} elements")

# ---- Boundary conditions and load --------------------------------------
left = nodes_on_line(nodes, x=0.0)
right = nodes_on_line(nodes, x=L)
fixed = fixed_dofs(left)                              # clamp the left edge

F = np.zeros(2 * len(nodes))
traction = P_total / (H * mat.t)                      # force per unit area
add_edge_traction(F, nodes, right, 0.0, traction, mat.t)

# ---- Analysis setup and solve -------------------------------------------
K = assemble_stiffness(nodes, elements, mat)
u, R = solve(K, F, fixed)

# ---- Result evaluation ---------------------------------------------------
strains, stresses = element_strain_stress(nodes, elements, u, mat.D)
vm_elem = von_mises(stresses)
vm_nodal = nodal_average(nodes, elements, vm_elem)
sxx_nodal = nodal_average(nodes, elements, stresses[:, 0])

tip_nodes = right
v_fem = u[2 * tip_nodes + 1].mean()
v_theory_timo = cantilever_tip_deflection(P_total, L, H, T, mat.E, mat.nu, shear_correction=True)
v_theory_euler = cantilever_tip_deflection(P_total, L, H, T, mat.E, mat.nu, shear_correction=False)
sigma_max_theory = cantilever_max_bending_stress(P_total, L, H, T)
sigma_max_fem = np.abs(sxx_nodal).max()

error_timo = abs(v_fem - v_theory_timo) / abs(v_theory_timo) * 100
error_euler = abs(v_fem - v_theory_euler) / abs(v_theory_euler) * 100

print(f"\nTip deflection (FEM):              {v_fem:.6e} m")
print(f"Tip deflection (Timoshenko theory): {v_theory_timo:.6e} m   error: {error_timo:.2f}%")
print(f"Tip deflection (Euler-Bernoulli):   {v_theory_euler:.6e} m   error: {error_euler:.2f}%")
print(f"Max bending stress (FEM):           {sigma_max_fem:.4e} Pa")
print(f"Max bending stress (theory):        {sigma_max_theory:.4e} Pa")

reaction_fy = R[1::2].sum()
print(f"\nSum of applied Fy:  {F[1::2].sum():.4e} N")
print(f"Sum of reaction Fy: {reaction_fy:.4e} N  (should be equal and opposite)")

# ---- Plots ----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 3))
plot_mesh(nodes, elements, ax=ax)
ax.set_title(f"Mesh ({NX}x{NY} cells, {len(elements)} elements)")
fig.tight_layout()
fig.savefig(OUT / "mesh.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 3))
scale = 0.1 * L / max(abs(u.min()), abs(u.max()))
plot_deformed(nodes, elements, u, scale=scale, ax=ax)
fig.tight_layout()
fig.savefig(OUT / "deformed_shape.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 3))
plot_contour(nodes, elements, vm_nodal, ax=ax, label="von Mises stress [Pa]")
ax.set_title("Von Mises stress")
fig.tight_layout()
fig.savefig(OUT / "von_mises.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 3))
plot_contour(nodes, elements, sxx_nodal, ax=ax, label="sigma_xx [Pa]", cmap="RdBu_r")
ax.set_title("Bending stress (sigma_xx)")
fig.tight_layout()
fig.savefig(OUT / "sigma_xx.png", dpi=150)
plt.close(fig)

print(f"\nPlots saved to {OUT}")