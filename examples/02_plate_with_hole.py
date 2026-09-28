"""Plate with a circular hole under uniaxial tension: solve with FEM,
validate the stress concentration factor against the Kirsch solution,
and save plots to results/plate_with_hole/."""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from fem.assembly import assemble_stiffness
from fem.boundary import add_edge_traction, fixed_dofs, nodes_on_line
from fem.material import Material
from fem.mesh_gmsh import plate_with_hole_mesh
from fem.plotting import plot_contour, plot_deformed, plot_mesh
from fem.postprocess import element_strain_stress, nodal_average, von_mises
from fem.solver import solve
from validation.analytical import kirsch_stress_concentration_factor

OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "plate_with_hole"
OUT.mkdir(parents=True, exist_ok=True)

# ---- Problem definition -----------------------------------------------
W, H, R = 20.0, 10.0, 1.0       # plate width, height, hole radius [m]
SIGMA = 1.0e6                    # applied far-field tensile stress [Pa]
T = 0.01                         # thickness [m]
MESH_SIZE_FAR = 0.5              # target element size away from the hole
mat = Material(E=200e9, nu=0.3, t=T)
cx, cy = W / 2.0, 0.0            # hole center

# ---- Model prep and meshing (gmsh, refined near the hole) --------------
nodes, elements = plate_with_hole_mesh(W, H, R, MESH_SIZE_FAR)
print(f"Mesh: {len(nodes)} nodes, {len(elements)} elements")

# ---- Boundary conditions and load --------------------------------------
left = nodes_on_line(nodes, x=0.0)
right = nodes_on_line(nodes, x=W)
fixed = np.concatenate([
    fixed_dofs(left, x=True, y=False),        # no x-motion on the left edge
    fixed_dofs([left[0]], x=False, y=True),   # pin one point to stop y-drift
])

F = np.zeros(2 * len(nodes))
add_edge_traction(F, nodes, right, SIGMA, 0.0, mat.t)

# ---- Analysis setup and solve --------------------------------------------
K = assemble_stiffness(nodes, elements, mat)
u, R_reac = solve(K, F, fixed)

# ---- Result evaluation ---------------------------------------------------
strains, stresses = element_strain_stress(nodes, elements, u, mat.D)
vm_elem = von_mises(stresses)
vm_nodal = nodal_average(nodes, elements, vm_elem)
sxx_nodal = nodal_average(nodes, elements, stresses[:, 0])

dx, dy = nodes[:, 0] - cx, nodes[:, 1] - cy
radius = np.hypot(dx, dy)
theta_deg = np.degrees(np.arctan2(dy, dx))
near_hole_90 = (radius < R * 1.15) & (np.abs(np.abs(theta_deg) - 90.0) < 15.0)

sigma_max_fem = sxx_nodal[near_hole_90].max()
kt_fem = sigma_max_fem / SIGMA
kt_theory = kirsch_stress_concentration_factor()
error = abs(kt_fem - kt_theory) / kt_theory * 100

print(f"\nApplied far-field stress:        {SIGMA:.4e} Pa")
print(f"Max hoop stress near hole (FEM): {sigma_max_fem:.4e} Pa")
print(f"Stress concentration Kt (FEM):    {kt_fem:.3f}")
print(f"Stress concentration Kt (theory): {kt_theory:.3f}   error: {error:.2f}%")

applied_fx = F[0::2].sum()
reaction_fx = R_reac[0::2].sum()
print(f"\nSum of applied Fx:  {applied_fx:.4e} N")
print(f"Sum of reaction Fx: {reaction_fx:.4e} N  (should be equal and opposite)")

# ---- Plots ----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
plot_mesh(nodes, elements, ax=ax)
ax.set_title(f"Mesh with hole ({len(elements)} elements)")
fig.tight_layout()
fig.savefig(OUT / "mesh.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5))
scale = 0.05 * W / max(abs(u.min()), abs(u.max()))
plot_deformed(nodes, elements, u, scale=scale, ax=ax)
fig.tight_layout()
fig.savefig(OUT / "deformed_shape.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5))
plot_contour(nodes, elements, vm_nodal, ax=ax, label="von Mises stress [Pa]")
ax.set_title("Von Mises stress")
fig.tight_layout()
fig.savefig(OUT / "von_mises.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5))
plot_contour(nodes, elements, sxx_nodal, ax=ax, label="sigma_xx [Pa]", cmap="RdBu_r")
ax.set_title(f"sigma_xx (Kt_FEM = {kt_fem:.2f}, Kt_theory = {kt_theory:.1f})")
fig.tight_layout()
fig.savefig(OUT / "sigma_xx.png", dpi=150)
plt.close(fig)

print(f"\nPlots saved to {OUT}")