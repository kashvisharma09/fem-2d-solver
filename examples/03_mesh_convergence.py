"""Mesh convergence study for the cantilever beam: solve at several mesh
densities and plot how the deflection error shrinks as the mesh is refined."""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from fem.assembly import assemble_stiffness
from fem.boundary import add_edge_traction, fixed_dofs, nodes_on_line
from fem.material import Material
from fem.mesh import rectangle_mesh
from fem.solver import solve
from validation.analytical import cantilever_tip_deflection

OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "convergence"
OUT.mkdir(parents=True, exist_ok=True)

# ---- Problem definition -----------------------------------------------
L, H, T = 10.0, 1.0, 0.1
P_total = -1.0e4
mat = Material(E=200e9, nu=0.3, t=T)
v_theory = cantilever_tip_deflection(P_total, L, H, T, mat.E, mat.nu, shear_correction=True)

# ---- Sweep over mesh densities ------------------------------------------
# (nx, ny) pairs, coarsest to finest, keeping the element aspect ratio ~1:1
mesh_sizes = [(4, 1), (8, 2), (16, 3), (32, 5), (64, 8), (100, 12)]

n_elements_list = []
errors_pct = []
dofs_list = []

for nx, ny in mesh_sizes:
    nodes, elements = rectangle_mesh(L, H, nx, ny)
    left = nodes_on_line(nodes, x=0.0)
    right = nodes_on_line(nodes, x=L)
    fixed = fixed_dofs(left)

    F = np.zeros(2 * len(nodes))
    add_edge_traction(F, nodes, right, 0.0, P_total / (H * mat.t), mat.t)

    K = assemble_stiffness(nodes, elements, mat)
    u, _ = solve(K, F, fixed)

    v_fem = u[2 * right + 1].mean()
    error_pct = abs(v_fem - v_theory) / abs(v_theory) * 100

    n_elements_list.append(len(elements))
    errors_pct.append(error_pct)
    dofs_list.append(2 * len(nodes))

    print(f"nx={nx:>3}, ny={ny:>2}  |  elements={len(elements):>5}  |  "
          f"dofs={2*len(nodes):>5}  |  v_fem={v_fem:.6e} m  |  error={error_pct:.2f}%")

# ---- Plot: error vs. element count (log-log) -----------------------------
fig, ax = plt.subplots(figsize=(7, 5))
ax.loglog(n_elements_list, errors_pct, "o-", color="C0")
ax.set_xlabel("Number of elements")
ax.set_ylabel("Tip deflection error vs. Timoshenko theory [%]")
ax.set_title("Mesh convergence: cantilever beam tip deflection")
ax.grid(True, which="both", linestyle=":", alpha=0.6)
fig.tight_layout()
fig.savefig(OUT / "convergence_error.png", dpi=150)
plt.close(fig)

# ---- Plot: absolute deflection vs. element count, with the theory line ---
fig, ax = plt.subplots(figsize=(7, 5))
v_fem_list = [
    v_theory * (1 - e / 100 * np.sign(v_theory))  # placeholder, replaced below
    for e in errors_pct
]
# recompute exact v_fem values for the plot (avoid relying on the sign trick above)
v_fem_exact = []
for nx, ny in mesh_sizes:
    nodes, elements = rectangle_mesh(L, H, nx, ny)
    left = nodes_on_line(nodes, x=0.0)
    right = nodes_on_line(nodes, x=L)
    fixed = fixed_dofs(left)
    F = np.zeros(2 * len(nodes))
    add_edge_traction(F, nodes, right, 0.0, P_total / (H * mat.t), mat.t)
    K = assemble_stiffness(nodes, elements, mat)
    u, _ = solve(K, F, fixed)
    v_fem_exact.append(u[2 * right + 1].mean())

ax.semilogx(n_elements_list, v_fem_exact, "o-", color="C1", label="FEM (CST)")
ax.axhline(v_theory, color="k", linestyle="--", label="Timoshenko theory")
ax.set_xlabel("Number of elements")
ax.set_ylabel("Tip deflection [m]")
ax.set_title("Convergence toward the analytical solution")
ax.legend()
ax.grid(True, which="both", linestyle=":", alpha=0.6)
fig.tight_layout()
fig.savefig(OUT / "convergence_deflection.png", dpi=150)
plt.close(fig)

print(f"\nPlots saved to {OUT}")