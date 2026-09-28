import numpy as np
import pytest

from fem.assembly import assemble_stiffness
from fem.boundary import add_edge_traction, fixed_dofs, nodes_on_line
from fem.material import Material
from fem.mesh import rectangle_mesh
from fem.solver import solve
from validation.analytical import cantilever_tip_deflection

MAT = Material(E=200e9, nu=0.3, t=0.1)


def _run_cantilever(nx, ny, L=10.0, H=1.0, P=-1e4):
    nodes, elems = rectangle_mesh(L, H, nx, ny)
    K = assemble_stiffness(nodes, elems, MAT)
    F = np.zeros(2 * len(nodes))
    add_edge_traction(F, nodes, nodes_on_line(nodes, x=L), 0.0, P / (H * MAT.t), MAT.t)
    fixed = fixed_dofs(nodes_on_line(nodes, x=0.0))
    u, R = solve(K, F, fixed)
    tip = nodes_on_line(nodes, x=L)
    return u[2 * tip + 1].mean(), R


def test_reactions_balance_tip_load():
    P = -1e4
    _, R = _run_cantilever(40, 4, P=P)
    assert abs(R[1::2].sum() + P) < 1e-3 * abs(P)


def test_fine_mesh_matches_timoshenko_theory():
    """A reasonably fine mesh should be within 8% of Timoshenko beam theory.

    CST elements are known to be overly stiff in bending (shear locking),
    so some gap from theory is expected even at fine mesh resolution;
    this is exactly what the mesh convergence study (Step 8) investigates.
    """
    L, H, T, P = 10.0, 1.0, 0.1, -1e4
    v_fem, _ = _run_cantilever(80, 8, L, H, P)
    v_theory = cantilever_tip_deflection(P, L, H, T, MAT.E, MAT.nu, shear_correction=True)
    error = abs(v_fem - v_theory) / abs(v_theory)
    assert error < 0.08


def test_coarse_mesh_is_less_accurate_than_fine_mesh():
    """CST is overly stiff in bending: coarse meshes under-predict deflection
    more than fine meshes. This motivates the mesh convergence study."""
    L, H, T, P = 10.0, 1.0, 0.1, -1e4
    v_coarse, _ = _run_cantilever(4, 1, L, H, P)
    v_fine, _ = _run_cantilever(80, 8, L, H, P)
    v_theory = cantilever_tip_deflection(P, L, H, T, MAT.E, MAT.nu, shear_correction=True)
    assert abs(v_coarse - v_theory) > abs(v_fine - v_theory)