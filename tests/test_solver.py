import numpy as np
import pytest

from fem.assembly import assemble_stiffness
from fem.boundary import add_edge_traction, fixed_dofs, nodes_on_line
from fem.material import Material
from fem.mesh import rectangle_mesh
from fem.solver import solve

MAT = Material(E=200e9, nu=0.3, t=0.01)


def _tension_problem(sigma=1e6, Lx=4.0, Ly=1.0, nx=8, ny=2):
    nodes, elems = rectangle_mesh(Lx, Ly, nx, ny)
    K = assemble_stiffness(nodes, elems, MAT)
    F = np.zeros(2 * len(nodes))
    right = nodes_on_line(nodes, x=Lx)
    left = nodes_on_line(nodes, x=0.0)
    add_edge_traction(F, nodes, right, sigma, 0.0, MAT.t)
    fixed = np.concatenate([
        fixed_dofs(left, x=True, y=False),   # left edge: no x-motion
        fixed_dofs([0], x=False, y=True),    # origin node: no y-motion (stops rigid drift)
    ])
    return nodes, K, F, fixed


def test_edge_traction_total_force():
    sigma, Ly = 1e6, 1.0
    nodes, K, F, fixed = _tension_problem(sigma=sigma, Ly=Ly)
    assert F[0::2].sum() == pytest.approx(sigma * Ly * MAT.t)
    assert F[1::2].sum() == pytest.approx(0.0, abs=1e-9)


def test_uniaxial_tension_matches_exact_solution():
    """Uniform stress is inside the CST element space, so the FE answer is exact."""
    sigma = 1e6
    nodes, K, F, fixed = _tension_problem(sigma=sigma)
    u, _ = solve(K, F, fixed)

    ux_exact = sigma * nodes[:, 0] / MAT.E
    uy_exact = -MAT.nu * sigma * nodes[:, 1] / MAT.E
    assert np.allclose(u[0::2], ux_exact, rtol=1e-6, atol=1e-12)
    assert np.allclose(u[1::2], uy_exact, rtol=1e-6, atol=1e-12)


def test_reactions_balance_applied_load():
    sigma, Ly = 1e6, 1.0
    nodes, K, F, fixed = _tension_problem(sigma=sigma, Ly=Ly)
    _, R = solve(K, F, fixed)
    assert R[0::2].sum() == pytest.approx(-sigma * Ly * MAT.t, rel=1e-6)
    assert R[1::2].sum() == pytest.approx(0.0, abs=1e-3)


def test_cantilever_tip_moves_down_under_downward_load():
    Lx, Ly = 10.0, 1.0
    nodes, elems = rectangle_mesh(Lx, Ly, 20, 2)
    K = assemble_stiffness(nodes, elems, MAT)
    F = np.zeros(2 * len(nodes))
    add_edge_traction(F, nodes, nodes_on_line(nodes, x=Lx), 0.0, -1e5, MAT.t)
    fixed = fixed_dofs(nodes_on_line(nodes, x=0.0))          # clamp the left edge

    u, R = solve(K, F, fixed)
    tip = nodes_on_line(nodes, x=Lx)
    assert np.all(u[2 * tip + 1] < 0)                        # tip moves down
    assert R[1::2].sum() == pytest.approx(-F[1::2].sum(), rel=1e-6)