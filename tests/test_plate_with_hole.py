import numpy as np
import pytest

from fem.assembly import assemble_stiffness
from fem.boundary import add_edge_traction, fixed_dofs, nodes_on_line
from fem.elements import signed_area
from fem.material import Material
from fem.mesh_gmsh import plate_with_hole_mesh
from fem.postprocess import element_strain_stress, nodal_average
from fem.solver import solve
from validation.analytical import kirsch_stress_concentration_factor

MAT = Material(E=200e9, nu=0.3, t=0.01)


def _build_and_solve(W=20.0, H=10.0, r=1.0, sigma=1e6, mesh_size_far=1.0):
    nodes, elems = plate_with_hole_mesh(W, H, r, mesh_size_far)
    K = assemble_stiffness(nodes, elems, MAT)
    F = np.zeros(2 * len(nodes))
    right = nodes_on_line(nodes, x=W)
    left = nodes_on_line(nodes, x=0.0)
    add_edge_traction(F, nodes, right, sigma, 0.0, MAT.t)
    fixed = np.concatenate([
        fixed_dofs(left, x=True, y=False),
        fixed_dofs([left[0]], x=False, y=True),
    ])
    u, R = solve(K, F, fixed)
    strains, stresses = element_strain_stress(nodes, elems, u, MAT.D)
    return nodes, elems, u, R, stresses


def _kt_from_fem(nodes, elems, stresses, cx, cy, r, sigma, angle_tol_deg=15.0):
    """Estimate the FEM stress concentration factor near theta = +-90 deg,
    the points of maximum hoop stress for uniaxial tension."""
    sxx_nodal = nodal_average(nodes, elems, stresses[:, 0])
    dx, dy = nodes[:, 0] - cx, nodes[:, 1] - cy
    radius = np.hypot(dx, dy)
    theta = np.degrees(np.arctan2(dy, dx))
    near_hole = radius < r * 1.15
    near_90 = np.abs(np.abs(theta) - 90.0) < angle_tol_deg
    mask = near_hole & near_90
    assert mask.sum() > 0, "No nodes found near the hole boundary at theta=90"
    return sxx_nodal[mask].max() / sigma


def test_mesh_has_a_hole():
    """The meshed area should match (rectangle area - hole area), confirming
    the hole was actually cut out of the plate."""
    W, H, r = 20.0, 10.0, 1.0
    nodes, elems = plate_with_hole_mesh(W, H, r, mesh_size_far=1.0)
    area = sum(signed_area(nodes[e]) for e in elems)
    expected = W * H - np.pi * r**2
    assert area == pytest.approx(expected, rel=0.02)


def test_reactions_balance_applied_load():
    W, H, r, sigma = 20.0, 10.0, 1.0, 1e6
    _, _, _, R, _ = _build_and_solve(W, H, r, sigma, mesh_size_far=1.0)
    assert R[0::2].sum() == pytest.approx(-sigma * H * MAT.t, rel=0.02)


def test_stress_concentration_matches_kirsch():
    """FEM hoop stress at the hole equator should be close to Kt = 3
    (Kirsch solution for a small hole in a wide plate under tension)."""
    W, H, r, sigma = 20.0, 10.0, 1.0, 1e6
    nodes, elems, u, R, stresses = _build_and_solve(W, H, r, sigma, mesh_size_far=0.6)
    kt_fem = _kt_from_fem(nodes, elems, stresses, cx=W / 2.0, cy=0.0, r=r, sigma=sigma)
    kt_theory = kirsch_stress_concentration_factor()
    assert kt_fem == pytest.approx(kt_theory, rel=0.25)