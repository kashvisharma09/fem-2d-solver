import numpy as np
import pytest

from fem.assembly import assemble_stiffness
from fem.elements import signed_area
from fem.material import Material
from fem.mesh import rectangle_mesh

MAT = Material(E=200e9, nu=0.3, t=0.01)


def test_mesh_counts():
    nodes, elems = rectangle_mesh(4.0, 1.0, 8, 2)
    assert nodes.shape == (9 * 3, 2)
    assert elems.shape == (2 * 8 * 2, 3)


def test_mesh_area_and_orientation():
    Lx, Ly = 4.0, 1.0
    nodes, elems = rectangle_mesh(Lx, Ly, 8, 2)
    areas = np.array([signed_area(nodes[e]) for e in elems])
    assert np.all(areas > 0)                       # all counter-clockwise
    assert areas.sum() == pytest.approx(Lx * Ly)   # triangles tile the rectangle


def test_K_is_symmetric():
    nodes, elems = rectangle_mesh(2.0, 1.0, 4, 2)
    K = assemble_stiffness(nodes, elems, MAT)
    assert K.shape == (2 * len(nodes), 2 * len(nodes))
    assert abs(K - K.T).max() < 1e-9 * abs(K).max()


def test_K_has_exactly_three_rigid_body_modes():
    nodes, elems = rectangle_mesh(2.0, 1.0, 3, 2)
    K = assemble_stiffness(nodes, elems, MAT).toarray()
    eig = np.linalg.eigvalsh(K)
    assert np.sum(np.abs(eig) < 1e-9 * eig.max()) == 3


def test_translation_and_rotation_give_zero_force():
    nodes, elems = rectangle_mesh(2.0, 1.0, 4, 2)
    K = assemble_stiffness(nodes, elems, MAT)
    tol = 1e-8 * abs(K).max()

    u_trans = np.tile([1.0, 2.0], len(nodes))
    assert np.abs(K @ u_trans).max() < tol

    # small rigid rotation: u = -theta*y, v = theta*x
    u_rot = np.column_stack([-nodes[:, 1], nodes[:, 0]]).ravel() * 1e-3
    assert np.abs(K @ u_rot).max() < tol