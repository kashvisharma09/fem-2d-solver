import numpy as np
import pytest

from fem.elements import b_matrix, element_stiffness, signed_area
from fem.material import Material

MAT = Material(E=200e9, nu=0.3, t=0.01)
TRI = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]])   # right triangle, area = 1


def test_area():
    assert signed_area(TRI) == pytest.approx(1.0)


def test_ke_is_symmetric():
    Ke = element_stiffness(TRI, MAT.D, MAT.t)
    assert np.allclose(Ke, Ke.T)


def test_rigid_body_modes():
    """A free element has exactly 3 zero-energy modes (2 translations + 1 rotation)."""
    Ke = element_stiffness(TRI, MAT.D, MAT.t)
    eig = np.linalg.eigvalsh(Ke)
    assert np.all(np.abs(eig[:3]) < 1e-9 * eig.max())
    assert eig[3] > 1e-6 * eig.max()


def test_translation_gives_zero_force():
    Ke = element_stiffness(TRI, MAT.D, MAT.t)
    u = np.array([1.0, 2.0] * 3)          # every node moves by (1, 2)
    assert np.allclose(Ke @ u, 0.0, atol=1e-6)


def test_constant_strain_recovery():
    """Patch test: a linear displacement field must reproduce its exact strain."""
    exx, eyy, gxy = 1e-3, -2e-4, 5e-4
    u = []
    for x, y in TRI:
        u += [exx * x + 0.5 * gxy * y,     # u(x, y)
              eyy * y + 0.5 * gxy * x]     # v(x, y)
    B, _ = b_matrix(TRI)
    assert np.allclose(B @ np.array(u), [exx, eyy, gxy])


def test_node_order_does_not_change_ke_energy():
    """Clockwise vs counter-clockwise ordering must give the same strain energy."""
    u = np.array([0.0, 0.0, 1e-3, 0.0, 0.0, 5e-4])
    Ke_ccw = element_stiffness(TRI, MAT.D, MAT.t)
    Ke_cw = element_stiffness(TRI[[0, 2, 1]], MAT.D, MAT.t)
    u_cw = u[[0, 1, 4, 5, 2, 3]]
    assert u @ Ke_ccw @ u == pytest.approx(u_cw @ Ke_cw @ u_cw)