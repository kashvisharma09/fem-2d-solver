import numpy as np
import pytest

from fem.material import Material
from fem.mesh import rectangle_mesh
from fem.postprocess import element_strain_stress, nodal_average, von_mises

MAT = Material(E=200e9, nu=0.3, t=0.01)


def test_uniform_strain_field_recovers_stress():
    """Impose an exact uniaxial-tension displacement field and check sigma_xx = E*exx."""
    nodes, elems = rectangle_mesh(2.0, 1.0, 4, 2)
    sigma = 1e6
    exx = sigma / MAT.E
    eyy = -MAT.nu * exx
    u = np.zeros(2 * len(nodes))
    u[0::2] = exx * nodes[:, 0]
    u[1::2] = eyy * nodes[:, 1]

    strains, stresses = element_strain_stress(nodes, elems, u, MAT.D)
    assert np.allclose(strains[:, 0], exx, rtol=1e-6)
    assert np.allclose(stresses[:, 0], sigma, rtol=1e-6)
    assert np.allclose(stresses[:, 1], 0.0, atol=1.0)   # sigma_yy ~ 0


def test_von_mises_matches_uniaxial_formula():
    """For pure uniaxial stress, von Mises equals |sigma_xx|."""
    stresses = np.array([[5e6, 0.0, 0.0], [-3e6, 0.0, 0.0]])
    vm = von_mises(stresses)
    assert vm == pytest.approx([5e6, 3e6])


def test_von_mises_pure_shear():
    """For pure shear tau, von Mises = sqrt(3) * tau."""
    tau = 2e6
    stresses = np.array([[0.0, 0.0, tau]])
    assert von_mises(stresses) == pytest.approx([np.sqrt(3) * tau])


def test_nodal_average_shape_and_value():
    nodes, elems = rectangle_mesh(2.0, 1.0, 4, 2)
    element_values = np.ones(len(elems)) * 7.0
    avg = nodal_average(nodes, elems, element_values)
    assert avg.shape == (len(nodes),)
    assert np.allclose(avg, 7.0)   # constant field averages to the same constant