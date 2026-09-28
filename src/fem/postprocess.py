"""Post-processing: element strains/stresses and von Mises, nodal averaging."""
import numpy as np

from .elements import b_matrix


def element_strain_stress(nodes: np.ndarray, elements: np.ndarray, u: np.ndarray, D: np.ndarray):
    """Constant strain/stress per element.

    Returns
    -------
    strains : (M, 3) array, columns [exx, eyy, gxy]
    stresses : (M, 3) array, columns [sxx, syy, sxy]
    """
    M = len(elements)
    strains = np.zeros((M, 3))
    stresses = np.zeros((M, 3))
    for e, tri in enumerate(elements):
        B, _ = b_matrix(nodes[tri])
        dofs = np.array([[2 * n, 2 * n + 1] for n in tri]).ravel()
        eps = B @ u[dofs]
        strains[e] = eps
        stresses[e] = D @ eps
    return strains, stresses


def von_mises(stresses: np.ndarray) -> np.ndarray:
    """Plane-stress von Mises stress from columns [sxx, syy, sxy]."""
    sxx, syy, sxy = stresses[:, 0], stresses[:, 1], stresses[:, 2]
    return np.sqrt(sxx**2 - sxx * syy + syy**2 + 3.0 * sxy**2)


def nodal_average(nodes: np.ndarray, elements: np.ndarray, element_values: np.ndarray) -> np.ndarray:
    """Average a per-element scalar onto nodes (simple unweighted average),
    for smoother contour plots. Returns a (N,) array."""
    sums = np.zeros(len(nodes))
    counts = np.zeros(len(nodes))
    for tri, val in zip(elements, element_values):
        sums[tri] += val
        counts[tri] += 1
    counts[counts == 0] = 1.0   # avoid division by zero for unused nodes
    return sums / counts