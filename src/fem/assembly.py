"""Global stiffness matrix assembly (sparse)."""
import numpy as np
from scipy.sparse import coo_matrix

from .elements import element_stiffness
from .material import Material


def element_dofs(tri: np.ndarray) -> np.ndarray:
    """Global DOF indices [2n, 2n+1] for each node of a triangle (length 6)."""
    return np.array([[2 * n, 2 * n + 1] for n in tri]).ravel()


def assemble_stiffness(nodes: np.ndarray, elements: np.ndarray, material: Material):
    """Assemble the global stiffness matrix K (2N x 2N, CSR format)."""
    n_dof = 2 * len(nodes)
    D = material.D

    rows, cols, vals = [], [], []
    for tri in elements:
        Ke = element_stiffness(nodes[tri], D, material.t)
        dofs = element_dofs(tri)
        rows.append(np.repeat(dofs, 6))
        cols.append(np.tile(dofs, 6))
        vals.append(Ke.ravel())

    # duplicate (row, col) entries are summed when converting to CSR
    K = coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_dof, n_dof),
    ).tocsr()
    return K