"""Linear static solver: K u = F with fixed DOFs."""
import numpy as np
from scipy.sparse.linalg import spsolve


def solve(K, F: np.ndarray, fixed):
    """Solve for displacements with zero-displacement supports.

    Returns
    -------
    u : (2N,) displacement vector
    R : (2N,) reaction vector (nonzero only at fixed DOFs), R = K u - F
    """
    n = K.shape[0]
    fixed = np.unique(np.asarray(fixed, dtype=int))
    free = np.setdiff1d(np.arange(n), fixed)

    Kff = K[free][:, free].tocsc()
    u = np.zeros(n)
    u[free] = spsolve(Kff, F[free])

    R = K @ u - F
    R[free] = 0.0     # remove numerical noise at free DOFs
    return u, R