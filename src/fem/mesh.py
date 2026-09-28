"""Structured triangular mesh for a rectangle."""
import numpy as np


def rectangle_mesh(Lx: float, Ly: float, nx: int, ny: int):
    """Mesh the rectangle [0, Lx] x [0, Ly] with nx * ny cells, 2 triangles each.

    Returns
    -------
    nodes : (N, 2) float array of node coordinates
    elements : (M, 3) int array of node indices, counter-clockwise
    """
    xs = np.linspace(0.0, Lx, nx + 1)
    ys = np.linspace(0.0, Ly, ny + 1)
    X, Y = np.meshgrid(xs, ys)                       # shape (ny+1, nx+1)
    nodes = np.column_stack([X.ravel(), Y.ravel()])  # node (i, j) -> j*(nx+1) + i

    elements = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i          # bottom-left
            n1 = n0 + 1                    # bottom-right
            n2 = n0 + (nx + 1) + 1         # top-right
            n3 = n0 + (nx + 1)             # top-left
            elements.append([n0, n1, n2])
            elements.append([n0, n2, n3])
    return nodes, np.array(elements, dtype=int)