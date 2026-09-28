"""Boundary conditions: selecting nodes, fixing DOFs, applying loads.

DOF numbering: node n has DOFs 2n (x) and 2n+1 (y).
"""
import numpy as np


def nodes_on_line(nodes: np.ndarray, *, x=None, y=None, tol: float = 1e-9) -> np.ndarray:
    """Indices of nodes on the vertical line x=const or horizontal line y=const."""
    if (x is None) == (y is None):
        raise ValueError("Give exactly one of x= or y=")
    coord = nodes[:, 0] if x is not None else nodes[:, 1]
    value = x if x is not None else y
    return np.where(np.abs(coord - value) < tol)[0]


def fixed_dofs(node_ids, x: bool = True, y: bool = True) -> np.ndarray:
    """DOF indices to constrain at the given nodes (x and/or y direction)."""
    dofs = []
    for n in np.atleast_1d(node_ids):
        if x:
            dofs.append(2 * int(n))
        if y:
            dofs.append(2 * int(n) + 1)
    return np.array(dofs, dtype=int)


def add_point_load(F: np.ndarray, node_id: int, fx: float = 0.0, fy: float = 0.0) -> None:
    """Add a concentrated force [N] at a node (modifies F in place)."""
    F[2 * node_id] += fx
    F[2 * node_id + 1] += fy


def add_edge_traction(F: np.ndarray, nodes: np.ndarray, edge_nodes,
                      tx: float, ty: float, t: float) -> None:
    """Add a uniform traction (tx, ty) [Pa] along a straight edge (modifies F in place).

    Consistent nodal loads for linear elements: every segment between two
    neighbouring edge nodes puts half of (traction * thickness * length) on each end.
    """
    edge_nodes = np.asarray(edge_nodes, dtype=int)
    pts = nodes[edge_nodes]
    axis = 0 if np.ptp(pts[:, 0]) >= np.ptp(pts[:, 1]) else 1
    ids = edge_nodes[np.argsort(pts[:, axis])]        # order along the edge

    for a, b in zip(ids[:-1], ids[1:]):
        length = np.linalg.norm(nodes[b] - nodes[a])
        f = 0.5 * length * t * np.array([tx, ty])
        F[2 * a: 2 * a + 2] += f
        F[2 * b: 2 * b + 2] += f