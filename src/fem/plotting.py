"""Plotting helpers: mesh, deformed shape, and stress contours."""
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np


def plot_mesh(nodes: np.ndarray, elements: np.ndarray, ax=None, **kwargs):
    ax = ax or plt.gca()
    triang = mtri.Triangulation(nodes[:, 0], nodes[:, 1], elements)
    ax.triplot(triang, color="0.4", linewidth=0.5, **kwargs)
    ax.set_aspect("equal")
    return ax


def plot_deformed(nodes: np.ndarray, elements: np.ndarray, u: np.ndarray,
                   scale: float = 1.0, ax=None):
    ax = ax or plt.gca()
    plot_mesh(nodes, elements, ax=ax)
    deformed = nodes + scale * u.reshape(-1, 2)
    triang = mtri.Triangulation(deformed[:, 0], deformed[:, 1], elements)
    ax.triplot(triang, color="crimson", linewidth=0.8)
    ax.set_aspect("equal")
    ax.set_title(f"Deformed shape (scale = {scale:g}x)")
    return ax


def plot_contour(nodes: np.ndarray, elements: np.ndarray, nodal_values: np.ndarray,
                  ax=None, label: str = "", cmap: str = "viridis"):
    ax = ax or plt.gca()
    triang = mtri.Triangulation(nodes[:, 0], nodes[:, 1], elements)
    tpc = ax.tricontourf(triang, nodal_values, levels=20, cmap=cmap)
    plt.colorbar(tpc, ax=ax, label=label)
    ax.set_aspect("equal")
    return ax