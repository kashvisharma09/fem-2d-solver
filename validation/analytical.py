"""Closed-form solutions used to validate the FE solver."""
import numpy as np


def cantilever_tip_deflection(P: float, L: float, h: float, t: float,
                               E: float, nu: float, shear_correction: bool = True) -> float:
    """Downward tip deflection of a cantilever beam under a tip force P [N].

    L: length, h: height (in the bending plane), t: out-of-plane thickness.
    Euler-Bernoulli term plus an optional Timoshenko shear term (rectangular
    cross-section shear correction factor k = 5/6).
    """
    I = t * h**3 / 12.0
    v_bending = P * L**3 / (3.0 * E * I)

    if not shear_correction:
        return v_bending

    G = E / (2.0 * (1.0 + nu))
    A = h * t
    k = 5.0 / 6.0
    v_shear = P * L / (k * G * A)
    return v_bending + v_shear


def cantilever_max_bending_stress(P: float, L: float, h: float, t: float) -> float:
    """Maximum bending stress (at the fixed end, outer fiber) for a tip-loaded
    cantilever, from sigma = M*c/I."""
    I = t * h**3 / 12.0
    M = P * L          # bending moment at the fixed end
    c = h / 2.0
    return M * c / I