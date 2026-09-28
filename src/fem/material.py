"""Material model for 2D plane-stress linear elasticity."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Material:
    E: float          # Young's modulus [Pa]
    nu: float         # Poisson's ratio [-]
    t: float = 1.0    # plate thickness [m]

    @property
    def D(self) -> np.ndarray:
        """Plane-stress constitutive matrix (3x3): sigma = D @ epsilon,
        with the ordering [xx, yy, xy] (engineering shear strain)."""
        c = self.E / (1.0 - self.nu ** 2)
        return c * np.array([
            [1.0,     self.nu, 0.0],
            [self.nu, 1.0,     0.0],
            [0.0,     0.0,     (1.0 - self.nu) / 2.0],
        ])