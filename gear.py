"""Gear-domain objects independent of rendering and GUI code."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pitch_curve import PitchCurve


@dataclass
class Gear:
    """A gear shaft, pitch curve, and current orientation."""

    name: str
    pitch_curve: PitchCurve
    center_x: float
    center_y: float
    angle_deg: float = 0.0

    @property
    def center(self) -> NDArray[np.float64]:
        """Return the shaft center as a two-element array."""

        return np.array([self.center_x, self.center_y], dtype=float)

    def world_points(self, sample_count: int = 256) -> NDArray[np.float64]:
        """Rotate and translate local pitch-curve points into world space."""

        local = self.pitch_curve.points(sample_count)
        angle = np.radians(self.angle_deg)
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        return local @ rotation.T + self.center

