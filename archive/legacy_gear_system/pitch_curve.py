"""Pitch-curve abstractions used by gears and future conjugate geometry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


class PitchCurve(ABC):
    """Abstract two-dimensional pitch curve centered on its shaft."""

    @abstractmethod
    def points(self, sample_count: int = 256) -> NDArray[np.float64]:
        """Return a closed Nx2 point array in local coordinates."""

    @abstractmethod
    def radius_at(self, angle_rad: float) -> float:
        """Return polar radius at *angle_rad*."""


@dataclass
class CircularPitchCurve(PitchCurve):
    """Circular Phase-1 pitch curve."""

    radius: float

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError("Pitch radius must be positive.")

    def points(self, sample_count: int = 256) -> NDArray[np.float64]:
        if sample_count < 16:
            raise ValueError("At least 16 samples are required.")
        angles = np.linspace(0.0, 2.0 * np.pi, sample_count, endpoint=True)
        return np.column_stack((self.radius * np.cos(angles), self.radius * np.sin(angles)))

    def radius_at(self, angle_rad: float) -> float:
        del angle_rad
        return self.radius


@dataclass
class SampledPitchCurve(PitchCurve):
    """Exact sampled noncircular pitch curve in gear-local coordinates."""

    coordinates: NDArray[np.float64]
    parameter_angles: NDArray[np.float64]
    radii: NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.coordinates.ndim != 2 or self.coordinates.shape[1] != 2:
            raise ValueError("Coordinates must be an Nx2 array.")
        if len(self.coordinates) < 2001:
            raise ValueError("A synthesized pitch curve requires at least 2001 points.")
        if len(self.parameter_angles) != len(self.coordinates):
            raise ValueError("Parameter and coordinate arrays must have equal lengths.")
        if len(self.radii) != len(self.coordinates):
            raise ValueError("Radius and coordinate arrays must have equal lengths.")
        if np.any(self.radii <= 0):
            raise ValueError("Every pitch radius must be positive.")

    def points(self, sample_count: int = 256) -> NDArray[np.float64]:
        """Return exact data or an evenly indexed display subset."""

        if sample_count >= len(self.coordinates):
            return self.coordinates.copy()
        indices = np.linspace(0, len(self.coordinates) - 1, sample_count).astype(int)
        return self.coordinates[indices]

    def radius_at(self, angle_rad: float) -> float:
        """Interpolate radius by synthesis parameter, periodically."""

        wrapped = angle_rad % (2.0 * np.pi)
        return float(np.interp(wrapped, self.parameter_angles, self.radii))
