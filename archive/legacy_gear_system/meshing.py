"""Gear-pair relationships and Phase-1 contact geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from gear import Gear


@dataclass
class GearPair:
    """Two pitch curves mounted at a fixed center distance."""

    input_gear: Gear
    output_gear: Gear

    @property
    def center_distance(self) -> float:
        """Return the actual shaft-center distance."""

        return float(np.linalg.norm(self.output_gear.center - self.input_gear.center))

    def contact_point(self) -> NDArray[np.float64]:
        """Return the circular Phase-1 pitch contact point.

        Future noncircular implementations can replace this with the
        instantaneous pole calculated from conjugate pitch curves.
        """

        direction = self.output_gear.center - self.input_gear.center
        magnitude = np.linalg.norm(direction)
        if magnitude == 0:
            raise ValueError("Gear centers cannot coincide.")
        unit = direction / magnitude
        radius = self.input_gear.pitch_curve.radius_at(0.0)
        return self.input_gear.center + unit * radius

    def set_angles(self, input_deg: float, output_deg: float) -> None:
        """Set both shaft orientations."""

        self.input_gear.angle_deg = input_deg
        self.output_gear.angle_deg = output_deg


@dataclass
class ConjugateGearPair(GearPair):
    """Noncircular pair with an explicitly tracked instantaneous pitch point."""

    input_pitch_radius: float = 0.0
    output_pitch_radius: float = 0.0

    def contact_point(self) -> NDArray[np.float64]:
        """Return the instantaneous pitch point on the fixed line of centers."""

        direction = self.output_gear.center - self.input_gear.center
        magnitude = np.linalg.norm(direction)
        if magnitude == 0:
            raise ValueError("Gear centers cannot coincide.")
        return self.input_gear.center + direction / magnitude * self.input_pitch_radius

    def set_contact_radii(self, input_radius: float, output_radius: float) -> None:
        """Update instantaneous radii after validating their fixed-distance sum."""

        if input_radius <= 0 or output_radius <= 0:
            raise ValueError("Pitch radii must be positive.")
        self.input_pitch_radius = input_radius
        self.output_pitch_radius = output_radius
