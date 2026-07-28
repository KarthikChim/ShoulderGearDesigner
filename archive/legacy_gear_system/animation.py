"""GUI-independent animation timing."""

from __future__ import annotations

from dataclasses import dataclass

from utils import clamp


@dataclass
class Animator:
    """Advance, pause, reset, and step a bounded elevation coordinate."""

    maximum_deg: float
    speed_deg_s: float
    current_deg: float = 0.0
    playing: bool = False
    direction: float = 1.0

    def start(self) -> None:
        self.playing = True

    def pause(self) -> None:
        self.playing = False

    def reset(self) -> None:
        self.playing = False
        self.current_deg = 0.0
        self.direction = 1.0

    def set_position(self, elevation_deg: float) -> None:
        self.current_deg = clamp(elevation_deg, 0.0, self.maximum_deg)

    def step(self, amount_deg: float) -> None:
        self.set_position(self.current_deg + amount_deg)

    def advance(self, elapsed_s: float) -> float:
        if elapsed_s < 0:
            raise ValueError("Elapsed time cannot be negative.")
        if not self.playing:
            return self.current_deg
        next_value = self.current_deg + self.direction * self.speed_deg_s * elapsed_s
        if next_value >= self.maximum_deg:
            next_value = self.maximum_deg
            self.direction = -1.0
        elif next_value <= 0.0:
            next_value = 0.0
            self.direction = 1.0
        self.current_deg = next_value
        return self.current_deg
