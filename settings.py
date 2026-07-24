"""Application configuration and validated default values."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RatioRegion:
    """One total-elevation interval with a GH-to-ST incremental ratio."""

    start_deg: float
    end_deg: float
    gh_to_st_ratio: float

    def __post_init__(self) -> None:
        if self.start_deg < 0 or self.end_deg <= self.start_deg:
            raise ValueError("Ratio-region bounds must be increasing and non-negative.")
        if self.gh_to_st_ratio <= 0:
            raise ValueError("A GH-to-ST ratio must be positive.")


def default_ratio_regions() -> tuple[RatioRegion, ...]:
    """Return the configurable default shoulder-rhythm schedule."""

    return (
        RatioRegion(0.0, 30.0, 4.0),
        RatioRegion(30.0, 90.0, 2.0),
        RatioRegion(90.0, 114.0, 1.0),
        RatioRegion(114.0, 180.0, 2.0),
    )


@dataclass
class Settings:
    """Mutable settings shared by simulation, GUI, rendering, and exporters."""

    center_distance: float = 100.0
    input_radius: float = 50.0
    output_radius: float = 50.0
    simulation_speed_deg_s: float = 30.0
    max_elevation_deg: float = 180.0
    input_revolutions_per_elevation_cycle: float = 1.0
    pitch_curve_samples: int = 4097
    frame_interval_ms: int = 16
    ratio_regions: tuple[RatioRegion, ...] = field(default_factory=default_ratio_regions)

    show_pitch_curves: bool = True
    show_teeth: bool = True
    show_tooth_centers: bool = False
    show_tooth_numbers: bool = False
    show_base_curve: bool = False
    show_root_curve: bool = False
    show_addendum_curve: bool = False
    show_normals: bool = False
    show_tangents: bool = False
    show_gear_centers: bool = True
    show_contact_point: bool = True
    show_instantaneous_ratio: bool = True
    show_velocity_vectors: bool = False
    show_radius_vectors: bool = True
    show_ratio_graph: bool = False
    show_gh_angle: bool = True
    show_st_angle: bool = True
    advanced_debug: bool = False

    def validate(self) -> None:
        """Validate values that may be edited through the GUI."""

        if self.center_distance <= 0:
            raise ValueError("Center distance must be positive.")
        if self.input_radius <= 0 or self.output_radius <= 0:
            raise ValueError("Gear radii must be positive.")
        if self.simulation_speed_deg_s <= 0:
            raise ValueError("Simulation speed must be positive.")
        if self.max_elevation_deg <= 0:
            raise ValueError("Maximum elevation must be positive.")
        if self.frame_interval_ms <= 0:
            raise ValueError("Frame interval must be positive.")
        if self.pitch_curve_samples < 2001:
            raise ValueError("At least 2001 pitch-curve samples are required.")
