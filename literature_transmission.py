"""Full-cycle and partial-sector transmission alternatives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline

from biomechanics.literature_model import LiteratureShoulderModel


@dataclass(frozen=True)
class TransmissionComparison:
    elevation_deg: np.ndarray
    target_st_deg: np.ndarray
    full_cycle_st_deg: np.ndarray
    sector_st_deg: np.ndarray
    target_derivative: np.ndarray
    full_cycle_derivative: np.ndarray
    sector_derivative: np.ndarray
    full_cycle_ratio: np.ndarray
    sector_ratio: np.ndarray
    full_cycle_input_radius: np.ndarray
    full_cycle_output_radius: np.ndarray
    sector_input_radius: np.ndarray
    sector_output_radius: np.ndarray
    confidence_lower_deg: np.ndarray
    confidence_upper_deg: np.ndarray
    full_cycle_max_error_deg: float
    full_cycle_rms_error_deg: float
    sector_max_error_deg: float
    sector_rms_error_deg: float


class SectorTransmission:
    """Non-periodic transmission over only the measured shoulder sector."""

    def __init__(self, model: LiteratureShoulderModel) -> None:
        self.model = model
        self.valid_range_deg = model.valid_range_deg
        self.requires_hard_stops = True
        self.reset_behavior = (
            "Return to the lower mechanical hard stop without wrapping; "
            "no periodic biological continuation is defined."
        )

    def output_angle_deg(self, elevation_deg):
        start = self.model.st_angle_at(self.valid_range_deg[0])
        return np.asarray(self.model.st_angle_at(elevation_deg)) - start

    def ratio(self, elevation_deg):
        return self.model.dst_delevation_at(elevation_deg)


def compare_transmission_alternatives(
    model: LiteratureShoulderModel,
    center_distance: float = 100.0,
    sample_count: int = 1001,
) -> TransmissionComparison:
    low, high = model.valid_range_deg
    elevation = np.linspace(low, high, sample_count)
    target_st = np.asarray(model.st_angle_at(elevation))
    target_derivative = np.asarray(model.dst_delevation_at(elevation))
    st_start = float(target_st[0])
    excursion = float(target_st[-1] - target_st[0])
    phase = (elevation - low) / (high - low) * 2.0 * np.pi

    # Closed-cycle alternative: impose periodic C2 closure on the normalized
    # residual, exposing the endpoint distortion rather than hiding it.
    knots = model.control_elevations_deg
    knot_phase = (knots - low) / (high - low) * 2.0 * np.pi
    knot_st = np.asarray(model.st_angle_at(knots))
    normalized_output = (knot_st - st_start) / excursion * 2.0 * np.pi
    residual = normalized_output - knot_phase
    residual[-1] = residual[0]
    closed = CubicSpline(knot_phase, residual, bc_type="periodic")
    output_phase = phase + closed(phase)
    full_st = st_start + output_phase / (2.0 * np.pi) * excursion
    mechanical_ratio = 1.0 + closed(phase, 1)
    full_derivative = mechanical_ratio * excursion / (high - low)

    sector = SectorTransmission(model)
    sector_st = st_start + sector.output_angle_deg(elevation)
    sector_derivative = np.asarray(sector.ratio(elevation))
    sector_ratio = sector_derivative

    def radii(ratio):
        if np.any(ratio <= 0):
            return np.full_like(ratio, np.nan), np.full_like(ratio, np.nan)
        first = center_distance * ratio / (1.0 + ratio)
        return first, center_distance - first

    full_r1, full_r2 = radii(mechanical_ratio)
    sector_r1, sector_r2 = radii(sector_ratio)
    uncertainty = model.uncertainty_at(elevation)
    full_error = full_st - target_st
    sector_error = sector_st - target_st
    return TransmissionComparison(
        elevation_deg=elevation,
        target_st_deg=target_st,
        full_cycle_st_deg=full_st,
        sector_st_deg=sector_st,
        target_derivative=target_derivative,
        full_cycle_derivative=full_derivative,
        sector_derivative=sector_derivative,
        full_cycle_ratio=mechanical_ratio,
        sector_ratio=sector_ratio,
        full_cycle_input_radius=full_r1,
        full_cycle_output_radius=full_r2,
        sector_input_radius=sector_r1,
        sector_output_radius=sector_r2,
        confidence_lower_deg=np.asarray(uncertainty["confidence_lower_deg"]),
        confidence_upper_deg=np.asarray(uncertainty["confidence_upper_deg"]),
        full_cycle_max_error_deg=float(np.max(np.abs(full_error))),
        full_cycle_rms_error_deg=float(np.sqrt(np.mean(full_error**2))),
        sector_max_error_deg=float(np.max(np.abs(sector_error))),
        sector_rms_error_deg=float(np.sqrt(np.mean(sector_error**2))),
    )
