"""Smooth transmission synthesis and conjugate noncircular pitch curves.

For external gears with fixed center distance ``a`` and transmission
``m(phi) = d(psi)/d(phi) > 0``, equality of pitch-point tangential velocities
gives ``r1 * omega1 = r2 * omega2``. Therefore:

``m = r1 / r2``

Combined with ``r1 + r2 = a``:

``r1 = a*m/(1+m)`` and ``r2 = a/(1+m)``.

The pitch curves are centrodes expressed in each rotating gear's local frame.
When Gear A rotates by ``+phi`` and Gear B by ``-psi``, their sampled contact
rays align with the fixed line of centers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
from shapely.geometry import Polygon

from kinematics import ShoulderModel, ShoulderState


@dataclass(frozen=True)
class TransmissionSample:
    """Smooth kinematics and instantaneous ratio at one input phase."""

    input_rad: float
    output_rad: float
    gear_ratio: float
    elevation_deg: float
    gh_deg: float
    st_deg: float
    gh_to_st_ratio: float


class SmoothTransmission:
    """Periodic cubic transmission generated from editable biomechanics.

    A closed 1:1-cycle noncircular pair requires ``psi(2*pi)=2*pi``. The
    biological ST contribution finishes at 60 degrees, so output gear angle is
    normalized such that one output revolution represents the final ST angle.

    To obtain a truly periodic C2 function, a spline is fit to the residual
    ``g(phi)=psi(phi)-phi``. Since ``g(0)=g(2*pi)=0``, SciPy's periodic cubic
    boundary condition makes position, velocity, and acceleration continuous
    across the revolution seam.
    """

    def __init__(self, shoulder_model) -> None:
        self.shoulder_model = shoulder_model
        elevations = np.asarray(shoulder_model.control_elevations_deg, dtype=float)
        self.elevation_start_deg = float(elevations[0])
        self.elevation_end_deg = float(elevations[-1])
        self.elevation_span_deg = (
            self.elevation_end_deg - self.elevation_start_deg
        )
        st_values = np.asarray(shoulder_model.st_angle_at(elevations), dtype=float)
        self.st_start_deg = float(st_values[0])
        self.final_st_deg = float(st_values[-1] - st_values[0])
        if self.final_st_deg <= 0:
            raise ValueError("Final scapular rotation must be positive.")

        input_rad = (
            (elevations - self.elevation_start_deg)
            / self.elevation_span_deg
            * 2.0
            * np.pi
        )
        output_rad = np.radians(
            (st_values - self.st_start_deg) / self.final_st_deg * 360.0
        )
        residual = output_rad - input_rad
        if abs(residual[0] - residual[-1]) > 1e-12:
            raise ValueError("Normalized transmission must close after one revolution.")
        self._residual_spline = CubicSpline(input_rad, residual, bc_type="periodic")

    def output_angle(self, input_rad: NDArray[np.float64] | float) -> NDArray[np.float64] | float:
        """Return unwrapped driven-gear angle ``psi(phi)``."""

        value = np.asarray(input_rad)
        revolutions = np.floor(value / (2.0 * np.pi))
        wrapped = value - revolutions * 2.0 * np.pi
        result = value + self._residual_spline(wrapped)
        return float(result) if np.ndim(input_rad) == 0 else result

    def ratio(self, input_rad: NDArray[np.float64] | float) -> NDArray[np.float64] | float:
        """Return positive instantaneous gear ratio ``dpsi/dphi``."""

        wrapped = np.asarray(input_rad) % (2.0 * np.pi)
        result = 1.0 + self._residual_spline(wrapped, 1)
        return float(result) if np.ndim(input_rad) == 0 else result

    def ratio_derivative(
        self, input_rad: NDArray[np.float64] | float
    ) -> NDArray[np.float64] | float:
        """Return ``d²psi/dphi²``, used for biomechanical acceleration."""

        wrapped = np.asarray(input_rad) % (2.0 * np.pi)
        result = self._residual_spline(wrapped, 2)
        return float(result) if np.ndim(input_rad) == 0 else result

    def evaluate(self, input_rad: float) -> TransmissionSample:
        """Map gear phase back to physical GH and ST angles."""

        wrapped = input_rad % (2.0 * np.pi)
        # Preserve the upper endpoint for a 0..180-degree animation sweep.
        phase = (
            2.0 * np.pi
            if input_rad > 0.0 and abs(wrapped) < 1e-12
            else wrapped
        )
        output = float(self.output_angle(phase))
        gear_ratio = float(self.ratio(phase))
        elevation = (
            self.elevation_start_deg
            + phase / (2.0 * np.pi) * self.elevation_span_deg
        )
        st = self.st_start_deg + output / (2.0 * np.pi) * self.final_st_deg
        try:
            gh = float(self.shoulder_model.gh_angle_at(elevation))
        except RuntimeError:
            gh = elevation - (st - self.st_start_deg)

        # dST/dE follows from normalized output and input revolutions.
        st_fraction = gear_ratio * self.final_st_deg / self.elevation_span_deg
        biomechanical_ratio = (1.0 - st_fraction) / st_fraction
        return TransmissionSample(
            input_rad=phase,
            output_rad=output,
            gear_ratio=gear_ratio,
            elevation_deg=elevation,
            gh_deg=gh,
            st_deg=st,
            gh_to_st_ratio=biomechanical_ratio,
        )


@dataclass(frozen=True)
class PitchCurveData:
    """Double-precision synthesis arrays for both conjugate gears."""

    input_rad: NDArray[np.float64]
    output_rad: NDArray[np.float64]
    ratio: NDArray[np.float64]
    input_radii: NDArray[np.float64]
    output_radii: NDArray[np.float64]
    input_points: NDArray[np.float64]
    output_points: NDArray[np.float64]
    center_distance: float


@dataclass(frozen=True)
class MeshingValidation:
    """Numerical validation results shown in the engineering panel."""

    constant_center_distance: bool
    continuous_motion: bool
    no_pitch_curve_overlap: bool
    smooth_velocity_ratio: bool
    ready_for_tooth_generation: bool
    maximum_center_error: float
    minimum_ratio: float
    maximum_ratio: float
    input_curve_valid: bool
    output_curve_valid: bool
    warnings: tuple[str, ...]


def synthesize_pitch_curves(
    transmission: SmoothTransmission,
    center_distance: float,
    sample_count: int,
) -> PitchCurveData:
    """Generate a full fixed-center conjugate pitch-curve pair."""

    if center_distance <= 0:
        raise ValueError("Center distance must be positive.")
    if sample_count < 2001:
        raise ValueError("At least 2001 samples are required.")

    phi = np.linspace(0.0, 2.0 * np.pi, sample_count, endpoint=True, dtype=np.float64)
    psi = np.asarray(transmission.output_angle(phi), dtype=np.float64)
    ratio = np.asarray(transmission.ratio(phi), dtype=np.float64)
    if np.any(ratio <= 0):
        raise ValueError("Transmission derivative must remain positive.")

    r_input = center_distance * ratio / (1.0 + ratio)
    r_output = center_distance / (1.0 + ratio)

    # Gear-local contact rays rotate opposite the respective world rotations.
    input_points = np.column_stack(
        (r_input * np.cos(-phi), r_input * np.sin(-phi))
    )
    output_local_angle = np.pi + psi
    output_points = np.column_stack(
        (r_output * np.cos(output_local_angle), r_output * np.sin(output_local_angle))
    )
    return PitchCurveData(
        input_rad=phi,
        output_rad=psi,
        ratio=ratio,
        input_radii=r_input,
        output_radii=r_output,
        input_points=input_points,
        output_points=output_points,
        center_distance=center_distance,
    )


def validate_pitch_curves(data: PitchCurveData) -> MeshingValidation:
    """Check closure, conjugacy, simplicity, convexity, and smooth velocity."""

    warnings: list[str] = []
    center_error = np.abs(data.input_radii + data.output_radii - data.center_distance)
    maximum_center_error = float(np.max(center_error))
    constant_center = maximum_center_error <= max(1e-10, data.center_distance * 1e-12)

    ratio_finite = bool(np.all(np.isfinite(data.ratio)))
    ratio_positive = bool(np.min(data.ratio) > 0)
    seam_ratio_error = abs(float(data.ratio[0] - data.ratio[-1]))
    smooth_ratio = ratio_finite and ratio_positive and seam_ratio_error < 1e-9

    input_polygon = Polygon(data.input_points)
    output_polygon = Polygon(data.output_points)
    input_valid = input_polygon.is_valid and input_polygon.exterior.is_simple
    output_valid = output_polygon.is_valid and output_polygon.exterior.is_simple
    input_convex = abs(input_polygon.area - input_polygon.convex_hull.area) <= input_polygon.area * 1e-9
    output_convex = abs(output_polygon.area - output_polygon.convex_hull.area) <= output_polygon.area * 1e-9

    placed_output = Polygon(
        data.output_points + np.array([data.center_distance, 0.0])
    )
    overlap_area = input_polygon.intersection(placed_output).area
    overlap_free = (
        input_valid
        and output_valid
        and input_convex
        and output_convex
        and overlap_area <= data.center_distance**2 * 1e-12
    )

    closure_error = max(
        float(np.linalg.norm(data.input_points[0] - data.input_points[-1])),
        float(np.linalg.norm(data.output_points[0] - data.output_points[-1])),
        abs(float(data.output_rad[-1] - data.output_rad[0] - 2.0 * np.pi)),
    )
    continuous = closure_error < 1e-8

    if not constant_center:
        warnings.append("Pitch-radius sum varies beyond numerical tolerance.")
    if not continuous:
        warnings.append("Pitch curves or transmission do not close continuously.")
    if not overlap_free:
        warnings.append("Pitch curves are invalid, non-convex, or overlap.")
    if not smooth_ratio:
        warnings.append("Velocity ratio is non-positive or discontinuous.")

    ready = constant_center and continuous and overlap_free and smooth_ratio
    return MeshingValidation(
        constant_center_distance=constant_center,
        continuous_motion=continuous,
        no_pitch_curve_overlap=overlap_free,
        smooth_velocity_ratio=smooth_ratio,
        ready_for_tooth_generation=ready,
        maximum_center_error=maximum_center_error,
        minimum_ratio=float(np.min(data.ratio)),
        maximum_ratio=float(np.max(data.ratio)),
        input_curve_valid=input_valid,
        output_curve_valid=output_valid,
        warnings=tuple(warnings),
    )
