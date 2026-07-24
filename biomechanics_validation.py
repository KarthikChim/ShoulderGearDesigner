"""Independent engineering validation of shoulder motion reproduction.

Mechanical ratio and biomechanical ratio are deliberately kept separate:

``m = dpsi/dphi``

The normalized driven revolution represents 60 degrees of ST rotation while
one input revolution represents 180 degrees of arm elevation. Therefore

``dST/dE = m * 60/180 = m/3``
``dGH/dE = 1 - m/3``
``GH:ST = (dGH/dE)/(dST/dE) = 3/m - 1``
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from noncircular import SmoothTransmission
from settings import RatioRegion


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ValidationCheckpoint:
    elevation_deg: float
    requested_gh_deg: float
    requested_st_deg: float
    schedule_gh_deg: float
    schedule_st_deg: float
    actual_gh_deg: float
    actual_st_deg: float
    gh_difference_deg: float
    st_difference_deg: float
    gh_percent_error: float
    st_percent_error: float
    specification_consistent: bool


@dataclass(frozen=True)
class BiomechanicsValidation:
    elevation_deg: FloatArray
    mechanical_ratio: FloatArray
    target_ratio: FloatArray
    actual_ratio: FloatArray
    target_gh_deg: FloatArray
    actual_gh_deg: FloatArray
    target_st_deg: FloatArray
    actual_st_deg: FloatArray
    gh_error_deg: FloatArray
    st_error_deg: FloatArray
    ratio_error: FloatArray
    checkpoints: tuple[ValidationCheckpoint, ...]
    maximum_gh_error: float
    maximum_st_error: float
    maximum_ratio_error: float
    maximum_velocity_error: float
    maximum_acceleration_error: float
    rms_error: float
    gh_endpoint_valid: bool
    st_endpoint_valid: bool
    elevation_sum_valid: bool
    no_discontinuities: bool
    no_negative_ratios: bool
    no_velocity_spikes: bool
    continuous_first_derivative: bool
    continuous_second_derivative: bool
    specification_consistent: bool
    passed: bool
    warnings: tuple[str, ...]


REQUESTED_CHECKPOINTS = (
    (30.0, 24.0, 6.0),
    (90.0, 72.0, 18.0),
    (114.0, 84.0, 30.0),
    (180.0, 120.0, 60.0),
)


def validate_biomechanics(
    transmission: SmoothTransmission,
    regions: tuple[RatioRegion, ...],
    maximum_elevation_deg: float = 180.0,
    sample_count: int = 3601,
) -> BiomechanicsValidation:
    """Compare the smooth synthesized motion with the independent step target."""

    elevation = np.linspace(0.0, maximum_elevation_deg, sample_count)
    phase = elevation / maximum_elevation_deg * 2.0 * np.pi
    mechanical = np.asarray(transmission.ratio(phase), dtype=np.float64)
    output = np.asarray(transmission.output_angle(phase), dtype=np.float64)
    actual_st = output / (2.0 * np.pi) * transmission.final_st_deg
    actual_gh = elevation - actual_st
    actual_ratio = 3.0 / mechanical - 1.0

    target_ratio = np.empty_like(elevation)
    target_gh = np.zeros_like(elevation)
    target_st = np.zeros_like(elevation)
    target_gh_velocity = np.empty_like(elevation)
    target_st_velocity = np.empty_like(elevation)
    for index, value in enumerate(elevation):
        gh, st = transmission.shoulder_model.contributions_at(float(value))
        target_gh[index], target_st[index] = gh, st
        region = next(
            (
                item
                for item in regions
                if item.start_deg <= value < item.end_deg
            ),
            regions[-1],
        )
        ratio = region.gh_to_st_ratio
        target_ratio[index] = ratio
        target_gh_velocity[index] = ratio / (ratio + 1.0)
        target_st_velocity[index] = 1.0 / (ratio + 1.0)

    gh_error = actual_gh - target_gh
    st_error = actual_st - target_st
    ratio_error = actual_ratio - target_ratio
    actual_st_velocity = mechanical / 3.0
    actual_gh_velocity = 1.0 - actual_st_velocity
    velocity_error = np.maximum(
        np.abs(actual_gh_velocity - target_gh_velocity),
        np.abs(actual_st_velocity - target_st_velocity),
    )

    radians_per_degree = 2.0 * np.pi / maximum_elevation_deg
    wrapped = np.mod(phase, 2.0 * np.pi)
    dm_dphi = transmission.ratio_derivative(wrapped)
    actual_st_acceleration = dm_dphi * radians_per_degree / 3.0
    actual_gh_acceleration = -actual_st_acceleration
    maximum_acceleration_error = float(
        np.max(np.maximum(np.abs(actual_gh_acceleration), np.abs(actual_st_acceleration)))
    )

    checkpoints: list[ValidationCheckpoint] = []
    for checkpoint_elevation, requested_gh, requested_st in REQUESTED_CHECKPOINTS:
        schedule_gh, schedule_st = transmission.shoulder_model.contributions_at(
            checkpoint_elevation
        )
        sample = transmission.evaluate(
            checkpoint_elevation / maximum_elevation_deg * 2.0 * np.pi
        )
        consistent = (
            abs(schedule_gh - requested_gh) <= 1e-9
            and abs(schedule_st - requested_st) <= 1e-9
        )
        checkpoints.append(
            ValidationCheckpoint(
                elevation_deg=checkpoint_elevation,
                requested_gh_deg=requested_gh,
                requested_st_deg=requested_st,
                schedule_gh_deg=schedule_gh,
                schedule_st_deg=schedule_st,
                actual_gh_deg=sample.gh_deg,
                actual_st_deg=sample.st_deg,
                gh_difference_deg=sample.gh_deg - requested_gh,
                st_difference_deg=sample.st_deg - requested_st,
                gh_percent_error=100.0 * (sample.gh_deg - requested_gh) / requested_gh,
                st_percent_error=100.0 * (sample.st_deg - requested_st) / requested_st,
                specification_consistent=consistent,
            )
        )

    endpoint_tolerance = 1e-8
    gh_endpoint = bool(abs(actual_gh[-1] - 120.0) <= endpoint_tolerance)
    st_endpoint = bool(abs(actual_st[-1] - 60.0) <= endpoint_tolerance)
    elevation_sum = bool(
        np.max(np.abs(actual_gh + actual_st - elevation)) <= endpoint_tolerance
    )
    negative_free = bool(np.min(mechanical) > 0.0 and np.min(actual_ratio) > 0.0)

    # Evaluate derivative jumps immediately to either side of every spline knot.
    epsilon = 1e-7
    knot_phases = np.radians(
        np.array([region.end_deg for region in regions[:-1]])
        / maximum_elevation_deg
        * 360.0
    )
    velocity_jumps = []
    acceleration_jumps = []
    for knot in knot_phases:
        left = float(transmission.ratio(knot - epsilon))
        right = float(transmission.ratio(knot + epsilon))
        velocity_jumps.append(abs(right - left) / 3.0)
        left_acc = float(transmission.ratio_derivative(knot - epsilon))
        right_acc = float(transmission.ratio_derivative(knot + epsilon))
        acceleration_jumps.append(abs(right_acc - left_acc) * radians_per_degree / 3.0)
    continuous_first = bool(max(velocity_jumps, default=0.0) < 1e-5)
    continuous_second = bool(max(acceleration_jumps, default=0.0) < 1e-5)
    no_discontinuities = continuous_first and continuous_second
    no_spikes = bool(
        np.all(np.isfinite(actual_gh_velocity))
        and np.all(np.isfinite(actual_st_velocity))
        and np.max(np.abs(actual_gh_velocity)) < 1.5
        and np.max(np.abs(actual_st_velocity)) < 1.5
    )

    max_gh = float(np.max(np.abs(gh_error)))
    max_st = float(np.max(np.abs(st_error)))
    max_ratio = float(np.max(np.abs(ratio_error)))
    max_velocity = float(np.max(velocity_error))
    rms = float(np.sqrt(np.mean(np.r_[gh_error, st_error] ** 2)))
    specification_consistent = all(item.specification_consistent for item in checkpoints)
    warnings: list[str] = []
    if not specification_consistent:
        warnings.append(
            "Requested 90°/114° checkpoints conflict with the incremental ratio schedule."
        )
    if max_ratio > 0.25:
        warnings.append(
            "C² spline smoothing does not reproduce the step target ratio within tolerance."
        )

    passed = all(
        (
            gh_endpoint,
            st_endpoint,
            elevation_sum,
            no_discontinuities,
            negative_free,
            no_spikes,
            continuous_first,
            continuous_second,
            specification_consistent,
            max_gh <= 0.5,
            max_st <= 0.5,
            max_ratio <= 0.25,
            max_velocity <= 0.10,
            maximum_acceleration_error <= 0.05,
            rms <= 0.25,
        )
    )
    return BiomechanicsValidation(
        elevation_deg=elevation,
        mechanical_ratio=mechanical,
        target_ratio=target_ratio,
        actual_ratio=actual_ratio,
        target_gh_deg=target_gh,
        actual_gh_deg=actual_gh,
        target_st_deg=target_st,
        actual_st_deg=actual_st,
        gh_error_deg=gh_error,
        st_error_deg=st_error,
        ratio_error=ratio_error,
        checkpoints=tuple(checkpoints),
        maximum_gh_error=max_gh,
        maximum_st_error=max_st,
        maximum_ratio_error=max_ratio,
        maximum_velocity_error=max_velocity,
        maximum_acceleration_error=maximum_acceleration_error,
        rms_error=rms,
        gh_endpoint_valid=gh_endpoint,
        st_endpoint_valid=st_endpoint,
        elevation_sum_valid=elevation_sum,
        no_discontinuities=no_discontinuities,
        no_negative_ratios=negative_free,
        no_velocity_spikes=no_spikes,
        continuous_first_derivative=continuous_first,
        continuous_second_derivative=continuous_second,
        specification_consistent=specification_consistent,
        passed=passed,
        warnings=tuple(warnings),
    )
