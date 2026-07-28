"""Automatic tooth design and assembled-pair validation.

This module deliberately exposes no general-purpose CAD knobs.  It selects one
shared rack standard from center distance, applies it to both conjugate pitch
curves, and verifies the resulting cached polygons through a full revolution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely import affinity
from shapely.geometry import Polygon

from noncircular import PitchCurveData
from tooth_geometry import GeneratedGear, NonCircularToothGenerator, ToothParameters


@dataclass(frozen=True)
class AutomaticToothDesign:
    module: float
    pressure_angle_deg: float
    addendum: float
    dedendum: float
    root_fillet_radius: float
    tooth_count: int
    envelope_samples: int


@dataclass(frozen=True)
class PairToothValidation:
    synchronized: bool
    interference_free: bool
    constant_center_distance: bool
    conjugacy_preserved: bool
    maximum_overlap_area: float
    minimum_clearance: float
    checked_positions: int
    warnings: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return (
            self.synchronized
            and self.interference_free
            and self.constant_center_distance
            and self.conjugacy_preserved
        )


@dataclass(frozen=True)
class GeneratedGearPair:
    input_gear: GeneratedGear
    output_gear: GeneratedGear
    design: AutomaticToothDesign
    validation: PairToothValidation


def _polyline_length(points: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def select_automatic_design(data: PitchCurveData) -> AutomaticToothDesign:
    """Choose a robust FDM-prototype rack standard from overall gear scale.

    A nominal module of 3% of center distance keeps the shoulder profile near
    33 teeth over the supported scale.  It is clamped to 1.5 model units so
    tooth roots are not excessively thin on ordinary prototype printers.
    Tooth count is shared and the effective module is then snapped to the mean
    closed pitch length, making the circular pitch close exactly at the seam.
    """

    input_length = _polyline_length(data.input_points)
    output_length = _polyline_length(data.output_points)
    mean_length = 0.5 * (input_length + output_length)
    nominal_module = max(1.5, 0.03 * data.center_distance)
    count = max(18, int(np.rint(mean_length / (np.pi * nominal_module))))
    module = mean_length / (np.pi * count)
    return AutomaticToothDesign(
        module=module,
        pressure_angle_deg=20.0,
        addendum=0.90 * module,
        dedendum=1.25 * module,
        root_fillet_radius=0.30 * module,
        tooth_count=count,
        envelope_samples=2048,
    )


def generate_gear_pair(data: PitchCurveData) -> GeneratedGearPair:
    """Rack-generate both gears with tooth-to-gap phase at the initial contact."""

    design = select_automatic_design(data)
    generator = NonCircularToothGenerator()

    common = dict(
        module=design.module,
        pressure_angle_deg=design.pressure_angle_deg,
        addendum=design.addendum,
        dedendum=design.dedendum,
        root_fillet_radius=design.root_fillet_radius,
        tooth_count=design.tooth_count,
        envelope_samples=design.envelope_samples,
    )
    input_gear = generator.generate(
        data.input_points, ToothParameters(**common, tooth_phase=0.5)
    )
    output_gear = generator.generate(
        data.output_points, ToothParameters(**common, tooth_phase=0.0)
    )
    validation = validate_generated_pair(data, input_gear, output_gear)
    return GeneratedGearPair(input_gear, output_gear, design, validation)


def validate_generated_pair(
    data: PitchCurveData,
    input_gear: GeneratedGear,
    output_gear: GeneratedGear,
    position_count: int = 361,
) -> PairToothValidation:
    """Check rigidly transformed finished boundaries over one input revolution.

    Full-resolution polygons are retained for display/export.  A topology-
    preserving 0.01%-of-center-distance simplification makes this startup-only
    collision sweep fast without altering the cached manufacturing boundary.
    """

    synchronized = (
        input_gear.tooth_count == output_gear.tooth_count
        and abs(input_gear.circular_pitch - output_gear.circular_pitch)
        <= max(1e-6, input_gear.circular_pitch * 1e-5)
    )
    center_error = np.max(
        np.abs(
            data.input_radii + data.output_radii - data.center_distance
        )
    )
    constant_center = bool(center_error <= data.center_distance * 1e-10)

    tolerance = data.center_distance * 1e-4
    first = Polygon(input_gear.polygon).simplify(tolerance, preserve_topology=True)
    second_local = Polygon(output_gear.polygon).simplify(
        tolerance, preserve_topology=True
    )
    sample_indices = np.linspace(
        0, len(data.input_rad) - 1, position_count, dtype=int
    )
    maximum_overlap = 0.0
    minimum_clearance = np.inf
    for index in sample_indices:
        input_angle = float(data.input_rad[index])
        output_angle = -float(data.output_rad[index])
        placed_input = affinity.rotate(
            first, np.degrees(input_angle), origin=(0.0, 0.0)
        )
        placed_output = affinity.translate(
            affinity.rotate(
                second_local, np.degrees(output_angle), origin=(0.0, 0.0)
            ),
            xoff=data.center_distance,
        )
        overlap = placed_input.intersection(placed_output).area
        maximum_overlap = max(maximum_overlap, float(overlap))
        minimum_clearance = min(
            minimum_clearance, float(placed_input.distance(placed_output))
        )

    # Touching flanks have zero distance.  Area, rather than distance, detects
    # solid interference; the tolerance covers polygonal envelope sampling.
    area_tolerance = (data.center_distance * 2e-4) ** 2
    interference_free = maximum_overlap <= area_tolerance
    conjugacy = synchronized and input_gear.validation.valid and output_gear.validation.valid
    warnings: list[str] = []
    if not synchronized:
        warnings.append("Mating tooth pitches do not close in synchronization.")
    if not interference_free:
        warnings.append(
            f"Solid interference detected (area {maximum_overlap:.6g})."
        )
    if not constant_center:
        warnings.append("Pitch-radius sum violates the fixed center distance.")
    if not conjugacy:
        warnings.append("One generated rack envelope failed tooth validation.")
    return PairToothValidation(
        synchronized=synchronized,
        interference_free=interference_free,
        constant_center_distance=constant_center,
        conjugacy_preserved=conjugacy,
        maximum_overlap_area=maximum_overlap,
        minimum_clearance=float(minimum_clearance),
        checked_positions=position_count,
        warnings=tuple(warnings),
    )
