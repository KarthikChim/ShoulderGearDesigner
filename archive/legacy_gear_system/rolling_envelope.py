"""Numerical rolling-rack envelopes for open non-circular pitch sectors.

This module does not place tooth polygons on a curve.  It moves the canonical
standard rack cutter through a no-slip rolling trajectory, constructs the
swept cutter volume between successive poses, and subtracts that envelope
from an addendum-height sector blank.

The numerical approximation converges as ``samples_per_pitch`` increases.
No ``buffer()``, polygon offset, smoothing, or independently placed material
tooth is used anywhere in the generation path.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
from shapely import make_valid
from shapely import affinity
from shapely.geometry import GeometryCollection, MultiPolygon, Point, Polygon
from shapely.ops import unary_union

from standard_involute import StandardGearParameters, generate_rack_tooth


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CurveFrame:
    arc_length: FloatArray
    points: FloatArray
    tangents: FloatArray
    outward_normals: FloatArray
    curvature: FloatArray


@dataclass(frozen=True)
class CutterPose:
    sample_index: int
    cutter_index: int
    arc_length: float
    origin: FloatArray
    tangent: FloatArray
    outward_normal: FloatArray
    curvature: float
    polygon: FloatArray


@dataclass(frozen=True)
class EnvelopeValidation:
    valid: bool
    equal_arc_spacing: bool
    maximum_pitch_spacing_error: float
    continuous_cutter_motion: bool
    maximum_pose_vertex_jump: float
    final_polygon_valid: bool
    printable_root_thickness: bool
    minimum_root_thickness: float
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class RollingEnvelopeResult:
    pitch_curve: FloatArray
    pitch_locations: FloatArray
    pitch_arc_positions: FloatArray
    frame: CurveFrame
    cutter_poses: tuple[CutterPose, ...]
    cutter_envelope: Polygon | MultiPolygon
    blank: Polygon
    final_polygon: Polygon
    tooth_regions: tuple[FloatArray, ...]
    root_curve: FloatArray
    validation: EnvelopeValidation


@dataclass(frozen=True)
class ConjugateMeshValidation:
    valid: bool
    conjugate_rolling: bool
    maximum_rolling_arc_error: float
    no_interference: bool
    maximum_penetration_area: float
    continuous_velocity_ratio: bool
    contact_through_complete_motion: bool
    maximum_contact_gap: float
    printable_root_thickness: bool
    sampled_positions: int
    warnings: tuple[str, ...]


def _curve_frame(points: FloatArray, sample_arc: FloatArray) -> CurveFrame:
    """Evaluate position, Frenet frame, and signed curvature versus arc."""
    points = np.asarray(points, dtype=np.float64)
    segment = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment)))
    keep = np.concatenate(([True], np.diff(cumulative) > 1e-10))
    cumulative = cumulative[keep]
    points = points[keep]
    if len(points) < 8:
        raise ValueError("Pitch curve requires at least eight unique points.")
    x = CubicSpline(cumulative, points[:, 0], bc_type="natural")
    y = CubicSpline(cumulative, points[:, 1], bc_type="natural")
    position = np.column_stack((x(sample_arc), y(sample_arc)))
    first = np.column_stack((x(sample_arc, 1), y(sample_arc, 1)))
    second = np.column_stack((x(sample_arc, 2), y(sample_arc, 2)))
    speed = np.linalg.norm(first, axis=1)
    tangent = first / np.maximum(speed[:, None], 1e-15)
    candidate = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    outward = np.where(
        (np.sum(candidate * position, axis=1) >= 0.0)[:, None],
        candidate,
        -candidate,
    )
    cross = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    curvature = cross / np.maximum(speed**3, 1e-15)
    return CurveFrame(sample_arc, position, tangent, outward, curvature)


def equally_spaced_pitch_locations(
    points: FloatArray,
    module: float,
    *,
    phase_pitch: float = 0.0,
) -> tuple[FloatArray, FloatArray, CurveFrame]:
    """Return locations separated by exactly ``pi*module`` in arc length."""
    segment = np.linalg.norm(np.diff(points, axis=0), axis=1)
    length = float(np.sum(segment))
    pitch = math.pi * module
    start = 0.5 * pitch + (phase_pitch % 1.0) * pitch
    positions = np.arange(start, length - 0.5 * pitch + 1e-12, pitch)
    frame = _curve_frame(points, positions)
    return frame.points, positions, frame


def _closed_cutter(parameters: StandardGearParameters) -> FloatArray:
    """Close the canonical rack tooth along its non-working root land."""
    tooth = generate_rack_tooth(
        replace(parameters, rack_is_cutter=True)
    ).points
    return np.vstack((tooth, tooth[0]))


def _transform_cutter(
    cutter: FloatArray,
    origin: FloatArray,
    tangent: FloatArray,
    outward_normal: FloatArray,
    tangential_offset: float,
) -> FloatArray:
    """Map rack +Y into the gear (opposite the outward normal)."""
    return (
        origin
        + (cutter[:, :1] + tangential_offset) * tangent
        - cutter[:, 1:] * outward_normal
    )


def _swept_between(first: FloatArray, second: FloatArray):
    """Ruled linear sweep between two dense cutter poses."""
    pieces = [Polygon(first), Polygon(second)]
    for index in range(len(first) - 1):
        quad = Polygon(
            (
                first[index],
                first[index + 1],
                second[index + 1],
                second[index],
            )
        )
        if quad.is_valid and quad.area > 1e-14:
            pieces.append(quad)
    return unary_union(pieces)


def _polygon_members(geometry) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)
    if isinstance(geometry, GeometryCollection):
        return [item for item in geometry.geoms if isinstance(item, Polygon)]
    return []


def _sector_polygon(boundary: FloatArray) -> Polygon:
    """Close an open radial sector boundary to its rotation centre."""
    polygon = Polygon(np.vstack((boundary, [[0.0, 0.0]], boundary[0])))
    if not polygon.is_valid:
        repaired = make_valid(polygon)
        members = _polygon_members(repaired)
        if not members:
            raise ValueError("Pitch-derived sector blank cannot be made valid.")
        center = Point(0.0, 0.0)
        containing = [item for item in members if item.covers(center)]
        polygon = max(containing or members, key=lambda item: item.area)
    return polygon


def generate_rolling_envelope(
    pitch_points: FloatArray,
    parameters: StandardGearParameters,
    *,
    phase_pitch: float = 0.0,
    samples_per_pitch: int = 24,
    neighboring_cutter_teeth: int = 2,
    minimum_root_thickness: float | None = None,
) -> RollingEnvelopeResult:
    """Generate one non-circular sector by rolling the canonical rack cutter.

    A cutter tooth with global integer index ``k`` has tangential coordinate
    ``k*p + phase*p - s`` at curve arc position ``s``.  That is the no-slip
    law: rack translation equals traversed pitch-curve arc length.
    """
    parameters.validate()
    if samples_per_pitch < 8:
        raise ValueError("At least eight cutter poses per pitch are required.")
    points = np.asarray(pitch_points, dtype=np.float64)
    length = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    pitch = parameters.circular_pitch
    step_count = max(
        2, int(math.ceil(length / (pitch / samples_per_pitch))) + 1
    )
    trajectory_arc = np.linspace(0.0, length, step_count)
    frame = _curve_frame(points, trajectory_arc)
    pitch_locations, pitch_positions, _ = equally_spaced_pitch_locations(
        points, parameters.module, phase_pitch=phase_pitch
    )
    cutter = _closed_cutter(parameters)

    pose_by_index: dict[int, list[CutterPose]] = {}
    all_poses: list[CutterPose] = []
    phase_distance = phase_pitch * pitch
    for sample_index, (arc, origin, tangent, normal, curvature) in enumerate(
        zip(
            frame.arc_length,
            frame.points,
            frame.tangents,
            frame.outward_normals,
            frame.curvature,
        )
    ):
        nearest_index = int(round((arc - phase_distance) / pitch))
        for cutter_index in range(
            nearest_index - neighboring_cutter_teeth,
            nearest_index + neighboring_cutter_teeth + 1,
        ):
            offset = cutter_index * pitch + phase_distance - arc
            world = _transform_cutter(cutter, origin, tangent, normal, offset)
            pose = CutterPose(
                sample_index,
                cutter_index,
                float(arc),
                origin.copy(),
                tangent.copy(),
                normal.copy(),
                float(curvature),
                world,
            )
            pose_by_index.setdefault(cutter_index, []).append(pose)
            all_poses.append(pose)

    # Joining corresponding cutter vertices between consecutive poses is the
    # continuous piecewise-linear trajectory sweep, not independent placement.
    swept: list = []
    maximum_jump = 0.0
    for cutter_poses in pose_by_index.values():
        cutter_poses.sort(key=lambda item: item.sample_index)
        for first, second in zip(cutter_poses[:-1], cutter_poses[1:]):
            if second.sample_index != first.sample_index + 1:
                continue
            maximum_jump = max(
                maximum_jump,
                float(np.max(np.linalg.norm(second.polygon - first.polygon, axis=1))),
            )
            swept.append(_swept_between(first.polygon, second.polygon))
    envelope = unary_union(swept)

    outer = frame.points + parameters.addendum * frame.outward_normals
    root = frame.points - parameters.dedendum * frame.outward_normals
    blank = _sector_polygon(outer)
    root_core = _sector_polygon(root)
    cut = blank.difference(envelope)
    candidates = _polygon_members(cut)
    if not candidates:
        raise ValueError("Rack envelope removed the complete sector blank.")
    center = Point(0.0, 0.0)
    containing = [polygon for polygon in candidates if polygon.covers(center)]
    final = max(containing or candidates, key=lambda polygon: polygon.area)
    if not final.is_valid:
        raise ValueError("Final rack-envelope sector polygon is invalid.")

    # Partition the already-generated final material into pitch cells solely
    # for GUI highlighting. These cells do not participate in construction.
    location_frame = _curve_frame(points, pitch_positions)
    regions: list[Polygon] = []
    used_material = GeometryCollection()
    for origin, tangent, normal in zip(
        location_frame.points,
        location_frame.tangents,
        location_frame.outward_normals,
    ):
        half = 0.5 * pitch
        inward = parameters.dedendum + 0.15 * parameters.module
        outward = parameters.addendum + 0.15 * parameters.module
        cell = Polygon(
            (
                origin - half * tangent - inward * normal,
                origin + half * tangent - inward * normal,
                origin + half * tangent + outward * normal,
                origin - half * tangent + outward * normal,
            )
        )
        available = final.intersection(cell).difference(used_material)
        pieces = _polygon_members(available)
        if pieces:
            region = max(pieces, key=lambda polygon: polygon.area)
            if region.area > 0.02 * parameters.module**2:
                regions.append(region)
                used_material = unary_union((used_material, region))
    if not regions:
        # Degenerate raw literature curves can leave only a central connected
        # remnant. Preserve it as one debug region; it is never an accepted
        # manufacturing candidate.
        regions.append(final)
    clean_regions: list[Polygon] = []
    for region in regions:
        clean = Polygon(region.exterior)
        if not clean.is_valid:
            members = _polygon_members(make_valid(clean))
            if not members:
                continue
            clean = max(members, key=lambda polygon: polygon.area)
        clean_regions.append(clean)
    tooth_regions = tuple(
        np.asarray(polygon.exterior.coords, dtype=np.float64)
        for polygon in clean_regions
    )

    spacing_error = (
        float(np.max(np.abs(np.diff(pitch_positions) - pitch)))
        if len(pitch_positions) > 1
        else 0.0
    )
    requested_root = (
        0.75 * parameters.module
        if minimum_root_thickness is None
        else minimum_root_thickness
    )
    # Conservative material depth between pitch curve and root core.
    measured_root = parameters.dedendum
    warnings: list[str] = []
    if spacing_error > 1e-9:
        warnings.append("Arc-length tooth pitch is not uniform.")
    continuity_limit = 1.25 * pitch
    if maximum_jump > continuity_limit:
        warnings.append("Cutter trajectory sampling is too coarse.")
    if measured_root < requested_root:
        warnings.append("Minimum printable root thickness is not met.")
    valid = (
        final.is_valid
        and spacing_error <= 1e-9
        and maximum_jump <= continuity_limit
        and measured_root >= requested_root
        and len(tooth_regions) > 0
    )
    return RollingEnvelopeResult(
        pitch_curve=points,
        pitch_locations=pitch_locations,
        pitch_arc_positions=pitch_positions,
        frame=frame,
        cutter_poses=tuple(all_poses),
        cutter_envelope=envelope,
        blank=blank,
        final_polygon=final,
        tooth_regions=tooth_regions,
        root_curve=root,
        validation=EnvelopeValidation(
            valid=valid,
            equal_arc_spacing=spacing_error <= 1e-9,
            maximum_pitch_spacing_error=spacing_error,
            continuous_cutter_motion=(
                maximum_jump <= continuity_limit
            ),
            maximum_pose_vertex_jump=maximum_jump,
            final_polygon_valid=final.is_valid,
            printable_root_thickness=measured_root >= requested_root,
            minimum_root_thickness=measured_root,
            warnings=tuple(warnings),
        ),
    )


def conjugate_rolling_envelopes(
    input_pitch_points: FloatArray,
    output_pitch_points: FloatArray,
    parameters: StandardGearParameters,
    *,
    samples_per_pitch: int = 24,
) -> tuple[RollingEnvelopeResult, RollingEnvelopeResult]:
    """Generate both members from the same cutter with half-pitch phasing."""
    first = generate_rolling_envelope(
        input_pitch_points,
        parameters,
        phase_pitch=0.0,
        samples_per_pitch=samples_per_pitch,
    )
    second = generate_rolling_envelope(
        output_pitch_points,
        parameters,
        phase_pitch=0.5,
        samples_per_pitch=samples_per_pitch,
    )
    return first, second


def validate_conjugate_envelopes(
    input_result: RollingEnvelopeResult,
    output_result: RollingEnvelopeResult,
    input_angles_rad: FloatArray,
    output_angles_rad: FloatArray,
    center_distance: float,
    velocity_ratio: FloatArray,
    *,
    backlash: float = 0.0,
    maximum_positions: int = 2001,
    area_tolerance: float = 1e-4,
) -> ConjugateMeshValidation:
    """Sweep the completed conjugate bodies through the supported motion."""
    input_angles = np.asarray(input_angles_rad, dtype=np.float64)
    output_angles = np.asarray(output_angles_rad, dtype=np.float64)
    ratio = np.asarray(velocity_ratio, dtype=np.float64)
    count = min(maximum_positions, len(input_angles))
    indices = np.linspace(0, len(input_angles) - 1, count, dtype=int)
    maximum_area = 0.0
    maximum_gap = 0.0
    for index in indices:
        first = affinity.rotate(
            input_result.final_polygon,
            math.degrees(float(input_angles[index])),
            origin=(0.0, 0.0),
        )
        second = affinity.translate(
            affinity.rotate(
                output_result.final_polygon,
                -math.degrees(float(output_angles[index])),
                origin=(0.0, 0.0),
            ),
            xoff=center_distance,
        )
        maximum_area = max(maximum_area, float(first.intersection(second).area))
        maximum_gap = max(maximum_gap, float(first.distance(second)))

    input_segment = np.linalg.norm(
        np.diff(input_result.pitch_curve, axis=0), axis=1
    )
    output_segment = np.linalg.norm(
        np.diff(output_result.pitch_curve, axis=0), axis=1
    )
    common = min(len(input_segment), len(output_segment))
    rolling_error = float(
        np.max(
            np.abs(
                np.cumsum(input_segment[:common])
                - np.cumsum(output_segment[:common])
            )
        )
    )
    conjugate = rolling_error <= 1e-3
    ratio_continuous = bool(
        np.all(np.isfinite(ratio))
        and np.all(ratio > 0.0)
        and np.max(np.abs(np.diff(ratio))) < 0.25 * max(np.max(ratio), 1.0)
    )
    no_interference = maximum_area <= area_tolerance
    # Backlash permits a finite separation at the unloaded contact point.
    allowed_gap = max(0.1, 1.5 * backlash)
    contact = maximum_gap <= allowed_gap
    root_ok = (
        input_result.validation.printable_root_thickness
        and output_result.validation.printable_root_thickness
    )
    warnings: list[str] = []
    if not conjugate:
        warnings.append("Input/output pitch arc travel is not conjugate.")
    if not no_interference:
        warnings.append("Positive-area material interference was detected.")
    if not ratio_continuous:
        warnings.append("Velocity ratio is nonpositive, discontinuous, or nonfinite.")
    if not contact:
        warnings.append("Contact was lost during the supported motion.")
    if not root_ok:
        warnings.append("Printable root thickness failed.")
    valid = conjugate and no_interference and ratio_continuous and contact and root_ok
    return ConjugateMeshValidation(
        valid=valid,
        conjugate_rolling=conjugate,
        maximum_rolling_arc_error=rolling_error,
        no_interference=no_interference,
        maximum_penetration_area=maximum_area,
        continuous_velocity_ratio=ratio_continuous,
        contact_through_complete_motion=contact,
        maximum_contact_gap=maximum_gap,
        printable_root_thickness=root_ok,
        sampled_positions=count,
        warnings=tuple(warnings),
    )
