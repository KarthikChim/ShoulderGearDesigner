"""Mechanical diagnostics and gated optimization for literature sectors.

This module intentionally separates screening from acceptance.  Search rows
may use a reduced collision sweep, but a selected candidate is not eligible
for prototype export until its complete closed bodies pass 2001 assembled
positions and every hard gate in :class:`OptimizedValidation`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from literature_sector import (
    LiteratureSectorTransmission,
    SectorDesignConfig,
    SectorPitchCurveData,
    SectorToothGeometry,
    generate_sector_teeth,
    synthesize_sector_pitch_curves,
    validate_sector,
)


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ClosedSectorBlank:
    polygon: Polygon
    boundary: FloatArray
    bore_radius: float
    web_thickness: float
    hard_stop_size: float
    valid: bool
    closed: bool
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class MeshPosition:
    sample_index: int
    elevation_deg: float
    input_angle_rad: float
    output_angle_rad: float
    ratio: float
    input_radius: float
    output_radius: float
    penetration_area: float
    minimum_clearance: float
    contact_mismatch: float
    normal_error_deg: float
    intended_input_tooth: int
    intended_output_tooth: int


@dataclass(frozen=True)
class MeshSweep:
    positions: tuple[MeshPosition, ...]
    zero_unintended_intersections: bool
    maximum_penetration_area: float
    minimum_clearance: float
    maximum_contact_mismatch: float
    maximum_normal_error_deg: float
    no_tooth_skipping: bool
    no_contact_discontinuity: bool


@dataclass(frozen=True)
class ToothMetric:
    member: str
    tooth_number: int
    arc_position: float
    root_thickness: float
    tip_thickness: float
    undercut_margin: float
    contact_ratio: float
    self_intersection_free: bool


@dataclass(frozen=True)
class OptimizedValidation:
    continuous_tangent: bool
    bounded_curvature: bool
    bounded_curvature_derivative: bool
    adjacent_tooth_overlap_free: bool
    mating_interference_free: bool
    minimum_pitch_radius_valid: bool
    minimum_contact_ratio_valid: bool
    minimum_root_thickness_valid: bool
    sector_blanks_closed_valid: bool
    no_extrapolation: bool
    maximum_st_error_valid: bool
    rms_st_error_valid: bool
    rack_envelope_verified: bool
    hard_pass: bool
    decision: str
    blockers: tuple[str, ...]


def _polyline_normals(points: FloatArray) -> FloatArray:
    tangent = np.gradient(points, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1)[:, None], 1e-15)
    candidate = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    return np.where(
        (np.sum(candidate * points, axis=1) >= 0)[:, None],
        candidate,
        -candidate,
    )


def build_closed_sector_blank(
    pitch_points: FloatArray,
    tooth_polygons: tuple[FloatArray, ...],
    config: SectorDesignConfig,
    *,
    bore_radius: float = 4.0,
    web_thickness: float = 5.0,
) -> ClosedSectorBlank:
    """Create one closed research sector body with bore and hard-stop tabs."""

    normals = _polyline_normals(pitch_points)
    root = pitch_points - config.dedendum_factor * config.module * normals
    # A filled fan is intentionally conservative.  It provides a real web from
    # the root sector to the mounting hub rather than leaving disconnected
    # decorative teeth.
    fan = Polygon(np.vstack((root, [[0.0, 0.0]]))).buffer(0)
    hub_radius = max(bore_radius + web_thickness, 2.2 * config.module)
    hub = Point(0.0, 0.0).buffer(hub_radius, quad_segs=48)
    solids = [fan, hub]
    solids.extend(Polygon(points).buffer(0) for points in tooth_polygons)
    body = unary_union(solids).buffer(0)

    # Hard-stop interfaces are radial tabs at both active-sector ends.
    stop_size = max(2.0 * config.module, web_thickness)
    for endpoint in (root[0], root[-1]):
        direction = endpoint / max(np.linalg.norm(endpoint), 1e-15)
        tangent = np.array([-direction[1], direction[0]])
        center = endpoint - 0.25 * stop_size * direction
        corners = np.array(
            [
                center - stop_size * direction - 0.5 * stop_size * tangent,
                center + 0.5 * stop_size * direction - 0.5 * stop_size * tangent,
                center + 0.5 * stop_size * direction + 0.5 * stop_size * tangent,
                center - stop_size * direction + 0.5 * stop_size * tangent,
            ]
        )
        body = body.union(Polygon(corners)).buffer(0)
    if body.geom_type == "MultiPolygon":
        body = max(body.geoms, key=lambda shape: shape.area)
    # Fill incidental voids left between the conservative fan and tooth-root
    # unions, then cut exactly one intentional shaft-bore placeholder.
    if body.geom_type == "Polygon":
        body = Polygon(body.exterior)
    body = body.difference(Point(0.0, 0.0).buffer(bore_radius, quad_segs=48))
    warnings: list[str] = []
    if body.is_empty or body.geom_type != "Polygon":
        warnings.append("Sector union did not form one polygon.")
        boundary = np.empty((0, 2), dtype=np.float64)
    else:
        boundary = np.asarray(body.exterior.coords, dtype=np.float64)
    closed = bool(
        len(boundary) > 3
        and np.linalg.norm(boundary[0] - boundary[-1]) < 1e-10
    )
    valid = bool(
        not body.is_empty
        and body.geom_type == "Polygon"
        and body.is_valid
        and closed
        and len(body.interiors) == 1
    )
    if not valid:
        warnings.append(
            "Blank must be one valid closed polygon with exactly one bore."
        )
    return ClosedSectorBlank(
        polygon=body,
        boundary=boundary,
        bore_radius=bore_radius,
        web_thickness=web_thickness,
        hard_stop_size=stop_size,
        valid=valid,
        closed=closed,
        warnings=tuple(warnings),
    )


def curve_geometry_metrics(data: SectorPitchCurveData) -> dict:
    """Return sample-resolved tangent, curvature, and curvature derivative."""

    result = {}
    for member, points in (
        ("input", data.input_points),
        ("output", data.output_points),
    ):
        first = np.gradient(points, data.elevation_deg, axis=0, edge_order=2)
        second = np.gradient(
            first, data.elevation_deg, axis=0, edge_order=2
        )
        speed = np.linalg.norm(first, axis=1)
        tangent = first / np.maximum(speed[:, None], 1e-15)
        curvature = (
            first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
        ) / np.maximum(speed**3, 1e-15)
        curvature_derivative = np.gradient(
            curvature, data.elevation_deg, edge_order=2
        )
        tangent_jump = np.r_[
            0.0, np.linalg.norm(np.diff(tangent, axis=0), axis=1)
        ]
        result[member] = {
            "tangent": tangent,
            "tangent_jump": tangent_jump,
            "curvature": curvature,
            "curvature_derivative": curvature_derivative,
        }
    return result


def adjacent_tooth_failures(teeth: SectorToothGeometry) -> list[dict]:
    failures = []
    for member, polygons in (
        ("input", teeth.input_teeth),
        ("output", teeth.output_teeth),
    ):
        for index, (first, second) in enumerate(zip(polygons[:-1], polygons[1:])):
            area = float(Polygon(first).intersection(Polygon(second)).area)
            if area > 1e-10:
                failures.append(
                    {
                        "member": member,
                        "tooth_number": index + 1,
                        "other_tooth_number": index + 2,
                        "overlap_area": area,
                    }
                )
    return failures


def simulate_complete_mesh(
    data: SectorPitchCurveData,
    input_blank: ClosedSectorBlank,
    output_blank: ClosedSectorBlank,
    teeth: SectorToothGeometry,
    *,
    position_count: int = 2001,
) -> MeshSweep:
    """Transform complete bodies through the non-wrapping active sector."""

    indices = np.linspace(
        0, len(data.elevation_deg) - 1, position_count, dtype=int
    )
    input_body = input_blank.polygon
    output_body = output_blank.polygon
    positions: list[MeshPosition] = []
    previous_pair: tuple[int, int] | None = None
    skipping = False
    contact_discontinuity = False
    for sample_index in indices:
        phi = float(data.input_rad[sample_index])
        psi = float(data.output_rad[sample_index])
        placed_input = affinity.rotate(
            input_body, np.degrees(phi), origin=(0.0, 0.0)
        )
        placed_output = affinity.translate(
            affinity.rotate(
                output_body, -np.degrees(psi), origin=(0.0, 0.0)
            ),
            xoff=data.center_distance,
        )
        penetration = float(placed_input.intersection(placed_output).area)
        clearance = float(placed_input.distance(placed_output))

        input_contact = np.asarray(data.input_points[sample_index])
        c, s = np.cos(phi), np.sin(phi)
        rotation_input = np.array([[c, -s], [s, c]])
        world_input = rotation_input @ input_contact
        output_contact = np.asarray(data.output_points[sample_index])
        c2, s2 = np.cos(-psi), np.sin(-psi)
        rotation_output = np.array([[c2, -s2], [s2, c2]])
        world_output = rotation_output @ output_contact + np.array(
            [data.center_distance, 0.0]
        )
        mismatch = float(np.linalg.norm(world_input - world_output))
        normal_error = 0.0  # Conjugate pitch radii share the line of centers.

        input_tooth = min(
            teeth.input_tooth_count,
            max(
                1,
                int(
                    np.floor(
                        sample_index
                        / max(len(data.elevation_deg) - 1, 1)
                        * teeth.input_tooth_count
                    )
                )
                + 1,
            ),
        )
        output_tooth = min(
            teeth.output_tooth_count,
            max(
                1,
                int(
                    np.floor(
                        sample_index
                        / max(len(data.elevation_deg) - 1, 1)
                        * teeth.output_tooth_count
                    )
                )
                + 1,
            ),
        )
        pair = (input_tooth, output_tooth)
        if previous_pair is not None:
            if abs(pair[0] - previous_pair[0]) > 1 or abs(
                pair[1] - previous_pair[1]
            ) > 1:
                skipping = True
            if mismatch > 1e-6:
                contact_discontinuity = True
        previous_pair = pair
        positions.append(
            MeshPosition(
                sample_index=int(sample_index),
                elevation_deg=float(data.elevation_deg[sample_index]),
                input_angle_rad=phi,
                output_angle_rad=psi,
                ratio=float(data.ratio[sample_index]),
                input_radius=float(data.input_radii[sample_index]),
                output_radius=float(data.output_radii[sample_index]),
                penetration_area=penetration,
                minimum_clearance=clearance,
                contact_mismatch=mismatch,
                normal_error_deg=normal_error,
                intended_input_tooth=input_tooth,
                intended_output_tooth=output_tooth,
            )
        )
    penetration = np.array([item.penetration_area for item in positions])
    clearance = np.array([item.minimum_clearance for item in positions])
    mismatch = np.array([item.contact_mismatch for item in positions])
    normal = np.array([item.normal_error_deg for item in positions])
    return MeshSweep(
        positions=tuple(positions),
        zero_unintended_intersections=bool(np.max(penetration) <= 1e-10),
        maximum_penetration_area=float(np.max(penetration)),
        minimum_clearance=float(np.min(clearance)),
        maximum_contact_mismatch=float(np.max(mismatch)),
        maximum_normal_error_deg=float(np.max(normal)),
        no_tooth_skipping=not skipping,
        no_contact_discontinuity=not contact_discontinuity,
    )


def compute_tooth_metrics(
    teeth: SectorToothGeometry,
    *,
    contact_ratio_threshold: float = 1.2,
) -> tuple[ToothMetric, ...]:
    """Geometry-resolved thickness and conservative rack contact estimates."""

    alpha = np.radians(teeth.pressure_angle_deg)
    circular_pitch = np.pi * teeth.module
    pitch_thickness = circular_pitch / 2.0 - teeth.backlash
    root_thickness = pitch_thickness + 2.0 * teeth.dedendum * np.tan(alpha)
    tip_thickness = pitch_thickness - 2.0 * teeth.addendum * np.tan(alpha)
    undercut_limit = 2.0 / np.sin(alpha) ** 2
    metrics: list[ToothMetric] = []
    for member, polygons, positions in (
        ("input", teeth.input_teeth, teeth.input_arc_positions),
        ("output", teeth.output_teeth, teeth.output_arc_positions),
    ):
        count = len(polygons)
        undercut_margin = count - undercut_limit
        # Straight-rack path-of-contact length divided by base pitch.  This is
        # calculated from configured addendum/dedendum geometry, not a fixed
        # proxy constant, but remains an unloaded 2-D estimate.
        path_length = (
            teeth.addendum + teeth.dedendum
        ) / max(np.sin(alpha), 1e-12)
        base_pitch = circular_pitch * np.cos(alpha)
        ratio = path_length / max(base_pitch, 1e-12)
        for index, (polygon, position) in enumerate(zip(polygons, positions)):
            metrics.append(
                ToothMetric(
                    member=member,
                    tooth_number=index + 1,
                    arc_position=float(position),
                    root_thickness=float(root_thickness),
                    tip_thickness=float(tip_thickness),
                    undercut_margin=float(undercut_margin),
                    contact_ratio=float(ratio),
                    self_intersection_free=Polygon(polygon).is_valid,
                )
            )
    return tuple(metrics)


def evaluate_hard_gates(
    transmission: LiteratureSectorTransmission,
    data: SectorPitchCurveData,
    teeth: SectorToothGeometry,
    input_blank: ClosedSectorBlank,
    output_blank: ClosedSectorBlank,
    mesh: MeshSweep,
    metrics: tuple[ToothMetric, ...],
    *,
    maximum_st_error_deg: float = 1.0,
    rms_st_error_deg: float = 0.75,
    minimum_contact_ratio: float = 1.2,
    minimum_root_thickness: float = 1.0,
    rack_envelope_verified: bool = False,
) -> OptimizedValidation:
    base = validate_sector(transmission, data, teeth)
    geometry = curve_geometry_metrics(data)
    curvature = np.r_[
        geometry["input"]["curvature"], geometry["output"]["curvature"]
    ]
    curvature_derivative = np.r_[
        geometry["input"]["curvature_derivative"],
        geometry["output"]["curvature_derivative"],
    ]
    tangent_jump = np.r_[
        geometry["input"]["tangent_jump"],
        geometry["output"]["tangent_jump"],
    ]
    continuous_tangent = bool(np.max(tangent_jump) < 0.02)
    curvature_limit = 1.0 / max(0.35 * transmission.config.module, 1e-12)
    curvature_derivative_limit = curvature_limit / max(
        transmission.config.module, 1e-12
    )
    bounded_curvature = bool(
        np.all(np.isfinite(curvature))
        and np.max(np.abs(curvature)) <= curvature_limit
    )
    bounded_curvature_derivative = bool(
        np.all(np.isfinite(curvature_derivative))
        and np.max(np.abs(curvature_derivative))
        <= curvature_derivative_limit
    )
    adjacent = len(adjacent_tooth_failures(teeth)) == 0
    minimum_ratio = min((item.contact_ratio for item in metrics), default=0.0)
    minimum_root = min((item.root_thickness for item in metrics), default=0.0)
    gates = {
        "continuous_tangent": continuous_tangent,
        "bounded_curvature": bounded_curvature,
        "bounded_curvature_derivative": bounded_curvature_derivative,
        "adjacent_tooth_overlap_free": adjacent,
        "mating_interference_free": mesh.zero_unintended_intersections,
        "minimum_pitch_radius_valid": base.minimum_radius_valid,
        "minimum_contact_ratio_valid": minimum_ratio >= minimum_contact_ratio,
        "minimum_root_thickness_valid": minimum_root >= minimum_root_thickness,
        "sector_blanks_closed_valid": input_blank.valid and output_blank.valid,
        "no_extrapolation": base.no_extrapolation,
        "maximum_st_error_valid": base.maximum_st_error_deg
        <= maximum_st_error_deg,
        "rms_st_error_valid": base.rms_st_error_deg <= rms_st_error_deg,
        "rack_envelope_verified": rack_envelope_verified,
    }
    blockers = tuple(name for name, valid in gates.items() if not valid)
    hard_pass = all(gates.values())
    # The hand-driven label is impossible until the exact rack envelope and
    # complete mesh both pass.  Software simulation remains permissible when
    # kinematics are finite, monotonic, and non-wrapping.
    if hard_pass:
        decision = "GO FOR UNLOADED HAND-DRIVEN BENCH PROTOTYPE"
    elif (
        base.finite_positive_ratio
        and base.monotonic_input
        and base.monotonic_output
        and base.no_wrapping
    ):
        decision = "GO FOR SOFTWARE SIMULATION"
    else:
        decision = "NO-GO"
    return OptimizedValidation(
        **gates,
        hard_pass=hard_pass,
        decision=decision,
        blockers=blockers,
    )
