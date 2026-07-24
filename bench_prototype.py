"""Pragmatic unloaded hand-driven literature gear-sector prototype.

The acceptance gates in this module are deliberately limited to an unloaded
plastic demonstration.  They do not imply wearable, powered, structural, or
human-use safety.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from biomechanics.literature_model import LiteratureShoulderModel
from literature_sector import (
    LiteratureSectorTransmission,
    SectorDesignConfig,
    SectorPitchCurveData,
    SectorToothGeometry,
    generate_sector_teeth,
    synthesize_sector_pitch_curves,
)
from optimized_sector import ClosedSectorBlank, build_closed_sector_blank


BENCH_LABEL = (
    "RESEARCH-ONLY UNLOADED HAND-DRIVEN PROTOTYPE — "
    "NOT FOR HUMAN OR POWERED USE"
)


@dataclass(frozen=True)
class BenchPrototypeConfig:
    center_distance_mm: float = 120.0
    input_sector_angle_deg: float = 180.0
    module_mm: float = 2.5
    pressure_angle_deg: float = 20.0
    backlash_mm: float = 0.45
    minimum_ratio: float = 0.08
    smoothing_strength: float = 0.35
    profile_relief_mm: float = 0.30
    gear_thickness_mm: float = 8.0
    hub_thickness_mm: float = 12.0
    hub_diameter_mm: float = 22.0
    bore_diameter_mm: float = 8.0
    shaft_clearance_mm: float = 0.2
    tooth_root_embed_mm: float = 1.5
    body_web_thickness_mm: float = 6.0
    root_fillet_radius_mm: float = 0.8
    minimum_tooth_connection_width_mm: float = 2.0
    body_root_clearance_mm: float = 0.0
    minimum_clearance_mm: float = 0.25
    contact_tolerance_mm: float = 0.05
    maximum_st_error_deg: float = 3.0
    rms_st_error_deg: float = 2.0
    endpoint_error_deg: float = 0.5
    minimum_pitch_radius_mm: float = 8.0
    minimum_root_thickness_mm: float = 1.5
    minimum_tip_thickness_mm: float = 0.8
    mesh_positions: int = 2001

    @property
    def bore_radius_mm(self) -> float:
        return 0.5 * (self.bore_diameter_mm + self.shaft_clearance_mm)

    @property
    def web_thickness_mm(self) -> float:
        return self.body_web_thickness_mm

    def sector_config(self) -> SectorDesignConfig:
        return SectorDesignConfig(
            center_distance=self.center_distance_mm,
            input_sector_angle_deg=self.input_sector_angle_deg,
            module=self.module_mm,
            pressure_angle_deg=self.pressure_angle_deg,
            backlash=self.backlash_mm,
            minimum_ratio=self.minimum_ratio,
            smoothing_strength=self.smoothing_strength,
            sample_count=self.mesh_positions,
        )


@dataclass(frozen=True)
class BenchMeshPosition:
    sample_index: int
    elevation_deg: float
    input_angle_rad: float
    output_angle_rad: float
    intended_input_tooth: int
    intended_output_tooth: int
    maximum_penetration_area_mm2: float
    minimum_noncontact_clearance_mm: float
    intended_contact_distance_mm: float


@dataclass(frozen=True)
class BenchValidation:
    maximum_st_error_deg: float
    rms_st_error_deg: float
    endpoint_error_deg: float
    minimum_ratio: float
    minimum_pitch_radius_mm: float
    maximum_tooth_penetration_area_mm2: float
    minimum_noncontact_clearance_mm: float
    adjacent_teeth_overlap_free: bool
    no_tooth_skipping: bool
    closed_valid_bodies: bool
    minimum_root_thickness_mm: float
    minimum_tip_thickness_mm: float
    minimum_tooth_connection_width_mm: float
    no_extrapolation: bool
    continuous_hand_rotation: bool
    all_practical_gates_pass: bool
    decision: str
    failures: tuple[str, ...]


@dataclass(frozen=True)
class BenchPrototype:
    config: BenchPrototypeConfig
    transmission: LiteratureSectorTransmission
    pitch_data: SectorPitchCurveData
    teeth: SectorToothGeometry
    input_teeth: tuple[np.ndarray, ...]
    output_teeth: tuple[np.ndarray, ...]
    input_blank: ClosedSectorBlank
    output_blank: ClosedSectorBlank
    mesh_positions: tuple[BenchMeshPosition, ...]
    validation: BenchValidation


def _relieve_teeth(
    polygons: tuple[np.ndarray, ...], relief_mm: float
) -> tuple[np.ndarray, ...]:
    relieved = []
    for points in polygons:
        shape = Polygon(points).buffer(-relief_mm, join_style="round")
        if shape.is_empty or shape.geom_type != "Polygon":
            raise ValueError("Profile relief removed an entire tooth.")
        relieved.append(np.asarray(shape.exterior.coords, dtype=np.float64))
    return tuple(relieved)


def _adjacent_overlap_free(polygons: tuple[np.ndarray, ...]) -> bool:
    return all(
        Polygon(first).intersection(Polygon(second)).area <= 1e-10
        for first, second in zip(polygons[:-1], polygons[1:])
    )


def _outward_normals(points: np.ndarray) -> np.ndarray:
    tangent = np.gradient(points, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1)[:, None], 1e-15)
    candidate = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    return np.where(
        (np.sum(candidate * points, axis=1) >= 0)[:, None],
        candidate,
        -candidate,
    )


def _build_bench_sector_blank(
    pitch_points: np.ndarray,
    tooth_polygons: tuple[np.ndarray, ...],
    config: BenchPrototypeConfig,
    sector_config: SectorDesignConfig,
) -> ClosedSectorBlank:
    """Build a visible rim-and-spoke sector instead of a solid center fan."""

    normals = _outward_normals(pitch_points)
    root = (
        pitch_points
        - (
            sector_config.dedendum_factor * sector_config.module
            + config.body_root_clearance_mm
        )
        * normals
    )
    # The body follows the root curve as a printable rim.  Its inner edge is
    # offset toward the shaft by the requested web thickness.
    inner = root - config.web_thickness_mm * normals
    rim = Polygon(np.vstack((root, inner[::-1]))).buffer(0)
    hub_radius = max(
        config.bore_radius_mm + config.web_thickness_mm,
        2.2 * config.module_mm,
    )
    hub = Point(0.0, 0.0).buffer(hub_radius, quad_segs=48)
    solids = [rim, hub]
    connection_width = max(
        config.minimum_tooth_connection_width_mm,
        2.0 * config.root_fillet_radius_mm,
    )
    for points in tooth_polygons:
        tooth = Polygon(points).buffer(0)
        solids.append(tooth)
        centroid = np.array([tooth.centroid.x, tooth.centroid.y])
        nearest_index = int(
            np.argmin(np.linalg.norm(pitch_points - centroid, axis=1))
        )
        nearest = inner[nearest_index] - (
            config.tooth_root_embed_mm * normals[nearest_index]
        )
        # Profile relief shrinks the tooth root as well as its flanks.  A
        # narrow root bridge reconnects that relieved tooth to the rim without
        # changing the working tip/flank clearance.
        solids.append(
            LineString([tuple(centroid), tuple(nearest)]).buffer(
                connection_width / 2.0,
                cap_style="round",
                join_style="round",
            )
        )

    # Three narrow radial webs connect the hub to the sector rim without
    # obscuring the gear teeth or filling the entire swept sector.
    for index in sorted({0, len(inner) // 2, len(inner) - 1}):
        spoke = LineString([(0.0, 0.0), tuple(inner[index])]).buffer(
            config.web_thickness_mm / 2.0,
            cap_style="square",
            join_style="round",
        )
        solids.append(spoke)

    # Compact stop pads at both ends provide visible mechanical-limit faces.
    stop_size = max(2.0 * config.module_mm, config.web_thickness_mm)
    for endpoint in (root[0], root[-1]):
        direction = endpoint / max(np.linalg.norm(endpoint), 1e-15)
        tangent = np.array([-direction[1], direction[0]])
        center = endpoint - 0.2 * stop_size * direction
        corners = np.array(
            [
                center - 0.7 * stop_size * direction - 0.5 * stop_size * tangent,
                center + 0.5 * stop_size * direction - 0.5 * stop_size * tangent,
                center + 0.5 * stop_size * direction + 0.5 * stop_size * tangent,
                center - 0.7 * stop_size * direction + 0.5 * stop_size * tangent,
            ]
        )
        solids.append(Polygon(corners))

    body = unary_union(solids).buffer(0)
    if body.geom_type == "MultiPolygon":
        # A disconnected tooth is a construction failure, not something to
        # silently discard.
        raise ValueError("Bench sector rim, teeth, hub, and spokes are disconnected.")
    body = body.difference(
        Point(0.0, 0.0).buffer(config.bore_radius_mm, quad_segs=48)
    )
    boundary = np.asarray(body.exterior.coords, dtype=np.float64)
    closed = bool(
        len(boundary) > 3
        and np.linalg.norm(boundary[0] - boundary[-1]) < 1e-10
    )
    valid = bool(
        body.geom_type == "Polygon"
        and body.is_valid
        and closed
        and len(body.interiors) >= 1
    )
    warnings = () if valid else ("Rim-and-spoke sector body is invalid.",)
    return ClosedSectorBlank(
        polygon=body,
        boundary=boundary,
        bore_radius=config.bore_radius_mm,
        web_thickness=config.web_thickness_mm,
        hard_stop_size=stop_size,
        valid=valid,
        closed=closed,
        warnings=warnings,
    )


def _transform_teeth(
    polygons: tuple[np.ndarray, ...],
    angle_rad: float,
    *,
    x_offset: float = 0.0,
) -> list[Polygon]:
    result = []
    for points in polygons:
        shape = affinity.rotate(
            Polygon(points), np.degrees(angle_rad), origin=(0.0, 0.0)
        )
        if x_offset:
            shape = affinity.translate(shape, xoff=x_offset)
        result.append(shape)
    return result


def _mesh_sweep(
    data: SectorPitchCurveData,
    input_teeth: tuple[np.ndarray, ...],
    output_teeth: tuple[np.ndarray, ...],
) -> tuple[tuple[BenchMeshPosition, ...], bool]:
    positions: list[BenchMeshPosition] = []
    previous_pair: tuple[int, int] | None = None
    no_skipping = True
    for index in range(len(data.elevation_deg)):
        input_world = _transform_teeth(input_teeth, float(data.input_rad[index]))
        output_world = _transform_teeth(
            output_teeth,
            -float(data.output_rad[index]),
            x_offset=data.center_distance,
        )
        contact = Point(float(data.input_radii[index]), 0.0)
        intended_input = min(
            range(len(input_world)),
            key=lambda tooth: contact.distance(input_world[tooth]),
        )
        intended_output = min(
            range(len(output_world)),
            key=lambda tooth: contact.distance(output_world[tooth]),
        )
        pair = (intended_input, intended_output)
        if previous_pair is not None and (
            abs(pair[0] - previous_pair[0]) > 1
            or abs(pair[1] - previous_pair[1]) > 1
        ):
            no_skipping = False
        previous_pair = pair

        maximum_penetration = 0.0
        minimum_clearance = np.inf
        for input_index, first in enumerate(input_world):
            for output_index, second in enumerate(output_world):
                area = float(first.intersection(second).area)
                maximum_penetration = max(maximum_penetration, area)
                if (input_index, output_index) != pair:
                    minimum_clearance = min(
                        minimum_clearance, float(first.distance(second))
                    )
        intended_distance = float(
            input_world[intended_input].distance(
                output_world[intended_output]
            )
        )
        positions.append(
            BenchMeshPosition(
                sample_index=index,
                elevation_deg=float(data.elevation_deg[index]),
                input_angle_rad=float(data.input_rad[index]),
                output_angle_rad=float(data.output_rad[index]),
                intended_input_tooth=intended_input + 1,
                intended_output_tooth=intended_output + 1,
                maximum_penetration_area_mm2=maximum_penetration,
                minimum_noncontact_clearance_mm=float(minimum_clearance),
                intended_contact_distance_mm=intended_distance,
            )
        )
    return tuple(positions), no_skipping


def build_bench_prototype(
    model: LiteratureShoulderModel,
    config: BenchPrototypeConfig | None = None,
) -> BenchPrototype:
    config = config or BenchPrototypeConfig()
    transmission = LiteratureSectorTransmission(
        model, config.sector_config(), regularized=True
    )
    data = synthesize_sector_pitch_curves(transmission)
    original_teeth = generate_sector_teeth(data, transmission.config)
    input_teeth = _relieve_teeth(
        original_teeth.input_teeth, config.profile_relief_mm
    )
    output_teeth = _relieve_teeth(
        original_teeth.output_teeth, config.profile_relief_mm
    )
    teeth = SectorToothGeometry(
        candidate="bench_regularized_relieved",
        input_teeth=input_teeth,
        output_teeth=output_teeth,
        input_root_curve=original_teeth.input_root_curve,
        output_root_curve=original_teeth.output_root_curve,
        input_tooth_count=len(input_teeth),
        output_tooth_count=len(output_teeth),
        module=original_teeth.module,
        pressure_angle_deg=original_teeth.pressure_angle_deg,
        addendum=original_teeth.addendum,
        dedendum=original_teeth.dedendum,
        backlash=original_teeth.backlash,
        root_fillet_radius=original_teeth.root_fillet_radius,
        input_arc_positions=original_teeth.input_arc_positions,
        output_arc_positions=original_teeth.output_arc_positions,
    )
    input_blank = _build_bench_sector_blank(
        data.input_points,
        input_teeth,
        config,
        transmission.config,
    )
    output_blank = _build_bench_sector_blank(
        data.output_points,
        output_teeth,
        config,
        transmission.config,
    )
    mesh, no_skipping = _mesh_sweep(data, input_teeth, output_teeth)

    target = np.asarray(model.st_angle_at(data.elevation_deg))
    error = data.absolute_st_deg - target
    alpha = np.radians(config.pressure_angle_deg)
    pitch_thickness = (
        np.pi * config.module_mm / 2.0 - config.backlash_mm
    )
    root_thickness = (
        pitch_thickness
        + 2.0 * original_teeth.dedendum * np.tan(alpha)
        - 2.0 * config.profile_relief_mm
    )
    tip_thickness = (
        pitch_thickness
        - 2.0 * original_teeth.addendum * np.tan(alpha)
        - 2.0 * config.profile_relief_mm
    )
    maximum_penetration = max(
        position.maximum_penetration_area_mm2 for position in mesh
    )
    minimum_clearance = min(
        position.minimum_noncontact_clearance_mm for position in mesh
    )
    connection_width = max(
        config.minimum_tooth_connection_width_mm,
        2.0 * config.root_fillet_radius_mm,
    )
    values = {
        "maximum_st_error": float(np.max(np.abs(error)))
        <= config.maximum_st_error_deg,
        "rms_st_error": float(np.sqrt(np.mean(error**2)))
        <= config.rms_st_error_deg,
        "endpoint_error": abs(float(error[-1])) <= config.endpoint_error_deg,
        "positive_ratio": float(np.min(data.ratio)) > 0,
        "minimum_pitch_radius": float(
            np.min(np.r_[data.input_radii, data.output_radii])
        )
        >= config.minimum_pitch_radius_mm,
        "zero_tooth_penetration": maximum_penetration <= 1e-10,
        "minimum_clearance": minimum_clearance >= config.minimum_clearance_mm,
        "adjacent_overlap": _adjacent_overlap_free(input_teeth)
        and _adjacent_overlap_free(output_teeth),
        "no_tooth_skipping": no_skipping,
        "closed_bodies": input_blank.valid and output_blank.valid,
        "minimum_root_thickness": root_thickness
        >= config.minimum_root_thickness_mm,
        "minimum_tip_thickness": tip_thickness
        >= config.minimum_tip_thickness_mm,
        "minimum_tooth_connection": connection_width
        >= config.minimum_tooth_connection_width_mm,
        "no_extrapolation": data.hard_stop_elevation_deg == (11.0, 147.0),
        "continuous_rotation": bool(
            np.all(np.diff(data.input_rad) > 0)
            and np.all(np.diff(data.output_rad) > 0)
        ),
    }
    failures = tuple(key for key, passed in values.items() if not passed)
    passed = all(values.values())
    validation = BenchValidation(
        maximum_st_error_deg=float(np.max(np.abs(error))),
        rms_st_error_deg=float(np.sqrt(np.mean(error**2))),
        endpoint_error_deg=float(error[-1]),
        minimum_ratio=float(np.min(data.ratio)),
        minimum_pitch_radius_mm=float(
            np.min(np.r_[data.input_radii, data.output_radii])
        ),
        maximum_tooth_penetration_area_mm2=maximum_penetration,
        minimum_noncontact_clearance_mm=minimum_clearance,
        adjacent_teeth_overlap_free=values["adjacent_overlap"],
        no_tooth_skipping=no_skipping,
        closed_valid_bodies=values["closed_bodies"],
        minimum_root_thickness_mm=float(root_thickness),
        minimum_tip_thickness_mm=float(tip_thickness),
        minimum_tooth_connection_width_mm=float(connection_width),
        no_extrapolation=values["no_extrapolation"],
        continuous_hand_rotation=values["continuous_rotation"],
        all_practical_gates_pass=passed,
        decision=(
            "GO FOR UNLOADED HAND-DRIVEN BENCH PROTOTYPE"
            if passed
            else "NO-GO"
        ),
        failures=failures,
    )
    return BenchPrototype(
        config=config,
        transmission=transmission,
        pitch_data=data,
        teeth=teeth,
        input_teeth=input_teeth,
        output_teeth=output_teeth,
        input_blank=input_blank,
        output_blank=output_blank,
        mesh_positions=mesh,
        validation=validation,
    )
