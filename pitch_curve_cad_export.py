"""CAD export of the committed literature pitch curves with no teeth.

The input is the committed ``LiteratureSectorPitchCurves.csv`` artifact.  No
biomechanics, transmission, tooth, rack, or envelope calculation is imported
or executed here.

The verified literature curves are open partial sectors.  A straight endpoint
chord closes each unchanged sampled outer curve into a manufacturable planar
profile.  The STEP solid uses a B-spline through every committed pitch sample
and one exact line for that closure chord.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import cadquery as cq
import ezdxf
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
from shapely.geometry import LineString, Polygon


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PitchCurveExportValidation:
    member: str
    source_candidate: str
    sample_count: int
    source_curve_unchanged: bool
    open_literature_curve: bool
    closure_method: str
    closed_profile: bool
    no_self_intersections: bool
    valid_step_solid: bool
    one_body: bool
    thickness_mm: float
    measured_thickness_mm: float
    center_distance_mm: float
    maximum_center_distance_error_mm: float
    volume_mm3: float
    passed: bool


@dataclass(frozen=True)
class PitchCurveArtifact:
    member: str
    source_points: FloatArray
    closed_profile: FloatArray
    arc_length: FloatArray
    tangent_angle_deg: FloatArray
    curvature: FloatArray
    solid: cq.Workplane
    validation: PitchCurveExportValidation


@dataclass(frozen=True)
class DualPitchPathValidation:
    passed: bool
    solid_count: int
    separate_bodies: bool
    thickness_mm: float
    input_axis: tuple[float, float]
    output_axis: tuple[float, float]
    shaft_hole_diameter_mm: float
    input_hole_clear: bool
    output_hole_clear: bool
    center_distance_mm: float
    body_distance_mm: float
    operating_contact: bool
    input_pitch_samples_unchanged: bool
    output_pitch_samples_unchanged: bool


def load_committed_pitch_curves(
    source_csv: str | Path,
    *,
    candidate: str = "regularized",
) -> tuple[FloatArray, FloatArray, float, float]:
    """Load the exact finalized arrays without running the synthesis pipeline."""
    path = Path(source_csv)
    with path.open(newline="", encoding="utf-8") as stream:
        first = stream.readline()
        if first.startswith("RESEARCH"):
            reader = csv.DictReader(stream)
        else:
            stream.seek(0)
            reader = csv.DictReader(stream)
        rows = [row for row in reader if row["candidate"] == candidate]
    if not rows:
        raise ValueError(f"No {candidate!r} pitch-curve rows in {path}.")
    input_points = np.array(
        [[float(row["input_x"]), float(row["input_y"])] for row in rows],
        dtype=np.float64,
    )
    output_points = np.array(
        [
            [float(row["output_local_x"]), float(row["output_local_y"])]
            for row in rows
        ],
        dtype=np.float64,
    )
    center = np.array([float(row["center_distance"]) for row in rows])
    radial_sum = np.array(
        [float(row["input_radius"]) + float(row["output_radius"]) for row in rows]
    )
    center_distance = float(center[0])
    maximum_error = float(
        max(
            np.max(np.abs(center - center_distance)),
            np.max(np.abs(radial_sum - center_distance)),
        )
    )
    return input_points, output_points, center_distance, maximum_error


def _differential_geometry(points: FloatArray) -> tuple[FloatArray, ...]:
    segment = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(segment)))
    x = CubicSpline(arc, points[:, 0], bc_type="natural")
    y = CubicSpline(arc, points[:, 1], bc_type="natural")
    first = np.column_stack((x(arc, 1), y(arc, 1)))
    second = np.column_stack((x(arc, 2), y(arc, 2)))
    speed = np.linalg.norm(first, axis=1)
    tangent = np.degrees(np.arctan2(first[:, 1], first[:, 0]))
    curvature = (
        first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    ) / np.maximum(speed**3, 1e-15)
    return arc, tangent, curvature


def _cad_solid(points: FloatArray, thickness_mm: float) -> cq.Workplane:
    vectors = [cq.Vector(float(x), float(y), 0.0) for x, y in points]
    spline = cq.Edge.makeSpline(vectors)
    closure = cq.Edge.makeLine(vectors[-1], vectors[0])
    wire = cq.Wire.assembleEdges([spline, closure])
    if not wire.IsClosed():
        raise ValueError("B-spline plus endpoint chord did not close.")
    solid = cq.Solid.extrudeLinear(
        wire,
        [],
        cq.Vector(0.0, 0.0, thickness_mm),
    )
    return cq.Workplane("XY").newObject([solid])


def build_pitch_curve_artifact(
    member: str,
    points: FloatArray,
    *,
    thickness_mm: float,
    center_distance_mm: float,
    maximum_center_error_mm: float,
) -> PitchCurveArtifact:
    """Build one toothless, hubless, boreless pitch-curve sector solid."""
    source = np.asarray(points, dtype=np.float64)
    if len(source) < 8 or not np.all(np.isfinite(source)):
        raise ValueError("Pitch curve needs at least eight finite points.")
    closed = np.vstack((source, source[0]))
    line = LineString(source)
    polygon = Polygon(closed)
    solid = _cad_solid(source, thickness_mm)
    shape = solid.val()
    z_length = float(shape.BoundingBox().zlen)
    valid_solid = bool(shape.isValid())
    one_body = len(shape.Solids()) == 1
    validation = PitchCurveExportValidation(
        member=member,
        source_candidate="regularized",
        sample_count=len(source),
        source_curve_unchanged=bool(np.array_equal(closed[:-1], source)),
        open_literature_curve=not np.array_equal(source[0], source[-1]),
        closure_method="straight endpoint chord; pitch samples unchanged",
        closed_profile=bool(np.array_equal(closed[0], closed[-1])),
        no_self_intersections=bool(line.is_simple and polygon.is_valid),
        valid_step_solid=valid_solid,
        one_body=one_body,
        thickness_mm=thickness_mm,
        measured_thickness_mm=z_length,
        center_distance_mm=center_distance_mm,
        maximum_center_distance_error_mm=maximum_center_error_mm,
        volume_mm3=float(shape.Volume()),
        passed=bool(
            line.is_simple
            and polygon.is_valid
            and valid_solid
            and one_body
            and math.isclose(z_length, thickness_mm, abs_tol=1e-6)
            and maximum_center_error_mm <= 1e-9
        ),
    )
    arc, tangent, curvature = _differential_geometry(source)
    return PitchCurveArtifact(
        member,
        source.copy(),
        closed,
        arc,
        tangent,
        curvature,
        solid,
        validation,
    )


def _export_dxf(artifact: PitchCurveArtifact, path: Path) -> None:
    document = ezdxf.new("R2010")
    document.units = ezdxf.units.MM
    layer = document.layers.new("PITCH_CURVE")
    layer.dxf.color = 7
    modelspace = document.modelspace()
    modelspace.add_spline(
        fit_points=artifact.source_points.tolist(),
        dxfattribs={"layer": "PITCH_CURVE"},
    )
    modelspace.add_line(
        artifact.source_points[-1].tolist(),
        artifact.source_points[0].tolist(),
        dxfattribs={"layer": "PITCH_CURVE"},
    )
    document.saveas(path)


def _export_csv(artifact: PitchCurveArtifact, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["ArcLength_mm", "X_mm", "Y_mm", "TangentAngle_deg", "Curvature"]
        )
        writer.writerows(
            zip(
                artifact.arc_length,
                artifact.source_points[:, 0],
                artifact.source_points[:, 1],
                artifact.tangent_angle_deg,
                artifact.curvature,
            )
        )


def export_pitch_curve_solids(
    source_csv: str | Path,
    output_directory: str | Path,
    *,
    thickness_mm: float = 8.0,
) -> tuple[PitchCurveArtifact, PitchCurveArtifact, tuple[Path, ...]]:
    """Export the two committed pitch curves without invoking tooth code."""
    input_points, output_points, center, center_error = (
        load_committed_pitch_curves(source_csv)
    )
    input_artifact = build_pitch_curve_artifact(
        "Input",
        input_points,
        thickness_mm=thickness_mm,
        center_distance_mm=center,
        maximum_center_error_mm=center_error,
    )
    output_artifact = build_pitch_curve_artifact(
        "Output",
        output_points,
        thickness_mm=thickness_mm,
        center_distance_mm=center,
        maximum_center_error_mm=center_error,
    )
    if not input_artifact.validation.passed or not output_artifact.validation.passed:
        raise ValueError("Pitch-curve CAD validation failed; export blocked.")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for artifact in (input_artifact, output_artifact):
        stem = f"{artifact.member}PitchCurve"
        step_path = output / f"{stem}.step"
        dxf_path = output / f"{stem}.dxf"
        csv_path = output / f"{stem}.csv"
        cq.exporters.export(
            artifact.solid,
            str(step_path),
            exportType="STEP",
            opt={"write_pcurves": True},
        )
        _export_dxf(artifact, dxf_path)
        _export_csv(artifact, csv_path)
        paths.extend((step_path, dxf_path, csv_path))
    validation_path = output / "PitchCurveExportValidation.json"
    validation_path.write_text(
        json.dumps(
            {
                "source": str(Path(source_csv)),
                "source_candidate": "regularized",
                "teeth_generated": False,
                "thickness_mm": thickness_mm,
                "input": asdict(input_artifact.validation),
                "output": asdict(output_artifact.validation),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    paths.append(validation_path)
    return input_artifact, output_artifact, tuple(paths)


def _minimal_axis_support(
    artifact: PitchCurveArtifact,
    *,
    thickness_mm: float,
    hole_radius_mm: float,
    land_radius_mm: float,
) -> cq.Workplane:
    """Connect an axis land to the existing pitch face with two narrow webs."""
    base = artifact.solid
    land = (
        cq.Workplane("XY")
        .circle(land_radius_mm)
        .extrude(thickness_mm)
    )
    connectors = []
    for endpoint in (
        artifact.source_points[0],
        artifact.source_points[-1],
    ):
        direction = endpoint / np.linalg.norm(endpoint)
        tangent = np.array([-direction[1], direction[0]])
        half_width = max(1.5, 0.55 * land_radius_mm)
        near = direction * (0.65 * land_radius_mm)
        far = endpoint
        polygon = [
            tuple(near - half_width * tangent),
            tuple(near + half_width * tangent),
            tuple(far + half_width * tangent),
            tuple(far - half_width * tangent),
        ]
        connectors.append(
            cq.Workplane("XY").polyline(polygon).close().extrude(thickness_mm)
        )
    result = base.union(land)
    for connector in connectors:
        result = result.union(connector)
    hole = (
        cq.Workplane("XY")
        .circle(hole_radius_mm)
        .extrude(thickness_mm + 2.0)
        .translate((0.0, 0.0, -1.0))
    )
    return result.cut(hole).clean()


def export_dual_operating_pitch_paths(
    source_csv: str | Path,
    destination_step: str | Path,
    *,
    thickness_mm: float = 2.0,
    shaft_hole_diameter_mm: float = 4.0,
) -> DualPitchPathValidation:
    """Export two separate printable pitch-path bodies in operating position."""
    input_points, output_points, center, center_error = (
        load_committed_pitch_curves(source_csv)
    )
    input_artifact = build_pitch_curve_artifact(
        "Input",
        input_points,
        thickness_mm=thickness_mm,
        center_distance_mm=center,
        maximum_center_error_mm=center_error,
    )
    output_artifact = build_pitch_curve_artifact(
        "Output",
        output_points,
        thickness_mm=thickness_mm,
        center_distance_mm=center,
        maximum_center_error_mm=center_error,
    )
    hole_radius = shaft_hole_diameter_mm / 2.0
    land_radius = max(5.0, hole_radius + 2.0)
    input_body = _minimal_axis_support(
        input_artifact,
        thickness_mm=thickness_mm,
        hole_radius_mm=hole_radius,
        land_radius_mm=land_radius,
    )
    output_local = _minimal_axis_support(
        output_artifact,
        thickness_mm=thickness_mm,
        hole_radius_mm=hole_radius,
        land_radius_mm=land_radius,
    )
    output_body = output_local.translate((center, 0.0, 0.0))
    first_shape = input_body.val()
    second_shape = output_body.val()
    compound = cq.Compound.makeCompound([first_shape, second_shape])
    destination = Path(destination_step)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cq.exporters.export(
        compound,
        str(destination),
        exportType="STEP",
        opt={"write_pcurves": True},
    )

    imported = cq.importers.importStep(str(destination)).val()
    solids = imported.Solids()
    input_probe = (
        cq.Workplane("XY")
        .circle(hole_radius * 0.95)
        .extrude(thickness_mm + 2.0)
        .translate((0.0, 0.0, -1.0))
        .val()
    )
    output_probe = input_probe.moved(
        cq.Location(cq.Vector(center, 0.0, 0.0))
    )
    input_clear = first_shape.intersect(input_probe).Volume() <= 1e-8
    output_clear = second_shape.intersect(output_probe).Volume() <= 1e-8
    body_distance = float(first_shape.distance(second_shape))
    validation = DualPitchPathValidation(
        passed=False,
        solid_count=len(solids),
        separate_bodies=len(solids) == 2,
        thickness_mm=thickness_mm,
        input_axis=(0.0, 0.0),
        output_axis=(center, 0.0),
        shaft_hole_diameter_mm=shaft_hole_diameter_mm,
        input_hole_clear=input_clear,
        output_hole_clear=output_clear,
        center_distance_mm=center,
        body_distance_mm=body_distance,
        operating_contact=body_distance <= 1e-5,
        input_pitch_samples_unchanged=np.array_equal(
            input_points, input_artifact.source_points
        ),
        output_pitch_samples_unchanged=np.array_equal(
            output_points, output_artifact.source_points
        ),
    )
    passed = all(
        (
            validation.separate_bodies,
            input_clear,
            output_clear,
            validation.operating_contact,
            validation.input_pitch_samples_unchanged,
            validation.output_pitch_samples_unchanged,
            math.isclose(
                validation.center_distance_mm, 120.0, abs_tol=1e-9
            ),
        )
    )
    return DualPitchPathValidation(
        **{**asdict(validation), "passed": passed}
    )
