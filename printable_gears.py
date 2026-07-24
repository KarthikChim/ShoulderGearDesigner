"""CadQuery solids for the connected McClure literature gear sectors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cadquery as cq
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from shapely.geometry import Polygon

from bench_prototype import (
    BENCH_LABEL,
    BenchPrototype,
    BenchPrototypeConfig,
    build_bench_prototype,
)
from biomechanics.literature_model import LiteratureShoulderModel


@dataclass(frozen=True)
class PrintableGearParameters:
    gear_thickness_mm: float = 8.0
    hub_thickness_mm: float = 12.0
    hub_diameter_mm: float = 22.0
    bore_diameter_mm: float = 8.0
    shaft_clearance_mm: float = 0.2
    tooth_root_embed_mm: float = 1.5
    body_web_thickness_mm: float = 6.0
    root_fillet_radius_mm: float = 0.8
    keyway_width_mm: float = 0.0

    @property
    def bore_radius_mm(self) -> float:
        return 0.5 * (self.bore_diameter_mm + self.shaft_clearance_mm)


@dataclass(frozen=True)
class GearSolidValidation:
    valid: bool
    one_connected_solid: bool
    positive_volume: bool
    bore_through: bool
    volume_mm3: float
    solid_count: int
    detached_tooth_count: int

    @property
    def passed(self) -> bool:
        return all(
            (
                self.valid,
                self.one_connected_solid,
                self.positive_volume,
                self.bore_through,
                self.detached_tooth_count == 0,
            )
        )


@dataclass(frozen=True)
class PrintableGearSolid:
    name: str
    cad: cq.Workplane
    validation: GearSolidValidation

    @property
    def shape(self) -> cq.Shape:
        return self.cad.val()


@dataclass(frozen=True)
class PrintableGearPair:
    input_gear: PrintableGearSolid
    output_gear: PrintableGearSolid
    prototype: BenchPrototype
    parameters: PrintableGearParameters
    warning: str = BENCH_LABEL


def _profile_from_polygon(polygon: Polygon) -> cq.Workplane:
    """Create a CadQuery pending-wire profile including every open cutout."""

    exterior = list(polygon.exterior.coords)[:-1]
    profile = cq.Workplane("XY").polyline(exterior).close()
    for interior in polygon.interiors:
        points = list(interior.coords)[:-1]
        profile = profile.moveTo(*points[0]).polyline(points[1:]).close()
    return profile


def _validate_solid(
    cad: cq.Workplane,
    parameters: PrintableGearParameters,
) -> GearSolidValidation:
    shape = cad.val()
    solids = shape.Solids()
    volume = float(shape.Volume())
    bore_probe = (
        cq.Workplane("XY")
        .circle(parameters.bore_radius_mm * 0.92)
        .extrude(parameters.hub_thickness_mm + 2.0, both=True)
        .val()
    )
    bore_intersection = float(shape.intersect(bore_probe).Volume())
    return GearSolidValidation(
        valid=bool(shape.isValid()),
        one_connected_solid=len(solids) == 1,
        positive_volume=volume > 0.0,
        bore_through=bore_intersection <= 1e-7,
        volume_mm3=volume,
        solid_count=len(solids),
        # 2-D connectivity is checked before extrusion; a one-solid CadQuery
        # result proves no tooth became a detached 3-D solid.
        detached_tooth_count=0 if len(solids) == 1 else len(solids) - 1,
    )


def _make_solid(
    name: str,
    polygon: Polygon,
    parameters: PrintableGearParameters,
) -> PrintableGearSolid:
    base = _profile_from_polygon(polygon).extrude(
        parameters.gear_thickness_mm
    )
    hub_extension = (
        cq.Workplane("XY")
        .circle(parameters.hub_diameter_mm / 2.0)
        .circle(parameters.bore_radius_mm)
        .extrude(parameters.hub_thickness_mm)
    )
    solid = base.union(hub_extension).clean()
    if parameters.keyway_width_mm > 0:
        keyway = (
            cq.Workplane("XY")
            .box(
                parameters.keyway_width_mm,
                parameters.bore_radius_mm * 1.5,
                parameters.hub_thickness_mm + 2.0,
            )
            .translate(
                (
                    0.0,
                    parameters.bore_radius_mm,
                    parameters.hub_thickness_mm / 2.0,
                )
            )
        )
        solid = solid.cut(keyway).clean()
    validation = _validate_solid(solid, parameters)
    return PrintableGearSolid(name, solid, validation)


def generate_printable_gears(
    model_path: str | Path = "ConsensusShoulderModel.json",
    *,
    prototype: BenchPrototype | None = None,
    parameters: PrintableGearParameters | None = None,
) -> PrintableGearPair:
    """Return two genuine CadQuery solids without writing any files."""

    parameters = parameters or PrintableGearParameters()
    if prototype is None:
        config = BenchPrototypeConfig(
            gear_thickness_mm=parameters.gear_thickness_mm,
            hub_thickness_mm=parameters.hub_thickness_mm,
            hub_diameter_mm=parameters.hub_diameter_mm,
            bore_diameter_mm=parameters.bore_diameter_mm,
            shaft_clearance_mm=parameters.shaft_clearance_mm,
            tooth_root_embed_mm=parameters.tooth_root_embed_mm,
            body_web_thickness_mm=parameters.body_web_thickness_mm,
            root_fillet_radius_mm=parameters.root_fillet_radius_mm,
        )
        prototype = build_bench_prototype(
            LiteratureShoulderModel(model_path), config
        )
    if not prototype.validation.all_practical_gates_pass:
        raise ValueError("The 2-D prototype did not pass its practical gates.")
    input_gear = _make_solid(
        "Literature input gear", prototype.input_blank.polygon, parameters
    )
    output_gear = _make_solid(
        "Literature output gear", prototype.output_blank.polygon, parameters
    )
    if not input_gear.validation.passed or not output_gear.validation.passed:
        raise ValueError("CadQuery did not produce two valid connected solids.")
    return PrintableGearPair(
        input_gear=input_gear,
        output_gear=output_gear,
        prototype=prototype,
        parameters=parameters,
    )


def export_printable_gears(
    pair: PrintableGearPair,
    directory: str | Path,
    *,
    export_stl: bool = True,
    export_step: bool = True,
) -> tuple[Path, ...]:
    """Optionally export genuine STL and STEP files from CadQuery solids."""

    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for prefix, gear in (
        ("BenchPrototype_InputGear", pair.input_gear),
        ("BenchPrototype_OutputGear", pair.output_gear),
    ):
        if export_stl:
            path = destination / f"{prefix}.stl"
            cq.exporters.export(gear.cad, str(path), exportType="STL")
            paths.append(path)
        if export_step:
            path = destination / f"{prefix}.step"
            cq.exporters.export(gear.cad, str(path), exportType="STEP")
            paths.append(path)
    return tuple(paths)


def preview_printable_gears(
    pair: PrintableGearPair,
    *,
    show: bool = True,
) -> plt.Figure:
    """Render a lightweight Matplotlib 3-D preview from CadQuery tessellation."""

    figure = plt.figure(figsize=(11, 7))
    axes = figure.add_subplot(111, projection="3d")
    for gear, offset, color in (
        (pair.input_gear, np.array([0.0, 0.0, 0.0]), "#f4a24c"),
        (
            pair.output_gear,
            np.array([pair.prototype.pitch_data.center_distance, 0.0, 0.0]),
            "#5aa9df",
        ),
    ):
        vertices, triangles = gear.shape.tessellate(0.25)
        points = np.array([[v.x, v.y, v.z] for v in vertices]) + offset
        faces = [points[list(triangle)] for triangle in triangles]
        axes.add_collection3d(
            Poly3DCollection(
                faces,
                facecolor=color,
                edgecolor="#25364a",
                linewidth=0.08,
                alpha=0.88,
            )
        )
    axes.autoscale_view()
    axes.set_box_aspect((2.0, 1.0, 0.35))
    axes.set_title(BENCH_LABEL)
    axes.set_xlabel("X (mm)")
    axes.set_ylabel("Y (mm)")
    axes.set_zlabel("Z (mm)")
    if show:
        plt.show()
    return figure


def export_featurescript(
    pair: PrintableGearPair,
    destination: str | Path,
) -> Path:
    """Write an Onshape script containing the exact connected outer profiles."""

    destination = Path(destination)

    def vectors(points: np.ndarray) -> str:
        return ",\n".join(
            f"        vector({x:.12g}, {y:.12g}) * millimeter"
            for x, y in points
        )

    input_points = np.asarray(
        pair.prototype.input_blank.polygon.exterior.coords[:-1],
        dtype=np.float64,
    )
    output_points = np.asarray(
        pair.prototype.output_blank.polygon.exterior.coords[:-1],
        dtype=np.float64,
    )
    source = f'''FeatureScript 3029;
import(path : "onshape/std/common.fs", version : "3029.0");

// RESEARCH-ONLY UNLOADED HAND-DRIVEN PROTOTYPE.
// NOT FOR HUMAN OR POWERED USE.
const INPUT_OUTLINE = [
{vectors(input_points)}
    ];
const OUTPUT_OUTLINE = [
{vectors(output_points)}
    ];

function closedOutline(sketch is Sketch, prefix is string, points is array,
        angle is ValueWithUnits, offset is Vector)
{{
    const c = cos(angle);
    const s = sin(angle);
    for (var index = 0; index < size(points); index += 1)
    {{
        const first = points[index];
        const second = points[(index + 1) % size(points)];
        skLineSegment(sketch, prefix ~ index, {{
            "start" : vector(
                first[0] * c - first[1] * s,
                first[0] * s + first[1] * c) + offset,
            "end" : vector(
                second[0] * c - second[1] * s,
                second[0] * s + second[1] * c) + offset
        }});
    }}
}}

function profileAt(context is Context, sketchId is Id, prefix is string,
        points is array, z is ValueWithUnits, angle is ValueWithUnits,
        offset is Vector) returns Query
{{
    const sketch = newSketchOnPlane(context, sketchId, {{
        "sketchPlane" : plane(
            vector(0, 0, z / meter) * meter,
            vector(0, 0, 1))
    }});
    closedOutline(sketch, prefix, points, angle, offset);
    skSolve(sketch);
    return qSketchRegion(sketchId);
}}

function herringboneBody(context is Context, bodyId is Id, prefix is string,
        points is array, thickness is ValueWithUnits,
        twist is ValueWithUnits, offset is Vector)
{{
    // Five sections create a symmetric double helix. Rotation rises from
    // zero at the lower face to the requested twist at the center plane,
    // then reverses back to zero at the upper face.
    const profiles = [
        profileAt(context, bodyId + "s0", prefix ~ "s0", points,
            0 * millimeter, 0 * degree, offset),
        profileAt(context, bodyId + "s1", prefix ~ "s1", points,
            thickness / 4, twist / 2, offset),
        profileAt(context, bodyId + "s2", prefix ~ "s2", points,
            thickness / 2, twist, offset),
        profileAt(context, bodyId + "s3", prefix ~ "s3", points,
            3 * thickness / 4, twist / 2, offset),
        profileAt(context, bodyId + "s4", prefix ~ "s4", points,
            thickness, 0 * degree, offset)
    ];
    opLoft(context, bodyId, {{
        "profileSubqueries" : profiles
    }});
}}

annotation {{ "Feature Type Name" : "Literature sector gear pair" }}
export const literatureSectorGearPair = defineFeature(function(context is Context,
        id is Id, definition is map)
    precondition
    {{
        annotation {{ "Name" : "Thickness", "Default" : 8 * millimeter }}
        isLength(definition.thickness, LENGTH_BOUNDS);
        annotation {{ "Name" : "Bore diameter", "Default" : 8 * millimeter }}
        isLength(definition.boreDiameter, LENGTH_BOUNDS);
        annotation {{ "Name" : "Center distance", "Default" : 120 * millimeter }}
        isLength(definition.centerDistance, LENGTH_BOUNDS);
        annotation {{ "Name" : "Herringbone half-angle", "Default" : 8 * degree }}
        isAngle(definition.helixAngle, ANGLE_360_BOUNDS);
    }}
    {{
        const inputOffset = vector(0, 0) * millimeter;
        const outputOffset = vector(definition.centerDistance / millimeter, 0)
            * millimeter;
        herringboneBody(context, id + "inputGear", "input", INPUT_OUTLINE,
            definition.thickness, definition.helixAngle, inputOffset);
        herringboneBody(context, id + "outputGear", "output", OUTPUT_OUTLINE,
            definition.thickness, -definition.helixAngle, outputOffset);

        // Cut bores only after the lofts exist. This guarantees genuine
        // cylindrical through-holes instead of accidentally extruding the
        // inner sketch region as a plug.
        fCylinder(context, id + "inputBoreTool", {{
            "bottomCenter" : vector(0, 0, -1) * millimeter,
            "topCenter" : vector(0, 0,
                definition.thickness / millimeter + 1) * millimeter,
            "radius" : definition.boreDiameter / 2
        }});
        opBoolean(context, id + "inputBoreCut", {{
            "tools" : qCreatedBy(id + "inputBoreTool", EntityType.BODY),
            "targets" : qCreatedBy(id + "inputGear", EntityType.BODY),
            "operationType" : BooleanOperationType.SUBTRACTION
        }});
        fCylinder(context, id + "outputBoreTool", {{
            "bottomCenter" : vector(
                definition.centerDistance / millimeter, 0, -1) * millimeter,
            "topCenter" : vector(
                definition.centerDistance / millimeter, 0,
                definition.thickness / millimeter + 1) * millimeter,
            "radius" : definition.boreDiameter / 2
        }});
        opBoolean(context, id + "outputBoreCut", {{
            "tools" : qCreatedBy(id + "outputBoreTool", EntityType.BODY),
            "targets" : qCreatedBy(id + "outputGear", EntityType.BODY),
            "operationType" : BooleanOperationType.SUBTRACTION
        }});
    }});
'''
    destination.write_text(source, encoding="utf-8")
    return destination
