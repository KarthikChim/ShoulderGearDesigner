"""Search, validate, and export an unloaded hand-driven bench prototype."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import ezdxf
import numpy as np
import svgwrite
from shapely.geometry import Point, Polygon

from bench_prototype import (
    BENCH_LABEL,
    BenchPrototype,
    BenchPrototypeConfig,
    build_bench_prototype,
)
from biomechanics.literature_model import LiteratureShoulderModel


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "validation_outputs"
MODEL = ROOT / "ConsensusShoulderModel.json"


def candidate_configs():
    """Cover practical ranges while keeping the bench search reproducible."""

    # The shared sector model requires at least 1001 samples.  Candidates that
    # pass this screening sweep are then rechecked at 2001 positions.
    base = BenchPrototypeConfig(mesh_positions=1001)
    values = [
        # center, sector, module, pressure, backlash, min ratio, smoothing, relief
        (110, 150, 2.0, 20, 0.25, 0.08, 0.10, 0.25),
        (120, 165, 2.5, 20, 0.35, 0.08, 0.25, 0.25),
        (120, 180, 2.5, 20, 0.45, 0.08, 0.35, 0.30),
        (130, 180, 2.5, 25, 0.35, 0.10, 0.25, 0.30),
        (140, 195, 3.0, 25, 0.45, 0.12, 0.35, 0.30),
        (150, 210, 3.0, 25, 0.35, 0.12, 0.50, 0.35),
        (160, 220, 3.5, 30, 0.45, 0.16, 0.50, 0.35),
        (115, 175, 2.0, 30, 0.45, 0.10, 0.25, 0.30),
        (125, 185, 3.5, 20, 0.25, 0.08, 0.35, 0.25),
        (135, 200, 2.5, 30, 0.35, 0.14, 0.25, 0.30),
        (145, 160, 3.0, 20, 0.45, 0.10, 0.10, 0.30),
        (155, 215, 2.0, 25, 0.35, 0.16, 0.35, 0.30),
    ]
    for item in values:
        yield replace(
            base,
            center_distance_mm=item[0],
            input_sector_angle_deg=item[1],
            module_mm=item[2],
            pressure_angle_deg=item[3],
            backlash_mm=item[4],
            minimum_ratio=item[5],
            smoothing_strength=item[6],
            profile_relief_mm=item[7],
        )


def search(model):
    results = []
    candidates = []
    for index, config in enumerate(candidate_configs(), start=1):
        try:
            prototype = build_bench_prototype(model, config)
            result = {
                "candidate_id": index,
                **asdict(config),
                **asdict(prototype.validation),
            }
            results.append(result)
            candidates.append((result, prototype))
        except Exception as error:
            results.append(
                {
                    "candidate_id": index,
                    **asdict(config),
                    "all_practical_gates_pass": False,
                    "decision": "NO-GO",
                    "failures": [f"{type(error).__name__}: {error}"],
                }
            )
    candidates.sort(
        key=lambda item: (
            not item[1].validation.all_practical_gates_pass,
            item[1].validation.maximum_tooth_penetration_area_mm2 > 0,
            item[1].validation.rms_st_error_deg,
            -item[1].validation.minimum_root_thickness_mm,
            -item[1].validation.minimum_noncontact_clearance_mm,
            item[1].config.center_distance_mm,
        )
    )
    return results, candidates


def select_full_resolution(model, candidates):
    """Recheck ranked coarse-pass candidates at exactly 2001 positions."""

    full_results = []
    for _, coarse in candidates:
        if not coarse.validation.all_practical_gates_pass:
            continue
        full_config = replace(coarse.config, mesh_positions=2001)
        full = build_bench_prototype(model, full_config)
        full_results.append(full)
        if full.validation.all_practical_gates_pass:
            return full, full_results
    return None, full_results


def _write_dxf(prototype: BenchPrototype, member: str, path: Path) -> None:
    blank = (
        prototype.input_blank
        if member == "input"
        else prototype.output_blank
    )
    teeth = (
        prototype.input_teeth
        if member == "input"
        else prototype.output_teeth
    )
    pitch = (
        prototype.pitch_data.input_points
        if member == "input"
        else prototype.pitch_data.output_points
    )
    document = ezdxf.new("R2010")
    document.header["$INSUNITS"] = 4
    modelspace = document.modelspace()
    modelspace.add_text(
        BENCH_LABEL,
        height=2.5,
        dxfattribs={"layer": "RESEARCH_ONLY_WARNING"},
    )
    modelspace.add_lwpolyline(
        blank.boundary.tolist(),
        close=True,
        dxfattribs={"layer": "CLOSED_SECTOR_BODY"},
    )
    for interior in blank.polygon.interiors:
        ring = Polygon(interior)
        layer = (
            "SHAFT_BORE_PLACEHOLDER"
            if ring.contains(Point(0.0, 0.0))
            else "OPEN_WEB_CUTOUT"
        )
        modelspace.add_lwpolyline(
            list(interior.coords),
            close=True,
            dxfattribs={"layer": layer},
        )
    modelspace.add_lwpolyline(
        pitch.tolist(),
        close=False,
        dxfattribs={"layer": "OPEN_PITCH_SECTOR_REFERENCE"},
    )
    for index, tooth in enumerate(teeth, start=1):
        modelspace.add_lwpolyline(
            tooth.tolist(),
            close=True,
            dxfattribs={"layer": f"TOOTH_PROFILE_{index:03d}"},
        )
    document.saveas(path)


def _write_svg(prototype: BenchPrototype, path: Path) -> None:
    output_offset = np.array([prototype.pitch_data.center_distance, 0.0])
    input_boundary = prototype.input_blank.boundary
    output_boundary = prototype.output_blank.boundary + output_offset
    all_points = np.vstack((input_boundary, output_boundary))
    low = np.min(all_points, axis=0)
    high = np.max(all_points, axis=0)
    margin = 12.0
    size = high - low + 2 * margin
    drawing = svgwrite.Drawing(
        path,
        size=(f"{size[0]}mm", f"{size[1]}mm"),
        viewBox=f"0 0 {size[0]} {size[1]}",
    )

    def converted(points):
        return [
            (
                float(point[0] - low[0] + margin),
                float(high[1] - point[1] + margin),
            )
            for point in points
        ]

    drawing.add(
        drawing.text(
            BENCH_LABEL,
            insert=(4, 7),
            fill="#b00020",
            font_size="3px",
            id="research-only-warning",
        )
    )
    drawing.add(
        drawing.polygon(
            converted(input_boundary),
            fill="#f6ad55",
            stroke="#7c2d12",
            stroke_width=0.25,
            id="input-closed-sector",
        )
    )
    # Preserve every open region between the rim, hub, and spokes.
    for member, blank, offset in (
        ("input", prototype.input_blank, np.array([0.0, 0.0])),
        ("output", prototype.output_blank, output_offset),
    ):
        for index, interior in enumerate(blank.polygon.interiors, start=1):
            points = np.asarray(interior.coords, dtype=np.float64) + offset
            drawing.add(
                drawing.polygon(
                    converted(points),
                    fill="white",
                    stroke="#111827",
                    stroke_width=0.18,
                    id=f"{member}-body-cutout-{index:02d}",
                )
            )
    drawing.add(
        drawing.polygon(
            converted(output_boundary),
            fill="#63b3ed",
            stroke="#1e3a8a",
            stroke_width=0.25,
            id="output-closed-sector",
        )
    )
    # SVG polygons do not carry Shapely interior rings, so render the two
    # shaft bores explicitly on top of the filled sector bodies.
    input_center = converted(np.array([[0.0, 0.0]]))[0]
    output_center = converted(
        np.array([[prototype.pitch_data.center_distance, 0.0]])
    )[0]
    for center, identifier in (
        (input_center, "input-shaft-bore"),
        (output_center, "output-shaft-bore"),
    ):
        drawing.add(
            drawing.circle(
                center=center,
                r=prototype.config.bore_radius_mm,
                fill="white",
                stroke="#111827",
                stroke_width=0.25,
                id=identifier,
            )
        )
    # Keep individual flanks visible even where the completed body union
    # merges their roots into the rim.
    for index, tooth in enumerate(prototype.input_teeth, start=1):
        drawing.add(
            drawing.polyline(
                converted(tooth),
                fill="none",
                stroke="#7c2d12",
                stroke_width=0.12,
                id=f"input-tooth-outline-{index:03d}",
            )
        )
    for index, tooth in enumerate(prototype.output_teeth, start=1):
        drawing.add(
            drawing.polyline(
                converted(tooth + output_offset),
                fill="none",
                stroke="#1e3a8a",
                stroke_width=0.12,
                id=f"output-tooth-outline-{index:03d}",
            )
        )
    drawing.save()


def export_passing_prototype(
    prototype: BenchPrototype, search_results: list[dict]
) -> tuple[Path, ...]:
    if not prototype.validation.all_practical_gates_pass:
        raise RuntimeError("Prototype export blocked: practical gates did not pass.")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    input_path = OUTPUT / "BenchPrototype_InputSector.dxf"
    output_path = OUTPUT / "BenchPrototype_OutputSector.dxf"
    svg_path = OUTPUT / "BenchPrototype_Pair.svg"
    validation_path = OUTPUT / "BenchPrototype_Validation.json"
    guide_path = OUTPUT / "BenchPrototype_PrintGuide.md"
    _write_dxf(prototype, "input", input_path)
    _write_dxf(prototype, "output", output_path)
    _write_svg(prototype, svg_path)
    validation_path.write_text(
        json.dumps(
            {
                "warning": BENCH_LABEL,
                "source": "McClure2001",
                "condition": prototype.transmission.model.selected["condition"],
                "valid_range_deg": list(
                    prototype.transmission.valid_range_deg
                ),
                "selected_config": asdict(prototype.config),
                "validation": asdict(prototype.validation),
                "mesh_position_count": len(prototype.mesh_positions),
                "input_tooth_count": len(prototype.input_teeth),
                "output_tooth_count": len(prototype.output_teeth),
                "search_results": search_results,
            },
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    guide_path.write_text(
        "\n".join(
            [
                f"# {BENCH_LABEL}",
                "",
                "This is an unloaded visual/mechanical demonstration only.",
                "",
                "## First-print recommendations",
                "",
                "- Print in PLA or PETG.",
                "- Test only by turning slowly by hand.",
                "- Use adjustable center-distance slots.",
                "- Add 0.3–0.5 mm extra physical backlash for the first print.",
                "- Use removable shaft hubs.",
                "- Do not attach motors.",
                "- Do not attach the mechanism to a body.",
                "- Stop immediately if teeth bind, crack, or climb.",
                "",
                "The DXFs include a shaft-bore placeholder, not a finished bearing "
                "or shaft interface. Verify scale and clearances with a small test "
                "coupon before printing full sectors.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return input_path, output_path, svg_path, validation_path, guide_path


def main() -> None:
    model = LiteratureShoulderModel(MODEL)
    results, candidates = search(model)
    prototype, full_results = select_full_resolution(model, candidates)
    if prototype is None:
        print(BENCH_LABEL)
        print("Decision: NO-GO")
        print("No prototype files exported.")
        raise SystemExit(1)
    paths = export_passing_prototype(prototype, results)
    print(BENCH_LABEL)
    print(f"Candidates searched: {len(results)}")
    print(f"Full-resolution candidates checked: {len(full_results)}")
    print(f"Mesh positions: {len(prototype.mesh_positions)}")
    print(f"Decision: {prototype.validation.decision}")
    print(f"Maximum ST error: {prototype.validation.maximum_st_error_deg:.6f}°")
    print(f"RMS ST error: {prototype.validation.rms_st_error_deg:.6f}°")
    print(
        "Minimum non-contacting clearance: "
        f"{prototype.validation.minimum_noncontact_clearance_mm:.6f} mm"
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
