"""Export the validated planetary pitch curves as separate STEP wire files.

Run from PyCharm or a terminal with:

    python export_planetary_pitch_steps.py

Generated files contain pitch-curve spline edges and 4 mm reference circles at
the exact rotation axes. They contain no teeth, faces, solids, bores, hubs,
spokes, or carrier bars.
"""

from __future__ import annotations

from pathlib import Path

from literature_planetary import (
    rotate_points,
    synthesize_literature_planetary_pitch_curves,
    validate_planetary_pitch_curves,
)


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "validation_outputs" / "LiteratureSectorTransmission.csv"
OUTPUT_DIRECTORY = ROOT / "planetary_pitch_exports" / "separate_step_curves"


def _spline_edge(points):
    """Create one smooth open CAD spline through every sampled pitch point."""
    import cadquery as cq

    vectors = [cq.Vector(float(x), float(y), 0.0) for x, y in points]
    return cq.Edge.makeSpline(vectors)


def _reference_circle(center_x: float, center_y: float):
    """Create a 4 mm diameter wire circle at a rotation axis."""
    import cadquery as cq

    return cq.Edge.makeCircle(
        2.0,
        cq.Vector(float(center_x), float(center_y), 0.0),
        cq.Vector(0.0, 0.0, 1.0),
    )


def _export_edges(edges, destination: Path) -> None:
    """Write independent wire edges as STEP geometry."""
    import cadquery as cq

    compound = cq.Compound.makeCompound(list(edges))
    cq.exporters.export(compound, str(destination), exportType="STEP")


def export_planetary_pitch_steps(
    *,
    center_distance_mm: float = 120.0,
    source_csv: str | Path = SOURCE,
    output_directory: str | Path = OUTPUT_DIRECTORY,
) -> dict[str, Path]:
    """Export separate sun, planet, and assembled-pair STEP wire files.

    The sun curve is exported in its stationary local/world frame. The planet
    curve is exported in two useful forms:

    - local: centered on the planet rotation axis;
    - assembled: placed at its starting carrier position relative to the sun.
    """
    import cadquery as cq

    data = synthesize_literature_planetary_pitch_curves(
        source_csv, center_distance_mm=center_distance_mm
    )
    validation = validate_planetary_pitch_curves(data)
    if not validation.passed:
        raise RuntimeError(f"Planetary pitch curves failed validation: {validation}")

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)

    sun_edge = _spline_edge(data.sun_pitch_points_local)
    planet_local_edge = _spline_edge(data.planet_pitch_points_local)

    start_index = 0
    planet_assembled_points = rotate_points(
        data.planet_pitch_points_local,
        data.planet_absolute_angle_rad[start_index],
    ) + data.planet_center_points_world[start_index]
    planet_assembled_edge = _spline_edge(planet_assembled_points)

    paths = {
        "sun": output / "StationarySunPitchCurve.step",
        "planet_local": output / "ShoulderPlanetPitchCurve_Local.step",
        "planet_assembled": output / "ShoulderPlanetPitchCurve_Assembled.step",
        "assembled_pair": output / "PlanetaryPitchCurvePair_Assembled.step",
    }
    sun_axis = _reference_circle(0.0, 0.0)
    planet_local_axis = _reference_circle(0.0, 0.0)
    planet_assembled_axis = _reference_circle(
        float(data.planet_center_points_world[start_index, 0]),
        float(data.planet_center_points_world[start_index, 1]),
    )

    _export_edges((sun_edge, sun_axis), paths["sun"])
    _export_edges((planet_local_edge, planet_local_axis), paths["planet_local"])
    _export_edges(
        (planet_assembled_edge, planet_assembled_axis),
        paths["planet_assembled"],
    )

    pair = cq.Compound.makeCompound(
        [sun_edge, planet_assembled_edge, sun_axis, planet_assembled_axis]
    )
    cq.exporters.export(pair, str(paths["assembled_pair"]), exportType="STEP")

    return paths


def main() -> None:
    paths = export_planetary_pitch_steps()
    print("Exported validated planetary STEP pitch curves:")
    for label, path in paths.items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()
