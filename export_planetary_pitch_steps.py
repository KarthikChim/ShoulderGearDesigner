"""Export the validated planetary pitch curves as separate STEP wire files.

Run from PyCharm or a terminal with:

    python export_planetary_pitch_steps.py

Generated files contain spline edges only. They contain no teeth, faces,
solids, bores, hubs, spokes, carrier bars, or reference circles.
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


def _export_edge(edge, destination: Path) -> None:
    """Write a single edge as STEP geometry."""
    import cadquery as cq

    compound = cq.Compound.makeCompound([edge])
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
    _export_edge(sun_edge, paths["sun"])
    _export_edge(planet_local_edge, paths["planet_local"])
    _export_edge(planet_assembled_edge, paths["planet_assembled"])

    pair = cq.Compound.makeCompound([sun_edge, planet_assembled_edge])
    cq.exporters.export(pair, str(paths["assembled_pair"]), exportType="STEP")

    return paths


def main() -> None:
    paths = export_planetary_pitch_steps()
    print("Exported validated planetary STEP pitch curves:")
    for label, path in paths.items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()
