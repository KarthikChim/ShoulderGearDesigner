"""Export the stationary-sun planetary pathway as wire geometry only."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from literature_planetary import (
    rotate_points,
    synthesize_literature_planetary_pitch_curves,
    validate_planetary_pitch_curves,
    write_planetary_report,
    write_planetary_validation_json,
)


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "validation_outputs" / "LiteratureSectorTransmission.csv"
OUTPUT = ROOT / "planetary_pitch_exports"


def export_csv(data, destination: str | Path) -> Path:
    destination = Path(destination)
    columns = (
        "elevation_deg",
        "carrier_angle_rad",
        "sun_angle_rad",
        "planet_absolute_angle_rad",
        "planet_relative_angle_rad",
        "dcarrier_dE",
        "dplanet_absolute_dE",
        "dplanet_relative_dE",
        "signed_ratio",
        "sun_pitch_radius_mm",
        "planet_pitch_radius_mm",
        "sun_pitch_x_local_mm",
        "sun_pitch_y_local_mm",
        "planet_pitch_x_local_mm",
        "planet_pitch_y_local_mm",
        "planet_center_x_world_mm",
        "planet_center_y_world_mm",
        "contact_x_world_mm",
        "contact_y_world_mm",
        "center_distance_mm",
    )
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for i in range(len(data.elevation_deg)):
            writer.writerow(
                (
                    data.elevation_deg[i],
                    data.carrier_angle_rad[i],
                    data.sun_angle_rad[i],
                    data.planet_absolute_angle_rad[i],
                    data.planet_relative_angle_rad[i],
                    data.dcarrier_dE[i],
                    data.dplanet_absolute_dE[i],
                    data.dplanet_relative_dE[i],
                    data.signed_ratio[i],
                    data.sun_pitch_radius_mm[i],
                    data.planet_pitch_radius_mm[i],
                    *data.sun_pitch_points_local[i],
                    *data.planet_pitch_points_local[i],
                    *data.planet_center_points_world[i],
                    *data.contact_points_world[i],
                    data.center_distance_mm,
                )
            )
    return destination


def export_wire_step(data, destination: str | Path) -> Path:
    """Write four independent STEP edges: two splines and two circles."""
    import cadquery as cq

    destination = Path(destination)
    start = 0
    planet_start = rotate_points(
        data.planet_pitch_points_local,
        data.planet_absolute_angle_rad[start],
    ) + data.planet_center_points_world[start]

    sun_edge = cq.Edge.makeSpline(
        [cq.Vector(float(x), float(y), 0.0) for x, y in data.sun_pitch_points_local]
    )
    planet_edge = cq.Edge.makeSpline(
        [cq.Vector(float(x), float(y), 0.0) for x, y in planet_start]
    )
    sun_reference = cq.Edge.makeCircle(
        2.0, cq.Vector(0.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)
    )
    planet_center = data.planet_center_points_world[start]
    planet_reference = cq.Edge.makeCircle(
        2.0,
        cq.Vector(float(planet_center[0]), float(planet_center[1]), 0.0),
        cq.Vector(0.0, 0.0, 1.0),
    )
    compound = cq.Compound.makeCompound(
        [sun_edge, planet_edge, sun_reference, planet_reference]
    )
    cq.exporters.export(compound, str(destination), exportType="STEP")
    return destination


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = synthesize_literature_planetary_pitch_curves(SOURCE)
    validation = validate_planetary_pitch_curves(data)
    if not validation.passed:
        raise RuntimeError(f"Planetary pitch-curve validation failed: {validation}")

    export_csv(data, OUTPUT / "LiteraturePlanetaryPitchCurves.csv")
    export_wire_step(data, OUTPUT / "LiteraturePlanetaryPitchPaths_WireOnly.step")
    write_planetary_report(
        validation, OUTPUT / "LiteraturePlanetaryPitchCurveReport.md"
    )
    write_planetary_validation_json(
        validation, OUTPUT / "LiteraturePlanetaryPitchCurveValidation.json"
    )
    print(f"Exported validated planetary pitch paths to {OUTPUT}")


if __name__ == "__main__":
    main()
