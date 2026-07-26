"""Create one two-body STEP for testing Onshape Arbitrary Gearpath."""

from pathlib import Path

from pitch_curve_cad_export import export_dual_operating_pitch_paths


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "validation_outputs" / "LiteratureSectorPitchCurves.csv"
DESTINATION = ROOT / "pitch_curve_exports" / "LiteraturePitchPaths.step"


def main() -> None:
    validation = export_dual_operating_pitch_paths(
        SOURCE,
        DESTINATION,
        thickness_mm=2.0,
        shaft_hole_diameter_mm=4.0,
    )
    print("PASS" if validation.passed else "FAIL")
    print(f"Solids: {validation.solid_count}")
    print(f"Center distance: {validation.center_distance_mm:.9f} mm")
    print(f"Body contact distance: {validation.body_distance_mm:.9f} mm")
    print(DESTINATION)
    if not validation.passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

