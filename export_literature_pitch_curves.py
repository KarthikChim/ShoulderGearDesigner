"""Export the committed literature pitch curves as toothless CAD solids."""

from pathlib import Path

from pitch_curve_cad_export import export_pitch_curve_solids


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "validation_outputs" / "LiteratureSectorPitchCurves.csv"
OUTPUT = ROOT / "pitch_curve_exports"


def main() -> None:
    input_artifact, output_artifact, paths = export_pitch_curve_solids(
        SOURCE,
        OUTPUT,
        thickness_mm=8.0,
    )
    for artifact in (input_artifact, output_artifact):
        result = artifact.validation
        print(
            f"{artifact.member}: PASS, {result.sample_count} unchanged samples, "
            f"{result.volume_mm3:.3f} mm^3"
        )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()

