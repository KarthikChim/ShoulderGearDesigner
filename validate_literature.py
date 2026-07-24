"""Rebuild literature model and generate non-manufacturing validation outputs."""

from pathlib import Path

from biomechanics.audit import write_coordinate_audit
from biomechanics.engine import BiomechanicsEngine
from biomechanics.engineering_report import generate_engineering_report


def main() -> None:
    root = Path(__file__).resolve().parent
    source = root / "biomechanics" / "data" / "HumanShoulderGroundTruth_v2.csv"
    consensus = root / "ConsensusShoulderModel.json"
    engine = BiomechanicsEngine(source)
    engine.build()
    engine.export(consensus)
    write_coordinate_audit(
        engine.normalized_observations, root / "CoordinateConventionAudit.md"
    )
    report, plot = generate_engineering_report(
        consensus,
        root / "validation_outputs",
        root / "biomechanics" / "config" / "transmission_acceptance.json",
    )
    print(f"Consensus: {consensus}")
    print(f"Audit: {root / 'CoordinateConventionAudit.md'}")
    print(f"Report: {report}")
    print(f"Plot: {plot}")
    print("Engineering decision: NO-GO for manufacturing or human use")


if __name__ == "__main__":
    main()
