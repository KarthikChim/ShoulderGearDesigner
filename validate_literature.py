"""Rebuild literature model and generate non-manufacturing validation outputs."""

import json
from pathlib import Path

import numpy as np

from biomechanics.audit import write_coordinate_audit
from biomechanics.engine import BiomechanicsEngine
from biomechanics.engineering_report import generate_engineering_report
from biomechanics.literature_model import LiteratureShoulderModel


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
    # Keep a small, reviewable validation artifact beside the full model.  It
    # proves that the exact JSON written above can be loaded by the production
    # adapter and evaluated throughout (but never outside) its support.
    payload = json.loads(consensus.read_text(encoding="utf-8"))
    selected = payload["selected_design"]
    literature_model = LiteratureShoulderModel(consensus)
    valid_min, valid_max = literature_model.valid_range_deg
    evaluation_grid = np.linspace(valid_min, valid_max, 1001)
    st_values = np.asarray(literature_model.st_angle_at(evaluation_grid))
    dst_values = np.asarray(literature_model.dst_delevation_at(evaluation_grid))
    outside_range_rejected = False
    try:
        literature_model.st_angle_at(valid_min - 0.001)
    except ValueError:
        outside_range_rejected = True
    validation_output = {
        "validation_valid": payload["validation_valid"],
        "conventions_verified": payload["conventions_verified"],
        "selected_condition_id": selected["condition"]["condition_id"],
        "selected_source": selected["contributing_papers"],
        "valid_range_deg": list(literature_model.valid_range_deg),
        "evaluation_sample_count": int(evaluation_grid.size),
        "all_st_angles_finite": bool(np.all(np.isfinite(st_values))),
        "all_dst_de_finite": bool(np.all(np.isfinite(dst_values))),
        "extrapolated_point_count": selected["extrapolated_point_count"],
        "outside_range_rejected": outside_range_rejected,
        "consensus_json_bytes": consensus.stat().st_size,
        "engineering_decision": "NO-GO for manufacturing or human use",
    }
    validation_path = (
        root / "validation_outputs" / "LiteraturePipelineValidation.json"
    )
    validation_path.write_text(
        json.dumps(validation_output, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"Consensus: {consensus}")
    print(f"Audit: {root / 'CoordinateConventionAudit.md'}")
    print(f"Report: {report}")
    print(f"Plot: {plot}")
    print(f"Validation: {validation_path}")
    print("Engineering decision: NO-GO for manufacturing or human use")


if __name__ == "__main__":
    main()
