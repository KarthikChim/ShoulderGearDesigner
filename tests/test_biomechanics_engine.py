"""Literature database, consensus, provenance, and export tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from biomechanics.engine import BiomechanicsEngine


SOURCE = Path("/Users/karth/Downloads/HumanShoulderGroundTruth_v2.csv")


@pytest.fixture(scope="module")
def engine() -> BiomechanicsEngine:
    result = BiomechanicsEngine(SOURCE)
    result.build()
    return result


def test_raw_rows_are_preserved_exactly(engine: BiomechanicsEngine) -> None:
    first = engine.raw_rows[0]
    original = first.original_dict()
    assert original["PaperID"] == "Forte2009"
    assert original["HT_Elevation_deg"] == "10"
    assert original["ST_InternalRotation_deg"] == "37.367"
    assert len(engine.raw_rows) == 1181


def test_validation_and_motion_grouping(engine: BiomechanicsEngine) -> None:
    report = engine.validation_report
    assert report.valid
    assert report.paper_count == 4
    assert report.duplicate_row_count == 0
    assert len(engine.motion_datasets) == 9
    keys = {dataset.key.identifier for dataset in engine.motion_datasets}
    assert any("Raising" in key for key in keys)
    assert any("Lowering" in key for key in keys)
    assert any("loaded" in key for key in keys)


def test_normalization_retains_complete_provenance(engine: BiomechanicsEngine) -> None:
    observation = engine.normalized_observations[0]
    assert observation.original_value == 37.367
    assert observation.normalized_value == 37.367
    assert observation.transformation.scale == 1.0
    assert not observation.original_convention.verified
    assert observation.row_number == 2


def test_consensus_is_motion_specific_and_shape_preserving(
    engine: BiomechanicsEngine,
) -> None:
    model = engine.model
    assert model is not None
    assert len(model.datasets) == len(model.splines) == 34
    assert all(len(dataset.source_curves) >= 1 for dataset in model.datasets)
    spline = model.splines[0]
    midpoint = 0.5 * (spline.knots[0] + spline.knots[-1])
    assert np.isfinite(spline.evaluate(midpoint))
    assert np.isfinite(spline.evaluate(midpoint, derivative=1))
    assert np.isfinite(spline.evaluate(midpoint, derivative=2))


def test_json_contains_coefficients_metadata_and_uncertainty(
    engine: BiomechanicsEngine, tmp_path: Path
) -> None:
    path = engine.export(tmp_path / "ConsensusShoulderModel.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["format"] == "ConsensusShoulderModel"
    assert payload["source"]["sha256"] == engine.model.source_sha256
    assert payload["consensus_datasets"][0]["variance_deg2"]
    assert payload["splines"][0]["coefficients"]
    assert payload["splines"][0]["first_derivative_coefficients"]
    assert payload["splines"][0]["second_derivative_coefficients"]
    assert payload["metadata"]["weighting_policy"]
    assert payload["coordinate_conventions"]
