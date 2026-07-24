"""Acceptance tests for the exact literature artifact committed to the repo."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from biomechanics.literature_model import LiteratureShoulderModel
from model_pathways import create_shoulder_model
from settings import Settings


ROOT = Path(__file__).resolve().parents[1]
COMMITTED_MODEL = ROOT / "ConsensusShoulderModel.json"


def test_committed_consensus_artifact_is_complete_and_loadable() -> None:
    assert COMMITTED_MODEL.stat().st_size > 1000
    payload = json.loads(COMMITTED_MODEL.read_text(encoding="utf-8"))
    assert payload["validation_valid"] is True
    assert payload["conventions_verified"] is True

    selected = payload["selected_design"]
    assert selected
    assert selected["conventions_verified"] is True
    assert selected["contributing_papers"] == ["McClure2001"]
    assert selected["condition"]["condition_id"]
    assert selected["valid_range_deg"] == [11.0, 147.0]
    assert selected["spline"]["knots_deg"]
    assert selected["spline"]["coefficients"]
    assert selected["elevation_deg"]
    assert selected["confidence_lower_deg"]
    assert selected["confidence_upper_deg"]
    assert selected["study_contribution"]["McClure2001"]
    assert selected["source_rows"]
    assert selected["gh_decomposition"]["warning"]
    assert selected["extrapolation"] == "forbidden"
    assert selected["extrapolated_point_count"] == 0
    assert selected["all_points_within_supported_range"] is True

    model = LiteratureShoulderModel(COMMITTED_MODEL)
    elevations = np.linspace(*model.valid_range_deg, 1001)
    assert np.all(np.isfinite(model.st_angle_at(elevations)))
    assert np.all(np.isfinite(model.dst_delevation_at(elevations)))

    with pytest.raises(ValueError, match="extrapolation is forbidden"):
        model.st_angle_at(model.valid_range_deg[0] - 0.001)
    with pytest.raises(ValueError, match="extrapolation is forbidden"):
        model.st_angle_at(model.valid_range_deg[1] + 0.001)


def test_committed_literature_and_legacy_pathways_remain_separate() -> None:
    legacy = create_shoulder_model("legacy", Settings())
    literature = create_shoulder_model(
        "literature", consensus_json=COMMITTED_MODEL
    )
    assert legacy.pathway_name == "legacy_piecewise"
    assert literature.pathway_name == "literature"
    assert legacy.valid_range_deg == (0.0, 180.0)
    assert literature.valid_range_deg == (11.0, 147.0)
    assert type(legacy) is not type(literature)
