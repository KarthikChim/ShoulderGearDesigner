"""Acceptance tests for the gated literature-based synthesis pathway."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from biomechanics.engine import BiomechanicsEngine
from biomechanics.literature_model import LiteratureShoulderModel
from biomechanics.normalization import sem_to_sd
from literature_transmission import (
    SectorTransmission,
    compare_transmission_alternatives,
)
from model_pathways import create_shoulder_model
from noncircular import SmoothTransmission
from settings import Settings


SOURCE = Path("biomechanics/data/HumanShoulderGroundTruth_v2.csv")


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    directory = tmp_path_factory.mktemp("literature")
    engine = BiomechanicsEngine(SOURCE)
    engine.build()
    path = engine.export(directory / "ConsensusShoulderModel.json")
    return engine, path


def test_consensus_json_is_nonempty_valid_and_reproducible(tmp_path) -> None:
    first = BiomechanicsEngine(SOURCE)
    first.build()
    first_path = first.export(tmp_path / "first.json")
    second = BiomechanicsEngine(SOURCE)
    second.build()
    second_path = second.export(tmp_path / "second.json")
    assert first_path.stat().st_size > 1000
    assert first_path.read_bytes() == second_path.read_bytes()
    payload = json.loads(first_path.read_text())
    assert payload["validation_valid"]
    assert payload["splines"]
    assert payload["selected_design"]["conventions_verified"]


def test_cross_paper_consensus_requires_verified_conventions(generated) -> None:
    engine, _ = generated
    for dataset in engine.consensus_datasets:
        papers = {curve.paper_id for curve in dataset.source_curves}
        if len(papers) > 1:
            assert dataset.conventions_verified


def test_design_condition_never_mixes_protocols(generated) -> None:
    _, path = generated
    selected = json.loads(path.read_text())["selected_design"]
    motion = selected["motion"]
    assert motion["healthy_only"] is True
    assert motion["loaded"] is False
    assert motion["direction"] == "Raising"
    assert motion["motion_type"] == "Dynamic scapular-plane abduction"
    assert motion["motion_plane"] == "40 degrees anterior to frontal"
    assert selected["contributing_papers"] == ["McClure2001"]


def test_sem_is_converted_to_sd(generated) -> None:
    assert sem_to_sd(2.0, 16) == pytest.approx(8.0)
    with pytest.raises(ValueError):
        sem_to_sd(2.0, 0)


def test_literature_adapter_reproduces_knots_and_forbids_extrapolation(
    generated,
) -> None:
    _, path = generated
    model = LiteratureShoulderModel(path)
    payload = json.loads(path.read_text())["selected_design"]
    knots = np.asarray(payload["spline"]["knots_deg"])
    expected = np.interp(knots, payload["elevation_deg"], payload["mean_deg"])
    assert np.allclose(model.st_angle_at(knots), expected, atol=1e-10)
    with pytest.raises(ValueError, match="extrapolation is forbidden"):
        model.st_angle_at(model.valid_range_deg[0] - 0.1)
    with pytest.raises(RuntimeError, match="not formally identical"):
        model.gh_angle_at(60.0)


def test_sector_derivative_positive_and_center_distance_preserved(generated) -> None:
    _, path = generated
    model = LiteratureShoulderModel(path)
    comparison = compare_transmission_alternatives(model)
    assert np.min(comparison.sector_ratio) > 0
    assert np.allclose(
        comparison.sector_input_radius + comparison.sector_output_radius,
        100.0,
        atol=1e-10,
    )
    # The closed alternative is retained for comparison and correctly rejected.
    assert np.min(comparison.full_cycle_ratio) < 0


def test_error_thresholds_are_configurable_and_reported(generated) -> None:
    _, path = generated
    comparison = compare_transmission_alternatives(LiteratureShoulderModel(path))
    assert comparison.sector_max_error_deg < 1e-10
    assert comparison.sector_rms_error_deg < 1e-10
    assert comparison.full_cycle_max_error_deg > 0


def test_leave_one_out_status_is_explicit_for_single_study(generated) -> None:
    _, path = generated
    analyses = json.loads(path.read_text())["sensitivity_analyses"]
    assert analyses["leave_one_study_out"]
    assert analyses["leave_one_study_out"][0]["status"] == (
        "not_estimable_single_study_condition"
    )


def test_legacy_and_literature_pathways_are_independent(generated) -> None:
    _, path = generated
    legacy = create_shoulder_model("legacy", Settings())
    literature = create_shoulder_model("literature", consensus_json=path)
    assert legacy.pathway_name == "legacy_piecewise"
    assert literature.pathway_name == "literature"
    assert legacy.valid_range_deg == (0.0, 180.0)
    assert literature.valid_range_deg == (11.0, 147.0)
    legacy_transmission = SmoothTransmission(legacy)
    assert np.min(legacy_transmission.ratio(np.linspace(0, 2 * np.pi, 1000))) > 0
    literature_full_cycle = SmoothTransmission(literature)
    assert np.min(
        literature_full_cycle.ratio(np.linspace(0, 2 * np.pi, 1000))
    ) < 0
