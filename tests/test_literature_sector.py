"""Acceptance tests for non-wrapping McClure literature sector synthesis."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from biomechanics.literature_model import LiteratureShoulderModel
from literature_sector import (
    LiteratureSectorTransmission,
    SectorDesignConfig,
    generate_sector_teeth,
    synthesize_sector_pitch_curves,
    validate_sector,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "ConsensusShoulderModel.json"


@pytest.fixture(scope="module")
def sector_pair():
    model = LiteratureShoulderModel(MODEL_PATH)
    config = SectorDesignConfig(sample_count=2001)
    raw = LiteratureSectorTransmission(model, config, regularized=False)
    regularized = LiteratureSectorTransmission(
        model, config, regularized=True
    )
    raw_data = synthesize_sector_pitch_curves(raw)
    reg_data = synthesize_sector_pitch_curves(regularized)
    raw_teeth = generate_sector_teeth(raw_data, config)
    reg_teeth = generate_sector_teeth(reg_data, config)
    return (
        model,
        config,
        raw,
        regularized,
        raw_data,
        reg_data,
        raw_teeth,
        reg_teeth,
    )


def test_loads_committed_mcclure_condition_and_exact_range(sector_pair) -> None:
    model, _, raw, *_ = sector_pair
    assert raw.model.model_path.resolve() == MODEL_PATH.resolve()
    assert model.selected["contributing_papers"] == ["McClure2001"]
    assert (
        model.selected["condition"]["condition_id"]
        == "healthy_unloaded_raising_scapular_plane"
    )
    assert raw.valid_range_deg == (11.0, 147.0)


def test_literature_sector_does_not_use_legacy_ratio_schedule(
    monkeypatch,
) -> None:
    import settings

    monkeypatch.setattr(
        settings,
        "default_ratio_regions",
        lambda: (_ for _ in ()).throw(AssertionError("legacy path called")),
    )
    model = LiteratureShoulderModel(MODEL_PATH)
    transmission = LiteratureSectorTransmission(model)
    assert transmission.candidate == "raw"


def test_excursion_endpoints_and_no_extrapolation(sector_pair) -> None:
    model, _, raw, regularized, *_ = sector_pair
    assert raw.st_excursion(11.0) == pytest.approx(0.0, abs=1e-12)
    expected = model.st_angle_at(147.0) - model.st_angle_at(11.0)
    assert raw.st_excursion(147.0) == pytest.approx(expected, abs=1e-12)
    assert regularized.st_excursion(147.0) == pytest.approx(
        expected, abs=1e-9
    )
    for transmission in (raw, regularized):
        with pytest.raises(ValueError, match="extrapolation is forbidden"):
            transmission.input_angle(10.999)
        with pytest.raises(ValueError, match="extrapolation is forbidden"):
            transmission.output_angle(147.001)


@pytest.mark.parametrize("regularized", [False, True])
def test_analytical_ratio_matches_finite_difference(
    sector_pair, regularized
) -> None:
    transmission = sector_pair[3] if regularized else sector_pair[2]
    # Avoid evaluating exactly at PCHIP knots, where the second derivative is
    # piecewise but the analytical first derivative remains well defined.
    elevation = np.linspace(12.013, 145.987, 500)
    step = 1e-5
    psi_low = np.asarray(transmission.output_angle(elevation - step))
    psi_high = np.asarray(transmission.output_angle(elevation + step))
    phi_low = np.asarray(transmission.input_angle(elevation - step))
    phi_high = np.asarray(transmission.input_angle(elevation + step))
    finite = (psi_high - psi_low) / (phi_high - phi_low)
    analytical = np.asarray(transmission.ratio(elevation))
    assert np.max(np.abs(finite - analytical)) < 2e-6


def test_raw_target_unchanged_and_regularization_separate(sector_pair) -> None:
    model, _, raw, regularized, raw_data, reg_data, *_ = sector_pair
    expected = np.asarray(model.st_angle_at(raw_data.elevation_deg))
    assert np.array_equal(raw_data.absolute_st_deg, expected)
    assert raw.regularized is False
    assert regularized.regularized is True
    assert raw.candidate == "raw"
    assert regularized.candidate == "regularized"
    assert np.max(np.abs(reg_data.absolute_st_deg - expected)) > 0
    assert reg_data.st_excursion_deg[-1] == pytest.approx(
        raw_data.st_excursion_deg[-1], abs=1e-9
    )


def test_fixed_center_distance_and_open_nonwrapping_curves(sector_pair) -> None:
    for data in (sector_pair[4], sector_pair[5]):
        assert np.max(
            np.abs(
                data.input_radii
                + data.output_radii
                - data.center_distance
            )
        ) < 1e-10
        assert data.wraps_or_closes is False
        assert not np.allclose(data.input_points[0], data.input_points[-1])
        assert data.input_rad[-1] - data.input_rad[0] < 2 * np.pi
        assert data.hard_stop_elevation_deg == (11.0, 147.0)


def test_teeth_exist_only_on_active_open_sectors(sector_pair) -> None:
    config = sector_pair[1]
    for data, teeth in (
        (sector_pair[4], sector_pair[6]),
        (sector_pair[5], sector_pair[7]),
    ):
        assert teeth.input_tooth_count > 0
        assert teeth.output_tooth_count > 0
        input_length = np.sum(
            np.linalg.norm(np.diff(data.input_points, axis=0), axis=1)
        )
        output_length = np.sum(
            np.linalg.norm(np.diff(data.output_points, axis=0), axis=1)
        )
        assert np.all(teeth.input_arc_positions >= 0)
        assert np.all(teeth.input_arc_positions <= input_length)
        assert np.all(teeth.output_arc_positions >= 0)
        assert np.all(teeth.output_arc_positions <= output_length)
        assert data.transition_input_bounds_rad[0] < data.active_input_bounds_rad[0]
        assert data.transition_input_bounds_rad[1] > data.active_input_bounds_rad[1]


def test_regularized_candidate_is_positive_and_validated(sector_pair) -> None:
    transmission, data, teeth = sector_pair[3], sector_pair[5], sector_pair[7]
    validation = validate_sector(transmission, data, teeth)
    assert np.min(data.ratio) >= transmission.config.minimum_ratio - 1e-10
    assert validation.finite_positive_ratio
    assert validation.constant_center_distance
    assert validation.minimum_radius_valid
    assert validation.no_extrapolation
    assert validation.no_wrapping
    assert validation.decision == "GO FOR SOFTWARE SIMULATION"
    assert validation.decision != "GO FOR HUMAN USE"


def test_target_realized_errors_are_exported() -> None:
    path = ROOT / "validation_outputs" / "LiteratureSectorTransmission.csv"
    assert path.exists() and path.stat().st_size > 1000
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    assert rows[0][0] == "RESEARCH BENCH PROTOTYPE — NOT FOR HUMAN USE"
    assert "absolute_st_deg" in rows[1]
    assert "candidate" in rows[1]
    validation = (
        ROOT / "validation_outputs" / "LiteratureSectorValidation.json"
    ).read_text(encoding="utf-8")
    assert '"maximum_st_error_deg"' in validation
    assert '"rms_st_error_deg"' in validation
