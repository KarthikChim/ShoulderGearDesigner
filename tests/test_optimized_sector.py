"""Hard-gate tests for optimized literature-sector mechanical geometry."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from biomechanics.literature_model import LiteratureShoulderModel
from literature_sector import (
    LiteratureSectorTransmission,
    SectorDesignConfig,
    generate_sector_teeth,
    synthesize_sector_pitch_curves,
)
from optimized_sector import (
    build_closed_sector_blank,
    compute_tooth_metrics,
    curve_geometry_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "validation_outputs"
MODEL = ROOT / "ConsensusShoulderModel.json"


@pytest.fixture(scope="module")
def regularized_geometry():
    model = LiteratureShoulderModel(MODEL)
    config = SectorDesignConfig(sample_count=2001)
    transmission = LiteratureSectorTransmission(
        model, config, regularized=True
    )
    data = synthesize_sector_pitch_curves(transmission)
    teeth = generate_sector_teeth(data, config)
    input_blank = build_closed_sector_blank(
        data.input_points, teeth.input_teeth, config
    )
    output_blank = build_closed_sector_blank(
        data.output_points, teeth.output_teeth, config
    )
    return model, config, transmission, data, teeth, input_blank, output_blank


def test_regularization_preserves_source_range_endpoint_and_raw_target(
    regularized_geometry,
) -> None:
    model, _, transmission, data, *_ = regularized_geometry
    assert model.selected["contributing_papers"] == ["McClure2001"]
    assert model.valid_range_deg == (11.0, 147.0)
    raw = np.asarray(model.st_angle_at(data.elevation_deg))
    assert transmission.st_excursion(147.0) == pytest.approx(
        raw[-1] - raw[0], abs=1e-9
    )
    assert np.min(transmission.dst_de(data.elevation_deg)) > 0
    with pytest.raises(ValueError, match="extrapolation is forbidden"):
        transmission.st_angle(10.99)


def test_regularized_pitch_curve_derivatives_are_finite(
    regularized_geometry,
) -> None:
    geometry = curve_geometry_metrics(regularized_geometry[3])
    for member in ("input", "output"):
        assert np.all(np.isfinite(geometry[member]["tangent"]))
        assert np.all(np.isfinite(geometry[member]["curvature"]))
        assert np.all(
            np.isfinite(geometry[member]["curvature_derivative"])
        )


def test_complete_sector_blanks_are_closed_valid_with_one_bore(
    regularized_geometry,
) -> None:
    for blank in regularized_geometry[-2:]:
        assert blank.closed
        assert blank.valid
        assert blank.polygon.is_valid
        assert blank.polygon.geom_type == "Polygon"
        assert len(blank.polygon.interiors) == 1
        assert np.allclose(blank.boundary[0], blank.boundary[-1])


def test_tooth_metrics_cover_every_generated_tooth(
    regularized_geometry,
) -> None:
    teeth = regularized_geometry[4]
    metrics = compute_tooth_metrics(teeth)
    assert len(metrics) == (
        teeth.input_tooth_count + teeth.output_tooth_count
    )
    assert all(np.isfinite(item.root_thickness) for item in metrics)
    assert all(np.isfinite(item.tip_thickness) for item in metrics)
    assert all(np.isfinite(item.undercut_margin) for item in metrics)
    assert all(np.isfinite(item.contact_ratio) for item in metrics)


def test_search_covers_required_sector_and_mechanical_ranges() -> None:
    path = OUTPUT / "OptimizedSectorSearch.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rows = [row for row in rows if row.get("candidate_id", "").isdigit()]
    assert len(rows) >= 81
    assert {float(row["sector_angle_deg"]) for row in rows} == {
        120,
        135,
        150,
        165,
        180,
        195,
        210,
        225,
        240,
    }
    assert min(float(row["center_distance_mm"]) for row in rows) == 100
    assert max(float(row["center_distance_mm"]) for row in rows) == 180
    assert min(float(row["module_mm"]) for row in rows) == 1.5
    assert max(float(row["module_mm"]) for row in rows) == 4.0
    assert {float(row["pressure_angle_deg"]) for row in rows} == {20, 25, 30}


def test_full_mesh_and_failure_diagnostics_are_sample_resolved() -> None:
    validation = json.loads(
        (OUTPUT / "OptimizedSectorValidation.json").read_text()
    )
    assert validation["full_mesh_position_count"] == 2001
    diagnostics_path = OUTPUT / "SectorFailureDiagnostics.csv"
    with diagnostics_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rows = [row for row in rows if row.get("sample_index", "").isdigit()]
    assert rows
    required = {
        "sample_index",
        "ht_elevation_deg",
        "input_angle_rad",
        "output_angle_rad",
        "local_ratio",
        "input_radius",
        "output_radius",
        "curvature",
        "tooth_number",
        "geometric_reason",
    }
    assert required.issubset(rows[0])


def test_every_hard_acceptance_gate_is_explicit_and_controls_export() -> None:
    payload = json.loads(
        (OUTPUT / "OptimizedSectorValidation.json").read_text()
    )
    validation = payload["validation"]
    gates = {
        "continuous_tangent",
        "bounded_curvature",
        "bounded_curvature_derivative",
        "adjacent_tooth_overlap_free",
        "mating_interference_free",
        "minimum_pitch_radius_valid",
        "minimum_contact_ratio_valid",
        "minimum_root_thickness_valid",
        "sector_blanks_closed_valid",
        "no_extrapolation",
        "maximum_st_error_valid",
        "rms_st_error_valid",
        "rack_envelope_verified",
    }
    assert gates.issubset(validation)
    assert validation["hard_pass"] == all(validation[key] for key in gates)
    assert payload["prototype_geometry_exported"] == validation["hard_pass"]
    if not validation["hard_pass"]:
        assert not (OUTPUT / "prototype_input_sector.dxf").exists()
        assert not (OUTPUT / "prototype_output_sector.dxf").exists()
        assert not (OUTPUT / "prototype_pair.svg").exists()
    assert validation["decision"] in {
        "NO-GO",
        "GO FOR SOFTWARE SIMULATION",
        "GO FOR UNLOADED HAND-DRIVEN BENCH PROTOTYPE",
    }
    assert validation["decision"] != "GO FOR HUMAN USE"
