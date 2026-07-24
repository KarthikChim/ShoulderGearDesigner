"""Practical gates for the unloaded hand-driven literature prototype."""

from __future__ import annotations

import json
from pathlib import Path

import ezdxf
import pytest

from bench_prototype import BenchPrototypeConfig, build_bench_prototype
from biomechanics.literature_model import LiteratureShoulderModel


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "ConsensusShoulderModel.json"
OUTPUT = ROOT / "validation_outputs"


@pytest.fixture(scope="module")
def prototype():
    return build_bench_prototype(
        LiteratureShoulderModel(MODEL),
        BenchPrototypeConfig(mesh_positions=2001),
    )


def test_uses_verified_mcclure_regularized_range(prototype) -> None:
    assert prototype.transmission.model.selected["contributing_papers"] == [
        "McClure2001"
    ]
    assert prototype.transmission.valid_range_deg == (11.0, 147.0)
    assert prototype.transmission.regularized
    assert prototype.validation.no_extrapolation


def test_biomechanical_error_and_ratio_gates(prototype) -> None:
    validation = prototype.validation
    assert validation.maximum_st_error_deg <= 3.0
    assert validation.rms_st_error_deg <= 2.0
    assert abs(validation.endpoint_error_deg) <= 0.5
    assert validation.minimum_ratio > 0
    assert validation.minimum_pitch_radius_mm >= 8.0


def test_full_2001_position_mesh_clears(prototype) -> None:
    validation = prototype.validation
    assert len(prototype.mesh_positions) == 2001
    assert validation.maximum_tooth_penetration_area_mm2 == pytest.approx(0.0)
    assert validation.minimum_noncontact_clearance_mm >= 0.25
    assert validation.no_tooth_skipping
    assert validation.continuous_hand_rotation


def test_printable_tooth_and_body_gates(prototype) -> None:
    validation = prototype.validation
    assert validation.adjacent_teeth_overlap_free
    assert validation.closed_valid_bodies
    assert prototype.input_blank.valid and prototype.input_blank.closed
    assert prototype.output_blank.valid and prototype.output_blank.closed
    assert validation.minimum_root_thickness_mm >= 1.5
    assert validation.minimum_tip_thickness_mm >= 0.8


def test_practical_decision_is_hand_driven_only(prototype) -> None:
    assert prototype.validation.all_practical_gates_pass
    assert (
        prototype.validation.decision
        == "GO FOR UNLOADED HAND-DRIVEN BENCH PROTOTYPE"
    )
    assert "HUMAN" not in prototype.validation.decision


def test_committed_validation_matches_live_result(prototype) -> None:
    payload = json.loads(
        (OUTPUT / "BenchPrototype_Validation.json").read_text()
    )
    assert payload["source"] == "McClure2001"
    assert payload["valid_range_deg"] == [11.0, 147.0]
    assert payload["mesh_position_count"] == 2001
    assert payload["validation"]["all_practical_gates_pass"] is True
    assert payload["validation"]["decision"] == prototype.validation.decision
    assert len(payload["search_results"]) == 12


@pytest.mark.parametrize(
    "filename",
    ["BenchPrototype_InputSector.dxf", "BenchPrototype_OutputSector.dxf"],
)
def test_exported_dxf_has_closed_body_bore_and_warning(filename) -> None:
    document = ezdxf.readfile(OUTPUT / filename)
    entities = list(document.modelspace())
    layers = {entity.dxf.layer for entity in entities}
    assert "CLOSED_SECTOR_BODY" in layers
    assert "SHAFT_BORE_PLACEHOLDER" in layers
    assert "RESEARCH_ONLY_WARNING" in layers
    bodies = [
        entity
        for entity in entities
        if entity.dxf.layer == "CLOSED_SECTOR_BODY"
    ]
    assert len(bodies) == 1
    assert bodies[0].closed


def test_svg_and_print_guide_are_explicitly_research_only() -> None:
    svg = (OUTPUT / "BenchPrototype_Pair.svg").read_text()
    guide = (OUTPUT / "BenchPrototype_PrintGuide.md").read_text()
    assert "NOT FOR HUMAN OR POWERED USE" in svg
    assert "NOT FOR HUMAN OR POWERED USE" in guide
    for requirement in (
        "PLA or PETG",
        "turning slowly by hand",
        "adjustable center-distance slots",
        "0.3–0.5 mm extra physical backlash",
        "removable shaft hubs",
        "Do not attach motors",
        "Do not attach the mechanism to a body",
    ):
        assert requirement in guide
