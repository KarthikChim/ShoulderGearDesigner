"""Regression tests for the connected literature printable-gear pathway."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest
from matplotlib.figure import Figure
from shapely.geometry import Point, Polygon

matplotlib.use("Agg")

from drawing import Renderer
from gui import PATHWAY_OPTIONS
from literature_printable_pair import LiteratureGearPair
from printable_gears import (
    export_featurescript,
    export_printable_gears,
    generate_printable_gears,
)


@pytest.fixture(scope="module")
def printable_pair():
    model_path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "ConsensusShoulderModel.json"
    )
    return generate_printable_gears(model_path)


@pytest.fixture(scope="module")
def moving_pair(printable_pair):
    return LiteratureGearPair(printable_pair.prototype)


def test_final_polygons_are_single_valid_closed_components(printable_pair):
    for blank in (
        printable_pair.prototype.input_blank,
        printable_pair.prototype.output_blank,
    ):
        assert blank.polygon.geom_type == "Polygon"
        assert blank.polygon.is_valid
        assert blank.polygon.exterior.is_ring


def test_every_tooth_is_materially_embedded_in_body(printable_pair):
    config = printable_pair.prototype.config
    for blank, teeth in (
        (printable_pair.prototype.input_blank, printable_pair.prototype.input_teeth),
        (printable_pair.prototype.output_blank, printable_pair.prototype.output_teeth),
    ):
        for tooth in teeth:
            assert blank.polygon.intersection(Polygon(tooth)).area > 0.0
    assert (
        printable_pair.prototype.validation.minimum_tooth_connection_width_mm
        >= config.minimum_tooth_connection_width_mm
    )


def test_bore_exists_and_is_centered(printable_pair):
    bore = printable_pair.parameters.bore_radius_mm
    for blank in (
        printable_pair.prototype.input_blank,
        printable_pair.prototype.output_blank,
    ):
        assert not blank.polygon.contains(Point(0.0, 0.0))
        assert any(
            abs(Point(0.0, 0.0).distance(ring) - bore) < 0.15
            for ring in blank.polygon.interiors
        )


def test_cadquery_solids_are_valid_and_manifold(printable_pair):
    for gear in (printable_pair.input_gear, printable_pair.output_gear):
        report = gear.validation
        assert report.passed
        assert report.valid
        assert report.solid_count == 1
        assert report.volume_mm3 > 0.0
        assert report.bore_through


def test_pathway_is_explicit_and_does_not_call_legacy(moving_pair):
    assert "Literature printable gears" in PATHWAY_OPTIONS
    assert moving_pair.pathway_name == "literature_printable_gears"
    assert moving_pair.source == "McClure2001"


def test_animation_is_reversible_and_never_extrapolates(moving_pair):
    low, high = moving_pair.valid_range_deg
    forward = moving_pair.render_state_at(80.0)
    moving_pair.render_state_at(120.0)
    reverse = moving_pair.render_state_at(80.0)
    assert forward.input_polygon.equals_exact(reverse.input_polygon, 1e-9)
    assert forward.output_polygon.equals_exact(reverse.output_polygon, 1e-9)
    assert moving_pair.input_angle_at(high) > moving_pair.input_angle_at(low)
    assert moving_pair.output_angle_at(high) < moving_pair.output_angle_at(low)
    with pytest.raises(ValueError):
        moving_pair.render_state_at(low - 0.01)
    with pytest.raises(ValueError):
        moving_pair.render_state_at(high + 0.01)


def test_no_tooth_polygon_penetration_at_2001_positions(moving_pair):
    low, high = moving_pair.valid_range_deg
    collision_areas = np.array(
        [
            moving_pair.render_state_at(elevation).collision_area
            for elevation in np.linspace(low, high, 2001)
        ]
    )
    assert np.max(collision_areas) <= 1e-7


def test_gui_renderer_draws_complete_connected_bodies(moving_pair):
    figure = Figure()
    axes = figure.add_subplot(111)
    renderer = Renderer(axes)
    state = renderer.draw_literature_pair(moving_pair, 80.0)
    assert state.input_polygon.is_valid
    assert state.output_polygon.is_valid
    assert len(axes.patches) >= 2
    assert "NOT FOR HUMAN OR POWERED USE" in axes.get_title()


def test_exported_stl_step_and_featurescript_are_nonempty(
    printable_pair, tmp_path
):
    outputs = export_printable_gears(printable_pair, tmp_path)
    for path in outputs:
        assert path.exists()
        assert path.stat().st_size > 100
    script = export_featurescript(
        printable_pair, tmp_path / "LiteratureSectorGearPair.fs"
    )
    text = script.read_text(encoding="utf-8")
    assert "FeatureScript 3029;" in text
    assert "INPUT_OUTLINE" in text
    assert "OUTPUT_OUTLINE" in text
    assert "NOT FOR HUMAN OR POWERED USE" in text
    assert "straightBody" in text
    assert "const inputBody = straightBody" in text
    assert "function straightBody" in text
    assert "export function straightBody" not in text
    assert text.count("fCylinder") == 2
    assert "opCylinder" not in text
    assert text.count("BooleanOperationType.SUBTRACTION") == 2
    assert "CENTER_DISTANCE_BOUNDS" in text
    assert "Whole-profile lofting is intentionally not used" in text
