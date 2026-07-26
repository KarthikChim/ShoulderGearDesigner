from pathlib import Path

import cadquery as cq
import pytest

from pitch_curve_cad_export import export_dual_operating_pitch_paths


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "validation_outputs" / "LiteratureSectorPitchCurves.csv"


def test_dual_operating_step_has_two_solids_two_holes_and_true_centers(tmp_path):
    destination = tmp_path / "LiteraturePitchPaths.step"
    result = export_dual_operating_pitch_paths(SOURCE, destination)
    assert result.passed
    assert result.solid_count == 2
    assert result.separate_bodies
    assert result.input_hole_clear
    assert result.output_hole_clear
    assert result.input_axis == (0.0, 0.0)
    assert result.output_axis == (120.0, 0.0)
    assert result.center_distance_mm == 120.0
    assert result.body_distance_mm == pytest.approx(0.0, abs=1e-5)
    imported = cq.importers.importStep(str(destination)).val()
    assert len(imported.Solids()) == 2
    assert all(solid.isValid() for solid in imported.Solids())
    assert all(
        solid.BoundingBox().zlen == pytest.approx(2.0, abs=1e-6)
        for solid in imported.Solids()
    )
