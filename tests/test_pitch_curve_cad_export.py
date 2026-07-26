from pathlib import Path

import cadquery as cq
import ezdxf
import numpy as np
import pytest

from pitch_curve_cad_export import (
    export_pitch_curve_solids,
    load_committed_pitch_curves,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "validation_outputs" / "LiteratureSectorPitchCurves.csv"


def test_loads_only_committed_regularized_pitch_arrays():
    input_points, output_points, center, error = load_committed_pitch_curves(SOURCE)
    assert len(input_points) == 4001
    assert len(output_points) == 4001
    assert center == 120.0
    assert error < 1e-10
    assert not np.array_equal(input_points[0], input_points[-1])
    assert not np.array_equal(output_points[0], output_points[-1])


def test_exports_toothless_valid_step_dxf_and_csv(tmp_path):
    input_artifact, output_artifact, paths = export_pitch_curve_solids(
        SOURCE, tmp_path
    )
    assert input_artifact.validation.passed
    assert output_artifact.validation.passed
    assert len(paths) == 7
    for stem in ("InputPitchCurve", "OutputPitchCurve"):
        step = tmp_path / f"{stem}.step"
        imported = cq.importers.importStep(str(step)).val()
        assert imported.isValid()
        assert len(imported.Solids()) == 1
        assert imported.BoundingBox().zlen == pytest.approx(8.0, abs=1e-6)
        dxf = ezdxf.readfile(tmp_path / f"{stem}.dxf")
        entities = list(dxf.modelspace())
        assert [entity.dxftype() for entity in entities] == ["SPLINE", "LINE"]
        csv_path = tmp_path / f"{stem}.csv"
        assert csv_path.read_text().splitlines()[0] == (
            "ArcLength_mm,X_mm,Y_mm,TangentAngle_deg,Curvature"
        )


def test_source_samples_are_bitwise_unchanged_after_export(tmp_path):
    before = load_committed_pitch_curves(SOURCE)
    input_artifact, output_artifact, _ = export_pitch_curve_solids(SOURCE, tmp_path)
    assert np.array_equal(before[0], input_artifact.source_points)
    assert np.array_equal(before[1], output_artifact.source_points)
