"""Tests that the original fixed-axis literature pathway remains available."""

from pathlib import Path

import numpy as np

from literature_pitch_gui import FIXED_MODE, PLANETARY_MODE, FixedAxisPitchMotion


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "validation_outputs" / "LiteratureSectorPitchCurves.csv"


def test_original_fixed_axis_pitch_motion_is_unchanged() -> None:
    motion = FixedAxisPitchMotion.load(SOURCE)
    assert motion.elevation_deg[0] == 11.0
    assert motion.elevation_deg[-1] == 147.0
    assert motion.center_distance_mm == 120.0
    assert np.all(np.isfinite(motion.ratio))
    assert np.all(motion.ratio > 0.0)


def test_both_gui_pathways_are_explicit() -> None:
    assert FIXED_MODE == "Literature fixed-axis pitch curves"
    assert PLANETARY_MODE == "Literature planetary pitch curves"


def test_viewer_does_not_import_tooth_generation() -> None:
    source = (ROOT / "literature_pitch_gui.py").read_text(encoding="utf-8")
    forbidden = (
        "tooth_geometry",
        "rolling_envelope",
        "gear_mesh_optimizer",
        "literature_printable_pair",
    )
    assert all(name not in source for name in forbidden)
