"""Tests for the intentionally minimal, toothless literature viewer."""

from pathlib import Path

import numpy as np

from literature_pitch_gui import LiteraturePitchMotion, _rotate


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "validation_outputs" / "LiteratureSectorPitchCurves.csv"


def test_committed_literature_pitch_motion_is_valid() -> None:
    motion = LiteraturePitchMotion.load(SOURCE)
    assert motion.elevation_deg[0] == 11.0
    assert motion.elevation_deg[-1] == 147.0
    assert motion.center_distance_mm == 120.0
    assert np.all(np.isfinite(motion.ratio))
    assert np.all(motion.ratio > 0.0)
    assert np.allclose(
        motion.input_radius_mm + motion.output_radius_mm,
        motion.center_distance_mm,
        atol=1e-8,
    )


def test_operating_contact_is_shared_throughout_motion() -> None:
    motion = LiteraturePitchMotion.load(SOURCE)
    for index in np.linspace(0, len(motion.elevation_deg) - 1, 25, dtype=int):
        input_path = _rotate(
            motion.input_points[[index]], motion.input_angle_rad[index]
        )
        output_path = _rotate(
            motion.output_points[[index]], -motion.output_angle_rad[index]
        )
        output_path[:, 0] += motion.center_distance_mm
        assert np.allclose(input_path[0], output_path[0], atol=1e-7)


def test_viewer_has_no_tooth_generation_dependencies() -> None:
    source = (ROOT / "literature_pitch_gui.py").read_text(encoding="utf-8")
    forbidden_imports = (
        "literature_printable_pair",
        "tooth_geometry",
        "rolling_rack_envelope",
        "gear_mesh_optimizer",
    )
    assert all(name not in source for name in forbidden_imports)
