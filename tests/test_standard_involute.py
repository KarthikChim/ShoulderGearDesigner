import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from standard_involute import (
    StandardGearParameters,
    export_pair,
    generate_involute_tooth,
    generate_rack,
    generate_rack_profile,
    generate_rack_tooth,
    involute_xy,
)


def test_analytical_involute_satisfies_radius_identity():
    rb = 20.0
    t = np.linspace(0.0, 1.2, 100)
    points = involute_xy(rb, t)
    assert np.allclose(
        np.linalg.norm(points, axis=1),
        rb * np.sqrt(1.0 + t * t),
        atol=1e-12,
    )


def test_rack_pitch_thickness_and_flank_angles_are_standard():
    parameters = StandardGearParameters()
    tooth = generate_rack_tooth(parameters)
    thickness = tooth.right_pitch_point[0] - tooth.left_pitch_point[0]
    assert thickness == pytest.approx(parameters.pitch_tooth_thickness, abs=1e-12)
    assert tooth.left_flank_angle_deg == parameters.pressure_angle_deg
    assert tooth.right_flank_angle_deg == -parameters.pressure_angle_deg
    profile = generate_rack_profile(parameters)
    assert np.allclose(profile[0], profile[-1])


def test_involute_crosses_pitch_circle_at_correct_thickness():
    parameters = StandardGearParameters(profile_samples=256)
    tooth = generate_involute_tooth(parameters)
    radii = np.linalg.norm(tooth, axis=1)
    near = tooth[np.argsort(np.abs(radii - parameters.pitch_radius))[:8]]
    angles = np.abs(np.arctan2(near[:, 1], near[:, 0]))
    expected = parameters.pitch_tooth_thickness / (
        2.0 * parameters.pitch_radius
    )
    assert np.min(np.abs(angles - expected)) < 1e-3


def test_generated_cad_solids_are_valid():
    pytest.importorskip("cadquery")
    code = (
        "from standard_involute import StandardGearParameters,generate_rack;"
        "g=generate_rack(StandardGearParameters(),validate=False);"
        "assert g.rack_solid.val().isValid();"
        "assert g.pinion_solid.val().isValid();"
        "assert g.rack_solid.val().Volume()>0;"
        "assert g.pinion_solid.val().Volume()>0"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_undercut_guard_reports_too_few_teeth():
    generated = generate_rack(StandardGearParameters(pinion_teeth=12))
    assert not generated.validation.undercut_free
    assert not generated.validation.valid


def test_step_stl_svg_dxf_exports(tmp_path: Path):
    pytest.importorskip("cadquery")
    code = (
        "from pathlib import Path;"
        "from standard_involute import StandardGearParameters,generate_rack,export_pair;"
        f"g=generate_rack(StandardGearParameters(),validate=False);"
        f"export_pair(g,Path({str(tmp_path)!r}))"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
    for path in tmp_path.iterdir():
        assert path.exists()
        assert path.stat().st_size > 100
    assert len(list(tmp_path.iterdir())) == 7
