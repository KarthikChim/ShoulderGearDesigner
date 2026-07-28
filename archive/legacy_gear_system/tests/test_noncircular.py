"""Tests for smooth transmission and conjugate pitch-curve synthesis."""

from __future__ import annotations

import numpy as np
import pytest

from kinematics import ShoulderModel
from noncircular import (
    SmoothTransmission,
    synthesize_pitch_curves,
    validate_pitch_curves,
)
from settings import default_ratio_regions


@pytest.fixture
def transmission() -> SmoothTransmission:
    return SmoothTransmission(ShoulderModel(default_ratio_regions()))


def test_transmission_closes_one_to_one(transmission: SmoothTransmission) -> None:
    assert transmission.output_angle(0.0) == pytest.approx(0.0)
    assert transmission.output_angle(2.0 * np.pi) == pytest.approx(2.0 * np.pi)
    assert transmission.ratio(0.0) == pytest.approx(transmission.ratio(2.0 * np.pi))


def test_transmission_is_positive_and_smooth(transmission: SmoothTransmission) -> None:
    phase = np.linspace(0.0, 2.0 * np.pi, 5001)
    ratio = np.asarray(transmission.ratio(phase))
    assert np.all(np.isfinite(ratio))
    assert np.min(ratio) > 0.0
    assert np.max(np.abs(np.diff(ratio))) < 0.01


def test_spline_passes_biomechanical_control_points(
    transmission: SmoothTransmission,
) -> None:
    for elevation, expected_st in (
        (0.0, 0.0),
        (30.0, 6.0),
        (90.0, 26.0),
        (114.0, 38.0),
        (180.0, 60.0),
    ):
        phase = elevation / 180.0 * 2.0 * np.pi
        state = transmission.evaluate(phase)
        assert state.st_deg == pytest.approx(expected_st)
        assert state.gh_deg + state.st_deg == pytest.approx(elevation)


def test_pitch_curves_are_conjugate_and_valid(
    transmission: SmoothTransmission,
) -> None:
    data = synthesize_pitch_curves(transmission, center_distance=100.0, sample_count=4097)
    assert len(data.input_points) == 4097
    assert np.max(np.abs(data.input_radii + data.output_radii - 100.0)) < 1e-12
    report = validate_pitch_curves(data)
    assert report.constant_center_distance
    assert report.continuous_motion
    assert report.no_pitch_curve_overlap
    assert report.smooth_velocity_ratio
    assert report.ready_for_tooth_generation
