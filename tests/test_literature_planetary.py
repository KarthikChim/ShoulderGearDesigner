"""Validation tests for the stationary-sun planetary literature pathway."""

from pathlib import Path

import numpy as np
import pytest

from literature_planetary import (
    assembled_planet_points,
    rotate_points,
    synthesize_literature_planetary_pitch_curves,
    validate_planetary_pitch_curves,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "validation_outputs" / "LiteratureSectorTransmission.csv"


@pytest.fixture(scope="module")
def planetary():
    return synthesize_literature_planetary_pitch_curves(SOURCE)


def test_verified_range_and_stationary_sun(planetary) -> None:
    assert planetary.elevation_deg[0] == 11.0
    assert planetary.elevation_deg[-1] == 147.0
    assert np.all(planetary.sun_angle_rad == 0.0)
    with pytest.raises(ValueError, match="outside verified"):
        planetary.index_at(10.9)
    with pytest.raises(ValueError, match="outside verified"):
        planetary.index_at(147.1)


def test_planet_center_tracks_carrier_at_constant_distance(planetary) -> None:
    center_radius = np.linalg.norm(planetary.planet_center_points_world, axis=1)
    assert np.allclose(center_radius, planetary.center_distance_mm, atol=1e-10)
    center_angles = np.unwrap(
        np.arctan2(
            planetary.planet_center_points_world[:, 1],
            planetary.planet_center_points_world[:, 0],
        )
    )
    assert np.allclose(center_angles, planetary.carrier_angle_rad, atol=1e-10)


def test_pitch_contact_is_coincident_for_complete_motion(planetary) -> None:
    for index in np.linspace(0, len(planetary.elevation_deg) - 1, 81, dtype=int):
        sun_contact = planetary.sun_pitch_points_local[index]
        planet_local_contact = planetary.planet_pitch_points_local[[index]]
        planet_contact = rotate_points(
            planet_local_contact, planetary.planet_absolute_angle_rad[index]
        )[0] + planetary.planet_center_points_world[index]
        assert np.allclose(sun_contact, planet_contact, atol=1e-9)


def test_external_planetary_rolling_equation(planetary) -> None:
    residual = (
        -planetary.dcarrier_dE * planetary.sun_pitch_radius_mm
        + planetary.dplanet_relative_dE * planetary.planet_pitch_radius_mm
    )
    assert np.max(np.abs(residual)) < 1e-9
    assert np.all(planetary.signed_ratio < 0.0)
    assert np.all(planetary.sun_pitch_radius_mm > 0.0)
    assert np.all(planetary.planet_pitch_radius_mm > 0.0)
    assert np.allclose(
        planetary.sun_pitch_radius_mm + planetary.planet_pitch_radius_mm,
        planetary.center_distance_mm,
        atol=1e-10,
    )


def test_planet_curve_assembles_in_starting_position(planetary) -> None:
    assembled = assembled_planet_points(planetary, 0)
    expected = (
        planetary.planet_pitch_points_local
        + planetary.planet_center_points_world[0]
    )
    assert np.allclose(assembled, expected, atol=1e-12)


def test_complete_validation_passes(planetary) -> None:
    report = validate_planetary_pitch_curves(planetary)
    assert report.passed
    assert report.sun_stationary
    assert report.no_extrapolation
    assert report.sun_curve_simple
    assert report.planet_curve_simple
    assert report.maximum_elevation_error_deg < 0.02
    assert report.rms_elevation_error_deg < 0.02
    assert report.endpoint_elevation_error_deg < 0.02
