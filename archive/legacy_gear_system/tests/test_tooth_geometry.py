"""Tests for rack-generated non-circular tooth geometry."""

import numpy as np
from shapely.geometry import Polygon

from settings import Settings
from simulation import Simulation
from tooth_geometry import NonCircularToothGenerator, ToothParameters


def generated():
    settings = Settings()
    simulation = Simulation(settings)
    return simulation.generated_gear


def test_arc_length_controls_automatic_tooth_count() -> None:
    gear = generated()
    expected = round(
        gear.pitch_length
        / (np.pi * Simulation(Settings()).generated_pair.design.module)
    )
    assert gear.tooth_count == expected
    assert np.isclose(gear.circular_pitch, gear.pitch_length / expected)


def test_tooth_locations_are_equally_spaced_in_arc_length() -> None:
    gear = generated()
    locations = np.array([item.arc_length for item in gear.locations])
    cyclic = np.diff(np.r_[locations, locations[0] + gear.pitch_length])
    assert np.allclose(cyclic, gear.circular_pitch, rtol=0.0, atol=1e-9)
    assert all(np.isclose(np.linalg.norm(item.tangent), 1.0) for item in gear.locations)
    assert all(
        np.isclose(np.dot(item.tangent, item.outward_normal), 0.0, atol=1e-10)
        for item in gear.locations
    )


def test_rack_envelope_is_one_valid_closed_polygon() -> None:
    gear = generated()
    polygon = Polygon(gear.polygon)
    assert polygon.is_valid
    assert polygon.exterior.is_simple
    assert np.allclose(gear.polygon[0], gear.polygon[-1])
    assert gear.validation.valid


def test_explicit_tooth_count_is_respected() -> None:
    simulation = Simulation(Settings())
    gear = NonCircularToothGenerator().generate(
        simulation.pitch_data.input_points,
        ToothParameters(module=3.0, tooth_count=36, envelope_samples=1024),
    )
    assert gear.tooth_count == 36
    assert len(gear.locations) == 36
