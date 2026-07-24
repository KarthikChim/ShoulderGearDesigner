"""Engineering proof of the synthesized shoulder-motion mapping."""

import numpy as np

from settings import Settings
from simulation import Simulation


def test_mechanical_and_biomechanical_ratios_are_distinct() -> None:
    simulation = Simulation(Settings())
    report = simulation.biomechanics_validation
    assert not np.allclose(report.mechanical_ratio, report.actual_ratio)
    assert np.allclose(report.actual_ratio, 3.0 / report.mechanical_ratio - 1.0)


def test_endpoint_sum_and_smoothness_checks_pass() -> None:
    report = Simulation(Settings()).biomechanics_validation
    assert report.gh_endpoint_valid
    assert report.st_endpoint_valid
    assert report.elevation_sum_valid
    assert report.no_discontinuities
    assert report.no_negative_ratios
    assert report.no_velocity_spikes
    assert report.continuous_first_derivative
    assert report.continuous_second_derivative


def test_schedule_checkpoint_values_are_computed_independently() -> None:
    report = Simulation(Settings()).biomechanics_validation
    by_elevation = {item.elevation_deg: item for item in report.checkpoints}
    assert by_elevation[30.0].schedule_gh_deg == 24.0
    assert by_elevation[30.0].schedule_st_deg == 6.0
    assert by_elevation[90.0].schedule_gh_deg == 64.0
    assert by_elevation[90.0].schedule_st_deg == 26.0
    assert by_elevation[114.0].schedule_gh_deg == 76.0
    assert by_elevation[114.0].schedule_st_deg == 38.0
    assert by_elevation[180.0].schedule_gh_deg == 120.0
    assert by_elevation[180.0].schedule_st_deg == 60.0


def test_conflicting_requested_checkpoints_force_honest_failure() -> None:
    report = Simulation(Settings()).biomechanics_validation
    assert not report.specification_consistent
    assert not report.passed
    assert any("conflict" in warning for warning in report.warnings)
    assert report.maximum_gh_error > 2.0
    assert report.maximum_ratio_error > 1.0
