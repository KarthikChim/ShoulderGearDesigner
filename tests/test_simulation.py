"""Tests for simulation orchestration."""

from __future__ import annotations

import pytest

from settings import Settings
from simulation import Simulation


def test_simulation_synchronizes_state_and_gears() -> None:
    simulation = Simulation(Settings())
    state = simulation.set_elevation(90.0)
    assert simulation.gear_pair.input_gear.angle_deg == pytest.approx(state.input_rotation_deg)
    assert simulation.gear_pair.output_gear.angle_deg == pytest.approx(state.output_rotation_deg)
    assert state.gh_deg + state.st_deg == pytest.approx(90.0)

