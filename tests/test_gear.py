"""Tests for gear rotation and pair geometry."""

from __future__ import annotations

import numpy as np
import pytest

from gear import Gear
from meshing import GearPair
from pitch_curve import CircularPitchCurve


def test_world_points_rotate_with_gear() -> None:
    gear = Gear("test", CircularPitchCurve(10.0), 5.0, 2.0, angle_deg=90.0)
    points = gear.world_points(17)
    assert points[0] == pytest.approx(np.array([5.0, 12.0]))


def test_contact_point_for_equal_circles() -> None:
    first = Gear("A", CircularPitchCurve(20.0), 0.0, 0.0)
    second = Gear("B", CircularPitchCurve(20.0), 40.0, 0.0)
    pair = GearPair(first, second)
    assert pair.center_distance == pytest.approx(40.0)
    assert pair.contact_point() == pytest.approx(np.array([20.0, 0.0]))

