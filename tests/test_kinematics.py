"""Tests for configurable shoulder-rhythm integration."""

from __future__ import annotations

import pytest

from kinematics import ShoulderModel
from settings import default_ratio_regions


@pytest.fixture
def model() -> ShoulderModel:
    return ShoulderModel(default_ratio_regions())


@pytest.mark.parametrize(
    ("elevation", "ratio"),
    [(0.0, 4.0), (29.9, 4.0), (30.0, 2.0), (90.0, 1.0), (114.0, 2.0)],
)
def test_ratio_regions(model: ShoulderModel, elevation: float, ratio: float) -> None:
    assert model.ratio_at(elevation) == pytest.approx(ratio)


def test_contributions_sum_to_total(model: ShoulderModel) -> None:
    for elevation in (0.0, 15.0, 30.0, 75.0, 114.0, 180.0):
        gh, st = model.contributions_at(elevation)
        assert gh + st == pytest.approx(elevation)


def test_default_final_contributions(model: ShoulderModel) -> None:
    gh, st = model.contributions_at(180.0)
    assert gh == pytest.approx(120.0)
    assert st == pytest.approx(60.0)

