"""Tests for GUI-independent timing behavior."""

from __future__ import annotations

import pytest

from animation import Animator


def test_paused_animator_does_not_move() -> None:
    animator = Animator(maximum_deg=180.0, speed_deg_s=30.0)
    assert animator.advance(1.0) == 0.0


def test_running_animator_uses_elapsed_time() -> None:
    animator = Animator(maximum_deg=180.0, speed_deg_s=30.0)
    animator.start()
    assert animator.advance(0.5) == pytest.approx(15.0)


def test_animator_reverses_at_endpoint() -> None:
    animator = Animator(maximum_deg=180.0, speed_deg_s=30.0, current_deg=175.0, playing=True)
    assert animator.advance(1.0) == 180.0
    assert animator.direction == -1.0

