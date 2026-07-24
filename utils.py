"""Small reusable mathematical utilities."""

from __future__ import annotations

import math


def clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp *value* to the closed interval [minimum, maximum]."""

    return max(minimum, min(maximum, value))


def degrees_to_radians(value: float) -> float:
    """Convert degrees to radians."""

    return math.radians(value)


def wrap_degrees(value: float) -> float:
    """Wrap an angle into [0, 360)."""

    return value % 360.0

