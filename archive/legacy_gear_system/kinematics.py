"""Configurable scapulohumeral rhythm calculations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from settings import RatioRegion
from utils import clamp


@dataclass(frozen=True)
class ShoulderState:
    """Kinematic values evaluated at one commanded arm elevation."""

    elevation_deg: float
    gh_deg: float
    st_deg: float
    instantaneous_ratio: float
    input_rotation_deg: float
    output_rotation_deg: float


class ShoulderModel:
    """Integrate arbitrary piecewise GH:ST ratios over total elevation.

    For an incremental total elevation ``dE`` and ratio ``k = dGH/dST``:

    ``dGH = k/(k+1) * dE`` and ``dST = 1/(k+1) * dE``.

    This guarantees ``GH + ST = total elevation`` at every point.
    """

    def __init__(
        self,
        regions: tuple[RatioRegion, ...],
        max_elevation_deg: float = 180.0,
        input_revolutions_per_cycle: float = 1.0,
    ) -> None:
        self._regions = self._validate_regions(regions, max_elevation_deg)
        self.max_elevation_deg = max_elevation_deg
        self.input_revolutions_per_cycle = input_revolutions_per_cycle
        self.valid_range_deg = (0.0, max_elevation_deg)
        self.control_elevations_deg = np.asarray(
            [0.0, *(region.end_deg for region in self._regions)], dtype=float
        )
        self.pathway_name = "legacy_piecewise"

    @property
    def regions(self) -> tuple[RatioRegion, ...]:
        """Return the immutable ratio schedule."""

        return self._regions

    @staticmethod
    def _validate_regions(
        regions: tuple[RatioRegion, ...], max_elevation_deg: float
    ) -> tuple[RatioRegion, ...]:
        if not regions:
            raise ValueError("At least one ratio region is required.")
        ordered = tuple(sorted(regions, key=lambda region: region.start_deg))
        cursor = 0.0
        for region in ordered:
            if abs(region.start_deg - cursor) > 1e-9:
                raise ValueError("Ratio regions must be contiguous and start at zero.")
            cursor = region.end_deg
        if abs(cursor - max_elevation_deg) > 1e-9:
            raise ValueError("Ratio regions must cover the full elevation range.")
        return ordered

    def ratio_at(self, elevation_deg: float) -> float:
        """Return the active incremental GH-to-ST ratio."""

        elevation = clamp(elevation_deg, 0.0, self.max_elevation_deg)
        for region in self._regions:
            if elevation < region.end_deg or region is self._regions[-1]:
                return region.gh_to_st_ratio
        return self._regions[-1].gh_to_st_ratio

    def contributions_at(self, elevation_deg: float) -> tuple[float, float]:
        """Integrate GH and ST contributions from zero to *elevation_deg*."""

        target = clamp(elevation_deg, 0.0, self.max_elevation_deg)
        gh = 0.0
        st = 0.0
        for region in self._regions:
            covered = max(0.0, min(target, region.end_deg) - region.start_deg)
            if covered <= 0:
                continue
            ratio = region.gh_to_st_ratio
            gh += covered * ratio / (ratio + 1.0)
            st += covered / (ratio + 1.0)
            if target <= region.end_deg:
                break
        return gh, st

    def st_angle_at(self, elevation_deg):
        values = np.asarray(elevation_deg)
        result = np.array(
            [self.contributions_at(float(value))[1] for value in values.flat]
        ).reshape(values.shape)
        return float(result) if np.ndim(elevation_deg) == 0 else result

    def gh_angle_at(self, elevation_deg):
        values = np.asarray(elevation_deg)
        result = np.array(
            [self.contributions_at(float(value))[0] for value in values.flat]
        ).reshape(values.shape)
        return float(result) if np.ndim(elevation_deg) == 0 else result

    def dst_delevation_at(self, elevation_deg):
        values = np.asarray(elevation_deg, dtype=float)
        result = np.empty_like(values)
        for index, value in np.ndenumerate(values):
            result[index] = 1.0 / (self.ratio_at(float(value)) + 1.0)
        return float(result) if np.ndim(elevation_deg) == 0 else result

    def dgh_delevation_at(self, elevation_deg):
        result = 1.0 - np.asarray(self.dst_delevation_at(elevation_deg))
        return float(result) if np.ndim(elevation_deg) == 0 else result

    def evaluate(self, elevation_deg: float) -> ShoulderState:
        """Evaluate the complete shoulder and schematic gear state."""

        elevation = clamp(elevation_deg, 0.0, self.max_elevation_deg)
        gh, st = self.contributions_at(elevation)
        input_rotation = (
            elevation
            / self.max_elevation_deg
            * 360.0
            * self.input_revolutions_per_cycle
        )
        # The Phase-1 circular output is a schematic visualization of ST motion.
        output_rotation = -(
            st
            / self.max_elevation_deg
            * 360.0
            * self.input_revolutions_per_cycle
        )
        return ShoulderState(
            elevation_deg=elevation,
            gh_deg=gh,
            st_deg=st,
            instantaneous_ratio=self.ratio_at(elevation),
            input_rotation_deg=input_rotation,
            output_rotation_deg=output_rotation,
        )
