"""Adapter from ConsensusShoulderModel.json to transmission synthesis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.interpolate import PPoly


class LiteratureShoulderModel:
    """Verified, range-limited ST trajectory with explicit provenance.

    GH = HT - ST is not a formal identity for 3-D shoulder rotations.  GH
    methods therefore raise unless the caller explicitly opts into the
    approximation.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        allow_approximate_gh: bool = False,
    ) -> None:
        self.model_path = Path(model_path)
        payload = json.loads(self.model_path.read_text(encoding="utf-8"))
        if not payload.get("validation_valid", False):
            raise ValueError("Consensus model validation is not valid.")
        selected = payload.get("selected_design")
        if not selected:
            raise ValueError("Consensus model has no selected design condition.")
        if not selected.get("conventions_verified", False):
            raise ValueError("Selected design uses unverified coordinate conventions.")
        self.selected = selected
        self.allow_approximate_gh = allow_approximate_gh
        self.valid_range_deg = tuple(float(value) for value in selected["valid_range_deg"])
        self.control_elevations_deg = np.asarray(
            selected["spline"]["knots_deg"], dtype=np.float64
        )
        coefficients = np.asarray(
            selected["spline"]["coefficients"], dtype=np.float64
        )
        self._spline = PPoly(
            coefficients, self.control_elevations_deg, extrapolate=False
        )
        self._confidence_lower = np.asarray(
            selected["confidence_lower_deg"], dtype=np.float64
        )
        self._confidence_upper = np.asarray(
            selected["confidence_upper_deg"], dtype=np.float64
        )
        self._consensus_elevation = np.asarray(
            selected["elevation_deg"], dtype=np.float64
        )
        self.pathway_name = "literature"
        self.gh_approximation_label = selected["gh_decomposition"]["warning"]

    def _check_range(self, elevation_deg):
        values = np.asarray(elevation_deg, dtype=np.float64)
        if np.any(values < self.valid_range_deg[0]) or np.any(
            values > self.valid_range_deg[1]
        ):
            raise ValueError(
                f"Elevation outside verified literature range {self.valid_range_deg}; "
                "extrapolation is forbidden."
            )
        return values

    def st_angle_at(self, elevation_deg):
        values = self._check_range(elevation_deg)
        result = self._spline(values)
        return float(result) if np.ndim(elevation_deg) == 0 else result

    def gh_angle_at(self, elevation_deg):
        if not self.allow_approximate_gh:
            raise RuntimeError(self.gh_approximation_label)
        values = self._check_range(elevation_deg)
        # Offset ST at the lower supported endpoint so GH starts at the
        # measured HT angle rather than subtracting absolute scapular posture.
        excursion = self._spline(values) - self._spline(self.valid_range_deg[0])
        result = values - excursion
        return float(result) if np.ndim(elevation_deg) == 0 else result

    def dst_delevation_at(self, elevation_deg):
        values = self._check_range(elevation_deg)
        result = self._spline(values, nu=1)
        return float(result) if np.ndim(elevation_deg) == 0 else result

    def dgh_delevation_at(self, elevation_deg):
        if not self.allow_approximate_gh:
            raise RuntimeError(self.gh_approximation_label)
        result = 1.0 - np.asarray(self.dst_delevation_at(elevation_deg))
        return float(result) if np.ndim(elevation_deg) == 0 else result

    def uncertainty_at(self, elevation_deg) -> dict:
        values = self._check_range(elevation_deg)
        lower = np.interp(
            values, self._consensus_elevation, self._confidence_lower
        )
        upper = np.interp(
            values, self._consensus_elevation, self._confidence_upper
        )
        mean = np.asarray(self.st_angle_at(elevation_deg))
        return {
            "confidence_lower_deg": float(lower) if np.ndim(elevation_deg) == 0 else lower,
            "confidence_upper_deg": float(upper) if np.ndim(elevation_deg) == 0 else upper,
            "half_width_deg": (
                float((upper - lower) / 2.0)
                if np.ndim(elevation_deg) == 0
                else (upper - lower) / 2.0
            ),
            "mean_deg": float(mean) if np.ndim(elevation_deg) == 0 else mean,
        }

    def provenance_at(self, elevation_deg) -> dict:
        value = float(self._check_range(elevation_deg))
        index = int(np.argmin(np.abs(self._consensus_elevation - value)))
        contributions = {
            paper: weights[index]
            for paper, weights in self.selected["study_contribution"].items()
            if weights[index] > 0
        }
        return {
            "elevation_deg": value,
            "motion": self.selected["motion"],
            "variable": self.selected["variable"],
            "contributing_papers": contributions,
            "source_rows": self.selected["source_rows"],
            "conventions_verified": self.selected["conventions_verified"],
            "extrapolated": False,
        }
