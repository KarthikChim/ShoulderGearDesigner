"""Immutable domain models for raw literature and derived consensus data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import PchipInterpolator


FloatArray = NDArray[np.float64]

KINEMATIC_VARIABLES = (
    "GH_Elevation_deg",
    "ST_UpwardRotation_deg",
    "ST_PosteriorTilt_deg",
    "ST_InternalRotation_deg",
    "SC_Elevation_deg",
    "SC_PosteriorRotation_deg",
    "AC_UpwardRotation_deg",
    "GH_Plane_deg",
    "GH_ExternalRotation_deg",
)


@dataclass(frozen=True)
class PaperMetadata:
    paper_id: str
    title: str
    authors: str
    year: int
    measurement_method: str
    sample_size: int | None
    healthy_only: bool | None
    data_sources: tuple[str, ...]


@dataclass(frozen=True)
class MotionKey:
    motion_type: str
    motion_plane: str
    direction: str
    loaded: bool | None
    healthy_only: bool | None

    @property
    def identifier(self) -> str:
        parts = (
            self.motion_type,
            self.motion_plane,
            self.direction,
            "loaded" if self.loaded else "unloaded",
            "healthy" if self.healthy_only else "mixed",
        )
        return " | ".join(parts)


@dataclass(frozen=True)
class RawLiteratureRow:
    """One CSV row retained exactly, plus parsed values for safe computation."""

    row_number: int
    fields: tuple[tuple[str, str], ...]
    paper_id: str
    figure_or_table: str
    motion_key: MotionKey
    ht_elevation_deg: float | None
    values: tuple[tuple[str, float], ...]
    sd: float | None
    sem: float | None
    extraction_method: str
    notes: str

    def original_dict(self) -> dict[str, str]:
        return dict(self.fields)


@dataclass(frozen=True)
class MotionDataset:
    key: MotionKey
    rows: tuple[RawLiteratureRow, ...]


@dataclass(frozen=True)
class CoordinateConvention:
    convention_id: str
    variable: str
    description: str
    positive_direction: str
    reference_frame: str
    verified: bool


@dataclass(frozen=True)
class Transformation:
    transformation_id: str
    scale: float
    offset_deg: float
    description: str

    def apply(self, value: float) -> float:
        return self.scale * value + self.offset_deg


@dataclass(frozen=True)
class NormalizedObservation:
    row_number: int
    paper_id: str
    figure_or_table: str
    motion_key: MotionKey
    variable: str
    ht_elevation_deg: float
    original_value: float
    original_convention: CoordinateConvention
    transformation: Transformation
    normalized_value: float
    sd: float | None
    sem: float | None
    reported_uncertainty_deg: float | None
    extraction_method: str
    notes: str


@dataclass(frozen=True)
class DigitizedCurve:
    paper_id: str
    motion_key: MotionKey
    variable: str
    elevation_deg: FloatArray
    value_deg: FloatArray
    uncertainty_deg: FloatArray
    source_rows: tuple[int, ...]
    study_weight: float
    sample_size: int | None


@dataclass(frozen=True)
class UncertaintyModel:
    variance: FloatArray
    standard_deviation: FloatArray
    confidence_lower: FloatArray
    confidence_upper: FloatArray
    effective_study_count: FloatArray


@dataclass(frozen=True)
class ConsensusDataset:
    motion_key: MotionKey
    variable: str
    elevation_deg: FloatArray
    mean_deg: FloatArray
    uncertainty: UncertaintyModel
    available_study_count: NDArray[np.int64]
    available_sample_count: NDArray[np.int64]
    study_contribution: dict[str, FloatArray]
    source_curves: tuple[DigitizedCurve, ...]


@dataclass(frozen=True)
class SplineModel:
    motion_key: MotionKey
    variable: str
    knots: FloatArray
    coefficients: FloatArray
    interpolation: str = "PCHIP"

    def interpolator(self) -> PchipInterpolator:
        return PchipInterpolator.construct_fast(
            self.coefficients.copy(), self.knots.copy(), extrapolate=False
        )

    def evaluate(self, elevation_deg: FloatArray | float, derivative: int = 0):
        return self.interpolator()(elevation_deg, nu=derivative)


@dataclass(frozen=True)
class ConsensusModel:
    datasets: tuple[ConsensusDataset, ...]
    splines: tuple[SplineModel, ...]
    generated_utc: str
    source_csv: Path
    source_sha256: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    message: str
    row_numbers: tuple[int, ...] = ()


@dataclass(frozen=True)
class ValidationReport:
    source: Path
    row_count: int
    paper_count: int
    duplicate_row_count: int
    missing_by_column: dict[str, int]
    repeated_elevation_groups: int
    issues: tuple[ValidationIssue, ...]

    @property
    def valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)
