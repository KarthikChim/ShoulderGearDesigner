"""Top-level orchestration for the literature-to-consensus pipeline."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .consensus import build_consensus_datasets, build_digitized_curves
from .exporter import export_consensus_model
from .loader import load_literature_csv
from .models import ConsensusModel, MotionDataset
from .normalization import CoordinateNormalizer
from .selection import load_design_condition, select_design_dataset
from .sensitivity import run_sensitivity_analyses
from .splines import fit_shape_preserving_splines
from .weighting import StudyWeightingSystem


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONVENTIONS = PACKAGE_DIR / "config" / "coordinate_conventions.json"
DEFAULT_WEIGHTING = PACKAGE_DIR / "config" / "default_weighting.json"
DEFAULT_DESIGN_CONDITION = PACKAGE_DIR / "config" / "design_condition.json"


class BiomechanicsEngine:
    """Authoritative, non-destructive shoulder biomechanics pipeline."""

    def __init__(
        self,
        source_csv: str | Path,
        coordinate_configuration: str | Path = DEFAULT_CONVENTIONS,
        weighting_configuration: str | Path = DEFAULT_WEIGHTING,
        design_condition: str | Path = DEFAULT_DESIGN_CONDITION,
    ) -> None:
        self.source_csv = Path(source_csv).resolve()
        self.coordinate_configuration = Path(coordinate_configuration).resolve()
        self.weighting_configuration = Path(weighting_configuration).resolve()
        self.design_condition_path = Path(design_condition).resolve()
        self.raw_rows = ()
        self.papers = ()
        self.validation_report = None
        self.motion_datasets = ()
        self.normalized_observations = ()
        self.study_weights = ()
        self.curves = ()
        self.consensus_datasets = ()
        self.model: ConsensusModel | None = None
        self.selected_design = None
        self.sensitivity_analyses = None

    def build(self) -> ConsensusModel:
        """Run validation, normalization, weighting, consensus, and spline fitting."""

        rows, papers, report = load_literature_csv(self.source_csv)
        self.raw_rows = rows
        self.papers = papers
        self.validation_report = report

        motion_groups = {}
        for row in rows:
            motion_groups.setdefault(row.motion_key, []).append(row)
        self.motion_datasets = tuple(
            MotionDataset(key, tuple(items))
            for key, items in sorted(
                motion_groups.items(), key=lambda item: item[0].identifier
            )
        )

        normalizer = CoordinateNormalizer.from_json(self.coordinate_configuration)
        self.normalized_observations = normalizer.normalize(rows)
        weighting = StudyWeightingSystem.from_json(self.weighting_configuration)
        self.study_weights = tuple(
            weighting.weight(paper, motion.key.motion_type)
            for motion in self.motion_datasets
            for paper in papers
            if any(row.paper_id == paper.paper_id for row in motion.rows)
        )
        self.curves = build_digitized_curves(
            self.normalized_observations, papers, weighting
        )
        weighting_data = weighting.configuration
        self.consensus_datasets = build_consensus_datasets(
            self.curves, weighting_data
        )
        splines = fit_shape_preserving_splines(self.consensus_datasets)
        digest = hashlib.sha256(self.source_csv.read_bytes()).hexdigest()
        unverified = sum(
            not item.original_convention.verified
            for item in self.normalized_observations
        )
        condition = load_design_condition(self.design_condition_path)
        preliminary = ConsensusModel(
            datasets=self.consensus_datasets,
            splines=splines,
            generated_utc="omitted-for-reproducible-build",
            source_csv=self.source_csv,
            source_sha256=digest,
            metadata={
                "schema_version": 1,
                "raw_row_count": len(rows),
                "paper_count": len(papers),
                "motion_group_count": len(self.motion_datasets),
                "normalized_observation_count": len(self.normalized_observations),
                "unverified_coordinate_observation_count": unverified,
                "validation_valid": report.valid,
                "coordinate_configuration": self.coordinate_configuration.name,
                "weighting_configuration": self.weighting_configuration.name,
                "weighting_policy": weighting_data,
                "papers": [asdict(paper) for paper in papers],
            },
        )
        selected_dataset, selected_spline = select_design_dataset(
            preliminary, condition
        )
        self.selected_design = {
            "condition": condition,
            "motion": {
                "identifier": selected_dataset.motion_key.identifier,
                "motion_type": selected_dataset.motion_key.motion_type,
                "motion_plane": selected_dataset.motion_key.motion_plane,
                "direction": selected_dataset.motion_key.direction,
                "loaded": selected_dataset.motion_key.loaded,
                "healthy_only": selected_dataset.motion_key.healthy_only,
            },
            "variable": selected_dataset.variable,
            "valid_range_deg": list(condition["supported_range_deg"]),
            "elevation_deg": selected_dataset.elevation_deg.tolist(),
            "mean_deg": selected_dataset.mean_deg.tolist(),
            "confidence_lower_deg": selected_dataset.uncertainty.confidence_lower.tolist(),
            "confidence_upper_deg": selected_dataset.uncertainty.confidence_upper.tolist(),
            "available_study_count": selected_dataset.available_study_count.tolist(),
            "available_sample_count": selected_dataset.available_sample_count.tolist(),
            "study_contribution": {
                key: value.tolist()
                for key, value in selected_dataset.study_contribution.items()
            },
            "contributing_papers": sorted(
                {curve.paper_id for curve in selected_dataset.source_curves}
            ),
            "source_rows": sorted(
                {
                    row
                    for curve in selected_dataset.source_curves
                    for row in curve.source_rows
                }
            ),
            "conventions_verified": selected_dataset.conventions_verified,
            # Every exported design point comes directly from the supported
            # literature interval.  The adapter rejects values outside this
            # interval instead of silently extending the PCHIP.
            "extrapolated_point_count": 0,
            "all_points_within_supported_range": bool(
                np.all(
                    (selected_dataset.elevation_deg >= condition["supported_range_deg"][0])
                    & (
                        selected_dataset.elevation_deg
                        <= condition["supported_range_deg"][1]
                    )
                )
            ),
            "spline": {
                "interpolation": selected_spline.interpolation,
                "knots_deg": selected_spline.knots.tolist(),
                "coefficients": selected_spline.coefficients.tolist(),
            },
            "extrapolation": "forbidden",
            "gh_decomposition": condition["gh_decomposition"],
        }
        self.sensitivity_analyses = run_sensitivity_analyses(
            self.curves, weighting_data, condition
        )
        preliminary.metadata["selected_design_condition"] = condition["condition_id"]
        preliminary.metadata["selected_design_conventions_verified"] = (
            selected_dataset.conventions_verified
        )
        self.model = preliminary
        return self.model

    def export(self, destination: str | Path) -> Path:
        if self.model is None:
            self.build()
        assert self.model is not None
        return export_consensus_model(
            self.model,
            self.validation_report,
            self.normalized_observations,
            self.selected_design,
            self.sensitivity_analyses,
            destination,
        )
