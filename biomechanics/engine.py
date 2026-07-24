"""Top-level orchestration for the literature-to-consensus pipeline."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .consensus import build_consensus_datasets, build_digitized_curves
from .exporter import export_consensus_model
from .loader import load_literature_csv
from .models import ConsensusModel, MotionDataset
from .normalization import CoordinateNormalizer
from .splines import fit_shape_preserving_splines
from .weighting import StudyWeightingSystem


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONVENTIONS = PACKAGE_DIR / "config" / "coordinate_conventions.json"
DEFAULT_WEIGHTING = PACKAGE_DIR / "config" / "default_weighting.json"


class BiomechanicsEngine:
    """Authoritative, non-destructive shoulder biomechanics pipeline."""

    def __init__(
        self,
        source_csv: str | Path,
        coordinate_configuration: str | Path = DEFAULT_CONVENTIONS,
        weighting_configuration: str | Path = DEFAULT_WEIGHTING,
    ) -> None:
        self.source_csv = Path(source_csv).resolve()
        self.coordinate_configuration = Path(coordinate_configuration).resolve()
        self.weighting_configuration = Path(weighting_configuration).resolve()
        self.raw_rows = ()
        self.papers = ()
        self.validation_report = None
        self.motion_datasets = ()
        self.normalized_observations = ()
        self.study_weights = ()
        self.curves = ()
        self.consensus_datasets = ()
        self.model: ConsensusModel | None = None

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
        self.model = ConsensusModel(
            datasets=self.consensus_datasets,
            splines=splines,
            generated_utc=datetime.now(timezone.utc).isoformat(),
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
                "coordinate_configuration": str(self.coordinate_configuration),
                "weighting_configuration": str(self.weighting_configuration),
                "weighting_policy": weighting_data,
                "papers": [asdict(paper) for paper in papers],
            },
        )
        return self.model

    def export(self, destination: str | Path) -> Path:
        if self.model is None:
            self.build()
        assert self.model is not None
        return export_consensus_model(
            self.model,
            self.validation_report,
            self.normalized_observations,
            destination,
        )
