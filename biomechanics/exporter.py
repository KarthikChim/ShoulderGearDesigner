"""Deterministic JSON serialization for future transmission synthesis."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .models import ConsensusModel, NormalizedObservation, ValidationReport


def _motion(key) -> dict:
    return {
        "identifier": key.identifier,
        "motion_type": key.motion_type,
        "motion_plane": key.motion_plane,
        "direction": key.direction,
        "loaded": key.loaded,
        "healthy_only": key.healthy_only,
    }


def _array(values) -> list:
    return np.asarray(values).tolist()


def export_consensus_model(
    model: ConsensusModel,
    validation: ValidationReport,
    observations: tuple[NormalizedObservation, ...],
    destination: str | Path,
) -> Path:
    path = Path(destination)
    conventions = {}
    transformations = {}
    for item in observations:
        conventions[item.original_convention.convention_id] = asdict(
            item.original_convention
        )
        transformations[item.transformation.transformation_id] = asdict(
            item.transformation
        )

    datasets = []
    for dataset in model.datasets:
        datasets.append(
            {
                "motion": _motion(dataset.motion_key),
                "variable": dataset.variable,
                "elevation_deg": _array(dataset.elevation_deg),
                "mean_deg": _array(dataset.mean_deg),
                "variance_deg2": _array(dataset.uncertainty.variance),
                "standard_deviation_deg": _array(
                    dataset.uncertainty.standard_deviation
                ),
                "confidence_lower_deg": _array(
                    dataset.uncertainty.confidence_lower
                ),
                "confidence_upper_deg": _array(
                    dataset.uncertainty.confidence_upper
                ),
                "effective_study_count": _array(
                    dataset.uncertainty.effective_study_count
                ),
                "available_study_count": _array(dataset.available_study_count),
                "available_sample_count": _array(dataset.available_sample_count),
                "study_contribution": {
                    paper: _array(values)
                    for paper, values in dataset.study_contribution.items()
                },
                "source_curves": [
                    {
                        "paper_id": curve.paper_id,
                        "source_rows": list(curve.source_rows),
                        "study_weight": curve.study_weight,
                        "sample_size": curve.sample_size,
                    }
                    for curve in dataset.source_curves
                ],
            }
        )

    splines = []
    for spline in model.splines:
        interpolator = spline.interpolator()
        first = interpolator.derivative(1)
        second = interpolator.derivative(2)
        splines.append(
            {
                "motion": _motion(spline.motion_key),
                "variable": spline.variable,
                "interpolation": spline.interpolation,
                "coefficient_order": "descending polynomial power per interval",
                "knots_deg": _array(spline.knots),
                "coefficients": _array(spline.coefficients),
                "first_derivative_coefficients": _array(first.c),
                "second_derivative_coefficients": _array(second.c),
                "extrapolation": "disabled",
            }
        )

    payload = {
        "format": "ConsensusShoulderModel",
        "schema_version": 1,
        "generated_utc": model.generated_utc,
        "source": {
            "csv_path": str(model.source_csv),
            "sha256": model.source_sha256,
        },
        "metadata": model.metadata,
        "validation": {
            "valid": validation.valid,
            "row_count": validation.row_count,
            "paper_count": validation.paper_count,
            "duplicate_row_count": validation.duplicate_row_count,
            "missing_by_column": validation.missing_by_column,
            "repeated_elevation_groups": validation.repeated_elevation_groups,
            "issues": [asdict(issue) for issue in validation.issues],
        },
        "coordinate_conventions": list(conventions.values()),
        "transformations": list(transformations.values()),
        "consensus_datasets": datasets,
        "splines": splines,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    return path
