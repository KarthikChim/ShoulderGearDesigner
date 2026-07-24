"""Motion-specific curve construction and weighted consensus synthesis."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.interpolate import PchipInterpolator

from .models import (
    ConsensusDataset,
    DigitizedCurve,
    NormalizedObservation,
    PaperMetadata,
    UncertaintyModel,
)
from .weighting import StudyWeightingSystem


def build_digitized_curves(
    observations: tuple[NormalizedObservation, ...],
    papers: tuple[PaperMetadata, ...],
    weighting: StudyWeightingSystem,
) -> tuple[DigitizedCurve, ...]:
    paper_map = {paper.paper_id: paper for paper in papers}
    groups: dict[tuple, list[NormalizedObservation]] = defaultdict(list)
    for item in observations:
        groups[
            (
                item.motion_key,
                item.variable,
                item.paper_id,
                item.figure_or_table,
            )
        ].append(item)

    # Divide a paper's quality weight across parallel curves in the same
    # motion/variable group so multiple figures cannot create pseudoreplication.
    paper_curve_counts: dict[tuple, int] = defaultdict(int)
    for motion_key, variable, paper_id, _ in groups:
        paper_curve_counts[(motion_key, variable, paper_id)] += 1

    curves: list[DigitizedCurve] = []
    for (motion_key, variable, paper_id, _figure), items in groups.items():
        by_elevation: dict[float, list[NormalizedObservation]] = defaultdict(list)
        for item in items:
            by_elevation[item.ht_elevation_deg].append(item)
        elevations = np.array(sorted(by_elevation), dtype=np.float64)
        if len(elevations) < 2:
            continue
        values = []
        uncertainties = []
        source_rows: list[int] = []
        for elevation in elevations:
            samples = by_elevation[float(elevation)]
            values.append(np.mean([sample.normalized_value for sample in samples]))
            available = [
                (
                    sample.sd
                    if sample.sd is not None
                    else sample.sem
                    if sample.sem is not None
                    else sample.reported_uncertainty_deg
                )
                for sample in samples
                if (
                    sample.sd is not None
                    or sample.sem is not None
                    or sample.reported_uncertainty_deg is not None
                )
            ]
            uncertainties.append(np.mean(available) if available else np.nan)
            source_rows.extend(sample.row_number for sample in samples)
        study_weight = weighting.weight(paper_map[paper_id], motion_key.motion_type).total
        study_weight /= paper_curve_counts[(motion_key, variable, paper_id)]
        curves.append(
            DigitizedCurve(
                paper_id=paper_id,
                motion_key=motion_key,
                variable=variable,
                elevation_deg=elevations,
                value_deg=np.asarray(values, dtype=np.float64),
                uncertainty_deg=np.asarray(uncertainties, dtype=np.float64),
                source_rows=tuple(source_rows),
                study_weight=study_weight,
                sample_size=paper_map[paper_id].sample_size,
            )
        )
    return tuple(curves)


def build_consensus_datasets(
    curves: tuple[DigitizedCurve, ...],
    weighting_configuration: dict,
    grid_step_deg: float = 1.0,
) -> tuple[ConsensusDataset, ...]:
    groups: dict[tuple, list[DigitizedCurve]] = defaultdict(list)
    for curve in curves:
        groups[(curve.motion_key, curve.variable)].append(curve)
    default_sd = float(weighting_configuration["uncertainty"]["default_sd_deg"])
    minimum_sd = float(weighting_configuration["uncertainty"]["minimum_sd_deg"])
    z_value = float(weighting_configuration["uncertainty"]["confidence_z"])
    results: list[ConsensusDataset] = []

    for (motion_key, variable), source_curves in groups.items():
        minimum = np.floor(min(curve.elevation_deg[0] for curve in source_curves))
        maximum = np.ceil(max(curve.elevation_deg[-1] for curve in source_curves))
        grid = np.arange(minimum, maximum + grid_step_deg * 0.5, grid_step_deg)
        values = np.full((len(source_curves), len(grid)), np.nan)
        measurement_variance = np.full_like(values, np.nan)
        weights = np.zeros_like(values)
        for row, curve in enumerate(source_curves):
            inside = (grid >= curve.elevation_deg[0]) & (grid <= curve.elevation_deg[-1])
            interpolation = PchipInterpolator(
                curve.elevation_deg, curve.value_deg, extrapolate=False
            )
            values[row, inside] = interpolation(grid[inside])
            known_uncertainty = np.isfinite(curve.uncertainty_deg)
            if np.count_nonzero(known_uncertainty) >= 2:
                uncertainty_interpolation = PchipInterpolator(
                    curve.elevation_deg[known_uncertainty],
                    curve.uncertainty_deg[known_uncertainty],
                    extrapolate=False,
                )
                interpolated_uncertainty = uncertainty_interpolation(grid[inside])
                interpolated_uncertainty = np.where(
                    np.isfinite(interpolated_uncertainty),
                    interpolated_uncertainty,
                    default_sd,
                )
            elif np.count_nonzero(known_uncertainty) == 1:
                interpolated_uncertainty = np.full(
                    np.count_nonzero(inside),
                    curve.uncertainty_deg[known_uncertainty][0],
                )
            else:
                interpolated_uncertainty = np.full(
                    np.count_nonzero(inside), default_sd
                )
            measurement_variance[row, inside] = np.maximum(
                interpolated_uncertainty, minimum_sd
            ) ** 2
            weights[row, inside] = curve.study_weight
        weight_sum = np.sum(weights, axis=0)
        valid = weight_sum > 0.0
        mean = np.full(len(grid), np.nan)
        mean[valid] = np.nansum(values[:, valid] * weights[:, valid], axis=0) / weight_sum[valid]
        deviations = values - mean
        variance = np.full(len(grid), np.nan)
        variance[valid] = (
            np.nansum(
                weights[:, valid]
                * (
                    deviations[:, valid] ** 2
                    + measurement_variance[:, valid]
                ),
                axis=0,
            )
            / weight_sum[valid]
        )
        weight_square_sum = np.sum(weights**2, axis=0)
        effective_n = np.zeros(len(grid))
        effective_n[valid] = weight_sum[valid] ** 2 / weight_square_sum[valid]
        standard_deviation = np.sqrt(variance)
        standard_error = standard_deviation / np.sqrt(np.maximum(effective_n, 1.0))
        lower = mean - z_value * standard_error
        upper = mean + z_value * standard_error
        contribution: dict[str, np.ndarray] = {}
        for index, curve in enumerate(source_curves):
            normalized = np.divide(
                weights[index],
                weight_sum,
                out=np.zeros_like(weight_sum),
                where=weight_sum > 0,
            )
            contribution[curve.paper_id] = (
                contribution.get(curve.paper_id, np.zeros_like(weight_sum))
                + normalized
            )
        papers = sorted({curve.paper_id for curve in source_curves})
        study_count = np.zeros(len(grid), dtype=np.int64)
        sample_count = np.zeros(len(grid), dtype=np.int64)
        for paper in papers:
            indices = [
                index
                for index, curve in enumerate(source_curves)
                if curve.paper_id == paper
            ]
            present = np.any(np.isfinite(values[indices]), axis=0)
            study_count += present.astype(np.int64)
            sample_size = next(
                (source_curves[index].sample_size for index in indices
                 if source_curves[index].sample_size is not None),
                0,
            )
            sample_count += present.astype(np.int64) * sample_size
        results.append(
            ConsensusDataset(
                motion_key=motion_key,
                variable=variable,
                elevation_deg=grid,
                mean_deg=mean,
                uncertainty=UncertaintyModel(
                    variance=variance,
                    standard_deviation=standard_deviation,
                    confidence_lower=lower,
                    confidence_upper=upper,
                    effective_study_count=effective_n,
                ),
                available_study_count=study_count,
                available_sample_count=sample_count,
                study_contribution=contribution,
                source_curves=tuple(source_curves),
            )
        )
    return tuple(results)
