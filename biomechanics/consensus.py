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
        biological_sd = []
        digitization_uncertainty = []
        verified = []
        convention_ids = []
        compatibility_groups = []
        source_rows: list[int] = []
        for elevation in elevations:
            samples = by_elevation[float(elevation)]
            values.append(np.mean([sample.normalized_value for sample in samples]))
            biological = [
                sample.biological_sd_deg
                for sample in samples
                if sample.biological_sd_deg is not None
            ]
            digitization = [
                sample.reported_uncertainty_deg
                for sample in samples
                if sample.reported_uncertainty_deg is not None
            ]
            biological_sd.append(np.mean(biological) if biological else np.nan)
            digitization_uncertainty.append(
                np.mean(digitization) if digitization else np.nan
            )
            verified.extend(sample.original_convention.verified for sample in samples)
            convention_ids.extend(
                sample.original_convention.convention_id for sample in samples
            )
            compatibility_groups.extend(
                sample.original_convention.compatibility_group for sample in samples
            )
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
                biological_sd_deg=np.asarray(biological_sd, dtype=np.float64),
                digitization_uncertainty_deg=np.asarray(
                    digitization_uncertainty, dtype=np.float64
                ),
                source_rows=tuple(source_rows),
                study_weight=study_weight,
                sample_size=paper_map[paper_id].sample_size,
                conventions_verified=all(verified),
                convention_ids=tuple(sorted(set(convention_ids))),
                compatibility_groups=tuple(sorted(set(compatibility_groups))),
            )
        )
    return tuple(curves)


def build_consensus_datasets(
    curves: tuple[DigitizedCurve, ...],
    weighting_configuration: dict,
    grid_step_deg: float = 1.0,
    uncertainty_scenario: str | None = None,
    equal_study_weighting: bool = False,
) -> tuple[ConsensusDataset, ...]:
    groups: dict[tuple, list[DigitizedCurve]] = defaultdict(list)
    for curve in curves:
        groups[(curve.motion_key, curve.variable)].append(curve)
    uncertainty_config = weighting_configuration["uncertainty"]
    scenario = uncertainty_scenario or uncertainty_config["default_scenario"]
    missing_assumption = uncertainty_config["missing_sd_scenarios_deg"][scenario]
    z_value = float(weighting_configuration["uncertainty"]["confidence_z"])
    results: list[ConsensusDataset] = []

    for (motion_key, variable), source_curves in groups.items():
        contributing_papers = {curve.paper_id for curve in source_curves}
        all_verified = all(curve.conventions_verified for curve in source_curves)
        compatibility_groups = {
            group
            for curve in source_curves
            for group in curve.compatibility_groups
        }
        compatible = (
            all_verified
            and "unresolved" not in compatibility_groups
            and len(compatibility_groups) == 1
        )
        if len(contributing_papers) > 1 and not compatible:
            # Cross-paper averaging is prohibited until every convention is
            # explicitly audited and verified compatible.
            continue
        minimum = np.floor(min(curve.elevation_deg[0] for curve in source_curves))
        maximum = np.ceil(max(curve.elevation_deg[-1] for curve in source_curves))
        grid = np.arange(minimum, maximum + grid_step_deg * 0.5, grid_step_deg)
        values = np.full((len(source_curves), len(grid)), np.nan)
        biological_variance_samples = np.full_like(values, np.nan)
        digitization_variance_samples = np.full_like(values, np.nan)
        weights = np.zeros_like(values)
        for row, curve in enumerate(source_curves):
            inside = (grid >= curve.elevation_deg[0]) & (grid <= curve.elevation_deg[-1])
            interpolation = PchipInterpolator(
                curve.elevation_deg, curve.value_deg, extrapolate=False
            )
            values[row, inside] = interpolation(grid[inside])
            biological_variance_samples[row, inside] = _interpolate_uncertainty(
                curve.elevation_deg,
                curve.biological_sd_deg,
                grid[inside],
                missing_assumption,
            ) ** 2
            digitization_variance_samples[row, inside] = _interpolate_uncertainty(
                curve.elevation_deg,
                curve.digitization_uncertainty_deg,
                grid[inside],
                0.0,
            ) ** 2
            weights[row, inside] = 1.0 if equal_study_weighting else curve.study_weight
        weight_sum = np.sum(weights, axis=0)
        valid = weight_sum > 0.0
        mean = np.full(len(grid), np.nan)
        mean[valid] = np.nansum(values[:, valid] * weights[:, valid], axis=0) / weight_sum[valid]
        deviations = values - mean
        between_variance = np.full(len(grid), np.nan)
        between_variance[valid] = (
            np.nansum(weights[:, valid] * deviations[:, valid] ** 2, axis=0)
            / weight_sum[valid]
        )
        biological_variance = _weighted_available_variance(
            biological_variance_samples, weights
        )
        digitization_variance = _weighted_available_variance(
            digitization_variance_samples, weights
        )
        variance = between_variance + np.nan_to_num(biological_variance) + np.nan_to_num(
            digitization_variance
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
                    biological_variance=biological_variance,
                    digitization_variance=digitization_variance,
                    standard_deviation=standard_deviation,
                    confidence_lower=lower,
                    confidence_upper=upper,
                    effective_study_count=effective_n,
                ),
                available_study_count=study_count,
                available_sample_count=sample_count,
                study_contribution=contribution,
                source_curves=tuple(source_curves),
                conventions_verified=compatible if len(contributing_papers) > 1 else all_verified,
                uncertainty_scenario=scenario,
            )
        )
    return tuple(results)


def _interpolate_uncertainty(
    elevation: np.ndarray,
    uncertainty: np.ndarray,
    target: np.ndarray,
    missing_assumption: float | None,
) -> np.ndarray:
    known = np.isfinite(uncertainty)
    if np.count_nonzero(known) >= 2:
        result = PchipInterpolator(
            elevation[known], uncertainty[known], extrapolate=False
        )(target)
    elif np.count_nonzero(known) == 1:
        result = np.full(len(target), uncertainty[known][0])
    else:
        fill = np.nan if missing_assumption is None else missing_assumption
        result = np.full(len(target), fill)
    if missing_assumption is not None:
        result = np.where(np.isfinite(result), result, missing_assumption)
    return result


def _weighted_available_variance(
    samples: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    available_weights = np.where(np.isfinite(samples), weights, 0.0)
    denominator = np.sum(available_weights, axis=0)
    return np.divide(
        np.nansum(available_weights * samples, axis=0),
        denominator,
        out=np.full(samples.shape[1], np.nan),
        where=denominator > 0,
    )
