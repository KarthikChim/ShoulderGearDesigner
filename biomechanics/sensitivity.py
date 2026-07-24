"""Consensus sensitivity analyses required before mechanism synthesis."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .consensus import build_consensus_datasets
from .models import ConsensusDataset, DigitizedCurve


def run_sensitivity_analyses(
    curves: tuple[DigitizedCurve, ...],
    weighting_configuration: dict,
    selected_condition: dict,
) -> dict:
    variants = {}
    for name, equal, scenario in (
        ("method_quality", False, "exclude_missing"),
        ("equal_study", True, "exclude_missing"),
        ("missing_sd_0_5_deg", False, "assume_0_5_deg"),
        ("missing_sd_1_0_deg", False, "assume_1_0_deg"),
    ):
        datasets = build_consensus_datasets(
            curves,
            weighting_configuration,
            uncertainty_scenario=scenario,
            equal_study_weighting=equal,
        )
        variants[name] = _selected_curve_summary(datasets, selected_condition)

    selected_curves = tuple(
        curve for curve in curves if _curve_matches(curve, selected_condition)
    )
    papers = sorted({curve.paper_id for curve in selected_curves})
    leave_one_out = []
    for paper in papers:
        remaining = tuple(curve for curve in selected_curves if curve.paper_id != paper)
        if not remaining:
            leave_one_out.append(
                {
                    "omitted_paper": paper,
                    "status": "not_estimable_single_study_condition",
                    "elevation_deg": [],
                    "mean_deg": [],
                }
            )
            continue
        datasets = build_consensus_datasets(remaining, weighting_configuration)
        summary = _selected_curve_summary(datasets, selected_condition)
        summary["omitted_paper"] = paper
        summary["status"] = "generated"
        leave_one_out.append(summary)
    return {"variants": variants, "leave_one_study_out": leave_one_out}


def _curve_matches(curve: DigitizedCurve, condition: dict) -> bool:
    key = curve.motion_key
    return (
        key.motion_type == condition["motion_type"]
        and key.motion_plane == condition["motion_plane"]
        and key.direction == condition["direction"]
        and key.loaded is condition["loaded"]
        and key.healthy_only is condition["healthy_only"]
        and curve.variable == condition["primary_variable"]
        and curve.paper_id in condition["allowed_papers"]
    )


def _selected_curve_summary(
    datasets: tuple[ConsensusDataset, ...], condition: dict
) -> dict:
    matches = [
        dataset
        for dataset in datasets
        if dataset.motion_key.motion_type == condition["motion_type"]
        and dataset.motion_key.motion_plane == condition["motion_plane"]
        and dataset.motion_key.direction == condition["direction"]
        and dataset.motion_key.loaded is condition["loaded"]
        and dataset.motion_key.healthy_only is condition["healthy_only"]
        and dataset.variable == condition["primary_variable"]
    ]
    if not matches:
        return {"status": "unavailable", "elevation_deg": [], "mean_deg": []}
    dataset = matches[0]
    return {
        "status": "generated",
        "elevation_deg": dataset.elevation_deg.tolist(),
        "mean_deg": dataset.mean_deg.tolist(),
        "confidence_lower_deg": dataset.uncertainty.confidence_lower.tolist(),
        "confidence_upper_deg": dataset.uncertainty.confidence_upper.tolist(),
        "papers": sorted({curve.paper_id for curve in dataset.source_curves}),
    }
