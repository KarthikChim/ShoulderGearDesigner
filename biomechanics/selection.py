"""Strict selection of one literature-supported design condition."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .models import ConsensusDataset, ConsensusModel, SplineModel


def load_design_condition(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def select_design_dataset(
    model: ConsensusModel, condition: dict
) -> tuple[ConsensusDataset, SplineModel]:
    matches = [
        dataset
        for dataset in model.datasets
        if dataset.motion_key.motion_type == condition["motion_type"]
        and dataset.motion_key.motion_plane == condition["motion_plane"]
        and dataset.motion_key.direction == condition["direction"]
        and dataset.motion_key.loaded is condition["loaded"]
        and dataset.motion_key.healthy_only is condition["healthy_only"]
        and dataset.variable == condition["primary_variable"]
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Design condition matched {len(matches)} datasets; exactly one is required."
        )
    dataset = matches[0]
    allowed = set(condition["allowed_papers"])
    contributors = {curve.paper_id for curve in dataset.source_curves}
    if not contributors or not contributors <= allowed:
        raise ValueError(
            f"Unexpected design contributors: {sorted(contributors - allowed)}"
        )
    if not dataset.conventions_verified:
        raise ValueError("Selected design dataset has unverified coordinate conventions.")
    requested_range = tuple(float(value) for value in condition["supported_range_deg"])
    actual_range = (
        float(np.nanmin(dataset.elevation_deg)),
        float(np.nanmax(dataset.elevation_deg)),
    )
    if requested_range[0] < actual_range[0] or requested_range[1] > actual_range[1]:
        raise ValueError(
            f"Requested range {requested_range} exceeds literature range {actual_range}."
        )
    spline = next(
        item
        for item in model.splines
        if item.motion_key == dataset.motion_key and item.variable == dataset.variable
    )
    return dataset, spline
