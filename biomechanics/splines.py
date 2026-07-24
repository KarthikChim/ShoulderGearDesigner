"""Shape-preserving spline fitting and analytical derivatives."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator

from .models import ConsensusDataset, SplineModel


def fit_shape_preserving_splines(
    datasets: tuple[ConsensusDataset, ...],
) -> tuple[SplineModel, ...]:
    models: list[SplineModel] = []
    for dataset in datasets:
        valid = np.isfinite(dataset.mean_deg)
        x = dataset.elevation_deg[valid]
        y = dataset.mean_deg[valid]
        if len(x) < 2:
            continue
        spline = PchipInterpolator(x, y, extrapolate=False)
        models.append(
            SplineModel(
                motion_key=dataset.motion_key,
                variable=dataset.variable,
                knots=spline.x.copy(),
                coefficients=spline.c.copy(),
            )
        )
    return tuple(models)
