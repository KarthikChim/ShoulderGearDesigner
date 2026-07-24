"""Coordinate-convention registry and provenance-preserving transforms."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .models import (
    CoordinateConvention,
    NormalizedObservation,
    RawLiteratureRow,
    Transformation,
)


class CoordinateNormalizer:
    """Create transformed copies; raw literature rows are never mutated."""

    def __init__(self, configuration: dict) -> None:
        self.configuration = configuration

    @classmethod
    def from_json(cls, path: str | Path) -> "CoordinateNormalizer":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def normalize(
        self, rows: tuple[RawLiteratureRow, ...]
    ) -> tuple[NormalizedObservation, ...]:
        observations: list[NormalizedObservation] = []
        transforms = self.configuration.get("transforms", {})
        conventions = self.configuration.get("conventions", {})
        for row in rows:
            if row.ht_elevation_deg is None:
                continue
            for variable, value in row.values:
                key = f"{row.paper_id}:{variable}"
                item = transforms.get(key, transforms.get(f"*:{variable}", {}))
                scale = float(item.get("scale", 1.0))
                offset = float(item.get("offset_deg", 0.0))
                transformation = Transformation(
                    transformation_id=item.get("id", f"identity:{key}"),
                    scale=scale,
                    offset_deg=offset,
                    description=item.get(
                        "description",
                        "Identity transform; published sign convention awaits explicit review.",
                    ),
                )
                convention_item = conventions.get(
                    key, conventions.get(f"*:{variable}", {})
                )
                convention = CoordinateConvention(
                    convention_id=convention_item.get("id", f"published:{key}"),
                    variable=variable,
                    description=convention_item.get(
                        "description", "Original published convention."
                    ),
                    positive_direction=convention_item.get(
                        "positive_direction", "unverified"
                    ),
                    reference_frame=convention_item.get(
                        "reference_frame", "as published"
                    ),
                    verified=bool(convention_item.get("verified", False)),
                )
                uncertainty_match = re.search(
                    r"uncertainty\s*(?:\+/-|±)\s*([0-9]*\.?[0-9]+)\s*deg",
                    row.notes,
                    flags=re.IGNORECASE,
                )
                note_uncertainty = (
                    float(uncertainty_match.group(1))
                    if uncertainty_match
                    else None
                )
                observations.append(
                    NormalizedObservation(
                        row_number=row.row_number,
                        paper_id=row.paper_id,
                        figure_or_table=row.figure_or_table,
                        motion_key=row.motion_key,
                        variable=variable,
                        ht_elevation_deg=row.ht_elevation_deg,
                        original_value=value,
                        original_convention=convention,
                        transformation=transformation,
                        normalized_value=transformation.apply(value),
                        sd=row.sd,
                        sem=row.sem,
                        reported_uncertainty_deg=note_uncertainty,
                        extraction_method=row.extraction_method,
                        notes=row.notes,
                    )
                )
        return tuple(observations)
