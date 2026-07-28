"""Selectable legacy and literature shoulder-model pathways."""

from __future__ import annotations

from pathlib import Path

from biomechanics.literature_model import LiteratureShoulderModel
from kinematics import ShoulderModel
from settings import Settings


def create_shoulder_model(
    pathway: str,
    settings: Settings | None = None,
    consensus_json: str | Path = "ConsensusShoulderModel.json",
    *,
    allow_approximate_gh: bool = False,
):
    """Create either pathway without coupling their source data or assumptions."""

    if pathway == "legacy":
        configuration = settings or Settings()
        return ShoulderModel(
            configuration.ratio_regions,
            configuration.max_elevation_deg,
            configuration.input_revolutions_per_elevation_cycle,
        )
    if pathway == "literature":
        return LiteratureShoulderModel(
            consensus_json, allow_approximate_gh=allow_approximate_gh
        )
    raise ValueError("pathway must be 'legacy' or 'literature'")
