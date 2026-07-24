"""Configurable, auditable study-quality weighting."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from .models import PaperMetadata


@dataclass(frozen=True)
class StudyWeight:
    paper_id: str
    total: float
    factors: dict[str, float]
    rationale: tuple[str, ...]


class StudyWeightingSystem:
    """Compute weights entirely from an editable JSON policy."""

    def __init__(self, configuration: dict) -> None:
        self.configuration = configuration

    @classmethod
    def from_json(cls, path: str | Path) -> "StudyWeightingSystem":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def weight(self, paper: PaperMetadata, motion_type: str) -> StudyWeight:
        config = self.configuration
        factors: dict[str, float] = {}
        rationale: list[str] = []
        method = paper.measurement_method.lower()
        method_factor = float(config["measurement_method"]["default"])
        for rule in config["measurement_method"]["rules"]:
            if any(keyword.lower() in method for keyword in rule["keywords"]):
                method_factor = float(rule["factor"])
                rationale.append(rule["label"])
                break
        factors["measurement_method"] = method_factor

        healthy_key = (
            "healthy" if paper.healthy_only is True
            else "mixed" if paper.healthy_only is False
            else "unknown"
        )
        factors["population"] = float(config["population"][healthy_key])
        dynamic = "dynamic" in motion_type.lower()
        factors["motion_protocol"] = float(
            config["motion_protocol"]["dynamic" if dynamic else "quasi_static"]
        )
        if paper.sample_size:
            exponent = float(config["sample_size"]["exponent"])
            reference = float(config["sample_size"]["reference"])
            raw = (paper.sample_size / reference) ** exponent
            factors["sample_size"] = min(
                float(config["sample_size"]["maximum"]),
                max(float(config["sample_size"]["minimum"]), raw),
            )
        else:
            factors["sample_size"] = float(config["sample_size"]["unknown"])
        total = math.prod(factors.values())
        return StudyWeight(paper.paper_id, total, factors, tuple(rationale))
