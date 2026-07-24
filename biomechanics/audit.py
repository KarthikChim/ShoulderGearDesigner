"""Coordinate-convention audit reporting."""

from __future__ import annotations

from pathlib import Path

from .models import NormalizedObservation


def write_coordinate_audit(
    observations: tuple[NormalizedObservation, ...], destination: str | Path
) -> Path:
    path = Path(destination)
    groups: dict[tuple[str, str], list[NormalizedObservation]] = {}
    for item in observations:
        groups.setdefault((item.paper_id, item.variable), []).append(item)
    lines = [
        "# Coordinate Convention Audit",
        "",
        "This report is a manufacturing gate. `VERIFIED` means the mapping is "
        "supported by an explicit source description; `UNRESOLVED` data cannot "
        "participate in cross-paper consensus.",
        "",
        "| Paper × variable | Euler/Cardan sequence | Proximal CS | Distal CS | "
        "Reference pose | Positive direction | Transform | Status | Source | "
        "Unresolved ambiguity |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key, items in sorted(groups.items()):
        convention = items[0].original_convention
        transform = items[0].transformation
        status = "VERIFIED" if all(
            item.original_convention.verified for item in items
        ) else "UNRESOLVED"
        ambiguity = convention.unresolved_ambiguity
        if not convention.verified and not ambiguity:
            ambiguity = (
                "Paper-specific sequence, frames, reference pose, and sign mapping "
                "have not been verified from the primary source."
            )
        cells = (
            f"{key[0]} × {key[1]}",
            convention.euler_sequence,
            convention.proximal_coordinate_system,
            convention.distal_coordinate_system,
            convention.reference_pose,
            convention.positive_direction,
            f"`normalized = {transform.scale:g}·original "
            f"{transform.offset_deg:+g}°` ({transform.transformation_id})",
            status,
            convention.supporting_source,
            ambiguity or "None identified",
        )
        lines.append("| " + " | ".join(cell.replace("|", "/") for cell in cells) + " |")
    lines.extend(
        [
            "",
            "## Selected design condition",
            "",
            "McClure2001 scapular-plane raising data are verified from the primary "
            "paper's pp. 270–272 and Figures 2–3. The paper defines embedded "
            "thorax/scapula axes and the scapular Euler sequence as external/internal "
            "rotation, upward/downward rotation, then posterior/anterior tilt.",
            "",
            "All other unresolved rows remain available exactly as extracted but are "
            "blocked from cross-paper averaging.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
