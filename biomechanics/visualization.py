"""Interactive literature/consensus plots with per-paper toggles."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons

from .models import ConsensusDataset


@dataclass
class ConsensusPlot:
    figure: object
    axes: object
    toggles: CheckButtons


def plot_consensus(dataset: ConsensusDataset, show: bool = True) -> ConsensusPlot:
    figure, axes = plt.subplots(figsize=(10, 6))
    figure.subplots_adjust(right=0.79)
    paper_lines = {}
    for curve in dataset.source_curves:
        line, = axes.plot(
            curve.elevation_deg,
            curve.value_deg,
            marker=".",
            linewidth=1.0,
            alpha=0.55,
            label=f"{curve.paper_id} (w={curve.study_weight:.2f})",
        )
        paper_lines.setdefault(curve.paper_id, []).append(line)
    axes.fill_between(
        dataset.elevation_deg,
        dataset.uncertainty.confidence_lower,
        dataset.uncertainty.confidence_upper,
        color="#6d70b3",
        alpha=0.18,
        label="95% confidence envelope",
    )
    axes.plot(
        dataset.elevation_deg,
        dataset.mean_deg,
        color="#392f8a",
        linewidth=2.5,
        label="Weighted consensus / PCHIP knots",
    )
    axes.set_title(f"{dataset.variable}\n{dataset.motion_key.identifier}")
    axes.set_xlabel("HT elevation (deg)")
    axes.set_ylabel("Angle (deg)")
    axes.grid(True, alpha=0.3)
    axes.legend(fontsize=8, loc="best")
    toggle_axes = figure.add_axes((0.81, 0.25, 0.18, 0.5))
    labels = list(paper_lines)
    toggles = CheckButtons(toggle_axes, labels, [True] * len(labels))

    def changed(label: str) -> None:
        visible = not paper_lines[label][0].get_visible()
        for line in paper_lines[label]:
            line.set_visible(visible)
        figure.canvas.draw_idle()

    toggles.on_clicked(changed)
    if show:
        plt.show()
    return ConsensusPlot(figure, axes, toggles)
