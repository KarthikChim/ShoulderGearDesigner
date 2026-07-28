"""Exact SVG export for synthesized pitch curves."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import svgwrite

from simulation import Simulation


class SvgExporter:
    """Export exact double-precision pitch-curve samples as SVG polylines."""

    def export(self, simulation: Simulation, destination: str | Path) -> Path:
        path = Path(destination)
        data = simulation.pitch_data
        output = data.output_points + np.array([data.center_distance, 0.0])
        teeth = simulation.generated_pair.input_gear.polygon
        output_teeth = (
            simulation.generated_pair.output_gear.polygon
            + np.array([data.center_distance, 0.0])
        )
        all_points = np.vstack((teeth, output_teeth))
        minimum = np.min(all_points, axis=0)
        maximum = np.max(all_points, axis=0)
        margin = data.center_distance * 0.08
        width, height = maximum - minimum + 2.0 * margin
        drawing = svgwrite.Drawing(
            path,
            size=(f"{width}mm", f"{height}mm"),
            viewBox=f"0 0 {width} {height}",
        )

        def convert(points: np.ndarray) -> list[tuple[float, float]]:
            return [
                (
                    float(point[0] - minimum[0] + margin),
                    float(maximum[1] - point[1] + margin),
                )
                for point in points
            ]

        drawing.add(drawing.polygon(
            convert(teeth), fill="none", stroke="#111111", stroke_width=0.25,
            id="input-complete-tooth-boundary",
        ))
        drawing.add(drawing.polyline(
            convert(data.input_points), fill="none", stroke="#e47d24",
            stroke_width=0.15, id="input-pitch-curve",
        ))
        drawing.add(drawing.polygon(
            convert(output_teeth), fill="none", stroke="#1e5d91",
            stroke_width=0.25, id="output-complete-tooth-boundary",
        ))
        drawing.add(
            drawing.polyline(
                convert(output),
                fill="none",
                stroke="#397eaf",
                stroke_width=0.35,
                id="output-pitch-curve",
            )
        )
        drawing.save()
        return path
