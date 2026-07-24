"""Exact DXF export for synthesized pitch curves."""

from __future__ import annotations

from pathlib import Path

import ezdxf
import numpy as np

from simulation import Simulation


class DxfExporter:
    """Export exact conjugate pitch curves, centers, and center line."""

    def export(self, simulation: Simulation, destination: str | Path) -> Path:
        path = Path(destination)
        data = simulation.pitch_data
        document = ezdxf.new("R2010")
        document.header["$INSUNITS"] = 4
        model = document.modelspace()
        output = data.output_points + np.array([data.center_distance, 0.0])
        model.add_lwpolyline(
            simulation.generated_gear.polygon.tolist(),
            close=True,
            dxfattribs={"layer": "INPUT_COMPLETE_TEETH"},
        )
        output_teeth = (
            simulation.generated_pair.output_gear.polygon
            + np.array([data.center_distance, 0.0])
        )
        model.add_lwpolyline(
            output_teeth.tolist(),
            close=True,
            dxfattribs={"layer": "OUTPUT_COMPLETE_TEETH"},
        )
        model.add_lwpolyline(
            data.input_points.tolist(),
            close=True,
            dxfattribs={"layer": "INPUT_PITCH_CURVE"},
        )
        model.add_lwpolyline(
            output.tolist(),
            close=True,
            dxfattribs={"layer": "OUTPUT_PITCH_CURVE"},
        )
        model.add_point((0.0, 0.0), dxfattribs={"layer": "SHAFT_CENTERS"})
        model.add_point(
            (data.center_distance, 0.0), dxfattribs={"layer": "SHAFT_CENTERS"}
        )
        model.add_line(
            (0.0, 0.0),
            (data.center_distance, 0.0),
            dxfattribs={"layer": "CENTER_DISTANCE"},
        )
        document.saveas(path)
        return path
