"""Full-precision synthesis-data CSV export."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from simulation import Simulation


class CsvExporter:
    """Export every synthesis sample and both local pitch-curve coordinates."""

    def export(self, simulation: Simulation, destination: str | Path) -> Path:
        path = Path(destination)
        data = simulation.pitch_data
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                (
                    "input_rotation_deg",
                    "output_rotation_deg",
                    "instantaneous_gear_ratio",
                    "input_pitch_radius",
                    "output_pitch_radius",
                    "input_local_x",
                    "input_local_y",
                    "output_local_x",
                    "output_local_y",
                    "radius_sum",
                )
            )
            for index in range(len(data.input_rad)):
                writer.writerow(
                    (
                        f"{np.degrees(data.input_rad[index]):.15g}",
                        f"{np.degrees(data.output_rad[index]):.15g}",
                        f"{data.ratio[index]:.15g}",
                        f"{data.input_radii[index]:.15g}",
                        f"{data.output_radii[index]:.15g}",
                        f"{data.input_points[index, 0]:.15g}",
                        f"{data.input_points[index, 1]:.15g}",
                        f"{data.output_points[index, 0]:.15g}",
                        f"{data.output_points[index, 1]:.15g}",
                        f"{data.input_radii[index] + data.output_radii[index]:.15g}",
                    )
                )
        return path
