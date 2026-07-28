"""Minimal literature-only viewer for the finalized shoulder pitch paths.

This module deliberately contains no tooth, rack-cutter, optimization, legacy,
or manufacturing code.  It reads the committed McClure2001 regularized pitch
curves and displays those curves in their operating pose.
"""

from __future__ import annotations

import csv
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


@dataclass(frozen=True)
class LiteraturePitchMotion:
    """The committed regularized pitch paths and their sampled motion."""

    elevation_deg: np.ndarray
    input_points: np.ndarray
    output_points: np.ndarray
    input_angle_rad: np.ndarray
    output_angle_rad: np.ndarray
    input_radius_mm: np.ndarray
    output_radius_mm: np.ndarray
    ratio: np.ndarray
    center_distance_mm: float

    @classmethod
    def load(cls, path: str | Path) -> "LiteraturePitchMotion":
        with Path(path).open(newline="", encoding="utf-8") as stream:
            if stream.readline().startswith("RESEARCH"):
                reader = csv.DictReader(stream)
            else:
                stream.seek(0)
                reader = csv.DictReader(stream)
            rows = [row for row in reader if row["candidate"] == "regularized"]
        if not rows:
            raise ValueError("The committed regularized literature curve is missing.")

        elevation = np.asarray([float(row["ht_elevation_deg"]) for row in rows])
        input_points = np.asarray(
            [[float(row["input_x"]), float(row["input_y"])] for row in rows]
        )
        output_points = np.asarray(
            [
                [float(row["output_local_x"]), float(row["output_local_y"])]
                for row in rows
            ]
        )
        input_radius = np.asarray([float(row["input_radius"]) for row in rows])
        output_radius = np.asarray([float(row["output_radius"]) for row in rows])
        center = np.asarray([float(row["center_distance"]) for row in rows])
        if not np.allclose(input_radius + output_radius, center, atol=1e-8):
            raise ValueError("Committed pitch radii do not preserve center distance.")

        input_angle = -np.unwrap(np.arctan2(input_points[:, 1], input_points[:, 0]))
        output_angle = np.unwrap(
            np.arctan2(output_points[:, 1], output_points[:, 0]) - np.pi
        )
        ratio = np.gradient(output_angle, input_angle)
        return cls(
            elevation_deg=elevation,
            input_points=input_points,
            output_points=output_points,
            input_angle_rad=input_angle,
            output_angle_rad=output_angle,
            input_radius_mm=input_radius,
            output_radius_mm=output_radius,
            ratio=ratio,
            center_distance_mm=float(center[0]),
        )

    def index_at(self, elevation_deg: float) -> int:
        return int(np.argmin(np.abs(self.elevation_deg - elevation_deg)))


def _rotate(points: np.ndarray, angle_rad: float) -> np.ndarray:
    cosine = np.cos(angle_rad)
    sine = np.sin(angle_rad)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    return points @ rotation.T


class LiteraturePitchGUI:
    """Small viewer showing only the verified literature pitch curves."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Literature Shoulder Pitch Curves")
        source = (
            Path(__file__).resolve().parent
            / "validation_outputs"
            / "LiteratureSectorPitchCurves.csv"
        )
        self.motion = LiteraturePitchMotion.load(source)
        self.angle = tk.DoubleVar(value=float(self.motion.elevation_deg[0]))
        self._build()
        self._render()

    def _build(self) -> None:
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
        controls = ttk.Frame(self.root, padding=16)
        controls.grid(row=0, column=0, sticky="ns")
        view = ttk.Frame(self.root, padding=(0, 8, 8, 8))
        view.grid(row=0, column=1, sticky="nsew")
        view.columnconfigure(0, weight=1)
        view.rowconfigure(0, weight=1)

        ttk.Label(
            controls, text="McClure2001 literature model",
            font=("TkDefaultFont", 13, "bold"),
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 14))
        row = 1
        ttk.Scale(
            controls,
            from_=float(self.motion.elevation_deg[0]),
            to=float(self.motion.elevation_deg[-1]),
            variable=self.angle,
            command=lambda _value: self._render(),
            length=230,
        ).grid(row=row, column=0, columnspan=2, sticky="ew")
        row += 1

        self.value_labels: dict[str, ttk.Label] = {}
        for name in (
            "Elevation",
            "Input rotation",
            "Output rotation",
            "Instantaneous ratio",
            "Center distance",
        ):
            ttk.Label(controls, text=name).grid(row=row, column=0, sticky="w", pady=3)
            label = ttk.Label(controls, font=("TkDefaultFont", 10, "bold"))
            label.grid(row=row, column=1, sticky="e", pady=3)
            self.value_labels[name] = label
            row += 1

        self.figure = Figure(figsize=(9.0, 6.5), dpi=100)
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=view)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def _render(self) -> None:
        index = self.motion.index_at(self.angle.get())
        phi = self.motion.input_angle_rad[index]
        psi = self.motion.output_angle_rad[index]
        input_path = _rotate(self.motion.input_points, phi)
        output_path = _rotate(self.motion.output_points, -psi)
        output_path[:, 0] += self.motion.center_distance_mm

        axes = self.axes
        axes.clear()
        axes.plot(
            input_path[:, 0], input_path[:, 1],
            color="#e58a2b", linewidth=3.0,
        )
        axes.plot(
            output_path[:, 0], output_path[:, 1],
            color="#2878b5", linewidth=3.0,
        )
        axes.scatter(
            [0.0, self.motion.center_distance_mm], [0.0, 0.0],
            color=["#b75d00", "#075c9c"], s=70, zorder=4,
        )
        contact = self.motion.input_radius_mm[index]
        axes.scatter([contact], [0.0], color="#20b95b", s=80, zorder=5)
        axes.set_aspect("equal", adjustable="datalim")
        axes.grid(alpha=0.22)
        axes.margins(0.12)
        self.figure.tight_layout()
        self.canvas.draw_idle()

        self.value_labels["Elevation"].configure(
            text=f"{self.motion.elevation_deg[index]:.1f}°"
        )
        self.value_labels["Input rotation"].configure(
            text=f"{np.degrees(phi):.1f}°"
        )
        self.value_labels["Output rotation"].configure(
            text=f"{np.degrees(psi):.1f}°"
        )
        self.value_labels["Instantaneous ratio"].configure(
            text=f"{self.motion.ratio[index]:.3f}:1"
        )
        self.value_labels["Center distance"].configure(
            text=f"{self.motion.center_distance_mm:.1f} mm"
        )
