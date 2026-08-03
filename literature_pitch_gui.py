"""Viewer for the original fixed-axis and new stationary-sun pathways."""

from __future__ import annotations

import csv
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from literature_planetary import (
    PlanetaryPitchCurveData,
    assembled_planet_points,
    rotate_points,
    synthesize_literature_planetary_pitch_curves,
)


FIXED_MODE = "Literature fixed-axis pitch curves"
PLANETARY_MODE = "Literature planetary pitch curves"


@dataclass(frozen=True)
class FixedAxisPitchMotion:
    elevation_deg: np.ndarray
    input_points: np.ndarray
    output_points: np.ndarray
    input_angle_rad: np.ndarray
    output_angle_rad: np.ndarray
    input_radius_mm: np.ndarray
    ratio: np.ndarray
    center_distance_mm: float

    @classmethod
    def load(cls, path: str | Path) -> "FixedAxisPitchMotion":
        with Path(path).open(newline="", encoding="utf-8") as stream:
            if stream.readline().startswith("RESEARCH"):
                reader = csv.DictReader(stream)
            else:
                stream.seek(0)
                reader = csv.DictReader(stream)
            rows = [row for row in reader if row["candidate"] == "regularized"]
        if not rows:
            raise ValueError("Committed fixed-axis literature curve is missing.")
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
            raise ValueError("Fixed-axis center distance is inconsistent.")
        input_angle = -np.unwrap(np.arctan2(input_points[:, 1], input_points[:, 0]))
        output_angle = np.unwrap(
            np.arctan2(output_points[:, 1], output_points[:, 0]) - np.pi
        )
        return cls(
            elevation_deg=elevation,
            input_points=input_points,
            output_points=output_points,
            input_angle_rad=input_angle,
            output_angle_rad=output_angle,
            input_radius_mm=input_radius,
            ratio=np.gradient(output_angle, input_angle),
            center_distance_mm=float(center[0]),
        )

    def index_at(self, elevation_deg: float) -> int:
        return int(np.argmin(np.abs(self.elevation_deg - elevation_deg)))


class LiteraturePitchGUI:
    """Display either literature pathway without generating teeth."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Literature Shoulder Pitch Curves")
        base = Path(__file__).resolve().parent / "validation_outputs"
        self.fixed = FixedAxisPitchMotion.load(base / "LiteratureSectorPitchCurves.csv")
        self.planetary: PlanetaryPitchCurveData = (
            synthesize_literature_planetary_pitch_curves(
                base / "LiteratureSectorTransmission.csv"
            )
        )
        self.mode = tk.StringVar(value=FIXED_MODE)
        self.angle = tk.DoubleVar(value=11.0)
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
            controls,
            text="McClure2001 literature model",
            font=("TkDefaultFont", 13, "bold"),
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        selector = ttk.Combobox(
            controls,
            textvariable=self.mode,
            values=(FIXED_MODE, PLANETARY_MODE),
            state="readonly",
            width=31,
        )
        selector.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        selector.bind("<<ComboboxSelected>>", lambda _event: self._render())

        ttk.Scale(
            controls,
            from_=11.0,
            to=147.0,
            variable=self.angle,
            command=lambda _value: self._render(),
            length=250,
        ).grid(row=2, column=0, columnspan=2, sticky="ew")

        self.name_labels: list[ttk.Label] = []
        self.value_labels: list[ttk.Label] = []
        for row in range(3, 9):
            name = ttk.Label(controls)
            name.grid(row=row, column=0, sticky="w", pady=3)
            value = ttk.Label(controls, font=("TkDefaultFont", 10, "bold"))
            value.grid(row=row, column=1, sticky="e", pady=3)
            self.name_labels.append(name)
            self.value_labels.append(value)

        self.figure = Figure(figsize=(9.0, 6.5), dpi=100)
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=view)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def _set_values(self, pairs: tuple[tuple[str, str], ...]) -> None:
        for name_label, value_label, (name, value) in zip(
            self.name_labels, self.value_labels, pairs, strict=True
        ):
            name_label.configure(text=name)
            value_label.configure(text=value)

    def _render(self) -> None:
        if self.mode.get() == PLANETARY_MODE:
            self._render_planetary()
        else:
            self._render_fixed()
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _render_fixed(self) -> None:
        data = self.fixed
        index = data.index_at(self.angle.get())
        phi = data.input_angle_rad[index]
        psi = data.output_angle_rad[index]
        input_path = rotate_points(data.input_points, phi)
        output_path = rotate_points(data.output_points, -psi)
        output_path[:, 0] += data.center_distance_mm
        contact = np.array([data.input_radius_mm[index], 0.0])

        axes = self.axes
        axes.clear()
        axes.plot(input_path[:, 0], input_path[:, 1], color="#e58a2b", linewidth=3)
        axes.plot(output_path[:, 0], output_path[:, 1], color="#2878b5", linewidth=3)
        axes.scatter([0, data.center_distance_mm], [0, 0], s=80, zorder=4)
        axes.scatter(*contact, color="#20b95b", s=75, zorder=5)
        self._finish_axes()
        self._set_values(
            (
                ("Elevation", f"{data.elevation_deg[index]:.1f}°"),
                ("Input rotation", f"{np.degrees(phi):.1f}°"),
                ("Output rotation", f"{np.degrees(psi):.1f}°"),
                ("Ratio", f"{data.ratio[index]:.3f}:1"),
                ("Center distance", f"{data.center_distance_mm:.1f} mm"),
                ("Pathway", "Fixed-axis"),
            )
        )

    def _render_planetary(self) -> None:
        data = self.planetary
        index = data.index_at(self.angle.get())
        planet_path = assembled_planet_points(data, index)
        planet_center = data.planet_center_points_world[index]
        contact = data.contact_points_world[index]

        sun_tangents = np.gradient(data.sun_pitch_points_local, axis=0)
        tangent = sun_tangents[index] / np.linalg.norm(sun_tangents[index])
        normal = np.array([-tangent[1], tangent[0]])
        line_length = 20.0

        axes = self.axes
        axes.clear()
        axes.plot(
            data.sun_pitch_points_local[:, 0],
            data.sun_pitch_points_local[:, 1],
            color="#e58a2b",
            linewidth=3,
        )
        axes.plot(planet_path[:, 0], planet_path[:, 1], color="#2878b5", linewidth=3)
        axes.plot(
            [0.0, planet_center[0]],
            [0.0, planet_center[1]],
            color="#696969",
            linewidth=5,
            solid_capstyle="round",
        )
        axes.scatter(
            [0.0, planet_center[0]], [0.0, planet_center[1]], s=85, zorder=4
        )
        axes.scatter(*contact, color="#20b95b", s=80, zorder=5)
        axes.plot(
            [contact[0] - tangent[0] * line_length, contact[0] + tangent[0] * line_length],
            [contact[1] - tangent[1] * line_length, contact[1] + tangent[1] * line_length],
            color="#7f3fbf",
            linewidth=2,
        )
        axes.plot(
            [contact[0] - normal[0] * line_length, contact[0] + normal[0] * line_length],
            [contact[1] - normal[1] * line_length, contact[1] + normal[1] * line_length],
            color="#2f9b66",
            linewidth=2,
        )
        self._finish_axes()
        self._set_values(
            (
                ("Shoulder target", f"{data.elevation_deg[index]:.1f}°"),
                ("Scapular carrier", f"{np.degrees(data.carrier_angle_rad[index]):.1f}°"),
                ("Sun angle", "0.0°"),
                ("Planet absolute", f"{np.degrees(data.planet_absolute_angle_rad[index]):.1f}°"),
                ("Signed ratio", f"{data.signed_ratio[index]:.3f}"),
                ("Center distance", f"{data.center_distance_mm:.1f} mm"),
            )
        )

    def _finish_axes(self) -> None:
        self.axes.set_aspect("equal", adjustable="datalim")
        self.axes.grid(alpha=0.22)
        self.axes.margins(0.20)
