"""Tkinter desktop interface for Shoulder Gear Designer."""

from __future__ import annotations

import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from drawing import Renderer
from export_csv import CsvExporter
from export_dxf import DxfExporter
from export_svg import SvgExporter
from settings import Settings
from simulation import Simulation


class ShoulderGearDesignerGUI:
    """Application window, controls, viewport, and live-value display."""

    def __init__(self, root: tk.Tk, settings: Settings | None = None) -> None:
        self.root = root
        self.root.title("Shoulder Gear Designer")
        self.root.minsize(1100, 700)
        self.settings = settings or Settings()
        self.simulation = Simulation(self.settings)
        self._last_tick = time.perf_counter()
        self._fps = 0.0
        self._camera_limits: tuple[tuple[float, float], tuple[float, float]] | None = None
        self._pan_anchor: tuple[float, float] | None = None

        self._create_variables()
        self._build_layout()
        self._connect_canvas_events()
        self._render()
        self._schedule_tick()

    def _create_variables(self) -> None:
        self.center_distance_var = tk.DoubleVar(value=self.settings.center_distance)
        self.speed_var = tk.DoubleVar(value=self.settings.simulation_speed_deg_s)
        self.angle_var = tk.DoubleVar(value=0.0)
        self.advanced_debug_var = tk.BooleanVar(value=self.settings.advanced_debug)

        self.live_vars = {
            "Mechanical ratio": tk.StringVar(),
            "Current GH": tk.StringVar(),
            "Current ST": tk.StringVar(),
            "Actual GH:ST": tk.StringVar(),
            "Target GH:ST": tk.StringVar(),
            "GH error": tk.StringVar(),
            "ST error": tk.StringVar(),
            "Ratio error": tk.StringVar(),
        }

    def _build_layout(self) -> None:
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        controls_shell = ttk.Frame(self.root)
        controls_shell.grid(row=0, column=0, sticky="nsew")
        controls_shell.rowconfigure(0, weight=1)
        controls_shell.columnconfigure(0, weight=1)
        controls_canvas = tk.Canvas(
            controls_shell, width=285, highlightthickness=0, borderwidth=0
        )
        controls_canvas.grid(row=0, column=0, sticky="nsew")
        controls_scroll = ttk.Scrollbar(
            controls_shell, orient="vertical", command=controls_canvas.yview
        )
        controls_scroll.grid(row=0, column=1, sticky="ns")
        controls_canvas.configure(yscrollcommand=controls_scroll.set)
        controls = ttk.Frame(controls_canvas, padding=12)
        controls_window = controls_canvas.create_window(
            (0, 0), window=controls, anchor="nw"
        )
        controls.bind(
            "<Configure>",
            lambda event: controls_canvas.configure(
                scrollregion=controls_canvas.bbox("all")
            ),
        )
        controls_canvas.bind(
            "<Configure>",
            lambda event: controls_canvas.itemconfigure(
                controls_window, width=event.width
            ),
        )
        viewport = ttk.Frame(self.root, padding=(0, 8, 8, 0))
        viewport.grid(row=0, column=1, sticky="nsew")
        viewport.columnconfigure(0, weight=1)
        viewport.rowconfigure(0, weight=1)

        ttk.Label(controls, text="Inputs", font=("TkDefaultFont", 13, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )
        row = 1
        for label, variable in (
            ("Center Distance", self.center_distance_var),
            ("Simulation Speed", self.speed_var),
        ):
            ttk.Label(controls, text=label).grid(row=row, column=0, sticky="w", pady=3)
            ttk.Entry(controls, textvariable=variable, width=12).grid(
                row=row, column=1, sticky="ew", pady=3
            )
            row += 1

        ttk.Button(controls, text="Apply Geometry", command=self._apply_inputs).grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=(6, 12)
        )
        row += 1

        button_commands: tuple[tuple[str, Callable[[], None]], ...] = (
            ("Start", self.simulation.animator.start),
            ("Pause", self.simulation.animator.pause),
            ("Reset", self._reset),
            ("Step Forward", lambda: self._step(1.0)),
            ("Step Backward", lambda: self._step(-1.0)),
        )
        for label, command in button_commands:
            ttk.Button(controls, text=label, command=command).grid(
                row=row, column=0, columnspan=2, sticky="ew", pady=2
            )
            row += 1

        ttk.Label(controls, text="Arm elevation").grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(12, 0)
        )
        row += 1
        self.angle_slider = ttk.Scale(
            controls,
            from_=0.0,
            to=self.settings.max_elevation_deg,
            variable=self.angle_var,
            command=self._slider_changed,
        )
        self.angle_slider.grid(row=row, column=0, columnspan=2, sticky="ew")
        row += 1

        ttk.Separator(controls).grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=10
        )
        row += 1
        ttk.Checkbutton(
            controls,
            text="Advanced Debug",
            variable=self.advanced_debug_var,
            command=self._display_changed,
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
        row += 1

        ttk.Button(controls, text="Reset Camera", command=self._reset_camera).grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=(12, 2)
        )
        row += 1
        ttk.Button(controls, text="Export SVG", command=self._export_svg).grid(
            row=row, column=0, sticky="ew", pady=2
        )
        ttk.Button(controls, text="Export DXF", command=self._export_dxf).grid(
            row=row, column=1, sticky="ew", pady=2
        )
        row += 1
        ttk.Button(controls, text="Export CSV", command=self._export_csv).grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=2
        )
        row += 1
        self.validation_label = ttk.Label(
            controls,
            justify="left",
            font=("TkDefaultFont", 8),
            padding=(0, 10),
        )
        self.validation_label.grid(row=row, column=0, columnspan=2, sticky="w")

        self.figure = Figure(figsize=(10, 7), dpi=100, constrained_layout=True)
        grid = self.figure.add_gridspec(
            2, 2, width_ratios=(2.35, 1.35), height_ratios=(1.0, 1.0)
        )
        self.axes = self.figure.add_subplot(grid[:, 0])
        self.ratio_axes = self.figure.add_subplot(grid[0, 1])
        self.contribution_axes = self.figure.add_subplot(grid[1, 1])
        self.renderer = Renderer(
            self.axes, self.ratio_axes, self.contribution_axes
        )
        self.canvas = FigureCanvasTkAgg(self.figure, master=viewport)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        values = ttk.Frame(viewport, padding=(8, 6))
        values.grid(row=1, column=0, sticky="ew")
        for column, (label, variable) in enumerate(self.live_vars.items()):
            values.columnconfigure(column, weight=1)
            ttk.Label(values, text=label, font=("TkDefaultFont", 8)).grid(
                row=0, column=column
            )
            ttk.Label(values, textvariable=variable, font=("TkDefaultFont", 10, "bold")).grid(
                row=1, column=column
            )

    def _connect_canvas_events(self) -> None:
        self.canvas.mpl_connect("scroll_event", self._zoom)
        self.canvas.mpl_connect("button_press_event", self._pan_start)
        self.canvas.mpl_connect("button_release_event", self._pan_end)
        self.canvas.mpl_connect("motion_notify_event", self._pan_move)

    def _apply_inputs(self) -> None:
        try:
            self.settings.center_distance = self.center_distance_var.get()
            self.settings.simulation_speed_deg_s = self.speed_var.get()
            self.settings.validate()
            self.simulation.rebuild_geometry()
            self._camera_limits = None
            self._render()
        except (tk.TclError, ValueError) as error:
            messagebox.showerror("Invalid settings", str(error))

    def _display_changed(self) -> None:
        enabled = self.advanced_debug_var.get()
        self.settings.advanced_debug = enabled
        self.settings.show_ratio_graph = enabled
        self.settings.show_pitch_curves = enabled
        self._render()

    def _slider_changed(self, raw_value: str) -> None:
        self.simulation.animator.pause()
        self.simulation.set_elevation(float(raw_value))
        self._render()

    def _step(self, amount: float) -> None:
        self.simulation.animator.pause()
        self.simulation.animator.step(amount)
        self.simulation.set_elevation(self.simulation.animator.current_deg)
        self.angle_var.set(self.simulation.animator.current_deg)
        self._render()

    def _reset(self) -> None:
        self.simulation.animator.reset()
        self.simulation.set_elevation(0.0)
        self.angle_var.set(0.0)
        self._render()

    def _schedule_tick(self) -> None:
        self.root.after(self.settings.frame_interval_ms, self._tick)

    def _tick(self) -> None:
        now = time.perf_counter()
        elapsed = min(0.1, now - self._last_tick)
        self._last_tick = now
        if elapsed > 0:
            instantaneous_fps = 1.0 / elapsed
            self._fps = instantaneous_fps if self._fps == 0 else 0.9 * self._fps + 0.1 * instantaneous_fps
        if self.simulation.animator.playing:
            self.simulation.update(elapsed)
            self.angle_var.set(self.simulation.animator.current_deg)
            self._render()
        else:
            self._update_live_values()
        self._schedule_tick()

    def _render(self) -> None:
        preserved = self._camera_limits
        self.renderer.draw(self.simulation)
        if preserved is not None:
            self.axes.set_xlim(*preserved[0])
            self.axes.set_ylim(*preserved[1])
        else:
            self._camera_limits = (self.axes.get_xlim(), self.axes.get_ylim())
        self._update_live_values()
        self.canvas.draw_idle()

    def _update_live_values(self) -> None:
        state = self.simulation.state
        phase = (
            state.elevation_deg
            / self.settings.max_elevation_deg
            * 2.0
            * 3.141592653589793
        )
        mechanical = float(self.simulation.transmission.ratio(phase))
        target_gh, target_st = self.simulation.shoulder_model.contributions_at(
            state.elevation_deg
        )
        region = next(
            (
                item
                for item in self.settings.ratio_regions
                if item.start_deg <= state.elevation_deg < item.end_deg
            ),
            self.settings.ratio_regions[-1],
        )
        target_ratio = region.gh_to_st_ratio
        ratio_error = state.instantaneous_ratio - target_ratio
        self.live_vars["Mechanical ratio"].set(f"{mechanical:.4f}")
        self.live_vars["Current GH"].set(f"{state.gh_deg:6.2f}°")
        self.live_vars["Current ST"].set(f"{state.st_deg:6.2f}°")
        self.live_vars["Actual GH:ST"].set(f"{state.instantaneous_ratio:.3f}:1")
        self.live_vars["Target GH:ST"].set(f"{target_ratio:.1f}:1")
        self.live_vars["GH error"].set(f"{state.gh_deg - target_gh:+.3f}°")
        self.live_vars["ST error"].set(f"{state.st_deg - target_st:+.3f}°")
        self.live_vars["Ratio error"].set(f"{ratio_error:+.3f}")
        self._update_validation_panel()

    def _update_validation_panel(self) -> None:
        report = self.simulation.biomechanics_validation
        rows = (
            ("GH endpoint = 120°", report.gh_endpoint_valid),
            ("ST endpoint = 60°", report.st_endpoint_valid),
            ("GH + ST = elevation", report.elevation_sum_valid),
            ("No discontinuities", report.no_discontinuities),
            ("No negative ratios", report.no_negative_ratios),
            ("No velocity spikes", report.no_velocity_spikes),
            ("Continuous first derivative", report.continuous_first_derivative),
            ("Continuous second derivative", report.continuous_second_derivative),
            ("Specification consistent", report.specification_consistent),
        )
        text = (
            f"BIOMECHANICS VALIDATION: {'PASS' if report.passed else 'FAIL'}\n"
            + "\n".join(
            f"{'✓' if valid else '⚠'} {label}" for label, valid in rows
            )
        )
        text += (
            f"\n\nMaximum GH error: {report.maximum_gh_error:.4f}°"
            f"\nMaximum ST error: {report.maximum_st_error:.4f}°"
            f"\nMaximum ratio error: {report.maximum_ratio_error:.4f}"
            f"\nMaximum velocity error: {report.maximum_velocity_error:.5f}"
            f"\nMaximum acceleration error: {report.maximum_acceleration_error:.5f}"
            f"\nRMS angular error: {report.rms_error:.4f}°"
        )
        text += "\n\nCheckpoints (requested → actual; error)"
        for checkpoint in report.checkpoints:
            text += (
                f"\n{checkpoint.elevation_deg:.0f}°: "
                f"GH {checkpoint.requested_gh_deg:.0f}→{checkpoint.actual_gh_deg:.2f}"
                f" ({checkpoint.gh_difference_deg:+.2f}°, {checkpoint.gh_percent_error:+.1f}%)"
                f"\n     ST {checkpoint.requested_st_deg:.0f}→{checkpoint.actual_st_deg:.2f}"
                f" ({checkpoint.st_difference_deg:+.2f}°, {checkpoint.st_percent_error:+.1f}%)"
            )
            if not checkpoint.specification_consistent:
                text += (
                    f"\n     schedule implies "
                    f"{checkpoint.schedule_gh_deg:.0f}/{checkpoint.schedule_st_deg:.0f}°"
                )
        if report.warnings:
            text += "\n\n" + "\n".join(f"⚠ {warning}" for warning in report.warnings)
        self.validation_label.configure(text=text)

    def _zoom(self, event: object) -> None:
        if event.xdata is None or event.ydata is None:
            return
        scale = 0.85 if event.button == "up" else 1.18
        x_limits = self.axes.get_xlim()
        y_limits = self.axes.get_ylim()
        x_width = (x_limits[1] - x_limits[0]) * scale
        y_width = (y_limits[1] - y_limits[0]) * scale
        relative_x = (event.xdata - x_limits[0]) / (x_limits[1] - x_limits[0])
        relative_y = (event.ydata - y_limits[0]) / (y_limits[1] - y_limits[0])
        self.axes.set_xlim(event.xdata - x_width * relative_x, event.xdata + x_width * (1 - relative_x))
        self.axes.set_ylim(event.ydata - y_width * relative_y, event.ydata + y_width * (1 - relative_y))
        self._camera_limits = (self.axes.get_xlim(), self.axes.get_ylim())
        self.canvas.draw_idle()

    def _pan_start(self, event: object) -> None:
        if event.button == 1 and event.xdata is not None and event.ydata is not None:
            self._pan_anchor = (event.xdata, event.ydata)

    def _pan_end(self, event: object) -> None:
        del event
        self._pan_anchor = None

    def _pan_move(self, event: object) -> None:
        if self._pan_anchor is None or event.xdata is None or event.ydata is None:
            return
        dx = self._pan_anchor[0] - event.xdata
        dy = self._pan_anchor[1] - event.ydata
        x_limits = self.axes.get_xlim()
        y_limits = self.axes.get_ylim()
        self.axes.set_xlim(x_limits[0] + dx, x_limits[1] + dx)
        self.axes.set_ylim(y_limits[0] + dy, y_limits[1] + dy)
        self._camera_limits = (self.axes.get_xlim(), self.axes.get_ylim())
        self.canvas.draw_idle()

    def _reset_camera(self) -> None:
        self._camera_limits = None
        self._render()

    def _export_svg(self) -> None:
        destination = filedialog.asksaveasfilename(
            defaultextension=".svg", filetypes=[("SVG drawing", "*.svg")]
        )
        if destination:
            SvgExporter().export(self.simulation, destination)

    def _export_dxf(self) -> None:
        destination = filedialog.asksaveasfilename(
            defaultextension=".dxf", filetypes=[("DXF drawing", "*.dxf")]
        )
        if destination:
            DxfExporter().export(self.simulation, destination)

    def _export_csv(self) -> None:
        destination = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV data", "*.csv")]
        )
        if destination:
            CsvExporter().export(self.simulation, destination)
