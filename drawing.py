"""Professional Matplotlib rendering for synthesized noncircular gears."""

from __future__ import annotations

import math
from dataclasses import dataclass

import matplotlib.axes
import matplotlib.patches as patches
import numpy as np
from shapely.geometry import Polygon

from gear import Gear
from simulation import Simulation


@dataclass
class Renderer:
    """Draw the conjugate pair, contact geometry, and live ratio graph."""

    axes: matplotlib.axes.Axes
    ratio_axes: matplotlib.axes.Axes | None = None
    contribution_axes: matplotlib.axes.Axes | None = None

    def draw(self, simulation: Simulation) -> None:
        """Redraw all geometry from the current simulation state."""

        axes = self.axes
        axes.clear()
        settings = simulation.settings
        pair = simulation.gear_pair
        state = simulation.state
        contact = pair.contact_point()

        if settings.show_teeth:
            self._draw_teeth(simulation)

        if settings.advanced_debug and settings.show_pitch_curves:
            self._draw_gear(pair.input_gear, "#e47d24", "Input pitch curve")
            self._draw_gear(pair.output_gear, "#397eaf", "Output pitch curve")
            self._draw_reference_circles(simulation)

        if settings.advanced_debug:
            self._draw_tooth_overlays(simulation)

        if settings.advanced_debug and settings.show_gear_centers:
            for gear in (pair.input_gear, pair.output_gear):
                axes.plot(gear.center_x, gear.center_y, marker="+", markersize=13, color="#18212b")
                axes.text(
                    gear.center_x,
                    gear.center_y - settings.center_distance * 0.58,
                    gear.name,
                    ha="center",
                    va="top",
                    fontsize=9,
                )

        if settings.advanced_debug and settings.show_radius_vectors:
            axes.plot(
                [pair.input_gear.center_x, contact[0]],
                [pair.input_gear.center_y, contact[1]],
                color="#e47d24",
                linewidth=2,
                label=f"r₁={pair.input_pitch_radius:.3f}",
            )
            axes.plot(
                [pair.output_gear.center_x, contact[0]],
                [pair.output_gear.center_y, contact[1]],
                color="#397eaf",
                linewidth=2,
                label=f"r₂={pair.output_pitch_radius:.3f}",
            )

        if settings.show_contact_point:
            axes.plot(contact[0], contact[1], "o", color="#d83b3b", markersize=7, zorder=6)
            axes.annotate(
                f"Contact ({contact[0]:.3f}, {contact[1]:.3f})",
                contact,
                xytext=(8, 10),
                textcoords="offset points",
                fontsize=8,
            )
            if settings.advanced_debug:
                normal_length = settings.center_distance * 0.16
                axes.plot(
                    [contact[0] - normal_length, contact[0] + normal_length],
                    [contact[1], contact[1]],
                    color="#8c4fb5", linestyle="--", linewidth=1.5,
                    label="Contact normal",
                )

        if settings.advanced_debug and settings.show_velocity_vectors:
            self._draw_velocity_vectors(contact, settings.center_distance)

        metric_lines = []
        metric_lines.append(
            f"Actual GH:ST: {state.instantaneous_ratio:.3f}:1"
        )
        if settings.advanced_debug:
            gear_ratio = simulation.transmission.ratio(
                math.radians(state.input_rotation_deg)
            )
            metric_lines.append(f"Mechanical dψ/dφ: {gear_ratio:.5f}")
        if settings.advanced_debug and settings.show_gh_angle:
            metric_lines.append(f"GH angle: {state.gh_deg:.3f}°")
        if settings.advanced_debug and settings.show_st_angle:
            metric_lines.append(f"Scapular angle: {state.st_deg:.3f}°")
        if metric_lines:
            axes.text(
                0.018,
                0.98,
                "\n".join(metric_lines),
                transform=axes.transAxes,
                va="top",
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "alpha": 0.9},
            )

        if settings.advanced_debug:
            axes.plot(
                [pair.input_gear.center_x, pair.output_gear.center_x],
                [pair.input_gear.center_y, pair.output_gear.center_y],
                linestyle=":", color="#69737d", linewidth=1,
            )
        margin = settings.center_distance * 0.65
        axes.set_xlim(-margin, settings.center_distance + margin)
        axes.set_ylim(-margin, margin)
        axes.set_aspect("equal", adjustable="box")
        axes.grid(True, color="#dce1e5", linewidth=0.55)
        axes.set_title("Rack-Generated Non-Circular Gear Teeth")
        axes.set_xlabel("X (model units)")
        axes.set_ylabel("Y (model units)")
        if settings.advanced_debug:
            axes.legend(loc="lower left", fontsize=8)

        self._draw_ratio_graph(simulation)
        self._draw_contribution_graph(simulation)

    def draw_literature_pair(
        self,
        pair,
        elevation_deg: float,
        *,
        show_pitch_curves: bool = False,
        show_tooth_outlines: bool = True,
        show_contact_point: bool = True,
    ):
        """Draw the final connected literature-sector body polygons."""

        state = pair.render_state_at(elevation_deg)
        axes = self.axes
        axes.clear()
        self._fill_polygon(
            state.input_polygon,
            "#f4a24c",
            "#7c2d12",
            "Literature input gear",
        )
        self._fill_polygon(
            state.output_polygon,
            "#5aa9df",
            "#1e3a8a",
            "Literature output gear",
        )
        if show_tooth_outlines:
            for polygon, color in (
                (state.active_input_tooth, "#f6e05e"),
                (state.active_output_tooth, "#f6e05e"),
            ):
                points = np.asarray(polygon.exterior.coords)
                axes.plot(
                    points[:, 0],
                    points[:, 1],
                    color=color,
                    linewidth=2.5,
                    zorder=8,
                )
        if show_pitch_curves:
            axes.plot(
                state.input_pitch_curve[:, 0],
                state.input_pitch_curve[:, 1],
                "--",
                color="#9a3412",
                linewidth=1.0,
                label="Input pitch sector",
            )
            axes.plot(
                state.output_pitch_curve[:, 0],
                state.output_pitch_curve[:, 1],
                "--",
                color="#1d4ed8",
                linewidth=1.0,
                label="Output pitch sector",
            )
        if show_contact_point:
            axes.plot(
                *state.contact_point,
                "o",
                color="#dc2626",
                markersize=7,
                zorder=10,
                label="Intended contact",
            )
        if state.collision_area > 1e-8 and not state.collision_polygon.is_empty:
            self._fill_polygon(
                state.collision_polygon,
                "#ef4444",
                "#991b1b",
                "Unintended overlap",
                alpha=0.75,
                zorder=20,
            )
        for center in (state.input_center, state.output_center):
            axes.plot(*center, "+", color="#111827", markersize=10, zorder=12)
        all_points = np.vstack(
            (
                np.asarray(state.input_polygon.exterior.coords),
                np.asarray(state.output_polygon.exterior.coords),
            )
        )
        low = np.min(all_points, axis=0)
        high = np.max(all_points, axis=0)
        margin = max(8.0, 0.08 * np.max(high - low))
        axes.set_xlim(low[0] - margin, high[0] + margin)
        axes.set_ylim(low[1] - margin, high[1] + margin)
        axes.set_aspect("equal", adjustable="box")
        axes.grid(True, color="#dce1e5", linewidth=0.55)
        axes.set_title(
            "Literature Printable Gears — Research Visualization\n"
            "NOT FOR HUMAN OR POWERED USE"
        )
        axes.set_xlabel("X (mm)")
        axes.set_ylabel("Y (mm)")
        axes.legend(loc="lower left", fontsize=8)
        if self.ratio_axes is not None:
            self.ratio_axes.clear()
            self.ratio_axes.set_visible(False)
        if self.contribution_axes is not None:
            self.contribution_axes.clear()
            self.contribution_axes.set_visible(False)
        return state

    def _fill_polygon(
        self,
        polygon: Polygon,
        face: str,
        edge: str,
        label: str,
        *,
        alpha: float = 0.9,
        zorder: int = 2,
    ) -> None:
        if polygon.is_empty:
            return
        if polygon.geom_type == "MultiPolygon":
            for index, component in enumerate(polygon.geoms):
                self._fill_polygon(
                    component,
                    face,
                    edge,
                    label if index == 0 else "_nolegend_",
                    alpha=alpha,
                    zorder=zorder,
                )
            return
        exterior = np.asarray(polygon.exterior.coords)
        self.axes.fill(
            exterior[:, 0],
            exterior[:, 1],
            facecolor=face,
            edgecolor=edge,
            linewidth=1.1,
            alpha=alpha,
            label=label,
            zorder=zorder,
        )
        for interior in polygon.interiors:
            points = np.asarray(interior.coords)
            self.axes.fill(
                points[:, 0],
                points[:, 1],
                facecolor="white",
                edgecolor=edge,
                linewidth=0.8,
                zorder=zorder + 1,
            )

    @staticmethod
    def _world(points: np.ndarray, gear: Gear) -> np.ndarray:
        angle = np.radians(gear.angle_deg)
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        return points @ rotation.T + gear.center

    def _draw_teeth(self, simulation: Simulation) -> None:
        pair = simulation.gear_pair
        for generated, gear, face, edge, label in (
            (
                simulation.generated_pair.input_gear,
                pair.input_gear,
                "#efa45c",
                "#98480d",
                "Input gear",
            ),
            (
                simulation.generated_pair.output_gear,
                pair.output_gear,
                "#6faada",
                "#1e5d91",
                "Output gear",
            ),
        ):
            points = self._world(generated.display_polygon, gear)
            self.axes.fill(
                points[:, 0], points[:, 1], facecolor=face, edgecolor=edge,
                linewidth=1.0, alpha=0.84, label=label, zorder=1,
            )
            self._highlight_active_tooth(generated, gear, simulation, edge)

    def _highlight_active_tooth(
        self, generated, gear: Gear, simulation: Simulation, color: str
    ) -> None:
        """Emphasize the boundary samples belonging to the contact tooth."""

        contact = simulation.gear_pair.contact_point()
        angle = np.radians(-gear.angle_deg)
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        local_contact = (contact - gear.center) @ rotation.T
        centers = np.vstack([location.origin for location in generated.locations])
        index = int(np.argmin(np.linalg.norm(centers - local_contact, axis=1)))
        boundary = generated.tooth_polygons[index]
        if len(boundary) > 1:
            active = self._world(boundary, gear)
            self.axes.plot(
                active[:, 0], active[:, 1], color="#f4e04d", linewidth=3.0,
                solid_capstyle="round", zorder=8,
            )

    def _draw_tooth_overlays(self, simulation: Simulation) -> None:
        settings = simulation.settings
        generated = simulation.generated_gear
        gear = simulation.gear_pair.input_gear
        for enabled, curve, color, label in (
            (settings.show_base_curve, generated.base_curve, "#7b4ab0", "Base curve"),
            (settings.show_root_curve, generated.root_curve, "#8f5c36", "Root curve"),
            (settings.show_addendum_curve, generated.addendum_curve, "#c83349", "Addendum curve"),
        ):
            if enabled:
                points = self._world(curve, gear)
                self.axes.plot(points[:, 0], points[:, 1], "--", color=color, lw=1.0, label=label)

        if not any(
            (
                settings.show_tooth_centers,
                settings.show_tooth_numbers,
                settings.show_normals,
                settings.show_tangents,
            )
        ):
            return
        scale = simulation.generated_pair.design.module * 1.8
        for location in generated.locations:
            origin = self._world(location.origin[None, :], gear)[0]
            tangent_tip = self._world(
                (location.origin + scale * location.tangent)[None, :], gear
            )[0]
            normal_tip = self._world(
                (location.origin + scale * location.outward_normal)[None, :], gear
            )[0]
            if settings.show_tooth_centers:
                self.axes.plot(*origin, ".", color="#222222", markersize=3, zorder=7)
            if settings.show_tooth_numbers:
                self.axes.text(*normal_tip, str(location.index), fontsize=6, ha="center")
            if settings.show_tangents:
                self.axes.plot(
                    [origin[0], tangent_tip[0]], [origin[1], tangent_tip[1]],
                    color="#168b8b", lw=0.7,
                )
            if settings.show_normals:
                self.axes.plot(
                    [origin[0], normal_tip[0]], [origin[1], normal_tip[1]],
                    color="#cf3e67", lw=0.7,
                )

    def _draw_gear(self, gear: Gear, color: str, label: str) -> None:
        points = gear.world_points(2048)
        self.axes.plot(points[:, 0], points[:, 1], color=color, linewidth=2.2, label=label)
        mean_radius = float(np.mean(gear.pitch_curve.radii))
        arrow = patches.Arc(
            (gear.center_x, gear.center_y),
            mean_radius * 0.75,
            mean_radius * 0.75,
            theta1=20,
            theta2=120,
            color=color,
            linewidth=1.4,
        )
        self.axes.add_patch(arrow)

    def _draw_reference_circles(self, simulation: Simulation) -> None:
        for gear, color in (
            (simulation.gear_pair.input_gear, "#e47d24"),
            (simulation.gear_pair.output_gear, "#397eaf"),
        ):
            radius = float(np.mean(gear.pitch_curve.radii))
            circle = patches.Circle(
                (gear.center_x, gear.center_y),
                radius,
                fill=False,
                linestyle="--",
                linewidth=0.8,
                alpha=0.45,
                edgecolor=color,
            )
            self.axes.add_patch(circle)

    def _draw_velocity_vectors(self, contact: np.ndarray, center_distance: float) -> None:
        scale = center_distance * 0.12
        for sign, color in ((1.0, "#e47d24"), (-1.0, "#397eaf")):
            self.axes.arrow(
                contact[0],
                contact[1],
                0.0,
                sign * scale,
                width=center_distance * 0.004,
                color=color,
                length_includes_head=True,
            )

    def _draw_ratio_graph(self, simulation: Simulation) -> None:
        if self.ratio_axes is None:
            return
        graph = self.ratio_axes
        graph.clear()
        graph.set_visible(True)
        report = simulation.biomechanics_validation
        display = slice(None, None, 4)
        current_elevation = simulation.state.elevation_deg
        current_ratio = simulation.state.instantaneous_ratio
        graph.step(
            report.elevation_deg[display],
            report.target_ratio[display],
            where="post",
            color="#333333",
            linestyle="--",
            linewidth=1.7,
            label="Target GH:ST",
        )
        graph.plot(
            report.elevation_deg[display],
            report.actual_ratio[display],
            color="#a047a8",
            linewidth=1.8,
            label="Actual GH:ST",
        )
        graph.plot(current_elevation, current_ratio, "o", color="#d83b3b", markersize=6)
        graph.axvline(current_elevation, color="#d83b3b", linewidth=0.7, alpha=0.5)
        graph.set_xlim(0.0, 180.0)
        graph.set_xlabel("Arm elevation (deg)")
        graph.set_ylabel("GH:ST ratio")
        graph.set_title("Target vs Actual Rhythm")
        graph.grid(True, color="#e0e3e7", linewidth=0.5)
        graph.legend(fontsize=7, loc="upper right")

    def _draw_contribution_graph(self, simulation: Simulation) -> None:
        if self.contribution_axes is None:
            return
        graph = self.contribution_axes
        graph.clear()
        report = simulation.biomechanics_validation
        display = slice(None, None, 4)
        elevation = report.elevation_deg[display]
        graph.plot(elevation, report.target_gh_deg[display], "--", color="#c45d19", lw=1.4, label="Target GH")
        graph.plot(elevation, report.actual_gh_deg[display], color="#e47d24", lw=1.8, label="Actual GH")
        graph.plot(elevation, report.target_st_deg[display], "--", color="#275d8c", lw=1.4, label="Target ST")
        graph.plot(elevation, report.actual_st_deg[display], color="#397eaf", lw=1.8, label="Actual ST")
        graph.axvline(
            simulation.state.elevation_deg, color="#d83b3b", linewidth=0.7, alpha=0.5
        )
        graph.set_xlim(0.0, 180.0)
        graph.set_xlabel("Arm elevation (deg)")
        graph.set_ylabel("Contribution (deg)")
        graph.set_title("GH / ST Contributions")
        graph.grid(True, color="#e0e3e7", linewidth=0.5)
        graph.legend(fontsize=7, ncol=2, loc="upper left")
