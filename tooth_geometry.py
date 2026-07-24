"""Generalized involute teeth for one non-circular pitch curve.

For a circular gear an involute may be written from a base circle.  For a
non-circular gear the corresponding exact construction is the envelope swept
by a straight-sided rack cutter while its pitch line rolls, without slip,
along the gear pitch curve.  The numerical envelope below implements that
manufacturing construction directly.  Increasing ``envelope_samples``
converges to the continuous rack-generated flank.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
from shapely.geometry import Polygon
from shapely.ops import unary_union


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ToothParameters:
    """Editable rack and tooth-layout parameters."""

    module: float
    pressure_angle_deg: float = 20.0
    addendum: float | None = None
    dedendum: float | None = None
    root_fillet_radius: float = 0.0
    tooth_count: int | None = None
    envelope_samples: int = 2048
    tooth_phase: float = 0.5

    def resolved(self) -> tuple[float, float]:
        return (
            self.module if self.addendum is None else self.addendum,
            1.25 * self.module if self.dedendum is None else self.dedendum,
        )


@dataclass(frozen=True)
class ToothLocation:
    """Arc-length location and right-handed local frame of one tooth."""

    index: int
    arc_length: float
    polar_angle: float
    pitch_radius: float
    origin: FloatArray
    tangent: FloatArray
    outward_normal: FloatArray
    curvature: float


@dataclass(frozen=True)
class ToothValidation:
    tooth_overlap_free: bool
    root_overlap_free: bool
    tooth_spacing_valid: bool
    arc_length_spacing_valid: bool
    minimum_root_thickness_valid: bool
    minimum_tooth_thickness_valid: bool
    minimum_root_thickness: float
    minimum_pitch_thickness: float
    maximum_spacing_error: float
    warnings: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return all(
            (
                self.tooth_overlap_free,
                self.root_overlap_free,
                self.tooth_spacing_valid,
                self.arc_length_spacing_valid,
                self.minimum_root_thickness_valid,
                self.minimum_tooth_thickness_valid,
            )
        )


@dataclass(frozen=True)
class GeneratedGear:
    """Complete one-gear result reusable by a future conjugate generator."""

    polygon: FloatArray
    display_polygon: FloatArray
    pitch_curve: FloatArray
    root_curve: FloatArray
    addendum_curve: FloatArray
    base_curve: FloatArray
    cumulative_arc_length: FloatArray
    pitch_length: float
    circular_pitch: float
    tooth_count: int
    locations: tuple[ToothLocation, ...]
    tooth_polygons: tuple[FloatArray, ...]
    validation: ToothValidation


class NonCircularToothGenerator:
    """Generate generalized involute teeth using a rack-cutter envelope."""

    def generate(
        self, pitch_points: FloatArray, parameters: ToothParameters
    ) -> GeneratedGear:
        points = self._open_curve(pitch_points)
        if len(points) < 16:
            raise ValueError("Pitch curve requires at least 16 unique points.")
        if parameters.module <= 0:
            raise ValueError("Module must be positive.")

        curve, cumulative, length = self._arc_length_spline(points)
        tooth_count = parameters.tooth_count or max(
            6, int(np.rint(length / (np.pi * parameters.module)))
        )
        circular_pitch = length / tooth_count
        addendum, dedendum = parameters.resolved()
        locations = self._locations(
            curve, length, tooth_count, parameters.tooth_phase
        )

        # Parallel curves are offsets of the pitch curve, not concentric
        # circles.  The base curve is the inward normal offset p*sin(alpha),
        # the generalized counterpart of rb = rp*cos(alpha).
        dense_s = np.linspace(0.0, length, parameters.envelope_samples, endpoint=False)
        dense_points, tangents, normals, _ = self._frame(curve, dense_s)
        alpha = np.radians(parameters.pressure_angle_deg)
        base_offset = circular_pitch * np.sin(alpha) / (2.0 * np.pi)
        addendum_curve = dense_points + addendum * normals
        root_curve = dense_points - dedendum * normals
        base_curve = dense_points - base_offset * normals

        blank = Polygon(addendum_curve).buffer(0)
        if blank.is_empty or not blank.is_valid:
            raise ValueError("Addendum offset does not form a valid gear blank.")

        cutters = self._rack_cutters(
            dense_s,
            dense_points,
            tangents,
            normals,
            length,
            circular_pitch,
            alpha,
            dedendum,
            addendum,
            parameters.root_fillet_radius,
            parameters.tooth_phase,
        )
        cut_union = unary_union(cutters)
        finished = blank.difference(cut_union).buffer(0)
        if finished.geom_type == "MultiPolygon":
            finished = max(finished.geoms, key=lambda item: item.area)
        if finished.is_empty or finished.geom_type != "Polygon":
            raise ValueError("Rack envelope did not produce one closed gear polygon.")
        polygon = np.asarray(finished.exterior.coords, dtype=np.float64)
        display_boundary = finished.simplify(
            parameters.module * 0.008, preserve_topology=True
        )
        display_polygon = np.asarray(
            display_boundary.exterior.coords, dtype=np.float64
        )
        tooth_polygons = self._partition_boundary(polygon, locations)
        validation = self._validate(
            locations,
            tooth_polygons,
            circular_pitch,
            parameters.module,
            parameters.pressure_angle_deg,
            dedendum,
            polygon,
        )
        return GeneratedGear(
            polygon=polygon,
            display_polygon=display_polygon,
            pitch_curve=np.vstack((points, points[0])),
            root_curve=np.vstack((root_curve, root_curve[0])),
            addendum_curve=np.vstack((addendum_curve, addendum_curve[0])),
            base_curve=np.vstack((base_curve, base_curve[0])),
            cumulative_arc_length=cumulative,
            pitch_length=length,
            circular_pitch=circular_pitch,
            tooth_count=tooth_count,
            locations=locations,
            tooth_polygons=tooth_polygons,
            validation=validation,
        )

    @staticmethod
    def _open_curve(points: FloatArray) -> FloatArray:
        array = np.asarray(points, dtype=np.float64)
        return array[:-1] if np.linalg.norm(array[0] - array[-1]) < 1e-10 else array

    @staticmethod
    def _arc_length_spline(
        points: FloatArray,
    ) -> tuple[tuple[CubicSpline, CubicSpline], FloatArray, float]:
        closed = np.vstack((points, points[0]))
        segment_lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        keep = np.concatenate(([True], np.diff(cumulative) > 1e-12))
        cumulative = cumulative[keep]
        closed = closed[keep]
        length = float(cumulative[-1])
        sx = CubicSpline(cumulative, closed[:, 0], bc_type="periodic")
        sy = CubicSpline(cumulative, closed[:, 1], bc_type="periodic")
        return (sx, sy), cumulative, length

    def _frame(
        self, curve: tuple[CubicSpline, CubicSpline], arc: FloatArray
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        sx, sy = curve
        points = np.column_stack((sx(arc), sy(arc)))
        first = np.column_stack((sx(arc, 1), sy(arc, 1)))
        speed = np.linalg.norm(first, axis=1)
        tangents = first / speed[:, None]
        # The synthesized input curve is clockwise, so its left normal is
        # outward.  Determine orientation robustly and select that side.
        probe = np.linspace(0.0, sx.x[-1], 512, endpoint=False)
        px, py = sx(probe), sy(probe)
        signed_twice_area = np.sum(px * np.roll(py, -1) - np.roll(px, -1) * py)
        left = np.column_stack((-tangents[:, 1], tangents[:, 0]))
        normals = left if signed_twice_area < 0.0 else -left
        second = np.column_stack((sx(arc, 2), sy(arc, 2)))
        cross = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
        curvature = cross / np.maximum(speed**3, 1e-15)
        return points, tangents, normals, curvature

    def _locations(
        self,
        curve: tuple[CubicSpline, CubicSpline],
        length: float,
        count: int,
        tooth_phase: float,
    ) -> tuple[ToothLocation, ...]:
        arc = (
            np.arange(count, dtype=np.float64) + tooth_phase
        ) * length / count
        arc %= length
        points, tangents, normals, curvature = self._frame(curve, arc)
        return tuple(
            ToothLocation(
                index=index + 1,
                arc_length=float(arc[index]),
                polar_angle=float(np.arctan2(points[index, 1], points[index, 0])),
                pitch_radius=float(np.linalg.norm(points[index])),
                origin=points[index],
                tangent=tangents[index],
                outward_normal=normals[index],
                curvature=float(curvature[index]),
            )
            for index in range(count)
        )

    @staticmethod
    def _rack_cutters(
        arc: FloatArray,
        points: FloatArray,
        tangents: FloatArray,
        normals: FloatArray,
        length: float,
        pitch: float,
        alpha: float,
        dedendum: float,
        addendum: float,
        fillet: float,
        tooth_phase: float,
    ) -> list[Polygon]:
        """Return cutter positions whose union is the discrete flank envelope.

        Pure rolling makes rack displacement equal to pitch-curve arc length.
        Thus cutter centers occur at local tangent coordinates ``k*p-s``.
        A standard rack tooth has pitch-line thickness ``p/2`` and straight
        flanks inclined by pressure angle ``alpha``.
        """

        cutters: list[Polygon] = []
        outside = 2.5 * addendum
        for s, origin, tangent, normal in zip(arc, points, tangents, normals):
            # Tooth centers use ``tooth_phase``; gap/cutter centers are half a
            # pitch away.  This phase is what lets the mating gear start with
            # a gap opposite an input-gear tooth.
            cutter_phase = np.remainder(tooth_phase + 0.5, 1.0) * pitch
            phase = np.remainder(s - cutter_phase, pitch)
            nearest = -phase if phase <= pitch / 2.0 else pitch - phase
            for center in (nearest - pitch, nearest, nearest + pitch):
                half_tip = pitch / 4.0 - dedendum * np.tan(alpha)
                if half_tip <= 0.0:
                    continue
                half_outer = pitch / 4.0 + outside * np.tan(alpha)
                local = np.array(
                    [
                        [center - half_tip, -dedendum],
                        [center + half_tip, -dedendum],
                        [center + half_outer, outside],
                        [center - half_outer, outside],
                    ],
                    dtype=np.float64,
                )
                world = origin + local[:, :1] * tangent + local[:, 1:] * normal
                cutter = Polygon(world)
                # Rounding the rack tip creates the physical root trochoid.
                if fillet > 0.0:
                    radius = min(fillet, dedendum * 0.45, half_tip * 0.45)
                    cutter = cutter.buffer(radius, quad_segs=4).buffer(
                        -radius, quad_segs=4
                    )
                cutters.append(cutter)
        return cutters

    @staticmethod
    def _partition_boundary(
        polygon: FloatArray, locations: tuple[ToothLocation, ...]
    ) -> tuple[FloatArray, ...]:
        """Assign final boundary samples to the nearest tooth pitch point."""

        centers = np.vstack([location.origin for location in locations])
        nearest = np.argmin(
            np.sum((polygon[:, None, :] - centers[None, :, :]) ** 2, axis=2), axis=1
        )
        return tuple(polygon[nearest == index] for index in range(len(locations)))

    @staticmethod
    def _validate(
        locations: tuple[ToothLocation, ...],
        tooth_polygons: tuple[FloatArray, ...],
        pitch: float,
        module: float,
        pressure_angle_deg: float,
        dedendum: float,
        polygon: FloatArray,
    ) -> ToothValidation:
        spacing = np.diff(
            np.r_[
                [location.arc_length for location in locations],
                locations[0].arc_length + pitch * len(locations),
            ]
        )
        max_error = float(np.max(np.abs(spacing - pitch)))
        spacing_ok = max_error <= max(1e-9, pitch * 1e-8)
        boundary_ok = Polygon(polygon).is_valid and Polygon(polygon).exterior.is_simple
        root_thickness = pitch / 2.0 - 2.0 * dedendum * np.tan(
            np.radians(pressure_angle_deg)
        )
        pitch_thickness = pitch / 2.0
        minimum_ok = pitch_thickness >= 0.25 * module
        root_ok = root_thickness > 0.10 * module
        warnings: list[str] = []
        if not boundary_ok:
            warnings.append("Generated tooth boundary self-intersects.")
        if not spacing_ok:
            warnings.append("Arc-length tooth spacing exceeds tolerance.")
        if not root_ok:
            warnings.append("Rack-tip/root thickness is below the minimum.")
        if not minimum_ok:
            warnings.append("Pitch-line tooth thickness is below the minimum.")
        if any(len(points) < 2 for points in tooth_polygons):
            warnings.append("One or more teeth have insufficient envelope samples.")
        return ToothValidation(
            tooth_overlap_free=boundary_ok,
            root_overlap_free=boundary_ok and root_ok,
            tooth_spacing_valid=spacing_ok,
            arc_length_spacing_valid=spacing_ok,
            minimum_root_thickness_valid=root_ok,
            minimum_tooth_thickness_valid=minimum_ok,
            minimum_root_thickness=float(root_thickness),
            minimum_pitch_thickness=float(pitch_thickness),
            maximum_spacing_error=max_error,
            warnings=tuple(warnings),
        )
