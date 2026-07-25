"""GUI-facing connected literature gear pair and render states."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from shapely import affinity
from shapely.geometry import Point, Polygon

from bench_prototype import BenchPrototype, BenchPrototypeConfig, build_bench_prototype
from biomechanics.literature_model import LiteratureShoulderModel


@dataclass(frozen=True)
class GearRenderState:
    elevation_deg: float
    input_polygon: Polygon
    output_polygon: Polygon
    input_pitch_curve: np.ndarray
    output_pitch_curve: np.ndarray
    input_angle_rad: float
    output_angle_rad: float
    contact_point: np.ndarray
    active_input_tooth: Polygon
    active_output_tooth: Polygon
    collision_polygon: Polygon
    collision_area: float
    input_center: np.ndarray
    output_center: np.ndarray


class LiteratureGearPair:
    """Two connected bodies driven only by the verified literature sector."""

    pathway_name = "literature_printable_gears"
    display_label = "Research visualization"

    def __init__(
        self,
        prototype: BenchPrototype,
        assembly_center_offset_mm: float = 0.0,
        tooth_style: str = "Spur",
    ) -> None:
        self.prototype = prototype
        self.transmission = prototype.transmission
        self.valid_range_deg = self.transmission.valid_range_deg
        self.nominal_center_distance = prototype.pitch_data.center_distance
        self.assembly_center_offset_mm = float(assembly_center_offset_mm)
        self.center_distance = (
            self.nominal_center_distance + self.assembly_center_offset_mm
        )
        self.source = "McClure2001"
        self.tooth_style = tooth_style

    @staticmethod
    def _check(elevation_deg: float, valid_range: tuple[float, float]) -> float:
        value = float(elevation_deg)
        if value < valid_range[0] or value > valid_range[1]:
            raise ValueError(
                f"Elevation outside literature range {valid_range}; "
                "extrapolation is forbidden."
            )
        return value

    def _sample_index(self, elevation_deg: float) -> int:
        value = self._check(elevation_deg, self.valid_range_deg)
        return int(
            np.argmin(
                np.abs(self.prototype.pitch_data.elevation_deg - value)
            )
        )

    def input_angle_at(self, elevation_deg: float) -> float:
        return float(self.transmission.input_angle(elevation_deg))

    def output_angle_at(self, elevation_deg: float) -> float:
        # External gears rotate oppositely.
        return -float(self.transmission.output_angle(elevation_deg))

    def input_polygon_at(self, elevation_deg: float) -> Polygon:
        return affinity.rotate(
            self.prototype.input_blank.polygon,
            np.degrees(self.input_angle_at(elevation_deg)),
            origin=(0.0, 0.0),
        )

    def output_polygon_at(self, elevation_deg: float) -> Polygon:
        return affinity.translate(
            affinity.rotate(
                self.prototype.output_blank.polygon,
                np.degrees(self.output_angle_at(elevation_deg)),
                origin=(0.0, 0.0),
            ),
            xoff=self.center_distance,
        )

    def contact_point_at(self, elevation_deg: float) -> np.ndarray:
        index = self._sample_index(elevation_deg)
        return np.array(
            [self.prototype.pitch_data.input_radii[index], 0.0],
            dtype=np.float64,
        )

    def collision_at(self, elevation_deg: float) -> Polygon:
        value = self._check(elevation_deg, self.valid_range_deg)
        input_teeth = self._transform_teeth(
            self.prototype.input_teeth, self.input_angle_at(value)
        )
        output_teeth = self._transform_teeth(
            self.prototype.output_teeth,
            self.output_angle_at(value),
            self.center_distance,
        )
        overlaps = [
            first.intersection(second)
            for first in input_teeth
            for second in output_teeth
            if first.intersects(second)
        ]
        if not overlaps:
            return Polygon()
        from shapely.ops import unary_union

        return unary_union(overlaps)

    @staticmethod
    def _transform_teeth(
        teeth: tuple[np.ndarray, ...],
        angle_rad: float,
        offset: float = 0.0,
    ) -> list[Polygon]:
        result = []
        for points in teeth:
            polygon = affinity.rotate(
                Polygon(points), np.degrees(angle_rad), origin=(0.0, 0.0)
            )
            if offset:
                polygon = affinity.translate(polygon, xoff=offset)
            result.append(polygon)
        return result

    def render_state_at(self, elevation_deg: float) -> GearRenderState:
        value = self._check(elevation_deg, self.valid_range_deg)
        input_angle = self.input_angle_at(value)
        output_angle = self.output_angle_at(value)
        input_polygon = self.input_polygon_at(value)
        output_polygon = self.output_polygon_at(value)
        # Transform arrays directly because the pitch curve is deliberately
        # open and must never be coerced into a polygon.
        def transform(points, angle, offset=0.0):
            c, s = np.cos(angle), np.sin(angle)
            rotation = np.array([[c, -s], [s, c]])
            return points @ rotation.T + np.array([offset, 0.0])

        input_pitch_points = transform(
            self.prototype.pitch_data.input_points, input_angle
        )
        output_pitch_points = transform(
            self.prototype.pitch_data.output_points,
            output_angle,
            self.center_distance,
        )
        contact = self.contact_point_at(value)
        input_teeth = self._transform_teeth(
            self.prototype.input_teeth, input_angle
        )
        output_teeth = self._transform_teeth(
            self.prototype.output_teeth,
            output_angle,
            self.center_distance,
        )
        point = Point(*contact)
        input_index = min(
            range(len(input_teeth)),
            key=lambda index: point.distance(input_teeth[index]),
        )
        output_index = min(
            range(len(output_teeth)),
            key=lambda index: point.distance(output_teeth[index]),
        )
        collision = self.collision_at(value)
        return GearRenderState(
            elevation_deg=value,
            input_polygon=input_polygon,
            output_polygon=output_polygon,
            input_pitch_curve=input_pitch_points,
            output_pitch_curve=output_pitch_points,
            input_angle_rad=input_angle,
            output_angle_rad=output_angle,
            contact_point=contact,
            active_input_tooth=input_teeth[input_index],
            active_output_tooth=output_teeth[output_index],
            collision_polygon=collision,
            collision_area=float(collision.area),
            input_center=np.array([0.0, 0.0]),
            output_center=np.array([self.center_distance, 0.0]),
        )


@lru_cache(maxsize=4)
def load_literature_gear_pair(
    model_path: str,
    center_distance: float = 120.0,
    module_mm: float = 2.5,
    pressure_angle_deg: float = 20.0,
    backlash_mm: float = 0.45,
    profile_relief_mm: float = 0.30,
    face_width_mm: float = 8.0,
    root_fillet_mm: float = 0.8,
    tooth_root_embed_mm: float = 1.5,
    center_distance_offset_mm: float = 0.0,
    tooth_style: str = "Spur",
) -> LiteratureGearPair:
    model = LiteratureShoulderModel(Path(model_path))
    prototype = build_bench_prototype(
        model,
        BenchPrototypeConfig(
            center_distance_mm=center_distance,
            module_mm=module_mm,
            pressure_angle_deg=pressure_angle_deg,
            backlash_mm=backlash_mm,
            profile_relief_mm=profile_relief_mm,
            gear_thickness_mm=face_width_mm,
            root_fillet_radius_mm=root_fillet_mm,
            tooth_root_embed_mm=tooth_root_embed_mm,
        ),
    )
    return LiteratureGearPair(
        prototype,
        center_distance_offset_mm,
        tooth_style,
    )
