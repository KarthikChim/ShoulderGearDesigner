"""Tests proving the literature teeth are explicit rack profiles, not blobs."""

from __future__ import annotations

import inspect

import numpy as np
from shapely.geometry import Polygon

import literature_sector
from bench_prototype import BenchPrototypeConfig, build_bench_prototype
from biomechanics.literature_model import LiteratureShoulderModel


def _prototype(config: BenchPrototypeConfig):
    return build_bench_prototype(
        LiteratureShoulderModel("ConsensusShoulderModel.json"),
        config,
    )


def test_tooth_generator_uses_no_buffer_offset_or_smoothing():
    source = inspect.getsource(literature_sector.generate_sector_teeth)
    assert ".buffer(" not in source
    assert "offset_curve" not in source
    assert "spline" not in source.lower()


def test_every_tooth_has_root_tip_and_two_straight_distinct_flanks():
    prototype = _prototype(BenchPrototypeConfig())
    for tooth in (*prototype.input_teeth, *prototype.output_teeth):
        assert tooth.shape == (5, 2)
        assert np.allclose(tooth[0], tooth[-1])
        root_left, tip_left, tip_right, root_right = tooth[:4]
        left_flank = tip_left - root_left
        right_flank = tip_right - root_right
        assert np.linalg.norm(left_flank) > 0
        assert np.linalg.norm(right_flank) > 0
        assert not np.allclose(left_flank, right_flank)
        assert np.linalg.norm(tip_right - tip_left) > 0
        assert np.linalg.norm(root_right - root_left) > 0
        assert Polygon(tooth).is_valid


def test_pressure_angle_and_pitch_thickness_are_consistent():
    config = BenchPrototypeConfig(
        module_mm=2.0,
        pressure_angle_deg=25.0,
        backlash_mm=0.20,
        profile_relief_mm=0.10,
    )
    prototype = _prototype(config)
    expected_thickness = (
        np.pi * config.module_mm / 2.0
        - config.backlash_mm
        - 2.0
        * config.profile_relief_mm
        / np.cos(np.radians(config.pressure_angle_deg))
    )
    thicknesses = []
    angles = []
    fraction = prototype.teeth.dedendum / (
        prototype.teeth.addendum + prototype.teeth.dedendum
    )
    for tooth in (*prototype.input_teeth, *prototype.output_teeth):
        root_left, tip_left, tip_right, root_right = tooth[:4]
        pitch_left = root_left + fraction * (tip_left - root_left)
        pitch_right = root_right + fraction * (tip_right - root_right)
        thicknesses.append(np.linalg.norm(pitch_right - pitch_left))
        root_mid = 0.5 * (root_left + root_right)
        tip_mid = 0.5 * (tip_left + tip_right)
        centerline = tip_mid - root_mid
        centerline /= np.linalg.norm(centerline)
        flank = tip_left - root_left
        angles.append(
            np.degrees(
                np.arccos(
                    np.clip(
                        abs(np.dot(flank, centerline))
                        / np.linalg.norm(flank),
                        -1.0,
                        1.0,
                    )
                )
            )
        )
    assert np.allclose(thicknesses, expected_thickness, atol=1e-8)
    assert np.allclose(angles, config.pressure_angle_deg, atol=1e-8)


def test_tooth_changes_do_not_change_locked_pitch_arrays():
    baseline = _prototype(BenchPrototypeConfig())
    changed = _prototype(
        BenchPrototypeConfig(
            module_mm=2.0,
            pressure_angle_deg=25.0,
            backlash_mm=0.20,
            profile_relief_mm=0.10,
        )
    )
    for first, second in (
        (baseline.pitch_data.elevation_deg, changed.pitch_data.elevation_deg),
        (baseline.pitch_data.input_rad, changed.pitch_data.input_rad),
        (baseline.pitch_data.output_rad, changed.pitch_data.output_rad),
        (baseline.pitch_data.ratio, changed.pitch_data.ratio),
        (baseline.pitch_data.input_radii, changed.pitch_data.input_radii),
        (baseline.pitch_data.output_radii, changed.pitch_data.output_radii),
        (baseline.pitch_data.input_points, changed.pitch_data.input_points),
        (baseline.pitch_data.output_points, changed.pitch_data.output_points),
    ):
        assert np.array_equal(first, second)


def test_final_gear_has_connected_non_blob_teeth():
    prototype = _prototype(BenchPrototypeConfig())
    assert prototype.input_blank.polygon.is_valid
    assert prototype.output_blank.polygon.is_valid
    assert prototype.input_blank.polygon.geom_type == "Polygon"
    assert prototype.output_blank.polygon.geom_type == "Polygon"
    assert prototype.validation.adjacent_teeth_overlap_free
    assert prototype.validation.maximum_tooth_penetration_area_mm2 == 0.0
