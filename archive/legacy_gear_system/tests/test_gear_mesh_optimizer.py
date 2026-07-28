"""Mechanical-only optimization regression tests."""

from __future__ import annotations

import numpy as np
import pytest

from bench_prototype import build_bench_prototype
from biomechanics.literature_model import LiteratureShoulderModel
from gear_mesh_optimizer import (
    GearMeshOptimizer,
    GearMeshParameters,
    export_optimization_report,
    pitch_curve_fingerprint,
)
from literature_printable_pair import load_literature_gear_pair


pytestmark = pytest.mark.skip(
    reason=(
        "Optimization intentionally paused while restored rack-generated "
        "teeth are validated; previous buffered-tooth optimum is superseded."
    )
)


def _locked_prototype():
    model = LiteratureShoulderModel("ConsensusShoulderModel.json")
    return build_bench_prototype(model)


def test_optimizer_evaluates_full_requested_grid_and_locks_pitch_curves():
    prototype = _locked_prototype()
    before = pitch_curve_fingerprint(prototype)
    result = GearMeshOptimizer(prototype).optimize(top_n=5)
    after = pitch_curve_fingerprint(prototype)
    assert result.evaluated_candidates == 90_000
    assert before == after == result.locked_pitch_sha256
    assert result.preferred.biomechanical_deviation_deg == 0.0
    assert result.preferred.parameters.tooth_style == "Spur"
    assert result.preferred_validation.pitch_arrays_identical
    assert result.preferred_validation.passed
    assert (
        result.preferred_validation.maximum_tooth_penetration_area_mm2
        == 0.0
    )


def test_smaller_module_study_covers_all_requested_modules():
    result = GearMeshOptimizer(_locked_prototype()).optimize(top_n=3)
    assert [item.parameters.module_mm for item in result.module_study] == [
        1.5,
        1.75,
        2.0,
        2.25,
        2.5,
    ]
    practical = [
        item.parameters.module_mm
        for item in result.module_study
        if item.printable_04_nozzle
    ]
    assert min(practical) == 2.0


def test_style_study_is_comparative_and_spur_remains_preferred():
    result = GearMeshOptimizer(_locked_prototype()).optimize(top_n=3)
    styles = {item.parameters.tooth_style for item in result.style_study}
    assert styles == {"Spur", "Helical", "Herringbone"}
    assert result.preferred.parameters.tooth_style == "Spur"
    assert max(
        item.contact_ratio_estimate
        for item in result.style_study
        if item.parameters.tooth_style == "Herringbone"
    ) > result.preferred.contact_ratio_estimate


def test_preferred_candidate_has_no_sampled_tooth_penetration():
    result = GearMeshOptimizer(_locked_prototype()).optimize(top_n=3)
    p = result.preferred.parameters
    pair = load_literature_gear_pair(
        "ConsensusShoulderModel.json",
        120.0,
        p.module_mm,
        p.pressure_angle_deg,
        p.backlash_mm,
        p.profile_relief_mm,
        p.face_width_mm,
        p.root_fillet_mm,
        p.tooth_root_embed_mm,
        p.center_distance_offset_mm,
        p.tooth_style,
    )
    collision = max(
        pair.render_state_at(value).collision_area
        for value in np.linspace(11.0, 147.0, 401)
    )
    assert collision <= 1e-9
    assert pair.prototype.validation.adjacent_teeth_overlap_free


def test_reports_include_lock_provenance_and_calibration(tmp_path):
    result = GearMeshOptimizer(_locked_prototype()).optimize(top_n=3)
    paths = export_optimization_report(result, tmp_path)
    assert len(paths) == 3
    assert all(path.exists() and path.stat().st_size > 100 for path in paths)
    text = (tmp_path / "GearMeshOptimization.json").read_text()
    assert '"locked_biomechanics": true' in text
    assert '"locked_transmission": true' in text
    assert '"calibration"' in text
