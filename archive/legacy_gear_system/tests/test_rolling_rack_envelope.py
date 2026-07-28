"""Tests for the canonical rack-cutter rolling envelope."""

from __future__ import annotations

import inspect
from functools import lru_cache

import numpy as np

import rolling_envelope
from biomechanics.literature_model import LiteratureShoulderModel
from literature_sector import (
    LiteratureSectorTransmission,
    SectorDesignConfig,
    generate_sector_teeth,
    synthesize_sector_pitch_curves,
)
from standard_involute import StandardGearParameters, generate_rack_tooth


@lru_cache(maxsize=1)
def _envelopes():
    model = LiteratureShoulderModel("ConsensusShoulderModel.json")
    config = SectorDesignConfig(
        sample_count=1001,
        module=2.5,
        backlash=0.45,
    )
    transmission = LiteratureSectorTransmission(
        model, config, regularized=True
    )
    data = synthesize_sector_pitch_curves(transmission)
    return data, config, generate_sector_teeth(data, config)


def test_generator_uses_only_canonical_rack_and_no_buffer_or_offsets():
    source = inspect.getsource(rolling_envelope)
    assert "generate_rack_tooth" in source
    assert ".buffer(" not in source
    assert "offset_curve" not in source
    assert "_swept_between" in source


def test_standard_cutter_has_two_flanks_tip_and_tangent_root_fillets():
    parameters = StandardGearParameters(
        module=2.5,
        backlash=0.45,
        root_fillet_radius=0.75,
        rack_is_cutter=True,
    )
    tooth = generate_rack_tooth(parameters)
    assert len(tooth.points) > 20
    assert tooth.left_flank_angle_deg == 20.0
    assert tooth.right_flank_angle_deg == -20.0
    assert np.all(np.isfinite(tooth.points))


def test_pitch_locations_are_exactly_equal_in_arc_length():
    _, config, teeth = _envelopes()
    expected = np.pi * config.module
    for positions in (
        teeth.input_arc_positions,
        teeth.output_arc_positions,
    ):
        assert np.allclose(np.diff(positions), expected, atol=1e-10)


def test_frames_have_unit_tangents_normals_and_finite_curvature():
    _, _, teeth = _envelopes()
    for result in (teeth.input_envelope, teeth.output_envelope):
        frame = result.frame
        assert np.allclose(np.linalg.norm(frame.tangents, axis=1), 1.0)
        assert np.allclose(np.linalg.norm(frame.outward_normals, axis=1), 1.0)
        assert np.max(
            np.abs(np.sum(frame.tangents * frame.outward_normals, axis=1))
        ) < 1e-12
        assert np.all(np.isfinite(frame.curvature))


def test_envelopes_and_complete_conjugate_mesh_validate():
    _, _, teeth = _envelopes()
    assert teeth.envelope_verified
    assert teeth.input_envelope.validation.valid
    assert teeth.output_envelope.validation.valid
    mesh = teeth.envelope_mesh_validation
    assert mesh.valid
    assert mesh.conjugate_rolling
    assert mesh.no_interference
    assert mesh.continuous_velocity_ratio
    assert mesh.contact_through_complete_motion
    assert mesh.printable_root_thickness
    assert mesh.sampled_positions == 1001


def test_locked_pitch_curves_are_not_modified():
    data, _, teeth = _envelopes()
    assert np.array_equal(data.input_points, teeth.input_envelope.pitch_curve)
    assert np.array_equal(data.output_points, teeth.output_envelope.pitch_curve)
