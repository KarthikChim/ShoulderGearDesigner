"""Full conjugate gear-pair generation and assembly validation."""

import numpy as np

from settings import Settings
from simulation import Simulation


def test_both_gears_have_complete_synchronized_teeth() -> None:
    simulation = Simulation(Settings())
    generated = simulation.generated_pair
    assert generated.input_gear.tooth_count == generated.output_gear.tooth_count
    assert len(generated.input_gear.polygon) > 1000
    assert len(generated.output_gear.polygon) > 1000
    assert np.allclose(
        generated.input_gear.polygon[0], generated.input_gear.polygon[-1]
    )
    assert np.allclose(
        generated.output_gear.polygon[0], generated.output_gear.polygon[-1]
    )


def test_complete_pair_passes_one_revolution_validation() -> None:
    validation = Simulation(Settings()).generated_pair.validation
    assert validation.checked_positions >= 181
    assert validation.synchronized
    assert validation.interference_free
    assert validation.constant_center_distance
    assert validation.conjugacy_preserved
    assert validation.valid
