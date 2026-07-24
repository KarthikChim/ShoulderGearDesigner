"""High-level simulation orchestration."""

from __future__ import annotations

from animation import Animator
from biomechanics_validation import BiomechanicsValidation, validate_biomechanics
from gear import Gear
from kinematics import ShoulderModel, ShoulderState
from meshing import ConjugateGearPair
from noncircular import (
    MeshingValidation,
    PitchCurveData,
    SmoothTransmission,
    synthesize_pitch_curves,
    validate_pitch_curves,
)
from pitch_curve import SampledPitchCurve
from pair_teeth import GeneratedGearPair, generate_gear_pair
from settings import Settings


class Simulation:
    """Own settings, kinematics, gears, and animation state."""

    def __init__(self, settings: Settings) -> None:
        settings.validate()
        self.settings = settings
        self.shoulder_model = ShoulderModel(
            settings.ratio_regions,
            settings.max_elevation_deg,
            settings.input_revolutions_per_elevation_cycle,
        )
        self.transmission = SmoothTransmission(self.shoulder_model)
        self.biomechanics_validation = self._validate_biomechanics()
        self.animator = Animator(
            settings.max_elevation_deg, settings.simulation_speed_deg_s
        )
        self.pitch_data = self._synthesize()
        self.validation = validate_pitch_curves(self.pitch_data)
        self.generated_pair = self._generate_teeth()
        self.generated_gear = self.generated_pair.input_gear
        self.gear_pair = self._create_gear_pair()
        self.state = self._state_at_elevation(0.0)
        self.set_elevation(0.0)

    def _synthesize(self) -> PitchCurveData:
        return synthesize_pitch_curves(
            self.transmission,
            self.settings.center_distance,
            self.settings.pitch_curve_samples,
        )

    def _validate_biomechanics(self) -> BiomechanicsValidation:
        """Run the independent 0–180-degree engineering validation sweep."""

        return validate_biomechanics(
            self.transmission,
            self.settings.ratio_regions,
            self.settings.max_elevation_deg,
        )

    def _create_gear_pair(self) -> ConjugateGearPair:
        input_gear = Gear(
            "Input / GH",
            SampledPitchCurve(
                self.pitch_data.input_points,
                self.pitch_data.input_rad,
                self.pitch_data.input_radii,
            ),
            0.0,
            0.0,
        )
        output_gear = Gear(
            "Output / ST",
            SampledPitchCurve(
                self.pitch_data.output_points,
                self.pitch_data.input_rad,
                self.pitch_data.output_radii,
            ),
            self.settings.center_distance,
            0.0,
        )
        return ConjugateGearPair(
            input_gear,
            output_gear,
            input_pitch_radius=float(self.pitch_data.input_radii[0]),
            output_pitch_radius=float(self.pitch_data.output_radii[0]),
        )

    def _generate_teeth(self) -> GeneratedGearPair:
        """Generate and validate both gears once per geometry rebuild."""

        return generate_gear_pair(self.pitch_data)

    def rebuild_geometry(self) -> None:
        """Recreate Phase-1 circular geometry after settings change."""

        self.settings.validate()
        self.transmission = SmoothTransmission(self.shoulder_model)
        self.biomechanics_validation = self._validate_biomechanics()
        self.pitch_data = self._synthesize()
        self.validation = validate_pitch_curves(self.pitch_data)
        self.generated_pair = self._generate_teeth()
        self.generated_gear = self.generated_pair.input_gear
        self.gear_pair = self._create_gear_pair()
        self.animator.speed_deg_s = self.settings.simulation_speed_deg_s
        self.set_elevation(self.animator.current_deg)

    def set_elevation(self, elevation_deg: float) -> ShoulderState:
        """Set elevation and synchronize kinematics with displayed gears."""

        self.animator.set_position(elevation_deg)
        self.state = self._state_at_elevation(self.animator.current_deg)
        sample = self.transmission.evaluate(
            self.animator.current_deg / self.settings.max_elevation_deg * 2.0 * 3.141592653589793
        )
        input_radius = (
            self.settings.center_distance
            * sample.gear_ratio
            / (1.0 + sample.gear_ratio)
        )
        output_radius = self.settings.center_distance - input_radius
        self.gear_pair.set_angles(
            self.state.input_rotation_deg, self.state.output_rotation_deg
        )
        self.gear_pair.set_contact_radii(input_radius, output_radius)
        return self.state

    def update(self, elapsed_s: float) -> ShoulderState:
        """Advance animation and return the new state."""

        return self.set_elevation(self.animator.advance(elapsed_s))

    def _state_at_elevation(self, elevation_deg: float) -> ShoulderState:
        """Evaluate the smooth periodic transmission at a commanded elevation."""

        input_rad = (
            elevation_deg
            / self.settings.max_elevation_deg
            * 2.0
            * 3.141592653589793
        )
        sample = self.transmission.evaluate(input_rad)
        return ShoulderState(
            elevation_deg=sample.elevation_deg,
            gh_deg=sample.gh_deg,
            st_deg=sample.st_deg,
            instantaneous_ratio=sample.gh_to_st_ratio,
            input_rotation_deg=input_rad * 180.0 / 3.141592653589793,
            output_rotation_deg=-sample.output_rad * 180.0 / 3.141592653589793,
        )
