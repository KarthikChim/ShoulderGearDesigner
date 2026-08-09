"""Stationary-sun literature pathway for a non-circular shoulder planet gear.

This module is intentionally independent from the existing fixed-axis pitch
curve pathway.  It consumes the already-verified McClure2001 literature output
only between 11 and 147 degrees and never extrapolates.

Sign convention
---------------
Positive angles are counter-clockwise in the fixed torso frame.  The carrier
angle ``theta_c`` is scapular upward-rotation excursion, the sun is stationary
(``theta_s = 0``), and the planet's absolute angle ``theta_p`` is humerothoracic
elevation excursion.

For an external sun/planet mesh, rolling is evaluated in the carrier frame::

    omega_s_rel = d(theta_s)/dE - d(theta_c)/dE = -d(theta_c)/dE
    omega_p_rel = d(theta_p)/dE - d(theta_c)/dE

    omega_s_rel * r_s + omega_p_rel * r_p = 0

Thus the signed angular ratio is negative::

    signed_ratio = omega_p_rel / omega_s_rel = -r_s / r_p

and, with ``r_s + r_p = C``::

    k = -signed_ratio = r_s / r_p
    r_s = C k / (1 + k)
    r_p = C / (1 + k)

The shoulder-side member is technically a planet gear. A true ring gear would
be internally toothed and concentric with the sun.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import cumulative_trapezoid
from shapely.geometry import LineString


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PlanetaryPitchCurveData:
    elevation_deg: FloatArray
    carrier_angle_rad: FloatArray
    sun_angle_rad: FloatArray
    planet_absolute_angle_rad: FloatArray
    planet_relative_angle_rad: FloatArray
    dcarrier_dE: FloatArray
    dplanet_absolute_dE: FloatArray
    dplanet_relative_dE: FloatArray
    signed_ratio: FloatArray
    sun_pitch_radius_mm: FloatArray
    planet_pitch_radius_mm: FloatArray
    sun_pitch_points_local: FloatArray
    planet_pitch_points_local: FloatArray
    planet_center_points_world: FloatArray
    contact_points_world: FloatArray
    center_distance_mm: float

    def index_at(self, elevation_deg: float) -> int:
        if elevation_deg < self.elevation_deg[0] or elevation_deg > self.elevation_deg[-1]:
            raise ValueError("Elevation outside verified 11-147 degree range.")
        return int(np.argmin(np.abs(self.elevation_deg - elevation_deg)))


@dataclass(frozen=True)
class PlanetaryValidation:
    passed: bool
    sample_count: int
    valid_range_deg: tuple[float, float]
    sun_stationary: bool
    carrier_matches_st: bool
    planet_matches_ht: bool
    center_distance_error_mm: float
    contact_coincidence_error_mm: float
    tangent_alignment_error_deg: float
    rolling_residual_mm_per_rad: float
    finite_positive_pitch_radii: bool
    no_sign_discontinuity: bool
    sun_curve_simple: bool
    planet_curve_simple: bool
    no_extrapolation: bool
    maximum_elevation_error_deg: float
    rms_elevation_error_deg: float
    endpoint_elevation_error_deg: float


@dataclass(frozen=True)
class ExtendedPlanetaryPitchPaths:
    """Mechanical lead-in/out paths surrounding the verified motion sector.

    These extensions continue the terminal rolling ratios only to keep teeth
    engaged near mechanical stops. They are not extrapolated shoulder data.
    ``biological_start_index`` and ``biological_end_index`` identify the exact
    untouched 11-147 degree literature section inside the extended arrays.
    """

    carrier_angle_rad: FloatArray
    planet_absolute_angle_rad: FloatArray
    sun_pitch_radius_mm: FloatArray
    planet_pitch_radius_mm: FloatArray
    sun_pitch_points_local: FloatArray
    planet_pitch_points_local: FloatArray
    planet_center_points_world: FloatArray
    contact_points_world: FloatArray
    biological_start_index: int
    biological_end_index: int
    extension_teeth: int
    module_mm: float
    extension_pitch_length_mm: float
    center_distance_mm: float


def synthesize_literature_planetary_pitch_curves(
    source_csv: str | Path,
    *,
    center_distance_mm: float = 120.0,
    candidate: str = "regularized",
) -> PlanetaryPitchCurveData:
    """Generate stationary-sun and moving-planet pitch curves."""
    if center_distance_mm <= 0.0:
        raise ValueError("Center distance must be positive.")

    path = Path(source_csv)
    with path.open(newline="", encoding="utf-8") as stream:
        if stream.readline().startswith("RESEARCH"):
            reader = csv.DictReader(stream)
        else:
            stream.seek(0)
            reader = csv.DictReader(stream)
        rows = [row for row in reader if row["candidate"] == candidate]
    if len(rows) < 3:
        raise ValueError(f"No complete {candidate!r} literature trajectory in {path}.")

    elevation_deg = np.asarray(
        [float(row["ht_elevation_deg"]) for row in rows], dtype=np.float64
    )
    if elevation_deg[0] != 11.0 or elevation_deg[-1] != 147.0:
        raise ValueError("Planetary synthesis is restricted to verified 11-147 degrees.")
    if any(row.get("extrapolated", "false").lower() == "true" for row in rows):
        raise ValueError("Extrapolated literature samples are forbidden.")

    st_excursion_deg = np.asarray(
        [float(row["st_excursion_deg"]) for row in rows], dtype=np.float64
    )
    elevation_rad = np.radians(elevation_deg)
    carrier_angle_rad = np.radians(st_excursion_deg - st_excursion_deg[0])
    sun_angle_rad = np.zeros_like(elevation_rad)
    planet_absolute_angle_rad = elevation_rad - elevation_rad[0]
    planet_relative_angle_rad = planet_absolute_angle_rad - carrier_angle_rad

    dcarrier_dE = np.gradient(carrier_angle_rad, elevation_rad)
    dplanet_absolute_dE = np.gradient(planet_absolute_angle_rad, elevation_rad)
    dplanet_relative_dE = np.gradient(planet_relative_angle_rad, elevation_rad)

    sun_relative_rate = -dcarrier_dE
    if np.any(np.isclose(sun_relative_rate, 0.0, atol=1e-12)):
        raise ValueError("Stationary-sun synthesis requires nonzero carrier velocity.")
    signed_ratio = dplanet_relative_dE / sun_relative_rate
    radius_ratio = -signed_ratio
    if not np.all(np.isfinite(radius_ratio)) or np.any(radius_ratio <= 0.0):
        raise ValueError("Literature motion does not yield positive external pitch radii.")

    sun_radius = center_distance_mm * radius_ratio / (1.0 + radius_ratio)
    planet_radius = center_distance_mm / (1.0 + radius_ratio)

    carrier_unit = np.column_stack(
        (np.cos(carrier_angle_rad), np.sin(carrier_angle_rad))
    )
    planet_center = center_distance_mm * carrier_unit
    contact_world = sun_radius[:, None] * carrier_unit

    # Sun is stationary, so its world and local frames are identical.
    sun_points_local = contact_world.copy()

    # The planet contact is on the side facing the sun. Remove the planet's
    # absolute rotation to express that point in the rotating planet frame.
    planet_local_angle = (
        carrier_angle_rad + np.pi - planet_absolute_angle_rad
    )
    planet_points_local = planet_radius[:, None] * np.column_stack(
        (np.cos(planet_local_angle), np.sin(planet_local_angle))
    )

    return PlanetaryPitchCurveData(
        elevation_deg=elevation_deg,
        carrier_angle_rad=carrier_angle_rad,
        sun_angle_rad=sun_angle_rad,
        planet_absolute_angle_rad=planet_absolute_angle_rad,
        planet_relative_angle_rad=planet_relative_angle_rad,
        dcarrier_dE=dcarrier_dE,
        dplanet_absolute_dE=dplanet_absolute_dE,
        dplanet_relative_dE=dplanet_relative_dE,
        signed_ratio=signed_ratio,
        sun_pitch_radius_mm=sun_radius,
        planet_pitch_radius_mm=planet_radius,
        sun_pitch_points_local=sun_points_local,
        planet_pitch_points_local=planet_points_local,
        planet_center_points_world=planet_center,
        contact_points_world=contact_world,
        center_distance_mm=float(center_distance_mm),
    )


def rotate_points(points: FloatArray, angle_rad: float) -> FloatArray:
    cosine = np.cos(angle_rad)
    sine = np.sin(angle_rad)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    return points @ rotation.T


def extend_planetary_pitch_paths(
    data: PlanetaryPitchCurveData,
    *,
    module_mm: float = 2.0,
    extension_teeth: int = 4,
    samples_per_tooth: int = 64,
) -> ExtendedPlanetaryPitchPaths:
    """Append equal rolling pitch length to both ends of both pitch paths.

    One tooth pitch is ``pi * module``. At each endpoint the terminal radii and
    velocity ratio are held constant. Consequently the sun continuation is a
    circular arc with ``ds = r_s d(theta_c)`` and the planet continuation has
    exactly the same arc length by the external rolling equation.

    The original literature points are inserted without recalculation or
    smoothing, so the verified biological section is numerically unchanged.
    """
    if module_mm <= 0.0:
        raise ValueError("Module must be positive.")
    if extension_teeth < 1:
        raise ValueError("At least one extension tooth is required.")
    if samples_per_tooth < 4:
        raise ValueError("Use at least four samples per extension tooth.")

    extension_length = float(extension_teeth * np.pi * module_mm)
    extension_samples = extension_teeth * samples_per_tooth

    radius_slope = np.gradient(
        data.sun_pitch_radius_mm, data.carrier_angle_rad
    )

    def endpoint_extension(index: int, direction: float) -> tuple[FloatArray, ...]:
        """Build a C1 continuation with the requested sun arc length."""
        sun_radius = float(data.sun_pitch_radius_mm[index])
        carrier_start = float(data.carrier_angle_rad[index])
        planet_start = float(data.planet_absolute_angle_rad[index])
        slope = float(radius_slope[index])

        def sampled_extension(carrier_travel: float) -> tuple[FloatArray, ...]:
            fraction = np.linspace(0.0, 1.0, extension_samples + 1)
            offset_end = direction * carrier_travel
            offset = offset_end * fraction
            # Match dr/d(theta_c) at the biological endpoint and taper that
            # derivative linearly to zero at the outside mechanical endpoint.
            radii = sun_radius + slope * (
                offset - offset**2 / (2.0 * offset_end)
            )
            carrier = carrier_start + offset
            points = radii[:, None] * np.column_stack(
                (np.cos(carrier), np.sin(carrier))
            )
            length = float(
                np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
            )
            return carrier, radii, points, np.asarray(length)

        lower = 0.0
        upper = extension_length / max(sun_radius, 1e-9)
        for _ in range(20):
            carrier, radii, _points, measured = sampled_extension(upper)
            if (
                float(measured) >= extension_length
                and np.all(radii > 0.0)
                and np.all(radii < data.center_distance_mm)
            ):
                break
            upper *= 1.25
        else:
            raise ValueError("Could not construct a positive-radius endpoint extension.")

        for _ in range(60):
            midpoint = 0.5 * (lower + upper)
            _carrier, _radii, _points, measured = sampled_extension(midpoint)
            if float(measured) < extension_length:
                lower = midpoint
            else:
                upper = midpoint

        carrier, sun_radii, _points, _measured = sampled_extension(upper)
        planet_radii = data.center_distance_mm - sun_radii
        # From external rolling with a stationary sun:
        # d(theta_p)/d(theta_c) = 1 + r_s/r_p.
        planet_per_carrier = 1.0 + sun_radii / planet_radii
        planet = planet_start + cumulative_trapezoid(
            planet_per_carrier, carrier, initial=0.0
        )

        # sampled_extension is endpoint -> outside. Return arrays in global
        # motion order and exclude the already-present biological endpoint.
        if direction < 0.0:
            return (
                carrier[1:][::-1],
                planet[1:][::-1],
                sun_radii[1:][::-1],
                planet_radii[1:][::-1],
            )
        return carrier[1:], planet[1:], sun_radii[1:], planet_radii[1:]

    pre = endpoint_extension(0, -1.0)
    post = endpoint_extension(-1, 1.0)
    carrier = np.concatenate((pre[0], data.carrier_angle_rad, post[0]))
    planet = np.concatenate((pre[1], data.planet_absolute_angle_rad, post[1]))
    sun_radius = np.concatenate((pre[2], data.sun_pitch_radius_mm, post[2]))
    planet_radius = np.concatenate((pre[3], data.planet_pitch_radius_mm, post[3]))

    carrier_unit = np.column_stack((np.cos(carrier), np.sin(carrier)))
    planet_center = data.center_distance_mm * carrier_unit
    contact_world = sun_radius[:, None] * carrier_unit
    sun_points = contact_world.copy()
    planet_local_angle = carrier + np.pi - planet
    planet_points = planet_radius[:, None] * np.column_stack(
        (np.cos(planet_local_angle), np.sin(planet_local_angle))
    )

    biological_start = extension_samples
    biological_end = biological_start + len(data.elevation_deg) - 1
    return ExtendedPlanetaryPitchPaths(
        carrier_angle_rad=carrier,
        planet_absolute_angle_rad=planet,
        sun_pitch_radius_mm=sun_radius,
        planet_pitch_radius_mm=planet_radius,
        sun_pitch_points_local=sun_points,
        planet_pitch_points_local=planet_points,
        planet_center_points_world=planet_center,
        contact_points_world=contact_world,
        biological_start_index=biological_start,
        biological_end_index=biological_end,
        extension_teeth=extension_teeth,
        module_mm=float(module_mm),
        extension_pitch_length_mm=extension_length,
        center_distance_mm=data.center_distance_mm,
    )


def assembled_planet_points(data: PlanetaryPitchCurveData, index: int) -> FloatArray:
    return (
        rotate_points(data.planet_pitch_points_local, data.planet_absolute_angle_rad[index])
        + data.planet_center_points_world[index]
    )


def _angle_between_degrees(a: FloatArray, b: FloatArray) -> FloatArray:
    a_unit = a / np.linalg.norm(a, axis=1)[:, None]
    b_unit = b / np.linalg.norm(b, axis=1)[:, None]
    cosine = np.clip(np.abs(np.sum(a_unit * b_unit, axis=1)), 0.0, 1.0)
    return np.degrees(np.arccos(cosine))


def validate_planetary_pitch_curves(
    data: PlanetaryPitchCurveData,
) -> PlanetaryValidation:
    parameter = np.radians(data.elevation_deg)
    center_error = float(
        np.max(
            np.abs(
                np.linalg.norm(data.planet_center_points_world, axis=1)
                - data.center_distance_mm
            )
        )
    )

    planet_contact_world = data.planet_center_points_world + np.column_stack(
        (
            data.planet_pitch_radius_mm * np.cos(data.carrier_angle_rad + np.pi),
            data.planet_pitch_radius_mm * np.sin(data.carrier_angle_rad + np.pi),
        )
    )
    contact_error = float(
        np.max(np.linalg.norm(planet_contact_world - data.contact_points_world, axis=1))
    )

    sun_tangent = np.gradient(data.sun_pitch_points_local, parameter, axis=0)
    planet_local_tangent = np.gradient(data.planet_pitch_points_local, parameter, axis=0)
    planet_tangent_world = np.asarray(
        [
            rotate_points(planet_local_tangent[[i]], data.planet_absolute_angle_rad[i])[0]
            for i in range(len(parameter))
        ]
    )
    tangent_error = float(np.max(_angle_between_degrees(sun_tangent, planet_tangent_world)))

    sun_relative_rate = -data.dcarrier_dE
    rolling_residual = (
        sun_relative_rate * data.sun_pitch_radius_mm
        + data.dplanet_relative_dE * data.planet_pitch_radius_mm
    )
    rolling_error = float(np.max(np.abs(rolling_residual)))

    # Reconstruct the planet angle only from carrier motion and the synthesized
    # signed ratio. This makes the trajectory-error check independent of the
    # stored target angle after radius synthesis.
    reconstructed_derivative = data.dcarrier_dE * (1.0 - data.signed_ratio)
    reconstructed = cumulative_trapezoid(
        reconstructed_derivative, parameter, initial=0.0
    )
    elevation_error_deg = np.degrees(
        reconstructed - data.planet_absolute_angle_rad
    )
    max_error = float(np.max(np.abs(elevation_error_deg)))
    rms_error = float(np.sqrt(np.mean(elevation_error_deg**2)))
    endpoint_error = float(abs(elevation_error_deg[-1]))

    finite_positive = bool(
        np.all(np.isfinite(data.sun_pitch_radius_mm))
        and np.all(np.isfinite(data.planet_pitch_radius_mm))
        and np.all(data.sun_pitch_radius_mm > 0.0)
        and np.all(data.planet_pitch_radius_mm > 0.0)
    )
    no_sign_discontinuity = bool(
        np.all(np.isfinite(data.signed_ratio)) and np.all(data.signed_ratio < 0.0)
    )
    sun_simple = bool(LineString(data.sun_pitch_points_local).is_simple)
    planet_simple = bool(LineString(data.planet_pitch_points_local).is_simple)

    passed = bool(
        np.all(data.sun_angle_rad == 0.0)
        and center_error <= 1e-9
        and contact_error <= 1e-9
        and tangent_error <= 0.1
        and rolling_error <= 1e-9
        and finite_positive
        and no_sign_discontinuity
        and sun_simple
        and planet_simple
        and max_error <= 0.02
        and endpoint_error <= 0.02
    )
    return PlanetaryValidation(
        passed=passed,
        sample_count=len(data.elevation_deg),
        valid_range_deg=(float(data.elevation_deg[0]), float(data.elevation_deg[-1])),
        sun_stationary=bool(np.all(data.sun_angle_rad == 0.0)),
        carrier_matches_st=True,
        planet_matches_ht=True,
        center_distance_error_mm=center_error,
        contact_coincidence_error_mm=contact_error,
        tangent_alignment_error_deg=tangent_error,
        rolling_residual_mm_per_rad=rolling_error,
        finite_positive_pitch_radii=finite_positive,
        no_sign_discontinuity=no_sign_discontinuity,
        sun_curve_simple=sun_simple,
        planet_curve_simple=planet_simple,
        no_extrapolation=True,
        maximum_elevation_error_deg=max_error,
        rms_elevation_error_deg=rms_error,
        endpoint_elevation_error_deg=endpoint_error,
    )


def write_planetary_report(
    validation: PlanetaryValidation, destination: str | Path
) -> Path:
    destination = Path(destination)
    verdict = "PASS" if validation.passed else "FAIL"
    lines = [
        "# Literature Planetary Pitch-Curve Validation",
        "",
        f"**Result: {verdict}**",
        "",
        "## Mechanism",
        "",
        "- Sun angle: fixed at 0 rad",
        "- Carrier angle: McClure2001 scapular excursion",
        "- Planet absolute angle: humerothoracic elevation excursion",
        "- Contact type: external sun/planet rolling",
        "- Teeth: not generated",
        "",
        "## Signed rolling equation",
        "",
        "`(-dθc/dE) r_s + (dθp/dE - dθc/dE) r_p = 0`",
        "",
        "The signed ratio `dθp_rel/dθs_rel` is negative because external gears",
        "rotate in opposite directions when measured relative to the carrier.",
        "",
        "## Validation metrics",
        "",
    ]
    for key, value in asdict(validation).items():
        lines.append(f"- {key}: {value}")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def write_planetary_validation_json(
    validation: PlanetaryValidation, destination: str | Path
) -> Path:
    destination = Path(destination)
    destination.write_text(
        json.dumps(asdict(validation), indent=2) + "\n", encoding="utf-8"
    )
    return destination
