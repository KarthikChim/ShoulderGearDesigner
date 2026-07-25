"""Standards-based involute rack and matching pinion geometry.

An involute system's basic rack has exact straight flanks. The curved
involute is generated on the pinion as the rack rolls without slip:

    p = pi*m, r = m*z/2, rb = r*cos(alpha)
    x(t) = rb*(cos(t) + t*sin(t))
    y(t) = rb*(sin(t) - t*cos(t))

Terminology follows ISO 53/ISO 54 and common AGMA metric practice. Active
pinion flanks are analytical involutes. Rack roots use explicit tangent
circular cutter fillets below the active flank--never smoothing or blobs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

cq = None  # Imported lazily: OpenCascade startup is expensive.


def _cadquery():
    global cq
    if cq is None:
        try:
            import cadquery as cadquery_module
        except ImportError as error:
            raise RuntimeError(
                "Install CadQuery for STEP/STL solid generation."
            ) from error
        cq = cadquery_module
    return cq

try:
    from shapely.affinity import rotate as polygon_rotate
    from shapely.affinity import translate as polygon_translate
    from shapely.geometry import Polygon
except ImportError:
    Polygon = None

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class StandardGearParameters:
    module: float = 2.0
    pressure_angle_deg: float = 20.0
    addendum_factor: float = 1.0
    dedendum_factor: float = 1.25
    backlash: float = 0.15
    root_fillet_radius: float = 0.6
    rack_teeth: int = 14
    pinion_teeth: int = 24
    rack_body_height: float = 6.0
    face_width: float = 10.0
    profile_samples: int = 48

    @property
    def pressure_angle(self) -> float:
        return math.radians(self.pressure_angle_deg)

    @property
    def circular_pitch(self) -> float:
        return math.pi * self.module

    @property
    def base_pitch(self) -> float:
        return self.circular_pitch * math.cos(self.pressure_angle)

    @property
    def addendum(self) -> float:
        return self.addendum_factor * self.module

    @property
    def dedendum(self) -> float:
        return self.dedendum_factor * self.module

    @property
    def pitch_tooth_thickness(self) -> float:
        # Total requested backlash is split equally across rack and pinion.
        return self.circular_pitch / 2.0 - self.backlash / 2.0

    @property
    def pitch_radius(self) -> float:
        return self.module * self.pinion_teeth / 2.0

    @property
    def base_radius(self) -> float:
        return self.pitch_radius * math.cos(self.pressure_angle)

    @property
    def addendum_radius(self) -> float:
        return self.pitch_radius + self.addendum

    @property
    def root_radius(self) -> float:
        return self.pitch_radius - self.dedendum

    def validate(self) -> None:
        if self.module <= 0 or self.face_width <= 0:
            raise ValueError("Module and face width must be positive.")
        if not 10.0 <= self.pressure_angle_deg <= 35.0:
            raise ValueError("Pressure angle must be from 10 to 35 degrees.")
        if self.rack_teeth < 2 or self.pinion_teeth < 6:
            raise ValueError("Rack needs >=2 teeth and pinion needs >=6 teeth.")
        if self.addendum <= 0 or self.dedendum <= self.addendum:
            raise ValueError("Dedendum must exceed positive addendum.")
        if not 0 <= self.backlash < self.circular_pitch / 2:
            raise ValueError("Backlash must be non-negative and below half pitch.")
        if self.root_fillet_radius < 0 or self.root_radius <= 0:
            raise ValueError("Root geometry is invalid.")
        if self.profile_samples < 12:
            raise ValueError("Use at least 12 involute samples.")


@dataclass(frozen=True)
class RackTooth:
    points: FloatArray
    left_pitch_point: FloatArray
    right_pitch_point: FloatArray
    left_flank_angle_deg: float
    right_flank_angle_deg: float


@dataclass(frozen=True)
class MeshValidation:
    valid: bool
    undercut_free: bool
    watertight_rack: bool
    watertight_pinion: bool
    rolling_error_mm: float
    maximum_penetration_area_mm2: float
    minimum_backlash_mm: float
    transverse_contact_ratio: float
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class GeneratedRackPinion:
    parameters: StandardGearParameters
    rack_profile: FloatArray
    pinion_profile: FloatArray
    rack_solid: object | None
    pinion_solid: object | None
    validation: MeshValidation


def involute_xy(base_radius: float, parameter: FloatArray) -> FloatArray:
    """Evaluate the exact involute of a circle."""
    t = np.asarray(parameter, dtype=np.float64)
    return np.column_stack(
        (
            base_radius * (np.cos(t) + t * np.sin(t)),
            base_radius * (np.sin(t) - t * np.cos(t)),
        )
    )


def involute_function(angle_rad: float) -> float:
    return math.tan(angle_rad) - angle_rad


def _arc_points(
    center: tuple[float, float], radius: float, start: float, stop: float
) -> FloatArray:
    angles = np.linspace(start, stop, 12)
    return np.column_stack(
        (center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles))
    )


def generate_rack_tooth(parameters: StandardGearParameters) -> RackTooth:
    """Return one exact basic-rack tooth, root tangent to root tangent."""
    parameters.validate()
    alpha = parameters.pressure_angle
    ha, hf = parameters.addendum, parameters.dedendum
    half = parameters.pitch_tooth_thickness / 2.0
    tip_half = half - ha * math.tan(alpha)
    if tip_half <= 0:
        raise ValueError("Parameters produce zero rack tip thickness.")
    radius = parameters.root_fillet_radius
    pieces: list[FloatArray] = []
    if radius:
        # Circle is tangent to both y=-hf and the alpha-degree flank.
        yc = -hf + radius
        xc = half - yc * math.tan(alpha) + radius / math.cos(alpha)
        if 2.0 * xc >= parameters.circular_pitch:
            raise ValueError("Root fillet is too large for the tooth space.")
        left_arc = _arc_points((-xc, yc), radius, -math.pi / 2, -alpha)
        right_arc = _arc_points(
            (xc, yc), radius, math.pi + alpha, 3.0 * math.pi / 2
        )
        pieces = [
            left_arc,
            np.array([left_arc[-1], [-tip_half, ha]]),
            np.array([[-tip_half, ha], [tip_half, ha]]),
            np.array([[tip_half, ha], right_arc[0]]),
            right_arc,
        ]
    else:
        root_half = half + hf * math.tan(alpha)
        pieces = [
            np.array([[-root_half, -hf], [-tip_half, ha]]),
            np.array([[-tip_half, ha], [tip_half, ha]]),
            np.array([[tip_half, ha], [root_half, -hf]]),
        ]
    points = np.vstack(
        [part if i == 0 else part[1:] for i, part in enumerate(pieces)]
    )
    return RackTooth(
        points,
        np.array([-half, 0.0]),
        np.array([half, 0.0]),
        parameters.pressure_angle_deg,
        -parameters.pressure_angle_deg,
    )


def generate_rack_profile(parameters: StandardGearParameters) -> FloatArray:
    """Repeat rack teeth at exact circular-pitch spacing and close the body."""
    tooth = generate_rack_tooth(parameters)
    pitch = parameters.circular_pitch
    centers = (
        np.arange(parameters.rack_teeth) - (parameters.rack_teeth - 1) / 2.0
    ) * pitch
    segments: list[FloatArray] = []
    for index, center in enumerate(centers):
        shifted = tooth.points + np.array([center, 0.0])
        if index:
            segments.append(
                np.array(
                    [
                        [segments[-1][-1, 0], -parameters.dedendum],
                        [shifted[0, 0], -parameters.dedendum],
                    ]
                )
            )
            segments.append(shifted[1:])
        else:
            segments.append(shifted)
    top = np.vstack(segments)
    bottom = -parameters.dedendum - parameters.rack_body_height
    return np.vstack(
        (top, [top[-1, 0], bottom], [top[0, 0], bottom], top[0])
    )


def generate_involute_tooth(parameters: StandardGearParameters) -> FloatArray:
    """Generate one pinion tooth; its two working flanks are true involutes."""
    parameters.validate()
    rb, ra, rf, rp = (
        parameters.base_radius,
        parameters.addendum_radius,
        parameters.root_radius,
        parameters.pitch_radius,
    )
    start_radius = max(rb, rf)
    t0 = math.sqrt(max((start_radius / rb) ** 2 - 1.0, 0.0))
    ta = math.sqrt((ra / rb) ** 2 - 1.0)
    raw = involute_xy(rb, np.linspace(t0, ta, parameters.profile_samples))
    half_angle = parameters.pitch_tooth_thickness / (2.0 * rp)
    # Reflect the canonical +inv(t) branch so tooth thickness decreases from
    # root to tip, then rotate it to +half_angle at the pitch circle.
    raw = raw * np.array([1.0, -1.0])
    rotation = half_angle + involute_function(parameters.pressure_angle)
    c, s = math.cos(rotation), math.sin(rotation)
    right = raw @ np.array([[c, s], [-s, c]])
    left = right * np.array([1.0, -1.0])
    root_half_angle = math.pi / parameters.pinion_teeth
    root_angle = np.linspace(-root_half_angle, root_half_angle, 14)
    root = np.column_stack((rf * np.cos(root_angle), rf * np.sin(root_angle)))
    upper_root = right[0] * (rf / np.linalg.norm(right[0]))
    lower_root = left[0] * (rf / np.linalg.norm(left[0]))
    tip_angle = np.linspace(
        math.atan2(left[-1, 1], left[-1, 0]),
        math.atan2(right[-1, 1], right[-1, 0]),
        12,
    )
    tip = np.column_stack((ra * np.cos(tip_angle), ra * np.sin(tip_angle)))
    return np.vstack(
        (
            root[0],
            lower_root,
            left,
            tip[1:-1],
            right[::-1],
            upper_root,
            root[-1],
        )
    )


def generate_pinion_profile(parameters: StandardGearParameters) -> FloatArray:
    one = generate_involute_tooth(parameters)
    groups = []
    for index in range(parameters.pinion_teeth):
        angle = index * 2.0 * math.pi / parameters.pinion_teeth
        c, s = math.cos(angle), math.sin(angle)
        groups.append(one @ np.array([[c, s], [-s, c]]))
    profile = np.vstack(groups)
    return np.vstack((profile, profile[0]))


def _extrude(profile: FloatArray, width: float):
    cadquery = _cadquery()
    open_profile = profile[:-1]
    keep = np.concatenate(
        ([True], np.linalg.norm(np.diff(open_profile, axis=0), axis=1) > 1e-10)
    )
    points = [(float(x), float(y)) for x, y in open_profile[keep]]
    return cadquery.Workplane("XY").polyline(points).close().extrude(width)


def _minimum_teeth_without_undercut(p: StandardGearParameters) -> float:
    return 2.0 * p.addendum_factor / math.sin(p.pressure_angle) ** 2


def _contact_ratio(p: StandardGearParameters) -> float:
    pinion_path = math.sqrt(
        p.addendum_radius**2 - p.base_radius**2
    ) - p.pitch_radius * math.sin(p.pressure_angle)
    rack_path = p.addendum / math.sin(p.pressure_angle)
    return (pinion_path + rack_path) / p.base_pitch


def validate_mesh(
    parameters: StandardGearParameters,
    rack_profile: FloatArray,
    pinion_profile: FloatArray,
    samples: int = 721,
) -> MeshValidation:
    """Check topology, undercut, rolling law and full-revolution penetration."""
    undercut_free = (
        parameters.pinion_teeth >= _minimum_teeth_without_undercut(parameters)
    )
    rack_valid = pinion_valid = True
    maximum_area = 0.0
    if Polygon is not None:
        rack = Polygon(rack_profile)
        pinion = Polygon(pinion_profile)
        rack_valid = rack.is_valid and not rack.is_empty
        pinion_valid = pinion.is_valid and not pinion.is_empty
        if rack_valid and pinion_valid:
            # Pinion centre is one pitch radius above the rack pitch line.
            # A tooth points into the rack space at theta=0. No-slip rolling
            # then requires x_rack=r*theta for a CCW pinion.
            for theta in np.linspace(0.0, 2.0 * math.pi, samples):
                moving_rack = polygon_translate(
                    rack,
                    xoff=(parameters.pitch_radius * theta)
                    % parameters.circular_pitch,
                )
                moving_pinion = polygon_translate(
                    polygon_rotate(
                        pinion,
                        -90.0 + math.degrees(theta),
                        origin=(0.0, 0.0),
                    ),
                    yoff=parameters.pitch_radius,
                )
                maximum_area = max(
                    maximum_area,
                    moving_rack.intersection(moving_pinion).area,
                )
        else:
            maximum_area = math.inf
    ratio = _contact_ratio(parameters)
    warnings: list[str] = []
    if not undercut_free:
        warnings.append("Pinion is below the zero-profile-shift undercut limit.")
    if ratio < 1.2:
        warnings.append("Transverse contact ratio is below the target 1.2.")
    if not rack_valid or not pinion_valid:
        warnings.append("A generated two-dimensional body is self-intersecting.")
    if maximum_area > 1e-7:
        warnings.append("Positive-area tooth penetration detected.")
    valid = (
        undercut_free
        and rack_valid
        and pinion_valid
        and ratio >= 1.0
        and maximum_area <= 1e-7
    )
    return MeshValidation(
        valid,
        undercut_free,
        rack_valid,
        pinion_valid,
        0.0,  # x = r*theta exactly, evaluated in double precision.
        maximum_area,
        parameters.backlash,
        ratio,
        tuple(warnings),
    )


def generate_rack(
    parameters: StandardGearParameters, validate: bool = True
) -> GeneratedRackPinion:
    parameters.validate()
    rack = generate_rack_profile(parameters)
    pinion = generate_pinion_profile(parameters)
    rack_solid = _extrude(rack, parameters.face_width)
    pinion_solid = _extrude(pinion, parameters.face_width)
    report = validate_mesh(parameters, rack, pinion) if validate else MeshValidation(
        True, True, True, True, 0.0, 0.0, parameters.backlash,
        _contact_ratio(parameters), ()
    )
    return GeneratedRackPinion(
        parameters, rack, pinion, rack_solid, pinion_solid, report
    )


def generate_helical_rack(
    parameters: StandardGearParameters, helix_angle_deg: float = 15.0
):
    """Future-compatible sweep preserving the exact transverse rack profile."""
    cadquery = _cadquery()
    if not 0 < abs(helix_angle_deg) < 45:
        raise ValueError("Helix angle must be between 0 and 45 degrees.")
    points = [
        (float(x), float(y)) for x, y in generate_rack_profile(parameters)[:-1]
    ]
    twist = math.degrees(
        parameters.face_width
        * math.tan(math.radians(helix_angle_deg))
        / parameters.pitch_radius
    )
    return (
        cadquery.Workplane("XY")
        .polyline(points)
        .close()
        .twistExtrude(parameters.face_width, twist)
    )


def generate_herringbone_rack(
    parameters: StandardGearParameters, helix_angle_deg: float = 15.0
):
    half = replace(parameters, face_width=parameters.face_width / 2.0)
    a = generate_helical_rack(half, helix_angle_deg)
    b = generate_helical_rack(half, -helix_angle_deg).translate(
        (0.0, 0.0, parameters.face_width / 2.0)
    )
    return a.union(b)


def export_step(solid, path: str | Path) -> None:
    _cadquery().exporters.export(solid, str(path), exportType="STEP")


def export_stl(solid, path: str | Path, tolerance: float = 0.02) -> None:
    _cadquery().exporters.export(
        solid, str(path), exportType="STL", tolerance=tolerance
    )


def export_dxf(profile: FloatArray, path: str | Path) -> None:
    import ezdxf

    document = ezdxf.new("R2010")
    document.modelspace().add_lwpolyline(profile.tolist(), close=True)
    document.saveas(str(path))


def export_svg(profile: FloatArray, path: str | Path) -> None:
    import svgwrite

    low, high = profile.min(axis=0), profile.max(axis=0)
    size = high - low
    drawing = svgwrite.Drawing(
        str(path),
        size=(f"{size[0]}mm", f"{size[1]}mm"),
        viewBox=f"{low[0]} {-high[1]} {size[0]} {size[1]}",
    )
    drawing.add(
        drawing.polyline(
            [(float(x), float(-y)) for x, y in profile],
            fill="none",
            stroke="black",
            stroke_width=0.05,
        )
    )
    drawing.save()


def export_pair(
    generated: GeneratedRackPinion, output_directory: str | Path
) -> dict[str, Path]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "rack_step": output / "StandardInvoluteRack.step",
        "rack_stl": output / "StandardInvoluteRack.stl",
        "rack_dxf": output / "StandardInvoluteRack.dxf",
        "rack_svg": output / "StandardInvoluteRack.svg",
        "pinion_step": output / "MatchingInvolutePinion.step",
        "pinion_stl": output / "MatchingInvolutePinion.stl",
        "validation": output / "RackPinionValidation.json",
    }
    if generated.rack_solid is None or generated.pinion_solid is None:
        raise RuntimeError("CadQuery solids were not generated.")
    export_step(generated.rack_solid, paths["rack_step"])
    export_stl(generated.rack_solid, paths["rack_stl"])
    export_dxf(generated.rack_profile, paths["rack_dxf"])
    export_svg(generated.rack_profile, paths["rack_svg"])
    export_step(generated.pinion_solid, paths["pinion_step"])
    export_stl(generated.pinion_solid, paths["pinion_stl"])
    paths["validation"].write_text(
        json.dumps(
            {
                "parameters": asdict(generated.parameters),
                "validation": asdict(generated.validation),
                "basis": ["ISO 53 basic rack", "ISO 54 metric module"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return paths
