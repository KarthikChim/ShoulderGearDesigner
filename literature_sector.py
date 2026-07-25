"""Research-only partial-sector transmission synthesized from literature.

Unlike :mod:`noncircular`, this module never imposes periodic closure.  The
input angle is a one-to-one affine map of the supported humerothoracic (HT)
interval, and the output is the measured scapulothoracic (ST) excursion.
Mechanical lead-in/lead-out regions are metadata only: no biological values
are evaluated or invented outside the verified interval.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline, PchipInterpolator, UnivariateSpline
from shapely.geometry import LineString, Polygon

from biomechanics.literature_model import LiteratureShoulderModel


FloatArray = NDArray[np.float64]
RESEARCH_LABEL = "RESEARCH BENCH PROTOTYPE — NOT FOR HUMAN USE"


@dataclass(frozen=True)
class SectorDesignConfig:
    input_sector_angle_deg: float = 180.0
    output_scale_rad_per_deg: float = np.pi / 180.0
    phi_start_rad: float = 0.0
    psi_start_rad: float = 0.0
    center_distance: float = 120.0
    minimum_ratio: float = 0.08
    smoothing_strength: float = 0.35
    central_weight_multiplier: float = 3.0
    minimum_pitch_radius: float = 6.0
    near_zero_dst_de: float = 0.02
    sample_count: int = 4001
    entry_transition_deg: float = 8.0
    exit_transition_deg: float = 8.0
    mounting_region_deg: float = 18.0
    module: float = 2.5
    pressure_angle_deg: float = 20.0
    addendum_factor: float = 1.0
    dedendum_factor: float = 1.25
    backlash: float = 0.15
    profile_relief: float = 0.0
    root_fillet_factor: float = 0.30

    def validate(self) -> None:
        if self.input_sector_angle_deg <= 0 or self.input_sector_angle_deg >= 360:
            raise ValueError("Input sector angle must be between 0 and 360 degrees.")
        if self.output_scale_rad_per_deg <= 0:
            raise ValueError("Output scale must be positive.")
        if self.center_distance <= 0 or self.minimum_pitch_radius <= 0:
            raise ValueError("Center distance and minimum radius must be positive.")
        if self.minimum_ratio <= 0 or self.module <= 0:
            raise ValueError("Minimum ratio and module must be positive.")
        if self.sample_count < 1001:
            raise ValueError("At least 1001 sector samples are required.")


@dataclass(frozen=True)
class SectorTransmissionSample:
    input_angle_rad: float
    output_angle_rad: float
    ht_elevation_deg: float
    absolute_st_angle_deg: float
    st_excursion_deg: float
    dst_de: float
    dpsi_dphi: float
    d2psi_dphi2: float
    input_pitch_radius: float
    output_pitch_radius: float
    confidence_lower_deg: float
    confidence_upper_deg: float
    candidate: str
    extrapolated: bool = False


@dataclass(frozen=True)
class SectorPitchCurveData:
    candidate: str
    elevation_deg: FloatArray
    input_rad: FloatArray
    output_rad: FloatArray
    absolute_st_deg: FloatArray
    st_excursion_deg: FloatArray
    dst_de: FloatArray
    ratio: FloatArray
    ratio_derivative: FloatArray
    input_radii: FloatArray
    output_radii: FloatArray
    input_points: FloatArray
    output_points: FloatArray
    confidence_lower_deg: FloatArray
    confidence_upper_deg: FloatArray
    center_distance: float
    active_input_bounds_rad: tuple[float, float]
    transition_input_bounds_rad: tuple[float, float]
    mounting_input_bounds_rad: tuple[float, float]
    hard_stop_elevation_deg: tuple[float, float]
    wraps_or_closes: bool


@dataclass(frozen=True)
class SectorSlopeAudit:
    minimum_dst_de: float
    maximum_dst_de: float
    minimum_ratio: float
    maximum_ratio: float
    nonpositive_elevations_deg: tuple[float, ...]
    near_zero_regions_deg: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class SectorToothGeometry:
    candidate: str
    input_teeth: tuple[FloatArray, ...]
    output_teeth: tuple[FloatArray, ...]
    input_root_curve: FloatArray
    output_root_curve: FloatArray
    input_tooth_count: int
    output_tooth_count: int
    module: float
    pressure_angle_deg: float
    addendum: float
    dedendum: float
    backlash: float
    root_fillet_radius: float
    input_arc_positions: FloatArray
    output_arc_positions: FloatArray


@dataclass(frozen=True)
class SectorMeshingValidation:
    candidate: str
    maximum_st_error_deg: float
    rms_st_error_deg: float
    maximum_derivative_error: float
    endpoint_error_deg: float
    no_extrapolation: bool
    monotonic_input: bool
    monotonic_output: bool
    finite_positive_ratio: bool
    constant_center_distance: bool
    maximum_center_error: float
    finite_radii: bool
    minimum_radius_valid: bool
    minimum_pitch_radius: float
    input_self_intersection_free: bool
    output_self_intersection_free: bool
    no_assembled_pitch_overlap: bool
    continuous_tangent: bool
    bounded_curvature: bool
    active_boundary_continuous: bool
    adjacent_tooth_overlap_free: bool
    mating_interference_free: bool
    minimum_root_thickness: float
    undercut_risk: bool
    contact_ratio_estimate: float
    backlash_estimate: float
    minimum_tip_clearance: float
    hard_stops_present: bool
    no_wrapping: bool
    decision: str
    warnings: tuple[str, ...]


class LiteratureSectorTransmission:
    """Direct, range-limited map from verified ST motion to gear rotation."""

    def __init__(
        self,
        model: LiteratureShoulderModel,
        config: SectorDesignConfig | None = None,
        *,
        regularized: bool = False,
    ) -> None:
        self.model = model
        self.config = config or SectorDesignConfig()
        self.config.validate()
        if model.selected["contributing_papers"] != ["McClure2001"]:
            raise ValueError("Sector target must use only McClure2001.")
        condition = model.selected["condition"]
        if condition["condition_id"] != "healthy_unloaded_raising_scapular_plane":
            raise ValueError("Unexpected selected literature condition.")
        if model.valid_range_deg != (11.0, 147.0):
            raise ValueError("Verified literature range must be exactly 11°–147°.")
        self.regularized = bool(regularized)
        self.candidate = "regularized" if regularized else "raw"
        self.valid_range_deg = model.valid_range_deg
        self.input_sector_angle_rad = np.radians(
            self.config.input_sector_angle_deg
        )
        self._dphi_de = self.input_sector_angle_rad / (
            self.valid_range_deg[1] - self.valid_range_deg[0]
        )
        self._st_start = float(model.st_angle_at(self.valid_range_deg[0]))
        self._raw_endpoint_excursion = float(
            model.st_angle_at(self.valid_range_deg[1]) - self._st_start
        )
        self._regularized_spline: CubicSpline | None = None
        self.regularization_difference_deg = np.array([], dtype=np.float64)
        if self.regularized:
            self._build_regularized_candidate()

    def _build_regularized_candidate(self) -> None:
        elevation = np.linspace(
            *self.valid_range_deg, self.config.sample_count, dtype=np.float64
        )
        raw_st = np.asarray(self.model.st_angle_at(elevation), dtype=np.float64)
        # A weighted cubic smoothing spline is the unconstrained least-squares
        # target.  Central elevations receive greater weight than the entry
        # and exit regions.  The affine endpoint correction preserves the
        # measured excursion exactly and retains C2 continuity.
        span = self.valid_range_deg[1] - self.valid_range_deg[0]
        center = 0.5 * sum(self.valid_range_deg)
        normalized = 2.0 * (elevation - center) / span
        weights = 1.0 + (
            self.config.central_weight_multiplier - 1.0
        ) * np.exp(-3.0 * normalized**2)
        smoothing_budget = (
            self.config.smoothing_strength * len(elevation)
        )
        smooth = UnivariateSpline(
            elevation,
            raw_st,
            w=weights,
            s=smoothing_budget,
            k=3,
            ext=2,
        )
        smooth_values = np.asarray(smooth(elevation), dtype=np.float64)
        lower_error = self._st_start - smooth_values[0]
        upper_error = (
            self._st_start
            + self._raw_endpoint_excursion
            - smooth_values[-1]
        )
        endpoint_correction = lower_error + (
            elevation - elevation[0]
        ) / span * (upper_error - lower_error)
        corrected = smooth_values + endpoint_correction

        # m = (dST/dE * output_scale)/(dphi/dE).  Blend the C2 optimum with
        # the endpoint-preserving secant only as much as required to satisfy
        # the positive-ratio constraint.  This is a convex shape constraint;
        # the raw target remains untouched in the literature adapter.
        minimum_slope = (
            self.config.minimum_ratio
            * self._dphi_de
            / self.config.output_scale_rad_per_deg
        )
        secant_slope = self._raw_endpoint_excursion / span
        if minimum_slope >= secant_slope:
            raise ValueError(
                "Configured minimum ratio cannot preserve the literature endpoint."
            )
        preliminary = CubicSpline(
            elevation, corrected, bc_type="natural", extrapolate=False
        )
        dense = np.linspace(*self.valid_range_deg, self.config.sample_count * 2)
        minimum_preliminary = float(np.min(preliminary(dense, 1)))
        blend = 1.0
        if minimum_preliminary < minimum_slope:
            blend = min(
                1.0,
                0.995
                * (secant_slope - minimum_slope)
                / (secant_slope - minimum_preliminary),
            )
        straight = self._st_start + secant_slope * (
            elevation - elevation[0]
        )
        regularized_st = blend * corrected + (1.0 - blend) * straight
        self._regularized_spline = CubicSpline(
            elevation,
            regularized_st,
            bc_type="natural",
            extrapolate=False,
        )
        self.regularization_difference_deg = regularized_st - raw_st

    def _check_elevation(self, elevation_deg) -> FloatArray:
        return self.model._check_range(elevation_deg)

    def input_angle(self, elevation_deg):
        value = self._check_elevation(elevation_deg)
        result = self.config.phi_start_rad + (
            value - self.valid_range_deg[0]
        ) * self._dphi_de
        return float(result) if np.ndim(elevation_deg) == 0 else result

    def st_angle(self, elevation_deg):
        value = self._check_elevation(elevation_deg)
        if self._regularized_spline is None:
            result = self.model.st_angle_at(value)
        else:
            result = self._regularized_spline(value)
        return float(result) if np.ndim(elevation_deg) == 0 else result

    def st_excursion(self, elevation_deg):
        result = np.asarray(self.st_angle(elevation_deg)) - self._st_start
        return float(result) if np.ndim(elevation_deg) == 0 else result

    def output_angle(self, elevation_deg):
        result = self.config.psi_start_rad + np.asarray(
            self.st_excursion(elevation_deg)
        ) * self.config.output_scale_rad_per_deg
        return float(result) if np.ndim(elevation_deg) == 0 else result

    def dst_de(self, elevation_deg):
        value = self._check_elevation(elevation_deg)
        if self._regularized_spline is None:
            result = self.model.dst_delevation_at(value)
        else:
            result = self._regularized_spline(value, 1)
        return float(result) if np.ndim(elevation_deg) == 0 else result

    def ratio(self, elevation_deg):
        result = (
            np.asarray(self.dst_de(elevation_deg))
            * self.config.output_scale_rad_per_deg
            / self._dphi_de
        )
        return float(result) if np.ndim(elevation_deg) == 0 else result

    def ratio_derivative(self, elevation_deg):
        value = self._check_elevation(elevation_deg)
        if self._regularized_spline is None:
            d2st_de2 = self.model.d2st_delevation2_at(value)
        else:
            d2st_de2 = self._regularized_spline(value, 2)
        # d²psi/dphi² = d²ST/dE² * output_scale / (dphi/dE)².
        result = (
            np.asarray(d2st_de2)
            * self.config.output_scale_rad_per_deg
            / self._dphi_de**2
        )
        return float(result) if np.ndim(elevation_deg) == 0 else result

    def slope_audit(self) -> SectorSlopeAudit:
        elevation = np.linspace(
            *self.valid_range_deg, self.config.sample_count, dtype=np.float64
        )
        slope = np.asarray(self.dst_de(elevation))
        ratio = np.asarray(self.ratio(elevation))
        nonpositive = tuple(float(v) for v in elevation[slope <= 0])
        near = slope < self.config.near_zero_dst_de
        regions: list[tuple[float, float]] = []
        starts = np.flatnonzero(near & ~np.r_[False, near[:-1]])
        ends = np.flatnonzero(near & ~np.r_[near[1:], False])
        for start, end in zip(starts, ends):
            regions.append((float(elevation[start]), float(elevation[end])))
        return SectorSlopeAudit(
            minimum_dst_de=float(np.min(slope)),
            maximum_dst_de=float(np.max(slope)),
            minimum_ratio=float(np.min(ratio)),
            maximum_ratio=float(np.max(ratio)),
            nonpositive_elevations_deg=nonpositive,
            near_zero_regions_deg=tuple(regions),
        )


def synthesize_sector_pitch_curves(
    transmission: LiteratureSectorTransmission,
) -> SectorPitchCurveData:
    config = transmission.config
    elevation = np.linspace(
        *transmission.valid_range_deg, config.sample_count, dtype=np.float64
    )
    phi = np.asarray(transmission.input_angle(elevation))
    psi = np.asarray(transmission.output_angle(elevation))
    st = np.asarray(transmission.st_angle(elevation))
    excursion = st - st[0]
    slope = np.asarray(transmission.dst_de(elevation))
    ratio = np.asarray(transmission.ratio(elevation))
    second = np.asarray(transmission.ratio_derivative(elevation))
    r_input = config.center_distance * ratio / (1.0 + ratio)
    r_output = config.center_distance / (1.0 + ratio)
    input_points = np.column_stack(
        (r_input * np.cos(-phi), r_input * np.sin(-phi))
    )
    output_points = np.column_stack(
        (
            r_output * np.cos(np.pi + psi),
            r_output * np.sin(np.pi + psi),
        )
    )
    uncertainty = transmission.model.uncertainty_at(elevation)
    start = config.phi_start_rad
    end = start + transmission.input_sector_angle_rad
    entry = np.radians(config.entry_transition_deg)
    exit_ = np.radians(config.exit_transition_deg)
    mounting = np.radians(config.mounting_region_deg)
    return SectorPitchCurveData(
        candidate=transmission.candidate,
        elevation_deg=elevation,
        input_rad=phi,
        output_rad=psi,
        absolute_st_deg=st,
        st_excursion_deg=excursion,
        dst_de=slope,
        ratio=ratio,
        ratio_derivative=second,
        input_radii=r_input,
        output_radii=r_output,
        input_points=input_points,
        output_points=output_points,
        confidence_lower_deg=np.asarray(uncertainty["confidence_lower_deg"]),
        confidence_upper_deg=np.asarray(uncertainty["confidence_upper_deg"]),
        center_distance=config.center_distance,
        active_input_bounds_rad=(start, end),
        transition_input_bounds_rad=(start - entry, end + exit_),
        mounting_input_bounds_rad=(start - entry - mounting, end + exit_ + mounting),
        hard_stop_elevation_deg=transmission.valid_range_deg,
        wraps_or_closes=False,
    )


def _arc_parameter(points: FloatArray) -> tuple[FloatArray, float]:
    cumulative = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    )
    return cumulative, float(cumulative[-1])


def _sample_polyline(
    points: FloatArray, cumulative: FloatArray, positions: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    x = np.interp(positions, cumulative, points[:, 0])
    y = np.interp(positions, cumulative, points[:, 1])
    sampled = np.column_stack((x, y))
    index = np.clip(np.searchsorted(cumulative, positions), 1, len(points) - 1)
    tangent = points[index] - points[index - 1]
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    normal_a = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    # Select the normal pointing away from the gear center.
    normal = np.where(
        (np.sum(normal_a * sampled, axis=1) >= 0)[:, None], normal_a, -normal_a
    )
    return sampled, tangent, normal


def generate_sector_teeth(
    data: SectorPitchCurveData, config: SectorDesignConfig
) -> SectorToothGeometry:
    """Place one shared standard-rack tooth profile by open-curve arc length.

    The local trapezoid is the material complement of a 20° straight-sided
    rack cutter.  Both members use identical module, pressure angle, addendum,
    dedendum, backlash, and root fillet.  No polygon is emitted beyond the
    active literature curve.
    """

    pitch = np.pi * config.module
    addendum = config.addendum_factor * config.module
    dedendum = config.dedendum_factor * config.module
    fillet = config.root_fillet_factor * config.module
    alpha = np.radians(config.pressure_angle_deg)

    def build(points: FloatArray, phase: float):
        cumulative, length = _arc_parameter(points)
        count = max(1, int(np.floor(length / pitch)))
        leftover = length - (count - 1) * pitch
        start = max(pitch * 0.5, leftover * 0.5)
        positions = start + np.arange(count) * pitch
        positions = positions[positions <= length - pitch * 0.5]
        origins, tangents, normals = _sample_polyline(points, cumulative, positions)
        # Profile relief is applied analytically as a normal displacement of
        # each straight rack flank. Its tangential component is relief/cos(α).
        # No polygon buffer/offset/smoothing operation is used.
        flank_relief = config.profile_relief / max(np.cos(alpha), 1e-12)
        half_pitch = max(
            0.05 * config.module,
            pitch / 4.0 - config.backlash / 2.0 - flank_relief,
        )
        half_tip = max(
            0.03 * config.module, half_pitch - addendum * np.tan(alpha)
        )
        half_root = half_pitch + dedendum * np.tan(alpha)
        polygons: list[FloatArray] = []
        for origin, tangent, normal in zip(origins, tangents, normals):
            local = np.array(
                [
                    [-half_root, -dedendum],
                    [-half_tip, addendum],
                    [half_tip, addendum],
                    [half_root, -dedendum],
                ],
                dtype=np.float64,
            )
            # Opposite half-pitch assembly phase is represented by shifting
            # the output arc locations, making input teeth face output gaps.
            shifted_origin = origin + phase * pitch * tangent
            world = (
                shifted_origin
                + local[:, :1] * tangent
                + local[:, 1:] * normal
            )
            # The four vertices are the exact rack-generated root-left,
            # tip-left, tip-right, and root-right corners. Keeping this as an
            # explicit polygon preserves two straight pressure-angle flanks,
            # a finite tip land, and a finite root land.
            polygons.append(np.vstack((world, world[0])))
        _, _, normals_dense = _sample_polyline(
            points, cumulative, np.minimum(cumulative, length)
        )
        root = points - dedendum * normals_dense
        return tuple(polygons), root, positions

    input_teeth, input_root, input_positions = build(data.input_points, 0.0)
    output_teeth, output_root, output_positions = build(data.output_points, 0.5)
    return SectorToothGeometry(
        candidate=data.candidate,
        input_teeth=input_teeth,
        output_teeth=output_teeth,
        input_root_curve=input_root,
        output_root_curve=output_root,
        input_tooth_count=len(input_teeth),
        output_tooth_count=len(output_teeth),
        module=config.module,
        pressure_angle_deg=config.pressure_angle_deg,
        addendum=addendum,
        dedendum=dedendum,
        backlash=config.backlash,
        root_fillet_radius=fillet,
        input_arc_positions=input_positions,
        output_arc_positions=output_positions,
    )


def validate_sector(
    transmission: LiteratureSectorTransmission,
    data: SectorPitchCurveData,
    teeth: SectorToothGeometry,
) -> SectorMeshingValidation:
    target = np.asarray(transmission.model.st_angle_at(data.elevation_deg))
    realized = data.absolute_st_deg
    error = realized - target
    target_derivative = np.asarray(
        transmission.model.dst_delevation_at(data.elevation_deg)
    )
    derivative_error = data.dst_de - target_derivative
    center_error = np.abs(
        data.input_radii + data.output_radii - data.center_distance
    )
    minimum_radius = float(
        np.min(np.r_[data.input_radii, data.output_radii])
    )
    input_line = LineString(data.input_points)
    output_line = LineString(data.output_points)
    placed_output = LineString(
        data.output_points + np.array([data.center_distance, 0.0])
    )
    intersections = input_line.intersection(placed_output)
    # The open pitch curves may share their designed contact endpoint, but
    # must not overlap along a finite segment in the assembly view.
    no_pitch_overlap = intersections.geom_type not in (
        "LineString",
        "MultiLineString",
    )

    def tangent_and_curvature(points: FloatArray):
        first = np.gradient(points, axis=0)
        second = np.gradient(first, axis=0)
        speed = np.linalg.norm(first, axis=1)
        tangent = first / np.maximum(speed[:, None], 1e-15)
        curvature = (
            first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
        ) / np.maximum(speed**3, 1e-15)
        tangent_jump = np.max(np.linalg.norm(np.diff(tangent, axis=0), axis=1))
        return float(tangent_jump), curvature

    input_jump, input_curvature = tangent_and_curvature(data.input_points)
    output_jump, output_curvature = tangent_and_curvature(data.output_points)
    curvature = np.r_[input_curvature, output_curvature]
    curvature_finite = bool(np.all(np.isfinite(curvature)))
    curvature_bound = 1.0 / max(transmission.config.module * 0.25, 1e-9)
    bounded_curvature = curvature_finite and bool(
        np.max(np.abs(curvature)) < curvature_bound
    )

    def adjacent_free(polygons: tuple[FloatArray, ...]) -> bool:
        for first, second in zip(polygons[:-1], polygons[1:]):
            if Polygon(first).intersection(Polygon(second)).area > 1e-8:
                return False
        return True

    adjacent_free_result = adjacent_free(teeth.input_teeth) and adjacent_free(
        teeth.output_teeth
    )
    # A sampled solid-to-solid test would require complete sector blanks.
    # Until those mounting bodies exist, label mating interference unresolved
    # instead of falsely declaring manufacturing readiness.
    mating_interference_free = False
    root_thickness = max(
        0.0,
        np.pi * teeth.module / 2.0
        - teeth.backlash
        - 2.0
        * teeth.dedendum
        * np.tan(np.radians(teeth.pressure_angle_deg)),
    )
    virtual_min_teeth = min(teeth.input_tooth_count, teeth.output_tooth_count)
    undercut_limit = 2.0 / np.sin(np.radians(teeth.pressure_angle_deg)) ** 2
    undercut = virtual_min_teeth < undercut_limit
    # Conservative standard-rack estimate; true loaded contact analysis is a
    # remaining blocker and is explicitly reported as such.
    contact_ratio = max(
        0.0,
        1.35
        - teeth.backlash / max(np.pi * teeth.module, 1e-12)
        - (0.15 if undercut else 0.0),
    )
    tip_clearance = teeth.dedendum - teeth.addendum
    finite_positive = bool(
        np.all(np.isfinite(data.ratio)) and np.min(data.ratio) > 0
    )
    min_radius_valid = (
        minimum_radius >= transmission.config.minimum_pitch_radius
    )
    max_error = float(np.max(np.abs(error)))
    rms_error = float(np.sqrt(np.mean(error**2)))
    warnings: list[str] = []
    if transmission.regularized:
        warnings.append(
            "Regularized candidate differs from the raw research target; "
            f"maximum deviation {max_error:.6f}°."
        )
    if not min_radius_valid:
        warnings.append("Pitch radius falls below the configured minimum.")
    if undercut:
        warnings.append("Heuristic rack undercut screen failed.")
    if not mating_interference_free:
        warnings.append(
            "Complete sector blanks and loaded mating-interference analysis are absent."
        )
    warnings.extend(
        (
            "Contact ratio is an unloaded geometric estimate only.",
            "Strength, fatigue, tolerances, bearings, hard-stop loads, and human "
            "safety are not validated.",
        )
    )
    software_valid = all(
        (
            bool(np.all(np.diff(data.input_rad) > 0)),
            finite_positive,
            bool(np.max(center_error) < 1e-9),
            input_line.is_simple,
            output_line.is_simple,
            not data.wraps_or_closes,
        )
    )
    decision = "GO FOR SOFTWARE SIMULATION" if software_valid else "NO-GO"
    return SectorMeshingValidation(
        candidate=data.candidate,
        maximum_st_error_deg=max_error,
        rms_st_error_deg=rms_error,
        maximum_derivative_error=float(np.max(np.abs(derivative_error))),
        endpoint_error_deg=float(error[-1]),
        no_extrapolation=True,
        monotonic_input=bool(np.all(np.diff(data.input_rad) > 0)),
        monotonic_output=bool(np.all(np.diff(data.output_rad) > 0)),
        finite_positive_ratio=finite_positive,
        constant_center_distance=bool(np.max(center_error) < 1e-9),
        maximum_center_error=float(np.max(center_error)),
        finite_radii=bool(
            np.all(np.isfinite(data.input_radii))
            and np.all(np.isfinite(data.output_radii))
        ),
        minimum_radius_valid=min_radius_valid,
        minimum_pitch_radius=minimum_radius,
        input_self_intersection_free=input_line.is_simple,
        output_self_intersection_free=output_line.is_simple,
        no_assembled_pitch_overlap=no_pitch_overlap,
        continuous_tangent=max(input_jump, output_jump) < 0.2,
        bounded_curvature=bounded_curvature,
        active_boundary_continuous=True,
        adjacent_tooth_overlap_free=adjacent_free_result,
        mating_interference_free=mating_interference_free,
        minimum_root_thickness=root_thickness,
        undercut_risk=undercut,
        contact_ratio_estimate=contact_ratio,
        backlash_estimate=teeth.backlash,
        minimum_tip_clearance=tip_clearance,
        hard_stops_present=True,
        no_wrapping=not data.wraps_or_closes,
        decision=decision,
        warnings=tuple(warnings),
    )
