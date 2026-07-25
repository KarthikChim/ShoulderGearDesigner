"""Physical tooth optimization on locked McClure non-circular pitch curves.

This module deliberately does not import or call pitch-curve synthesis.  A
``GearMeshOptimizer`` receives the already-verified literature prototype and
uses its immutable ``SectorPitchCurveData`` as geometry ground truth.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from bench_prototype import (
    BenchPrototype,
    BenchPrototypeConfig,
    build_bench_prototype,
)


MODULES = (1.5, 1.75, 2.0, 2.25, 2.5)
PRESSURE_ANGLES = (20.0, 25.0, 30.0)
BACKLASHES = (0.10, 0.15, 0.20, 0.25, 0.30)
PROFILE_RELIEFS = (0.00, 0.05, 0.10, 0.15)
FACE_WIDTHS = (8.0, 10.0, 12.0, 14.0, 16.0)
CENTER_OFFSETS = (-0.50, -0.25, 0.0, 0.25, 0.50)
ROOT_FILLETS = (0.4, 0.6, 0.8, 1.0)
ROOT_EMBEDS = (1.0, 1.5, 2.0)
HELIX_ANGLES = (10.0, 15.0, 20.0, 25.0)
TOOTH_STYLES = ("Spur", "Helical", "Herringbone")


@dataclass(frozen=True)
class GearMeshParameters:
    module_mm: float = 2.0
    pressure_angle_deg: float = 25.0
    backlash_mm: float = 0.15
    profile_relief_mm: float = 0.05
    face_width_mm: float = 14.0
    center_distance_offset_mm: float = 0.0
    root_fillet_mm: float = 0.8
    tooth_root_embed_mm: float = 2.0
    tooth_style: str = "Spur"
    helix_angle_deg: float = 0.0

    @property
    def print_style_penalty(self) -> float:
        return {"Spur": 0.0, "Helical": 0.12, "Herringbone": 0.28}[
            self.tooth_style
        ]


@dataclass(frozen=True)
class GearMeshCandidate:
    parameters: GearMeshParameters
    input_tooth_count: int
    output_tooth_count: int
    pitch_tooth_thickness_mm: float
    minimum_root_thickness_mm: float
    minimum_tip_thickness_mm: float
    contact_ratio_estimate: float
    effective_backlash_mm: float
    angular_lost_motion_deg: float
    minimum_clearance_mm: float
    undercut_risk: bool
    printable_04_nozzle: bool
    strength_index: float
    engagement_smoothness: float
    print_sensitivity: float
    biomechanical_deviation_deg: float
    no_interference: bool
    score: float


@dataclass(frozen=True)
class GearOptimizationResult:
    preferred: GearMeshCandidate
    top_candidates: tuple[GearMeshCandidate, ...]
    evaluated_candidates: int
    locked_pitch_sha256: str
    module_study: tuple[GearMeshCandidate, ...]
    style_study: tuple[GearMeshCandidate, ...]
    preferred_validation: "PreferredPhysicalValidation"


@dataclass(frozen=True)
class PreferredPhysicalValidation:
    pitch_arrays_identical: bool
    maximum_tooth_penetration_area_mm2: float
    minimum_noncontact_clearance_mm: float
    adjacent_teeth_overlap_free: bool
    no_tooth_skipping: bool
    valid_closed_bodies: bool
    passed: bool


def pitch_curve_fingerprint(prototype: BenchPrototype) -> str:
    """Hash every locked motion/pitch array used by the physical search."""

    digest = hashlib.sha256()
    data = prototype.pitch_data
    for values in (
        data.elevation_deg,
        data.input_rad,
        data.output_rad,
        data.ratio,
        data.input_radii,
        data.output_radii,
        data.input_points,
        data.output_points,
    ):
        digest.update(np.ascontiguousarray(values, dtype=np.float64).tobytes())
    digest.update(repr(data.hard_stop_elevation_deg).encode())
    return digest.hexdigest()


class GearMeshOptimizer:
    """Search printable tooth geometry while preserving the pitch curves."""

    def __init__(self, locked_prototype: BenchPrototype) -> None:
        self.prototype = locked_prototype
        self.pitch_data = locked_prototype.pitch_data
        self.transmission = locked_prototype.transmission
        self.locked_pitch_sha256 = pitch_curve_fingerprint(locked_prototype)
        self._input_length = self._arc_length(self.pitch_data.input_points)
        self._output_length = self._arc_length(self.pitch_data.output_points)
        self._minimum_radius = float(
            np.min(
                np.r_[
                    self.pitch_data.input_radii,
                    self.pitch_data.output_radii,
                ]
            )
        )
        self._mean_output_radius = float(
            np.mean(self.pitch_data.output_radii)
        )

    @staticmethod
    def _arc_length(points: np.ndarray) -> float:
        return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))

    def assert_pitch_curves_locked(self) -> None:
        current = pitch_curve_fingerprint(self.prototype)
        if current != self.locked_pitch_sha256:
            raise RuntimeError("Locked literature pitch curves were modified.")

    def evaluate(self, parameters: GearMeshParameters) -> GearMeshCandidate:
        """Evaluate print and mesh proxies without changing pitch geometry."""

        self.assert_pitch_curves_locked()
        p = parameters
        if p.tooth_style not in TOOTH_STYLES:
            raise ValueError(f"Unknown tooth style: {p.tooth_style}")
        module = p.module_mm
        alpha = np.radians(p.pressure_angle_deg)
        circular_pitch = np.pi * module
        input_count = max(1, int(np.floor(self._input_length / circular_pitch)))
        output_count = max(1, int(np.floor(self._output_length / circular_pitch)))

        # Standard rack thickness relations at pitch, root, and tip.
        pitch_thickness = circular_pitch / 2.0 - p.backlash_mm
        root_thickness = (
            pitch_thickness
            + 2.0 * 1.25 * module * np.tan(alpha)
            - 2.0 * p.profile_relief_mm
        )
        tip_thickness = (
            pitch_thickness
            - 2.0 * module * np.tan(alpha)
            - 2.0 * p.profile_relief_mm
        )

        # Positive center offset opens the mesh; negative offset tightens it.
        effective_backlash = max(
            0.05,
            p.backlash_mm
            + 2.0 * p.profile_relief_mm
            + 2.0 * p.center_distance_offset_mm * np.tan(alpha),
        )
        angular_lost_motion = np.degrees(
            effective_backlash / max(self._mean_output_radius, 1e-12)
        )
        # Empirical non-circular flank envelope from the existing 2,001-frame
        # polygon sweep: about 0.30 module of combined backlash + bilateral
        # relief is consumed by curvature/placement variation before free
        # clearance appears.
        clearance = (
            p.backlash_mm
            + 2.0 * p.profile_relief_mm
            - 0.30 * module
            + p.center_distance_offset_mm
            + 0.03 * module
        )
        flank_envelope_ok = (
            p.backlash_mm + 2.0 * p.profile_relief_mm
            >= 0.30 * module - 1e-12
        )

        virtual_teeth = 2.0 * self._minimum_radius / module
        undercut_limit = 2.0 / max(np.sin(alpha) ** 2, 1e-12)
        undercut = virtual_teeth < undercut_limit
        transverse_contact = (
            1.38
            - effective_backlash / max(circular_pitch, 1e-12)
            + 0.015 * max(0, min(input_count, output_count) - 10)
            - (0.12 if undercut else 0.0)
        )
        overlap_ratio = 0.0
        if p.tooth_style in ("Helical", "Herringbone"):
            overlap_ratio = (
                p.face_width_mm
                * np.tan(np.radians(p.helix_angle_deg))
                / max(circular_pitch, 1e-12)
            )
        contact_ratio = max(0.0, transverse_contact + overlap_ratio)

        # A 0.4 mm nozzle needs at least two extrusion widths at the tip, with
        # extra margin retained here for dimensional scatter and elephant foot.
        printable = (
            tip_thickness >= 0.9
            and root_thickness >= 1.5
            and module >= 2.0
            and clearance >= 0.05
            and flank_envelope_ok
        )
        no_interference = (
            clearance >= 0.0
            and tip_thickness > 0.0
            and flank_envelope_ok
        )
        strength = (
            root_thickness
            * p.face_width_mm
            * (1.0 + 0.08 * p.root_fillet_mm / module)
            * (1.0 + 0.05 * p.tooth_root_embed_mm)
        )
        smoothness = contact_ratio * (
            1.0 + 0.015 * min(input_count, output_count)
        ) / (
            1.0 + angular_lost_motion + 0.4 * p.print_style_penalty
        )
        sensitivity = (
            0.4 / module
            + effective_backlash / max(module, 1e-12)
            + (0.20 if tip_thickness < 1.0 else 0.0)
            + p.print_style_penalty
        )

        # Pitch geometry is immutable, so physical candidates have exactly
        # zero modeled biomechanical deviation by construction.
        biomechanical_deviation = 0.0
        gate_penalty = 0.0 if (no_interference and printable) else 1_000_000.0
        score = (
            gate_penalty
            + 20_000.0 * angular_lost_motion
            - 2_000.0 * contact_ratio
            - 2.0 * strength
            - 200.0 * smoothness
            + 500.0 * sensitivity
            + (750.0 if undercut else 0.0)
        )
        return GearMeshCandidate(
            parameters=p,
            input_tooth_count=input_count,
            output_tooth_count=output_count,
            pitch_tooth_thickness_mm=float(pitch_thickness),
            minimum_root_thickness_mm=float(root_thickness),
            minimum_tip_thickness_mm=float(tip_thickness),
            contact_ratio_estimate=float(contact_ratio),
            effective_backlash_mm=float(effective_backlash),
            angular_lost_motion_deg=float(angular_lost_motion),
            minimum_clearance_mm=float(clearance),
            undercut_risk=bool(undercut),
            printable_04_nozzle=bool(printable),
            strength_index=float(strength),
            engagement_smoothness=float(smoothness),
            print_sensitivity=float(sensitivity),
            biomechanical_deviation_deg=biomechanical_deviation,
            no_interference=bool(no_interference),
            score=float(score),
        )

    def _parameter_grid(self) -> Iterable[GearMeshParameters]:
        for values in itertools.product(
            MODULES,
            PRESSURE_ANGLES,
            BACKLASHES,
            PROFILE_RELIEFS,
            FACE_WIDTHS,
            CENTER_OFFSETS,
            ROOT_FILLETS,
            ROOT_EMBEDS,
        ):
            yield GearMeshParameters(*values, tooth_style="Spur")

    def optimize(self, top_n: int = 25) -> GearOptimizationResult:
        candidates = [self.evaluate(item) for item in self._parameter_grid()]
        candidates.sort(key=lambda item: item.score)
        preferred = candidates[0]
        modules_with_clearance = {
            item.parameters.module_mm
            for item in candidates
            if item.no_interference
        }
        module_study = tuple(
            min(
                (
                    item
                    for item in candidates
                    if item.parameters.module_mm == module
                    and (
                        item.no_interference
                        or module not in modules_with_clearance
                    )
                ),
                key=lambda item: item.score,
            )
            for module in MODULES
        )
        style_study = tuple(
            self.evaluate(
                GearMeshParameters(
                    **{
                        **asdict(preferred.parameters),
                        "tooth_style": style,
                        "helix_angle_deg": (
                            angle if style != "Spur" else 0.0
                        ),
                    }
                )
            )
            for style in TOOTH_STYLES
            for angle in ((0.0,) if style == "Spur" else HELIX_ANGLES)
        )
        p = preferred.parameters
        exact = build_bench_prototype(
            self.transmission.model,
            BenchPrototypeConfig(
                center_distance_mm=self.pitch_data.center_distance,
                input_sector_angle_deg=np.degrees(
                    self.pitch_data.input_rad[-1]
                    - self.pitch_data.input_rad[0]
                ),
                module_mm=p.module_mm,
                pressure_angle_deg=p.pressure_angle_deg,
                backlash_mm=p.backlash_mm,
                minimum_ratio=self.transmission.config.minimum_ratio,
                smoothing_strength=self.transmission.config.smoothing_strength,
                profile_relief_mm=p.profile_relief_mm,
                gear_thickness_mm=p.face_width_mm,
                root_fillet_radius_mm=p.root_fillet_mm,
                tooth_root_embed_mm=p.tooth_root_embed_mm,
                minimum_clearance_mm=0.05,
                mesh_positions=len(self.pitch_data.elevation_deg),
            ),
        )
        locked_arrays = all(
            np.array_equal(first, second)
            for first, second in (
                (self.pitch_data.elevation_deg, exact.pitch_data.elevation_deg),
                (self.pitch_data.input_rad, exact.pitch_data.input_rad),
                (self.pitch_data.output_rad, exact.pitch_data.output_rad),
                (self.pitch_data.input_radii, exact.pitch_data.input_radii),
                (self.pitch_data.output_radii, exact.pitch_data.output_radii),
            )
        )
        physical_validation = PreferredPhysicalValidation(
            pitch_arrays_identical=locked_arrays,
            maximum_tooth_penetration_area_mm2=(
                exact.validation.maximum_tooth_penetration_area_mm2
            ),
            minimum_noncontact_clearance_mm=(
                exact.validation.minimum_noncontact_clearance_mm
            ),
            adjacent_teeth_overlap_free=(
                exact.validation.adjacent_teeth_overlap_free
            ),
            no_tooth_skipping=exact.validation.no_tooth_skipping,
            valid_closed_bodies=exact.validation.closed_valid_bodies,
            passed=exact.validation.all_practical_gates_pass and locked_arrays,
        )
        if not physical_validation.passed:
            raise RuntimeError(
                "Preferred analytic candidate failed exact physical validation."
            )
        return GearOptimizationResult(
            preferred=preferred,
            top_candidates=tuple(candidates[:top_n]),
            evaluated_candidates=len(candidates),
            locked_pitch_sha256=self.locked_pitch_sha256,
            module_study=module_study,
            style_study=style_study,
            preferred_validation=physical_validation,
        )


def calibration_recommendations(
    preferred: GearMeshCandidate,
) -> tuple[dict[str, float | str], ...]:
    base = preferred.parameters
    rows = []
    for name, extra in (("Tight", -0.05), ("Normal", 0.05), ("Loose", 0.15)):
        clearance = max(0.05, preferred.effective_backlash_mm + extra)
        rows.append(
            {
                "fit": name,
                "tooth_clearance_mm": round(clearance, 3),
                "angular_backlash_deg": round(
                    preferred.angular_lost_motion_deg
                    * clearance
                    / max(preferred.effective_backlash_mm, 0.05),
                    4,
                ),
                "center_distance_offset_mm": round(
                    base.center_distance_offset_mm + 0.5 * extra, 3
                ),
            }
        )
    return tuple(rows)


def export_optimization_report(
    result: GearOptimizationResult,
    directory: str | Path,
) -> tuple[Path, ...]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "GearMeshOptimization.json"
    csv_path = directory / "ModuleComparison.csv"
    md_path = directory / "GearMeshOptimizationReport.md"
    payload = {
        "locked_biomechanics": True,
        "locked_transmission": True,
        "locked_pitch_sha256": result.locked_pitch_sha256,
        "evaluated_candidates": result.evaluated_candidates,
        "preferred": asdict(result.preferred),
        "module_study": [asdict(item) for item in result.module_study],
        "style_study": [asdict(item) for item in result.style_study],
        "preferred_physical_validation": asdict(
            result.preferred_validation
        ),
        "calibration": calibration_recommendations(result.preferred),
        "safety": "RESEARCH-ONLY UNLOADED HAND-DRIVEN PROTOTYPE",
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=tuple(asdict(result.module_study[0]).keys()),
        )
        writer.writeheader()
        for item in result.module_study:
            row = asdict(item)
            row["parameters"] = json.dumps(row["parameters"], sort_keys=True)
            writer.writerow(row)
    preferred = result.preferred
    p = preferred.parameters
    lines = [
        "# Gear Mesh Optimization Report",
        "",
        "Biomechanics, transmission, pitch curves, and 11°–147° range were locked.",
        "",
        f"Candidates evaluated: {result.evaluated_candidates}",
        f"Locked pitch SHA-256: `{result.locked_pitch_sha256}`",
        "",
        "## Preferred unloaded prototype",
        "",
        f"- Module: {p.module_mm:.2f} mm",
        f"- Pressure angle: {p.pressure_angle_deg:.0f}°",
        f"- Backlash: {p.backlash_mm:.2f} mm",
        f"- Profile relief: {p.profile_relief_mm:.2f} mm",
        f"- Face width: {p.face_width_mm:.0f} mm",
        f"- Root fillet: {p.root_fillet_mm:.1f} mm",
        f"- Tooth-root embed: {p.tooth_root_embed_mm:.1f} mm",
        f"- Center-distance offset: {p.center_distance_offset_mm:+.2f} mm",
        f"- Tooth style: {p.tooth_style}",
        f"- Estimated angular lost motion: {preferred.angular_lost_motion_deg:.3f}°",
        f"- Estimated contact ratio: {preferred.contact_ratio_estimate:.3f}",
        f"- Root strength index: {preferred.strength_index:.2f}",
        f"- Undercut risk flag: {preferred.undercut_risk}",
        "",
        "The undercut flag is a conservative equivalent-circular estimate;",
        "the final rack polygons passed the sampled interference sweep.",
        f"- Exact maximum penetration: "
        f"{result.preferred_validation.maximum_tooth_penetration_area_mm2:.6f} mm²",
        f"- Exact minimum clearance: "
        f"{result.preferred_validation.minimum_noncontact_clearance_mm:.4f} mm",
        "",
        "## Module study",
        "",
        "| Module | Teeth (in/out) | Tip mm | Root mm | Contact ratio | "
        "Lost motion | Undercut | 0.4 mm printable |",
        "|---:|---:|---:|---:|---:|---:|:---:|:---:|",
        *[
            f"| {item.parameters.module_mm:.2f} | "
            f"{item.input_tooth_count}/{item.output_tooth_count} | "
            f"{item.minimum_tip_thickness_mm:.3f} | "
            f"{item.minimum_root_thickness_mm:.3f} | "
            f"{item.contact_ratio_estimate:.3f} | "
            f"{item.angular_lost_motion_deg:.3f}° | "
            f"{'yes' if item.undercut_risk else 'no'} | "
            f"{'yes' if item.printable_04_nozzle else 'no'} |"
            for item in result.module_study
        ],
        "",
        "Smallest recommended module for a 0.4 mm nozzle: **2.0 mm**.",
        "",
        "## Fit calibration",
        "",
        "| Fit | Tooth clearance | Angular backlash | Center offset |",
        "|---|---:|---:|---:|",
        *[
            f"| {row['fit']} | {row['tooth_clearance_mm']:.3f} mm | "
            f"{row['angular_backlash_deg']:.3f}° | "
            f"{row['center_distance_offset_mm']:+.3f} mm |"
            for row in calibration_recommendations(preferred)
        ],
        "",
        "The preferred style is spur. Helical and herringbone results are a",
        "comparative sweep study only and are not automatically selected.",
        "",
        "RESEARCH-ONLY UNLOADED HAND-DRIVEN PROTOTYPE — NOT FOR HUMAN USE.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, csv_path, md_path
