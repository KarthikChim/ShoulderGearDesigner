"""Build, validate, compare, and export the research-only literature sector."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import ezdxf
import matplotlib.pyplot as plt
import numpy as np
import svgwrite
from shapely.geometry import LineString

from biomechanics.literature_model import LiteratureShoulderModel
from kinematics import ShoulderModel
from literature_sector import (
    RESEARCH_LABEL,
    LiteratureSectorTransmission,
    SectorDesignConfig,
    generate_sector_teeth,
    synthesize_sector_pitch_curves,
    validate_sector,
)
from literature_transmission import compare_transmission_alternatives
from noncircular import SmoothTransmission
from settings import default_ratio_regions


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "ConsensusShoulderModel.json"
OUTPUT = ROOT / "validation_outputs"


def _write_transmission_csv(raw, regularized) -> Path:
    path = OUTPUT / "LiteratureSectorTransmission.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([RESEARCH_LABEL])
        writer.writerow(
            [
                "candidate",
                "ht_elevation_deg",
                "input_angle_rad",
                "output_angle_rad",
                "absolute_st_deg",
                "st_excursion_deg",
                "dst_de",
                "dpsi_dphi",
                "d2psi_dphi2",
                "confidence_lower_deg",
                "confidence_upper_deg",
                "extrapolated",
            ]
        )
        for data in (raw, regularized):
            for index in range(len(data.elevation_deg)):
                writer.writerow(
                    [
                        data.candidate,
                        *[
                            f"{array[index]:.17g}"
                            for array in (
                                data.elevation_deg,
                                data.input_rad,
                                data.output_rad,
                                data.absolute_st_deg,
                                data.st_excursion_deg,
                                data.dst_de,
                                data.ratio,
                                data.ratio_derivative,
                                data.confidence_lower_deg,
                                data.confidence_upper_deg,
                            )
                        ],
                        "false",
                    ]
                )
    return path


def _write_pitch_csv(raw, regularized) -> Path:
    path = OUTPUT / "LiteratureSectorPitchCurves.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([RESEARCH_LABEL])
        writer.writerow(
            [
                "candidate",
                "ht_elevation_deg",
                "input_x",
                "input_y",
                "output_local_x",
                "output_local_y",
                "output_assembled_x",
                "output_assembled_y",
                "input_radius",
                "output_radius",
                "center_distance",
            ]
        )
        for data in (raw, regularized):
            assembled = data.output_points + np.array(
                [data.center_distance, 0.0]
            )
            for index in range(len(data.elevation_deg)):
                writer.writerow(
                    [
                        data.candidate,
                        f"{data.elevation_deg[index]:.17g}",
                        f"{data.input_points[index, 0]:.17g}",
                        f"{data.input_points[index, 1]:.17g}",
                        f"{data.output_points[index, 0]:.17g}",
                        f"{data.output_points[index, 1]:.17g}",
                        f"{assembled[index, 0]:.17g}",
                        f"{assembled[index, 1]:.17g}",
                        f"{data.input_radii[index]:.17g}",
                        f"{data.output_radii[index]:.17g}",
                        f"{data.center_distance:.17g}",
                    ]
                )
    return path


def _legacy_metrics(model, config):
    elevation = np.linspace(*model.valid_range_deg, config.sample_count)
    target = np.asarray(model.st_angle_at(elevation))
    legacy_model = ShoulderModel(default_ratio_regions())
    legacy_st = np.array(
        [legacy_model.contributions_at(float(value))[1] for value in elevation]
    )
    error = legacy_st - target
    transmission = SmoothTransmission(legacy_model)
    phase = elevation / 180.0 * 2.0 * np.pi
    ratio = np.asarray(transmission.ratio(phase))
    return {
        "name": "legacy_periodic_full_revolution",
        "maximum_st_error_deg": float(np.max(np.abs(error))),
        "rms_st_error_deg": float(np.sqrt(np.mean(error**2))),
        "maximum_derivative_error": float(
            np.max(
                np.abs(
                    np.gradient(legacy_st, elevation)
                    - model.dst_delevation_at(elevation)
                )
            )
        ),
        "endpoint_distortion_deg": float(error[-1]),
        "minimum_ratio": float(np.min(ratio)),
        "maximum_ratio": float(np.max(ratio)),
        "minimum_pitch_radius": float(
            np.min(config.center_distance * ratio / (1.0 + ratio))
        ),
        "maximum_pitch_radius": float(
            np.max(config.center_distance / (1.0 + ratio))
        ),
        "required_center_distance": config.center_distance,
        "estimated_undercut_risk": False,
        "estimated_contact_ratio": None,
        "warnings": ["Uses the prohibited legacy ratio schedule; comparison only."],
    }


def _comparison_rows(model, raw, regularized, raw_validation, reg_validation):
    old = compare_transmission_alternatives(
        model, center_distance=raw.center_distance, sample_count=len(raw.elevation_deg)
    )
    config = SectorDesignConfig(center_distance=raw.center_distance)
    legacy = _legacy_metrics(model, config)

    def sector_row(data, validation):
        curvature = np.gradient(
            np.unwrap(np.arctan2(data.input_points[:, 1], data.input_points[:, 0])),
            data.input_rad,
        )
        return {
            "name": f"literature_partial_sector_{data.candidate}",
            "maximum_st_error_deg": validation.maximum_st_error_deg,
            "rms_st_error_deg": validation.rms_st_error_deg,
            "maximum_derivative_error": validation.maximum_derivative_error,
            "endpoint_distortion_deg": validation.endpoint_error_deg,
            "minimum_ratio": float(np.min(data.ratio)),
            "maximum_ratio": float(np.max(data.ratio)),
            "minimum_pitch_radius": validation.minimum_pitch_radius,
            "maximum_pitch_radius": float(
                np.max(np.r_[data.input_radii, data.output_radii])
            ),
            "maximum_curvature_proxy": float(np.max(np.abs(curvature))),
            "required_center_distance": data.center_distance,
            "estimated_undercut_risk": validation.undercut_risk,
            "estimated_contact_ratio": validation.contact_ratio_estimate,
            "warnings": list(validation.warnings),
        }

    full_radius = np.r_[
        old.full_cycle_input_radius, old.full_cycle_output_radius
    ]
    full = {
        "name": "literature_forced_full_revolution",
        "maximum_st_error_deg": old.full_cycle_max_error_deg,
        "rms_st_error_deg": old.full_cycle_rms_error_deg,
        "maximum_derivative_error": float(
            np.max(np.abs(old.full_cycle_derivative - old.target_derivative))
        ),
        "endpoint_distortion_deg": float(
            old.full_cycle_st_deg[-1] - old.target_st_deg[-1]
        ),
        "minimum_ratio": float(np.min(old.full_cycle_ratio)),
        "maximum_ratio": float(np.max(old.full_cycle_ratio)),
        "minimum_pitch_radius": (
            float(np.nanmin(full_radius))
            if np.any(np.isfinite(full_radius))
            else None
        ),
        "maximum_pitch_radius": (
            float(np.nanmax(full_radius))
            if np.any(np.isfinite(full_radius))
            else None
        ),
        "required_center_distance": raw.center_distance,
        "estimated_undercut_risk": True,
        "estimated_contact_ratio": None,
        "warnings": ["Negative ratio at periodic closure; mechanically invalid."],
    }
    return [
        legacy,
        full,
        sector_row(raw, raw_validation),
        sector_row(regularized, reg_validation),
    ]


def _write_comparison_csv(rows) -> Path:
    path = OUTPUT / "LiteratureSectorComparison.csv"
    scalar_keys = [
        "name",
        "maximum_st_error_deg",
        "rms_st_error_deg",
        "maximum_derivative_error",
        "endpoint_distortion_deg",
        "minimum_ratio",
        "maximum_ratio",
        "minimum_pitch_radius",
        "maximum_pitch_radius",
        "required_center_distance",
        "estimated_undercut_risk",
        "estimated_contact_ratio",
        "warnings",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=scalar_keys)
        writer.writerow({key: RESEARCH_LABEL if key == "name" else "" for key in scalar_keys})
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        " | ".join(row.get(key, []))
                        if key == "warnings"
                        else row.get(key)
                    )
                    for key in scalar_keys
                }
            )
    return path


def _dxf_polyline(path: Path, points: np.ndarray, layer: str) -> None:
    document = ezdxf.new("R2010")
    document.header["$INSUNITS"] = 4
    model = document.modelspace()
    model.add_text(
        RESEARCH_LABEL,
        height=3.0,
        dxfattribs={"layer": "RESEARCH_ONLY"},
    )
    model.add_lwpolyline(points.tolist(), close=False, dxfattribs={"layer": layer})
    document.saveas(path)


def _dxf_teeth(path: Path, teeth, center_offset: float = 0.0) -> None:
    document = ezdxf.new("R2010")
    document.header["$INSUNITS"] = 4
    model = document.modelspace()
    model.add_text(
        RESEARCH_LABEL,
        height=3.0,
        dxfattribs={"layer": "RESEARCH_ONLY"},
    )
    for index, polygon in enumerate(teeth):
        shifted = polygon + np.array([center_offset, 0.0])
        model.add_lwpolyline(
            shifted.tolist(),
            close=True,
            dxfattribs={"layer": f"TOOTH_{index + 1:03d}"},
        )
    document.saveas(path)


def _write_svg(data, teeth) -> Path:
    path = OUTPUT / "literature_sector_pair.svg"
    output_pitch = data.output_points + np.array([data.center_distance, 0.0])
    output_teeth = tuple(
        item + np.array([data.center_distance, 0.0]) for item in teeth.output_teeth
    )
    all_points = np.vstack(
        (data.input_points, output_pitch, *teeth.input_teeth, *output_teeth)
    )
    low = np.min(all_points, axis=0)
    high = np.max(all_points, axis=0)
    margin = 15.0
    size = high - low + 2 * margin
    drawing = svgwrite.Drawing(
        path,
        size=(f"{size[0]}mm", f"{size[1]}mm"),
        viewBox=f"0 0 {size[0]} {size[1]}",
    )

    def convert(points):
        return [
            (
                float(point[0] - low[0] + margin),
                float(high[1] - point[1] + margin),
            )
            for point in points
        ]

    drawing.add(
        drawing.text(
            RESEARCH_LABEL,
            insert=(5, 8),
            fill="#b00020",
            font_size="4px",
            id="research-only-watermark",
        )
    )
    drawing.add(
        drawing.polyline(
            convert(data.input_points),
            fill="none",
            stroke="#d97706",
            stroke_width=0.3,
            id="input-open-pitch-sector",
        )
    )
    drawing.add(
        drawing.polyline(
            convert(output_pitch),
            fill="none",
            stroke="#2563eb",
            stroke_width=0.3,
            id="output-open-pitch-sector",
        )
    )
    for index, polygon in enumerate(teeth.input_teeth):
        drawing.add(
            drawing.polygon(
                convert(polygon),
                fill="none",
                stroke="#9a3412",
                stroke_width=0.18,
                id=f"input-tooth-{index + 1}",
            )
        )
    for index, polygon in enumerate(output_teeth):
        drawing.add(
            drawing.polygon(
                convert(polygon),
                fill="none",
                stroke="#1d4ed8",
                stroke_width=0.18,
                id=f"output-tooth-{index + 1}",
            )
        )
    drawing.save()
    return path


def _write_plots(model, raw, regularized) -> Path:
    path = OUTPUT / "LiteratureSectorPlots.png"
    target = np.asarray(model.st_angle_at(raw.elevation_deg))
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    figure.suptitle(RESEARCH_LABEL, color="#b00020", fontweight="bold")
    axes[0, 0].fill_between(
        raw.elevation_deg,
        raw.confidence_lower_deg,
        raw.confidence_upper_deg,
        alpha=0.18,
        color="#6366f1",
        label="McClure confidence band",
    )
    axes[0, 0].plot(raw.elevation_deg, target, "k--", label="Raw target")
    axes[0, 0].plot(
        regularized.elevation_deg,
        regularized.absolute_st_deg,
        label="Regularized",
    )
    axes[0, 0].set_title("Absolute ST upward rotation")
    axes[0, 0].legend()
    axes[0, 1].plot(raw.elevation_deg, raw.dst_de, label="Raw dST/dE")
    axes[0, 1].plot(
        regularized.elevation_deg,
        regularized.dst_de,
        label="Regularized dST/dE",
    )
    axes[0, 1].axhline(0, color="red", lw=0.8)
    axes[0, 1].set_title("Literature derivative")
    axes[0, 1].legend()
    axes[1, 0].plot(raw.elevation_deg, raw.ratio, label="Raw ratio")
    axes[1, 0].plot(
        regularized.elevation_deg,
        regularized.ratio,
        label="Regularized ratio",
    )
    axes[1, 0].set_title("dψ/dφ")
    axes[1, 0].legend()
    axes[1, 1].plot(
        raw.input_points[:, 0], raw.input_points[:, 1], label="Raw input"
    )
    axes[1, 1].plot(
        raw.output_points[:, 0] + raw.center_distance,
        raw.output_points[:, 1],
        label="Raw output assembled",
    )
    axes[1, 1].set_aspect("equal")
    axes[1, 1].set_title("Open partial pitch sectors")
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.set_xlabel("HT elevation (deg)" if axis is not axes[1, 1] else "x")
    figure.savefig(path, dpi=170)
    plt.close(figure)
    return path


def _write_report(
    config,
    raw_transmission,
    reg_transmission,
    raw_data,
    reg_data,
    raw_teeth,
    reg_teeth,
    raw_validation,
    reg_validation,
    comparisons,
) -> Path:
    path = OUTPUT / "LiteratureSectorEngineeringReport.md"
    raw_audit = raw_transmission.slope_audit()
    reg_audit = reg_transmission.slope_audit()
    preferred = min(
        (raw_validation, reg_validation),
        key=lambda item: (
            item.decision == "NO-GO",
            item.maximum_st_error_deg,
            -item.minimum_pitch_radius,
        ),
    )
    lines = [
        f"# {RESEARCH_LABEL}",
        "",
        "## Scope and provenance",
        "",
        "- Source: McClure2001 only",
        "- Condition: healthy, unloaded, raising, scapular-plane elevation",
        "- Verified HT range: 11°–147°",
        "- No extrapolation, wrapping, or periodic continuation",
        "- Transitions are mechanical placeholders and excluded from biomechanical error",
        "",
        "## Slope audit",
        "",
        "| Candidate | min dST/dE | max dST/dE | min dψ/dφ | max dψ/dφ | near-zero regions |",
        "|---|---:|---:|---:|---:|---|",
        f"| Raw | {raw_audit.minimum_dst_de:.8g} | {raw_audit.maximum_dst_de:.8g} | "
        f"{raw_audit.minimum_ratio:.8g} | {raw_audit.maximum_ratio:.8g} | "
        f"{raw_audit.near_zero_regions_deg or 'none'} |",
        f"| Regularized | {reg_audit.minimum_dst_de:.8g} | {reg_audit.maximum_dst_de:.8g} | "
        f"{reg_audit.minimum_ratio:.8g} | {reg_audit.maximum_ratio:.8g} | "
        f"{reg_audit.near_zero_regions_deg or 'none'} |",
        "",
        "The raw literature curve is preserved unchanged. The regularized candidate "
        "enforces a positive mechanical ratio while preserving endpoint excursion.",
        "",
        "## Sector validation",
        "",
        "| Candidate | Decision | Max ST error | RMS ST error | Min radius | Contact ratio estimate | Teeth input/output |",
        "|---|---|---:|---:|---:|---:|---:|",
        f"| Raw | {raw_validation.decision} | {raw_validation.maximum_st_error_deg:.8g}° | "
        f"{raw_validation.rms_st_error_deg:.8g}° | {raw_validation.minimum_pitch_radius:.5g} | "
        f"{raw_validation.contact_ratio_estimate:.3f} | "
        f"{raw_teeth.input_tooth_count}/{raw_teeth.output_tooth_count} |",
        f"| Regularized | {reg_validation.decision} | {reg_validation.maximum_st_error_deg:.8g}° | "
        f"{reg_validation.rms_st_error_deg:.8g}° | {reg_validation.minimum_pitch_radius:.5g} | "
        f"{reg_validation.contact_ratio_estimate:.3f} | "
        f"{reg_teeth.input_tooth_count}/{reg_teeth.output_tooth_count} |",
        "",
        "## Four-way comparison",
        "",
        "| Alternative | Max ST error | RMS ST error | Ratio range | Radius range |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in comparisons:
        lines.append(
            f"| {item['name']} | {item['maximum_st_error_deg']:.6g}° | "
            f"{item['rms_st_error_deg']:.6g}° | "
            f"{item['minimum_ratio']:.6g}–{item['maximum_ratio']:.6g} | "
            f"{item.get('minimum_pitch_radius')}–{item.get('maximum_pitch_radius')} |"
        )
    lines.extend(
        [
            "",
            "## Final decision",
            "",
            f"**{preferred.decision}**",
            "",
            f"- Preferred candidate from numerical gates: {preferred.candidate}",
            f"- Preferred input sector angle: {config.input_sector_angle_deg:.1f}°",
            f"- Preferred center distance: {config.center_distance:.1f} model units",
            f"- Preferred module: {config.module:.3f}",
            f"- Hard stops: 11° and 147° HT equivalents",
            "",
            "## Remaining blockers",
            "",
            "- Complete sector blanks and hard-stop solid geometry are placeholders.",
            "- Mating tooth interference is not cleared without complete blanks.",
            "- Contact ratio is an unloaded estimate, not loaded contact analysis.",
            "- Strength, fatigue, tolerances, bearings, backlash under load, and actuator "
            "failure behavior are not validated.",
            "- Single-study target prevents leave-one-study-out validation.",
            "- No human-subject or wearable safety validation exists.",
            "",
            "**Never interpret this report as GO FOR HUMAN USE.**",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model = LiteratureShoulderModel(MODEL_PATH)
    config = SectorDesignConfig()
    raw_transmission = LiteratureSectorTransmission(
        model, config, regularized=False
    )
    regularized_transmission = LiteratureSectorTransmission(
        model, config, regularized=True
    )
    raw_data = synthesize_sector_pitch_curves(raw_transmission)
    reg_data = synthesize_sector_pitch_curves(regularized_transmission)
    raw_teeth = generate_sector_teeth(raw_data, config)
    reg_teeth = generate_sector_teeth(reg_data, config)
    raw_validation = validate_sector(
        raw_transmission, raw_data, raw_teeth
    )
    reg_validation = validate_sector(
        regularized_transmission, reg_data, reg_teeth
    )
    comparisons = _comparison_rows(
        model, raw_data, reg_data, raw_validation, reg_validation
    )

    _write_transmission_csv(raw_data, reg_data)
    _write_pitch_csv(raw_data, reg_data)
    _write_comparison_csv(comparisons)
    _write_plots(model, raw_data, reg_data)
    _dxf_polyline(
        OUTPUT / "literature_sector_input_pitch_curve.dxf",
        reg_data.input_points,
        "INPUT_OPEN_PITCH_SECTOR",
    )
    _dxf_polyline(
        OUTPUT / "literature_sector_output_pitch_curve.dxf",
        reg_data.output_points,
        "OUTPUT_OPEN_PITCH_SECTOR",
    )
    _dxf_teeth(
        OUTPUT / "literature_sector_input_teeth.dxf",
        reg_teeth.input_teeth,
    )
    _dxf_teeth(
        OUTPUT / "literature_sector_output_teeth.dxf",
        reg_teeth.output_teeth,
    )
    _write_svg(reg_data, reg_teeth)
    report = _write_report(
        config,
        raw_transmission,
        regularized_transmission,
        raw_data,
        reg_data,
        raw_teeth,
        reg_teeth,
        raw_validation,
        reg_validation,
        comparisons,
    )
    validation_path = OUTPUT / "LiteratureSectorValidation.json"
    payload = {
        "watermark": RESEARCH_LABEL,
        "source": "McClure2001",
        "condition": model.selected["condition"],
        "valid_range_deg": list(model.valid_range_deg),
        "config": asdict(config),
        "boundaries": {
            "active_input_bounds_rad": list(raw_data.active_input_bounds_rad),
            "transition_input_bounds_rad": list(
                raw_data.transition_input_bounds_rad
            ),
            "mounting_input_bounds_rad": list(
                raw_data.mounting_input_bounds_rad
            ),
            "hard_stop_elevation_deg": list(
                raw_data.hard_stop_elevation_deg
            ),
            "transition_biomechanics_included": False,
            "wraps_or_closes": False,
        },
        "raw_slope_audit": asdict(raw_transmission.slope_audit()),
        "regularized_slope_audit": asdict(
            regularized_transmission.slope_audit()
        ),
        "regularization": {
            "separately_identified": True,
            "endpoint_excursion_preserved": bool(
                abs(
                    reg_data.st_excursion_deg[-1]
                    - raw_data.st_excursion_deg[-1]
                )
                < 1e-9
            ),
            "maximum_difference_deg": float(
                np.max(
                    np.abs(
                        reg_data.absolute_st_deg - raw_data.absolute_st_deg
                    )
                )
            ),
            "rms_difference_deg": float(
                np.sqrt(
                    np.mean(
                        (
                            reg_data.absolute_st_deg
                            - raw_data.absolute_st_deg
                        )
                        ** 2
                    )
                )
            ),
        },
        "raw_validation": asdict(raw_validation),
        "regularized_validation": asdict(reg_validation),
        "tooth_counts": {
            "raw_input": raw_teeth.input_tooth_count,
            "raw_output": raw_teeth.output_tooth_count,
            "regularized_input": reg_teeth.input_tooth_count,
            "regularized_output": reg_teeth.output_tooth_count,
        },
        "comparison": comparisons,
        "final_decision": reg_validation.decision,
        "remaining_blockers": list(reg_validation.warnings),
    }
    validation_path.write_text(
        json.dumps(
            payload,
            indent=2,
            allow_nan=False,
            default=lambda value: (
                value.item() if isinstance(value, np.generic) else str(value)
            ),
        ),
        encoding="utf-8",
    )
    print(RESEARCH_LABEL)
    print(f"Raw target feasibility: {raw_validation.decision}")
    print(f"Regularized candidate feasibility: {reg_validation.decision}")
    print(f"Preferred sector angle: {config.input_sector_angle_deg:.1f} deg")
    print(f"Preferred center distance: {config.center_distance:.1f}")
    print(f"Preferred module: {config.module:.3f}")
    print(
        "Tooth counts (input/output): "
        f"{reg_teeth.input_tooth_count}/{reg_teeth.output_tooth_count}"
    )
    print(
        f"Maximum/RMS biomechanical error: "
        f"{reg_validation.maximum_st_error_deg:.6g}/"
        f"{reg_validation.rms_st_error_deg:.6g} deg"
    )
    print(
        "Pitch-radius range: "
        f"{reg_validation.minimum_pitch_radius:.6g}/"
        f"{np.max(np.r_[reg_data.input_radii, reg_data.output_radii]):.6g}"
    )
    print(
        f"Contact-ratio estimate: {reg_validation.contact_ratio_estimate:.4f}"
    )
    print("Remaining blockers:")
    for warning in reg_validation.warnings:
        print(f"- {warning}")
    print(f"Report: {report}")
    print(f"Final decision: {reg_validation.decision}")


if __name__ == "__main__":
    main()
