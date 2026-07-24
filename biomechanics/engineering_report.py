"""Generate literature-transmission comparison plots and GO/NO-GO report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from literature_transmission import compare_transmission_alternatives
from .literature_model import LiteratureShoulderModel


def generate_engineering_report(
    consensus_json: str | Path,
    output_directory: str | Path,
    acceptance_path: str | Path,
) -> tuple[Path, Path]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    acceptance = json.loads(Path(acceptance_path).read_text(encoding="utf-8"))
    model = LiteratureShoulderModel(consensus_json)
    approximate_model = LiteratureShoulderModel(
        consensus_json, allow_approximate_gh=True
    )
    comparison = compare_transmission_alternatives(model)
    elevation = comparison.elevation_deg
    gh_estimate = np.asarray(approximate_model.gh_angle_at(elevation))

    figure, axes = plt.subplots(3, 2, figsize=(13, 12), constrained_layout=True)
    axes[0, 0].fill_between(
        elevation,
        comparison.confidence_lower_deg,
        comparison.confidence_upper_deg,
        color="#7f83bd",
        alpha=0.2,
        label="Confidence envelope",
    )
    axes[0, 0].plot(elevation, comparison.target_st_deg, "k--", label="Target ST")
    axes[0, 0].plot(elevation, comparison.full_cycle_st_deg, label="Closed cycle")
    axes[0, 0].plot(elevation, comparison.sector_st_deg, label="Sector")
    axes[0, 0].set_title("ST angle")
    axes[0, 0].set_ylabel("deg")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(elevation, comparison.target_derivative, "k--", label="Target")
    axes[0, 1].plot(elevation, comparison.full_cycle_derivative, label="Closed cycle")
    axes[0, 1].plot(elevation, comparison.sector_derivative, label="Sector")
    axes[0, 1].axhline(0, color="red", lw=0.8)
    axes[0, 1].set_title("dST / dElevation")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(elevation, gh_estimate, color="#d37722")
    axes[1, 0].set_title("GH estimate (explicit HT−ST approximation)")
    axes[1, 0].set_ylabel("deg")

    axes[1, 1].plot(elevation, comparison.full_cycle_ratio, label="Closed cycle")
    axes[1, 1].plot(elevation, comparison.sector_ratio, label="Sector")
    axes[1, 1].axhline(0, color="red", lw=0.8)
    axes[1, 1].set_title("Mechanical transmission ratio")
    axes[1, 1].legend(fontsize=8)

    axes[2, 0].plot(elevation, comparison.full_cycle_input_radius, label="Closed r1")
    axes[2, 0].plot(elevation, comparison.full_cycle_output_radius, label="Closed r2")
    axes[2, 0].plot(elevation, comparison.sector_input_radius, "--", label="Sector r1")
    axes[2, 0].plot(elevation, comparison.sector_output_radius, "--", label="Sector r2")
    axes[2, 0].set_title("Pitch radii at 100-unit center distance")
    axes[2, 0].set_ylabel("model units")
    axes[2, 0].legend(fontsize=8)

    axes[2, 1].plot(
        elevation,
        comparison.full_cycle_st_deg - comparison.target_st_deg,
        label="Closed-cycle error",
    )
    axes[2, 1].plot(
        elevation,
        comparison.sector_st_deg - comparison.target_st_deg,
        label="Sector error",
    )
    axes[2, 1].set_title("Endpoint and trajectory distortion")
    axes[2, 1].set_ylabel("deg")
    axes[2, 1].legend(fontsize=8)
    for axis in axes.flat:
        axis.set_xlabel("HT elevation (deg)")
        axis.grid(True, alpha=0.3)
    plot_path = output / "LiteratureTransmissionComparison.png"
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)

    payload = json.loads(Path(consensus_json).read_text(encoding="utf-8"))
    loo = payload["sensitivity_analyses"]["leave_one_study_out"]
    loo_generated = any(item["status"] == "generated" for item in loo)
    closed_positive = bool(np.min(comparison.full_cycle_ratio) > acceptance["minimum_positive_derivative"])
    sector_positive = bool(np.min(comparison.sector_ratio) > acceptance["minimum_positive_derivative"])
    sector_error_ok = (
        comparison.sector_max_error_deg <= acceptance["maximum_st_error_deg"]
        and comparison.sector_rms_error_deg <= acceptance["rms_st_error_deg"]
    )
    minimum_sector_radius = float(
        np.nanmin(
            np.r_[
                comparison.sector_input_radius,
                comparison.sector_output_radius,
            ]
        )
    )
    no_go_reasons = []
    if not closed_positive:
        no_go_reasons.append(
            "Closed full-revolution mapping has a negative ratio near periodic closure."
        )
    if not loo_generated and acceptance["require_leave_one_study_out"]:
        no_go_reasons.append(
            "Selected condition has one contributing study, so leave-one-study-out "
            "validation is not estimable."
        )
    if not acceptance["allow_gh_decomposition_approximation"]:
        no_go_reasons.append(
            "GH=HT−ST is not verified as an exact 3-D rotational decomposition."
        )
    if minimum_sector_radius < acceptance["minimum_pitch_radius_units"]:
        no_go_reasons.append(
            f"Sector mapping reaches a {minimum_sector_radius:.6f}-unit pitch radius, "
            f"below the configured {acceptance['minimum_pitch_radius_units']:.3f}-unit "
            "minimum."
        )
    no_go_reasons.append(
        "No strength, backlash, fatigue, tolerance, or human-subject safety validation exists."
    )
    report_path = output / "LiteratureTransmissionComparison.md"
    lines = [
        "# Literature Transmission Comparison",
        "",
        "## Selected condition",
        "",
        "- Healthy, unloaded, raising",
        "- Dynamic scapular-plane abduction, 40° anterior to frontal",
        "- McClure2001 Figure 3B",
        f"- Supported HT range: {model.valid_range_deg[0]:.1f}°–{model.valid_range_deg[1]:.1f}°",
        "- Extrapolation: forbidden",
        "",
        "## Numerical comparison",
        "",
        "| Alternative | Max ST error | RMS ST error | Minimum ratio | Result |",
        "|---|---:|---:|---:|---|",
        f"| Closed full revolution | {comparison.full_cycle_max_error_deg:.6f}° | "
        f"{comparison.full_cycle_rms_error_deg:.6f}° | "
        f"{np.min(comparison.full_cycle_ratio):.6f} | "
        f"{'PASS' if closed_positive else 'FAIL'} |",
        f"| Partial sector | {comparison.sector_max_error_deg:.6e}° | "
        f"{comparison.sector_rms_error_deg:.6e}° | "
        f"{np.min(comparison.sector_ratio):.6e} | "
        f"{'PASS' if sector_positive and sector_error_ok else 'FAIL'} |",
        "",
        f"Minimum sector pitch radius at 100-unit center distance: "
        f"**{minimum_sector_radius:.6f} units**.",
        "",
        "The sector alternative requires independent lower/upper mechanical hard "
        "stops and a non-wrapping reset to the lower stop.",
        "",
        "## GO / NO-GO",
        "",
        "**NO-GO for manufacturing or human use.**",
        "",
        *[f"- {reason}" for reason in no_go_reasons],
        "",
        "The sector mapping is suitable only for further bench-top research. This "
        "report does not authorize final gear geometry or powered wearable testing.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path, plot_path
