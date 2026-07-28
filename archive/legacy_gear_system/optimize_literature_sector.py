"""Search and diagnose partial-sector mechanics without optimistic export."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from biomechanics.literature_model import LiteratureShoulderModel
from literature_sector import (
    RESEARCH_LABEL,
    LiteratureSectorTransmission,
    SectorDesignConfig,
    generate_sector_teeth,
    synthesize_sector_pitch_curves,
    validate_sector,
)
from optimized_sector import (
    adjacent_tooth_failures,
    build_closed_sector_blank,
    compute_tooth_metrics,
    curve_geometry_metrics,
    evaluate_hard_gates,
    simulate_complete_mesh,
)


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "validation_outputs"
MODEL = ROOT / "ConsensusShoulderModel.json"


def _candidate_configs():
    sector_angles = [120, 135, 150, 165, 180, 195, 210, 225, 240]
    center_distances = [100, 120, 140, 160, 180]
    modules = [1.5, 2.0, 2.5, 3.0, 4.0]
    pressure_angles = [20, 25, 30]
    backlashes = [0.10, 0.20, 0.30, 0.40]
    minimum_ratios = [0.05, 0.08, 0.12, 0.16, 0.20]
    smoothing = [0.10, 0.25, 0.50, 1.00]
    # Nine deterministic cross-sections at every sector angle cover the full
    # range of every parameter without pretending that 13,500 combinations
    # constitute manufacturing optimization.
    for angle in sector_angles:
        for index in range(9):
            yield SectorDesignConfig(
                input_sector_angle_deg=float(angle),
                center_distance=float(center_distances[index % len(center_distances)]),
                module=float(modules[(2 * index) % len(modules)]),
                pressure_angle_deg=float(
                    pressure_angles[index % len(pressure_angles)]
                ),
                backlash=float(backlashes[(3 * index) % len(backlashes)]),
                minimum_ratio=float(
                    minimum_ratios[(4 * index) % len(minimum_ratios)]
                ),
                smoothing_strength=float(
                    smoothing[(5 * index) % len(smoothing)]
                ),
                sample_count=2001,
            )


def _geometry_extrema(data):
    metrics = curve_geometry_metrics(data)
    curvature = np.r_[
        metrics["input"]["curvature"], metrics["output"]["curvature"]
    ]
    derivative = np.r_[
        metrics["input"]["curvature_derivative"],
        metrics["output"]["curvature_derivative"],
    ]
    tangent = np.r_[
        metrics["input"]["tangent_jump"],
        metrics["output"]["tangent_jump"],
    ]
    return (
        float(np.max(np.abs(curvature))),
        float(np.max(np.abs(derivative))),
        float(np.max(tangent)),
    )


def search(model):
    rows = []
    candidates = []
    for candidate_id, config in enumerate(_candidate_configs(), start=1):
        try:
            transmission = LiteratureSectorTransmission(
                model, config, regularized=True
            )
            data = synthesize_sector_pitch_curves(transmission)
            teeth = generate_sector_teeth(data, config)
            base = validate_sector(transmission, data, teeth)
            input_blank = build_closed_sector_blank(
                data.input_points, teeth.input_teeth, config
            )
            output_blank = build_closed_sector_blank(
                data.output_points, teeth.output_teeth, config
            )
            tooth_metrics = compute_tooth_metrics(teeth)
            curvature, curvature_derivative, tangent_jump = _geometry_extrema(
                data
            )
            adjacent = len(adjacent_tooth_failures(teeth)) == 0
            coarse_mesh = simulate_complete_mesh(
                data,
                input_blank,
                output_blank,
                teeth,
                position_count=41,
            )
            min_contact = min(
                (item.contact_ratio for item in tooth_metrics), default=0.0
            )
            min_root = min(
                (item.root_thickness for item in tooth_metrics), default=0.0
            )
            min_tip = min(
                (item.tip_thickness for item in tooth_metrics), default=0.0
            )
            undercut_margin = min(
                (item.undercut_margin for item in tooth_metrics), default=-np.inf
            )
            row = {
                "candidate_id": candidate_id,
                "sector_angle_deg": config.input_sector_angle_deg,
                "center_distance_mm": config.center_distance,
                "module_mm": config.module,
                "pressure_angle_deg": config.pressure_angle_deg,
                "backlash_mm": config.backlash,
                "minimum_ratio_constraint": config.minimum_ratio,
                "smoothing_strength": config.smoothing_strength,
                "minimum_ratio": float(np.min(data.ratio)),
                "maximum_ratio": float(np.max(data.ratio)),
                "minimum_pitch_radius_mm": float(
                    np.min(np.r_[data.input_radii, data.output_radii])
                ),
                "maximum_pitch_radius_mm": float(
                    np.max(np.r_[data.input_radii, data.output_radii])
                ),
                "maximum_curvature": curvature,
                "maximum_curvature_derivative": curvature_derivative,
                "maximum_tangent_jump": tangent_jump,
                "maximum_st_error_deg": base.maximum_st_error_deg,
                "rms_st_error_deg": base.rms_st_error_deg,
                "input_tooth_count": teeth.input_tooth_count,
                "output_tooth_count": teeth.output_tooth_count,
                "minimum_undercut_margin": undercut_margin,
                "minimum_contact_ratio": min_contact,
                "minimum_root_thickness_mm": min_root,
                "minimum_tip_thickness_mm": min_tip,
                "adjacent_tooth_overlap_free": adjacent,
                "coarse_mating_interference_free": coarse_mesh.zero_unintended_intersections,
                "maximum_coarse_penetration_area": coarse_mesh.maximum_penetration_area,
                "input_blank_valid": input_blank.valid,
                "output_blank_valid": output_blank.valid,
                "rack_envelope_verified": False,
                "hard_geometry_pass": False,
                "rejection_reason": "exact shared-rack envelope not verified",
            }
            rows.append(row)
            candidates.append(
                (
                    row,
                    transmission,
                    data,
                    teeth,
                    input_blank,
                    output_blank,
                    tooth_metrics,
                )
            )
        except Exception as error:
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "sector_angle_deg": config.input_sector_angle_deg,
                    "center_distance_mm": config.center_distance,
                    "module_mm": config.module,
                    "pressure_angle_deg": config.pressure_angle_deg,
                    "backlash_mm": config.backlash,
                    "minimum_ratio_constraint": config.minimum_ratio,
                    "smoothing_strength": config.smoothing_strength,
                    "hard_geometry_pass": False,
                    "rejection_reason": f"{type(error).__name__}: {error}",
                }
            )
    # Feasibility-first ranking: interference, tangent/curvature, contact
    # ratio, root thickness, biomechanical error, then compactness.
    candidates.sort(
        key=lambda item: (
            not item[0]["coarse_mating_interference_free"],
            item[0]["maximum_tangent_jump"],
            item[0]["maximum_curvature"],
            -item[0]["minimum_contact_ratio"],
            -item[0]["minimum_root_thickness_mm"],
            item[0]["rms_st_error_deg"],
            item[0]["center_distance_mm"],
        )
    )
    return rows, candidates


def write_search(rows):
    path = OUTPUT / "OptimizedSectorSearch.csv"
    keys = ["watermark", *sorted({key for row in rows for key in row})]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(
            {"watermark": RESEARCH_LABEL, **row} for row in rows
        )
    return path


def diagnostic_rows(transmission, data, teeth, mesh):
    geometry = curve_geometry_metrics(data)
    rows = []
    curvature_limit = 1.0 / max(0.35 * transmission.config.module, 1e-12)
    derivative_limit = curvature_limit / transmission.config.module
    for member in ("input", "output"):
        for failure, values, limit in (
            ("continuous_tangent", geometry[member]["tangent_jump"], 0.02),
            ("bounded_curvature", np.abs(geometry[member]["curvature"]), curvature_limit),
            (
                "bounded_curvature_derivative",
                np.abs(geometry[member]["curvature_derivative"]),
                derivative_limit,
            ),
        ):
            for index in np.flatnonzero(values > limit):
                rows.append(
                    {
                        "failure": failure,
                        "sample_index": int(index),
                        "member": member,
                        "ht_elevation_deg": float(data.elevation_deg[index]),
                        "input_angle_rad": float(data.input_rad[index]),
                        "output_angle_rad": float(data.output_rad[index]),
                        "local_ratio": float(data.ratio[index]),
                        "input_radius": float(data.input_radii[index]),
                        "output_radius": float(data.output_radii[index]),
                        "curvature": float(geometry[member]["curvature"][index]),
                        "curvature_derivative": float(
                            geometry[member]["curvature_derivative"][index]
                        ),
                        "tooth_number": "",
                        "geometric_reason": f"{failure} metric exceeds {limit:.8g}",
                    }
                )
    for failure in adjacent_tooth_failures(teeth):
        member = failure["member"]
        positions = (
            teeth.input_arc_positions
            if member == "input"
            else teeth.output_arc_positions
        )
        position = positions[failure["tooth_number"] - 1]
        points = data.input_points if member == "input" else data.output_points
        cumulative = np.r_[
            0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))
        ]
        index = int(np.argmin(np.abs(cumulative - position)))
        rows.append(
            {
                "failure": "adjacent_tooth_overlap_free",
                "sample_index": index,
                "member": member,
                "ht_elevation_deg": float(data.elevation_deg[index]),
                "input_angle_rad": float(data.input_rad[index]),
                "output_angle_rad": float(data.output_rad[index]),
                "local_ratio": float(data.ratio[index]),
                "input_radius": float(data.input_radii[index]),
                "output_radius": float(data.output_radii[index]),
                "curvature": float(geometry[member]["curvature"][index]),
                "curvature_derivative": float(
                    geometry[member]["curvature_derivative"][index]
                ),
                "tooth_number": failure["tooth_number"],
                "geometric_reason": (
                    f"tooth {failure['tooth_number']} overlaps tooth "
                    f"{failure['other_tooth_number']} by "
                    f"{failure['overlap_area']:.9g} mm²"
                ),
            }
        )
    for position in mesh.positions:
        if position.penetration_area > 1e-10:
            index = position.sample_index
            rows.append(
                {
                    "failure": "mating_interference_free",
                    "sample_index": index,
                    "member": "pair",
                    "ht_elevation_deg": position.elevation_deg,
                    "input_angle_rad": position.input_angle_rad,
                    "output_angle_rad": position.output_angle_rad,
                    "local_ratio": position.ratio,
                    "input_radius": position.input_radius,
                    "output_radius": position.output_radius,
                    "curvature": "",
                    "curvature_derivative": "",
                    "tooth_number": (
                        f"{position.intended_input_tooth}/"
                        f"{position.intended_output_tooth}"
                    ),
                    "geometric_reason": (
                        f"complete sector bodies penetrate by "
                        f"{position.penetration_area:.9g} mm²"
                    ),
                }
            )
    return rows


def write_diagnostics(rows):
    csv_path = OUTPUT / "SectorFailureDiagnostics.csv"
    keys = [
        "watermark",
        "failure",
        "sample_index",
        "member",
        "ht_elevation_deg",
        "input_angle_rad",
        "output_angle_rad",
        "local_ratio",
        "input_radius",
        "output_radius",
        "curvature",
        "curvature_derivative",
        "tooth_number",
        "geometric_reason",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(
            {"watermark": RESEARCH_LABEL, **row} for row in rows
        )
    markdown = OUTPUT / "SectorFailureDiagnostics.md"
    counts = {}
    for row in rows:
        counts[row["failure"]] = counts.get(row["failure"], 0) + 1
    lines = [
        f"# {RESEARCH_LABEL}",
        "",
        "Every row in `SectorFailureDiagnostics.csv` is sample-resolved. "
        "Mechanical lead-in/out regions are excluded.",
        "",
        "## Failure counts",
        "",
        "| Gate | Failing samples |",
        "|---|---:|",
        *[f"| {key} | {value} |" for key, value in sorted(counts.items())],
        "",
        "The exact shared-rack envelope is not yet verified. Consequently no "
        "candidate may be called printable even if a polygon collision screen clears.",
    ]
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, markdown


def write_mesh(mesh):
    path = OUTPUT / "OptimizedSectorMeshClearance.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        keys = ["watermark", *list(asdict(mesh.positions[0]))]
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for position in mesh.positions:
            writer.writerow(
                {"watermark": RESEARCH_LABEL, **asdict(position)}
            )
    return path


def write_tooth_metrics(metrics):
    path = OUTPUT / "OptimizedSectorToothMetrics.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        keys = ["watermark", *list(asdict(metrics[0]))] if metrics else []
        writer = csv.DictWriter(stream, fieldnames=keys)
        if keys:
            writer.writeheader()
            for metric in metrics:
                writer.writerow(
                    {"watermark": RESEARCH_LABEL, **asdict(metric)}
                )
    return path


def write_plots(rows, selected_data, mesh):
    path = OUTPUT / "OptimizedSectorPlots.png"
    valid_rows = [row for row in rows if "minimum_pitch_radius_mm" in row]
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    figure.suptitle(RESEARCH_LABEL, color="#b00020", fontweight="bold")
    axes[0, 0].scatter(
        [row["sector_angle_deg"] for row in valid_rows],
        [row["minimum_pitch_radius_mm"] for row in valid_rows],
        c=[row["minimum_ratio_constraint"] for row in valid_rows],
        s=20,
    )
    axes[0, 0].set_title("Search: minimum pitch radius")
    axes[0, 1].scatter(
        [row["rms_st_error_deg"] for row in valid_rows],
        [row["maximum_curvature"] for row in valid_rows],
        c=[row["center_distance_mm"] for row in valid_rows],
        s=20,
    )
    axes[0, 1].set_title("Biomechanical error vs curvature")
    axes[1, 0].plot(
        selected_data.elevation_deg, selected_data.ratio, label="Selected dψ/dφ"
    )
    axes[1, 0].set_title("Selected screened candidate ratio")
    axes[1, 1].plot(
        [item.elevation_deg for item in mesh.positions],
        [item.penetration_area for item in mesh.positions],
    )
    axes[1, 1].set_title("2001-position body penetration")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    figure.savefig(path, dpi=170)
    plt.close(figure)
    return path


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model = LiteratureShoulderModel(MODEL)
    rows, candidates = search(model)
    write_search(rows)
    if not candidates:
        raise RuntimeError("No candidate could be synthesized for diagnostics.")
    (
        selected_row,
        transmission,
        data,
        teeth,
        input_blank,
        output_blank,
        tooth_metrics,
    ) = candidates[0]
    mesh = simulate_complete_mesh(
        data,
        input_blank,
        output_blank,
        teeth,
        position_count=2001,
    )
    validation = evaluate_hard_gates(
        transmission,
        data,
        teeth,
        input_blank,
        output_blank,
        mesh,
        tooth_metrics,
        rack_envelope_verified=False,
    )
    diagnostics = diagnostic_rows(transmission, data, teeth, mesh)
    write_diagnostics(diagnostics)
    write_mesh(mesh)
    write_tooth_metrics(tooth_metrics)
    write_plots(rows, data, mesh)

    validation_path = OUTPUT / "OptimizedSectorValidation.json"
    payload = {
        "watermark": RESEARCH_LABEL,
        "source": "McClure2001",
        "condition": model.selected["condition"],
        "valid_range_deg": list(model.valid_range_deg),
        "selected_screened_candidate": selected_row,
        "full_mesh_position_count": len(mesh.positions),
        "mesh_summary": {
            key: value
            for key, value in asdict(mesh).items()
            if key != "positions"
        },
        "validation": asdict(validation),
        "prototype_geometry_exported": validation.hard_pass,
        "decision": validation.decision,
    }
    validation_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    report_path = OUTPUT / "OptimizedSectorEngineeringReport.md"
    report_path.write_text(
        "\n".join(
            [
                f"# {RESEARCH_LABEL}",
                "",
                "## Search",
                "",
                f"- Candidates evaluated: {len(rows)}",
                "- Sector angles: 120°, 135°, 150°, 165°, 180°, 195°, "
                "210°, 225°, 240°",
                "- Center distance, module, pressure angle, backlash, minimum "
                "ratio, and smoothing ranges were all sampled.",
                "- Ranking order: interference, tangent/curvature, contact "
                "ratio, root thickness, biomechanical RMS error, compactness.",
                "",
                "## Selected screened candidate",
                "",
                *[
                    f"- {key}: {value}"
                    for key, value in selected_row.items()
                    if key
                    in (
                        "sector_angle_deg",
                        "center_distance_mm",
                        "module_mm",
                        "pressure_angle_deg",
                        "backlash_mm",
                        "minimum_ratio_constraint",
                        "smoothing_strength",
                        "maximum_st_error_deg",
                        "rms_st_error_deg",
                        "minimum_pitch_radius_mm",
                        "minimum_contact_ratio",
                    )
                ],
                "",
                "## Full 2001-position mesh",
                "",
                f"- Maximum penetration area: {mesh.maximum_penetration_area}",
                f"- Minimum clearance: {mesh.minimum_clearance}",
                f"- Maximum contact mismatch: {mesh.maximum_contact_mismatch}",
                f"- No tooth skipping: {mesh.no_tooth_skipping}",
                f"- No contact discontinuity: {mesh.no_contact_discontinuity}",
                "",
                "## Hard gates",
                "",
                *[
                    f"- {'PASS' if value else 'FAIL'} — {key}"
                    for key, value in asdict(validation).items()
                    if isinstance(value, bool)
                ],
                "",
                f"## Decision: {validation.decision}",
                "",
                "Prototype DXF/SVG was not exported because every hard gate "
                "did not pass." if not validation.hard_pass else
                "All prototype export gates passed.",
                "",
                "Remaining blockers:",
                *[f"- {item}" for item in validation.blockers],
                "",
                "**Never GO FOR HUMAN USE.**",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(RESEARCH_LABEL)
    print(f"Candidates evaluated: {len(rows)}")
    print(f"Full mesh positions: {len(mesh.positions)}")
    print(f"Passing hard candidates: {sum(bool(row.get('hard_geometry_pass')) for row in rows)}")
    print(f"Decision: {validation.decision}")
    print("Blockers:")
    for blocker in validation.blockers:
        print(f"- {blocker}")


if __name__ == "__main__":
    main()
