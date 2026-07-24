"""Command-line entry point for model generation and inspection."""

from __future__ import annotations

import argparse
from pathlib import Path

from .engine import BiomechanicsEngine
from .visualization import plot_consensus


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ConsensusShoulderModel.json")
    parser.add_argument("csv", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("ConsensusShoulderModel.json")
    )
    parser.add_argument("--plot-index", type=int)
    args = parser.parse_args()
    engine = BiomechanicsEngine(args.csv)
    model = engine.build()
    destination = engine.export(args.output)
    report = engine.validation_report
    print(f"Rows: {report.row_count}; papers: {report.paper_count}")
    print(f"Motion groups: {len(engine.motion_datasets)}")
    print(f"Consensus datasets: {len(model.datasets)}; splines: {len(model.splines)}")
    print(f"Coordinate observations awaiting verification: "
          f"{model.metadata['unverified_coordinate_observation_count']}")
    print(f"Validation: {'PASS' if report.valid else 'FAIL'}")
    for issue in report.issues:
        print(f"{issue.severity.upper()}: {issue.message}")
    print(f"Written: {destination.resolve()}")
    if args.plot_index is not None:
        plot_consensus(model.datasets[args.plot_index])


if __name__ == "__main__":
    main()
