"""Tests for exact pitch-curve data exports."""

from __future__ import annotations

import csv

import ezdxf

from export_csv import CsvExporter
from export_dxf import DxfExporter
from export_svg import SvgExporter
from settings import Settings
from simulation import Simulation


def test_csv_svg_and_dxf_exports(tmp_path) -> None:
    simulation = Simulation(Settings())
    csv_path = CsvExporter().export(simulation, tmp_path / "curves.csv")
    svg_path = SvgExporter().export(simulation, tmp_path / "curves.svg")
    dxf_path = DxfExporter().export(simulation, tmp_path / "curves.dxf")

    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    assert len(rows) == simulation.settings.pitch_curve_samples + 1
    assert "instantaneous_gear_ratio" in rows[0]
    assert svg_path.stat().st_size > 1000

    document = ezdxf.readfile(dxf_path)
    layers = {entity.dxf.layer for entity in document.modelspace()}
    assert "INPUT_PITCH_CURVE" in layers
    assert "OUTPUT_PITCH_CURVE" in layers
    assert "INPUT_COMPLETE_TEETH" in layers
    assert "OUTPUT_COMPLETE_TEETH" in layers
    assert "input-complete-tooth-boundary" in svg_path.read_text(encoding="utf-8")
    assert "output-complete-tooth-boundary" in svg_path.read_text(encoding="utf-8")
