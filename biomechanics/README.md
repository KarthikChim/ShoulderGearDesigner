# Shoulder Biomechanics Engine

This package converts the literature extraction database into a reusable,
motion-specific `ConsensusShoulderModel.json`. It does not import or modify
gear-generation code.

## Build the model

From the project directory:

```bash
./.venv/bin/python -m biomechanics.cli \
  biomechanics/data/HumanShoulderGroundTruth_v2.csv \
  --output ConsensusShoulderModel.json
```

Add `--plot-index 0` to open an interactive consensus plot. Each paper can be
toggled independently; labels display the applied study weight.

To rebuild the selected pathway, coordinate audit, comparison plot, and
GO/NO-GO report without generating gear geometry:

```bash
./.venv/bin/python validate_literature.py
```

## Scientific safeguards

- Every CSV row is retained exactly as text in `RawLiteratureRow.fields`.
- Interval ratios and excursion-only rows with blank angle fields are not
  treated as absolute kinematic trajectories.
- Motion type, plane, direction, loading, and population status are all part
  of the consensus key. Raising/lowering and loaded/unloaded data never merge.
- Repeated elevations are preserved in raw data and averaged only in derived
  within-study curves.
- Study quality is configured in `config/default_weighting.json`; no quality
  score is buried in Python code.
- Transformations are configured in `config/coordinate_conventions.json`.
  Every normalized observation retains original value, convention,
  transformation, and normalized value.
- The supplied CSV does not fully specify mappings from every published
  coordinate system into one verified standard. Defaults are therefore
  identity transformations marked `verified: false`. These conventions must
  be reviewed before cross-paper values are treated as anatomically identical.
  Cross-paper consensus is blocked unless every contributing mapping is
  verified.
- McClure2001 Figure 3 data used by the selected scapular-plane condition were
  audited against the primary paper's pp. 270–272 and Figures 2–3.
- SD and SEM remain distinct. SEM is converted with
  `SD = SEM * sqrt(sample_size)`. Digitization uncertainty remains separate
  from biological variance. Missing SD is excluded by default; 0.5° and 1.0°
  assumptions exist only as named sensitivity scenarios.
- PCHIP is used for study interpolation and consensus splines to avoid
  oscillatory cubic overshoot. First and second derivative coefficients are
  included in the exported JSON.

## Package layout

- `models.py` — immutable raw, provenance, curve, consensus, spline models
- `loader.py` — CSV parsing and validation
- `normalization.py` — non-destructive coordinate transformations
- `weighting.py` — configurable quality policy
- `consensus.py` — within-study curves and weighted confidence envelopes
- `splines.py` — shape-preserving spline models and derivatives
- `visualization.py` — interactive paper/consensus plots
- `exporter.py` — deterministic JSON schema
- `engine.py` — complete orchestration API
- `cli.py` — command-line interface
- `literature_model.py` — verified, range-limited synthesis adapter
- `selection.py` — single-condition selection gate
- `sensitivity.py` — weighting, missing-SD, and leave-one-out analyses
- `audit.py` — coordinate convention audit
- `engineering_report.py` — full-cycle/sector comparison and NO-GO report

Future transmission synthesis should consume `ConsensusShoulderModel.json`;
it should not parse the literature CSV directly.
