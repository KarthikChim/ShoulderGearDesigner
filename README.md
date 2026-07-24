# Shoulder Gear Designer

## Literature biomechanics engine

The authoritative literature pipeline is isolated in
`biomechanics/`. Run it to create `ConsensusShoulderModel.json`; it does not
modify the current transmission or gear synthesis. See
`biomechanics/README.md` for scientific safeguards and configuration.

`model_pathways.py` preserves the independently testable legacy pathway and
adds the verified, range-limited literature pathway. Run
`validate_literature.py` to rebuild the JSON, coordinate audit, comparison
plot, and engineering report. Current result: **NO-GO for manufacturing or
human use**; the partial-sector mapping is research-only.

## Phase 5 biomechanics validation

The normal display now distinguishes the mechanical derivative `dψ/dφ` from
the actual incremental biomechanical ratio:

```text
dST/dE = (dψ/dφ) / 3
dGH/dE = 1 - (dψ/dφ) / 3
GH:ST  = 3 / (dψ/dφ) - 1
```

At startup an independent 3601-point sweep compares target and actual GH/ST
motion, checks endpoints and derivative continuity, reports maximum/RMS
errors, and evaluates the requested checkpoints. The current result is
honestly **FAIL**, not because of broken gear closure, but because:

1. the piecewise incremental ratios integrate to 64° GH / 26° ST at 90° and
   76° / 38° at 114°, conflicting with the supplied 72° / 18° and 84° / 30°
   checkpoints; and
2. a periodic C² cubic transmission cannot exactly overlay a discontinuous
   step-ratio target at every breakpoint.

The gears still finish at exactly 120° GH / 60° ST, maintain GH + ST equal to
total elevation, remain positive, and pass first/second derivative continuity.

## Phase 4 complete pair

Both gears are complete rack-generated non-circular gears. Their teeth are
spaced by pitch-curve arc length, not polar angle. The generator sweeps one
shared 20-degree straight-sided rack cutter under pure rolling, and a
half-pitch phase offset places a tooth opposite a mating gap at assembly.

The application automatically chooses module, addendum, dedendum, root
fillet, tooth count, and envelope resolution from center distance. These
values are deliberately not exposed in the interface. SVG and DXF exports
contain both closed tooth boundaries at full floating-point precision.

The displayed base, root, and addendum references are parallel non-circular
curves.  Calling them circles would be geometrically incorrect for this gear.
The numerical validation is a design-screening check; final manufacturing
still requires contact-ratio analysis, backlash, strength, tolerance, and
physical safety validation.

Shoulder Gear Designer is a modular Python desktop application for exploring
gear-based approximations of scapulohumeral coordination during shoulder
abduction.

The current version synthesizes a smooth periodic transmission and
rack-generates both members of the fixed-center conjugate non-circular pair.

## Installation

Python 3.12 or newer is recommended. Tkinter must be provided by the Python
installation.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

On Windows, activate with `.venv\Scripts\activate`.

## Controls

- Edit center distance and simulation speed, then select
  **Apply Geometry**.
- Start, pause, reset, or step the simulation.
- Scrub the arm-elevation slider.
- Use **Advanced Debug** to reveal pitch curves, centers, radius vectors,
  kinematic values, and the live ratio graph.
- Scroll over the viewport to zoom and drag with the left mouse button to pan.
- Export both complete tooth boundaries and pitch curves as SVG or DXF; CSV
  retains the full transmission synthesis samples.

## Simulation model

The commanded arm elevation runs from 0 to 180 degrees. The default
incremental GH-to-ST regions are configurable in `settings.py`:

| Elevation | GH:ST |
|---|---:|
| 0–30° | 4:1 |
| 30–90° | 2:1 |
| 90–114° | 1:1 |
| 114–180° | 2:1 |

For ratio `k`, each incremental elevation `dE` is divided as:

```text
dGH = k / (k + 1) * dE
dST = 1 / (k + 1) * dE
```

This produces 120 degrees GH and 60 degrees ST at 180 degrees total elevation.

Because a closed pair must return to its initial geometry after one revolution,
the final 60 degrees of scapular motion is normalized to one output-gear
revolution. Thus 6 degrees of output-gear rotation represents 1 degree of
scapular rotation.

For transmission `m = dψ/dφ` and fixed center distance `a`, pitch radii are:

```text
r_input  = a m / (1 + m)
r_output = a / (1 + m)
The program samples 4097 double-precision points, checks closure and velocity
continuity, verifies the fixed-radius sum, and uses Shapely to confirm that both
pitch curves are valid convex loops that do not overlap at assembly.

## Project structure

- `main.py` — application entry point
- `gui.py` — Tkinter interface
- `settings.py` — editable configuration
- `kinematics.py` — shoulder-rhythm integration
- `pitch_curve.py` — extensible pitch-curve interface
- `noncircular.py` — periodic spline, conjugate synthesis, and validation
- `tooth_geometry.py` — arc-length frames and rack-cutter tooth envelope
- `pair_teeth.py` — automatic shared rack design and assembled-pair validation
- `biomechanics_validation.py` — independent target/actual engineering report
- `gear.py`, `meshing.py` — mechanical domain objects
- `animation.py`, `simulation.py` — timing and orchestration
- `drawing.py` — Matplotlib renderer
- `export_csv.py`, `export_svg.py`, `export_dxf.py` — exact-data exporters
- `tests/` — GUI-independent pytest suite

## Roadmap

1. Contact-ratio and detailed undercut analysis
2. Load, stress, backlash, and tolerance analysis
3. Helical and herringbone geometry
4. Manufacturing-oriented CAD export

## Limitations and safety

This application is an engineering research tool, not a validated medical
device design system. Its geometry must not be used for powered wearable
testing or human loading without independent mechanical and clinical review.
