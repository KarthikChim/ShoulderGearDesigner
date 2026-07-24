# RESEARCH BENCH PROTOTYPE — NOT FOR HUMAN USE

## Scope and provenance

- Source: McClure2001 only
- Condition: healthy, unloaded, raising, scapular-plane elevation
- Verified HT range: 11°–147°
- No extrapolation, wrapping, or periodic continuation
- Transitions are mechanical placeholders and excluded from biomechanical error

## Slope audit

| Candidate | min dST/dE | max dST/dE | min dψ/dφ | max dψ/dφ | near-zero regions |
|---|---:|---:|---:|---:|---|
| Raw | 3.4912275e-05 | 0.95245811 | 2.6378163e-05 | 0.71963501 | ((11.0, 14.57),) |
| Regularized | 0.10588235 | 0.93011648 | 0.08 | 0.70275467 | none |

The raw literature curve is preserved unchanged. The regularized candidate enforces a positive mechanical ratio while preserving endpoint excursion.

## Sector validation

| Candidate | Decision | Max ST error | RMS ST error | Min radius | Contact ratio estimate | Teeth input/output |
|---|---|---:|---:|---:|---:|---:|
| Raw | GO FOR SOFTWARE SIMULATION | 0° | 0° | 0.0031653 | 1.331 | 29/29 |
| Regularized | GO FOR SOFTWARE SIMULATION | 0.97138856° | 0.68141698° | 8.8889 | 1.331 | 27/27 |

## Four-way comparison

| Alternative | Max ST error | RMS ST error | Ratio range | Radius range |
|---|---:|---:|---:|---:|
| legacy_periodic_full_revolution | 19.073° | 11.9261° | 0.577879–1.53275 | 43.94854986912697–76.05145013087304 |
| literature_forced_full_revolution | 0.0814942° | 0.00711294° | -0.456746–2.99683 | None–None |
| literature_partial_sector_raw | 0° | 0° | 2.63782e-05–0.719635 | 0.0031652961233264846–119.99683470387667 |
| literature_partial_sector_regularized | 0.971389° | 0.681417° | 0.08–0.702755 | 8.888888888881509–111.1111111111185 |

## Final decision

**GO FOR SOFTWARE SIMULATION**

- Preferred candidate from numerical gates: raw
- Preferred input sector angle: 180.0°
- Preferred center distance: 120.0 model units
- Preferred module: 2.500
- Hard stops: 11° and 147° HT equivalents

## Remaining blockers

- Complete sector blanks and hard-stop solid geometry are placeholders.
- Mating tooth interference is not cleared without complete blanks.
- Contact ratio is an unloaded estimate, not loaded contact analysis.
- Strength, fatigue, tolerances, bearings, backlash under load, and actuator failure behavior are not validated.
- Single-study target prevents leave-one-study-out validation.
- No human-subject or wearable safety validation exists.

**Never interpret this report as GO FOR HUMAN USE.**
