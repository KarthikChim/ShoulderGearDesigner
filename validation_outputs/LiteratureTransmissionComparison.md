# Literature Transmission Comparison

## Selected condition

- Healthy, unloaded, raising
- Dynamic scapular-plane abduction, 40° anterior to frontal
- McClure2001 Figure 3B
- Supported HT range: 11.0°–147.0°
- Extrapolation: forbidden

## Numerical comparison

| Alternative | Max ST error | RMS ST error | Minimum ratio | Result |
|---|---:|---:|---:|---|
| Closed full revolution | 0.081194° | 0.007108° | -0.454251 | FAIL |
| Partial sector | 7.105427e-15° | 1.931919e-15° | 3.491228e-05 | PASS |

Minimum sector pitch radius at 100-unit center distance: **0.003491 units**.

The sector alternative requires independent lower/upper mechanical hard stops and a non-wrapping reset to the lower stop.

## GO / NO-GO

**NO-GO for manufacturing or human use.**

- Closed full-revolution mapping has a negative ratio near periodic closure.
- Selected condition has one contributing study, so leave-one-study-out validation is not estimable.
- GH=HT−ST is not verified as an exact 3-D rotational decomposition.
- Sector mapping reaches a 0.003491-unit pitch radius, below the configured 2.000-unit minimum.
- No strength, backlash, fatigue, tolerance, or human-subject safety validation exists.

The sector mapping is suitable only for further bench-top research. This report does not authorize final gear geometry or powered wearable testing.
