# Literature Planetary Pitch-Curve Validation

**Result: PASS**

## Mechanism

- Sun angle: fixed at 0 rad
- Carrier angle: McClure2001 scapular excursion
- Planet absolute angle: humerothoracic elevation excursion
- Contact type: external sun/planet rolling
- Teeth: not generated

## Signed rolling equation

`(-dθc/dE) r_s + (dθp/dE - dθc/dE) r_p = 0`

The signed ratio `dθp_rel/dθs_rel` is negative because external gears
rotate in opposite directions when measured relative to the carrier.

## Validation metrics

- passed: True
- sample_count: 4001
- valid_range_deg: (11.0, 147.0)
- sun_stationary: True
- carrier_matches_st: True
- planet_matches_ht: True
- center_distance_error_mm: 1.4210854715202004e-14
- contact_coincidence_error_mm: 5.826883674891095e-14
- tangent_alignment_error_deg: 0.02759597761041167
- rolling_residual_mm_per_rad: 1.0658141036401503e-14
- finite_positive_pitch_radii: True
- no_sign_discontinuity: True
- sun_curve_simple: True
- planet_curve_simple: True
- no_extrapolation: True
- maximum_elevation_error_deg: 1.6538884343610288e-13
- rms_elevation_error_deg: 7.706243880822002e-14
- endpoint_elevation_error_deg: 1.526666247102488e-13
