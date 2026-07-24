# RESEARCH BENCH PROTOTYPE — NOT FOR HUMAN USE

## Search

- Candidates evaluated: 81
- Sector angles: 120°, 135°, 150°, 165°, 180°, 195°, 210°, 225°, 240°
- Center distance, module, pressure angle, backlash, minimum ratio, and smoothing ranges were all sampled.
- Ranking order: interference, tangent/curvature, contact ratio, root thickness, biomechanical RMS error, compactness.

## Selected screened candidate

- sector_angle_deg: 240.0
- center_distance_mm: 120.0
- module_mm: 2.5
- pressure_angle_deg: 20.0
- backlash_mm: 0.3
- minimum_ratio_constraint: 0.2
- smoothing_strength: 0.5
- minimum_pitch_radius_mm: 20.00388179374964
- maximum_st_error_deg: 9.633490766343854
- rms_st_error_deg: 7.353035201022116
- minimum_contact_ratio: 2.2284102341760708

## Full 2001-position mesh

- Maximum penetration area: 1.5467700772325454
- Minimum clearance: 0.0
- Maximum contact mismatch: 5.783398860650275e-14
- No tooth skipping: True
- No contact discontinuity: True

## Hard gates

- FAIL — continuous_tangent
- PASS — bounded_curvature
- FAIL — bounded_curvature_derivative
- PASS — adjacent_tooth_overlap_free
- FAIL — mating_interference_free
- PASS — minimum_pitch_radius_valid
- PASS — minimum_contact_ratio_valid
- PASS — minimum_root_thickness_valid
- PASS — sector_blanks_closed_valid
- PASS — no_extrapolation
- FAIL — maximum_st_error_valid
- FAIL — rms_st_error_valid
- FAIL — rack_envelope_verified
- FAIL — hard_pass

## Decision: GO FOR SOFTWARE SIMULATION

Prototype DXF/SVG was not exported because every hard gate did not pass.

Remaining blockers:
- continuous_tangent
- bounded_curvature_derivative
- mating_interference_free
- maximum_st_error_valid
- rms_st_error_valid
- rack_envelope_verified

**Never GO FOR HUMAN USE.**
