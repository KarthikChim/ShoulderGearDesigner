# Gear Mesh Optimization Report

Biomechanics, transmission, pitch curves, and 11°–147° range were locked.

Candidates evaluated: 90000
Locked pitch SHA-256: `b0c65e7bd0e333d55e9d79cbc3d3c24ee81b3ce18b9c04dc6e81277802108a64`

## Preferred unloaded prototype

- Module: 2.00 mm
- Pressure angle: 20°
- Backlash: 0.30 mm
- Profile relief: 0.15 mm
- Face width: 16 mm
- Root fillet: 1.0 mm
- Tooth-root embed: 2.0 mm
- Center-distance offset: +0.00 mm
- Tooth style: Spur
- Estimated angular lost motion: 0.363°
- Estimated contact ratio: 1.210
- Root strength index: 79.83
- Undercut risk flag: True

The undercut flag is a conservative equivalent-circular estimate;
the final rack polygons passed the sampled interference sweep.
- Exact maximum penetration: 0.000000 mm²
- Exact minimum clearance: 0.0679 mm

## Module study

| Module | Teeth (in/out) | Tip mm | Root mm | Contact ratio | Lost motion | Undercut | 0.4 mm printable |
|---:|---:|---:|---:|---:|---:|:---:|:---:|
| 1.50 | 18/18 | 0.174 | 4.071 | 1.405 | 0.273° | no | no |
| 1.75 | 15/15 | 0.178 | 4.725 | 1.355 | 0.333° | no | no |
| 2.00 | 13/13 | 1.086 | 4.361 | 1.210 | 0.363° | yes | yes |
| 2.25 | 12/12 | 0.836 | 6.682 | 1.403 | 0.030° | no | no |
| 2.50 | 10/10 | 0.940 | 7.435 | 1.374 | 0.030° | no | no |

Smallest recommended module for a 0.4 mm nozzle: **2.0 mm**.

## Fit calibration

| Fit | Tooth clearance | Angular backlash | Center offset |
|---|---:|---:|---:|
| Tight | 0.550 mm | 0.333° | -0.025 mm |
| Normal | 0.650 mm | 0.394° | +0.025 mm |
| Loose | 0.750 mm | 0.454° | +0.075 mm |

The preferred style is spur. Helical and herringbone results are a
comparative sweep study only and are not automatically selected.

RESEARCH-ONLY UNLOADED HAND-DRIVEN PROTOTYPE — NOT FOR HUMAN USE.
