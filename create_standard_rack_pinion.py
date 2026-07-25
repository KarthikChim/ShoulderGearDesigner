"""Generate the standalone standards-based rack and matching pinion."""

from __future__ import annotations

import argparse
from pathlib import Path

from standard_involute import StandardGearParameters, export_pair, generate_rack


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("standard_rack_outputs"))
    parser.add_argument("--module", type=float, default=2.0)
    parser.add_argument("--pressure-angle", type=float, default=20.0)
    parser.add_argument("--rack-teeth", type=int, default=14)
    parser.add_argument("--pinion-teeth", type=int, default=24)
    parser.add_argument("--backlash", type=float, default=0.15)
    parser.add_argument("--face-width", type=float, default=10.0)
    args = parser.parse_args()
    generated = generate_rack(
        StandardGearParameters(
            module=args.module,
            pressure_angle_deg=args.pressure_angle,
            rack_teeth=args.rack_teeth,
            pinion_teeth=args.pinion_teeth,
            backlash=args.backlash,
            face_width=args.face_width,
        )
    )
    paths = export_pair(generated, args.output)
    print("Validation:", "PASS" if generated.validation.valid else "FAIL")
    print(f"Contact ratio: {generated.validation.transverse_contact_ratio:.4f}")
    for warning in generated.validation.warnings:
        print("WARNING:", warning)
    for name, path in paths.items():
        print(f"{name}: {path.resolve()}")
    return 0 if generated.validation.valid else 2


if __name__ == "__main__":
    raise SystemExit(main())

