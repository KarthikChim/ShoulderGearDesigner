"""CSV ingestion and non-destructive validation."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

from .models import (
    KINEMATIC_VARIABLES,
    MotionKey,
    PaperMetadata,
    RawLiteratureRow,
    ValidationIssue,
    ValidationReport,
)


REQUIRED_COLUMNS = (
    "PaperID",
    "Paper",
    "Authors",
    "Year",
    "MotionType",
    "MotionPlane",
    "MeasurementMethod",
    "Direction",
    "Loaded",
    "HT_Elevation_deg",
    *KINEMATIC_VARIABLES,
)

REQUIRED_METADATA_VALUES = (
    "PaperID",
    "Paper",
    "Authors",
    "Year",
    "MotionType",
    "MotionPlane",
    "MeasurementMethod",
    "HealthyOnly",
    "SampleSize",
    "Direction",
    "Loaded",
    "DataSource",
    "ExtractionMethod",
)


def _optional_float(value: str) -> float | None:
    return None if not value.strip() else float(value)


def _optional_int(value: str) -> int | None:
    return None if not value.strip() else int(float(value))


def _optional_bool(value: str) -> bool | None:
    if not value.strip():
        return None
    if value.strip().lower() in {"true", "1", "yes"}:
        return True
    if value.strip().lower() in {"false", "0", "no"}:
        return False
    raise ValueError(f"Invalid boolean {value!r}")


def load_literature_csv(
    path: str | Path,
) -> tuple[tuple[RawLiteratureRow, ...], tuple[PaperMetadata, ...], ValidationReport]:
    source = Path(path)
    issues: list[ValidationIssue] = []
    with source.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        columns = tuple(reader.fieldnames or ())
        missing_columns = [name for name in REQUIRED_COLUMNS if name not in columns]
        if missing_columns:
            raise ValueError(f"CSV is missing required columns: {missing_columns}")
        raw_dicts = list(reader)

    missing_by_column = {
        column: sum(not row.get(column, "").strip() for row in raw_dicts)
        for column in columns
    }
    for column in REQUIRED_METADATA_VALUES:
        missing_rows = tuple(
            index + 2
            for index, row in enumerate(raw_dicts)
            if not row.get(column, "").strip()
        )
        if missing_rows:
            issues.append(
                ValidationIssue(
                    "error",
                    "missing_required_value",
                    f"{column} is missing in {len(missing_rows)} rows.",
                    missing_rows,
                )
            )
    signatures = Counter(tuple(row.get(column, "") for column in columns) for row in raw_dicts)
    duplicate_count = sum(count - 1 for count in signatures.values() if count > 1)
    if duplicate_count:
        issues.append(
            ValidationIssue(
                "warning", "duplicate_rows", f"{duplicate_count} exact duplicate rows found."
            )
        )

    parsed: list[RawLiteratureRow] = []
    paper_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    repeated: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for offset, row in enumerate(raw_dicts, start=2):
        paper_rows[row["PaperID"]].append(row)
        try:
            elevation = _optional_float(row["HT_Elevation_deg"])
            values = tuple(
                (variable, float(row[variable]))
                for variable in KINEMATIC_VARIABLES
                if row[variable].strip()
            )
            if elevation is not None and not 0.0 <= elevation <= 180.0:
                issues.append(
                    ValidationIssue(
                        "error", "invalid_ht_angle",
                        f"HT elevation {elevation} is outside 0–180 degrees.", (offset,)
                    )
                )
            for variable, value in values:
                if not -180.0 <= value <= 180.0:
                    issues.append(
                        ValidationIssue(
                            "error", "out_of_range_angle",
                            f"{variable}={value} is outside ±180 degrees.", (offset,)
                        )
                    )
            motion_key = MotionKey(
                row["MotionType"].strip(),
                row["MotionPlane"].strip(),
                row["Direction"].strip(),
                _optional_bool(row["Loaded"]),
                _optional_bool(row["HealthyOnly"]),
            )
            parsed_row = RawLiteratureRow(
                row_number=offset,
                fields=tuple((column, row.get(column, "")) for column in columns),
                paper_id=row["PaperID"].strip(),
                figure_or_table=row["FigureOrTable"].strip(),
                motion_key=motion_key,
                ht_elevation_deg=elevation,
                values=values,
                sd=_optional_float(row["SD"]),
                sem=_optional_float(row["SEM"]),
                sample_size=_optional_int(row["SampleSize"]),
                extraction_method=row["ExtractionMethod"].strip(),
                notes=row["Notes"],
            )
            parsed.append(parsed_row)
            if elevation is not None:
                for variable, _ in values:
                    key = (
                        parsed_row.paper_id,
                        parsed_row.figure_or_table,
                        motion_key.identifier,
                        variable,
                        f"{elevation:.12g}",
                    )
                    repeated[key].append(offset)
        except (ValueError, TypeError) as error:
            issues.append(
                ValidationIssue("error", "invalid_value", str(error), (offset,))
            )

    repeated_groups = [rows for rows in repeated.values() if len(rows) > 1]
    if repeated_groups:
        issues.append(
            ValidationIssue(
                "warning",
                "repeated_ht_elevations",
                f"{len(repeated_groups)} paper/figure/variable groups repeat an HT elevation; "
                "values are preserved and averaged only in derived study curves.",
                tuple(number for group in repeated_groups for number in group[:2]),
            )
        )

    papers: list[PaperMetadata] = []
    for paper_id, rows in sorted(paper_rows.items()):
        first = rows[0]
        sizes = [_optional_int(row["SampleSize"]) for row in rows]
        sizes = [value for value in sizes if value is not None]
        sources = tuple(sorted({row["DataSource"].strip() for row in rows if row["DataSource"].strip()}))
        papers.append(
            PaperMetadata(
                paper_id=paper_id,
                title=first["Paper"],
                authors=first["Authors"],
                year=int(first["Year"]),
                measurement_method=first["MeasurementMethod"],
                sample_size=max(sizes) if sizes else None,
                healthy_only=_optional_bool(first["HealthyOnly"]),
                data_sources=sources,
            )
        )
    report = ValidationReport(
        source=source,
        row_count=len(raw_dicts),
        paper_count=len(papers),
        duplicate_row_count=duplicate_count,
        missing_by_column=missing_by_column,
        repeated_elevation_groups=len(repeated_groups),
        issues=tuple(issues),
    )
    return tuple(parsed), tuple(papers), report
