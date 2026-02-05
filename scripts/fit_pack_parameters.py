"""Fit pack parameters from standardized CSVs and write JSON outputs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data_ingestion import normalize_pack_dataframe
from parameter_identification import PackParameterIdentifier

REQUIRED_COLUMNS = ("time", "pack_voltage", "pack_current")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit pack parameters from standardized CSVs.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/standardized"),
        help="Root directory containing standardized CSV files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/standardized_params"),
        help="Output directory for parameter JSON files.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.csv",
        help="Glob pattern for CSV files.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row limit per input file.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on the first file with missing columns or fit errors.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path to write a summary JSON of outputs/skips.",
    )
    args = parser.parse_args()

    input_root = args.input
    output_root = args.output
    output_root.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_root.rglob(args.pattern))
    if not csv_files:
        raise SystemExit(f"No CSV files found under {input_root}")

    summary = {"outputs": [], "skipped": []}

    for csv_path in csv_files:
        try:
            payload = fit_file(csv_path, input_root, args.max_rows)
        except Exception as exc:
            if args.strict:
                raise
            summary["skipped"].append({"file": str(csv_path), "reason": str(exc)})
            print(f"Skipping {csv_path}: {exc}")
            continue

        out_path = output_root / csv_path.relative_to(input_root)
        out_path = out_path.with_suffix(".json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(to_builtin(payload), handle, indent=2, ensure_ascii=True)
        summary["outputs"].append({"file": str(csv_path), "output": str(out_path)})
        print(f"Saved {out_path}")

    if args.summary_json:
        summary_path = args.summary_json
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(to_builtin(summary), handle, indent=2, ensure_ascii=True)
        print(f"Wrote summary to {summary_path}")


def fit_file(csv_path: Path, input_root: Path, max_rows: Optional[int]) -> Dict[str, Any]:
    data = pd.read_csv(csv_path, low_memory=False)
    data = normalize_pack_dataframe(data)
    data = _coerce_numeric(data, REQUIRED_COLUMNS + ("pack_temperature",))
    data = data.dropna(subset=REQUIRED_COLUMNS)
    if max_rows is not None:
        data = data.head(max_rows)

    if "pack_temperature" in data.columns and data["pack_temperature"].isna().all():
        data = data.drop(columns=["pack_temperature"])

    missing = [col for col in REQUIRED_COLUMNS if col not in data.columns]
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")
    if data.empty:
        raise ValueError("no usable rows after cleaning")

    results = PackParameterIdentifier.identify_from_pack_data(data)

    payload = {
        "source_file": str(csv_path),
        "source_relpath": _safe_relpath(csv_path, input_root),
        "n_rows": int(len(data)),
        "columns": list(data.columns),
        "results": results,
    }
    return payload


def _coerce_numeric(data: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for column in columns:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def _safe_relpath(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: to_builtin(val) for key, val in value.items()}
    if isinstance(value, list):
        return [to_builtin(val) for val in value]
    if isinstance(value, tuple):
        return [to_builtin(val) for val in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


if __name__ == "__main__":
    main()
