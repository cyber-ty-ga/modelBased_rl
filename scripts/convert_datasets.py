"""Convert raw datasets into standardized CSV files."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data_ingestion import load_calce_zip, load_matrio_zip, load_nasa_csv, normalize_pack_dataframe

STANDARD_COLUMNS = ["time", "pack_voltage", "pack_current", "pack_temperature"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert datasets to standardized CSVs.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root data directory containing dataset folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/standardized"),
        help="Output directory for standardized CSV files.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row limit per output file (useful for quick tests).",
    )
    parser.add_argument(
        "--max-calce-files",
        type=int,
        default=None,
        help="Optional limit on the number of CALCE XLSX files per zip.",
    )
    parser.add_argument(
        "--matr-channel",
        type=int,
        default=None,
        help="Optional MATR.io channel number to convert (default: all).",
    )
    args = parser.parse_args()

    data_root = args.data_root
    output_root = args.output
    output_root.mkdir(parents=True, exist_ok=True)

    convert_nasa(data_root / "NASA Dataset", output_root, args.max_rows)
    convert_matr(data_root / "matr.io", output_root, args.max_rows, args.matr_channel)
    convert_calce(
        data_root / "CALCE battery", output_root, args.max_rows, args.max_calce_files
    )


def convert_nasa(nasa_root: Path, output_root: Path, max_rows: Optional[int]) -> None:
    if not nasa_root.exists():
        print(f"NASA dataset not found at {nasa_root}")
        return

    output_time = output_root / "nasa"
    output_impedance = output_root / "nasa_impedance"
    output_time.mkdir(parents=True, exist_ok=True)
    output_impedance.mkdir(parents=True, exist_ok=True)

    for csv_path in sorted(nasa_root.rglob("*.csv")):
        df = load_nasa_csv(csv_path)
        df = _annotate_source(df, "nasa", csv_path, nasa_root)
        if max_rows is not None:
            df = df.head(max_rows)

        if _has_pack_timeseries(df):
            df = _ensure_standard_columns(df)
            out_path = output_time / f"{csv_path.stem}.csv"
        else:
            out_path = output_impedance / f"{csv_path.stem}_impedance.csv"

        _write_csv(df, out_path)
        print(f"NASA -> {out_path}")


def convert_matr(
    matr_root: Path, output_root: Path, max_rows: Optional[int], channel: Optional[int]
) -> None:
    if not matr_root.exists():
        print(f"MATR.io dataset not found at {matr_root}")
        return

    output_dir = output_root / "matr"
    output_dir.mkdir(parents=True, exist_ok=True)

    for zip_path in sorted(matr_root.glob("*.zip")):
        frames = load_matrio_zip(zip_path, channel=channel, max_rows=max_rows)
        for channel_id, df in frames.items():
            df = _annotate_source(df, "matr", zip_path, matr_root)
            if _has_pack_timeseries(df):
                df = _ensure_standard_columns(df)
            out_path = output_dir / f"{zip_path.stem}_{channel_id}.csv"
            _write_csv(df, out_path)
            print(f"MATR -> {out_path}")


def convert_calce(
    calce_root: Path,
    output_root: Path,
    max_rows: Optional[int],
    max_files: Optional[int],
) -> None:
    if not calce_root.exists():
        print(f"CALCE dataset not found at {calce_root}")
        return

    output_dir = output_root / "calce"
    output_dir.mkdir(parents=True, exist_ok=True)

    for zip_path in sorted(calce_root.rglob("*.zip")):
        df = load_calce_zip(zip_path, max_files=max_files, max_rows=max_rows)
        if df.empty:
            print(f"CALCE -> no data in {zip_path}")
            continue
        df = _annotate_source(df, "calce", zip_path, calce_root)
        if _has_pack_timeseries(df):
            df = _ensure_standard_columns(df)
        out_path = output_dir / f"{zip_path.stem}.csv"
        _write_csv(df, out_path)
        print(f"CALCE -> {out_path}")


def _annotate_source(df: pd.DataFrame, dataset: str, path: Path, root: Path) -> pd.DataFrame:
    data = df.copy()
    data["source_dataset"] = dataset
    data["source_file"] = str(path)
    try:
        data["source_relpath"] = str(path.relative_to(root.parent))
    except ValueError:
        data["source_relpath"] = str(path)
    return data


def _has_pack_timeseries(df: pd.DataFrame) -> bool:
    return {"time", "pack_voltage", "pack_current"}.issubset(df.columns)


def _ensure_standard_columns(df: pd.DataFrame) -> pd.DataFrame:
    data = normalize_pack_dataframe(df)
    for column in STANDARD_COLUMNS:
        if column not in data.columns:
            data[column] = pd.NA
    ordered = STANDARD_COLUMNS + [c for c in data.columns if c not in STANDARD_COLUMNS]
    return data[ordered]


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


if __name__ == "__main__":
    main()
