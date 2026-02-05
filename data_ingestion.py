"""Dataset loaders and normalization helpers for battery datasets."""
from __future__ import annotations

import io
import json
import zipfile
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

PACK_COLUMN_ALIASES = {
    "time": [
        "time",
        "Time",
        "test_time",
        "Test_Time",
        "Test_Time(s)",
        "Step_Time(s)",
        "step_time",
        "time_s",
        "Time(s)",
    ],
    "pack_voltage": [
        "pack_voltage",
        "voltage",
        "Voltage",
        "Voltage_measured",
        "Voltage_load",
        "Voltage_charge",
        "Voltage(V)",
        "voltage_v",
    ],
    "pack_current": [
        "pack_current",
        "current",
        "Current",
        "Current_measured",
        "Current_load",
        "Current_charge",
        "Current(A)",
        "current_a",
    ],
    "pack_temperature": [
        "pack_temperature",
        "temperature",
        "Temperature",
        "Temperature_measured",
        "Temp",
        "Temp(C)",
        "Temperature(C)",
        "temperature_c",
    ],
}


def normalize_pack_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the dataframe with standard pack column names applied."""
    data = dataframe.copy()
    rename_map = {}
    for target, candidates in PACK_COLUMN_ALIASES.items():
        if target in data.columns:
            continue
        match = _find_column(data.columns, candidates)
        if match:
            rename_map[match] = target
    if rename_map:
        data = data.rename(columns=rename_map)
    return data


def load_nasa_csv(path: str | Path) -> pd.DataFrame:
    """Load a NASA dataset CSV and normalize the column names when possible."""
    data = pd.read_csv(path)
    if _is_nasa_impedance(data):
        return _parse_nasa_impedance(data)
    return normalize_pack_dataframe(data)


def load_matrio_zip(
    path: str | Path, channel: Optional[int | str] = None, max_rows: Optional[int] = None
) -> Dict[str, pd.DataFrame]:
    """Load MATR.io JSON-in-zip data into dataframes keyed by channel."""
    results: Dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(path) as zf:
        json_files = [name for name in zf.namelist() if name.endswith("_structure.json")]
        if channel is not None:
            channel_tag = f"_CH{channel}_"
            json_files = [name for name in json_files if channel_tag in name]
            if not json_files:
                raise ValueError(f"No channel {channel} found in {path}.")
        for name in json_files:
            payload = json.loads(zf.read(name).decode("utf-8"))
            raw_data = payload.get("raw_data", {})
            df = pd.DataFrame(raw_data)
            df = normalize_pack_dataframe(df)
            if max_rows is not None:
                df = df.head(max_rows)
            channel_id = _extract_channel_id(name)
            channel_number = payload.get("channel_id")
            df["channel_id"] = channel_id
            if channel_number is not None:
                df["channel_number"] = channel_number
            df["source_file"] = name
            results[channel_id] = df
    return results


def load_calce_zip(
    path: str | Path,
    max_files: Optional[int] = None,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Load CALCE zip archives that contain XLSX files (sheet2 is used as data)."""
    frames = []
    with zipfile.ZipFile(path) as zf:
        xlsx_files = [name for name in zf.namelist() if name.endswith(".xlsx")]
        if max_files is not None:
            xlsx_files = xlsx_files[:max_files]
        for name in xlsx_files:
            df = _read_calce_sheet(zf.read(name))
            if df.empty:
                continue
            df = normalize_pack_dataframe(df)
            if "pack_temperature" not in df.columns:
                df["pack_temperature"] = np.nan
            df["source_file"] = name
            if max_rows is not None:
                df = df.head(max_rows)
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _find_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    lookup = {col.lower(): col for col in columns}
    for candidate in candidates:
        key = candidate.lower()
        if key in lookup:
            return lookup[key]
    return None


def _extract_channel_id(name: str) -> str:
    match = re.search(r"_CH(\d+)_", name)
    return f"CH{match.group(1)}" if match else name


def _is_nasa_impedance(dataframe: pd.DataFrame) -> bool:
    impedance_cols = {"Battery_impedance", "Rectified_Impedance"}
    return impedance_cols.issubset(set(dataframe.columns))


def _parse_nasa_impedance(dataframe: pd.DataFrame) -> pd.DataFrame:
    data = dataframe.copy()
    complex_cols = [
        "Sense_current",
        "Battery_current",
        "Current_ratio",
        "Battery_impedance",
        "Rectified_Impedance",
    ]
    for col in complex_cols:
        if col not in data.columns:
            continue
        parsed = data[col].apply(_parse_complex)
        data[f"{col}_real"] = np.real(parsed)
        data[f"{col}_imag"] = np.imag(parsed)
        data[f"{col}_abs"] = np.abs(parsed)
        data[f"{col}_phase_deg"] = np.degrees(np.angle(parsed))
    return data


def _parse_complex(value: object) -> complex:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return complex(np.nan, np.nan)
    text = str(value).strip()
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1]
    try:
        return complex(text)
    except ValueError:
        return complex(np.nan, np.nan)


def _read_calce_sheet(xlsx_bytes: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(xlsx_bytes)) as zf:
        shared_strings = _load_shared_strings(zf)
        sheet_path = _pick_calce_sheet(zf)
        sheet_root = ET.fromstring(zf.read(sheet_path))
        namespace = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        rows = sheet_root.findall(".//s:sheetData/s:row", namespace)

        row_dicts = []
        max_cols = 0
        for row in rows:
            row_values: Dict[int, object] = {}
            for cell in row.findall("s:c", namespace):
                cell_ref = cell.get("r")
                if not cell_ref:
                    continue
                col_idx = _column_index(cell_ref)
                value = _read_cell_value(cell, shared_strings, namespace)
                row_values[col_idx] = value
                if col_idx + 1 > max_cols:
                    max_cols = col_idx + 1
            if row_values:
                row_dicts.append(row_values)

        if not row_dicts:
            return pd.DataFrame()

        rows_out = []
        for row_values in row_dicts:
            row_list = [""] * max_cols
            for col_idx, value in row_values.items():
                row_list[col_idx] = value
            rows_out.append(row_list)

        headers = [str(item).strip() for item in rows_out[0]]
        valid_indices = [i for i, name in enumerate(headers) if name]
        columns = [headers[i] for i in valid_indices]
        data_rows = [
            [row[i] if i < len(row) else "" for i in valid_indices] for row in rows_out[1:]
        ]
        df = pd.DataFrame(data_rows, columns=columns)
        for col in df.columns:
            raw = df[col]
            coerced = pd.to_numeric(raw, errors="coerce")
            non_empty = raw.astype(str).str.strip() != ""
            if coerced.notna().sum() == non_empty.sum():
                df[col] = coerced
        return df


def _load_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    shared_strings: list[str] = []
    namespace = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    for si in root.findall("s:si", namespace):
        parts = [node.text or "" for node in si.findall(".//s:t", namespace)]
        shared_strings.append("".join(parts))
    return shared_strings


def _pick_calce_sheet(zf: zipfile.ZipFile) -> str:
    candidates = [name for name in zf.namelist() if name.startswith("xl/worksheets/sheet")]
    if "xl/worksheets/sheet2.xml" in candidates:
        return "xl/worksheets/sheet2.xml"
    if candidates:
        return sorted(candidates)[0]
    raise ValueError("No worksheet found in CALCE XLSX file.")


def _read_cell_value(cell: ET.Element, shared_strings: list[str], namespace: dict) -> object:
    cell_type = cell.get("t")
    value_node = cell.find("s:v", namespace)
    if cell_type == "inlineStr":
        inline_node = cell.find(".//s:t", namespace)
        return inline_node.text if inline_node is not None else ""
    if value_node is None:
        return ""
    value_text = value_node.text or ""
    if cell_type == "s":
        idx = int(value_text) if value_text.isdigit() else None
        return shared_strings[idx] if idx is not None and idx < len(shared_strings) else ""
    return value_text


def _column_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    index = 0
    for ch in letters:
        index = index * 26 + (ord(ch.upper()) - ord("A") + 1)
    return index - 1
