"""
Upload service — parses Apple Health XML exports, CSV, and JSON files
into flat dicts compatible with save_wearable_sync().
"""

import csv
import io
import json
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime


# ---------------------------------------------------------------------------
# Column alias map: friendly / vendor names → WearableSync column names
# ---------------------------------------------------------------------------

_ALIASES = {
    # Apple Health HealthKit identifiers
    "HKQuantityTypeIdentifierHeartRateVariabilitySDNN": "hrv_rmssd",
    "HKQuantityTypeIdentifierRestingHeartRate": "resting_hr",
    "HKQuantityTypeIdentifierHeartRate": "avg_hr",
    "HKQuantityTypeIdentifierStepCount": "steps",
    "HKQuantityTypeIdentifierActiveEnergyBurned": "active_calories",
    "HKQuantityTypeIdentifierBasalEnergyBurned": "_basal_calories",
    "HKQuantityTypeIdentifierDistanceWalkingRunning": "distance_meters",
    "HKQuantityTypeIdentifierFlightsClimbed": "floors_climbed",
    "HKQuantityTypeIdentifierOxygenSaturation": "spo2",
    "HKQuantityTypeIdentifierRespiratoryRate": "respiration_rate",
    "HKQuantityTypeIdentifierVO2Max": "vo2_max",
    "HKQuantityTypeIdentifierAppleExerciseTime": "active_minutes",
    "HKQuantityTypeIdentifierBodyTemperature": "skin_temp_delta",

    # Friendly CSV column names
    "date": "sync_date",
    "hrv": "hrv_rmssd",
    "hrv_sdnn": "hrv_rmssd",
    "hrv_rmssd": "hrv_rmssd",
    "resting_heart_rate": "resting_hr",
    "resting_hr": "resting_hr",
    "heart_rate_avg": "avg_hr",
    "avg_hr": "avg_hr",
    "heart_rate_max": "hr_max",
    "hr_max": "hr_max",
    "steps": "steps",
    "active_energy_kcal": "active_calories",
    "active_calories": "active_calories",
    "total_energy_kcal": "calories_total",
    "calories_total": "calories_total",
    "active_minutes": "active_minutes",
    "exercise_minutes": "active_minutes",
    "distance_meters": "distance_meters",
    "distance_km": "_distance_km",
    "flights_climbed": "floors_climbed",
    "floors_climbed": "floors_climbed",
    "spo2": "spo2",
    "spo2_avg": "spo2",
    "respiratory_rate": "respiration_rate",
    "respiration_rate": "respiration_rate",
    "vo2_max": "vo2_max",
    "sleep_duration_hours": "sleep_duration_hours",
    "sleep_hours": "sleep_duration_hours",
    "sleep_deep_minutes": "_sleep_deep_min",
    "sleep_rem_minutes": "_sleep_rem_min",
    "sleep_light_minutes": "_sleep_light_min",
    "sleep_deep_pct": "sleep_deep_pct",
    "sleep_rem_pct": "sleep_rem_pct",
    "sleep_light_pct": "sleep_light_pct",
    "sleep_score": "sleep_score",
    "recovery_score": "recovery_score",
    "stress_avg": "stress_avg",
    "strain_score": "strain_score",
    "skin_temp_delta": "skin_temp_delta",

    # --- Garmin Connect CSV export column names (per-report downloads) ---
    # HRV report
    "avg hrv":                      "hrv_rmssd",
    "average hrv":                  "hrv_rmssd",
    "last night 5-minute high":     "hrv_rmssd",  # Garmin HRV CSV header variant
    # Resting HR report
    "resting heart rate (bpm)":     "resting_hr",
    "resting hr (bpm)":             "resting_hr",
    # Sleep report
    "sleep score":                  "sleep_score",
    "total sleep time":             "sleep_duration_hours",   # Garmin gives minutes; normalized below
    "total sleep (seconds)":        "_sleep_seconds",
    "rem sleep time":               "_sleep_rem_min",
    "light sleep time":             "_sleep_light_min",
    "deep sleep time":              "_sleep_deep_min",
    "awake sleep time":             "_awake_min",
    # Activity / Steps report
    "total steps":                  "steps",
    "steps (steps)":                "steps",
    "distance (km)":                "_distance_km",
    "distance (meters)":            "distance_meters",
    "active calories (kcal)":       "active_calories",
    "calories (kcal)":              "calories_total",
    "floors climbed (floors)":      "floors_climbed",
    "moderate intensity minutes":   "_mod_min",
    "vigorous intensity minutes":   "_vig_min",
    # Stress report
    "avg stress level":             "stress_avg",
    "average stress":               "stress_avg",
    # SpO2 / Pulse Ox
    "avg spo2 (%)":                 "spo2",
    "spo2 (%)":                     "spo2",
    # Body Battery
    "body battery charged":         "_bb_charged",
    "body battery drained":         "_bb_drained",
    # Respiration
    "avg waking respiration rate":  "respiration_rate",
    "avg sleep respiration rate":   "respiration_rate",
}

# WearableSync columns that accept numeric values
_NUMERIC_COLS = {
    "hrv_rmssd", "resting_hr", "recovery_score", "active_minutes",
    "active_calories", "strain_score", "sleep_score", "stress_avg",
    "steps", "spo2", "respiration_rate", "vo2_max",
    "sleep_deep_pct", "sleep_rem_pct", "sleep_light_pct",
    "sleep_duration_hours", "skin_temp_delta", "avg_hr", "hr_max",
    "calories_total", "distance_meters", "floors_climbed",
}


# ---------------------------------------------------------------------------
# Derivation logic — compute missing proprietary metrics
# ---------------------------------------------------------------------------

def derive_missing_fields(row: dict) -> dict:
    """Fill in sleep_score, recovery_score, stress_avg from available data."""

    # --- sleep_score from sleep stages + duration ---
    if not row.get("sleep_score"):
        deep = row.get("sleep_deep_pct")
        rem = row.get("sleep_rem_pct")
        dur = row.get("sleep_duration_hours")
        if deep is not None and rem is not None and dur is not None:
            # Weighted formula: deep sleep and REM are most restorative,
            # duration contributes up to 40 points (optimal at 8h)
            duration_score = min(dur / 8.0, 1.0) * 40
            stage_score = min(deep * 1.5 + rem * 1.0, 60)
            row["sleep_score"] = int(min(100, duration_score + stage_score))
        elif dur is not None:
            # Duration-only fallback
            row["sleep_score"] = int(min(100, (dur / 8.0) * 70))

    # --- recovery_score from HRV ---
    if not row.get("recovery_score"):
        hrv = row.get("hrv_rmssd")
        rhr = row.get("resting_hr")
        if hrv is not None:
            # Simple percentile mapping: 20ms → 20, 50ms → 55, 80ms+ → 85+
            base = min(100, max(0, (hrv - 10) * 1.1))
            # Bonus if RHR is low (athlete range)
            if rhr is not None and rhr < 60:
                base = min(100, base + 5)
            row["recovery_score"] = int(base)

    # --- stress_avg as inverse HRV proxy ---
    if not row.get("stress_avg"):
        hrv = row.get("hrv_rmssd")
        if hrv is not None:
            # Higher HRV → lower stress. 80ms → ~25 stress, 30ms → ~70 stress
            row["stress_avg"] = int(max(0, min(100, 100 - hrv * 1.0)))

    return row


# ---------------------------------------------------------------------------
# Apple Health XML parser
# ---------------------------------------------------------------------------

def _parse_date(date_str: str) -> str:
    """Extract YYYY-MM-DD from Apple Health date strings like '2026-04-14 08:30:00 -0500'."""
    return date_str[:10]


def _aggregate_sleep_analysis(records: list[dict]) -> dict[str, dict]:
    """
    Aggregate HKCategoryTypeIdentifierSleepAnalysis records into daily summaries.
    Returns {date_str: {sleep_duration_hours, sleep_deep_pct, sleep_rem_pct, sleep_light_pct}}.
    """
    daily = defaultdict(lambda: {"deep_sec": 0, "rem_sec": 0, "core_sec": 0, "total_sec": 0})

    for rec in records:
        start_str = rec.get("startDate", "")
        end_str = rec.get("endDate", "")
        value = rec.get("value", "")
        if not start_str or not end_str:
            continue

        try:
            fmt = "%Y-%m-%d %H:%M:%S %z"
            start_dt = datetime.strptime(start_str, fmt)
            end_dt = datetime.strptime(end_str, fmt)
        except ValueError:
            try:
                fmt2 = "%Y-%m-%d %H:%M:%S"
                start_dt = datetime.strptime(start_str[:19], fmt2)
                end_dt = datetime.strptime(end_str[:19], fmt2)
            except ValueError:
                continue

        duration_sec = (end_dt - start_dt).total_seconds()
        if duration_sec <= 0 or duration_sec > 86400:
            continue

        # Wakeup date is the date the sleep is attributed to
        wake_date = end_dt.strftime("%Y-%m-%d")
        day = daily[wake_date]
        day["total_sec"] += duration_sec

        if "Deep" in value or "AsleepDeep" in value:
            day["deep_sec"] += duration_sec
        elif "REM" in value or "AsleepREM" in value:
            day["rem_sec"] += duration_sec
        elif "Core" in value or "AsleepCore" in value or "Asleep" in value:
            day["core_sec"] += duration_sec

    result = {}
    for day_str, d in daily.items():
        total = d["total_sec"]
        if total < 1800:  # less than 30 min — skip noise
            continue
        result[day_str] = {
            "sleep_duration_hours": round(total / 3600, 2),
            "sleep_deep_pct": round(d["deep_sec"] / total * 100, 1) if total else 0,
            "sleep_rem_pct": round(d["rem_sec"] / total * 100, 1) if total else 0,
            "sleep_light_pct": round(d["core_sec"] / total * 100, 1) if total else 0,
        }
    return result


def parse_apple_health_xml(file_bytes: bytes) -> tuple[list[dict], list[str]]:
    """
    Parse an Apple Health export (.zip or raw .xml).
    Returns (rows: list of WearableSync-keyed dicts, errors: list of error strings).
    Each row has a 'sync_date' key.
    """
    xml_data = None
    errors = []

    # Handle zip or raw XML
    if file_bytes[:4] == b'PK\x03\x04':
        try:
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
                xml_names = [n for n in zf.namelist() if n.endswith('.xml') and 'export' in n.lower()]
                if not xml_names:
                    xml_names = [n for n in zf.namelist() if n.endswith('.xml')]
                if not xml_names:
                    return [], ["No XML file found in zip archive"]
                xml_data = zf.read(xml_names[0])
        except zipfile.BadZipFile:
            return [], ["Invalid zip file"]
    else:
        xml_data = file_bytes

    # Parse XML iteratively to handle large files
    daily_data = defaultdict(lambda: defaultdict(list))
    sleep_records = []

    try:
        for event, elem in ET.iterparse(io.BytesIO(xml_data), events=("end",)):
            if elem.tag == "Record":
                rec_type = elem.get("type", "")
                start_date = elem.get("startDate", "")
                value = elem.get("value", "")
                day = _parse_date(start_date)

                if not day or not rec_type:
                    elem.clear()
                    continue

                if rec_type == "HKCategoryTypeIdentifierSleepAnalysis":
                    sleep_records.append({
                        "startDate": start_date,
                        "endDate": elem.get("endDate", ""),
                        "value": value,
                    })
                elif rec_type in _ALIASES:
                    col = _ALIASES[rec_type]
                    try:
                        numeric_val = float(value)
                        daily_data[day][col].append(numeric_val)
                    except (ValueError, TypeError):
                        pass

                elem.clear()
    except ET.ParseError as e:
        return [], [f"XML parse error: {e}"]

    # Aggregate sleep
    sleep_by_day = _aggregate_sleep_analysis(sleep_records)

    # Build per-day rows
    rows = []
    for day_str in sorted(daily_data.keys()):
        cols = daily_data[day_str]
        row = {"sync_date": day_str}

        for col_name, values in cols.items():
            if col_name == "_basal_calories":
                continue  # handled separately
            if not values:
                continue

            # Aggregation strategy per metric
            if col_name in ("steps", "active_calories", "calories_total",
                            "active_minutes", "floors_climbed"):
                row[col_name] = int(sum(values))
            elif col_name in ("distance_meters",):
                row[col_name] = round(sum(values), 1)
            elif col_name in ("hr_max",):
                row[col_name] = int(max(values))
            elif col_name == "spo2":
                # Apple reports SpO2 as 0-1 fraction
                avg_val = sum(values) / len(values)
                row[col_name] = round(avg_val * 100 if avg_val <= 1 else avg_val, 1)
            else:
                # Default: daily average (HR, HRV, RHR, respiratory rate, etc.)
                row[col_name] = round(sum(values) / len(values), 1)

        # Merge basal + active for total calories
        basal = daily_data[day_str].get("_basal_calories", [])
        if basal and "calories_total" not in row:
            active = row.get("active_calories", 0)
            row["calories_total"] = int(sum(basal) + active)

        # Merge sleep data
        if day_str in sleep_by_day:
            row.update(sleep_by_day[day_str])

        # Derive missing proprietary fields
        row = derive_missing_fields(row)

        # Cast to int where appropriate
        for k in ("hrv_rmssd", "resting_hr", "avg_hr", "hr_max", "steps",
                   "active_calories", "active_minutes", "calories_total",
                   "floors_climbed", "sleep_score", "recovery_score", "stress_avg"):
            if k in row and row[k] is not None:
                row[k] = int(row[k])

        rows.append(row)

    return rows, errors


# ---------------------------------------------------------------------------
# CSV / JSON parsers
# ---------------------------------------------------------------------------

def _normalize_row(raw: dict) -> "tuple[dict | None, str | None]":
    """Map a single raw dict (from CSV or JSON) to WearableSync column names.
    Returns (mapped_dict, error_or_None)."""
    mapped = {}
    sync_date = None

    for key, value in raw.items():
        clean_key = key.strip().lower().replace(" ", "_")
        # Also try original spacing for multi-word Garmin headers
        clean_key_spaced = key.strip().lower()
        col = _ALIASES.get(clean_key_spaced, _ALIASES.get(clean_key, clean_key))

        if col == "sync_date":
            sync_date = str(value).strip()
            continue

        if col.startswith("_"):
            # Internal temp columns
            if col == "_distance_km":
                try:
                    mapped["distance_meters"] = round(float(value) * 1000, 1)
                except (ValueError, TypeError):
                    pass
            elif col == "_sleep_seconds":
                try:
                    mapped["sleep_duration_hours"] = round(float(value) / 3600, 2)
                except (ValueError, TypeError):
                    pass
            elif col in ("_sleep_deep_min", "_sleep_rem_min", "_sleep_light_min", "_awake_min"):
                try:
                    mapped[col] = float(value)
                except (ValueError, TypeError):
                    pass
            elif col in ("_mod_min", "_vig_min"):
                try:
                    mapped[col] = float(value)
                except (ValueError, TypeError):
                    pass
            elif col in ("_bb_charged", "_bb_drained"):
                try:
                    mapped[col] = float(value)
                except (ValueError, TypeError):
                    pass
            continue

        if col not in _NUMERIC_COLS:
            continue

        try:
            v = str(value).strip()
            if v == "" or v.lower() in ("na", "n/a", "null", "none", ""):
                continue
            mapped[col] = float(v)
        except (ValueError, TypeError):
            continue

    if not sync_date:
        return None, "Missing 'date' column"

    # Validate date format
    try:
        datetime.strptime(sync_date, "%Y-%m-%d")
    except ValueError:
        try:
            dt = datetime.strptime(sync_date, "%m/%d/%Y")
            sync_date = dt.strftime("%Y-%m-%d")
        except ValueError:
            return None, f"Invalid date format: {sync_date}"

    mapped["sync_date"] = sync_date

    # Convert sleep stage minutes → percentages if duration is available
    deep_min = mapped.pop("_sleep_deep_min", None)
    rem_min = mapped.pop("_sleep_rem_min", None)
    light_min = mapped.pop("_sleep_light_min", None)
    dur_h = mapped.get("sleep_duration_hours")

    if dur_h and dur_h > 0:
        total_min = dur_h * 60
        if deep_min is not None:
            mapped["sleep_deep_pct"] = round(deep_min / total_min * 100, 1)
        if rem_min is not None:
            mapped["sleep_rem_pct"] = round(rem_min / total_min * 100, 1)
        if light_min is not None:
            mapped["sleep_light_pct"] = round(light_min / total_min * 100, 1)
    elif deep_min is not None and rem_min is not None and light_min is not None:
        total_min = deep_min + rem_min + light_min
        if total_min > 0:
            mapped["sleep_deep_pct"] = round(deep_min / total_min * 100, 1)
            mapped["sleep_rem_pct"] = round(rem_min / total_min * 100, 1)
            mapped["sleep_light_pct"] = round(light_min / total_min * 100, 1)
            mapped["sleep_duration_hours"] = round(total_min / 60, 2)

    # Garmin: moderate + vigorous intensity minutes → active_minutes
    mod_min = mapped.pop("_mod_min", None)
    vig_min = mapped.pop("_vig_min", None)
    if mod_min is not None or vig_min is not None:
        if "active_minutes" not in mapped:
            mapped["active_minutes"] = (mod_min or 0) + (vig_min or 0)

    # Garmin: body battery charged/drained → recovery_score proxy
    bb_charged = mapped.pop("_bb_charged", None)
    if bb_charged is not None and "recovery_score" not in mapped:
        mapped["recovery_score"] = min(100.0, bb_charged)

    # Derive missing proprietary fields
    mapped = derive_missing_fields(mapped)

    return mapped, None


def parse_csv_upload(file_bytes: bytes) -> tuple[list[dict], list[str]]:
    """Parse a CSV file into WearableSync-keyed row dicts."""
    rows = []
    errors = []

    try:
        text = file_bytes.decode("utf-8-sig")  # handle BOM
    except UnicodeDecodeError:
        text = file_bytes.decode("latin-1")

    reader = csv.DictReader(io.StringIO(text))
    for i, raw_row in enumerate(reader, start=2):  # line 2 = first data row
        mapped, err = _normalize_row(raw_row)
        if err:
            errors.append(f"Row {i}: {err}")
            continue
        if mapped:
            rows.append(mapped)

    return rows, errors


def parse_json_upload(file_bytes: bytes) -> tuple[list[dict], list[str]]:
    """Parse a JSON file (array of objects) into WearableSync-keyed row dicts."""
    rows = []
    errors = []

    try:
        data = json.loads(file_bytes)
    except json.JSONDecodeError as e:
        return [], [f"Invalid JSON: {e}"]

    if isinstance(data, dict):
        data = [data]

    if not isinstance(data, list):
        return [], ["JSON must be an array of objects"]

    for i, raw_row in enumerate(data, start=1):
        if not isinstance(raw_row, dict):
            errors.append(f"Item {i}: not an object")
            continue
        mapped, err = _normalize_row(raw_row)
        if err:
            errors.append(f"Item {i}: {err}")
            continue
        if mapped:
            rows.append(mapped)

    return rows, errors
