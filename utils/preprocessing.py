"""Dataset loading, inspection and cleaning (Pandas + NumPy)."""
import csv
import warnings

import numpy as np
import pandas as pd

NA_VALUES = ["?", "NA", "N/A", "n/a", "null", "NULL", "None", "nan", "NaN", "-", ""]
DATE_HINTS = ["datetime", "timestamp", "date", "time"]
VALUE_HINTS = ["global_active_power", "consumption", "kwh", "energy", "usage",
               "power", "load", "demand", "value"]
MIN_ROWS = 30


class DataError(Exception):
    """A problem with the user's data that can be shown as a friendly message."""


# ----------------------------------------------------------------- loading
def read_csv_file(path):
    """Read a CSV file (auto-detects , ; tab | separators)."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            sample = fh.read(4096)
        if not sample.strip():
            raise DataError("The uploaded file is empty.")
        try:
            sep = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        except csv.Error:
            sep = ","
        df = pd.read_csv(path, sep=sep, na_values=NA_VALUES, encoding_errors="replace")
    except DataError:
        raise
    except pd.errors.EmptyDataError:
        raise DataError("The uploaded file is empty.")
    except Exception as exc:  # parser errors, binary files, etc.
        raise DataError(f"The file could not be read as a CSV file ({exc}).")
    df.columns = [str(c).strip() for c in df.columns]
    if df.shape[1] < 2:
        raise DataError("The CSV needs at least two columns: a date/time column and a consumption column.")
    if len(df) < MIN_ROWS:
        raise DataError(f"The CSV has only {len(df)} rows. At least {MIN_ROWS} rows are required.")
    return df


def inspect_dataframe(df):
    """Summary of the raw dataset shown on the upload page."""
    missing = df.isna().sum()
    preview = df.head(10).astype(str).replace({"nan": "", "NaT": "", "None": "", "<NA>": ""})
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing": {c: int(missing[c]) for c in df.columns},
        "missing_total": int(missing.sum()),
        "duplicates": int(df.duplicated().sum()),
        "preview": preview.to_dict(orient="records"),
    }


# ------------------------------------------------------- column detection
def parse_dates(series):
    """Parse date strings; chooses day-first or month-first by whichever parses more values."""
    text = series.astype(str).str.strip()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = pd.to_datetime(text, errors="coerce", dayfirst=False)
        b = pd.to_datetime(text, errors="coerce", dayfirst=True)
    na, nb = a.notna().sum(), b.notna().sum()
    if na != nb:
        return a if na > nb else b
    iso = text.head(50).str.match(r"^\d{4}-").all()
    return a if iso else b  # ambiguous dd/mm vs mm/dd -> assume day-first


def _looks_like_dates(series):
    sample = series.dropna().astype(str).head(200)
    if sample.empty or pd.to_numeric(sample, errors="coerce").notna().mean() > 0.5:
        return False
    return parse_dates(sample).notna().mean() >= 0.8


def detect_columns(df):
    """Guess the date column, optional separate time column and consumption column."""
    cols = list(df.columns)
    low = {c: c.lower() for c in cols}
    date_col = None
    for hint in DATE_HINTS:
        for c in cols:
            if hint in low[c] and _looks_like_dates(df[c]):
                date_col = c
                break
        if date_col:
            break
    if date_col is None:
        for c in cols:
            if _looks_like_dates(df[c]):
                date_col = c
                break
    time_col = None
    if date_col and "time" not in low[date_col]:
        for c in cols:
            if c != date_col and low[c] == "time":
                time_col = c
    candidates = []
    for c in cols:
        if c in (date_col, time_col):
            continue
        if pd.to_numeric(df[c], errors="coerce").notna().mean() >= 0.7:
            candidates.append(c)
    value_col = None
    for hint in VALUE_HINTS:
        for c in candidates:
            if hint in low[c]:
                value_col = c
                break
        if value_col:
            break
    if value_col is None and candidates:
        value_col = candidates[0]
    return {"date_col": date_col, "time_col": time_col, "value_col": value_col}


# ---------------------------------------------------------------- cleaning
def clean_dataset(df, date_col, value_col, time_col=None):
    """Clean the raw data and return (clean_df, freq, report).

    clean_df : DataFrame indexed by regular timestamps with one column 'consumption'.
    freq     : 'h' (hourly) or 'D' (daily).
    report   : dictionary describing every cleaning action (nothing is hidden).
    """
    for col in (date_col, value_col):
        if col not in df.columns:
            raise DataError(f"Column '{col}' was not found in the dataset.")
    if date_col == value_col:
        raise DataError("The date column and the consumption column must be different.")

    text = df[date_col].astype(str).str.strip()
    if time_col and time_col in df.columns and time_col != date_col:
        text = text + " " + df[time_col].astype(str).str.strip()
    ts = parse_dates(text)
    raw_values = df[value_col]
    values = pd.to_numeric(raw_values, errors="coerce")
    work = pd.DataFrame({"timestamp": ts.values, "consumption": values.values})

    report = {"raw_rows": int(len(df)), "actions": []}
    report["non_numeric"] = int((values.isna() & raw_values.notna()).sum())
    report["missing_raw"] = int(raw_values.isna().sum())
    report["negative"] = int((work["consumption"] < 0).sum())
    work.loc[work["consumption"] < 0, "consumption"] = np.nan
    report["invalid_dates"] = int(work["timestamp"].isna().sum())
    work = work.dropna(subset=["timestamp"])
    if len(work) < MIN_ROWS:
        raise DataError("Fewer than 30 rows have a valid date. Please check the selected date column.")
    if work["consumption"].notna().sum() < MIN_ROWS:
        raise DataError("Fewer than 30 rows have a valid numeric consumption value. Please check the selected consumption column.")

    # duplicates
    report["duplicate_rows"] = int(work.duplicated().sum())
    work = work.drop_duplicates()
    report["duplicate_timestamps"] = int(work["timestamp"].duplicated().sum())
    work = work.groupby("timestamp", as_index=True)["consumption"].mean().sort_index().to_frame()

    # frequency
    steps = work.index.to_series().diff().dropna()
    median_step = steps.median()
    if median_step <= pd.Timedelta(hours=3):
        freq, label = "h", "Hourly"
    elif median_step <= pd.Timedelta(days=3):
        freq, label = "D", "Daily"
    else:
        raise DataError("The data must be recorded at least every 3 days. Hourly, minute-level or daily data is supported.")
    report["frequency"] = label
    report["median_step"] = str(median_step)

    # outliers: cap extreme values with a wide IQR fence (3 x IQR)
    s = work["consumption"]
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lower, upper = max(0.0, q1 - 3 * iqr), q3 + 3 * iqr
    mask = (s < lower) | (s > upper)
    report["outliers"] = int(mask.sum())
    report["outlier_bounds"] = [round(float(lower), 3), round(float(upper), 3)]
    work["consumption"] = s.clip(lower=lower, upper=upper)

    # regular time grid (mean per interval) and fill the gaps
    regular = work["consumption"].resample(freq).mean()
    report["gaps_filled"] = int(regular.isna().sum())
    regular = regular.interpolate(method="time", limit_direction="both")
    clean = regular.to_frame("consumption")
    clean.index.name = "timestamp"

    report["final_rows"] = int(len(clean))
    report["start"] = str(clean.index[0])
    report["end"] = str(clean.index[-1])
    a = report["actions"]
    a.append(f"Converted '{date_col}' to date/time and sorted the data chronologically.")
    a.append(f"Invalid dates removed: {report['invalid_dates']}.")
    a.append(f"Non-numeric consumption values converted to missing: {report['non_numeric']}.")
    a.append(f"Negative consumption values treated as invalid: {report['negative']}.")
    a.append(f"Duplicate rows removed: {report['duplicate_rows']}; repeated timestamps averaged: {report['duplicate_timestamps']}.")
    a.append(f"Outliers capped using a 3 x IQR fence [{report['outlier_bounds'][0]}, {report['outlier_bounds'][1]}]: {report['outliers']}.")
    a.append(f"Detected {label.lower()} data; built a regular time grid ({report['final_rows']} records).")
    a.append(f"Missing/invalid intervals filled by time interpolation: {report['gaps_filled']}.")
    return clean, freq, report
