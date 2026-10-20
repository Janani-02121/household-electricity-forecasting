"""Feature engineering and Random Forest training (time-aware, no data leakage)."""
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils.preprocessing import DataError

# Settings per data frequency. Lags/rolling windows are measured in records.
FREQ_CONFIG = {
    "h": {"label": "Hourly", "lags": [1, 2, 3, 24, 48, 168], "windows": [24, 168],
          "step": pd.Timedelta(hours=1), "min_rows": 24 * 21, "per_day": 24, "max_days": 30},
    "D": {"label": "Daily", "lags": [1, 2, 7, 14], "windows": [7, 30],
          "step": pd.Timedelta(days=1), "min_rows": 90, "per_day": 1, "max_days": 365},
}

FEATURE_INFO = {
    "year": "Calendar year", "month": "Month of the year (1-12)", "day": "Day of the month",
    "dayofweek": "Day of week (0 = Monday)", "dayofyear": "Day of the year (1-366)",
    "weekofyear": "ISO week number", "quarter": "Quarter of the year",
    "is_weekend": "1 if Saturday/Sunday, else 0", "hour": "Hour of the day (0-23)",
    "hour_sin": "Cyclical encoding of hour (sine)", "hour_cos": "Cyclical encoding of hour (cosine)",
}


def build_features(df, freq):
    """Create the feature table. Every lag/rolling feature uses PAST values only."""
    cfg = FREQ_CONFIG[freq]
    idx = df.index
    s = df["consumption"]
    X = pd.DataFrame(index=idx)
    X["year"] = idx.year
    X["month"] = idx.month
    X["day"] = idx.day
    X["dayofweek"] = idx.dayofweek
    X["dayofyear"] = idx.dayofyear
    X["weekofyear"] = idx.isocalendar().week.to_numpy().astype(int)
    X["quarter"] = idx.quarter
    X["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    if freq == "h":
        X["hour"] = idx.hour
        X["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
        X["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    for lag in cfg["lags"]:
        X[f"lag_{lag}"] = s.shift(lag)                       # value 'lag' steps ago
    past = s.shift(1)                                          # exclude the current value
    for w in cfg["windows"]:
        X[f"roll_mean_{w}"] = past.rolling(w).mean()
    X[f"roll_std_{cfg['windows'][0]}"] = past.rolling(cfg["windows"][0]).std()
    return X


def describe_feature(name):
    if name in FEATURE_INFO:
        return FEATURE_INFO[name]
    if name.startswith("lag_"):
        return f"Consumption {name[4:]} records earlier"
    if name.startswith("roll_mean_"):
        return f"Mean of the previous {name[10:]} records (current value excluded)"
    if name.startswith("roll_std_"):
        return f"Std. deviation of the previous {name[9:]} records (current value excluded)"
    return name


def _new_model(n_estimators, max_depth):
    return RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                 min_samples_leaf=4, random_state=42, n_jobs=-1)


def train_model(df, freq, dataset_name, n_estimators=100, max_depth=None, test_size=0.2):
    """Chronological split -> train -> evaluate -> refit on all data for forecasting."""
    cfg = FREQ_CONFIG[freq]
    if len(df) < cfg["min_rows"]:
        raise DataError(f"Not enough data to train: {len(df)} {cfg['label'].lower()} records found, "
                        f"at least {cfg['min_rows']} are required.")
    X = build_features(df, freq)
    data = X.join(df["consumption"]).dropna()
    features = list(X.columns)
    split = int(len(data) * (1 - test_size))
    train, test = data.iloc[:split], data.iloc[split:]          # test = the LAST part in time
    if len(train) < 50 or len(test) < 10:
        raise DataError("Not enough records to create both a training and a testing set.")

    model = _new_model(n_estimators, max_depth).fit(train[features], train["consumption"])
    pred = model.predict(test[features])
    actual = test["consumption"].to_numpy()
    mse = mean_squared_error(actual, pred)
    naive_mae = mean_absolute_error(actual, test["lag_1"])      # baseline: "same as previous record"
    metrics = {"mae": float(mean_absolute_error(actual, pred)), "mse": float(mse),
               "rmse": float(np.sqrt(mse)), "r2": float(r2_score(actual, pred)),
               "naive_mae": float(naive_mae), "mean_actual": float(actual.mean())}

    # Final model for forecasting: refit on ALL available data.
    final = _new_model(n_estimators, max_depth).fit(data[features], data["consumption"])
    importance = sorted(zip(features, final.feature_importances_), key=lambda t: -t[1])
    keep = max(max(cfg["lags"]), max(cfg["windows"])) + 5

    bundle = {
        "model": final, "features": features, "freq": freq, "dataset": dataset_name,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "params": {"n_estimators": n_estimators, "max_depth": max_depth if max_depth else "None (unlimited)",
                   "min_samples_leaf": 4, "random_state": 42},
        "n_train": int(len(train)), "n_test": int(len(test)), "n_all": int(len(data)),
        "train_range": [str(train.index[0]), str(train.index[-1])],
        "test_range": [str(test.index[0]), str(test.index[-1])],
        "metrics": metrics,
        "test_index": [str(t) for t in test.index],
        "test_actual": np.round(actual, 4).tolist(), "test_pred": np.round(pred, 4).tolist(),
        "importance": [(n, round(float(v), 5)) for n, v in importance],
        "history_tail": df.iloc[-keep:].copy(), "keep": keep,
        "preview": X.dropna().head(8).round(3).reset_index().astype(str).to_dict(orient="records"),
    }
    return bundle


def save_bundle(bundle, path):
    joblib.dump(bundle, path)


def load_bundle(path):
    return joblib.load(path)
