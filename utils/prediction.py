"""Recursive multi-step forecasting with the trained Random Forest."""
import numpy as np
import pandas as pd

from utils.model_training import FREQ_CONFIG, build_features
from utils.preprocessing import DataError


def steps_for_request(bundle, days=None, end_date=None):
    """Convert 'N days' or 'up to a date' into the number of future records."""
    cfg = FREQ_CONFIG[bundle["freq"]]
    last = bundle["history_tail"].index[-1]
    if end_date:
        end = pd.Timestamp(end_date)
        end = end + pd.Timedelta(hours=23) if bundle["freq"] == "h" else end
        steps = int((end - last) / cfg["step"])
        if steps < 1:
            raise DataError(f"The forecast end date must be after the last data point ({last:%Y-%m-%d %H:%M}).")
    else:
        if not days or days < 1:
            raise DataError("Please choose the number of days to forecast.")
        steps = int(days) * cfg["per_day"]
    if steps > cfg["max_days"] * cfg["per_day"]:
        raise DataError(f"For {cfg['label'].lower()} data the maximum forecast period is {cfg['max_days']} days.")
    return steps


def forecast(bundle, steps):
    """Predict 'steps' records ahead. Each prediction is fed back as history for the next one."""
    freq, features, model = bundle["freq"], bundle["features"], bundle["model"]
    step = FREQ_CONFIG[freq]["step"]
    model.set_params(n_jobs=1)   # single-row predictions are faster without thread overhead
    hist = bundle["history_tail"].copy()
    keep = bundle["keep"]
    out_ts, out_val = [], []
    for _ in range(steps):
        next_ts = hist.index[-1] + step
        row = pd.DataFrame({"consumption": [np.nan]}, index=pd.DatetimeIndex([next_ts]))
        hist = pd.concat([hist, row])
        X = build_features(hist, freq).iloc[[-1]][features]     # lags only look backwards
        pred = max(float(model.predict(X)[0]), 0.0)
        hist.iloc[-1, hist.columns.get_loc("consumption")] = pred
        hist = hist.iloc[-keep:]
        out_ts.append(next_ts)
        out_val.append(pred)
    result = pd.DataFrame({"timestamp": out_ts, "predicted": out_val}).set_index("timestamp")
    result.index.name = "timestamp"
    return result
