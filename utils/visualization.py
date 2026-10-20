"""Statistics, chart data, energy analysis and recommendations (all computed from real data)."""
import numpy as np
import pandas as pd

from utils.model_training import FREQ_CONFIG

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _r(x, n=3):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, n) if np.isfinite(x) else None


def _series(s, n=3):
    return [_r(v, n) for v in s]


def daily_totals(df, freq):
    """One value per day (sum of the day's records). Incomplete first/last days are dropped."""
    s = df["consumption"]
    if freq == "D":
        return s
    daily = s.resample("D").sum()
    full = s.resample("D").count() == 24
    return daily[full] if full.sum() >= 2 else daily


def summary_stats(df):
    s = df["consumption"]
    return {"records": int(len(s)), "mean": _r(s.mean()), "min": _r(s.min()), "max": _r(s.max()),
            "total": _r(s.sum(), 2), "std": _r(s.std()), "latest": _r(s.iloc[-1]),
            "latest_time": f"{s.index[-1]:%Y-%m-%d %H:%M}", "min_time": f"{s.idxmin():%Y-%m-%d %H:%M}",
            "max_time": f"{s.idxmax():%Y-%m-%d %H:%M}", "start": f"{s.index[0]:%Y-%m-%d %H:%M}",
            "end": f"{s.index[-1]:%Y-%m-%d %H:%M}"}


def thresholds(s):
    p33, p66, p95 = np.percentile(s, [33, 66, 95])
    return {"low": _r(p33), "high": _r(p66), "peak": _r(p95)}


def classify(value, t):
    if value >= t["peak"]:
        return "Peak"
    if value >= t["high"]:
        return "High"
    if value >= t["low"]:
        return "Moderate"
    return "Low"


def distribution(values, t):
    labels = ["Low", "Moderate", "High", "Peak"]
    v = np.asarray(values, dtype=float)
    counts = [int((v < t["low"]).sum()), int(((v >= t["low"]) & (v < t["high"])).sum()),
              int(((v >= t["high"]) & (v < t["peak"])).sum()), int((v >= t["peak"]).sum())]
    total = max(len(v), 1)
    return [{"label": l, "count": c, "percent": round(100 * c / total, 1)} for l, c in zip(labels, counts)]


def trend_info(daily):
    n = len(daily)
    w = 30 if n >= 60 else max(n // 2, 1)
    recent = float(daily.iloc[-w:].mean())
    prev = float(daily.iloc[-2 * w:-w].mean()) if n >= 2 * w else float(daily.iloc[:max(n - w, 1)].mean())
    change = (recent - prev) / prev * 100 if prev else 0.0
    label = "Increasing" if change > 5 else "Decreasing" if change < -5 else "Stable"
    slope = float(np.polyfit(np.arange(n), daily.to_numpy(dtype=float), 1)[0]) if n > 2 else 0.0
    return {"window_days": int(w), "recent_avg": _r(recent), "previous_avg": _r(prev),
            "change_pct": _r(change, 1), "label": label, "slope_per_day": _r(slope, 4)}


def analyze(df, freq, forecast_df=None):
    """Full energy analysis used by the Analysis, Dashboard and Report pages."""
    s = df["consumption"]
    per_day = FREQ_CONFIG[freq]["per_day"]
    daily = daily_totals(df, freq)
    t = thresholds(s)
    a = {"stats": summary_stats(df), "thresholds": t, "distribution": distribution(s, t),
         "trend": trend_info(daily), "freq": freq, "per_day": per_day,
         "daily_avg": _r(daily.mean())}

    # peak usage periods
    weekday = daily.groupby(daily.index.dayofweek).mean().reindex(range(7))
    a["peak_weekday"] = DAYS[int(weekday.idxmax())]
    a["low_weekday"] = DAYS[int(weekday.idxmin())]
    a["weekday_avg"] = _r(daily[daily.index.dayofweek < 5].mean())
    a["weekend_avg"] = _r(daily[daily.index.dayofweek >= 5].mean()) if (daily.index.dayofweek >= 5).any() else None
    if freq == "h":
        hourly = s.groupby(s.index.hour).mean()
        top = hourly.sort_values(ascending=False).head(3).index.tolist()
        a["peak_hour"] = int(hourly.idxmax())
        a["peak_hour_value"] = _r(hourly.max())
        a["peak_hours"] = sorted(int(h) for h in top)
        a["lowest_hour"] = int(hourly.idxmin())
    top_day = daily.idxmax()
    a["peak_day"] = f"{top_day:%Y-%m-%d}"
    a["peak_day_value"] = _r(daily.max())
    spikes = s[s > s.mean() + 3 * s.std()]
    a["spike_count"] = int(len(spikes))

    # forecast section
    a["forecast"] = None
    if forecast_df is not None and len(forecast_df):
        f = forecast_df["predicted"]
        fd = _r(f.mean() * per_day)
        recent = a["trend"]["recent_avg"]
        a["forecast"] = {
            "mean": _r(f.mean()), "min": _r(f.min()), "max": _r(f.max()), "total": _r(f.sum(), 2),
            "peak_time": f"{f.idxmax():%Y-%m-%d %H:%M}", "peak_value": _r(f.max()),
            "mean_class": classify(f.mean(), t), "daily_equiv": fd,
            "distribution": distribution(f, t),
            "high_points": int((f >= t["high"]).sum()), "points": int(len(f)),
            "vs_recent_pct": _r((fd - recent) / recent * 100, 1) if recent else None,
        }
    a["recommendations"] = recommendations(a)
    return a


def recommendations(a):
    """Rules driven by the numbers above - every tip lists the evidence it is based on."""
    recs = []
    mean = a["stats"]["mean"]
    if a["freq"] == "h" and a["peak_hour_value"] and mean and a["peak_hour_value"] > mean * 1.10:
        pct = (a["peak_hour_value"] / mean - 1) * 100
        hrs = ", ".join(f"{h:02d}:00" for h in a["peak_hours"])
        recs.append({"level": "warning", "title": "Reduce usage during peak hours",
                     "text": f"Consumption is highest around {hrs}. Try to run flexible appliances (washing machine, iron, water heater) outside these hours.",
                     "evidence": f"Average at {a['peak_hour']:02d}:00 is {a['peak_hour_value']} vs overall average {mean} (+{pct:.0f}%)."})
    wd, we = a["weekday_avg"], a["weekend_avg"]
    if wd and we and abs(we - wd) / wd > 0.08:
        higher = "weekends" if we > wd else "weekdays"
        recs.append({"level": "info", "title": f"Usage is higher on {higher}",
                     "text": f"Review which appliances run on {higher} and switch off devices that are not in use.",
                     "evidence": f"Average daily consumption: weekdays {wd}, weekends {we}."})
    tr = a["trend"]
    if tr["label"] == "Increasing":
        recs.append({"level": "warning", "title": "Consumption is trending upwards",
                     "text": "Check for appliances left on, ageing equipment or changes in household habits.",
                     "evidence": f"Last {tr['window_days']} days average {tr['recent_avg']} per day vs {tr['previous_avg']} before ({tr['change_pct']:+}%)."})
    elif tr["label"] == "Decreasing":
        recs.append({"level": "success", "title": "Consumption is trending downwards",
                     "text": "Your recent usage is lower than before. Keep up the current habits.",
                     "evidence": f"Last {tr['window_days']} days average {tr['recent_avg']} per day vs {tr['previous_avg']} before ({tr['change_pct']:+}%)."})
    if a["spike_count"]:
        recs.append({"level": "warning", "title": "Monitor unusually high consumption",
                     "text": "Some records are far above normal. Check what was running at those times (heavy appliances, faulty devices).",
                     "evidence": f"{a['spike_count']} record(s) exceed mean + 3 standard deviations; highest day was {a['peak_day']} ({a['peak_day_value']})."})
    fc = a["forecast"]
    if fc:
        if fc["high_points"] and fc["points"]:
            share = fc["high_points"] / fc["points"] * 100
            recs.append({"level": "warning" if share > 40 else "info", "title": "Plan for predicted high-consumption periods",
                         "text": f"The forecast shows the highest usage around {fc['peak_time']}. Plan heavy appliance use away from it.",
                         "evidence": f"{fc['high_points']} of {fc['points']} forecast records ({share:.0f}%) are in the High/Peak class (>= {a['thresholds']['high']})."})
        if fc["vs_recent_pct"] is not None and abs(fc["vs_recent_pct"]) > 5:
            word = "higher" if fc["vs_recent_pct"] > 0 else "lower"
            recs.append({"level": "info", "title": f"Forecast is {word} than recent usage",
                         "text": "Compare this with your plans (guests, seasonal appliances) and adjust usage if needed.",
                         "evidence": f"Forecast daily average {fc['daily_equiv']} vs recent {a['trend']['recent_avg']} ({fc['vs_recent_pct']:+}%)."})
    recs.append({"level": "success", "title": "Build energy-efficient habits",
                 "text": "Switch off appliances at the plug instead of standby, use energy-efficient lighting and appliances, and avoid running devices unnecessarily.",
                 "evidence": "General best practice (not derived from the dataset)."})
    return recs


def build_charts(df, freq, bundle=None, forecast_df=None, analysis=None):
    """JSON-ready data for every Chart.js chart."""
    s = df["consumption"]
    daily = daily_totals(df, freq)
    charts = {}
    hist = daily if len(daily) <= 800 else daily.resample("W").mean()
    charts["historical"] = {"labels": [f"{d:%Y-%m-%d}" for d in hist.index], "values": _series(hist),
                            "title": "Daily consumption" if len(daily) <= 800 else "Weekly average of daily consumption"}
    last = daily.iloc[-60:]
    charts["daily"] = {"labels": [f"{d:%Y-%m-%d}" for d in last.index], "values": _series(last)}
    wk = daily.groupby(daily.index.dayofweek).mean().reindex(range(7))
    charts["weekly"] = {"labels": DAYS, "values": _series(wk)}
    mo = daily.groupby(daily.index.to_period("M")).mean()
    charts["monthly"] = {"labels": [str(p) for p in mo.index], "values": _series(mo)}
    if freq == "h":
        hr = s.groupby(s.index.hour).mean()
        charts["hourly"] = {"labels": [f"{h:02d}:00" for h in hr.index], "values": _series(hr)}
    top = daily.sort_values(ascending=False).head(10)
    charts["peak"] = {"labels": [f"{d:%Y-%m-%d}" for d in top.index], "values": _series(top)}
    t = analysis["thresholds"] if analysis else thresholds(s)
    charts["classes"] = {"labels": ["Low", "Moderate", "High", "Peak"],
                         "values": [d["percent"] for d in distribution(s, t)]}

    comp_l, comp_v = ["Overall average"], [_r(daily.mean())]
    tr = trend_info(daily)
    comp_l += ["Previous period", "Recent period"]
    comp_v += [tr["previous_avg"], tr["recent_avg"]]
    comp_l += ["Weekdays", "Weekends"]
    comp_v += [_r(daily[daily.index.dayofweek < 5].mean()),
               _r(daily[daily.index.dayofweek >= 5].mean()) if (daily.index.dayofweek >= 5).any() else None]
    if forecast_df is not None and len(forecast_df):
        comp_l.append("Forecast")
        comp_v.append(_r(forecast_df["predicted"].mean() * FREQ_CONFIG[freq]["per_day"]))
    charts["comparison"] = {"labels": comp_l, "values": comp_v}

    if bundle:
        idx, act, prd = bundle["test_index"], bundle["test_actual"], bundle["test_pred"]
        step = max(len(idx) // 1500, 1)
        charts["actual_pred"] = {"labels": idx[::step], "actual": act[::step], "pred": prd[::step]}
        charts["importance"] = {"labels": [n for n, _ in bundle["importance"][:12]],
                                "values": [v for _, v in bundle["importance"][:12]]}
    if forecast_df is not None and len(forecast_df):
        tail = s.iloc[-(7 * FREQ_CONFIG[freq]["per_day"]):]
        charts["forecast"] = {"hist_labels": [str(i) for i in tail.index], "hist": _series(tail),
                              "labels": [str(i) for i in forecast_df.index], "values": _series(forecast_df["predicted"])}
    return charts
