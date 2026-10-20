"""Household Electricity Consumption Forecasting - Flask application.

Routes only handle web requests; the machine-learning work lives in the utils/ package.
Run with:  python app.py   then open  http://127.0.0.1:5000
"""
import json
import os
import traceback
from datetime import datetime

import pandas as pd
from flask import (Flask, Response, flash, redirect, render_template, request,
                   url_for)
from werkzeug.utils import secure_filename

from utils import model_training as mt
from utils import preprocessing as pp
from utils import prediction as pr
from utils import visualization as viz

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
MODEL_PATH = os.path.join(BASE_DIR, "model", "trained_model.pkl")
HISTORY_PATH = os.path.join(BASE_DIR, "model", "prediction_history.json")
SAMPLE_PATH = os.path.join(BASE_DIR, "dataset", "electricity_consumption.csv")
SAMPLE_NAME = "electricity_consumption.csv"

app = Flask(__name__)
app.secret_key = "household-electricity-forecasting-secret"
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB upload limit
os.makedirs(UPLOAD_DIR, exist_ok=True)

# In-memory state of the running application (single local user).
STATE = {"raw": None, "info": None, "detected": None, "filename": None, "is_sample": False,
         "df": None, "freq": None, "report": None, "unit": "kWh",
         "bundle": None, "forecast": None, "history": []}


# ------------------------------------------------------------------ helpers
def load_dataset(path, filename, is_sample=False):
    """Read + inspect + auto-detect columns (does not clean yet)."""
    raw = pp.read_csv_file(path)
    STATE.update(raw=raw, info=pp.inspect_dataframe(raw), detected=pp.detect_columns(raw),
                 filename=filename, is_sample=is_sample)


def process_dataset(date_col, value_col, time_col, unit):
    """Clean the raw data and reset anything that depended on the previous dataset."""
    df, freq, report = pp.clean_dataset(STATE["raw"], date_col, value_col, time_col or None)
    STATE.update(df=df, freq=freq, report=report, unit=unit or "kWh", bundle=None, forecast=None)
    STATE["report"]["columns"] = {"date": date_col, "time": time_col or "-", "value": value_col}


def try_load_saved_model():
    if not os.path.exists(MODEL_PATH):
        return
    try:
        bundle = mt.load_bundle(MODEL_PATH)
        if bundle.get("dataset") == STATE["filename"] and bundle.get("freq") == STATE["freq"]:
            STATE["bundle"] = bundle
    except Exception:
        print("Saved model could not be loaded - please train the model again.")


def load_sample():
    load_dataset(SAMPLE_PATH, SAMPLE_NAME, is_sample=True)
    d = STATE["detected"]
    process_dataset(d["date_col"], d["value_col"], d["time_col"], "kWh")
    try_load_saved_model()


def load_history():
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH, encoding="utf-8") as fh:
                STATE["history"] = json.load(fh)
        except Exception:
            STATE["history"] = []


def save_history():
    try:
        with open(HISTORY_PATH, "w", encoding="utf-8") as fh:
            json.dump(STATE["history"], fh)
    except OSError:
        pass


def has_data():
    return STATE["df"] is not None


def forecast_df():
    return STATE["forecast"]["df"] if STATE["forecast"] else None


def current_analysis():
    return viz.analyze(STATE["df"], STATE["freq"], forecast_df())


def need_data():
    flash("Please upload and preprocess a dataset first.", "error")
    return redirect(url_for("upload"))


def explain_metrics(m):
    """Plain-language reading of the evaluation metrics."""
    r2 = m["r2"]
    level = ("excellent" if r2 >= 0.9 else "good" if r2 >= 0.75 else "moderate" if r2 >= 0.5 else "weak")
    rel = m["rmse"] / m["mean_actual"] * 100 if m["mean_actual"] else 0
    gain = (1 - m["mae"] / m["naive_mae"]) * 100 if m["naive_mae"] else 0
    lines = [
        f"R² = {r2:.3f}: the model explains about {max(r2, 0) * 100:.1f}% of the variation in the unseen test data - a {level} fit.",
        f"MAE = {m['mae']:.3f}: on average a prediction is off by {m['mae']:.3f} units.",
        f"RMSE = {m['rmse']:.3f} ({rel:.1f}% of the average test consumption). RMSE is larger than MAE when there are some big errors.",
        f"MSE = {m['mse']:.4f} is the average squared error (RMSE squared).",
    ]
    if gain > 0:
        lines.append(f"Compared with a naive baseline (predict the previous value, MAE {m['naive_mae']:.3f}) the model's MAE is {gain:.1f}% lower.")
    else:
        lines.append(f"The naive baseline (predict the previous value) has MAE {m['naive_mae']:.3f}, which is not worse than the model - "
                     "more data or tuning may help. Multi-day forecasts also depend on recursive predictions, so errors can grow with the horizon.")
    return lines


@app.template_filter("num")
def num_filter(v, digits=2):
    if v is None:
        return "—"
    try:
        return f"{float(v):,.{digits}f}"
    except (TypeError, ValueError):
        return str(v)


@app.context_processor
def inject_globals():
    return {"g_has_data": has_data(), "g_has_model": STATE["bundle"] is not None,
            "g_dataset": STATE["filename"], "g_sample": STATE["is_sample"], "unit": STATE["unit"],
            "g_has_forecast": STATE["forecast"] is not None, "now_year": datetime.now().year}


# ------------------------------------------------------------------- routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    if not has_data():
        return need_data()
    a = current_analysis()
    charts = viz.build_charts(STATE["df"], STATE["freq"], STATE["bundle"], forecast_df(), a)
    return render_template("dashboard.html", a=a, charts=charts, bundle=STATE["bundle"],
                           forecast=STATE["forecast"], report=STATE["report"])


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        action = request.form.get("action")
        try:
            if action == "upload":
                f = request.files.get("file")
                if not f or not f.filename:
                    raise pp.DataError("Please choose a CSV file to upload.")
                if not f.filename.lower().endswith(".csv"):
                    raise pp.DataError("Invalid file type. Only .csv files are accepted.")
                name = secure_filename(f.filename) or "uploaded.csv"
                path = os.path.join(UPLOAD_DIR, name)
                f.save(path)
                load_dataset(path, name, is_sample=False)
                flash(f"'{name}' uploaded: {STATE['info']['rows']:,} rows, {STATE['info']['columns']} columns. "
                      "Check the selected columns below, then click 'Apply preprocessing'.", "success")
            elif action == "process":
                if STATE["raw"] is None:
                    raise pp.DataError("Upload a dataset first.")
                process_dataset(request.form.get("date_col"), request.form.get("value_col"),
                                request.form.get("time_col"), request.form.get("unit", "kWh").strip()[:12])
                r = STATE["report"]
                flash(f"Preprocessing finished: {r['final_rows']:,} clean {r['frequency'].lower()} records "
                      f"({r['start'][:10]} to {r['end'][:10]}). Next: train the model.", "success")
                if STATE["filename"] == SAMPLE_NAME:
                    try_load_saved_model()
            elif action == "sample":
                load_sample()
                flash("Sample dataset loaded (SAMPLE DATA - synthetic, for testing only).", "success")
        except pp.DataError as exc:
            flash(str(exc), "error")
        except Exception as exc:
            traceback.print_exc()
            flash(f"Unexpected error while processing the file: {exc}", "error")
        return redirect(url_for("upload"))
    return render_template("upload.html", info=STATE["info"], detected=STATE["detected"],
                           report=STATE["report"], filename=STATE["filename"],
                           clean=STATE["df"].head(8).reset_index().astype(str).to_dict("records") if has_data() else None)


@app.route("/analysis")
def analysis():
    if not has_data():
        return need_data()
    a = current_analysis()
    charts = viz.build_charts(STATE["df"], STATE["freq"], STATE["bundle"], forecast_df(), a)
    return render_template("analysis.html", a=a, charts=charts)


@app.route("/train", methods=["GET", "POST"])
def train():
    if request.method == "POST":
        if not has_data():
            return need_data()
        try:
            n_est = min(max(int(request.form.get("n_estimators", 100)), 10), 500)
            depth_txt = request.form.get("max_depth", "").strip()
            depth = min(max(int(depth_txt), 2), 50) if depth_txt else None
            test_size = float(request.form.get("test_size", 0.2))
            if not 0.1 <= test_size <= 0.4:
                raise ValueError
        except ValueError:
            flash("Invalid training parameters. Use whole numbers for trees/depth and a test size of 10%-40%.", "error")
            return redirect(url_for("train"))
        try:
            bundle = mt.train_model(STATE["df"], STATE["freq"], STATE["filename"], n_est, depth, test_size)
            mt.save_bundle(bundle, MODEL_PATH)
            STATE["bundle"], STATE["forecast"] = bundle, None
            m = bundle["metrics"]
            flash(f"Model trained successfully (R² = {m['r2']:.3f}, RMSE = {m['rmse']:.3f}). The model was saved to model/trained_model.pkl.", "success")
        except pp.DataError as exc:
            flash(str(exc), "error")
        except Exception as exc:
            traceback.print_exc()
            flash(f"Training failed: {exc}", "error")
        return redirect(url_for("train"))
    cfg = mt.FREQ_CONFIG.get(STATE["freq"]) if has_data() else None
    features = None
    if has_data():
        feats = list(mt.build_features(STATE["df"].head(400), STATE["freq"]).columns)
        features = [(f, mt.describe_feature(f)) for f in feats]
    return render_template("train.html", bundle=STATE["bundle"], cfg=cfg, features=features,
                           n_records=len(STATE["df"]) if has_data() else 0)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if STATE["bundle"] is None:
            flash("Train the model before forecasting.", "error")
            return redirect(url_for("train"))
        try:
            bundle = STATE["bundle"]
            end_date = request.form.get("end_date", "").strip()
            days = int(request.form.get("days", 7) or 7)
            steps = pr.steps_for_request(bundle, days=days, end_date=end_date or None)
            result = pr.forecast(bundle, steps)
            per_day = mt.FREQ_CONFIG[bundle["freq"]]["per_day"]
            label = (f"Up to {end_date}" if end_date else f"Next {days} day(s)")
            STATE["forecast"] = {"df": result, "label": label, "made_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                 "steps": steps, "days": round(steps / per_day, 1)}
            f = result["predicted"]
            STATE["history"].insert(0, {
                "time": STATE["forecast"]["made_at"], "period": label, "steps": steps,
                "start": f"{result.index[0]:%Y-%m-%d %H:%M}", "end": f"{result.index[-1]:%Y-%m-%d %H:%M}",
                "avg": round(float(f.mean()), 3), "peak": round(float(f.max()), 3),
                "peak_time": f"{f.idxmax():%Y-%m-%d %H:%M}", "total": round(float(f.sum()), 2),
                "model": "RandomForestRegressor", "dataset": STATE["filename"],
                "results": [[f"{t:%Y-%m-%d %H:%M}", round(float(v), 3)] for t, v in f.items()]})
            STATE["history"] = STATE["history"][:50]
            save_history()
            flash(f"Forecast generated for {steps} {'hour' if bundle['freq'] == 'h' else 'day'}(s).", "success")
        except pp.DataError as exc:
            flash(str(exc), "error")
        except ValueError:
            flash("Please enter a valid number of days.", "error")
        except Exception as exc:
            traceback.print_exc()
            flash(f"Forecast failed: {exc}", "error")
        return redirect(url_for("predict"))

    ctx = {"bundle": STATE["bundle"], "forecast": STATE["forecast"], "rows": None, "daily": None,
           "charts": {}, "a": None, "cfg": None}
    if STATE["bundle"] is not None:
        ctx["cfg"] = mt.FREQ_CONFIG[STATE["bundle"]["freq"]]
        ctx["last_ts"] = f"{STATE['bundle']['history_tail'].index[-1]:%Y-%m-%d %H:%M}"
    if STATE["forecast"] is not None:
        a = current_analysis()
        fdf = STATE["forecast"]["df"]
        t = a["thresholds"]
        ctx["rows"] = [{"time": f"{i:%Y-%m-%d %H:%M}", "value": v, "level": viz.classify(v, t)}
                       for i, v in fdf["predicted"].items()]
        g = fdf["predicted"].groupby(fdf.index.date)
        d = pd.DataFrame({"avg": g.mean(), "total": g.sum(), "peak": g.max(), "n": g.count()})
        ctx["daily"] = [{"date": str(i), "avg": r.avg, "total": r.total, "peak": r.peak,
                         "level": viz.classify(r.avg, t)} for i, r in d.iterrows()]
        ctx["a"] = a
        ctx["charts"] = viz.build_charts(STATE["df"], STATE["freq"], STATE["bundle"], fdf, a)
    return render_template("prediction.html", **ctx)


@app.route("/results")
def results():
    if STATE["bundle"] is None:
        flash("Train the model first to see the evaluation results.", "error")
        return redirect(url_for("train"))
    a = current_analysis()
    charts = viz.build_charts(STATE["df"], STATE["freq"], STATE["bundle"], forecast_df(), a)
    return render_template("results.html", bundle=STATE["bundle"], charts=charts,
                           explanation=explain_metrics(STATE["bundle"]["metrics"]))


@app.route("/history", methods=["GET", "POST"])
def history():
    if request.method == "POST":
        STATE["history"] = []
        save_history()
        flash("Prediction history cleared.", "success")
        return redirect(url_for("history"))
    return render_template("history.html", history=STATE["history"])


def report_context():
    a = current_analysis()
    b = STATE["bundle"]
    return {"a": a, "bundle": b, "rep": STATE["report"], "info": STATE["info"], "filename": STATE["filename"],
            "forecast": STATE["forecast"], "explanation": explain_metrics(b["metrics"]) if b else None,
            "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


@app.route("/report")
def report():
    if not has_data():
        return need_data()
    fmt = request.args.get("download")
    if fmt == "csv":
        if not STATE["forecast"]:
            flash("Generate a forecast first to download it as CSV.", "error")
            return redirect(url_for("report"))
        out = STATE["forecast"]["df"].round(4).to_csv()
        return Response(out, mimetype="text/csv",
                        headers={"Content-Disposition": "attachment; filename=forecast.csv"})
    if fmt == "html":
        with open(os.path.join(BASE_DIR, "static", "css", "style.css"), encoding="utf-8") as fh:
            css = fh.read()
        html = render_template("report.html", standalone=True, css=css, **report_context())
        return Response(html, mimetype="text/html",
                        headers={"Content-Disposition": "attachment; filename=electricity_forecast_report.html"})
    return render_template("report.html", standalone=False, **report_context())


@app.route("/about")
def about():
    return render_template("about.html")


# ----------------------------------------------------------- error handlers
@app.errorhandler(404)
def not_found(_e):
    return render_template("error.html", code=404, message="The page you are looking for does not exist."), 404


@app.errorhandler(413)
def too_large(_e):
    flash("The file is too large (maximum 200 MB).", "error")
    return redirect(url_for("upload"))


@app.errorhandler(500)
def server_error(_e):
    return render_template("error.html", code=500, message="Something went wrong on the server. Please check the terminal for details."), 500


# --------------------------------------------------------------------- main
load_history()
try:
    load_sample()
except Exception as exc:  # the app still starts; the user can upload a dataset
    print(f"Sample dataset could not be loaded: {exc}")

if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)
