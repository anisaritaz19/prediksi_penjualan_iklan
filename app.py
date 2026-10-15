"""
app.py
======
Aplikasi web Flask: Prediksi Penjualan Berdasarkan Biaya Iklan TV
Model: Regresi Linear Sederhana (Scikit-Learn) yang disimpan dengan Joblib.

Jalankan lokal:   python app.py
Produksi (Render): gunicorn app:app
"""

import json
import math
import os
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

import train  # dipakai untuk melatih ulang otomatis bila model belum ada

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)

MAX_INPUT = 10_000  # batas atas input (ribu USD) agar tidak absurd


# ----------------------------------------------------------------------------
# Memuat model + metadata
# ----------------------------------------------------------------------------
def load_artifacts():
    """Muat model dan metadata. Latih ulang bila file belum ada / tidak cocok."""
    if not (train.MODEL_PATH.exists() and train.META_PATH.exists()):
        train.run()
    try:
        model = joblib.load(train.MODEL_PATH)
    except Exception:  # mis. versi scikit-learn berbeda
        train.run()
        model = joblib.load(train.MODEL_PATH)
    meta = json.loads(train.META_PATH.read_text(encoding="utf-8"))
    return model, meta


MODEL, META = load_artifacts()


# ----------------------------------------------------------------------------
# Filter Jinja untuk format angka
# ----------------------------------------------------------------------------
@app.template_filter("num")
def num_filter(value, digits=2):
    try:
        return f"{float(value):,.{int(digits)}f}"
    except (TypeError, ValueError):
        return value


# ----------------------------------------------------------------------------
# Logika prediksi
# ----------------------------------------------------------------------------
def predict_sales(raw_value):
    """Kembalikan (hasil, error). Salah satunya selalu None."""
    text = str(raw_value or "").strip().replace(",", ".")
    if not text:
        return None, "Biaya iklan TV wajib diisi."
    try:
        tv = float(text)
    except ValueError:
        return None, "Masukkan angka yang valid, contoh: 150 atau 150.5"
    if not math.isfinite(tv):
        return None, "Masukkan angka yang valid."
    if tv < 0:
        return None, "Biaya iklan tidak boleh negatif."
    if tv > MAX_INPUT:
        return None, f"Nilai terlalu besar. Maksimum {MAX_INPUT:,} (ribu USD)."

    X = pd.DataFrame({train.FEATURE: [tv]})
    pred = max(float(MODEL.predict(X)[0]), 0.0)
    rmse = META["metrics"]["rmse_test"]
    rng = META["range"]

    warning = None
    if tv < rng["tv_min"] or tv > rng["tv_max"]:
        warning = (
            f"Nilai {tv:g} berada di luar rentang data latih "
            f"({rng['tv_min']:g} - {rng['tv_max']:g}). Prediksi berupa "
            "ekstrapolasi sehingga kurang dapat diandalkan."
        )

    return {
        "tv": tv,
        "tv_usd": tv * 1000,
        "sales": pred,
        "sales_units": pred * 1000,
        "low": max(pred - rmse, 0.0),
        "high": pred + rmse,
        "warning": warning,
    }, None


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result, error, tv_value = None, None, ""
    if request.method == "POST":
        tv_value = request.form.get("tv", "").strip()
        result, error = predict_sales(tv_value)

    app_data = {
        "points": META["points"],
        "line": META["line"],
        "prediction": (
            {"x": result["tv"], "y": round(result["sales"], 3)} if result else None
        ),
    }
    return render_template(
        "index.html",
        meta=META,
        result=result,
        error=error,
        tv_value=tv_value,
        app_data=app_data,
        active="home",
    )


@app.route("/analisis")
def analisis():
    m = META["model"]
    interpretasi = {
        "per_1": m["coef"],
        "per_1_unit": m["coef"] * 1000,
        "per_10": m["coef"] * 10,
        "per_10_unit": m["coef"] * 10 * 1000,
        "intercept": m["intercept"],
        "intercept_unit": m["intercept"] * 1000,
    }
    return render_template(
        "analisis.html", meta=META, it=interpretasi, active="analisis"
    )


# ----------------------------------------------------------------------------
# API JSON (opsional, untuk demo / pengujian)
# ----------------------------------------------------------------------------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True) or {}
    result, error = predict_sales(payload.get("tv"))
    if error:
        return jsonify({"ok": False, "error": error}), 400
    return jsonify({"ok": True, **result})


@app.route("/api/metrics")
def api_metrics():
    return jsonify(
        {
            "model": META["model"],
            "metrics": META["metrics"],
            "split": META["split"],
            "range": META["range"],
        }
    )


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG") == "1")
