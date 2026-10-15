"""
train.py
========
Melatih model Regresi Linear Sederhana:  Sales = b0 + b1 * TV

Alur kerja (sesuai TA-2):
1. Memuat dataset  (data/advertising.csv)
2. Eksplorasi + pembersihan data (missing value, pengecekan outlier)
3. Membagi data latih / uji
4. Melatih LinearRegression (Scikit-Learn)
5. Evaluasi: R-squared, RMSE (+ MAE)
6. Menyimpan model dengan Joblib + metadata (JSON) + gambar scatterplot

Jalankan:  python train.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # backend tanpa GUI (aman untuk server / Render)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "advertising.csv"
MODEL_PATH = BASE_DIR / "model" / "model_regresi_tv.joblib"
META_PATH = BASE_DIR / "model" / "metadata.json"
PLOT_PATH = BASE_DIR / "static" / "img" / "scatter_tv_sales.png"

FEATURE = "TV"
TARGET = "Sales"
TEST_SIZE = 0.2
RANDOM_STATE = 42


# ----------------------------------------------------------------------------
# 1. Memuat data
# ----------------------------------------------------------------------------
def _find_column(df: pd.DataFrame, name: str) -> str:
    """Cari kolom tanpa peduli huruf besar/kecil atau spasi."""
    for col in df.columns:
        if str(col).strip().lower() == name.lower():
            return col
    raise KeyError(
        f"Kolom '{name}' tidak ditemukan. Kolom yang ada: {list(df.columns)}"
    )


def load_raw() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan di {DATA_PATH}.\n"
            "Jalankan dulu:  python download_dataset.py\n"
            "atau salin file CSV Advertising secara manual ke folder data/ "
            "dengan nama advertising.csv"
        )
    df = pd.read_csv(DATA_PATH)
    # Buang kolom indeks bawaan CSV (mis. 'Unnamed: 0') bila ada
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, [c != "" and not c.startswith("Unnamed") for c in df.columns]]
    return df


# ----------------------------------------------------------------------------
# Fungsi bantu
# ----------------------------------------------------------------------------
def _iqr_outliers(series: pd.Series) -> int:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())


def _save_scatter(df, X_train, y_train, X_test, y_test, b0, b1):
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=150)
    ax.scatter(X_train[FEATURE], y_train, s=28, alpha=0.7, color="#6366f1",
               label="Data latih")
    ax.scatter(X_test[FEATURE], y_test, s=34, alpha=0.9, color="#f59e0b",
               edgecolor="white", linewidth=0.6, label="Data uji")
    xs = np.linspace(0, df[FEATURE].max() * 1.05, 100)
    ax.plot(xs, b0 + b1 * xs, color="#ef4444", linewidth=2.2,
            label=f"Garis regresi: y = {b0:.2f} + {b1:.4f}x")
    ax.set_xlabel("Biaya Iklan TV (ribu USD)")
    ax.set_ylabel("Penjualan (ribu unit)")
    ax.set_title("Scatterplot TV vs Sales dan Garis Regresi Linear")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(PLOT_PATH)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 2-6. Pipeline utama
# ----------------------------------------------------------------------------
def run() -> dict:
    raw = load_raw()
    tv_col = _find_column(raw, FEATURE)
    sales_col = _find_column(raw, TARGET)

    # Bukti akses data (untuk TP-02: 5 baris pertama + nama kolom)
    head_records = raw.head(5).round(2).to_dict(orient="records")
    all_columns = list(raw.columns)

    # --- Pembersihan data ---
    df = raw[[tv_col, sales_col]].rename(columns={tv_col: FEATURE, sales_col: TARGET})
    df = df.apply(pd.to_numeric, errors="coerce")
    rows_raw = len(df)
    n_missing = int(df.isna().sum().sum())
    df = df.dropna().reset_index(drop=True)
    rows_clean = len(df)
    if rows_clean < 10:
        raise ValueError("Data terlalu sedikit setelah pembersihan.")

    out_tv = _iqr_outliers(df[FEATURE])
    out_sales = _iqr_outliers(df[TARGET])
    corr = float(df[FEATURE].corr(df[TARGET]))

    desc = df.describe().round(2)
    describe = {
        col: {stat: float(desc.loc[stat, col]) for stat in desc.index}
        for col in desc.columns
    }

    # --- Split data ---
    X = df[[FEATURE]]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # --- Latih model ---
    model = LinearRegression()
    model.fit(X_train, y_train)
    b0 = float(model.intercept_)
    b1 = float(model.coef_[0])

    # --- Evaluasi ---
    pred_test = model.predict(X_test)
    pred_train = model.predict(X_train)
    metrics = {
        "r2_test": float(r2_score(y_test, pred_test)),
        "rmse_test": float(np.sqrt(mean_squared_error(y_test, pred_test))),
        "mae_test": float(mean_absolute_error(y_test, pred_test)),
        "r2_train": float(r2_score(y_train, pred_train)),
        "rmse_train": float(np.sqrt(mean_squared_error(y_train, pred_train))),
    }

    # --- Simpan model (Joblib) ---
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    # --- Data untuk grafik di web ---
    points = [
        {"x": round(float(a), 2), "y": round(float(b), 2), "split": "train"}
        for a, b in zip(X_train[FEATURE], y_train)
    ] + [
        {"x": round(float(a), 2), "y": round(float(b), 2), "split": "test"}
        for a, b in zip(X_test[FEATURE], y_test)
    ]
    x_max = float(df[FEATURE].max())
    line = [
        {"x": round(x, 2), "y": round(b0 + b1 * x, 3)}
        for x in (0.0, x_max * 1.05)
    ]

    meta = {
        "trained_at": datetime.now().strftime("%d %B %Y %H:%M"),
        "sklearn_version": sklearn.__version__,
        "dataset": {
            "source": "Kaggle: tawfikelmetwally/advertising-dataset",
            "columns": all_columns,
            "head": head_records,
            "rows_raw": rows_raw,
            "rows_clean": rows_clean,
            "missing_values": n_missing,
            "outliers_tv": out_tv,
            "outliers_sales": out_sales,
            "correlation": round(corr, 4),
            "describe": describe,
        },
        "model": {
            "feature": FEATURE,
            "target": TARGET,
            "intercept": b0,
            "coef": b1,
        },
        "metrics": metrics,
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
        },
        "range": {
            "tv_min": float(df[FEATURE].min()),
            "tv_max": x_max,
            "sales_min": float(df[TARGET].min()),
            "sales_max": float(df[TARGET].max()),
            "sales_mean": float(df[TARGET].mean()),
        },
        "points": points,
        "line": line,
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    _save_scatter(df, X_train, y_train, X_test, y_test, b0, b1)
    return meta


def main():
    try:
        meta = run()
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"\n[ERROR] {exc}\n", file=sys.stderr)
        sys.exit(1)

    m = meta["metrics"]
    print("=" * 56)
    print(" MODEL REGRESI LINEAR SEDERHANA: Sales = b0 + b1 * TV")
    print("=" * 56)
    print(f" Jumlah data         : {meta['dataset']['rows_clean']}")
    print(f" Korelasi TV-Sales   : {meta['dataset']['correlation']}")
    print(f" Intercept (b0)      : {meta['model']['intercept']:.4f}")
    print(f" Koefisien (b1)      : {meta['model']['coef']:.6f}")
    print(f" R-squared (uji)     : {m['r2_test']:.4f}")
    print(f" RMSE (uji)          : {m['rmse_test']:.4f}")
    print(f" MAE (uji)           : {m['mae_test']:.4f}")
    print(f" Model disimpan di   : {MODEL_PATH.relative_to(BASE_DIR)}")
    print("=" * 56)


if __name__ == "__main__":
    main()
