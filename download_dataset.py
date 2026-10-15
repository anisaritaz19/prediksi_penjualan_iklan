"""
download_dataset.py
===================
Mengunduh dataset Advertising dari Kaggle dengan kagglehub, lalu menyalin
file CSV-nya ke  data/advertising.csv  agar bisa dipakai train.py dan
ikut di-commit ke GitHub (supaya build di Render tidak butuh akses Kaggle).

Persiapan (sekali saja, hanya di komputer lokal):
    pip install kagglehub

Jalankan:
    python download_dataset.py
"""

import shutil
import sys
from pathlib import Path

import kagglehub

BASE_DIR = Path(__file__).resolve().parent
TARGET = BASE_DIR / "data" / "advertising.csv"


def main():
    # Download latest version
    path = kagglehub.dataset_download("tawfikelmetwally/advertising-dataset")
    print("Path to dataset files:", path)

    csv_files = sorted(Path(path).rglob("*.csv"))
    if not csv_files:
        print("[ERROR] Tidak ada file CSV di folder dataset.", file=sys.stderr)
        sys.exit(1)

    source = csv_files[0]
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(source, TARGET)
    print(f"File '{source.name}' disalin ke: {TARGET.relative_to(BASE_DIR)}")
    print("Selanjutnya jalankan:  python train.py")


if __name__ == "__main__":
    main()
