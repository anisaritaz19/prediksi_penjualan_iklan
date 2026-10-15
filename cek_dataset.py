"""
cek_dataset.py
==============
Bukti akses dan validasi data (format TP-02): memuat dataset dengan Pandas,
menampilkan 5 baris pertama dan nama-nama kolomnya.

Jalankan, lalu screenshot kode + outputnya untuk laporan:
    python cek_dataset.py
"""

import pandas as pd

df = pd.read_csv("data/advertising.csv")

print("--- 5 Baris Pertama ---")
print(df.head())

print("\n--- Nama Kolom ---")
print(df.columns)

print("\n--- Info Data ---")
df.info()

print("\n--- Cek Missing Values ---")
print(df.isnull().sum())
