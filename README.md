# Prediksi Penjualan Berdasarkan Biaya Iklan TV

Aplikasi web **Regresi Linear Sederhana** untuk Tugas Akhir 2 (TA-2) mata praktikum Pembelajaran Mesin.
Pengguna memasukkan biaya iklan TV, lalu aplikasi menampilkan estimasi penjualan, grafik data, serta evaluasi model (R² dan RMSE).

- **Model:** `Sales = b0 + b1 × TV` (`LinearRegression` Scikit-Learn, disimpan dengan Joblib)
- **Dataset:** [Advertising Dataset (Kaggle)](https://www.kaggle.com/datasets/tawfikelmetwally/advertising-dataset), kolom yang dipakai `TV` (X) dan `Sales` (y)
- **Stack:** Python, Flask, HTML, CSS, Bootstrap 5, Chart.js, Pandas, NumPy, Matplotlib
- **Satuan data:** TV dalam *ribu USD*, Sales dalam *ribu unit*

---

## 1. Struktur Folder

```
ta2-prediksi-penjualan-iklan-tv/
├── app.py                  # Aplikasi Flask (route, validasi input, prediksi)
├── train.py                # Pipeline: load data -> EDA -> split -> train -> evaluasi -> simpan Joblib
├── download_dataset.py     # Unduh dataset dari Kaggle (kagglehub) -> data/advertising.csv
├── cek_dataset.py          # Bukti akses data TP-02 (head() + columns)
├── requirements.txt        # Dependensi aplikasi (dipakai Render)
├── render.yaml             # Blueprint deployment Render (opsional)
├── Procfile                # Perintah start (alternatif)
├── .python-version         # Versi Python untuk Render
├── .gitignore
├── data/
│   └── advertising.csv     # (dibuat oleh download_dataset.py, WAJIB ikut di-commit)
├── model/                  # (dibuat oleh train.py)
│   ├── model_regresi_tv.joblib
│   └── metadata.json       # parameter, metrik, data grafik
├── static/
│   ├── css/style.css
│   ├── js/app.js           # grafik Chart.js + interaksi form
│   └── img/scatter_tv_sales.png   # (dibuat oleh train.py)
└── templates/
    ├── base.html           # layout + navbar + footer
    ├── index.html          # input, hasil, grafik, evaluasi
    └── analisis.html       # dataset, EDA, interpretasi parameter, keterbatasan
```

---

## 2. Menjalankan di VS Code (Lokal)

### Persiapan
1. Install **Python 3.10+** (centang *Add Python to PATH*) dan **VS Code** dengan ekstensi **Python** (Microsoft).
2. Buka folder proyek: `File > Open Folder...` lalu pilih `ta2-prediksi-penjualan-iklan-tv`.
3. Buka terminal VS Code: `` Ctrl + ` ``.

### Langkah
```bash
# 1) Buat virtual environment
python -m venv .venv

# 2) Aktifkan
#    Windows (PowerShell):
.venv\Scripts\Activate.ps1
#    Windows (CMD):
.venv\Scripts\activate.bat
#    macOS / Linux:
source .venv/bin/activate

# 3) Install dependensi
pip install -r requirements.txt

# 4) Unduh dataset dari Kaggle (sekali saja)
pip install kagglehub
python download_dataset.py

# 5) (Opsional, untuk screenshot TP-02) cek akses data
python cek_dataset.py

# 6) Latih model -> membuat model/*.joblib, metadata.json, dan scatter_tv_sales.png
python train.py

# 7) Jalankan aplikasi
python app.py
```
Buka **http://127.0.0.1:5000** di browser.

> **Memilih interpreter:** `Ctrl+Shift+P` → *Python: Select Interpreter* → pilih yang berlokasi di `.venv`.
>
> **PowerShell menolak activate?** Jalankan sekali: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
>
> **Mode debug (auto-reload):** `set FLASK_DEBUG=1` (CMD) / `$env:FLASK_DEBUG=1` (PowerShell) / `export FLASK_DEBUG=1` (macOS/Linux), lalu `python app.py`.
>
> Jika `model/` belum ada saat `app.py` dijalankan, aplikasi akan melatih model otomatis (asal `data/advertising.csv` sudah ada).

### Alternatif: unduh manual
Jika `kagglehub` bermasalah, unduh CSV dari halaman Kaggle, lalu simpan sebagai `data/advertising.csv`. Kolom `TV` dan `Sales` harus ada.

---

## 3. Deployment ke Render

### A. Upload ke GitHub
```bash
git init
git add .
git commit -m "TA-2: prediksi penjualan iklan TV (regresi linear)"
git branch -M main
git remote add origin https://github.com/USERNAME/NAMA-REPO.git
git push -u origin main
```
Pastikan `data/advertising.csv` ikut ter-commit (cek dengan `git ls-files data`). Build di Render membutuhkannya untuk melatih model.

### B. Buat Web Service di Render
1. Login ke [render.com](https://render.com) dan hubungkan akun GitHub.
2. **New +** → **Web Service** → pilih repositori Anda.
3. Isi pengaturan:

| Kolom | Nilai |
|---|---|
| Language / Runtime | `Python 3` |
| Branch | `main` |
| Build Command | `pip install -r requirements.txt && python train.py` |
| Start Command | `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120` |
| Instance Type | `Free` |
| Environment Variable | `PYTHON_VERSION` = `3.11.9` |
| Health Check Path (opsional) | `/health` |

4. Klik **Create Web Service**. Setelah build selesai, URL berbentuk `https://nama-layanan.onrender.com`.

> Alternatif: pilih **New + → Blueprint**, Render akan membaca `render.yaml` secara otomatis.
>
> **Catatan paket Free:** layanan tidur setelah ±15 menit tanpa trafik, sehingga akses pertama bisa lambat (±30-60 detik). Buka URL sebelum presentasi/demo.
>
> **Update aplikasi:** cukup `git push`, Render akan deploy ulang otomatis.

---

## 4. Checklist Kesesuaian TA-2

| Ketentuan TA-2 | Lokasi di proyek |
|---|---|
| Regresi Linear **Sederhana** (1 X, 1 y) | `train.py` (X = `TV`, y = `Sales`) |
| Dataset publik dari Kaggle + justifikasi | `download_dataset.py`, halaman **Analisis** bagian 1 |
| Scatterplot validasi linearitas | Grafik di halaman utama + `static/img/scatter_tv_sales.png` |
| Pembersihan data (missing, outlier) | `train.py` (`dropna`, hitung outlier IQR), ringkasan di halaman Analisis |
| Split data latih/uji | `train_test_split(test_size=0.2, random_state=42)` |
| Evaluasi R² dan RMSE | Kartu metrik di halaman utama + tabel di halaman Analisis |
| Interpretasi intercept dan koefisien | Halaman Analisis bagian 3 (dihitung otomatis dari model) |
| Simpan model dengan Joblib | `model/model_regresi_tv.joblib` |
| Flask + Bootstrap, form sesuai X | `app.py`, `templates/index.html` |
| Deploy ke Render | Bagian 3 di atas |
| Repo GitHub, URL aplikasi, video YouTube | Lampirkan tautan di laporan |

### Saran isi laporan (bab justifikasi studi kasus & data)
- **Alasan studi kasus:** anggaran iklan TV adalah keputusan bisnis nyata, hubungannya dengan penjualan masuk akal secara sebab-akibat dan berpotensi linear.
- **Alasan dataset:** sumber publik (Kaggle), ±200 observasi, tanpa missing value, kolom `TV` dan `Sales` numerik.
- **Screenshot yang perlu diambil:** output `cek_dataset.py`, output `train.py` (metrik), halaman utama dengan hasil prediksi, halaman Analisis, dan halaman Render yang sudah live.
- Angka R², RMSE, intercept, dan koefisien di laporan **harus sama** dengan yang tampil di aplikasi (lihat `model/metadata.json`).

---

## 5. Endpoint

| Method | URL | Keterangan |
|---|---|---|
| GET/POST | `/` | Halaman prediksi |
| GET | `/analisis` | Analisis dataset dan model |
| POST | `/api/predict` | JSON `{"tv": 150}` → estimasi penjualan |
| GET | `/api/metrics` | Parameter dan metrik model |
| GET | `/health` | Health check |

Contoh:
```bash
curl -X POST http://127.0.0.1:5000/api/predict -H "Content-Type: application/json" -d '{"tv": 150}'
```

---

## 6. Troubleshooting

| Masalah | Solusi |
|---|---|
| `Dataset tidak ditemukan di .../data/advertising.csv` | Jalankan `python download_dataset.py` atau salin CSV manual ke `data/advertising.csv` |
| `KeyError: Kolom 'TV' tidak ditemukan` | Periksa nama kolom CSV dengan `python cek_dataset.py` |
| `ModuleNotFoundError` | Aktifkan `.venv`, lalu `pip install -r requirements.txt` |
| Build Render gagal di `train.py` | Pastikan `data/advertising.csv` ada di GitHub (`git ls-files data`) |
| Peringatan versi scikit-learn saat load model | Hapus folder `model/` lalu `python train.py`, atau biarkan app melatih ulang otomatis |
| Port 5000 dipakai | `set PORT=5001` (Windows) / `export PORT=5001`, lalu `python app.py` |
| Grafik tidak muncul | Perangkat harus online (Bootstrap dan Chart.js dimuat dari CDN) |

---

*Catatan: model hanya memakai satu variabel (TV). Hasil prediksi di luar rentang data latih merupakan ekstrapolasi dan kurang dapat diandalkan.*
