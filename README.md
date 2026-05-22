# 🔍 TenderGuard

**Sistem Deteksi Kolusi Tender Berbasis Graph Mining**  
Data Pengadaan LPSE Nasional · GEMASTIK 2026 — Kategori Pelayanan Publik

---

## 📋 Deskripsi

TenderGuard adalah pipeline otomatis untuk mendeteksi pola kolusi (bid-rigging) pada data pengadaan pemerintah Indonesia (LPSE/LKPP). Sistem memodelkan relasi antar vendor sebagai graf dan menerapkan Graph ML untuk mengidentifikasi kluster vendor mencurigakan.

### Pola Kolusi yang Dideteksi
- **Rotasi kemenangan** — sekelompok vendor bergantian menang secara terkoordinasi
- **Cover bidding** — vendor pendamping mengajukan penawaran yang dirancang kalah
- **Bid suppression** — penawaran harga terlalu mirip antar peserta dalam satu kluster
- **Network anomaly** — vendor dengan posisi struktural tidak wajar dalam jaringan

---

## 🏗️ Arsitektur

```
INPUT (opentender.net / LKPP API / inaproc.id)
    ↓
Module 1: Data Ingestion & Scraping  (modules/scraper.py)
    ↓
Module 2: Graph Construction          (modules/graph_builder.py)
    ↓
Module 3: Anomaly Detection           (modules/detector.py)
    ↓
Module 4: Risk Scoring                (modules/scorer.py)
    ↓
Module 5: Visualisasi & Dashboard     (modules/visualizer.py + dashboard/app.py)
```

---

## 🚀 Cara Penggunaan

### 1. Instalasi

```bash
# Clone repository
git clone https://github.com/<team>/tender-guard.git
cd tender-guard

# Buat virtual environment
python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Demo Cepat (Data Sintetis)

```bash
python generate_demo_data.py
streamlit run dashboard/app.py
```

### 3. Pipeline Penuh (Data Nyata)

```bash
# Konfigurasi target di config.yaml terlebih dahulu

# Step 1: Scraping data
python -m modules.scraper --config config.yaml --opentender

# Step 2: Bangun graf
python -m modules.graph_builder --config config.yaml

# Step 3: Deteksi anomali
python -m modules.detector --config config.yaml

# Step 4 (opsional, dengan GNN):
python -m modules.detector --config config.yaml --gnn

# Step 5: Hitung risk score
python -m modules.scorer --config config.yaml

# Step 6: Visualisasi
python -m modules.visualizer --config config.yaml

# Step 7: Jalankan dashboard
streamlit run dashboard/app.py
```

### 4. Unit Tests

```bash
pytest -v
```

---

## 📁 Struktur Direktori

```
tender-guard/
├── data/
│   ├── raw/                    # Data mentah dari scraping
│   ├── processed/              # Data bersih + hasil analisis
│   └── vendor_network.html     # Visualisasi interaktif
├── modules/
│   ├── __init__.py
│   ├── scraper.py              # Modul 1: Data Ingestion
│   ├── graph_builder.py        # Modul 2: Graph Construction
│   ├── detector.py             # Modul 3: Anomaly Detection
│   ├── scorer.py               # Modul 4: Risk Scoring
│   └── visualizer.py           # Modul 5: Visualisasi
├── models/
│   ├── isolation_forest.pkl    # Model tersimpan
│   └── gnn_checkpoint.pt       # GNN checkpoint (opsional)
├── dashboard/
│   └── app.py                  # Streamlit Dashboard
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_graph_analysis.ipynb
│   └── 03_model_eval.ipynb
├── tests/
│   ├── test_graph_builder.py
│   └── test_scorer.py
├── generate_demo_data.py       # Generator data sintetis
├── config.yaml                 # Konfigurasi sistem
├── requirements.txt
└── README.md
```

---

## ⚙️ Konfigurasi

Edit `config.yaml` untuk mengatur target wilayah, tahun, dan parameter deteksi:

```yaml
data:
  target_region: "DKI Jakarta"
  target_years: [2023, 2024]
  cpv_codes: ["45"]             # Kode CPV konstruksi
  min_tender_value: 100000000   # Minimum 100 juta rupiah

detection:
  isolation_forest:
    contamination: 0.05         # Estimasi 5% vendor mencurigakan
  community:
    min_community_size: 3

scoring:
  weights:
    isolation_forest: 0.35
    community: 0.30
    heuristic: 0.35
  thresholds:
    high: 0.70
    medium: 0.40
```

---

## 🧠 Metode Deteksi

| Metode | Kegunaan |
|---|---|
| **Isolation Forest** | Deteksi node/vendor anomali secara statistik |
| **Louvain Community Detection** | Identifikasi kluster vendor terlalu kohesif |
| **Graph Neural Network (GNN)** | Klasifikasi node berdasarkan struktur lokal (opsional) |
| **Heuristic Indicators** | Win-rate tidak wajar, cover bidding, clustering tinggi |

### Risk Score Formula

```
risk_score = 0.35 × IF_score + 0.30 × community_score + 0.35 × heuristic_score
```

| Level | Threshold |
|---|---|
| 🔴 HIGH   | score ≥ 0.70 |
| 🟡 MEDIUM | score ≥ 0.40 |
| 🟢 LOW    | score < 0.40 |

---

## 📊 Sumber Data

| Sumber | URL | Tipe |
|---|---|---|
| Opentender | opentender.net | REST API / CSV |
| LKPP API | lpse.lkpp.go.id/api | REST API |
| InaProc | inaproc.id | REST API |

---

## 📦 Tech Stack

- **Python ≥ 3.10**
- **NetworkX** — Graph construction & analysis
- **scikit-learn** — Isolation Forest
- **PyTorch Geometric** — GNN (opsional)
- **python-louvain** — Community detection
- **Pyvis / Plotly** — Visualisasi
- **Streamlit** — Dashboard
- **rapidfuzz** — Fuzzy string matching
- **SQLite / DuckDB** — Storage

---

## 📄 Lisensi

MIT License — Proyek penelitian terbuka untuk kepentingan antikorupsi Indonesia.

---

*TenderGuard dibangun untuk GEMASTIK 2026 — Kategori Pelayanan Publik*
