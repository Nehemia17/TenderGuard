# Technical Requirements Document (TRD)
## Graph Mining untuk Deteksi Kolusi Tender di Data Pengadaan LPSE Nasional

---

**Versi:** 1.0  
**Tanggal:** Mei 2026  
**Status:** Draft  
**Referensi PRD:** PRD_GraphMining_Kolusi_Tender v1.0

---

## 1. Arsitektur Sistem

### 1.1 Overview

Sistem dibangun sebagai pipeline modular berbasis Python dengan 5 komponen utama yang dapat dijalankan secara sekuensial maupun independen.

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                              │
│   opentender.net  │  LKPP API  │  inaproc.id                   │
└────────────┬────────────────────────────────────────────────────┘
             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   MODULE 1: DATA INGESTION                      │
│   Scraper  │  API Client  │  Normalizer  │  SQLite Storage      │
└────────────┬────────────────────────────────────────────────────┘
             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  MODULE 2: GRAPH CONSTRUCTION                   │
│   Co-participation Graph  │  Win-Loss Graph  │  Feature Eng.    │
└────────────┬────────────────────────────────────────────────────┘
             ▼
┌─────────────────────────────────────────────────────────────────┐
│                MODULE 3: ANOMALY DETECTION                      │
│   Isolation Forest  │  Community Detection  │  GNN              │
└────────────┬────────────────────────────────────────────────────┘
             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   MODULE 4: RISK SCORING                        │
│   Score Aggregator  │  Threshold Engine  │  Report Generator    │
└────────────┬────────────────────────────────────────────────────┘
             ▼
┌─────────────────────────────────────────────────────────────────┐
│                 MODULE 5: VISUALIZATION LAYER                   │
│   Network Map  │  Streamlit Dashboard  │  CSV/PDF Export        │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Tech Stack

| Layer | Teknologi | Versi |
|---|---|---|
| Bahasa | Python | ≥ 3.10 |
| Scraping | Scrapy / requests + BeautifulSoup4 | Latest stable |
| Data Processing | pandas, numpy | ≥ 2.0 |
| Graph Library | NetworkX | ≥ 3.2 |
| ML (tabular) | scikit-learn (Isolation Forest) | ≥ 1.4 |
| ML (graph) | PyTorch Geometric (PyG) | ≥ 2.5 |
| Deep Learning | PyTorch | ≥ 2.2 |
| Visualisasi Graf | Pyvis / Gephi (export) | Latest |
| Dashboard | Streamlit | ≥ 1.30 |
| Database | SQLite / DuckDB | Latest |
| String Matching | rapidfuzz | ≥ 3.0 |
| Environment | Docker (opsional) | Latest |

---

## 2. Spesifikasi Modul

### 2.1 Modul 1 — Data Ingestion

#### 2.1.1 Komponen Scraper

**Target 1: opentender.net**

```python
# Endpoint utama
BASE_URL = "https://opentender.net/api/search"

# Parameter request
params = {
    "country": "ID",
    "region": "<kode_wilayah>",
    "year": "<tahun>",
    "cpvs": "45*",  # Kode CPV untuk konstruksi
    "page": 0,
    "pageSize": 100
}

# Field yang diambil
FIELDS_REQUIRED = [
    "id",                    # ID tender
    "title",                 # Judul tender
    "buyers.name",           # Nama instansi
    "lots.bids.bidders.name",# Nama peserta
    "lots.bids.isWinning",   # Status menang/kalah
    "lots.bids.price.netAmount",  # Nilai penawaran
    "date"                   # Tanggal
]
```

**Target 2: LKPP API**

```
GET https://lpse.lkpp.go.id/eproc4/dt/lelang
    ?idLpse=<kode_lpse>
    &tahunAnggaran=<tahun>
    &draw=1&start=0&length=100

# Rate limit: max 10 req/detik (estimasi)
# Autentikasi: tidak diperlukan (data publik)
```

**Target 3: inaproc.id (pelengkap)**

```
GET https://inaproc.id/api/contract
    ?tahun=<tahun>
    &satker=<kode_satker>
    &format=json
```

#### 2.1.2 Schema Data Output (SQLite)

```sql
-- Tabel utama tender
CREATE TABLE tender (
    tender_id       TEXT PRIMARY KEY,
    judul           TEXT,
    instansi        TEXT,
    tanggal         DATE,
    nilai_hps       REAL,          -- Harga Perkiraan Sendiri
    sumber          TEXT           -- 'opentender' | 'lkpp' | 'inaproc'
);

-- Tabel peserta per tender
CREATE TABLE peserta (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    tender_id       TEXT REFERENCES tender(tender_id),
    vendor_id       TEXT,          -- ID unik setelah normalisasi
    nama_asli       TEXT,          -- Nama mentah dari sumber
    nilai_penawaran REAL,
    is_winner       BOOLEAN,
    rank            INTEGER        -- Peringkat penawaran
);

-- Tabel master vendor (setelah deduplication)
CREATE TABLE vendor (
    vendor_id       TEXT PRIMARY KEY,
    nama_canonical  TEXT,
    nama_aliases    TEXT,          -- JSON array nama-nama alternatif
    alamat          TEXT,
    npwp            TEXT
);
```

#### 2.1.3 Normalisasi Nama Vendor

```python
from rapidfuzz import fuzz, process

def normalize_vendor_name(name: str) -> str:
    """Normalisasi dasar nama perusahaan."""
    name = name.upper().strip()
    # Hapus prefiks badan hukum yang tidak relevan
    prefixes = ["PT.", "PT ", "CV.", "CV ", "UD.", "UD ", "KOPERASI "]
    for p in prefixes:
        if name.startswith(p):
            name = name[len(p):].strip()
    return name

def deduplicate_vendors(names: list[str], threshold: int = 85) -> dict:
    """
    Kelompokkan nama vendor yang mirip menggunakan fuzzy matching.
    threshold: skor kemiripan minimum (0-100) untuk dianggap sama.
    """
    canonical_map = {}  # nama_asli -> nama_canonical
    canonical_list = []
    
    for name in names:
        norm = normalize_vendor_name(name)
        match = process.extractOne(norm, canonical_list, scorer=fuzz.ratio)
        if match and match[1] >= threshold:
            canonical_map[name] = match[0]
        else:
            canonical_list.append(norm)
            canonical_map[name] = norm
    
    return canonical_map
```

---

### 2.2 Modul 2 — Graph Construction

#### 2.2.1 Co-participation Graph

Graf utama yang merepresentasikan hubungan keikutsertaan bersama.

```python
import networkx as nx

def build_co_participation_graph(df_peserta: pd.DataFrame) -> nx.Graph:
    """
    df_peserta: DataFrame dengan kolom [tender_id, vendor_id, is_winner, nilai_penawaran]
    
    Edge (A, B) ada jika A dan B mengikuti tender yang sama.
    Edge weight = jumlah tender yang sama-sama diikuti.
    """
    G = nx.Graph()
    
    for tender_id, group in df_peserta.groupby("tender_id"):
        vendors = group["vendor_id"].tolist()
        winner = group[group["is_winner"]]["vendor_id"].values
        
        for i in range(len(vendors)):
            for j in range(i + 1, len(vendors)):
                v1, v2 = vendors[i], vendors[j]
                if G.has_edge(v1, v2):
                    G[v1][v2]["weight"] += 1
                    G[v1][v2]["tenders"].append(tender_id)
                else:
                    G.add_edge(v1, v2, weight=1, tenders=[tender_id])
    
    return G
```

#### 2.2.2 Win-Loss Relationship Graph

Graf diarahkan untuk mendeteksi pola cover bidding.

```python
def build_win_loss_graph(df_peserta: pd.DataFrame) -> nx.DiGraph:
    """
    Edge terarah: A → B berarti A menang ketika B kalah dalam tender yang sama.
    Edge weight = berapa kali ini terjadi.
    
    Sinyal kolusi: jika A → B memiliki weight sangat tinggi dan B tidak pernah menang.
    """
    DG = nx.DiGraph()
    
    for tender_id, group in df_peserta.groupby("tender_id"):
        winners = group[group["is_winner"]]["vendor_id"].tolist()
        losers = group[~group["is_winner"]]["vendor_id"].tolist()
        
        for winner in winners:
            for loser in losers:
                if DG.has_edge(winner, loser):
                    DG[winner][loser]["weight"] += 1
                else:
                    DG.add_edge(winner, loser, weight=1)
    
    return DG
```

#### 2.2.3 Node Feature Engineering

```python
def compute_node_features(
    G: nx.Graph,
    DG: nx.DiGraph,
    df_peserta: pd.DataFrame
) -> pd.DataFrame:
    """Hitung fitur statistik per vendor sebagai node attributes."""
    
    features = {}
    for vendor in G.nodes():
        vendor_data = df_peserta[df_peserta["vendor_id"] == vendor]
        total = len(vendor_data)
        wins = vendor_data["is_winner"].sum()
        
        features[vendor] = {
            # Statistik dasar
            "total_tenders"     : total,
            "win_count"         : wins,
            "win_rate"          : wins / total if total > 0 else 0,
            
            # Fitur graf (co-participation)
            "degree"            : G.degree(vendor),
            "weighted_degree"   : G.degree(vendor, weight="weight"),
            "clustering_coef"   : nx.clustering(G, vendor),
            "betweenness"       : nx.betweenness_centrality(G).get(vendor, 0),
            
            # Fitur win-loss
            "times_beat_others" : sum(DG[vendor][v]["weight"] for v in DG.successors(vendor)),
            "times_lost_to"     : sum(DG[v][vendor]["weight"] for v in DG.predecessors(vendor)),
            
            # Fitur penawaran harga
            "avg_bid_ratio"     : (
                vendor_data["nilai_penawaran"].mean() /
                vendor_data["nilai_hps"].mean()
                if vendor_data["nilai_hps"].mean() > 0 else 0
            ),
            "bid_std"           : vendor_data["nilai_penawaran"].std()
        }
    
    return pd.DataFrame(features).T
```

---

### 2.3 Modul 3 — Anomaly Detection

#### 2.3.1 Isolation Forest (Baseline)

```python
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def detect_anomalous_vendors(node_features: pd.DataFrame) -> pd.Series:
    """
    Deteksi vendor anomali berdasarkan fitur statistik.
    Return: Series dengan anomaly score per vendor (-1 = anomali, 1 = normal)
    """
    feature_cols = [
        "win_rate", "degree", "weighted_degree",
        "clustering_coef", "betweenness",
        "times_beat_others", "times_lost_to",
        "avg_bid_ratio", "bid_std"
    ]
    
    X = node_features[feature_cols].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    clf = IsolationForest(
        n_estimators=200,
        contamination=0.05,   # Estimasi 5% vendor mencurigakan
        random_state=42
    )
    
    scores = clf.fit_predict(X_scaled)
    anomaly_scores = clf.score_samples(X_scaled)  # Raw anomaly score
    
    return pd.Series(anomaly_scores, index=node_features.index)
```

#### 2.3.2 Community Detection (Louvain)

```python
import community as community_louvain  # python-louvain

def detect_suspicious_communities(G: nx.Graph, node_features: pd.DataFrame) -> dict:
    """
    Temukan komunitas vendor menggunakan Louvain.
    Komunitas mencurigakan: rata-rata win_rate anggota sangat tinggi.
    """
    partition = community_louvain.best_partition(G, weight="weight")
    
    # Evaluasi tiap komunitas
    community_stats = {}
    for comm_id in set(partition.values()):
        members = [v for v, c in partition.items() if c == comm_id]
        if len(members) < 3:
            continue
        
        stats = node_features.loc[
            [m for m in members if m in node_features.index]
        ]
        
        community_stats[comm_id] = {
            "members"       : members,
            "size"          : len(members),
            "avg_win_rate"  : stats["win_rate"].mean(),
            "total_wins"    : stats["win_count"].sum(),
            # Komunitas mencurigakan jika avg win rate anggota > 3x rata-rata global
            "suspicion_flag": stats["win_rate"].mean() > (
                node_features["win_rate"].mean() * 3
            )
        }
    
    return community_stats
```

#### 2.3.3 Graph Neural Network (GNN)

Diimplementasikan menggunakan PyTorch Geometric sebagai enhancement opsional.

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data

class TenderCollusionGNN(torch.nn.Module):
    """
    Graph Convolutional Network untuk klasifikasi node (vendor) mencurigakan.
    Arsitektur: 2-layer GCN + 1 output layer.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 2):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = torch.nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        return self.classifier(x)


def prepare_graph_data(
    G: nx.Graph,
    node_features: pd.DataFrame
) -> Data:
    """Konversi NetworkX graph ke format PyTorch Geometric."""
    
    # Node index mapping
    node_list = list(G.nodes())
    node_idx = {v: i for i, v in enumerate(node_list)}
    
    # Edge index (COO format)
    edges = [(node_idx[u], node_idx[v]) for u, v in G.edges()]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # Edge weights
    edge_weights = torch.tensor(
        [G[u][v]["weight"] for u, v in G.edges()], dtype=torch.float
    )
    
    # Node feature matrix
    feature_cols = [
        "win_rate", "degree", "clustering_coef",
        "betweenness", "times_beat_others", "avg_bid_ratio"
    ]
    X = torch.tensor(
        node_features.reindex(node_list)[feature_cols].fillna(0).values,
        dtype=torch.float
    )
    
    return Data(x=X, edge_index=edge_index, edge_attr=edge_weights)
```

**Strategi Label untuk Training GNN:**
- Label positif (mencurigakan): vendor yang sudah ditandai sebagai anomali oleh Isolation Forest (semi-supervised)
- Label negatif: vendor dengan win_rate sangat rendah dan degree rendah (baseline normal)
- Opsi: manual labeling pada subset kecil berdasarkan pengetahuan domain

---

### 2.4 Modul 4 — Risk Scoring

#### 2.4.1 Formula Risk Score

```python
def compute_risk_score(
    vendor_id: str,
    isolation_score: float,   # dari Isolation Forest (0-1, lebih tinggi = lebih normal)
    community_flag: bool,     # True jika berada di komunitas mencurigakan
    heuristic_flags: dict     # dict sinyal heuristik
) -> float:
    """
    Agregasi skor risiko dari berbagai sinyal.
    Output: float 0.0 – 1.0 (semakin tinggi = semakin mencurigakan)
    """
    
    # Isolation Forest: inversi karena output negatif = anomali
    if_score = max(0, min(1, (- isolation_score + 0.5) * 2))
    
    # Community flag
    comm_score = 1.0 if community_flag else 0.0
    
    # Heuristic scoring
    heuristic_score = sum([
        0.3 if heuristic_flags.get("win_rate_above_3x_avg") else 0,
        0.3 if heuristic_flags.get("always_loses_to_same_vendor") else 0,
        0.2 if heuristic_flags.get("bid_price_too_close") else 0,
        0.2 if heuristic_flags.get("high_clustering_coefficient") else 0,
    ])
    
    # Weighted aggregation
    WEIGHTS = {
        "isolation_forest"  : 0.35,
        "community"         : 0.30,
        "heuristic"         : 0.35
    }
    
    risk_score = (
        WEIGHTS["isolation_forest"]  * if_score +
        WEIGHTS["community"]         * comm_score +
        WEIGHTS["heuristic"]         * heuristic_score
    )
    
    return round(min(1.0, risk_score), 3)


def classify_risk(score: float) -> str:
    if score >= 0.7:
        return "HIGH"
    elif score >= 0.4:
        return "MEDIUM"
    else:
        return "LOW"
```

#### 2.4.2 Heuristic Flags

```python
def compute_heuristic_flags(
    vendor_id: str,
    node_features: pd.DataFrame,
    DG: nx.DiGraph
) -> dict:
    row = node_features.loc[vendor_id]
    global_avg_win_rate = node_features["win_rate"].mean()
    
    # Flag: win rate jauh di atas rata-rata
    win_rate_above = row["win_rate"] > (global_avg_win_rate * 3)
    
    # Flag: selalu kalah terhadap 1 vendor tertentu
    preds = list(DG.predecessors(vendor_id))
    always_loses = any(
        DG[pred][vendor_id]["weight"] >= 5 and 
        row["win_count"] == 0
        for pred in preds
    ) if preds else False
    
    # Flag: koefisien kluster tinggi (sangat terhubung dalam kluster kecil)
    high_clustering = row["clustering_coef"] > 0.7
    
    # Flag: harga penawaran terlalu mirip antar sesama komunitas
    # (diimplementasikan terpisah di analisis per tender)
    bid_too_close = row["bid_std"] < (row.get("avg_bid_ratio", 0) * 0.01)
    
    return {
        "win_rate_above_3x_avg"       : bool(win_rate_above),
        "always_loses_to_same_vendor" : bool(always_loses),
        "high_clustering_coefficient" : bool(high_clustering),
        "bid_price_too_close"         : bool(bid_too_close)
    }
```

---

### 2.5 Modul 5 — Visualisasi & Dashboard

#### 2.5.1 Visualisasi Graf (Pyvis)

```python
from pyvis.network import Network

def visualize_vendor_network(
    G: nx.Graph,
    risk_scores: dict,     # {vendor_id: risk_score}
    output_path: str = "vendor_network.html"
):
    """Render graf vendor dengan warna berdasarkan risk score."""
    
    net = Network(height="700px", width="100%", bgcolor="#1a1a2e", font_color="white")
    
    COLOR_MAP = {"HIGH": "#e74c3c", "MEDIUM": "#f39c12", "LOW": "#2ecc71"}
    
    for node in G.nodes():
        score = risk_scores.get(node, 0)
        level = classify_risk(score)
        net.add_node(
            node,
            label=node[:20],
            color=COLOR_MAP[level],
            size=10 + score * 30,
            title=f"Vendor: {node}<br>Risk Score: {score}<br>Level: {level}"
        )
    
    for u, v, data in G.edges(data=True):
        net.add_edge(u, v, width=min(data["weight"] * 0.5, 5))
    
    net.save_graph(output_path)
```

#### 2.5.2 Streamlit Dashboard

```python
# dashboard/app.py

import streamlit as st
import pandas as pd
from modules import data_loader, graph_builder, scorer, visualizer

st.set_page_config(page_title="TENDER GUARD", layout="wide")

st.title("🔍 TENDER GUARD — Sistem Deteksi Kolusi Pengadaan")

# Sidebar filters
with st.sidebar:
    instansi = st.selectbox("Pilih Instansi", options=load_instansi_list())
    tahun = st.slider("Tahun", 2020, 2025, (2023, 2024))
    min_risk = st.slider("Minimum Risk Score", 0.0, 1.0, 0.5)

# Main content
tab1, tab2, tab3 = st.tabs(["📋 Tender Berisiko", "🕸️ Peta Jaringan", "📊 Statistik"])

with tab1:
    df_risky = load_risky_tenders(instansi, tahun, min_risk)
    st.dataframe(df_risky, use_container_width=True)
    st.download_button("Export CSV", df_risky.to_csv(), "tender_berisiko.csv")

with tab2:
    st.components.v1.html(open("vendor_network.html").read(), height=700)

with tab3:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Tender Dianalisis", "1,247")
    col2.metric("Tender Risiko Tinggi", "89", "+12 dari bulan lalu")
    col3.metric("Kluster Mencurigakan", "7")
```

---

## 3. Struktur Direktori Proyek

```
tender-guard/
├── data/
│   ├── raw/                    # Data mentah dari scraping
│   ├── processed/              # Data setelah normalisasi
│   └── tender.db               # SQLite database
├── modules/
│   ├── __init__.py
│   ├── scraper.py              # Modul 1: Scraping
│   ├── graph_builder.py        # Modul 2: Graph construction
│   ├── detector.py             # Modul 3: Anomaly detection
│   ├── scorer.py               # Modul 4: Risk scoring
│   └── visualizer.py           # Modul 5: Visualisasi
├── models/
│   ├── isolation_forest.pkl    # Model tersimpan
│   └── gnn_checkpoint.pt       # GNN checkpoint
├── dashboard/
│   └── app.py                  # Streamlit app
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory Data Analysis
│   ├── 02_graph_analysis.ipynb # Analisis graf
│   └── 03_model_eval.ipynb     # Evaluasi model
├── tests/
│   └── test_*.py               # Unit tests
├── requirements.txt
├── config.yaml                 # Konfigurasi (wilayah, threshold, dll)
└── README.md
```

---

## 4. Konfigurasi Sistem

```yaml
# config.yaml

data:
  target_region: "DKI Jakarta"
  target_years: [2023, 2024]
  cpv_codes: ["45"]            # Kode CPV konstruksi
  min_tender_value: 100000000  # Minimum 100 juta rupiah

graph:
  min_co_participation: 2     # Minimum berapa kali untuk buat edge
  max_nodes: 5000             # Batas node untuk performance

detection:
  isolation_forest:
    contamination: 0.05
    n_estimators: 200
  community:
    resolution: 1.0           # Resolusi Louvain
  gnn:
    hidden_dim: 64
    epochs: 100
    lr: 0.001

scoring:
  weights:
    isolation_forest: 0.35
    community: 0.30
    heuristic: 0.35
  thresholds:
    high: 0.70
    medium: 0.40

output:
  dashboard_port: 8501
  export_format: ["csv", "html"]
```

---

## 5. Persyaratan Non-Fungsional

| Aspek | Persyaratan |
|---|---|
| **Performance** | Pipeline penuh selesai < 30 menit untuk 1.000 tender |
| **Skalabilitas** | Arsitektur mendukung penambahan provinsi tanpa refactor besar |
| **Reproduksibilitas** | `requirements.txt` + `config.yaml` cukup untuk menjalankan ulang |
| **Portabilitas** | Dapat dijalankan di Windows / Linux / Mac tanpa konfigurasi tambahan |
| **Akurasi** | Precision ≥ 70% pada labeled test set (berdasarkan Isolation Forest) |

---

## 6. Setup & Instalasi

```bash
# 1. Clone repository
git clone https://github.com/<team>/tender-guard.git
cd tender-guard

# 2. Buat virtual environment
python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Jalankan pipeline scraping
python -m modules.scraper --config config.yaml

# 5. Bangun graf dan deteksi
python -m modules.graph_builder
python -m modules.detector
python -m modules.scorer

# 6. Jalankan dashboard
streamlit run dashboard/app.py
```

---

## 7. Dependencies Utama

```txt
# requirements.txt

# Data
pandas>=2.0
numpy>=1.26
requests>=2.31
beautifulsoup4>=4.12
scrapy>=2.11
rapidfuzz>=3.0
duckdb>=0.10

# Graph
networkx>=3.2
python-louvain>=0.16
pyvis>=0.3

# ML
scikit-learn>=1.4
torch>=2.2
torch-geometric>=2.5

# Visualization & Dashboard
streamlit>=1.30
plotly>=5.18
matplotlib>=3.8
seaborn>=0.13

# Utils
pyyaml>=6.0
tqdm>=4.66
pytest>=8.0
```

---

## 8. Evaluasi Model

### 8.1 Metrik

Karena data berlabel sangat terbatas (unsupervised dominant), evaluasi menggunakan kombinasi:

| Metrik | Metode |
|---|---|
| Silhouette Score | Kualitas kluster komunitas |
| Precision@K | Pada subset berlabel manual |
| Anomaly Recall | Berapa % kasus kolusi historis tertangkap |
| Expert Review | Validasi dengan analis domain (KPK / ICW) |

### 8.2 Baseline Perbandingan

| Sistem | Metode | Keterbatasan |
|---|---|---|
| ICW Opentender | Indikator statistik sederhana | Tidak memodelkan relasi |
| Paper ResearchGate 2016 | Rule-based | Tidak menggunakan ML |
| **TenderGuard (ours)** | Graph ML + GNN | Memodelkan struktur jaringan |

---

## 9. Roadmap Teknis

| Fase | Sprint | Task Teknis |
|---|---|---|
| **Fase 1** | Sprint 1–2 | Scraper, normalisasi, SQLite schema |
| **Fase 2** | Sprint 3–4 | Graph construction, feature engineering |
| **Fase 3** | Sprint 5–6 | Isolation Forest, community detection |
| **Fase 4** | Sprint 7 | GNN training (opsional di MVP) |
| **Fase 5** | Sprint 8 | Risk scoring, Streamlit dashboard |
| **Fase 6** | Sprint 9 | Testing, validasi, dokumentasi |

---

*Dokumen ini adalah living document dan akan diperbarui seiring iterasi pengembangan. Setiap perubahan arsitektur mayor harus direfleksikan di sini sebelum implementasi.*
