"""
TenderGuard — Demo Data Generator
====================================
Membuat data sintetis realistis untuk demonstrasi dashboard
tanpa memerlukan akses internet ke opentender.net / LKPP API.

Pola kolusi yang disimulasikan:
1. Ring A (PT KARYA MAJU GROUP) : 4 vendor bergantian menang di 40 tender
2. Ring B (CV BANGUN JAYA GROUP): 3 vendor pola cover-bidding
3. Vendor normal                 : ~50 vendor dengan pola kompetitif wajar

Jalankan: python generate_demo_data.py
"""

import os
import random
import sqlite3
import pickle
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import networkx as nx

random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────
# Konfigurasi Data Demo
# ─────────────────────────────────────────────

N_TENDER_NORMAL      = 800    # Tender kompetitif normal
N_TENDER_RING_A      = 120    # Tender yang dikuasai Ring A
N_TENDER_RING_B      = 80     # Tender yang dikuasai Ring B
N_VENDOR_NORMAL      = 60     # Vendor non-kolusi
INSTANSI_LIST = [
    "Dinas PU DKI Jakarta",
    "Dinas Cipta Karya Jakarta Selatan",
    "Dinas Bina Marga Jakarta Utara",
    "Dinas Perumahan Rakyat DKI Jakarta",
    "BPTJ DKI Jakarta",
    "Dinas SDA DKI Jakarta",
    "Satuan Kerja PUPR DKI",
    "Kementerian PUPR — Balai Besar",
]

# Vendor kolusi Ring A (bergantian menang)
RING_A_VENDORS = [
    "PT KARYA MAJU BERSAMA",
    "PT KARYA MAJU JAYA",
    "PT MAJU KARYA KONSTRUKSI",
    "CV KARYA BERSAMA MAJU",
]

# Vendor kolusi Ring B (cover bidding)
RING_B_VENDORS = [
    "CV BANGUN JAYA ABADI",
    "PT BANGUN JAYA KONSTRUKSI",
    "UD JAYA BANGUN PERKASA",
]

# ─────────────────────────────────────────────
# Generator Helpers
# ─────────────────────────────────────────────

def random_date(start_year: int = 2023, end_year: int = 2024) -> str:
    start = datetime(start_year, 1, 1)
    end   = datetime(end_year, 12, 31)
    delta = end - start
    return (start + timedelta(days=random.randint(0, delta.days))).strftime("%Y-%m-%d")


def random_vendor_names(n: int) -> list[str]:
    prefixes  = ["PT", "CV", "UD", "PT"]
    adjective = ["SEJAHTERA", "MANDIRI", "UNGGUL", "PRIMA", "UTAMA", "ABADI",
                 "SENTOSA", "PERDANA", "NUSANTARA", "MULIA", "SUKSES", "MAKMUR",
                 "PERKASA", "GEMILANG", "ANDALAN", "TERBAIK", "GLOBAL", "INTI"]
    noun      = ["KONSTRUKSI", "BANGUN", "CIPTA", "KARYA", "JAYA", "RAYA",
                 "GRAHA", "INDO", "MULTI", "TEKNIK", "ENGINERING", "PERSADA"]
    names = set()
    while len(names) < n:
        name = f"{random.choice(prefixes)} {random.choice(adjective)} {random.choice(noun)}"
        names.add(name)
    return list(names)


NORMAL_VENDORS = random_vendor_names(N_VENDOR_NORMAL)
ALL_VENDORS    = RING_A_VENDORS + RING_B_VENDORS + NORMAL_VENDORS


def make_tender_id(prefix: str, i: int) -> str:
    return f"{prefix}-2024-{i:05d}"


def random_hps() -> float:
    return round(random.uniform(500_000_000, 50_000_000_000), 0)


def add_noise_to_price(base: float, noise_pct: float = 0.15) -> float:
    return max(0, round(base * (1 + random.uniform(-noise_pct, noise_pct)), 0))


# ─────────────────────────────────────────────
# Generate Data
# ─────────────────────────────────────────────

def generate_normal_tenders() -> list[dict]:
    """Buat tender kompetitif normal dengan 3–8 peserta acak."""
    records = []
    for i in range(1, N_TENDER_NORMAL + 1):
        tid        = make_tender_id("NORMAL", i)
        instansi   = random.choice(INSTANSI_LIST)
        tanggal    = random_date()
        nilai_hps  = random_hps()
        n_peserta  = random.randint(3, 8)
        vendors    = random.sample(NORMAL_VENDORS, min(n_peserta, len(NORMAL_VENDORS)))
        winner_idx = random.randint(0, len(vendors) - 1)

        for rank, vendor in enumerate(vendors, 1):
            # Harga penawaran bervariasi, tidak terpola
            harga = add_noise_to_price(nilai_hps * random.uniform(0.75, 0.98), 0.08)
            records.append({
                "tender_id"       : tid,
                "judul"           : f"Pekerjaan Konstruksi {instansi} Paket {i}",
                "instansi"        : instansi,
                "tanggal"         : tanggal,
                "nama_asli"       : vendor,
                "vendor_id"       : vendor.upper().strip(),
                "nilai_penawaran" : harga,
                "nilai_hps"       : nilai_hps,
                "is_winner"       : rank == winner_idx + 1,
                "rank"            : rank,
                "sumber"          : "demo",
            })
    return records


def generate_ring_a_tenders() -> list[dict]:
    """
    Ring A: 4 vendor bergantian menang (rotasi kemenangan).
    Vendor yang tidak menang selalu ikut serta sebagai 'boneka'.
    """
    records = []
    winner_rotation = 0

    for i in range(1, N_TENDER_RING_A + 1):
        tid       = make_tender_id("RINGA", i)
        instansi  = random.choice(INSTANSI_LIST[:4])
        tanggal   = random_date()
        nilai_hps = random_hps()

        winner = RING_A_VENDORS[winner_rotation % len(RING_A_VENDORS)]
        winner_rotation += 1

        # Tambahkan juga beberapa vendor normal (decoy)
        decoys = random.sample(NORMAL_VENDORS, 2)
        peserta = RING_A_VENDORS + decoys
        random.shuffle(peserta)

        for rank, vendor in enumerate(peserta, 1):
            is_winner = (vendor == winner)
            # Harga penawaran ring: pemenang sedikit lebih rendah, yang lain agak lebih tinggi
            if is_winner:
                harga = nilai_hps * random.uniform(0.88, 0.93)
            elif vendor in RING_A_VENDORS:
                # Harga 'boneka' sedikit lebih tinggi dari pemenang, konsisten
                harga = nilai_hps * random.uniform(0.94, 0.97)
            else:
                harga = add_noise_to_price(nilai_hps * 0.9, 0.1)

            records.append({
                "tender_id"       : tid,
                "judul"           : f"Konstruksi Jalan & Jembatan Paket RING-A {i}",
                "instansi"        : instansi,
                "tanggal"         : tanggal,
                "nama_asli"       : vendor,
                "vendor_id"       : vendor.upper().strip(),
                "nilai_penawaran" : round(harga, 0),
                "nilai_hps"       : nilai_hps,
                "is_winner"       : is_winner,
                "rank"            : rank,
                "sumber"          : "demo",
            })
    return records


def generate_ring_b_tenders() -> list[dict]:
    """
    Ring B: 3 vendor pola cover-bidding.
    Vendor B dan C selalu ikut dan selalu kalah dari vendor A.
    """
    records = []

    for i in range(1, N_TENDER_RING_B + 1):
        tid       = make_tender_id("RINGB", i)
        instansi  = random.choice(INSTANSI_LIST[4:])
        tanggal   = random_date()
        nilai_hps = random_hps()

        # Vendor A selalu menang
        winner  = RING_B_VENDORS[0]
        decoys  = random.sample(NORMAL_VENDORS, 1)
        peserta = RING_B_VENDORS + decoys

        for rank, vendor in enumerate(peserta, 1):
            is_winner = (vendor == winner)
            if is_winner:
                harga = nilai_hps * random.uniform(0.85, 0.90)
            elif vendor in RING_B_VENDORS:
                # Cover bids: harga hampir sama satu sama lain, tapi lebih tinggi dari pemenang
                harga = nilai_hps * random.uniform(0.92, 0.95)
            else:
                harga = add_noise_to_price(nilai_hps * 0.93, 0.07)

            records.append({
                "tender_id"       : tid,
                "judul"           : f"Pembangunan Gedung Pemerintah Paket RING-B {i}",
                "instansi"        : instansi,
                "tanggal"         : tanggal,
                "nama_asli"       : vendor,
                "vendor_id"       : vendor.upper().strip(),
                "nilai_penawaran" : round(harga, 0),
                "nilai_hps"       : nilai_hps,
                "is_winner"       : is_winner,
                "rank"            : rank,
                "sumber"          : "demo",
            })
    return records


# ─────────────────────────────────────────────
# Build Graph & Compute Features
# ─────────────────────────────────────────────

def build_graphs(df: pd.DataFrame):
    """Build co-participation dan win-loss graph dari DataFrame."""
    from modules.graph_builder import (
        build_co_participation_graph,
        build_win_loss_graph,
        compute_node_features,
    )
    G  = build_co_participation_graph(df, min_co_participation=2)
    DG = build_win_loss_graph(df)
    nf = compute_node_features(G, DG, df)
    return G, DG, nf


# ─────────────────────────────────────────────
# Compute Detection & Scores
# ─────────────────────────────────────────────

def run_detection(G, DG, nf):
    from modules.detector import (
        detect_anomalous_vendors,
        detect_suspicious_communities,
        get_vendor_community_flags,
    )
    try:
        import community as community_louvain
        partition = community_louvain.best_partition(G, weight="weight")
    except Exception:
        partition = {}

    if_labels, if_scores, _, _ = detect_anomalous_vendors(nf)
    community_stats = detect_suspicious_communities(G, nf)
    community_flags = get_vendor_community_flags(partition, community_stats)

    return if_labels, if_scores, community_stats, community_flags, partition


def run_scoring(nf, if_scores, community_flags, DG, df):
    from modules.scorer import (
        compute_all_vendor_scores,
        compute_tender_risk_scores,
    )
    vendor_scores = compute_all_vendor_scores(nf, if_scores, community_flags, DG)
    tender_scores = compute_tender_risk_scores(df, vendor_scores)

    # Join info tender
    tender_info = df.drop_duplicates("tender_id").set_index("tender_id")[
        ["judul", "instansi", "tanggal", "nilai_hps", "sumber"]
    ]
    tender_scores = tender_scores.join(tender_info, how="left")

    return vendor_scores, tender_scores


# ─────────────────────────────────────────────
# Visualisasi
# ─────────────────────────────────────────────

def run_visualization(G, vendor_scores):
    from modules.visualizer import (
        visualize_vendor_network,
        plot_risk_distribution,
        plot_top_risky_vendors,
    )
    visualize_vendor_network(G, vendor_scores, output_path="data/vendor_network.html", max_nodes=300)
    plot_risk_distribution(vendor_scores)
    plot_top_risky_vendors(vendor_scores)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  TenderGuard — Demo Data Generator")
    print("=" * 60)

    # Buat folder
    for d in ["data/raw", "data/processed", "models", "logs"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # 1. Generate data
    print("\n[1/6] Membuat data sintetis...")
    records = []
    records.extend(generate_normal_tenders())
    records.extend(generate_ring_a_tenders())
    records.extend(generate_ring_b_tenders())
    df = pd.DataFrame(records)
    df["is_winner"] = df["is_winner"].astype(bool)
    df.to_csv("data/processed/peserta_clean.csv", index=False)
    print(f"      Total: {len(df)} baris, {df['tender_id'].nunique()} tender, "
          f"{df['vendor_id'].nunique()} vendor unik")

    # 2. Build graphs
    print("\n[2/6] Membangun graf...")
    G, DG, nf = build_graphs(df)
    nf.to_csv("data/processed/node_features.csv")
    with open("data/processed/graph_G.pkl", "wb") as f:
        pickle.dump(G, f)
    with open("data/processed/graph_DG.pkl", "wb") as f:
        pickle.dump(DG, f)
    print(f"      Co-participation: {G.number_of_nodes()} node, {G.number_of_edges()} edge")
    print(f"      Win-loss        : {DG.number_of_nodes()} node, {DG.number_of_edges()} edge")

    # 3. Anomaly detection
    print("\n[3/6] Menjalankan deteksi anomali...")
    if_labels, if_scores, community_stats, community_flags, partition = run_detection(G, DG, nf)
    detection_results = pd.DataFrame({
        "if_label"               : if_labels,
        "if_score"               : if_scores,
        "in_suspicious_community": community_flags,
        "gnn_suspicious_prob"    : 0.0,
    })
    detection_results.to_csv("data/processed/detection_results.csv")
    n_anomaly = (if_labels == -1).sum()
    n_suspicious_comm = sum(1 for c in community_stats.values() if c["suspicion_flag"])
    print(f"      Vendor anomali (IF)  : {n_anomaly}")
    print(f"      Komunitas mencurigakan: {n_suspicious_comm}")

    # 4. Risk scoring
    print("\n[4/6] Menghitung risk score...")
    vendor_scores, tender_scores = run_scoring(nf, if_scores, community_flags, DG, df)
    vendor_scores.to_csv("data/processed/vendor_risk_scores.csv")
    tender_scores.to_csv("data/processed/tender_risk_scores.csv")

    high_v  = (vendor_scores["risk_level"] == "HIGH").sum()
    high_t  = (tender_scores["tender_risk_level"] == "HIGH").sum()
    print(f"      Vendor HIGH risk   : {high_v}")
    print(f"      Tender HIGH risk   : {high_t}")

    # 5. Visualisasi
    print("\n[5/6] Membuat visualisasi...")
    try:
        run_visualization(G, vendor_scores)
        print("      vendor_network.html dibuat ✅")
    except Exception as e:
        print(f"      Visualisasi gagal (opsional): {e}")

    # 6. Verifikasi kolusi terdeteksi
    print("\n[6/6] Verifikasi deteksi kolusi...")
    ring_a_detected = vendor_scores.loc[
        [v for v in RING_A_VENDORS if v.upper() in vendor_scores.index],
        "risk_level"
    ] if any(v.upper() in vendor_scores.index for v in RING_A_VENDORS) else pd.Series()

    ring_b_detected = vendor_scores.loc[
        [v for v in RING_B_VENDORS if v.upper() in vendor_scores.index],
        "risk_level"
    ] if any(v.upper() in vendor_scores.index for v in RING_B_VENDORS) else pd.Series()

    print("\n  Ring A (Rotasi Kemenangan):")
    for v in RING_A_VENDORS:
        vid = v.upper()
        if vid in vendor_scores.index:
            row = vendor_scores.loc[vid]
            print(f"    {v}: score={row['risk_score']:.3f}  level={row['risk_level']}")

    print("\n  Ring B (Cover Bidding):")
    for v in RING_B_VENDORS:
        vid = v.upper()
        if vid in vendor_scores.index:
            row = vendor_scores.loc[vid]
            print(f"    {v}: score={row['risk_score']:.3f}  level={row['risk_level']}")

    print("\n" + "=" * 60)
    print("  ✅ Demo data berhasil dibuat!")
    print("\n  Jalankan dashboard dengan perintah:")
    print("  streamlit run dashboard/app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
