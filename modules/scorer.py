"""
Modul 4 — Risk Scoring
========================
Setiap tender/vendor diberi skor risiko kolusi berdasarkan gabungan
sinyal dari Modul 3 (Isolation Forest, Community Detection, Heuristics).

Sesuai TRD Section 2.4
"""

import logging
from pathlib import Path

import networkx as nx
import pandas as pd
import yaml

logger = logging.getLogger("tenderguard.scorer")


# ─────────────────────────────────────────────
# 2.4.2 Heuristic Flags (TRD 2.4.2)
# ─────────────────────────────────────────────

def compute_heuristic_flags(
    vendor_id: str,
    node_features: pd.DataFrame,
    DG: nx.DiGraph,
    clustering_threshold: float = 0.7,
    always_loses_min_times: int = 5,
) -> dict:
    """
    Hitung flag heuristik untuk satu vendor.

    Flags yang diperiksa:
    - win_rate_above_3x_avg       : Win rate jauh di atas rata-rata global
    - always_loses_to_same_vendor : Selalu kalah terhadap 1 vendor tertentu
    - high_clustering_coefficient : Sangat terhubung dalam kluster kecil
    - bid_price_too_close         : Standar deviasi harga penawaran terlalu kecil

    Returns:
        dict {flag_name: bool}
    """
    if vendor_id not in node_features.index:
        return {
            "win_rate_above_3x_avg"       : False,
            "always_loses_to_same_vendor" : False,
            "high_clustering_coefficient" : False,
            "bid_price_too_close"         : False,
        }

    row = node_features.loc[vendor_id]
    global_avg_win_rate = node_features["win_rate"].mean() if "win_rate" in node_features.columns else 0

    # Flag 1: Win rate jauh di atas rata-rata
    win_rate_above = bool(row.get("win_rate", 0) > (global_avg_win_rate * 3))

    # Flag 2: Selalu kalah terhadap 1 vendor tertentu
    always_loses = False
    if DG.has_node(vendor_id):
        preds = list(DG.predecessors(vendor_id))
        for pred in preds:
            if (
                DG[pred][vendor_id].get("weight", 0) >= always_loses_min_times
                and row.get("win_count", 0) == 0
            ):
                always_loses = True
                break

    # Flag 3: Clustering coefficient tinggi
    high_clustering = bool(row.get("clustering_coef", 0) > clustering_threshold)

    # Flag 4: Standar deviasi harga penawaran sangat kecil (harga terlalu mirip)
    avg_bid_ratio = row.get("avg_bid_ratio", 0)
    bid_std       = row.get("bid_std", 0)
    bid_too_close = bool(
        bid_std < (avg_bid_ratio * 0.01) and avg_bid_ratio > 0 and bid_std > 0
    )

    return {
        "win_rate_above_3x_avg"       : win_rate_above,
        "always_loses_to_same_vendor" : always_loses,
        "high_clustering_coefficient" : high_clustering,
        "bid_price_too_close"         : bid_too_close,
    }


# ─────────────────────────────────────────────
# 2.4.1 Risk Score Formula (TRD 2.4.1)
# ─────────────────────────────────────────────

def compute_risk_score(
    isolation_score: float,
    community_flag: bool,
    heuristic_flags: dict,
    weights: dict | None = None,
) -> float:
    """
    Agregasi skor risiko dari berbagai sinyal.

    Args:
        isolation_score  : Raw score dari Isolation Forest (lebih negatif = lebih anomali)
        community_flag   : True jika berada di komunitas mencurigakan
        heuristic_flags  : Output dari compute_heuristic_flags
        weights          : Dict bobot per komponen (default: IF=0.35, comm=0.30, heur=0.35)

    Returns:
        float 0.0–1.0 (semakin tinggi = semakin mencurigakan)
    """
    if weights is None:
        weights = {
            "isolation_forest" : 0.35,
            "community"        : 0.30,
            "heuristic"        : 0.35,
        }

    # Isolation Forest: inversi karena output negatif = anomali
    # score_samples range biasanya -0.5 to 0 untuk anomali
    if_score = max(0.0, min(1.0, (-isolation_score + 0.5) * 2))

    # Community flag
    comm_score = 1.0 if community_flag else 0.0

    # Heuristic scoring
    heuristic_score = sum([
        0.30 if heuristic_flags.get("win_rate_above_3x_avg")       else 0,
        0.30 if heuristic_flags.get("always_loses_to_same_vendor")  else 0,
        0.20 if heuristic_flags.get("bid_price_too_close")          else 0,
        0.20 if heuristic_flags.get("high_clustering_coefficient")  else 0,
    ])
    heuristic_score = min(1.0, heuristic_score)

    # Weighted aggregation
    risk_score = (
        weights["isolation_forest"] * if_score +
        weights["community"]        * comm_score +
        weights["heuristic"]        * heuristic_score
    )

    return round(min(1.0, risk_score), 3)


def classify_risk(score: float) -> str:
    """
    Klasifikasikan skor risiko ke dalam kategori.

    Returns:
        "HIGH" (≥0.7) | "MEDIUM" (≥0.4) | "LOW" (<0.4)
    """
    if score >= 0.70:
        return "HIGH"
    elif score >= 0.40:
        return "MEDIUM"
    else:
        return "LOW"


# ─────────────────────────────────────────────
# Bulk Scoring per Vendor
# ─────────────────────────────────────────────

def compute_all_vendor_scores(
    node_features: pd.DataFrame,
    if_scores: pd.Series,
    community_flags: pd.Series,
    DG: nx.DiGraph,
    weights: dict | None = None,
) -> pd.DataFrame:
    """
    Hitung risk score untuk semua vendor sekaligus.

    Args:
        node_features   : DataFrame fitur per vendor
        if_scores       : Series raw IF scores
        community_flags : Series boolean komunitas mencurigakan
        DG              : Win-loss directed graph
        weights         : Bobot per komponen

    Returns:
        pd.DataFrame dengan kolom:
        vendor_id, risk_score, risk_level, if_score, community_flag,
        + semua flag heuristik
    """
    records = []
    vendors = node_features.index.tolist()

    logger.info(f"Menghitung risk score untuk {len(vendors)} vendor...")

    for vendor in vendors:
        heuristic_flags = compute_heuristic_flags(vendor, node_features, DG)

        iso_score = float(if_scores.get(vendor, 0))
        comm_flag = bool(community_flags.get(vendor, False))

        score = compute_risk_score(
            isolation_score=iso_score,
            community_flag=comm_flag,
            heuristic_flags=heuristic_flags,
            weights=weights,
        )
        level = classify_risk(score)

        records.append({
            "vendor_id"                  : vendor,
            "risk_score"                 : score,
            "risk_level"                 : level,
            "if_score_raw"               : iso_score,
            "in_suspicious_community"    : comm_flag,
            "win_rate"                   : node_features.loc[vendor].get("win_rate", 0),
            "total_tenders"              : node_features.loc[vendor].get("total_tenders", 0),
            "win_count"                  : node_features.loc[vendor].get("win_count", 0),
            **{f"flag_{k}": v for k, v in heuristic_flags.items()},
        })

    df_scores = pd.DataFrame(records).set_index("vendor_id")
    df_scores = df_scores.sort_values("risk_score", ascending=False)

    # Statistik
    high_risk   = (df_scores["risk_level"] == "HIGH").sum()
    medium_risk = (df_scores["risk_level"] == "MEDIUM").sum()
    low_risk    = (df_scores["risk_level"] == "LOW").sum()

    logger.info(
        f"Risk scoring selesai — HIGH: {high_risk}, MEDIUM: {medium_risk}, LOW: {low_risk}"
    )

    return df_scores


# ─────────────────────────────────────────────
# Risk Score per Tender
# ─────────────────────────────────────────────

def compute_tender_risk_scores(
    df_peserta: pd.DataFrame,
    vendor_scores: pd.DataFrame,
) -> pd.DataFrame:
    """
    Agregasi risk score per tender berdasarkan risk score vendor peserta.

    Skor tender = max risk score peserta * 0.7 + avg risk score peserta * 0.3

    Args:
        df_peserta    : DataFrame [tender_id, vendor_id]
        vendor_scores : Output dari compute_all_vendor_scores

    Returns:
        pd.DataFrame dengan kolom:
        tender_id, tender_risk_score, tender_risk_level,
        vendor_count, max_vendor_risk, avg_vendor_risk,
        high_risk_vendors, suspicious_vendors
    """
    logger.info("Menghitung risk score per tender...")

    records = []
    for tender_id, group in df_peserta.groupby("tender_id"):
        vendors_in_tender = group["vendor_id"].dropna().unique().tolist()
        vendor_scores_in  = vendor_scores.loc[
            [v for v in vendors_in_tender if v in vendor_scores.index],
            "risk_score"
        ]

        if vendor_scores_in.empty:
            continue

        max_risk  = vendor_scores_in.max()
        avg_risk  = vendor_scores_in.mean()
        n_high    = (vendor_scores.loc[vendor_scores_in.index, "risk_level"] == "HIGH").sum()

        tender_score = round(max_risk * 0.7 + avg_risk * 0.3, 3)
        tender_level = classify_risk(tender_score)

        # Identifikasi vendor mencurigakan
        suspicious = [
            v for v in vendor_scores_in.index
            if vendor_scores.loc[v, "risk_level"] in ("HIGH", "MEDIUM")
        ]

        records.append({
            "tender_id"          : tender_id,
            "tender_risk_score"  : tender_score,
            "tender_risk_level"  : tender_level,
            "vendor_count"       : len(vendors_in_tender),
            "max_vendor_risk"    : round(max_risk, 3),
            "avg_vendor_risk"    : round(avg_risk, 3),
            "high_risk_vendors"  : int(n_high),
            "suspicious_vendors" : ", ".join(suspicious[:5]),  # Maks 5 vendor pertama
        })

    df_tender_scores = (
        pd.DataFrame(records)
        .set_index("tender_id")
        .sort_values("tender_risk_score", ascending=False)
    )

    return df_tender_scores


# ─────────────────────────────────────────────
# Pipeline Risk Scoring
# ─────────────────────────────────────────────

class RiskScoringPipeline:
    """Orkestrasi risk scoring untuk vendor dan tender."""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        self.scoring_cfg = self.cfg.get("scoring", {})

    def run(
        self,
        node_features: pd.DataFrame,
        if_scores: pd.Series,
        community_flags: pd.Series,
        DG: nx.DiGraph,
        df_peserta: pd.DataFrame,
    ) -> dict:
        """
        Jalankan pipeline risk scoring end-to-end.

        Returns:
            dict berisi: vendor_scores, tender_scores
        """
        weights = self.scoring_cfg.get("weights", None)

        # Skor per vendor
        vendor_scores = compute_all_vendor_scores(
            node_features, if_scores, community_flags, DG, weights=weights
        )

        # Skor per tender
        tender_scores = compute_tender_risk_scores(df_peserta, vendor_scores)

        # Merge info tender (judul, instansi, tanggal) jika tersedia
        if "judul" in df_peserta.columns:
            tender_info = df_peserta.drop_duplicates("tender_id").set_index("tender_id")[
                [c for c in ["judul", "instansi", "tanggal", "nilai_hps", "sumber"]
                 if c in df_peserta.columns]
            ]
            tender_scores = tender_scores.join(tender_info, how="left")

        # Simpan output
        vendor_scores.to_csv("data/processed/vendor_risk_scores.csv")
        tender_scores.to_csv("data/processed/tender_risk_scores.csv")

        logger.info("Risk scoring selesai.")
        logger.info(f"  Tender HIGH risk   : {(tender_scores['tender_risk_level'] == 'HIGH').sum()}")
        logger.info(f"  Tender MEDIUM risk : {(tender_scores['tender_risk_level'] == 'MEDIUM').sum()}")

        return {
            "vendor_scores" : vendor_scores,
            "tender_scores" : tender_scores,
        }


# ─────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from modules.graph_builder import GraphBuilderPipeline

    parser = argparse.ArgumentParser(description="TenderGuard — Modul 4: Risk Scoring")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    # Load data
    graphs        = GraphBuilderPipeline.load_graphs()
    DG            = graphs["DG"]
    node_features = graphs["node_features"]

    detection     = pd.read_csv("data/processed/detection_results.csv", index_col=0)
    df_peserta    = pd.read_csv("data/processed/peserta_clean.csv")

    if_scores       = detection["if_score"]
    community_flags = detection["in_suspicious_community"].astype(bool)

    pipeline = RiskScoringPipeline(config_path=args.config)
    results  = pipeline.run(
        node_features=node_features,
        if_scores=if_scores,
        community_flags=community_flags,
        DG=DG,
        df_peserta=df_peserta,
    )

    vs = results["vendor_scores"]
    ts = results["tender_scores"]

    print(f"\n✅ Risk scoring selesai.")
    print(f"   Vendor HIGH risk   : {(vs['risk_level'] == 'HIGH').sum()}")
    print(f"   Vendor MEDIUM risk : {(vs['risk_level'] == 'MEDIUM').sum()}")
    print(f"   Tender HIGH risk   : {(ts['tender_risk_level'] == 'HIGH').sum()}")
    print(f"\nTop 5 Vendor Paling Mencurigakan:")
    print(vs.head(5)[["risk_score", "risk_level", "win_rate", "total_tenders"]])
