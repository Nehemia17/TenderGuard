"""
Modul 2 — Graph Construction
==============================
Pembangunan graf dari data tender untuk merepresentasikan
relasi antar vendor: co-participation graph, win-loss graph,
dan node feature engineering.

Sesuai TRD Section 2.2
"""

import logging
import sqlite3
from pathlib import Path
from typing import Optional

import networkx as nx
import pandas as pd
import numpy as np
import pickle
import yaml

logger = logging.getLogger("tenderguard.graph_builder")


# ─────────────────────────────────────────────
# 2.2.1 Co-participation Graph
# ─────────────────────────────────────────────

def build_co_participation_graph(
    df_peserta: pd.DataFrame,
    min_co_participation: int = 2,
) -> nx.Graph:
    """
    Bangun co-participation graph dari data peserta tender.

    Node  : Perusahaan/vendor
    Edge  : Dua vendor pernah mengikuti tender yang sama
    Weight: Frekuensi co-participation (semakin sering, semakin besar)

    Args:
        df_peserta            : DataFrame [tender_id, vendor_id, is_winner, nilai_penawaran]
        min_co_participation  : Minimum edge weight agar edge disimpan

    Returns:
        nx.Graph dengan edge attributes: weight, tenders
    """
    G = nx.Graph()

    # Tambahkan semua node terlebih dahulu
    all_vendors = df_peserta["vendor_id"].unique()
    G.add_nodes_from(all_vendors)

    logger.info(f"Membangun co-participation graph dari {df_peserta['tender_id'].nunique()} tender...")

    for tender_id, group in df_peserta.groupby("tender_id"):
        vendors = group["vendor_id"].dropna().unique().tolist()
        if len(vendors) < 2:
            continue

        for i in range(len(vendors)):
            for j in range(i + 1, len(vendors)):
                v1, v2 = vendors[i], vendors[j]
                if G.has_edge(v1, v2):
                    G[v1][v2]["weight"] += 1
                    G[v1][v2]["tenders"].append(tender_id)
                else:
                    G.add_edge(v1, v2, weight=1, tenders=[tender_id])

    # Filter edge dengan weight < min_co_participation
    edges_to_remove = [
        (u, v) for u, v, d in G.edges(data=True)
        if d["weight"] < min_co_participation
    ]
    G.remove_edges_from(edges_to_remove)

    logger.info(
        f"Co-participation graph: {G.number_of_nodes()} node, "
        f"{G.number_of_edges()} edge (min_co_participation={min_co_participation})"
    )
    return G


# ─────────────────────────────────────────────
# 2.2.2 Win-Loss Relationship Graph
# ─────────────────────────────────────────────

def build_win_loss_graph(df_peserta: pd.DataFrame) -> nx.DiGraph:
    """
    Bangun directed win-loss graph untuk deteksi cover bidding.

    Edge terarah: A → B berarti A menang ketika B kalah dalam tender yang sama.
    Edge weight = berapa kali pola ini terjadi.

    Sinyal kolusi: jika A → B sangat sering dan B tidak pernah menang.

    Args:
        df_peserta: DataFrame [tender_id, vendor_id, is_winner]

    Returns:
        nx.DiGraph
    """
    DG = nx.DiGraph()

    logger.info("Membangun win-loss directed graph...")

    for tender_id, group in df_peserta.groupby("tender_id"):
        winners = group[group["is_winner"] == True]["vendor_id"].tolist()
        losers  = group[group["is_winner"] == False]["vendor_id"].tolist()

        for winner in winners:
            for loser in losers:
                if winner == loser:
                    continue
                if DG.has_edge(winner, loser):
                    DG[winner][loser]["weight"] += 1
                    DG[winner][loser]["tenders"].append(tender_id)
                else:
                    DG.add_edge(winner, loser, weight=1, tenders=[tender_id])

    logger.info(
        f"Win-loss graph: {DG.number_of_nodes()} node, "
        f"{DG.number_of_edges()} edge"
    )
    return DG


# ─────────────────────────────────────────────
# 2.2.3 Node Feature Engineering
# ─────────────────────────────────────────────

def compute_node_features(
    G: nx.Graph,
    DG: nx.DiGraph,
    df_peserta: pd.DataFrame,
) -> pd.DataFrame:
    """
    Hitung fitur statistik per vendor sebagai node attributes.

    Fitur yang dihitung:
    - Statistik dasar    : total_tenders, win_count, win_rate
    - Fitur graf         : degree, weighted_degree, clustering_coef, betweenness
    - Fitur win-loss     : times_beat_others, times_lost_to
    - Fitur harga        : avg_bid_ratio, bid_std

    Returns:
        pd.DataFrame dengan vendor_id sebagai index
    """
    logger.info("Menghitung fitur node vendor...")

    # Pre-compute centralities untuk efisiensi
    betweenness = nx.betweenness_centrality(G, weight="weight") if G.number_of_nodes() > 0 else {}
    clustering   = nx.clustering(G, weight="weight") if G.number_of_nodes() > 0 else {}

    features = {}
    for vendor in G.nodes():
        vendor_data = df_peserta[df_peserta["vendor_id"] == vendor]
        total = len(vendor_data)
        wins  = vendor_data["is_winner"].sum()

        # Hitung nilai penawaran vs HPS
        has_hps = "nilai_hps" in vendor_data.columns
        avg_hps = vendor_data["nilai_hps"].mean() if has_hps else 0

        avg_bid_ratio = (
            vendor_data["nilai_penawaran"].mean() / avg_hps
            if has_hps and avg_hps > 0 else 0
        )

        features[vendor] = {
            # Statistik dasar
            "total_tenders"     : total,
            "win_count"         : int(wins),
            "win_rate"          : wins / total if total > 0 else 0,

            # Fitur graf (co-participation)
            "degree"            : G.degree(vendor),
            "weighted_degree"   : G.degree(vendor, weight="weight"),
            "clustering_coef"   : clustering.get(vendor, 0),
            "betweenness"       : betweenness.get(vendor, 0),

            # Fitur win-loss
            "times_beat_others" : sum(
                DG[vendor][v]["weight"] for v in DG.successors(vendor)
            ) if DG.has_node(vendor) else 0,
            "times_lost_to"     : sum(
                DG[v][vendor]["weight"] for v in DG.predecessors(vendor)
            ) if DG.has_node(vendor) else 0,

            # Fitur penawaran harga
            "avg_bid_ratio"     : avg_bid_ratio,
            "bid_std"           : vendor_data["nilai_penawaran"].std()
                                    if len(vendor_data) > 1 else 0,
        }

    df_features = pd.DataFrame(features).T
    df_features.index.name = "vendor_id"
    logger.info(f"Fitur node dihitung untuk {len(df_features)} vendor")
    return df_features


# ─────────────────────────────────────────────
# Pipeline Graph Builder
# ─────────────────────────────────────────────

class GraphBuilderPipeline:
    """
    Orkestrasi pembangunan graf dari data peserta tender.
    """

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        self.graph_cfg = self.cfg.get("graph", {})
        Path("data/processed").mkdir(parents=True, exist_ok=True)

    def run(self, df_peserta: pd.DataFrame) -> dict:
        """
        Jalankan pipeline graph building.

        Args:
            df_peserta: DataFrame bersih dari Modul 1

        Returns:
            dict berisi: G (co-participation), DG (win-loss), node_features
        """
        min_co = self.graph_cfg.get("min_co_participation", 2)
        max_nodes = self.graph_cfg.get("max_nodes", 5000)

        # Batasi jumlah vendor jika terlalu besar
        if df_peserta["vendor_id"].nunique() > max_nodes:
            logger.warning(
                f"Jumlah vendor ({df_peserta['vendor_id'].nunique()}) melebihi max_nodes={max_nodes}. "
                "Mengambil vendor dengan aktivitas terbanyak."
            )
            top_vendors = (
                df_peserta.groupby("vendor_id").size()
                .nlargest(max_nodes).index.tolist()
            )
            df_peserta = df_peserta[df_peserta["vendor_id"].isin(top_vendors)]

        # Bangun kedua graf
        G  = build_co_participation_graph(df_peserta, min_co_participation=min_co)
        DG = build_win_loss_graph(df_peserta)

        # Hitung fitur node
        node_features = compute_node_features(G, DG, df_peserta)

        # Simpan ke file
        nx.write_gml(G,  "data/processed/co_participation.gml")
        nx.write_gml(DG, "data/processed/win_loss.gml")
        node_features.to_csv("data/processed/node_features.csv")

        with open("data/processed/graph_G.pkl", "wb") as f:
            pickle.dump(G, f)
        with open("data/processed/graph_DG.pkl", "wb") as f:
            pickle.dump(DG, f)

        logger.info("Graph building selesai. File disimpan di data/processed/")

        return {
            "G"             : G,
            "DG"            : DG,
            "node_features" : node_features,
        }

    @staticmethod
    def load_graphs() -> dict:
        """Load graph yang sudah tersimpan."""
        with open("data/processed/graph_G.pkl", "rb") as f:
            G = pickle.load(f)
        with open("data/processed/graph_DG.pkl", "rb") as f:
            DG = pickle.load(f)
        node_features = pd.read_csv("data/processed/node_features.csv", index_col="vendor_id")
        return {"G": G, "DG": DG, "node_features": node_features}


# ─────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TenderGuard — Modul 2: Graph Construction")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data",   default="data/processed/peserta_clean.csv", help="Path CSV peserta")
    args = parser.parse_args()

    df_peserta = pd.read_csv(args.data)
    df_peserta["is_winner"] = df_peserta["is_winner"].astype(bool)

    pipeline = GraphBuilderPipeline(config_path=args.config)
    result   = pipeline.run(df_peserta)

    G  = result["G"]
    DG = result["DG"]
    nf = result["node_features"]

    print(f"\n✅ Graph construction selesai.")
    print(f"   Co-participation graph : {G.number_of_nodes()} node, {G.number_of_edges()} edge")
    print(f"   Win-loss graph         : {DG.number_of_nodes()} node, {DG.number_of_edges()} edge")
    print(f"   Node features shape    : {nf.shape}")
