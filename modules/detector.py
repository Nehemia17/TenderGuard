"""
Modul 3 — Anomaly Detection
=============================
Penerapan algoritma ML untuk mendeteksi pola abnormal dalam graf vendor.

Metode yang diimplementasikan:
1. Isolation Forest  — Deteksi node anomali secara statistik
2. Louvain Community Detection — Identifikasi kluster terlalu kohesif
3. GNN (Graph Neural Network) — Klasifikasi node opsional (PyTorch Geometric)

Sesuai TRD Section 2.3
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import networkx as nx
import yaml
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

try:
    import community as community_louvain  # python-louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False
    logging.warning("python-louvain tidak tersedia. Community detection dinonaktifkan.")

try:
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv
    from torch_geometric.data import Data
    TORCH_GEO_AVAILABLE = True
except ImportError:
    TORCH_GEO_AVAILABLE = False
    logging.warning("PyTorch Geometric tidak tersedia. GNN dinonaktifkan.")

logger = logging.getLogger("tenderguard.detector")

FEATURE_COLS = [
    "win_rate", "degree", "weighted_degree",
    "clustering_coef", "betweenness",
    "times_beat_others", "times_lost_to",
    "avg_bid_ratio", "bid_std",
]


# ─────────────────────────────────────────────
# 2.3.1 Isolation Forest (Baseline)
# ─────────────────────────────────────────────

def detect_anomalous_vendors(
    node_features: pd.DataFrame,
    contamination: float = 0.05,
    n_estimators: int = 200,
    random_state: int = 42,
) -> tuple[pd.Series, pd.Series, IsolationForest, StandardScaler]:
    """
    Deteksi vendor anomali berdasarkan fitur statistik menggunakan Isolation Forest.

    Args:
        node_features : DataFrame fitur per vendor
        contamination : Estimasi proporsi vendor mencurigakan (0–0.5)
        n_estimators  : Jumlah pohon dalam ensemble
        random_state  : Seed untuk reproduksibilitas

    Returns:
        Tuple (labels, scores, model, scaler)
        - labels : pd.Series (-1=anomali, 1=normal)
        - scores : pd.Series (raw anomaly score, lebih negatif = lebih anomali)
        - model  : model terlatih
        - scaler : scaler yang digunakan
    """
    # Gunakan hanya kolom fitur yang tersedia
    cols_available = [c for c in FEATURE_COLS if c in node_features.columns]
    X = node_features[cols_available].fillna(0)

    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    labels = clf.fit_predict(X_scaled)
    scores = clf.score_samples(X_scaled)

    n_anomaly = (labels == -1).sum()
    logger.info(f"Isolation Forest: {n_anomaly} vendor terdeteksi anomali "
                f"dari {len(node_features)} total vendor")

    return (
        pd.Series(labels, index=node_features.index, name="if_label"),
        pd.Series(scores, index=node_features.index, name="if_score"),
        clf,
        scaler,
    )


# ─────────────────────────────────────────────
# 2.3.2 Community Detection (Louvain)
# ─────────────────────────────────────────────

def detect_suspicious_communities(
    G: nx.Graph,
    node_features: pd.DataFrame,
    min_size: int = 3,
    suspicion_multiplier: float = 3.0,
) -> dict:
    """
    Temukan komunitas vendor menggunakan Louvain Community Detection.
    Komunitas mencurigakan ditandai jika rata-rata win_rate anggota
    jauh di atas rata-rata global.

    Args:
        G                    : Co-participation graph
        node_features        : DataFrame fitur per vendor
        min_size             : Ukuran minimum komunitas untuk dipertimbangkan
        suspicion_multiplier : Multiplier threshold win_rate

    Returns:
        dict {comm_id: {members, size, avg_win_rate, suspicion_flag, ...}}
    """
    if not LOUVAIN_AVAILABLE:
        logger.warning("python-louvain tidak tersedia — mengembalikan dict kosong")
        return {}

    if G.number_of_nodes() == 0:
        logger.warning("Graph kosong — skip community detection")
        return {}

    logger.info("Menjalankan Louvain community detection...")
    partition = community_louvain.best_partition(G, weight="weight")

    global_avg_win_rate = node_features["win_rate"].mean() if "win_rate" in node_features.columns else 0

    community_stats = {}
    for comm_id in set(partition.values()):
        members = [v for v, c in partition.items() if c == comm_id]
        if len(members) < min_size:
            continue

        valid_members = [m for m in members if m in node_features.index]
        if not valid_members:
            continue

        stats = node_features.loc[valid_members]
        avg_wr = stats["win_rate"].mean() if "win_rate" in stats.columns else 0

        community_stats[comm_id] = {
            "members"        : members,
            "size"           : len(members),
            "avg_win_rate"   : avg_wr,
            "total_wins"     : int(stats["win_count"].sum()) if "win_count" in stats.columns else 0,
            "suspicion_flag" : avg_wr > (global_avg_win_rate * suspicion_multiplier),
        }

    suspicious = sum(1 for c in community_stats.values() if c["suspicion_flag"])
    logger.info(
        f"Community detection: {len(community_stats)} komunitas ditemukan, "
        f"{suspicious} mencurigakan"
    )

    return community_stats


def get_vendor_community_flags(
    partition: dict,
    community_stats: dict,
) -> pd.Series:
    """
    Buat Series boolean: vendor berada di komunitas mencurigakan atau tidak.

    Args:
        partition       : {vendor_id: community_id}
        community_stats : Output dari detect_suspicious_communities

    Returns:
        pd.Series {vendor_id: True/False}
    """
    suspicious_comms = {
        cid for cid, stats in community_stats.items() if stats["suspicion_flag"]
    }
    flags = {
        vendor: (partition.get(vendor, -1) in suspicious_comms)
        for vendor in partition
    }
    return pd.Series(flags, name="in_suspicious_community")


# ─────────────────────────────────────────────
# 2.3.3 Graph Neural Network (GNN)
# ─────────────────────────────────────────────

if TORCH_GEO_AVAILABLE:
    class TenderCollusionGNN(torch.nn.Module):
        """
        Graph Convolutional Network untuk klasifikasi node (vendor) mencurigakan.
        Arsitektur: 2-layer GCN + 1 output layer.
        Input: node features dari Modul 2
        Output: 2-class (normal=0, suspicious=1)
        """

        def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 2):
            super().__init__()
            self.conv1      = GCNConv(input_dim, hidden_dim)
            self.conv2      = GCNConv(hidden_dim, hidden_dim)
            self.classifier = torch.nn.Linear(hidden_dim, output_dim)

        def forward(self, x, edge_index, edge_weight=None):
            x = F.relu(self.conv1(x, edge_index, edge_weight))
            x = F.dropout(x, p=0.3, training=self.training)
            x = F.relu(self.conv2(x, edge_index, edge_weight))
            return self.classifier(x)


    def prepare_graph_data(
        G: nx.Graph,
        node_features: pd.DataFrame,
    ) -> "Data":
        """
        Konversi NetworkX graph ke format PyTorch Geometric Data.

        Args:
            G             : Co-participation graph
            node_features : DataFrame fitur per vendor

        Returns:
            torch_geometric.data.Data
        """
        feature_cols_gnn = [
            "win_rate", "degree", "clustering_coef",
            "betweenness", "times_beat_others", "avg_bid_ratio",
        ]
        cols_available = [c for c in feature_cols_gnn if c in node_features.columns]

        # Node index mapping
        node_list = [n for n in G.nodes() if n in node_features.index]
        node_idx  = {v: i for i, v in enumerate(node_list)}

        # Edge index (COO format)
        edges = [
            (node_idx[u], node_idx[v])
            for u, v in G.edges()
            if u in node_idx and v in node_idx
        ]
        if not edges:
            logger.warning("Tidak ada edge dalam graph — GNN tidak dapat dijalankan")
            return None

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        # Edge weights
        edge_weights = torch.tensor(
            [G[u][v].get("weight", 1) for u, v in G.edges()
             if u in node_idx and v in node_idx],
            dtype=torch.float,
        )

        # Node feature matrix
        X_np = node_features.reindex(node_list)[cols_available].fillna(0).values
        X    = torch.tensor(X_np, dtype=torch.float)

        return Data(x=X, edge_index=edge_index, edge_attr=edge_weights)


    def train_gnn(
        G: nx.Graph,
        node_features: pd.DataFrame,
        if_labels: pd.Series,
        hidden_dim: int = 64,
        epochs: int = 100,
        lr: float = 0.001,
        device: str = "cpu",
    ) -> Optional["TenderCollusionGNN"]:
        """
        Latih GNN menggunakan label semi-supervised dari Isolation Forest.

        Label Training:
        - Anomali dari IF (label=-1) → kelas 1 (suspicious)
        - Normal dari IF (label=+1) dengan low win_rate → kelas 0 (normal)

        Returns:
            Model terlatih atau None jika data tidak cukup
        """
        data = prepare_graph_data(G, node_features)
        if data is None:
            return None

        node_list = [n for n in G.nodes() if n in node_features.index]

        # Buat label training
        y_list = []
        mask_list = []
        for vendor in node_list:
            if vendor in if_labels.index:
                label = 0 if if_labels[vendor] == 1 else 1  # IF: -1=anomali → 1
                y_list.append(label)
                mask_list.append(True)
            else:
                y_list.append(0)
                mask_list.append(False)

        y    = torch.tensor(y_list, dtype=torch.long)
        mask = torch.tensor(mask_list, dtype=torch.bool)

        input_dim = data.x.shape[1]
        model     = TenderCollusionGNN(input_dim, hidden_dim=hidden_dim)
        model.to(device)
        data  = data.to(device)
        y     = y.to(device)
        mask  = mask.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()

        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            out  = model(data.x, data.edge_index, data.edge_attr)
            loss = criterion(out[mask], y[mask])
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 20 == 0:
                logger.info(f"GNN Epoch {epoch+1}/{epochs} — Loss: {loss.item():.4f}")

        logger.info("GNN training selesai")
        return model


    def gnn_predict(
        model: "TenderCollusionGNN",
        G: nx.Graph,
        node_features: pd.DataFrame,
        device: str = "cpu",
    ) -> pd.Series:
        """
        Jalankan inferensi GNN untuk mendapatkan probabilitas mencurigakan per vendor.

        Returns:
            pd.Series {vendor_id: prob_suspicious}
        """
        data      = prepare_graph_data(G, node_features)
        node_list = [n for n in G.nodes() if n in node_features.index]

        model.eval()
        with torch.no_grad():
            data  = data.to(device)
            out   = model(data.x, data.edge_index, data.edge_attr)
            probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()

        return pd.Series(dict(zip(node_list, probs)), name="gnn_suspicious_prob")


# ─────────────────────────────────────────────
# Pipeline Detector
# ─────────────────────────────────────────────

class AnomalyDetectionPipeline:
    """
    Orkestrasi deteksi anomali menggunakan Isolation Forest,
    Community Detection (Louvain), dan GNN (opsional).
    """

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        self.det_cfg = self.cfg.get("detection", {})
        Path("models").mkdir(exist_ok=True)

    def run(
        self,
        G: nx.Graph,
        DG: nx.DiGraph,
        node_features: pd.DataFrame,
        use_gnn: bool = False,
    ) -> dict:
        """
        Jalankan seluruh pipeline deteksi anomali.

        Args:
            G             : Co-participation graph
            DG            : Win-loss directed graph
            node_features : DataFrame fitur per vendor
            use_gnn       : Aktifkan GNN (butuh PyTorch Geometric)

        Returns:
            dict berisi: if_labels, if_scores, community_stats, community_flags,
                         gnn_scores (jika aktif), partition
        """
        if_cfg  = self.det_cfg.get("isolation_forest", {})
        com_cfg = self.det_cfg.get("community", {})
        gnn_cfg = self.det_cfg.get("gnn", {})

        # 1. Isolation Forest
        logger.info("=== Menjalankan Isolation Forest ===")
        if_labels, if_scores, if_model, if_scaler = detect_anomalous_vendors(
            node_features,
            contamination=if_cfg.get("contamination", 0.05),
            n_estimators =if_cfg.get("n_estimators", 200),
            random_state =if_cfg.get("random_state", 42),
        )
        # Simpan model
        with open("models/isolation_forest.pkl", "wb") as f:
            pickle.dump({"model": if_model, "scaler": if_scaler}, f)

        # 2. Community Detection (Louvain)
        logger.info("=== Menjalankan Community Detection (Louvain) ===")
        partition = {}
        community_flags = pd.Series(False, index=node_features.index, name="in_suspicious_community")
        community_stats = {}

        if LOUVAIN_AVAILABLE:
            community_stats = detect_suspicious_communities(
                G, node_features,
                min_size=com_cfg.get("min_community_size", 3),
            )
            try:
                partition = community_louvain.best_partition(G, weight="weight")
                community_flags = get_vendor_community_flags(partition, community_stats)
            except Exception as e:
                logger.warning(f"Community detection gagal: {e}")

        # 3. GNN (opsional)
        gnn_scores = pd.Series(0.0, index=node_features.index, name="gnn_suspicious_prob")
        gnn_model  = None

        if use_gnn and TORCH_GEO_AVAILABLE:
            logger.info("=== Menjalankan GNN Training ===")
            gnn_model = train_gnn(
                G, node_features, if_labels,
                hidden_dim=gnn_cfg.get("hidden_dim", 64),
                epochs    =gnn_cfg.get("epochs", 100),
                lr        =gnn_cfg.get("lr", 0.001),
            )
            if gnn_model is not None:
                gnn_scores = gnn_predict(gnn_model, G, node_features)
                torch.save(gnn_model.state_dict(), "models/gnn_checkpoint.pt")

        # Gabungkan hasil ke satu DataFrame
        detection_results = pd.DataFrame({
            "if_label"               : if_labels,
            "if_score"               : if_scores,
            "in_suspicious_community": community_flags,
            "gnn_suspicious_prob"    : gnn_scores,
        })
        detection_results.to_csv("data/processed/detection_results.csv")

        logger.info("Deteksi anomali selesai. Hasil disimpan di data/processed/detection_results.csv")

        return {
            "if_labels"       : if_labels,
            "if_scores"       : if_scores,
            "community_stats" : community_stats,
            "community_flags" : community_flags,
            "gnn_scores"      : gnn_scores,
            "partition"       : partition,
        }


# ─────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from modules.graph_builder import GraphBuilderPipeline

    parser = argparse.ArgumentParser(description="TenderGuard — Modul 3: Anomaly Detection")
    parser.add_argument("--config",  default="config.yaml")
    parser.add_argument("--gnn",     action="store_true", default=False, help="Aktifkan GNN (butuh PyTorch Geometric)")
    args = parser.parse_args()

    # Load graph dari file
    graphs       = GraphBuilderPipeline.load_graphs()
    G            = graphs["G"]
    DG           = graphs["DG"]
    node_features= graphs["node_features"]

    pipeline = AnomalyDetectionPipeline(config_path=args.config)
    results  = pipeline.run(G, DG, node_features, use_gnn=args.gnn)

    n_anomaly   = (results["if_labels"] == -1).sum()
    n_suspicious_comm = sum(1 for c in results["community_stats"].values() if c["suspicion_flag"])

    print(f"\n✅ Deteksi anomali selesai.")
    print(f"   Vendor anomali (IF)           : {n_anomaly}")
    print(f"   Komunitas mencurigakan        : {n_suspicious_comm}")
    print(f"   Vendor di komunitas mencurigakan: {results['community_flags'].sum()}")
