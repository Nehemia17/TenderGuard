"""
Modul 5 — Visualisasi
======================
Render graf vendor interaktif menggunakan Pyvis dan
ekspor laporan ke format CSV/HTML.

Sesuai TRD Section 2.5.1
"""

import logging
from pathlib import Path

import networkx as nx
import pandas as pd

try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False
    logging.warning("pyvis tidak tersedia. Visualisasi HTML dinonaktifkan.")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

logger = logging.getLogger("tenderguard.visualizer")

# Peta warna risk level
COLOR_MAP = {
    "HIGH"   : "#e74c3c",  # Merah
    "MEDIUM" : "#f39c12",  # Oranye
    "LOW"    : "#2ecc71",  # Hijau
}


# ─────────────────────────────────────────────
# Classify risk helper (duplikasi agar modul independen)
# ─────────────────────────────────────────────

def _classify_risk(score: float) -> str:
    if score >= 0.70:
        return "HIGH"
    elif score >= 0.40:
        return "MEDIUM"
    return "LOW"


# ─────────────────────────────────────────────
# 2.5.1 Pyvis Interactive Network
# ─────────────────────────────────────────────

def visualize_vendor_network(
    G: nx.Graph,
    vendor_scores: pd.DataFrame,
    output_path: str = "data/vendor_network.html",
    max_nodes: int = 500,
) -> str:
    """
    Render graf vendor dengan warna berdasarkan risk score menggunakan Pyvis.

    Warna node:
    - Merah (#e74c3c)  : HIGH risk
    - Oranye (#f39c12) : MEDIUM risk
    - Hijau (#2ecc71)  : LOW risk

    Ukuran node proporsional dengan risk score.

    Args:
        G             : Co-participation graph
        vendor_scores : DataFrame dengan kolom risk_score, risk_level, win_rate, total_tenders
        output_path   : Path file HTML output
        max_nodes     : Batas node yang dirender (untuk performa)

    Returns:
        Path file yang disimpan
    """
    if not PYVIS_AVAILABLE:
        logger.error("pyvis tidak tersedia. Install dengan: pip install pyvis")
        return ""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Ambil subgraf jika node terlalu banyak (tampilkan yg paling berisiko)
    if G.number_of_nodes() > max_nodes:
        if not vendor_scores.empty:
            top_vendors = vendor_scores.nlargest(max_nodes, "risk_score").index.tolist()
        else:
            top_vendors = list(G.nodes())[:max_nodes]
        G = G.subgraph(top_vendors).copy()
        logger.info(f"Menampilkan subgraf dengan {max_nodes} node teratas")

    net = Network(
        height="750px",
        width="100%",
        bgcolor="#0f0f1a",
        font_color="white",
        notebook=False,
    )
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "stabilization": {"iterations": 200},
        "barnesHut": {
          "gravitationalConstant": -8000,
          "springLength": 120
        }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 200
      }
    }
    """)

    for node in G.nodes():
        if node in vendor_scores.index:
            score = float(vendor_scores.loc[node, "risk_score"])
            level = str(vendor_scores.loc[node, "risk_level"])
            wrate = vendor_scores.loc[node].get("win_rate", 0)
            total = int(vendor_scores.loc[node].get("total_tenders", 0))
        else:
            score = 0.0
            level = "LOW"
            wrate = 0.0
            total = 0

        color = COLOR_MAP.get(level, "#95a5a6")
        size  = 10 + score * 40

        tooltip = (
            f"<b>{node}</b><br>"
            f"Risk Score: <b>{score:.3f}</b><br>"
            f"Level: <b>{level}</b><br>"
            f"Win Rate: {wrate:.1%}<br>"
            f"Total Tender: {total}"
        )

        net.add_node(
            node,
            label=str(node)[:25],
            color=color,
            size=size,
            title=tooltip,
            borderWidth=2,
            borderWidthSelected=4,
        )

    for u, v, data in G.edges(data=True):
        weight = data.get("weight", 1)
        net.add_edge(
            u, v,
            width=min(weight * 0.4, 6),
            color="#3d3d5c",
            title=f"Co-participation: {weight}x",
        )

    net.save_graph(output_path)
    logger.info(f"Network visualization disimpan: {output_path}")
    return output_path


# ─────────────────────────────────────────────
# Plotly Static Charts
# ─────────────────────────────────────────────

def plot_risk_distribution(
    vendor_scores: pd.DataFrame,
    output_path: str = "data/processed/risk_distribution.html",
) -> str:
    """Histogram distribusi risk score vendor."""
    if not PLOTLY_AVAILABLE:
        logger.warning("Plotly tidak tersedia")
        return ""

    fig = px.histogram(
        vendor_scores.reset_index(),
        x="risk_score",
        color="risk_level",
        color_discrete_map={"HIGH": "#e74c3c", "MEDIUM": "#f39c12", "LOW": "#2ecc71"},
        nbins=30,
        title="Distribusi Risk Score Vendor",
        labels={"risk_score": "Risk Score", "risk_level": "Level Risiko"},
        template="plotly_dark",
    )
    fig.update_layout(
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="#16213e",
        font_color="white",
    )
    fig.write_html(output_path)
    logger.info(f"Plot distribusi disimpan: {output_path}")
    return output_path


def plot_top_risky_vendors(
    vendor_scores: pd.DataFrame,
    top_n: int = 20,
    output_path: str = "data/processed/top_risky_vendors.html",
) -> str:
    """Bar chart top N vendor paling berisiko."""
    if not PLOTLY_AVAILABLE:
        logger.warning("Plotly tidak tersedia")
        return ""

    top = vendor_scores.head(top_n).reset_index()

    fig = px.bar(
        top,
        x="risk_score",
        y="vendor_id",
        orientation="h",
        color="risk_level",
        color_discrete_map={"HIGH": "#e74c3c", "MEDIUM": "#f39c12", "LOW": "#2ecc71"},
        title=f"Top {top_n} Vendor Paling Berisiko",
        labels={"risk_score": "Risk Score", "vendor_id": "Vendor"},
        template="plotly_dark",
    )
    fig.update_layout(
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="#16213e",
        font_color="white",
        yaxis={"categoryorder": "total ascending"},
        height=600,
    )
    fig.write_html(output_path)
    logger.info(f"Plot top vendor disimpan: {output_path}")
    return output_path


def plot_tender_risk_timeline(
    tender_scores: pd.DataFrame,
    output_path: str = "data/processed/tender_risk_timeline.html",
) -> str:
    """Scatter plot risk score tender berdasarkan tanggal."""
    if not PLOTLY_AVAILABLE or "tanggal" not in tender_scores.columns:
        return ""

    df = tender_scores.reset_index().dropna(subset=["tanggal"])
    df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce")
    df = df.dropna(subset=["tanggal"])

    fig = px.scatter(
        df,
        x="tanggal",
        y="tender_risk_score",
        color="tender_risk_level",
        color_discrete_map={"HIGH": "#e74c3c", "MEDIUM": "#f39c12", "LOW": "#2ecc71"},
        size="tender_risk_score",
        hover_data=["tender_id", "instansi", "vendor_count"],
        title="Timeline Risk Score Tender",
        template="plotly_dark",
    )
    fig.update_layout(
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="#16213e",
        font_color="white",
    )
    fig.write_html(output_path)
    return output_path


# ─────────────────────────────────────────────
# Matplotlib Static Export (backup)
# ─────────────────────────────────────────────

def render_network_matplotlib(
    G: nx.Graph,
    vendor_scores: pd.DataFrame,
    output_path: str = "data/processed/network_static.png",
    max_nodes: int = 200,
) -> str:
    """Render static network map menggunakan matplotlib (backup jika pyvis tidak tersedia)."""
    if not MPL_AVAILABLE:
        return ""

    if G.number_of_nodes() > max_nodes:
        top_vendors = vendor_scores.nlargest(max_nodes, "risk_score").index.tolist()
        G = G.subgraph(top_vendors).copy()

    node_colors = []
    node_sizes  = []
    for node in G.nodes():
        if node in vendor_scores.index:
            score = float(vendor_scores.loc[node, "risk_score"])
            level = str(vendor_scores.loc[node, "risk_level"])
        else:
            score = 0.0
            level = "LOW"
        color_hex = COLOR_MAP.get(level, "#95a5a6")
        node_colors.append(color_hex)
        node_sizes.append(50 + score * 300)

    pos = nx.spring_layout(G, k=0.5, seed=42)
    plt.figure(figsize=(16, 12), facecolor="#0f0f1a")
    ax = plt.gca()
    ax.set_facecolor("#0f0f1a")

    nx.draw_networkx(
        G, pos,
        node_color=node_colors,
        node_size=node_sizes,
        edge_color="#3d3d5c",
        width=0.5,
        font_size=6,
        font_color="white",
        with_labels=False,
        ax=ax,
    )

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e74c3c", label="HIGH Risk"),
        Patch(facecolor="#f39c12", label="MEDIUM Risk"),
        Patch(facecolor="#2ecc71", label="LOW Risk"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", facecolor="#1a1a2e", labelcolor="white")
    plt.title("Vendor Network — TenderGuard", color="white", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
    plt.close()

    logger.info(f"Static network plot disimpan: {output_path}")
    return output_path


# ─────────────────────────────────────────────
# Export Laporan
# ─────────────────────────────────────────────

def export_csv_report(
    vendor_scores: pd.DataFrame,
    tender_scores: pd.DataFrame,
    output_dir: str = "data/processed",
) -> list[str]:
    """Export laporan CSV lengkap."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    paths = []

    vp = f"{output_dir}/vendor_risk_scores.csv"
    tp = f"{output_dir}/tender_risk_scores.csv"

    vendor_scores.to_csv(vp)
    tender_scores.to_csv(tp)

    paths.extend([vp, tp])
    logger.info(f"CSV reports disimpan: {vp}, {tp}")
    return paths


# ─────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from modules.graph_builder import GraphBuilderPipeline

    parser = argparse.ArgumentParser(description="TenderGuard — Modul 5: Visualisasi")
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--max-nodes",  type=int, default=300)
    args = parser.parse_args()

    # Load data
    graphs        = GraphBuilderPipeline.load_graphs()
    G             = graphs["G"]
    vendor_scores = pd.read_csv("data/processed/vendor_risk_scores.csv", index_col="vendor_id")
    tender_scores = pd.read_csv("data/processed/tender_risk_scores.csv", index_col="tender_id")

    # Render
    visualize_vendor_network(G, vendor_scores, max_nodes=args.max_nodes)
    plot_risk_distribution(vendor_scores)
    plot_top_risky_vendors(vendor_scores)
    plot_tender_risk_timeline(tender_scores)
    render_network_matplotlib(G, vendor_scores, max_nodes=200)

    print("\n✅ Visualisasi selesai.")
    print("   - data/vendor_network.html")
    print("   - data/processed/risk_distribution.html")
    print("   - data/processed/top_risky_vendors.html")
    print("   - data/processed/network_static.png")
