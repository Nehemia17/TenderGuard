"""
TenderGuard — Streamlit Dashboard
===================================
Antarmuka visual utama untuk eksplorasi hasil analisis kolusi tender.

Tabs:
1. 📋 Tender Berisiko  — Tabel ranking tender berdasarkan risk score
2. 🕸️ Peta Jaringan   — Visualisasi interaktif graf vendor
3. 📊 Statistik       — Ringkasan dan chart analisis
4. 🔍 Detail Vendor   — Profil individual vendor

Sesuai TRD Section 2.5.2
"""

import os
import json
import pickle
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx

# ── Konfigurasi halaman ────────────────────────────────────────
st.set_page_config(
    page_title="TenderGuard — Deteksi Kolusi Tender",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS: Load from style.css (ClickHouse Design System) ───────
_css_path = Path(__file__).parent / "style.css"
if _css_path.exists():
    with open(_css_path, "r", encoding="utf-8") as _f:
        st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)
else:
    st.warning("style.css not found. Design might be broken.")


# ══════════════════════════════════════════════
# Data Loading Helpers
# ══════════════════════════════════════════════

@st.cache_data(ttl=300)
def load_vendor_scores() -> pd.DataFrame:
    path = "data/processed/vendor_risk_scores.csv"
    if not Path(path).exists():
        return pd.DataFrame()
    df = pd.read_csv(path, index_col="vendor_id")
    return df


@st.cache_data(ttl=300)
def load_tender_scores() -> pd.DataFrame:
    path = "data/processed/tender_risk_scores.csv"
    if not Path(path).exists():
        return pd.DataFrame()
    df = pd.read_csv(path, index_col="tender_id")
    return df


@st.cache_data(ttl=300)
def load_peserta() -> pd.DataFrame:
    path = "data/processed/peserta_clean.csv"
    if not Path(path).exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_resource
def load_graphs():
    try:
        with open("data/processed/graph_G.pkl", "rb") as f:
            G = pickle.load(f)
        with open("data/processed/graph_DG.pkl", "rb") as f:
            DG = pickle.load(f)
        nf = pd.read_csv("data/processed/node_features.csv", index_col="vendor_id")
        return G, DG, nf
    except FileNotFoundError:
        return None, None, None


def load_network_html() -> str:
    path = "data/vendor_network.html"
    if Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


# ══════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════

def render_sidebar(vendor_scores: pd.DataFrame, tender_scores: pd.DataFrame):
    with st.sidebar:
        # Brand header
        st.markdown("""
        <div style="padding:1.5rem 0 1rem;">
            <div style="font-size:1.5rem;font-weight:700;letter-spacing:-1px;color:#ffffff;">
                <span style="color:#faff69;">Tender</span>Guard
            </div>
            <div style="font-size:0.75rem;font-weight:600;color:#888888;text-transform:uppercase;
                letter-spacing:1.5px;margin-top:4px;">Graph Mining · Kolusi Deteksi</div>
        </div>
        """, unsafe_allow_html=True)
        st.divider()

        # Filter risk level
        st.markdown("<div class='section-label'>Filter Level Risiko</div>", unsafe_allow_html=True)
        risk_levels = st.multiselect(
            "", options=["HIGH", "MEDIUM", "LOW"], default=["HIGH", "MEDIUM"],
            key="filter_risk_levels",
        )

        st.markdown("<div class='section-label' style='margin-top:1rem;'>Minimum Risk Score</div>", unsafe_allow_html=True)
        min_risk = st.slider(
            "", min_value=0.0, max_value=1.0, value=0.0, step=0.05,
            key="filter_min_risk",
        )

        instansi_list = []
        if not tender_scores.empty and "instansi" in tender_scores.columns:
            instansi_list = ["Semua"] + sorted(tender_scores["instansi"].dropna().unique().tolist())
        st.markdown("<div class='section-label' style='margin-top:1rem;'>Instansi</div>", unsafe_allow_html=True)
        selected_instansi = st.selectbox("", instansi_list or ["Semua"], key="filter_instansi")

        st.divider()

        if not vendor_scores.empty:
            n_high   = int((vendor_scores["risk_level"] == "HIGH").sum())
            n_medium = int((vendor_scores["risk_level"] == "MEDIUM").sum())
            n_low    = int((vendor_scores["risk_level"] == "LOW").sum())
            st.markdown("<div class='section-label'>Ringkasan</div>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            col1.metric("HIGH",   n_high)
            col1.metric("MEDIUM", n_medium)
            col2.metric("LOW",    n_low)
            col2.metric("Total",  len(vendor_scores))

        st.divider()
        st.markdown(
            "<p style='font-size:0.7rem;color:#5a5a5a;text-transform:uppercase;"
            "letter-spacing:1px;'>TenderGuard v1.0 · GEMASTIK 2026</p>",
            unsafe_allow_html=True,
        )

    return risk_levels, min_risk, selected_instansi


# ══════════════════════════════════════════════
# Dashboard Sections
# ══════════════════════════════════════════════

def render_global_metrics(tender_scores, vendor_scores, G):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        n_tender = len(tender_scores) if not tender_scores.empty else 0
        st.markdown(f"<div class='stat-label'>Total Tender</div><div class='stat-callout'>{n_tender:,}</div>", unsafe_allow_html=True)
    with col2:
        n_high = (tender_scores["tender_risk_level"] == "HIGH").sum() if not tender_scores.empty else 0
        st.markdown(f"<div class='stat-label'>Tender Risiko Tinggi</div><div class='stat-callout' style='color:#ef4444;'>{n_high:,}</div>", unsafe_allow_html=True)
    with col3:
        n_vendor = len(vendor_scores) if not vendor_scores.empty else 0
        st.markdown(f"<div class='stat-label'>Total Vendor</div><div class='stat-callout'>{n_vendor:,}</div>", unsafe_allow_html=True)
    with col4:
        n_nodes = G.number_of_nodes() if G else 0
        n_edges = G.number_of_edges() if G else 0
        st.markdown(f"<div class='stat-label'>Node / Edge Graf</div><div class='stat-callout'>{n_nodes:,} <span style='font-size:1rem;color:#888;'>/ {n_edges:,}</span></div>", unsafe_allow_html=True)


def render_network_view():
    st.markdown("<div class='section-title'><span>🕸️ Peta Jaringan</span> Kolusi</div>", unsafe_allow_html=True)
    network_html = load_network_html()
    if not network_html:
        st.info("Jalankan pipeline untuk membuat peta jaringan.")
        return

    st.markdown(
        "<p style='font-size:0.75rem;color:#888;margin-bottom:1rem;'>"
        "🔴 HIGH Risk &nbsp;·&nbsp; 🟡 MEDIUM Risk &nbsp;·&nbsp; 🟢 LOW Risk "
        "| Ukuran node = risk score. Ketebalan edge = frekuensi co-participation.</p>",
        unsafe_allow_html=True
    )
    st.components.v1.html(network_html, height=750, scrolling=False)


def render_charts_view(vendor_scores):
    st.markdown("<div class='section-title'><span>📊 Distribusi</span> Risiko</div>", unsafe_allow_html=True)
    if vendor_scores.empty:
        st.info("Data belum tersedia.")
        return

    try:
        import plotly.graph_objects as go
        top10 = vendor_scores.nlargest(10, "risk_score").reset_index()
        colors_bar = [
            "#ef4444" if l == "HIGH" else
            "#faff69" if l == "MEDIUM" else "#22c55e"
            for l in top10["risk_level"]
        ]
        fig_bar = go.Figure(go.Bar(
            x=top10["risk_score"],
            y=top10["vendor_id"].str[:25],
            orientation="h",
            marker_color=colors_bar,
            marker_line_width=0,
            text=top10["risk_score"].apply(lambda x: f"{x:.3f}"),
            textposition="outside",
            textfont=dict(color="#cccccc", size=11),
        ))
        fig_bar.update_layout(
            title=dict(text="Top 10 Vendor Paling Berisiko", font=dict(size=14, color="#888888")),
            xaxis=dict(title="Risk Score", color="#888888", gridcolor="#2a2a2a", title_font=dict(color="#888888")),
            yaxis=dict(color="#cccccc", categoryorder="total ascending"),
            paper_bgcolor="#0a0a0a",
            plot_bgcolor="#0a0a0a",
            font=dict(color="#cccccc", family="Inter"),
            height=360,
            margin=dict(t=40, b=10, l=10, r=40),
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    except ImportError:
        pass

    try:
        import plotly.express as px
        fig_hist = px.histogram(
            vendor_scores.reset_index(),
            x="risk_score",
            color="risk_level",
            color_discrete_map={"HIGH": "#ef4444", "MEDIUM": "#faff69", "LOW": "#22c55e"},
            nbins=25,
            title="Distribusi Risk Score Vendor",
            template="plotly_dark",
            opacity=0.85,
        )
        fig_hist.update_layout(
            paper_bgcolor="#0a0a0a",
            plot_bgcolor="#121212",
            font=dict(color="#cccccc", family="Inter"),
            title_font=dict(size=14, color="#888888"),
            xaxis=dict(gridcolor="#2a2a2a", color="#888888"),
            yaxis=dict(gridcolor="#2a2a2a", color="#888888"),
            bargap=0.05,
            height=350,
            margin=dict(t=40, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    except ImportError:
        pass


def render_tender_table(tender_scores, risk_levels, min_risk, selected_instansi):
    st.markdown("<div class='section-title'><span>📋 Daftar Tender</span> Berisiko Tinggi</div>", unsafe_allow_html=True)

    if tender_scores.empty:
        st.info("Data belum tersedia.")
        return

    df = tender_scores.copy()
    df = df[df["tender_risk_level"].isin(risk_levels)]
    df = df[df["tender_risk_score"] >= min_risk]
    if selected_instansi != "Semua" and "instansi" in df.columns:
        df = df[df["instansi"] == selected_instansi]

    st.markdown(f"<p style='color:#888; font-size:0.85rem;'>Menampilkan <b>{len(df)}</b> tender sesuai filter</p>", unsafe_allow_html=True)

    display_cols = ["tender_risk_score", "tender_risk_level", "vendor_count", "high_risk_vendors", "suspicious_vendors"]
    if "judul" in df.columns: display_cols = ["judul", "instansi"] + display_cols
    if "tanggal" in df.columns: display_cols.append("tanggal")
    available_cols = [c for c in display_cols if c in df.columns]

    def highlight_risk(val):
        if val == "HIGH": return "background-color:rgba(239,68,68,0.2); color:#ef4444; font-weight:700"
        if val == "MEDIUM": return "background-color:rgba(250,255,105,0.15); color:#faff69; font-weight:700"
        if val == "LOW": return "background-color:rgba(34,197,94,0.15); color:#22c55e; font-weight:700"
        return ""

    fmt_dict = {}
    if "tender_risk_score" in available_cols: fmt_dict["tender_risk_score"] = "{:.3f}"
    
    styled = df[available_cols].style.map(
        highlight_risk, subset=["tender_risk_level"]
    ).background_gradient(
        subset=["tender_risk_score"], cmap="YlOrRd"
    ).format(fmt_dict)

    st.dataframe(styled, use_container_width=True, height=400)


def render_vendor_detail(vendor_scores, peserta):
    st.markdown("<div class='section-title'><span>🔍 Profil Detail</span> Vendor</div>", unsafe_allow_html=True)

    if vendor_scores.empty:
        return

    vendor_list = vendor_scores.sort_values("risk_score", ascending=False).index.tolist()
    selected = st.selectbox(
        "",
        options=vendor_list,
        format_func=lambda x: f"{x} — {vendor_scores.loc[x, 'risk_level']} ({vendor_scores.loc[x, 'risk_score']:.3f})",
        key="vendor_selector",
    )

    if not selected:
        return

    row = vendor_scores.loc[selected]
    level = row.get("risk_level", "LOW")
    badge_class = f"badge-{level.lower()}"
    
    st.markdown(f"<h3 style='margin-bottom:1rem;'>{selected} &nbsp; <span class='{badge_class}'>{level}</span></h3>", unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Risk Score",    f"{row.get('risk_score', 0):.3f}")
    c2.metric("Win Rate",      f"{row.get('win_rate', 0):.1%}")
    c3.metric("Total Tender",  f"{int(row.get('total_tenders', 0))}")
    c4.metric("Menang",        f"{int(row.get('win_count', 0))}")
    c5.metric("IF Score Raw",  f"{row.get('if_score_raw', 0):.4f}")

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("<h4 style='color:#faff69; margin-bottom:1rem;'>🚩 Flag Heuristik</h4>", unsafe_allow_html=True)
        flag_cols = [c for c in row.index if c.startswith("flag_")]
        if flag_cols:
            for fc in flag_cols:
                label = fc.replace("flag_", "").replace("_", " ").title()
                val   = bool(row.get(fc, False))
                icon  = "✅" if val else "⬜"
                color = "#fc8181" if val else "#68d391"
                st.markdown(f"<div style='margin-bottom:0.5rem;'><span style='color:{color}'>{icon}</span> <span style='color:#ccc;'>{label}</span></div>", unsafe_allow_html=True)
        else:
            st.info("Tidak ada flag heuristik tersedia")

    with col_r:
        st.markdown("<h4 style='color:#faff69; margin-bottom:1rem;'>📑 Riwayat Tender</h4>", unsafe_allow_html=True)
        if not peserta.empty and "vendor_id" in peserta.columns:
            v_tenders = peserta[peserta["vendor_id"] == selected][["tender_id", "is_winner", "nilai_penawaran", "rank"]].head(20)
            if not v_tenders.empty:
                st.dataframe(v_tenders, use_container_width=True, hide_index=True)
            else:
                st.info("Tidak ada riwayat tender ditemukan")


# ══════════════════════════════════════════════
# Main App
# ══════════════════════════════════════════════

def main():
    # Hero Header
    st.markdown("""
    <div class="tg-header">
        <h1><span class="brand">Tender</span>Guard</h1>
        <p>Sistem Deteksi Kolusi Pengadaan Berbasis Graph Mining &nbsp;·&nbsp; Data LPSE Nasional</p>
    </div>
    """, unsafe_allow_html=True)

    if not Path("data/processed/vendor_risk_scores.csv").exists():
        st.warning("⚠️ **Data belum tersedia.** Jalankan pipeline atau `python generate_demo_data.py` untuk mengisi dashboard.", icon="⚠️")

    # Load data
    vendor_scores = load_vendor_scores()
    tender_scores = load_tender_scores()
    peserta       = load_peserta()
    G, DG, nf     = load_graphs()

    # Sidebar
    risk_levels, min_risk, selected_instansi = render_sidebar(vendor_scores, tender_scores)

    # Dashboard Layout: Single Page Command Center
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 1. Top Metrics
    render_global_metrics(tender_scores, vendor_scores, G)
    st.markdown("<hr>", unsafe_allow_html=True)

    # 2. Graph & Charts (Side by Side)
    col_map, col_charts = st.columns([5.5, 4.5], gap="large")
    with col_map:
        render_network_view()
    with col_charts:
        render_charts_view(vendor_scores)
    
    st.markdown("<hr>", unsafe_allow_html=True)

    # 3. Tables (Full Width)
    render_tender_table(tender_scores, risk_levels, min_risk, selected_instansi)
    
    st.markdown("<hr>", unsafe_allow_html=True)

    # 4. Deep Dive
    render_vendor_detail(vendor_scores, peserta)


if __name__ == "__main__":
    main()
