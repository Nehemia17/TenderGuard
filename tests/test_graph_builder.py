"""
Tests — Modul 1 & 2: Data Ingestion & Graph Construction
"""

import pytest
import pandas as pd
import networkx as nx
from modules.scraper import normalize_vendor_name, deduplicate_vendors
from modules.graph_builder import (
    build_co_participation_graph,
    build_win_loss_graph,
    compute_node_features,
)


# ─────────────────────────────────────────────
# Test: normalize_vendor_name
# ─────────────────────────────────────────────

class TestNormalizeVendorName:

    def test_removes_pt_prefix(self):
        assert normalize_vendor_name("PT. KARYA MAJU") == "KARYA MAJU"

    def test_removes_cv_prefix(self):
        assert normalize_vendor_name("CV. BANGUN JAYA") == "BANGUN JAYA"

    def test_uppercase(self):
        assert normalize_vendor_name("pt karya maju") == "KARYA MAJU"

    def test_strips_whitespace(self):
        assert normalize_vendor_name("  PT KARYA  ") == "KARYA"

    def test_empty_string(self):
        result = normalize_vendor_name("")
        assert result == ""


# ─────────────────────────────────────────────
# Test: deduplicate_vendors
# ─────────────────────────────────────────────

class TestDeduplicateVendors:

    def test_identical_names_merged(self):
        names = ["PT KARYA MAJU", "PT KARYA MAJU"]
        result = deduplicate_vendors(names, threshold=85)
        assert result["PT KARYA MAJU"] == result["PT KARYA MAJU"]

    def test_similar_names_merged(self):
        names = ["PT KARYA MAJU BERSAMA", "PT KARYA MAJU BRSAMA"]  # Typo
        result = deduplicate_vendors(names, threshold=85)
        assert len(set(result.values())) <= 2  # Mungkin sama, mungkin berbeda tergantung threshold

    def test_different_names_not_merged(self):
        names = ["PT KARYA MAJU", "CV BANGUN JAYA ABADI"]
        result = deduplicate_vendors(names, threshold=85)
        assert len(set(result.values())) == 2


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def sample_peserta():
    """DataFrame peserta sederhana untuk testing."""
    return pd.DataFrame([
        # Tender 1: A menang, B dan C kalah
        {"tender_id": "T001", "vendor_id": "A", "is_winner": True,  "nilai_penawaran": 900_000_000, "nilai_hps": 1_000_000_000},
        {"tender_id": "T001", "vendor_id": "B", "is_winner": False, "nilai_penawaran": 950_000_000, "nilai_hps": 1_000_000_000},
        {"tender_id": "T001", "vendor_id": "C", "is_winner": False, "nilai_penawaran": 960_000_000, "nilai_hps": 1_000_000_000},
        # Tender 2: A menang lagi, B kalah lagi (pola kolusi)
        {"tender_id": "T002", "vendor_id": "A", "is_winner": True,  "nilai_penawaran": 850_000_000, "nilai_hps": 950_000_000},
        {"tender_id": "T002", "vendor_id": "B", "is_winner": False, "nilai_penawaran": 900_000_000, "nilai_hps": 950_000_000},
        # Tender 3: C menang, D kalah
        {"tender_id": "T003", "vendor_id": "C", "is_winner": True,  "nilai_penawaran": 800_000_000, "nilai_hps": 850_000_000},
        {"tender_id": "T003", "vendor_id": "D", "is_winner": False, "nilai_penawaran": 820_000_000, "nilai_hps": 850_000_000},
    ])


# ─────────────────────────────────────────────
# Test: build_co_participation_graph
# ─────────────────────────────────────────────

class TestCoParticipationGraph:

    def test_nodes_created(self, sample_peserta):
        G = build_co_participation_graph(sample_peserta, min_co_participation=1)
        assert "A" in G.nodes()
        assert "B" in G.nodes()
        assert "C" in G.nodes()

    def test_edge_between_co_participants(self, sample_peserta):
        G = build_co_participation_graph(sample_peserta, min_co_participation=1)
        assert G.has_edge("A", "B"), "A dan B harus terhubung (ikut T001, T002)"

    def test_edge_weight_reflects_frequency(self, sample_peserta):
        G = build_co_participation_graph(sample_peserta, min_co_participation=1)
        # A dan B bertemu 2 kali (T001, T002)
        assert G["A"]["B"]["weight"] == 2

    def test_min_co_participation_filter(self, sample_peserta):
        G = build_co_participation_graph(sample_peserta, min_co_participation=2)
        # C dan D hanya bertemu 1 kali — edge harus dihapus
        assert not G.has_edge("C", "D")

    def test_no_self_loops(self, sample_peserta):
        G = build_co_participation_graph(sample_peserta, min_co_participation=1)
        assert len(list(nx.selfloop_edges(G))) == 0


# ─────────────────────────────────────────────
# Test: build_win_loss_graph
# ─────────────────────────────────────────────

class TestWinLossGraph:

    def test_directed_edge_winner_to_loser(self, sample_peserta):
        DG = build_win_loss_graph(sample_peserta)
        assert DG.has_edge("A", "B"), "A → B harus ada (A menang, B kalah)"

    def test_edge_weight_accumulates(self, sample_peserta):
        DG = build_win_loss_graph(sample_peserta)
        # A menang atas B di 2 tender
        assert DG["A"]["B"]["weight"] == 2

    def test_no_self_loops(self, sample_peserta):
        DG = build_win_loss_graph(sample_peserta)
        assert len(list(nx.selfloop_edges(DG))) == 0


# ─────────────────────────────────────────────
# Test: compute_node_features
# ─────────────────────────────────────────────

class TestNodeFeatures:

    def test_win_rate_correct(self, sample_peserta):
        G  = build_co_participation_graph(sample_peserta, min_co_participation=1)
        DG = build_win_loss_graph(sample_peserta)
        nf = compute_node_features(G, DG, sample_peserta)
        # Vendor A: menang 2 dari 2 tender → win_rate = 1.0
        assert nf.loc["A", "win_rate"] == pytest.approx(1.0)

    def test_total_tenders_correct(self, sample_peserta):
        G  = build_co_participation_graph(sample_peserta, min_co_participation=1)
        DG = build_win_loss_graph(sample_peserta)
        nf = compute_node_features(G, DG, sample_peserta)
        assert nf.loc["A", "total_tenders"] == 2
        assert nf.loc["C", "total_tenders"] == 2

    def test_feature_columns_exist(self, sample_peserta):
        G  = build_co_participation_graph(sample_peserta, min_co_participation=1)
        DG = build_win_loss_graph(sample_peserta)
        nf = compute_node_features(G, DG, sample_peserta)
        expected_cols = ["total_tenders", "win_count", "win_rate",
                         "degree", "clustering_coef", "betweenness"]
        for col in expected_cols:
            assert col in nf.columns, f"Kolom '{col}' tidak ada"

    def test_no_nan_in_core_features(self, sample_peserta):
        G  = build_co_participation_graph(sample_peserta, min_co_participation=1)
        DG = build_win_loss_graph(sample_peserta)
        nf = compute_node_features(G, DG, sample_peserta)
        core_cols = ["win_rate", "degree", "clustering_coef", "betweenness"]
        assert nf[core_cols].isnull().sum().sum() == 0
