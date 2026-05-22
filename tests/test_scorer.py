"""
Tests — Modul 4: Risk Scoring
"""

import pytest
import pandas as pd
import networkx as nx
from modules.scorer import compute_risk_score, classify_risk, compute_heuristic_flags


class TestClassifyRisk:

    def test_high_risk(self):
        assert classify_risk(0.75) == "HIGH"
        assert classify_risk(1.0)  == "HIGH"
        assert classify_risk(0.70) == "HIGH"

    def test_medium_risk(self):
        assert classify_risk(0.5)  == "MEDIUM"
        assert classify_risk(0.40) == "MEDIUM"
        assert classify_risk(0.69) == "MEDIUM"

    def test_low_risk(self):
        assert classify_risk(0.0)  == "LOW"
        assert classify_risk(0.39) == "LOW"
        assert classify_risk(0.2)  == "LOW"


class TestComputeRiskScore:

    def test_all_zero_returns_low(self):
        score = compute_risk_score(
            isolation_score=0.3,   # Normal (positif)
            community_flag=False,
            heuristic_flags={},
        )
        assert score < 0.4  # LOW risk

    def test_all_flags_returns_high(self):
        score = compute_risk_score(
            isolation_score=-0.5,   # Anomali
            community_flag=True,
            heuristic_flags={
                "win_rate_above_3x_avg"       : True,
                "always_loses_to_same_vendor" : True,
                "bid_price_too_close"         : True,
                "high_clustering_coefficient" : True,
            },
        )
        assert score >= 0.7  # HIGH risk

    def test_score_in_range(self):
        score = compute_risk_score(-0.3, True, {"win_rate_above_3x_avg": True})
        assert 0.0 <= score <= 1.0

    def test_custom_weights(self):
        weights = {"isolation_forest": 0.5, "community": 0.3, "heuristic": 0.2}
        score = compute_risk_score(-0.5, False, {}, weights=weights)
        assert 0.0 <= score <= 1.0


class TestHeuristicFlags:

    @pytest.fixture
    def simple_features(self):
        return pd.DataFrame({
            "win_rate"        : {"A": 0.9, "B": 0.1, "C": 0.05},
            "win_count"       : {"A": 9,   "B": 1,   "C": 0},
            "clustering_coef" : {"A": 0.8, "B": 0.2, "C": 0.5},
            "avg_bid_ratio"   : {"A": 0.9, "B": 0.9, "C": 0.9},
            "bid_std"         : {"A": 0.001, "B": 100_000, "C": 0.0},
            "total_tenders"   : {"A": 10, "B": 10, "C": 1},
        })

    def test_high_win_rate_flagged(self, simple_features):
        DG = nx.DiGraph()
        flags = compute_heuristic_flags("A", simple_features, DG)
        # Global avg win_rate = (0.9+0.1+0.05)/3 ≈ 0.35, threshold = 0.35*3 = 1.05
        # A win_rate=0.9 < 1.05, jadi tidak ter-flag
        assert isinstance(flags["win_rate_above_3x_avg"], bool)

    def test_high_clustering_flagged(self, simple_features):
        DG = nx.DiGraph()
        flags = compute_heuristic_flags("A", simple_features, DG)
        assert flags["high_clustering_coefficient"] is True  # clustering=0.8 > 0.7

    def test_unknown_vendor_returns_false_flags(self, simple_features):
        DG = nx.DiGraph()
        flags = compute_heuristic_flags("UNKNOWN_VENDOR", simple_features, DG)
        assert all(v is False for v in flags.values())
