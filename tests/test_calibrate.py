"""Tests for the selective calibration layer (calibrate.py)."""

import json
import math

from haruspex.calibrate import (
    GRID, SelectiveCalibrator, _binom_cdf, _fit_set, _fit_threshold,
    fit_temperature,
)


def test_binom_cdf_edge_cases():
    assert _binom_cdf(0, 0, 0.5) == 1.0
    assert _binom_cdf(10, 10, 0.0) == 1.0
    assert abs(_binom_cdf(5, 10, 0.5) - 0.623) < 0.01


def test_binom_cdf_certain():
    assert _binom_cdf(0, 10, 1.0) == 0.0
    assert _binom_cdf(10, 10, 1.0) == 1.0


def test_fit_threshold_perfect():
    data = [(0.9, True)] * 50
    result = _fit_threshold(data, target_risk=0.1, delta=0.1)
    assert result["threshold"] is not None
    assert result["achieved_risk"] == 0.0
    assert result["coverage"] > 0


def test_fit_threshold_no_data():
    result = _fit_threshold([], target_risk=0.1, delta=0.1)
    assert result["threshold"] is None


def test_fit_threshold_all_wrong():
    data = [(0.9, False)] * 50
    result = _fit_threshold(data, target_risk=0.05, delta=0.05)
    assert result["threshold"] is None  # cannot certify any threshold


def test_selective_calibrator_fit_pooled():
    # All correct at high confidence: threshold must certify
    data = [(0.95, True)] * 48 + [(0.2, True)] * 2
    cal = SelectiveCalibrator.fit(data)
    assert cal.threshold is not None
    assert cal.n == 50


def test_selective_calibrator_fit_typed():
    data = [(0.9, True, "boolean")] * 30 + [(0.8, True, "choice")] * 20
    cal = SelectiveCalibrator.fit(data)
    assert cal.by_type is not None
    assert "boolean" in cal.by_type
    assert "choice" in cal.by_type


def test_apply_accept():
    cal = SelectiveCalibrator(threshold=0.7, target_risk=0.1)
    result = cal.apply(0.85)
    assert result["accept"] is True
    assert result["abstain"] is False


def test_apply_abstain():
    cal = SelectiveCalibrator(threshold=0.7, target_risk=0.1)
    result = cal.apply(0.5)
    assert result["accept"] is False
    assert result["abstain"] is True


def test_apply_no_threshold():
    cal = SelectiveCalibrator(threshold=None, target_risk=0.1)
    result = cal.apply(0.99)
    assert result["accept"] is False  # no threshold = abstain on everything


def test_apply_typed():
    cal = SelectiveCalibrator(None, 0.1, by_type={"boolean": {"threshold": 0.6},
                                                    "choice": {"threshold": 0.8}})
    assert cal.apply(0.7, "boolean")["accept"] is True
    assert cal.apply(0.7, "choice")["accept"] is False


def test_temperature_for():
    cal = SelectiveCalibrator(0.7, 0.1, temperature=2.0, temperature_scope="all")
    assert cal.temperature_for("boolean") == 2.0
    assert cal.temperature_for("choice") == 2.0

    cal2 = SelectiveCalibrator(0.7, 0.1, temperature=2.0, temperature_scope="boolean")
    assert cal2.temperature_for("boolean") == 2.0
    assert cal2.temperature_for("choice") == 1.0


def test_prediction_set():
    cal = SelectiveCalibrator(0.7, 0.1, set_q=0.4)
    probs = {"yes": 0.8, "no": 0.2}
    ps = cal.prediction_set(probs)
    assert ps is not None
    assert "yes" in ps


def test_prediction_set_none():
    cal = SelectiveCalibrator(0.7, 0.1)  # no set_q
    assert cal.prediction_set({"yes": 0.8, "no": 0.2}) is None


def test_fit_set():
    probs = [0.9, 0.95, 0.85, 0.92, 0.88, 0.91, 0.87, 0.93, 0.86, 0.94]
    q = _fit_set(probs, alpha=0.1)
    assert q is not None
    assert 0.0 <= q <= 1.0


def test_fit_set_empty():
    assert _fit_set([], 0.1) is None


def test_save_load(tmp_path):
    cal = SelectiveCalibrator.fit([(0.9, True)] * 30 + [(0.3, False)] * 10)
    path = tmp_path / "calibrator.json"
    cal.save(path)
    loaded = SelectiveCalibrator.load(path)
    assert loaded.threshold == cal.threshold
    assert loaded.target_risk == cal.target_risk
    assert loaded.n == cal.n


def test_fit_temperature_no_data():
    assert fit_temperature([]) == 1.0


def test_grid_is_sorted():
    assert list(GRID) == sorted(GRID)
    assert GRID[0] == 0.0
