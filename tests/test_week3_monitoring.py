import numpy as np
import pandas as pd
import pytest

from src.week3_monitoring import calculate_business_cost, calculate_psi, drift_summary


def test_business_cost_and_slider_sensitivity():
    y = [0, 0, 1, 1]
    pred = [1, 0, 0, 1]
    amounts = [100, 20, 35, 10]
    low = calculate_business_cost(y, pred, amounts, fp_cost=2)
    high = calculate_business_cost(y, pred, amounts, fp_cost=5)
    assert low["fp"] == 1
    assert low["fn"] == 1
    assert low["total_estimated_cost"] == pytest.approx(37)
    assert high["total_estimated_cost"] - low["total_estimated_cost"] == pytest.approx(3)


def test_drift_summary_schema():
    ref = pd.DataFrame({"amount": [1, 2, 3, 4, 5, 6]})
    current = pd.DataFrame({"amount": [10, 20, 30, 40, 50, 60]})
    result = drift_summary(ref, current, ["amount"])
    assert len(result) == 1
    assert {"feature", "psi", "ks_p_value", "psi_level"}.issubset(result.columns)
    assert result.loc[0, "feature"] == "amount"


def test_constant_feature_psi_is_not_calculated():
    assert np.isnan(calculate_psi([1, 1, 1], [1, 1, 2]))
