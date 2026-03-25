"""Unit tests for FeaturePipeline — fit, transform, save/load, edge cases."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.preprocessing.feature_pipeline import FeaturePipeline


@pytest.fixture(scope="module")
def fitted_pipeline() -> FeaturePipeline:
    """Fit a pipeline on a small synthetic sample."""
    rng = np.random.RandomState(42)
    n = 200
    df = pd.DataFrame({
        "TransactionAmt": rng.exponential(100, n),
        "card1": rng.randint(1000, 9999, n),
        "card2": rng.choice([100, 200, np.nan], n),
        "card3": rng.choice([150, 185, np.nan], n),
        "card4": rng.choice(["visa", "mastercard", None], n),
        "card5": rng.choice([200, 226, np.nan], n),
        "card6": rng.choice(["debit", "credit", None], n),
        "ProductCD": rng.choice(["W", "H", "C", "S"], n),
        "addr1": rng.choice([300, 400, np.nan], n),
        "addr2": rng.choice([87, 60, np.nan], n),
        "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", None], n),
        "R_emaildomain": rng.choice(["gmail.com", None], n),
        "dist1": rng.choice([0, 50, np.nan], n),
        "C1": rng.randint(0, 5, n).astype(float),
        "C13": rng.randint(0, 5, n).astype(float),
        "D1": rng.choice([0, 14, np.nan], n),
        "D2": rng.choice([0, 14, np.nan], n),
        "V12": rng.rand(n),
        "V258": rng.rand(n),
    })
    y = pd.Series(rng.choice([0, 1], n, p=[0.95, 0.05]))
    pipeline = FeaturePipeline()
    pipeline.fit_transform(df, y)
    return pipeline


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

class TestPipelineFit:
    def test_is_fitted(self, fitted_pipeline: FeaturePipeline):
        assert fitted_pipeline.is_fitted is True

    def test_final_feature_names_populated(self, fitted_pipeline: FeaturePipeline):
        assert len(fitted_pipeline.final_feature_names_) > 0

    def test_all_feature_names_are_strings(self, fitted_pipeline: FeaturePipeline):
        for name in fitted_pipeline.final_feature_names_:
            assert isinstance(name, str)


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------

class TestPipelineTransform:
    def test_no_nan(self, fitted_pipeline: FeaturePipeline):
        df = pd.DataFrame([{"TransactionAmt": 100.0, "card1": 1234}])
        result = fitted_pipeline.transform(df)
        assert not result.isnull().any().any(), "Transform output contains NaN"

    def test_no_inf(self, fitted_pipeline: FeaturePipeline):
        df = pd.DataFrame([{"TransactionAmt": 100.0, "card1": 1234}])
        result = fitted_pipeline.transform(df)
        arr = result.values
        assert np.isfinite(arr).all(), "Transform output contains Inf"

    def test_feature_count_matches(self, fitted_pipeline: FeaturePipeline):
        df = pd.DataFrame([{"TransactionAmt": 100.0}])
        result = fitted_pipeline.transform(df)
        assert result.shape[1] == len(fitted_pipeline.final_feature_names_)

    def test_single_row_matches_batch(self, fitted_pipeline: FeaturePipeline):
        txn = {"TransactionAmt": 250.0, "card1": 5555, "card4": "visa"}
        single = fitted_pipeline.transform(pd.DataFrame([txn]))
        batch = fitted_pipeline.transform(pd.DataFrame([txn, txn]))
        np.testing.assert_array_almost_equal(
            single.values[0], batch.values[0],
            err_msg="Single-row and batch-row outputs differ",
        )


# ---------------------------------------------------------------------------
# Sparse / empty input
# ---------------------------------------------------------------------------

class TestPipelineEdgeCases:
    def test_sparse_input(self, fitted_pipeline: FeaturePipeline):
        """Only TransactionAmt + card1 — shouldn't crash."""
        df = pd.DataFrame([{"TransactionAmt": 42.0, "card1": 9999}])
        result = fitted_pipeline.transform(df)
        assert result.shape == (1, len(fitted_pipeline.final_feature_names_))
        assert np.isfinite(result.values).all()

    def test_completely_empty_row(self, fitted_pipeline: FeaturePipeline):
        """Empty dict — should produce a full-width row of zeros (no crash)."""
        df = pd.DataFrame([{}])
        result = fitted_pipeline.transform(df)
        assert result.shape == (1, len(fitted_pipeline.final_feature_names_))
        # All imputed to median or 0 — no NaN/Inf
        assert np.isfinite(result.values).all()


# ---------------------------------------------------------------------------
# Save / load roundtrip
# ---------------------------------------------------------------------------

class TestPipelineSaveLoad:
    def test_roundtrip_identical_output(self, fitted_pipeline: FeaturePipeline):
        txn = {"TransactionAmt": 300.0, "card1": 7777, "card4": "mastercard"}
        before = fitted_pipeline.transform(pd.DataFrame([txn]))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "pipeline.pkl")
            fitted_pipeline.save(path)
            loaded = FeaturePipeline.load(path)

        after = loaded.transform(pd.DataFrame([txn]))
        np.testing.assert_array_equal(
            before.values, after.values,
            err_msg="Save/load roundtrip changed transform output",
        )

    def test_loaded_feature_names_match(self, fitted_pipeline: FeaturePipeline):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "pipeline.pkl")
            fitted_pipeline.save(path)
            loaded = FeaturePipeline.load(path)
        assert loaded.final_feature_names_ == fitted_pipeline.final_feature_names_
