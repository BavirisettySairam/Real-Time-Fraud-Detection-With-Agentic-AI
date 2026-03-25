"""Shared fixtures for unit tests."""

import numpy as np
import pandas as pd
import pytest

from ml.preprocessing import FeaturePipeline


@pytest.fixture(scope="session")
def pipeline() -> FeaturePipeline:
    """Load the fitted FeaturePipeline once per test session."""
    return FeaturePipeline.load("models/feature_pipeline.pkl")


def preprocess(pipeline: FeaturePipeline, txn: dict) -> np.ndarray:
    """Transform a single transaction dict → 1-D float32 feature array."""
    df = pipeline.transform(pd.DataFrame([txn]))
    return df.values[0].astype(np.float32)
