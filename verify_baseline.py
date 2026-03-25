"""Quick verification: baseline-only pipeline on full data."""
import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from ml.preprocessing import FeaturePipeline

print("=" * 60)
print("Baseline-Only Pipeline Verification (full data)")
print("=" * 60)

print("Loading transactions (all rows)...")
txn = pd.read_csv("data/train_transaction.csv")
print(f"  Transactions: {txn.shape}")

print("Loading identity...")
idf = pd.read_csv("data/train_identity.csv")
df = txn.merge(idf, on="TransactionID", how="left")
del txn, idf
print(f"  Merged: {df.shape[0]} rows, {df.shape[1]} columns")

y = df["isFraud"]
print(f"  Fraud rate: {y.mean():.4f}")

print("\nFitting pipeline (steps 1-6 only, no engineering)...")
pipeline = FeaturePipeline()
X = pipeline.fit_transform(df, y)

summary = pipeline.get_pipeline_summary()
print("\nPipeline filtering steps:")
for step, count in summary["original_to_baseline"].items():
    print(f"  {step}: {count}")

total = summary["total_features"]
print(f"\nTotal output features: {total}")
print(f"Output shape: {X.shape}")

nan_count = X.isnull().sum().sum()
inf_count = np.isinf(X.values).sum()
print(f"NaN count: {nan_count}")
print(f"Inf count: {inf_count}")

assert nan_count == 0, "NaN values found!"
assert inf_count == 0, "Inf values found!"

pipeline.save("models/feature_pipeline.pkl")
print("\nSaved to models/feature_pipeline.pkl")

pipeline2 = FeaturePipeline.load("models/feature_pipeline.pkl")
xs = pipeline2.transform(df.iloc[[0]])
print(f"Single-row transform: {xs.shape[1]} features")
print(f"Feature count match: {xs.shape[1] == X.shape[1]}")

assert xs.shape[1] == X.shape[1], "Mismatch!"

print("\n" + "=" * 60)
print("ALL CHECKS PASSED")
print("=" * 60)
