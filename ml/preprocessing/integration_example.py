"""
Integration Guide: FeaturePipeline into your project
=====================================================

This file shows exactly how to integrate the preprocessing pipeline
into your existing train_agents.py and API inference path.

Run this script standalone to verify the pipeline works on your data:
    python ml/preprocessing/integration_example.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ══════════════════════════════════════════════
# PART 1: HOW TO MODIFY train_agents.py
# ══════════════════════════════════════════════
"""
In your train_agents.py, BEFORE training any agent, add:

    from ml.preprocessing import FeaturePipeline

    # Load raw data
    df_train = pd.read_csv("data/final_transaction.csv")
    # ... merge with identity if needed ...
    
    y_train = df_train["isFraud"]

    # Fit and transform
    pipeline = FeaturePipeline()
    X_train = pipeline.fit_transform(df_train, y_train)
    
    # Save the fitted pipeline (CRITICAL — inference needs this)
    pipeline.save("models/feature_pipeline.pkl")
    
    # Now train your agents on X_train instead of df_train
    # The feature names are in pipeline.get_feature_names()

For validation/test splits:
    
    X_val = pipeline.transform(df_val)    # uses fitted params, no leakage
    X_test = pipeline.transform(df_test)  # uses fitted params, no leakage

IMPORTANT: 
- fit() only on training data. transform() on val/test.
- This prevents data leakage (medians, target encodings, freq counts all from train only).
"""


# ══════════════════════════════════════════════
# PART 2: HOW TO MODIFY THE API INFERENCE PATH
# ══════════════════════════════════════════════
"""
In your services/api_gateway/main.py or wherever the orchestrator is initialized:

    from ml.preprocessing import FeaturePipeline
    
    # Load once at startup (in FastAPI lifespan)
    pipeline = FeaturePipeline.load("models/feature_pipeline.pkl")
    
    # In your predict endpoint, BEFORE passing to orchestrator:
    async def predict(request: TransactionRequest):
        raw_df = pd.DataFrame([request.dict()])
        processed_df = pipeline.transform(raw_df)
        
        # Now pass processed_df to your agents
        result = orchestrator.analyze(processed_df.iloc[0].to_dict())
        ...

For batch endpoint:
    async def predict_batch(requests: List[TransactionRequest]):
        raw_df = pd.DataFrame([r.dict() for r in requests])
        processed_df = pipeline.transform(raw_df)
        
        results = []
        for idx, row in processed_df.iterrows():
            result = orchestrator.analyze(row.to_dict())
            results.append(result)
        ...
"""


# ══════════════════════════════════════════════
# PART 3: HOW TO MODIFY EACH AGENT
# ══════════════════════════════════════════════
"""
Your agents currently expect raw features. After integration, they'll receive
preprocessed features. You need to:

1. RETRAIN all models on preprocessed features (the feature set will be different)
2. Update each agent's analyze() to accept the preprocessed dict directly
   (no internal feature engineering — the pipeline handles it)

Example for Vibe Checker:
    
    class VibeChecker:
        def analyze(self, txn: dict) -> float:
            # BEFORE: txn has raw fields, agent does its own transforms
            # AFTER:  txn has preprocessed fields from pipeline
            
            features = [txn.get(f, 0) for f in self.feature_names]
            score = self.model.predict_proba([features])[0][1]
            return score
"""


# ══════════════════════════════════════════════
# PART 4: VERIFICATION SCRIPT
# ══════════════════════════════════════════════

def verify_pipeline():
    """
    Run this to verify the pipeline works on your actual data.
    """
    import pandas as pd
    import numpy as np
    from ml.preprocessing import FeaturePipeline
    
    print("=" * 60)
    print("Feature Pipeline Verification")
    print("=" * 60)
    
    # ── Load data (split across transaction + identity files) ──
    txn_path = Path("data/train_transaction.csv")
    id_path = Path("data/train_identity.csv")

    if not txn_path.exists():
        print(f"ERROR: {txn_path} not found")
        return False

    print(f"Loading transactions from {txn_path} (all rows)...")
    train_txn = pd.read_csv(txn_path)

    if id_path.exists():
        print(f"Loading identity from {id_path}...")
        train_id = pd.read_csv(id_path)
        df = train_txn.merge(train_id, on="TransactionID", how="left")
        print(f"Merged: {df.shape[0]} rows, {df.shape[1]} columns")
    else:
        print(f"WARNING: {id_path} not found — using transactions only")
        df = train_txn
        print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Check for isFraud column
    if "isFraud" not in df.columns:
        print("ERROR: 'isFraud' column not found in data")
        return False
    
    y = df["isFraud"]
    print(f"Fraud rate: {y.mean():.4f}")
    
    # ── Fit pipeline ──
    print("\nFitting pipeline...")
    pipeline = FeaturePipeline()
    X = pipeline.fit_transform(df, y)
    
    print(f"\n{'-' * 40}")
    print("Pipeline Summary:")
    summary = pipeline.get_pipeline_summary()
    for step, count in summary["original_to_baseline"].items():
        print(f"  {step}: {count}")
    print(f"\nEngineered features:")
    for feat_type, count in summary["engineered_features"].items():
        print(f"  {feat_type}: {count}")
    print(f"\nTotal features: {summary['total_features']}")
    print(f"{'-' * 40}")
    
    # ── Verify output ──
    assert X.shape[0] == len(df), f"Row count mismatch: {X.shape[0]} vs {len(df)}"
    assert X.isnull().sum().sum() == 0, "NaN values found in output"
    assert np.isinf(X.values).sum() == 0, "Inf values found in output"
    assert X.shape[1] > 0, "No features in output"
    print(f"\n✓ Output shape: {X.shape}")
    print(f"✓ No NaN values")
    print(f"✓ No Inf values")
    
    # ── Save and reload ──
    save_path = "models/feature_pipeline.pkl"
    pipeline.save(save_path)
    print(f"\n✓ Saved to {save_path}")
    
    pipeline2 = FeaturePipeline.load(save_path)
    print(f"✓ Reloaded from {save_path}")
    
    # ── Verify single-row transform (simulates inference) ──
    single_row = df.iloc[[0]]
    X_single = pipeline2.transform(single_row)
    assert X_single.shape[0] == 1, "Single-row transform failed"
    assert X_single.shape[1] == X.shape[1], (
        f"Feature count mismatch: single={X_single.shape[1]} vs batch={X.shape[1]}"
    )
    print(f"✓ Single-row transform works ({X_single.shape[1]} features)")
    
    # ── Verify transform on data with missing columns (simulates API input) ──
    sparse_input = pd.DataFrame({
        "TransactionAmt": [150.0],
        "TransactionDT": [86400],
        "card1": [1234],
        "card4": ["visa"],
        "card6": ["debit"],
        "ProductCD": ["W"],
    })
    X_sparse = pipeline2.transform(sparse_input)
    assert X_sparse.shape[0] == 1, "Sparse input transform failed"
    assert X_sparse.isnull().sum().sum() == 0, "NaN in sparse input output"
    print(f"✓ Sparse input transform works ({X_sparse.shape[1]} features)")
    
    print(f"\n{'=' * 60}")
    print("ALL CHECKS PASSED")
    print(f"{'=' * 60}")
    print(f"\nNext steps:")
    print(f"1. Run full training: python train_agents.py --agent all --folds 5")
    print(f"   (after integrating pipeline into train_agents.py)")
    print(f"2. Verify models load and score correctly")
    print(f"3. Update API inference path to use pipeline.transform()")
    return True


if __name__ == "__main__":
    success = verify_pipeline()
    sys.exit(0 if success else 1)
