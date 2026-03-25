# Training Workflow

Complete reference for the model training pipeline, dataset preparation, feature engineering, and evaluation methodology.

**Author:** Bavirisetty Sairam
**Last Updated:** March 2026

---

## Dataset

**Source:** IEEE-CIS Fraud Detection (Kaggle)
**Records:** 590,540 transactions with identity data

### Split Strategy

Stratified 60/20/20 split on `isFraud` to preserve the ~3.5% fraud rate across all sets:

| Set | Records | Fraud Rate | Files |
|---|---:|---:|---|
| Train | 354,324 | 3.50% | `data/train_transaction.csv`, `data/train_identity.csv` |
| Validation | 118,108 | 3.50% | `data/val_transaction.csv`, `data/val_identity.csv` |
| Test | 118,108 | 3.50% | `data/test_transaction.csv`, `data/test_identity.csv` |

The original Kaggle test set (unlabelled) is retained as `data/final_transaction.csv` and `data/final_identity.csv` for submission generation.

---

## Preprocessing Pipeline

All agents share a common `FeaturePipeline` (`ml/preprocessing/feature_pipeline.py`) that transforms raw transaction + identity data into 175 clean features. The pipeline is fitted once during training and serialised to `models/feature_pipeline.pkl` for inference.

### Pipeline Steps

1. **NaN / Inf handling** — Replace `inf`/`-inf` with `NaN`, drop columns > 90% missing.
2. **Imputation** — Median for numeric, mode for categorical.
3. **Target encoding** — Smoothed target encoding for high-cardinality categoricals (card1–6, addr1–2, P/R_emaildomain, etc.).
4. **Feature engineering** — `TransactionAmt_log`, `TransactionAmt_cents`, time features (`hour`, `day`, `is_weekend`, `is_night`), interaction features, missing-count indicators.
5. **Feature selection** — Remove zero-variance and highly correlated (>0.95) features.

### Output

| Metric | Value |
|---|---|
| Input columns (raw) | 209 baseline |
| Output features | 175 |
| Artifact | `models/feature_pipeline.pkl` |

Each agent then adds its own specialised features on top of the 175 pipeline features:
- **Vibe Checker**: Uses 175 pipeline features directly.
- **Era Tracker**: 175 pipeline + 24 sliding-window behavioural features = **199 total**.
- **OG Check**: 175 pipeline + 19 rule-engineered features = **194 total**.

---

## Training Pipeline

All training is handled by `train_agents.py` with CLI arguments:

```bash
# Train all agents
python train_agents.py --agent all --folds 5

# Train individual agent
python train_agents.py --agent vibe --folds 5
python train_agents.py --agent era --folds 3
python train_agents.py --agent og --folds 5
```

### Outputs

Each training run produces:
- Model artifacts in `models/`
- Metrics appended to `logs/training_metrics.log` (timestamped, includes confusion matrix)
- Threshold optimised via F1 grid search on validation set

---

## Agent Training Details

### Vibe Checker (LightGBM + XGBoost Ensemble)

| Parameter | Value |
|---|---|
| LightGBM objective | `binary` (log loss) |
| XGBoost objective | `binary:logistic` |
| Cross-validation | Stratified K-Fold |
| Feature count | 175 (from FeaturePipeline) |
| Blend weights | 90% LightGBM / 10% XGBoost (optimised via grid search) |

**Feature engineering:**
- All 175 features from FeaturePipeline (see Preprocessing Pipeline section above)
- Includes `TransactionAmt_log`, `TransactionAmt_sqrt`, `TransactionAmt_cents`
- Time features: `hour`, `day`, `is_weekend`, `is_night`
- Target-encoded categoricals, interaction features, missing-count indicators

**Artifacts:** `models/vibe_lgb.txt`, `models/vibe_xgb.json`, `models/vibe_metrics.json`

### Era Tracker (CatBoost)

| Parameter | Value |
|---|---|
| Model | CatBoostClassifier |
| Class weights | `auto_class_weights="Balanced"` |
| Depth | 8 |
| Iterations | 2000 (early stopping) |
| Feature count | 175 pipeline + 24 sliding-window = 199 |

**Numeric features (25 sliding-window + 175 pipeline):**
Pipeline provides 175 base features. Era Tracker adds 24 sliding-window behavioural features:
`amt_zscore_user`, `amt_ratio_user_mean`, `amt_ratio_user_median`, `amt_ratio_user_max`, `amt_log`, `user_txn_count_24h`, `user_txn_count_1h`, `user_total_amt_24h`, `user_mean_amt_24h`, `user_max_amt_24h`, `user_std_amt_24h`, `time_since_last`, `avg_gap_24h`, `min_gap_24h`, `hour_sin`, `hour_cos`, `is_night`, `is_weekend`, `rapid_succession`, `is_new_user`, `night_first_time`, `increasing_amounts`, `hour_deviation`, `product_diversity_24h`

**Categorical features (3):** `ProductCD`, `card4`, `card6`

**Artifacts:** `models/era_tracker_catboost.cbm`, `models/era_tracker_metrics.json`

### OG Check (LightGBM on Rule Features)

| Parameter | Value |
|---|---|
| Model | LightGBM |
| Feature count | 175 pipeline + 19 rule-engineered = 194 |
| Threshold | Optimised via F1 grid search |

**Features (19 rule-engineered + 175 pipeline):**
Pipeline provides 175 base features. OG Check adds 19 rule-engineered features:
12 binary rule indicators (`HIGH_AMOUNT`, `MAX_AMOUNT`, `LATE_NIGHT`, `MISSING_EMAIL`, `MISSING_ADDR`, `MISSING_DEVICE`, `VELOCITY_TXN_COUNT`, `VELOCITY_AMOUNT_1H`, `CARD_FREQ_LOW`, `AMOUNT_ROUND`, `SUSPICIOUS_DOMAIN`, `RISKY_PRODUCT`) + 7 engineered (`addr2_missing`, `D1_missing`, `D1_high`, `C1_high`, `C13_high`, `M_mismatch_count`, `id_missing`)

**Artifacts:** `models/og_check_lgb.txt`, `models/og_check_params.json`

---

## Current Metrics (Validation Set — 118,108 transactions)

| Agent | ROC-AUC | PR-AUC | Precision | Recall | F1 | Threshold |
|---|---:|---:|---:|---:|---:|---:|
| Vibe Checker (Ensemble) | 0.8991 | 0.6001 | 0.7714 | 0.4794 | 0.5913 | 0.8109 |
| Vibe Checker (LGB only) | 0.9699 | 0.8585 | 0.8782 | 0.7592 | 0.8144 | 0.4626 |
| Vibe Checker (XGB only) | 0.9669 | 0.8142 | 0.6118 | 0.8100 | 0.6971 | 0.4626 |
| OG Check | 0.8903 | 0.5352 | 0.6678 | 0.4417 | 0.5317 | 0.8144 |
| Era Tracker (CatBoost) | 0.8773 | 0.5162 | 0.6572 | 0.4240 | 0.5154 | 0.8063 |

### Confusion Matrices

**Vibe Checker (Ensemble):**
```
TN = 113,517    FP =    459
FN =    974     TP =  3,158
```

**OG Check:**
```
TN = 111,174    FP =  2,802
FN =   3,151    TP =    981
```

**Era Tracker (CatBoost):**
```
TN = 108,482    FP =  5,494
FN =   2,817    TP =  1,315
```

---

## Metric Progression

Training iterations documenting progressive improvements:

### Era Tracker

| Version | Model | ROC-AUC | PR-AUC | F1 | Recall |
|---|---|---:|---:|---:|---:|
| v1 | LightGBM (18 features) | 0.7077 | 0.0893 | 0.0985 | 0.073 |
| v2 | LightGBM (26 features, velocity) | 0.7077 | 0.0893 | 0.0985 | 0.073 |
| v3 | CatBoost (28 features) | 0.7813 | 0.1612 | 0.2404 | 0.318 |
| **v4** | **CatBoost (199 features, pipeline)** | **0.8773** | **0.5162** | **0.5154** | **0.424** |

Key change in v3: switched to CatBoost with native categorical handling, added circular time encoding, redesigned behavioural feature set.
Key change in v4: retrained on 175 FeaturePipeline features + 24 sliding-window features. PR-AUC +220%, F1 +114%.

### OG Check

| Version | Model | ROC-AUC | PR-AUC | F1 |
|---|---|---:|---:|---:|
| v1 | Logistic Regression (12 rules) | 0.7039 | 0.0924 | 0.0086 |
| v2 | LightGBM (20 features) | 0.7833 | 0.2311 | 0.2479 |
| **v3** | **LightGBM (194 features, pipeline)** | **0.8903** | **0.5352** | **0.5317** |

Key change in v2: replaced logistic regression with LightGBM, added 8 engineered features from transaction data signals.
Key change in v3: retrained on 175 FeaturePipeline features + 19 rule features. PR-AUC +132%, F1 +114%.

---

## Metrics Logging

All training runs are logged to `logs/training_metrics.log` in append-only format:

```
========================================================================
  Era Tracker (CatBoost)  —  2026-03-25 03:07:08 UTC
========================================================================
  roc_auc     : 0.781287
  pr_auc      : 0.161170
  precision   : 0.193127
  recall      : 0.318248
  f1          : 0.240380
  threshold   : 0.758730
  Confusion Matrix:
    TN= 108482  FP=   5494
    FN=   2817  TP=   1315
```

This file serves as the authoritative training history and is never overwritten.

---

## Retraining

To retrain after dataset changes:

1. Prepare the dataset split:
   ```bash
   python train_agents.py --agent all --folds 5
   ```

2. Verify metrics in `logs/training_metrics.log`.

3. Run unit tests:
   ```bash
   python -m pytest tests/unit -v
   ```

4. Verify SHAP compatibility:
   ```bash
   curl -X POST http://localhost:8000/api/v1/explain \
     -H "Content-Type: application/json" \
     -d '{"TransactionAmt": 450, "ProductCD": "W"}'
   ```

5. Confirm `shap_available: true` in the response.
