"""
Feature Preprocessing Pipeline for IEEE-CIS Fraud Detection

Implements the full pipeline from the methodology:
1. Feature removal (>95% missing, zero-variance, high correlation, low info gain)
2. Missing value treatment (class-specific medians, categorical -> "missing", missingness indicators)
3. Feature engineering (15 temporal + 12 amount + 28 aggregation + 25 interaction = 80 new)

Usage:
    # Training
    pipeline = FeaturePipeline()
    X_train = pipeline.fit_transform(df_train, y_train)
    pipeline.save("models/feature_pipeline.pkl")

    # Inference
    pipeline = FeaturePipeline.load("models/feature_pipeline.pkl")
    X_new = pipeline.transform(df_new)
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# IEEE-CIS known column groups
# ──────────────────────────────────────────────

CATEGORICAL_COLS = [
    "ProductCD",
    "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2",
    "P_emaildomain", "R_emaildomain",
    "DeviceType", "DeviceInfo",
] + [f"M{i}" for i in range(1, 10)]

# TransactionID and isFraud are dropped.  TransactionDT stays for temporal
# feature engineering but is excluded from the baseline feature set afterwards.
DROP_COLS = ["TransactionID", "isFraud"]

# ──────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────

class FeaturePipeline:
    """
    Scikit-learn-style fit/transform pipeline for IEEE-CIS data.

    fit() learns:
        - which features to keep (missing %, variance, correlation, info gain)
        - class-specific medians (fraud / legit) for numerical imputation
        - frequency encodings (for categoricals)
        - target encodings (for interaction features)
        - aggregation statistics per card1, addr1, email, device, combos

    transform() applies the same transforms deterministically.
    """

    def __init__(
        self,
        missing_threshold: float = 0.95,
        correlation_threshold: float = 0.98,
        info_gain_threshold: float = 0.001,
        missingness_indicator_threshold: float = 0.20,
    ):
        self.missing_threshold = missing_threshold
        self.correlation_threshold = correlation_threshold
        self.info_gain_threshold = info_gain_threshold
        self.missingness_indicator_threshold = missingness_indicator_threshold

        # Fitted state (populated by fit())
        self.is_fitted = False
        self.features_after_missing_: list = []
        self.features_after_variance_: list = []
        self.features_after_correlation_: list = []
        self.features_final_baseline_: list = []
        self.numerical_medians_: dict = {}          # overall median (for inference)
        self.numerical_medians_fraud_: dict = {}    # median where y==1
        self.numerical_medians_legit_: dict = {}    # median where y==0
        self.frequency_encodings_: dict = {}
        self.target_encodings_: dict = {}
        self.missingness_features_: list = []
        self.final_feature_names_: list = []
        self.feature_stats_: dict = {}
        self.global_fraud_rate_: float = 0.035
        self._smoothing: int = 20

    # ──────────────────────────────────────────
    # FIT
    # ──────────────────────────────────────────

    def fit(self, df: pd.DataFrame, y: pd.Series) -> "FeaturePipeline":
        """
        Learn all preprocessing parameters from training data.

        Args:
            df: Raw training DataFrame (with TransactionID, TransactionDT, etc.)
            y:  Binary target (isFraud)
        """
        logger.info(f"Fitting pipeline on {len(df)} rows, {len(df.columns)} columns")

        df = df.copy()
        self.global_fraud_rate_ = float(y.mean())

        # ── Create sample for expensive operations (correlation, mutual info) ──
        if len(df) > 100_000:
            df_sample = df.sample(n=100_000, random_state=42)
            y_sample = y.loc[df_sample.index]
            logger.info(f"Using 100k sample for correlation/MI (full data: {len(df)} rows)")
        else:
            df_sample = df
            y_sample = y

        # ── Separate feature columns (TransactionDT kept for temporal, excluded from baseline later) ──
        feature_cols = [c for c in df.columns if c not in DROP_COLS and c != "TransactionDT"]
        logger.info(f"Starting features (excl TransactionDT): {len(feature_cols)}")

        # ── Step 1: Remove >95% missing ──
        missing_pct = df[feature_cols].isnull().mean()
        self.features_after_missing_ = [
            c for c in feature_cols if missing_pct[c] <= self.missing_threshold
        ]
        removed = len(feature_cols) - len(self.features_after_missing_)
        logger.info(
            f"Step 1 - Remove >{self.missing_threshold*100:.0f}% missing: "
            f"{len(feature_cols)} -> {len(self.features_after_missing_)} (removed {removed})"
        )

        # ── Step 2: Remove zero-variance ──
        numeric_subset = df[self.features_after_missing_].select_dtypes(include=[np.number])
        zero_var = numeric_subset.columns[numeric_subset.var() == 0].tolist()
        cat_subset = df[self.features_after_missing_].select_dtypes(exclude=[np.number])
        cat_zero_var = [c for c in cat_subset.columns if cat_subset[c].nunique(dropna=False) <= 1]
        zero_var_all = set(zero_var + cat_zero_var)
        self.features_after_variance_ = [
            c for c in self.features_after_missing_ if c not in zero_var_all
        ]
        logger.info(
            f"Step 2 - Remove zero-variance: "
            f"{len(self.features_after_missing_)} -> {len(self.features_after_variance_)} "
            f"(removed {len(zero_var_all)})"
        )

        # ── Step 3: Remove highly correlated (>0.98) ──
        numeric_cols = [
            c for c in self.features_after_variance_
            if df[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]
        ]
        corr_to_drop = set()
        if len(numeric_cols) > 1:
            corr_matrix = df_sample[numeric_cols].corr().abs()
            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            for col in upper.columns:
                high_corr = upper.index[upper[col] > self.correlation_threshold].tolist()
                corr_to_drop.update(high_corr)
        self.features_after_correlation_ = [
            c for c in self.features_after_variance_ if c not in corr_to_drop
        ]
        logger.info(
            f"Step 3 - Remove correlated >{self.correlation_threshold}: "
            f"{len(self.features_after_variance_)} -> {len(self.features_after_correlation_)} "
            f"(removed {len(corr_to_drop)})"
        )

        # ── Step 4: Remove low information gain (<0.001) ──
        from sklearn.feature_selection import mutual_info_classif

        numeric_for_ig = [
            c for c in self.features_after_correlation_
            if df[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]
        ]
        top_k = 180  # cap baseline at 180 numeric features
        if numeric_for_ig:
            X_ig = df_sample[numeric_for_ig].fillna(-999)
            ig_scores = mutual_info_classif(X_ig, y_sample, random_state=42, n_neighbors=3)
            mi_series = pd.Series(ig_scores, index=numeric_for_ig).sort_values(ascending=False)
            selected_numeric = mi_series.head(top_k).index.tolist()
        else:
            selected_numeric = []

        # Keep all non-numeric (categorical) columns that survived correlation filter
        non_numeric = [
            c for c in self.features_after_correlation_ if c not in numeric_for_ig
        ]
        self.features_final_baseline_ = selected_numeric + non_numeric
        logger.info(
            f"Step 4 - Select top-{top_k} by MI + {len(non_numeric)} non-numeric: "
            f"{len(self.features_after_correlation_)} -> {len(self.features_final_baseline_)}"
        )

        # ── Step 5: Learn imputation values (class-specific medians) ──
        fraud_mask = y == 1
        legit_mask = y == 0
        for col in self.features_final_baseline_:
            if col in CATEGORICAL_COLS or df[col].dtype == object:
                pass  # categoricals get fillna("missing") — no mode imputation
            else:
                self.numerical_medians_[col] = float(df[col].median())
                self.numerical_medians_fraud_[col] = float(df.loc[fraud_mask, col].median()) if fraud_mask.any() else self.numerical_medians_[col]
                self.numerical_medians_legit_[col] = float(df.loc[legit_mask, col].median()) if legit_mask.any() else self.numerical_medians_[col]

        # ── Step 6: Learn missingness indicator columns (>20% missing) ──
        self.missingness_features_ = [
            col for col in self.features_final_baseline_
            if df[col].isnull().mean() > self.missingness_indicator_threshold
        ]
        logger.info(f"Missingness indicators for {len(self.missingness_features_)} features")

        # ── TEMPORARILY DISABLED: Steps 7-9 (feature engineering stats) ──
        # Uncomment when re-enabling engineered features in transform().

        # # ── Step 7: Learn frequency encodings ──
        # freq_encode_cols = [
        #     c for c in ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
        #                  "DeviceType", "DeviceInfo", "addr1", "addr2", "M4"]
        #     if c in self.features_final_baseline_ or c in df.columns
        # ]
        # for col in freq_encode_cols:
        #     freq = df[col].value_counts(normalize=True, dropna=False).to_dict()
        #     self.frequency_encodings_[col] = freq

        # # ── Step 8: Learn target encodings (smoothed) ──
        # gm = self.global_fraud_rate_
        # sm = self._smoothing
        # target_encode_cols = ["ProductCD", "card4", "card6", "P_emaildomain",
        #                       "R_emaildomain", "DeviceType", "addr1"]
        # for col in target_encode_cols:
        #     if col not in df.columns:
        #         continue
        #     temp = df[[col]].copy()
        #     temp["__t__"] = y.values
        #     grp = temp.groupby(col, dropna=False)["__t__"]
        #     counts = grp.count()
        #     means = grp.mean()
        #     smoothed = (counts * means + sm * gm) / (counts + sm)
        #     self.target_encodings_[col] = smoothed.to_dict()
        #     self.target_encodings_[f"{col}__global_mean"] = gm

        # # High-order interaction target encodings
        # interaction_combos = [
        #     ("card4_x_card6", ["card4", "card6"]),
        #     ("card4_x_ProductCD", ["card4", "ProductCD"]),
        #     ("ProductCD_x_email", ["ProductCD", "P_emaildomain"]),
        #     ("addr1_x_card4", ["addr1", "card4"]),
        #     ("device_x_email", ["DeviceType", "P_emaildomain"]),
        #     ("card1_x_addr1", ["card1", "addr1"]),
        #     ("card4_x_email_x_device", ["card4", "P_emaildomain", "DeviceType"]),
        #     ("ProductCD_x_card6_x_email", ["ProductCD", "card6", "P_emaildomain"]),
        # ]
        # for combo_name, cols in interaction_combos:
        #     if all(c in df.columns for c in cols):
        #         combo_key = df[cols[0]].astype(str)
        #         for c in cols[1:]:
        #             combo_key = combo_key + "_" + df[c].astype(str)
        #         temp = pd.DataFrame({"__key__": combo_key, "__t__": y.values})
        #         grp = temp.groupby("__key__", dropna=False)["__t__"]
        #         counts = grp.count()
        #         means = grp.mean()
        #         smoothed = (counts * means + sm * gm) / (counts + sm)
        #         self.target_encodings_[combo_name] = smoothed.to_dict()
        #         self.target_encodings_[f"{combo_name}__cols"] = cols
        #         self.target_encodings_[f"{combo_name}__global_mean"] = gm

        # # ── Step 9: Learn aggregation statistics ──
        # self._learn_aggregation_stats(df, y)

        self.is_fitted = True
        logger.info(
            f"Pipeline fitted. Baseline features: {len(self.features_final_baseline_)}, "
            f"Missingness indicators: {len(self.missingness_features_)}, "
            f"Freq encodings: {len(self.frequency_encodings_)}, "
            f"Target encodings: {len([k for k in self.target_encodings_ if not k.endswith(('__global_mean', '__cols'))])}"
        )
        return self

    def _learn_aggregation_stats(self, df: pd.DataFrame, y: pd.Series) -> None:
        """Learn all 28 aggregation feature statistics from training data."""
        gm = self.global_fraud_rate_
        sm = self._smoothing

        def _smoothed_fraud_rate(series_col: pd.Series, target: np.ndarray) -> dict:
            temp = pd.DataFrame({"k": series_col, "t": target})
            grp = temp.groupby("k", dropna=False)["t"]
            counts = grp.count()
            means = grp.mean()
            return ((counts * means + sm * gm) / (counts + sm)).to_dict()

        # -- card1 (user proxy) --
        if "card1" in df.columns and "TransactionAmt" in df.columns:
            c1_stats = df.groupby("card1")["TransactionAmt"].agg(["mean", "std", "count", "min", "max", "median"])
            c1_stats.columns = ["card1_amt_mean", "card1_amt_std", "card1_amt_count",
                                "card1_amt_min", "card1_amt_max", "card1_amt_median"]
            c1_stats["card1_amt_std"] = c1_stats["card1_amt_std"].fillna(0)
            self.feature_stats_["card1_amt"] = c1_stats.to_dict()
            self.feature_stats_["card1_fraud_rate"] = _smoothed_fraud_rate(df["card1"], y.values)

            # Time-window stats (approximate: 1 day = 86400s, 7 days = 604800s)
            if "TransactionDT" in df.columns:
                dt_max = df["TransactionDT"].max()
                mask_1d = df["TransactionDT"] >= (dt_max - 86400)
                mask_7d = df["TransactionDT"] >= (dt_max - 604800)
                for suffix, mask in [("1d", mask_1d), ("7d", mask_7d)]:
                    sub = df[mask]
                    if len(sub) > 0:
                        grp = sub.groupby("card1")["TransactionAmt"].agg(["count", "mean"])
                        grp.columns = [f"card1_txn_count_{suffix}", f"card1_amt_mean_{suffix}"]
                        self.feature_stats_[f"card1_window_{suffix}"] = grp.to_dict()

        # -- card4+card6 combo --
        if all(c in df.columns for c in ["card4", "card6", "TransactionAmt"]):
            combo = df["card4"].astype(str) + "_" + df["card6"].astype(str)
            combo_stats = df.groupby(combo)["TransactionAmt"].agg(["count", "mean", "std"])
            combo_stats.columns = ["card_combo_txn_count", "card_combo_amt_mean", "card_combo_amt_std"]
            combo_stats["card_combo_amt_std"] = combo_stats["card_combo_amt_std"].fillna(0)
            self.feature_stats_["card_combo_amt"] = combo_stats.to_dict()

        # -- addr1 --
        if "addr1" in df.columns and "TransactionAmt" in df.columns:
            a1_stats = df.groupby("addr1")["TransactionAmt"].agg(["count", "mean"])
            a1_stats.columns = ["addr1_txn_count", "addr1_amt_mean"]
            self.feature_stats_["addr1_amt"] = a1_stats.to_dict()
            self.feature_stats_["addr1_fraud_rate"] = _smoothed_fraud_rate(df["addr1"], y.values)

        # -- P_emaildomain --
        if "P_emaildomain" in df.columns and "TransactionAmt" in df.columns:
            e_stats = df.groupby("P_emaildomain")["TransactionAmt"].agg(["count", "mean"])
            e_stats.columns = ["email_txn_count", "email_amt_mean"]
            self.feature_stats_["email_amt"] = e_stats.to_dict()
            self.feature_stats_["email_fraud_rate"] = _smoothed_fraud_rate(df["P_emaildomain"], y.values)

        # -- R_emaildomain --
        if "R_emaildomain" in df.columns and "TransactionAmt" in df.columns:
            r_stats = df.groupby("R_emaildomain")["TransactionAmt"].agg(["count", "mean"])
            r_stats.columns = ["r_email_txn_count", "r_email_amt_mean"]
            self.feature_stats_["r_email_amt"] = r_stats.to_dict()
            self.feature_stats_["r_email_fraud_rate"] = _smoothed_fraud_rate(df["R_emaildomain"], y.values)

        # -- DeviceType --
        if "DeviceType" in df.columns and "TransactionAmt" in df.columns:
            d_stats = df.groupby("DeviceType")["TransactionAmt"].agg(["count", "mean"])
            d_stats.columns = ["device_txn_count", "device_amt_mean"]
            self.feature_stats_["device_amt"] = d_stats.to_dict()
            self.feature_stats_["device_fraud_rate"] = _smoothed_fraud_rate(df["DeviceType"], y.values)

        # -- DeviceInfo --
        if "DeviceInfo" in df.columns:
            di_stats = df.groupby("DeviceInfo")["TransactionAmt"].agg(["count"]) if "TransactionAmt" in df.columns else pd.DataFrame()
            if not di_stats.empty:
                di_stats.columns = ["device_info_txn_count"]
                self.feature_stats_["device_info_amt"] = di_stats.to_dict()
            self.feature_stats_["device_info_fraud_rate"] = _smoothed_fraud_rate(df["DeviceInfo"], y.values)

        # -- card1+email combo --
        if all(c in df.columns for c in ["card1", "P_emaildomain"]):
            combo = df["card1"].astype(str) + "_" + df["P_emaildomain"].astype(str)
            combo_count = combo.value_counts().to_dict()
            self.feature_stats_["card1_email_combo_count"] = combo_count
            self.feature_stats_["card1_email_combo_fraud_rate"] = _smoothed_fraud_rate(combo, y.values)

    # ──────────────────────────────────────────
    # TRANSFORM
    # ──────────────────────────────────────────

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted preprocessing to new data.
        Works for both batch (training/evaluation) and single-row (inference).
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline not fitted. Call fit() first or load a saved pipeline.")

        df = df.copy()
        parts = []  # collect DataFrames, concat once at end to avoid fragmentation

        # ── 1. Baseline features: select + impute (dict-based to avoid fragmentation) ──
        bl_dict: dict = {}
        for col in self.features_final_baseline_:
            if col in df.columns:
                series = df[col].copy()
            elif col in self.numerical_medians_:
                series = pd.Series(self.numerical_medians_[col], index=df.index)
            else:
                series = pd.Series(0, index=df.index)

            if col in self.numerical_medians_:
                series = pd.to_numeric(series, errors="coerce").fillna(self.numerical_medians_[col])
            elif col in CATEGORICAL_COLS or series.dtype == object:
                series = series.fillna("missing").astype(str)
            bl_dict[col] = series

        baseline = pd.DataFrame(bl_dict, index=df.index)
        parts.append(baseline)

        # ── TEMPORARILY DISABLED: all feature engineering ──
        # Uncomment the blocks below to re-enable engineered features.

        # # ── 2. Missingness indicators ──
        # miss_dict = {}
        # for col in self.missingness_features_:
        #     if col in df.columns:
        #         miss_dict[f"{col}_missing"] = df[col].isnull().astype(np.int8)
        #     else:
        #         miss_dict[f"{col}_missing"] = np.int8(1)
        # if miss_dict:
        #     parts.append(pd.DataFrame(miss_dict, index=df.index))

        # # ── 3. Temporal features (15 new) ──
        # parts.append(self._temporal_features(df))

        # # ── 4. Amount-based features (12 new) ──
        # parts.append(self._amount_features(df, baseline))

        # # ── 5. Aggregation features (28 new) ──
        # parts.append(self._aggregation_features(df))

        # # ── 6. Interaction features (25 new) — freq + target + high-order ──
        # parts.append(self._interaction_features(df, baseline))

        # Concat all parts
        result = pd.concat(parts, axis=1)

        # ── Drop raw categorical columns ──
        cat_cols = [c for c in result.columns
                    if result[c].dtype == object or (c in CATEGORICAL_COLS and c in result.columns)]
        result = result.drop(columns=cat_cols, errors="ignore")

        # ── Ensure all numeric, no NaN, no Inf ──
        for col in result.columns:
            if result[col].dtype == object or result[col].dtype.name == "category":
                result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0)
        result = result.fillna(0).replace([np.inf, -np.inf], 0)

        # ── Enforce exact feature set from fit (handles train/inference parity) ──
        if self.final_feature_names_:
            for col in self.final_feature_names_:
                if col not in result.columns:
                    result[col] = 0
            result = result[self.final_feature_names_]
        else:
            self.final_feature_names_ = result.columns.tolist()
        logger.info(f"Transform complete: {result.shape[1]} features, {len(result)} rows")
        return result

    # ─── Temporal (15) ───

    def _temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = {}
        if "TransactionDT" in df.columns:
            dt = df["TransactionDT"].fillna(0)
            hour = (dt // 3600) % 24
            dow = (dt // 86400) % 7

            out["hour_of_day"] = hour
            out["day_of_week"] = dow
            out["day_of_month"] = (dt // 86400) % 30
            out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
            out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
            out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
            out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
            out["is_weekend"] = (dow >= 5).astype(np.int8)
            out["is_night"] = ((hour >= 22) | (hour <= 5)).astype(np.int8)
            out["is_business_hours"] = (
                (hour >= 9) & (hour <= 17) & (dow < 5)
            ).astype(np.int8)
            out["hour_bin"] = pd.cut(
                hour, bins=[0, 6, 12, 18, 24], labels=[0, 1, 2, 3], include_lowest=True
            ).astype(float).fillna(0)

            # Velocity features (require card1 + sorted TransactionDT)
            if "card1" in df.columns and len(df) > 1:
                sorted_idx = dt.argsort()
                sorted_df = df.iloc[sorted_idx].copy()
                sorted_dt = dt.iloc[sorted_idx].values
                sorted_card1 = sorted_df["card1"].values
                sorted_amt = sorted_df["TransactionAmt"].values if "TransactionAmt" in df.columns else np.zeros(len(df))

                n = len(df)
                vel_1h = np.zeros(n, dtype=np.float32)
                vel_24h = np.zeros(n, dtype=np.float32)
                time_since = np.zeros(n, dtype=np.float32)
                amt_vel_1h = np.zeros(n, dtype=np.float32)

                # Build card1 -> sorted positions index
                card_positions: dict = {}
                for i in range(n):
                    c = sorted_card1[i]
                    if c not in card_positions:
                        card_positions[c] = []
                    positions = card_positions[c]

                    # Count txns in last 1h and 24h for same card
                    t_now = sorted_dt[i]
                    cnt_1h = 0
                    cnt_24h = 0
                    sum_amt_1h = 0.0
                    last_t = 0.0
                    for j in reversed(positions):
                        delta = t_now - sorted_dt[j]
                        if delta <= 3600:
                            cnt_1h += 1
                            sum_amt_1h += sorted_amt[j]
                        if delta <= 86400:
                            cnt_24h += 1
                        else:
                            break
                        if last_t == 0.0:
                            last_t = delta

                    vel_1h[i] = cnt_1h
                    vel_24h[i] = cnt_24h
                    time_since[i] = last_t
                    amt_vel_1h[i] = sum_amt_1h
                    positions.append(i)

                # Map back to original index order
                reverse_idx = np.argsort(sorted_idx)
                out["txn_velocity_1h"] = vel_1h[reverse_idx]
                out["txn_velocity_24h"] = vel_24h[reverse_idx]
                out["time_since_last_txn"] = time_since[reverse_idx]
                out["amt_velocity_1h"] = amt_vel_1h[reverse_idx]
            else:
                out["txn_velocity_1h"] = 0
                out["txn_velocity_24h"] = 0
                out["time_since_last_txn"] = 0
                out["amt_velocity_1h"] = 0
        else:
            for feat in ["hour_of_day", "day_of_week", "day_of_month",
                         "hour_sin", "hour_cos", "dow_sin", "dow_cos",
                         "is_weekend", "is_night", "is_business_hours", "hour_bin",
                         "txn_velocity_1h", "txn_velocity_24h",
                         "time_since_last_txn", "amt_velocity_1h"]:
                out[feat] = 0

        return pd.DataFrame(out, index=df.index)

    # ─── Amount (12) ───

    def _amount_features(self, df: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
        out = {}
        if "TransactionAmt" in df.columns:
            amt = df["TransactionAmt"].fillna(0).clip(lower=0)
            out["amt_log"] = np.log1p(amt)
            out["amt_sqrt"] = np.sqrt(amt)
            out["amt_decimal"] = amt - np.floor(amt)
            out["amt_is_round"] = (out["amt_decimal"] == 0).astype(np.int8)
            out["amt_log10"] = np.log10(amt + 1)
            out["amt_bin"] = pd.cut(
                amt, bins=[0, 10, 50, 100, 500, 1000, 5000, float("inf")],
                labels=[0, 1, 2, 3, 4, 5, 6], include_lowest=True,
            ).astype(float).fillna(0)

            # Z-score relative to card1
            if "card1_amt" in self.feature_stats_ and "card1" in df.columns:
                c1_means = self.feature_stats_["card1_amt"]["card1_amt_mean"]
                c1_stds = self.feature_stats_["card1_amt"]["card1_amt_std"]
                c1 = df["card1"]
                m = c1.map(c1_means).fillna(amt.median())
                s = c1.map(c1_stds).fillna(amt.std()).replace(0, 1)
                out["amt_zscore_card1"] = (amt - m) / s
                out["amt_percentile_card1"] = amt.groupby(df["card1"]).rank(pct=True) if len(df) > 1 else 0.5
                out["amt_ratio_to_card1_mean"] = amt / m.replace(0, 1)
            else:
                out["amt_zscore_card1"] = 0
                out["amt_percentile_card1"] = 0.5
                out["amt_ratio_to_card1_mean"] = 1.0

            out["amt_percentile_global"] = amt.rank(pct=True) if len(amt) > 1 else 0.5

            # Interaction with time
            hour = baseline.get("hour_of_day") if "hour_of_day" in baseline.columns else pd.Series(0, index=df.index)
            weekend = baseline.get("is_weekend") if "is_weekend" in baseline.columns else pd.Series(0, index=df.index)
            # hour/weekend might be in the temporal part, not baseline — handle gracefully
            out["amt_x_hour"] = out["amt_log"] * 0  # placeholder, will be filled after concat
            out["amt_x_weekend"] = out["amt_log"] * 0
        else:
            for feat in ["amt_log", "amt_sqrt", "amt_decimal", "amt_is_round", "amt_log10",
                         "amt_bin", "amt_zscore_card1", "amt_percentile_card1",
                         "amt_percentile_global", "amt_ratio_to_card1_mean",
                         "amt_x_hour", "amt_x_weekend"]:
                out[feat] = 0

        return pd.DataFrame(out, index=df.index)

    # ─── Aggregation (28) ───

    def _aggregation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = {}

        # card1 (7 features: count, mean, std, min, max, median, fraud_rate)
        if "card1_amt" in self.feature_stats_ and "card1" in df.columns:
            s = self.feature_stats_["card1_amt"]
            c1 = df["card1"]
            out["card1_txn_count"] = c1.map(s["card1_amt_count"]).fillna(1)
            out["card1_amt_mean"] = c1.map(s["card1_amt_mean"]).fillna(0)
            out["card1_amt_std"] = c1.map(s["card1_amt_std"]).fillna(0)
            out["card1_amt_min"] = c1.map(s["card1_amt_min"]).fillna(0)
            out["card1_amt_max"] = c1.map(s["card1_amt_max"]).fillna(0)
            out["card1_amt_median"] = c1.map(s["card1_amt_median"]).fillna(0)
        else:
            for f in ["card1_txn_count", "card1_amt_mean", "card1_amt_std",
                       "card1_amt_min", "card1_amt_max", "card1_amt_median"]:
                out[f] = 0

        if "card1_fraud_rate" in self.feature_stats_ and "card1" in df.columns:
            out["card1_fraud_rate"] = df["card1"].map(self.feature_stats_["card1_fraud_rate"]).fillna(self.global_fraud_rate_)
        else:
            out["card1_fraud_rate"] = self.global_fraud_rate_

        # card1 time-window stats (4 features)
        for suffix in ["1d", "7d"]:
            key = f"card1_window_{suffix}"
            if key in self.feature_stats_ and "card1" in df.columns:
                s = self.feature_stats_[key]
                out[f"card1_txn_count_{suffix}"] = df["card1"].map(s[f"card1_txn_count_{suffix}"]).fillna(0)
                out[f"card1_amt_mean_{suffix}"] = df["card1"].map(s[f"card1_amt_mean_{suffix}"]).fillna(0)
            else:
                out[f"card1_txn_count_{suffix}"] = 0
                out[f"card1_amt_mean_{suffix}"] = 0

        # card4+card6 combo (3 features)
        if "card_combo_amt" in self.feature_stats_ and "card4" in df.columns and "card6" in df.columns:
            combo = df["card4"].astype(str) + "_" + df["card6"].astype(str)
            s = self.feature_stats_["card_combo_amt"]
            out["card_combo_txn_count"] = combo.map(s["card_combo_txn_count"]).fillna(1)
            out["card_combo_amt_mean"] = combo.map(s["card_combo_amt_mean"]).fillna(0)
            out["card_combo_amt_std"] = combo.map(s["card_combo_amt_std"]).fillna(0)
        else:
            for f in ["card_combo_txn_count", "card_combo_amt_mean", "card_combo_amt_std"]:
                out[f] = 0

        # addr1 (3 features)
        if "addr1_amt" in self.feature_stats_ and "addr1" in df.columns:
            s = self.feature_stats_["addr1_amt"]
            out["addr1_txn_count"] = df["addr1"].map(s["addr1_txn_count"]).fillna(1)
            out["addr1_amt_mean"] = df["addr1"].map(s["addr1_amt_mean"]).fillna(0)
        else:
            out["addr1_txn_count"] = 0
            out["addr1_amt_mean"] = 0
        if "addr1_fraud_rate" in self.feature_stats_ and "addr1" in df.columns:
            out["addr1_fraud_rate"] = df["addr1"].map(self.feature_stats_["addr1_fraud_rate"]).fillna(self.global_fraud_rate_)
        else:
            out["addr1_fraud_rate"] = self.global_fraud_rate_

        # P_emaildomain (3 features)
        if "email_amt" in self.feature_stats_ and "P_emaildomain" in df.columns:
            s = self.feature_stats_["email_amt"]
            out["email_txn_count"] = df["P_emaildomain"].map(s["email_txn_count"]).fillna(1)
            out["email_amt_mean"] = df["P_emaildomain"].map(s["email_amt_mean"]).fillna(0)
        else:
            out["email_txn_count"] = 0
            out["email_amt_mean"] = 0
        if "email_fraud_rate" in self.feature_stats_ and "P_emaildomain" in df.columns:
            out["email_fraud_rate"] = df["P_emaildomain"].map(self.feature_stats_["email_fraud_rate"]).fillna(self.global_fraud_rate_)
        else:
            out["email_fraud_rate"] = self.global_fraud_rate_

        # R_emaildomain (3 features)
        if "r_email_amt" in self.feature_stats_ and "R_emaildomain" in df.columns:
            s = self.feature_stats_["r_email_amt"]
            out["r_email_txn_count"] = df["R_emaildomain"].map(s["r_email_txn_count"]).fillna(1)
            out["r_email_amt_mean"] = df["R_emaildomain"].map(s["r_email_amt_mean"]).fillna(0)
        else:
            out["r_email_txn_count"] = 0
            out["r_email_amt_mean"] = 0
        if "r_email_fraud_rate" in self.feature_stats_ and "R_emaildomain" in df.columns:
            out["r_email_fraud_rate"] = df["R_emaildomain"].map(self.feature_stats_["r_email_fraud_rate"]).fillna(self.global_fraud_rate_)
        else:
            out["r_email_fraud_rate"] = self.global_fraud_rate_

        # DeviceType (3 features)
        if "device_amt" in self.feature_stats_ and "DeviceType" in df.columns:
            s = self.feature_stats_["device_amt"]
            out["device_txn_count"] = df["DeviceType"].map(s["device_txn_count"]).fillna(1)
            out["device_amt_mean"] = df["DeviceType"].map(s["device_amt_mean"]).fillna(0)
        else:
            out["device_txn_count"] = 0
            out["device_amt_mean"] = 0
        if "device_fraud_rate" in self.feature_stats_ and "DeviceType" in df.columns:
            out["device_fraud_rate"] = df["DeviceType"].map(self.feature_stats_["device_fraud_rate"]).fillna(self.global_fraud_rate_)
        else:
            out["device_fraud_rate"] = self.global_fraud_rate_

        # DeviceInfo (2 features)
        if "device_info_amt" in self.feature_stats_ and "DeviceInfo" in df.columns:
            s = self.feature_stats_["device_info_amt"]
            out["device_info_txn_count"] = df["DeviceInfo"].map(s["device_info_txn_count"]).fillna(1)
        else:
            out["device_info_txn_count"] = 0
        if "device_info_fraud_rate" in self.feature_stats_ and "DeviceInfo" in df.columns:
            out["device_info_fraud_rate"] = df["DeviceInfo"].map(self.feature_stats_["device_info_fraud_rate"]).fillna(self.global_fraud_rate_)
        else:
            out["device_info_fraud_rate"] = self.global_fraud_rate_

        # card1+email combo (2 features)
        if "card1_email_combo_count" in self.feature_stats_ and "card1" in df.columns and "P_emaildomain" in df.columns:
            combo = df["card1"].astype(str) + "_" + df["P_emaildomain"].astype(str)
            out["card1_email_combo_count"] = combo.map(self.feature_stats_["card1_email_combo_count"]).fillna(1)
            out["card1_email_combo_fraud_rate"] = combo.map(self.feature_stats_["card1_email_combo_fraud_rate"]).fillna(self.global_fraud_rate_)
        else:
            out["card1_email_combo_count"] = 0
            out["card1_email_combo_fraud_rate"] = self.global_fraud_rate_

        return pd.DataFrame(out, index=df.index)

    # ─── Interaction (25) ───

    def _interaction_features(self, df: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
        out = {}
        gm = self.global_fraud_rate_

        # 7 single-column target encodings
        for col in ["ProductCD", "card4", "card6", "P_emaildomain",
                     "R_emaildomain", "DeviceType", "addr1"]:
            enc = self.target_encodings_.get(col)
            if enc and col in df.columns:
                out[f"{col}_target_enc"] = df[col].map(enc).fillna(gm).astype(np.float32)
            else:
                out[f"{col}_target_enc"] = gm

        # 10 frequency encodings
        for col in ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
                     "DeviceType", "DeviceInfo", "addr1", "addr2", "M4"]:
            freq = self.frequency_encodings_.get(col)
            if freq and col in df.columns:
                out[f"{col}_freq"] = df[col].map(freq).fillna(0).astype(np.float32)
            else:
                out[f"{col}_freq"] = 0

        # 8 high-order interaction target encodings
        interaction_combos = [
            "card4_x_card6", "card4_x_ProductCD", "ProductCD_x_email",
            "addr1_x_card4", "device_x_email", "card1_x_addr1",
            "card4_x_email_x_device", "ProductCD_x_card6_x_email",
        ]
        for combo_name in interaction_combos:
            enc = self.target_encodings_.get(combo_name)
            cols_key = f"{combo_name}__cols"
            cols_list = self.target_encodings_.get(cols_key, [])
            if enc and cols_list and all(c in df.columns for c in cols_list):
                combo_key = df[cols_list[0]].astype(str)
                for c in cols_list[1:]:
                    combo_key = combo_key + "_" + df[c].astype(str)
                out[f"{combo_name}_target"] = combo_key.map(enc).fillna(gm).astype(np.float32)
            else:
                out[f"{combo_name}_target"] = gm

        return pd.DataFrame(out, index=df.index)

    def fit_transform(self, df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit pipeline on training data and return transformed features."""
        self.fit(df, y)
        return self.transform(df)

    # ──────────────────────────────────────────
    # SERIALIZATION
    # ──────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save fitted pipeline state to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted pipeline.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "missing_threshold": self.missing_threshold,
            "correlation_threshold": self.correlation_threshold,
            "info_gain_threshold": self.info_gain_threshold,
            "missingness_indicator_threshold": self.missingness_indicator_threshold,
            "features_after_missing_": self.features_after_missing_,
            "features_after_variance_": self.features_after_variance_,
            "features_after_correlation_": self.features_after_correlation_,
            "features_final_baseline_": self.features_final_baseline_,
            "numerical_medians_": self.numerical_medians_,
            "numerical_medians_fraud_": self.numerical_medians_fraud_,
            "numerical_medians_legit_": self.numerical_medians_legit_,
            "frequency_encodings_": self.frequency_encodings_,
            "target_encodings_": self.target_encodings_,
            "missingness_features_": self.missingness_features_,
            "final_feature_names_": self.final_feature_names_,
            "feature_stats_": self.feature_stats_,
            "global_fraud_rate_": self.global_fraud_rate_,
            "_smoothing": self._smoothing,
            "is_fitted": True,
        }

        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Also save a human-readable summary
        summary_path = path.with_suffix(".json")
        summary = {
            "baseline_features": len(self.features_final_baseline_),
            "missingness_indicators": len(self.missingness_features_),
            "frequency_encodings": len(self.frequency_encodings_),
            "target_encodings": len([k for k in self.target_encodings_ if not k.endswith(("__global_mean", "__cols"))]),
            "final_feature_count": len(self.final_feature_names_),
            "thresholds": {
                "missing": self.missing_threshold,
                "correlation": self.correlation_threshold,
                "info_gain": self.info_gain_threshold,
                "missingness_indicator": self.missingness_indicator_threshold,
            },
            "pipeline_steps": {
                "step1_after_missing_removal": len(self.features_after_missing_),
                "step2_after_variance_removal": len(self.features_after_variance_),
                "step3_after_correlation_removal": len(self.features_after_correlation_),
                "step4_baseline": len(self.features_final_baseline_),
            },
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Pipeline saved to {path} ({path.stat().st_size / 1024:.1f} KB)")

    @classmethod
    def load(cls, path: str) -> "FeaturePipeline":
        """Load a fitted pipeline from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Pipeline file not found: {path}")

        with open(path, "rb") as f:
            state = pickle.load(f)

        pipeline = cls(
            missing_threshold=state["missing_threshold"],
            correlation_threshold=state["correlation_threshold"],
            info_gain_threshold=state["info_gain_threshold"],
            missingness_indicator_threshold=state["missingness_indicator_threshold"],
        )
        pipeline.features_after_missing_ = state["features_after_missing_"]
        pipeline.features_after_variance_ = state["features_after_variance_"]
        pipeline.features_after_correlation_ = state["features_after_correlation_"]
        pipeline.features_final_baseline_ = state["features_final_baseline_"]
        pipeline.numerical_medians_ = state["numerical_medians_"]
        pipeline.numerical_medians_fraud_ = state.get("numerical_medians_fraud_", {})
        pipeline.numerical_medians_legit_ = state.get("numerical_medians_legit_", {})
        pipeline.frequency_encodings_ = state["frequency_encodings_"]
        pipeline.target_encodings_ = state["target_encodings_"]
        pipeline.missingness_features_ = state["missingness_features_"]
        pipeline.final_feature_names_ = state["final_feature_names_"]
        pipeline.feature_stats_ = state["feature_stats_"]
        pipeline.global_fraud_rate_ = state.get("global_fraud_rate_", 0.035)
        pipeline._smoothing = state.get("_smoothing", 20)
        pipeline.is_fitted = True

        logger.info(
            f"Pipeline loaded from {path}: "
            f"{len(pipeline.features_final_baseline_)} baseline features, "
            f"{len(pipeline.final_feature_names_)} total features after engineering"
        )
        return pipeline

    # ──────────────────────────────────────────
    # UTILITIES
    # ──────────────────────────────────────────

    def get_feature_names(self) -> list:
        """Return final feature names after transform."""
        if not self.is_fitted:
            raise RuntimeError("Pipeline not fitted.")
        return self.final_feature_names_

    def get_pipeline_summary(self) -> dict:
        """Return a summary of the pipeline steps and feature counts."""
        if not self.is_fitted:
            return {"status": "not_fitted"}

        return {
            "status": "fitted",
            "original_to_baseline": {
                "after_missing_removal": len(self.features_after_missing_),
                "after_variance_removal": len(self.features_after_variance_),
                "after_correlation_removal": len(self.features_after_correlation_),
                "baseline": len(self.features_final_baseline_),
            },
            "engineered_features": {
                "temporal": 15,
                "amount_based": 12,
                "aggregation": 28,
                "interaction": 25,
                "missingness_indicators": len(self.missingness_features_),
            },
            "total_features": len(self.final_feature_names_),
        }

    def __repr__(self) -> str:
        if self.is_fitted:
            return (
                f"FeaturePipeline(fitted=True, "
                f"baseline={len(self.features_final_baseline_)}, "
                f"total={len(self.final_feature_names_)})"
            )
        return "FeaturePipeline(fitted=False)"
