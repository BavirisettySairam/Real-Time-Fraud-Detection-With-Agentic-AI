#!/usr/bin/env python
"""
Train LightGBM Fraud Detection Model
Optimizes for PR-AUC and ROC-AUC with threshold tuning
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import logging
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    precision_recall_curve
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LightGBMTrainer:
    """LightGBM trainer with comprehensive metric tracking"""
    
    def __init__(self):
        self.model = None
        self.metrics_history = {}
        self.best_threshold = 0.5
        self.best_f1 = 0.0
        self.val_metrics = {}
        self.cv_scores = {'roc_auc': [], 'pr_auc': [], 'f1': []}
        self.category_mappings: Dict[str, Dict[str, int]] = {}
        self.feature_columns = []
        self.test_inference_snapshot: Dict[str, Any] = {}
        self.top_test_predictions = []

    @staticmethod
    def _add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add lightweight, high-signal engineered features."""
        out = df.copy()

        if 'TransactionAmt' in out.columns:
            amt = pd.to_numeric(out['TransactionAmt'], errors='coerce').fillna(0.0)
            out['TransactionAmt_log'] = np.log1p(np.clip(amt, 0, None))
            out['TransactionAmt_sqrt'] = np.sqrt(np.clip(amt, 0, None))
            out['TransactionAmt_cents'] = np.mod(np.round(amt * 100), 100).astype(np.float32)

        if 'TransactionDT' in out.columns:
            tdt = pd.to_numeric(out['TransactionDT'], errors='coerce').fillna(0)
            out['hour'] = ((tdt // 3600) % 24).astype(np.int16)
            out['day'] = ((tdt // 86400) % 7).astype(np.int16)
            out['is_weekend'] = out['day'].isin([5, 6]).astype(np.int8)
            out['is_night'] = out['hour'].isin([0, 1, 2, 3, 4, 5, 22, 23]).astype(np.int8)

        if {'card1', 'addr1'}.issubset(out.columns):
            out['card_addr_combo'] = out['card1'].astype(str) + '_' + out['addr1'].astype(str)

        if {'P_emaildomain', 'R_emaildomain'}.issubset(out.columns):
            out['email_domain_pair'] = out['P_emaildomain'].astype(str) + '_' + out['R_emaildomain'].astype(str)

        out['missing_count'] = out.isna().sum(axis=1).astype(np.float32)
        return out
    
    def extract_features(self, df: pd.DataFrame, fit: bool = True) -> Tuple[pd.DataFrame, np.ndarray]:
        """Extract and engineer features from raw data with consistent train/test encoding."""
        X = self._add_engineered_features(df)
        
        # Drop non-numeric and target columns
        drop_cols = ['TransactionID', 'isFraud']
        X = X.drop(columns=[c for c in drop_cols if c in X.columns])
        
        # Handle missing values
        X = X.fillna(0)

        # Ordinal encode categorical columns with stable mappings.
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            as_text = X[col].astype(str)
            if fit:
                categories = pd.Index(as_text.unique())
                mapping = {value: idx for idx, value in enumerate(categories)}
                self.category_mappings[col] = mapping
            else:
                mapping = self.category_mappings.get(col, {})

            X[col] = as_text.map(mapping).fillna(-1).astype(np.int32)

        if fit:
            self.feature_columns = list(X.columns)
        else:
            for name in self.feature_columns:
                if name not in X.columns:
                    X[name] = 0
            X = X[self.feature_columns]
        
        # Ensure all columns are numeric
        X = X.astype(np.float32)
        
        # Extract target
        y = df['isFraud'].values if 'isFraud' in df.columns else np.zeros(len(df))
        
        logger.info(f"Features: {X.shape}, fraud rate: {y.mean():.2%}")
        return X, y

    def load_test_data(self, data_dir: Path):
        """Load and merge test transaction and identity data."""
        trans_path = data_dir / "test_transaction.csv"
        identity_path = data_dir / "test_identity.csv"

        logger.info(f"Loading test transactions from {trans_path}...")
        df_trans = pd.read_csv(trans_path)

        if identity_path.exists():
            logger.info(f"Loading test identities from {identity_path}...")
            df_identity = pd.read_csv(identity_path)
            df = df_trans.merge(df_identity, on="TransactionID", how="left")
        else:
            df = df_trans

        logger.info(f"Loaded {len(df)} test transactions")
        return df

    def evaluate_unlabeled_test_set(
        self,
        df_test: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Generate inference-only metrics on unlabeled Kaggle-style test data."""
        if self.model is None:
            raise RuntimeError("Model is not trained")

        logger.info("Scoring unlabeled test set for inference snapshot...")
        scores = self.model.predict(X_test.values).astype(float)
        scores = np.clip(scores, 0.0, 1.0)

        predicted_fraud = (scores >= self.best_threshold).astype(int)
        tx_ids = df_test.get("TransactionID", pd.Series(range(len(df_test)))).astype(int).tolist()

        ranked_idx = np.argsort(scores)[::-1][:10]
        top_entries = []
        for idx in ranked_idx:
            top_entries.append({
                "TransactionID": int(tx_ids[idx]),
                "fraud_score": float(scores[idx]),
                "TransactionAmt": float(df_test.iloc[idx].get("TransactionAmt", 0.0) or 0.0),
                "ProductCD": str(df_test.iloc[idx].get("ProductCD", "")),
                "hour": int(df_test.iloc[idx].get("hour", -1)) if "hour" in df_test.columns else -1,
            })

        snapshot = {
            "rows": int(len(scores)),
            "threshold": float(self.best_threshold),
            "predicted_fraud_count": int(predicted_fraud.sum()),
            "predicted_fraud_rate": float(predicted_fraud.mean()),
            "score_min": float(np.min(scores)),
            "score_max": float(np.max(scores)),
            "score_mean": float(np.mean(scores)),
            "score_std": float(np.std(scores)),
            "score_p50": float(np.quantile(scores, 0.50)),
            "score_p90": float(np.quantile(scores, 0.90)),
            "score_p95": float(np.quantile(scores, 0.95)),
            "score_p99": float(np.quantile(scores, 0.99)),
        }

        self.test_inference_snapshot = snapshot
        self.top_test_predictions = top_entries

        return snapshot
    
    def load_and_prepare_data(self, data_dir: Path, max_samples: int | None = None):
        """Load and prepare training data"""
        trans_path = data_dir / "train_transaction.csv"
        identity_path = data_dir / "train_identity.csv"
        
        logger.info(f"Loading from {trans_path}...")
        read_kwargs = {'nrows': max_samples} if max_samples is not None else {}
        df_trans = pd.read_csv(trans_path, **read_kwargs)
        
        if identity_path.exists():
            logger.info(f"Loading from {identity_path}...")
            df_identity = pd.read_csv(identity_path)
            df = df_trans.merge(df_identity, on="TransactionID", how="left")
        else:
            df = df_trans
        
        logger.info(f"Loaded {len(df)} transactions")
        return df
    
    def find_optimal_threshold(self, y_true: np.ndarray, y_pred_proba: np.ndarray):
        """Find optimal threshold balancing precision and recall"""
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
        
        # F1 score that balances both metrics
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        
        # Filter for reasonable precision (at least 20% to avoid extreme false positive rates)
        valid_mask = precisions >= 0.2
        if valid_mask.sum() == 0:
            valid_mask = precisions >= precisions.min()  # Fallback if none pass
        
        valid_f1 = f1_scores.copy()
        valid_f1[~valid_mask] = -1  # Mark invalid thresholds
        
        best_idx = np.argmax(valid_f1)
        if valid_f1[best_idx] < 0:  # If still invalid, use standard approach
            best_idx = np.argmax(f1_scores)
        
        optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        best_f1 = f1_scores[best_idx]
        
        return optimal_threshold, best_f1
    
    def compute_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5):
        """Compute all evaluation metrics"""
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        metrics = {
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'pr_auc': average_precision_score(y_true, y_pred_proba),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'threshold': threshold,
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        return metrics
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_folds: int = 5
    ) -> dict:
        """Train LightGBM with cross-validation"""
        
        logger.info("=" * 70)
        logger.info("LIGHTGBM TRAINING")
        logger.info("=" * 70)
        
        # Convert to numpy if pandas DataFrame
        if hasattr(X_train, 'values'):
            X_train = X_train.values
        if hasattr(X_val, 'values'):
            X_val = X_val.values
        
        X_train = X_train.astype(np.float32)
        X_val = X_val.astype(np.float32)
        y_train = y_train.astype(np.int32)
        y_val = y_val.astype(np.int32)
        
        # LightGBM parameters optimized for fraud detection (improved for recall)
        base_weight = (1 - y_train.mean()) / max(y_train.mean(), 0.01)
        params = {
            'objective': 'binary',
            'metric': ['auc', 'aucpr'],
            'boosting_type': 'gbdt',
            'num_leaves': 127,
            'learning_rate': 0.03,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.9,
            'bagging_freq': 1,
            'seed': 42,
            'verbose': -1,
            'scale_pos_weight': base_weight,
            'lambda_l1': 0.5,
            'lambda_l2': 0.5,
            'min_data_in_leaf': 40,
            'num_threads': -1,
        }
        
        logger.info(f"Scale pos weight: {params['scale_pos_weight']:.2f}")
        
        # Cross-validation
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        cv_scores = {'roc_auc': [], 'pr_auc': [], 'f1': []}
        
        fold_idx = 1
        for train_idx, val_idx in skf.split(X_train, y_train):
            logger.info(f"\nFold {fold_idx}/{n_folds}")
            
            X_fold_train = X_train[train_idx]
            y_fold_train = y_train[train_idx]
            X_fold_val = X_train[val_idx]
            y_fold_val = y_train[val_idx]
            
            train_data = lgb.Dataset(
                X_fold_train, label=y_fold_train,
                feature_name=[f'f_{i}' for i in range(X_fold_train.shape[1])]
            )
            val_data = lgb.Dataset(X_fold_val, label=y_fold_val, reference=train_data)
            
            # Train
            model = lgb.train(
                params,
                train_data,
                num_boost_round=2500,
                valid_sets=[train_data, val_data],
                callbacks=[
                    lgb.early_stopping(200),
                    lgb.log_evaluation(period=100)
                ]
            )
            
            # Evaluate
            y_pred_proba = model.predict(X_fold_val)
            opt_threshold, best_f1 = self.find_optimal_threshold(y_fold_val, y_pred_proba)
            
            roc_auc = roc_auc_score(y_fold_val, y_pred_proba)
            pr_auc = average_precision_score(y_fold_val, y_pred_proba)
            
            cv_scores['roc_auc'].append(roc_auc)
            cv_scores['pr_auc'].append(pr_auc)
            cv_scores['f1'].append(best_f1)
            
            logger.info(f"  ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}, F1: {best_f1:.4f}, Threshold: {opt_threshold:.4f}")
            
            fold_idx += 1
        
        logger.info("\n" + "=" * 70)
        logger.info("CROSS-VALIDATION RESULTS")
        logger.info("=" * 70)
        logger.info(f"ROC-AUC:  {np.mean(cv_scores['roc_auc']):.4f} ± {np.std(cv_scores['roc_auc']):.4f}")
        logger.info(f"PR-AUC:   {np.mean(cv_scores['pr_auc']):.4f} ± {np.std(cv_scores['pr_auc']):.4f}")
        logger.info(f"F1-Score: {np.mean(cv_scores['f1']):.4f} ± {np.std(cv_scores['f1']):.4f}")
        
        # Final model on full training set
        logger.info("\n" + "=" * 70)
        logger.info("FINAL MODEL TRAINING")
        logger.info("=" * 70)
        
        train_data = lgb.Dataset(
            X_train, label=y_train,
            feature_name=[f'f_{i}' for i in range(X_train.shape[1])]
        )
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=2500,
            valid_sets=[train_data, val_data],
            callbacks=[
                lgb.early_stopping(200),
                lgb.log_evaluation(period=100)
            ]
        )
        
        # Evaluate on validation set
        y_val_pred = self.model.predict(X_val)
        opt_threshold, best_f1 = self.find_optimal_threshold(y_val, y_val_pred)
        self.best_threshold = opt_threshold
        self.best_f1 = best_f1
        
        val_metrics = self.compute_metrics(y_val, y_val_pred, opt_threshold)
        self.val_metrics = val_metrics
        self.cv_scores = cv_scores
        
        logger.info("\n" + "=" * 70)
        logger.info("VALIDATION SET EVALUATION")
        logger.info("=" * 70)
        logger.info(f"ROC-AUC:   {val_metrics['roc_auc']:.4f}")
        logger.info(f"PR-AUC:    {val_metrics['pr_auc']:.4f}")
        logger.info(f"Precision: {val_metrics['precision']:.4f}")
        logger.info(f"Recall:    {val_metrics['recall']:.4f}")
        logger.info(f"F1-Score:  {val_metrics['f1']:.4f}")
        logger.info(f"Threshold: {val_metrics['threshold']:.4f}")
        
        cm = np.array(val_metrics['confusion_matrix'])
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"  TN={cm[0,0]}, FP={cm[0,1]}")
        logger.info(f"  FN={cm[1,0]}, TP={cm[1,1]}")
        
        # Feature importance
        feature_importance = self.model.feature_importance(importance_type='gain')
        top_features_idx = np.argsort(feature_importance)[-20:][::-1]
        
        logger.info(f"\nTop 20 Important Features:")
        for rank, idx in enumerate(top_features_idx, 1):
            logger.info(f"  {rank}. Feature {idx}: {feature_importance[idx]:.0f}")
        
        return {
            'cv_scores': cv_scores,
            'val_metrics': val_metrics,
            'best_threshold': opt_threshold,
            'feature_importance': feature_importance.tolist()
        }
    
    def save(self, save_dir: Path):
        """Save model and metrics"""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = save_dir / "lightgbm_model.txt"
        self.model.save_model(str(model_path))
        logger.info(f"Model saved to {model_path}")
        
        # Save metrics
        metrics_path = save_dir / "lightgbm_metrics.json"
        cv_summary = {
            'roc_auc_mean': float(np.mean(self.cv_scores.get('roc_auc', [0.0]))),
            'roc_auc_std': float(np.std(self.cv_scores.get('roc_auc', [0.0]))),
            'pr_auc_mean': float(np.mean(self.cv_scores.get('pr_auc', [0.0]))),
            'pr_auc_std': float(np.std(self.cv_scores.get('pr_auc', [0.0]))),
            'f1_mean': float(np.mean(self.cv_scores.get('f1', [0.0]))),
            'f1_std': float(np.std(self.cv_scores.get('f1', [0.0]))),
        }
        val_cm = self.val_metrics.get('confusion_matrix', [[0, 0], [0, 0]])
        runtime_eval_snapshot = {
            'rows': int(self.val_metrics.get('rows', 0)),
            'fraud_rate': float(self.val_metrics.get('fraud_rate', 0.0)),
            'threshold': float(self.val_metrics.get('threshold', self.best_threshold)),
            'roc_auc': float(self.val_metrics.get('roc_auc', 0.0)),
            'pr_auc': float(self.val_metrics.get('pr_auc', 0.0)),
            'precision': float(self.val_metrics.get('precision', 0.0)),
            'recall': float(self.val_metrics.get('recall', 0.0)),
            'f1': float(self.val_metrics.get('f1', 0.0)),
            'log_loss': float(self.val_metrics.get('log_loss', 0.0)),
            'confusion_matrix': {
                'tn': int(val_cm[0][0]),
                'fp': int(val_cm[0][1]),
                'fn': int(val_cm[1][0]),
                'tp': int(val_cm[1][1]),
            },
        }
        metrics_data = {
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'best_threshold': self.best_threshold,
            'best_f1': self.best_f1,
            'runtime_eval_snapshot': runtime_eval_snapshot,
            'test_inference_snapshot': self.test_inference_snapshot,
            'top_test_predictions': self.top_test_predictions,
            'metrics_source': 'train_lightgbm_train_and_test_inference',
            'metrics_generated_at': datetime.now(timezone.utc).date().isoformat(),
            'validation_metrics': {
                'roc_auc': float(self.val_metrics.get('roc_auc', 0.0)),
                'pr_auc': float(self.val_metrics.get('pr_auc', 0.0)),
                'precision': float(self.val_metrics.get('precision', 0.0)),
                'recall': float(self.val_metrics.get('recall', 0.0)),
                'f1': float(self.val_metrics.get('f1', 0.0)),
                'threshold': float(self.val_metrics.get('threshold', self.best_threshold)),
                'rows': int(self.val_metrics.get('rows', 0)),
                'fraud_rate': float(self.val_metrics.get('fraud_rate', 0.0)),
                'log_loss': float(self.val_metrics.get('log_loss', 0.0)),
                'confusion_matrix': self.val_metrics.get('confusion_matrix', [[0, 0], [0, 0]]),
            },
            'cv_summary': cv_summary,
        }
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")


def main():
    project_root = Path(__file__).parent
    data_dir = project_root / "data"
    model_dir = project_root / "models"

    max_samples_env = os.getenv("MAX_TRAIN_SAMPLES", "").strip()
    max_samples = int(max_samples_env) if max_samples_env else None
    n_folds_env = os.getenv("CV_FOLDS", "3").strip()
    n_folds = max(2, int(n_folds_env))
    
    # Load data
    trainer = LightGBMTrainer()
    df = trainer.load_and_prepare_data(data_dir, max_samples=max_samples)
    
    # Prepare features
    X, y = trainer.extract_features(df, fit=True)
    
    # Stratified split improves class-balance consistency between train/val.
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    
    logger.info(f"Train: {len(X_train)} samples ({y_train.mean():.2%} fraud)")
    logger.info(f"Val:   {len(X_val)} samples ({y_val.mean():.2%} fraud)")
    logger.info(f"CV folds: {n_folds}, max_samples: {'all' if max_samples is None else max_samples}")
    
    # Train
    results = trainer.train(X_train, y_train, X_val, y_val, n_folds=n_folds)

    # Add validation metadata expected by runtime dashboards.
    results['val_metrics']['rows'] = int(len(y_val))
    results['val_metrics']['fraud_rate'] = float(y_val.mean())

    # Compute log loss manually to avoid extra imports.
    y_val_scores = trainer.model.predict(X_val.values).astype(float)
    y_val_scores = np.clip(y_val_scores, 1e-12, 1 - 1e-12)
    log_loss = -np.mean(y_val * np.log(y_val_scores) + (1 - y_val) * np.log(1 - y_val_scores))
    results['val_metrics']['log_loss'] = float(log_loss)
    trainer.val_metrics = results['val_metrics']

    # Score unlabeled test set.
    df_test = trainer.load_test_data(data_dir)
    X_test, _ = trainer.extract_features(df_test, fit=False)
    test_snapshot = trainer.evaluate_unlabeled_test_set(df_test, X_test)

    logger.info("\n" + "=" * 70)
    logger.info("TEST SET INFERENCE SNAPSHOT (UNLABELED)")
    logger.info("=" * 70)
    logger.info(f"Rows:                {test_snapshot['rows']}")
    logger.info(f"Predicted fraud cnt: {test_snapshot['predicted_fraud_count']}")
    logger.info(f"Predicted fraud rate:{test_snapshot['predicted_fraud_rate']:.4%}")
    logger.info(f"Score mean/std:      {test_snapshot['score_mean']:.6f} / {test_snapshot['score_std']:.6f}")
    logger.info(f"Score quantiles:     p90={test_snapshot['score_p90']:.6f}, p95={test_snapshot['score_p95']:.6f}, p99={test_snapshot['score_p99']:.6f}")
    
    # Save
    trainer.save(model_dir)
    
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
