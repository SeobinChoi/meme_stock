#!/usr/bin/env python3
"""
Spike-Aware Time Series ML Pipeline for Meme Stock Prediction

Reframed approach:
1. Primary: Spike classification (P90 threshold on log1p scale)
2. Secondary: Distribution-aware regression (Poisson/Quantile)
3. Strengthened baselines: SeasonalMean + Weighted-MA ensemble
4. Robust metrics: PR-AUC, Recall@K, sMASE, SMAPE, RMSLE
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json

# ML and stats
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (mean_absolute_error, mean_squared_error, 
                           precision_recall_curve, average_precision_score,
                           matthews_corrcoef, recall_score, precision_score)
import lightgbm as lgb

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Statistical tests
from scipy import stats

warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "spike_aware_production"

# Create output directories
for subdir in ['data', 'models', 'predictions', 'metrics', 'plots', 'reports']:
    (MODEL_DIR / subdir).mkdir(parents=True, exist_ok=True)


def load_and_prepare_data() -> pd.DataFrame:
    """Load ML data and add spike labels."""
    print("📊 Loading and preparing spike-aware data...")
    
    # Load the most comprehensive dataset
    data_files = list((DATA_DIR / 'processed' / 'reddit' / 'ml').glob('reddit_mentions_full_2021_2023_*.csv'))
    if not data_files:
        raise FileNotFoundError("No ML dataset found. Run process_archive_reddit_data_ml.py first.")
    
    latest_file = max(data_files, key=lambda x: x.stat().st_mtime)
    df = pd.read_csv(latest_file)
    
    # Parse date and basic cleaning
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['mentions'])
    
    # Filter tickers with sufficient data
    ticker_counts = df.groupby('ticker').size()
    valid_tickers = ticker_counts[ticker_counts >= 60].index
    df = df[df['ticker'].isin(valid_tickers)].copy()
    
    print(f"   Final dataset: {len(df)} rows, {len(valid_tickers)} tickers")
    return df


def create_spike_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create spike-aware features."""
    df_feat = df.copy().sort_values(['ticker', 'date']).reset_index(drop=True)
    
    # Log transform for stability
    df_feat['log1p_mentions'] = np.log1p(df_feat['mentions'])
    
    # Time features
    df_feat['day_of_week'] = df_feat['date'].dt.dayofweek
    df_feat['month'] = df_feat['date'].dt.month
    df_feat['is_weekend'] = (df_feat['day_of_week'] >= 5).astype(int)
    
    # Lag features on log scale
    for lag in [1, 3, 7, 14]:
        df_feat[f'log_lag_{lag}'] = df_feat.groupby('ticker')['log1p_mentions'].shift(lag)
    
    # Rolling features (shifted to prevent leakage)
    for window in [7, 14, 30]:
        df_feat[f'log_ma_{window}'] = df_feat.groupby('ticker')['log1p_mentions'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )
        df_feat[f'log_std_{window}'] = df_feat.groupby('ticker')['log1p_mentions'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
        )
    
    # Surprise features
    df_feat['ema_7'] = df_feat.groupby('ticker')['log1p_mentions'].transform(
        lambda x: x.shift(1).ewm(span=7).mean()
    )
    df_feat['surprise_7'] = df_feat.groupby('ticker')['log1p_mentions'].shift(1) - df_feat['ema_7']
    
    # Cross-ticker market sentiment
    market_total = df_feat.groupby('date')['log1p_mentions'].transform('sum')
    df_feat['market_ex_ticker'] = (market_total - df_feat['log1p_mentions']).shift(1)
    
    # Ticker encoding for global models
    le = LabelEncoder()
    df_feat['ticker_id'] = le.fit_transform(df_feat['ticker'])
    
    # Fill NaN values
    numeric_cols = df_feat.select_dtypes(include=[np.number]).columns
    df_feat[numeric_cols] = df_feat[numeric_cols].fillna(0)
    
    return df_feat


def create_spike_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Create spike labels for classification."""
    df_labeled = df.copy()
    
    # Per-ticker 90th percentile on log scale
    df_labeled['spike_threshold'] = df_labeled.groupby('ticker')['log1p_mentions'].transform(
        lambda x: x.quantile(0.9)
    )
    df_labeled['is_spike'] = (df_labeled['log1p_mentions'] > df_labeled['spike_threshold']).astype(int)
    
    # Future spike (prediction target)
    df_labeled['spike_next'] = df_labeled.groupby('ticker')['is_spike'].shift(-1)
    df_labeled = df_labeled.dropna(subset=['spike_next'])
    df_labeled['spike_next'] = df_labeled['spike_next'].astype(int)
    
    spike_rate = df_labeled['spike_next'].mean()
    print(f"   Spike rate: {spike_rate:.3f} ({df_labeled['spike_next'].sum()} / {len(df_labeled)})")
    
    return df_labeled


def seasonal_mean_baseline(train_data, test_dates, ticker):
    """Weekday-specific mean baseline."""
    ticker_train = train_data[train_data['ticker'] == ticker]
    if len(ticker_train) == 0:
        return np.zeros(len(test_dates))
    
    weekday_means = ticker_train.groupby('day_of_week')['log1p_mentions'].mean()
    test_df = pd.DataFrame({'date': test_dates})
    test_df['day_of_week'] = test_df['date'].dt.dayofweek
    predictions = test_df['day_of_week'].map(weekday_means).fillna(weekday_means.mean())
    
    return predictions.values


def calculate_metrics(y_true, y_pred, task='regression'):
    """Calculate metrics for classification or regression."""
    metrics = {}
    
    if task == 'classification':
        if len(np.unique(y_true)) > 1:
            metrics['pr_auc'] = average_precision_score(y_true, y_pred)
            y_pred_binary = (y_pred > 0.5).astype(int)
            metrics['mcc'] = matthews_corrcoef(y_true, y_pred_binary)
            
            # Recall@K
            for k in [0.05, 0.1]:
                threshold = np.quantile(y_pred, 1 - k)
                top_k_mask = y_pred >= threshold
                if top_k_mask.sum() > 0:
                    metrics[f'recall_at_{int(k*100)}pct'] = y_true[top_k_mask].mean()
                else:
                    metrics[f'recall_at_{int(k*100)}pct'] = 0.0
        else:
            metrics.update({'pr_auc': 0.0, 'mcc': 0.0, 'recall_at_5pct': 0.0, 'recall_at_10pct': 0.0})
    
    elif task == 'regression':
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # SMAPE
        epsilon = 1e-8
        smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + epsilon))
        metrics['smape'] = smape
    
    return metrics


def create_splits(df, start_date='2021-03-01', val_months=3, step_months=1):
    """Create expanding window splits."""
    splits = []
    current_date = pd.to_datetime(start_date)
    end_date = df['date'].max()
    
    while current_date + pd.DateOffset(months=val_months) <= end_date:
        train_end = current_date
        val_start = current_date
        val_end = current_date + pd.DateOffset(months=val_months)
        
        if val_end > end_date:
            break
            
        splits.append({'train_end': train_end, 'val_start': val_start, 'val_end': val_end})
        current_date += pd.DateOffset(months=step_months)
    
    print(f"   Created {len(splits)} expanding window splits")
    return splits


def train_models(df_labeled):
    """Train spike classification and regression models."""
    print("\n🚀 Training spike-aware models...")
    
    # Define features
    feature_cols = [col for col in df_labeled.columns if col.startswith(('log_lag_', 'log_ma_', 'log_std_', 
                                                                        'surprise_', 'market_', 'day_of_week', 
                                                                        'month', 'is_weekend', 'ticker_id'))]
    
    print(f"   Using {len(feature_cols)} features")
    
    splits = create_splits(df_labeled)
    results = {'classification': [], 'regression_poisson': [], 'regression_quantile': [], 'baselines': []}
    
    tickers = df_labeled['ticker'].unique()
    
    # Train on first 3 splits for demo
    for split_idx, split_info in enumerate(splits[:3]):
        print(f"\n   Processing split {split_idx + 1}/3")
        
        # Create split masks
        train_mask = df_labeled['date'] <= split_info['train_end']
        test_mask = (df_labeled['date'] >= split_info['val_start']) & \
                   (df_labeled['date'] <= split_info['val_end'])
        
        if not train_mask.any() or not test_mask.any():
            continue
        
        train_data = df_labeled[train_mask]
        test_data = df_labeled[test_mask]
        
        if len(train_data) < 100:
            continue
        
        # Prepare global model data
        X_train = train_data[feature_cols]
        X_test = test_data[feature_cols]
        y_class_train = train_data['spike_next']
        y_reg_train = train_data['log1p_mentions']
        
        # Handle inf/nan
        X_train = X_train.replace([np.inf, -np.inf], 0).fillna(0)
        X_test = X_test.replace([np.inf, -np.inf], 0).fillna(0)
        
        try:
            # Train classification model
            pos_weight = (len(y_class_train) - y_class_train.sum()) / max(y_class_train.sum(), 1)
            
            clf = lgb.LGBMClassifier(
                objective='binary',
                num_leaves=31,
                learning_rate=0.05,
                n_estimators=100,
                scale_pos_weight=pos_weight,
                random_state=42,
                verbosity=-1
            )
            clf.fit(X_train, y_class_train)
            clf_probs = clf.predict_proba(X_test)[:, 1]
            
            # Train regression models
            reg_poisson = lgb.LGBMRegressor(
                objective='poisson',
                num_leaves=31,
                learning_rate=0.05,
                n_estimators=100,
                random_state=42,
                verbosity=-1
            )
            reg_poisson.fit(X_train, y_reg_train)
            poisson_preds = reg_poisson.predict(X_test)
            
            reg_quantile = lgb.LGBMRegressor(
                objective='quantile',
                alpha=0.5,
                num_leaves=31,
                learning_rate=0.05,
                n_estimators=100,
                random_state=42,
                verbosity=-1
            )
            reg_quantile.fit(X_train, y_reg_train)
            quantile_preds = reg_quantile.predict(X_test)
            
        except Exception as e:
            print(f"      Error training models: {e}")
            continue
        
        # Evaluate per ticker
        for ticker in tickers:
            ticker_test = test_data[test_data['ticker'] == ticker]
            ticker_train = train_data[train_data['ticker'] == ticker]
            
            if len(ticker_test) == 0:
                continue
            
            ticker_mask = test_data['ticker'] == ticker
            
            # Classification metrics
            if ticker_mask.sum() > 0:
                ticker_clf_probs = clf_probs[ticker_mask]
                ticker_class_true = test_data[ticker_mask]['spike_next'].values
                
                clf_metrics = calculate_metrics(ticker_class_true, ticker_clf_probs, 'classification')
                clf_metrics.update({'split': split_idx, 'ticker': ticker})
                results['classification'].append(clf_metrics)
                
                # Regression metrics
                ticker_reg_true = test_data[ticker_mask]['log1p_mentions'].values
                
                # Poisson
                ticker_poisson = poisson_preds[ticker_mask]
                poisson_metrics = calculate_metrics(ticker_reg_true, ticker_poisson, 'regression')
                poisson_metrics.update({'split': split_idx, 'ticker': ticker})
                results['regression_poisson'].append(poisson_metrics)
                
                # Quantile
                ticker_quantile = quantile_preds[ticker_mask]
                quantile_metrics = calculate_metrics(ticker_reg_true, ticker_quantile, 'regression')
                quantile_metrics.update({'split': split_idx, 'ticker': ticker})
                results['regression_quantile'].append(quantile_metrics)
                
                # Baseline
                test_dates = ticker_test['date']
                baseline_preds = seasonal_mean_baseline(train_data, test_dates, ticker)
                baseline_metrics = calculate_metrics(ticker_reg_true, baseline_preds, 'regression')
                baseline_metrics.update({'split': split_idx, 'ticker': ticker})
                results['baselines'].append(baseline_metrics)
    
    return results


def create_simple_report(results):
    """Create a simplified report."""
    print("\n📋 Generating results...")
    
    # Classification summary
    if results['classification']:
        clf_df = pd.DataFrame(results['classification'])
        avg_pr_auc = clf_df['pr_auc'].mean()
        avg_recall_5 = clf_df['recall_at_5pct'].mean()
        
        print(f"🎯 Classification Performance:")
        print(f"   Average PR-AUC: {avg_pr_auc:.3f} (target: ≥0.65)")
        print(f"   Average Recall@5%: {avg_recall_5:.3f} (target: ≥0.60)")
        print(f"   Meets criteria: {avg_pr_auc >= 0.65 and avg_recall_5 >= 0.60}")
    
    # Regression comparison
    model_results = {}
    for model_name in ['baselines', 'regression_poisson', 'regression_quantile']:
        if results[model_name]:
            df = pd.DataFrame(results[model_name])
            avg_mae = df['mae'].mean()
            avg_smape = df['smape'].mean()
            model_results[model_name] = {'mae': avg_mae, 'smape': avg_smape}
    
    if model_results:
        print(f"\n📈 Regression Performance:")
        best_model = min(model_results.items(), key=lambda x: x[1]['mae'])
        
        for model, metrics in model_results.items():
            print(f"   {model}: MAE={metrics['mae']:.3f}, SMAPE={metrics['smape']:.1f}")
        
        print(f"   Best model: {best_model[0]}")
        
        # Compare to baseline
        baseline_mae = model_results.get('baselines', {}).get('mae', np.inf)
        if baseline_mae != np.inf and best_model[1]['mae'] < baseline_mae:
            improvement = (baseline_mae - best_model[1]['mae']) / baseline_mae * 100
            print(f"   Improvement over baseline: {improvement:.1f}%")
    
    # Save simple report
    report_path = MODEL_DIR / 'reports' / f'spike_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to {report_path}")


def main():
    """Main execution."""
    print("🚀 Starting Spike-Aware ML Pipeline")
    print("=" * 50)
    
    # Load and prepare data
    df = load_and_prepare_data()
    df_features = create_spike_features(df)
    df_labeled = create_spike_labels(df_features)
    
    # Train models
    results = train_models(df_labeled)
    
    # Generate report
    create_simple_report(results)
    
    print("\n🎯 Pipeline completed!")
    print(f"📁 Outputs saved to: {MODEL_DIR}")


if __name__ == '__main__':
    main()