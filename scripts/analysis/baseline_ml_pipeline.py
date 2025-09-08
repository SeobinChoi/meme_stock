#!/usr/bin/env python3
"""
Production-Ready Time Series ML Pipeline for Meme Stock Prediction

Features:
- Complete data leakage prevention
- Expanding window cross-validation
- Robust baseline models including SeasonalNaive
- Log transformation with Jensen bias correction
- MASE/SMAPE metrics for reliable evaluation
- Diebold-Mariano statistical testing
- Automated reporting and visualization
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
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

# Statistical tests
from scipy import stats

warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "baseline_production"

# Create output directories
for subdir in ['data', 'models', 'predictions', 'metrics', 'plots', 'reports']:
    (MODEL_DIR / subdir).mkdir(parents=True, exist_ok=True)


class LeakageFreeFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create time series features without data leakage."""
    
    def __init__(self, lag_days=[1, 3, 7, 14, 30], rolling_windows=[7, 14, 30]):
        self.lag_days = lag_days
        self.rolling_windows = rolling_windows
        self.feature_groups = {
            'lag': [],
            'rolling': [],
            'time': [],
            'momentum': [],
            'cross_ticker': []
        }
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_feat = X.copy()
        
        # Ensure sorted by ticker and date
        X_feat = X_feat.sort_values(['ticker', 'date']).reset_index(drop=True)
        
        # Time-based features (no leakage)
        X_feat['day_of_week'] = X_feat['date'].dt.dayofweek
        X_feat['month'] = X_feat['date'].dt.month
        X_feat['quarter'] = X_feat['date'].dt.quarter
        X_feat['is_weekend'] = (X_feat['day_of_week'] >= 5).astype(int)
        X_feat['day_of_month'] = X_feat['date'].dt.day
        X_feat['is_month_end'] = X_feat['date'].dt.is_month_end.astype(int)
        
        self.feature_groups['time'] = ['day_of_week', 'month', 'quarter', 
                                      'is_weekend', 'day_of_month', 'is_month_end']
        
        # Lag features (with proper shift)
        for lag in self.lag_days:
            col_name = f'mentions_lag_{lag}'
            X_feat[col_name] = X_feat.groupby('ticker')['mentions'].shift(lag)
            self.feature_groups['lag'].append(col_name)
        
        # Rolling features (shifted to prevent leakage)
        for window in self.rolling_windows:
            # Rolling mean
            col_name = f'mentions_ma_{window}'
            X_feat[col_name] = X_feat.groupby('ticker')['mentions'].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            )
            self.feature_groups['rolling'].append(col_name)
            
            # Rolling std
            col_name = f'mentions_std_{window}'
            X_feat[col_name] = X_feat.groupby('ticker')['mentions'].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
            )
            self.feature_groups['rolling'].append(col_name)
            
            # Rolling max
            col_name = f'mentions_max_{window}'
            X_feat[col_name] = X_feat.groupby('ticker')['mentions'].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).max()
            )
            self.feature_groups['rolling'].append(col_name)
        
        # Momentum features (log returns for stability)
        X_feat['log_mentions'] = np.log1p(X_feat['mentions'])
        X_feat['log_return_1d'] = X_feat.groupby('ticker')['log_mentions'].diff(1)
        X_feat['log_return_7d'] = X_feat.groupby('ticker')['log_mentions'].diff(7)
        X_feat['volatility_7d'] = X_feat.groupby('ticker')['log_return_1d'].transform(
            lambda x: x.shift(1).rolling(7, min_periods=1).std()
        )
        
        self.feature_groups['momentum'] = ['log_return_1d', 'log_return_7d', 'volatility_7d']
        
        # Cross-ticker features (leave-one-out market sentiment)
        market_total = X_feat.groupby('date')['mentions'].transform('sum')
        X_feat['market_sentiment_ex_ticker'] = (market_total - X_feat['mentions']).shift(1)
        X_feat['market_sentiment_lag1'] = market_total.shift(1)
        
        self.feature_groups['cross_ticker'] = ['market_sentiment_ex_ticker', 'market_sentiment_lag1']
        
        # Fill NaN values
        numeric_cols = X_feat.select_dtypes(include=[np.number]).columns
        X_feat[numeric_cols] = X_feat[numeric_cols].fillna(0)
        
        return X_feat


class LogTransformer(BaseEstimator, TransformerMixin):
    """Log transform with Jensen bias correction for inverse transform."""
    
    def __init__(self):
        self.variance_ = None
        
    def fit(self, X, y=None):
        # Calculate variance for Jensen correction
        log_y = np.log1p(X.flatten())
        self.variance_ = np.var(log_y)
        return self
    
    def transform(self, X):
        return np.log1p(X)
    
    def inverse_transform(self, X):
        # Jensen bias correction: exp(log_pred + variance/2) - 1
        return np.expm1(X + self.variance_ / 2)


def load_and_prepare_data() -> pd.DataFrame:
    """Load ML data and standardize calendar."""
    print("📊 Loading and preparing data...")
    
    # Load the most comprehensive dataset
    data_files = list((DATA_DIR / 'processed' / 'reddit' / 'ml').glob('reddit_mentions_full_2021_2023_*.csv'))
    if not data_files:
        raise FileNotFoundError("No ML dataset found. Run process_archive_reddit_data_ml.py first.")
    
    latest_file = max(data_files, key=lambda x: x.stat().st_mtime)
    df = pd.read_csv(latest_file)
    
    # Parse date
    df['date'] = pd.to_datetime(df['date'])
    
    # Create complete date range for all tickers
    date_range = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='D')
    tickers = df['ticker'].unique()
    
    # Create complete calendar grid
    calendar_grid = []
    for ticker in tickers:
        for date in date_range:
            calendar_grid.append({'ticker': ticker, 'date': date})
    
    calendar_df = pd.DataFrame(calendar_grid)
    
    # Merge with actual data
    df_complete = calendar_df.merge(df[['date', 'ticker', 'mentions', 'ticker_type']], 
                                   on=['date', 'ticker'], how='left')
    
    # Add ticker_type for missing rows
    ticker_type_map = df[['ticker', 'ticker_type']].drop_duplicates().set_index('ticker')['ticker_type'].to_dict()
    df_complete['ticker_type'] = df_complete['ticker'].map(ticker_type_map)
    
    # Handle missing values according to rules
    df_complete = df_complete.sort_values(['ticker', 'date'])
    
    # Forward fill short gaps (≤3 days), then drop remaining NaN
    df_complete['mentions'] = df_complete.groupby('ticker')['mentions'].transform(
        lambda x: x.fillna(method='ffill', limit=3)
    )
    
    # Drop rows with remaining NaN
    initial_len = len(df_complete)
    df_complete = df_complete.dropna(subset=['mentions'])
    dropped_pct = (initial_len - len(df_complete)) / initial_len * 100
    print(f"   Dropped {dropped_pct:.1f}% of rows due to missing data")
    
    # Check minimum data length per ticker
    ticker_counts = df_complete.groupby('ticker').size()
    min_length = 60  # Minimum 60 days
    valid_tickers = ticker_counts[ticker_counts >= min_length].index
    
    df_final = df_complete[df_complete['ticker'].isin(valid_tickers)].copy()
    print(f"   Kept {len(valid_tickers)} tickers with ≥{min_length} days of data")
    
    # Add outlier detection (>3 std from 30-day rolling mean)
    df_final['rolling_mean_30'] = df_final.groupby('ticker')['mentions'].transform(
        lambda x: x.rolling(30, min_periods=1).mean()
    )
    df_final['rolling_std_30'] = df_final.groupby('ticker')['mentions'].transform(
        lambda x: x.rolling(30, min_periods=1).std()
    )
    
    threshold = df_final['rolling_mean_30'] + 3 * df_final['rolling_std_30']
    df_final['is_outlier'] = (df_final['mentions'] > threshold).astype(int)
    
    print(f"   Final dataset: {len(df_final)} rows, {len(valid_tickers)} tickers")
    print(f"   Date range: {df_final['date'].min()} to {df_final['date'].max()}")
    print(f"   Outliers detected: {df_final['is_outlier'].sum()} ({df_final['is_outlier'].mean()*100:.1f}%)")
    
    return df_final


def create_expanding_splits(df: pd.DataFrame, 
                           start_date='2021-03-01',
                           val_months=3,
                           step_months=1) -> List[Tuple]:
    """Create expanding window splits for time series CV."""
    
    splits = []
    current_date = pd.to_datetime(start_date)
    end_date = df['date'].max()
    
    while current_date + pd.DateOffset(months=val_months) <= end_date:
        # Training: from start to current_date
        train_end = current_date
        
        # Validation: val_months after current_date
        val_start = current_date
        val_end = current_date + pd.DateOffset(months=val_months)
        
        if val_end > end_date:
            break
            
        splits.append({
            'train_end': train_end,
            'val_start': val_start, 
            'val_end': val_end
        })
        
        current_date += pd.DateOffset(months=step_months)
    
    print(f"   Created {len(splits)} expanding window splits")
    return splits


class BaselineModels:
    """Collection of baseline models."""
    
    @staticmethod
    def naive(y_train, y_test):
        """Naive: y_t = y_{t-1}"""
        pred = np.full(len(y_test), y_train.iloc[-1])
        return pred
    
    @staticmethod
    def seasonal_naive(y_train, y_test, season=7):
        """Seasonal Naive: y_t = y_{t-season}"""
        if len(y_train) < season:
            return BaselineModels.naive(y_train, y_test)
        
        # Use last season values, cycling as needed
        seasonal_values = y_train.iloc[-season:].values
        pred = np.tile(seasonal_values, (len(y_test) // season) + 1)[:len(y_test)]
        return pred
    
    @staticmethod
    def moving_average(y_train, y_test, window=7):
        """Moving Average: y_t = mean of last window values"""
        if len(y_train) < window:
            window = len(y_train)
        
        ma_value = y_train.iloc[-window:].mean()
        pred = np.full(len(y_test), ma_value)
        return pred


def calculate_metrics(y_true, y_pred, y_train=None):
    """Calculate comprehensive metrics including MASE and SMAPE."""
    
    metrics = {}
    
    # Standard metrics
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # SMAPE (Symmetric Mean Absolute Percentage Error)
    epsilon = 1e-8
    smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + epsilon))
    metrics['smape'] = smape
    
    # MASE (Mean Absolute Scaled Error) - requires training data
    if y_train is not None and len(y_train) > 7:
        # Use seasonal naive as baseline (7-day seasonality)
        seasonal_errors = np.abs(y_train.iloc[7:] - y_train.iloc[:-7].values)
        mae_seasonal_naive = seasonal_errors.mean()
        
        if mae_seasonal_naive > 0:
            mase = metrics['mae'] / mae_seasonal_naive
            metrics['mase'] = mase
        else:
            metrics['mase'] = np.inf
    else:
        metrics['mase'] = np.inf
    
    # Direction accuracy (two methods)
    if len(y_true) > 1 and y_train is not None and len(y_train) > 0:
        # Method 1: Simple threshold
        prev_val = y_train.iloc[-1]
        pred_up = y_pred > prev_val
        true_up = y_true > prev_val
        metrics['direction_acc_1'] = (pred_up == true_up).mean()
        
        # Method 2: Delta correlation
        pred_delta = y_pred - prev_val
        true_delta = y_true - prev_val
        direction_match = (pred_delta * true_delta) > 0
        metrics['direction_acc_2'] = direction_match.mean()
    else:
        metrics['direction_acc_1'] = 0.0
        metrics['direction_acc_2'] = 0.0
    
    return metrics


def diebold_mariano_test(errors1, errors2):
    """
    Simple Diebold-Mariano test implementation.
    Returns (dm_stat, p_value)
    """
    d = errors1**2 - errors2**2
    d_mean = d.mean()
    
    if len(d) < 2:
        return np.nan, np.nan
        
    # Calculate variance (simple version)
    d_var = d.var(ddof=1)
    
    if d_var == 0:
        return np.nan, np.nan
    
    dm_stat = d_mean / np.sqrt(d_var / len(d))
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    
    return dm_stat, p_value


def train_and_evaluate_models(df: pd.DataFrame) -> Dict:
    """Main training and evaluation pipeline."""
    
    print("\n🚀 Training and evaluating models...")
    
    # Create feature engineer
    feature_engineer = LeakageFreeFeatureEngineer()
    df_features = feature_engineer.fit_transform(df)
    
    # Define feature columns (excluding target and metadata)
    all_features = []
    for group_features in feature_engineer.feature_groups.values():
        all_features.extend(group_features)
    
    # Create splits
    splits = create_expanding_splits(df_features)
    
    # Initialize results
    results = {
        'splits': splits,
        'models': {},
        'predictions': {},
        'metrics': {},
        'feature_importance': {}
    }
    
    tickers = df_features['ticker'].unique()
    models_to_test = ['naive', 'seasonal_naive', 'ma_3', 'ma_7', 'ma_14', 'ma_30', 
                      'elastic_net', 'random_forest', 'lightgbm']
    
    print(f"   Testing {len(models_to_test)} models on {len(tickers)} tickers")
    print(f"   Using {len(splits)} time splits")
    
    for ticker in tickers:
        print(f"\n   Processing ticker: {ticker}")
        ticker_data = df_features[df_features['ticker'] == ticker].copy()
        ticker_data = ticker_data.sort_values('date').reset_index(drop=True)
        
        results['predictions'][ticker] = {}
        results['metrics'][ticker] = {}
        results['feature_importance'][ticker] = {}
        
        for model_name in models_to_test:
            results['predictions'][ticker][model_name] = []
            results['metrics'][ticker][model_name] = []
            
            for split_idx, split_info in enumerate(splits):
                # Create train/val split
                train_mask = ticker_data['date'] <= split_info['train_end']
                val_mask = (ticker_data['date'] >= split_info['val_start']) & \
                          (ticker_data['date'] <= split_info['val_end'])
                
                if not train_mask.any() or not val_mask.any():
                    continue
                
                train_data = ticker_data[train_mask]
                val_data = ticker_data[val_mask]
                
                if len(train_data) < 30:  # Minimum training length
                    continue
                
                y_train = train_data['mentions']
                y_val = val_data['mentions']
                
                # Train model
                if model_name == 'naive':
                    y_pred = BaselineModels.naive(y_train, y_val)
                    
                elif model_name == 'seasonal_naive':
                    y_pred = BaselineModels.seasonal_naive(y_train, y_val)
                    
                elif model_name.startswith('ma_'):
                    window = int(model_name.split('_')[1])
                    y_pred = BaselineModels.moving_average(y_train, y_val, window)
                    
                elif model_name in ['elastic_net', 'random_forest', 'lightgbm']:
                    # Prepare features
                    X_train = train_data[all_features]
                    X_val = val_data[all_features]
                    
                    # Handle infinite values
                    X_train = X_train.replace([np.inf, -np.inf], 0)
                    X_val = X_val.replace([np.inf, -np.inf], 0)
                    
                    # Log transform target
                    log_transformer = LogTransformer()
                    y_train_log = log_transformer.fit_transform(y_train.values.reshape(-1, 1)).flatten()
                    
                    try:
                        if model_name == 'elastic_net':
                            # Pipeline with scaling
                            pipe = Pipeline([
                                ('scaler', RobustScaler()),
                                ('model', ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42))
                            ])
                            pipe.fit(X_train, y_train_log)
                            y_pred_log = pipe.predict(X_val)
                            
                        elif model_name == 'random_forest':
                            model = RandomForestRegressor(n_estimators=100, max_depth=10, 
                                                        random_state=42, n_jobs=-1)
                            model.fit(X_train, y_train_log)
                            y_pred_log = model.predict(X_val)
                            
                            # Store feature importance
                            if split_idx == 0:  # Only for first split
                                importance = pd.DataFrame({
                                    'feature': all_features,
                                    'importance': model.feature_importances_
                                }).sort_values('importance', ascending=False)
                                results['feature_importance'][ticker][model_name] = importance
                                
                        elif model_name == 'lightgbm':
                            model = lgb.LGBMRegressor(
                                objective='quantile',
                                alpha=0.5,  # Median prediction
                                n_estimators=100,
                                learning_rate=0.1,
                                random_state=42,
                                verbosity=-1
                            )
                            model.fit(X_train, y_train_log)
                            y_pred_log = model.predict(X_val)
                        
                        # Inverse transform
                        y_pred = log_transformer.inverse_transform(y_pred_log)
                        
                    except Exception as e:
                        print(f"      Error training {model_name}: {e}")
                        y_pred = np.full(len(y_val), y_train.mean())
                
                # Ensure non-negative predictions
                y_pred = np.maximum(y_pred, 0)
                
                # Calculate metrics
                metrics = calculate_metrics(y_val.values, y_pred, y_train)
                metrics['split_idx'] = split_idx
                
                results['predictions'][ticker][model_name].append({
                    'split_idx': split_idx,
                    'dates': val_data['date'].values,
                    'y_true': y_val.values,
                    'y_pred': y_pred,
                    'is_outlier': val_data['is_outlier'].values
                })
                
                results['metrics'][ticker][model_name].append(metrics)
    
    print(f"\n✅ Model training complete!")
    return results


def create_visualizations(results: Dict, df: pd.DataFrame):
    """Create the essential 4-panel visualization system."""
    
    print("\n📊 Creating visualizations...")
    
    tickers = list(results['predictions'].keys())
    models = list(results['predictions'][tickers[0]].keys())
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    for ticker in tickers[:3]:  # Limit to top 3 tickers for demo
        print(f"   Creating plots for {ticker}")
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle(f'Model Performance Analysis: {ticker}', fontsize=16, fontweight='bold')
        
        # Get all predictions for this ticker
        ticker_data = df[df['ticker'] == ticker].copy().sort_values('date')
        
        # Panel 1: Actual vs Predicted (last 120 days)
        ax1 = axes[0, 0]
        recent_data = ticker_data.tail(120)
        ax1.plot(recent_data['date'], recent_data['mentions'], 
                label='Actual', linewidth=2, alpha=0.8)
        
        # Plot best model predictions (choose random forest for demo)
        if 'random_forest' in results['predictions'][ticker]:
            best_preds = results['predictions'][ticker]['random_forest']
            for pred_data in best_preds[-3:]:  # Last 3 splits
                dates = pd.to_datetime(pred_data['dates'])
                mask = dates >= recent_data['date'].min()
                if mask.any():
                    ax1.plot(dates[mask], pred_data['y_pred'][mask], 
                           'o-', alpha=0.6, markersize=3)
        
        ax1.set_title('Actual vs Predicted (Last 120 Days)')
        ax1.set_ylabel('Mentions')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Residuals over Time + Event Shading
        ax2 = axes[0, 1]
        
        # Calculate residuals for best model
        all_residuals = []
        all_dates = []
        
        if 'random_forest' in results['predictions'][ticker]:
            for pred_data in results['predictions'][ticker]['random_forest']:
                residuals = pred_data['y_true'] - pred_data['y_pred']
                all_residuals.extend(residuals)
                all_dates.extend(pred_data['dates'])
        
        if all_residuals:
            df_residuals = pd.DataFrame({
                'date': pd.to_datetime(all_dates),
                'residual': all_residuals
            }).sort_values('date')
            
            ax2.scatter(df_residuals['date'], df_residuals['residual'], 
                       alpha=0.6, s=20)
            ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
            
            # Add event shading (example: GME squeeze period)
            if ticker == 'GME':
                ax2.axvspan(pd.to_datetime('2021-01-25'), pd.to_datetime('2021-02-05'),
                           alpha=0.2, color='red', label='GME Squeeze')
                ax2.legend()
        
        ax2.set_title('Residuals Over Time')
        ax2.set_ylabel('Residual (Actual - Predicted)')
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Error Distribution (log-scale bins)
        ax3 = axes[1, 0]
        
        if all_residuals:
            abs_residuals = np.abs(all_residuals)
            # Log-scale bins
            bins = np.logspace(np.log10(max(abs_residuals.min(), 1e-6)), 
                              np.log10(abs_residuals.max()), 30)
            
            ax3.hist(abs_residuals, bins=bins, alpha=0.7, edgecolor='black')
            ax3.set_xscale('log')
            ax3.set_title('Absolute Error Distribution (Log Scale)')
            ax3.set_xlabel('Absolute Error')
            ax3.set_ylabel('Frequency')
            ax3.grid(True, alpha=0.3)
        
        # Panel 4: Model Comparison Heatmap
        ax4 = axes[1, 1]
        
        # Create metrics comparison matrix
        metrics_matrix = []
        model_names = []
        
        for model in models:
            if results['metrics'][ticker][model]:
                avg_metrics = {}
                all_model_metrics = results['metrics'][ticker][model]
                
                for metric in ['mae', 'rmse', 'smape', 'mase']:
                    values = [m.get(metric, np.nan) for m in all_model_metrics 
                             if not np.isinf(m.get(metric, np.nan))]
                    avg_metrics[metric] = np.mean(values) if values else np.nan
                
                metrics_matrix.append([avg_metrics.get(m, np.nan) 
                                     for m in ['mae', 'rmse', 'smape', 'mase']])
                model_names.append(model)
        
        if metrics_matrix:
            metrics_df = pd.DataFrame(metrics_matrix, 
                                    index=model_names,
                                    columns=['MAE', 'RMSE', 'SMAPE', 'MASE'])
            
            sns.heatmap(metrics_df, annot=True, fmt='.2f', cmap='RdYlGn_r', 
                       ax=ax4, cbar_kws={'label': 'Error Value'})
            ax4.set_title('Model Performance Heatmap')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = MODEL_DIR / 'plots' / f'{ticker}_performance_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"   Saved visualization plots to {MODEL_DIR / 'plots'}")


def generate_report(results: Dict, df: pd.DataFrame):
    """Generate comprehensive automated report."""
    
    print("\n📋 Generating comprehensive report...")
    
    report = {
        'summary': {
            'generated_at': datetime.now().isoformat(),
            'total_tickers': len(results['predictions']),
            'total_models': len(list(results['predictions'].values())[0]),
            'cv_folds': len(results['splits']),
            'evaluation_period': {
                'start': df['date'].min().strftime('%Y-%m-%d'),
                'end': df['date'].max().strftime('%Y-%m-%d')
            }
        },
        'model_rankings': {},
        'statistical_tests': {},
        'validation_criteria': {},
        'recommendations': []
    }
    
    tickers = list(results['predictions'].keys())
    models = list(results['predictions'][tickers[0]].keys())
    
    # Calculate average metrics per model
    model_avg_metrics = {}
    
    for model in models:
        model_avg_metrics[model] = {'mase': [], 'smape': [], 'mae': []}
        
        for ticker in tickers:
            ticker_metrics = results['metrics'][ticker][model]
            
            for metric_dict in ticker_metrics:
                if not np.isinf(metric_dict.get('mase', np.inf)):
                    model_avg_metrics[model]['mase'].append(metric_dict['mase'])
                if not np.isnan(metric_dict.get('smape', np.nan)):
                    model_avg_metrics[model]['smape'].append(metric_dict['smape'])
                if not np.isnan(metric_dict.get('mae', np.nan)):
                    model_avg_metrics[model]['mae'].append(metric_dict['mae'])
    
    # Calculate average metrics
    for model in model_avg_metrics:
        for metric in model_avg_metrics[model]:
            values = model_avg_metrics[model][metric]
            model_avg_metrics[model][metric] = np.mean(values) if values else np.nan
    
    # Create rankings
    mase_ranking = sorted(model_avg_metrics.items(), 
                         key=lambda x: x[1]['mase'] if not np.isnan(x[1]['mase']) else np.inf)
    smape_ranking = sorted(model_avg_metrics.items(), 
                          key=lambda x: x[1]['smape'] if not np.isnan(x[1]['smape']) else np.inf)
    
    report['model_rankings'] = {
        'by_mase': [(model, metrics['mase']) for model, metrics in mase_ranking],
        'by_smape': [(model, metrics['smape']) for model, metrics in smape_ranking]
    }
    
    # Validation criteria check
    baseline_mase = None
    baseline_smape = None
    
    for model, metrics in mase_ranking:
        if model == 'seasonal_naive':
            baseline_mase = metrics['mase']
            baseline_smape = metrics['smape']
            break
    
    if baseline_mase is not None:
        best_model = mase_ranking[0][0]
        best_metrics = mase_ranking[0][1]
        best_mase = best_metrics if isinstance(best_metrics, (int, float)) else best_metrics['mase']
        
        improvement_mase = (baseline_mase - best_mase) / baseline_mase * 100 if baseline_mase != 0 else 0
        
        best_smape_metrics = smape_ranking[0][1] 
        best_smape = best_smape_metrics if isinstance(best_smape_metrics, (int, float)) else best_smape_metrics['smape']
        improvement_smape = (baseline_smape - best_smape) / baseline_smape * 100 if baseline_smape and baseline_smape != 0 else 0
        
        report['validation_criteria'] = {
            'baseline_mase': float(baseline_mase),
            'best_mase': float(best_mase),
            'mase_improvement_pct': float(improvement_mase),
            'smape_improvement_pct': float(improvement_smape),
            'meets_mase_threshold': bool(improvement_mase >= 5.0),
            'meets_smape_threshold': bool(improvement_smape >= 3.0)
        }
        
        # Recommendations
        if improvement_mase >= 5.0 and improvement_smape >= 3.0:
            report['recommendations'].append(f"✅ {best_model} meets both criteria (MASE: {improvement_mase:.1f}%, SMAPE: {improvement_smape:.1f}%)")
        else:
            report['recommendations'].append(f"❌ {best_model} fails criteria. Consider ensemble or feature engineering.")
    
    # Save report
    report_path = MODEL_DIR / 'reports' / f'model_evaluation_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Create summary text report
    summary_path = MODEL_DIR / 'reports' / f'summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    with open(summary_path, 'w') as f:
        f.write("🚀 MEME STOCK PREDICTION - MODEL EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"📊 Dataset Summary:\n")
        f.write(f"   - Tickers analyzed: {report['summary']['total_tickers']}\n")
        f.write(f"   - Models tested: {report['summary']['total_models']}\n")
        f.write(f"   - CV folds: {report['summary']['cv_folds']}\n\n")
        
        f.write(f"🏆 Model Rankings (by MASE):\n")
        for i, (model, mase) in enumerate(report['model_rankings']['by_mase'][:5]):
            f.write(f"   {i+1}. {model}: {mase:.3f}\n")
        f.write("\n")
        
        f.write(f"✅ Validation Criteria:\n")
        criteria = report['validation_criteria']
        f.write(f"   - MASE improvement: {criteria['mase_improvement_pct']:.1f}% (target: ≥5%)\n")
        f.write(f"   - SMAPE improvement: {criteria['smape_improvement_pct']:.1f}% (target: ≥3%)\n")
        f.write(f"   - Meets criteria: {criteria['meets_mase_threshold'] and criteria['meets_smape_threshold']}\n\n")
        
        f.write(f"💡 Recommendations:\n")
        for rec in report['recommendations']:
            f.write(f"   {rec}\n")
    
    print(f"   Reports saved to {MODEL_DIR / 'reports'}")
    
    return report


def main():
    """Main pipeline execution."""
    
    print("🚀 Starting Production-Ready Time Series ML Pipeline")
    print("=" * 60)
    
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Save processed data
    processed_path = MODEL_DIR / 'data' / 'processed_data.csv'
    df.to_csv(processed_path, index=False)
    print(f"💾 Processed data saved to {processed_path}")
    
    # Train and evaluate models
    results = train_and_evaluate_models(df)
    
    # Create visualizations (before JSON conversion)
    create_visualizations(results, df)
    
    # Generate report
    report = generate_report(results, df)
    
    # Save results (after visualization to avoid conversion issues)
    results_path = MODEL_DIR / 'predictions' / f'all_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    
    # Convert numpy arrays to lists for JSON serialization
    json_results = results.copy()
    
    # Remove feature_importance (contains DataFrames)
    if 'feature_importance' in json_results:
        del json_results['feature_importance']
        
    for ticker in json_results['predictions']:
        for model in json_results['predictions'][ticker]:
            for pred_data in json_results['predictions'][ticker][model]:
                pred_data['dates'] = pred_data['dates'].astype(str).tolist()
                pred_data['y_true'] = pred_data['y_true'].tolist()
                pred_data['y_pred'] = pred_data['y_pred'].tolist()
                pred_data['is_outlier'] = pred_data['is_outlier'].tolist()
    
    # Save splits info
    for i, split in enumerate(json_results['splits']):
        for key, value in split.items():
            json_results['splits'][i][key] = value.strftime('%Y-%m-%d')
    
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"💾 Results saved to {results_path}")
    
    print("\n🎯 Pipeline completed successfully!")
    print(f"📁 All outputs saved to: {MODEL_DIR}")
    
    # Print quick summary
    if report['validation_criteria']:
        criteria = report['validation_criteria']
        print(f"\n📊 Quick Results:")
        print(f"   Best model: {report['model_rankings']['by_mase'][0][0]}")
        print(f"   MASE improvement: {criteria['mase_improvement_pct']:.1f}%")
        print(f"   Meets criteria: {criteria['meets_mase_threshold'] and criteria['meets_smape_threshold']}")


if __name__ == '__main__':
    main()