#!/usr/bin/env python3
"""
Enhanced Price Prediction Training with Advanced Reddit Features

Improvements:
1. Advanced Reddit feature engineering
2. Ensemble models  
3. Cross-validation improvements
4. Feature selection and regularization
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
from scipy import stats
from scipy.stats import spearmanr

# ML
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
import lightgbm as lgb
import xgboost as xgb

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "price_prediction_enhanced"
PROCESSED_DIR = DATA_DIR / "processed" / "targets"

# Create directories
MODEL_DIR.mkdir(parents=True, exist_ok=True)
for subdir in ['models', 'predictions', 'metrics', 'plots', 'reports']:
    (MODEL_DIR / subdir).mkdir(parents=True, exist_ok=True)


def load_price_targets() -> pd.DataFrame:
    """Load the latest price targets dataset."""
    print("📊 Loading price targets...")
    
    # Find latest aligned targets file
    target_files = list(PROCESSED_DIR.glob('price_targets_aligned_*.csv'))
    if not target_files:
        raise FileNotFoundError("No price targets found. Run make_targets_price.py first.")
    
    latest_file = max(target_files, key=lambda x: x.stat().st_mtime)
    df = pd.read_csv(latest_file)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"   Loaded {len(df)} records from {latest_file.name}")
    
    # Clean infinite/nan values
    before_clean = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Remove rows with inf/nan targets
    df = df.dropna(subset=['y1d', 'y5d'])
    
    print(f"   After cleaning: {len(df)} records ({before_clean - len(df)} removed)")
    print(f"   Tickers: {sorted(df['ticker'].unique())}")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    
    return df


def engineer_advanced_reddit_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create advanced Reddit features for improved prediction."""
    print("🔧 Engineering advanced Reddit features...")
    
    df = df.copy().sort_values(['ticker', 'date'])
    
    # Basic features (already present)
    reddit_features = ['log_mentions', 'reddit_ema_3', 'reddit_ema_5', 'reddit_ema_10',
                      'reddit_surprise', 'reddit_market_ex', 'reddit_spike_p95']
    
    # Advanced momentum features
    for window in [7, 14, 21]:
        df[f'reddit_momentum_{window}'] = df.groupby('ticker')['log_mentions'].transform(
            lambda x: x.shift(1).rolling(window).mean() - x.shift(window+1).rolling(window).mean()
        )
    
    # Volatility of mentions (proxy for sentiment uncertainty)
    for window in [5, 10, 20]:
        df[f'reddit_vol_{window}'] = df.groupby('ticker')['log_mentions'].transform(
            lambda x: x.shift(1).rolling(window).std()
        )
    
    # Acceleration (second derivative of mentions)
    df['reddit_accel'] = df.groupby('ticker')['log_mentions'].transform(
        lambda x: x.shift(1) - 2*x.shift(2) + x.shift(3)
    )
    
    # Cross-sectional rank features
    for lag in [1, 3, 5]:
        df[f'reddit_rank_{lag}d'] = df.groupby('date')['log_mentions'].transform(
            lambda x: x.shift(lag).rank(pct=True)
        )
    
    # Regime change indicators
    df['reddit_regime_high'] = df.groupby('ticker')['log_mentions'].transform(
        lambda x: (x.shift(1) > x.shift(1).rolling(60).quantile(0.8)).astype(int)
    )
    
    df['reddit_regime_low'] = df.groupby('ticker')['log_mentions'].transform(
        lambda x: (x.shift(1) < x.shift(1).rolling(60).quantile(0.2)).astype(int)
    )
    
    # Market correlation feature
    market_mentions = df.groupby('date')['log_mentions'].sum()
    df['market_mentions_total'] = df['date'].map(market_mentions)
    
    df['reddit_vs_market_corr'] = df.groupby('ticker').apply(
        lambda x: x['log_mentions'].shift(1).rolling(30).corr(x['market_mentions_total'].shift(1))
    ).values
    
    # Weekend/weekday effects (if we have day-of-week data)
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_monday'] = (df['day_of_week'] == 0).astype(int)
    
    print(f"   Created {len([col for col in df.columns if 'reddit' in col])} Reddit features")
    
    return df


def create_expanding_cv_splits(df: pd.DataFrame, min_train_months=6, val_months=1, step_months=1) -> List[Dict]:
    """Create expanding window CV splits with 1-day gap."""
    print(f"📅 Creating expanding CV splits...")
    
    splits = []
    start_date = df['date'].min()
    end_date = df['date'].max()
    
    # First training window starts after min_train_months
    first_split_date = start_date + pd.DateOffset(months=min_train_months)
    current_date = first_split_date
    
    while current_date + pd.DateOffset(months=val_months) <= end_date:
        # Training: from start to current_date
        train_end = current_date - pd.Timedelta(days=1)  # 1-day gap
        
        # Validation: val_months starting from current_date
        val_start = current_date
        val_end = current_date + pd.DateOffset(months=val_months) - pd.Timedelta(days=1)
        
        if val_end > end_date:
            break
        
        splits.append({
            'train_start': start_date,
            'train_end': train_end,
            'val_start': val_start,
            'val_end': val_end
        })
        
        current_date += pd.DateOffset(months=step_months)
    
    print(f"   Created {len(splits)} expanding splits")
    if splits:
        print(f"   First split: train until {splits[0]['train_end'].date()}, val {splits[0]['val_start'].date()}-{splits[0]['val_end'].date()}")
        print(f"   Last split: train until {splits[-1]['train_end'].date()}, val {splits[-1]['val_start'].date()}-{splits[-1]['val_end'].date()}")
    
    return splits


def calculate_ic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Calculate Information Coefficient and related metrics."""
    
    # Remove nan/inf values
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return {'ic': 0.0, 'ic_p_value': 1.0, 'rank_ic': 0.0, 'n_valid': 0}
    
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    # Spearman correlation (rank IC) - more robust
    rank_ic, rank_p = spearmanr(y_pred_clean, y_true_clean)
    
    # Pearson correlation (regular IC)
    if len(y_true_clean) > 2:
        ic, ic_p = stats.pearsonr(y_pred_clean, y_true_clean)
    else:
        ic, ic_p = 0.0, 1.0
    
    return {
        'ic': ic,
        'ic_p_value': ic_p,
        'rank_ic': rank_ic,
        'rank_ic_p_value': rank_p,
        'n_valid': len(y_true_clean)
    }


class EnhancedRedditLGBM:
    """Enhanced LGBM with advanced Reddit features and feature selection."""
    
    def __init__(self, name: str = 'Enhanced Reddit LGBM'):
        self.name = name
        self.model = None
        self.feature_selector = None
        self.selected_features = None
        
        # Base features
        self.price_features = [
            'returns_1d', 'returns_3d', 'returns_5d', 'returns_10d',
            'vol_5d', 'vol_10d', 'vol_20d',
            'price_ratio_sma10', 'price_ratio_sma20', 'rsi_14',
            'volume_ratio', 'turnover'
        ]
        
        # Enhanced Reddit features
        self.reddit_features = [
            'log_mentions', 'reddit_ema_3', 'reddit_ema_5', 'reddit_ema_10',
            'reddit_surprise', 'reddit_market_ex', 'reddit_spike_p95',
            'reddit_momentum_7', 'reddit_momentum_14', 'reddit_momentum_21',
            'reddit_vol_5', 'reddit_vol_10', 'reddit_vol_20',
            'reddit_accel', 'reddit_rank_1d', 'reddit_rank_3d', 'reddit_rank_5d',
            'reddit_regime_high', 'reddit_regime_low', 'reddit_vs_market_corr',
            'is_monday'
        ]
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        # Combine available features
        available_features = []
        available_features.extend([col for col in self.price_features if col in X.columns])
        available_features.extend([col for col in self.reddit_features if col in X.columns])
        
        if not available_features:
            self.selected_features = []
            return
        
        # Prepare training data
        X_train = X[available_features].fillna(0)
        
        # Feature selection using SelectKBest
        k = min(20, len(available_features))  # Select top 20 features or all if less
        self.feature_selector = SelectKBest(score_func=f_regression, k=k)
        X_selected = self.feature_selector.fit_transform(X_train, y)
        
        # Get selected feature names
        selected_mask = self.feature_selector.get_support()
        self.selected_features = [feat for feat, selected in zip(available_features, selected_mask) if selected]
        
        # Train LightGBM with optimized hyperparameters
        self.model = lgb.LGBMRegressor(
            objective='regression',  # Changed from quantile for better IC optimization
            num_leaves=127,
            max_depth=8,
            learning_rate=0.03,
            n_estimators=500,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbosity=-1
        )
        
        self.model.fit(X_selected, y)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None or not self.selected_features:
            return np.zeros(len(X))
        
        X_pred = X[self.selected_features].fillna(0)
        X_selected = self.feature_selector.transform(X_pred)
        
        return self.model.predict(X_selected)


class EnsembleModel:
    """Ensemble of multiple models for improved performance."""
    
    def __init__(self, name: str = 'Ensemble Model'):
        self.name = name
        self.models = []
        self.weights = []
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        # Initialize individual models
        models_config = [
            ('lgbm', EnhancedRedditLGBM('LGBM')),
            ('xgb', self._create_xgb_model()),
            ('ridge', self._create_ridge_model())
        ]
        
        self.models = []
        model_scores = []
        
        # Train each model and calculate validation score
        from sklearn.model_selection import cross_val_score
        
        for name, model in models_config:
            try:
                model.fit(X, y)
                
                # Simple validation score (could be improved with proper CV)
                predictions = model.predict(X)
                ic_score = calculate_ic_metrics(y.values, predictions)['rank_ic']
                
                if not np.isnan(ic_score):
                    self.models.append((name, model))
                    model_scores.append(max(0, ic_score))  # Ensure non-negative
                    
            except Exception as e:
                print(f"   Warning: Model {name} failed to train: {e}")
                continue
        
        # Calculate ensemble weights based on IC scores
        if model_scores:
            total_score = sum(model_scores)
            self.weights = [score / total_score if total_score > 0 else 1/len(model_scores) 
                          for score in model_scores]
        else:
            self.weights = [1.0]  # Fallback if no models work
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.models:
            return np.zeros(len(X))
        
        predictions = []
        for (name, model), weight in zip(self.models, self.weights):
            try:
                pred = model.predict(X)
                predictions.append(pred * weight)
            except:
                continue
        
        if predictions:
            return np.sum(predictions, axis=0)
        else:
            return np.zeros(len(X))
    
    def _create_xgb_model(self):
        """Create XGBoost model wrapper."""
        class XGBWrapper:
            def __init__(self):
                self.model = None
                self.feature_cols = []
                
            def fit(self, X: pd.DataFrame, y: pd.Series):
                # Use available features
                price_features = [
                    'returns_1d', 'returns_3d', 'returns_5d',
                    'vol_5d', 'vol_10d', 'price_ratio_sma10', 'rsi_14'
                ]
                reddit_features = [
                    'log_mentions', 'reddit_ema_5', 'reddit_surprise', 
                    'reddit_market_ex', 'reddit_momentum_14'
                ]
                
                available_features = []
                available_features.extend([col for col in price_features if col in X.columns])
                available_features.extend([col for col in reddit_features if col in X.columns])
                
                if available_features:
                    self.feature_cols = available_features
                    X_train = X[self.feature_cols].fillna(0)
                    
                    self.model = xgb.XGBRegressor(
                        n_estimators=200,
                        max_depth=6,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42,
                        verbosity=0
                    )
                    
                    self.model.fit(X_train, y)
            
            def predict(self, X: pd.DataFrame) -> np.ndarray:
                if self.model is None or not self.feature_cols:
                    return np.zeros(len(X))
                
                X_pred = X[self.feature_cols].fillna(0)
                return self.model.predict(X_pred)
        
        return XGBWrapper()
    
    def _create_ridge_model(self):
        """Create Ridge regression wrapper."""
        class RidgeWrapper:
            def __init__(self):
                self.model = None
                self.feature_cols = []
                
            def fit(self, X: pd.DataFrame, y: pd.Series):
                # Simple feature set for Ridge
                features = ['returns_1d', 'returns_3d', 'vol_5d', 'log_mentions', 'reddit_ema_5']
                available_features = [col for col in features if col in X.columns]
                
                if available_features:
                    self.feature_cols = available_features
                    X_train = X[self.feature_cols].fillna(0)
                    
                    self.model = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0], cv=5)
                    self.model.fit(X_train, y)
            
            def predict(self, X: pd.DataFrame) -> np.ndarray:
                if self.model is None or not self.feature_cols:
                    return np.zeros(len(X))
                
                X_pred = X[self.feature_cols].fillna(0)
                return self.model.predict(X_pred)
        
        return RidgeWrapper()


class RandomWalkBaseline:
    """Random walk baseline: predict 0 return."""
    
    def __init__(self):
        self.name = 'Random Walk'
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        pass  # No training needed
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X))


class PriceOnlyLGBM:
    """Price-only LightGBM with technical indicators."""
    
    def __init__(self):
        self.name = 'Price-Only LGBM'
        self.model = None
        self.feature_cols = [
            'returns_1d', 'returns_3d', 'returns_5d', 'returns_10d',
            'vol_5d', 'vol_10d', 'vol_20d',
            'price_ratio_sma10', 'price_ratio_sma20',
            'rsi_14', 'volume_ratio', 'turnover'
        ]
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        # Use available features
        available_features = [col for col in self.feature_cols if col in X.columns]
        self.feature_cols = available_features
        
        if self.feature_cols:
            X_train = X[self.feature_cols].fillna(0)
            self.model = lgb.LGBMRegressor(
                objective='regression',
                num_leaves=31,
                learning_rate=0.05,
                n_estimators=200,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=-1
            )
            self.model.fit(X_train, y)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None or not self.feature_cols:
            return np.zeros(len(X))
        
        X_pred = X[self.feature_cols].fillna(0)
        return self.model.predict(X_pred)


def evaluate_enhanced_models(df: pd.DataFrame, target_col: str = 'y1d') -> Dict:
    """Evaluate enhanced models using expanding CV."""
    
    print(f"\n🚀 Evaluating enhanced models for {target_col}...")
    
    # Engineer advanced features
    df_enhanced = engineer_advanced_reddit_features(df)
    
    # Create CV splits
    splits = create_expanding_cv_splits(df_enhanced)
    
    # Initialize models
    models = {
        'rw': RandomWalkBaseline(),
        'price_lgbm': PriceOnlyLGBM(),
        'enhanced_reddit': EnhancedRedditLGBM(),
        'ensemble': EnsembleModel()
    }
    
    # Store results
    results = {
        'predictions': {name: [] for name in models.keys()},
        'metrics': {name: [] for name in models.keys()},
    }
    
    print(f"   Training {len(models)} models on {min(8, len(splits))} CV splits...")
    
    for split_idx, split_info in enumerate(splits[:8]):  # Limit to 8 splits
        print(f"\n   Split {split_idx + 1}/{min(8, len(splits))}: {split_info['val_start'].date()} to {split_info['val_end'].date()}")
        
        # Create train/val sets
        train_mask = (df_enhanced['date'] >= split_info['train_start']) & (df_enhanced['date'] <= split_info['train_end'])
        val_mask = (df_enhanced['date'] >= split_info['val_start']) & (df_enhanced['date'] <= split_info['val_end'])
        
        train_data = df_enhanced[train_mask].copy()
        val_data = df_enhanced[val_mask].copy()
        
        if len(train_data) == 0 or len(val_data) == 0:
            continue
        
        print(f"      Train: {len(train_data)} samples, Val: {len(val_data)} samples")
        
        # Train and evaluate each model
        for model_name, model in models.items():
            try:
                # Train model
                model.fit(train_data, train_data[target_col])
                
                # Make predictions
                predictions = model.predict(val_data)
                
                # Store predictions
                pred_df = val_data[['date', 'ticker', target_col]].copy()
                pred_df['y_pred'] = predictions
                pred_df['model'] = model_name
                pred_df['split'] = split_idx
                
                results['predictions'][model_name].append(pred_df)
                
                # Calculate IC metrics
                ic_metrics = calculate_ic_metrics(val_data[target_col].values, predictions)
                ic_metrics.update({
                    'split': split_idx,
                    'model': model_name,
                    'rmse': np.sqrt(mean_squared_error(val_data[target_col], predictions)),
                    'mae': mean_absolute_error(val_data[target_col], predictions)
                })
                
                results['metrics'][model_name].append(ic_metrics)
                
                print(f"         {model_name}: IC={ic_metrics['rank_ic']:.3f}, RMSE={ic_metrics['rmse']:.4f}")
                
            except Exception as e:
                print(f"         {model_name}: Error - {e}")
                continue
    
    return results


def generate_enhanced_report(results: Dict, target_col: str = 'y1d') -> Dict:
    """Generate enhanced comparison report."""
    
    print(f"\n📋 Generating enhanced report...")
    
    report = {
        'target': target_col,
        'generated_at': datetime.now().isoformat(),
        'model_performance': {},
        'go_no_go_decision': {},
        'recommendations': []
    }
    
    # Aggregate metrics across splits
    for model_name in results['metrics'].keys():
        if results['metrics'][model_name]:
            metrics_df = pd.DataFrame(results['metrics'][model_name])
            
            # Model performance summary
            perf_summary = {
                'avg_ic': float(metrics_df['ic'].mean()),
                'avg_rank_ic': float(metrics_df['rank_ic'].mean()),
                'ic_hit_rate': float((metrics_df['rank_ic'] > 0).mean()),
                'avg_rmse': float(metrics_df['rmse'].mean()),
                'n_splits': len(metrics_df),
                'ic_std': float(metrics_df['rank_ic'].std())
            }
            
            report['model_performance'][model_name] = perf_summary
    
    # Go/No-Go decision
    baseline_ic = report['model_performance'].get('price_lgbm', {}).get('avg_rank_ic', 0.0)
    enhanced_ic = report['model_performance'].get('enhanced_reddit', {}).get('avg_rank_ic', 0.0)
    ensemble_ic = report['model_performance'].get('ensemble', {}).get('avg_rank_ic', 0.0)
    
    best_reddit_ic = max(enhanced_ic, ensemble_ic)
    ic_improvement = best_reddit_ic - baseline_ic
    
    report['go_no_go_decision'] = {
        'ic_improvement': float(ic_improvement),
        'ic_improvement_threshold': 0.03,
        'meets_ic_threshold': bool(ic_improvement >= 0.03),
        'best_model': 'ensemble' if ensemble_ic > enhanced_ic else 'enhanced_reddit',
        'best_ic': float(best_reddit_ic),
        'overall_decision': 'GO' if ic_improvement >= 0.03 else 'CONTINUE-IMPROVING'
    }
    
    # Recommendations
    decision = report['go_no_go_decision']
    if decision['overall_decision'] == 'GO':
        report['recommendations'].append("✅ Enhanced Reddit features meet Go/No-Go criteria!")
        report['recommendations'].append(f"📈 IC improvement: {ic_improvement:.3f}")
        report['recommendations'].append("🚀 Ready for deep learning experiments on Colab.")
    else:
        report['recommendations'].append(f"🔄 IC improvement ({ic_improvement:.3f}) - continue feature engineering")
        report['recommendations'].append("📊 Current best features show promise")
        report['recommendations'].append("🧠 Ready to try deep learning approaches")
    
    return report


def main():
    """Main execution."""
    
    print("🚀 Enhanced Price Prediction Training and Evaluation")
    print("=" * 60)
    
    # Load data
    df = load_price_targets()
    
    # Evaluate enhanced models
    results = evaluate_enhanced_models(df, target_col='y1d')
    
    # Generate report
    report = generate_enhanced_report(results, target_col='y1d')
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save report
    report_path = MODEL_DIR / 'reports' / f'enhanced_report_{timestamp}.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Enhanced evaluation complete!")
    print(f"📁 Results saved to: {MODEL_DIR}")
    
    # Print key results
    decision = report['go_no_go_decision']
    print(f"\n🎯 DECISION: {decision['overall_decision']}")
    print(f"   IC Improvement: {decision['ic_improvement']:.3f} (target: ≥0.03)")
    print(f"   Best Model: {decision['best_model']}")
    print(f"   Best IC: {decision['best_ic']:.3f}")
    
    # Model comparison
    print(f"\n📊 Model Performance:")
    for model, perf in report['model_performance'].items():
        print(f"   {model}: IC={perf['avg_rank_ic']:.3f} (±{perf['ic_std']:.3f})")
    
    return report


if __name__ == '__main__':
    main()