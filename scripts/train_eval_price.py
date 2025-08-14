#!/usr/bin/env python3
"""
Price Prediction Training and Evaluation Pipeline

Implements:
1. Price-only baselines (RW, AR, Price-only LGBM)
2. Reddit-enhanced models
3. Expanding monthly CV with 1-day gap
4. IC/IR metrics and portfolio backtesting
5. DM tests and Go/No-Go validation
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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "price_prediction"
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


def calculate_portfolio_metrics(returns_df: pd.DataFrame, strategy_name: str = 'equal_weight') -> Dict:
    """Calculate portfolio-level metrics including IR."""
    
    if len(returns_df) == 0:
        return {'ir': 0.0, 'sharpe': 0.0, 'total_return': 0.0, 'max_dd': 0.0, 'n_trades': 0}
    
    # Portfolio daily returns (equal weight or market cap weight)
    daily_returns = returns_df.mean(axis=1)  # Equal weight across tickers
    
    # Calculate metrics
    total_return = (1 + daily_returns).prod() - 1
    annualized_return = (1 + daily_returns.mean()) ** 252 - 1
    annualized_vol = daily_returns.std() * np.sqrt(252)
    
    sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0.0
    
    # IR is often used interchangeably with Sharpe for long-only strategies
    ir = sharpe
    
    # Maximum drawdown
    cumulative = (1 + daily_returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative / rolling_max) - 1
    max_dd = drawdown.min()
    
    # Number of trades (positions changes)
    n_trades = len(daily_returns)
    
    return {
        'ir': ir,
        'sharpe': sharpe,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'annualized_vol': annualized_vol,
        'max_dd': max_dd,
        'n_trades': n_trades,
        'avg_daily_return': daily_returns.mean(),
        'daily_vol': daily_returns.std()
    }


def simulate_strategy(predictions_df: pd.DataFrame, target_col: str = 'y1d', 
                     strategy_type: str = 'long_only', top_k: float = 0.1) -> pd.DataFrame:
    """Simulate trading strategy based on predictions."""
    
    strategy_returns = []
    
    # Group by date for daily rebalancing
    for date, day_data in predictions_df.groupby('date'):
        if len(day_data) < 2:
            continue
        
        # Rank by predictions
        day_data = day_data.copy()
        day_data['pred_rank'] = day_data['y_pred'].rank(ascending=False, pct=True)
        
        if strategy_type == 'long_only':
            # Go long top K% of predictions
            selected = day_data[day_data['pred_rank'] <= top_k]
            if len(selected) > 0:
                # Equal weight portfolio return
                portfolio_return = selected[target_col].mean()
                strategy_returns.append({'date': date, 'portfolio_return': portfolio_return, 'n_positions': len(selected)})
        
        elif strategy_type == 'long_short':
            # Long top decile, short bottom decile
            long_positions = day_data[day_data['pred_rank'] <= 0.1]
            short_positions = day_data[day_data['pred_rank'] >= 0.9]
            
            long_return = long_positions[target_col].mean() if len(long_positions) > 0 else 0.0
            short_return = -short_positions[target_col].mean() if len(short_positions) > 0 else 0.0  # Short position
            
            portfolio_return = (long_return + short_return) / 2  # Equal weight long/short
            n_positions = len(long_positions) + len(short_positions)
            
            strategy_returns.append({'date': date, 'portfolio_return': portfolio_return, 'n_positions': n_positions})
    
    return pd.DataFrame(strategy_returns)


class PricePredictor:
    """Base class for price prediction models."""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.feature_cols = []
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        raise NotImplementedError
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class RandomWalkBaseline(PricePredictor):
    """Random walk baseline: predict 0 return."""
    
    def __init__(self):
        super().__init__('Random Walk')
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        pass  # No training needed
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X))


class ARBaseline(PricePredictor):
    """Autoregressive baseline using lagged returns."""
    
    def __init__(self, lags: List[int] = [1, 3, 5]):
        super().__init__(f'AR({max(lags)})')
        self.lags = lags
        self.feature_cols = [f'returns_{lag}d' for lag in lags if lag <= 10]  # Use available return features
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        available_features = [col for col in self.feature_cols if col in X.columns]
        if not available_features:
            available_features = [col for col in X.columns if 'returns' in col and 'lag' not in col][:3]
        
        self.feature_cols = available_features[:3]  # Use first 3 available return features
        
        if self.feature_cols:
            self.model = LinearRegression()
            X_train = X[self.feature_cols].fillna(0)
            self.model.fit(X_train, y)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None or not self.feature_cols:
            return np.zeros(len(X))
        
        X_pred = X[self.feature_cols].fillna(0)
        return self.model.predict(X_pred)


class PriceOnlyLGBM(PricePredictor):
    """Price-only LightGBM with technical indicators."""
    
    def __init__(self):
        super().__init__('Price-Only LGBM')
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


class RedditEnhancedLGBM(PricePredictor):
    """LGBM with Reddit features added."""
    
    def __init__(self):
        super().__init__('Reddit-Enhanced LGBM')
        self.price_features = [
            'returns_1d', 'returns_3d', 'returns_5d',
            'vol_5d', 'vol_10d', 'price_ratio_sma10', 'rsi_14'
        ]
        self.reddit_features = [
            'log_mentions', 'reddit_ema_3', 'reddit_ema_5', 'reddit_ema_10',
            'reddit_surprise', 'reddit_market_ex', 'reddit_spike_p95'
        ]
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        # Combine available price and Reddit features
        available_features = []
        available_features.extend([col for col in self.price_features if col in X.columns])
        available_features.extend([col for col in self.reddit_features if col in X.columns])
        
        self.feature_cols = available_features
        
        if self.feature_cols:
            X_train = X[self.feature_cols].fillna(0)
            self.model = lgb.LGBMRegressor(
                objective='quantile',
                alpha=0.5,  # Median regression for robustness
                num_leaves=63,
                learning_rate=0.05,
                n_estimators=300,
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


def diebold_mariano_test(errors1: np.ndarray, errors2: np.ndarray) -> Tuple[float, float]:
    """Diebold-Mariano test for comparing forecast accuracy."""
    
    d = errors1**2 - errors2**2
    d_mean = d.mean()
    
    if len(d) < 2 or d.var() == 0:
        return np.nan, 1.0
    
    dm_stat = d_mean / np.sqrt(d.var() / len(d))
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    
    return dm_stat, p_value


def evaluate_models(df: pd.DataFrame, target_col: str = 'y1d') -> Dict:
    """Evaluate all models using expanding CV."""
    
    print(f"\n🚀 Evaluating models for {target_col}...")
    
    # Create CV splits
    splits = create_expanding_cv_splits(df)
    
    # Initialize models
    models = {
        'rw': RandomWalkBaseline(),
        'ar': ARBaseline(),
        'price_lgbm': PriceOnlyLGBM(),
        'reddit_lgbm': RedditEnhancedLGBM()
    }
    
    # Store results
    results = {
        'predictions': {name: [] for name in models.keys()},
        'metrics': {name: [] for name in models.keys()},
        'strategies': {name: [] for name in models.keys()}
    }
    
    print(f"   Training {len(models)} models on {len(splits)} CV splits...")
    
    for split_idx, split_info in enumerate(splits[:6]):  # Limit to 6 splits for demo
        print(f"\n   Split {split_idx + 1}/{min(6, len(splits))}: {split_info['val_start'].date()} to {split_info['val_end'].date()}")
        
        # Create train/val sets
        train_mask = (df['date'] >= split_info['train_start']) & (df['date'] <= split_info['train_end'])
        val_mask = (df['date'] >= split_info['val_start']) & (df['date'] <= split_info['val_end'])
        
        train_data = df[train_mask].copy()
        val_data = df[val_mask].copy()
        
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
                
                # Strategy simulation
                strategy_returns = simulate_strategy(pred_df, target_col, 'long_only', 0.2)  # Top 20%
                if len(strategy_returns) > 0:
                    portfolio_metrics = calculate_portfolio_metrics(strategy_returns.set_index('date')[['portfolio_return']])
                    portfolio_metrics.update({'split': split_idx, 'model': model_name})
                    results['strategies'][model_name].append(portfolio_metrics)
                
                print(f"         {model_name}: IC={ic_metrics['rank_ic']:.3f}, RMSE={ic_metrics['rmse']:.4f}")
                
            except Exception as e:
                print(f"         {model_name}: Error - {e}")
                continue
    
    return results


def generate_comparison_report(results: Dict, target_col: str = 'y1d') -> Dict:
    """Generate comprehensive comparison report."""
    
    print(f"\n📋 Generating comparison report...")
    
    report = {
        'target': target_col,
        'generated_at': datetime.now().isoformat(),
        'model_performance': {},
        'statistical_tests': {},
        'go_no_go_decision': {},
        'recommendations': []
    }
    
    # Aggregate metrics across splits
    for model_name in results['metrics'].keys():
        if results['metrics'][model_name]:
            metrics_df = pd.DataFrame(results['metrics'][model_name])
            strategy_df = pd.DataFrame(results['strategies'][model_name]) if results['strategies'][model_name] else pd.DataFrame()
            
            # Model performance summary
            perf_summary = {
                'avg_ic': float(metrics_df['ic'].mean()),
                'avg_rank_ic': float(metrics_df['rank_ic'].mean()),
                'ic_hit_rate': float((metrics_df['rank_ic'] > 0).mean()),
                'avg_rmse': float(metrics_df['rmse'].mean()),
                'n_splits': len(metrics_df)
            }
            
            if not strategy_df.empty:
                perf_summary.update({
                    'avg_ir': float(strategy_df['ir'].mean()),
                    'avg_sharpe': float(strategy_df['sharpe'].mean()),
                    'avg_total_return': float(strategy_df['total_return'].mean()),
                    'avg_max_dd': float(strategy_df['max_dd'].mean())
                })
            
            report['model_performance'][model_name] = perf_summary
    
    # Statistical tests (DM test against Random Walk)
    if 'rw' in results['predictions'] and results['predictions']['rw']:
        rw_preds = pd.concat(results['predictions']['rw'])
        
        for model_name in results['predictions'].keys():
            if model_name != 'rw' and results['predictions'][model_name]:
                model_preds = pd.concat(results['predictions'][model_name])
                
                # Align predictions by date and ticker
                merged = rw_preds.merge(model_preds, on=['date', 'ticker'], suffixes=('_rw', '_model'))
                
                if len(merged) > 10:
                    rw_errors = np.abs(merged[f'{target_col}_rw'] - merged['y_pred_rw'])
                    model_errors = np.abs(merged[f'{target_col}_model'] - merged['y_pred_model'])
                    
                    dm_stat, dm_p = diebold_mariano_test(rw_errors, model_errors)
                    
                    report['statistical_tests'][model_name] = {
                        'dm_statistic': float(dm_stat) if not np.isnan(dm_stat) else 0.0,
                        'dm_p_value': float(dm_p) if not np.isnan(dm_p) else 1.0,
                        'is_significant': bool(dm_p < 0.05) if not np.isnan(dm_p) else False
                    }
    
    # Go/No-Go decision
    baseline_ic = report['model_performance'].get('price_lgbm', {}).get('avg_rank_ic', 0.0)
    reddit_ic = report['model_performance'].get('reddit_lgbm', {}).get('avg_rank_ic', 0.0)
    
    baseline_ir = report['model_performance'].get('price_lgbm', {}).get('avg_ir', 0.0)
    reddit_ir = report['model_performance'].get('reddit_lgbm', {}).get('avg_ir', 0.0)
    
    ic_improvement = reddit_ic - baseline_ic
    ir_improvement = reddit_ir - baseline_ir
    
    # DM test significance
    dm_significant = report['statistical_tests'].get('reddit_lgbm', {}).get('is_significant', False)
    
    report['go_no_go_decision'] = {
        'ic_improvement': float(ic_improvement),
        'ic_improvement_threshold': 0.03,
        'meets_ic_threshold': bool(ic_improvement >= 0.03),
        'ir_improvement': float(ir_improvement),
        'ir_threshold': 0.3,
        'meets_ir_threshold': bool(reddit_ir >= 0.3),
        'dm_test_significant': dm_significant,
        'overall_decision': 'GO' if (ic_improvement >= 0.03 and reddit_ir >= 0.3 and dm_significant) else 'NO-GO'
    }
    
    # Recommendations
    decision = report['go_no_go_decision']
    if decision['overall_decision'] == 'GO':
        report['recommendations'].append("✅ Reddit features significantly improve price prediction!")
        report['recommendations'].append(f"📈 IC improvement: {ic_improvement:.3f}, IR: {reddit_ir:.3f}")
        report['recommendations'].append("🚀 Ready to deploy Reddit-enhanced model for trading.")
    else:
        if not decision['meets_ic_threshold']:
            report['recommendations'].append(f"❌ IC improvement ({ic_improvement:.3f}) below threshold (0.03)")
        if not decision['meets_ir_threshold']:
            report['recommendations'].append(f"❌ IR ({reddit_ir:.3f}) below threshold (0.3)")
        if not decision['dm_test_significant']:
            report['recommendations'].append("❌ Improvement not statistically significant (DM test)")
        
        report['recommendations'].append("🔄 Consider: better Reddit features, longer training, or alternative strategies.")
    
    return report


def main():
    """Main execution."""
    
    print("🚀 Price Prediction Training and Evaluation Pipeline")
    print("=" * 60)
    
    # Load data
    df = load_price_targets()
    
    # Evaluate models
    results = evaluate_models(df, target_col='y1d')
    
    # Generate report
    report = generate_comparison_report(results, target_col='y1d')
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save detailed results
    results_path = MODEL_DIR / 'predictions' / f'evaluation_results_{timestamp}.json'
    # Convert DataFrames to dict for JSON serialization
    json_results = {}
    for key, value in results.items():
        if key in ['predictions', 'metrics', 'strategies']:
            json_results[key] = {}
            for model_name, model_results in value.items():
                if model_results and isinstance(model_results[0], pd.DataFrame):
                    # Convert DataFrame to dict and handle datetime objects
                    serializable_results = []
                    for df in model_results:
                        df_dict = df.copy()
                        # Convert datetime columns to strings
                        for col in df_dict.columns:
                            if pd.api.types.is_datetime64_any_dtype(df_dict[col]):
                                df_dict[col] = df_dict[col].dt.strftime('%Y-%m-%d')
                        serializable_results.append(df_dict.to_dict('records'))
                    json_results[key][model_name] = serializable_results
                else:
                    json_results[key][model_name] = model_results
        else:
            json_results[key] = value
    
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    # Save report
    report_path = MODEL_DIR / 'reports' / f'price_prediction_report_{timestamp}.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Save summary
    summary_path = MODEL_DIR / 'reports' / f'summary_{timestamp}.txt'
    with open(summary_path, 'w') as f:
        f.write("🚀 PRICE PREDICTION EVALUATION REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"📊 Model Performance Summary:\n")
        for model, perf in report['model_performance'].items():
            f.write(f"   {model}:\n")
            f.write(f"      Rank IC: {perf['avg_rank_ic']:.3f}\n")
            f.write(f"      IR: {perf.get('avg_ir', 0.0):.3f}\n")
            f.write(f"      Hit Rate: {perf['ic_hit_rate']:.1%}\n")
        
        f.write(f"\n🎯 Go/No-Go Decision: {report['go_no_go_decision']['overall_decision']}\n")
        f.write(f"   IC Improvement: {report['go_no_go_decision']['ic_improvement']:.3f}\n")
        f.write(f"   Reddit IR: {report['model_performance'].get('reddit_lgbm', {}).get('avg_ir', 0.0):.3f}\n")
        f.write(f"   DM Test Significant: {report['go_no_go_decision']['dm_test_significant']}\n")
        
        f.write(f"\n💡 Recommendations:\n")
        for rec in report['recommendations']:
            f.write(f"   {rec}\n")
    
    print(f"\n✅ Evaluation complete!")
    print(f"📁 Results saved to: {MODEL_DIR}")
    print(f"📊 Report: {report_path.name}")
    
    # Print key results
    decision = report['go_no_go_decision']
    print(f"\n🎯 DECISION: {decision['overall_decision']}")
    print(f"   IC Improvement: {decision['ic_improvement']:.3f} (target: ≥0.03)")
    reddit_ir = report['model_performance'].get('reddit_lgbm', {}).get('avg_ir', 0.0)
    print(f"   Reddit Model IR: {reddit_ir:.3f} (target: ≥0.3)")
    print(f"   Statistical Significance: {decision['dm_test_significant']}")


if __name__ == '__main__':
    main()