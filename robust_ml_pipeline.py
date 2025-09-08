#!/usr/bin/env python3
"""
Robust Meme Stock ML Pipeline - Revised Implementation
Addresses overfitting, leakage, unrealistic costs, and causality concerns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
from pathlib import Path
import logging
from datetime import datetime, timedelta
import json

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobustMemeStockML:
    """Robust implementation of meme stock ML pipeline with proper validation."""
    
    def __init__(self):
        self.results = {}
        self.feature_importance = {}
        self.causality_results = {}
        
    def generate_realistic_data(self, n_days=1000, n_tickers=3):
        """Generate realistic market data with proper temporal structure."""
        logger.info("Generating realistic market data...")
        
        np.random.seed(42)
        dates = pd.date_range('2021-01-01', periods=n_days, freq='D')
        
        # Generate market factors with realistic correlations
        market_factor = np.random.randn(n_days) * 0.02
        sector_factor = np.random.randn(n_days) * 0.015
        
        data = {}
        tickers = ['GME', 'AMC', 'BB']
        
        for ticker in tickers:
            base_price = {'GME': 100, 'AMC': 50, 'BB': 10}[ticker]
            beta = {'GME': 1.5, 'AMC': 1.2, 'BB': 1.0}[ticker]
            
            # Generate returns with realistic patterns
            returns = []
            for i in range(n_days):
                market_comp = beta * market_factor[i]
                sector_comp = sector_factor[i]
                idiosyncratic = np.random.randn() * 0.03
                
                # Add some mean reversion
                if i > 0:
                    mean_reversion = -0.001 * np.random.randn()
                else:
                    mean_reversion = 0
                
                daily_return = market_comp + sector_comp + idiosyncratic + mean_reversion
                returns.append(daily_return)
            
            # Generate prices
            prices = [base_price]
            for ret in returns:
                prices.append(prices[-1] * (1 + ret))
            
            # Generate Reddit activity with realistic patterns
            reddit_activity = []
            for i, ret in enumerate(returns):
                base_activity = np.random.poisson(50)
                
                # Activity increases with volatility and extreme returns
                volatility_mult = 1 + abs(ret) * 10
                extreme_return_mult = 1 + max(0, abs(ret) - 0.05) * 5
                
                # Add some persistence
                if i > 0:
                    persistence = 0.1 * reddit_activity[-1] / 100
                else:
                    persistence = 0
                
                activity = base_activity * volatility_mult * extreme_return_mult + persistence
                reddit_activity.append(int(max(0, activity)))
            
            # Create DataFrame with proper temporal structure
            df = pd.DataFrame({
                'date': dates,
                'close': prices[:-1],  # Remove last price to match length
                'volume': np.random.lognormal(12, 1, n_days),
                'reddit_mentions': reddit_activity,
                'returns': returns
            })
            
            # Calculate technical indicators (NO LOOK-AHEAD BIAS)
            df['sma_20'] = df['close'].rolling(20).mean()
            df['volatility'] = df['returns'].rolling(20).std()
            df['rsi'] = self._calculate_rsi(df['close'])
            
            data[ticker] = df
        
        return data
    
    def _calculate_rsi(self, prices, window=14):
        """Calculate RSI indicator without look-ahead bias."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def create_leakage_free_features(self, df):
        """Create features without look-ahead bias."""
        logger.info("Creating leakage-free features...")
        
        # Reddit features (using only T-day info)
        df['reddit_ma_5'] = df['reddit_mentions'].rolling(5).mean()
        df['reddit_ma_20'] = df['reddit_mentions'].rolling(20).mean()
        df['reddit_surprise'] = (df['reddit_mentions'] - df['reddit_ma_20']) / df['reddit_ma_20']
        df['reddit_momentum'] = df['reddit_mentions'].rolling(5).sum() / df['reddit_mentions'].rolling(20).sum()
        
        # Technical features (using only T-day info)
        df['price_ratio_sma20'] = df['close'] / df['sma_20']
        df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(50).mean()
        df['rsi_signal'] = (df['rsi'] - 50) / 50
        
        # Calendar features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        
        # Target: next day return (T+1)
        df['target'] = df['returns'].shift(-1)
        
        return df
    
    def test_granger_causality(self, data):
        """Test Granger causality between Reddit signals and returns."""
        logger.info("Testing Granger causality...")
        
        try:
            from statsmodels.tsa.stattools import grangercausalitytests
        except ImportError:
            logger.warning("statsmodels not available, skipping Granger tests")
            return {}
        
        causality_results = {}
        
        for ticker, df in data.items():
            df_clean = df.dropna()
            
            if len(df_clean) > 100:
                # Test: Do Reddit mentions Granger-cause returns?
                test_data = df_clean[['returns', 'reddit_mentions']].values
                
                try:
                    gc_result = grangercausalitytests(test_data, maxlag=5, verbose=False)
                    
                    # Extract p-values
                    p_values = []
                    for lag in range(1, 6):
                        if lag in gc_result:
                            p_value = gc_result[lag][0]['ssr_ftest'][1]
                            p_values.append(p_value)
                    
                    causality_results[ticker] = {
                        'p_values': p_values,
                        'min_p_value': min(p_values) if p_values else 1.0,
                        'significant_lags': [i+1 for i, p in enumerate(p_values) if p < 0.05],
                        'causality_confirmed': min(p_values) < 0.05 if p_values else False
                    }
                    
                except Exception as e:
                    logger.warning(f"Granger test failed for {ticker}: {e}")
                    causality_results[ticker] = {
                        'p_values': [1.0],
                        'min_p_value': 1.0,
                        'significant_lags': [],
                        'causality_confirmed': False
                    }
        
        self.causality_results = causality_results
        return causality_results
    
    def run_purged_kfold_validation(self, data, n_splits=5, embargo_days=5):
        """Run purged K-fold validation with embargo."""
        logger.info("Running purged K-fold validation...")
        
        results = {}
        
        for ticker, df in data.items():
            # Create features
            df_features = self.create_leakage_free_features(df.copy())
            df_clean = df_features.dropna()
            
            if len(df_clean) < 100:
                continue
            
            # Prepare data
            feature_cols = ['reddit_surprise', 'reddit_momentum', 'price_ratio_sma20', 
                          'volatility_ratio', 'rsi_signal', 'day_of_week', 'month']
            X = df_clean[feature_cols]
            y = df_clean['target']
            
            # Purged K-fold with embargo
            n_samples = len(X)
            fold_size = n_samples // n_splits
            
            fold_results = []
            
            for fold in range(n_splits):
                # Define fold boundaries
                start_idx = fold * fold_size
                end_idx = start_idx + fold_size
                
                # Training set: before fold
                train_start = 0
                train_end = start_idx - embargo_days
                
                # Test set: fold
                test_start = start_idx
                test_end = end_idx
                
                if train_end <= 0 or test_start >= n_samples:
                    continue
                
                X_train = X.iloc[train_start:train_end]
                y_train = y.iloc[train_start:train_end]
                X_test = X.iloc[test_start:test_end]
                y_test = y.iloc[test_start:test_end]
                
                if len(X_train) < 50 or len(X_test) < 10:
                    continue
                
                # Train model
                model = Ridge(alpha=1.0)
                scaler = StandardScaler()
                
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Calculate IC (Information Coefficient)
                ic = np.corrcoef(y_test, y_pred)[0, 1]
                
                fold_results.append({
                    'fold': fold,
                    'mse': mse,
                    'r2': r2,
                    'ic': ic,
                    'n_train': len(X_train),
                    'n_test': len(X_test)
                })
            
            results[ticker] = fold_results
        
        return results
    
    def run_walk_forward_validation(self, data, train_window=252, test_window=63):
        """Run walk-forward validation."""
        logger.info("Running walk-forward validation...")
        
        results = {}
        
        for ticker, df in data.items():
            df_features = self.create_leakage_free_features(df.copy())
            df_clean = df_features.dropna()
            
            if len(df_clean) < train_window + test_window:
                continue
            
            feature_cols = ['reddit_surprise', 'reddit_momentum', 'price_ratio_sma20', 
                          'volatility_ratio', 'rsi_signal', 'day_of_week', 'month']
            
            walk_results = []
            
            for start_idx in range(0, len(df_clean) - train_window - test_window, test_window):
                # Training set
                train_end = start_idx + train_window
                X_train = df_clean.iloc[start_idx:train_end][feature_cols]
                y_train = df_clean.iloc[start_idx:train_end]['target']
                
                # Test set
                test_start = train_end
                test_end = test_start + test_window
                X_test = df_clean.iloc[test_start:test_end][feature_cols]
                y_test = df_clean.iloc[test_start:test_end]['target']
                
                if len(X_train) < 50 or len(X_test) < 10:
                    continue
                
                # Train model
                model = Ridge(alpha=1.0)
                scaler = StandardScaler()
                
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                ic = np.corrcoef(y_test, y_pred)[0, 1]
                
                walk_results.append({
                    'start_date': df_clean.iloc[test_start]['date'],
                    'end_date': df_clean.iloc[test_end-1]['date'],
                    'mse': mse,
                    'r2': r2,
                    'ic': ic,
                    'n_train': len(X_train),
                    'n_test': len(X_test)
                })
            
            results[ticker] = walk_results
        
        return results
    
    def run_feature_ablation(self, data):
        """Run feature ablation studies."""
        logger.info("Running feature ablation studies...")
        
        results = {}
        
        for ticker, df in data.items():
            df_features = self.create_leakage_free_features(df.copy())
            df_clean = df_features.dropna()
            
            if len(df_clean) < 100:
                continue
            
            # Define feature sets
            reddit_features = ['reddit_surprise', 'reddit_momentum']
            technical_features = ['price_ratio_sma20', 'volatility_ratio', 'rsi_signal']
            calendar_features = ['day_of_week', 'month']
            
            feature_sets = {
                'reddit_only': reddit_features,
                'technical_only': technical_features,
                'calendar_only': calendar_features,
                'reddit_technical': reddit_features + technical_features,
                'all_features': reddit_features + technical_features + calendar_features
            }
            
            ticker_results = {}
            
            for set_name, features in feature_sets.items():
                X = df_clean[features]
                y = df_clean['target']
                
                # Use time series split for proper validation
                tscv = TimeSeriesSplit(n_splits=5)
                
                fold_results = []
                for train_idx, test_idx in tscv.split(X):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    
                    if len(X_train) < 50 or len(X_test) < 10:
                        continue
                    
                    # Train model
                    model = Ridge(alpha=1.0)
                    scaler = StandardScaler()
                    
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    
                    # Calculate metrics
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    ic = np.corrcoef(y_test, y_pred)[0, 1]
                    
                    fold_results.append({
                        'mse': mse,
                        'r2': r2,
                        'ic': ic
                    })
                
                if fold_results:
                    avg_mse = np.mean([r['mse'] for r in fold_results])
                    avg_r2 = np.mean([r['r2'] for r in fold_results])
                    avg_ic = np.mean([r['ic'] for r in fold_results])
                    
                    ticker_results[set_name] = {
                        'mse': avg_mse,
                        'r2': avg_r2,
                        'ic': avg_ic,
                        'n_folds': len(fold_results)
                    }
            
            results[ticker] = ticker_results
        
        return results
    
    def run_realistic_backtest(self, data, transaction_costs=[0.002, 0.0035, 0.005]):
        """Run realistic backtest with proper transaction costs."""
        logger.info("Running realistic backtest...")
        
        results = {}
        
        for ticker, df in data.items():
            df_features = self.create_leakage_free_features(df.copy())
            df_clean = df_features.dropna()
            
            if len(df_clean) < 100:
                continue
            
            ticker_results = {}
            
            for cost in transaction_costs:
                # Generate signals
                df_clean['signal'] = np.where(
                    df_clean['reddit_surprise'] > 0.5,  # High Reddit activity
                    -1,  # Short (contrarian)
                    np.where(
                        df_clean['reddit_surprise'] < -0.5,  # Low Reddit activity
                        1,  # Long (contrarian)
                        0   # Neutral
                    )
                )
                
                # Calculate strategy returns
                df_clean['strategy_return'] = df_clean['signal'].shift(1) * df_clean['returns']
                
                # Apply transaction costs
                position_changes = df_clean['signal'].diff().abs()
                df_clean['net_return'] = df_clean['strategy_return'] - position_changes * cost
                
                # Calculate metrics
                net_returns = df_clean['net_return'].dropna()
                
                if len(net_returns) > 0:
                    sharpe = net_returns.mean() / net_returns.std() * np.sqrt(252)
                    total_return = (1 + net_returns).prod() - 1
                    hit_rate = (net_returns > 0).mean()
                    turnover = position_changes.mean()
                    
                    # Calculate drawdown
                    cumulative = (1 + net_returns).cumprod()
                    running_max = cumulative.expanding().max()
                    drawdown = (cumulative - running_max) / running_max
                    max_drawdown = drawdown.min()
                    
                    ticker_results[f"{cost:.1%}"] = {
                        'sharpe': sharpe,
                        'total_return': total_return,
                        'hit_rate': hit_rate,
                        'turnover': turnover,
                        'max_drawdown': max_drawdown,
                        'volatility': net_returns.std() * np.sqrt(252)
                    }
            
            results[ticker] = ticker_results
        
        return results
    
    def evaluate_yearly_stability(self, data):
        """Evaluate performance stability across years."""
        logger.info("Evaluating yearly stability...")
        
        results = {}
        
        for ticker, df in data.items():
            df_features = self.create_leakage_free_features(df.copy())
            df_clean = df_features.dropna()
            
            if len(df_clean) < 365:  # Need at least 1 year
                continue
            
            # Add year column
            df_clean['year'] = df_clean['date'].dt.year
            
            yearly_results = {}
            
            for year in df_clean['year'].unique():
                year_data = df_clean[df_clean['year'] == year]
                
                if len(year_data) < 50:
                    continue
                
                # Generate signals
                year_data['signal'] = np.where(
                    year_data['reddit_surprise'] > 0.5,
                    -1,
                    np.where(
                        year_data['reddit_surprise'] < -0.5,
                        1,
                        0
                    )
                )
                
                # Calculate returns
                year_data['strategy_return'] = year_data['signal'].shift(1) * year_data['returns']
                
                # Apply realistic costs
                position_changes = year_data['signal'].diff().abs()
                year_data['net_return'] = year_data['strategy_return'] - position_changes * 0.0035
                
                net_returns = year_data['net_return'].dropna()
                
                if len(net_returns) > 0:
                    sharpe = net_returns.mean() / net_returns.std() * np.sqrt(252)
                    total_return = (1 + net_returns).prod() - 1
                    hit_rate = (net_returns > 0).mean()
                    
                    yearly_results[year] = {
                        'sharpe': sharpe,
                        'total_return': total_return,
                        'hit_rate': hit_rate,
                        'n_days': len(net_returns)
                    }
            
            results[ticker] = yearly_results
        
        return results
    
    def generate_comprehensive_report(self, kfold_results, walk_forward_results, 
                                    ablation_results, backtest_results, yearly_results):
        """Generate comprehensive validation report."""
        logger.info("Generating comprehensive report...")
        
        report = []
        report.append("🔍 ROBUST MEME STOCK ML VALIDATION REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # K-Fold Results
        report.append("\n📊 PURGED K-FOLD VALIDATION RESULTS")
        report.append("-" * 50)
        
        for ticker, folds in kfold_results.items():
            if folds:
                avg_ic = np.mean([f['ic'] for f in folds])
                avg_r2 = np.mean([f['r2'] for f in folds])
                ic_std = np.std([f['ic'] for f in folds])
                
                report.append(f"{ticker}:")
                report.append(f"  Average IC: {avg_ic:.3f} ± {ic_std:.3f}")
                report.append(f"  Average R²: {avg_r2:.3f}")
                report.append(f"  Number of folds: {len(folds)}")
        
        # Walk-Forward Results
        report.append("\n📈 WALK-FORWARD VALIDATION RESULTS")
        report.append("-" * 50)
        
        for ticker, walks in walk_forward_results.items():
            if walks:
                avg_ic = np.mean([w['ic'] for w in walks])
                ic_std = np.std([w['ic'] for w in walks])
                ic_range = [min([w['ic'] for w in walks]), max([w['ic'] for w in walks])]
                
                report.append(f"{ticker}:")
                report.append(f"  Average IC: {avg_ic:.3f} ± {ic_std:.3f}")
                report.append(f"  IC Range: [{ic_range[0]:.3f}, {ic_range[1]:.3f}]")
                report.append(f"  Number of walks: {len(walks)}")
        
        # Feature Ablation Results
        report.append("\n🔬 FEATURE ABLATION RESULTS")
        report.append("-" * 50)
        
        for ticker, ablation in ablation_results.items():
            report.append(f"{ticker}:")
            for feature_set, metrics in ablation.items():
                report.append(f"  {feature_set}: IC = {metrics['ic']:.3f}, R² = {metrics['r2']:.3f}")
        
        # Backtest Results
        report.append("\n💰 REALISTIC BACKTEST RESULTS")
        report.append("-" * 50)
        
        for ticker, backtest in backtest_results.items():
            report.append(f"{ticker}:")
            for cost, metrics in backtest.items():
                report.append(f"  {cost} cost: Sharpe = {metrics['sharpe']:.3f}, Return = {metrics['total_return']:.1%}")
        
        # Yearly Stability
        report.append("\n📅 YEARLY STABILITY ANALYSIS")
        report.append("-" * 50)
        
        for ticker, yearly in yearly_results.items():
            report.append(f"{ticker}:")
            for year, metrics in yearly.items():
                report.append(f"  {year}: Sharpe = {metrics['sharpe']:.3f}, Return = {metrics['total_return']:.1%}")
        
        # Causality Results
        if self.causality_results:
            report.append("\n🔗 GRANGER CAUSALITY ANALYSIS")
            report.append("-" * 50)
            
            for ticker, causality in self.causality_results.items():
                report.append(f"{ticker}:")
                report.append(f"  Min P-value: {causality['min_p_value']:.4f}")
                report.append(f"  Significant Lags: {causality['significant_lags']}")
                report.append(f"  Causality Confirmed: {causality['causality_confirmed']}")
        
        # Overall Assessment
        report.append("\n🎯 OVERALL ASSESSMENT")
        report.append("-" * 50)
        
        # Calculate overall metrics
        all_ics = []
        all_sharpes = []
        
        for ticker, folds in kfold_results.items():
            if folds:
                all_ics.extend([f['ic'] for f in folds])
        
        for ticker, backtest in backtest_results.items():
            for cost, metrics in backtest.items():
                all_sharpes.append(metrics['sharpe'])
        
        if all_ics:
            avg_ic = np.mean(all_ics)
            ic_std = np.std(all_ics)
            report.append(f"Overall IC: {avg_ic:.3f} ± {ic_std:.3f}")
        
        if all_sharpes:
            avg_sharpe = np.mean(all_sharpes)
            sharpe_std = np.std(all_sharpes)
            report.append(f"Overall Sharpe: {avg_sharpe:.3f} ± {sharpe_std:.3f}")
        
        # Recommendations
        report.append("\n💡 RECOMMENDATIONS")
        report.append("-" * 50)
        
        if all_ics and np.mean(all_ics) < 0.05:
            report.append("• IC < 0.05 suggests weak predictive power")
        
        if all_sharpes and np.mean(all_sharpes) < 0.5:
            report.append("• Sharpe < 0.5 indicates poor risk-adjusted returns")
        
        report.append("• Use realistic transaction costs (20-50bps)")
        report.append("• Implement proper time-series validation")
        report.append("• Monitor causality relationships")
        report.append("• Test on out-of-sample data")
        
        return "\n".join(report)
    
    def run_complete_validation(self):
        """Run complete validation pipeline."""
        logger.info("🚀 Starting complete validation pipeline...")
        
        # Generate data
        data = self.generate_realistic_data()
        
        # Run all validations
        kfold_results = self.run_purged_kfold_validation(data)
        walk_forward_results = self.run_walk_forward_validation(data)
        ablation_results = self.run_feature_ablation(data)
        backtest_results = self.run_realistic_backtest(data)
        yearly_results = self.evaluate_yearly_stability(data)
        causality_results = self.test_granger_causality(data)
        
        # Generate report
        report = self.generate_comprehensive_report(
            kfold_results, walk_forward_results, ablation_results, 
            backtest_results, yearly_results
        )
        
        # Save results
        output_dir = Path("robust_validation_results")
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / "validation_report.txt", "w", encoding='utf-8') as f:
            f.write(report)
        
        # Save detailed results
        results_dict = {
            'kfold_results': kfold_results,
            'walk_forward_results': walk_forward_results,
            'ablation_results': ablation_results,
            'backtest_results': backtest_results,
            'yearly_results': yearly_results,
            'causality_results': causality_results
        }
        
        # Save results as pickle instead of JSON to avoid serialization issues
        import pickle
        with open(output_dir / "detailed_results.pkl", "wb") as f:
            pickle.dump(results_dict, f)
        
        return report, results_dict

def main():
    """Run complete validation pipeline."""
    validator = RobustMemeStockML()
    
    try:
        report, results = validator.run_complete_validation()
        
        print(report)
        
        logger.info("\n📁 Results saved to: robust_validation_results/")
        logger.info("🎉 Complete validation completed!")
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise

if __name__ == "__main__":
    main()
