#!/usr/bin/env python3
"""
Comprehensive validation and robustness testing for meme stock ML upgrade.
Addresses critical concerns about overfitting, causality, and performance metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobustnessValidator:
    """Comprehensive validation framework for meme stock ML results."""
    
    def __init__(self):
        self.results = {}
        
    def generate_realistic_data(self, n_days=1000, n_tickers=3):
        """Generate more realistic market data with proper noise and correlations."""
        logger.info("Generating realistic market data...")
        
        np.random.seed(42)
        dates = pd.date_range('2021-01-01', periods=n_days, freq='D')
        tickers = ['GME', 'AMC', 'BB']
        
        # Generate correlated market factors
        market_factor = np.random.randn(n_days) * 0.02  # Market-wide movements
        sector_factor = np.random.randn(n_days) * 0.015  # Meme stock sector factor
        
        data = {}
        
        for ticker in tickers:
            # Base price
            base_price = {'GME': 100, 'AMC': 50, 'BB': 10}[ticker]
            
            # Generate realistic price series with mean reversion
            returns = []
            price = base_price
            
            for i in range(n_days):
                # Market beta
                beta = {'GME': 1.5, 'AMC': 1.2, 'BB': 1.0}[ticker]
                
                # Daily return components
                market_component = beta * market_factor[i]
                sector_component = sector_factor[i]
                idiosyncratic = np.random.randn() * 0.03
                
                # Mean reversion component
                mean_reversion = -0.001 * (price / base_price - 1)
                
                daily_return = market_component + sector_component + idiosyncratic + mean_reversion
                returns.append(daily_return)
                
                price *= (1 + daily_return)
            
            # Generate Reddit activity with realistic patterns
            reddit_activity = []
            for i, ret in enumerate(returns):
                # Base activity
                base_activity = np.random.poisson(50)
                
                # Activity increases with volatility and extreme returns
                volatility_multiplier = 1 + abs(ret) * 10
                extreme_return_multiplier = 1 + max(0, abs(ret) - 0.05) * 5
                
                activity = base_activity * volatility_multiplier * extreme_return_multiplier
                reddit_activity.append(int(activity))
            
            # Create DataFrame
            prices = [base_price]
            for ret in returns:
                prices.append(prices[-1] * (1 + ret))
            
            df = pd.DataFrame({
                'date': dates,
                'close': prices,
                'volume': np.random.lognormal(12, 1, len(prices)),
                'reddit_mentions': reddit_activity,
                'returns': [0] + returns
            })
            
            # Calculate technical indicators
            df['sma_20'] = df['close'].rolling(20).mean()
            df['rsi'] = self._calculate_rsi(df['close'])
            df['volatility'] = df['returns'].rolling(20).std()
            
            data[ticker] = df
        
        return data
    
    def _calculate_rsi(self, prices, window=14):
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def test_transaction_cost_sensitivity(self, data):
        """Test strategy performance across different transaction cost assumptions."""
        logger.info("Testing transaction cost sensitivity...")
        
        cost_scenarios = [0.0005, 0.001, 0.002, 0.0035, 0.005]  # 5bp to 50bp
        results = {}
        
        for cost in cost_scenarios:
            logger.info(f"Testing transaction cost: {cost:.1%}")
            
            scenario_results = {}
            
            for ticker, df in data.items():
                # Generate signals (simple contrarian based on Reddit activity)
                df['signal'] = np.where(
                    df['reddit_mentions'] > df['reddit_mentions'].rolling(20).mean() * 1.5,
                    -1,  # Short when high activity
                    np.where(
                        df['reddit_mentions'] < df['reddit_mentions'].rolling(20).mean() * 0.5,
                        1,  # Long when low activity
                        0   # Neutral
                    )
                )
                
                # Calculate strategy returns with transaction costs
                df['strategy_return'] = df['signal'].shift(1) * df['returns']
                
                # Apply transaction costs
                position_changes = df['signal'].diff().abs()
                df['net_return'] = df['strategy_return'] - position_changes * cost
                
                # Calculate metrics
                net_returns = df['net_return'].dropna()
                if len(net_returns) > 0:
                    sharpe = net_returns.mean() / net_returns.std() * np.sqrt(252)
                    total_return = (1 + net_returns).prod() - 1
                    hit_rate = (net_returns > 0).mean()
                    
                    scenario_results[ticker] = {
                        'sharpe': sharpe,
                        'total_return': total_return,
                        'hit_rate': hit_rate,
                        'volatility': net_returns.std() * np.sqrt(252)
                    }
            
            # Aggregate results
            if scenario_results:
                avg_sharpe = np.mean([r['sharpe'] for r in scenario_results.values()])
                avg_return = np.mean([r['total_return'] for r in scenario_results.values()])
                avg_hit_rate = np.mean([r['hit_rate'] for r in scenario_results.values()])
                
                results[f"{cost:.1%}"] = {
                    'avg_sharpe': avg_sharpe,
                    'avg_return': avg_return,
                    'avg_hit_rate': avg_hit_rate,
                    'ticker_results': scenario_results
                }
        
        self.results['transaction_costs'] = results
        return results
    
    def test_feature_ablation(self, data):
        """Test performance when Reddit features are removed."""
        logger.info("Testing feature ablation...")
        
        ablation_results = {}
        
        for ticker, df in data.items():
            # Full model features
            df['reddit_signal'] = np.where(
                df['reddit_mentions'] > df['reddit_mentions'].rolling(20).mean() * 1.5,
                -1, 0
            )
            df['technical_signal'] = np.where(
                df['rsi'] < 30, 1, np.where(df['rsi'] > 70, -1, 0)
            )
            
            # Full model
            df['full_signal'] = df['reddit_signal'] + df['technical_signal']
            df['full_return'] = df['full_signal'].shift(1) * df['returns']
            
            # Reddit-only model
            df['reddit_return'] = df['reddit_signal'].shift(1) * df['returns']
            
            # Technical-only model
            df['technical_return'] = df['technical_signal'].shift(1) * df['returns']
            
            # Calculate Sharpe ratios
            full_returns = df['full_return'].dropna()
            reddit_returns = df['reddit_return'].dropna()
            technical_returns = df['technical_return'].dropna()
            
            ablation_results[ticker] = {
                'full_sharpe': full_returns.mean() / full_returns.std() * np.sqrt(252) if len(full_returns) > 0 else 0,
                'reddit_sharpe': reddit_returns.mean() / reddit_returns.std() * np.sqrt(252) if len(reddit_returns) > 0 else 0,
                'technical_sharpe': technical_returns.mean() / technical_returns.std() * np.sqrt(252) if len(technical_returns) > 0 else 0,
                'reddit_contribution': (full_returns.mean() / full_returns.std() - technical_returns.mean() / technical_returns.std()) * np.sqrt(252) if len(full_returns) > 0 and len(technical_returns) > 0 else 0
            }
        
        self.results['feature_ablation'] = ablation_results
        return ablation_results
    
    def test_granger_causality(self, data):
        """Test Granger causality between Reddit mentions and returns."""
        logger.info("Testing Granger causality...")
        
        from statsmodels.tsa.stattools import grangercausalitytests
        
        causality_results = {}
        
        for ticker, df in data.items():
            # Prepare data for Granger test
            df_clean = df.dropna()
            
            if len(df_clean) > 100:  # Need sufficient data
                # Test: Do Reddit mentions Granger-cause returns?
                test_data = df_clean[['returns', 'reddit_mentions']].values
                
                try:
                    # Test with different lags
                    gc_result = grangercausalitytests(test_data, maxlag=5, verbose=False)
                    
                    # Extract p-values
                    p_values = []
                    for lag in range(1, 6):
                        if lag in gc_result:
                            p_value = gc_result[lag][0]['ssr_ftest'][1]  # F-test p-value
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
        
        self.results['granger_causality'] = causality_results
        return causality_results
    
    def test_walk_forward_distribution(self, data):
        """Analyze walk-forward validation fold distribution."""
        logger.info("Testing walk-forward fold distribution...")
        
        fold_results = {}
        
        for ticker, df in data.items():
            # Simulate walk-forward validation
            n_folds = 10
            fold_size = len(df) // n_folds
            
            fold_sharpes = []
            fold_returns = []
            
            for fold in range(n_folds):
                start_idx = fold * fold_size
                end_idx = start_idx + fold_size
                
                fold_data = df.iloc[start_idx:end_idx].copy()
                
                if len(fold_data) > 20:  # Minimum data requirement
                    # Generate signals
                    fold_data['signal'] = np.where(
                        fold_data['reddit_mentions'] > fold_data['reddit_mentions'].rolling(10).mean() * 1.5,
                        -1, 0
                    )
                    
                    # Calculate returns
                    fold_data['strategy_return'] = fold_data['signal'].shift(1) * fold_data['returns']
                    returns = fold_data['strategy_return'].dropna()
                    
                    if len(returns) > 0:
                        sharpe = returns.mean() / returns.std() * np.sqrt(252)
                        total_return = (1 + returns).prod() - 1
                        
                        fold_sharpes.append(sharpe)
                        fold_returns.append(total_return)
            
            fold_results[ticker] = {
                'sharpe_distribution': fold_sharpes,
                'return_distribution': fold_returns,
                'sharpe_mean': np.mean(fold_sharpes),
                'sharpe_std': np.std(fold_sharpes),
                'sharpe_max': np.max(fold_sharpes),
                'sharpe_min': np.min(fold_sharpes),
                'outlier_folds': [i for i, s in enumerate(fold_sharpes) if abs(s - np.mean(fold_sharpes)) > 2 * np.std(fold_sharpes)]
            }
        
        self.results['walk_forward'] = fold_results
        return fold_results
    
    def test_r2_validation(self, data):
        """Investigate R² values for potential feature leakage."""
        logger.info("Testing R² validation...")
        
        r2_results = {}
        
        for ticker, df in data.items():
            # Create features
            df['reddit_ma'] = df['reddit_mentions'].rolling(20).mean()
            df['reddit_signal'] = df['reddit_mentions'] / df['reddit_ma'] - 1
            df['rsi_signal'] = (df['rsi'] - 50) / 50
            df['vol_signal'] = df['volatility'] / df['volatility'].rolling(50).mean() - 1
            
            # Target: next day return
            df['target'] = df['returns'].shift(-1)
            
            # Clean data
            df_clean = df.dropna()
            
            if len(df_clean) > 50:
                # Test different feature combinations
                features = {
                    'reddit_only': ['reddit_signal'],
                    'technical_only': ['rsi_signal', 'vol_signal'],
                    'all_features': ['reddit_signal', 'rsi_signal', 'vol_signal'],
                    'lagged_features': ['reddit_signal', 'rsi_signal', 'vol_signal', 'returns']  # Potential leakage
                }
                
                ticker_results = {}
                
                for feature_set, cols in features.items():
                    X = df_clean[cols]
                    y = df_clean['target']
                    
                    if len(X) > 0 and len(y) > 0:
                        # Simple linear regression
                        from sklearn.linear_model import LinearRegression
                        from sklearn.model_selection import train_test_split
                        
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                        
                        model = LinearRegression()
                        model.fit(X_train, y_train)
                        
                        y_pred = model.predict(X_test)
                        r2 = r2_score(y_test, y_pred)
                        mse = mean_squared_error(y_test, y_pred)
                        
                        ticker_results[feature_set] = {
                            'r2': r2,
                            'mse': mse,
                            'feature_importance': dict(zip(cols, model.coef_))
                        }
                
                r2_results[ticker] = ticker_results
        
        self.results['r2_validation'] = r2_results
        return r2_results
    
    def generate_comprehensive_report(self):
        """Generate comprehensive validation report."""
        logger.info("Generating comprehensive validation report...")
        
        report = []
        report.append("🔍 COMPREHENSIVE ROBUSTNESS VALIDATION REPORT")
        report.append("=" * 60)
        
        # Transaction Cost Analysis
        if 'transaction_costs' in self.results:
            report.append("\n📊 TRANSACTION COST SENSITIVITY ANALYSIS")
            report.append("-" * 50)
            
            for cost, results in self.results['transaction_costs'].items():
                report.append(f"Cost {cost}: Sharpe = {results['avg_sharpe']:.3f}, Return = {results['avg_return']:.1%}")
            
            # Check if Sharpe drops significantly with realistic costs
            low_cost_sharpe = self.results['transaction_costs']['0.1%']['avg_sharpe']
            high_cost_sharpe = self.results['transaction_costs']['0.5%']['avg_sharpe']
            
            if high_cost_sharpe < low_cost_sharpe * 0.5:
                report.append("⚠️  WARNING: Strategy highly sensitive to transaction costs!")
            else:
                report.append("✅ Strategy robust to transaction cost variations")
        
        # Feature Ablation Analysis
        if 'feature_ablation' in self.results:
            report.append("\n🔬 FEATURE ABLATION ANALYSIS")
            report.append("-" * 50)
            
            for ticker, results in self.results['feature_ablation'].items():
                report.append(f"{ticker}:")
                report.append(f"  Full Model Sharpe: {results['full_sharpe']:.3f}")
                report.append(f"  Reddit Only Sharpe: {results['reddit_sharpe']:.3f}")
                report.append(f"  Technical Only Sharpe: {results['technical_sharpe']:.3f}")
                
                if results['technical_sharpe'] < 1.0:
                    report.append("  ✅ Technical-only model Sharpe < 1.0 (as expected)")
                else:
                    report.append("  ⚠️  WARNING: Technical-only model Sharpe > 1.0")
        
        # Granger Causality Analysis
        if 'granger_causality' in self.results:
            report.append("\n🔗 GRANGER CAUSALITY ANALYSIS")
            report.append("-" * 50)
            
            for ticker, results in self.results['granger_causality'].items():
                report.append(f"{ticker}:")
                report.append(f"  Min P-value: {results['min_p_value']:.4f}")
                report.append(f"  Significant Lags: {results['significant_lags']}")
                
                if results['causality_confirmed']:
                    report.append("  ✅ Reddit mentions Granger-cause returns")
                else:
                    report.append("  ❌ No Granger causality detected")
        
        # Walk-Forward Analysis
        if 'walk_forward' in self.results:
            report.append("\n📈 WALK-FORWARD DISTRIBUTION ANALYSIS")
            report.append("-" * 50)
            
            for ticker, results in self.results['walk_forward'].items():
                report.append(f"{ticker}:")
                report.append(f"  Sharpe Mean: {results['sharpe_mean']:.3f}")
                report.append(f"  Sharpe Std: {results['sharpe_std']:.3f}")
                report.append(f"  Sharpe Range: [{results['sharpe_min']:.3f}, {results['sharpe_max']:.3f}]")
                report.append(f"  Outlier Folds: {results['outlier_folds']}")
                
                if results['sharpe_std'] > results['sharpe_mean'] * 0.5:
                    report.append("  ⚠️  WARNING: High Sharpe volatility suggests overfitting")
                else:
                    report.append("  ✅ Stable Sharpe distribution")
        
        # R² Validation
        if 'r2_validation' in self.results:
            report.append("\n📊 R² VALIDATION ANALYSIS")
            report.append("-" * 50)
            
            for ticker, results in self.results['r2_validation'].items():
                report.append(f"{ticker}:")
                
                for feature_set, metrics in results.items():
                    report.append(f"  {feature_set}: R² = {metrics['r2']:.3f}")
                    
                    if feature_set == 'lagged_features' and metrics['r2'] > 0.3:
                        report.append("    ⚠️  WARNING: High R² with lagged features suggests leakage")
        
        # Overall Assessment
        report.append("\n🎯 OVERALL ASSESSMENT")
        report.append("-" * 50)
        
        # Check for red flags
        red_flags = []
        
        if 'transaction_costs' in self.results:
            if self.results['transaction_costs']['0.5%']['avg_sharpe'] < 1.0:
                red_flags.append("Strategy fails with realistic transaction costs")
        
        if 'feature_ablation' in self.results:
            for ticker, results in self.results['feature_ablation'].items():
                if results['technical_sharpe'] > 1.0:
                    red_flags.append(f"Technical-only model too strong for {ticker}")
        
        if 'granger_causality' in self.results:
            causality_confirmed = any(r['causality_confirmed'] for r in self.results['granger_causality'].values())
            if not causality_confirmed:
                red_flags.append("No Granger causality detected")
        
        if red_flags:
            report.append("🚨 RED FLAGS DETECTED:")
            for flag in red_flags:
                report.append(f"  • {flag}")
        else:
            report.append("✅ No major red flags detected")
        
        return "\n".join(report)
    
    def run_comprehensive_validation(self):
        """Run all validation tests."""
        logger.info("Starting comprehensive validation...")
        
        # Generate realistic data
        data = self.generate_realistic_data()
        
        # Run all tests
        self.test_transaction_cost_sensitivity(data)
        self.test_feature_ablation(data)
        self.test_granger_causality(data)
        self.test_walk_forward_distribution(data)
        self.test_r2_validation(data)
        
        # Generate report
        report = self.generate_comprehensive_report()
        
        return report, data

def main():
    """Run comprehensive validation."""
    validator = RobustnessValidator()
    
    logger.info("🚀 Starting Comprehensive Robustness Validation")
    logger.info("=" * 60)
    
    try:
        report, data = validator.run_comprehensive_validation()
        
        print(report)
        
        # Save results
        output_dir = Path("validation_results")
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / "robustness_report.txt", "w") as f:
            f.write(report)
        
        logger.info(f"\n📁 Results saved to: {output_dir}")
        logger.info("🎉 Comprehensive validation completed!")
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise

if __name__ == "__main__":
    main()
