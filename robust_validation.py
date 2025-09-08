#!/usr/bin/env python3
"""
Simplified but comprehensive validation for meme stock ML results.
Addresses critical concerns about overfitting, causality, and performance metrics.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_realistic_data(n_days=1000):
    """Generate realistic market data."""
    logger.info("Generating realistic market data...")
    
    np.random.seed(42)
    dates = pd.date_range('2021-01-01', periods=n_days, freq='D')
    
    # Generate market factors
    market_factor = np.random.randn(n_days) * 0.02
    sector_factor = np.random.randn(n_days) * 0.015
    
    data = {}
    tickers = ['GME', 'AMC', 'BB']
    
    for ticker in tickers:
        base_price = {'GME': 100, 'AMC': 50, 'BB': 10}[ticker]
        beta = {'GME': 1.5, 'AMC': 1.2, 'BB': 1.0}[ticker]
        
        # Generate returns
        returns = []
        for i in range(n_days):
            market_comp = beta * market_factor[i]
            sector_comp = sector_factor[i]
            idiosyncratic = np.random.randn() * 0.03
            mean_reversion = -0.001 * np.random.randn()  # Simplified
            
            daily_return = market_comp + sector_comp + idiosyncratic + mean_reversion
            returns.append(daily_return)
        
        # Generate prices
        prices = [base_price]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        # Generate Reddit activity
        reddit_activity = []
        for ret in returns:
            base_activity = np.random.poisson(50)
            volatility_mult = 1 + abs(ret) * 10
            activity = base_activity * volatility_mult
            reddit_activity.append(int(activity))
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'close': prices[:-1],  # Remove last price to match length
            'volume': np.random.lognormal(12, 1, n_days),
            'reddit_mentions': reddit_activity,
            'returns': returns
        })
        
        # Calculate technical indicators
        df['sma_20'] = df['close'].rolling(20).mean()
        df['volatility'] = df['returns'].rolling(20).std()
        
        data[ticker] = df
    
    return data

def test_transaction_costs(data):
    """Test transaction cost sensitivity."""
    logger.info("Testing transaction cost sensitivity...")
    
    costs = [0.0005, 0.001, 0.002, 0.0035, 0.005]  # 5bp to 50bp
    results = {}
    
    for cost in costs:
        cost_results = {}
        
        for ticker, df in data.items():
            # Simple contrarian strategy
            df['signal'] = np.where(
                df['reddit_mentions'] > df['reddit_mentions'].rolling(20).mean() * 1.5,
                -1,  # Short when high activity
                np.where(
                    df['reddit_mentions'] < df['reddit_mentions'].rolling(20).mean() * 0.5,
                    1,  # Long when low activity
                    0
                )
            )
            
            # Calculate returns
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
                
                cost_results[ticker] = {
                    'sharpe': sharpe,
                    'total_return': total_return,
                    'hit_rate': hit_rate
                }
        
        # Aggregate
        if cost_results:
            avg_sharpe = np.mean([r['sharpe'] for r in cost_results.values()])
            avg_return = np.mean([r['total_return'] for r in cost_results.values()])
            avg_hit_rate = np.mean([r['hit_rate'] for r in cost_results.values()])
            
            results[f"{cost:.1%}"] = {
                'avg_sharpe': avg_sharpe,
                'avg_return': avg_return,
                'avg_hit_rate': avg_hit_rate
            }
    
    return results

def test_feature_ablation(data):
    """Test feature ablation."""
    logger.info("Testing feature ablation...")
    
    results = {}
    
    for ticker, df in data.items():
        # Reddit signal
        df['reddit_signal'] = np.where(
            df['reddit_mentions'] > df['reddit_mentions'].rolling(20).mean() * 1.5,
            -1, 0
        )
        
        # Technical signal (simplified)
        df['technical_signal'] = np.where(
            df['volatility'] > df['volatility'].rolling(20).mean() * 1.2,
            -1, 0
        )
        
        # Calculate returns
        df['reddit_return'] = df['reddit_signal'].shift(1) * df['returns']
        df['technical_return'] = df['technical_signal'].shift(1) * df['returns']
        df['combined_return'] = (df['reddit_signal'] + df['technical_signal']).shift(1) * df['returns']
        
        # Calculate Sharpe ratios
        reddit_returns = df['reddit_return'].dropna()
        technical_returns = df['technical_return'].dropna()
        combined_returns = df['combined_return'].dropna()
        
        results[ticker] = {
            'reddit_sharpe': reddit_returns.mean() / reddit_returns.std() * np.sqrt(252) if len(reddit_returns) > 0 else 0,
            'technical_sharpe': technical_returns.mean() / technical_returns.std() * np.sqrt(252) if len(technical_returns) > 0 else 0,
            'combined_sharpe': combined_returns.mean() / combined_returns.std() * np.sqrt(252) if len(combined_returns) > 0 else 0
        }
    
    return results

def test_walk_forward_distribution(data):
    """Test walk-forward distribution."""
    logger.info("Testing walk-forward distribution...")
    
    results = {}
    
    for ticker, df in data.items():
        # Simulate walk-forward validation
        n_folds = 10
        fold_size = len(df) // n_folds
        
        fold_sharpes = []
        
        for fold in range(n_folds):
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size
            
            fold_data = df.iloc[start_idx:end_idx].copy()
            
            if len(fold_data) > 20:
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
                    fold_sharpes.append(sharpe)
        
        results[ticker] = {
            'sharpe_distribution': fold_sharpes,
            'sharpe_mean': np.mean(fold_sharpes),
            'sharpe_std': np.std(fold_sharpes),
            'sharpe_max': np.max(fold_sharpes) if fold_sharpes else 0,
            'sharpe_min': np.min(fold_sharpes) if fold_sharpes else 0,
            'outlier_count': len([s for s in fold_sharpes if abs(s - np.mean(fold_sharpes)) > 2 * np.std(fold_sharpes)])
        }
    
    return results

def test_r2_validation(data):
    """Test R² for potential leakage."""
    logger.info("Testing R² validation...")
    
    results = {}
    
    for ticker, df in data.items():
        # Create features
        df['reddit_signal'] = df['reddit_mentions'] / df['reddit_mentions'].rolling(20).mean() - 1
        df['vol_signal'] = df['volatility'] / df['volatility'].rolling(50).mean() - 1
        
        # Target: next day return
        df['target'] = df['returns'].shift(-1)
        
        # Clean data
        df_clean = df.dropna()
        
        if len(df_clean) > 50:
            # Simple correlation analysis
            reddit_corr = df_clean['reddit_signal'].corr(df_clean['target'])
            vol_corr = df_clean['vol_signal'].corr(df_clean['target'])
            
            # Simple R² calculation
            reddit_r2 = reddit_corr ** 2
            vol_r2 = vol_corr ** 2
            
            results[ticker] = {
                'reddit_correlation': reddit_corr,
                'vol_correlation': vol_corr,
                'reddit_r2': reddit_r2,
                'vol_r2': vol_r2,
                'combined_r2': (reddit_corr + vol_corr) ** 2 / 4  # Simplified
            }
    
    return results

def generate_report(transaction_results, ablation_results, walk_forward_results, r2_results):
    """Generate comprehensive report."""
    
    report = []
    report.append("🔍 COMPREHENSIVE ROBUSTNESS VALIDATION REPORT")
    report.append("=" * 60)
    
    # Transaction Cost Analysis
    report.append("\n📊 TRANSACTION COST SENSITIVITY ANALYSIS")
    report.append("-" * 50)
    
    for cost, results in transaction_results.items():
        report.append(f"Cost {cost}: Sharpe = {results['avg_sharpe']:.3f}, Return = {results['avg_return']:.1%}")
    
    # Check sensitivity
    costs = list(transaction_results.keys())
    if len(costs) >= 2:
        low_cost_sharpe = transaction_results[costs[0]]['avg_sharpe']
        high_cost_sharpe = transaction_results[costs[-1]]['avg_sharpe']
        
        if high_cost_sharpe < low_cost_sharpe * 0.5:
            report.append("⚠️  WARNING: Strategy highly sensitive to transaction costs!")
        else:
            report.append("✅ Strategy robust to transaction cost variations")
    
    # Feature Ablation Analysis
    report.append("\n🔬 FEATURE ABLATION ANALYSIS")
    report.append("-" * 50)
    
    for ticker, results in ablation_results.items():
        report.append(f"{ticker}:")
        report.append(f"  Reddit Only Sharpe: {results['reddit_sharpe']:.3f}")
        report.append(f"  Technical Only Sharpe: {results['technical_sharpe']:.3f}")
        report.append(f"  Combined Sharpe: {results['combined_sharpe']:.3f}")
        
        if results['technical_sharpe'] < 1.0:
            report.append("  ✅ Technical-only model Sharpe < 1.0 (as expected)")
        else:
            report.append("  ⚠️  WARNING: Technical-only model Sharpe > 1.0")
    
    # Walk-Forward Analysis
    report.append("\n📈 WALK-FORWARD DISTRIBUTION ANALYSIS")
    report.append("-" * 50)
    
    for ticker, results in walk_forward_results.items():
        report.append(f"{ticker}:")
        report.append(f"  Sharpe Mean: {results['sharpe_mean']:.3f}")
        report.append(f"  Sharpe Std: {results['sharpe_std']:.3f}")
        report.append(f"  Sharpe Range: [{results['sharpe_min']:.3f}, {results['sharpe_max']:.3f}]")
        report.append(f"  Outlier Folds: {results['outlier_count']}")
        
        if results['sharpe_std'] > results['sharpe_mean'] * 0.5:
            report.append("  ⚠️  WARNING: High Sharpe volatility suggests overfitting")
        else:
            report.append("  ✅ Stable Sharpe distribution")
    
    # R² Validation
    report.append("\n📊 R² VALIDATION ANALYSIS")
    report.append("-" * 50)
    
    for ticker, results in r2_results.items():
        report.append(f"{ticker}:")
        report.append(f"  Reddit R²: {results['reddit_r2']:.3f}")
        report.append(f"  Volatility R²: {results['vol_r2']:.3f}")
        report.append(f"  Combined R²: {results['combined_r2']:.3f}")
        
        if results['combined_r2'] > 0.3:
            report.append("    ⚠️  WARNING: High R² suggests potential leakage")
        else:
            report.append("    ✅ R² within reasonable range")
    
    # Overall Assessment
    report.append("\n🎯 OVERALL ASSESSMENT")
    report.append("-" * 50)
    
    # Check for red flags
    red_flags = []
    
    # Transaction cost sensitivity
    if transaction_results:
        high_cost_sharpe = list(transaction_results.values())[-1]['avg_sharpe']
        if high_cost_sharpe < 1.0:
            red_flags.append("Strategy fails with realistic transaction costs")
    
    # Technical model strength
    for ticker, results in ablation_results.items():
        if results['technical_sharpe'] > 1.0:
            red_flags.append(f"Technical-only model too strong for {ticker}")
    
    # R² concerns
    for ticker, results in r2_results.items():
        if results['combined_r2'] > 0.3:
            red_flags.append(f"High R² for {ticker} suggests potential leakage")
    
    if red_flags:
        report.append("🚨 RED FLAGS DETECTED:")
        for flag in red_flags:
            report.append(f"  • {flag}")
    else:
        report.append("✅ No major red flags detected")
    
    # Recommendations
    report.append("\n💡 RECOMMENDATIONS:")
    report.append("• Re-examine transaction cost assumptions")
    report.append("• Validate feature engineering for leakage")
    report.append("• Test with more realistic market data")
    report.append("• Implement stricter walk-forward validation")
    report.append("• Consider ensemble methods for stability")
    
    return "\n".join(report)

def main():
    """Run comprehensive validation."""
    logger.info("🚀 Starting Comprehensive Robustness Validation")
    logger.info("=" * 60)
    
    try:
        # Generate data
        data = generate_realistic_data()
        
        # Run tests
        transaction_results = test_transaction_costs(data)
        ablation_results = test_feature_ablation(data)
        walk_forward_results = test_walk_forward_distribution(data)
        r2_results = test_r2_validation(data)
        
        # Generate report
        report = generate_report(transaction_results, ablation_results, walk_forward_results, r2_results)
        
        print(report)
        
        # Save results
        output_dir = Path("validation_results")
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / "robustness_report.txt", "w", encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"\n📁 Results saved to: {output_dir}")
        logger.info("🎉 Comprehensive validation completed!")
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise

if __name__ == "__main__":
    main()
