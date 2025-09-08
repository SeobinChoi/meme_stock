#!/usr/bin/env python3
"""
Complete & Robust Reddit Mechanism Analysis
Uses ALL data (train + validation + test) for comprehensive results
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # No display backend
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def load_all_data():
    """Load ALL available data for comprehensive analysis"""
    print("Loading ALL data for comprehensive analysis...")
    
    # Load all datasets
    train_df = pd.read_csv('data/colab_datasets/tabular_train_20250814_031335.csv')
    val_df = pd.read_csv('data/colab_datasets/tabular_val_20250814_031335.csv')
    test_df = pd.read_csv('data/colab_datasets/tabular_test_20250814_031335.csv')
    
    # Combine all data
    df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"✅ Complete dataset loaded: {len(df)} total samples")
    print(f"   - Training: {len(train_df)} samples")
    print(f"   - Validation: {len(val_df)} samples") 
    print(f"   - Test: {len(test_df)} samples")
    print(f"   - Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   - Tickers: {df['ticker'].nunique()} unique")
    
    return df

def comprehensive_mechanism_analysis(df):
    """Comprehensive mechanism analysis using all data"""
    print("\n🔍 Running comprehensive mechanism analysis...")
    
    # 1. Basic correlations (all data)
    corr_price = df['log_mentions'].corr(df['price_ratio_sma20'])
    corr_rsi = df['log_mentions'].corr(df['rsi_14'])
    corr_volume = df['log_mentions'].corr(df['volume_ratio'])
    corr_returns = df['log_mentions'].corr(df['returns_1d'])
    
    # 2. Contrarian effect (all data)
    df['mentions_change'] = df.groupby('ticker')['log_mentions'].diff()
    df['next_return'] = df.groupby('ticker')['returns_1d'].shift(-1)
    df['next_return_5d'] = df.groupby('ticker')['returns_5d'].shift(-1)
    corr_reversal_1d = df['mentions_change'].corr(df['next_return'])
    corr_reversal_5d = df['mentions_change'].corr(df['next_return_5d'])
    
    # 3. Volatility effect (all data)
    df['next_vol'] = df.groupby('ticker')['vol_5d'].shift(-1)
    corr_volatility = df['log_mentions'].corr(df['next_vol'])
    
    # 4. Time-based analysis
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['year'] = df['date'].dt.year
    
    # Monthly correlations
    monthly_corr = df.groupby('month').apply(
        lambda x: x['log_mentions'].corr(x['returns_1d'])
    ).round(4)
    
    # Quarterly correlations
    quarterly_corr = df.groupby('quarter').apply(
        lambda x: x['log_mentions'].corr(x['returns_1d'])
    ).round(4)
    
    # 5. Ticker-specific analysis
    ticker_correlations = df.groupby('ticker').apply(
        lambda x: x['log_mentions'].corr(x['returns_1d'])
    ).round(4)
    
    # 6. Robustness checks
    # Remove outliers (top/bottom 1%)
    df_clean = df.copy()
    for col in ['log_mentions', 'returns_1d', 'price_ratio_sma20']:
        q1 = df_clean[col].quantile(0.01)
        q99 = df_clean[col].quantile(0.99)
        df_clean = df_clean[(df_clean[col] >= q1) & (df_clean[col] <= q99)]
    
    # Clean correlations
    clean_corr_price = df_clean['log_mentions'].corr(df_clean['price_ratio_sma20'])
    clean_corr_reversal = df_clean['mentions_change'].corr(df_clean['next_return'])
    
    # Results
    print("\n📊 COMPREHENSIVE MECHANISM RESULTS")
    print("=" * 60)
    print(f"Basic Correlations (All Data):")
    print(f"  - Mentions vs Price/SMA: {corr_price:.4f}")
    print(f"  - Mentions vs RSI: {corr_rsi:.4f}")
    print(f"  - Mentions vs Volume: {corr_volume:.4f}")
    print(f"  - Mentions vs Same-day Returns: {corr_returns:.4f}")
    
    print(f"\nContrarian Effects (All Data):")
    print(f"  - 1-day Reversal: {corr_reversal_1d:.4f}")
    print(f"  - 5-day Reversal: {corr_reversal_5d:.4f}")
    print(f"  - Volatility Response: {corr_volatility:.4f}")
    
    print(f"\nRobustness Checks (Outlier-removed):")
    print(f"  - Clean Price Correlation: {clean_corr_price:.4f}")
    print(f"  - Clean Reversal: {clean_corr_reversal:.4f}")
    
    return {
        'corr_price': corr_price,
        'corr_rsi': corr_rsi,
        'corr_volume': corr_volume,
        'corr_returns': corr_returns,
        'corr_reversal_1d': corr_reversal_1d,
        'corr_reversal_5d': corr_reversal_5d,
        'corr_volatility': corr_volatility,
        'clean_corr_price': clean_corr_price,
        'clean_corr_reversal': clean_corr_reversal,
        'monthly_corr': monthly_corr,
        'quarterly_corr': quarterly_corr,
        'ticker_correlations': ticker_correlations
    }

def comprehensive_trading_strategy(df):
    """Comprehensive trading strategy using all data"""
    print("\n💰 Running comprehensive trading strategy...")
    
    # Multiple strategy variations
    strategies = {}
    
    # Strategy 1: Basic contrarian
    df['strategy1'] = 0
    df.loc[df['log_mentions'] > df['log_mentions'].quantile(0.8), 'strategy1'] = -1  # Sell high
    df.loc[df['log_mentions'] < df['log_mentions'].quantile(0.2), 'strategy1'] = 1   # Buy low
    
    # Strategy 2: Momentum-based
    df['strategy2'] = 0
    df.loc[df['mentions_change'] > df['mentions_change'].quantile(0.8), 'strategy2'] = -1  # Sell momentum
    df.loc[df['mentions_change'] < df['mentions_change'].quantile(0.2), 'strategy2'] = 1   # Buy reversal
    
    # Strategy 3: RSI + Mentions combined
    df['strategy3'] = 0
    high_mentions_rsi = (df['log_mentions'] > df['log_mentions'].quantile(0.7)) & (df['rsi_14'] > 70)
    low_mentions_rsi = (df['log_mentions'] < df['log_mentions'].quantile(0.3)) & (df['rsi_14'] < 30)
    df.loc[high_mentions_rsi, 'strategy3'] = -1  # Sell overvalued
    df.loc[low_mentions_rsi, 'strategy3'] = 1    # Buy undervalued
    
    # Calculate returns for all strategies
    for i in range(1, 4):
        df[f'strategy{i}_return'] = df[f'strategy{i}'] * df['returns_1d']
    
    # Performance metrics
    buyhold_return = df['returns_1d'].sum()
    buyhold_sharpe = df['returns_1d'].mean() / df['returns_1d'].std() * np.sqrt(252) if df['returns_1d'].std() != 0 else 0
    
    strategy_results = {}
    for i in range(1, 4):
        strategy_return = df[f'strategy{i}_return'].sum()
        strategy_sharpe = df[f'strategy{i}_return'].mean() / df[f'strategy{i}_return'].std() * np.sqrt(252) if df[f'strategy{i}_return'].std() != 0 else 0
        improvement = strategy_return - buyhold_return
        
        strategy_results[f'strategy{i}'] = {
            'return': strategy_return,
            'sharpe': strategy_sharpe,
            'improvement': improvement,
            'improvement_pct': (improvement/buyhold_return*100) if buyhold_return != 0 else 0
        }
    
    # Results
    print("\n💰 COMPREHENSIVE TRADING STRATEGY RESULTS")
    print("=" * 60)
    print(f"Buy & Hold (All Data):")
    print(f"  - Total Return: {buyhold_return:.4f}")
    print(f"  - Sharpe Ratio: {buyhold_sharpe:.4f}")
    
    print(f"\nStrategy Performance:")
    for i, results in strategy_results.items():
        print(f"  {i.upper()}: Return={results['return']:.4f}, Sharpe={results['sharpe']:.4f}")
        print(f"    Improvement: {results['improvement']:.4f} ({results['improvement_pct']:.2f}%)")
    
    return {
        'buyhold': {'return': buyhold_return, 'sharpe': buyhold_sharpe},
        'strategies': strategy_results
    }

def create_comprehensive_plots(df, mechanism_results, strategy_results):
    """Create comprehensive visualization plots"""
    print("\n📈 Creating comprehensive plots...")
    
    # Plot 1: All correlations heatmap
    plt.figure(figsize=(15, 12))
    
    # Correlation matrix
    corr_cols = ['log_mentions', 'price_ratio_sma20', 'rsi_14', 'volume_ratio', 'returns_1d', 'vol_5d']
    corr_matrix = df[corr_cols].corr()
    
    plt.subplot(3, 3, 1)
    plt.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(corr_cols)), corr_cols, rotation=45)
    plt.yticks(range(len(corr_cols)), corr_cols)
    plt.title('Correlation Matrix (All Data)')
    
    # Plot 2: Monthly correlation pattern
    plt.subplot(3, 3, 2)
    mechanism_results['monthly_corr'].plot(kind='bar')
    plt.title('Monthly Correlation Pattern')
    plt.xlabel('Month')
    plt.ylabel('Mentions vs Returns Correlation')
    plt.xticks(rotation=45)
    
    # Plot 3: Quarterly correlation pattern
    plt.subplot(3, 3, 3)
    mechanism_results['quarterly_corr'].plot(kind='bar')
    plt.title('Quarterly Correlation Pattern')
    plt.xlabel('Quarter')
    plt.ylabel('Mentions vs Returns Correlation')
    
    # Plot 4: Ticker-specific correlations
    plt.subplot(3, 3, 4)
    ticker_corr = mechanism_results['ticker_correlations']
    plt.bar(range(len(ticker_corr)), ticker_corr.values)
    plt.title('Ticker-Specific Correlations')
    plt.xlabel('Ticker Index')
    plt.ylabel('Correlation')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Plot 5: Strategy performance comparison
    plt.subplot(3, 3, 5)
    strategy_names = list(strategy_results['strategies'].keys())
    strategy_returns = [strategy_results['strategies'][s]['return'] for s in strategy_names]
    plt.bar(strategy_names, strategy_returns)
    plt.title('Strategy Returns Comparison')
    plt.ylabel('Total Return')
    plt.axhline(y=strategy_results['buyhold']['return'], color='red', linestyle='--', label='Buy & Hold')
    plt.legend()
    
    # Plot 6: Strategy Sharpe ratios
    plt.subplot(3, 3, 6)
    strategy_sharpes = [strategy_results['strategies'][s]['sharpe'] for s in strategy_names]
    plt.bar(strategy_names, strategy_sharpes)
    plt.title('Strategy Sharpe Ratios')
    plt.ylabel('Sharpe Ratio')
    plt.axhline(y=strategy_results['buyhold']['sharpe'], color='red', linestyle='--', label='Buy & Hold')
    plt.legend()
    
    # Plot 7: Mentions vs Returns scatter
    plt.subplot(3, 3, 7)
    plt.scatter(df['log_mentions'], df['returns_1d'], alpha=0.3, s=1)
    plt.title(f'Mentions vs Returns (r={mechanism_results["corr_returns"]:.3f})')
    plt.xlabel('Log Mentions')
    plt.ylabel('Returns')
    
    # Plot 8: Mentions change vs Next day return
    plt.subplot(3, 3, 8)
    plt.scatter(df['mentions_change'], df['next_return'], alpha=0.3, s=1)
    plt.title(f'Mention Change vs Next Return (r={mechanism_results["corr_reversal_1d"]:.3f})')
    plt.xlabel('Mention Change')
    plt.ylabel('Next Day Return')
    
    # Plot 9: Strategy improvement summary
    plt.subplot(3, 3, 9)
    plt.axis('off')
    
    summary_text = f"""COMPREHENSIVE ANALYSIS SUMMARY
    
Total Samples: {len(df):,}
Date Range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}
Tickers: {df['ticker'].nunique()}

Key Findings:
• Contrarian Effect: {mechanism_results['corr_reversal_1d']:.4f}
• Overvaluation: {mechanism_results['corr_rsi']:.4f}
• Volatility Response: {mechanism_results['corr_volatility']:.4f}

Best Strategy: {max(strategy_results['strategies'].items(), key=lambda x: x[1]['return'])[0]}
Strategy Improvement: {max(strategy_results['strategies'].items(), key=lambda x: x[1]['improvement'])[1]['improvement']:.4f}"""
    
    plt.text(0.1, 0.5, summary_text, fontsize=10, transform=plt.gca().transAxes,
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue'))
    
    plt.tight_layout()
    plt.savefig('comprehensive_mechanism_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Comprehensive plots saved: comprehensive_mechanism_analysis.png")

def paper_readiness_assessment(mechanism_results, strategy_results):
    """Assess paper readiness with comprehensive data"""
    print("\n📚 COMPREHENSIVE PAPER READINESS ASSESSMENT")
    print("=" * 60)
    
    # Contrarian effect assessment
    reversal_1d = mechanism_results['corr_reversal_1d']
    if reversal_1d < -0.05:
        print("🎯 CONTRARIAN EFFECT: VERY STRONG (Excellent for paper)")
        contrarian_score = 5
    elif reversal_1d < -0.03:
        print("🎯 CONTRARIAN EFFECT: STRONG (Very good for paper)")
        contrarian_score = 4
    elif reversal_1d < -0.01:
        print("🎯 CONTRARIAN EFFECT: MODERATE (Good for paper)")
        contrarian_score = 3
    else:
        print("🎯 CONTRARIAN EFFECT: WEAK (Needs more analysis)")
        contrarian_score = 2
    
    # Overvaluation mechanism assessment
    rsi_corr = abs(mechanism_results['corr_rsi'])
    if rsi_corr > 0.15:
        print("📊 OVERVALUATION MECHANISM: STRONG (Clear mechanism)")
        mechanism_score = 5
    elif rsi_corr > 0.10:
        print("📊 OVERVALUATION MECHANISM: MODERATE (Good mechanism)")
        mechanism_score = 4
    else:
        print("📊 OVERVALUATION MECHANISM: WEAK (Unclear mechanism)")
        mechanism_score = 3
    
    # Volatility response assessment
    vol_corr = mechanism_results['corr_volatility']
    if vol_corr > 0.20:
        print("🚨 VOLATILITY RESPONSE: STRONG (Clear regulatory effect)")
        volatility_score = 5
    elif vol_corr > 0.10:
        print("🚨 VOLATILITY RESPONSE: MODERATE (Some regulatory effect)")
        volatility_score = 4
    else:
        print("🚨 VOLATILITY RESPONSE: WEAK (Little regulatory effect)")
        volatility_score = 3
    
    # Data robustness assessment
    clean_vs_raw = abs(mechanism_results['clean_corr_reversal'] - mechanism_results['corr_reversal_1d'])
    if clean_vs_raw < 0.01:
        print("🔒 DATA ROBUSTNESS: EXCELLENT (Results stable)")
        robustness_score = 5
    elif clean_vs_raw < 0.02:
        print("🔒 DATA ROBUSTNESS: GOOD (Results mostly stable)")
        robustness_score = 4
    else:
        print("🔒 DATA ROBUSTNESS: POOR (Results sensitive to outliers)")
        robustness_score = 3
    
    # Overall paper readiness
    total_score = (contrarian_score + mechanism_score + volatility_score + robustness_score) / 4
    readiness_pct = (total_score / 5) * 100
    
    print(f"\n📊 OVERALL PAPER READINESS: {readiness_pct:.1f}%")
    print("=" * 60)
    
    if readiness_pct >= 80:
        print("🎉 EXCELLENT: Paper is ready for submission!")
        print("   - Strong empirical findings")
        print("   - Clear mechanisms identified")
        print("   - Robust to various checks")
    elif readiness_pct >= 60:
        print("✅ GOOD: Paper is mostly ready")
        print("   - Solid empirical findings")
        print("   - Some mechanisms clear")
        print("   - Minor improvements needed")
    else:
        print("⚠️ NEEDS WORK: Paper requires more analysis")
        print("   - Weak empirical findings")
        print("   - Mechanisms unclear")
        print("   - Significant improvements needed")
    
    return readiness_pct

def main():
    """Main execution function"""
    print("🚀 COMPLETE & ROBUST REDDIT MECHANISM ANALYSIS")
    print("=" * 80)
    
    # 1. Load ALL data
    df = load_all_data()
    
    # 2. Comprehensive mechanism analysis
    mechanism_results = comprehensive_mechanism_analysis(df)
    
    # 3. Comprehensive trading strategy
    strategy_results = comprehensive_trading_strategy(df)
    
    # 4. Create comprehensive plots
    create_comprehensive_plots(df, mechanism_results, strategy_results)
    
    # 5. Paper readiness assessment
    readiness_score = paper_readiness_assessment(mechanism_results, strategy_results)
    
    # 6. Final summary
    print(f"\n🎯 FINAL SUMMARY")
    print("=" * 80)
    print(f"Analysis completed using {len(df):,} total samples")
    print(f"Paper readiness: {readiness_score:.1f}%")
    print(f"Key finding: Contrarian effect = {mechanism_results['corr_reversal_1d']:.4f}")
    print(f"Best strategy improvement: {max(strategy_results['strategies'].items(), key=lambda x: x[1]['improvement'])[1]['improvement']:.4f}")
    
    print(f"\n✅ COMPREHENSIVE ANALYSIS COMPLETE!")
    print("📁 Files saved:")
    print("   - comprehensive_mechanism_analysis.png")
    print("   - All results printed above")

if __name__ == "__main__":
    main()
