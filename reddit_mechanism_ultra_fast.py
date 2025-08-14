#!/usr/bin/env python3
"""
Ultra-Fast Reddit Mechanism Analysis
No display, just save plots and print results
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # No display backend
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def ultra_fast_analysis():
    """Ultra-fast analysis without display"""
    print("Ultra-fast Reddit Mechanism Analysis")
    print("=" * 50)
    
    # Load data
    print("Loading data...")
    df = pd.read_csv('data/colab_datasets/tabular_train_20250814_031335.csv')
    df['date'] = pd.to_datetime(df['date'])
    print(f"Data loaded: {len(df)} samples")
    
    # Quick calculations
    print("Running analysis...")
    
    # 1. Basic correlations
    corr_price = df['log_mentions'].corr(df['price_ratio_sma20'])
    corr_rsi = df['log_mentions'].corr(df['rsi_14'])
    corr_volume = df['log_mentions'].corr(df['volume_ratio'])
    
    # 2. Contrarian effect
    df['mentions_change'] = df.groupby('ticker')['log_mentions'].diff()
    df['next_return'] = df.groupby('ticker')['returns_1d'].shift(-1)
    corr_reversal = df['mentions_change'].corr(df['next_return'])
    
    # 3. Volatility effect
    df['next_vol'] = df.groupby('ticker')['vol_5d'].shift(-1)
    corr_volatility = df['log_mentions'].corr(df['next_vol'])
    
    # 4. Trading strategy
    df['signal'] = 0
    df.loc[df['log_mentions'] > df['log_mentions'].quantile(0.8), 'signal'] = -1  # Sell high
    df.loc[df['log_mentions'] < df['log_mentions'].quantile(0.2), 'signal'] = 1   # Buy low
    
    df['strategy_return'] = df['signal'] * df['returns_1d']
    
    buyhold = df['returns_1d'].sum()
    strategy = df['strategy_return'].sum()
    improvement = strategy - buyhold
    
    # Results
    print("\n📊 MECHANISM ANALYSIS RESULTS")
    print("=" * 50)
    print(f"Overvaluation (Mentions vs Price/SMA): {corr_price:.4f}")
    print(f"Overvaluation (Mentions vs RSI): {corr_rsi:.4f}")
    print(f"Overvaluation (Mentions vs Volume): {corr_volume:.4f}")
    print(f"Contrarian Trading Effect: {corr_reversal:.4f}")
    print(f"Volatility Response: {corr_volatility:.4f}")
    
    print(f"\n💰 TRADING STRATEGY RESULTS")
    print("=" * 50)
    print(f"Buy & Hold Return: {buyhold:.4f}")
    print(f"Reddit Strategy Return: {strategy:.4f}")
    print(f"Strategy Improvement: {improvement:.4f}")
    print(f"Improvement %: {(improvement/buyhold*100):.2f}%" if buyhold != 0 else "N/A")
    
    # Quick plot (save only, no display)
    print("\n📈 Creating plots...")
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    axes[0,0].scatter(df['log_mentions'], df['price_ratio_sma20'], alpha=0.3, s=1)
    axes[0,0].set_title(f'Mentions vs Price/SMA (r={corr_price:.3f})')
    axes[0,0].set_xlabel('Log Mentions')
    axes[0,0].set_ylabel('Price/SMA Ratio')
    
    axes[0,1].scatter(df['log_mentions'], df['rsi_14'], alpha=0.3, s=1)
    axes[0,1].set_title(f'Mentions vs RSI (r={corr_rsi:.3f})')
    axes[0,1].set_xlabel('Log Mentions')
    axes[0,1].set_ylabel('RSI')
    
    axes[1,0].scatter(df['mentions_change'], df['next_return'], alpha=0.3, s=1)
    axes[1,0].set_title(f'Mention Change vs Next Return (r={corr_reversal:.3f})')
    axes[1,0].set_xlabel('Mention Change')
    axes[1,0].set_ylabel('Next Day Return')
    
    axes[1,1].scatter(df['log_mentions'], df['next_vol'], alpha=0.3, s=1)
    axes[1,1].set_title(f'Mentions vs Next Volatility (r={corr_volatility:.3f})')
    axes[1,1].set_xlabel('Log Mentions')
    axes[1,1].set_ylabel('Next Day Volatility')
    
    plt.tight_layout()
    plt.savefig('ultra_fast_mechanism_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()  # Close to free memory
    
    print("✅ Analysis complete!")
    print("📁 Plot saved: ultra_fast_mechanism_analysis.png")
    
    # Paper readiness assessment
    print(f"\n📚 PAPER READINESS ASSESSMENT")
    print("=" * 50)
    
    if corr_reversal < -0.05:
        print("🎯 CONTRARIAN EFFECT: STRONG (Good for paper)")
    elif corr_reversal < -0.02:
        print("🎯 CONTRARIAN EFFECT: MODERATE (Acceptable for paper)")
    else:
        print("🎯 CONTRARIAN EFFECT: WEAK (Needs more analysis)")
    
    if improvement > 0:
        print("💰 STRATEGY: PROFITABLE (Strong practical value)")
    else:
        print("💰 STRATEGY: UNPROFITABLE (Focus on theoretical insights)")
    
    if abs(corr_price) > 0.1 or abs(corr_rsi) > 0.1:
        print("📊 OVERVALUATION: DETECTED (Mechanism identified)")
    else:
        print("📊 OVERVALUATION: WEAK (Mechanism unclear)")
    
    return {
        'corr_price': corr_price,
        'corr_rsi': corr_rsi,
        'corr_reversal': corr_reversal,
        'corr_volatility': corr_volatility,
        'strategy_improvement': improvement
    }

if __name__ == "__main__":
    ultra_fast_analysis()
