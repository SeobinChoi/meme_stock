#!/usr/bin/env python3
"""
Fast Reddit Mention-Price Correlation Mechanism Analysis
Streamlined for quick execution and English output
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def load_data_fast():
    """Quick data loading"""
    print("Loading data...")
    
    # Load training data only for speed
    df = pd.read_csv('data/colab_datasets/tabular_train_20250814_031335.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"Data loaded: {len(df)} samples")
    return df

def quick_mechanism_analysis(df):
    """Fast mechanism analysis - all in one function"""
    print("Running quick mechanism analysis...")
    
    # 1. Overvaluation hypothesis
    df['price_sma_ratio'] = df['price_ratio_sma20']
    df['rsi_extreme'] = np.where(df['rsi_14'] > 70, 1, 0)
    
    # 2. Contrarian trading hypothesis
    df['mentions_change'] = df.groupby('ticker')['log_mentions'].diff()
    df['next_day_return'] = df.groupby('ticker')['returns_1d'].shift(-1)
    
    # 3. Regulatory response hypothesis
    df['next_day_volatility'] = df.groupby('ticker')['vol_5d'].shift(-1)
    
    # Quick correlations
    corr_price = df['log_mentions'].corr(df['price_sma_ratio'])
    corr_rsi = df['log_mentions'].corr(df['rsi_14'])
    corr_reversal = df['mentions_change'].corr(df['next_day_return'])
    corr_volatility = df['log_mentions'].corr(df['next_day_volatility'])
    
    # Simple visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot 1: Mentions vs Price/SMA
    axes[0,0].scatter(df['log_mentions'], df['price_sma_ratio'], alpha=0.3, s=1)
    axes[0,0].set_title(f'Mentions vs Price/SMA (r={corr_price:.3f})')
    axes[0,0].set_xlabel('Log Mentions')
    axes[0,0].set_ylabel('Price/SMA Ratio')
    
    # Plot 2: Mentions vs RSI
    axes[0,1].scatter(df['log_mentions'], df['rsi_14'], alpha=0.3, s=1)
    axes[0,1].set_title(f'Mentions vs RSI (r={corr_rsi:.3f})')
    axes[0,1].set_xlabel('Log Mentions')
    axes[0,1].set_ylabel('RSI')
    
    # Plot 3: Mention change vs Next day return
    axes[1,0].scatter(df['mentions_change'], df['next_day_return'], alpha=0.3, s=1)
    axes[1,0].set_title(f'Mention Change vs Next Day Return (r={corr_reversal:.3f})')
    axes[1,0].set_xlabel('Mention Change')
    axes[1,0].set_ylabel('Next Day Return')
    
    # Plot 4: Mentions vs Next day volatility
    axes[1,1].scatter(df['log_mentions'], df['next_day_volatility'], alpha=0.3, s=1)
    axes[1,1].set_title(f'Mentions vs Next Day Volatility (r={corr_volatility:.3f})')
    axes[1,1].set_xlabel('Log Mentions')
    axes[1,1].set_ylabel('Next Day Volatility')
    
    plt.tight_layout()
    plt.savefig('quick_mechanism_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return {
        'corr_price_sma': corr_price,
        'corr_rsi': corr_rsi,
        'corr_reversal': corr_reversal,
        'corr_volatility': corr_volatility
    }

def quick_trading_strategy(df):
    """Fast trading strategy simulation"""
    print("Running quick trading strategy...")
    
    # Simple strategy: sell on high mentions, buy on low mentions
    df['strategy_signal'] = 0
    df.loc[df['log_mentions'] > df['log_mentions'].quantile(0.8), 'strategy_signal'] = -1  # Sell
    df.loc[df['log_mentions'] < df['log_mentions'].quantile(0.2), 'strategy_signal'] = 1   # Buy
    
    df['strategy_return'] = df['strategy_signal'] * df['returns_1d']
    
    # Quick performance metrics
    buyhold_return = df['returns_1d'].sum()
    strategy_return = df['strategy_return'].sum()
    improvement = strategy_return - buyhold_return
    
    print(f"Buy & Hold: {buyhold_return:.4f}")
    print(f"Strategy: {strategy_return:.4f}")
    print(f"Improvement: {improvement:.4f}")
    
    return {
        'buyhold': buyhold_return,
        'strategy': strategy_return,
        'improvement': improvement
    }

def main():
    """Main execution"""
    print("Fast Reddit Mechanism Analysis")
    print("=" * 50)
    
    # Load data
    df = load_data_fast()
    
    # Quick analysis
    mechanism_results = quick_mechanism_analysis(df)
    
    # Quick strategy
    strategy_results = quick_trading_strategy(df)
    
    # Summary
    print("\nQuick Results Summary:")
    print("=" * 50)
    print(f"Overvaluation (Mentions vs Price/SMA): {mechanism_results['corr_price_sma']:.4f}")
    print(f"Overvaluation (Mentions vs RSI): {mechanism_results['corr_rsi']:.4f}")
    print(f"Contrarian Trading: {mechanism_results['corr_reversal']:.4f}")
    print(f"Regulatory Response: {mechanism_results['corr_volatility']:.4f}")
    print(f"Strategy Improvement: {strategy_results['improvement']:.4f}")
    
    print("\nAnalysis complete! Check 'quick_mechanism_analysis.png'")

if __name__ == "__main__":
    main()
