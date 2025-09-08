#!/usr/bin/env python3
"""
Fast Reddit-Price Correlation Analysis for M1 Mac with 8GB RAM
Optimized for speed and memory efficiency
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Font settings
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_data_fast():
    """Load data with memory optimization"""
    print("Loading data efficiently...")
    
    # Minimal columns needed
    cols = ['date', 'ticker', 'log_mentions', 'returns_1d', 'returns_5d', 
            'reddit_surprise', 'reddit_momentum_3', 'reddit_momentum_7']
    
    # Optimized dtypes
    dtypes = {
        'ticker': 'category',
        'log_mentions': 'float32',
        'returns_1d': 'float32', 
        'returns_5d': 'float32',
        'reddit_surprise': 'float32',
        'reddit_momentum_3': 'float32',
        'reddit_momentum_7': 'float32'
    }
    
    # Load all data at once (faster than separate loads)
    files = [
        'data/colab_datasets/tabular_train_20250814_031335.csv',
        'data/colab_datasets/tabular_val_20250814_031335.csv', 
        'data/colab_datasets/tabular_test_20250814_031335.csv'
    ]
    
    dfs = []
    for file in files:
        chunk = pd.read_csv(file, usecols=cols, dtype=dtypes)
        dfs.append(chunk)
    
    df = pd.concat(dfs, ignore_index=True)
    del dfs  # Free memory
    
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"Loaded {len(df)} samples, {df.memory_usage(deep=True).sum()/1024**2:.1f} MB")
    return df

def analyze_key_correlations(df):
    """Analyze key correlations focusing on main meme stocks"""
    print("Analyzing key correlations...")
    
    # Focus on main meme stocks
    main_stocks = ['GME', 'AMC', 'BB']
    df_main = df[df['ticker'].isin(main_stocks)].copy()
    
    results = {}
    
    # Calculate correlations for each ticker
    for ticker in main_stocks:
        ticker_data = df_main[df_main['ticker'] == ticker]
        
        if len(ticker_data) > 50:
            # Key correlations
            corr_surprise = ticker_data['reddit_surprise'].corr(ticker_data['returns_1d'])
            corr_momentum3 = ticker_data['reddit_momentum_3'].corr(ticker_data['returns_1d'])
            corr_momentum7 = ticker_data['reddit_momentum_7'].corr(ticker_data['returns_1d'])
            
            results[ticker] = {
                'reddit_surprise': corr_surprise,
                'reddit_momentum_3': corr_momentum3, 
                'reddit_momentum_7': corr_momentum7,
                'sample_size': len(ticker_data)
            }
            
            print(f"{ticker}:")
            print(f"  reddit_surprise: {corr_surprise:.3f}")
            print(f"  reddit_momentum_3: {corr_momentum3:.3f}")
            print(f"  reddit_momentum_7: {corr_momentum7:.3f}")
            print(f"  samples: {len(ticker_data)}")
    
    return results

def create_simple_visualization(results):
    """Create simple but clear visualization"""
    print("Creating visualization...")
    
    tickers = list(results.keys())
    surprise_corrs = [results[t]['reddit_surprise'] for t in tickers]
    momentum3_corrs = [results[t]['reddit_momentum_3'] for t in tickers]
    momentum7_corrs = [results[t]['reddit_momentum_7'] for t in tickers]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Reddit Surprise correlations
    bars1 = axes[0].bar(tickers, surprise_corrs, color=['red' if x < 0 else 'green' for x in surprise_corrs])
    axes[0].set_title('Reddit Surprise vs Returns')
    axes[0].set_ylabel('Correlation')
    axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Add values on bars
    for bar, val in zip(bars1, surprise_corrs):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom')
    
    # Reddit Momentum 3 correlations
    bars2 = axes[1].bar(tickers, momentum3_corrs, color=['red' if x < 0 else 'green' for x in momentum3_corrs])
    axes[1].set_title('Reddit Momentum 3d vs Returns')
    axes[1].set_ylabel('Correlation')
    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    for bar, val in zip(bars2, momentum3_corrs):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom')
    
    # Reddit Momentum 7 correlations  
    bars3 = axes[2].bar(tickers, momentum7_corrs, color=['red' if x < 0 else 'green' for x in momentum7_corrs])
    axes[2].set_title('Reddit Momentum 7d vs Returns')
    axes[2].set_ylabel('Correlation')
    axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    for bar, val in zip(bars3, momentum7_corrs):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('fast_correlation_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Visualization saved: fast_correlation_analysis.png")

def main():
    """Main execution - fast and efficient"""
    print("=== FAST REDDIT-PRICE CORRELATION ANALYSIS ===")
    
    # 1. Load data efficiently
    df = load_data_fast()
    
    # 2. Analyze key correlations
    results = analyze_key_correlations(df)
    
    # 3. Create visualization
    create_simple_visualization(results)
    
    # 4. Summary
    print("\n=== SUMMARY ===")
    print("Key findings (negative = contrarian effect):")
    for ticker, data in results.items():
        print(f"{ticker}: surprise={data['reddit_surprise']:.3f}, "
              f"momentum3={data['reddit_momentum_3']:.3f}, "
              f"momentum7={data['reddit_momentum_7']:.3f}")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
