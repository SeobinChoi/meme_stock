#!/usr/bin/env python3
"""
Price and Reddit mentions visualization by ticker
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import seaborn as sns

def load_and_plot_data():
    """Load data and create visualizations"""
    
    print("📊 Loading data...")
    
    # Load all datasets
    datasets = []
    for split in ['train', 'val', 'test']:
        try:
            df = pd.read_csv(f'data/colab_datasets/tabular_{split}_20250814_031335.csv')
            df['split'] = split
            datasets.append(df)
        except FileNotFoundError:
            print(f"⚠️ {split} data file not found")
            continue
    
    if not datasets:
        print("❌ Cannot load data")
        return
    
    # Combine data
    df = pd.concat(datasets, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"✅ Loaded {len(df)} samples")
    print(f"   Period: {df['date'].min()} ~ {df['date'].max()}")
    
    # Check tickers
    tickers = df['ticker'].unique()
    print(f"   Tickers: {list(tickers)}")
    
    # Find Reddit mention columns
    reddit_cols = [col for col in df.columns if 'mentions' in col.lower() or 'reddit' in col.lower()]
    print(f"   Reddit columns: {reddit_cols[:5]}...")
    
    # Select main Reddit column
    main_reddit_col = None
    for col in ['log_mentions', 'mentions', 'reddit_mentions', 'reddit_ema_3']:
        if col in df.columns:
            main_reddit_col = col
            break
    
    if main_reddit_col is None:
        print("❌ Cannot find Reddit mention column")
        return
    
    print(f"   Main Reddit column: {main_reddit_col}")
    
    # Create visualization
    n_tickers = len(tickers)
    fig, axes = plt.subplots(n_tickers, 2, figsize=(15, 6 * n_tickers))
    
    if n_tickers == 1:
        axes = axes.reshape(1, -1)
    
    plt.style.use('default')
    
    for i, ticker in enumerate(tickers):
        ticker_data = df[df['ticker'] == ticker].copy()
        ticker_data = ticker_data.sort_values('date')
        
        print(f"\n📈 Processing {ticker} data...")
        print(f"   Samples: {len(ticker_data)}")
        
        # Prepare price data (calculate price index from returns)
        if 'returns_1d' in ticker_data.columns:
            returns = ticker_data['returns_1d'].fillna(0)
            price_index = (1 + returns).cumprod() * 100  # Start from 100
        else:
            price_index = pd.Series(range(100, 100 + len(ticker_data)), index=ticker_data.index)
        
        # Prepare Reddit mentions
        reddit_mentions = ticker_data[main_reddit_col].fillna(0)
        
        # Convert from log scale if needed
        if 'log' in main_reddit_col.lower():
            reddit_mentions = np.exp(reddit_mentions) - 1  # Inverse log(x+1)
        
        dates = ticker_data['date']
        
        # 1. Price chart
        ax1 = axes[i, 0]
        ax1.plot(dates, price_index, linewidth=2, color='blue', label='Price Index')
        ax1.set_title(f'{ticker} - Price Trend', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price Index (Base: 100)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Date formatting
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # 2. Reddit mentions chart
        ax2 = axes[i, 1]
        ax2.plot(dates, reddit_mentions, linewidth=2, color='red', label='Reddit Mentions')
        ax2.set_title(f'{ticker} - Reddit Mention Trend', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Mention Count')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Date formatting
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # Print statistics
        print(f"   Price index: {price_index.min():.2f} ~ {price_index.max():.2f}")
        print(f"   Reddit mentions: {reddit_mentions.min():.0f} ~ {reddit_mentions.max():.0f}")
    
    plt.tight_layout()
    
    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'price_reddit_analysis_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n💾 Chart saved: {filename}")
    
    plt.show()
    
    # Correlation analysis
    print("\n📊 Price-Reddit Correlation Analysis:")
    print("=" * 50)
    
    for ticker in tickers:
        ticker_data = df[df['ticker'] == ticker].copy()
        
        if len(ticker_data) < 10:
            continue
        
        # Correlation between returns and Reddit mentions
        if 'returns_1d' in ticker_data.columns:
            returns = ticker_data['returns_1d'].dropna()
            reddit_vals = ticker_data[main_reddit_col].dropna()
            
            if len(returns) > 10 and len(reddit_vals) > 10:
                # Use data from same dates only
                common_dates = set(ticker_data.dropna(subset=['returns_1d', main_reddit_col])['date'])
                if len(common_dates) > 10:
                    corr_data = ticker_data[ticker_data['date'].isin(common_dates)]
                    correlation = corr_data['returns_1d'].corr(corr_data[main_reddit_col])
                    
                    print(f"{ticker:8s}: Correlation = {correlation:6.3f}")
                    
                    if abs(correlation) > 0.1:
                        print(f"         {'Strong' if abs(correlation) > 0.3 else 'Moderate'} correlation!")

def create_combined_chart():
    """Create combined price and Reddit chart for each ticker"""
    
    print("\n📊 Creating combined chart...")
    
    # Load data
    datasets = []
    for split in ['train', 'val', 'test']:
        try:
            df = pd.read_csv(f'data/colab_datasets/tabular_{split}_20250814_031335.csv')
            datasets.append(df)
        except:
            continue
    
    if not datasets:
        return
    
    df = pd.concat(datasets, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    
    # Find Reddit column
    reddit_col = None
    for col in ['log_mentions', 'mentions', 'reddit_ema_3']:
        if col in df.columns:
            reddit_col = col
            break
    
    if reddit_col is None:
        return
    
    tickers = df['ticker'].unique()
    
    # Combined chart
    fig, axes = plt.subplots(len(tickers), 1, figsize=(15, 4 * len(tickers)))
    if len(tickers) == 1:
        axes = [axes]
    
    for i, ticker in enumerate(tickers):
        ticker_data = df[df['ticker'] == ticker].sort_values('date')
        
        ax = axes[i]
        
        # Price (left y-axis)
        if 'returns_1d' in ticker_data.columns:
            price_index = (1 + ticker_data['returns_1d'].fillna(0)).cumprod() * 100
        else:
            price_index = pd.Series(range(100, 100 + len(ticker_data)))
        
        ax.plot(ticker_data['date'], price_index, color='blue', linewidth=2, label='Price Index')
        ax.set_ylabel('Price Index', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        
        # Reddit mentions (right y-axis)
        ax2 = ax.twinx()
        reddit_data = ticker_data[reddit_col].fillna(0)
        if 'log' in reddit_col.lower():
            reddit_data = np.exp(reddit_data) - 1
        
        ax2.plot(ticker_data['date'], reddit_data, color='red', linewidth=2, alpha=0.7, label='Reddit Mentions')
        ax2.set_ylabel('Reddit Mentions', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        ax.set_title(f'{ticker} - Price vs Reddit Mentions', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Date format
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'combined_price_reddit_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"💾 Combined chart saved: {filename}")
    
    plt.show()

def main():
    """Main execution"""
    
    print("📊 Price & Reddit Mentions Visualization by Ticker")
    print("=" * 60)
    
    # Individual charts
    load_and_plot_data()
    
    # Combined chart
    create_combined_chart()
    
    print("\n✅ Visualization completed!")

if __name__ == "__main__":
    main()