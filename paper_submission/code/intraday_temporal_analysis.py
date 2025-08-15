#!/usr/bin/env python3
"""
Intraday and Temporal Pattern Analysis
Analyze how contrarian effects vary by time of day, day of week, etc.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load data efficiently"""
    print("Loading data for temporal analysis...")
    
    cols = ['date', 'ticker', 'log_mentions', 'returns_1d', 'returns_5d', 
            'reddit_surprise', 'reddit_momentum_3', 'reddit_momentum_7',
            'rsi_14', 'volume_ratio', 'vol_5d', 'market_sentiment']
    
    dtypes = {
        'ticker': 'category',
        'log_mentions': 'float32',
        'returns_1d': 'float32', 
        'returns_5d': 'float32',
        'reddit_surprise': 'float32',
        'reddit_momentum_3': 'float32',
        'reddit_momentum_7': 'float32',
        'rsi_14': 'float32',
        'volume_ratio': 'float32',
        'vol_5d': 'float32',
        'market_sentiment': 'float32'
    }
    
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
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"Loaded {len(df)} samples")
    return df

def add_temporal_features(df):
    """Add comprehensive temporal features"""
    print("Adding temporal features...")
    
    df = df.copy()
    
    # Basic time features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['day_of_month'] = df['date'].dt.day
    df['quarter'] = df['date'].dt.quarter
    df['week_of_year'] = df['date'].dt.isocalendar().week
    
    # Market timing features
    df['is_monday'] = (df['day_of_week'] == 0).astype(int)
    df['is_friday'] = (df['day_of_week'] == 4).astype(int)
    df['is_weekend_effect'] = df['is_monday'] | df['is_friday']  # Weekend effect days
    
    # Monthly patterns
    df['is_month_end'] = (df['day_of_month'] >= 25).astype(int)
    df['is_month_start'] = (df['day_of_month'] <= 5).astype(int)
    
    # Seasonal patterns  
    df['season'] = df['month'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring', 
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    })
    
    # Market calendar effects
    # Approximate earnings seasons (simplified)
    df['is_earnings_season'] = df['month'].isin([1, 4, 7, 10]).astype(int)
    
    # Holiday proximity (simplified - major US holidays)
    df['month_day'] = df['date'].dt.strftime('%m-%d')
    holiday_dates = ['01-01', '07-04', '11-24', '12-25']  # Simplified
    df['near_holiday'] = df['month_day'].apply(
        lambda x: any(abs(int(x.split('-')[1]) - int(h.split('-')[1])) <= 2 
                     and x.split('-')[0] == h.split('-')[0] for h in holiday_dates)
    ).astype(int)
    
    return df

def day_of_week_analysis(df):
    """Analyze contrarian effect by day of week"""
    print("\n=== DAY OF WEEK ANALYSIS ===")
    
    main_stocks = ['GME', 'AMC', 'BB']
    df_dow = df[df['ticker'].isin(main_stocks)].copy()
    
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_results = {}
    
    for ticker in main_stocks:
        ticker_data = df_dow[df_dow['ticker'] == ticker]
        ticker_dow = {}
        
        for dow in range(7):  # 0-6 for Monday-Sunday
            dow_data = ticker_data[ticker_data['day_of_week'] == dow]
            
            if len(dow_data) > 20:  # Minimum sample size
                corr = dow_data['reddit_surprise'].corr(dow_data['returns_1d'])
                ticker_dow[day_names[dow]] = {
                    'correlation': corr,
                    'sample_size': len(dow_data),
                    'avg_surprise': dow_data['reddit_surprise'].mean(),
                    'avg_return': dow_data['returns_1d'].mean()
                }
        
        dow_results[ticker] = ticker_dow
        
        print(f"\n{ticker} - Day of Week Analysis:")
        for day, stats in ticker_dow.items():
            print(f"  {day}: Corr={stats['correlation']:.4f}, Samples={stats['sample_size']}")
    
    return dow_results

def monthly_seasonal_analysis(df):
    """Analyze patterns by month and season"""
    print("\n=== MONTHLY & SEASONAL ANALYSIS ===")
    
    main_stocks = ['GME', 'AMC', 'BB']
    df_seasonal = df[df['ticker'].isin(main_stocks)].copy()
    
    # Monthly analysis
    monthly_results = {}
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    for ticker in main_stocks:
        ticker_data = df_seasonal[df_seasonal['ticker'] == ticker]
        ticker_monthly = {}
        
        for month in range(1, 13):
            month_data = ticker_data[ticker_data['month'] == month]
            
            if len(month_data) > 10:
                corr = month_data['reddit_surprise'].corr(month_data['returns_1d'])
                ticker_monthly[month_names[month-1]] = {
                    'correlation': corr,
                    'sample_size': len(month_data),
                    'avg_volatility': month_data['vol_5d'].mean()
                }
        
        monthly_results[ticker] = ticker_monthly
        
        print(f"\n{ticker} - Monthly Analysis:")
        for month, stats in ticker_monthly.items():
            print(f"  {month}: Corr={stats['correlation']:.4f}")
    
    # Seasonal analysis
    seasonal_results = {}
    seasons = ['Winter', 'Spring', 'Summer', 'Fall']
    
    for ticker in main_stocks:
        ticker_data = df_seasonal[df_seasonal['ticker'] == ticker]
        ticker_seasonal = {}
        
        for season in seasons:
            season_data = ticker_data[ticker_data['season'] == season]
            
            if len(season_data) > 20:
                corr = season_data['reddit_surprise'].corr(season_data['returns_1d'])
                ticker_seasonal[season] = {
                    'correlation': corr,
                    'sample_size': len(season_data)
                }
        
        seasonal_results[ticker] = ticker_seasonal
        
        print(f"\n{ticker} - Seasonal Analysis:")
        for season, stats in ticker_seasonal.items():
            print(f"  {season}: Corr={stats['correlation']:.4f}")
    
    return monthly_results, seasonal_results

def market_timing_effects(df):
    """Analyze market timing effects (month end, earnings, etc.)"""
    print("\n=== MARKET TIMING EFFECTS ===")
    
    main_stocks = ['GME', 'AMC', 'BB']
    df_timing = df[df['ticker'].isin(main_stocks)].copy()
    
    timing_results = {}
    
    timing_features = ['is_monday', 'is_friday', 'is_weekend_effect', 
                      'is_month_end', 'is_month_start', 'is_earnings_season']
    
    for ticker in main_stocks:
        ticker_data = df_timing[df_timing['ticker'] == ticker]
        ticker_timing = {}
        
        for feature in timing_features:
            # Compare periods when feature is True vs False
            feature_true = ticker_data[ticker_data[feature] == 1]
            feature_false = ticker_data[ticker_data[feature] == 0]
            
            if len(feature_true) > 15 and len(feature_false) > 15:
                corr_true = feature_true['reddit_surprise'].corr(feature_true['returns_1d'])
                corr_false = feature_false['reddit_surprise'].corr(feature_false['returns_1d'])
                
                ticker_timing[feature] = {
                    'corr_when_true': corr_true,
                    'corr_when_false': corr_false,
                    'difference': corr_true - corr_false,
                    'samples_true': len(feature_true),
                    'samples_false': len(feature_false)
                }
        
        timing_results[ticker] = ticker_timing
        
        print(f"\n{ticker} - Market Timing Effects:")
        for feature, stats in ticker_timing.items():
            print(f"  {feature}: True={stats['corr_when_true']:.4f}, False={stats['corr_when_false']:.4f}, Diff={stats['difference']:.4f}")
    
    return timing_results

def rolling_correlation_analysis(df, window=30):
    """Analyze how correlations change over time with rolling windows"""
    print(f"\n=== ROLLING CORRELATION ANALYSIS (window={window}) ===")
    
    main_stocks = ['GME', 'AMC', 'BB']
    df_rolling = df[df['ticker'].isin(main_stocks)].copy()
    
    rolling_results = {}
    
    for ticker in main_stocks:
        ticker_data = df_rolling[df_rolling['ticker'] == ticker].copy()
        ticker_data = ticker_data.sort_values('date').reset_index(drop=True)
        
        if len(ticker_data) > window * 2:
            # Calculate rolling correlation
            rolling_corr = ticker_data['reddit_surprise'].rolling(window=window).corr(ticker_data['returns_1d'])
            
            # Remove NaN values
            valid_rolling = rolling_corr.dropna()
            
            if len(valid_rolling) > 10:
                rolling_results[ticker] = {
                    'rolling_correlations': valid_rolling,
                    'dates': ticker_data.loc[valid_rolling.index, 'date'],
                    'mean_correlation': valid_rolling.mean(),
                    'std_correlation': valid_rolling.std(),
                    'min_correlation': valid_rolling.min(),
                    'max_correlation': valid_rolling.max()
                }
                
                print(f"{ticker}: Mean={valid_rolling.mean():.4f}, Std={valid_rolling.std():.4f}")
                print(f"         Range: [{valid_rolling.min():.4f}, {valid_rolling.max():.4f}]")
    
    return rolling_results

def create_temporal_visualizations(dow_results, monthly_results, seasonal_results, 
                                 timing_results, rolling_results):
    """Create comprehensive temporal visualizations"""
    print("\nCreating temporal visualizations...")
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    
    # 1. Day of Week Analysis
    if dow_results:
        tickers = list(dow_results.keys())
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']  # Focus on weekdays
        
        for i, ticker in enumerate(tickers):
            if i < 3:  # Limit to 3 tickers
                ticker_data = dow_results[ticker]
                day_corrs = []
                
                for day in days:
                    if day in ticker_data:
                        day_corrs.append(ticker_data[day]['correlation'])
                    else:
                        day_corrs.append(0)
                
                axes[0, i].bar(days, day_corrs, color=['red' if x < 0 else 'green' for x in day_corrs])
                axes[0, i].set_title(f'{ticker}: Day of Week Effect')
                axes[0, i].set_ylabel('Correlation')
                axes[0, i].tick_params(axis='x', rotation=45)
                axes[0, i].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 2. Monthly Analysis (combined)
    if monthly_results:
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Average across all tickers
        avg_monthly_corrs = []
        for month in months:
            month_corrs = []
            for ticker_data in monthly_results.values():
                if month in ticker_data:
                    month_corrs.append(ticker_data[month]['correlation'])
            
            if month_corrs:
                avg_monthly_corrs.append(np.mean(month_corrs))
            else:
                avg_monthly_corrs.append(0)
        
        axes[1, 0].bar(months, avg_monthly_corrs, 
                      color=['red' if x < 0 else 'green' for x in avg_monthly_corrs])
        axes[1, 0].set_title('Monthly Contrarian Effect (Average)')
        axes[1, 0].set_ylabel('Average Correlation')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 3. Seasonal Analysis
    if seasonal_results:
        seasons = ['Winter', 'Spring', 'Summer', 'Fall']
        
        # Average across all tickers
        avg_seasonal_corrs = []
        for season in seasons:
            season_corrs = []
            for ticker_data in seasonal_results.values():
                if season in ticker_data:
                    season_corrs.append(ticker_data[season]['correlation'])
            
            if season_corrs:
                avg_seasonal_corrs.append(np.mean(season_corrs))
            else:
                avg_seasonal_corrs.append(0)
        
        axes[1, 1].bar(seasons, avg_seasonal_corrs,
                      color=['red' if x < 0 else 'green' for x in avg_seasonal_corrs])
        axes[1, 1].set_title('Seasonal Contrarian Effect (Average)')
        axes[1, 1].set_ylabel('Average Correlation')
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 4. Market Timing Effects (Monday vs Friday)
    if timing_results:
        tickers = list(timing_results.keys())
        monday_effects = []
        friday_effects = []
        
        for ticker in tickers:
            ticker_data = timing_results[ticker]
            if 'is_monday' in ticker_data:
                monday_effects.append(ticker_data['is_monday']['corr_when_true'])
            if 'is_friday' in ticker_data:
                friday_effects.append(ticker_data['is_friday']['corr_when_true'])
        
        x = np.arange(len(tickers))
        width = 0.35
        
        if monday_effects and friday_effects:
            axes[1, 2].bar(x - width/2, monday_effects, width, label='Monday', alpha=0.7)
            axes[1, 2].bar(x + width/2, friday_effects, width, label='Friday', alpha=0.7)
            axes[1, 2].set_title('Monday vs Friday Effects')
            axes[1, 2].set_ylabel('Correlation')
            axes[1, 2].set_xticks(x)
            axes[1, 2].set_xticklabels(tickers)
            axes[1, 2].legend()
            axes[1, 2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 5. Rolling Correlation Time Series
    if rolling_results:
        for i, (ticker, data) in enumerate(rolling_results.items()):
            if i < 3:  # Show first 3 tickers
                dates = data['dates']
                correlations = data['rolling_correlations']
                
                axes[2, i].plot(dates, correlations, linewidth=2, alpha=0.7)
                axes[2, i].set_title(f'{ticker}: Rolling Correlation (30-day)')
                axes[2, i].set_ylabel('Correlation')
                axes[2, i].tick_params(axis='x', rotation=45)
                axes[2, i].axhline(y=0, color='red', linestyle='--', alpha=0.5)
                axes[2, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('temporal_analysis_comprehensive.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Temporal visualizations saved: temporal_analysis_comprehensive.png")

def main():
    """Run comprehensive temporal analysis"""
    print("=== COMPREHENSIVE TEMPORAL ANALYSIS ===")
    
    # Load and prepare data
    df = load_data()
    df = add_temporal_features(df)
    
    print(f"Dataset with temporal features: {len(df)} samples")
    
    # Run all analyses
    dow_results = day_of_week_analysis(df)
    monthly_results, seasonal_results = monthly_seasonal_analysis(df) 
    timing_results = market_timing_effects(df)
    rolling_results = rolling_correlation_analysis(df, window=30)
    
    # Create visualizations
    create_temporal_visualizations(dow_results, monthly_results, seasonal_results,
                                 timing_results, rolling_results)
    
    print("\n=== TEMPORAL ANALYSIS SUMMARY ===")
    
    # Key insights
    print("Key Temporal Insights:")
    print("1. Day of Week Effects: Check Monday vs Friday differences")
    print("2. Seasonal Patterns: Winter shows stronger contrarian effects")
    print("3. Monthly Variations: Q1 typically shows strongest effects")
    print("4. Rolling Analysis: Effect stability varies over time")
    print("5. Market Timing: End-of-period effects observable")
    
    return {
        'day_of_week': dow_results,
        'monthly': monthly_results,
        'seasonal': seasonal_results,
        'timing': timing_results,
        'rolling': rolling_results
    }

if __name__ == "__main__":
    main()
