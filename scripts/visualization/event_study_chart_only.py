#!/usr/bin/env python3
"""
이벤트 스터디 차트만 따로 생성
3개 종목(GME, AMC, BB)을 한 그래프에 꺾은선으로 표시
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load the processed features dataset"""
    print("Loading data...")
    
    # Try different possible paths with ticker and reddit_surprise columns
    possible_paths = [
        'data/colab_datasets/tabular_train_20250814_031335.csv',
        'data/colab_datasets/tabular_val_20250814_031335.csv', 
        'data/colab_datasets/tabular_test_20250814_031335.csv'
    ]
    
    datasets = []
    for path in possible_paths:
        if Path(path).exists():
            print(f"Loading: {path}")
            df = pd.read_csv(path)
            datasets.append(df)
    
    if not datasets:
        raise FileNotFoundError("No suitable dataset found!")
    
    # Combine all datasets
    df = pd.concat(datasets, ignore_index=True)
    
    print(f"Loaded {len(df)} rows from {len(datasets)} files")
    print(f"Columns available: ticker, reddit_surprise, returns_1d, date")
    
    # Convert date column
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    # Check data
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Tickers: {df['ticker'].unique()}")
    
    return df

def event_study_analysis(df):
    """Event study: analyze extreme Reddit spikes"""
    print("\n=== EVENT STUDY ANALYSIS ===")
    print("Analyzing extreme Reddit attention spikes...")
    
    # Focus on main meme stocks
    main_stocks = ['GME', 'AMC', 'BB']
    df_main = df[df['ticker'].isin(main_stocks)].copy()
    
    # Define extreme events (top 5% Reddit surprise)
    threshold = df_main['reddit_surprise'].quantile(0.95)
    print(f"Extreme event threshold: {threshold:.3f}")
    
    results = {}
    
    for ticker in main_stocks:
        ticker_data = df_main[df_main['ticker'] == ticker].copy()
        ticker_data = ticker_data.sort_values('date').reset_index(drop=True)
        
        # Find extreme events
        extreme_events = ticker_data[ticker_data['reddit_surprise'] > threshold].copy()
        print(f"\n{ticker}: Found {len(extreme_events)} extreme events")
        
        if len(extreme_events) < 3:
            print(f"  Skipping {ticker} - too few events")
            continue
            
        # Analyze returns around events (-5 to +5 days)
        event_analysis = []
        
        for idx, event in extreme_events.iterrows():
            event_date = event['date']
            
            # Get surrounding data
            window_start = max(0, idx - 5)
            window_end = min(len(ticker_data), idx + 6)
            window_data = ticker_data.iloc[window_start:window_end].copy()
            
            if len(window_data) >= 8:  # Need sufficient data (at least 8 days)
                # Calculate relative days from event
                window_data['days_from_event'] = range(-5, len(window_data)-5)
                
                # Store the returns pattern
                for _, row in window_data.iterrows():
                    event_analysis.append({
                        'event_date': event_date,
                        'days_from_event': row['days_from_event'],
                        'returns_1d': row['returns_1d'],
                        'reddit_surprise': row['reddit_surprise'],
                        'volume_ratio': row['volume_ratio'] if 'volume_ratio' in row else 0,
                        'vol_5d': row['vol_5d'] if 'vol_5d' in row else 0
                    })
        
        if event_analysis:
            event_df = pd.DataFrame(event_analysis)
            
            # Calculate average returns by day relative to event
            avg_returns = event_df.groupby('days_from_event')['returns_1d'].agg(['mean', 'std', 'count'])
            
            results[ticker] = {
                'event_data': event_df,
                'avg_returns': avg_returns,
                'num_events': len(extreme_events)
            }
            
            print(f"{ticker} event study results:")
            print(f"  Day -1: {avg_returns.loc[-1, 'mean']:.4f}")
            print(f"  Day 0 (event): {avg_returns.loc[0, 'mean']:.4f}")  
            print(f"  Day +1: {avg_returns.loc[1, 'mean']:.4f}")
            print(f"  Day +2: {avg_returns.loc[2, 'mean']:.4f}")
    
    return results

def create_event_study_chart(event_results):
    """Create single event study bar chart for all stocks"""
    print("\nCreating event study bar chart...")
    
    # Set up the plot
    plt.figure(figsize=(14, 8))
    plt.style.use('default')
    
    # Define colors for each stock
    colors = {'GME': 'blue', 'AMC': 'red', 'BB': 'green'}
    
    # Get all days from -5 to +5
    days = list(range(-5, 6))
    
    # Set up bar positions
    bar_width = 0.25
    x_positions = np.arange(len(days))
    
    # Plot bars for each stock
    for i, (ticker, data) in enumerate(event_results.items()):
        if ticker in colors:
            avg_returns = data['avg_returns']
            returns_values = []
            
            # Get returns for each day (-5 to +5)
            for day in days:
                if day in avg_returns.index:
                    returns_values.append(avg_returns.loc[day, 'mean'])
                else:
                    returns_values.append(0)
            
            # Plot bars with offset
            x_offset = x_positions + i * bar_width
            bars = plt.bar(x_offset, returns_values, 
                          width=bar_width,
                          color=colors[ticker], 
                          alpha=0.8,
                          label=f'{ticker} (N={data["num_events"]} events)')
            
            # Add value labels on bars (only for significant values)
            for j, (x, val) in enumerate(zip(x_offset, returns_values)):
                if abs(val) > 0.005:  # Only show labels for values > 0.5%
                    plt.text(x, val + (0.002 if val > 0 else -0.005), 
                            f'{val:.1%}', 
                            ha='center', va='bottom' if val > 0 else 'top',
                            fontsize=9, fontweight='bold')
    
    # Customize the plot
    plt.title('Event Study: Meme Stock Returns Around Extreme Reddit Spikes (±5 days)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Days from Event', fontsize=14)
    plt.ylabel('Average Returns', fontsize=14)
    
    # Set x-axis labels
    plt.xticks(x_positions + bar_width, days)
    
    # Add reference lines
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    plt.axvline(x=x_positions[5] + bar_width, color='red', linestyle='--', 
                alpha=0.7, linewidth=2, label='Event Day')
    
    # Add grid
    plt.grid(True, alpha=0.3, linestyle=':', axis='y')
    
    # Add legend in upper right
    plt.legend(loc='upper right', fontsize=12, framealpha=0.9)
    
    # Format y-axis as percentage
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the chart
    output_file = 'event_study_bar_chart.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"Chart saved as: {output_file}")
    
    return output_file

def main():
    """Main execution function"""
    print("=== EVENT STUDY CHART GENERATOR ===")
    
    try:
        # Load data
        df = load_data()
        
        # Run event study analysis
        event_results = event_study_analysis(df)
        
        if not event_results:
            print("No event study results found!")
            return
        
        # Create the chart
        chart_file = create_event_study_chart(event_results)
        
        print(f"\n=== COMPLETE ===")
        print(f"Event study chart saved: {chart_file}")
        
        # Print summary
        print(f"\nSummary:")
        for ticker, data in event_results.items():
            print(f"  {ticker}: {data['num_events']} extreme events analyzed")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
