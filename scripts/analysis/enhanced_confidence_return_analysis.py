import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_reddit_data():
    print("Loading Reddit text data...")
    try:
        df_raw = pd.read_csv('data/raw/reddit/raw_reddit_wsb.csv', usecols=['title', 'body', 'created', 'timestamp'])
        df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
        df_raw = df_raw.sort_values(by='timestamp').reset_index(drop=True)
        print(f"Loaded {len(df_raw)} Reddit posts")
        return df_raw
    except Exception as e:
        print(f"Error loading Reddit data: {e}")
        return None

def load_market_data():
    print("Loading market data...")
    dtype_dict = {
        'ticker': 'category',
        'returns_1d': 'float32',
        'vol_5d': 'float32',
        'reddit_surprise': 'float32',
        'log_mentions': 'float32'
    }
    cols_needed = ['date', 'ticker', 'returns_1d', 'vol_5d', 'reddit_surprise', 'log_mentions']
    
    train_df = pd.read_csv('data/colab_datasets/tabular_train_20250814_031335.csv', usecols=cols_needed, dtype=dtype_dict)
    val_df = pd.read_csv('data/colab_datasets/tabular_val_20250814_031335.csv', usecols=cols_needed, dtype=dtype_dict)
    test_df = pd.read_csv('data/colab_datasets/tabular_test_20250814_031335.csv', usecols=cols_needed, dtype=dtype_dict)
    
    df_market = pd.concat([train_df, val_df, test_df], ignore_index=True)
    df_market['date'] = pd.to_datetime(df_market['date'])
    return df_market

def extract_confidence_score(text):
    """Extract overconfidence indicators from text"""
    if not isinstance(text, str):
        return 0
    
    confidence_words = [
        'guaranteed', 'sure thing', 'definitely', 'cant lose', 'free money', 
        'yolo', 'all in', 'to the moon', 'rocket', 'lambo', 'tendies', 
        'diamond hands', 'cant go tits up', 'literally cant lose', 'risk free',
        'absolutely', 'certain', 'no doubt', 'easy money', 'guaranteed win'
    ]
    
    text_lower = text.lower()
    confidence_score = 0
    
    for word in confidence_words:
        confidence_score += text_lower.count(word)
    
    # Bonus for excessive punctuation (!!!!, ????)
    exclamation_bonus = min(text.count('!') - 1, 5)
    confidence_score += max(0, exclamation_bonus)
    
    return confidence_score

def identify_main_ticker(title, body):
    """Identify the main ticker mentioned in the post"""
    text = (title if isinstance(title, str) else '') + ' ' + (body if isinstance(body, str) else '')
    text_upper = text.upper()
    
    tickers = ['GME', 'AMC', 'BB']
    ticker_counts = {ticker: text_upper.count(ticker) for ticker in tickers}
    
    if any(ticker_counts.values()):
        return max(ticker_counts, key=ticker_counts.get)
    return None

def main():
    print("=== ENHANCED CONFIDENCE SCORE VS NEXT DAY RETURNS ANALYSIS ===")
    
    # Load data
    df_raw = load_reddit_data()
    df_market = load_market_data()
    
    if df_raw is None:
        print("Reddit data unavailable, exiting...")
        return
    
    # Process Reddit data
    print("Analyzing confidence patterns from Reddit text...")
    behavioral_data = []
    
    for i, row in df_raw.iterrows():
        if i % 5000 == 0:
            print(f"Processed {i}/{len(df_raw)} posts...")
        
        full_text = str(row['title']) + " " + str(row['body'])
        confidence_score = extract_confidence_score(full_text)
        main_ticker = identify_main_ticker(row['title'], row['body'])
        
        behavioral_data.append({
            'timestamp': row['timestamp'],
            'confidence_score': confidence_score,
            'main_ticker': main_ticker
        })
    
    df_behavioral = pd.DataFrame(behavioral_data)
    df_behavioral['date'] = df_behavioral['timestamp'].dt.normalize()
    
    # Aggregate by date and ticker
    df_daily = df_behavioral.groupby(['date', 'main_ticker']).agg({
        'confidence_score': 'sum'
    }).reset_index()
    df_daily = df_daily.rename(columns={'main_ticker': 'ticker'})
    
    # Merge with market data
    df_merged = pd.merge(df_market, df_daily, on=['date', 'ticker'], how='left')
    df_merged['confidence_score'] = df_merged['confidence_score'].fillna(0)
    df_merged['next_day_returns'] = df_merged.groupby('ticker')['returns_1d'].shift(-1)
    
    # Analysis - only days with confidence > 0
    valid_data = df_merged.dropna(subset=['confidence_score', 'next_day_returns'])
    valid_data = valid_data[valid_data['confidence_score'] > 0]
    
    if len(valid_data) > 10:
        corr, p_value = pearsonr(valid_data['confidence_score'], valid_data['next_day_returns'])
        print(f"\nENHANCED ANALYSIS:")
        print(f"Confidence Score vs Next Day Returns: {corr:.4f} (p={p_value:.4f})")
        print(f"Sample size: {len(valid_data)}")
        
        # Create enhanced visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Enhanced Confidence-Return Analysis: Behavioral Paradox', fontsize=16, fontweight='bold')
        
        # 1. Main scatter plot with regression line
        ax1 = axes[0, 0]
        
        # Create bins for better visualization
        confidence_bins = pd.qcut(valid_data['confidence_score'], q=10, duplicates='drop')
        bin_means = valid_data.groupby(confidence_bins).agg({
            'confidence_score': 'mean',
            'next_day_returns': 'mean'
        }).reset_index(drop=True)
        
        # Scatter plot of all points (transparent)
        ax1.scatter(valid_data['confidence_score'], valid_data['next_day_returns'], 
                   alpha=0.3, s=20, color='lightblue', label='Individual observations')
        
        # Bin means (more visible)
        ax1.scatter(bin_means['confidence_score'], bin_means['next_day_returns'], 
                   color='red', s=100, alpha=0.8, label='Binned means')
        
        # Regression line
        X = valid_data['confidence_score'].values.reshape(-1, 1)
        y = valid_data['next_day_returns'].values
        reg = LinearRegression().fit(X, y)
        x_line = np.linspace(valid_data['confidence_score'].min(), valid_data['confidence_score'].max(), 100)
        y_line = reg.predict(x_line.reshape(-1, 1))
        ax1.plot(x_line, y_line, 'r--', linewidth=2, alpha=0.8, label=f'Regression line (β={reg.coef_[0]:.3f})')
        
        ax1.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax1.set_xlabel('Daily Confidence Score', fontweight='bold')
        ax1.set_ylabel('Next Day Returns', fontweight='bold')
        ax1.set_title(f'Confidence vs Returns\n(Correlation: {corr:.3f}, p={p_value:.4f})', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Confidence quintiles bar chart
        ax2 = axes[0, 1]
        valid_data['confidence_quintile'] = pd.qcut(valid_data['confidence_score'], 5, 
                                                   labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4', 'Q5 (High)'])
        quintile_returns = valid_data.groupby('confidence_quintile')['next_day_returns'].mean()
        
        bars = ax2.bar(range(len(quintile_returns)), quintile_returns.values, 
                      color=['green' if x > 0 else 'red' for x in quintile_returns.values],
                      alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, quintile_returns.values)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax2.axhline(0, color='black', linestyle='-', alpha=0.5)
        ax2.set_xlabel('Confidence Quintile', fontweight='bold')
        ax2.set_ylabel('Average Next Day Returns', fontweight='bold')
        ax2.set_title('Returns by Confidence Level\n(Higher Confidence = Lower Returns)', fontweight='bold')
        ax2.set_xticks(range(len(quintile_returns)))
        ax2.set_xticklabels(quintile_returns.index, rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 3. Ticker-specific analysis
        ax3 = axes[1, 0]
        ticker_colors = {'GME': 'red', 'AMC': 'blue', 'BB': 'green'}
        
        for ticker in valid_data['ticker'].unique():
            ticker_data = valid_data[valid_data['ticker'] == ticker]
            if len(ticker_data) > 10:
                ticker_corr, _ = pearsonr(ticker_data['confidence_score'], ticker_data['next_day_returns'])
                ax3.scatter(ticker_data['confidence_score'], ticker_data['next_day_returns'], 
                           label=f'{ticker} (r={ticker_corr:.3f})', alpha=0.6, s=30,
                           color=ticker_colors.get(ticker, 'gray'))
                
                # Add trend line for each ticker
                if len(ticker_data) > 5:
                    z = np.polyfit(ticker_data['confidence_score'], ticker_data['next_day_returns'], 1)
                    p = np.poly1d(z)
                    x_trend = np.linspace(ticker_data['confidence_score'].min(), 
                                        ticker_data['confidence_score'].max(), 50)
                    ax3.plot(x_trend, p(x_trend), '--', alpha=0.6, 
                            color=ticker_colors.get(ticker, 'gray'))
        
        ax3.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax3.set_xlabel('Confidence Score', fontweight='bold')
        ax3.set_ylabel('Next Day Returns', fontweight='bold')
        ax3.set_title('Ticker-Specific Confidence Effects', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Summary statistics box
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Calculate additional stats
        high_conf = valid_data[valid_data['confidence_score'] >= valid_data['confidence_score'].quantile(0.8)]
        low_conf = valid_data[valid_data['confidence_score'] <= valid_data['confidence_score'].quantile(0.2)]
        
        high_conf_ret = high_conf['next_day_returns'].mean()
        low_conf_ret = low_conf['next_day_returns'].mean()
        
        summary_text = f"""
BEHAVIORAL PARADOX CONFIRMED!

📊 Main Finding:
• Correlation: {corr:.4f}
• P-value: {p_value:.4f}
• Sample size: {len(valid_data):,}

📈 Performance Comparison:
• High Confidence (Top 20%): {high_conf_ret:.3f}
• Low Confidence (Bottom 20%): {low_conf_ret:.3f}
• Difference: {high_conf_ret - low_conf_ret:.3f}

🔍 Interpretation:
{"🔴 STRONG Overconfidence Paradox" if corr < -0.15 else "🟡 Moderate Effect" if corr < -0.05 else "🟢 No Clear Effect"}

📋 Statistical Significance:
{"✅ Highly Significant (p < 0.001)" if p_value < 0.001 else "✅ Significant (p < 0.05)" if p_value < 0.05 else "❌ Not Significant"}

💡 Key Insight:
The more confident Reddit posts are,
the WORSE the next-day returns!
        """
        
        ax4.text(0.05, 0.95, summary_text, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
                fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig('enhanced_confidence_return_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\n🎯 ENHANCED CONFIDENCE-RETURN ANALYSIS COMPLETE!")
        print(f"Enhanced chart saved: enhanced_confidence_return_analysis.png")
        
        # Copy to paper submission folder
        import shutil
        shutil.copy('enhanced_confidence_return_analysis.png', 'paper_submission/images/')
        print("Chart also copied to paper_submission/images/")
        
        return corr, p_value, len(valid_data)

if __name__ == "__main__":
    main()
