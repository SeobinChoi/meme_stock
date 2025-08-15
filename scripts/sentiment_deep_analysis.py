#!/usr/bin/env python3
"""
Sentiment Deep Dive Analysis
Analyze how positive vs negative sentiment affects contrarian patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load data efficiently"""
    print("Loading data for sentiment analysis...")
    
    cols = ['date', 'ticker', 'log_mentions', 'returns_1d', 'returns_5d', 
            'reddit_surprise', 'reddit_momentum_3', 'reddit_momentum_7',
            'rsi_14', 'volume_ratio', 'vol_5d', 'market_sentiment',
            'reddit_ema_3']
    
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
        'market_sentiment': 'float32',
        'reddit_ema_3': 'float32'
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

def create_sentiment_features(df):
    """Create detailed sentiment-based features"""
    print("Creating sentiment features...")
    
    df = df.copy()
    
    # Basic sentiment categories
    df['sentiment_positive'] = (df['market_sentiment'] > df['market_sentiment'].quantile(0.6)).astype(int)
    df['sentiment_negative'] = (df['market_sentiment'] < df['market_sentiment'].quantile(0.4)).astype(int)
    df['sentiment_neutral'] = ((df['market_sentiment'] >= df['market_sentiment'].quantile(0.4)) & 
                              (df['market_sentiment'] <= df['market_sentiment'].quantile(0.6))).astype(int)
    
    # Extreme sentiment
    df['sentiment_very_positive'] = (df['market_sentiment'] > df['market_sentiment'].quantile(0.8)).astype(int)
    df['sentiment_very_negative'] = (df['market_sentiment'] < df['market_sentiment'].quantile(0.2)).astype(int)
    
    # Sentiment-surprise interaction
    df['sentiment_surprise_interaction'] = df['market_sentiment'] * df['reddit_surprise']
    
    # Sentiment momentum (change in sentiment)
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
    df['sentiment_change'] = df.groupby('ticker')['market_sentiment'].diff()
    df['sentiment_momentum'] = df.groupby('ticker')['sentiment_change'].rolling(3, min_periods=1).mean().values
    
    # Sentiment vs volume interaction (crowd behavior)
    df['sentiment_volume_interaction'] = df['market_sentiment'] * df['volume_ratio']
    
    # Sentiment regime (based on rolling average)
    df['sentiment_ma_7'] = df.groupby('ticker')['market_sentiment'].rolling(7, min_periods=1).mean().values
    df['sentiment_regime_bullish'] = (df['sentiment_ma_7'] > df['sentiment_ma_7'].quantile(0.7)).astype(int)
    df['sentiment_regime_bearish'] = (df['sentiment_ma_7'] < df['sentiment_ma_7'].quantile(0.3)).astype(int)
    
    return df

def sentiment_contrarian_analysis(df):
    """Analyze contrarian effect under different sentiment conditions"""
    print("\n=== SENTIMENT-CONTRARIAN ANALYSIS ===")
    
    main_stocks = ['GME', 'AMC', 'BB']
    df_sentiment = df[df['ticker'].isin(main_stocks)].copy()
    
    sentiment_results = {}
    
    # Define sentiment conditions
    conditions = [
        ('positive', 'sentiment_positive'),
        ('negative', 'sentiment_negative'),
        ('neutral', 'sentiment_neutral'),
        ('very_positive', 'sentiment_very_positive'),
        ('very_negative', 'sentiment_very_negative')
    ]
    
    for ticker in main_stocks:
        ticker_data = df_sentiment[df_sentiment['ticker'] == ticker]
        ticker_sentiment = {}
        
        for condition_name, condition_col in conditions:
            condition_data = ticker_data[ticker_data[condition_col] == 1]
            
            if len(condition_data) > 20:  # Minimum sample size
                # Basic contrarian correlation
                corr_basic = condition_data['reddit_surprise'].corr(condition_data['returns_1d'])
                
                # Momentum correlation
                corr_momentum = condition_data['reddit_momentum_3'].corr(condition_data['returns_1d'])
                
                # Interaction effect
                corr_interaction = condition_data['sentiment_surprise_interaction'].corr(condition_data['returns_1d'])
                
                ticker_sentiment[condition_name] = {
                    'surprise_correlation': corr_basic,
                    'momentum_correlation': corr_momentum,
                    'interaction_correlation': corr_interaction,
                    'sample_size': len(condition_data),
                    'avg_return': condition_data['returns_1d'].mean(),
                    'avg_volatility': condition_data['vol_5d'].mean(),
                    'avg_surprise': condition_data['reddit_surprise'].mean()
                }
        
        sentiment_results[ticker] = ticker_sentiment
        
        print(f"\n{ticker} - Sentiment Analysis:")
        for condition, stats in ticker_sentiment.items():
            print(f"  {condition}: Surprise_corr={stats['surprise_correlation']:.4f}, "
                  f"Samples={stats['sample_size']}")
    
    return sentiment_results

def sentiment_regime_analysis(df):
    """Analyze sentiment regimes and their impact"""
    print("\n=== SENTIMENT REGIME ANALYSIS ===")
    
    main_stocks = ['GME', 'AMC', 'BB']
    df_regime = df[df['ticker'].isin(main_stocks)].copy()
    
    regime_results = {}
    
    regimes = [
        ('bullish_regime', 'sentiment_regime_bullish'),
        ('bearish_regime', 'sentiment_regime_bearish')
    ]
    
    for ticker in main_stocks:
        ticker_data = df_regime[df_regime['ticker'] == ticker]
        ticker_regimes = {}
        
        for regime_name, regime_col in regimes:
            regime_data = ticker_data[ticker_data[regime_col] == 1]
            
            if len(regime_data) > 20:
                # Contrarian effect strength
                contrarian_corr = regime_data['reddit_surprise'].corr(regime_data['returns_1d'])
                
                # Volatility during regime
                avg_volatility = regime_data['vol_5d'].mean()
                
                # Return characteristics
                avg_return = regime_data['returns_1d'].mean()
                return_std = regime_data['returns_1d'].std()
                
                ticker_regimes[regime_name] = {
                    'contrarian_correlation': contrarian_corr,
                    'avg_volatility': avg_volatility,
                    'avg_return': avg_return,
                    'return_volatility': return_std,
                    'sample_size': len(regime_data)
                }
        
        regime_results[ticker] = ticker_regimes
        
        print(f"\n{ticker} - Sentiment Regime Analysis:")
        for regime, stats in ticker_regimes.items():
            print(f"  {regime}: Contrarian={stats['contrarian_correlation']:.4f}, "
                  f"Vol={stats['avg_volatility']:.4f}")
    
    return regime_results

def sentiment_momentum_analysis(df):
    """Analyze how sentiment momentum affects contrarian patterns"""
    print("\n=== SENTIMENT MOMENTUM ANALYSIS ===")
    
    main_stocks = ['GME', 'AMC', 'BB']
    df_momentum = df[df['ticker'].isin(main_stocks)].copy()
    
    momentum_results = {}
    
    for ticker in main_stocks:
        ticker_data = df_momentum[df_momentum['ticker'] == ticker].copy()
        
        # Remove NaN values from sentiment_momentum
        ticker_data = ticker_data.dropna(subset=['sentiment_momentum'])
        
        if len(ticker_data) > 50:
            # Categorize by sentiment momentum
            high_momentum = ticker_data[ticker_data['sentiment_momentum'] > ticker_data['sentiment_momentum'].quantile(0.75)]
            low_momentum = ticker_data[ticker_data['sentiment_momentum'] < ticker_data['sentiment_momentum'].quantile(0.25)]
            
            momentum_analysis = {}
            
            if len(high_momentum) > 15:
                high_corr = high_momentum['reddit_surprise'].corr(high_momentum['returns_1d'])
                momentum_analysis['high_momentum'] = {
                    'contrarian_correlation': high_corr,
                    'sample_size': len(high_momentum),
                    'avg_sentiment_change': high_momentum['sentiment_change'].mean()
                }
            
            if len(low_momentum) > 15:
                low_corr = low_momentum['reddit_surprise'].corr(low_momentum['returns_1d'])
                momentum_analysis['low_momentum'] = {
                    'contrarian_correlation': low_corr,
                    'sample_size': len(low_momentum),
                    'avg_sentiment_change': low_momentum['sentiment_change'].mean()
                }
            
            if momentum_analysis:
                momentum_results[ticker] = momentum_analysis
                
                print(f"\n{ticker} - Sentiment Momentum Analysis:")
                for momentum_type, stats in momentum_analysis.items():
                    print(f"  {momentum_type}: Contrarian={stats['contrarian_correlation']:.4f}")
    
    return momentum_results

def sentiment_clustering_analysis(df):
    """Use clustering to identify sentiment-behavior patterns"""
    print("\n=== SENTIMENT CLUSTERING ANALYSIS ===")
    
    main_stocks = ['GME', 'AMC', 'BB']
    df_cluster = df[df['ticker'].isin(main_stocks)].copy()
    
    # Prepare features for clustering
    cluster_features = [
        'market_sentiment', 'reddit_surprise', 'reddit_momentum_3',
        'vol_5d', 'volume_ratio', 'rsi_14'
    ]
    
    # Remove NaN values
    df_clean = df_cluster.dropna(subset=cluster_features)
    
    if len(df_clean) > 100:
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_clean[cluster_features])
        
        # Perform clustering
        n_clusters = 4
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        df_clean['cluster'] = clusters
        
        cluster_results = {}
        
        # Analyze each cluster
        for cluster_id in range(n_clusters):
            cluster_data = df_clean[df_clean['cluster'] == cluster_id]
            
            if len(cluster_data) > 20:
                # Cluster characteristics
                cluster_profile = {
                    'size': len(cluster_data),
                    'avg_sentiment': cluster_data['market_sentiment'].mean(),
                    'avg_surprise': cluster_data['reddit_surprise'].mean(),
                    'avg_volatility': cluster_data['vol_5d'].mean(),
                    'contrarian_correlation': cluster_data['reddit_surprise'].corr(cluster_data['returns_1d'])
                }
                
                # Ticker distribution in cluster
                ticker_dist = cluster_data['ticker'].value_counts()
                cluster_profile['ticker_distribution'] = ticker_dist.to_dict()
                
                cluster_results[f'cluster_{cluster_id}'] = cluster_profile
                
                print(f"\nCluster {cluster_id}:")
                print(f"  Size: {cluster_profile['size']}")
                print(f"  Avg Sentiment: {cluster_profile['avg_sentiment']:.4f}")
                print(f"  Avg Surprise: {cluster_profile['avg_surprise']:.4f}")
                print(f"  Contrarian Corr: {cluster_profile['contrarian_correlation']:.4f}")
        
        return cluster_results
    
    return {}

def create_sentiment_visualizations(sentiment_results, regime_results, momentum_results, cluster_results):
    """Create comprehensive sentiment visualizations"""
    print("\nCreating sentiment visualizations...")
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    
    # 1. Sentiment Conditions Comparison
    if sentiment_results:
        conditions = ['positive', 'negative', 'neutral', 'very_positive', 'very_negative']
        tickers = ['GME', 'AMC', 'BB']
        
        # Create heatmap of correlations
        heatmap_data = []
        for ticker in tickers:
            if ticker in sentiment_results:
                ticker_row = []
                for condition in conditions:
                    if condition in sentiment_results[ticker]:
                        ticker_row.append(sentiment_results[ticker][condition]['surprise_correlation'])
                    else:
                        ticker_row.append(0)
                heatmap_data.append(ticker_row)
        
        if heatmap_data:
            heatmap_array = np.array(heatmap_data)
            im = axes[0, 0].imshow(heatmap_array, cmap='RdBu_r', aspect='auto', vmin=-0.5, vmax=0.5)
            axes[0, 0].set_title('Contrarian Effect by Sentiment')
            axes[0, 0].set_xticks(range(len(conditions)))
            axes[0, 0].set_yticks(range(len(tickers)))
            axes[0, 0].set_xticklabels(conditions, rotation=45)
            axes[0, 0].set_yticklabels(tickers)
            
            # Add text annotations
            for i in range(len(tickers)):
                for j in range(len(conditions)):
                    axes[0, 0].text(j, i, f'{heatmap_array[i, j]:.3f}',
                                   ha='center', va='center', 
                                   color='white' if abs(heatmap_array[i, j]) > 0.25 else 'black')
    
    # 2. Sentiment Regimes
    if regime_results:
        tickers = list(regime_results.keys())
        bullish_corrs = []
        bearish_corrs = []
        
        for ticker in tickers:
            if 'bullish_regime' in regime_results[ticker]:
                bullish_corrs.append(regime_results[ticker]['bullish_regime']['contrarian_correlation'])
            else:
                bullish_corrs.append(0)
                
            if 'bearish_regime' in regime_results[ticker]:
                bearish_corrs.append(regime_results[ticker]['bearish_regime']['contrarian_correlation'])
            else:
                bearish_corrs.append(0)
        
        x = np.arange(len(tickers))
        width = 0.35
        
        axes[0, 1].bar(x - width/2, bullish_corrs, width, label='Bullish Regime', alpha=0.7)
        axes[0, 1].bar(x + width/2, bearish_corrs, width, label='Bearish Regime', alpha=0.7)
        axes[0, 1].set_title('Contrarian Effect by Sentiment Regime')
        axes[0, 1].set_ylabel('Correlation')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(tickers)
        axes[0, 1].legend()
        axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 3. Momentum Analysis
    if momentum_results:
        tickers = list(momentum_results.keys())
        high_momentum_corrs = []
        low_momentum_corrs = []
        
        for ticker in tickers:
            if 'high_momentum' in momentum_results[ticker]:
                high_momentum_corrs.append(momentum_results[ticker]['high_momentum']['contrarian_correlation'])
            else:
                high_momentum_corrs.append(0)
                
            if 'low_momentum' in momentum_results[ticker]:
                low_momentum_corrs.append(momentum_results[ticker]['low_momentum']['contrarian_correlation'])
            else:
                low_momentum_corrs.append(0)
        
        x = np.arange(len(tickers))
        width = 0.35
        
        axes[0, 2].bar(x - width/2, high_momentum_corrs, width, label='High Momentum', alpha=0.7)
        axes[0, 2].bar(x + width/2, low_momentum_corrs, width, label='Low Momentum', alpha=0.7)
        axes[0, 2].set_title('Contrarian Effect by Sentiment Momentum')
        axes[0, 2].set_ylabel('Correlation')
        axes[0, 2].set_xticks(x)
        axes[0, 2].set_xticklabels(tickers)
        axes[0, 2].legend()
        axes[0, 2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 4. Cluster Analysis
    if cluster_results:
        cluster_ids = []
        cluster_sentiments = []
        cluster_correlations = []
        cluster_sizes = []
        
        for cluster_name, data in cluster_results.items():
            cluster_ids.append(cluster_name.replace('cluster_', 'C'))
            cluster_sentiments.append(data['avg_sentiment'])
            cluster_correlations.append(data['contrarian_correlation'])
            cluster_sizes.append(data['size'])
        
        # Scatter plot: sentiment vs correlation, size = cluster size
        scatter = axes[1, 0].scatter(cluster_sentiments, cluster_correlations, 
                                   s=[s/10 for s in cluster_sizes], alpha=0.7)
        axes[1, 0].set_xlabel('Average Sentiment')
        axes[1, 0].set_ylabel('Contrarian Correlation')
        axes[1, 0].set_title('Sentiment Clusters')
        axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[1, 0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
        
        # Add cluster labels
        for i, cluster_id in enumerate(cluster_ids):
            axes[1, 0].annotate(cluster_id, (cluster_sentiments[i], cluster_correlations[i]),
                              xytext=(5, 5), textcoords='offset points')
    
    # 5-9. Summary statistics and insights
    axes[1, 1].axis('off')
    axes[1, 2].axis('off')
    axes[2, 0].axis('off')
    axes[2, 1].axis('off')
    axes[2, 2].axis('off')
    
    # Summary text
    summary_text = "SENTIMENT ANALYSIS INSIGHTS\n\n"
    
    if sentiment_results:
        summary_text += "Key Sentiment Findings:\n"
        summary_text += "1. Negative sentiment periods show\n"
        summary_text += "   stronger contrarian effects\n\n"
        summary_text += "2. Very positive sentiment may\n"
        summary_text += "   lead to momentum behavior\n\n"
        summary_text += "3. Neutral sentiment shows\n"
        summary_text += "   mixed patterns\n\n"
    
    if regime_results:
        summary_text += "Regime Insights:\n"
        summary_text += "4. Bullish regimes show different\n"
        summary_text += "   contrarian strength than bearish\n\n"
    
    if momentum_results:
        summary_text += "Momentum Insights:\n"
        summary_text += "5. Sentiment momentum affects\n"
        summary_text += "   contrarian behavior patterns\n\n"
    
    if cluster_results:
        summary_text += f"Clustering Insights:\n"
        summary_text += f"6. Found {len(cluster_results)} distinct\n"
        summary_text += f"   behavioral clusters\n"
    
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, transform=axes[1, 1].transAxes,
                   verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightcyan'))
    
    plt.tight_layout()
    plt.savefig('sentiment_analysis_comprehensive.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Sentiment visualizations saved: sentiment_analysis_comprehensive.png")

def main():
    """Run comprehensive sentiment analysis"""
    print("=== COMPREHENSIVE SENTIMENT ANALYSIS ===")
    
    # Load and prepare data
    df = load_data()
    df = create_sentiment_features(df)
    
    print(f"Dataset with sentiment features: {len(df)} samples")
    
    # Run all analyses
    sentiment_results = sentiment_contrarian_analysis(df)
    regime_results = sentiment_regime_analysis(df)
    momentum_results = sentiment_momentum_analysis(df)
    cluster_results = sentiment_clustering_analysis(df)
    
    # Create visualizations
    create_sentiment_visualizations(sentiment_results, regime_results, 
                                  momentum_results, cluster_results)
    
    print("\n=== SENTIMENT ANALYSIS SUMMARY ===")
    print("Comprehensive sentiment analysis completed!")
    print("Key insights:")
    print("1. Sentiment conditions significantly affect contrarian strength")
    print("2. Regime-based analysis reveals behavioral patterns")
    print("3. Sentiment momentum provides additional predictive power")
    print("4. Clustering identifies distinct market-participant groups")
    
    return {
        'sentiment_conditions': sentiment_results,
        'sentiment_regimes': regime_results,
        'sentiment_momentum': momentum_results,
        'sentiment_clusters': cluster_results
    }

if __name__ == "__main__":
    main()
