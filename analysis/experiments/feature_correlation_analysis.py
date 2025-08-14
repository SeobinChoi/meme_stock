#!/usr/bin/env python3
"""
Feature Correlation Analysis
Analyzes correlation between Reddit features and stock movements
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def analyze_feature_correlations():
    """
    Analyze correlations between Reddit features and stock movements
    """
    print("🔍 Analyzing Feature Correlations with Stock Movements...")
    
    # Load the features dataset
    features_file = Path("data/features/engineered_features_dataset.csv")
    if not features_file.exists():
        print("❌ Features dataset not found")
        return
    
    df = pd.read_csv(features_file)
    print(f"✅ Loaded dataset with {len(df)} samples and {len(df.columns)} features")
    
    # Identify Reddit features and stock return features
    reddit_features = [col for col in df.columns if col.startswith('reddit_')]
    stock_returns = [col for col in df.columns if 'returns_1d' in col and col.endswith('_1d')]
    
    print(f"\n📊 Found {len(reddit_features)} Reddit features")
    print(f"📈 Found {len(stock_returns)} stock return features: {stock_returns}")
    
    # Calculate correlations
    correlation_results = {}
    
    for stock_return in stock_returns:
        if stock_return in df.columns:
            correlations = {}
            for reddit_feature in reddit_features:
                if reddit_feature in df.columns:
                    # Calculate correlation
                    corr = df[reddit_feature].corr(df[stock_return])
                    correlations[reddit_feature] = corr
            
            # Sort by absolute correlation
            sorted_correlations = sorted(correlations.items(), 
                                       key=lambda x: abs(x[1]), reverse=True)
            
            correlation_results[stock_return] = sorted_correlations
    
    # Print top correlations for each stock
    print("\n" + "="*80)
    print("TOP REDDIT FEATURE CORRELATIONS WITH STOCK RETURNS")
    print("="*80)
    
    for stock_return, correlations in correlation_results.items():
        print(f"\n🔴 {stock_return.upper()}:")
        print("-" * 40)
        
        # Show top 10 positive and negative correlations
        positive_corr = [c for c in correlations if c[1] > 0][:10]
        negative_corr = [c for c in correlations if c[1] < 0][:10]
        
        print("📈 Top Positive Correlations:")
        for feature, corr in positive_corr:
            print(f"   {feature}: {corr:.4f}")
        
        print("\n📉 Top Negative Correlations:")
        for feature, corr in negative_corr:
            print(f"   {feature}: {corr:.4f}")
        
        print(f"\n💡 Strongest Overall: {correlations[0][0]} ({correlations[0][1]:.4f})")
    
    # Analyze feature quality
    print("\n" + "="*80)
    print("FEATURE QUALITY ANALYSIS")
    print("="*80)
    
    # Check for constant features
    constant_features = []
    for feature in reddit_features:
        if feature in df.columns:
            unique_values = df[feature].nunique()
            if unique_values <= 1:
                constant_features.append(feature)
    
    if constant_features:
        print(f"⚠️  Found {len(constant_features)} constant Reddit features:")
        for feature in constant_features:
            print(f"   - {feature}")
    else:
        print("✅ No constant Reddit features found")
    
    # Check for high correlation between features (multicollinearity)
    print("\n🔍 Checking for Multicollinearity...")
    
    # Create correlation matrix for Reddit features only
    reddit_df = df[reddit_features].select_dtypes(include=[np.number])
    
    if len(reddit_df.columns) > 1:
        corr_matrix = reddit_df.corr()
        
        # Find highly correlated feature pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.95:
                    high_corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_val
                    ))
        
        if high_corr_pairs:
            print(f"⚠️  Found {len(high_corr_pairs)} highly correlated Reddit feature pairs:")
            for feat1, feat2, corr in high_corr_pairs[:10]:  # Show first 10
                print(f"   - {feat1} ↔ {feat2}: {corr:.4f}")
        else:
            print("✅ No highly correlated Reddit feature pairs found")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print(f"📊 Total Features: {len(df.columns)}")
    print(f"🔴 Reddit Features: {len(reddit_features)}")
    print(f"📈 Stock Return Features: {len(stock_returns)}")
    print(f"📅 Total Samples: {len(df)}")
    
    # Data quality metrics
    missing_data = df.isnull().sum().sum()
    total_cells = len(df) * len(df.columns)
    missing_percentage = (missing_data / total_cells) * 100
    
    print(f"❌ Missing Data: {missing_percentage:.2f}%")
    print(f"✅ Data Completeness: {100 - missing_percentage:.2f}%")
    
    return correlation_results

def generate_correlation_report(correlation_results):
    """
    Generate a detailed correlation report
    """
    print("\n" + "="*80)
    print("DETAILED CORRELATION REPORT")
    print("="*80)
    
    for stock_return, correlations in correlation_results.items():
        print(f"\n🎯 {stock_return.upper()} ANALYSIS:")
        print("=" * 50)
        
        # Strongest correlations
        strongest_positive = max(correlations, key=lambda x: x[1] if x[1] > 0 else -1)
        strongest_negative = min(correlations, key=lambda x: x[1] if x[1] < 0 else 1)
        
        print(f"📈 Strongest Positive: {strongest_positive[0]} ({strongest_positive[1]:.4f})")
        print(f"📉 Strongest Negative: {strongest_negative[0]} ({strongest_negative[1]:.4f})")
        
        # Count significant correlations
        significant_positive = len([c for c in correlations if c[1] > 0.1])
        significant_negative = len([c for c in correlations if c[1] < -0.1])
        
        print(f"🔍 Significant Positive Correlations (>0.1): {significant_positive}")
        print(f"🔍 Significant Negative Correlations (<-0.1): {significant_negative}")
        
        # Feature categories analysis
        sentiment_features = [c for c in correlations if 'sentiment' in c[0].lower()]
        engagement_features = [c for c in correlations if any(x in c[0].lower() for x in ['score', 'comment', 'post'])]
        linguistic_features = [c for c in correlations if any(x in c[0].lower() for x in ['length', 'word', 'uppercase', 'exclamation'])]
        
        if sentiment_features:
            avg_sentiment_corr = np.mean([abs(c[1]) for c in sentiment_features])
            print(f"😊 Average Sentiment Feature Correlation: {avg_sentiment_corr:.4f}")
        
        if engagement_features:
            avg_engagement_corr = np.mean([abs(c[1]) for c in engagement_features])
            print(f"👥 Average Engagement Feature Correlation: {avg_engagement_corr:.4f}")
        
        if linguistic_features:
            avg_linguistic_corr = np.mean([abs(c[1]) for c in linguistic_features])
            print(f"📝 Average Linguistic Feature Correlation: {avg_linguistic_corr:.4f}")

def main():
    """
    Main function
    """
    print("🚀 Starting Feature Correlation Analysis...")
    
    # Run correlation analysis
    correlation_results = analyze_feature_correlations()
    
    if correlation_results:
        # Generate detailed report
        generate_correlation_report(correlation_results)
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print("✅ Feature correlation analysis completed successfully")
        print("📊 Check the output above for detailed insights")
    else:
        print("❌ Analysis failed - no correlation results generated")

if __name__ == "__main__":
    main()
