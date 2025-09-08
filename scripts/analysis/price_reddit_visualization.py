#!/usr/bin/env python3
"""
Price vs Reddit Interest Visualization
주가와 Reddit 관심도 간의 관계 시각화
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_and_prepare_data():
    """데이터 로드 및 준비"""
    print("📊 Loading data for price vs Reddit visualization...")
    
    # 통합 데이터셋 로드
    train_df = pd.read_csv('data/colab_datasets/tabular_train_20250814_031335.csv')
    val_df = pd.read_csv('data/colab_datasets/tabular_val_20250814_031335.csv')
    test_df = pd.read_csv('data/colab_datasets/tabular_test_20250814_031335.csv')
    
    # 데이터 통합
    df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
    
    print(f"   ✅ Total data: {len(df)} records")
    print(f"   ✅ Date range: {df['date'].min()} ~ {df['date'].max()}")
    print(f"   ✅ Tickers: {df['ticker'].unique()}")
    
    return df

def create_price_reddit_plots(df):
    """주가와 Reddit 관심도 플롯 생성"""
    print("📈 Creating price vs Reddit interest plots...")
    
    # 종목별로 플롯 생성
    tickers = df['ticker'].unique()
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Price vs Reddit Interest Analysis', fontsize=16, fontweight='bold')
    
    for i, ticker in enumerate(tickers):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        # 해당 종목 데이터 필터링
        ticker_data = df[df['ticker'] == ticker].copy()
        ticker_data = ticker_data.sort_values('date').reset_index(drop=True)
        
        # 주가와 Reddit 관심도 플롯
        ax2 = ax.twinx()
        
        # 주가 수익률 플롯 (왼쪽 y축)
        line1 = ax.plot(ticker_data['date'], ticker_data['returns_1d'], 
                       color='blue', linewidth=2, label='Daily Returns', alpha=0.8)
        ax.set_ylabel('Daily Returns', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        
        # Reddit 관심도 플롯 (오른쪽 y축)
        line2 = ax2.plot(ticker_data['date'], ticker_data['log_mentions'], 
                        color='red', linewidth=2, label='Reddit Interest', alpha=0.8)
        ax2.set_ylabel('Log Mentions', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        ax.set_title(f'{ticker} - Returns vs Reddit Interest', fontweight='bold')
        ax.set_xlabel('Date')
        
        # 범례
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')
        
        # 격자
        ax.grid(True, alpha=0.3)
        
        # x축 날짜 회전
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('results/price_vs_reddit_interest.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("   ✅ Price vs Reddit interest plot saved to results/price_vs_reddit_interest.png")

def create_correlation_plots(df):
    """상관관계 플롯 생성"""
    print("📊 Creating correlation plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Price vs Reddit Interest Correlation Analysis', fontsize=16, fontweight='bold')
    
    tickers = df['ticker'].unique()
    
    for i, ticker in enumerate(tickers):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        # 해당 종목 데이터 필터링
        ticker_data = df[df['ticker'] == ticker].copy()
        
        # 산점도 그리기
        scatter = ax.scatter(ticker_data['log_mentions'], ticker_data['returns_1d'], 
                           alpha=0.6, s=30, c=ticker_data['date'], cmap='viridis')
        
        # 상관계수 계산
        correlation = ticker_data['log_mentions'].corr(ticker_data['returns_1d'])
        
        ax.set_xlabel('Log Mentions (Reddit Interest)')
        ax.set_ylabel('Daily Returns')
        ax.set_title(f'{ticker} - Correlation: {correlation:.3f}', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 추세선 추가
        z = np.polyfit(ticker_data['log_mentions'], ticker_data['returns_1d'], 1)
        p = np.poly1d(z)
        ax.plot(ticker_data['log_mentions'], p(ticker_data['log_mentions']), 
               "r--", alpha=0.8, linewidth=2)
    
    plt.tight_layout()
    plt.savefig('results/price_reddit_correlation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("   ✅ Correlation plot saved to results/price_reddit_correlation.png")

def create_contrarian_analysis(df):
    """Contrarian Effect 분석 플롯"""
    print("🔄 Creating contrarian effect analysis...")
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Contrarian Effect Analysis', fontsize=16, fontweight='bold')
    
    tickers = df['ticker'].unique()
    
    for i, ticker in enumerate(tickers):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        # 해당 종목 데이터 필터링
        ticker_data = df[df['ticker'] == ticker].copy()
        ticker_data = ticker_data.sort_values('date').reset_index(drop=True)
        
        # 다음날 수익률 계산
        ticker_data['next_day_return'] = ticker_data['returns_1d'].shift(-1)
        
        # Reddit 관심도 구간별 분석
        ticker_data['reddit_quartile'] = pd.qcut(ticker_data['log_mentions'], 
                                                q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        
        # 구간별 평균 수익률 계산
        quartile_returns = ticker_data.groupby('reddit_quartile')['next_day_return'].mean()
        
        # 막대 그래프
        bars = ax.bar(quartile_returns.index, quartile_returns.values, 
                     alpha=0.7, color=['lightblue', 'lightgreen', 'orange', 'red'])
        
        ax.set_xlabel('Reddit Interest Quartile')
        ax.set_ylabel('Average Next Day Return')
        ax.set_title(f'{ticker} - Contrarian Effect', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 값 표시
        for bar, value in zip(bars, quartile_returns.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                   f'{value:.3f}', ha='center', va='bottom')
        
        # Contrarian Effect 확인
        q1_return = quartile_returns['Q1']
        q4_return = quartile_returns['Q4']
        contrarian_effect = q1_return - q4_return
        
        ax.text(0.5, 0.95, f'Contrarian Effect: {contrarian_effect:.3f}', 
               transform=ax.transAxes, ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('results/contrarian_effect_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("   ✅ Contrarian effect analysis saved to results/contrarian_effect_analysis.png")

def create_reddit_spike_analysis(df):
    """Reddit 스파이크 분석"""
    print("📈 Creating Reddit spike analysis...")
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Reddit Spike Analysis', fontsize=16, fontweight='bold')
    
    tickers = df['ticker'].unique()
    
    for i, ticker in enumerate(tickers):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        # 해당 종목 데이터 필터링
        ticker_data = df[df['ticker'] == ticker].copy()
        ticker_data = ticker_data.sort_values('date').reset_index(drop=True)
        
        # Reddit 스파이크 식별 (상위 5%)
        spike_threshold = ticker_data['log_mentions'].quantile(0.95)
        ticker_data['is_spike'] = ticker_data['log_mentions'] > spike_threshold
        
        # 스파이크 이벤트 시각화
        spike_dates = ticker_data[ticker_data['is_spike']]['date']
        
        # 주가 수익률 플롯
        ax.plot(ticker_data['date'], ticker_data['returns_1d'], 
               color='blue', linewidth=1, alpha=0.7, label='Daily Returns')
        
        # 스파이크 이벤트 표시
        spike_returns = ticker_data[ticker_data['is_spike']]['returns_1d']
        ax.scatter(spike_dates, spike_returns, 
                 color='red', s=100, alpha=0.8, label='Reddit Spike', zorder=5)
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Daily Returns')
        ax.set_title(f'{ticker} - Reddit Spike Events', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # x축 날짜 회전
        ax.tick_params(axis='x', rotation=45)
        
        # 스파이크 통계
        n_spikes = len(spike_dates)
        ax.text(0.5, 0.95, f'Spikes: {n_spikes}', 
               transform=ax.transAxes, ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('results/reddit_spike_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("   ✅ Reddit spike analysis saved to results/reddit_spike_analysis.png")

def create_summary_statistics(df):
    """요약 통계 생성"""
    print("📊 Creating summary statistics...")
    
    # 종목별 상관계수 계산
    correlations = []
    
    for ticker in df['ticker'].unique():
        ticker_data = df[df['ticker'] == ticker].copy()
        
        # 다양한 상관계수 계산
        corr_current = ticker_data['log_mentions'].corr(ticker_data['returns_1d'])
        corr_next = ticker_data['log_mentions'].corr(ticker_data['returns_1d'].shift(-1))
        corr_lag1 = ticker_data['log_mentions'].shift(1).corr(ticker_data['returns_1d'])
        
        correlations.append({
            'Ticker': ticker,
            'Current_Day_Correlation': corr_current,
            'Next_Day_Correlation': corr_next,
            'Lag_1_Day_Correlation': corr_lag1
        })
    
    corr_df = pd.DataFrame(correlations)
    
    print("\n" + "="*80)
    print("PRICE vs REDDIT INTEREST CORRELATION SUMMARY")
    print("="*80)
    print(corr_df.to_string(index=False))
    print("="*80)
    
    # 전체 요약
    print(f"\n📊 OVERALL SUMMARY:")
    print(f"   Average Current Day Correlation: {corr_df['Current_Day_Correlation'].mean():.4f}")
    print(f"   Average Next Day Correlation: {corr_df['Next_Day_Correlation'].mean():.4f}")
    print(f"   Average Lag 1 Day Correlation: {corr_df['Lag_1_Day_Correlation'].mean():.4f}")
    
    # Contrarian Effect 확인
    negative_corr_count = (corr_df['Next_Day_Correlation'] < 0).sum()
    total_count = len(corr_df)
    
    print(f"\n🔄 CONTRARIAN EFFECT ANALYSIS:")
    print(f"   Negative Correlation Count: {negative_corr_count}/{total_count}")
    print(f"   Contrarian Effect Rate: {negative_corr_count/total_count*100:.1f}%")
    
    if negative_corr_count > total_count / 2:
        print("   ✅ Contrarian Effect DETECTED")
    else:
        print("   ❌ Contrarian Effect NOT DETECTED")
    
    return corr_df

def main():
    """메인 실행 함수"""
    print("🚀 Starting Price vs Reddit Interest Visualization")
    print("=" * 80)
    
    # 결과 디렉토리 생성
    import os
    os.makedirs('results', exist_ok=True)
    
    # 1. 데이터 로드
    df = load_and_prepare_data()
    
    # 2. 주가 vs Reddit 관심도 플롯
    print("\n" + "="*50)
    print("PRICE VS REDDIT INTEREST PLOTS")
    print("="*50)
    create_price_reddit_plots(df)
    
    # 3. 상관관계 플롯
    print("\n" + "="*50)
    print("CORRELATION ANALYSIS")
    print("="*50)
    create_correlation_plots(df)
    
    # 4. Contrarian Effect 분석
    print("\n" + "="*50)
    print("CONTRARIAN EFFECT ANALYSIS")
    print("="*50)
    create_contrarian_analysis(df)
    
    # 5. Reddit 스파이크 분석
    print("\n" + "="*50)
    print("REDDIT SPIKE ANALYSIS")
    print("="*50)
    create_reddit_spike_analysis(df)
    
    # 6. 요약 통계
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    corr_df = create_summary_statistics(df)
    
    print("\n🎉 Price vs Reddit visualization completed!")
    print("📁 Results saved in 'results/' directory")
    
    return df, corr_df

if __name__ == "__main__":
    df, corr_df = main()
