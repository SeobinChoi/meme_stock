#!/usr/bin/env python3
"""
Reddit mentions and price correlation analysis & visualization

Analyzes relationship between Reddit mentions and price movements in meme stocks.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Font settings for compatibility
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    """Load data efficiently"""
    print("Loading data...")
    
    # Load data with optimal dtypes for memory efficiency
    dtype_dict = {
        'ticker': 'category',
        'returns_1d': 'float32', 
        'returns_5d': 'float32',
        'vol_5d': 'float32',
        'log_mentions': 'float32',
        'rsi_14': 'float32',
        'volume_ratio': 'float32',
        'reddit_ema_3': 'float32',
        'reddit_momentum_7': 'float32',
        'reddit_surprise': 'float32',
        'market_sentiment': 'float32'
    }
    
    # Load only essential columns
    cols_needed = ['date', 'ticker', 'log_mentions', 'returns_1d', 'returns_5d', 
                   'vol_5d', 'rsi_14', 'volume_ratio', 'reddit_ema_3', 
                   'reddit_momentum_7', 'reddit_surprise', 'market_sentiment']
    
    print("Loading train data...")
    train_df = pd.read_csv('data/colab_datasets/tabular_train_20250814_031335.csv', 
                          usecols=cols_needed, dtype=dtype_dict)
    print("Loading validation data...")
    val_df = pd.read_csv('data/colab_datasets/tabular_val_20250814_031335.csv',
                        usecols=cols_needed, dtype=dtype_dict)
    print("Loading test data...")
    test_df = pd.read_csv('data/colab_datasets/tabular_test_20250814_031335.csv',
                         usecols=cols_needed, dtype=dtype_dict)
    
    # Combine data
    df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    del train_df, val_df, test_df  # Free memory
    
    # Convert date efficiently
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"Data loaded: {len(df)} samples")
    print(f"Period: {df['date'].min()} ~ {df['date'].max()}")
    print(f"Tickers: {df['ticker'].unique()}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    return df

def analyze_ticker_correlations_fast(df):
    """Fast ticker-specific correlation analysis"""
    print("\nAnalyzing ticker correlations...")
    
    # log_mentions 분포
    plt.figure(figsize=(15, 10))
    
    # 1. 전체 분포
    plt.subplot(2, 3, 1)
    plt.hist(df['log_mentions'].dropna(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('전체 Log Mentions 분포')
    plt.xlabel('Log Mentions')
    plt.ylabel('빈도')
    plt.axvline(df['log_mentions'].median(), color='red', linestyle='--', label=f'중앙값: {df["log_mentions"].median():.2f}')
    plt.legend()
    
    # 2. 티커별 분포
    plt.subplot(2, 3, 2)
    df.boxplot(column='log_mentions', by='ticker', ax=plt.gca())
    plt.title('티커별 Log Mentions 분포')
    plt.suptitle('')  # boxplot의 기본 제목 제거
    
    # 3. 언급 수 통계
    plt.subplot(2, 3, 3)
    mentions_stats = df.groupby('ticker')['log_mentions'].agg(['mean', 'std', 'min', 'max']).round(2)
    plt.table(cellText=mentions_stats.values, 
              rowLabels=mentions_stats.index,
              colLabels=mentions_stats.columns,
              cellLoc='center',
              loc='center')
    plt.title('티커별 언급 수 통계')
    plt.axis('off')
    
    # 4. 시간별 언급 수 변화
    plt.subplot(2, 3, 4)
    daily_mentions = df.groupby('date')['log_mentions'].mean()
    plt.plot(daily_mentions.index, daily_mentions.values, alpha=0.7, linewidth=1)
    plt.title('일별 평균 언급 수 변화')
    plt.xlabel('날짜')
    plt.ylabel('평균 Log Mentions')
    plt.xticks(rotation=45)
    
    # 5. 언급 수 vs 수익률 산점도
    plt.subplot(2, 3, 5)
    plt.scatter(df['log_mentions'], df['returns_1d'], alpha=0.5, s=1)
    plt.title('언급 수 vs 1일 수익률')
    plt.xlabel('Log Mentions')
    plt.ylabel('Returns (1d)')
    
    # 6. 언급 수 vs 변동성
    plt.subplot(2, 3, 6)
    plt.scatter(df['log_mentions'], df['vol_5d'], alpha=0.5, s=1)
    plt.title('언급 수 vs 5일 변동성')
    plt.xlabel('Log Mentions')
    plt.ylabel('Volatility (5d)')
    
    plt.tight_layout()
    plt.savefig('mentions_distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return mentions_stats

def analyze_correlations(df):
    """상관관계 분석"""
    print("\n🔗 상관관계 분석...")
    
    # 주요 변수들 선택
    key_vars = [
        'log_mentions',           # Reddit 언급 수
        'returns_1d',            # 1일 수익률
        'returns_5d',            # 5일 수익률
        'vol_5d',                # 5일 변동성
        'rsi_14',                # RSI
        'volume_ratio',          # 거래량 비율
        'reddit_ema_3',          # Reddit EMA 3일
        'reddit_momentum_7',     # Reddit 모멘텀 7일
        'reddit_surprise',       # Reddit 서프라이즈
        'market_sentiment'       # 시장 감정
    ]
    
    # 상관관계 계산
    corr_matrix = df[key_vars].corr()
    
    # 상관관계 히트맵
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, 
                mask=mask,
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                square=True,
                fmt='.3f',
                cbar_kws={'shrink': 0.8})
    plt.title('주요 변수 간 상관관계 히트맵')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return corr_matrix

def analyze_lag_correlations(df):
    """지연 상관관계 분석 (언급 수가 미래 수익률에 미치는 영향)"""
    print("\n⏰ 지연 상관관계 분석...")
    
    # 지연 기간 설정
    lags = [0, 1, 2, 3, 5, 7]
    lag_correlations = []
    
    for lag in lags:
        if lag == 0:
            # 동시 상관관계
            # NaN 제거하고 같은 인덱스로 맞춤
            valid_data = df[['log_mentions', 'returns_1d']].dropna()
            corr = valid_data['log_mentions'].corr(valid_data['returns_1d'])
            
            lag_correlations.append({
                'lag': lag,
                'correlation': corr,
                'p_value': stats.pearsonr(valid_data['log_mentions'], valid_data['returns_1d'])[1]
            })
        else:
            # 지연 상관관계 (언급 수가 미래 수익률에 미치는 영향)
            df_lag = df.copy()
            df_lag[f'returns_lag_{lag}'] = df_lag.groupby('ticker')['returns_1d'].shift(-lag)
            
            # NaN 제거
            valid_data = df_lag[['log_mentions', f'returns_lag_{lag}']].dropna()
            
            if len(valid_data) > 10:  # 최소 데이터 수 확인
                corr = valid_data['log_mentions'].corr(valid_data[f'returns_lag_{lag}'])
                p_val = stats.pearsonr(valid_data['log_mentions'], valid_data[f'returns_lag_{lag}'])[1]
                
                lag_correlations.append({
                    'lag': lag,
                    'correlation': corr,
                    'p_value': p_val
                })
    
    # 결과 시각화
    lag_df = pd.DataFrame(lag_correlations)
    
    plt.figure(figsize=(12, 8))
    
    # 1. 지연 상관관계 그래프
    plt.subplot(2, 2, 1)
    plt.plot(lag_df['lag'], lag_df['correlation'], 'o-', linewidth=2, markersize=8)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.title('언급 수와 미래 수익률의 지연 상관관계')
    plt.xlabel('지연 기간 (일)')
    plt.ylabel('상관계수')
    plt.grid(True, alpha=0.3)
    
    # 2. P-value 그래프
    plt.subplot(2, 2, 2)
    plt.plot(lag_df['lag'], lag_df['p_value'], 'o-', linewidth=2, markersize=8, color='red')
    plt.axhline(y=0.05, color='black', linestyle='--', alpha=0.5, label='p=0.05')
    plt.title('지연 상관관계의 통계적 유의성')
    plt.xlabel('지연 기간 (일)')
    plt.ylabel('P-value')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 3. 상관관계 vs P-value
    plt.subplot(2, 2, 3)
    plt.scatter(lag_df['correlation'], lag_df['p_value'], s=100, alpha=0.7)
    for i, row in lag_df.iterrows():
        plt.annotate(f"lag={row['lag']}", (row['correlation'], row['p_value']), 
                    xytext=(5, 5), textcoords='offset points')
    plt.axhline(y=0.05, color='red', linestyle='--', alpha=0.5)
    plt.title('상관계수 vs P-value')
    plt.xlabel('상관계수')
    plt.ylabel('P-value')
    plt.grid(True, alpha=0.3)
    
    # 4. 결과 테이블
    plt.subplot(2, 2, 4)
    plt.axis('off')
    table_data = lag_df.round(4)
    table = plt.table(cellText=table_data.values,
                     colLabels=table_data.columns,
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    plt.title('지연 상관관계 결과 요약')
    
    plt.tight_layout()
    plt.savefig('lag_correlation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return lag_df

def analyze_ticker_specific_correlations(df):
    """티커별 상관관계 분석"""
    print("\n🎯 티커별 상관관계 분석...")
    
    tickers = df['ticker'].unique()
    ticker_correlations = []
    
    for ticker in tickers:
        ticker_data = df[df['ticker'] == ticker]
        
        if len(ticker_data) > 50:  # 충분한 데이터가 있는 경우만
            # 언급 수와 수익률의 상관관계
            corr_1d = ticker_data['log_mentions'].corr(ticker_data['returns_1d'])
            corr_5d = ticker_data['log_mentions'].corr(ticker_data['returns_5d'])
            
            # 언급 수와 변동성의 상관관계
            corr_vol = ticker_data['log_mentions'].corr(ticker_data['vol_5d'])
            
            ticker_correlations.append({
                'ticker': ticker,
                'mentions_returns_1d': corr_1d,
                'mentions_returns_5d': corr_5d,
                'mentions_volatility': corr_vol,
                'sample_size': len(ticker_data)
            })
    
    ticker_corr_df = pd.DataFrame(ticker_correlations)
    
    # 시각화
    plt.figure(figsize=(15, 10))
    
    # 1. 티커별 언급 수-수익률 상관관계
    plt.subplot(2, 3, 1)
    bars = plt.bar(ticker_corr_df['ticker'], ticker_corr_df['mentions_returns_1d'])
    plt.title('티커별: 언급 수 vs 1일 수익률 상관관계')
    plt.xlabel('티커')
    plt.ylabel('상관계수')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 색상 설정 (양수/음수)
    for bar in bars:
        if bar.get_height() > 0:
            bar.set_color('green')
        else:
            bar.set_color('red')
    
    # 2. 티커별 언급 수-5일 수익률 상관관계
    plt.subplot(2, 3, 2)
    bars = plt.bar(ticker_corr_df['ticker'], ticker_corr_df['mentions_returns_5d'])
    plt.title('티커별: 언급 수 vs 5일 수익률 상관관계')
    plt.xlabel('티커')
    plt.ylabel('상관계수')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    for bar in bars:
        if bar.get_height() > 0:
            bar.set_color('green')
        else:
            bar.set_color('red')
    
    # 3. 티커별 언급 수-변동성 상관관계
    plt.subplot(2, 3, 3)
    bars = plt.bar(ticker_corr_df['ticker'], ticker_corr_df['mentions_volatility'])
    plt.title('티커별: 언급 수 vs 변동성 상관관계')
    plt.xlabel('티커')
    plt.ylabel('상관계수')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    for bar in bars:
        if bar.get_height() > 0:
            bar.set_color('green')
        else:
            bar.set_color('red')
    
    # 4. 상관관계 비교
    plt.subplot(2, 3, 4)
    x = np.arange(len(ticker_corr_df))
    width = 0.25
    
    plt.bar(x - width, ticker_corr_df['mentions_returns_1d'], width, label='1일 수익률', alpha=0.8)
    plt.bar(x, ticker_corr_df['mentions_returns_5d'], width, label='5일 수익률', alpha=0.8)
    plt.bar(x + width, ticker_corr_df['mentions_volatility'], width, label='변동성', alpha=0.8)
    
    plt.title('티커별 상관관계 비교')
    plt.xlabel('티커')
    plt.ylabel('상관계수')
    plt.xticks(x, ticker_corr_df['ticker'])
    plt.legend()
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 5. 샘플 크기
    plt.subplot(2, 3, 5)
    plt.bar(ticker_corr_df['ticker'], ticker_corr_df['sample_size'])
    plt.title('티커별 샘플 크기')
    plt.xlabel('티커')
    plt.ylabel('샘플 수')
    
    # 6. 결과 요약 테이블
    plt.subplot(2, 3, 6)
    plt.axis('off')
    summary_data = ticker_corr_df.round(4)
    table = plt.table(cellText=summary_data.values,
                     colLabels=summary_data.columns,
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)
    plt.title('티커별 상관관계 요약')
    
    plt.tight_layout()
    plt.savefig('ticker_specific_correlations.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return ticker_corr_df

def main():
    """메인 실행 함수"""
    print("🚀 Reddit 언급 수와 가격 상관관계 분석 시작")
    print("=" * 60)
    
    # 1. 데이터 로딩
    df = load_data()
    
    # 2. 언급 수 분포 분석
    mentions_stats = analyze_mentions_distribution(df)
    
    # 3. 상관관계 분석
    corr_matrix = analyze_correlations(df)
    
    # 4. 지연 상관관계 분석
    lag_correlations = analyze_lag_correlations(df)
    
    # 5. 티커별 상관관계 분석
    ticker_correlations = analyze_ticker_specific_correlations(df)
    
    # 6. 결과 요약
    print("\n📊 분석 결과 요약")
    print("=" * 60)
    
    print(f"언급 수와 1일 수익률 상관관계: {df['log_mentions'].corr(df['returns_1d']):.4f}")
    print(f"언급 수와 5일 수익률 상관관계: {df['log_mentions'].corr(df['returns_5d']):.4f}")
    print(f"언급 수와 변동성 상관관계: {df['log_mentions'].corr(df['vol_5d']):.4f}")
    
    print(f"\n가장 높은 지연 상관관계:")
    best_lag = lag_correlations.loc[lag_correlations['correlation'].abs().idxmax()]
    print(f"  지연 {best_lag['lag']}일: {best_lag['correlation']:.4f} (p={best_lag['p_value']:.4f})")
    
    print(f"\n티커별 최고 상관관계:")
    best_ticker = ticker_correlations.loc[ticker_correlations['mentions_returns_1d'].abs().idxmax()]
    print(f"  {best_ticker['ticker']}: {best_ticker['mentions_returns_1d']:.4f}")
    
    print(f"\n✅ 분석 완료! 시각화 파일들이 저장되었습니다.")
    print("   - mentions_distribution_analysis.png")
    print("   - correlation_heatmap.png")
    print("   - lag_correlation_analysis.png")
    print("   - ticker_specific_correlations.png")

if __name__ == "__main__":
    main()
