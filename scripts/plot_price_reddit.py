#!/usr/bin/env python3
"""
종목별 가격과 Reddit 언급 수 시각화
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import seaborn as sns

def load_and_plot_data():
    """데이터 로드 및 시각화"""
    
    print("📊 데이터 로드 중...")
    
    # 모든 데이터셋 로드
    datasets = []
    for split in ['train', 'val', 'test']:
        try:
            df = pd.read_csv(f'data/colab_datasets/tabular_{split}_20250814_031335.csv')
            df['split'] = split
            datasets.append(df)
        except FileNotFoundError:
            print(f"⚠️ {split} 데이터 파일을 찾을 수 없습니다")
            continue
    
    if not datasets:
        print("❌ 데이터를 로드할 수 없습니다")
        return
    
    # 데이터 결합
    df = pd.concat(datasets, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"✅ 총 {len(df)} 샘플 로드")
    print(f"   기간: {df['date'].min()} ~ {df['date'].max()}")
    
    # 종목별 데이터 확인
    tickers = df['ticker'].unique()
    print(f"   종목: {list(tickers)}")
    
    # Reddit 언급 관련 컬럼 찾기
    reddit_cols = [col for col in df.columns if 'mentions' in col.lower() or 'reddit' in col.lower()]
    print(f"   Reddit 컬럼: {reddit_cols[:5]}...")  # 처음 5개만 표시
    
    # 가격 관련 컬럼 추정 (수익률에서 역계산하거나 직접 찾기)
    price_cols = [col for col in df.columns if 'price' in col.lower() or 'close' in col.lower()]
    
    # 메인 Reddit 언급 컬럼 선택 (log_mentions 또는 mentions)
    main_reddit_col = None
    for col in ['log_mentions', 'mentions', 'reddit_mentions']:
        if col in df.columns:
            main_reddit_col = col
            break
    
    if main_reddit_col is None:
        # Reddit EMA나 다른 지표 사용
        for col in df.columns:
            if 'reddit' in col.lower() and ('ema' in col or 'momentum' in col):
                main_reddit_col = col
                break
    
    if main_reddit_col is None:
        print("❌ Reddit 언급 컬럼을 찾을 수 없습니다")
        return
    
    print(f"   메인 Reddit 컬럼: {main_reddit_col}")
    
    # 종목별 시각화
    n_tickers = len(tickers)
    fig, axes = plt.subplots(n_tickers, 2, figsize=(15, 6 * n_tickers))
    
    if n_tickers == 1:
        axes = axes.reshape(1, -1)
    
    plt.style.use('default')
    
    for i, ticker in enumerate(tickers):
        ticker_data = df[df['ticker'] == ticker].copy()
        ticker_data = ticker_data.sort_values('date')
        
        print(f"\n📈 {ticker} 데이터 처리 중...")
        print(f"   샘플 수: {len(ticker_data)}")
        
        # 가격 데이터 준비 (수익률에서 가격 지수 계산)
        if 'returns_1d' in ticker_data.columns:
            # 수익률을 이용한 누적 가격 지수 계산
            returns = ticker_data['returns_1d'].fillna(0)
            price_index = (1 + returns).cumprod() * 100  # 100을 기준으로 시작
        else:
            # 임의의 가격 생성 (실제 데이터가 없을 경우)
            price_index = pd.Series(range(100, 100 + len(ticker_data)), index=ticker_data.index)
        
        # Reddit 언급 수 준비
        reddit_mentions = ticker_data[main_reddit_col].fillna(0)
        
        # 로그 스케일이면 원래대로 변환
        if 'log' in main_reddit_col.lower():
            reddit_mentions = np.exp(reddit_mentions) - 1  # log(x+1) 역변환
        
        dates = ticker_data['date']
        
        # 1. 가격 차트
        ax1 = axes[i, 0]
        ax1.plot(dates, price_index, linewidth=2, color='blue', label='가격 지수')
        ax1.set_title(f'{ticker} - 가격 추이', fontsize=14, fontweight='bold')
        ax1.set_xlabel('날짜')
        ax1.set_ylabel('가격 지수 (기준: 100)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 날짜 포맷팅
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # 2. Reddit 언급 수 차트
        ax2 = axes[i, 1]
        ax2.plot(dates, reddit_mentions, linewidth=2, color='red', label='Reddit 언급 수')
        ax2.set_title(f'{ticker} - Reddit 언급 추이', fontsize=14, fontweight='bold')
        ax2.set_xlabel('날짜')
        ax2.set_ylabel('언급 수')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 날짜 포맷팅
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # 통계 출력
        print(f"   가격 지수: {price_index.min():.2f} ~ {price_index.max():.2f}")
        print(f"   Reddit 언급: {reddit_mentions.min():.0f} ~ {reddit_mentions.max():.0f}")
    
    plt.tight_layout()
    
    # 저장
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'price_reddit_analysis_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n💾 그래프 저장: {filename}")
    
    plt.show()
    
    # 상관관계 분석
    print("\n📊 가격-Reddit 상관관계 분석:")
    print("=" * 50)
    
    for ticker in tickers:
        ticker_data = df[df['ticker'] == ticker].copy()
        
        if len(ticker_data) < 10:
            continue
        
        # 수익률과 Reddit 언급의 상관관계
        if 'returns_1d' in ticker_data.columns:
            returns = ticker_data['returns_1d'].dropna()
            reddit_vals = ticker_data[main_reddit_col].dropna()
            
            if len(returns) > 10 and len(reddit_vals) > 10:
                # 같은 날짜의 데이터만 사용
                common_dates = set(ticker_data.dropna(subset=['returns_1d', main_reddit_col])['date'])
                if len(common_dates) > 10:
                    corr_data = ticker_data[ticker_data['date'].isin(common_dates)]
                    correlation = corr_data['returns_1d'].corr(corr_data[main_reddit_col])
                    
                    print(f"{ticker:8s}: 상관계수 = {correlation:6.3f}")
                    
                    if abs(correlation) > 0.1:
                        print(f"         {'강한' if abs(correlation) > 0.3 else '중간'} 상관관계!")

def create_combined_chart():
    """종목별 가격과 Reddit을 하나의 차트에 표시"""
    
    print("\n📊 통합 차트 생성 중...")
    
    # 데이터 로드
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
    
    # Reddit 컬럼 찾기
    reddit_col = None
    for col in ['log_mentions', 'mentions', 'reddit_ema_3']:
        if col in df.columns:
            reddit_col = col
            break
    
    if reddit_col is None:
        return
    
    tickers = df['ticker'].unique()
    
    # 통합 차트
    fig, axes = plt.subplots(len(tickers), 1, figsize=(15, 4 * len(tickers)))
    if len(tickers) == 1:
        axes = [axes]
    
    for i, ticker in enumerate(tickers):
        ticker_data = df[df['ticker'] == ticker].sort_values('date')
        
        ax = axes[i]
        
        # 가격 (왼쪽 y축)
        if 'returns_1d' in ticker_data.columns:
            price_index = (1 + ticker_data['returns_1d'].fillna(0)).cumprod() * 100
        else:
            price_index = pd.Series(range(100, 100 + len(ticker_data)))
        
        ax.plot(ticker_data['date'], price_index, color='blue', linewidth=2, label='가격 지수')
        ax.set_ylabel('가격 지수', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        
        # Reddit 언급 (오른쪽 y축)
        ax2 = ax.twinx()
        reddit_data = ticker_data[reddit_col].fillna(0)
        if 'log' in reddit_col.lower():
            reddit_data = np.exp(reddit_data) - 1
        
        ax2.plot(ticker_data['date'], reddit_data, color='red', linewidth=2, alpha=0.7, label='Reddit 언급')
        ax2.set_ylabel('Reddit 언급 수', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        ax.set_title(f'{ticker} - 가격 vs Reddit 언급', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 날짜 포맷
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    # 저장
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'combined_price_reddit_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"💾 통합 차트 저장: {filename}")
    
    plt.show()

def main():
    """메인 실행"""
    
    print("📊 종목별 가격 & Reddit 언급 수 시각화")
    print("=" * 60)
    
    # 개별 차트
    load_and_plot_data()
    
    # 통합 차트
    create_combined_chart()
    
    print("\n✅ 시각화 완료!")

if __name__ == "__main__":
    main()