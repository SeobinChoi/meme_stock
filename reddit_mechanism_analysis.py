#!/usr/bin/env python3
"""
Reddit 언급 수와 가격 간의 음의 상관관계 메커니즘 분석

논문 완성을 위한 고급 분석:
1. 메커니즘 분석 (왜 음의 상관관계가 발생하는가?)
2. Robustness Check
3. 실용적 응용 (투자 전략)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    """데이터 로딩"""
    print("📊 데이터 로딩 중...")
    
    # 훈련 데이터 로딩
    train_df = pd.read_csv('data/colab_datasets/tabular_train_20250814_031335.csv')
    val_df = pd.read_csv('data/colab_datasets/tabular_val_20250814_031335.csv')
    test_df = pd.read_csv('data/colab_datasets/tabular_test_20250814_031335.csv')
    
    # 데이터 합치기
    df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    # 날짜 변환
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"✅ 데이터 로딩 완료: {len(df)} 샘플")
    return df

def analyze_overvaluation_hypothesis(df):
    """과평가 가설 분석: Reddit 언급 증가 = 과평가 신호?"""
    print("\n🔍 과평가 가설 분석...")
    
    # 과평가 지표 계산
    df['price_to_sma_ratio'] = df['price_ratio_sma20']  # 20일 이동평균 대비 가격
    df['rsi_extreme'] = np.where(df['rsi_14'] > 70, 1, 0)  # RSI 과매수
    df['volume_spike'] = np.where(df['volume_ratio'] > 2, 1, 0)  # 거래량 급증
    
    # Reddit 언급 수준별 과평가 지표 분석
    df['mentions_quartile'] = pd.qcut(df['log_mentions'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    
    # 시각화
    plt.figure(figsize=(15, 10))
    
    # 1. 언급 수준별 과평가 지표
    plt.subplot(2, 3, 1)
    overvaluation_by_mentions = df.groupby('mentions_quartile').agg({
        'price_to_sma_ratio': 'mean',
        'rsi_extreme': 'mean',
        'volume_spike': 'mean'
    })
    
    overvaluation_by_mentions.plot(kind='bar', ax=plt.gca())
    plt.title('언급 수준별 과평가 지표')
    plt.xlabel('Reddit 언급 수준 (Quartile)')
    plt.ylabel('과평가 지표 값')
    plt.legend(['가격/SMA 비율', 'RSI 과매수', '거래량 급증'])
    plt.xticks(rotation=45)
    
    # 2. 언급 수 vs 가격/SMA 비율
    plt.subplot(2, 3, 2)
    plt.scatter(df['log_mentions'], df['price_to_sma_ratio'], alpha=0.5, s=1)
    plt.title('Reddit 언급 수 vs 가격/SMA 비율')
    plt.xlabel('Log Mentions')
    plt.ylabel('Price/SMA Ratio')
    
    # 3. 언급 수 vs RSI
    plt.subplot(2, 3, 3)
    plt.scatter(df['log_mentions'], df['rsi_14'], alpha=0.5, s=1)
    plt.axhline(y=70, color='red', linestyle='--', label='RSI 70 (과매수)')
    plt.axhline(y=30, color='green', linestyle='--', label='RSI 30 (과매도)')
    plt.title('Reddit 언급 수 vs RSI')
    plt.xlabel('Log Mentions')
    plt.ylabel('RSI')
    plt.legend()
    
    # 4. 언급 수 vs 거래량 비율
    plt.subplot(2, 3, 4)
    plt.scatter(df['log_mentions'], df['volume_ratio'], alpha=0.5, s=1)
    plt.axhline(y=2, color='red', linestyle='--', label='거래량 급증 (2x)')
    plt.title('Reddit 언급 수 vs 거래량 비율')
    plt.xlabel('Log Mentions')
    plt.ylabel('Volume Ratio')
    plt.legend()
    
    # 5. 과평가 지표 상관관계
    plt.subplot(2, 3, 5)
    overvaluation_corr = df[['log_mentions', 'price_to_sma_ratio', 'rsi_14', 'volume_ratio']].corr()
    sns.heatmap(overvaluation_corr, annot=True, cmap='RdBu_r', center=0, square=True)
    plt.title('과평가 지표 상관관계')
    
    # 6. 결과 요약
    plt.subplot(2, 3, 6)
    plt.axis('off')
    
    # 과평가 가설 검증 결과
    corr_price = df['log_mentions'].corr(df['price_to_sma_ratio'])
    corr_rsi = df['log_mentions'].corr(df['rsi_14'])
    corr_volume = df['log_mentions'].corr(df['volume_ratio'])
    
    summary_text = f"""과평가 가설 검증 결과:
    
언급 수 vs 가격/SMA: {corr_price:.4f}
언급 수 vs RSI: {corr_rsi:.4f}
언급 수 vs 거래량: {corr_volume:.4f}

결론: {'과평가 가설 지지' if corr_price > 0.1 else '과평가 가설 약함'}"""
    
    plt.text(0.1, 0.5, summary_text, fontsize=12, transform=plt.gca().transAxes,
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue'))
    
    plt.tight_layout()
    plt.savefig('overvaluation_hypothesis_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'corr_price_sma': corr_price,
        'corr_rsi': corr_rsi,
        'corr_volume': corr_volume
    }

def analyze_contrarian_trading_hypothesis(df):
    """반대 거래 가설 분석: 전문가들의 역방향 거래?"""
    print("\n🔄 반대 거래 가설 분석...")
    
    # 반대 거래 지표 계산
    df['returns_reversal'] = df.groupby('ticker')['returns_1d'].shift(1)  # 전일 수익률
    df['volume_reversal'] = df.groupby('ticker')['volume_ratio'].shift(1)  # 전일 거래량
    
    # Reddit 언급 증가 후의 가격 반전 패턴
    df['mentions_increase'] = df.groupby('ticker')['log_mentions'].diff()  # 언급 수 변화
    df['price_reversal'] = df.groupby('ticker')['returns_1d'].shift(-1)  # 다음날 수익률
    
    # 시각화
    plt.figure(figsize=(15, 10))
    
    # 1. 언급 수 증가 후 가격 반전
    plt.subplot(2, 3, 1)
    plt.scatter(df['mentions_increase'], df['price_reversal'], alpha=0.5, s=1)
    plt.title('Reddit 언급 증가 vs 다음날 가격 반전')
    plt.xlabel('언급 수 변화')
    plt.ylabel('다음날 수익률')
    
    # 2. 언급 수 증가 구간별 분석
    plt.subplot(2, 3, 2)
    df['mentions_change_group'] = pd.cut(df['mentions_increase'], 
                                        bins=[-np.inf, -0.5, 0, 0.5, np.inf], 
                                        labels=['대폭 감소', '감소', '증가', '대폭 증가'])
    
    reversal_by_group = df.groupby('mentions_change_group')['price_reversal'].agg(['mean', 'std', 'count'])
    reversal_by_group['mean'].plot(kind='bar', ax=plt.gca())
    plt.title('언급 수 변화별 가격 반전 패턴')
    plt.xlabel('언급 수 변화 구간')
    plt.ylabel('평균 다음날 수익률')
    plt.xticks(rotation=45)
    
    # 3. 티커별 반대 거래 패턴
    plt.subplot(2, 3, 3)
    ticker_reversal = df.groupby('ticker').agg({
        'mentions_increase': 'corr',
        'price_reversal': 'corr'
    }).round(4)
    
    plt.bar(ticker_reversal.index, ticker_reversal['mentions_increase'])
    plt.title('티커별: 언급 증가 vs 가격 반전 상관관계')
    plt.xlabel('티커')
    plt.ylabel('상관계수')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 4. 시간별 반대 거래 패턴
    plt.subplot(2, 3, 4)
    df['hour'] = pd.to_datetime(df['date']).dt.hour
    hourly_reversal = df.groupby('hour').agg({
        'mentions_increase': 'corr',
        'price_reversal': 'corr'
    }).round(4)
    
    plt.plot(hourly_reversal.index, hourly_reversal['mentions_increase'], 'o-')
    plt.title('시간별 반대 거래 패턴')
    plt.xlabel('시간')
    plt.ylabel('언급 증가 vs 가격 반전 상관계수')
    plt.grid(True, alpha=0.3)
    
    # 5. 거래량과 반대 거래
    plt.subplot(2, 3, 5)
    plt.scatter(df['volume_ratio'], df['price_reversal'], alpha=0.5, s=1)
    plt.title('거래량 vs 가격 반전')
    plt.xlabel('거래량 비율')
    plt.ylabel('다음날 수익률')
    
    # 6. 결과 요약
    plt.subplot(2, 3, 6)
    plt.axis('off')
    
    # 반대 거래 가설 검증 결과
    overall_corr = df['mentions_increase'].corr(df['price_reversal'])
    ticker_avg_corr = ticker_reversal['mentions_increase'].mean()
    
    summary_text = f"""반대 거래 가설 검증 결과:
    
전체 상관계수: {overall_corr:.4f}
티커별 평균: {ticker_avg_corr:.4f}

결론: {'반대 거래 가설 지지' if overall_corr < -0.05 else '반대 거래 가설 약함'}"""
    
    plt.text(0.1, 0.5, summary_text, fontsize=12, transform=plt.gca().transAxes,
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightgreen'))
    
    plt.tight_layout()
    plt.savefig('contrarian_trading_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'overall_corr': overall_corr,
        'ticker_avg_corr': ticker_avg_corr,
        'ticker_reversal': ticker_reversal
    }

def analyze_regulatory_response_hypothesis(df):
    """규제 대응 가설 분석: 규제 기관의 개입?"""
    print("\n🚨 규제 대응 가설 분석...")
    
    # 규제 대응 지표 계산
    df['volatility_spike'] = np.where(df['vol_5d'] > df['vol_5d'].quantile(0.9), 1, 0)  # 변동성 급증
    df['volume_spike'] = np.where(df['volume_ratio'] > df['volume_ratio'].quantile(0.9), 1, 0)  # 거래량 급증
    
    # Reddit 언급 급증 후 규제 대응 패턴
    df['mentions_extreme'] = np.where(df['log_mentions'] > df['log_mentions'].quantile(0.9), 1, 0)  # 언급 급증
    
    # 시각화
    plt.figure(figsize=(15, 10))
    
    # 1. 언급 급증 후 변동성 변화
    plt.subplot(2, 3, 1)
    df['volatility_next'] = df.groupby('ticker')['vol_5d'].shift(-1)  # 다음날 변동성
    plt.scatter(df['log_mentions'], df['volatility_next'], alpha=0.5, s=1)
    plt.title('Reddit 언급 수 vs 다음날 변동성')
    plt.xlabel('Log Mentions')
    plt.ylabel('다음날 변동성')
    
    # 2. 언급 급증 구간별 변동성 변화
    plt.subplot(2, 3, 2)
    volatility_by_mentions = df.groupby('mentions_extreme')['volatility_next'].agg(['mean', 'std', 'count'])
    volatility_by_mentions['mean'].plot(kind='bar', ax=plt.gca())
    plt.title('언급 급증별 다음날 변동성')
    plt.xlabel('언급 급증 여부')
    plt.ylabel('평균 다음날 변동성')
    plt.xticks([0, 1], ['일반', '급증'])
    
    # 3. 티커별 규제 대응 패턴
    plt.subplot(2, 3, 3)
    ticker_regulatory = df.groupby('ticker').agg({
        'log_mentions': 'corr',
        'volatility_next': 'corr'
    }).round(4)
    
    plt.bar(ticker_regulatory.index, ticker_regulatory['log_mentions'])
    plt.title('티커별: 언급 수 vs 다음날 변동성 상관관계')
    plt.xlabel('티커')
    plt.ylabel('상관계수')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 4. 시간별 규제 대응 패턴
    plt.subplot(2, 3, 4)
    df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
    weekly_regulatory = df.groupby('day_of_week').agg({
        'log_mentions': 'corr',
        'volatility_next': 'corr'
    }).round(4)
    
    day_names = ['월', '화', '수', '목', '금', '토', '일']
    plt.plot(range(len(weekly_regulatory)), weekly_regulatory['log_mentions'], 'o-')
    plt.title('요일별 규제 대응 패턴')
    plt.xlabel('요일')
    plt.ylabel('언급 수 vs 다음날 변동성 상관계수')
    plt.xticks(range(len(weekly_regulatory)), day_names)
    plt.grid(True, alpha=0.3)
    
    # 5. 거래량과 규제 대응
    plt.subplot(2, 3, 5)
    plt.scatter(df['volume_ratio'], df['volatility_next'], alpha=0.5, s=1)
    plt.title('거래량 vs 다음날 변동성')
    plt.xlabel('거래량 비율')
    plt.ylabel('다음날 변동성')
    
    # 6. 결과 요약
    plt.subplot(2, 3, 6)
    plt.axis('off')
    
    # 규제 대응 가설 검증 결과
    overall_corr = df['log_mentions'].corr(df['volatility_next'])
    ticker_avg_corr = ticker_regulatory['log_mentions'].mean()
    
    summary_text = f"""규제 대응 가설 검증 결과:
    
전체 상관계수: {overall_corr:.4f}
티커별 평균: {ticker_avg_corr:.4f}

결론: {'규제 대응 가설 지지' if overall_corr > 0.05 else '규제 대응 가설 약함'}"""
    
    plt.text(0.1, 0.5, summary_text, fontsize=12, transform=plt.gca().transAxes,
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightcoral'))
    
    plt.tight_layout()
    plt.savefig('regulatory_response_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'overall_corr': overall_corr,
        'ticker_avg_corr': ticker_avg_corr,
        'ticker_regulatory': ticker_regulatory
    }

def develop_trading_strategy(df):
    """실용적 응용: 역방향 신호 기반 투자 전략"""
    print("\n💰 역방향 신호 기반 투자 전략 개발...")
    
    # 투자 전략 시뮬레이션
    df['strategy_signal'] = np.where(df['log_mentions'] > df['log_mentions'].quantile(0.8), -1, 0)  # 언급 급증 시 매도 신호
    df['strategy_signal'] = np.where(df['log_mentions'] < df['log_mentions'].quantile(0.2), 1, df['strategy_signal'])  # 언급 감소 시 매수 신호
    
    # 전략 수익률 계산
    df['strategy_returns'] = df['strategy_signal'] * df['returns_1d']
    
    # 시각화
    plt.figure(figsize=(15, 10))
    
    # 1. 전략 신호 분포
    plt.subplot(2, 3, 1)
    signal_counts = df['strategy_signal'].value_counts().sort_index()
    plt.bar(['매도(-1)', '관망(0)', '매수(1)'], signal_counts.values)
    plt.title('전략 신호 분포')
    plt.ylabel('신호 수')
    
    # 2. 신호별 수익률
    plt.subplot(2, 3, 2)
    returns_by_signal = df.groupby('strategy_signal')['returns_1d'].agg(['mean', 'std', 'count'])
    returns_by_signal['mean'].plot(kind='bar', ax=plt.gca())
    plt.title('신호별 평균 수익률')
    plt.xlabel('전략 신호')
    plt.ylabel('평균 수익률')
    plt.xticks([-1, 0, 1], ['매도', '관망', '매수'])
    
    # 3. 누적 수익률 비교
    plt.subplot(2, 3, 3)
    df['cumulative_returns'] = df.groupby('ticker')['returns_1d'].cumsum()
    df['cumulative_strategy'] = df.groupby('ticker')['strategy_returns'].cumsum()
    
    # 전체 평균
    daily_avg_returns = df.groupby('date')[['cumulative_returns', 'cumulative_strategy']].mean()
    plt.plot(daily_avg_returns.index, daily_avg_returns['cumulative_returns'], label='Buy & Hold', alpha=0.7)
    plt.plot(daily_avg_returns.index, daily_avg_returns['cumulative_strategy'], label='Reddit Strategy', alpha=0.7)
    plt.title('누적 수익률 비교')
    plt.xlabel('날짜')
    plt.ylabel('누적 수익률')
    plt.legend()
    plt.xticks(rotation=45)
    
    # 4. 티커별 전략 성과
    plt.subplot(2, 3, 4)
    ticker_performance = df.groupby('ticker').agg({
        'returns_1d': 'mean',
        'strategy_returns': 'mean'
    }).round(4)
    
    x = np.arange(len(ticker_performance))
    width = 0.35
    
    plt.bar(x - width/2, ticker_performance['returns_1d'], width, label='Buy & Hold', alpha=0.8)
    plt.bar(x + width/2, ticker_performance['strategy_returns'], width, label='Reddit Strategy', alpha=0.8)
    
    plt.title('티커별 전략 성과 비교')
    plt.xlabel('티커')
    plt.ylabel('평균 수익률')
    plt.xticks(x, ticker_performance.index)
    plt.legend()
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 5. 리스크-수익률 분석
    plt.subplot(2, 3, 5)
    risk_return = df.groupby('ticker').agg({
        'returns_1d': ['mean', 'std'],
        'strategy_returns': ['mean', 'std']
    }).round(4)
    
    # Buy & Hold
    plt.scatter(risk_return[('returns_1d', 'std')], risk_return[('returns_1d', 'mean')], 
                label='Buy & Hold', s=100, alpha=0.7)
    
    # Reddit Strategy
    plt.scatter(risk_return[('strategy_returns', 'std')], risk_return[('strategy_returns', 'mean')], 
                label='Reddit Strategy', s=100, alpha=0.7)
    
    plt.title('리스크-수익률 분석')
    plt.xlabel('표준편차 (리스크)')
    plt.ylabel('평균 수익률')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. 전략 성과 요약
    plt.subplot(2, 3, 6)
    plt.axis('off')
    
    # 전략 성과 계산
    total_strategy_return = df['strategy_returns'].sum()
    total_buyhold_return = df['returns_1d'].sum()
    strategy_improvement = total_strategy_return - total_buyhold_return
    
    # Sharpe Ratio 계산
    strategy_sharpe = df['strategy_returns'].mean() / df['strategy_returns'].std() * np.sqrt(252)
    buyhold_sharpe = df['returns_1d'].mean() / df['returns_1d'].std() * np.sqrt(252)
    
    summary_text = f"""전략 성과 요약:

총 수익률:
- Buy & Hold: {total_buyhold_return:.4f}
- Reddit Strategy: {total_strategy_return:.4f}
- 개선도: {strategy_improvement:.4f}

Sharpe Ratio:
- Buy & Hold: {buyhold_sharpe:.4f}
- Reddit Strategy: {strategy_sharpe:.4f}

결론: {'전략 성과 우수' if strategy_improvement > 0 else '전략 성과 열악'}"""
    
    plt.text(0.1, 0.5, summary_text, fontsize=10, transform=plt.gca().transAxes,
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightyellow'))
    
    plt.tight_layout()
    plt.savefig('trading_strategy_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'total_strategy_return': total_strategy_return,
        'total_buyhold_return': total_buyhold_return,
        'strategy_improvement': strategy_improvement,
        'strategy_sharpe': strategy_sharpe,
        'buyhold_sharpe': buyhold_sharpe
    }

def main():
    """메인 실행 함수"""
    print("🚀 Reddit 언급 수와 가격 상관관계 메커니즘 분석 시작")
    print("=" * 80)
    
    # 1. 데이터 로딩
    df = load_data()
    
    # 2. 과평가 가설 분석
    overvaluation_results = analyze_overvaluation_hypothesis(df)
    
    # 3. 반대 거래 가설 분석
    contrarian_results = analyze_contrarian_trading_hypothesis(df)
    
    # 4. 규제 대응 가설 분석
    regulatory_results = analyze_regulatory_response_hypothesis(df)
    
    # 5. 투자 전략 개발
    strategy_results = develop_trading_strategy(df)
    
    # 6. 종합 결과 요약
    print("\n📊 메커니즘 분석 결과 요약")
    print("=" * 80)
    
    print("🔍 과평가 가설:")
    print(f"  - 언급 수 vs 가격/SMA: {overvaluation_results['corr_price_sma']:.4f}")
    print(f"  - 언급 수 vs RSI: {overvaluation_results['corr_rsi']:.4f}")
    print(f"  - 언급 수 vs 거래량: {overvaluation_results['corr_volume']:.4f}")
    
    print("\n🔄 반대 거래 가설:")
    print(f"  - 전체 상관계수: {contrarian_results['overall_corr']:.4f}")
    print(f"  - 티커별 평균: {contrarian_results['ticker_avg_corr']:.4f}")
    
    print("\n🚨 규제 대응 가설:")
    print(f"  - 전체 상관계수: {regulatory_results['overall_corr']:.4f}")
    print(f"  - 티커별 평균: {regulatory_results['ticker_avg_corr']:.4f}")
    
    print("\n💰 투자 전략 성과:")
    print(f"  - Buy & Hold 수익률: {strategy_results['total_buyhold_return']:.4f}")
    print(f"  - Reddit Strategy 수익률: {strategy_results['total_strategy_return']:.4f}")
    print(f"  - 전략 개선도: {strategy_results['strategy_improvement']:.4f}")
    print(f"  - Reddit Strategy Sharpe: {strategy_results['strategy_sharpe']:.4f}")
    
    print(f"\n✅ 메커니즘 분석 완료! 시각화 파일들이 저장되었습니다.")
    print("   - overvaluation_hypothesis_analysis.png")
    print("   - contrarian_trading_analysis.png")
    print("   - regulatory_response_analysis.png")
    print("   - trading_strategy_analysis.png")

if __name__ == "__main__":
    main()
