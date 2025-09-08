#!/usr/bin/env python3
"""
Meme Stocks Reddit-Price Correlation Analysis
AMC, BB, GME 종목의 Reddit 반응과 주가 상관관계 분석
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class MemeStocksCorrelationAnalysis:
    """밈스톡 Reddit-주가 상관관계 분석 클래스"""
    
    def __init__(self):
        self.df = None
        self.correlation_results = {}
        
    def load_data(self):
        """데이터 로드"""
        print("📊 Loading meme stocks data for correlation analysis...")
        
        # 통합 데이터셋 로드
        train_df = pd.read_csv('data/colab_datasets/tabular_train_20250814_031335.csv')
        val_df = pd.read_csv('data/colab_datasets/tabular_val_20250814_031335.csv')
        test_df = pd.read_csv('data/colab_datasets/tabular_test_20250814_031335.csv')
        
        # 데이터 통합
        df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
        
        # AMC, BB, GME만 필터링
        meme_stocks = ['AMC', 'BB', 'GME']
        df = df[df['ticker'].isin(meme_stocks)].copy()
        
        print(f"   ✅ Total data: {len(df)} records")
        print(f"   ✅ Date range: {df['date'].min()} ~ {df['date'].max()}")
        print(f"   ✅ Tickers: {df['ticker'].unique()}")
        
        # 종목별 데이터 수
        for ticker in meme_stocks:
            count = len(df[df['ticker'] == ticker])
            print(f"   📊 {ticker}: {count} records")
        
        self.df = df
        return df
    
    def calculate_correlations(self):
        """상관관계 계산"""
        print("🔍 Calculating Reddit-Price correlations...")
        
        correlation_results = {}
        
        for ticker in ['AMC', 'BB', 'GME']:
            print(f"   📈 Analyzing {ticker}...")
            
            ticker_data = self.df[self.df['ticker'] == ticker].copy()
            ticker_data = ticker_data.sort_values('date').reset_index(drop=True)
            
            # 기본 통계
            basic_stats = {
                'total_records': len(ticker_data),
                'date_range': f"{ticker_data['date'].min()} ~ {ticker_data['date'].max()}",
                'avg_log_mentions': ticker_data['log_mentions'].mean(),
                'avg_returns_1d': ticker_data['returns_1d'].mean(),
                'std_log_mentions': ticker_data['log_mentions'].std(),
                'std_returns_1d': ticker_data['returns_1d'].std()
            }
            
            # 상관관계 계산
            correlations = {}
            
            # 1. 동일 시점 상관관계
            corr_same_day, p_same_day = pearsonr(ticker_data['log_mentions'], ticker_data['returns_1d'])
            corr_spearman_same, p_spearman_same = spearmanr(ticker_data['log_mentions'], ticker_data['returns_1d'])
            
            correlations['same_day'] = {
                'pearson': {'correlation': corr_same_day, 'p_value': p_same_day},
                'spearman': {'correlation': corr_spearman_same, 'p_value': p_spearman_same}
            }
            
            # 2. Reddit -> 주가 (Reddit가 선행)
            corr_lead1, p_lead1 = pearsonr(ticker_data['log_mentions'].shift(1), ticker_data['returns_1d'])
            corr_spearman_lead1, p_spearman_lead1 = spearmanr(ticker_data['log_mentions'].shift(1), ticker_data['returns_1d'])
            
            correlations['reddit_lead_1d'] = {
                'pearson': {'correlation': corr_lead1, 'p_value': p_lead1},
                'spearman': {'correlation': corr_spearman_lead1, 'p_value': p_spearman_lead1}
            }
            
            # 3. 주가 -> Reddit (주가가 선행)
            corr_lag1, p_lag1 = pearsonr(ticker_data['log_mentions'], ticker_data['returns_1d'].shift(1))
            corr_spearman_lag1, p_spearman_lag1 = spearmanr(ticker_data['log_mentions'], ticker_data['returns_1d'].shift(1))
            
            correlations['price_lead_1d'] = {
                'pearson': {'correlation': corr_lag1, 'p_value': p_lag1},
                'spearman': {'correlation': corr_spearman_lag1, 'p_value': p_spearman_lag1}
            }
            
            # 4. 다양한 지연 시간별 상관관계
            lag_correlations = {}
            for lag in range(1, 6):  # 1일~5일 지연
                # Reddit -> 주가
                reddit_lead_corr, reddit_lead_p = pearsonr(ticker_data['log_mentions'].shift(lag), ticker_data['returns_1d'])
                # 주가 -> Reddit
                price_lead_corr, price_lead_p = pearsonr(ticker_data['log_mentions'], ticker_data['returns_1d'].shift(lag))
                
                lag_correlations[f'lag_{lag}d'] = {
                    'reddit_lead': {'correlation': reddit_lead_corr, 'p_value': reddit_lead_p},
                    'price_lead': {'correlation': price_lead_corr, 'p_value': price_lead_p}
                }
            
            correlations['lag_analysis'] = lag_correlations
            
            # 5. Reddit 피처별 상관관계
            reddit_features = [
                'reddit_ema_3', 'reddit_ema_5', 'reddit_ema_10',
                'reddit_surprise', 'reddit_market_ex', 'reddit_spike_p95',
                'reddit_momentum_3', 'reddit_momentum_7', 'reddit_momentum_14',
                'reddit_vol_5', 'reddit_vol_10', 'reddit_percentile',
                'market_sentiment', 'price_reddit_momentum', 'vol_reddit_attention'
            ]
            
            feature_correlations = {}
            for feature in reddit_features:
                if feature in ticker_data.columns:
                    corr, p_val = pearsonr(ticker_data[feature].fillna(0), ticker_data['returns_1d'])
                    feature_correlations[feature] = {
                        'correlation': corr,
                        'p_value': p_val,
                        'abs_correlation': abs(corr)
                    }
            
            correlations['reddit_features'] = feature_correlations
            
            # 6. 시계열 분석 (월별, 분기별)
            ticker_data['year_month'] = ticker_data['date'].dt.to_period('M')
            monthly_correlations = {}
            
            for period in ticker_data['year_month'].unique():
                period_data = ticker_data[ticker_data['year_month'] == period]
                if len(period_data) > 5:  # 최소 5개 데이터 포인트
                    corr, p_val = pearsonr(period_data['log_mentions'], period_data['returns_1d'])
                    monthly_correlations[str(period)] = {
                        'correlation': corr,
                        'p_value': p_val,
                        'records': len(period_data)
                    }
            
            correlations['monthly'] = monthly_correlations
            
            correlation_results[ticker] = {
                'basic_stats': basic_stats,
                'correlations': correlations
            }
            
            print(f"      ✅ {ticker} analysis completed")
        
        self.correlation_results = correlation_results
        return correlation_results
    
    def create_correlation_summary_table(self):
        """상관관계 요약 표 생성"""
        print("📋 Creating correlation summary table...")
        
        summary_data = []
        
        for ticker in ['AMC', 'BB', 'GME']:
            result = self.correlation_results[ticker]
            correlations = result['correlations']
            
            # 주요 상관관계 추출
            same_day_corr = correlations['same_day']['pearson']['correlation']
            same_day_p = correlations['same_day']['pearson']['p_value']
            
            reddit_lead_corr = correlations['reddit_lead_1d']['pearson']['correlation']
            reddit_lead_p = correlations['reddit_lead_1d']['pearson']['p_value']
            
            price_lead_corr = correlations['price_lead_1d']['pearson']['correlation']
            price_lead_p = correlations['price_lead_1d']['pearson']['p_value']
            
            # 기본 통계
            avg_mentions = result['basic_stats']['avg_log_mentions']
            avg_returns = result['basic_stats']['avg_returns_1d']
            
            summary_data.append({
                'Ticker': ticker,
                'Avg_Log_Mentions': f"{avg_mentions:.4f}",
                'Avg_Daily_Returns': f"{avg_returns:.4f}",
                'Same_Day_Corr': f"{same_day_corr:.4f}",
                'Same_Day_P': f"{same_day_p:.4f}",
                'Reddit_Lead_Corr': f"{reddit_lead_corr:.4f}",
                'Reddit_Lead_P': f"{reddit_lead_p:.4f}",
                'Price_Lead_Corr': f"{price_lead_corr:.4f}",
                'Price_Lead_P': f"{price_lead_p:.4f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        print("\n" + "="*120)
        print("MEME STOCKS REDDIT-PRICE CORRELATION SUMMARY")
        print("="*120)
        print(summary_df.to_string(index=False))
        print("="*120)
        
        return summary_df
    
    def create_detailed_correlation_analysis(self):
        """상세 상관관계 분석"""
        print("🔍 Creating detailed correlation analysis...")
        
        detailed_results = []
        
        for ticker in ['AMC', 'BB', 'GME']:
            result = self.correlation_results[ticker]
            correlations = result['correlations']
            
            print(f"\n📊 {ticker} DETAILED CORRELATION ANALYSIS")
            print("-" * 60)
            
            # 1. 동일 시점 상관관계
            same_day = correlations['same_day']
            print(f"Same Day Correlation:")
            print(f"  Pearson: {same_day['pearson']['correlation']:.4f} (p={same_day['pearson']['p_value']:.4f})")
            print(f"  Spearman: {same_day['spearman']['correlation']:.4f} (p={same_day['spearman']['p_value']:.4f})")
            
            # 2. 지연 상관관계
            print(f"\nLag Analysis (Reddit -> Price):")
            for lag_key, lag_data in correlations['lag_analysis'].items():
                reddit_lead = lag_data['reddit_lead']
                print(f"  {lag_key}: {reddit_lead['correlation']:.4f} (p={reddit_lead['p_value']:.4f})")
            
            # 3. Reddit 피처별 상관관계 (상위 5개)
            feature_corrs = correlations['reddit_features']
            sorted_features = sorted(feature_corrs.items(), key=lambda x: x[1]['abs_correlation'], reverse=True)
            
            print(f"\nTop Reddit Features by Correlation:")
            for i, (feature, corr_data) in enumerate(sorted_features[:5]):
                print(f"  {i+1}. {feature}: {corr_data['correlation']:.4f} (p={corr_data['p_value']:.4f})")
            
            # 4. 월별 상관관계 요약
            monthly_corrs = correlations['monthly']
            monthly_values = [data['correlation'] for data in monthly_corrs.values()]
            if monthly_values:
                print(f"\nMonthly Correlation Summary:")
                print(f"  Mean: {np.mean(monthly_values):.4f}")
                print(f"  Std: {np.std(monthly_values):.4f}")
                print(f"  Positive months: {sum(1 for x in monthly_values if x > 0)}/{len(monthly_values)}")
            
            detailed_results.append({
                'ticker': ticker,
                'same_day_corr': same_day['pearson']['correlation'],
                'reddit_lead_1d': correlations['reddit_lead_1d']['pearson']['correlation'],
                'top_feature': sorted_features[0][0] if sorted_features else 'N/A',
                'top_feature_corr': sorted_features[0][1]['correlation'] if sorted_features else 0,
                'monthly_corr_mean': np.mean(monthly_values) if monthly_values else 0
            })
        
        return detailed_results
    
    def create_correlation_visualization(self):
        """상관관계 시각화"""
        print("📈 Creating correlation visualizations...")
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 18))
        fig.suptitle('Meme Stocks Reddit-Price Correlation Analysis', fontsize=16, fontweight='bold')
        
        tickers = ['AMC', 'BB', 'GME']
        
        for i, ticker in enumerate(tickers):
            ticker_data = self.df[self.df['ticker'] == ticker].copy()
            ticker_data = ticker_data.sort_values('date').reset_index(drop=True)
            
            # 1. 산점도 (동일 시점)
            ax1 = axes[i, 0]
            ax1.scatter(ticker_data['log_mentions'], ticker_data['returns_1d'], 
                       alpha=0.6, s=20, color='blue')
            
            # 상관관계 선 추가
            z = np.polyfit(ticker_data['log_mentions'], ticker_data['returns_1d'], 1)
            p = np.poly1d(z)
            ax1.plot(ticker_data['log_mentions'], p(ticker_data['log_mentions']), 
                    "r--", alpha=0.8, linewidth=2)
            
            corr = self.correlation_results[ticker]['correlations']['same_day']['pearson']['correlation']
            ax1.set_title(f'{ticker} - Same Day Correlation: {corr:.4f}', fontweight='bold')
            ax1.set_xlabel('Log Mentions')
            ax1.set_ylabel('Daily Returns')
            ax1.grid(True, alpha=0.3)
            
            # 2. 시계열 플롯
            ax2 = axes[i, 1]
            ax2_twin = ax2.twinx()
            
            # 수익률 플롯 (왼쪽 y축)
            line1 = ax2.plot(ticker_data['date'], ticker_data['returns_1d'], 
                           color='blue', linewidth=1, label='Daily Returns', alpha=0.7)
            ax2.set_ylabel('Daily Returns', color='blue')
            ax2.tick_params(axis='y', labelcolor='blue')
            
            # Reddit 관심도 플롯 (오른쪽 y축)
            line2 = ax2_twin.plot(ticker_data['date'], ticker_data['log_mentions'], 
                                color='red', linewidth=1, label='Log Mentions', alpha=0.7)
            ax2_twin.set_ylabel('Log Mentions', color='red')
            ax2_twin.tick_params(axis='y', labelcolor='red')
            
            ax2.set_title(f'{ticker} - Time Series', fontweight='bold')
            ax2.set_xlabel('Date')
            
            # 범례
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax2.legend(lines, labels, loc='upper left')
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='x', rotation=45)
            
            # 3. 지연 상관관계 플롯
            ax3 = axes[i, 2]
            lag_correlations = self.correlation_results[ticker]['correlations']['lag_analysis']
            
            lags = list(range(1, 6))
            reddit_lead_corrs = [lag_correlations[f'lag_{lag}d']['reddit_lead']['correlation'] for lag in lags]
            price_lead_corrs = [lag_correlations[f'lag_{lag}d']['price_lead']['correlation'] for lag in lags]
            
            ax3.plot(lags, reddit_lead_corrs, 'o-', label='Reddit → Price', linewidth=2, markersize=6)
            ax3.plot(lags, price_lead_corrs, 's-', label='Price → Reddit', linewidth=2, markersize=6)
            
            ax3.set_title(f'{ticker} - Lag Correlations', fontweight='bold')
            ax3.set_xlabel('Lag (Days)')
            ax3.set_ylabel('Correlation')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.axhline(0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/meme_stocks_correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("   ✅ Correlation visualization saved to results/meme_stocks_correlation_analysis.png")
    
    def create_reddit_features_heatmap(self):
        """Reddit 피처별 상관관계 히트맵"""
        print("🔥 Creating Reddit features correlation heatmap...")
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle('Reddit Features Correlation with Returns by Stock', fontsize=16, fontweight='bold')
        
        tickers = ['AMC', 'BB', 'GME']
        
        for i, ticker in enumerate(tickers):
            ax = axes[i]
            
            # Reddit 피처별 상관관계 추출
            feature_corrs = self.correlation_results[ticker]['correlations']['reddit_features']
            
            # 데이터 준비
            features = list(feature_corrs.keys())
            correlations = [feature_corrs[feature]['correlation'] for feature in features]
            
            # 히트맵 데이터 생성
            corr_matrix = np.array(correlations).reshape(-1, 1)
            
            # 히트맵 그리기
            im = ax.imshow(corr_matrix.T, cmap='RdBu_r', aspect='auto', vmin=-0.3, vmax=0.3)
            
            # 축 설정
            ax.set_xticks(range(len(features)))
            ax.set_xticklabels(features, rotation=45, ha='right')
            ax.set_yticks([0])
            ax.set_yticklabels(['Correlation'])
            ax.set_title(f'{ticker} - Reddit Features vs Returns', fontweight='bold')
            
            # 값 표시
            for j, corr in enumerate(correlations):
                ax.text(j, 0, f'{corr:.3f}', ha='center', va='center', 
                       color='white' if abs(corr) > 0.15 else 'black', fontweight='bold')
        
        # 컬러바 추가
        cbar = fig.colorbar(im, ax=axes, orientation='horizontal', pad=0.1)
        cbar.set_label('Correlation with Returns', fontsize=12)
        
        plt.tight_layout()
        plt.savefig('results/reddit_features_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("   ✅ Reddit features heatmap saved to results/reddit_features_correlation_heatmap.png")
    
    def generate_correlation_report(self, summary_df, detailed_results):
        """상관관계 분석 리포트 생성"""
        print("📝 Generating correlation analysis report...")
        
        report = []
        report.append("=" * 120)
        report.append("MEME STOCKS REDDIT-PRICE CORRELATION ANALYSIS REPORT")
        report.append("=" * 120)
        report.append("")
        report.append("Analysis Overview:")
        report.append("- Target Stocks: AMC, BB, GME (Meme Stocks)")
        report.append("- Analysis Period: 2021-2023")
        report.append("- Correlation Methods: Pearson, Spearman")
        report.append("- Lag Analysis: 1-5 days")
        report.append("")
        
        # 요약 표
        report.append("CORRELATION SUMMARY TABLE")
        report.append("-" * 50)
        report.append(summary_df.to_string(index=False))
        report.append("")
        
        # 종목별 상세 분석
        report.append("STOCK-SPECIFIC ANALYSIS")
        report.append("-" * 50)
        
        for ticker in ['AMC', 'BB', 'GME']:
            result = self.correlation_results[ticker]
            correlations = result['correlations']
            
            report.append(f"\n{ticker} ANALYSIS:")
            report.append(f"  Total Records: {result['basic_stats']['total_records']}")
            report.append(f"  Date Range: {result['basic_stats']['date_range']}")
            report.append(f"  Average Log Mentions: {result['basic_stats']['avg_log_mentions']:.4f}")
            report.append(f"  Average Daily Returns: {result['basic_stats']['avg_returns_1d']:.4f}")
            
            # 주요 상관관계
            same_day_corr = correlations['same_day']['pearson']['correlation']
            same_day_p = correlations['same_day']['pearson']['p_value']
            reddit_lead_corr = correlations['reddit_lead_1d']['pearson']['correlation']
            reddit_lead_p = correlations['reddit_lead_1d']['pearson']['p_value']
            
            report.append(f"  Same Day Correlation: {same_day_corr:.4f} (p={same_day_p:.4f})")
            report.append(f"  Reddit Lead (1d) Correlation: {reddit_lead_corr:.4f} (p={reddit_lead_p:.4f})")
            
            # 상관관계 해석
            if abs(same_day_corr) > 0.1 and same_day_p < 0.05:
                report.append(f"  → Significant same-day correlation detected")
            else:
                report.append(f"  → No significant same-day correlation")
            
            if abs(reddit_lead_corr) > 0.1 and reddit_lead_p < 0.05:
                report.append(f"  → Significant Reddit lead effect detected")
            else:
                report.append(f"  → No significant Reddit lead effect")
        
        # 전체 결론
        report.append("\nOVERALL CONCLUSIONS")
        report.append("-" * 50)
        
        # 평균 상관관계 계산
        avg_same_day_corr = np.mean([self.correlation_results[ticker]['correlations']['same_day']['pearson']['correlation'] 
                                   for ticker in ['AMC', 'BB', 'GME']])
        avg_reddit_lead_corr = np.mean([self.correlation_results[ticker]['correlations']['reddit_lead_1d']['pearson']['correlation'] 
                                      for ticker in ['AMC', 'BB', 'GME']])
        
        report.append(f"Average Same Day Correlation: {avg_same_day_corr:.4f}")
        report.append(f"Average Reddit Lead Correlation: {avg_reddit_lead_corr:.4f}")
        report.append("")
        
        # 상관관계 강도 해석
        if abs(avg_same_day_corr) > 0.1:
            report.append("✅ Moderate to strong same-day correlation detected")
        elif abs(avg_same_day_corr) > 0.05:
            report.append("⚠️ Weak same-day correlation detected")
        else:
            report.append("❌ No meaningful same-day correlation")
        
        if abs(avg_reddit_lead_corr) > 0.1:
            report.append("✅ Moderate to strong Reddit lead effect detected")
        elif abs(avg_reddit_lead_corr) > 0.05:
            report.append("⚠️ Weak Reddit lead effect detected")
        else:
            report.append("❌ No meaningful Reddit lead effect")
        
        report.append("")
        
        # 투자 전략 시사점
        report.append("INVESTMENT STRATEGY IMPLICATIONS")
        report.append("-" * 50)
        
        if avg_reddit_lead_corr > 0.05:
            report.append("📈 Positive Reddit lead effect suggests:")
            report.append("  - High Reddit attention may predict positive returns")
            report.append("  - Social media sentiment can be used as leading indicator")
            report.append("  - Contrarian strategy may be less effective")
        elif avg_reddit_lead_corr < -0.05:
            report.append("📉 Negative Reddit lead effect suggests:")
            report.append("  - High Reddit attention may predict negative returns")
            report.append("  - Contrarian strategy may be effective")
            report.append("  - Social media hype may signal overvaluation")
        else:
            report.append("📊 Neutral Reddit effect suggests:")
            report.append("  - Reddit attention has limited predictive power")
            report.append("  - Other factors may be more important")
            report.append("  - Traditional analysis methods may be more reliable")
        
        report_text = "\n".join(report)
        
        # 파일로 저장
        with open('results/meme_stocks_correlation_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print("   ✅ Correlation report saved to results/meme_stocks_correlation_report.txt")
        print("\n" + report_text)
        
        return report_text

def main():
    """메인 실행 함수"""
    print("🚀 Starting Meme Stocks Reddit-Price Correlation Analysis")
    print("=" * 80)
    
    # 결과 디렉토리 생성
    import os
    os.makedirs('results', exist_ok=True)
    
    # 분석 초기화
    analysis = MemeStocksCorrelationAnalysis()
    
    # 1. 데이터 로드
    df = analysis.load_data()
    
    # 2. 상관관계 계산
    correlation_results = analysis.calculate_correlations()
    
    # 3. 요약 표 생성
    print("\n" + "="*50)
    print("CORRELATION SUMMARY")
    print("="*50)
    summary_df = analysis.create_correlation_summary_table()
    
    # 4. 상세 분석
    print("\n" + "="*50)
    print("DETAILED CORRELATION ANALYSIS")
    print("="*50)
    detailed_results = analysis.create_detailed_correlation_analysis()
    
    # 5. 시각화
    print("\n" + "="*50)
    print("VISUALIZATION")
    print("="*50)
    analysis.create_correlation_visualization()
    analysis.create_reddit_features_heatmap()
    
    # 6. 최종 리포트 생성
    print("\n" + "="*50)
    print("FINAL REPORT GENERATION")
    print("="*50)
    analysis.generate_correlation_report(summary_df, detailed_results)
    
    print("\n🎉 Meme stocks correlation analysis completed!")
    print("📁 Results saved in 'results/' directory")
    
    return analysis, summary_df, detailed_results

if __name__ == "__main__":
    analysis, summary_df, detailed_results = main()
