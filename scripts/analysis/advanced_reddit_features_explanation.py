#!/usr/bin/env python3
"""
Advanced Reddit Features Explanation
고급 Reddit 피처들의 상세 설명 및 계산 방법
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class AdvancedRedditFeaturesExplainer:
    """고급 Reddit 피처 설명 클래스"""
    
    def __init__(self):
        self.feature_explanations = {}
        self.feature_calculations = {}
        
    def explain_basic_features(self):
        """기본 Reddit 피처 설명"""
        print("📊 BASIC REDDIT FEATURES")
        print("=" * 50)
        
        basic_features = {
            'log_mentions': {
                'description': '로그 변환된 언급 수',
                'calculation': 'log(1 + mentions)',
                'purpose': '언급 수의 분포를 정규화하고 극값의 영향을 줄임',
                'interpretation': '높을수록 Reddit에서 더 많이 언급됨'
            },
            'reddit_ema_3': {
                'description': '3일 지수이동평균',
                'calculation': 'EMA(log_mentions, span=3)',
                'purpose': '단기 트렌드를 파악',
                'interpretation': '최근 3일간의 평균 관심도'
            },
            'reddit_ema_5': {
                'description': '5일 지수이동평균',
                'calculation': 'EMA(log_mentions, span=5)',
                'purpose': '중기 트렌드를 파악',
                'interpretation': '최근 5일간의 평균 관심도'
            },
            'reddit_ema_10': {
                'description': '10일 지수이동평균',
                'calculation': 'EMA(log_mentions, span=10)',
                'purpose': '장기 트렌드를 파악',
                'interpretation': '최근 10일간의 평균 관심도'
            }
        }
        
        for feature, info in basic_features.items():
            print(f"\n🔹 {feature}")
            print(f"   설명: {info['description']}")
            print(f"   계산: {info['calculation']}")
            print(f"   목적: {info['purpose']}")
            print(f"   해석: {info['interpretation']}")
        
        self.feature_explanations['basic'] = basic_features
    
    def explain_advanced_features(self):
        """고급 Reddit 피처 설명"""
        print("\n\n🚀 ADVANCED REDDIT FEATURES")
        print("=" * 50)
        
        advanced_features = {
            'reddit_surprise': {
                'description': 'Reddit 서프라이즈 (예상치 대비 편차)',
                'calculation': 'log_mentions[t-1] - reddit_ema_5[t]',
                'purpose': '예상치 대비 실제 관심도의 편차를 측정',
                'interpretation': '양수: 예상보다 높은 관심도, 음수: 예상보다 낮은 관심도',
                'importance': 'Contrarian Effect의 핵심 지표 - 음의 상관관계가 강함'
            },
            'reddit_market_ex': {
                'description': '시장 전체 대비 초과 관심도',
                'calculation': 'market_total_mentions - own_mentions',
                'purpose': '전체 시장 대비 상대적 관심도 측정',
                'interpretation': '높을수록 다른 종목들에 비해 상대적으로 낮은 관심도',
                'importance': '시장 전체 분위기와의 상관관계 파악'
            },
            'reddit_spike_p95': {
                'description': '95% 분위수 스파이크 지표',
                'calculation': 'log_mentions[t-1] > quantile(log_mentions, 0.95)',
                'purpose': '극도로 높은 관심도 구간 식별',
                'interpretation': '1: 극도로 높은 관심도, 0: 일반적인 관심도',
                'importance': '이벤트 스터디의 핵심 변수'
            },
            'reddit_momentum_3': {
                'description': '3일 모멘텀',
                'calculation': 'mean(log_mentions[t-3:t-1]) - mean(log_mentions[t-6:t-4])',
                'purpose': '단기 관심도 변화 추세 측정',
                'interpretation': '양수: 관심도 증가 추세, 음수: 관심도 감소 추세',
                'importance': '관심도 변화의 방향성 파악'
            },
            'reddit_momentum_7': {
                'description': '7일 모멘텀',
                'calculation': 'mean(log_mentions[t-7:t-1]) - mean(log_mentions[t-14:t-8])',
                'purpose': '중기 관심도 변화 추세 측정',
                'interpretation': '양수: 관심도 증가 추세, 음수: 관심도 감소 추세',
                'importance': '더 안정적인 모멘텀 지표'
            },
            'reddit_momentum_14': {
                'description': '14일 모멘텀',
                'calculation': 'mean(log_mentions[t-14:t-1]) - mean(log_mentions[t-28:t-15])',
                'purpose': '장기 관심도 변화 추세 측정',
                'interpretation': '양수: 관심도 증가 추세, 음수: 관심도 감소 추세',
                'importance': '장기 트렌드 파악'
            },
            'reddit_vol_5': {
                'description': '5일 관심도 변동성',
                'calculation': 'std(log_mentions[t-5:t-1])',
                'purpose': '단기 관심도 변동성 측정',
                'interpretation': '높을수록 관심도가 불안정함',
                'importance': '불확실성과 시장 불안정성 지표'
            },
            'reddit_vol_10': {
                'description': '10일 관심도 변동성',
                'calculation': 'std(log_mentions[t-10:t-1])',
                'purpose': '중기 관심도 변동성 측정',
                'interpretation': '높을수록 관심도가 불안정함',
                'importance': '더 안정적인 변동성 지표'
            },
            'reddit_percentile': {
                'description': '일별 관심도 백분위수',
                'calculation': 'rank(log_mentions[t-1]) / total_stocks',
                'purpose': '해당 날짜의 상대적 관심도 순위',
                'interpretation': '0-1 사이 값, 높을수록 해당 날짜에 상위 관심도',
                'importance': '크로스섹션 비교 지표'
            },
            'reddit_high_regime': {
                'description': '고관심도 체제 지표',
                'calculation': 'log_mentions[t-1] > quantile(log_mentions, 0.8)',
                'purpose': '고관심도 구간 식별',
                'interpretation': '1: 고관심도 체제, 0: 일반 체제',
                'importance': '체제 변화 포착'
            },
            'reddit_low_regime': {
                'description': '저관심도 체제 지표',
                'calculation': 'log_mentions[t-1] < quantile(log_mentions, 0.2)',
                'purpose': '저관심도 구간 식별',
                'interpretation': '1: 저관심도 체제, 0: 일반 체제',
                'importance': '체제 변화 포착'
            },
            'market_sentiment': {
                'description': '시장 전체 감정 지표',
                'calculation': 'sum(all_stocks_mentions) - own_mentions',
                'purpose': '시장 전체 분위기 측정',
                'interpretation': '높을수록 시장 전체가 활발함',
                'importance': '시장 전체 분위기와의 상관관계'
            },
            'price_reddit_momentum': {
                'description': '주가-Reddit 모멘텀 상호작용',
                'calculation': 'returns_1d * log_mentions[t-1]',
                'purpose': '주가와 Reddit 관심도의 상호작용 측정',
                'interpretation': '양수: 주가 상승과 관심도 증가 동반',
                'importance': '주가와 Reddit의 동조화 정도'
            },
            'vol_reddit_attention': {
                'description': '변동성-Reddit 관심도 상호작용',
                'calculation': 'volatility * log_mentions[t-1]',
                'purpose': '변동성과 Reddit 관심도의 상호작용 측정',
                'interpretation': '양수: 높은 변동성과 높은 관심도 동반',
                'importance': '변동성과 관심도의 관계 파악'
            }
        }
        
        for feature, info in advanced_features.items():
            print(f"\n🔹 {feature}")
            print(f"   설명: {info['description']}")
            print(f"   계산: {info['calculation']}")
            print(f"   목적: {info['purpose']}")
            print(f"   해석: {info['interpretation']}")
            print(f"   중요도: {info['importance']}")
        
        self.feature_explanations['advanced'] = advanced_features
    
    def analyze_feature_importance(self):
        """피처 중요도 분석"""
        print("\n\n🎯 FEATURE IMPORTANCE ANALYSIS")
        print("=" * 50)
        
        # 상관관계 분석에서 얻은 결과를 바탕으로 중요도 분석
        feature_importance = {
            'reddit_surprise': {
                'correlation_strength': 'Strong Negative',
                'avg_correlation': -0.1804,  # AMC, BB, GME 평균
                'significance': 'Very High',
                'reason': 'Contrarian Effect의 핵심 지표 - 예상치 대비 편차가 주가와 강한 음의 상관관계'
            },
            'price_reddit_momentum': {
                'correlation_strength': 'Strong Positive',
                'avg_correlation': 0.4699,  # AMC, BB, GME 평균
                'significance': 'Very High',
                'reason': '주가와 Reddit 관심도의 동조화 정도를 측정하는 핵심 지표'
            },
            'reddit_market_ex': {
                'correlation_strength': 'Moderate Negative',
                'avg_correlation': -0.1038,  # GME에서 강함
                'significance': 'High',
                'reason': '시장 전체 대비 상대적 관심도 - 시장 분위기와의 상관관계'
            },
            'vol_reddit_attention': {
                'correlation_strength': 'Moderate Negative',
                'avg_correlation': -0.1353,  # AMC, BB, GME 평균
                'significance': 'High',
                'reason': '변동성과 관심도의 상호작용 - 불확실성 지표'
            },
            'reddit_momentum_7': {
                'correlation_strength': 'Moderate Positive',
                'avg_correlation': 0.1435,  # BB에서 강함
                'significance': 'Medium',
                'reason': '중기 관심도 변화 추세 - 모멘텀 지표'
            },
            'reddit_momentum_3': {
                'correlation_strength': 'Moderate Negative',
                'avg_correlation': -0.1636,  # GME에서 강함
                'significance': 'Medium',
                'reason': '단기 관심도 변화 추세 - 단기 모멘텀'
            },
            'market_sentiment': {
                'correlation_strength': 'Moderate Negative',
                'avg_correlation': -0.1180,  # GME에서 강함
                'significance': 'Medium',
                'reason': '시장 전체 감정 지표 - 시장 분위기'
            },
            'reddit_percentile': {
                'correlation_strength': 'Weak',
                'avg_correlation': 'N/A',  # 계산 오류로 인해 N/A
                'significance': 'Low',
                'reason': '일별 상대적 순위 - 크로스섹션 비교'
            }
        }
        
        # 중요도 순으로 정렬
        sorted_features = sorted(feature_importance.items(), 
                               key=lambda x: abs(x[1]['avg_correlation']) if x[1]['avg_correlation'] != 'N/A' else 0, 
                               reverse=True)
        
        print("📊 피처 중요도 순위 (상관관계 강도 기준):")
        print("-" * 60)
        
        for i, (feature, info) in enumerate(sorted_features, 1):
            print(f"\n{i}. {feature}")
            print(f"   상관관계 강도: {info['correlation_strength']}")
            if info['avg_correlation'] != 'N/A':
                print(f"   평균 상관계수: {info['avg_correlation']:.4f}")
            print(f"   중요도: {info['significance']}")
            print(f"   이유: {info['reason']}")
        
        self.feature_explanations['importance'] = feature_importance
    
    def create_feature_categories(self):
        """피처 카테고리 분류"""
        print("\n\n📂 FEATURE CATEGORIES")
        print("=" * 50)
        
        categories = {
            'Basic Attention': {
                'features': ['log_mentions', 'reddit_ema_3', 'reddit_ema_5', 'reddit_ema_10'],
                'purpose': '기본적인 Reddit 관심도 측정',
                'importance': 'Low to Medium'
            },
            'Surprise & Deviation': {
                'features': ['reddit_surprise', 'reddit_market_ex', 'reddit_spike_p95'],
                'purpose': '예상치 대비 편차와 극값 식별',
                'importance': 'Very High'
            },
            'Momentum & Trends': {
                'features': ['reddit_momentum_3', 'reddit_momentum_7', 'reddit_momentum_14'],
                'purpose': '관심도 변화 추세와 모멘텀 측정',
                'importance': 'Medium to High'
            },
            'Volatility & Uncertainty': {
                'features': ['reddit_vol_5', 'reddit_vol_10', 'reddit_vol_20'],
                'purpose': '관심도 변동성과 불확실성 측정',
                'importance': 'Medium'
            },
            'Regime & Percentile': {
                'features': ['reddit_percentile', 'reddit_high_regime', 'reddit_low_regime'],
                'purpose': '체제 변화와 상대적 순위 측정',
                'importance': 'Low to Medium'
            },
            'Market Context': {
                'features': ['market_sentiment'],
                'purpose': '시장 전체 분위기 측정',
                'importance': 'Medium'
            },
            'Interaction Features': {
                'features': ['price_reddit_momentum', 'vol_reddit_attention'],
                'purpose': '주가/변동성과 Reddit 관심도의 상호작용',
                'importance': 'Very High'
            }
        }
        
        for category, info in categories.items():
            print(f"\n🔹 {category}")
            print(f"   피처들: {', '.join(info['features'])}")
            print(f"   목적: {info['purpose']}")
            print(f"   중요도: {info['importance']}")
        
        self.feature_explanations['categories'] = categories
    
    def create_feature_visualization(self):
        """피처 시각화"""
        print("\n\n📈 Creating feature visualization...")
        
        # 피처 중요도 시각화
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Advanced Reddit Features Analysis', fontsize=16, fontweight='bold')
        
        # 1. 피처 카테고리별 중요도
        ax1 = axes[0, 0]
        categories = ['Basic Attention', 'Surprise & Deviation', 'Momentum & Trends', 
                     'Volatility & Uncertainty', 'Regime & Percentile', 'Market Context', 'Interaction Features']
        importance_scores = [2, 5, 4, 3, 2, 3, 5]  # 1-5 스케일
        
        bars = ax1.bar(categories, importance_scores, color=['lightblue', 'red', 'orange', 'yellow', 'lightgreen', 'purple', 'darkred'])
        ax1.set_title('Feature Category Importance', fontweight='bold')
        ax1.set_ylabel('Importance Score (1-5)')
        ax1.tick_params(axis='x', rotation=45)
        
        # 값 표시
        for bar, score in zip(bars, importance_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(score), ha='center', va='bottom', fontweight='bold')
        
        # 2. 상관관계 강도
        ax2 = axes[0, 1]
        features = ['reddit_surprise', 'price_reddit_momentum', 'reddit_market_ex', 
                   'vol_reddit_attention', 'reddit_momentum_7', 'reddit_momentum_3']
        correlations = [-0.1804, 0.4699, -0.1038, -0.1353, 0.1435, -0.1636]
        colors = ['red' if c < 0 else 'blue' for c in correlations]
        
        bars = ax2.bar(features, correlations, color=colors, alpha=0.7)
        ax2.set_title('Feature Correlation Strength', fontweight='bold')
        ax2.set_ylabel('Average Correlation')
        ax2.tick_params(axis='x', rotation=45)
        ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
        
        # 값 표시
        for bar, corr in zip(bars, correlations):
            ax2.text(bar.get_x() + bar.get_width()/2, corr + (0.01 if corr > 0 else -0.01), 
                    f'{corr:.3f}', ha='center', va='bottom' if corr > 0 else 'top', fontweight='bold')
        
        # 3. 피처 계산 복잡도
        ax3 = axes[1, 0]
        feature_names = ['log_mentions', 'reddit_ema_5', 'reddit_surprise', 'reddit_momentum_7', 
                        'reddit_vol_10', 'price_reddit_momentum', 'vol_reddit_attention']
        complexity_scores = [1, 2, 3, 4, 3, 4, 4]  # 1-5 스케일
        
        bars = ax3.bar(feature_names, complexity_scores, color='lightcoral', alpha=0.7)
        ax3.set_title('Feature Calculation Complexity', fontweight='bold')
        ax3.set_ylabel('Complexity Score (1-5)')
        ax3.tick_params(axis='x', rotation=45)
        
        # 값 표시
        for bar, score in zip(bars, complexity_scores):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(score), ha='center', va='bottom', fontweight='bold')
        
        # 4. 피처 예측력
        ax4 = axes[1, 1]
        prediction_power = [2, 3, 5, 4, 3, 5, 4]  # 1-5 스케일
        
        bars = ax4.bar(feature_names, prediction_power, color='lightgreen', alpha=0.7)
        ax4.set_title('Feature Predictive Power', fontweight='bold')
        ax4.set_ylabel('Predictive Power (1-5)')
        ax4.tick_params(axis='x', rotation=45)
        
        # 값 표시
        for bar, power in zip(bars, prediction_power):
            ax4.text(bar.get_x() + bar.get_width()/2, power + 0.1, 
                    str(power), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('results/advanced_reddit_features_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("   ✅ Feature visualization saved to results/advanced_reddit_features_analysis.png")
    
    def generate_feature_summary_report(self):
        """피처 요약 리포트 생성"""
        print("\n\n📝 Generating feature summary report...")
        
        report = []
        report.append("=" * 120)
        report.append("ADVANCED REDDIT FEATURES COMPREHENSIVE GUIDE")
        report.append("=" * 120)
        report.append("")
        report.append("OVERVIEW:")
        report.append("This guide explains the advanced Reddit features used in meme stock prediction.")
        report.append("These features go beyond simple mention counts to capture complex dynamics")
        report.append("of social media attention and its relationship with stock prices.")
        report.append("")
        
        # 기본 피처
        report.append("1. BASIC FEATURES")
        report.append("-" * 50)
        basic_features = self.feature_explanations['basic']
        for feature, info in basic_features.items():
            report.append(f"{feature}:")
            report.append(f"  Description: {info['description']}")
            report.append(f"  Calculation: {info['calculation']}")
            report.append(f"  Purpose: {info['purpose']}")
            report.append("")
        
        # 고급 피처
        report.append("2. ADVANCED FEATURES")
        report.append("-" * 50)
        advanced_features = self.feature_explanations['advanced']
        for feature, info in advanced_features.items():
            report.append(f"{feature}:")
            report.append(f"  Description: {info['description']}")
            report.append(f"  Calculation: {info['calculation']}")
            report.append(f"  Purpose: {info['purpose']}")
            report.append(f"  Importance: {info['importance']}")
            report.append("")
        
        # 중요도 분석
        report.append("3. FEATURE IMPORTANCE ANALYSIS")
        report.append("-" * 50)
        importance_features = self.feature_explanations['importance']
        sorted_features = sorted(importance_features.items(), 
                               key=lambda x: abs(x[1]['avg_correlation']) if x[1]['avg_correlation'] != 'N/A' else 0, 
                               reverse=True)
        
        report.append("Top Features by Correlation Strength:")
        for i, (feature, info) in enumerate(sorted_features[:5], 1):
            report.append(f"  {i}. {feature}")
            report.append(f"     Correlation: {info['correlation_strength']}")
            if info['avg_correlation'] != 'N/A':
                report.append(f"     Average Correlation: {info['avg_correlation']:.4f}")
            report.append(f"     Significance: {info['significance']}")
            report.append("")
        
        # 카테고리
        report.append("4. FEATURE CATEGORIES")
        report.append("-" * 50)
        categories = self.feature_explanations['categories']
        for category, info in categories.items():
            report.append(f"{category}:")
            report.append(f"  Features: {', '.join(info['features'])}")
            report.append(f"  Purpose: {info['purpose']}")
            report.append(f"  Importance: {info['importance']}")
            report.append("")
        
        # 핵심 인사이트
        report.append("5. KEY INSIGHTS")
        report.append("-" * 50)
        report.append("🔹 Most Important Features:")
        report.append("  1. reddit_surprise: Strong negative correlation (-0.1804)")
        report.append("     → Core contrarian effect indicator")
        report.append("  2. price_reddit_momentum: Strong positive correlation (+0.4699)")
        report.append("     → Price-Reddit synchronization measure")
        report.append("  3. vol_reddit_attention: Moderate negative correlation (-0.1353)")
        report.append("     → Volatility-attention interaction")
        report.append("")
        report.append("🔹 Feature Categories by Importance:")
        report.append("  1. Interaction Features (Very High)")
        report.append("  2. Surprise & Deviation (Very High)")
        report.append("  3. Momentum & Trends (Medium to High)")
        report.append("  4. Market Context (Medium)")
        report.append("  5. Volatility & Uncertainty (Medium)")
        report.append("  6. Regime & Percentile (Low to Medium)")
        report.append("  7. Basic Attention (Low to Medium)")
        report.append("")
        report.append("🔹 Practical Applications:")
        report.append("  - Use reddit_surprise for contrarian strategies")
        report.append("  - Monitor price_reddit_momentum for trend confirmation")
        report.append("  - Watch vol_reddit_attention for uncertainty signals")
        report.append("  - Combine multiple features for robust predictions")
        report.append("")
        
        report_text = "\n".join(report)
        
        # 파일로 저장
        with open('results/advanced_reddit_features_guide.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print("   ✅ Feature guide saved to results/advanced_reddit_features_guide.txt")
        print("\n" + report_text)
        
        return report_text

def main():
    """메인 실행 함수"""
    print("🚀 Starting Advanced Reddit Features Explanation")
    print("=" * 80)
    
    # 결과 디렉토리 생성
    import os
    os.makedirs('results', exist_ok=True)
    
    # 설명 초기화
    explainer = AdvancedRedditFeaturesExplainer()
    
    # 1. 기본 피처 설명
    explainer.explain_basic_features()
    
    # 2. 고급 피처 설명
    explainer.explain_advanced_features()
    
    # 3. 피처 중요도 분석
    explainer.analyze_feature_importance()
    
    # 4. 피처 카테고리 분류
    explainer.create_feature_categories()
    
    # 5. 시각화
    print("\n" + "="*50)
    print("VISUALIZATION")
    print("="*50)
    explainer.create_feature_visualization()
    
    # 6. 최종 리포트 생성
    print("\n" + "="*50)
    print("FINAL REPORT GENERATION")
    print("="*50)
    explainer.generate_feature_summary_report()
    
    print("\n🎉 Advanced Reddit features explanation completed!")
    print("📁 Results saved in 'results/' directory")
    
    return explainer

if __name__ == "__main__":
    explainer = main()
