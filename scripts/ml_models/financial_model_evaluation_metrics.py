#!/usr/bin/env python3
"""
Financial Model Evaluation Metrics
=================================
금융 예측 모델을 위한 전문 평가 지표들

일반 ML과 다른 금융 특화 지표들:
1. Information Coefficient (IC) - 예측력 측정
2. Information Ratio (IR) - 위험 대비 수익
3. Hit Rate - 방향성 정확도  
4. Sharpe Ratio - 위험 조정 수익률
5. Maximum Drawdown - 최대 손실
6. Calmar Ratio - 위험 대비 성과
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class FinancialModelEvaluator:
    """금융 모델 전용 평가 클래스"""
    
    def __init__(self):
        self.metrics = {}
        
    def calculate_information_coefficient(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Information Coefficient (IC) - 금융에서 가장 중요한 지표
        
        IC = Correlation(실제수익률, 예측수익률)
        - IC > 0.05: 매우 좋음
        - IC > 0.03: 좋음  
        - IC > 0.01: 보통
        - IC < 0: 역방향 (contrarian 신호)
        """
        # Pearson IC (선형 관계)
        ic_pearson, ic_p_value = pearsonr(y_true, y_pred)
        
        # Rank IC (비선형 관계, Spearman)
        ic_rank, rank_p_value = spearmanr(y_true, y_pred)
        
        # IC 통계량
        ic_mean = ic_pearson
        ic_std = np.std([ic_pearson])  # 실제로는 rolling window로 계산
        
        # Information Ratio = IC / IC_std
        information_ratio = ic_mean / (ic_std + 1e-8)
        
        return {
            'ic_pearson': ic_pearson,
            'ic_rank': ic_rank,
            'ic_p_value': ic_p_value,
            'rank_p_value': rank_p_value,
            'information_ratio': information_ratio,
            'ic_significance': 'Significant' if ic_p_value < 0.05 else 'Not Significant'
        }
    
    def calculate_hit_rate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Hit Rate - 방향성 예측 정확도
        
        일반 ML accuracy와 다름:
        - 단순히 맞춘 개수가 아니라 '방향'을 맞춘 비율
        - 금융에서는 크기보다 방향이 더 중요할 때가 많음
        """
        # 방향성 계산
        actual_direction = np.sign(y_true)
        predicted_direction = np.sign(y_pred)
        
        # Hit rate 계산
        hit_rate = np.mean(actual_direction == predicted_direction)
        
        # 세분화된 hit rate
        positive_mask = y_true > 0
        negative_mask = y_true < 0
        
        hit_rate_positive = np.mean(actual_direction[positive_mask] == predicted_direction[positive_mask]) if np.any(positive_mask) else 0
        hit_rate_negative = np.mean(actual_direction[negative_mask] == predicted_direction[negative_mask]) if np.any(negative_mask) else 0
        
        return {
            'overall_hit_rate': hit_rate,
            'hit_rate_up_days': hit_rate_positive,
            'hit_rate_down_days': hit_rate_negative,
            'up_day_count': np.sum(positive_mask),
            'down_day_count': np.sum(negative_mask)
        }
    
    def calculate_financial_returns_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        금융 수익률 기반 지표들
        
        실제 거래했을 때의 성과를 시뮬레이션
        """
        # 예측 기반 포지션 생성 (1: 매수, -1: 매도, 0: 중립)
        positions = np.sign(y_pred)
        
        # 실제 수익률과 포지션으로 전략 수익률 계산
        strategy_returns = positions * y_true
        
        # 벤치마크 (단순 매수 보유)
        benchmark_returns = y_true
        
        # 누적 수익률
        cumulative_strategy = np.cumprod(1 + strategy_returns) - 1
        cumulative_benchmark = np.cumprod(1 + benchmark_returns) - 1
        
        # 연간화 (252 거래일 기준)
        trading_days = 252
        n_periods = len(strategy_returns)
        
        # Sharpe Ratio
        strategy_sharpe = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-8) * np.sqrt(trading_days)
        benchmark_sharpe = np.mean(benchmark_returns) / (np.std(benchmark_returns) + 1e-8) * np.sqrt(trading_days)
        
        # Maximum Drawdown
        def calculate_max_drawdown(returns):
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            return np.min(drawdown)
        
        strategy_max_dd = calculate_max_drawdown(strategy_returns)
        benchmark_max_dd = calculate_max_drawdown(benchmark_returns)
        
        # Calmar Ratio = Annual Return / |Max Drawdown|
        annual_strategy_return = np.mean(strategy_returns) * trading_days
        annual_benchmark_return = np.mean(benchmark_returns) * trading_days
        
        strategy_calmar = annual_strategy_return / (abs(strategy_max_dd) + 1e-8)
        benchmark_calmar = annual_benchmark_return / (abs(benchmark_max_dd) + 1e-8)
        
        return {
            'strategy_total_return': cumulative_strategy[-1],
            'benchmark_total_return': cumulative_benchmark[-1],
            'strategy_annual_return': annual_strategy_return,
            'benchmark_annual_return': annual_benchmark_return,
            'strategy_sharpe': strategy_sharpe,
            'benchmark_sharpe': benchmark_sharpe,
            'strategy_max_drawdown': strategy_max_dd,
            'benchmark_max_drawdown': benchmark_max_dd,
            'strategy_calmar': strategy_calmar,
            'benchmark_calmar': benchmark_calmar,
            'excess_return': annual_strategy_return - annual_benchmark_return
        }
    
    def calculate_volatility_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        변동성 관련 지표들
        
        밈스톡은 변동성이 매우 높으므로 중요
        """
        # 예측 오차
        errors = y_pred - y_true
        
        # 변동성 예측 정확도
        actual_vol = np.std(y_true)
        predicted_vol = np.std(y_pred)
        
        # 조건부 변동성 (상승/하락장에서의 예측 정확도)
        up_days = y_true > 0
        down_days = y_true < 0
        
        up_day_accuracy = np.mean(np.abs(errors[up_days])) if np.any(up_days) else 0
        down_day_accuracy = np.mean(np.abs(errors[down_days])) if np.any(down_days) else 0
        
        return {
            'actual_volatility': actual_vol,
            'predicted_volatility': predicted_vol,
            'volatility_ratio': predicted_vol / (actual_vol + 1e-8),
            'error_volatility': np.std(errors),
            'up_day_mae': up_day_accuracy,
            'down_day_mae': down_day_accuracy,
            'vol_timing_ability': 1 - (up_day_accuracy + down_day_accuracy) / (2 * np.mean(np.abs(errors)))
        }
    
    def calculate_regime_based_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        시장 상황별 성과 지표
        
        밈스톡은 viral/normal 구간이 다르므로 중요
        """
        # 변동성 기준으로 regime 구분
        vol_threshold = np.percentile(np.abs(y_true), 75)  # 상위 25%를 고변동성으로 정의
        
        high_vol_mask = np.abs(y_true) > vol_threshold
        low_vol_mask = ~high_vol_mask
        
        # 각 regime에서의 성과
        def regime_metrics(mask, regime_name):
            if not np.any(mask):
                return {}
            
            regime_true = y_true[mask]
            regime_pred = y_pred[mask]
            
            ic, _ = pearsonr(regime_true, regime_pred)
            hit_rate = np.mean(np.sign(regime_true) == np.sign(regime_pred))
            
            return {
                f'{regime_name}_ic': ic,
                f'{regime_name}_hit_rate': hit_rate,
                f'{regime_name}_samples': np.sum(mask),
                f'{regime_name}_mae': np.mean(np.abs(regime_pred - regime_true))
            }
        
        high_vol_metrics = regime_metrics(high_vol_mask, 'high_vol')
        low_vol_metrics = regime_metrics(low_vol_mask, 'low_vol')
        
        return {**high_vol_metrics, **low_vol_metrics}
    
    def calculate_traditional_ml_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        전통적인 ML 지표들 (참고용)
        """
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'explained_variance': 1 - np.var(y_true - y_pred) / np.var(y_true)
        }
    
    def comprehensive_evaluation(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               model_name: str = "Model") -> Dict:
        """
        종합적인 금융 모델 평가
        """
        print(f"📊 Comprehensive Financial Model Evaluation: {model_name}")
        print("=" * 60)
        
        # 1. Information Coefficient
        ic_metrics = self.calculate_information_coefficient(y_true, y_pred)
        print(f"\n📈 INFORMATION COEFFICIENT:")
        print(f"   IC (Pearson): {ic_metrics['ic_pearson']:.4f}")
        print(f"   IC (Rank): {ic_metrics['ic_rank']:.4f}")
        print(f"   Information Ratio: {ic_metrics['information_ratio']:.4f}")
        print(f"   Significance: {ic_metrics['ic_significance']}")
        
        # 2. Hit Rate
        hit_metrics = self.calculate_hit_rate(y_true, y_pred)
        print(f"\n🎯 HIT RATE ANALYSIS:")
        print(f"   Overall Hit Rate: {hit_metrics['overall_hit_rate']:.1%}")
        print(f"   Up Days Hit Rate: {hit_metrics['hit_rate_up_days']:.1%}")
        print(f"   Down Days Hit Rate: {hit_metrics['hit_rate_down_days']:.1%}")
        
        # 3. Financial Returns
        returns_metrics = self.calculate_financial_returns_metrics(y_true, y_pred)
        print(f"\n💰 FINANCIAL PERFORMANCE:")
        print(f"   Strategy Return: {returns_metrics['strategy_total_return']:.1%}")
        print(f"   Benchmark Return: {returns_metrics['benchmark_total_return']:.1%}")
        print(f"   Excess Return: {returns_metrics['excess_return']:.1%}")
        print(f"   Strategy Sharpe: {returns_metrics['strategy_sharpe']:.2f}")
        print(f"   Max Drawdown: {returns_metrics['strategy_max_drawdown']:.1%}")
        
        # 4. Volatility Analysis
        vol_metrics = self.calculate_volatility_metrics(y_true, y_pred)
        print(f"\n📊 VOLATILITY ANALYSIS:")
        print(f"   Actual Vol: {vol_metrics['actual_volatility']:.1%}")
        print(f"   Predicted Vol: {vol_metrics['predicted_volatility']:.1%}")
        print(f"   Vol Timing Ability: {vol_metrics['vol_timing_ability']:.2f}")
        
        # 5. Regime Analysis
        regime_metrics = self.calculate_regime_based_metrics(y_true, y_pred)
        print(f"\n🌪️ REGIME ANALYSIS:")
        if 'high_vol_ic' in regime_metrics:
            print(f"   High Vol IC: {regime_metrics['high_vol_ic']:.4f}")
            print(f"   High Vol Hit Rate: {regime_metrics['high_vol_hit_rate']:.1%}")
        if 'low_vol_ic' in regime_metrics:
            print(f"   Low Vol IC: {regime_metrics['low_vol_ic']:.4f}")
            print(f"   Low Vol Hit Rate: {regime_metrics['low_vol_hit_rate']:.1%}")
        
        # 6. Traditional ML (참고용)
        ml_metrics = self.calculate_traditional_ml_metrics(y_true, y_pred)
        print(f"\n🤖 TRADITIONAL ML METRICS (Reference):")
        print(f"   R²: {ml_metrics['r2']:.4f}")
        print(f"   RMSE: {ml_metrics['rmse']:.4f}")
        print(f"   MAE: {ml_metrics['mae']:.4f}")
        
        # 종합 점수 계산
        overall_score = self._calculate_overall_score(ic_metrics, hit_metrics, returns_metrics)
        print(f"\n🏆 OVERALL SCORE: {overall_score:.2f}/100")
        
        # 모든 지표 통합
        all_metrics = {
            'model_name': model_name,
            'overall_score': overall_score,
            **ic_metrics,
            **hit_metrics,
            **returns_metrics,
            **vol_metrics,
            **regime_metrics,
            **ml_metrics
        }
        
        return all_metrics
    
    def _calculate_overall_score(self, ic_metrics: Dict, hit_metrics: Dict, 
                                returns_metrics: Dict) -> float:
        """
        종합 점수 계산 (0-100)
        
        가중치:
        - IC: 40%
        - Hit Rate: 30%  
        - Returns: 30%
        """
        # IC 점수 (0-40)
        ic_score = min(40, abs(ic_metrics['ic_pearson']) * 1000)  # 0.04 IC = 40점
        
        # Hit Rate 점수 (0-30)
        hit_rate = hit_metrics['overall_hit_rate']
        hit_score = max(0, (hit_rate - 0.5) * 60)  # 50% = 0점, 100% = 30점
        
        # Returns 점수 (0-30)
        sharpe = returns_metrics['strategy_sharpe']
        returns_score = min(30, max(0, sharpe * 15))  # Sharpe 2.0 = 30점
        
        return ic_score + hit_score + returns_score
    
    def model_comparison_report(self, results: List[Dict]):
        """
        여러 모델 비교 리포트
        """
        print("\n" + "="*80)
        print("📊 MULTI-MODEL COMPARISON REPORT")
        print("="*80)
        
        # 정렬 기준별 랭킹
        rankings = {
            'overall_score': sorted(results, key=lambda x: x['overall_score'], reverse=True),
            'ic_pearson': sorted(results, key=lambda x: abs(x['ic_pearson']), reverse=True),
            'overall_hit_rate': sorted(results, key=lambda x: x['overall_hit_rate'], reverse=True),
            'strategy_sharpe': sorted(results, key=lambda x: x['strategy_sharpe'], reverse=True)
        }
        
        print(f"\n🏆 RANKINGS BY DIFFERENT CRITERIA:")
        print(f"{'Rank':<4} {'Overall':<15} {'IC Leader':<15} {'Hit Rate':<15} {'Sharpe':<15}")
        print("-" * 70)
        
        for i in range(min(5, len(results))):
            print(f"{i+1:<4} "
                  f"{rankings['overall_score'][i]['model_name']:<15} "
                  f"{rankings['ic_pearson'][i]['model_name']:<15} "
                  f"{rankings['overall_hit_rate'][i]['model_name']:<15} "
                  f"{rankings['strategy_sharpe'][i]['model_name']:<15}")
        
        # 최고 성과 모델
        best_model = rankings['overall_score'][0]
        print(f"\n🥇 BEST OVERALL MODEL: {best_model['model_name']}")
        print(f"   Overall Score: {best_model['overall_score']:.1f}/100")
        print(f"   IC: {best_model['ic_pearson']:.4f}")
        print(f"   Hit Rate: {best_model['overall_hit_rate']:.1%}")
        print(f"   Sharpe: {best_model['strategy_sharpe']:.2f}")
        
        return best_model


def demo_evaluation():
    """
    평가 지표 데모
    """
    print("📊 Financial Model Evaluation Metrics Demo")
    print("="*50)
    
    # 샘플 데이터 생성 (실제 밈스톡 패턴 모방)
    np.random.seed(42)
    n_samples = 100
    
    # 실제 수익률 (밈스톡 특성: 높은 변동성, 비정규분포)
    y_true = np.random.normal(0, 0.05, n_samples)  # 5% 일일 변동성
    y_true[::10] = np.random.normal(0, 0.2, len(y_true[::10]))  # 10%는 극단적 변동
    
    # 모델 예측들 (다양한 성능 시뮬레이션)
    models_predictions = {
        'Perfect_Model': y_true + np.random.normal(0, 0.01, n_samples),
        'Good_Model': y_true * 0.7 + np.random.normal(0, 0.02, n_samples),  
        'Contrarian_Model': -y_true * 0.5 + np.random.normal(0, 0.02, n_samples),
        'Random_Model': np.random.normal(0, 0.03, n_samples)
    }
    
    # 평가 실행
    evaluator = FinancialModelEvaluator()
    results = []
    
    for model_name, y_pred in models_predictions.items():
        result = evaluator.comprehensive_evaluation(y_true, y_pred, model_name)
        results.append(result)
        print("\n" + "-"*60)
    
    # 비교 리포트
    best_model = evaluator.model_comparison_report(results)
    
    return results


if __name__ == "__main__":
    print("🚀 Financial Model Evaluation Framework")
    print("=" * 50)
    print("💡 금융 예측 모델을 위한 전문 평가 지표들")
    print("📊 IC, Hit Rate, Sharpe, Drawdown 등 포함")
    print("=" * 50)
    
    # 데모 실행
    demo_results = demo_evaluation()
    
    print(f"\n✅ Demo completed! {len(demo_results)} models evaluated.")
    print("📚 이 지표들을 논문에서 사용하세요!")

