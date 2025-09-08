#!/usr/bin/env python3
"""
Contrarian vs Non-Contrarian Model Performance Comparison
=========================================================
Contrarian 전략 적용 유무에 따른 성능 비교 분석

비교 항목:
1. Standard Models (원본)
2. Contrarian Models (예측값 반전)
3. Target-Flipped Models (타겟 반전 훈련)
4. Ensemble Strategies
"""

import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from typing import Dict, List, Tuple
import json
import matplotlib.pyplot as plt
import seaborn as sns

# ML Libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb

# Statistical Libraries
from scipy.stats import spearmanr, pearsonr

warnings.filterwarnings('ignore')
np.random.seed(42)

class ContrarianComparisonAnalyzer:
    """Contrarian vs Standard 모델 성능 비교 분석기"""
    
    def __init__(self):
        self.results = {}
        self.comparison_data = {}
        
        print("🔄 Contrarian vs Standard Model Comparison")
        print("=" * 50)
        
    def load_data(self):
        """데이터 로드"""
        print("📊 Loading datasets...")
        
        self.train_df = pd.read_csv("data/colab_datasets/tabular_train_20250814_031335.csv")
        self.test_df = pd.read_csv("data/colab_datasets/tabular_test_20250814_031335.csv")
        
        # Top features (이전 분석에서 확인된 중요 특성들)
        self.top_features = [
            'log_mentions', 'price_ratio_sma20', 'price_ratio_sma10', 
            'returns_3d', 'returns_1d', 'market_sentiment',
            'reddit_market_ex', 'price_reddit_momentum', 'reddit_momentum_3',
            'reddit_surprise', 'reddit_ema_3', 'reddit_momentum_7',
            'rsi_14', 'reddit_ema_5', 'returns_10d'
        ]
        
        # 존재하는 특성만 선택
        self.available_features = [f for f in self.top_features if f in self.train_df.columns]
        print(f"   ✅ Using {len(self.available_features)} features")
        
        # 데이터 준비
        self.X_train = self.train_df[self.available_features].fillna(0).values.astype(np.float32)
        self.y_train = self.train_df['y1d'].values.astype(np.float32)
        self.X_test = self.test_df[self.available_features].fillna(0).values.astype(np.float32)
        self.y_test = self.test_df['y1d'].values.astype(np.float32)
        
        # 스케일링
        self.scaler = RobustScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"   📈 Train: {len(self.X_train)}, Test: {len(self.X_test)}")
        
    def calculate_comprehensive_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                      strategy_name: str) -> Dict:
        """종합적인 성능 지표 계산"""
        
        # 기본 지표
        ic_pearson, ic_p = pearsonr(y_true, y_pred) if len(y_true) > 1 else (0, 1)
        ic_rank, rank_p = spearmanr(y_true, y_pred) if len(y_true) > 1 else (0, 1)
        
        # Hit Rate
        hit_rate = np.mean(np.sign(y_true) == np.sign(y_pred))
        
        # 상승/하락일 별 Hit Rate
        up_days = y_true > 0
        down_days = y_true < 0
        hit_rate_up = np.mean(np.sign(y_true[up_days]) == np.sign(y_pred[up_days])) if np.any(up_days) else 0
        hit_rate_down = np.mean(np.sign(y_true[down_days]) == np.sign(y_pred[down_days])) if np.any(down_days) else 0
        
        # 금융 성과 지표
        positions = np.sign(y_pred)
        strategy_returns = positions * y_true
        benchmark_returns = y_true
        
        # 누적 수익률
        cumulative_strategy = np.cumprod(1 + strategy_returns) - 1
        cumulative_benchmark = np.cumprod(1 + benchmark_returns) - 1
        
        # Sharpe Ratio (연간화)
        strategy_sharpe = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-8) * np.sqrt(252)
        
        # Maximum Drawdown
        def max_drawdown(returns):
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            return np.min(drawdown)
        
        max_dd = max_drawdown(strategy_returns)
        
        # Information Ratio
        info_ratio = ic_pearson / (np.std([ic_pearson]) + 1e-8)
        
        # 전통적 ML 지표 (참고용)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'strategy': strategy_name,
            'ic_pearson': ic_pearson,
            'ic_rank': ic_rank,
            'ic_p_value': ic_p,
            'hit_rate': hit_rate,
            'hit_rate_up': hit_rate_up,
            'hit_rate_down': hit_rate_down,
            'up_days_count': np.sum(up_days),
            'down_days_count': np.sum(down_days),
            'total_return': cumulative_strategy[-1],
            'benchmark_return': cumulative_benchmark[-1],
            'excess_return': cumulative_strategy[-1] - cumulative_benchmark[-1],
            'sharpe_ratio': strategy_sharpe,
            'max_drawdown': max_dd,
            'information_ratio': info_ratio,
            'mse': mse,
            'r2': r2,
            'n_samples': len(y_true)
        }
    
    def train_standard_models(self):
        """표준 모델들 (Contrarian 적용 X)"""
        print("\n🤖 Training Standard Models (No Contrarian)...")
        
        standard_results = {}
        
        # 1. LightGBM Standard
        lgb_model = lgb.LGBMRegressor(
            objective='regression',
            num_leaves=80,
            max_depth=6,
            learning_rate=0.05,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbosity=-1
        )
        lgb_model.fit(self.X_train_scaled, self.y_train)
        lgb_pred = lgb_model.predict(self.X_test_scaled)
        standard_results['LightGBM_Standard'] = self.calculate_comprehensive_metrics(
            self.y_test, lgb_pred, 'LightGBM_Standard')
        
        # 2. RandomForest Standard
        rf_model = RandomForestRegressor(
            n_estimators=150,
            max_depth=8,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(self.X_train_scaled, self.y_train)
        rf_pred = rf_model.predict(self.X_test_scaled)
        standard_results['RandomForest_Standard'] = self.calculate_comprehensive_metrics(
            self.y_test, rf_pred, 'RandomForest_Standard')
        
        # 3. XGBoost Standard
        xgb_model = xgb.XGBRegressor(
            max_depth=6,
            learning_rate=0.05,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )
        xgb_model.fit(self.X_train_scaled, self.y_train)
        xgb_pred = xgb_model.predict(self.X_test_scaled)
        standard_results['XGBoost_Standard'] = self.calculate_comprehensive_metrics(
            self.y_test, xgb_pred, 'XGBoost_Standard')
        
        # 4. Ensemble Standard
        ensemble_pred = (lgb_pred + rf_pred + xgb_pred) / 3
        standard_results['Ensemble_Standard'] = self.calculate_comprehensive_metrics(
            self.y_test, ensemble_pred, 'Ensemble_Standard')
        
        self.results['standard'] = standard_results
        return standard_results
    
    def train_contrarian_models(self):
        """Contrarian 모델들 (예측값 반전)"""
        print("\n🔄 Training Contrarian Models (Prediction Flip)...")
        
        # 먼저 standard 모델들의 예측값을 가져와서 반전
        standard_results = self.results['standard']
        contrarian_results = {}
        
        for model_name, metrics in standard_results.items():
            # 해당 모델을 다시 훈련 (예측값 구하기 위해)
            if 'LightGBM' in model_name:
                model = lgb.LGBMRegressor(
                    objective='regression', num_leaves=80, max_depth=6,
                    learning_rate=0.05, n_estimators=200, subsample=0.8,
                    colsample_bytree=0.9, reg_alpha=0.1, reg_lambda=0.1,
                    random_state=42, verbosity=-1
                )
                model.fit(self.X_train_scaled, self.y_train)
                predictions = model.predict(self.X_test_scaled)
                
            elif 'RandomForest' in model_name:
                model = RandomForestRegressor(
                    n_estimators=150, max_depth=8, min_samples_split=20,
                    min_samples_leaf=10, random_state=42, n_jobs=-1
                )
                model.fit(self.X_train_scaled, self.y_train)
                predictions = model.predict(self.X_test_scaled)
                
            elif 'XGBoost' in model_name:
                model = xgb.XGBRegressor(
                    max_depth=6, learning_rate=0.05, n_estimators=200,
                    subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
                )
                model.fit(self.X_train_scaled, self.y_train)
                predictions = model.predict(self.X_test_scaled)
                
            elif 'Ensemble' in model_name:
                # 앙상블 예측 재구성
                lgb_model = lgb.LGBMRegressor(objective='regression', num_leaves=80, max_depth=6,
                    learning_rate=0.05, n_estimators=200, random_state=42, verbosity=-1)
                rf_model = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1)
                xgb_model = xgb.XGBRegressor(max_depth=6, learning_rate=0.05, n_estimators=200, 
                    random_state=42, verbosity=0)
                
                lgb_model.fit(self.X_train_scaled, self.y_train)
                rf_model.fit(self.X_train_scaled, self.y_train)
                xgb_model.fit(self.X_train_scaled, self.y_train)
                
                predictions = (lgb_model.predict(self.X_test_scaled) + 
                             rf_model.predict(self.X_test_scaled) + 
                             xgb_model.predict(self.X_test_scaled)) / 3
            
            # Contrarian 예측값 (반전)
            contrarian_pred = -predictions
            contrarian_name = model_name.replace('_Standard', '_Contrarian')
            
            contrarian_results[contrarian_name] = self.calculate_comprehensive_metrics(
                self.y_test, contrarian_pred, contrarian_name)
        
        self.results['contrarian'] = contrarian_results
        return contrarian_results
    
    def train_target_flipped_models(self):
        """Target-Flipped 모델들 (타겟 반전하여 훈련)"""
        print("\n🔀 Training Target-Flipped Models...")
        
        # 타겟을 반전하여 훈련
        y_train_flipped = -self.y_train
        target_flipped_results = {}
        
        # 1. LightGBM Target-Flipped
        lgb_model = lgb.LGBMRegressor(
            objective='regression', num_leaves=80, max_depth=6,
            learning_rate=0.05, n_estimators=200, subsample=0.8,
            colsample_bytree=0.9, reg_alpha=0.1, reg_lambda=0.1,
            random_state=42, verbosity=-1
        )
        lgb_model.fit(self.X_train_scaled, y_train_flipped)
        lgb_pred = lgb_model.predict(self.X_test_scaled)
        target_flipped_results['LightGBM_TargetFlipped'] = self.calculate_comprehensive_metrics(
            self.y_test, lgb_pred, 'LightGBM_TargetFlipped')
        
        # 2. RandomForest Target-Flipped
        rf_model = RandomForestRegressor(
            n_estimators=150, max_depth=8, min_samples_split=20,
            min_samples_leaf=10, random_state=42, n_jobs=-1
        )
        rf_model.fit(self.X_train_scaled, y_train_flipped)
        rf_pred = rf_model.predict(self.X_test_scaled)
        target_flipped_results['RandomForest_TargetFlipped'] = self.calculate_comprehensive_metrics(
            self.y_test, rf_pred, 'RandomForest_TargetFlipped')
        
        self.results['target_flipped'] = target_flipped_results
        return target_flipped_results
    
    def create_comparison_table(self):
        """비교 테이블 생성"""
        print("\n📊 COMPREHENSIVE COMPARISON TABLE")
        print("=" * 100)
        
        # 모든 결과 통합
        all_results = []
        for category, results in self.results.items():
            for model_name, metrics in results.items():
                metrics['category'] = category
                all_results.append(metrics)
        
        # 테이블 헤더
        print(f"{'Strategy':<25} {'Category':<15} {'IC':<8} {'Hit Rate':<10} {'Sharpe':<8} {'Return':<10} {'MaxDD':<8}")
        print("-" * 100)
        
        # 카테고리별로 정렬해서 출력
        categories = ['standard', 'contrarian', 'target_flipped']
        for category in categories:
            category_results = [r for r in all_results if r['category'] == category]
            category_results.sort(key=lambda x: x['ic_pearson'], reverse=True)  # IC 기준 정렬
            
            for result in category_results:
                print(f"{result['strategy']:<25} {result['category']:<15} "
                      f"{result['ic_pearson']:<8.4f} {result['hit_rate']:<10.3f} "
                      f"{result['sharpe_ratio']:<8.2f} {result['total_return']:<10.1%} "
                      f"{result['max_drawdown']:<8.1%}")
        
        return all_results
    
    def analyze_contrarian_effectiveness(self, all_results: List[Dict]):
        """Contrarian 효과 분석"""
        print(f"\n🔍 CONTRARIAN EFFECTIVENESS ANALYSIS")
        print("=" * 60)
        
        # 같은 베이스 모델끼리 비교
        model_bases = ['LightGBM', 'RandomForest', 'XGBoost', 'Ensemble']
        
        improvements = []
        
        for base in model_bases:
            standard = next((r for r in all_results if r['strategy'] == f'{base}_Standard'), None)
            contrarian = next((r for r in all_results if r['strategy'] == f'{base}_Contrarian'), None)
            
            if standard and contrarian:
                # IC 개선도 (절댓값 기준)
                ic_improvement = abs(contrarian['ic_pearson']) - abs(standard['ic_pearson'])
                
                # Hit Rate 개선도
                hr_improvement = contrarian['hit_rate'] - standard['hit_rate']
                
                # Sharpe 개선도
                sharpe_improvement = contrarian['sharpe_ratio'] - standard['sharpe_ratio']
                
                # Return 개선도
                return_improvement = contrarian['total_return'] - standard['total_return']
                
                improvements.append({
                    'model': base,
                    'ic_improvement': ic_improvement,
                    'hr_improvement': hr_improvement,
                    'sharpe_improvement': sharpe_improvement,
                    'return_improvement': return_improvement,
                    'standard_ic': standard['ic_pearson'],
                    'contrarian_ic': contrarian['ic_pearson']
                })
                
                print(f"\n📈 {base} Model Comparison:")
                print(f"   Standard IC: {standard['ic_pearson']:.4f} → Contrarian IC: {contrarian['ic_pearson']:.4f}")
                print(f"   |IC| Change: {ic_improvement:+.4f}")
                print(f"   Hit Rate: {standard['hit_rate']:.1%} → {contrarian['hit_rate']:.1%} ({hr_improvement:+.1%})")
                print(f"   Sharpe: {standard['sharpe_ratio']:.2f} → {contrarian['sharpe_ratio']:.2f} ({sharpe_improvement:+.2f})")
                print(f"   Return: {standard['total_return']:.1%} → {contrarian['total_return']:.1%} ({return_improvement:+.1%})")
        
        # 전체 요약
        avg_ic_improvement = np.mean([imp['ic_improvement'] for imp in improvements])
        avg_hr_improvement = np.mean([imp['hr_improvement'] for imp in improvements])
        avg_sharpe_improvement = np.mean([imp['sharpe_improvement'] for imp in improvements])
        
        print(f"\n🏆 OVERALL CONTRARIAN IMPACT:")
        print(f"   Average |IC| Improvement: {avg_ic_improvement:+.4f}")
        print(f"   Average Hit Rate Improvement: {avg_hr_improvement:+.1%}")
        print(f"   Average Sharpe Improvement: {avg_sharpe_improvement:+.2f}")
        
        # 성공률 계산
        successful_models = len([imp for imp in improvements if imp['ic_improvement'] > 0 and imp['hr_improvement'] > 0])
        print(f"   Models Improved by Contrarian: {successful_models}/{len(improvements)} ({successful_models/len(improvements):.1%})")
        
        return improvements
    
    def find_best_strategy(self, all_results: List[Dict]):
        """최적 전략 찾기"""
        print(f"\n🏆 BEST STRATEGY IDENTIFICATION")
        print("=" * 50)
        
        # 다양한 기준으로 최고 모델 찾기
        best_ic = max(all_results, key=lambda x: abs(x['ic_pearson']))
        best_hit_rate = max(all_results, key=lambda x: x['hit_rate'])
        best_sharpe = max(all_results, key=lambda x: x['sharpe_ratio'])
        best_return = max(all_results, key=lambda x: x['total_return'])
        
        print(f"🎯 Best by |IC|: {best_ic['strategy']} (IC: {best_ic['ic_pearson']:.4f})")
        print(f"🎯 Best by Hit Rate: {best_hit_rate['strategy']} ({best_hit_rate['hit_rate']:.1%})")
        print(f"🎯 Best by Sharpe: {best_sharpe['strategy']} ({best_sharpe['sharpe_ratio']:.2f})")
        print(f"🎯 Best by Return: {best_return['strategy']} ({best_return['total_return']:.1%})")
        
        # 종합 점수 (가중 평균)
        for result in all_results:
            # 정규화된 점수들
            ic_score = abs(result['ic_pearson']) * 100  # IC 점수
            hr_score = (result['hit_rate'] - 0.5) * 200  # Hit Rate 점수 (50% 기준)
            sharpe_score = max(0, result['sharpe_ratio'] * 10)  # Sharpe 점수
            
            # 종합 점수 (가중치: IC 40%, Hit Rate 40%, Sharpe 20%)
            combined_score = ic_score * 0.4 + hr_score * 0.4 + sharpe_score * 0.2
            result['combined_score'] = combined_score
        
        best_combined = max(all_results, key=lambda x: x['combined_score'])
        
        print(f"\n🥇 BEST OVERALL STRATEGY: {best_combined['strategy']}")
        print(f"   Category: {best_combined['category']}")
        print(f"   Combined Score: {best_combined['combined_score']:.2f}")
        print(f"   IC: {best_combined['ic_pearson']:.4f}")
        print(f"   Hit Rate: {best_combined['hit_rate']:.1%}")
        print(f"   Sharpe: {best_combined['sharpe_ratio']:.2f}")
        print(f"   Total Return: {best_combined['total_return']:.1%}")
        
        return best_combined
    
    def generate_summary_report(self, all_results: List[Dict], improvements: List[Dict]):
        """최종 요약 리포트"""
        print(f"\n📋 FINAL SUMMARY REPORT")
        print("=" * 60)
        
        # 카테고리별 성과
        categories = ['standard', 'contrarian', 'target_flipped']
        
        print(f"\n📊 Performance by Category:")
        for category in categories:
            cat_results = [r for r in all_results if r['category'] == category]
            if cat_results:
                avg_ic = np.mean([abs(r['ic_pearson']) for r in cat_results])
                avg_hr = np.mean([r['hit_rate'] for r in cat_results])
                avg_sharpe = np.mean([r['sharpe_ratio'] for r in cat_results])
                
                print(f"   {category.title():<15}: |IC|={avg_ic:.4f}, HR={avg_hr:.1%}, Sharpe={avg_sharpe:.2f}")
        
        # Contrarian 효과 요약
        if improvements:
            positive_improvements = len([imp for imp in improvements if imp['ic_improvement'] > 0])
            print(f"\n🔄 Contrarian Strategy Impact:")
            print(f"   Models showing IC improvement: {positive_improvements}/{len(improvements)}")
            print(f"   Average IC improvement: {np.mean([imp['ic_improvement'] for imp in improvements]):+.4f}")
        
        # 목표 달성도
        target_ic = 0.03
        target_hr = 0.55
        
        passing_models = [r for r in all_results 
                         if abs(r['ic_pearson']) >= target_ic and r['hit_rate'] >= target_hr]
        close_models = [r for r in all_results 
                       if abs(r['ic_pearson']) >= target_ic * 0.7 or r['hit_rate'] >= target_hr - 0.05]
        
        print(f"\n🎯 Target Achievement (IC≥{target_ic:.3f}, HR≥{target_hr:.1%}):")
        print(f"   Models passing targets: {len(passing_models)}/{len(all_results)}")
        print(f"   Models close to targets: {len(close_models)}/{len(all_results)}")
        
        if passing_models:
            print(f"   🏆 Passing models:")
            for model in passing_models:
                print(f"      {model['strategy']}: IC={model['ic_pearson']:.4f}, HR={model['hit_rate']:.1%}")
    
    def run_comprehensive_comparison(self):
        """전체 비교 분석 실행"""
        print("🚀 CONTRARIAN vs STANDARD MODEL COMPARISON")
        print("=" * 60)
        
        # 1. 데이터 로드
        self.load_data()
        
        # 2. 모델 훈련
        self.train_standard_models()
        self.train_contrarian_models()
        self.train_target_flipped_models()
        
        # 3. 비교 분석
        all_results = self.create_comparison_table()
        improvements = self.analyze_contrarian_effectiveness(all_results)
        best_strategy = self.find_best_strategy(all_results)
        self.generate_summary_report(all_results, improvements)
        
        # 4. 결과 저장
        self.save_results(all_results, improvements, best_strategy)
        
        return all_results, improvements, best_strategy
    
    def save_results(self, all_results: List[Dict], improvements: List[Dict], 
                    best_strategy: Dict):
        """결과 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/contrarian_comparison_{timestamp}.json"
        
        save_data = {
            'timestamp': timestamp,
            'all_results': all_results,
            'improvements': improvements,
            'best_strategy': best_strategy,
            'summary': {
                'total_models': len(all_results),
                'categories': list(set(r['category'] for r in all_results)),
                'best_overall': best_strategy['strategy'],
                'best_ic': max(all_results, key=lambda x: abs(x['ic_pearson']))['strategy'],
                'best_hit_rate': max(all_results, key=lambda x: x['hit_rate'])['strategy']
            }
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(save_data, f, indent=2)
            print(f"\n💾 Results saved to {filename}")
        except Exception as e:
            print(f"❌ Error saving: {e}")


def main():
    """메인 실행"""
    print("🔄 Contrarian vs Standard Models Performance Comparison")
    print("=" * 60)
    print("🎯 Goal: Determine if contrarian strategy improves performance")
    print("📊 Comparing: Standard, Contrarian, Target-Flipped approaches")
    print("=" * 60)
    
    analyzer = ContrarianComparisonAnalyzer()
    all_results, improvements, best_strategy = analyzer.run_comprehensive_comparison()
    
    print("\n✅ Comprehensive comparison completed!")
    print("📈 Check results for detailed analysis")


if __name__ == "__main__":
    main()

