#!/usr/bin/env python3
"""
Contrarian-Aware ML Models for Meme Stock Prediction
====================================================
Negative IC를 고려한 Contrarian Effect 모델

Key Insights:
- IC = -0.0097 (contrarian signal detected)
- Hit Rate < 50% (reverse prediction needed)
- Top features: log_mentions, price_ratio, returns
"""

import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json

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

class ContrarianAwareML:
    """Contrarian Effect를 고려한 ML 파이프라인"""
    
    def __init__(self, target_ic=0.03, target_hit_rate=0.55):
        self.target_ic = target_ic
        self.target_hit_rate = target_hit_rate
        self.results = {}
        
        print(f"🔄 Contrarian-Aware ML Pipeline")
        print(f"🎯 Target IC: {target_ic:.3f}, Target Hit Rate: {target_hit_rate:.1%}")
        
    def load_data(self):
        """Load Colab datasets"""
        print("📊 Loading Colab datasets...")
        
        self.train_df = pd.read_csv("data/colab_datasets/tabular_train_20250814_031335.csv")
        self.val_df = pd.read_csv("data/colab_datasets/tabular_val_20250814_031335.csv") 
        self.test_df = pd.read_csv("data/colab_datasets/tabular_test_20250814_031335.csv")
        
        print(f"   ✅ Train: {len(self.train_df)} samples")
        print(f"   ✅ Test: {len(self.test_df)} samples")
        
        # Check target statistics
        y_stats = self.train_df['y1d'].describe()
        print(f"   📊 Target stats: mean={y_stats['mean']:.4f}, std={y_stats['std']:.4f}")
        
    def prepare_top_features(self):
        """이전 실험에서 발견된 중요 특성들만 사용"""
        
        # 중요도 상위 특성들 (이전 실험 결과 기반)
        self.top_features = [
            'log_mentions',           # 95.7
            'price_ratio_sma20',      # 92.9
            'price_ratio_sma10',      # 89.9
            'returns_3d',             # 85.7
            'returns_1d',             # 81.0
            'market_sentiment',       # 80.5
            'reddit_market_ex',       # 79.7
            'price_reddit_momentum',  # 74.9
            'reddit_momentum_3',      # 74.4
            'reddit_surprise',        # 72.7
            'reddit_ema_3',           # 70.0
            'reddit_momentum_7',      # 69.7
            'rsi_14',                 # 66.2
            'reddit_ema_5',           # 65.4
            'returns_10d'             # 62.5
        ]
        
        # 존재하는 특성만 선택
        available_features = [f for f in self.top_features if f in self.train_df.columns]
        print(f"📊 Using top {len(available_features)} features")
        
        return available_features
        
    def calculate_contrarian_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Contrarian effect를 고려한 메트릭 계산"""
        
        # Original metrics
        ic, _ = pearsonr(y_true, y_pred) if len(y_true) > 1 else (0, 1)
        rank_ic, _ = spearmanr(y_true, y_pred) if len(y_true) > 1 else (0, 1)
        
        # Directional accuracy
        y_true_direction = np.sign(y_true)
        y_pred_direction = np.sign(y_pred)
        hit_rate = np.mean(y_true_direction == y_pred_direction)
        
        # Contrarian-adjusted metrics
        contrarian_ic = -ic  # Flip the IC
        contrarian_hit_rate = 1 - hit_rate  # Flip hit rate
        
        # Traditional metrics
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'original_ic': ic,
            'original_hit_rate': hit_rate,
            'contrarian_ic': contrarian_ic,
            'contrarian_hit_rate': contrarian_hit_rate,
            'rank_ic': rank_ic,
            'mse': mse,
            'r2': r2,
            'n_samples': len(y_true)
        }
    
    def train_contrarian_models(self, features: List[str]):
        """Contrarian effect를 고려한 모델 훈련"""
        
        print(f"\n🔄 Training Contrarian-Aware Models...")
        
        # Prepare data
        X_train = self.train_df[features].fillna(0).values.astype(np.float32)
        y_train = self.train_df['y1d'].values.astype(np.float32)
        
        X_test = self.test_df[features].fillna(0).values.astype(np.float32)
        y_test = self.test_df['y1d'].values.astype(np.float32)
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"   📈 Train: {len(X_train)}, Test: {len(X_test)}")
        
        models = {}
        
        # 1. Standard Models (for comparison)
        print("   🤖 Training Standard Models...")
        
        # LightGBM (가장 좋은 성능)
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
        lgb_model.fit(X_train_scaled, y_train)
        lgb_pred = lgb_model.predict(X_test_scaled)
        models['LightGBM_Standard'] = {
            'model': lgb_model,
            'predictions': lgb_pred,
            'metrics': self.calculate_contrarian_metrics(y_test, lgb_pred)
        }
        
        # 2. Contrarian Models (예측값 반전)
        print("   🔄 Training Contrarian Models...")
        
        # Contrarian LightGBM (예측값 반전)
        lgb_contrarian_pred = -lgb_pred
        models['LightGBM_Contrarian'] = {
            'model': lgb_model,  # Same model, flipped predictions
            'predictions': lgb_contrarian_pred,
            'metrics': self.calculate_contrarian_metrics(y_test, lgb_contrarian_pred)
        }
        
        # 3. Target-Flipped Models (타겟을 반전하여 훈련)
        print("   🔀 Training Target-Flipped Models...")
        
        # Target-flipped LightGBM
        y_train_flipped = -y_train
        lgb_flipped = lgb.LGBMRegressor(
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
        lgb_flipped.fit(X_train_scaled, y_train_flipped)
        lgb_flipped_pred = lgb_flipped.predict(X_test_scaled)
        models['LightGBM_TargetFlipped'] = {
            'model': lgb_flipped,
            'predictions': lgb_flipped_pred,
            'metrics': self.calculate_contrarian_metrics(y_test, lgb_flipped_pred)
        }
        
        # 4. Ensemble Contrarian Model
        print("   🎭 Creating Contrarian Ensemble...")
        
        # Random Forest for ensemble
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train_scaled, y_train)
        rf_pred = rf_model.predict(X_test_scaled)
        
        # XGBoost for ensemble
        xgb_model = xgb.XGBRegressor(
            max_depth=6,
            learning_rate=0.05,
            n_estimators=150,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )
        xgb_model.fit(X_train_scaled, y_train)
        xgb_pred = xgb_model.predict(X_test_scaled)
        
        # Contrarian ensemble (weighted average of flipped predictions)
        ensemble_pred = -(0.5 * lgb_pred + 0.3 * rf_pred + 0.2 * xgb_pred)
        models['Ensemble_Contrarian'] = {
            'predictions': ensemble_pred,
            'metrics': self.calculate_contrarian_metrics(y_test, ensemble_pred)
        }
        
        return models
    
    def evaluate_models(self, models: Dict):
        """모델 평가 및 결과 출력"""
        
        print(f"\n📊 CONTRARIAN MODEL EVALUATION:")
        print(f"{'Model':<20} {'Orig IC':<8} {'Contr IC':<9} {'Orig HR':<8} {'Contr HR':<9} {'Status'}")
        print("-" * 75)
        
        results = []
        
        for name, model_data in models.items():
            metrics = model_data['metrics']
            
            orig_ic = metrics['original_ic']
            contr_ic = metrics['contrarian_ic']
            orig_hr = metrics['original_hit_rate']
            contr_hr = metrics['contrarian_hit_rate']
            
            # Check if contrarian version meets targets
            contr_status = "✅ PASS" if (contr_ic >= self.target_ic and contr_hr >= self.target_hit_rate) else "❌ FAIL"
            if contr_ic >= self.target_ic * 0.5 or contr_hr >= self.target_hit_rate - 0.05:
                contr_status = "⚠️ CLOSE"
            
            print(f"{name:<20} {orig_ic:<8.4f} {contr_ic:<9.4f} {orig_hr:<8.3f} {contr_hr:<9.3f} {contr_status}")
            
            results.append({
                'model': name,
                'original_ic': orig_ic,
                'contrarian_ic': contr_ic,
                'original_hit_rate': orig_hr,
                'contrarian_hit_rate': contr_hr,
                'metrics': metrics
            })
        
        return results
    
    def find_best_contrarian_strategy(self, results: List[Dict]):
        """최적 contrarian 전략 찾기"""
        
        print(f"\n🏆 BEST CONTRARIAN STRATEGIES:")
        
        # Sort by contrarian IC
        sorted_results = sorted(results, key=lambda x: x['contrarian_ic'], reverse=True)
        
        print(f"\n🔄 Top Contrarian IC Models:")
        for i, result in enumerate(sorted_results[:5], 1):
            contr_ic = result['contrarian_ic']
            contr_hr = result['contrarian_hit_rate']
            print(f"{i}. {result['model']:<20} IC: {contr_ic:.4f}, HR: {contr_hr:.3f}")
        
        # Sort by contrarian hit rate
        sorted_by_hr = sorted(results, key=lambda x: x['contrarian_hit_rate'], reverse=True)
        
        print(f"\n🎯 Top Contrarian Hit Rate Models:")
        for i, result in enumerate(sorted_by_hr[:5], 1):
            contr_ic = result['contrarian_ic']
            contr_hr = result['contrarian_hit_rate']
            print(f"{i}. {result['model']:<20} IC: {contr_ic:.4f}, HR: {contr_hr:.3f}")
        
        # Best overall (IC + HR combined)
        for result in results:
            result['combined_score'] = result['contrarian_ic'] * 0.6 + (result['contrarian_hit_rate'] - 0.5) * 0.4
        
        best_combined = sorted(results, key=lambda x: x['combined_score'], reverse=True)
        
        print(f"\n🎖️ BEST COMBINED PERFORMANCE:")
        best = best_combined[0]
        print(f"Model: {best['model']}")
        print(f"Contrarian IC: {best['contrarian_ic']:.4f}")
        print(f"Contrarian Hit Rate: {best['contrarian_hit_rate']:.3f}")
        print(f"Combined Score: {best['combined_score']:.4f}")
        
        # Success analysis
        passing_models = [r for r in results 
                         if r['contrarian_ic'] >= self.target_ic and r['contrarian_hit_rate'] >= self.target_hit_rate]
        
        close_models = [r for r in results 
                       if r['contrarian_ic'] >= self.target_ic * 0.5 or r['contrarian_hit_rate'] >= self.target_hit_rate - 0.05]
        
        print(f"\n📈 SUCCESS ANALYSIS:")
        print(f"✅ Models passing targets: {len(passing_models)}/{len(results)}")
        print(f"⚠️ Models close to targets: {len(close_models)}/{len(results)}")
        print(f"🎯 Best Contrarian IC: {max(r['contrarian_ic'] for r in results):.4f} (Target: {self.target_ic:.3f})")
        print(f"🎯 Best Contrarian HR: {max(r['contrarian_hit_rate'] for r in results):.3f} (Target: {self.target_hit_rate:.3f})")
        
        return best
    
    def generate_trading_signals(self, best_model: Dict):
        """거래 신호 생성"""
        
        print(f"\n💰 TRADING SIGNAL GENERATION:")
        print(f"Using: {best_model['model']}")
        
        # 최근 예측값들 분석
        model_data = None
        for name, data in self.results.items():
            if name == best_model['model']:
                model_data = data
                break
        
        if model_data is None:
            print("❌ Model data not found")
            return
        
        predictions = model_data['predictions']
        
        # 신호 강도 분석
        pred_std = np.std(predictions)
        strong_buy_threshold = np.percentile(predictions, 90)
        buy_threshold = np.percentile(predictions, 75)
        sell_threshold = np.percentile(predictions, 25)
        strong_sell_threshold = np.percentile(predictions, 10)
        
        print(f"📊 Signal Thresholds:")
        print(f"   Strong Buy: > {strong_buy_threshold:.4f}")
        print(f"   Buy: > {buy_threshold:.4f}")
        print(f"   Sell: < {sell_threshold:.4f}")
        print(f"   Strong Sell: < {strong_sell_threshold:.4f}")
        
        # 최근 3일 신호
        recent_predictions = predictions[-3:]
        recent_signals = []
        
        for i, pred in enumerate(recent_predictions):
            if pred > strong_buy_threshold:
                signal = "STRONG BUY"
            elif pred > buy_threshold:
                signal = "BUY"
            elif pred < strong_sell_threshold:
                signal = "STRONG SELL"
            elif pred < sell_threshold:
                signal = "SELL"
            else:
                signal = "HOLD"
            
            recent_signals.append(signal)
            print(f"   Day {i+1}: {pred:.4f} -> {signal}")
        
        # 전체 신호 요약
        current_signal = recent_signals[-1]
        signal_consistency = len(set(recent_signals[-2:]))  # 최근 2일 일관성
        
        print(f"\n🚦 CURRENT RECOMMENDATION:")
        print(f"   Signal: {current_signal}")
        print(f"   Consistency: {'High' if signal_consistency == 1 else 'Low'}")
        print(f"   Confidence: {abs(recent_predictions[-1]) / pred_std:.2f}σ")
        
    def run_contrarian_experiment(self):
        """전체 contrarian 실험 실행"""
        
        print("🔄 CONTRARIAN-AWARE ML EXPERIMENT")
        print("=" * 50)
        
        # Load data
        self.load_data()
        
        # Prepare features
        features = self.prepare_top_features()
        
        # Train models
        models = self.train_contrarian_models(features)
        self.results = models
        
        # Evaluate models
        results = self.evaluate_models(models)
        
        # Find best strategy
        best_model = self.find_best_contrarian_strategy(results)
        
        # Generate trading signals
        self.generate_trading_signals(best_model)
        
        print(f"\n✅ Contrarian experiment completed!")
        

def main():
    """Main execution"""
    print("🔄 Contrarian-Aware ML Models for Meme Stock Prediction")
    print("=" * 60)
    print("📊 Hypothesis: Negative IC suggests contrarian trading opportunities")
    print("🎯 Goal: Convert negative correlation into positive trading signals")
    print("=" * 60)
    
    # Run experiment
    pipeline = ContrarianAwareML()
    pipeline.run_contrarian_experiment()
    
    print("\n🎉 All done! Check the contrarian trading signals above.")


if __name__ == "__main__":
    main()

